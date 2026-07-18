"""Tests for adapters/fastapi/smells.py (docs/PhaseX/phase-7-fastapi-adapter-smells.md)."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from loupe_core.adapters.fastapi import smells as smells_module
from loupe_core.adapters.fastapi.smells import ALL_CATEGORIES, find_code_smells
from loupe_core.graph.builder import build_graph, parse_file

FIXTURES = Path(__file__).parent / "fixtures" / "phase7_smells"


@pytest.fixture(scope="module")
def repo():
    old_cwd = os.getcwd()
    os.chdir(FIXTURES)
    try:
        files = sorted(p.name for p in Path(".").glob("*.py"))
        parsed = [parse_file(f) for f in files]
        g = build_graph(parsed)
        yield parsed, g
    finally:
        os.chdir(old_cwd)


def _names(findings, category=None):
    return {f.qualified_name for f in findings if category is None or f.category == category}


def _run(repo, category=None):
    parsed, g = repo
    return find_code_smells(parsed, g.graph, g.unresolved, g.pagerank_scores, category=category)


# --------------------------------------------------------------------------
# a. Missing response model / return type
# --------------------------------------------------------------------------


def test_missing_response_model_flags_the_deliberate_instance(repo):
    names = _names(_run(repo), "missing_response_model")
    assert "list_items_smelly" in names


def test_missing_response_model_does_not_flag_the_clean_counterexample(repo):
    names = _names(_run(repo), "missing_response_model")
    assert "get_item_clean" not in names


# --------------------------------------------------------------------------
# b. Untyped route parameters
# --------------------------------------------------------------------------


def test_untyped_params_flags_the_deliberate_instance(repo):
    names = _names(_run(repo), "untyped_params")
    assert "create_item_smelly" in names


def test_untyped_params_does_not_flag_the_clean_counterexample(repo):
    names = _names(_run(repo), "untyped_params")
    assert "create_item_clean" not in names


# --------------------------------------------------------------------------
# c. Blocking calls in async handlers — the sync-vs-async distinction is the
#    acceptance criterion that actually matters here, not just "flags it."
# --------------------------------------------------------------------------


def test_blocking_call_flags_the_async_handler(repo):
    names = _names(_run(repo), "blocking_call_in_async")
    assert "slow_handler_smelly" in names


def test_blocking_call_does_not_flag_the_identical_call_from_a_sync_handler(repo):
    """Same time.sleep(1) call, same file shape — only the async context makes it a smell."""
    names = _names(_run(repo), "blocking_call_in_async")
    assert "slow_handler_clean_sync" not in names


# --------------------------------------------------------------------------
# d. Business logic embedded in a route handler
# --------------------------------------------------------------------------


def test_business_logic_in_handler_flags_the_deliberate_instance(repo):
    names = _names(_run(repo), "business_logic_in_handler")
    assert "create_order_smelly" in names


def test_business_logic_in_handler_does_not_flag_the_clean_counterexample(repo):
    names = _names(_run(repo), "business_logic_in_handler")
    assert "create_order_clean" not in names


# --------------------------------------------------------------------------
# e. N+1 query pattern
# --------------------------------------------------------------------------


def test_n_plus_one_flags_the_deliberate_instance(repo):
    names = _names(_run(repo), "n_plus_one")
    assert "get_all_with_details_smelly" in names


def test_n_plus_one_does_not_flag_the_clean_counterexample(repo):
    names = _names(_run(repo), "n_plus_one")
    assert "get_all_with_details_clean" not in names


# --------------------------------------------------------------------------
# f. Circular dependencies
# --------------------------------------------------------------------------


def test_circular_dependency_flags_both_symbols_in_the_real_cycle(repo):
    names = _names(_run(repo), "circular_dependency")
    assert {"helper_a", "helper_b"} <= names


def test_circular_dependency_does_not_flag_an_unrelated_acyclic_symbol(repo):
    names = _names(_run(repo), "circular_dependency")
    assert "format_response" not in names
    assert "validate_a" not in names


# --------------------------------------------------------------------------
# g. God-object / hub detection
# --------------------------------------------------------------------------


def test_god_object_flags_the_deliberate_extreme_outlier(repo):
    names = _names(_run(repo), "god_object")
    assert "format_response" in names


def test_god_object_does_not_flag_most_ordinary_symbols(repo):
    """The statistical rule must actually separate outliers from the rest, not just
    compute a number — most of the fixture's ~31 symbols should be unflagged."""
    parsed, _g = repo
    total_symbols = sum(len(pf.symbols) for pf in parsed)
    flagged = _names(_run(repo), "god_object")
    assert len(flagged) < total_symbols / 2


# --------------------------------------------------------------------------
# Conventions reuse — not a new technique, verified as reuse, not duplication
# --------------------------------------------------------------------------


def test_convention_violation_findings_come_from_e4s_real_mining_functions(repo):
    """Verified by spying on the exact underlying functions and confirming they're
    actually called — not by checking output shape alone, which a duplicated
    reimplementation could also satisfy."""
    parsed, g = repo
    with patch.object(smells_module, "mine_conventions", wraps=smells_module.mine_conventions) as spy:
        find_code_smells(parsed, g.graph, g.unresolved, g.pagerank_scores, category="convention_violation")
    assert spy.called


def test_convention_violation_surfaces_real_missing_docstrings(repo):
    names = _names(_run(repo), "convention_violation")
    assert "caller_one" in names  # has no docstring in the fixture


# --------------------------------------------------------------------------
# Category filter
# --------------------------------------------------------------------------


def test_category_filter_returns_only_that_category(repo):
    findings = _run(repo, category="god_object")
    assert findings
    assert all(f.category == "god_object" for f in findings)


def test_no_filter_returns_all_categories_combined(repo):
    findings = _run(repo)
    categories_present = {f.category for f in findings}
    # every category with a real deliberate fixture instance shows up when unfiltered
    assert categories_present == set(ALL_CATEGORIES)
