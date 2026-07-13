"""Tests for graph/test_linkage.py (docs/loupe-extensions.md E2 — Test-to-Code Linkage)."""

import os
import shutil
from pathlib import Path

import pytest

from loupe_core.graph.builder import EdgeType, build_graph, parse_file
from loupe_core.graph.test_linkage import TestConfidence, link_tests
from loupe_core.graph.traversal import expand_dependencies

FIXTURES = Path(__file__).parent / "fixtures" / "e2"
FILES = ["utils.py", "test_utils.py"]


@pytest.fixture
def repo(tmp_path):
    for f in FILES:
        shutil.copy(FIXTURES / f, tmp_path / f)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        parsed = [parse_file(f) for f in FILES]
        g = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        yield g, symbols_by_id
    finally:
        os.chdir(old_cwd)


def _id_by_name(symbols_by_id, qualified_name: str) -> str:
    return next(s.id for s in symbols_by_id.values() if s.qualified_name == qualified_name)


def _link(links, test_name, target_name, symbols_by_id):
    test_id = _id_by_name(symbols_by_id, test_name)
    target_id = _id_by_name(symbols_by_id, target_name)
    return next((l for l in links if l.test_symbol_id == test_id and l.target_symbol_id == target_id), None)


def test_naming_and_call_heuristics_agree_is_confirmed(repo):
    g, symbols_by_id = repo
    links = link_tests(g.graph, symbols_by_id)

    link = _link(links, "test_format_currency", "format_currency", symbols_by_id)
    assert link is not None
    assert link.confidence == TestConfidence.CONFIRMED


def test_naming_matches_but_target_is_mocked_not_called_is_naming_only(repo):
    g, symbols_by_id = repo
    links = link_tests(g.graph, symbols_by_id)

    link = _link(links, "test_validate_email", "validate_email", symbols_by_id)
    assert link is not None
    assert link.confidence == TestConfidence.NAMING_ONLY


def test_call_without_matching_naming_convention_is_call_only(repo):
    g, symbols_by_id = repo
    links = link_tests(g.graph, symbols_by_id)

    link = _link(links, "check_currency_formatting", "format_currency", symbols_by_id)
    assert link is not None
    assert link.confidence == TestConfidence.CALL_ONLY


def test_symbol_with_no_matching_test_returns_empty_via_expand_dependencies_not_an_error(repo):
    """Nothing in this fixture tests test_format_currency itself — the zero-result case
    must come back as an empty set, not raise, exactly like any other expand_dependencies call."""
    g, symbols_by_id = repo
    link_tests(g.graph, symbols_by_id)

    untested_id = _id_by_name(symbols_by_id, "test_format_currency")
    result = expand_dependencies(g.graph, untested_id, depth=1, direction="incoming", edge_type=EdgeType.TESTS)
    assert result == set()


def test_expand_dependencies_with_tests_edge_type_reaches_only_test_edges(repo):
    g, symbols_by_id = repo
    link_tests(g.graph, symbols_by_id)
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")

    via_tests = expand_dependencies(g.graph, format_currency_id, depth=1, direction="incoming", edge_type=EdgeType.TESTS)
    names = {symbols_by_id[i].qualified_name for i in via_tests}
    assert names == {"test_format_currency", "check_currency_formatting"}

    # unfiltered incoming traversal must include the same edges (TESTS is additive, not exclusive)
    via_all = expand_dependencies(g.graph, format_currency_id, depth=1, direction="incoming")
    assert names.issubset({symbols_by_id[i].qualified_name for i in via_all})


def test_expand_dependencies_edge_type_filter_does_not_change_default_behavior(repo):
    """Backward-compatibility check: calling expand_dependencies without edge_type
    must traverse exactly what it did before E2 introduced the filter parameter."""
    g, symbols_by_id = repo
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")

    before_link_tests = expand_dependencies(g.graph, format_currency_id, depth=1, direction="incoming")
    link_tests(g.graph, symbols_by_id)
    after_link_tests_no_filter = expand_dependencies(g.graph, format_currency_id, depth=1, direction="incoming")

    # TESTS edges add new incoming neighbors on top of, not instead of, the original CALLS ones
    assert before_link_tests.issubset(after_link_tests_no_filter)
