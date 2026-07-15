"""Tests for graph/impact.py (docs/loupe-extensions.md E1 — Blast-Radius / Impact Analysis)."""

import os
import shutil
from pathlib import Path

import pytest

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.graph.impact import analyze_impact, hub_threshold

FIXTURES = Path(__file__).parent / "fixtures" / "e1"
FILES = ["utils.py", "models.py", "services.py"]


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


def test_direct_and_transitive_affected_match_the_real_two_hop_call_chain(repo):
    """format_currency <- Order.total (direct) <- Order.describe (transitive) is a real,
    resolved chain in the fixture (see models.py's Order.describe docstring for why it's
    built via self.<name> rather than a local-variable chain)."""
    g, symbols_by_id = repo
    target_id = _id_by_name(symbols_by_id, "format_currency")

    report = analyze_impact(g.graph, symbols_by_id, g.pagerank_scores, target_id, depth=2)

    direct_names = {s.qualified_name for s in report.directly_affected}
    transitive_names = {s.qualified_name for s in report.transitively_affected}

    assert direct_names == {"Order.total", "format_receipt_amount"}
    assert transitive_names == {"Order.describe"}
    # never double-counted across both tiers
    assert direct_names.isdisjoint(transitive_names)


def test_leaf_symbol_with_zero_callers_returns_empty_not_an_error(repo):
    g, symbols_by_id = repo
    leaf_id = _id_by_name(symbols_by_id, "unused_utility")

    report = analyze_impact(g.graph, symbols_by_id, g.pagerank_scores, leaf_id, depth=2)

    assert report.directly_affected == []
    assert report.transitively_affected == []


def test_high_centrality_warnings_includes_the_queried_symbol_itself_when_it_is_a_hub(repo):
    """format_currency has two independent direct callers plus a transitive one, making its
    PageRank score a clear, unambiguous outlier in this small fixture (see impact.py's
    hub_threshold docstring for the mean + 1 stdev definition)."""
    g, symbols_by_id = repo
    target_id = _id_by_name(symbols_by_id, "format_currency")

    report = analyze_impact(g.graph, symbols_by_id, g.pagerank_scores, target_id, depth=2)

    assert target_id in report.high_centrality_warnings
    # and it's not just present but genuinely the most central entry
    assert report.high_centrality_warnings[0] == target_id


def test_hub_threshold_is_a_real_outlier_cutoff_not_just_the_top_of_any_list(repo):
    g, symbols_by_id = repo
    threshold = hub_threshold(g.pagerank_scores)

    above = [sid for sid, score in g.pagerank_scores.items() if score > threshold]
    # exactly the unambiguous outlier (format_currency), not most/all of the graph
    assert len(above) == 1
    assert symbols_by_id[above[0]].qualified_name == "format_currency"


def test_symbol_with_no_hub_neighbors_gets_no_warnings(repo):
    g, symbols_by_id = repo
    leaf_id = _id_by_name(symbols_by_id, "unused_utility")

    report = analyze_impact(g.graph, symbols_by_id, g.pagerank_scores, leaf_id, depth=2)

    assert report.high_centrality_warnings == []


def test_affected_route_count_is_zero_when_no_caller_has_a_route_shaped_decorator(repo):
    g, symbols_by_id = repo
    target_id = _id_by_name(symbols_by_id, "format_currency")

    report = analyze_impact(g.graph, symbols_by_id, g.pagerank_scores, target_id, depth=2)

    assert report.affected_route_count == 0


def test_affected_route_count_detects_fastapi_style_decorators(tmp_path):
    """Real gap found via a real ~900-symbol FastAPI codebase: this used to be hardcoded
    0 always. Now detects the <name>.<http_method>(...) decorator shape directly from
    Symbol.decorators — no FastAPI adapter needed for this much."""
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        (tmp_path / "core_logic.py").write_text(
            "def require_org_context(request):\n    return request.headers.get('org-id')\n"
        )
        (tmp_path / "kb.py").write_text(
            "from fastapi import APIRouter\n"
            "from core_logic import require_org_context\n\n"
            "router = APIRouter()\n\n\n"
            "@router.get('/search')\n"
            "async def search(q: str):\n"
            "    org_id = require_org_context(None)\n"
            "    return org_id\n"
        )
        (tmp_path / "admin.py").write_text(
            "from fastapi import APIRouter\n"
            "from core_logic import require_org_context\n\n"
            "router = APIRouter()\n\n\n"
            "@router.post('/reindex')\n"
            "async def reindex(payload: dict):\n"
            "    org_id = require_org_context(None)\n"
            "    return org_id\n"
        )
        (tmp_path / "helpers.py").write_text(
            "from core_logic import require_org_context\n\n\n"
            "def non_route_caller():\n"
            "    return require_org_context(None)\n"
        )

        files = ["core_logic.py", "kb.py", "admin.py", "helpers.py"]
        parsed = [parse_file(f) for f in files]
        g = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        target_id = _id_by_name(symbols_by_id, "require_org_context")

        report = analyze_impact(g.graph, symbols_by_id, g.pagerank_scores, target_id, depth=2)

        direct_names = {s.qualified_name for s in report.directly_affected}
        assert direct_names == {"search", "reindex", "non_route_caller"}
        assert report.affected_route_count == 2, "exactly the two @router.get/@router.post handlers, not the plain function"
    finally:
        os.chdir(old_cwd)
