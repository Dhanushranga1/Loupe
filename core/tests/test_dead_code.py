"""Tests for analysis/dead_code.py (docs/PhaseX/zero-cost-static-analysis-pack.md E6)."""

import os
from pathlib import Path

import pytest

from loupe_core.analysis.dead_code import find_dead_code
from loupe_core.graph.builder import build_graph, parse_file


@pytest.fixture
def dead_code_repo(tmp_path):
    _write(
        tmp_path,
        "app.py",
        "from fastapi import FastAPI\n\n"
        "app = FastAPI()\n\n\n"
        "def used_helper():\n    return 1\n\n\n"
        "def unused_helper():\n    return 2\n\n\n"
        "def caller():\n    return used_helper()\n\n\n"
        "@app.get('/status')\n"
        "def status_route():\n    return {'ok': True}\n\n\n"
        "class Widget:\n"
        "    def __init__(self):\n"
        "        self.value = 1\n\n"
        "    def used_method(self):\n        return self.value\n\n"
        "    def dead_method(self):\n        return self.value * 2\n\n"
        "    def caller_method(self):\n"
        # The call resolver deliberately does no type inference (docs/phase-1-graph-theory.md):
        # a `self.method()` call is a resolvable pattern, `Widget().used_method()` is not.
        "        return self.used_method()\n",
    )
    _write(tmp_path, "test_app.py", "def test_something():\n    assert unused_helper_never_called_directly() == 2\n")
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        parsed = [parse_file("app.py"), parse_file("test_app.py")]
        graph = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        yield graph, symbols_by_id
    finally:
        os.chdir(old_cwd)


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _qualified_names(findings) -> set[str]:
    return {f.qualified_name for f in findings}


def test_genuinely_unused_helper_is_flagged(dead_code_repo):
    graph, symbols_by_id = dead_code_repo
    findings = find_dead_code(graph.graph, symbols_by_id)
    assert "unused_helper" in _qualified_names(findings)


def test_used_helper_is_not_flagged(dead_code_repo):
    graph, symbols_by_id = dead_code_repo
    findings = find_dead_code(graph.graph, symbols_by_id)
    assert "used_helper" not in _qualified_names(findings)


def test_route_handler_with_zero_callers_is_excluded_not_flagged(dead_code_repo):
    """§6's own acceptance criterion: a route handler with zero incoming
    edges is correct (nothing in the codebase calls a route, the framework
    does) — must be excluded via the route-kind check, not flagged."""
    graph, symbols_by_id = dead_code_repo
    findings = find_dead_code(graph.graph, symbols_by_id)
    assert "status_route" not in _qualified_names(findings)


def test_test_symbols_never_flagged(dead_code_repo):
    """A test function typically has zero in-repo callers (the test runner
    calls it, not application code) but must never be flagged as dead."""
    graph, symbols_by_id = dead_code_repo
    findings = find_dead_code(graph.graph, symbols_by_id)
    assert "test_something" not in _qualified_names(findings)


def test_dunder_init_never_flagged(dead_code_repo):
    graph, symbols_by_id = dead_code_repo
    findings = find_dead_code(graph.graph, symbols_by_id)
    assert "Widget.__init__" not in _qualified_names(findings)


def test_used_method_not_flagged_dead_method_is(dead_code_repo):
    graph, symbols_by_id = dead_code_repo
    findings = find_dead_code(graph.graph, symbols_by_id)
    names = _qualified_names(findings)
    assert "Widget.used_method" not in names
    assert "Widget.dead_method" in names


def test_findings_sorted_deterministically(dead_code_repo):
    graph, symbols_by_id = dead_code_repo
    first = find_dead_code(graph.graph, symbols_by_id)
    second = find_dead_code(graph.graph, symbols_by_id)

    assert [f.symbol_id for f in first] == [f.symbol_id for f in second]
    assert first == sorted(first, key=lambda f: (f.file_path, f.qualified_name))
