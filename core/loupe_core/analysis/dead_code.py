"""E6 — Dead code detection (docs/PhaseX/zero-cost-static-analysis-pack.md).

Directly reuses E1's impact-analysis traversal, run in the direction nothing
currently reads a symbol — no new graph algorithm needed.
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from loupe_core.adapters.fastapi.routes import looks_like_http_route
from loupe_core.graph.test_linkage import is_test_file
from loupe_core.graph.traversal import expand_dependencies
from loupe_core.parsing.schema import Symbol


@dataclass(frozen=True)
class DeadCodeFinding:
    symbol_id: str
    qualified_name: str
    file_path: str


def find_dead_code(graph: nx.DiGraph, symbols_by_id: dict[str, Symbol]) -> list[DeadCodeFinding]:
    """A dead-code candidate: zero incoming edges (nothing calls or inherits
    from it — `expand_dependencies(..., direction="incoming")` returns
    empty), excluding:
    - route handlers (`looks_like_http_route`) — the framework calls these,
      not application code, so a route legitimately has zero *in-repo*
      callers;
    - test symbols (`is_test_file`) — a test runner calls these, not other
      application code;
    - `__init__` methods — the interpreter calls these implicitly on
      instantiation, never via an explicit call site the graph would resolve.

    Honest limitation, not solved here: a symbol registered as an entry
    point outside Python call syntax entirely (e.g. a `[project.scripts]`
    console-script target, a Celery task registered by decorator alone) has
    no real caller *in the graph* either, and will be flagged as a false
    positive — the same limitation essentially every static dead-code tool
    has, not something this check's simple exclusion list can fully close.
    """
    findings = []
    for symbol_id, symbol in symbols_by_id.items():
        if symbol_id not in graph:
            continue
        if looks_like_http_route(symbol):
            continue
        if is_test_file(symbol.file_path):
            continue
        if symbol.name == "__init__":
            continue
        incoming = expand_dependencies(graph, symbol_id, depth=1, direction="incoming")
        if not incoming:
            findings.append(
                DeadCodeFinding(symbol_id=symbol_id, qualified_name=symbol.qualified_name, file_path=symbol.file_path)
            )
    return sorted(findings, key=lambda f: (f.file_path, f.qualified_name))
