"""Untested high-impact symbols: risky code with a large blast radius and no linked test.

Composes two capabilities that already exist in this codebase rather than
inventing a third: E1's `analyze_impact` (graph/impact.py) for "how many
things call this, directly and transitively," and the same `TESTS`-edge-only
`expand_dependencies` query `ledger/build_ledger.py`'s `_has_linked_test`
already uses for "does anything test this" — any confidence level counts,
same as there, since this is "is there a linked test at all," not a
confidence gate. `graph` must already have `link_tests` (E2) run on it, same
precondition as every other TESTS-edge consumer in this codebase.

One `analyze_impact` call per candidate symbol rather than a new combined
traversal: this module is glue over two already-shipped analyses, matching
`build_ledger.py`'s own "glue, not a new algorithm" framing for the same
`has_test` reuse.
"""

from __future__ import annotations

import networkx as nx

from loupe_core.parsing.schema import Symbol

from .builder import EdgeType
from .impact import analyze_impact
from .traversal import expand_dependencies


def _has_linked_test(graph: nx.DiGraph, symbol_id: str) -> bool:
    return bool(expand_dependencies(graph, symbol_id, depth=1, direction="incoming", edge_type=EdgeType.TESTS))


def find_untested_high_impact_symbols(
    graph: nx.DiGraph,
    symbols_by_id: dict[str, Symbol],
    pagerank_scores: dict[str, float],
    min_impact: int = 1,
) -> list[tuple[str, int]]:
    """(symbol_id, impact_size) pairs for every symbol with no linked test and an impact_size
    (direct + transitive callers, E1's own two-tier count) at or above `min_impact`. Sorted by
    impact_size descending, ties broken by symbol_id for deterministic output — the same
    stability bar every other graph pass in this codebase holds itself to.
    """
    results: list[tuple[str, int]] = []
    for symbol_id in symbols_by_id:
        if _has_linked_test(graph, symbol_id):
            continue
        report = analyze_impact(graph, symbols_by_id, pagerank_scores, symbol_id)
        impact_size = len(report.directly_affected) + len(report.transitively_affected)
        if impact_size >= min_impact:
            results.append((symbol_id, impact_size))

    return sorted(results, key=lambda pair: (-pair[1], pair[0]))
