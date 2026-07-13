"""Bounded, cycle-safe graph traversal (docs/phase-1-graph-theory.md §7)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import networkx as nx

if TYPE_CHECKING:
    from .builder import EdgeType

Direction = Literal["outgoing", "incoming", "both"]


def expand_dependencies(
    graph: nx.DiGraph,
    symbol_id: str,
    depth: int = 1,
    direction: Direction = "outgoing",
    edge_type: "EdgeType | None" = None,
) -> set[str]:
    """Symbol ids reachable from `symbol_id` within `depth` hops (excludes `symbol_id` itself).

    outgoing = what this symbol depends on (successors) — dependency expansion.
    incoming = what depends on this symbol (predecessors) — impact analysis.
    both     = union of both directions.

    `edge_type=None` (the default) traverses every edge regardless of type,
    exactly Phase 1's original behavior. Passing a specific `EdgeType`
    (e.g. `EdgeType.TESTS`, added by E2) restricts traversal to only that
    kind of edge — added so a new edge type never needs a new tool, per
    docs/loupe-extensions.md E2's explicit design decision.

    Explicit visited set + layer-by-layer BFS: the graph is expected to
    contain cycles (mutual recursion, circular imports), and an unguarded
    traversal would loop forever or blow the stack.
    """
    visited: set[str] = {symbol_id}
    frontier: set[str] = {symbol_id}
    result: set[str] = set()

    def _matches(u: str, v: str) -> bool:
        return edge_type is None or graph[u][v].get("edge_type") == edge_type

    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            neighbors: set[str] = set()
            if direction in ("outgoing", "both"):
                neighbors.update(succ for succ in graph.successors(node) if _matches(node, succ))
            if direction in ("incoming", "both"):
                neighbors.update(pred for pred in graph.predecessors(node) if _matches(pred, node))
            next_frontier.update(neighbors - visited)
        if not next_frontier:
            break
        result.update(next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier

    return result
