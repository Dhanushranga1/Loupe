"""Bounded, cycle-safe graph traversal (docs/phase-1-graph-theory.md §7)."""

from __future__ import annotations

from typing import Literal

import networkx as nx

Direction = Literal["outgoing", "incoming", "both"]


def expand_dependencies(
    graph: nx.DiGraph,
    symbol_id: str,
    depth: int = 1,
    direction: Direction = "outgoing",
) -> set[str]:
    """Symbol ids reachable from `symbol_id` within `depth` hops (excludes `symbol_id` itself).

    outgoing = what this symbol depends on (successors) — dependency expansion.
    incoming = what depends on this symbol (predecessors) — impact analysis.
    both     = union of both directions.

    Explicit visited set + layer-by-layer BFS: the graph is expected to
    contain cycles (mutual recursion, circular imports), and an unguarded
    traversal would loop forever or blow the stack.
    """
    visited: set[str] = {symbol_id}
    frontier: set[str] = {symbol_id}
    result: set[str] = set()

    for _ in range(depth):
        next_frontier: set[str] = set()
        for node in frontier:
            neighbors: set[str] = set()
            if direction in ("outgoing", "both"):
                neighbors.update(graph.successors(node))
            if direction in ("incoming", "both"):
                neighbors.update(graph.predecessors(node))
            next_frontier.update(neighbors - visited)
        if not next_frontier:
            break
        result.update(next_frontier)
        visited.update(next_frontier)
        frontier = next_frontier

    return result
