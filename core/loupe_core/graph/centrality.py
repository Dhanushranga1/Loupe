"""PageRank centrality over the symbol graph (docs/phase-1-graph-theory.md §7).

No custom tuning of PageRank's damping factor for this phase — use the
library default (0.85) until there's a concrete reason to change it.
"""

from __future__ import annotations

import networkx as nx


def compute_pagerank(graph: nx.DiGraph) -> dict[str, float]:
    """PageRank score per symbol_id; empty dict for an empty graph (no nodes to rank)."""
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, weight="weight")
