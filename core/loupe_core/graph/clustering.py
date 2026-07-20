"""Louvain community detection over the symbol graph
(docs/PhaseX/phase-10.5-graph-clustering.md).

Computed once per full or incremental reindex — `build_graph` (this module's
only caller) is only ever invoked from the reindex pipeline, never per MCP
tool call, so storing the result on `LoupeGraph` alongside `pagerank_scores`
already satisfies "compute once, cache, never recompute live" (§4) without
any separate caching layer.

Community structure is about *connectedness*, not call direction, so
clustering runs over the undirected projection of the graph (§3) — not the
directed graph Phase 1 builds for call resolution.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx

LOUVAIN_SEED = 42

# networkx's own convention: resolution < 1 favors fewer, larger communities;
# > 1 favors more, smaller ones — verified empirically against a
# multi-subsystem fixture (test_clustering.py) before trusting the direction
# from the docstring alone. Coarse serves claude_md_generator's architecture
# overview (a handful of major subsystems); fine serves scope's precise
# per-symbol boundary detection (§3).
COARSE_RESOLUTION = 0.5
FINE_RESOLUTION = 2.0


@dataclass
class GraphClusters:
    coarse: list[set[str]] = field(default_factory=list)
    fine: list[set[str]] = field(default_factory=list)


def compute_clusters(graph: nx.DiGraph, seed: int = LOUVAIN_SEED) -> GraphClusters:
    """Both resolution levels, over the same fixed seed (§3's determinism fix,
    part 1) — repeated calls against an *unchanged* graph produce identical
    partitions. Empty graph short-circuits to empty clusters rather than
    handing Louvain a degenerate input.
    """
    if graph.number_of_nodes() == 0:
        return GraphClusters(coarse=[], fine=[])

    undirected = graph.to_undirected()
    coarse = nx.algorithms.community.louvain_communities(
        undirected, weight="weight", resolution=COARSE_RESOLUTION, seed=seed
    )
    fine = nx.algorithms.community.louvain_communities(
        undirected, weight="weight", resolution=FINE_RESOLUTION, seed=seed
    )
    return GraphClusters(coarse=coarse, fine=fine)


def jaccard_similarity(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def align_clusters(old_clusters: list[set[str]], new_clusters: list[set[str]]) -> dict[int, int | None]:
    """Maps each `new_clusters` index to whichever `old_clusters` index shares
    the highest Jaccard similarity (§3's determinism fix, part 2) — "the same
    cluster, evolved" rather than a coincidentally-labeled new one, since raw
    cluster indices/IDs are not meaningfully comparable across two separate
    Louvain runs on a graph that has genuinely changed.

    Maps to `None` when a new cluster shares zero members with every old
    cluster — a genuinely new community, not an aligned one. Not
    symmetric/injective by design: two new clusters could both best-align to
    the same old cluster if one old community legitimately split in two: this
    function reports the best match per new cluster, it doesn't resolve
    conflicts between competing new clusters.
    """
    alignment: dict[int, int | None] = {}
    for new_index, new_set in enumerate(new_clusters):
        best_old_index: int | None = None
        best_similarity = 0.0
        for old_index, old_set in enumerate(old_clusters):
            similarity = jaccard_similarity(new_set, old_set)
            if similarity > best_similarity:
                best_old_index, best_similarity = old_index, similarity
        alignment[new_index] = best_old_index
    return alignment
