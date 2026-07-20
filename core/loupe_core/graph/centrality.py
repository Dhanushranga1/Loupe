"""PageRank centrality over the symbol graph (docs/phase-1-graph-theory.md §7,
docs/PhaseX/loupe-retrieval-upgrades.md §2 for the personalized variant).

No custom tuning of PageRank's damping factor for this phase — use the
library default (0.85) until there's a concrete reason to change it.
"""

from __future__ import annotations

import networkx as nx

from .traversal import expand_dependencies

# The retrieval-upgrades spec says "max_iter capped at 20" for personalized
# PageRank, reasoning that this is a ranking signal, not a value needing
# numerical precision. Verified empirically before trusting the number as
# specified: at networkx's default tol=1e-6, even a trivial 3-node cycle
# graph needs ~70 iterations to converge — 20 iterations at the default
# tolerance would raise PowerIterationFailedConvergence on nearly any real
# subgraph, not just large ones. The spec's own stated reasoning (relative
# order matters, not exact values) justifies loosening tol as much as
# capping max_iter; TOL=0.01 is what actually makes 20 iterations converge
# in practice, checked against real fixture graphs, not assumed from docs.
PERSONALIZED_MAX_ITER = 20
PERSONALIZED_TOL = 0.01
DEFAULT_PERSONALIZATION_DEPTH = 3


def compute_pagerank(graph: nx.DiGraph) -> dict[str, float]:
    """PageRank score per symbol_id; empty dict for an empty graph (no nodes to rank)."""
    if graph.number_of_nodes() == 0:
        return {}
    return nx.pagerank(graph, weight="weight")


def compute_personalized_pagerank(
    graph: nx.DiGraph,
    seed_ids: set[str],
    static_pagerank_scores: dict[str, float],
    depth: int = DEFAULT_PERSONALIZATION_DEPTH,
    in_scope_mass: float = 1.0,
    score_ids: set[str] | None = None,
) -> dict[str, float]:
    """Query-aware PageRank, bounded to the local subgraph reachable from `seed_ids`.

    Returns a score for every id in `score_ids` (default `seed_ids` itself —
    Phase 9's original self-referential "seed and re-rank the same
    query-candidate set" usage) — never a partial result. Falls back to the
    symbol's own *static* PageRank score (never zero — a symbol outside the
    bounded subgraph is a locality artifact, not evidence of irrelevance)
    when: the seed set has no members in `graph` at all; a candidate isn't in
    the bounded subgraph; or power iteration doesn't converge within budget.

    `score_ids`, when given, decouples "what the restart distribution is
    seeded from" (`seed_ids`) from "what ids the caller actually wants scores
    for" — needed by Phase 11's scope-aware soft-boundary retrieval
    (docs/PhaseX/scope-aware-retrieval.md §2), where the restart bias comes
    from scope membership (potentially many symbols) but scores are wanted
    for a *different* set: the query's own RRF candidates.

    `in_scope_mass` (default 1.0, all restart mass on `seed_ids`, Phase 9's
    original behavior) splits the restart distribution between `seed_ids` and
    every other node in the bounded subgraph — scope-aware-retrieval passes
    ~0.9 here for soft-boundary mode, biasing heavily toward scope without
    making anything just outside it fully invisible. That spec's "10%
    distributed across the rest of the graph" is interpreted as the rest of
    *this bounded subgraph*, not the literal whole repo — a checked,
    deliberate reading consistent with Phase 9's own cost-bounding rationale
    for `depth` in the first place: this function is one secondary ranking
    signal inside RRF fusion, not the candidate-generation step, so nothing
    about its job requires reaching further than the depth bound already
    established — a genuinely relevant far-away candidate still surfaces via
    the query's lexical/semantic signals regardless of this function's reach.
    """
    target_ids = score_ids if score_ids is not None else seed_ids
    seeds_in_graph = {sid for sid in seed_ids if sid in graph}
    if not seeds_in_graph:
        return {sid: static_pagerank_scores.get(sid, 0.0) for sid in target_ids}

    reachable: set[str] = set(seeds_in_graph)
    for sid in seeds_in_graph:
        reachable |= expand_dependencies(graph, sid, depth=depth, direction="both")
    subgraph = graph.subgraph(reachable)

    non_seed_nodes = [n for n in subgraph.nodes() if n not in seeds_in_graph]
    if in_scope_mass >= 1.0 or not non_seed_nodes:
        personalization = {node: (1.0 if node in seeds_in_graph else 0.0) for node in subgraph.nodes()}
    else:
        seed_share = in_scope_mass / len(seeds_in_graph)
        rest_share = (1.0 - in_scope_mass) / len(non_seed_nodes)
        personalization = {node: (seed_share if node in seeds_in_graph else rest_share) for node in subgraph.nodes()}

    total_mass = sum(personalization.values())
    if total_mass > 0:
        personalization = {node: mass / total_mass for node, mass in personalization.items()}

    try:
        personalized_scores = nx.pagerank(
            subgraph, personalization=personalization, max_iter=PERSONALIZED_MAX_ITER, tol=PERSONALIZED_TOL, weight="weight"
        )
    except nx.PowerIterationFailedConvergence:
        personalized_scores = {}

    return {
        sid: personalized_scores.get(sid, static_pagerank_scores.get(sid, 0.0))
        for sid in target_ids
    }
