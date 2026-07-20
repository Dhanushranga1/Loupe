"""Reciprocal Rank Fusion over lexical + semantic + centrality (+ optional churn)
(docs/phase-2-retrieval.md §6, docs/PhaseX/loupe-retrieval-upgrades.md §2 for
the personalized-centrality variant, docs/PhaseX/phase-14-adaptive-context-compression.md
§2 for the churn signal).

Centrality is folded in as a third "ranking" even though it isn't
query-dependent — a deliberate, documented extension of textbook RRF (which
fuses multiple rankings *of the same query*): treating repo-wide PageRank
order as a fixed, always-available ranking gives structurally important
symbols a consistent, mild boost across every query, without a hand-tuned
weight term. Churn (optional, a fourth ranking, same "fixed, always-available"
shape as centrality) applies the identical reasoning to *temporal* relevance
instead of structural importance.
"""

from __future__ import annotations

import networkx as nx

from loupe_core.graph.centrality import compute_personalized_pagerank
from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.semantic import SemanticIndex

RRF_K = 60
CANDIDATE_POOL_SIZE = 50
FINAL_TOP_K = 20


def fuse(
    lexical_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    pagerank_scores: dict[str, float],
    graph: nx.DiGraph | None = None,
    top_k: int = FINAL_TOP_K,
    scope_seed_ids: set[str] | None = None,
    in_scope_mass: float = 1.0,
    churn_scores: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """RRF-fuse already-ranked lexical/semantic candidate lists, folding in centrality.

    `lexical_results`/`semantic_results` are (symbol_id, signal_score) pairs,
    already sorted best-first and truncated to their own candidate pool.
    Only symbols appearing in at least one of the two pools are eligible —
    centrality never introduces a candidate neither text-based signal found,
    it only re-orders among candidates one of them already surfaced.

    `graph` is optional: when given, the centrality term is *personalized*
    PageRank (retrieval-upgrades §2) seeded from this call's own candidate
    pool — a query-aware signal, replacing the static, query-blind one.
    `graph=None` keeps the original static-`pagerank_scores` behavior, which
    real callers should only choose deliberately (e.g. an ablation
    comparison against the personalized path), not by omission.

    `scope_seed_ids`/`in_scope_mass` (Phase 11's scope-aware-retrieval §2,
    soft-boundary mode): when `scope_seed_ids` is given, personalization is
    seeded from *scope membership* instead of the query's own candidate pool,
    with `in_scope_mass` controlling how much restart probability mass stays
    inside scope vs. spreads to the rest of the bounded subgraph — scores are
    still returned for this call's actual candidates, decoupling "what biases
    the walk" from "what gets ranked," exactly what
    `compute_personalized_pagerank`'s `score_ids` parameter exists for.
    Defaults preserve Phase 9's original self-referential behavior exactly.

    `churn_scores` (Phase 14 §2, optional, `None` by default — omitting it
    keeps every existing caller's behavior exactly unchanged): a fourth,
    fixed "ranking," folded in the same way centrality is (§6's own
    extension of textbook RRF) — a recency-decayed measure of how recently
    and frequently each candidate has actually been edited
    (`retrieval/churn.py`'s `compute_churn_scores`), giving temporally
    active symbols a consistent, mild boost alongside centrality's
    structural one.
    """
    lexical_rank = {symbol_id: i + 1 for i, (symbol_id, _) in enumerate(lexical_results)}
    semantic_rank = {symbol_id: i + 1 for i, (symbol_id, _) in enumerate(semantic_results)}
    candidates = set(lexical_rank) | set(semantic_rank)

    if graph is not None:
        personalization_seeds = scope_seed_ids if scope_seed_ids is not None else candidates
        centrality_scores = compute_personalized_pagerank(
            graph, personalization_seeds, pagerank_scores, in_scope_mass=in_scope_mass, score_ids=candidates
        )
    else:
        centrality_scores = pagerank_scores

    # Centrality ranking is local to this query's candidate set, not the whole repo (§6).
    # `candidates` is a set, whose iteration order depends on Python's per-process
    # string hash randomization — sorting on score alone would let equal-score ties
    # break differently across runs. `sid` as a secondary key makes the output
    # fully deterministic regardless of set iteration order.
    centrality_sorted = sorted(candidates, key=lambda sid: (-centrality_scores.get(sid, 0.0), sid))
    centrality_rank = {symbol_id: i + 1 for i, symbol_id in enumerate(centrality_sorted)}

    rank_maps = [lexical_rank, semantic_rank, centrality_rank]
    if churn_scores is not None:
        churn_sorted = sorted(candidates, key=lambda sid: (-churn_scores.get(sid, 0.0), sid))
        rank_maps.append({symbol_id: i + 1 for i, symbol_id in enumerate(churn_sorted)})

    scores: dict[str, float] = {}
    for symbol_id in candidates:
        total = 0.0
        for rank_map in rank_maps:
            rank = rank_map.get(symbol_id)
            if rank is not None:
                total += 1.0 / (RRF_K + rank)
        scores[symbol_id] = total

    ranked = sorted(candidates, key=lambda sid: (-scores[sid], sid))
    return [(symbol_id, scores[symbol_id]) for symbol_id in ranked[:top_k]]


def search(
    query: str,
    lexical_index: LexicalIndex,
    semantic_index: SemanticIndex,
    pagerank_scores: dict[str, float],
    graph: nx.DiGraph | None = None,
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    top_k: int = FINAL_TOP_K,
    churn_scores: dict[str, float] | None = None,
) -> list[tuple[str, float]]:
    """Convenience wrapper: run both signals for `query`, then fuse."""
    lexical_results = lexical_index.query(query, top_k=candidate_pool_size)
    semantic_results = semantic_index.query(query, top_k=candidate_pool_size)
    return fuse(lexical_results, semantic_results, pagerank_scores, graph=graph, top_k=top_k, churn_scores=churn_scores)
