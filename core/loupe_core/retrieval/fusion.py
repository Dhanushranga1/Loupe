"""Reciprocal Rank Fusion over lexical + semantic + centrality (docs/phase-2-retrieval.md §6).

Centrality is folded in as a third "ranking" even though it isn't
query-dependent — a deliberate, documented extension of textbook RRF (which
fuses multiple rankings *of the same query*): treating repo-wide PageRank
order as a fixed, always-available ranking gives structurally important
symbols a consistent, mild boost across every query, without a hand-tuned
weight term.
"""

from __future__ import annotations

from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.semantic import SemanticIndex

RRF_K = 60
CANDIDATE_POOL_SIZE = 50
FINAL_TOP_K = 20


def fuse(
    lexical_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    pagerank_scores: dict[str, float],
    top_k: int = FINAL_TOP_K,
) -> list[tuple[str, float]]:
    """RRF-fuse already-ranked lexical/semantic candidate lists, folding in centrality.

    `lexical_results`/`semantic_results` are (symbol_id, signal_score) pairs,
    already sorted best-first and truncated to their own candidate pool.
    Only symbols appearing in at least one of the two pools are eligible —
    centrality never introduces a candidate neither text-based signal found,
    it only re-orders among candidates one of them already surfaced.
    """
    lexical_rank = {symbol_id: i + 1 for i, (symbol_id, _) in enumerate(lexical_results)}
    semantic_rank = {symbol_id: i + 1 for i, (symbol_id, _) in enumerate(semantic_results)}
    candidates = set(lexical_rank) | set(semantic_rank)

    # Centrality ranking is local to this query's candidate set, not the whole repo (§6).
    # `candidates` is a set, whose iteration order depends on Python's per-process
    # string hash randomization — sorting on score alone would let equal-score ties
    # break differently across runs. `sid` as a secondary key makes the output
    # fully deterministic regardless of set iteration order.
    centrality_sorted = sorted(candidates, key=lambda sid: (-pagerank_scores.get(sid, 0.0), sid))
    centrality_rank = {symbol_id: i + 1 for i, symbol_id in enumerate(centrality_sorted)}

    scores: dict[str, float] = {}
    for symbol_id in candidates:
        total = 0.0
        for rank_map in (lexical_rank, semantic_rank, centrality_rank):
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
    candidate_pool_size: int = CANDIDATE_POOL_SIZE,
    top_k: int = FINAL_TOP_K,
) -> list[tuple[str, float]]:
    """Convenience wrapper: run both signals for `query`, then fuse."""
    lexical_results = lexical_index.query(query, top_k=candidate_pool_size)
    semantic_results = semantic_index.query(query, top_k=candidate_pool_size)
    return fuse(lexical_results, semantic_results, pagerank_scores, top_k=top_k)
