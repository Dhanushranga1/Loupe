"""MMR (Maximal Marginal Relevance) diversity-aware final selection
(docs/PhaseX/loupe-retrieval-upgrades.md §4).

Runs after cross-encoder reranking (§1's pipeline stage 4), operating on the
already-scored candidate set — trades a little relevance for diversity so
the final result set isn't five near-identical getters crowding out
genuinely distinct relevant symbols. `O(n^2)` in the candidate count, trivial
at `n <= 20` (RRF's own narrowing already bounds the input this small).
"""

from __future__ import annotations

import math

DEFAULT_LAMBDA = 0.7


def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def mmr_select(
    candidates: list[tuple[str, float]],
    embeddings: dict[str, list[float]],
    final_top_k: int,
    lambda_param: float = DEFAULT_LAMBDA,
) -> list[tuple[str, float]]:
    """Iteratively picks `final_top_k` candidates from `candidates` (symbol_id,
    relevance_score pairs — relevance is whatever upstream stage scored them,
    cross-encoder output in the real pipeline), trading relevance against
    redundancy with what's already been selected.

    `lambda_param=1.0` (diversity penalty fully off) must reduce to a plain
    top-k-by-relevance sort — a strict generalization of the original
    behavior, not a replacement that changes results when diversity isn't
    wanted (§4's own backward-compatibility acceptance criterion). Ties are
    broken by symbol_id, matching every other ranking function in this
    codebase (fuse(), _cap_symbols_by_pagerank, etc.) for deterministic output
    regardless of dict/set iteration order.

    A candidate id missing from `embeddings` is treated as having zero
    redundancy with anything already selected — defensive only; every
    candidate the real pipeline passes in was embedded at index time
    (`SemanticIndex.index()` embeds every symbol, not just query hits).
    """
    selected: list[tuple[str, float]] = []
    remaining = list(candidates)

    while remaining and len(selected) < final_top_k:
        best_index = None
        best_key: tuple[float, str] | None = None
        for i, (symbol_id, relevance) in enumerate(remaining):
            candidate_embedding = embeddings.get(symbol_id)
            if candidate_embedding is None or not selected:
                redundancy = 0.0
            else:
                redundancy = max(
                    cosine_similarity(candidate_embedding, embeddings[sid])
                    if embeddings.get(sid) is not None
                    else 0.0
                    for sid, _ in selected
                )
            score = lambda_param * relevance - (1 - lambda_param) * redundancy
            key = (-score, symbol_id)
            if best_key is None or key < best_key:
                best_index, best_key = i, key

        selected.append(remaining.pop(best_index))

    return selected
