"""Cross-encoder reranking of RRF's top-20 (docs/PhaseX/loupe-retrieval-upgrades.md §3).

Runs after `retrieval/fusion.py`'s RRF stage, on RRF's already-narrowed
candidate set only — a cross-encoder jointly encodes (query, candidate)
together, far more accurate than comparing independent embeddings, but too
slow to run against every candidate in a repo. Scores REPLACE RRF's score
for these candidates rather than blending with it (§3, a decided choice:
once you're comparing only within an already-narrowed set, a direct joint
relevance judgment is strictly better information than a fused proxy).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from loupe_core.parsing.schema import Symbol
from loupe_core.retrieval.lexical import symbol_document_text

CROSS_ENCODER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def get_default_cross_encoder() -> CrossEncoder:
    """Lazily load the real model once per process — same discipline as
    `retrieval/semantic.py`'s `get_default_model` for the embedding model."""
    global _model
    if _model is None:
        _model = CrossEncoder(CROSS_ENCODER_MODEL_NAME)
    return _model


@dataclass
class RerankResult:
    ranked: list[tuple[str, float]]
    latency_ms: float


def rerank(
    query: str,
    candidates: list[tuple[str, float]],
    symbols_by_id: dict[str, Symbol],
    cross_encoder: CrossEncoder | None = None,
) -> RerankResult:
    """Re-score `candidates` (RRF's top-20, (symbol_id, rrf_score) pairs) by jointly
    encoding (query, candidate_discovery_text) pairs in one batched `predict` call
    (§3 step 2 — one call for the whole batch, not one per candidate, matching the
    batching discipline already used for embedding calls in `retrieval/semantic.py`).

    "Discovery text" here is `lexical.py`'s `symbol_document_text` (name +
    qualified_name + signature + docstring + decorators), not the governor's
    similarly-named `_discovery_text` (signature + first docstring line only,
    a budget-estimation concept, not a retrieval one) — checked empirically,
    not assumed: the governor's narrower text drops the qualified name, which
    silently broke the "OrderService.log" adversarial case (Phase 2's exact
    dotted-name query) that this function's own acceptance criterion requires
    recovering into the top 2.

    A candidate id absent from `symbols_by_id` is dropped rather than raising —
    defensive only; every id `fuse()` produces has a backing symbol by construction.
    `latency_ms` times only the `predict()` call itself, the number the spec's own
    acceptance criterion asks to be measured and logged, not pair-building overhead.
    """
    model = cross_encoder or get_default_cross_encoder()
    pairs: list[tuple[str, str]] = []
    ids: list[str] = []
    for symbol_id, _score in candidates:
        symbol = symbols_by_id.get(symbol_id)
        if symbol is None:
            continue
        pairs.append((query, symbol_document_text(symbol)))
        ids.append(symbol_id)

    if not pairs:
        return RerankResult(ranked=[], latency_ms=0.0)

    start = time.perf_counter()
    scores = model.predict(pairs)
    latency_ms = (time.perf_counter() - start) * 1000

    ranked = sorted(zip(ids, (float(s) for s in scores)), key=lambda pair: (-pair[1], pair[0]))
    return RerankResult(ranked=ranked, latency_ms=latency_ms)
