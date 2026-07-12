"""Evaluation metrics (docs/phase-5-evaluation.md §4)."""

from __future__ import annotations

from loupe_core.governor.budget import estimate_tokens


def recall_at_k(retrieved_ids: list[str], ground_truth_ids: set[str], k: int) -> float | None:
    """|top_k(retrieved) ∩ ground_truth| / |ground_truth|.

    Returns None (not 0.0) if `ground_truth_ids` is empty — such a task is
    excluded from aggregation, not scored as a failure that would silently
    drag down an aggregate mean.
    """
    if not ground_truth_ids:
        return None
    top_k = set(retrieved_ids[:k])
    return len(top_k & ground_truth_ids) / len(ground_truth_ids)


def token_cost(retrieved_content: list[str]) -> int:
    """Token cost of the concatenated retrieved content, via Phase 3's `estimate_tokens`."""
    return estimate_tokens("\n".join(retrieved_content))


def chunk_containment(chunk_line_range: tuple[int, int], symbol_line_range: tuple[int, int]) -> float:
    """Fraction of the symbol's line range actually contained within the chunk (vector-RAG baseline only).

    1.0 = fully captured, 0.0 = no overlap; partial values expose the real
    cost of arbitrary chunk boundaries splitting a symbol — reported
    alongside recall, not hidden by crediting a chunked hit as complete
    regardless of how much of the symbol it actually contains. Both ranges
    are 1-indexed, inclusive (matching `Symbol.line_start`/`line_end`).
    """
    chunk_start, chunk_end = chunk_line_range
    symbol_start, symbol_end = symbol_line_range
    symbol_length = symbol_end - symbol_start + 1
    if symbol_length <= 0:
        return 0.0

    overlap_start = max(chunk_start, symbol_start)
    overlap_end = min(chunk_end, symbol_end)
    overlap_length = max(0, overlap_end - overlap_start + 1)

    return overlap_length / symbol_length
