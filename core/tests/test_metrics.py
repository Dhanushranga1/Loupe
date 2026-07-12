"""Tests for eval/metrics.py (docs/phase-5-evaluation.md §8 — Metrics)."""

from loupe_core.eval.metrics import chunk_containment, recall_at_k, token_cost


def test_recall_at_k_returns_none_for_empty_ground_truth():
    assert recall_at_k(["a", "b"], set(), k=5) is None


def test_recall_at_k_excluded_task_does_not_drag_down_aggregate_mean():
    # Constructing exactly the case the spec warns about: an empty-ground-truth
    # task must not silently count as a 0.0 in an aggregate.
    values = [recall_at_k(["a"], {"a"}, k=5), recall_at_k([], set(), k=5)]
    scored = [v for v in values if v is not None]
    assert scored == [1.0]
    assert sum(scored) / len(scored) == 1.0


def test_recall_at_k_partial_and_full_match():
    assert recall_at_k(["a", "b", "c"], {"a", "b"}, k=5) == 1.0
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=5) == 0.5
    assert recall_at_k(["x", "y"], {"a", "b"}, k=5) == 0.0


def test_recall_at_k_respects_k_cutoff():
    # ground truth 'b' is only reachable within top 3, not top 1
    assert recall_at_k(["x", "y", "b"], {"b"}, k=1) == 0.0
    assert recall_at_k(["x", "y", "b"], {"b"}, k=3) == 1.0


def test_token_cost_matches_estimate_tokens_on_concatenated_content():
    from loupe_core.governor.budget import estimate_tokens

    content = ["def f():\n    return 1", "def g():\n    return 2"]
    assert token_cost(content) == estimate_tokens("\n".join(content))


def test_chunk_containment_full_overlap():
    assert chunk_containment((1, 20), (5, 10)) == 1.0


def test_chunk_containment_no_overlap():
    assert chunk_containment((1, 4), (5, 10)) == 0.0
    assert chunk_containment((11, 20), (5, 10)) == 0.0


def test_chunk_containment_partial_overlap():
    # symbol spans lines 5-10 (6 lines); chunk only covers 5-7 (3 of those 6 lines)
    assert chunk_containment((5, 7), (5, 10)) == 3 / 6


def test_chunk_containment_chunk_fully_inside_symbol():
    # symbol spans 1-100; chunk covers only 40-60 (21 lines of the 100)
    assert chunk_containment((40, 60), (1, 100)) == 21 / 100
