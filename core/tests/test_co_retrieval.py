"""Tests for context/co_retrieval.py (docs/PhaseX/phase-14-adaptive-context-compression.md §3)."""

import json
from pathlib import Path

from loupe_core.context.co_retrieval import (
    MIN_SUPPORT,
    mine_co_retrieval_suggestions,
    sessions_from_retrieval_logs,
)


# --------------------------------------------------------------------------
# §6 acceptance criterion: A/B co-requested in the large majority of
# sessions, C requested independently — B surfaces for A, C does not.
# --------------------------------------------------------------------------


def test_co_requested_pair_surfaces_the_other_independently_requested_symbol_does_not():
    sessions = (
        [{"A", "B"} for _ in range(6)]  # A and B co-requested most of the time
        + [{"A"} for _ in range(2)]  # A requested alone sometimes too
        + [{"C"} for _ in range(3)]  # C is requested independently, never with A
    )

    suggestions = mine_co_retrieval_suggestions(sessions)
    suggested_ids = {s.symbol_id for s in suggestions["A"]}

    assert "B" in suggested_ids
    assert "C" not in suggested_ids


def test_confidence_is_the_standard_market_basket_ratio():
    sessions = [{"A", "B"} for _ in range(6)] + [{"A"} for _ in range(2)]

    suggestions = mine_co_retrieval_suggestions(sessions)
    b_suggestion = next(s for s in suggestions["A"] if s.symbol_id == "B")

    assert b_suggestion.confidence == 6 / 8  # count(A,B)/count(A alone)
    assert b_suggestion.support == 6


def test_confidence_is_directional_not_symmetric():
    """A is requested far more often than B overall, so B->A confidence
    should be higher than A->B confidence even for the identical pair."""
    sessions = [{"A", "B"} for _ in range(5)] + [{"A"} for _ in range(15)]

    suggestions = mine_co_retrieval_suggestions(sessions)
    a_to_b = next(s for s in suggestions["A"] if s.symbol_id == "B").confidence
    b_to_a = next(s for s in suggestions["B"] if s.symbol_id == "A").confidence

    assert a_to_b == 5 / 20
    assert b_to_a == 5 / 5
    assert b_to_a > a_to_b


# --------------------------------------------------------------------------
# §6 acceptance criterion: minimum-support boundary, exactly at the edge
# --------------------------------------------------------------------------


def test_pair_below_minimum_support_is_excluded():
    assert MIN_SUPPORT == 5
    sessions = [{"A", "B"} for _ in range(MIN_SUPPORT - 1)]

    suggestions = mine_co_retrieval_suggestions(sessions)

    assert "B" not in {s.symbol_id for s in suggestions.get("A", [])}


def test_pair_at_exactly_minimum_support_is_included():
    sessions = [{"A", "B"} for _ in range(MIN_SUPPORT)]

    suggestions = mine_co_retrieval_suggestions(sessions)

    assert "B" in {s.symbol_id for s in suggestions["A"]}


def test_suggestions_sorted_by_confidence_descending():
    sessions = (
        [{"A", "B"} for _ in range(10)]  # A->B confidence 10/10 = 1.0
        + [{"A", "C"} for _ in range(5)]  # A->C confidence 5/15 = 0.33
    )

    suggestions = mine_co_retrieval_suggestions(sessions)
    ordered_ids = [s.symbol_id for s in suggestions["A"]]

    assert ordered_ids == ["B", "C"]


def test_no_sessions_produces_no_suggestions():
    assert mine_co_retrieval_suggestions([]) == {}


# --------------------------------------------------------------------------
# sessions_from_retrieval_logs — real RetrievalLog JSONL parsing
# --------------------------------------------------------------------------


def _write_log(path: Path, entries: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def test_sessions_from_retrieval_logs_extracts_get_symbol_requests_only(tmp_path):
    _write_log(
        tmp_path / "session-1.jsonl",
        [
            {"tool_name": "search_symbols", "query_text": "some query"},
            {"tool_name": "get_symbol", "query_text": "symbol-a"},
            {"tool_name": "get_symbol", "query_text": "symbol-b"},
        ],
    )
    _write_log(
        tmp_path / "session-2.jsonl",
        [{"tool_name": "get_symbol", "query_text": "symbol-c"}],
    )

    sessions = sessions_from_retrieval_logs(tmp_path)

    assert {"symbol-a", "symbol-b"} in sessions
    assert {"symbol-c"} in sessions
    assert len(sessions) == 2


def test_sessions_from_retrieval_logs_skips_sessions_with_no_get_symbol_calls(tmp_path):
    _write_log(tmp_path / "session-1.jsonl", [{"tool_name": "search_symbols", "query_text": "q"}])

    sessions = sessions_from_retrieval_logs(tmp_path)

    assert sessions == []


def test_sessions_from_retrieval_logs_empty_dir_returns_empty_list(tmp_path):
    assert sessions_from_retrieval_logs(tmp_path) == []
