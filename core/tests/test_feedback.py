"""Tests for eval/feedback.py (docs/loupe-extensions.md E3 — Human Feedback Loop)."""

from loupe_core.eval.backfill import FileChangeEvent, RetrievalEvent
from loupe_core.eval.feedback import FeedbackEntry, backfill_all_with_feedback, resolve_training_label
from loupe_core.parsing.extractor import extract_symbols

CONTENT_V0 = b"def f():\n    return 1\n\n\ndef g():\n    return 2\n"
CONTENT_V1_F_CHANGED = b"def f():\n    return 999\n\n\ndef g():\n    return 2\n"


def _symbols_from(tmp_path, content: bytes, name: str = "a.py"):
    path = tmp_path / name
    path.write_bytes(content)
    return {s.name: s for s in extract_symbols(str(path))}


def test_feedback_wins_when_it_disagrees_with_the_proxy_signal():
    """The exact disagreement case the spec calls out: feedback says not_helpful,
    proxy says symbol_edited=True — feedback must win, not average, not proxy."""
    feedback = {"log-1": FeedbackEntry("log-1", rating="not_helpful", note=None, submitted_at=100.0, source="dashboard")}

    label = resolve_training_label("log-1", proxy_outcome=True, feedback_by_log_id=feedback)

    assert label is False


def test_feedback_wins_even_when_it_agrees_with_the_proxy_signal():
    feedback = {"log-1": FeedbackEntry("log-1", rating="helpful", note=None, submitted_at=100.0, source="dashboard")}

    label = resolve_training_label("log-1", proxy_outcome=True, feedback_by_log_id=feedback)

    assert label is True


def test_proxy_used_when_no_feedback_exists():
    label = resolve_training_label("log-1", proxy_outcome=True, feedback_by_log_id={})
    assert label is True

    label_false = resolve_training_label("log-2", proxy_outcome=False, feedback_by_log_id={})
    assert label_false is False


def test_no_feedback_and_no_proxy_signal_returns_none_not_a_negative_label():
    label = resolve_training_label("log-1", proxy_outcome=None, feedback_by_log_id={})
    assert label is None


def test_backfill_all_with_feedback_prefers_feedback_over_a_disagreeing_proxy_end_to_end(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )
    # proxy would say True: f's own range was edited within the window
    file_changes = [FileChangeEvent(file_path=symbols["f"].file_path, changed_at=1000.0 + 600, new_content=CONTENT_V1_F_CHANGED)]
    # but a human explicitly marked this retrieval not helpful
    feedback = {"log-1": FeedbackEntry("log-1", rating="not_helpful", note="wrong symbol", submitted_at=1001.0, source="dashboard")}

    examples = backfill_all_with_feedback([event], file_changes, feedback)

    assert len(examples) == 1
    assert examples[0].symbol_edited is False


def test_backfill_all_with_feedback_falls_back_to_proxy_when_no_feedback_for_that_log(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )
    file_changes = [FileChangeEvent(file_path=symbols["f"].file_path, changed_at=1000.0 + 600, new_content=CONTENT_V1_F_CHANGED)]

    examples = backfill_all_with_feedback([event], file_changes, feedback_by_log_id={})

    assert len(examples) == 1
    assert examples[0].symbol_edited is True


def test_backfill_all_with_feedback_excludes_events_with_no_signal_at_all(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )

    examples = backfill_all_with_feedback([event], file_changes=[], feedback_by_log_id={})

    assert examples == []
