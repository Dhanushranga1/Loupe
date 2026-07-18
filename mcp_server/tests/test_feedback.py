"""Tests for app/feedback.py's FeedbackStore (docs/loupe-extensions.md E3)."""

from app.feedback import FeedbackStore


def test_submit_writes_a_correctly_linked_entry(tmp_path):
    store = FeedbackStore(tmp_path / "logs" / "feedback")

    entry = store.submit("log-1", "helpful", "good match", source="dashboard")

    assert entry.retrieval_log_id == "log-1"
    assert entry.rating == "helpful"
    assert entry.note == "good match"
    assert entry.source == "dashboard"
    assert entry.submitted_at > 0


def test_all_by_log_id_reads_back_what_was_submitted(tmp_path):
    store = FeedbackStore(tmp_path / "logs" / "feedback")
    store.submit("log-1", "helpful", None, source="dashboard")
    store.submit("log-2", "not_helpful", "wrong symbol", source="claude_self_report")

    entries = store.all_by_log_id()

    assert set(entries) == {"log-1", "log-2"}
    assert entries["log-1"].rating == "helpful"
    assert entries["log-2"].rating == "not_helpful"
    assert entries["log-2"].source == "claude_self_report"


def test_a_later_submission_for_the_same_log_id_overwrites_the_earlier_one(tmp_path):
    store = FeedbackStore(tmp_path / "logs" / "feedback")
    store.submit("log-1", "not_helpful", "first guess", source="dashboard")
    store.submit("log-1", "helpful", "actually fine", source="dashboard")

    entries = store.all_by_log_id()

    assert len(entries) == 1
    assert entries["log-1"].rating == "helpful"
    assert entries["log-1"].note == "actually fine"


def test_all_by_log_id_on_a_fresh_store_returns_empty(tmp_path):
    store = FeedbackStore(tmp_path / "logs" / "feedback")
    assert store.all_by_log_id() == {}
