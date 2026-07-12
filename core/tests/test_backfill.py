"""Tests for eval/backfill.py (docs/phase-6-closing-the-loop.md §8 — Outcome backfill)."""

from loupe_core.eval.backfill import BACKFILL_WINDOW_SECONDS, FileChangeEvent, RetrievalEvent, backfill_all, backfill_outcome
from loupe_core.parsing.extractor import extract_symbols

CONTENT_V0 = b"def f():\n    return 1\n\n\ndef g():\n    return 2\n"
CONTENT_V1_F_CHANGED = b"def f():\n    return 999\n\n\ndef g():\n    return 2\n"
CONTENT_V2_G_CHANGED = b"def f():\n    return 1\n\n\ndef g():\n    return 888\n"


def _symbols_from(tmp_path, content: bytes, name: str = "a.py"):
    path = tmp_path / name
    path.write_bytes(content)
    return {s.name: s for s in extract_symbols(str(path))}


def test_edit_overlapping_symbol_within_window_backfills_true(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )
    changes = [FileChangeEvent(file_path=symbols["f"].file_path, changed_at=1000.0 + 600, new_content=CONTENT_V1_F_CHANGED)]

    assert backfill_outcome(event, changes) is True


def test_edit_not_overlapping_symbol_backfills_false(tmp_path):
    """The file changed, but not the specific retrieved symbol's range — must be False, not True."""
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )
    # g() changed, not f() — f is what was retrieved.
    changes = [FileChangeEvent(file_path=symbols["f"].file_path, changed_at=1000.0 + 600, new_content=CONTENT_V2_G_CHANGED)]

    assert backfill_outcome(event, changes) is False


def test_no_file_activity_within_window_leaves_outcome_none(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )
    assert backfill_outcome(event, file_changes=[]) is None


def test_change_outside_window_does_not_count(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )
    too_late = 1000.0 + BACKFILL_WINDOW_SECONDS + 1
    changes = [FileChangeEvent(file_path=symbols["f"].file_path, changed_at=too_late, new_content=CONTENT_V1_F_CHANGED)]

    assert backfill_outcome(event, changes) is None


def test_change_to_a_different_file_does_not_count(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    event = RetrievalEvent(
        log_id="log-1", session_id="sess-1", symbol=symbols["f"], content_at_retrieval=CONTENT_V0, retrieved_at=1000.0
    )
    changes = [FileChangeEvent(file_path="other_file.py", changed_at=1000.0 + 60, new_content=CONTENT_V1_F_CHANGED)]

    assert backfill_outcome(event, changes) is None


def test_backfill_all_excludes_unresolved_entries_not_as_negative_examples(tmp_path):
    symbols = _symbols_from(tmp_path, CONTENT_V0)
    resolved_true = RetrievalEvent("log-1", "sess-1", symbols["f"], CONTENT_V0, retrieved_at=1000.0)
    resolved_false = RetrievalEvent("log-2", "sess-1", symbols["g"], CONTENT_V0, retrieved_at=2000.0)
    unresolved = RetrievalEvent("log-3", "sess-1", symbols["f"], CONTENT_V0, retrieved_at=9999.0)

    changes = [
        FileChangeEvent(symbols["f"].file_path, 1000.0 + 60, CONTENT_V1_F_CHANGED),  # touches f, resolves log-1 True
        FileChangeEvent(symbols["g"].file_path, 2000.0 + 60, CONTENT_V1_F_CHANGED),  # touches f not g, resolves log-2 False
        # nothing resolves log-3 (unresolved) within its window
    ]

    examples = backfill_all([resolved_true, resolved_false, unresolved], changes)

    by_log_id = {e.log_id: e.symbol_edited for e in examples}
    assert by_log_id == {"log-1": True, "log-2": False}
    assert "log-3" not in by_log_id, "an unresolved outcome must be excluded, not defaulted to a negative label"
