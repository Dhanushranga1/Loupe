"""Outcome backfill: turn Phase 4's deferred `RetrievalLog.outcome` into a real,
honestly-labeled weak-supervision signal (docs/phase-6-closing-the-loop.md §3).

The proxy: was the retrieved symbol's own source code subsequently edited,
within a bounded time window, in the same session. Real and observable, but
imperfect (a symbol can be useful without being edited, or edited for
unrelated reasons) — accepted as noise, not solved. The ranker (§4) treats
this as weak supervision, not a clean label.

Deliberately framework-free, matching `core`'s design boundary: this module
operates on plain `RetrievalEvent`/`FileChangeEvent` records, not on Phase 4
server's actual `RetrievalLog` JSONL format or `IndexerWorker` directly — the
server-side job that reads real telemetry and calls `backfill_outcome`
periodically is a thin integration layer, not core logic.
"""

from __future__ import annotations

from dataclasses import dataclass

from loupe_core.eval.mine_history import diff_hunks_between_contents, hunk_overlaps_symbol
from loupe_core.parsing.schema import Symbol

BACKFILL_WINDOW_SECONDS = 30 * 60  # 30 minutes (§3)


@dataclass
class RetrievalEvent:
    """Enough of a `get_symbol` RetrievalLog entry to backfill its outcome."""

    log_id: str
    session_id: str
    symbol: Symbol
    content_at_retrieval: bytes  # full file content at the moment of retrieval
    retrieved_at: float  # unix timestamp


@dataclass
class FileChangeEvent:
    """A settled file change, as the indexer worker would observe it."""

    file_path: str
    changed_at: float
    new_content: bytes


def backfill_outcome(event: RetrievalEvent, file_changes: list[FileChangeEvent]) -> bool | None:
    """True/False if resolved within the window, None if still unresolved (§3 steps 1-3).

    None is not a negative label — it means "no signal yet," and must be
    excluded from training data, never treated as `symbol_edited: False`.
    """
    window_end = event.retrieved_at + BACKFILL_WINDOW_SECONDS
    relevant_changes = [
        c
        for c in file_changes
        if c.file_path == event.symbol.file_path and event.retrieved_at <= c.changed_at <= window_end
    ]
    if not relevant_changes:
        return None

    # Multiple edits within the window are evaluated as one outcome check
    # against the latest state, not one outcome per intermediate edit.
    latest_change = max(relevant_changes, key=lambda c: c.changed_at)
    hunks = diff_hunks_between_contents(event.content_at_retrieval, latest_change.new_content)

    return any(hunk_overlaps_symbol(old_start, old_count, event.symbol) for old_start, old_count, _, _ in hunks)


@dataclass
class LabeledExample:
    log_id: str
    symbol_id: str
    symbol_edited: bool


def backfill_all(events: list[RetrievalEvent], file_changes: list[FileChangeEvent]) -> list[LabeledExample]:
    """Backfill every event; only True/False outcomes become training examples (§4)."""
    examples = []
    for event in events:
        outcome = backfill_outcome(event, file_changes)
        if outcome is not None:
            examples.append(LabeledExample(log_id=event.log_id, symbol_id=event.symbol.id, symbol_edited=outcome))
    return examples
