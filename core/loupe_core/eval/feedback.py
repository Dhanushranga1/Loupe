"""Human feedback as a training label, taking strict precedence over the proxy signal
(docs/loupe-extensions.md E3, layered on top of Phase 6's `backfill.py`).

Deliberately framework-free, same design boundary as `backfill.py`: this
module operates on plain `FeedbackEntry`/`RetrievalEvent`/`FileChangeEvent`
records, not on the server's actual HTTP endpoint or JSONL storage format —
those are a thin server-side integration layer (`server/app/feedback.py`),
not core logic.

Precedence rule, stated explicitly in the spec and enforced here: when both
a `FeedbackEntry` and a proxy-derived `outcome.symbol_edited` exist for the
same `retrieval_log_id`, the human feedback entry wins outright and the
proxy signal is discarded for that entry — never averaged or blended. A
direct human judgment is strictly higher-confidence than "was the file
touched afterward," and blending a clean signal with a noisy one just
reintroduces the noise the clean label exists to avoid.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loupe_core.eval.backfill import FileChangeEvent, RetrievalEvent, backfill_outcome
from loupe_core.eval.backfill import LabeledExample

Rating = Literal["helpful", "not_helpful"]
Source = Literal["dashboard", "claude_self_report"]


@dataclass
class FeedbackEntry:
    retrieval_log_id: str
    rating: Rating
    note: str | None
    submitted_at: float
    source: Source


def resolve_training_label(
    retrieval_log_id: str,
    proxy_outcome: bool | None,
    feedback_by_log_id: dict[str, FeedbackEntry],
) -> bool | None:
    """The label to train on for one retrieval log entry, or None if there's no signal at all.

    Feedback, when present, always wins outright — it is never averaged or
    blended with the proxy signal, per the precedence rule above.
    """
    feedback = feedback_by_log_id.get(retrieval_log_id)
    if feedback is not None:
        return feedback.rating == "helpful"
    return proxy_outcome


def backfill_all_with_feedback(
    events: list[RetrievalEvent],
    file_changes: list[FileChangeEvent],
    feedback_by_log_id: dict[str, FeedbackEntry],
) -> list[LabeledExample]:
    """`backfill_all`, but consulting human feedback first for every event (§4's precedence rule).

    Appended alongside `backfill.py`'s own `backfill_all` rather than
    replacing it — training-data callers that don't yet have feedback data
    keep using the proxy-only path unchanged.
    """
    examples = []
    for event in events:
        proxy_outcome = backfill_outcome(event, file_changes)
        label = resolve_training_label(event.log_id, proxy_outcome, feedback_by_log_id)
        if label is not None:
            examples.append(LabeledExample(log_id=event.log_id, symbol_id=event.symbol.id, symbol_edited=label))
    return examples
