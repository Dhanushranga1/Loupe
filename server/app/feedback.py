"""Human feedback storage and the plain HTTP endpoint the Lens dashboard button calls
(docs/loupe-extensions.md E3).

Deliberately a plain FastAPI route (`POST /feedback`), not an MCP tool — a
human clicking a dashboard button never appears in Claude's context at all,
so it costs nothing against the MCP tool-count budget that governs LLM
selection accuracy (the same budget `analyze_impact`, E1's 5th tool, does
count against). `mcp_tools.py`'s `submit_feedback` is the secondary,
optional path for Claude to solicit feedback conversationally — this
module's `FeedbackStore` backs both.

One flat `.loupe/logs/feedback/feedback.jsonl`, not per-session like
`TelemetryWriter`'s retrieval logs: a `FeedbackEntry` references a
`retrieval_log_id` that already carries its session's identity, so there's
no natural per-session file boundary to mirror here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from loupe_core.eval.feedback import FeedbackEntry

FEEDBACK_LOG_FILENAME = "feedback.jsonl"


class FeedbackRequest(BaseModel):
    retrieval_log_id: str
    rating: Literal["helpful", "not_helpful"]
    note: str | None = None


class FeedbackStore:
    """Appends/reads `FeedbackEntry` records against `.loupe/logs/feedback/feedback.jsonl`."""

    def __init__(self, logs_dir: Path) -> None:
        self._path = logs_dir / FEEDBACK_LOG_FILENAME
        logs_dir.mkdir(parents=True, exist_ok=True)

    def submit(self, retrieval_log_id: str, rating: Literal["helpful", "not_helpful"], note: str | None, source: str) -> FeedbackEntry:
        entry = FeedbackEntry(
            retrieval_log_id=retrieval_log_id, rating=rating, note=note, submitted_at=time.time(), source=source
        )
        with open(self._path, "a") as f:
            f.write(
                json.dumps(
                    {
                        "retrieval_log_id": entry.retrieval_log_id,
                        "rating": entry.rating,
                        "note": entry.note,
                        "submitted_at": entry.submitted_at,
                        "source": entry.source,
                    }
                )
                + "\n"
            )
        return entry

    def all_by_log_id(self) -> dict[str, FeedbackEntry]:
        """Every submitted entry, keyed by `retrieval_log_id`. A later submission for the
        same log id overwrites an earlier one — the most recent human judgment wins."""
        if not self._path.exists():
            return {}
        entries: dict[str, FeedbackEntry] = {}
        with open(self._path) as f:
            for line in f:
                data = json.loads(line)
                entries[data["retrieval_log_id"]] = FeedbackEntry(**data)
        return entries
