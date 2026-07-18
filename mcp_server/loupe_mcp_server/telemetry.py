"""Structured RetrievalLog telemetry (docs/phase-4-systems.md §6 + addendum item d).

Every one of the four MCP tools is logged, governed or not — telemetry
doesn't distinguish, it records everything (§6). Written as an appended
JSONL line per session (`.loupe/logs/retrieval/<session_id>.jsonl`), so
writing is always a pure append, never a read-modify-rewrite.

Addendum item (d): `latency_ms`, `output_size_bytes`, and a nullable
`error_code` are included from the start — cheap now, annoying to retrofit
once logs already exist without them.

Simplification worth being explicit about: the original `RetrievalLog`
schema (loupe-project-guide.md §5) envisions `candidates` carrying full
per-signal (lexical/semantic/centrality) score breakdowns. That breakdown
only exists inside `search_symbols`' fusion call and isn't threaded back out
to a return value today. `log_tool_call` is a single generic wrapper reused
across all four tools, so `candidates`/`selected` here are derived from
whatever the route actually returned (symbol id + any score attached) —
correct and populated for every call, not a full RRF audit trail. Widening
this is a natural Phase 5/6 refinement once retrieval logs are actually
being trained on, not a Phase 4 requirement.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from fastapi import Request

from .session_manager import session_id_from_request


@dataclass
class RetrievalLog:
    session_id: str
    turn_index: int
    tool_name: str
    query_text: str | None
    query_intent: str | None
    candidates: list[dict[str, Any]]
    selected: list[dict[str, Any]]
    latency_ms: float
    output_size_bytes: int
    error_code: str | None = None
    outcome: dict[str, Any] | None = None  # always null at log time (§6) — backfilled in Phase 6
    log_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    timestamp: float = field(default_factory=time.time)

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "log_id": self.log_id,
                "timestamp": self.timestamp,
                "session_id": self.session_id,
                "turn_index": self.turn_index,
                "tool_name": self.tool_name,
                "query_text": self.query_text,
                "query_intent": self.query_intent,
                "candidates": self.candidates,
                "selected": self.selected,
                "latency_ms": self.latency_ms,
                "output_size_bytes": self.output_size_bytes,
                "error_code": self.error_code,
                "outcome": self.outcome,
            }
        )


class TelemetryWriter:
    """Appends RetrievalLog entries to `.loupe/logs/retrieval/<session_id>.jsonl`."""

    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir
        self._logs_dir.mkdir(parents=True, exist_ok=True)
        self._turn_counters: dict[str, int] = {}

    def next_turn_index(self, session_id: str) -> int:
        self._turn_counters[session_id] = self._turn_counters.get(session_id, 0) + 1
        return self._turn_counters[session_id]

    def write(self, log: RetrievalLog) -> None:
        path = self._logs_dir / f"{log.session_id}.jsonl"
        with open(path, "a") as f:
            f.write(log.to_json_line() + "\n")


def _entry_to_dict(item: Any) -> dict[str, Any]:
    if hasattr(item, "model_dump"):
        data = item.model_dump()
        return {"symbol_id": data.get("symbol_id"), "score": data.get("score")}
    return {"value": str(item)}


def _result_entries(result: Any) -> list[dict[str, Any]]:
    if result is None:
        return []
    if isinstance(result, list):
        return [_entry_to_dict(item) for item in result]
    return [_entry_to_dict(result)]


def _result_size_bytes(result: Any) -> int:
    if result is None:
        return 0
    if isinstance(result, list):
        return sum(_result_size_bytes(item) for item in result)
    if hasattr(result, "model_dump_json"):
        return len(result.model_dump_json().encode("utf-8"))
    return len(json.dumps(result, default=str).encode("utf-8"))


def log_tool_call(tool_name: str) -> Callable:
    """Wrap an MCP tool route: time it, log a RetrievalLog entry, log errors too.

    `query_text` is taken from whichever of the common argument names the
    route was called with (`query`, `path_or_glob`, `symbol_id`).
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        async def wrapper(request: Request, *args: Any, **kwargs: Any) -> Any:
            telemetry: TelemetryWriter = request.app.state.telemetry
            session_id = session_id_from_request(request)
            turn_index = telemetry.next_turn_index(session_id)

            query_text = kwargs.get("query") or kwargs.get("path_or_glob") or kwargs.get("symbol_id")
            start = time.perf_counter()
            error_code: str | None = None
            result: Any = None
            try:
                result = await fn(request, *args, **kwargs)
                return result
            except Exception as exc:
                error_code = type(exc).__name__
                raise
            finally:
                latency_ms = (time.perf_counter() - start) * 1000
                entries = _result_entries(result)
                telemetry.write(
                    RetrievalLog(
                        session_id=session_id,
                        turn_index=turn_index,
                        tool_name=tool_name,
                        query_text=query_text,
                        query_intent=None,
                        candidates=entries,
                        selected=entries,
                        latency_ms=latency_ms,
                        output_size_bytes=_result_size_bytes(result),
                        error_code=error_code,
                    )
                )

        return wrapper

    return decorator
