"""Maps MCP session identity to Phase 3 SessionState (docs/phase-4-systems.md §5).

Addendum item (b), resolved: session identity comes from `fastapi-mcp`'s
connection-scoped `Mcp-Session-Id` header — verified empirically against a
real MCP handshake (see docs/progress/phase-4/checklist.md's changelog), not
an explicit tool argument threaded through by the model. A fallback pseudo-
session key is used only when that header is genuinely absent (e.g. a raw
HTTP client bypassing the MCP transport entirely), per the addendum's
explicit instruction not to let a missing session id silently reset state.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from fastapi import Request

from loupe_core.governor.session import SessionState

from .config import DEFAULT_TOKEN_BUDGET

DEFAULT_TTL_SECONDS = 2 * 60 * 60  # 2 hours
FALLBACK_SESSION_KEY = "__no-mcp-session-id__"


@dataclass
class _SessionEntry:
    state: SessionState
    last_active: float


class SessionManager:
    """In-memory session_id -> SessionState map with TTL-based expiry.

    No persistence across process restarts (deliberately deferred, matching
    Phase 3's own scoping note — durable storage is a `storage/` concern for
    when it's actually needed).
    """

    def __init__(self, ttl_seconds: int = DEFAULT_TTL_SECONDS, clock=time.monotonic) -> None:
        self._ttl_seconds = ttl_seconds
        self._clock = clock
        self._sessions: dict[str, _SessionEntry] = {}

    def get_or_create(self, session_id: str, token_budget_total: int = DEFAULT_TOKEN_BUDGET) -> SessionState:
        entry = self._sessions.get(session_id)
        if entry is None:
            state = SessionState(session_id=session_id, token_budget_total=token_budget_total)
            entry = _SessionEntry(state=state, last_active=self._clock())
            self._sessions[session_id] = entry
        else:
            entry.last_active = self._clock()
        return entry.state

    def sweep_expired(self) -> list[str]:
        """Remove sessions idle past TTL; return the removed session ids."""
        now = self._clock()
        expired = [sid for sid, entry in self._sessions.items() if now - entry.last_active > self._ttl_seconds]
        for sid in expired:
            del self._sessions[sid]
        return expired

    def __len__(self) -> int:
        return len(self._sessions)

    def __contains__(self, session_id: str) -> bool:
        return session_id in self._sessions


def session_id_from_request(request: Request) -> str:
    """The MCP session id for this request, or a fallback key if genuinely absent."""
    return request.headers.get("mcp-session-id", FALLBACK_SESSION_KEY)
