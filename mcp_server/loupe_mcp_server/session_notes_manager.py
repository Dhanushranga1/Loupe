"""Maps MCP session identity to a `session_notes.py` `SessionNotesStore`
(docs/PhaseX/session-notes.md), mirroring `session_manager.py`'s own
session_id -> SessionState mapping — a second, independent piece of
per-session state, not a member of `SessionState` itself (notes and governed
symbol residency decay/evict on entirely separate schedules).
"""

from __future__ import annotations

from pathlib import Path

from loupe_core.context.session_notes import SessionNotesStore


class SessionNotesManager:
    def __init__(self, logs_dir: Path) -> None:
        self._logs_dir = logs_dir
        self._stores: dict[str, SessionNotesStore] = {}

    def get_or_create(self, session_id: str) -> SessionNotesStore:
        store = self._stores.get(session_id)
        if store is None:
            store = SessionNotesStore(session_id, logs_dir=self._logs_dir)
            self._stores[session_id] = store
        return store
