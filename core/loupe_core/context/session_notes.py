"""Decay-ranked, MMR-deduplicated session scratchpad (docs/PhaseX/session-notes.md).

Two-tier storage (§3): a full, append-only JSONL log — every note ever
written, never deleted, exactly Phase 4's `RetrievalLog` pattern — plus an
"active" working set managed by Phase 3's own `EvictionCache` (§1), reused
directly rather than reimplemented, so old low-importance notes stop
competing for attention without ever being lost from the log.

Retrieval (§4) composes three already-existing techniques, none of them new:
lexical+semantic search over note text (Phase 2's own tokenizer and
embedding model, reused directly) -> decayed-importance re-rank (§1) -> MMR
deduplication (retrieval-upgrades §4, reused directly).

The core property this all serves (§5): notes live server-side, entirely
independent of Claude Code's own conversation transcript — the same reason
Phase 3's `SessionState` already survives compaction untouched.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from loupe_core.governor.eviction import EvictionCache
from loupe_core.retrieval.lexical import tokenize
from loupe_core.retrieval.mmr import DEFAULT_LAMBDA, cosine_similarity, mmr_select

# Matches retrieval/fusion.py's constant. Duplicated, not imported, the same
# way eval/harness.py already duplicates it — this module combines only two
# signals (lexical + semantic, no centrality-equivalent for notes exists), a
# genuinely different combination shape from fusion.py's `fuse()`, not a
# smaller copy of the same function.
RRF_K = 60

# §3: "once the active set exceeds a size threshold" — no exact number given.
# A documented, revisit-eligible constant, same spirit as every other tuned
# number in this project (RRF's k=60, MMR's lambda=0.7).
DEFAULT_ACTIVE_SET_LIMIT = 50

MIN_IMPORTANCE = 1
MAX_IMPORTANCE = 5


@dataclass
class Note:
    note_id: str
    session_id: str
    content: str
    importance: int  # 1-5, self-assigned at write time (§1)
    turn_index: int
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "note_id": self.note_id,
            "session_id": self.session_id,
            "content": self.content,
            "importance": self.importance,
            "turn_index": self.turn_index,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Note":
        return cls(**data)


def _get_default_model():
    from loupe_core.retrieval.semantic import get_default_model

    return get_default_model()


def _search_notes(query: str, notes: list[Note], model: object | None = None) -> dict[str, float]:
    """Lexical (BM25, Phase 2's own tokenizer) + semantic (Phase 2's own
    embedding model) relevance-to-`query`, combined via a lightweight 2-signal
    RRF — `fusion.py`'s `fuse()` isn't reused directly here since it's built
    around a third, centrality signal notes have no equivalent of; the actual
    point of reuse is the tokenizer and embedding model, not that specific
    3-signal function shape.
    """
    if not notes:
        return {}

    from rank_bm25 import BM25Okapi

    corpus = [tokenize(n.content) for n in notes]
    bm25 = BM25Okapi(corpus)
    lexical_scores = bm25.get_scores(tokenize(query))
    lexical_order = sorted(range(len(notes)), key=lambda i: -lexical_scores[i])
    lexical_rank = {notes[i].note_id: rank + 1 for rank, i in enumerate(lexical_order)}

    embed_model = model if model is not None else _get_default_model()
    note_embeddings = embed_model.encode([n.content for n in notes], normalize_embeddings=True)
    query_embedding = list(embed_model.encode([query], normalize_embeddings=True)[0])
    semantic_scores = [cosine_similarity(query_embedding, list(e)) for e in note_embeddings]
    semantic_order = sorted(range(len(notes)), key=lambda i: -semantic_scores[i])
    semantic_rank = {notes[i].note_id: rank + 1 for rank, i in enumerate(semantic_order)}

    return {
        n.note_id: 1.0 / (RRF_K + lexical_rank[n.note_id]) + 1.0 / (RRF_K + semantic_rank[n.note_id]) for n in notes
    }


def _embed_notes(notes: list[Note], model: object | None = None) -> dict[str, list[float]]:
    if not notes:
        return {}
    embed_model = model if model is not None else _get_default_model()
    embeddings = embed_model.encode([n.content for n in notes], normalize_embeddings=True)
    return {n.note_id: list(e) for n, e in zip(notes, embeddings, strict=True)}


class SessionNotesStore:
    """Owns one session's notes: the full append-only log on disk, plus an
    in-memory active-set `EvictionCache`. One instance per live session,
    analogous to `SessionManager`'s per-session `SessionState` (Phase 3).
    """

    def __init__(
        self,
        session_id: str,
        logs_dir: Path,
        active_set_limit: int = DEFAULT_ACTIVE_SET_LIMIT,
        model: object | None = None,
    ) -> None:
        self.session_id = session_id
        self._log_path = logs_dir / session_id / "notes.jsonl"
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        self._active_set_limit = active_set_limit
        self._model = model
        self._active_notes: dict[str, Note] = {}
        self._eviction = EvictionCache()
        self._turn_index = 0

    def write(self, content: str, importance: int) -> Note:
        if not (MIN_IMPORTANCE <= importance <= MAX_IMPORTANCE):
            raise ValueError(f"importance must be between {MIN_IMPORTANCE} and {MAX_IMPORTANCE}, got {importance}")

        # One write = one turn tick (§1): decay every existing active note by
        # one turn *before* adding the new one, mirroring governor/session.py's
        # `request_symbols` "Step 1: decay every current resident by one turn"
        # — the same reused mechanism, the same "decay happens first" ordering.
        self._eviction.decay_step()

        self._turn_index += 1
        note = Note(
            note_id=uuid.uuid4().hex,
            session_id=self.session_id,
            content=content,
            importance=importance,
            turn_index=self._turn_index,
        )
        self._append_to_log(note)
        self._add_to_active_set(note)
        return note

    def _append_to_log(self, note: Note) -> None:
        with open(self._log_path, "a") as f:
            f.write(json.dumps(note.to_dict()) + "\n")

    def _add_to_active_set(self, note: Note) -> None:
        self._eviction.add_or_refresh(note.note_id, float(note.importance))
        self._active_notes[note.note_id] = note
        while len(self._active_notes) > self._active_set_limit:
            evicted_id = self._eviction.evict_lowest()
            if evicted_id is None:
                break
            self._active_notes.pop(evicted_id, None)

    def is_active(self, note_id: str) -> bool:
        return note_id in self._active_notes

    def read_recent(self, limit: int = 10) -> list[Note]:
        ordered = sorted(self._active_notes.values(), key=lambda n: -n.turn_index)
        return ordered[:limit]

    def list_all(self) -> list[Note]:
        """Every note ever written, from the full log — not just the active set."""
        if not self._log_path.exists():
            return []
        notes = []
        with open(self._log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    notes.append(Note.from_dict(json.loads(line)))
        return notes

    def read_relevant(self, query: str, top_k: int = 5, lambda_param: float = DEFAULT_LAMBDA) -> list[Note]:
        """§4's full pipeline: search the active set by relevance to `query` ->
        re-rank by decayed importance -> MMR-deduplicate the final candidates.
        """
        candidates = list(self._active_notes.values())
        if not candidates:
            return []

        relevance = _search_notes(query, candidates, model=self._model)
        decayed_importance = self._eviction.current_scores
        # Neither relevance-to-query nor decayed importance alone is the full
        # picture — a highly relevant but long-decayed note, or a currently
        # important but query-irrelevant one, are both weaker candidates than
        # a note that's genuinely both.
        combined = {
            note_id: relevance.get(note_id, 0.0) * decayed_importance.get(note_id, 0.0) for note_id in relevance
        }
        ranked = sorted(combined.items(), key=lambda pair: (-pair[1], pair[0]))

        embeddings = _embed_notes(candidates, model=self._model)
        selected = mmr_select(ranked, embeddings, final_top_k=top_k, lambda_param=lambda_param)

        notes_by_id = {n.note_id: n for n in candidates}
        return [notes_by_id[note_id] for note_id, _ in selected if note_id in notes_by_id]
