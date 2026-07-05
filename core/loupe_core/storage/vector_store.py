"""sqlite-vec wrapper for embedding storage/search (docs/phase-2-retrieval.md §5).

This module owns only the searchable vector index — a sqlite-vec virtual
table storing `(symbol_id, embedding)` and a KNN query over it. The
content-hash-keyed `embedding_cache` table (the "should I even re-embed this
symbol" decision) is a separate, plain table owned by `retrieval/semantic.py`
— the two are deliberately not merged, matching the spec's own two distinct
named tables.
"""

from __future__ import annotations

import sqlite3
import struct

import sqlite_vec


def _rowid_for_symbol(symbol_id: str) -> int:
    """Deterministic positive 63-bit rowid derived from a hex symbol_id.

    vec0 virtual tables require an integer rowid; `Symbol.id` is a 16-hex-char
    string (64 bits). Masking off the sign bit keeps it a valid positive
    SQLite INTEGER rowid; the collision risk from the 1-bit truncation is
    negligible at this project's scale.
    """
    return int(symbol_id, 16) & 0x7FFFFFFFFFFFFFFF


def _to_blob(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


class VectorStore:
    """A sqlite-vec-backed KNN index over symbol embeddings."""

    def __init__(self, dim: int = 384, db_path: str = ":memory:") -> None:
        self._dim = dim
        self._conn = sqlite3.connect(db_path)
        self._conn.enable_load_extension(True)
        sqlite_vec.load(self._conn)
        self._conn.enable_load_extension(False)
        self._conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vec_symbols USING vec0("
            f"symbol_id TEXT, embedding FLOAT[{dim}])"
        )
        self._conn.commit()

    def upsert(self, symbol_id: str, embedding: list[float]) -> None:
        """Insert or replace this symbol's embedding in the searchable index."""
        rowid = _rowid_for_symbol(symbol_id)
        self._conn.execute("DELETE FROM vec_symbols WHERE rowid = ?", (rowid,))
        self._conn.execute(
            "INSERT INTO vec_symbols(rowid, symbol_id, embedding) VALUES (?, ?, ?)",
            (rowid, symbol_id, _to_blob(embedding)),
        )
        self._conn.commit()

    def delete(self, symbol_id: str) -> None:
        self._conn.execute("DELETE FROM vec_symbols WHERE rowid = ?", (_rowid_for_symbol(symbol_id),))
        self._conn.commit()

    def sync(self, current_symbol_ids: set[str]) -> None:
        """Remove any indexed symbol not in `current_symbol_ids` (deleted-symbol cleanup)."""
        indexed_ids = {row[0] for row in self._conn.execute("SELECT symbol_id FROM vec_symbols")}
        for stale_id in indexed_ids - current_symbol_ids:
            self.delete(stale_id)

    def query(self, embedding: list[float], top_k: int = 50) -> list[tuple[str, float]]:
        """Top `top_k` (symbol_id, similarity_score) pairs, most similar first.

        Assumes L2-normalized embeddings (see retrieval/semantic.py), so
        L2 distance and cosine similarity rank candidates identically;
        `similarity = 1 - distance^2 / 2` converts distance to a
        higher-is-better cosine similarity for a normalized pair of vectors.
        """
        rows = self._conn.execute(
            "SELECT symbol_id, distance FROM vec_symbols WHERE embedding MATCH ? AND k = ? ORDER BY distance",
            (_to_blob(embedding), top_k),
        ).fetchall()
        return [(symbol_id, 1 - (distance**2) / 2) for symbol_id, distance in rows]

    def close(self) -> None:
        self._conn.close()
