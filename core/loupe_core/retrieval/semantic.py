"""Local sentence embeddings + content-hash cache (docs/phase-2-retrieval.md §5).

Two responsibilities live here, deliberately kept together since they're
always used as a pair: deciding *whether* a symbol needs re-embedding (the
`EmbeddingCache` table, keyed by content_hash) and actually calling the
model when it does. The searchable KNN index itself is a separate concern,
owned by `storage/vector_store.py`.
"""

from __future__ import annotations

import sqlite3
import struct

from sentence_transformers import SentenceTransformer

from loupe_core.parsing.schema import Symbol
from loupe_core.storage.vector_store import VectorStore

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM = 384
BATCH_SIZE = 64

_model: SentenceTransformer | None = None


def _get_default_model() -> SentenceTransformer:
    """Lazily load the real model once per process — it's ~130MB, not something to reload per call."""
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def embed_text_for_symbol(symbol: Symbol) -> str:
    """docstring+signature when present, else qualified_name+signature (§5's decided policy)."""
    if symbol.docstring:
        return f"{symbol.docstring}\n{symbol.signature}"
    return f"{symbol.qualified_name}\n{symbol.signature}"


def _pack(embedding: list[float]) -> bytes:
    return struct.pack(f"{len(embedding)}f", *embedding)


def _unpack(blob: bytes) -> list[float]:
    count = len(blob) // 4
    return list(struct.unpack(f"{count}f", blob))


class EmbeddingCache:
    """`embedding_cache(symbol_id TEXT PRIMARY KEY, content_hash TEXT, embedding BLOB)` (§5)."""

    def __init__(self, db_path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(db_path)
        self._conn.execute(
            "CREATE TABLE IF NOT EXISTS embedding_cache ("
            "symbol_id TEXT PRIMARY KEY, content_hash TEXT, embedding BLOB)"
        )
        self._conn.commit()

    def get(self, symbol_id: str) -> tuple[str, list[float]] | None:
        row = self._conn.execute(
            "SELECT content_hash, embedding FROM embedding_cache WHERE symbol_id = ?", (symbol_id,)
        ).fetchone()
        if row is None:
            return None
        content_hash, blob = row
        return content_hash, _unpack(blob)

    def put(self, symbol_id: str, content_hash: str, embedding: list[float]) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO embedding_cache(symbol_id, content_hash, embedding) VALUES (?, ?, ?)",
            (symbol_id, content_hash, _pack(embedding)),
        )
        self._conn.commit()

    def delete(self, symbol_id: str) -> None:
        self._conn.execute("DELETE FROM embedding_cache WHERE symbol_id = ?", (symbol_id,))
        self._conn.commit()

    def sync(self, current_symbol_ids: set[str]) -> None:
        """Remove cache rows for symbols that no longer exist (§5)."""
        cached_ids = {row[0] for row in self._conn.execute("SELECT symbol_id FROM embedding_cache")}
        for stale_id in cached_ids - current_symbol_ids:
            self.delete(stale_id)


class SemanticIndex:
    """Ties the embedding cache and the vector store together for a symbol set."""

    def __init__(
        self,
        dim: int = EMBEDDING_DIM,
        cache_db_path: str = ":memory:",
        vector_db_path: str = ":memory:",
        model: object | None = None,
    ) -> None:
        self._cache = EmbeddingCache(cache_db_path)
        self._store = VectorStore(dim=dim, db_path=vector_db_path)
        self._model = model  # injectable for tests; production uses the lazy real model

    def _encode(self, texts: list[str]) -> list[list[float]]:
        model = self._model if self._model is not None else _get_default_model()
        embeddings = model.encode(texts, batch_size=BATCH_SIZE, normalize_embeddings=True)
        return [list(row) for row in embeddings]

    def index(self, symbols: list[Symbol]) -> None:
        """Reuse cached embeddings for unchanged content; batch-embed only what changed."""
        to_embed_symbols: list[Symbol] = []
        to_embed_texts: list[str] = []

        for symbol in symbols:
            cached = self._cache.get(symbol.id)
            if cached is not None and cached[0] == symbol.content_hash:
                self._store.upsert(symbol.id, cached[1])
                continue
            to_embed_symbols.append(symbol)
            to_embed_texts.append(embed_text_for_symbol(symbol))

        if to_embed_texts:
            embeddings = self._encode(to_embed_texts)
            for symbol, embedding in zip(to_embed_symbols, embeddings, strict=True):
                self._cache.put(symbol.id, symbol.content_hash, embedding)
                self._store.upsert(symbol.id, embedding)

        current_ids = {s.id for s in symbols}
        self._cache.sync(current_ids)
        self._store.sync(current_ids)

    def query(self, query_text: str, top_k: int = 50) -> list[tuple[str, float]]:
        embedding = self._encode([query_text])[0]
        return self._store.query(embedding, top_k=top_k)

    def is_cached(self, symbol_id: str) -> bool:
        """Whether `symbol_id` currently has an embedding_cache row (for test/inspection use)."""
        return self._cache.get(symbol_id) is not None
