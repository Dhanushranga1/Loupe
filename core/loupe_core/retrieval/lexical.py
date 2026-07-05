"""BM25 lexical retrieval over symbol names, signatures, and docstrings.

Implements docs/phase-2-retrieval.md §4. The tokenizer is the one piece
everything downstream depends on getting right — see its docstring for the
exact, tested algorithm.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from loupe_core.parsing.schema import Symbol

_NON_WORD = re.compile(r"[^A-Za-z0-9_]+")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_MIN_TOKEN_LENGTH = 2


def tokenize(text: str) -> list[str]:
    """Snake_case- and camelCase-aware tokenizer, lowercased, drops tokens < 2 chars.

    Algorithm (docs/phase-2-retrieval.md §4), applied to every '_'-and-alnum
    "word" found in `text`:
      1. Split on '_' (snake_case boundaries).
      2. Within each piece, split on lowercase->uppercase transitions (camelCase).
      3. Lowercase every resulting token.
      4. Drop tokens shorter than 2 characters and pure-punctuation tokens.
    """
    tokens: list[str] = []
    for word in _NON_WORD.split(text):
        if not word:
            continue
        for piece in word.split("_"):
            if not piece:
                continue
            for sub in _CAMEL_BOUNDARY.split(piece):
                token = sub.lower()
                if len(token) >= _MIN_TOKEN_LENGTH:
                    tokens.append(token)
    return tokens


def symbol_document_text(symbol: Symbol) -> str:
    """The text indexed per symbol: name, qualified_name, signature, docstring, decorators."""
    parts = [symbol.name, symbol.qualified_name, symbol.signature]
    if symbol.docstring:
        parts.append(symbol.docstring)
    parts.extend(symbol.decorators)
    return " ".join(parts)


class LexicalIndex:
    """A BM25 index over a repo's symbols, rebuilt fully on every re-index (§1)."""

    def __init__(self, symbols: list[Symbol]) -> None:
        self._symbol_ids = [s.id for s in symbols]
        corpus = [tokenize(symbol_document_text(s)) for s in symbols]
        self._bm25 = BM25Okapi(corpus) if corpus else None

    def query(self, query_text: str, top_k: int = 50) -> list[tuple[str, float]]:
        """Top `top_k` (symbol_id, bm25_score) pairs, sorted by score descending."""
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(tokenize(query_text))
        ranked = sorted(zip(self._symbol_ids, scores), key=lambda pair: pair[1], reverse=True)
        return ranked[:top_k]
