"""Data contracts for Phase 0: the Symbol record and its id/hash helpers.

See docs/phase-0-foundations.md §3 for the authoritative spec this module implements.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum


class SymbolKind(str, Enum):
    """The kinds of symbols Phase 0 extracts from Python source."""

    FUNCTION = "function"
    ASYNC_FUNCTION = "async_function"
    METHOD = "method"
    CLASS = "class"


def compute_symbol_id(file_path: str, qualified_name: str, kind: SymbolKind) -> str:
    """Deterministic id from (file_path, qualified_name, kind) — stable across body edits.

    Truncated to 16 hex characters: short enough to be a practical join key, long
    enough that collisions within a single repo's symbol set are not a realistic concern.
    """
    digest = hashlib.sha256(f"{file_path}:{qualified_name}:{kind.value}".encode("utf-8"))
    return digest.hexdigest()[:16]


def compute_content_hash(source_bytes: bytes) -> str:
    """Full sha256 hex digest of a symbol's exact byte range — changes iff the body changes."""
    return hashlib.sha256(source_bytes).hexdigest()


@dataclass
class Symbol:
    """The atomic retrieval unit: one function, class, or method with an exact byte range."""

    id: str
    kind: SymbolKind
    name: str
    qualified_name: str
    file_path: str
    byte_start: int
    byte_end: int
    line_start: int
    line_end: int
    signature: str
    docstring: str | None
    decorators: list[str] = field(default_factory=list)
    parent_id: str | None = None
    content_hash: str = ""
    language: str = "python"
