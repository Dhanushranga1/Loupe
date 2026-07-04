"""File-level incremental cache: skip re-parsing files that haven't changed.

See docs/phase-0-foundations.md §5. Deliberately coarse — "did anything in
this file change at all" — not byte-range-level incremental re-parsing,
which is explicitly deferred (see phase-0-foundations.md §8).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from .schema import Symbol


@dataclass
class _FileEntry:
    file_hash: str
    symbol_ids: list[str]


class FileIndexCache:
    """Tracks, per file, whether its content has changed since it was last indexed."""

    def __init__(self) -> None:
        self._entries: dict[str, _FileEntry] = {}

    def is_stale(self, file_path: str) -> bool:
        """True if `file_path` has never been indexed, or its content has changed since."""
        entry = self._entries.get(file_path)
        if entry is None:
            return True
        return _hash_file(file_path) != entry.file_hash

    def update(self, file_path: str, symbols: list[Symbol]) -> None:
        """Record the current hash of `file_path` and the ids of the symbols it produced."""
        self._entries[file_path] = _FileEntry(
            file_hash=_hash_file(file_path),
            symbol_ids=[s.id for s in symbols],
        )

    def symbol_ids_for(self, file_path: str) -> list[str]:
        """The symbol ids recorded for `file_path` on its last `update`, or `[]` if never indexed."""
        entry = self._entries.get(file_path)
        return list(entry.symbol_ids) if entry is not None else []

    def to_json(self) -> str:
        return json.dumps(
            {path: {"file_hash": e.file_hash, "symbol_ids": e.symbol_ids} for path, e in self._entries.items()},
            indent=2,
        )

    @classmethod
    def from_json(cls, data: str) -> FileIndexCache:
        cache = cls()
        for path, entry in json.loads(data).items():
            cache._entries[path] = _FileEntry(file_hash=entry["file_hash"], symbol_ids=list(entry["symbol_ids"]))
        return cache

    def save(self, path: str) -> None:
        Path(path).write_text(self.to_json())

    @classmethod
    def load(cls, path: str) -> FileIndexCache:
        return cls.from_json(Path(path).read_text())


def _hash_file(file_path: str) -> str:
    return hashlib.sha256(Path(file_path).read_bytes()).hexdigest()
