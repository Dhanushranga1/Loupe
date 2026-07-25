"""Target-project build ledger (docs/target-project-build-ledger.md).

Tracks build/implementation progress on the project Loupe is pointed at —
not Loupe's own build. Every mechanism here is reused, not reinvented:
`depends_on` (§3) comes straight from Phase 1's `expand_dependencies` in the
outgoing direction; `has_test` (§5) from E2's `TESTS` edge; staleness (§4)
from Phase 0's `content_hash` plus, for the transitive case, E1's
`analyze_impact`. This module is glue over three already-shipped
capabilities, which is also why it stays small.

Opt-in by design: nothing here auto-seeds every symbol in a repo as a
ledger entry. A symbol is tracked because `set_status` was called for it —
same as a real task tracker not pre-populating a card for every file.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path

import networkx as nx

from loupe_core.graph.builder import EdgeType
from loupe_core.graph.impact import analyze_impact
from loupe_core.graph.traversal import expand_dependencies
from loupe_core.parsing.schema import Symbol


class LedgerStatus(str, Enum):
    PLANNED = "planned"
    IN_PROGRESS = "in_progress"
    DONE = "done"
    STALE = "stale"  # only ever reached automatically, from DONE — never set directly via set_status


@dataclass
class BuildLedgerEntry:
    symbol_id: str
    status: LedgerStatus
    depends_on: list[str] = field(default_factory=list)
    has_test: bool = False
    content_hash_at_done: str | None = None
    note: str | None = None
    updated_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["status"] = self.status.value
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "BuildLedgerEntry":
        d = dict(data)
        d["status"] = LedgerStatus(d["status"])
        return cls(**d)


def _direct_dependencies(graph: nx.DiGraph, symbol_id: str) -> list[str]:
    """§3: what this symbol calls (outgoing), not what calls it — the opposite
    direction from E1's impact analysis, and a deliberate choice: a build
    ledger's "depends on" means "must exist before this is meaningfully
    done," which is the outgoing direction.
    """
    return sorted(expand_dependencies(graph, symbol_id, depth=1, direction="outgoing", edge_type=EdgeType.CALLS))


def _has_linked_test(graph: nx.DiGraph, symbol_id: str) -> bool:
    """§5: any confidence level counts — this field is "is there anything
    linking a test to this," not a confidence gate. `graph` must already
    have `link_tests` (E2) run on it, same as any other TESTS-edge consumer.
    """
    return bool(expand_dependencies(graph, symbol_id, depth=1, direction="incoming", edge_type=EdgeType.TESTS))


class BuildLedgerStore:
    """Owns one project's ledger: `.loupe/build/ledger.json`, a flat
    `{symbol_id: BuildLedgerEntry}` map — matching the plain-JSON-map shape
    already used for `cache/churn.json`/`cache/co_retrieval.json`.
    """

    def __init__(self, ledger_path: Path) -> None:
        self._path = ledger_path
        self._entries: dict[str, BuildLedgerEntry] = self._load()

    def _load(self) -> dict[str, BuildLedgerEntry]:
        if not self._path.exists():
            return {}
        data = json.loads(self._path.read_text())
        return {sid: BuildLedgerEntry.from_dict(entry) for sid, entry in data.items()}

    def _save(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        data = {sid: entry.to_dict() for sid, entry in self._entries.items()}
        self._path.write_text(json.dumps(data, indent=2))

    def set_status(
        self,
        symbol_id: str,
        status: LedgerStatus,
        *,
        graph: nx.DiGraph,
        symbols_by_id: dict[str, Symbol],
        note: str | None = None,
    ) -> BuildLedgerEntry:
        """Creates the entry (auto-populating `depends_on`/`has_test` per
        §3/§5) if `symbol_id` isn't already tracked; otherwise updates status
        in place, reusing the previously-computed `depends_on`/`has_test`
        rather than re-querying the graph on every status change.
        """
        existing = self._entries.get(symbol_id)
        depends_on = existing.depends_on if existing is not None else _direct_dependencies(graph, symbol_id)
        has_test = existing.has_test if existing is not None else _has_linked_test(graph, symbol_id)

        content_hash_at_done = existing.content_hash_at_done if existing is not None else None
        if status == LedgerStatus.DONE:
            symbol = symbols_by_id.get(symbol_id)
            content_hash_at_done = symbol.content_hash if symbol is not None else None

        entry = BuildLedgerEntry(
            symbol_id=symbol_id,
            status=status,
            depends_on=depends_on,
            has_test=has_test,
            content_hash_at_done=content_hash_at_done,
            note=note if note is not None else (existing.note if existing is not None else None),
        )
        self._entries[symbol_id] = entry
        self._save()
        return entry

    def _refresh_staleness(
        self, graph: nx.DiGraph, symbols_by_id: dict[str, Symbol], pagerank_scores: dict[str, float]
    ) -> None:
        """§4, run lazily on every read: direct staleness first (§4.1, a
        DONE entry's own content_hash no longer matching), then transitive
        (§4.2) — for every entry just found directly stale, run E1's
        `analyze_impact` and flag any *other* DONE ledger entry in that
        blast radius, even though its own content_hash hasn't changed.
        """
        directly_stale: set[str] = set()
        for entry in self._entries.values():
            if entry.status != LedgerStatus.DONE:
                continue
            symbol = symbols_by_id.get(entry.symbol_id)
            current_hash = symbol.content_hash if symbol is not None else None
            if current_hash != entry.content_hash_at_done:
                entry.status = LedgerStatus.STALE
                entry.updated_at = time.time()
                directly_stale.add(entry.symbol_id)

        changed = bool(directly_stale)
        for changed_symbol_id in directly_stale:
            report = analyze_impact(graph, symbols_by_id, pagerank_scores, changed_symbol_id)
            affected_ids = {s.symbol_id for s in report.directly_affected} | {
                s.symbol_id for s in report.transitively_affected
            }
            for entry in self._entries.values():
                if entry.symbol_id in affected_ids and entry.status == LedgerStatus.DONE:
                    entry.status = LedgerStatus.STALE
                    entry.updated_at = time.time()
                    changed = True

        if changed:
            self._save()

    def get(
        self, symbol_id: str, *, graph: nx.DiGraph, symbols_by_id: dict[str, Symbol], pagerank_scores: dict[str, float]
    ) -> BuildLedgerEntry | None:
        self._refresh_staleness(graph, symbols_by_id, pagerank_scores)
        return self._entries.get(symbol_id)

    def list(
        self,
        *,
        graph: nx.DiGraph,
        symbols_by_id: dict[str, Symbol],
        pagerank_scores: dict[str, float],
        filter_status: LedgerStatus | None = None,
    ) -> list[BuildLedgerEntry]:
        self._refresh_staleness(graph, symbols_by_id, pagerank_scores)
        entries = list(self._entries.values())
        if filter_status is not None:
            entries = [e for e in entries if e.status == filter_status]
        return sorted(entries, key=lambda e: e.symbol_id)
