"""Test-to-code linkage (docs/loupe-extensions.md E2).

Adds a new `EdgeType.TESTS` edge, not a new tool — the moment a `TESTS` edge
exists in the same `nx.DiGraph` the rest of the graph lives in,
`expand_dependencies(..., edge_type=EdgeType.TESTS)` already traverses it
(see `traversal.py`'s `edge_type` filter, added for exactly this).

Two heuristics, combined into an honest three-way confidence label rather
than filtered down to one "best" answer (consistent with Phase 1's own
refusal to guess on ambiguous calls — see `docs/phase-1-graph-theory.md`):

1. Naming: a test symbol named `test_<name>`/`Test<Name>` matched
   case-insensitively against the target's bare name.
2. Call: the test symbol has an already-resolved `CALLS` edge to the target
   (Phase 1's own call resolution, reused as-is — no new resolution logic).

Both heuristics run only over symbols in files that look like test files
(`test_*.py` / `*_test.py`) — this is what makes `call_only` a meaningful,
distinct case from "any function that happens to call another function":
without a test-file scope, every ordinary caller in the whole graph would
otherwise qualify.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import networkx as nx

from loupe_core.parsing.schema import Symbol, SymbolKind

from .builder import EdgeType


class TestConfidence(str, Enum):
    __test__ = False  # not a pytest test class — pytest's collector otherwise flags any "Test*" name

    CONFIRMED = "confirmed"
    CALL_ONLY = "call_only"
    NAMING_ONLY = "naming_only"


@dataclass
class TestLink:
    test_symbol_id: str
    target_symbol_id: str
    confidence: TestConfidence


def _is_test_file(file_path: str) -> bool:
    filename = file_path.rsplit("/", 1)[-1]
    return filename.startswith("test_") or filename.endswith("_test.py")


def _naming_candidate_name(symbol: Symbol) -> str | None:
    if symbol.kind in (SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.ASYNC_FUNCTION) and symbol.name.startswith(
        "test_"
    ):
        suffix = symbol.name[len("test_") :]
        return suffix or None
    if symbol.kind == SymbolKind.CLASS and symbol.name.startswith("Test") and len(symbol.name) > len("Test"):
        return symbol.name[len("Test") :]
    return None


def link_tests(graph: nx.DiGraph, symbols_by_id: dict[str, Symbol]) -> list[TestLink]:
    """Find test<->target links, add them as `TESTS` edges into `graph` in place, and return them.

    Deterministic output order: symbols in `(file_path, byte_start)` order, then
    targets in sorted-id order within each — same "stable, reproducible" bar
    every other resolution pass in this codebase holds itself to.
    """
    by_name: dict[str, list[str]] = {}
    for sid, s in symbols_by_id.items():
        by_name.setdefault(s.name.casefold(), []).append(sid)

    test_symbols = sorted(
        (s for s in symbols_by_id.values() if _is_test_file(s.file_path)),
        key=lambda s: (s.file_path, s.byte_start),
    )

    links: list[TestLink] = []
    for symbol in test_symbols:
        sid = symbol.id
        candidate_name = _naming_candidate_name(symbol)
        naming_target_ids = set(by_name.get(candidate_name.casefold(), [])) if candidate_name else set()
        naming_target_ids.discard(sid)

        call_target_ids = {
            v for _, v, data in graph.out_edges(sid, data=True) if data.get("edge_type") == EdgeType.CALLS
        }

        confirmed = naming_target_ids & call_target_ids
        for target_id in sorted(confirmed):
            links.append(TestLink(sid, target_id, TestConfidence.CONFIRMED))
        for target_id in sorted(call_target_ids - confirmed):
            links.append(TestLink(sid, target_id, TestConfidence.CALL_ONLY))
        for target_id in sorted(naming_target_ids - confirmed):
            links.append(TestLink(sid, target_id, TestConfidence.NAMING_ONLY))

    for link in links:
        graph.add_edge(
            link.test_symbol_id,
            link.target_symbol_id,
            edge_type=EdgeType.TESTS,
            weight=1,
            confidence=link.confidence.value,
        )

    return links
