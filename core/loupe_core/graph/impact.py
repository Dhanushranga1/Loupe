"""Blast-radius / impact analysis (docs/loupe-extensions.md E1).

No new traversal algorithm — this is `expand_dependencies` (Phase 1) called
in the direction it already supports but nothing before E1 used:
`direction="incoming"`. "What calls this, transitively" is exactly what
"what breaks if I change this" means for a static call graph.

Hub threshold: the spec says to "reuse whatever cutoff Phase 1's
hub-detection already uses — don't invent a second one," but no such cutoff
exists anywhere in the codebase — Phase 1 computes PageRank and stores it,
but nothing before E1 thresholds it into "hub" vs. not. This introduces one,
documented here since the spec assumed it was inherited rather than new:
a symbol is a hub if its PageRank score is more than one standard deviation
above the graph's mean score. A fixed percentile (e.g. "top 10%") would
always flag *something* regardless of whether the graph actually has an
outlier; a stdev-based cutoff only fires when a symbol's centrality is
genuinely unusual relative to the rest of that specific repo's graph.

Route counting: no "FastAPI adapter" or `route` symbol kind exists anywhere
in this repo (E1's original scope note), and building one is real, separate
work. But Phase 0 already extracts each symbol's decorator text verbatim
(`Symbol.decorators`) regardless of framework — so "is this an HTTP route
handler" doesn't need a whole adapter, just a decorator-shape check:
`<name>.<http_method>(...)`, e.g. `router.get("/search")` or
`app.post("/orders")`. This is an honest heuristic (a decorator matching
that shape almost certainly is a route in FastAPI/Flask-style code; nothing
here confirms the base is really an `APIRouter` instance — that would be
real type inference, deliberately out of scope everywhere else in this
project too), not full framework awareness — real, found via a real
~900-symbol FastAPI codebase where this previously silently reported 0
routes affected out of 187 real route-handler callers.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

import networkx as nx

from loupe_core.parsing.schema import Symbol, SymbolKind

from .traversal import expand_dependencies

_HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}
_ROUTE_DECORATOR_PATTERN = re.compile(r"^\w+\.(" + "|".join(_HTTP_METHODS) + r")\(")


@dataclass
class SymbolSummary:
    symbol_id: str
    qualified_name: str
    file_path: str
    kind: SymbolKind


@dataclass
class ImpactReport:
    symbol_id: str
    directly_affected: list[SymbolSummary] = field(default_factory=list)
    transitively_affected: list[SymbolSummary] = field(default_factory=list)
    high_centrality_warnings: list[str] = field(default_factory=list)  # symbol_ids, ranked highest-pagerank first
    affected_route_count: int = 0  # decorator-shape heuristic — see module docstring


def hub_threshold(pagerank_scores: dict[str, float]) -> float:
    """Mean + one standard deviation of the graph's own PageRank distribution — see module docstring."""
    scores = list(pagerank_scores.values())
    if len(scores) < 2:
        return float("inf")  # can't call anything an "outlier" relative to itself/nothing
    return statistics.mean(scores) + statistics.pstdev(scores)


def _looks_like_http_route(symbol: Symbol) -> bool:
    """True if any of `symbol`'s decorators is shaped like `<name>.<http_method>(...)` —
    see module docstring for exactly what this does and doesn't claim to detect."""
    return any(_ROUTE_DECORATOR_PATTERN.match(d) for d in symbol.decorators)


def _summaries(symbol_ids: set[str], symbols_by_id: dict[str, Symbol]) -> list[SymbolSummary]:
    summaries = [
        SymbolSummary(symbol_id=sid, qualified_name=s.qualified_name, file_path=s.file_path, kind=s.kind)
        for sid in symbol_ids
        if (s := symbols_by_id.get(sid)) is not None
    ]
    return sorted(summaries, key=lambda s: (s.file_path, s.qualified_name))


def analyze_impact(
    graph: nx.DiGraph,
    symbols_by_id: dict[str, Symbol],
    pagerank_scores: dict[str, float],
    symbol_id: str,
    depth: int = 2,
) -> ImpactReport:
    """What would need re-checking if `symbol_id` changed: its direct + transitive callers."""
    direct_ids = expand_dependencies(graph, symbol_id, depth=1, direction="incoming")
    all_ids = expand_dependencies(graph, symbol_id, depth=depth, direction="incoming")
    transitive_ids = all_ids - direct_ids

    threshold = hub_threshold(pagerank_scores)
    warning_candidates = all_ids | ({symbol_id} if symbol_id in symbols_by_id else set())
    warnings = sorted(
        (sid for sid in warning_candidates if pagerank_scores.get(sid, 0.0) > threshold),
        key=lambda sid: -pagerank_scores.get(sid, 0.0),
    )

    affected_route_count = sum(
        1 for sid in (direct_ids | transitive_ids) if (s := symbols_by_id.get(sid)) is not None and _looks_like_http_route(s)
    )

    return ImpactReport(
        symbol_id=symbol_id,
        directly_affected=_summaries(direct_ids, symbols_by_id),
        transitively_affected=_summaries(transitive_ids, symbols_by_id),
        high_centrality_warnings=warnings,
        affected_route_count=affected_route_count,
    )
