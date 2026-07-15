"""MCP tool definitions — the four Phase 4 tools plus E1's `analyze_impact`
and E3's optional `submit_feedback`
(docs/phase-4-systems.md §3, docs/loupe-extensions.md E1/E3).

Governor scoping (§3, stated explicitly since it wasn't nailed down until
this phase): `list_symbols`, `search_symbols`, `expand_dependencies`, and
`analyze_impact` all return discovery-tier content only (signatures/ids, not
full source) and never touch Phase 3's session residency/eviction logic —
their cost is small enough to treat as always-affordable. Only `get_symbol`
— the one call returning full extracted source — is governed.

Split into pure `*_impl` functions (testable directly against a manually
constructed `LoupeIndex`, no HTTP needed) and thin `@router` HTTP wrappers
that pull `index`/`session_manager` from `request.app.state` — this is
deliberate: phase-4-systems.md §10's task order builds and tests the
contracts in isolation before wiring them to a live bootstrap flow.
"""

from __future__ import annotations

import fnmatch
import re
from typing import Literal

from fastapi import APIRouter, HTTPException, Request
from pydantic import BaseModel

from loupe_core.governor.budget import symbol_extraction_cost
from loupe_core.governor.knapsack import KnapsackCandidate
from loupe_core.governor.session import HARD_CEILING, request_symbols
from loupe_core.graph.builder import EdgeType
from loupe_core.graph.impact import analyze_impact as graph_analyze_impact
from loupe_core.graph.traversal import expand_dependencies as graph_expand_dependencies
from loupe_core.parsing.schema import Symbol

from .bootstrap import LoupeIndex
from .feedback import FeedbackStore
from .session_manager import SessionManager, session_id_from_request
from .telemetry import log_tool_call

router = APIRouter()

GET_SYMBOL_RELEVANCE = 1.0  # an explicit request is maximally relevant by definition (§3)

# Addendum item (a): not exhaustive or foolproof by design — exists and is
# documented as a deliberate mitigation against extracted source content
# containing text that resembles a role marker or an injected directive,
# not a claim of airtight prompt-injection defense.
_INJECTION_PATTERNS = [
    re.compile(r"(?im)^\s*(system|assistant)\s*:\s"),
    re.compile(r"(?i)ignore (all |any )?(previous|prior|above) instructions"),
    re.compile(r"(?i)disregard (all |any )?(previous|prior|above) instructions"),
]


def sanitize_source(text: str) -> tuple[str, bool]:
    """Neutralize sequences resembling prompt-injection role markers/directives.

    Returns (sanitized_text, was_modified) so callers can log when the
    sanitizer actually stripped something, rather than a silent no-op.
    """
    modified = False
    result = text
    for pattern in _INJECTION_PATTERNS:
        new_result = pattern.sub("[REDACTED]", result)
        if new_result != result:
            modified = True
            result = new_result
    return result, modified


class SymbolSummary(BaseModel):
    symbol_id: str
    kind: str
    name: str
    qualified_name: str
    file_path: str
    line_start: int
    line_end: int
    signature: str
    docstring: str | None = None
    score: float | None = None


class GetSymbolResponse(BaseModel):
    symbol_id: str
    source: str
    already_resident: bool


class DeniedResponse(BaseModel):
    status: Literal["denied"] = "denied"
    reason: Literal["session_budget_exhausted", "exceeds_hard_ceiling"]
    suggestion: str


class ImpactReportResponse(BaseModel):
    symbol_id: str
    directly_affected: list[SymbolSummary]
    directly_affected_total: int  # real count before any max_affected truncation
    transitively_affected: list[SymbolSummary]
    transitively_affected_total: int
    high_centrality_warnings: list[SymbolSummary]  # was raw symbol_id strings — not human-readable, a real gap
    affected_route_count: int


def _to_summary(symbol: Symbol, score: float | None = None) -> SymbolSummary:
    return SymbolSummary(
        symbol_id=symbol.id,
        kind=symbol.kind.value,
        name=symbol.name,
        qualified_name=symbol.qualified_name,
        file_path=symbol.file_path,
        line_start=symbol.line_start,
        line_end=symbol.line_end,
        signature=symbol.signature,
        docstring=symbol.docstring,
        score=score,
    )


def _sorted_by_file_and_byte(symbols: list[Symbol]) -> list[Symbol]:
    """Deterministic ordering (§5): same index state -> byte-identical JSON, every call."""
    return sorted(symbols, key=lambda s: (s.file_path, s.byte_start))


def list_symbols_impl(
    index: LoupeIndex, path_or_glob: str, kind_filter: list[str] | None = None
) -> list[SymbolSummary]:
    matches = [s for s in index.symbols if fnmatch.fnmatch(s.file_path, path_or_glob)]
    if kind_filter:
        matches = [s for s in matches if s.kind.value in kind_filter]
    return [_to_summary(s) for s in _sorted_by_file_and_byte(matches)]


def search_symbols_impl(index: LoupeIndex, query: str, top_k: int = 20) -> list[SymbolSummary]:
    from loupe_core.retrieval.fusion import search as fusion_search

    results = fusion_search(query, index.lexical_index, index.semantic_index, index.graph.pagerank_scores, top_k=top_k)
    summaries = []
    for symbol_id, score in results:
        symbol = index.symbol_by_id(symbol_id)
        if symbol is not None:
            summaries.append(_to_summary(symbol, score=score))
    return summaries


def expand_dependencies_impl(
    index: LoupeIndex,
    symbol_id: str,
    depth: int = 1,
    direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    edge_type: Literal["calls", "imports", "inherits", "tests"] | None = None,
) -> list[SymbolSummary]:
    edge_type_filter = EdgeType(edge_type) if edge_type is not None else None
    reachable_ids = graph_expand_dependencies(
        index.graph.graph, symbol_id, depth=depth, direction=direction, edge_type=edge_type_filter
    )
    symbols = [s for s in (index.symbol_by_id(sid) for sid in reachable_ids) if s is not None]
    return [_to_summary(s) for s in _sorted_by_file_and_byte(symbols)]


DEFAULT_MAX_AFFECTED = 30  # real gap found via a real ~900-symbol repo: an ungoverned, uncapped
# analyze_impact call on a widely-used symbol (187 direct callers) produced a 96K-char response
# that blew past the calling tool's own output-size limit. Core's analyze_impact() still computes
# and returns the full, untruncated result (correctness); this cap is presentation-only, applied
# here, with the real totals preserved in *_total so truncation is visible, not silent.


def _capped_by_pagerank(
    summaries: list, symbols_by_id: dict[str, Symbol], pagerank_scores: dict[str, float], limit: int
) -> list[SymbolSummary]:
    ranked = sorted(summaries, key=lambda s: (-pagerank_scores.get(s.symbol_id, 0.0), s.symbol_id))
    return [_to_summary(symbols_by_id[s.symbol_id]) for s in ranked[:limit]]


def analyze_impact_impl(
    index: LoupeIndex, symbol_id: str, depth: int = 2, max_affected: int = DEFAULT_MAX_AFFECTED
) -> ImpactReportResponse:
    if index.symbol_by_id(symbol_id) is None:
        raise HTTPException(status_code=404, detail=f"unknown symbol_id: {symbol_id!r}")

    symbols_by_id = {s.id: s for s in index.symbols}
    pagerank_scores = index.graph.pagerank_scores
    report = graph_analyze_impact(index.graph.graph, symbols_by_id, pagerank_scores, symbol_id, depth=depth)

    return ImpactReportResponse(
        symbol_id=symbol_id,
        directly_affected=_capped_by_pagerank(report.directly_affected, symbols_by_id, pagerank_scores, max_affected),
        directly_affected_total=len(report.directly_affected),
        transitively_affected=_capped_by_pagerank(
            report.transitively_affected, symbols_by_id, pagerank_scores, max_affected
        ),
        transitively_affected_total=len(report.transitively_affected),
        high_centrality_warnings=[_to_summary(symbols_by_id[sid]) for sid in report.high_centrality_warnings],
        affected_route_count=report.affected_route_count,
    )


class SubmitFeedbackResponse(BaseModel):
    status: Literal["recorded"] = "recorded"


def submit_feedback_impl(
    feedback_store: FeedbackStore, retrieval_log_id: str, rating: Literal["helpful", "not_helpful"], note: str | None = None
) -> SubmitFeedbackResponse:
    # E3 (docs/loupe-extensions.md): the secondary, optional MCP-visible path
    # for Claude to solicit feedback conversationally — the dashboard's plain
    # POST /feedback (main.py) is expected to carry most real usage, but both
    # write through the same FeedbackStore, tagged by `source` so a training
    # query can tell them apart if that ever matters.
    feedback_store.submit(retrieval_log_id, rating, note, source="claude_self_report")
    return SubmitFeedbackResponse()


def get_symbol_impl(
    index: LoupeIndex,
    session_manager: SessionManager,
    session_id: str,
    symbol_id: str,
) -> GetSymbolResponse | DeniedResponse:
    symbol = index.symbol_by_id(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"unknown symbol_id: {symbol_id!r}")

    parsed_file = index.parsed_files.get(symbol.file_path)
    source_bytes = parsed_file.source_bytes if parsed_file is not None else b""
    cost = symbol_extraction_cost(symbol, source_bytes)

    session = session_manager.get_or_create(session_id)
    was_already_resident = session.eviction.is_resident(symbol_id)

    result = request_symbols(session, [KnapsackCandidate(symbol_id, GET_SYMBOL_RELEVANCE, cost)])

    if symbol_id in result.denied:
        if cost > HARD_CEILING:
            return DeniedResponse(
                reason="exceeds_hard_ceiling",
                suggestion="call expand_dependencies for a lighter signature-only view",
            )
        return DeniedResponse(
            reason="session_budget_exhausted",
            suggestion="call expand_dependencies for a lighter signature-only view, "
            "or request less-relevant symbols be evicted by continuing the conversation",
        )

    source_text = source_bytes[symbol.byte_start : symbol.byte_end].decode("utf-8")
    sanitized_source, _was_sanitized = sanitize_source(source_text)

    return GetSymbolResponse(symbol_id=symbol_id, source=sanitized_source, already_resident=was_already_resident)


# --------------------------------------------------------------------------
# Thin HTTP wrappers — pull shared state from app.state, delegate to *_impl.
# --------------------------------------------------------------------------


@router.get("/list_symbols", operation_id="list_symbols")
@log_tool_call("list_symbols")
async def list_symbols_route(request: Request, path_or_glob: str, kind_filter: str | None = None) -> list[SymbolSummary]:
    index: LoupeIndex = request.app.state.index
    kinds = kind_filter.split(",") if kind_filter else None
    return list_symbols_impl(index, path_or_glob, kind_filter=kinds)


@router.get("/search_symbols", operation_id="search_symbols")
@log_tool_call("search_symbols")
async def search_symbols_route(request: Request, query: str, top_k: int = 20) -> list[SymbolSummary]:
    index: LoupeIndex = request.app.state.index
    return search_symbols_impl(index, query, top_k=top_k)


@router.get("/get_symbol", operation_id="get_symbol")
@log_tool_call("get_symbol")
async def get_symbol_route(request: Request, symbol_id: str) -> GetSymbolResponse | DeniedResponse:
    index: LoupeIndex = request.app.state.index
    session_manager: SessionManager = request.app.state.session_manager
    session_id = session_id_from_request(request)
    return get_symbol_impl(index, session_manager, session_id, symbol_id)


@router.get("/expand_dependencies", operation_id="expand_dependencies")
@log_tool_call("expand_dependencies")
async def expand_dependencies_route(
    request: Request,
    symbol_id: str,
    depth: int = 1,
    direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    edge_type: Literal["calls", "imports", "inherits", "tests"] | None = None,
) -> list[SymbolSummary]:
    index: LoupeIndex = request.app.state.index
    return expand_dependencies_impl(index, symbol_id, depth=depth, direction=direction, edge_type=edge_type)


@router.get("/analyze_impact", operation_id="analyze_impact")
@log_tool_call("analyze_impact")
async def analyze_impact_route(
    request: Request, symbol_id: str, depth: int = 2, max_affected: int = DEFAULT_MAX_AFFECTED
) -> ImpactReportResponse:
    index: LoupeIndex = request.app.state.index
    return analyze_impact_impl(index, symbol_id, depth=depth, max_affected=max_affected)


@router.get("/submit_feedback", operation_id="submit_feedback")
@log_tool_call("submit_feedback")
async def submit_feedback_route(
    request: Request, retrieval_log_id: str, rating: Literal["helpful", "not_helpful"], note: str | None = None
) -> SubmitFeedbackResponse:
    feedback_store: FeedbackStore = request.app.state.feedback_store
    return submit_feedback_impl(feedback_store, retrieval_log_id, rating, note=note)
