"""MCP tool definitions — the four Phase 4 tools plus E1's `analyze_impact`,
E3's optional `submit_feedback`, and Phase 7's `find_code_smells`
(docs/phase-4-systems.md §3, docs/loupe-extensions.md E1/E3,
docs/PhaseX/phase-7-fastapi-adapter-smells.md).

Governor scoping (§3, stated explicitly since it wasn't nailed down until
this phase): `list_symbols`, `search_symbols`, `expand_dependencies`,
`analyze_impact`, and `find_code_smells` all return discovery-tier content
only (signatures/ids/findings, not full source) and never touch Phase 3's
session residency/eviction logic — their cost is small enough to treat as
always-affordable. Only `get_symbol` — the one call returning full extracted
source — is governed.

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

from loupe_core.adapters.fastapi.smells import ALL_CATEGORIES, Category
from loupe_core.adapters.fastapi.smells import find_code_smells as graph_find_code_smells
from loupe_core.governor.budget import (
    ancestor_context_text,
    estimate_tokens,
    symbol_extraction_cost,
    symbol_extraction_marginal_cost,
)
from loupe_core.governor.knapsack import KnapsackCandidate
from loupe_core.governor.session import HARD_CEILING, request_symbols
from loupe_core.graph.builder import EdgeType
from loupe_core.graph.impact import analyze_impact as graph_analyze_impact
from loupe_core.graph.traversal import expand_dependencies as graph_expand_dependencies
from loupe_core.parsing.schema import Symbol

from .bootstrap import LoupeIndex
from .config import LoupeConfig
from .experimental_gate import is_experimental_feature_enabled, log_experimental_usage
from .feedback import FeedbackStore
from .session_manager import SessionManager, session_id_from_request
from .telemetry import log_tool_call, record_cross_encoder_latency

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


class FileSummary(BaseModel):
    """Phase 14 §1's L2 zoom level: one line per *file*, not per symbol —
    for scanning a cluster's file layout without paying for every symbol
    signature inside it."""

    file_path: str
    symbol_count: int
    symbol_names: list[str]


def _to_file_summaries(symbols: list[Symbol]) -> list[FileSummary]:
    by_file: dict[str, list[Symbol]] = {}
    for s in symbols:
        by_file.setdefault(s.file_path, []).append(s)
    return [
        FileSummary(file_path=file_path, symbol_count=len(syms), symbol_names=sorted(s.qualified_name for s in syms))
        for file_path, syms in sorted(by_file.items())
    ]


class GetSymbolResponse(BaseModel):
    symbol_id: str
    source: str
    already_resident: bool
    related_suggestions: list[SymbolSummary] = []  # Phase 14 §3, [] when no cache/no suggestions meet MIN_SUPPORT


# Phase 14 §1's L5 zoom level, decided threshold: a symbol whose full
# extraction would cost more than this many tokens gets decomposed by
# default rather than sent whole.
L5_DECOMPOSITION_THRESHOLD_TOKENS = 400


class SymbolDecomposition(BaseModel):
    """A response-*shaping* rule, not new parsing (§1) — Phase 0 already
    extracts methods as their own symbols with a `parent_id` pointing at
    their enclosing class; this just returns the parent's own signature
    plus its children at L3 (signature-only) granularity instead of the
    full concatenated body."""

    symbol_id: str
    signature: str
    docstring: str | None
    children: list[SymbolSummary]
    related_suggestions: list[SymbolSummary] = []  # Phase 14 §3, same as GetSymbolResponse's own field


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


def _resolve_scope(
    index: LoupeIndex, scope_path: str | None, scope_symbol_id: str | None, scope_mode: str = "soft"
) -> "Scope | None":
    """None when no scope was requested at all — every scope-aware *_impl
    function treats that as "behave exactly as if scope didn't exist,"
    the same backward-compatibility guarantee `context/scope.py`'s own
    tests check at the core layer.
    """
    from loupe_core.context.scope import Scope

    if scope_path is None and scope_symbol_id is None:
        return None
    return Scope(path_prefix=scope_path, seed_symbol_id=scope_symbol_id, mode=scope_mode)  # type: ignore[arg-type]


def list_symbols_impl(
    index: LoupeIndex,
    path_or_glob: str,
    kind_filter: list[str] | None = None,
    granularity: Literal["symbol", "file_summary"] = "symbol",
    scope_path: str | None = None,
    scope_symbol_id: str | None = None,
) -> list[SymbolSummary] | list[FileSummary]:
    matches = [s for s in index.symbols if fnmatch.fnmatch(s.file_path, path_or_glob)]
    if kind_filter:
        matches = [s for s in matches if s.kind.value in kind_filter]

    # scope-aware-retrieval §3's hard/soft distinction is a *ranking* concept —
    # list_symbols has no relevance ranking to soften, it's a plain glob
    # filter, so scope here is always a hard set-intersection regardless of
    # `mode` (there is no `mode` parameter on this tool at all, deliberately —
    # a resolved, documented simplification, not an oversight).
    scope = _resolve_scope(index, scope_path, scope_symbol_id)
    if scope is not None:
        from loupe_core.context.scope import apply_hard_scope, resolve_scope_membership

        symbols_by_id = {s.id: s for s in index.symbols}
        membership = resolve_scope_membership(scope, index.graph, symbols_by_id)
        in_scope_ids = apply_hard_scope({s.id for s in matches}, membership)
        matches = [s for s in matches if s.id in in_scope_ids]

    # Phase 14 §1's L2 zoom level: one line per file, not per symbol —
    # composes with `scope` above rather than needing its own cluster-scoping
    # mechanism (typical usage is `granularity=file_summary` *and* `scope`
    # together, but nothing here requires it).
    if granularity == "file_summary":
        return _to_file_summaries(matches)

    return [_to_summary(s) for s in _sorted_by_file_and_byte(matches)]


CHURN_CACHE_FILENAME = "cache/churn.json"


def _load_churn_scores(index: LoupeIndex) -> dict[str, float] | None:
    """`None` (not `{}`) when no `loupe update-churn` cache exists yet — `fuse()`
    treats `churn_scores=None` as "no churn signal at all," whereas an empty
    dict would still build a churn ranking from every candidate's `0.0`
    fallback, introducing an arbitrary id-ordering bias from nothing. Cheap
    enough to read fresh on every call (a small JSON file) rather than cached
    on `LoupeIndex`, since churn is deliberately not tied to the reindex
    lifecycle (Phase 14 §2) — the file's content only ever changes via an
    explicit `loupe update-churn` run.
    """
    churn_path = index.loupe_dir / CHURN_CACHE_FILENAME
    if not churn_path.exists():
        return None
    import json

    return json.loads(churn_path.read_text())


CO_RETRIEVAL_CACHE_FILENAME = "cache/co_retrieval.json"
MAX_RELATED_SUGGESTIONS = 5


def _load_related_suggestions(index: LoupeIndex, symbol_id: str) -> list[SymbolSummary]:
    """§3: `related_suggestions` on `get_symbol`'s own response, signatures
    only (L3-equivalent cost, never full bodies) — cheap regardless of how
    often it fires. `[]` (not an error) when no `loupe update-suggestions`
    cache exists yet, or `symbol_id` has no suggestions meeting the minimum
    support threshold — both are "nothing to suggest," not failure states.
    """
    suggestions_path = index.loupe_dir / CO_RETRIEVAL_CACHE_FILENAME
    if not suggestions_path.exists():
        return []
    import json

    all_suggestions = json.loads(suggestions_path.read_text())
    entries = all_suggestions.get(symbol_id, [])[:MAX_RELATED_SUGGESTIONS]

    summaries = []
    for entry in entries:
        suggested_symbol = index.symbol_by_id(entry["symbol_id"])
        if suggested_symbol is not None:
            summaries.append(_to_summary(suggested_symbol))
    return summaries


def search_symbols_impl(
    index: LoupeIndex,
    query: str,
    top_k: int = 20,
    scope_path: str | None = None,
    scope_symbol_id: str | None = None,
    scope_mode: Literal["hard", "soft"] = "soft",
    config: LoupeConfig | None = None,
    llm_client: object | None = None,
) -> list[SymbolSummary]:
    """`config`/`llm_client` gate docs/PhaseX/experimental-gate-and-hyde.md's
    HyDE signal: both must be given, *and* `config.experimental` must have
    `llm_assist` and `features.hyde_query_rewrite` both true, for HyDE to run
    at all. `llm_client` is never constructed anywhere in this project's own
    server startup today (see `hyde.py`'s module docstring) — real callers
    always pass `llm_client=None`, so HyDE stays fully inert in production
    regardless of manifest config until an operator wires a real client in
    themselves. Tests inject a fake client directly.
    """
    from loupe_core.retrieval.fusion import CANDIDATE_POOL_SIZE, FINAL_TOP_K
    from loupe_core.retrieval.mmr import mmr_select
    from loupe_core.retrieval.rerank import rerank

    scope = _resolve_scope(index, scope_path, scope_symbol_id, scope_mode)
    membership: set[str] = set()
    if scope is not None:
        from loupe_core.context.scope import resolve_scope_membership

        symbols_by_id_all = {s.id: s for s in index.symbols}
        membership = resolve_scope_membership(scope, index.graph, symbols_by_id_all)

    lexical_results = index.lexical_index.query(query, top_k=CANDIDATE_POOL_SIZE)
    semantic_results = index.semantic_index.query(query, top_k=CANDIDATE_POOL_SIZE)

    # §1's hard mode: filter the candidate universe *before* Phase 2's ranking
    # pipeline runs — applied to both signals' own result pools, not a
    # post-hoc filter on the fused output.
    if scope is not None and scope.mode == "hard":
        from loupe_core.context.scope import apply_hard_scope

        lexical_ids = apply_hard_scope({sid for sid, _ in lexical_results}, membership)
        semantic_ids = apply_hard_scope({sid for sid, _ in semantic_results}, membership)
        lexical_results = [(sid, score) for sid, score in lexical_results if sid in lexical_ids]
        semantic_results = [(sid, score) for sid, score in semantic_results if sid in semantic_ids]

    # §2's soft mode: bias the centrality term's restart distribution toward
    # scope membership instead of the query's own candidates — nothing is
    # filtered out of the candidate pool at all.
    from loupe_core.context.scope import DEFAULT_IN_SCOPE_MASS

    soft_scope_kwargs = (
        {"scope_seed_ids": membership, "in_scope_mass": DEFAULT_IN_SCOPE_MASS}
        if scope is not None and scope.mode == "soft" and membership
        else {}
    )

    # docs/PhaseX/experimental-gate-and-hyde.md §6: an optional fourth RRF
    # signal, gated by both `config` and `llm_client` being provided *and*
    # the two-level manifest flag being on (see this function's docstring).
    hyde_results: list[tuple[str, float]] | None = None
    if config is not None and llm_client is not None and is_experimental_feature_enabled(config, "hyde_query_rewrite"):
        from loupe_core.retrieval.hyde import hyde_search

        hyde_outcome = hyde_search(query, llm_client, index.semantic_index, top_k=CANDIDATE_POOL_SIZE)
        hyde_results = hyde_outcome.ranked
        log_experimental_usage(
            index.loupe_dir,
            "hyde_query_rewrite",
            hyde_outcome.total_tokens,
            cost_estimate_type="measured",
            query=query,
        )

    # Retrieval-upgrades §1's 4-stage pipeline: RRF always narrows to its own fixed
    # top-20 (stage 2) — the caller's `top_k` is the *final* result count (post-MMR,
    # stage 4), not RRF's own, so it must not be threaded into `fusion_search` here.
    from loupe_core.retrieval.fusion import fuse

    rrf_results = fuse(
        lexical_results,
        semantic_results,
        index.graph.pagerank_scores,
        graph=index.graph.graph,
        top_k=FINAL_TOP_K,
        churn_scores=_load_churn_scores(index),
        hyde_results=hyde_results,
        **soft_scope_kwargs,
    )
    symbols_by_id = {}
    for symbol_id, _score in rrf_results:
        symbol = index.symbol_by_id(symbol_id)
        if symbol is not None:
            symbols_by_id[symbol_id] = symbol

    rerank_result = rerank(query, rrf_results, symbols_by_id)
    record_cross_encoder_latency(rerank_result.latency_ms)

    embeddings = {
        symbol_id: embedding
        for symbol_id, _ in rerank_result.ranked
        if (embedding := index.semantic_index.get_embedding(symbol_id)) is not None
    }
    selected = mmr_select(rerank_result.ranked, embeddings, final_top_k=top_k)

    summaries = []
    for symbol_id, score in selected:
        symbol = symbols_by_id.get(symbol_id)
        if symbol is not None:
            summaries.append(_to_summary(symbol, score=score))
    return summaries


DEFAULT_MAX_AFFECTED = 30  # real gap found via a real ~900-symbol repo: an ungoverned, uncapped
# analyze_impact call on a widely-used symbol (187 direct callers) produced a 96K-char response
# that blew past the calling tool's own output-size limit. Core's analyze_impact() still computes
# and returns the full, untruncated result (correctness); this cap is presentation-only, applied
# here, with the real totals preserved in *_total so truncation is visible, not silent.
# expand_dependencies had the exact same class of gap (found immediately after the analyze_impact
# fix, by the same real-repo caller — a high-fanout symbol's incoming edges hit 91.8K chars), so
# it shares this constant and the same cap-with-visible-total shape.


class ExpandDependenciesResponse(BaseModel):
    results: list[SymbolSummary]
    total_count: int  # real count before any max_results truncation


def _cap_symbols_by_pagerank(symbols: list[Symbol], pagerank_scores: dict[str, float], limit: int) -> list[Symbol]:
    return sorted(symbols, key=lambda s: (-pagerank_scores.get(s.id, 0.0), s.id))[:limit]


def expand_dependencies_impl(
    index: LoupeIndex,
    symbol_id: str,
    depth: int = 1,
    direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    edge_type: Literal["calls", "imports", "inherits", "tests"] | None = None,
    max_results: int = DEFAULT_MAX_AFFECTED,
    scope_path: str | None = None,
    scope_symbol_id: str | None = None,
    scope_mode: Literal["hard", "soft"] = "soft",
) -> ExpandDependenciesResponse:
    edge_type_filter = EdgeType(edge_type) if edge_type is not None else None
    reachable_ids = graph_expand_dependencies(
        index.graph.graph, symbol_id, depth=depth, direction=direction, edge_type=edge_type_filter
    )
    symbols = [s for s in (index.symbol_by_id(sid) for sid in reachable_ids) if s is not None]

    scope = _resolve_scope(index, scope_path, scope_symbol_id, scope_mode)
    if scope is None:
        capped = _cap_symbols_by_pagerank(symbols, index.graph.pagerank_scores, max_results)
        return ExpandDependenciesResponse(results=[_to_summary(s) for s in capped], total_count=len(symbols))

    from loupe_core.context.scope import apply_hard_scope, resolve_scope_membership

    symbols_by_id_all = {s.id: s for s in index.symbols}
    membership = resolve_scope_membership(scope, index.graph, symbols_by_id_all)

    if scope.mode == "hard":
        # §1's genuine isolation guarantee: the reachable set itself is
        # narrowed to scope membership, not just the capped presentation —
        # matches list_symbols'/search_symbols' hard-mode treatment.
        in_scope_ids = apply_hard_scope({s.id for s in symbols}, membership)
        symbols = [s for s in symbols if s.id in in_scope_ids]
        capped = _cap_symbols_by_pagerank(symbols, index.graph.pagerank_scores, max_results)
    else:
        # §2's soft mode: nothing is filtered out of the reachable set at all
        # (total_count still reflects the true, unscoped traversal) — scope
        # only biases *which* symbols survive the max_results cap, via the
        # same scope-seeded personalized PageRank search_symbols uses.
        from loupe_core.context.scope import scoped_personalized_pagerank

        scope_scores = (
            scoped_personalized_pagerank(index.graph, membership, {s.id for s in symbols})
            if membership
            else index.graph.pagerank_scores
        )
        capped = _cap_symbols_by_pagerank(symbols, scope_scores, max_results)

    return ExpandDependenciesResponse(results=[_to_summary(s) for s in capped], total_count=len(symbols))


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


class SmellFindingResponse(BaseModel):
    category: Category
    symbol_id: str
    qualified_name: str
    file_path: str
    message: str
    severity: Literal["info", "warning", "high"]


class FindCodeSmellsResponse(BaseModel):
    findings: list[SmellFindingResponse]
    total_count: int  # real count before any max_findings truncation


_SEVERITY_RANK = {"high": 0, "warning": 1, "info": 2}


def find_code_smells_impl(
    index: LoupeIndex, category: Category | None = None, max_findings: int = DEFAULT_MAX_AFFECTED
) -> FindCodeSmellsResponse:
    # Real gap found dogfooding this tool against Loupe's own repo: an
    # unbounded, uncapped call produced a 172K-char response (513 findings,
    # mostly convention_violation on a real ~1,700-symbol index) — the exact
    # same output-size class of bug analyze_impact/expand_dependencies had.
    # Same fix: core still computes the full, correct, untruncated list;
    # capping (sorted highest-severity first, so nothing important is lost
    # to truncation) is a presentation-only concern applied here, with the
    # real total preserved so truncation is visible, not silent.
    parsed_files = list(index.parsed_files.values())
    findings = graph_find_code_smells(
        parsed_files, index.graph.graph, index.graph.unresolved, index.graph.pagerank_scores, category=category
    )
    ranked = sorted(findings, key=lambda f: (_SEVERITY_RANK[f.severity], f.qualified_name))
    capped = ranked[:max_findings]
    return FindCodeSmellsResponse(
        findings=[
            SmellFindingResponse(
                category=f.category,
                symbol_id=f.symbol_id,
                qualified_name=f.qualified_name,
                file_path=f.file_path,
                message=f.message,
                severity=f.severity,
            )
            for f in capped
        ],
        total_count=len(findings),
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


class NoteResponse(BaseModel):
    note_id: str
    content: str
    importance: int
    turn_index: int
    timestamp: float


class SessionNotesResponse(BaseModel):
    notes: list[NoteResponse]


def _note_to_response(note) -> NoteResponse:
    return NoteResponse(
        note_id=note.note_id, content=note.content, importance=note.importance, turn_index=note.turn_index, timestamp=note.timestamp
    )


def session_notes_impl(
    store,
    action: Literal["write", "read_recent", "read_relevant", "list"],
    content: str | None = None,
    importance: int = 3,
    query: str | None = None,
    top_k: int = 5,
    limit: int = 10,
) -> SessionNotesResponse:
    """One tool, four actions (session-notes.md §4) rather than four separate
    tools — keeps the tool-count budget from growing for what's fundamentally
    one capability with a few modes.
    """
    if action == "write":
        if content is None:
            raise HTTPException(status_code=400, detail="content is required for action='write'")
        note = store.write(content, importance)
        return SessionNotesResponse(notes=[_note_to_response(note)])

    if action == "read_recent":
        return SessionNotesResponse(notes=[_note_to_response(n) for n in store.read_recent(limit=limit)])

    if action == "read_relevant":
        if query is None:
            raise HTTPException(status_code=400, detail="query is required for action='read_relevant'")
        return SessionNotesResponse(notes=[_note_to_response(n) for n in store.read_relevant(query, top_k=top_k)])

    if action == "list":
        return SessionNotesResponse(notes=[_note_to_response(n) for n in store.list_all()])

    raise HTTPException(status_code=400, detail=f"unknown action: {action!r}")


def _denied_response(cost: int) -> DeniedResponse:
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


def get_symbol_impl(
    index: LoupeIndex,
    session_manager: SessionManager,
    session_id: str,
    symbol_id: str,
    full: bool = False,
) -> GetSymbolResponse | SymbolDecomposition | DeniedResponse:
    symbol = index.symbol_by_id(symbol_id)
    if symbol is None:
        raise HTTPException(status_code=404, detail=f"unknown symbol_id: {symbol_id!r}")

    parsed_file = index.parsed_files.get(symbol.file_path)
    source_bytes = parsed_file.source_bytes if parsed_file is not None else b""
    full_cost = symbol_extraction_cost(symbol, source_bytes)

    session = session_manager.get_or_create(session_id)

    # §1's L5 decomposition: a symbol whose full body exceeds the threshold
    # gets its signature + child (method) list at L3 granularity instead —
    # only when it actually has children to decompose into; a class with no
    # methods, or a plain function, has nothing to decompose and falls
    # through to the full-body path regardless of size. `full=True` always
    # forces the complete body, overriding this rule entirely.
    children = sorted(
        (s for s in index.symbols if s.parent_id == symbol_id), key=lambda s: (s.file_path, s.byte_start)
    )
    if not full and children and full_cost > L5_DECOMPOSITION_THRESHOLD_TOKENS:
        # Charge only for what's actually returned (signature + child
        # signatures), not the full body's cost — the governor should
        # reflect real spend, not the cost of content never sent.
        decomposition_cost = estimate_tokens(symbol.signature) + sum(estimate_tokens(c.signature) for c in children)
        result = request_symbols(session, [KnapsackCandidate(symbol_id, GET_SYMBOL_RELEVANCE, decomposition_cost)])
        if symbol_id in result.denied:
            return _denied_response(decomposition_cost)
        return SymbolDecomposition(
            symbol_id=symbol_id,
            signature=symbol.signature,
            docstring=symbol.docstring,
            children=[_to_summary(c) for c in children],
            related_suggestions=_load_related_suggestions(index, symbol_id),
        )

    # §4's differential extraction: a method whose enclosing class (or other
    # ancestor) already had its shared context (signature + docstring only,
    # never its body — see budget.py's ancestor_context_text) sent and
    # charged earlier this session is charged the *marginal* cost only —
    # its own body, not a re-billed copy of context Claude already has.
    ancestor = index.symbol_by_id(symbol.parent_id) if symbol.parent_id else None
    already_charged = ancestor is not None and ancestor.id in session.shared_context_charged
    marginal_cost = symbol_extraction_marginal_cost(symbol, source_bytes, ancestor, already_charged)

    was_already_resident = session.eviction.is_resident(symbol_id)
    result = request_symbols(session, [KnapsackCandidate(symbol_id, GET_SYMBOL_RELEVANCE, marginal_cost)])

    if symbol_id in result.denied:
        return _denied_response(marginal_cost)

    source_text = source_bytes[symbol.byte_start : symbol.byte_end].decode("utf-8")
    sanitized_source, _was_sanitized = sanitize_source(source_text)

    if ancestor is not None and not already_charged:
        # First time this ancestor's shared context is sent this session —
        # prepend it so Claude has enough class-level context to understand
        # the method, and mark it charged so a sibling method later doesn't
        # pay for it again.
        sanitized_source = f"{ancestor_context_text(ancestor)}\n\n{sanitized_source}"
        session.shared_context_charged.add(ancestor.id)

    return GetSymbolResponse(
        symbol_id=symbol_id,
        source=sanitized_source,
        already_resident=was_already_resident,
        related_suggestions=_load_related_suggestions(index, symbol_id),
    )


# --------------------------------------------------------------------------
# Thin HTTP wrappers — pull shared state from app.state, delegate to *_impl.
# --------------------------------------------------------------------------


@router.get("/list_symbols", operation_id="list_symbols")
@log_tool_call("list_symbols")
async def list_symbols_route(
    request: Request,
    path_or_glob: str,
    kind_filter: str | None = None,
    granularity: Literal["symbol", "file_summary"] = "symbol",
    scope_path: str | None = None,
    scope_symbol_id: str | None = None,
) -> list[SymbolSummary] | list[FileSummary]:
    index: LoupeIndex = request.app.state.index
    kinds = kind_filter.split(",") if kind_filter else None
    return list_symbols_impl(
        index, path_or_glob, kind_filter=kinds, granularity=granularity, scope_path=scope_path, scope_symbol_id=scope_symbol_id
    )


@router.get("/search_symbols", operation_id="search_symbols")
@log_tool_call("search_symbols")
async def search_symbols_route(
    request: Request,
    query: str,
    top_k: int = 20,
    scope_path: str | None = None,
    scope_symbol_id: str | None = None,
    scope_mode: Literal["hard", "soft"] = "soft",
) -> list[SymbolSummary]:
    index: LoupeIndex = request.app.state.index
    # `hyde_llm_client` is never set in this project's own `main.py` startup
    # (see `search_symbols_impl`'s docstring) — `getattr(..., None)` means
    # HyDE stays inert on every real server today, however the manifest is
    # configured, until an operator wires a real client into `app.state`.
    config = getattr(request.app.state, "config", None)
    llm_client = getattr(request.app.state, "hyde_llm_client", None)
    return search_symbols_impl(
        index,
        query,
        top_k=top_k,
        scope_path=scope_path,
        scope_symbol_id=scope_symbol_id,
        scope_mode=scope_mode,
        config=config,
        llm_client=llm_client,
    )


@router.get("/get_symbol", operation_id="get_symbol")
@log_tool_call("get_symbol")
async def get_symbol_route(
    request: Request, symbol_id: str, full: bool = False
) -> GetSymbolResponse | SymbolDecomposition | DeniedResponse:
    index: LoupeIndex = request.app.state.index
    session_manager: SessionManager = request.app.state.session_manager
    session_id = session_id_from_request(request)
    return get_symbol_impl(index, session_manager, session_id, symbol_id, full=full)


@router.get("/expand_dependencies", operation_id="expand_dependencies")
@log_tool_call("expand_dependencies")
async def expand_dependencies_route(
    request: Request,
    symbol_id: str,
    depth: int = 1,
    direction: Literal["outgoing", "incoming", "both"] = "outgoing",
    edge_type: Literal["calls", "imports", "inherits", "tests"] | None = None,
    max_results: int = DEFAULT_MAX_AFFECTED,
    scope_path: str | None = None,
    scope_symbol_id: str | None = None,
    scope_mode: Literal["hard", "soft"] = "soft",
) -> ExpandDependenciesResponse:
    index: LoupeIndex = request.app.state.index
    return expand_dependencies_impl(
        index,
        symbol_id,
        depth=depth,
        direction=direction,
        edge_type=edge_type,
        max_results=max_results,
        scope_path=scope_path,
        scope_symbol_id=scope_symbol_id,
        scope_mode=scope_mode,
    )


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


@router.get("/session_notes", operation_id="session_notes")
@log_tool_call("session_notes")
async def session_notes_route(
    request: Request,
    action: Literal["write", "read_recent", "read_relevant", "list"],
    content: str | None = None,
    importance: int = 3,
    query: str | None = None,
    top_k: int = 5,
    limit: int = 10,
) -> SessionNotesResponse:
    from .session_notes_manager import SessionNotesManager

    session_notes_manager: SessionNotesManager = request.app.state.session_notes_manager
    session_id = session_id_from_request(request)
    store = session_notes_manager.get_or_create(session_id)
    return session_notes_impl(store, action, content=content, importance=importance, query=query, top_k=top_k, limit=limit)


@router.get("/find_code_smells", operation_id="find_code_smells")
@log_tool_call("find_code_smells")
async def find_code_smells_route(
    request: Request, category: Category | None = None, max_findings: int = DEFAULT_MAX_AFFECTED
) -> FindCodeSmellsResponse:
    index: LoupeIndex = request.app.state.index
    return find_code_smells_impl(index, category=category, max_findings=max_findings)
