"""FastAPI static-analysis smell detection (docs/PhaseX/phase-7-fastapi-adapter-smells.md).

Seven named categories (a-g below), each backed by a real, specific
technique named in the spec — no vague pattern matching — plus an eighth,
explicitly *not* a new technique: error-handling/docstring inconsistency,
which calls directly into E4's existing `conventions/mining.py` rather than
re-implementing majority-pattern detection a second time. (The spec's own
section 1 says "seven categories" but its acceptance criteria separately
require the conventions-reuse findings too — recorded here as a real
discrepancy in the source doc, resolved by building both rather than
picking one reading and silently dropping the other.)

Detection and reporting only, the same "detection, not enforcement"
boundary E4 already drew for conventions — nothing here ever rewrites code.

Node-correlation: reuses `conventions/mining.py`'s `_symbol_nodes` helper
(the same `extract_symbols` capture-query/zip trick Phase 0's own module
docstring calls out as meant for exactly this) to get (AST node, Symbol)
pairs for the body-level checks (c, d, e).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import networkx as nx
import tree_sitter as ts

from loupe_core.adapters.fastapi.routes import looks_like_http_route
from loupe_core.conventions.mining import mine_conventions
from loupe_core.graph.builder import EdgeType, ParsedFile, UnresolvedReference
from loupe_core.graph.impact import hub_threshold
from loupe_core.parsing.ast_utils import find_all as _find_all
from loupe_core.parsing.ast_utils import node_text as _node_text
from loupe_core.parsing.ast_utils import symbol_nodes as _symbol_nodes
from loupe_core.parsing.schema import Symbol, SymbolKind

Severity = Literal["info", "warning", "high"]
Category = Literal[
    "missing_response_model",
    "untyped_params",
    "blocking_call_in_async",
    "business_logic_in_handler",
    "n_plus_one",
    "circular_dependency",
    "god_object",
    "convention_violation",
]

ALL_CATEGORIES: tuple[Category, ...] = (
    "missing_response_model",
    "untyped_params",
    "blocking_call_in_async",
    "business_logic_in_handler",
    "n_plus_one",
    "circular_dependency",
    "god_object",
    "convention_violation",
)

_FUNCTION_KINDS = {SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.ASYNC_FUNCTION}


@dataclass
class SmellFinding:
    category: Category
    symbol_id: str
    qualified_name: str
    file_path: str
    message: str
    severity: Severity


def _route_handler_nodes(parsed_files: list[ParsedFile]) -> list[tuple[ts.Node, Symbol, ParsedFile]]:
    return [
        (node, symbol, pf)
        for pf in parsed_files
        for node, symbol in _symbol_nodes(pf)
        if symbol.kind in _FUNCTION_KINDS and looks_like_http_route(symbol)
    ]


# --------------------------------------------------------------------------
# a. Missing response model / return type annotation — pure signature inspection
# --------------------------------------------------------------------------


def check_missing_response_model(symbols_by_id: dict[str, Symbol]) -> list[SmellFinding]:
    findings = []
    for symbol in symbols_by_id.values():
        if not looks_like_http_route(symbol):
            continue
        has_response_model = any("response_model" in d for d in symbol.decorators)
        has_return_annotation = "->" in symbol.signature
        if not has_response_model and not has_return_annotation:
            findings.append(
                SmellFinding(
                    category="missing_response_model",
                    symbol_id=symbol.id,
                    qualified_name=symbol.qualified_name,
                    file_path=symbol.file_path,
                    message="Route handler has neither a response_model on its decorator nor a return type annotation.",
                    severity="warning",
                )
            )
    return findings


# --------------------------------------------------------------------------
# b. Untyped route parameters — signature inspection against the real AST
#    parameter list, not the pre-rendered signature text (handles typed vs.
#    untyped vs. defaulted parameters correctly, including `x: dict = None`).
# --------------------------------------------------------------------------

_UNTYPED_ANNOTATION_NAMES = {"dict", "Dict", "Any"}
_IGNORED_PARAM_NAMES = {"self", "cls"}


def _param_is_untyped(child: ts.Node, source_bytes: bytes) -> bool:
    if child.type == "identifier":
        return _node_text(child, source_bytes) not in _IGNORED_PARAM_NAMES
    if child.type == "default_parameter":
        # a bare name with a default value and no annotation at all, e.g. `name="x"`
        return True
    if child.type in ("typed_parameter", "typed_default_parameter"):
        type_node = child.child_by_field_name("type")
        if type_node is None:
            return False
        return _node_text(type_node, source_bytes) in _UNTYPED_ANNOTATION_NAMES
    return False


def check_untyped_params(parsed_files: list[ParsedFile]) -> list[SmellFinding]:
    findings = []
    for node, symbol, pf in _route_handler_nodes(parsed_files):
        params_node = node.child_by_field_name("parameters")
        if params_node is None:
            continue
        if any(_param_is_untyped(child, pf.source_bytes) for child in params_node.children):
            findings.append(
                SmellFinding(
                    category="untyped_params",
                    symbol_id=symbol.id,
                    qualified_name=symbol.qualified_name,
                    file_path=symbol.file_path,
                    message="Route handler has a parameter typed as a bare dict/Any (or with no annotation) instead of a Pydantic schema.",
                    severity="warning",
                )
            )
    return findings


# --------------------------------------------------------------------------
# c. Blocking synchronous calls inside an async def handler — blocklist-based
#    call-target check. Matched against unresolved-external raw call text,
#    not resolved graph edges: Phase 1 deliberately never creates a CALLS
#    edge to a stdlib/third-party target (see graph/builder.py's own "no
#    guessing on external calls" design) — every one of these blocking
#    calls (time.sleep, requests.get, ...) is exactly that case, so it
#    shows up in `graph.unresolved` with reason "external", never as a
#    resolved edge. Checking resolved edges alone would silently find
#    nothing here; verified empirically against a real fixture before
#    writing this, not assumed from the spec's own (incomplete) wording.
# --------------------------------------------------------------------------

BLOCKING_CALL_PATTERNS = {
    "time.sleep",
    "requests.get",
    "requests.post",
    "requests.put",
    "requests.delete",
    "requests.patch",
    "urllib.request.urlopen",
    "psycopg2.connect",
    "sqlite3.connect",
    "socket.socket",
}


def check_blocking_calls_in_async(
    unresolved: list[UnresolvedReference], symbols_by_id: dict[str, Symbol]
) -> list[SmellFinding]:
    findings = []
    for ref in unresolved:
        caller = symbols_by_id.get(ref.from_symbol_id)
        if caller is None or caller.kind != SymbolKind.ASYNC_FUNCTION:
            continue
        if ref.raw_expression in BLOCKING_CALL_PATTERNS:
            findings.append(
                SmellFinding(
                    category="blocking_call_in_async",
                    symbol_id=caller.id,
                    qualified_name=caller.qualified_name,
                    file_path=caller.file_path,
                    message=f"Blocking call `{ref.raw_expression}(...)` inside an async def handler blocks the whole event loop.",
                    severity="high",
                )
            )
    return findings


# --------------------------------------------------------------------------
# d. Business logic embedded in a route handler — cyclomatic complexity +
#    call-target diversity, combined. Call-target diversity is counted from
#    *resolved* CALLS edges only, deliberately: a resolved edge is by
#    construction an in-repo, non-framework target (see check c's note —
#    framework/stdlib calls essentially never resolve), so this needs no
#    separate framework-noise exclusion list.
# --------------------------------------------------------------------------

_COMPLEXITY_NODE_TYPES = {
    "if_statement",
    "elif_clause",
    "for_statement",
    "while_statement",
    "except_clause",
    "boolean_operator",
    "conditional_expression",
}
COMPLEXITY_THRESHOLD = 5  # McCabe cyclomatic complexity strictly above this
CALL_DIVERSITY_THRESHOLD = 4  # distinct direct in-repo call targets at or above this


def _cyclomatic_complexity(node: ts.Node) -> int:
    return 1 + len(_find_all(node, _COMPLEXITY_NODE_TYPES))


def check_business_logic_in_handler(parsed_files: list[ParsedFile], graph: nx.DiGraph) -> list[SmellFinding]:
    findings = []
    for node, symbol, _pf in _route_handler_nodes(parsed_files):
        body = node.child_by_field_name("body")
        if body is None:
            continue
        complexity = _cyclomatic_complexity(body)
        call_targets = {v for _u, v, data in graph.out_edges(symbol.id, data=True) if data.get("edge_type") == EdgeType.CALLS}
        if complexity > COMPLEXITY_THRESHOLD and len(call_targets) >= CALL_DIVERSITY_THRESHOLD:
            findings.append(
                SmellFinding(
                    category="business_logic_in_handler",
                    symbol_id=symbol.id,
                    qualified_name=symbol.qualified_name,
                    file_path=symbol.file_path,
                    message=(
                        f"Route handler has complex control flow (cyclomatic complexity {complexity}) and calls "
                        f"{len(call_targets)} distinct symbols directly — looks like business logic that belongs "
                        "in a service layer, not inline in the route."
                    ),
                    severity="warning",
                )
            )
    return findings


# --------------------------------------------------------------------------
# e. N+1 query pattern — loop-nested call-site detection. A textual check on
#    the call's attribute name within a loop's subtree, the same static
#    pattern real N+1 linters use, not full call resolution.
# --------------------------------------------------------------------------

DB_QUERY_METHOD_NAMES = {"query", "execute", "filter", "filter_by", "find", "select", "fetchone", "fetchall"}
_LOOP_NODE_TYPES = {"for_statement", "while_statement"}


def _call_callee_name(call: ts.Node, source_bytes: bytes) -> str | None:
    fn = call.child_by_field_name("function")
    if fn is None:
        return None
    if fn.type == "attribute":
        attr = fn.child_by_field_name("attribute")
        return _node_text(attr, source_bytes) if attr is not None else None
    if fn.type == "identifier":
        return _node_text(fn, source_bytes)
    return None


def check_n_plus_one(parsed_files: list[ParsedFile]) -> list[SmellFinding]:
    findings = []
    for pf in parsed_files:
        for node, symbol in _symbol_nodes(pf):
            if symbol.kind not in _FUNCTION_KINDS:
                continue
            body = node.child_by_field_name("body")
            if body is None:
                continue
            for loop in _find_all(body, _LOOP_NODE_TYPES):
                call_names = {_call_callee_name(c, pf.source_bytes) for c in _find_all(loop, {"call"})}
                if call_names & DB_QUERY_METHOD_NAMES:
                    findings.append(
                        SmellFinding(
                            category="n_plus_one",
                            symbol_id=symbol.id,
                            qualified_name=symbol.qualified_name,
                            file_path=symbol.file_path,
                            message="A DB-query-shaped call is made inside a loop — likely one query per iteration (N+1) instead of one batched query.",
                            severity="warning",
                        )
                    )
                    break  # one finding per symbol, even with multiple offending loops
    return findings


# --------------------------------------------------------------------------
# f. Circular imports / circular dependencies — cycle detection over the
#    same directed call graph Phase 1 already builds (the circular_a/
#    circular_b fixtures from Phase 1's own test suite are real instances
#    of exactly this). Plain cycle enumeration via networkx; Tarjan's SCC
#    algorithm is the named, more-scalable alternative if this graph ever
#    needs to handle repos where naive cycle enumeration becomes a problem.
# --------------------------------------------------------------------------


def check_circular_dependencies(graph: nx.DiGraph, symbols_by_id: dict[str, Symbol]) -> list[SmellFinding]:
    findings = []
    seen_symbol_ids: set[str] = set()
    for cycle in nx.simple_cycles(graph):
        if len(cycle) < 2:
            continue  # a single self-recursive edge isn't a circular-dependency smell
        cycle_names = [symbols_by_id[sid].qualified_name for sid in cycle if sid in symbols_by_id]
        for sid in cycle:
            symbol = symbols_by_id.get(sid)
            if symbol is None or sid in seen_symbol_ids:
                continue
            seen_symbol_ids.add(sid)
            findings.append(
                SmellFinding(
                    category="circular_dependency",
                    symbol_id=sid,
                    qualified_name=symbol.qualified_name,
                    file_path=symbol.file_path,
                    message=f"Part of a circular call dependency: {' -> '.join(cycle_names)} -> {cycle_names[0]}.",
                    severity="warning",
                )
            )
    return findings


# --------------------------------------------------------------------------
# g. God-object / overloaded hub detection — direct reuse of E1's
#    hub_threshold (mean + 1 stdev over PageRank), not a second definition.
# --------------------------------------------------------------------------


def check_god_object(pagerank_scores: dict[str, float], symbols_by_id: dict[str, Symbol]) -> list[SmellFinding]:
    threshold = hub_threshold(pagerank_scores)
    findings = []
    for sid, score in pagerank_scores.items():
        if score <= threshold:
            continue
        symbol = symbols_by_id.get(sid)
        if symbol is None:
            continue
        findings.append(
            SmellFinding(
                category="god_object",
                symbol_id=sid,
                qualified_name=symbol.qualified_name,
                file_path=symbol.file_path,
                message=f"PageRank ({score:.4f}) is a statistical outlier relative to the rest of this repo's graph — a potential god-object/overloaded hub.",
                severity="high",
            )
        )
    return findings


# --------------------------------------------------------------------------
# Conventions reuse — not a new technique. Calls directly into E4's
# mine_error_handling/mine_docstrings; verified in tests by monkeypatching
# those exact functions and checking they were actually invoked, not
# reimplemented.
# --------------------------------------------------------------------------


def check_conventions(parsed_files: list[ParsedFile], symbols_by_id: dict[str, Symbol]) -> list[SmellFinding]:
    report = mine_conventions(parsed_files)
    findings = []
    for sid in report.error_handling.violating_symbol_ids:
        symbol = symbols_by_id.get(sid)
        if symbol is None:
            continue
        findings.append(
            SmellFinding(
                category="convention_violation",
                symbol_id=sid,
                qualified_name=symbol.qualified_name,
                file_path=symbol.file_path,
                message=f"Error-handling pattern deviates from this repo's majority pattern ({report.error_handling.majority_pattern!r}).",
                severity="info",
            )
        )
    for sid in report.docstrings.missing_symbol_ids:
        symbol = symbols_by_id.get(sid)
        if symbol is None:
            continue
        findings.append(
            SmellFinding(
                category="convention_violation",
                symbol_id=sid,
                qualified_name=symbol.qualified_name,
                file_path=symbol.file_path,
                message="Public symbol has no docstring, unlike most of this repo's public API.",
                severity="info",
            )
        )
    return findings


# --------------------------------------------------------------------------
# Combined entry point
# --------------------------------------------------------------------------


def find_code_smells(
    parsed_files: list[ParsedFile],
    graph: nx.DiGraph,
    unresolved: list[UnresolvedReference],
    pagerank_scores: dict[str, float],
    category: Category | None = None,
) -> list[SmellFinding]:
    """Run one, several, or all seven-plus-one smell checks, optionally filtered to
    a single `category`. Detection and reporting only — nothing here fixes anything."""
    symbols_by_id = {s.id: s for pf in parsed_files for s in pf.symbols}

    all_findings: list[SmellFinding] = []
    if category is None or category == "missing_response_model":
        all_findings += check_missing_response_model(symbols_by_id)
    if category is None or category == "untyped_params":
        all_findings += check_untyped_params(parsed_files)
    if category is None or category == "blocking_call_in_async":
        all_findings += check_blocking_calls_in_async(unresolved, symbols_by_id)
    if category is None or category == "business_logic_in_handler":
        all_findings += check_business_logic_in_handler(parsed_files, graph)
    if category is None or category == "n_plus_one":
        all_findings += check_n_plus_one(parsed_files)
    if category is None or category == "circular_dependency":
        all_findings += check_circular_dependencies(graph, symbols_by_id)
    if category is None or category == "god_object":
        all_findings += check_god_object(pagerank_scores, symbols_by_id)
    if category is None or category == "convention_violation":
        all_findings += check_conventions(parsed_files, symbols_by_id)

    return all_findings
