"""E9 — API contract diffing (docs/PhaseX/zero-cost-static-analysis-pack.md).

Reuses the exact structured/semantic diffing technique already built for
`claude_md_generator` (docs/PhaseX/claude-md-generator.md §3) — comparing
the *underlying data* between two snapshots, not rendered text — pointed at
route contract shapes instead of conventions/architecture data.

Scope, decided explicitly: tracks each route's HTTP method, response-model
name, status code, and response-model *required* fields (fields with no
default). A field disappearing from a response model is what actually
breaks client code that reads it; a field's own required/optional status
barely matters for a *response* model specifically (the client never
supplies it), so "narrowed" here means "used to be a guaranteed-present
field, now isn't" — tracking only the required-field set captures exactly
that, and nothing about a newly-added field (required or not) ever counts
as removed from that set. Not covered here, an honest scope limit: a
route's own *request* parameters becoming newly required (no default where
one existed before) — that needs request-parameter default extraction,
a genuinely separate capability from response-model field extraction.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loupe_core.adapters.fastapi.routes import HTTP_METHODS, looks_like_http_route
from loupe_core.graph.builder import ParsedFile
from loupe_core.parsing.ast_utils import class_field_annotations, symbol_nodes
from loupe_core.parsing.schema import Symbol, SymbolKind

_METHOD_PATTERN = re.compile(r"^\w+\.(" + "|".join(HTTP_METHODS) + r")\(")
_RESPONSE_MODEL_PATTERN = re.compile(r"response_model\s*=\s*(\w+)")
_STATUS_CODE_PATTERN = re.compile(r"status_code\s*=\s*(\d+)")
_RETURN_ANNOTATION_PATTERN = re.compile(r"->\s*([\w\[\], .]+?)\s*:?\s*$")

DEFAULT_STATUS_CODE = 200  # FastAPI's own real default when a route sets none explicitly


@dataclass(frozen=True)
class RouteContract:
    qualified_name: str
    file_path: str
    method: str | None
    response_model_name: str | None
    status_code: int
    required_fields: frozenset[str]


@dataclass(frozen=True)
class BreakingChange:
    qualified_name: str
    description: str


def _extract_method(symbol: Symbol) -> str | None:
    for decorator in symbol.decorators:
        match = _METHOD_PATTERN.match(decorator)
        if match:
            return match.group(1)
    return None


def _extract_response_model_name(symbol: Symbol) -> str | None:
    for decorator in symbol.decorators:
        match = _RESPONSE_MODEL_PATTERN.search(decorator)
        if match:
            return match.group(1)
    match = _RETURN_ANNOTATION_PATTERN.search(symbol.signature)
    return match.group(1).strip() if match else None


def _extract_status_code(symbol: Symbol) -> int:
    for decorator in symbol.decorators:
        match = _STATUS_CODE_PATTERN.search(decorator)
        if match:
            return int(match.group(1))
    return DEFAULT_STATUS_CODE


def extract_route_contracts(parsed_files: list[ParsedFile]) -> dict[str, RouteContract]:
    """One `RouteContract` per route handler, keyed by `qualified_name` — not
    `symbol_id`, which isn't a stable identity across two different commits'
    parses the way E9's own diffing needs (a route's byte range shifts on
    any nearby edit even when the route itself is unchanged).
    """
    symbols: list[Symbol] = []
    class_nodes_by_name: dict[str, tuple] = {}
    for pf in parsed_files:
        for node, symbol in symbol_nodes(pf):
            symbols.append(symbol)
            if symbol.kind == SymbolKind.CLASS:
                class_nodes_by_name[symbol.name] = (node, pf.source_bytes)

    contracts: dict[str, RouteContract] = {}
    for symbol in symbols:
        if not looks_like_http_route(symbol):
            continue

        response_model_name = _extract_response_model_name(symbol)
        required_fields: frozenset[str] = frozenset()
        if response_model_name and response_model_name in class_nodes_by_name:
            node, source_bytes = class_nodes_by_name[response_model_name]
            fields = class_field_annotations(node, source_bytes)
            required_fields = frozenset(f.name for f in fields if not f.has_default)

        contracts[symbol.qualified_name] = RouteContract(
            qualified_name=symbol.qualified_name,
            file_path=symbol.file_path,
            method=_extract_method(symbol),
            response_model_name=response_model_name,
            status_code=_extract_status_code(symbol),
            required_fields=required_fields,
        )
    return contracts


def diff_contracts(old: dict[str, RouteContract], new: dict[str, RouteContract]) -> list[BreakingChange]:
    """Structured diff between two contract snapshots — the underlying data,
    not rendered text, so the result names the specific change. Flags: a
    route removed entirely, a required response-model field removed, and a
    changed status code. A route or field's mere *addition* never appears
    here — additive changes are, by construction, not breaking.
    """
    changes: list[BreakingChange] = []

    for name, old_contract in sorted(old.items()):
        new_contract = new.get(name)
        if new_contract is None:
            changes.append(BreakingChange(name, "route removed"))
            continue

        removed_fields = old_contract.required_fields - new_contract.required_fields
        for field in sorted(removed_fields):
            changes.append(BreakingChange(name, f"required field {field!r} removed from the response model"))

        if old_contract.status_code != new_contract.status_code:
            changes.append(
                BreakingChange(name, f"status code changed: {old_contract.status_code} -> {new_contract.status_code}")
            )

    return changes
