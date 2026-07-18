"""Auto-derived coding conventions (docs/loupe-extensions.md E4).

Genuinely new logic, not a reuse of an earlier phase's algorithm: Phase 0
only extracts definition-level structure, and Phase 1's call resolution
only ever looks *at* a call expression to resolve it — nothing before E4
mines statement-level patterns inside a function body for their own sake.
Deliberately narrow scope, exactly the three categories the spec names, not
an open-ended pattern miner.

Exposed as an MCP *Resource* (`conventions://summary`), not a Tool — see
`server/app/main.py`'s resource registration for why that's a deliberate,
zero-tool-count-cost choice (a periodic whole-repo report is what MCP's
Resource primitive is for, not a per-query lookup).

Node-correlation trick: see `parsing/ast_utils.py`'s module docstring —
`symbol_nodes`/`find_all`/`node_text` used here were originally defined
privately in this module, promoted to a shared location once Phase 7's
smell detectors needed the identical mechanism (the same "extract once a
second consumer needs it" correction already applied elsewhere in this
project, e.g. `looks_like_http_route`).
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import tree_sitter as ts

from loupe_core.graph.builder import ParsedFile
from loupe_core.parsing.ast_utils import find_all as _find_all
from loupe_core.parsing.ast_utils import node_text as _node_text
from loupe_core.parsing.ast_utils import symbol_nodes as _symbol_nodes
from loupe_core.parsing.schema import Symbol, SymbolKind

_LOGGING_METHODS = {"debug", "info", "warning", "error", "exception", "critical"}
_FUNCTION_KINDS = {SymbolKind.FUNCTION, SymbolKind.METHOD, SymbolKind.ASYNC_FUNCTION}


# --------------------------------------------------------------------------
# 1. Error-handling convention
# --------------------------------------------------------------------------


@dataclass
class ErrorHandlingConvention:
    majority_pattern: str | None
    violation_count: int = 0
    violating_symbol_ids: list[str] = field(default_factory=list)


def _exception_type_text(except_clause: ts.Node, source_bytes: bytes) -> str:
    for child in except_clause.children:
        if child.type == "as_pattern":
            return _node_text(child.children[0], source_bytes)
        if child.type in ("identifier", "tuple", "attribute"):
            return _node_text(child, source_bytes)
    return "bare"


def _logging_pattern_in(block: ts.Node, source_bytes: bytes) -> str:
    for call in _find_all(block, {"call"}):
        fn = call.child_by_field_name("function")
        if fn is None:
            continue
        if fn.type == "attribute":
            attr = fn.child_by_field_name("attribute")
            if attr is not None and _node_text(attr, source_bytes) in _LOGGING_METHODS:
                return _node_text(fn, source_bytes)
        elif fn.type == "identifier" and _node_text(fn, source_bytes) == "print":
            return "print"
    return "none"


def _error_handling_pattern(except_clause: ts.Node, source_bytes: bytes) -> str:
    exc_type = _exception_type_text(except_clause, source_bytes)
    # except_clause has no named "body" field (verified against the real
    # grammar) — its block is just the last "block"-typed child.
    body = next((c for c in except_clause.children if c.type == "block"), None)
    logging_pattern = _logging_pattern_in(body, source_bytes) if body is not None else "none"
    return f"except {exc_type}: {logging_pattern}"


def mine_error_handling(parsed_files: list[ParsedFile]) -> ErrorHandlingConvention:
    """Majority (exception type, logging call) pattern across every `except` clause
    repo-wide, and which functions deviate from it — one vote per function, using
    its *first* except clause, so a function with several except blocks doesn't
    outvote functions with just one."""
    pattern_by_symbol: dict[str, str] = {}

    for parsed_file in parsed_files:
        for node, symbol in _symbol_nodes(parsed_file):
            if symbol.kind not in _FUNCTION_KINDS:
                continue
            except_clauses = _find_all(node, {"except_clause"})
            if not except_clauses:
                continue
            pattern_by_symbol[symbol.id] = _error_handling_pattern(except_clauses[0], parsed_file.source_bytes)

    if not pattern_by_symbol:
        return ErrorHandlingConvention(majority_pattern=None)

    counts = Counter(pattern_by_symbol.values())
    majority_pattern, _ = counts.most_common(1)[0]
    violating_symbol_ids = sorted(sid for sid, pattern in pattern_by_symbol.items() if pattern != majority_pattern)

    return ErrorHandlingConvention(
        majority_pattern=majority_pattern,
        violation_count=len(violating_symbol_ids),
        violating_symbol_ids=violating_symbol_ids,
    )


# --------------------------------------------------------------------------
# 2. Docstring convention
# --------------------------------------------------------------------------


@dataclass
class DocstringConvention:
    coverage_pct: float
    dominant_style: str  # "google" | "numpy" | "plain" | "none"
    missing_symbol_ids: list[str] = field(default_factory=list)  # public symbols with no docstring at all


def _docstring_style(docstring: str) -> str:
    lines = docstring.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("Args:") or stripped.startswith("Returns:") or stripped.startswith("Raises:"):
            return "google"
        if stripped == "Parameters" and i + 1 < len(lines) and set(lines[i + 1].strip()) <= {"-"} and lines[i + 1].strip():
            return "numpy"
    return "plain"


def mine_docstrings(parsed_files: list[ParsedFile]) -> DocstringConvention:
    public_symbols = [
        s
        for pf in parsed_files
        for s in pf.symbols
        if s.kind in _FUNCTION_KINDS | {SymbolKind.CLASS} and not s.name.startswith("_")
    ]
    if not public_symbols:
        return DocstringConvention(coverage_pct=0.0, dominant_style="none")

    documented = [s for s in public_symbols if s.docstring]
    missing_symbol_ids = sorted(s.id for s in public_symbols if not s.docstring)
    coverage_pct = 100.0 * len(documented) / len(public_symbols)

    if not documented:
        return DocstringConvention(coverage_pct=0.0, dominant_style="none", missing_symbol_ids=missing_symbol_ids)

    style_counts = Counter(_docstring_style(s.docstring) for s in documented)
    dominant_style, _ = style_counts.most_common(1)[0]

    return DocstringConvention(
        coverage_pct=coverage_pct, dominant_style=dominant_style, missing_symbol_ids=missing_symbol_ids
    )


# --------------------------------------------------------------------------
# 3. Import style convention
# --------------------------------------------------------------------------


@dataclass
class ImportConvention:
    dominant_style: str  # "relative" | "absolute"
    relative_count: int = 0
    absolute_count: int = 0


def mine_imports(parsed_files: list[ParsedFile]) -> ImportConvention:
    relative_count = 0
    absolute_count = 0

    for parsed_file in parsed_files:
        for node in _find_all(parsed_file.tree.root_node, {"import_statement", "import_from_statement"}):
            if node.type == "import_statement":
                absolute_count += 1
            else:
                has_relative = any(child.type == "relative_import" for child in node.children)
                if has_relative:
                    relative_count += 1
                else:
                    absolute_count += 1

    dominant_style = "relative" if relative_count > absolute_count else "absolute"
    return ImportConvention(dominant_style=dominant_style, relative_count=relative_count, absolute_count=absolute_count)


# --------------------------------------------------------------------------
# Combined report
# --------------------------------------------------------------------------


@dataclass
class ConventionsReport:
    error_handling: ErrorHandlingConvention
    docstrings: DocstringConvention
    imports: ImportConvention


def mine_conventions(parsed_files: list[ParsedFile]) -> ConventionsReport:
    return ConventionsReport(
        error_handling=mine_error_handling(parsed_files),
        docstrings=mine_docstrings(parsed_files),
        imports=mine_imports(parsed_files),
    )
