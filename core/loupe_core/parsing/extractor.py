"""AST -> Symbol extraction via tree-sitter queries.

Implements the algorithm in docs/phase-0-foundations.md §4. Byte-exact,
Python-only for Phase 0 (module-level functions, classes, methods, async
functions, decorators). Nested/closure functions and lambdas are out of
scope (see phase-0-foundations.md §1): the tree-sitter query below captures
every `function_definition`/`class_definition` regardless of nesting depth,
so closures nested inside a function body are explicitly filtered back out
by `is_nested_in_function` rather than never being captured in the first
place — found via the Phase 1 smoke test (docs/phase-1-graph-theory.md §10),
where a real closure in this project's own `graph/builder.py` leaked through
as a spurious top-level symbol; none of Phase 0's own fixtures happened to
contain a function-nested-in-a-function case.

One resolved inconsistency worth recording: phase-0-foundations.md §3's
illustrative comment shows a signature ending in ":", but §4's algorithm and
§7's acceptance criteria both explicitly require the trailing colon to be
excluded. This module follows §4/§7 (no trailing colon) as the authoritative,
tested behavior.
"""

from __future__ import annotations

import inspect
from pathlib import Path

import tree_sitter as ts

from .languages import get_language, get_parser
from .schema import Symbol, SymbolKind, compute_content_hash, compute_symbol_id

# Public: Phase 1 (graph/builder.py) re-parses each file independently and
# reuses this exact query + sort order to re-locate each Symbol's AST node
# without Phase 0 needing to keep tree-sitter Tree objects alive.
DEFINITION_QUERY_SOURCE = "(function_definition) @def (class_definition) @def"


def extract_symbols(file_path: str) -> list[Symbol]:
    """Parse `file_path` and return one Symbol per top-level/class-level definition.

    `file_path` is used verbatim as `Symbol.file_path` and as part of the id
    hash — callers are responsible for passing a normalized, repo-relative,
    forward-slash path (see phase-0-foundations.md §3's design notes).
    """
    source_bytes = Path(file_path).read_bytes()
    tree = get_parser("python").parse(source_bytes)

    query = ts.Query(get_language("python"), DEFINITION_QUERY_SOURCE)
    cursor = ts.QueryCursor(query)
    captures = cursor.captures(tree.root_node)
    nodes = sorted(captures.get("def", []), key=lambda n: n.start_byte)

    symbols: list[Symbol] = []
    symbol_id_by_start_byte: dict[int, str] = {}

    for node in nodes:
        if is_nested_in_function(node):
            continue

        name = _node_text(node.child_by_field_name("name"), source_bytes)
        container = _enclosing_container(node)
        is_method = container is not None and container.type == "class_definition"
        kind = _determine_kind(node, is_method)
        qualified_name = _qualified_name(node, name, source_bytes)

        decorated = node.parent if node.parent is not None and node.parent.type == "decorated_definition" else None
        byte_start = decorated.start_byte if decorated is not None else node.start_byte
        byte_end = node.end_byte
        line_start = (decorated if decorated is not None else node).start_point.row + 1
        line_end = node.end_point.row + 1

        parent_id = symbol_id_by_start_byte.get(container.start_byte) if is_method else None
        symbol_id = compute_symbol_id(file_path, qualified_name, kind)

        symbol = Symbol(
            id=symbol_id,
            kind=kind,
            name=name,
            qualified_name=qualified_name,
            file_path=file_path,
            byte_start=byte_start,
            byte_end=byte_end,
            line_start=line_start,
            line_end=line_end,
            signature=_extract_signature(node, source_bytes),
            docstring=_extract_docstring(node, source_bytes),
            decorators=_extract_decorators(decorated, source_bytes) if decorated is not None else [],
            parent_id=parent_id,
            content_hash=compute_content_hash(source_bytes[byte_start:byte_end]),
        )
        symbols.append(symbol)
        symbol_id_by_start_byte[node.start_byte] = symbol_id

    return symbols


def _determine_kind(node: ts.Node, is_method: bool) -> SymbolKind:
    """function/async_function/method/class — methods are never distinguished as async."""
    if node.type == "class_definition":
        return SymbolKind.CLASS
    if is_method:
        return SymbolKind.METHOD
    if node.children and node.children[0].type == "async":
        return SymbolKind.ASYNC_FUNCTION
    return SymbolKind.FUNCTION


def _enclosing_container(node: ts.Node) -> ts.Node | None:
    """The class_definition/function_definition whose body directly contains `node`.

    Returns None for a module-level definition. Skips decorated_definition
    wrappers at every level, since decoration doesn't change nesting.
    """
    parent = node.parent
    if parent is not None and parent.type == "decorated_definition":
        parent = parent.parent
    if parent is None or parent.type != "block":
        return None
    grandparent = parent.parent
    if grandparent is not None and grandparent.type == "decorated_definition":
        grandparent = grandparent.parent
    return grandparent


def is_nested_in_function(node: ts.Node) -> bool:
    """True if any enclosing definition is a function — i.e. `node` is a closure."""
    container = _enclosing_container(node)
    while container is not None:
        if container.type == "function_definition":
            return True
        container = _enclosing_container(container)
    return False


def _qualified_name(node: ts.Node, name: str, source_bytes: bytes) -> str:
    """Join enclosing class names with the symbol's own name via '.'."""
    parts = [name]
    container = _enclosing_container(node)
    while container is not None and container.type == "class_definition":
        class_name = _node_text(container.child_by_field_name("name"), source_bytes)
        parts.insert(0, class_name)
        container = _enclosing_container(container)
    return ".".join(parts)


def _extract_signature(node: ts.Node, source_bytes: bytes) -> str:
    """Verbatim source from the start of the def/class line up to (excluding) the trailing ':'."""
    colon = next(c for c in node.children if c.type == ":")
    return source_bytes[node.start_byte : colon.start_byte].decode("utf-8").rstrip()


def _extract_docstring(node: ts.Node, source_bytes: bytes) -> str | None:
    """The cleaned (dedented, stripped) docstring, if the body's first statement is a bare string."""
    body = node.child_by_field_name("body")
    if body is None or body.child_count == 0:
        return None
    first_statement = body.children[0]
    if first_statement.type != "expression_statement" or first_statement.child_count == 0:
        return None
    string_node = first_statement.children[0]
    if string_node.type != "string":
        return None
    content = "".join(
        _node_text(child, source_bytes) for child in string_node.children if child.type == "string_content"
    )
    return inspect.cleandoc(content) if content else ""


def _extract_decorators(decorated_node: ts.Node, source_bytes: bytes) -> list[str]:
    """Exact source text of each decorator's expression (no leading '@'), in written order."""
    decorators = []
    for child in decorated_node.children:
        if child.type != "decorator":
            continue
        expr = next(c for c in child.children if c.type != "@")
        decorators.append(_node_text(expr, source_bytes))
    return decorators


def _node_text(node: ts.Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")
