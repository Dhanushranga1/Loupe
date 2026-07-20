"""Shared tree-sitter AST helpers for modules that need to re-locate a `Symbol`'s
actual AST node and walk inside it — E4's conventions miner and Phase 7's smell
detectors both need this, so it lives here rather than being defined privately
in whichever one happened to need it first (the same "extract once a second
consumer needs it" correction already applied to `looks_like_http_route`).

Node-correlation trick: `extract_symbols`'s own module docstring (Phase 0,
`parsing/extractor.py`) says its capture query + sort order is "public...
reused to re-locate each Symbol's AST node" — `symbol_nodes` re-runs that
exact query/filter and zips the result against `ParsedFile.symbols` (built
by the same call, same order) to get (tree-sitter node, Symbol) pairs
without re-deriving symbol identity from scratch.
"""

from __future__ import annotations

from dataclasses import dataclass

import tree_sitter as ts

from loupe_core.graph.builder import ParsedFile
from loupe_core.parsing.extractor import DEFINITION_QUERY_SOURCE, is_nested_in_function
from loupe_core.parsing.languages import get_language
from loupe_core.parsing.schema import Symbol


def node_text(node: ts.Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")


def find_all(node: ts.Node, types: set[str]) -> list[ts.Node]:
    found: list[ts.Node] = []
    if node.type in types:
        found.append(node)
    for child in node.children:
        found.extend(find_all(child, types))
    return found


def symbol_nodes(parsed_file: ParsedFile) -> list[tuple[ts.Node, Symbol]]:
    """(tree-sitter node, Symbol) pairs, in the exact order/filter `extract_symbols` used."""
    query = ts.Query(get_language("python"), DEFINITION_QUERY_SOURCE)
    cursor = ts.QueryCursor(query)
    captures = cursor.captures(parsed_file.tree.root_node)
    nodes = sorted(captures.get("def", []), key=lambda n: n.start_byte)
    nodes = [n for n in nodes if not is_nested_in_function(n)]
    return list(zip(nodes, parsed_file.symbols))


@dataclass(frozen=True)
class ClassFieldInfo:
    name: str
    type_text: str
    has_default: bool


def class_field_annotations(class_node: ts.Node, source_bytes: bytes) -> list[ClassFieldInfo]:
    """Top-level annotated assignments in a class body (`name: Type` or
    `name: Type = default`) — the field shape shared by Pydantic
    `BaseModel`/`BaseSettings`, SQLAlchemy declarative models (type-annotated
    `mapped_column`-style fields), and plain dataclasses. Reused by the
    zero-cost static analysis pack's E7 (config/env-var drift), E8
    (migration drift), and E9 (API contract diffing) — three real consumers
    needing the identical "what fields does this class declare, and does
    each have a default" extraction, not duplicated three times over.
    Non-annotated assignments (`x = 5`, no type) are deliberately not
    fields here — Settings/model/response-model classes always
    type-annotate their real fields; an untyped class attribute is
    something else (a class-level constant, a private cache slot).
    """
    body = class_node.child_by_field_name("body")
    if body is None:
        return []

    fields: list[ClassFieldInfo] = []
    for stmt in body.children:
        if stmt.type != "expression_statement" or not stmt.children:
            continue
        assignment = stmt.children[0]
        if assignment.type != "assignment":
            continue
        left = assignment.child_by_field_name("left")
        type_node = assignment.child_by_field_name("type")
        right = assignment.child_by_field_name("right")
        if left is None or type_node is None or left.type != "identifier":
            continue
        fields.append(
            ClassFieldInfo(
                name=node_text(left, source_bytes),
                type_text=node_text(type_node, source_bytes),
                has_default=right is not None,
            )
        )
    return fields
