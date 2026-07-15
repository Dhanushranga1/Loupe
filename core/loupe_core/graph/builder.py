"""Builds the repo-wide symbol graph from Phase 0's per-file Symbol lists.

Implements docs/phase-1-graph-theory.md. Resolution is deliberately
conservative ("best effort, no guessing" — see §1/§6): every edge created is
one the algorithm is confident about; everything else is recorded in
`LoupeGraph.unresolved`, never silently mislinked.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

import networkx as nx
import tree_sitter as ts

from loupe_core.parsing.extractor import DEFINITION_QUERY_SOURCE
from loupe_core.parsing.extractor import extract_symbols
from loupe_core.parsing.extractor import is_nested_in_function
from loupe_core.parsing.languages import get_language, get_parser
from loupe_core.parsing.schema import Symbol, SymbolKind

from .centrality import compute_pagerank

UnresolvedReason = Literal["external", "ambiguous", "no_type_inference"]


class EdgeType(str, Enum):
    CALLS = "calls"
    IMPORTS = "imports"
    INHERITS = "inherits"
    TESTS = "tests"  # added by E2 (docs/loupe-extensions.md) — added post-hoc by graph/test_linkage.py,
    # not resolved here alongside calls/imports/inherits


@dataclass
class Edge:
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: int = 1


@dataclass
class UnresolvedReference:
    from_symbol_id: str
    raw_expression: str
    reason: UnresolvedReason


@dataclass
class ParsedFile:
    file_path: str
    tree: ts.Tree
    source_bytes: bytes
    symbols: list[Symbol]


@dataclass
class LoupeGraph:
    graph: nx.DiGraph
    unresolved: list[UnresolvedReference] = field(default_factory=list)
    pagerank_scores: dict[str, float] = field(default_factory=dict)


def parse_file(file_path: str) -> ParsedFile:
    """Read + parse `file_path` fresh and pair it with Phase 0's extracted symbols."""
    source_bytes = Path(file_path).read_bytes()
    tree = get_parser("python").parse(source_bytes)
    symbols = extract_symbols(file_path)
    return ParsedFile(file_path=file_path, tree=tree, source_bytes=source_bytes, symbols=symbols)


def build_graph(parsed_files: list[ParsedFile]) -> LoupeGraph:
    """Resolve inheritance, then calls, into a DiGraph; rank it; return it all."""
    symbols_by_id, qualified_index, bare_name_index = _build_name_index(parsed_files)
    import_indexes = {pf.file_path: _build_import_index(pf) for pf in parsed_files}
    def_nodes = {pf.file_path: _map_symbols_to_nodes(pf) for pf in parsed_files}
    repo_module_stems = {Path(pf.file_path).stem for pf in parsed_files}

    graph = nx.DiGraph()
    graph.add_nodes_from(symbols_by_id)

    unresolved: list[UnresolvedReference] = []
    base_class_ids: dict[str, list[str]] = {}

    # Pass 1: INHERITS, fully resolved before any CALLS resolution (rule 1 needs it).
    for pf in parsed_files:
        for symbol in pf.symbols:
            if symbol.kind != SymbolKind.CLASS:
                continue
            node = def_nodes[pf.file_path][symbol.id]
            bases: list[str] = []
            for base_text in _extract_base_class_texts(node, pf.source_bytes):
                outcome, target_id = _resolve_bare_identifier(
                    base_text, import_indexes[pf.file_path], qualified_index, bare_name_index
                )
                if outcome == "resolved":
                    _add_edge(graph, symbol.id, target_id, EdgeType.INHERITS)
                    bases.append(target_id)
                else:
                    unresolved.append(UnresolvedReference(symbol.id, base_text, outcome))
            base_class_ids[symbol.id] = bases

    # Pass 2: CALLS, using now-complete inheritance info for rule 1 (self/cls).
    for pf in parsed_files:
        for symbol in pf.symbols:
            if symbol.kind == SymbolKind.CLASS:
                continue
            node = def_nodes[pf.file_path][symbol.id]
            body = node.child_by_field_name("body")
            if body is None:
                continue
            for call_node in _iter_call_nodes(body, root_start_byte=node.start_byte):
                callee = call_node.child_by_field_name("function")
                raw_text = _node_text(callee, pf.source_bytes)
                outcome, target_id = _resolve_call(
                    callee=callee,
                    source_bytes=pf.source_bytes,
                    caller=symbol,
                    import_index=import_indexes[pf.file_path],
                    qualified_index=qualified_index,
                    bare_name_index=bare_name_index,
                    symbols_by_id=symbols_by_id,
                    base_class_ids=base_class_ids,
                    repo_module_stems=repo_module_stems,
                )
                if outcome == "resolved":
                    _add_edge(graph, symbol.id, target_id, EdgeType.CALLS)
                else:
                    unresolved.append(UnresolvedReference(symbol.id, raw_text, outcome))

    pagerank_scores = compute_pagerank(graph)
    return LoupeGraph(graph=graph, unresolved=unresolved, pagerank_scores=pagerank_scores)


# --------------------------------------------------------------------------
# Name index: qualified_index, bare_name_index (§5)
# --------------------------------------------------------------------------


def _build_name_index(
    parsed_files: list[ParsedFile],
) -> tuple[dict[str, Symbol], dict[str, str], dict[str, list[str]]]:
    symbols_by_id: dict[str, Symbol] = {}
    qualified_index: dict[str, str] = {}
    bare_name_index: dict[str, list[str]] = {}
    for pf in parsed_files:
        for symbol in pf.symbols:
            symbols_by_id[symbol.id] = symbol
            qualified_index[symbol.qualified_name] = symbol.id
            bare_name_index.setdefault(symbol.name, []).append(symbol.id)
    return symbols_by_id, qualified_index, bare_name_index


# --------------------------------------------------------------------------
# Per-file import index (§5)
# --------------------------------------------------------------------------


def _build_import_index(pf: ParsedFile) -> dict[str, str]:
    """local name used in this file -> qualified name it refers to (best-effort)."""
    index: dict[str, str] = {}
    for node in _find_all(pf.tree.root_node, {"import_statement", "import_from_statement"}):
        if node.type == "import_statement":
            for dotted in (c for c in node.children if c.type in ("dotted_name", "aliased_import")):
                local_name, target = _import_binding(dotted, pf.source_bytes)
                index[local_name] = target
        else:  # import_from_statement
            module_name_node = node.child_by_field_name("module_name")
            module_start = module_name_node.start_byte if module_name_node is not None else -1
            for child in node.children:
                if child.type in ("dotted_name", "aliased_import") and child.start_byte != module_start:
                    local_name, target = _import_binding(child, pf.source_bytes)
                    index[local_name] = target
    return index


def _import_binding(node: ts.Node, source_bytes: bytes) -> tuple[str, str]:
    """Return (local_name, target_name) for a dotted_name or aliased_import node."""
    if node.type == "aliased_import":
        target = _node_text(node.child_by_field_name("name"), source_bytes)
        alias = _node_text(node.child_by_field_name("alias"), source_bytes)
        return alias, target.rsplit(".", 1)[-1]
    text = _node_text(node, source_bytes)
    local_name = text.split(".", 1)[0]
    target_name = text.rsplit(".", 1)[-1]
    return local_name, target_name


# --------------------------------------------------------------------------
# Re-locating each Symbol's AST node in the freshly re-parsed tree
# --------------------------------------------------------------------------


def _map_symbols_to_nodes(pf: ParsedFile) -> dict[str, ts.Node]:
    """Zip Phase 0's symbols to definition nodes in this fresh parse.

    Safe because both use the identical query + start_byte sort order over
    the same source bytes (see extractor.DEFINITION_QUERY_SOURCE) — and both
    apply the identical nested-function filter, so a closure excluded from
    `pf.symbols` doesn't throw off the 1:1 zip below.
    """
    query = ts.Query(get_language("python"), DEFINITION_QUERY_SOURCE)
    cursor = ts.QueryCursor(query)
    captures = cursor.captures(pf.tree.root_node)
    nodes = sorted(captures.get("def", []), key=lambda n: n.start_byte)
    nodes = [n for n in nodes if not is_nested_in_function(n)]
    return {symbol.id: node for symbol, node in zip(pf.symbols, nodes, strict=True)}


# --------------------------------------------------------------------------
# Base class names (for INHERITS resolution)
# --------------------------------------------------------------------------


def _extract_base_class_texts(class_node: ts.Node, source_bytes: bytes) -> list[str]:
    superclasses = class_node.child_by_field_name("superclasses")
    if superclasses is None:
        return []
    return [_node_text(c, source_bytes) for c in superclasses.children if c.type not in ("(", ",", ")")]


# --------------------------------------------------------------------------
# Call-expression walking (§6)
# --------------------------------------------------------------------------


def _iter_call_nodes(root: ts.Node, root_start_byte: int) -> list[ts.Node]:
    """All `call` nodes within `root`, not descending into a nested def's own body."""
    calls: list[ts.Node] = []

    def walk(n: ts.Node) -> None:
        if n.type in ("function_definition", "class_definition") and n.start_byte != root_start_byte:
            return  # a nested scope's calls belong to that scope, not this one
        if n.type == "call":
            calls.append(n)
        for child in n.children:
            walk(child)

    walk(root)
    return calls


# --------------------------------------------------------------------------
# The six-case resolution algorithm (§6), exact priority order
# --------------------------------------------------------------------------


def _resolve_call(
    callee: ts.Node,
    source_bytes: bytes,
    caller: Symbol,
    import_index: dict[str, str],
    qualified_index: dict[str, str],
    bare_name_index: dict[str, list[str]],
    symbols_by_id: dict[str, Symbol],
    base_class_ids: dict[str, list[str]],
    repo_module_stems: set[str],
) -> tuple[str, str | None]:
    """Returns ("resolved", symbol_id) or (reason, None)."""
    if callee.type == "attribute":
        base = callee.child_by_field_name("object")
        attr_name = _node_text(callee.child_by_field_name("attribute"), source_bytes)

        # Rule 1: self.<name> / cls.<name>
        if base.type == "identifier" and _node_text(base, source_bytes) in ("self", "cls"):
            return _resolve_self_call(attr_name, caller, qualified_index, symbols_by_id, base_class_ids)

        # Extension of §5's "orders" example: base.attr where base is a known
        # *first-party* whole-module import (its target matches one of this
        # repo's own file stems) is resolved via bare_name_index on the
        # attribute name. Gating on repo_module_stems is load-bearing, not
        # cosmetic: without it, `import re; re.search(...)` in a large repo
        # that also happens to define an unrelated function literally named
        # `search` anywhere at all would silently resolve to that unrelated
        # symbol instead of being correctly classified "external" — a real
        # false-positive CALLS edge found via a real ~900-symbol codebase,
        # not a hypothetical. A stdlib/third-party import's target never
        # matches a file actually being indexed, so it now falls through to
        # Rule 6 (unresolved) instead of guessing.
        if base.type == "identifier" and _node_text(base, source_bytes) in import_index:
            local_name = _node_text(base, source_bytes)
            if import_index[local_name] in repo_module_stems:
                return _resolve_via_bare_name_index(attr_name, bare_name_index)
            return "external", None

        # Rule 6: attribute access on anything else (local var, call result, ...)
        return "no_type_inference", None

    if callee.type == "identifier":
        name = _node_text(callee, source_bytes)
        return _resolve_bare_identifier(name, import_index, qualified_index, bare_name_index)

    # Rule 6: call on a call result, subscript-then-call, etc.
    return "no_type_inference", None


def _resolve_self_call(
    attr_name: str,
    caller: Symbol,
    qualified_index: dict[str, str],
    symbols_by_id: dict[str, Symbol],
    base_class_ids: dict[str, list[str]],
) -> tuple[str, str | None]:
    if caller.parent_id is None or caller.parent_id not in symbols_by_id:
        return "no_type_inference", None
    enclosing_class = symbols_by_id[caller.parent_id]

    target = qualified_index.get(f"{enclosing_class.qualified_name}.{attr_name}")
    if target is not None:
        return "resolved", target

    # Walk the transitive base-class chain looking for <Base>.<name>.
    frontier = list(base_class_ids.get(enclosing_class.id, []))
    seen: set[str] = set()
    while frontier:
        base_id = frontier.pop(0)
        if base_id in seen:
            continue
        seen.add(base_id)
        base_symbol = symbols_by_id.get(base_id)
        if base_symbol is None:
            continue
        target = qualified_index.get(f"{base_symbol.qualified_name}.{attr_name}")
        if target is not None:
            return "resolved", target
        frontier.extend(base_class_ids.get(base_id, []))

    return "no_type_inference", None


def _resolve_bare_identifier(
    name: str,
    import_index: dict[str, str],
    qualified_index: dict[str, str],
    bare_name_index: dict[str, list[str]],
) -> tuple[str, str | None]:
    # Rule 2: bare name matches this file's import index and resolves in-repo.
    imported_target = import_index.get(name)
    if imported_target is not None and imported_target in qualified_index:
        return "resolved", qualified_index[imported_target]

    # Rules 3/4/5: fall back to the repo-wide bare-name index.
    return _resolve_via_bare_name_index(name, bare_name_index)


def _resolve_via_bare_name_index(name: str, bare_name_index: dict[str, list[str]]) -> tuple[str, str | None]:
    matches = bare_name_index.get(name, [])
    if len(matches) == 1:
        return "resolved", matches[0]
    if len(matches) > 1:
        return "ambiguous", None
    return "external", None


# --------------------------------------------------------------------------
# Small shared helpers
# --------------------------------------------------------------------------


def _add_edge(graph: nx.DiGraph, source_id: str, target_id: str, edge_type: EdgeType) -> None:
    if graph.has_edge(source_id, target_id) and graph[source_id][target_id]["edge_type"] == edge_type:
        graph[source_id][target_id]["weight"] += 1
    else:
        graph.add_edge(source_id, target_id, edge_type=edge_type, weight=1)


def _find_all(node: ts.Node, types: set[str]) -> list[ts.Node]:
    found: list[ts.Node] = []
    if node.type in types:
        found.append(node)
    for child in node.children:
        found.extend(_find_all(child, types))
    return found


def _node_text(node: ts.Node, source_bytes: bytes) -> str:
    return source_bytes[node.start_byte : node.end_byte].decode("utf-8")
