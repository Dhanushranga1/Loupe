"""Tests for graph/builder.py against the Phase 1 mock-project fixtures.

One test per docs/phase-1-graph-theory.md §9 bullet, so a future regression
points at the exact rule that broke, not a giant integration failure.
"""

from pathlib import Path

from loupe_core.graph.builder import EdgeType, build_graph, parse_file

FIXTURES = Path(__file__).parent / "fixtures" / "phase1"


def _edge_data(graph, source_id, target_id):
    assert graph.has_edge(source_id, target_id), f"expected an edge {source_id} -> {target_id}"
    return graph[source_id][target_id]


def test_order_inherits_base(loupe_graph, symbol_by_qn):
    order = symbol_by_qn["Order"]
    base = symbol_by_qn["Base"]
    data = _edge_data(loupe_graph.graph, order.id, base.id)
    assert data["edge_type"] == EdgeType.INHERITS

    inherits_edges = [
        (u, v) for u, v, d in loupe_graph.graph.edges(data=True) if d["edge_type"] == EdgeType.INHERITS
    ]
    assert inherits_edges == [(order.id, base.id)]


def test_create_order_calls_validate_email_via_import(loupe_graph, symbol_by_qn):
    caller = symbol_by_qn["OrderService.create_order"]
    target = symbol_by_qn["validate_email"]
    data = _edge_data(loupe_graph.graph, caller.id, target.id)
    assert data["edge_type"] == EdgeType.CALLS


def test_create_order_calls_own_log_via_self(loupe_graph, symbol_by_qn):
    caller = symbol_by_qn["OrderService.create_order"]
    target = symbol_by_qn["OrderService.log"]
    data = _edge_data(loupe_graph.graph, caller.id, target.id)
    assert data["edge_type"] == EdgeType.CALLS


def test_create_order_instantiates_order_class(loupe_graph, symbol_by_qn):
    caller = symbol_by_qn["OrderService.create_order"]
    order_class = symbol_by_qn["Order"]
    data = _edge_data(loupe_graph.graph, caller.id, order_class.id)
    assert data["edge_type"] == EdgeType.CALLS, "instantiation must point at the class symbol itself"


def test_json_dumps_is_unresolved_external(loupe_graph, symbol_by_qn):
    caller = symbol_by_qn["OrderService.create_order"]
    matches = [r for r in loupe_graph.unresolved if r.from_symbol_id == caller.id and r.raw_expression == "json.dumps"]
    assert len(matches) == 1
    assert matches[0].reason == "external"
    order_class = symbol_by_qn["Order"]
    assert not loupe_graph.graph.has_edge(caller.id, order_class.id) or loupe_graph.graph[caller.id][order_class.id][
        "edge_type"
    ] != EdgeType.IMPORTS, "json.dumps must never produce a spurious edge"


def test_bare_process_call_is_ambiguous_not_linked_to_either_handler(loupe_graph, symbol_by_qn):
    dispatch = symbol_by_qn["dispatch"]
    order_handler_process = symbol_by_qn["OrderHandler.process"]
    user_handler_process = symbol_by_qn["UserHandler.process"]

    assert not loupe_graph.graph.has_edge(dispatch.id, order_handler_process.id)
    assert not loupe_graph.graph.has_edge(dispatch.id, user_handler_process.id)

    matches = [r for r in loupe_graph.unresolved if r.from_symbol_id == dispatch.id and r.raw_expression == "process"]
    assert len(matches) == 1
    assert matches[0].reason == "ambiguous"


def test_pagerank_ranks_hub_utils_above_never_called_handlers(loupe_graph, symbol_by_qn):
    format_currency = symbol_by_qn["format_currency"]
    validate_email = symbol_by_qn["validate_email"]
    never_called = [symbol_by_qn["OrderHandler.process"], symbol_by_qn["UserHandler.process"]]

    baseline = max(loupe_graph.pagerank_scores[s.id] for s in never_called)
    assert loupe_graph.pagerank_scores[format_currency.id] > baseline
    assert loupe_graph.pagerank_scores[validate_email.id] > baseline


def test_duplicate_calls_collapse_to_one_edge_with_weight_at_least_two(tmp_path):
    caller_src = tmp_path / "caller.py"
    callee_src = tmp_path / "callee.py"
    callee_src.write_text("def target():\n    return 1\n")
    caller_src.write_text(
        "from callee import target\n\n\ndef run():\n    target()\n    target()\n    return target()\n"
    )

    parsed = [parse_file(str(callee_src)), parse_file(str(caller_src))]
    graph = build_graph(parsed)

    run_symbol = next(s for pf in parsed for s in pf.symbols if s.qualified_name == "run")
    target_symbol = next(s for pf in parsed for s in pf.symbols if s.qualified_name == "target")

    data = graph.graph[run_symbol.id][target_symbol.id]
    assert data["weight"] >= 2
    assert len([e for e in graph.graph.edges() if e == (run_symbol.id, target_symbol.id)]) == 1
