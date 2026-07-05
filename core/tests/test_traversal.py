"""Tests for graph/traversal.py's expand_dependencies (docs/phase-1-graph-theory.md §9)."""

from loupe_core.graph.traversal import expand_dependencies


def test_circular_traversal_terminates_and_includes_both_sides(loupe_graph, symbol_by_qn):
    helper_a = symbol_by_qn["helper_a"]
    helper_b = symbol_by_qn["helper_b"]

    result = expand_dependencies(loupe_graph.graph, helper_a.id, depth=5, direction="both")

    assert isinstance(result, set)
    assert helper_b.id in result
    # depth=5 on a 2-node cycle can only ever reach {helper_a, helper_b}; helper_a
    # itself is excluded by contract, so the only other reachable node is helper_b.
    assert result == {helper_b.id}


def test_outgoing_and_incoming_give_genuinely_different_results(loupe_graph, symbol_by_qn):
    create_order = symbol_by_qn["OrderService.create_order"]
    order_class = symbol_by_qn["Order"]

    outgoing = expand_dependencies(loupe_graph.graph, create_order.id, depth=1, direction="outgoing")
    incoming_to_order = expand_dependencies(loupe_graph.graph, order_class.id, depth=1, direction="incoming")

    assert order_class.id in outgoing
    assert create_order.id in incoming_to_order

    # outgoing-from-create_order and incoming-to-create_order must differ:
    # create_order calls things (outgoing), but nothing in this fixture calls create_order.
    incoming_to_create_order = expand_dependencies(loupe_graph.graph, create_order.id, depth=1, direction="incoming")
    assert outgoing != incoming_to_create_order
    assert incoming_to_create_order == set()


def test_depth_limits_the_traversal(loupe_graph, symbol_by_qn):
    create_order = symbol_by_qn["OrderService.create_order"]
    order_class = symbol_by_qn["Order"]
    base_class = symbol_by_qn["Base"]

    depth_one = expand_dependencies(loupe_graph.graph, create_order.id, depth=1, direction="outgoing")
    assert base_class.id not in depth_one, "Base is two hops away (create_order -> Order -> Base), not one"

    depth_two = expand_dependencies(loupe_graph.graph, create_order.id, depth=2, direction="outgoing")
    assert order_class.id in depth_two
    assert base_class.id in depth_two


def test_result_never_contains_the_starting_symbol():
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_edge("a", "b", edge_type="calls", weight=1)
    graph.add_edge("b", "a", edge_type="calls", weight=1)

    result = expand_dependencies(graph, "a", depth=10, direction="both")
    assert "a" not in result
    assert result == {"b"}
