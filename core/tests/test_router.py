"""Tests for retrieval/router.py (docs/phase-6-closing-the-loop.md §8 — Router)."""

from pathlib import Path

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.retrieval.router import classify_intent, detect_symbol_reference, seed_debug_candidates

PHASE1_FIXTURES = Path(__file__).parent / "fixtures" / "phase1"
PHASE1_FILES = ["utils.py", "models.py", "services.py", "handlers.py", "circular_a.py", "circular_b.py"]


def test_debug_intent_classification():
    for query in [
        "getting a KeyError in the config loader",
        "why does this crash on startup",
        "traceback shows a null pointer in the parser",
    ]:
        assert classify_intent(query) == "debug", query


def test_feature_intent_classification():
    for query in [
        "add support for OAuth login",
        "implement rate limiting on the API",
        "create a new endpoint for exporting reports",
    ]:
        assert classify_intent(query) == "feature", query


def test_refactor_intent_classification():
    for query in [
        "refactor the session manager for clarity",
        "rename this variable to something clearer",
        "clean up the duplicated validation logic",
    ]:
        assert classify_intent(query) == "refactor", query


def test_general_intent_default_and_ambiguous_queries():
    for query in [
        "what does this function do",
        "where is the database connection configured",
        "explain how retries are handled here",
        "how many symbols are in this repo",
    ]:
        assert classify_intent(query) == "general", query


def test_detect_symbol_reference_snake_case():
    assert detect_symbol_reference("crash inside validate_email when input is empty") == "validate_email"


def test_detect_symbol_reference_qualified_name():
    assert detect_symbol_reference("error thrown from OrderService.create_order") == "OrderService.create_order"


def test_detect_symbol_reference_filename():
    assert detect_symbol_reference("exception traced back to utils.py") == "utils.py"


def test_detect_symbol_reference_none_when_absent():
    assert detect_symbol_reference("why is this broken") is None


def _build_test_graph():
    parsed = [parse_file(str(PHASE1_FIXTURES / f)) for f in PHASE1_FILES]
    return build_graph(parsed), parsed


def test_seed_debug_candidates_expands_from_resolved_reference():
    loupe_graph, parsed = _build_test_graph()
    symbols_by_name = {s.qualified_name: s.id for pf in parsed for s in pf.symbols}

    def resolve(reference: str) -> str | None:
        return symbols_by_name.get(reference)

    result = seed_debug_candidates(
        "crash inside create_order when email is missing",
        resolve_reference=lambda ref: symbols_by_name.get("OrderService.create_order"),
        graph=loupe_graph.graph,
        depth=1,
    )
    expanded_names = {
        s.qualified_name for pf in parsed for s in pf.symbols if s.id in result
    }
    assert "validate_email" in expanded_names or "Order" in expanded_names


def test_seed_debug_candidates_empty_when_no_reference_detected():
    loupe_graph, _ = _build_test_graph()
    result = seed_debug_candidates("why is this broken", resolve_reference=lambda ref: None, graph=loupe_graph.graph)
    assert result == set()


def test_seed_debug_candidates_does_not_crash_on_unresolvable_reference():
    loupe_graph, _ = _build_test_graph()
    result = seed_debug_candidates(
        "crash inside totally_unknown_symbol_xyz", resolve_reference=lambda ref: None, graph=loupe_graph.graph
    )
    assert result == set()


def test_seed_debug_candidates_empty_for_non_debug_intent():
    loupe_graph, parsed = _build_test_graph()
    symbols_by_name = {s.qualified_name: s.id for pf in parsed for s in pf.symbols}
    result = seed_debug_candidates(
        "add support for validate_email length checks",  # feature-intent, even though it references a real symbol
        resolve_reference=lambda ref: symbols_by_name.get(ref),
        graph=loupe_graph.graph,
    )
    assert result == set()
