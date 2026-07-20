"""Tests for retrieval/router.py (docs/phase-6-closing-the-loop.md §8 — Router,
docs/PhaseX/loupe-retrieval-upgrades.md §5 for the nearest-centroid fallback)."""

from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

import loupe_core.retrieval.router as router_module
from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.retrieval.router import (
    NEAREST_CENTROID_THRESHOLD,
    classify_intent,
    classify_intent_semantic,
    compute_intent_centroids,
    detect_symbol_reference,
    get_cached_intent_centroids,
    seed_debug_candidates,
    select_starting_level,
)
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME

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


# --------------------------------------------------------------------------
# Phase 9 §5 — nearest-centroid semantic fallback
# --------------------------------------------------------------------------


class _EncodeSpy:
    """Delegates to a real model's encode() while counting calls — a spy, not a fake
    (same pattern as test_semantic.py's EncodeSpy)."""

    def __init__(self, real_model: SentenceTransformer):
        self._real_model = real_model
        self.encode_call_count = 0

    def encode(self, texts, **kwargs):
        self.encode_call_count += 1
        return self._real_model.encode(texts, **kwargs)


@pytest.fixture(scope="session")
def real_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="module")
def real_centroids(real_model):
    """Computed once per module — reused explicitly by every test below instead
    of relying on the module-global cache, so tests stay independent of cache
    ordering/state."""
    return compute_intent_centroids(real_model)


def test_paraphrase_with_no_keyword_overlap_is_recovered_as_debug_via_centroid_path(real_model, real_centroids):
    """§5's own headline example: no "error"/"crash"/"broken" token at all, so the
    regex-only router (classify_intent) falls through to "general" — the semantic
    fallback must recover the obviously-debug intent a human reader sees immediately.
    """
    query = "things stopped working after the last deploy"
    assert classify_intent(query) == "general", "sanity check: regex path must NOT catch this paraphrase"
    assert classify_intent_semantic(query, model=real_model, centroids=real_centroids) == "debug"


@pytest.mark.parametrize(
    "query",
    [
        "getting a KeyError in the config loader",
        "why does this crash on startup",
        "add support for OAuth login",
        "refactor the session manager for clarity",
        "rename this variable to something clearer",
    ],
)
def test_regex_fast_path_queries_classified_identically_through_semantic_entrypoint(query, real_model, real_centroids):
    """§5's consistency acceptance criterion: a query the regex path would already
    catch must be classified identically when routed through
    `classify_intent_semantic` instead of `classify_intent` directly.
    """
    assert classify_intent_semantic(query, model=real_model, centroids=real_centroids) == classify_intent(query)


def test_regex_fast_path_queries_never_call_the_embedding_model(real_model):
    """Stronger than "same answer": the fast path must return before any embedding
    call happens at all — spied directly, not inferred from timing.
    """
    spy = _EncodeSpy(real_model)
    dummy_centroids = compute_intent_centroids(real_model)  # built with the real model, not the spy

    classify_intent_semantic("why does this crash on startup", model=spy, centroids=dummy_centroids)

    assert spy.encode_call_count == 0, "regex-matched queries must not embed the query at all"


def test_unrelated_query_below_threshold_stays_general(real_model, real_centroids):
    """Empirically measured at ~0.45 best-centroid similarity against the real
    model — comfortably under NEAREST_CENTROID_THRESHOLD's calibrated 0.65 (see
    that constant's own comment on why the model's noise floor is too high for
    a threshold this low to ever reject anything)."""
    query = "purple elephants dance quietly under the moonlight"
    assert classify_intent(query) == "general"
    assert classify_intent_semantic(query, model=real_model, centroids=real_centroids) == "general"


def test_compute_intent_centroids_has_one_vector_per_intent_all_same_dimension(real_model):
    centroids = compute_intent_centroids(real_model)
    assert set(centroids.vectors) == {"debug", "feature", "refactor", "general"}
    dims = {len(vector) for vector in centroids.vectors.values()}
    assert dims == {384}


def test_get_cached_intent_centroids_computes_once_per_process(real_model, monkeypatch):
    monkeypatch.setattr(router_module, "_cached_centroids", None)
    spy = _EncodeSpy(real_model)

    first = get_cached_intent_centroids(spy)
    calls_after_first = spy.encode_call_count
    assert calls_after_first > 0

    second = get_cached_intent_centroids(spy)
    assert spy.encode_call_count == calls_after_first, "a second call must not re-embed the example queries"
    assert second is first


def test_nearest_centroid_threshold_is_the_documented_constant():
    """0.65, not the spec's literal 0.3 — see router.py's own comment on why the
    literal value doesn't work against the real embedding model."""
    assert NEAREST_CENTROID_THRESHOLD == 0.65


# --------------------------------------------------------------------------
# Phase 14 §1 — starting zoom-level selection
# --------------------------------------------------------------------------


def test_broad_architecture_shaped_query_starts_at_l0_l1():
    assert select_starting_level("how does auth work across this repo") == "L0_L1"


def test_debug_query_naming_a_specific_symbol_starts_at_l3_l4():
    assert select_starting_level("crash inside validate_email when input is empty") == "L3_L4"


def test_feature_query_starts_at_l3_l4_even_without_a_named_symbol():
    assert select_starting_level("add support for OAuth login") == "L3_L4"


def test_general_query_naming_a_specific_symbol_starts_at_l3_l4_not_l0_l1():
    """A general-intent query is only broad enough for L0/L1 when it *also*
    has no detected symbol anchor — naming OrderService.create_order gives
    the router somewhere concrete to start, so it should, same as today."""
    assert select_starting_level("what does OrderService.create_order do") == "L3_L4"


def test_general_ambiguous_query_with_no_symbol_reference_starts_at_l0_l1():
    assert select_starting_level("what does this function do") == "L0_L1"
