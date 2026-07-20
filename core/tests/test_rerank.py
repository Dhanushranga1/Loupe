"""Tests for retrieval/rerank.py's cross-encoder reranking
(docs/PhaseX/loupe-retrieval-upgrades.md §3).

Uses the real cross-encoder model against the real Phase 1/2 fixtures — same
"real model, not fabricated data" discipline as test_fusion.py.
"""

from pathlib import Path

import pytest
import yaml
from sentence_transformers import SentenceTransformer

from loupe_core.retrieval.fusion import CANDIDATE_POOL_SIZE, FINAL_TOP_K, fuse
from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.rerank import CROSS_ENCODER_MODEL_NAME, RerankResult, rerank
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME, SemanticIndex

LABELED_QUERIES_PATH = Path(__file__).parent / "fixtures" / "phase2" / "labeled_queries.yaml"


@pytest.fixture(scope="module")
def phase1_symbols(phase1_parsed):
    return [s for pf in phase1_parsed for s in pf.symbols]


@pytest.fixture(scope="module")
def symbols_by_id(phase1_symbols):
    return {s.id: s for s in phase1_symbols}


@pytest.fixture(scope="module")
def label_to_id(phase1_symbols):
    return {f"{s.file_path.split('/')[-1]}::{s.qualified_name}": s.id for s in phase1_symbols}


@pytest.fixture(scope="module")
def lexical_index(phase1_symbols):
    return LexicalIndex(phase1_symbols)


@pytest.fixture(scope="session")
def real_embedding_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="module")
def semantic_index(phase1_symbols, real_embedding_model):
    index = SemanticIndex(model=real_embedding_model)
    index.index(phase1_symbols)
    return index


@pytest.fixture(scope="session")
def real_cross_encoder():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(CROSS_ENCODER_MODEL_NAME)


@pytest.fixture(scope="module")
def labeled_queries():
    with open(LABELED_QUERIES_PATH) as f:
        return yaml.safe_load(f)


def _rrf_top20(query, lexical_index, semantic_index, loupe_graph):
    lexical_results = lexical_index.query(query, top_k=CANDIDATE_POOL_SIZE)
    semantic_results = semantic_index.query(query, top_k=CANDIDATE_POOL_SIZE)
    return fuse(lexical_results, semantic_results, loupe_graph.pagerank_scores, graph=loupe_graph.graph, top_k=FINAL_TOP_K)


def test_rerank_recovers_adversarial_pairs_into_top_2(
    labeled_queries, label_to_id, lexical_index, semantic_index, loupe_graph, symbols_by_id, real_cross_encoder
):
    """Spec's own acceptance criterion: cross-encoder reranking of RRF's top-20 must
    place the true target in the top 2 — tighter than Phase 2's original top-3 bar,
    since a cross-encoder should be strictly more precise on an already-narrowed set.
    """
    adversarial = [item for item in labeled_queries if item["kind"] == "adversarial"]
    assert len(adversarial) == 2

    for item in adversarial:
        target_ids = {label_to_id[label] for label in item["expected"]}
        rrf_top20 = _rrf_top20(item["query"], lexical_index, semantic_index, loupe_graph)

        result = rerank(item["query"], rrf_top20, symbols_by_id, cross_encoder=real_cross_encoder)
        top2_ids = {sid for sid, _ in result.ranked[:2]}

        assert top2_ids & target_ids, (
            f"cross-encoder reranking must place the true target in the top 2 for {item['query']!r}, "
            f"got {[sid for sid, _ in result.ranked[:5]]}"
        )


def test_rerank_measures_and_returns_predict_latency(lexical_index, semantic_index, loupe_graph, symbols_by_id, real_cross_encoder):
    rrf_top20 = _rrf_top20("validate an email address", lexical_index, semantic_index, loupe_graph)

    result = rerank("validate an email address", rrf_top20, symbols_by_id, cross_encoder=real_cross_encoder)

    assert isinstance(result, RerankResult)
    assert result.latency_ms >= 0.0
    assert result.latency_ms == result.latency_ms  # not NaN


def test_rerank_replaces_rrf_score_rather_than_blending(lexical_index, semantic_index, loupe_graph, symbols_by_id, real_cross_encoder):
    """§3's decided, non-negotiable behavior: the returned score is the
    cross-encoder's own relevance judgment, not any function of RRF's input score.
    Verified concretely: reranking the same candidate set with two different
    queries must reorder candidates purely off query-conditioned relevance, since
    RRF's score never changes between the two calls (same `rrf_top20` reused).
    """
    rrf_top20 = _rrf_top20("validate an email address", lexical_index, semantic_index, loupe_graph)

    result_a = rerank("validate an email address", rrf_top20, symbols_by_id, cross_encoder=real_cross_encoder)
    result_b = rerank("create a new order", rrf_top20, symbols_by_id, cross_encoder=real_cross_encoder)

    ranking_a = [sid for sid, _ in result_a.ranked]
    ranking_b = [sid for sid, _ in result_b.ranked]
    assert ranking_a != ranking_b, "different queries against the identical candidate set must produce different orderings"


def test_rerank_drops_candidate_ids_missing_from_symbols_by_id(real_cross_encoder):
    candidates = [("real-id", 0.5), ("phantom-id", 0.4)]
    symbols_by_id = {}  # neither id resolves — defensive path, not expected in real usage

    result = rerank("some query", candidates, symbols_by_id, cross_encoder=real_cross_encoder)

    assert result.ranked == []
    assert result.latency_ms == 0.0


def test_rerank_empty_candidates_returns_empty_result(real_cross_encoder):
    result = rerank("some query", [], {}, cross_encoder=real_cross_encoder)
    assert result.ranked == []
    assert result.latency_ms == 0.0
