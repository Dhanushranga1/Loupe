"""Tests for retrieval/fusion.py against the Phase 2 labeled query set (docs/phase-2-retrieval.md §8).

Uses the real embedding model (session-scoped, loaded once) — recall
numbers here reflect genuine model behavior, not fabricated data.
"""

from pathlib import Path

import pytest
import yaml
from sentence_transformers import SentenceTransformer

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.retrieval.fusion import fuse, search
from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME, SemanticIndex

PHASE1_FIXTURES = Path(__file__).parent / "fixtures" / "phase1"
PHASE1_FILES = ["utils.py", "models.py", "services.py", "handlers.py", "circular_a.py", "circular_b.py"]
LABELED_QUERIES_PATH = Path(__file__).parent / "fixtures" / "phase2" / "labeled_queries.yaml"


@pytest.fixture(scope="module")
def phase1_symbols():
    parsed = [parse_file(str(PHASE1_FIXTURES / f)) for f in PHASE1_FILES]
    return [s for pf in parsed for s in pf.symbols], parsed


@pytest.fixture(scope="module")
def label_to_id(phase1_symbols):
    symbols, _ = phase1_symbols
    return {f"{s.file_path.split('/')[-1]}::{s.qualified_name}": s.id for s in symbols}


@pytest.fixture(scope="module")
def pagerank_scores(phase1_symbols):
    _, parsed = phase1_symbols
    return build_graph(parsed).pagerank_scores


@pytest.fixture(scope="module")
def lexical_index(phase1_symbols):
    symbols, _ = phase1_symbols
    return LexicalIndex(symbols)


@pytest.fixture(scope="session")
def real_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="module")
def semantic_index(phase1_symbols, real_model):
    symbols, _ = phase1_symbols
    index = SemanticIndex(model=real_model)
    index.index(symbols)
    return index


@pytest.fixture(scope="module")
def labeled_queries():
    with open(LABELED_QUERIES_PATH) as f:
        return yaml.safe_load(f)


def _in_top_k(results: list[tuple[str, float]], target_ids: set[str], k: int) -> bool:
    top_ids = {symbol_id for symbol_id, _ in results[:k]}
    return bool(top_ids & target_ids)


def test_labeled_query_set_has_at_least_eight_queries(labeled_queries):
    assert len(labeled_queries) >= 8


def test_fused_recall_at_3_beats_or_matches_either_signal_alone(
    labeled_queries, label_to_id, lexical_index, semantic_index, pagerank_scores
):
    lexical_hits = semantic_hits = fused_hits = 0

    for item in labeled_queries:
        target_ids = {label_to_id[label] for label in item["expected"]}

        lexical_results = lexical_index.query(item["query"], top_k=50)
        semantic_results = semantic_index.query(item["query"], top_k=50)
        fused_results = fuse(lexical_results, semantic_results, pagerank_scores, top_k=3)

        lexical_hits += _in_top_k(lexical_results, target_ids, 3)
        semantic_hits += _in_top_k(semantic_results, target_ids, 3)
        fused_hits += _in_top_k(fused_results, target_ids, 3)

    n = len(labeled_queries)
    lexical_recall, semantic_recall, fused_recall = lexical_hits / n, semantic_hits / n, fused_hits / n

    assert fused_recall >= max(lexical_recall, semantic_recall), (
        f"fused recall@3 ({fused_recall}) must be >= max(lexical={lexical_recall}, semantic={semantic_recall})"
    )


def test_adversarial_pair_fusion_recovers_both_single_signal_failures(
    labeled_queries, label_to_id, lexical_index, semantic_index, pagerank_scores
):
    adversarial = [item for item in labeled_queries if item["kind"] == "adversarial"]
    assert len(adversarial) == 2, "expected exactly the two constructed adversarial queries"

    for item in adversarial:
        target_ids = {label_to_id[label] for label in item["expected"]}
        results = search(item["query"], lexical_index, semantic_index, pagerank_scores, top_k=3)
        assert _in_top_k(results, target_ids, 3), (
            f"fusion must recover the true target into top-3 for adversarial query {item['query']!r}"
        )

    # Confirm the adversarial construction is real: each query fails on exactly one signal alone.
    paraphrase_query = next(i for i in adversarial if "displayable price" in i["query"])
    exact_name_query = next(i for i in adversarial if i["query"] == "OrderService.log")

    # top_k=3 membership isn't a reliable check here: most of this tiny corpus scores a tied
    # 0.0 BM25 for this query, so which zero-score symbols land in an arbitrary top-3 window
    # depends on tie-break ordering, not relevance. Check the actual score instead.
    paraphrase_target_id = next(iter({label_to_id[label] for label in paraphrase_query["expected"]}))
    lexical_scores = dict(lexical_index.query(paraphrase_query["query"], top_k=50))
    assert lexical_scores.get(paraphrase_target_id, 0.0) == 0.0, (
        "lexical alone must assign zero relevance to the paraphrase target"
    )

    exact_name_target = {label_to_id[label] for label in exact_name_query["expected"]}
    semantic_only = semantic_index.query(exact_name_query["query"], top_k=3)
    assert not _in_top_k(semantic_only, exact_name_target, 3), "semantic alone must fail on the exact-name query"


def test_centrality_breaks_ties_between_equally_ranked_candidates():
    """Synthetic, model-independent check that centrality actually influences fusion output."""
    lexical_results = [("high-pagerank", 5.0), ("low-pagerank", 5.0)]
    semantic_results = [("high-pagerank", 0.8), ("low-pagerank", 0.8)]
    pagerank_scores = {"high-pagerank": 0.9, "low-pagerank": 0.1}

    fused = fuse(lexical_results, semantic_results, pagerank_scores, top_k=2)
    ranked_ids = [symbol_id for symbol_id, _ in fused]

    assert ranked_ids == ["high-pagerank", "low-pagerank"], (
        "with lexical/semantic tied, the higher-PageRank candidate must rank first after fusion"
    )


def test_fuse_never_introduces_a_candidate_neither_signal_found():
    lexical_results = [("a", 1.0)]
    semantic_results = [("b", 1.0)]
    pagerank_scores = {"a": 0.5, "b": 0.5, "c": 0.9}  # "c" is high-pagerank but absent from both signals

    fused = fuse(lexical_results, semantic_results, pagerank_scores, top_k=10)
    ranked_ids = {symbol_id for symbol_id, _ in fused}

    assert ranked_ids == {"a", "b"}
    assert "c" not in ranked_ids
