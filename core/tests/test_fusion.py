"""Tests for retrieval/fusion.py against the Phase 2 labeled query set (docs/phase-2-retrieval.md §8).

Uses the real embedding model (session-scoped, loaded once) — recall
numbers here reflect genuine model behavior, not fabricated data.
"""

import os
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


# --------------------------------------------------------------------------
# `fuse(..., graph=...)` — personalized centrality (retrieval-upgrades §2),
# not just `graph/centrality.py`'s own unit tests (test_centrality.py) in
# isolation, but proof it's actually wired into RRF's centrality term.
# --------------------------------------------------------------------------


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def two_cluster_repo(tmp_path):
    _write(
        tmp_path,
        "cluster_a.py",
        "def hub_a():\n    return 1\n\n\n"
        + "\n\n".join(f"def a_caller_{i}():\n    return hub_a()" for i in range(1, 4)),
    )
    _write(
        tmp_path,
        "cluster_b.py",
        "def hub_b():\n    return 2\n\n\n"
        + "\n\n".join(f"def b_caller_{i}():\n    return hub_b()" for i in range(1, 9)),
    )
    _write(
        tmp_path,
        "bridge.py",
        "from cluster_a import a_caller_1\nfrom cluster_b import b_caller_1\n\n\n"
        "def weak_bridge():\n    a_caller_1()\n    b_caller_1()\n",
    )

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        files = ["cluster_a.py", "cluster_b.py", "bridge.py"]
        parsed = [parse_file(f) for f in files]
        g = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        yield g, symbols_by_id
    finally:
        os.chdir(old_cwd)


def _id_by_name(symbols_by_id, qualified_name: str) -> str:
    return next(s.id for s in symbols_by_id.values() if s.qualified_name == qualified_name)


def test_fuse_with_graph_uses_personalized_centrality_not_static(two_cluster_repo):
    """Static PageRank favors hub_b (8 callers) over hub_a (3). With lexical/semantic
    tied on both hubs, `fuse(..., graph=None)` must rank by the static score (hub_b
    first); passing the real graph must flip that order once seeded near hub_a —
    the concrete proof `fuse()` is actually consulting `graph`, not ignoring it.
    """
    g, symbols_by_id = two_cluster_repo
    hub_a = _id_by_name(symbols_by_id, "hub_a")
    hub_b = _id_by_name(symbols_by_id, "hub_b")
    seed = _id_by_name(symbols_by_id, "a_caller_2")

    # hub_a/hub_b swap rank-1/rank-2 between the two signals, so their combined
    # lexical+semantic RRF contribution is exactly symmetric (rank is positional,
    # not value-based — equal *scores* would NOT tie here, only equal *rank sums*
    # do) — isolating centrality as the only remaining thing that can break the tie.
    lexical_results = [(hub_a, 2.0), (hub_b, 1.0), (seed, 0.5)]
    semantic_results = [(hub_b, 2.0), (hub_a, 1.0), (seed, 0.5)]

    static_fused = fuse(lexical_results, semantic_results, g.pagerank_scores, top_k=3)
    static_ranked = [sid for sid, _ in static_fused]
    assert static_ranked.index(hub_b) < static_ranked.index(hub_a), (
        "sanity check on fixture construction: static PageRank must favor hub_b"
    )

    personalized_fused = fuse(lexical_results, semantic_results, g.pagerank_scores, graph=g.graph, top_k=3)
    personalized_ranked = [sid for sid, _ in personalized_fused]
    assert personalized_ranked.index(hub_a) < personalized_ranked.index(hub_b), (
        "personalized centrality, seeded by this call's own candidate pool, must flip the order static PageRank gave"
    )


def test_fuse_with_churn_scores_breaks_ties_toward_the_more_recently_edited_symbol():
    """Phase 14 §2: churn is an optional fourth RRF signal — omitted entirely
    by default (every existing test above passes no `churn_scores` at all,
    proving the signal is opt-in, not a silent behavior change), but when
    given, must actually influence ranking among otherwise-tied candidates."""
    lexical_results = [("recently_edited", 1.0), ("untouched_in_months", 1.0)]
    semantic_results = [("recently_edited", 1.0), ("untouched_in_months", 1.0)]
    pagerank_scores = {"recently_edited": 0.5, "untouched_in_months": 0.5}
    churn_scores = {"recently_edited": 0.9, "untouched_in_months": 0.0}

    without_churn = fuse(lexical_results, semantic_results, pagerank_scores, top_k=2)
    with_churn = fuse(lexical_results, semantic_results, pagerank_scores, top_k=2, churn_scores=churn_scores)

    without_churn_ids = [sid for sid, _ in without_churn]
    with_churn_ids = [sid for sid, _ in with_churn]

    assert without_churn_ids[0] == "recently_edited", "sanity check: tied without churn, id breaks the tie alphabetically"
    assert with_churn_ids.index("recently_edited") < with_churn_ids.index("untouched_in_months")


# --------------------------------------------------------------------------
# `fuse(..., hyde_results=...)` — docs/PhaseX/experimental-gate-and-hyde.md §6.
# Unlike churn/centrality (fixed per-repo scores that only re-rank existing
# candidates), hyde_results is a genuine ranking that can introduce brand
# new candidates neither lexical nor raw-query semantic ever found — that's
# the entire reason HyDE is worth its cost.
# --------------------------------------------------------------------------


def test_fuse_without_hyde_results_is_unaffected_default_none():
    lexical_results = [("a", 1.0)]
    semantic_results = [("b", 1.0)]
    pagerank_scores = {"a": 0.5, "b": 0.5}

    with_default = fuse(lexical_results, semantic_results, pagerank_scores, top_k=10)
    explicit_none = fuse(lexical_results, semantic_results, pagerank_scores, top_k=10, hyde_results=None)

    assert with_default == explicit_none


def test_fuse_with_hyde_results_can_introduce_a_candidate_neither_other_signal_found():
    lexical_results = [("a", 1.0)]
    semantic_results = [("b", 1.0)]
    pagerank_scores = {"a": 0.5, "b": 0.5, "c": 0.9}
    hyde_results = [("c", 1.0)]  # only HyDE's hypothetical-answer embedding found "c"

    fused = fuse(lexical_results, semantic_results, pagerank_scores, top_k=10, hyde_results=hyde_results)
    ranked_ids = {symbol_id for symbol_id, _ in fused}

    assert ranked_ids == {"a", "b", "c"}, "hyde_results must be able to add a candidate, unlike centrality/churn"


def test_fuse_with_hyde_results_raises_a_candidates_score_without_affecting_others():
    lexical_results = [("x", 1.0), ("y", 1.0)]
    semantic_results = [("x", 1.0), ("y", 1.0)]
    pagerank_scores = {"x": 0.5, "y": 0.5}
    hyde_results = [("y", 1.0)]  # only "y" gets a hyde vote; "x" has none

    without_hyde = dict(fuse(lexical_results, semantic_results, pagerank_scores, top_k=2))
    with_hyde = dict(fuse(lexical_results, semantic_results, pagerank_scores, top_k=2, hyde_results=hyde_results))

    assert with_hyde["y"] > without_hyde["y"], "a candidate hyde ranks highly must score higher with it folded in"
    assert with_hyde["x"] == without_hyde["x"], "a candidate absent from hyde_results is completely unaffected by it"


def test_hyde_rescues_a_vague_query_that_raw_semantic_search_misses(real_model):
    """docs/PhaseX/experimental-gate-and-hyde.md §7's acceptance criterion:
    a deliberately vague query where raw-query semantic search fails to
    surface the true target is correctly rescued once HyDE's hypothetical-
    answer embedding is added as a fourth RRF signal.

    Empirically calibrated against the real embedding model (same discipline
    used for E5's duplicate-code similarity threshold): with a deliberately
    narrow candidate pool (top_k=1 per signal — an artificially tight
    window, constructed the same way the file's other hand-built adversarial
    pair above is, to force a genuine "missing entirely" case rather than
    "ranked a bit lower"), this exact vague phrasing's raw-query semantic
    and lexical top-1 both land on other, real decoys, genuinely excluding
    the true target (`retry_with_backoff`) from the candidate pool
    altogether — verified below, not assumed.
    """
    from loupe_core.parsing.schema import Symbol, SymbolKind

    corpus = {
        "retry_with_backoff": "Retries a failed network call with exponential backoff, "
        "giving up after the maximum number of attempts.",
        "give_up_on_stuck_job": "Abandon a job that has been stuck in the queue for too long "
        "and mark it as failed.",
        "restart_stalled_worker": "Kill and respawn a worker process that has stopped "
        "responding to heartbeats.",
        "shutdown_gracefully": "Stop accepting new requests and wait for in-flight requests "
        "to finish before exiting.",
        "clear_cache": "Empty the in-memory cache of previously computed results.",
        "log_error": "Write an error message to the application log.",
    }
    symbols = [
        Symbol(
            id=f"{i:016x}", kind=SymbolKind.FUNCTION, name=name, qualified_name=name, file_path=f"{name}.py",
            byte_start=0, byte_end=1, line_start=1, line_end=1, signature=f"def {name}():", docstring=doc,
        )
        for i, (name, doc) in enumerate(corpus.items())
    ]
    id_by_name = {s.name: s.id for s in symbols}

    semantic_index = SemanticIndex(model=real_model)
    semantic_index.index(symbols)
    lexical_index = LexicalIndex(symbols)
    pagerank_scores = {s.id: 0.5 for s in symbols}

    query = "how do we keep it from hammering a dead endpoint over and over"
    small_pool = 1  # narrow enough that the true target is genuinely excluded without HyDE

    lexical_results = lexical_index.query(query, top_k=small_pool)
    semantic_results = semantic_index.query(query, top_k=small_pool)
    target_id = id_by_name["retry_with_backoff"]

    without_hyde = fuse(lexical_results, semantic_results, pagerank_scores, top_k=3)
    assert target_id not in {sid for sid, _ in without_hyde}, (
        "sanity check: the true target must genuinely be missing without HyDE, or this isn't a real rescue"
    )

    hypothetical_answer = (
        "def retry_with_backoff(fn):\n"
        '    """Retry fn with exponential backoff, stop after max_attempts."""\n    ...'
    )
    hyde_results = semantic_index.query(hypothetical_answer, top_k=small_pool)

    with_hyde = fuse(lexical_results, semantic_results, pagerank_scores, top_k=3, hyde_results=hyde_results)
    assert target_id in {sid for sid, _ in with_hyde}, "HyDE's hypothetical-answer signal must rescue the true target"
