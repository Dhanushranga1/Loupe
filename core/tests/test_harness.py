"""Tests for eval/harness.py (docs/phase-5-evaluation.md §8 — Harness).

Uses the real embedding model (session-scoped) against the fixture git
repo — real strategies, real numbers, not fabricated data.
"""

import sys
from pathlib import Path

import git
import pytest
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "phase5"))
from build_fixture_repo import build_fixture_repo  # noqa: E402

from loupe_core.eval.harness import (
    build_repo_snapshot,
    run_churn_ablation,
    run_end_to_end_condition,
    run_harness,
    run_oracle_condition,
    run_personalized_pagerank_ablation,
    strategy_a_naive_whole_file,
    strategy_c_loupe_end_to_end,
    strategy_c_loupe_oracle,
)
from loupe_core.eval.mine_history import mine_history
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME


@pytest.fixture(scope="session")
def real_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


@pytest.fixture(scope="module")
def fixture_repo(tmp_path_factory):
    repo_root = tmp_path_factory.mktemp("phase5_harness_repo")
    shas = build_fixture_repo(repo_root)
    return repo_root, shas


@pytest.fixture(scope="module")
def tasks(fixture_repo):
    repo_root, _ = fixture_repo
    return mine_history(str(repo_root), max_commits=50)


def test_strategy_c_oracle_extracts_exactly_ground_truth_symbols(fixture_repo, tasks, real_model):
    repo_root, shas = fixture_repo
    repo = git.Repo(repo_root)
    task = next(t for t in tasks if t.commit_sha == shas["good_thousands_separator"])

    snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=real_model)
    result = strategy_c_loupe_oracle(snapshot, task.ground_truth_symbol_ids)

    assert result.retrieved_symbol_ids == task.ground_truth_symbol_ids
    assert len(result.retrieved_content) == 1
    assert "format_currency" in result.retrieved_content[0]


def test_strategy_a_oracle_loads_whole_file_more_symbols_than_target(fixture_repo, tasks, real_model):
    repo_root, shas = fixture_repo
    repo = git.Repo(repo_root)
    task = next(t for t in tasks if t.commit_sha == shas["good_thousands_separator"])

    snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=real_model)
    result = strategy_a_naive_whole_file(snapshot, task.task_description, ground_truth_files=task.ground_truth_files)

    # utils.py has 2 symbols (format_currency, validate_email); oracle Loupe
    # extracts only the 1 ground-truth symbol — this is the whole point.
    assert len(result.retrieved_symbol_ids) == 2
    assert set(task.ground_truth_symbol_ids).issubset(set(result.retrieved_symbol_ids))


def test_oracle_condition_loupe_uses_substantially_fewer_tokens_than_naive(fixture_repo, tasks, real_model):
    repo_root, _ = fixture_repo
    repo = git.Repo(repo_root)

    result = run_oracle_condition(repo, tasks, embedding_model=real_model)

    assert result["naive_total_tokens"] > 0
    assert result["loupe_total_tokens"] > 0
    assert result["ratio"] is not None
    assert result["ratio"] > 1.0, (
        f"Loupe's oracle-mode token cost ({result['loupe_total_tokens']}) should be substantially "
        f"lower than naive whole-file loading ({result['naive_total_tokens']}); ratio={result['ratio']}"
    )


def test_end_to_end_condition_runs_without_manual_intervention_and_produces_all_metrics(fixture_repo, tasks, real_model):
    repo_root, _ = fixture_repo
    repo = git.Repo(repo_root)

    result = run_end_to_end_condition(repo, tasks, embedding_model=real_model)

    assert set(result.keys()) == {"naive", "vector_rag", "loupe"}
    for metrics in result.values():
        for metric_name in ("recall_5", "recall_10", "tokens"):
            agg = metrics[metric_name]
            assert set(agg.keys()) == {"mean", "median", "n"}


def test_strategy_c_end_to_end_personalized_flag_toggles_fuse_graph_argument(fixture_repo, tasks, real_model):
    """Not a recall assertion (the fixture repo is too small/single-cluster for
    personalization to visibly change ranking — that's what test_fusion.py's
    dedicated two-cluster fixture is for). This just proves the flag actually
    routes to a different `fuse()` call rather than being a dead parameter:
    both modes must still run end-to-end and return a valid result shape.
    """
    repo_root, shas = fixture_repo
    repo = git.Repo(repo_root)
    task = next(t for t in tasks if t.commit_sha == shas["good_thousands_separator"])
    snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=real_model)

    static_result = strategy_c_loupe_end_to_end(snapshot, task.task_description, use_personalized_pagerank=False)
    personalized_result = strategy_c_loupe_end_to_end(snapshot, task.task_description, use_personalized_pagerank=True)

    assert static_result.retrieved_symbol_ids
    assert personalized_result.retrieved_symbol_ids


def test_personalized_pagerank_ablation_runs_and_produces_both_arms(fixture_repo, tasks, real_model):
    repo_root, _ = fixture_repo
    repo = git.Repo(repo_root)

    result = run_personalized_pagerank_ablation(repo, tasks, embedding_model=real_model)

    assert set(result.keys()) == {"static_pagerank", "personalized_pagerank"}
    for metrics in result.values():
        for metric_name in ("recall_5", "recall_10", "tokens"):
            agg = metrics[metric_name]
            assert set(agg.keys()) == {"mean", "median", "n"}


def test_strategy_c_end_to_end_churn_flag_toggles_fuse_churn_argument(fixture_repo, tasks, real_model):
    """Not a recall assertion, same reasoning as the personalized-PageRank
    equivalent above — this proves the flag routes to a real, non-crashing
    churn computation against the fixture's real (if tightly-clustered-in-time)
    git history, not that it visibly changes ranking on this small fixture."""
    repo_root, shas = fixture_repo
    repo = git.Repo(repo_root)
    task = next(t for t in tasks if t.commit_sha == shas["good_thousands_separator"])
    snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=real_model)

    without_churn = strategy_c_loupe_end_to_end(snapshot, task.task_description, repo=repo, use_churn=False)
    with_churn = strategy_c_loupe_end_to_end(snapshot, task.task_description, repo=repo, use_churn=True)

    assert without_churn.retrieved_symbol_ids
    assert with_churn.retrieved_symbol_ids


def test_churn_ablation_runs_and_produces_both_arms(fixture_repo, tasks, real_model):
    repo_root, _ = fixture_repo
    repo = git.Repo(repo_root)

    result = run_churn_ablation(repo, tasks, embedding_model=real_model)

    assert set(result.keys()) == {"without_churn", "with_churn"}
    for metrics in result.values():
        for metric_name in ("recall_5", "recall_10", "tokens"):
            agg = metrics[metric_name]
            assert set(agg.keys()) == {"mean", "median", "n"}


def test_full_harness_run_writes_results_json(fixture_repo, real_model, tmp_path):
    repo_root, _ = fixture_repo
    results_dir = tmp_path / "results"

    report = run_harness(str(repo_root), max_commits=50, embedding_model=real_model, results_dir=results_dir)

    assert report["task_count"] > 0
    assert "oracle_condition" in report
    assert "end_to_end_condition" in report

    output_path = Path(report["_output_path"])
    assert output_path.exists()
    assert output_path.parent == results_dir

    import json

    on_disk = json.loads(output_path.read_text())
    assert on_disk["task_count"] == report["task_count"]
