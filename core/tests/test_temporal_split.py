"""Tests for the temporal train/test split (docs/phase-6-closing-the-loop.md §8 —
End-to-end evaluation: 'the temporal split is verified explicitly')."""

import random
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "phase5"))
from build_fixture_repo import build_fixture_repo  # noqa: E402

from loupe_core.eval.mine_history import mine_history, temporal_split
from loupe_core.retrieval.ranker import Ranker, TrainingExample


@pytest.fixture(scope="module")
def tasks(tmp_path_factory):
    repo_root = tmp_path_factory.mktemp("phase6_temporal_repo")
    build_fixture_repo(repo_root)
    return mine_history(str(repo_root), max_commits=50)


def test_no_evaluation_task_commit_date_precedes_the_training_cutoff(tasks):
    assert len(tasks) >= 2, "need at least 2 tasks for a meaningful split"
    sorted_dates = sorted(t.committed_at for t in tasks)
    cutoff = sorted_dates[len(sorted_dates) // 2]

    training, evaluation = temporal_split(tasks, cutoff)

    assert training, "expected at least one training task"
    assert evaluation, "expected at least one evaluation task"
    assert all(t.committed_at < cutoff for t in training)
    assert all(t.committed_at >= cutoff for t in evaluation)
    assert max(t.committed_at for t in training) <= min(t.committed_at for t in evaluation)


def test_split_is_exhaustive_and_non_overlapping(tasks):
    cutoff = sorted(t.committed_at for t in tasks)[len(tasks) // 2]
    training, evaluation = temporal_split(tasks, cutoff)

    assert set(t.commit_sha for t in training) & set(t.commit_sha for t in evaluation) == set()
    assert len(training) + len(evaluation) == len(tasks)


# --------------------------------------------------------------------------
# Leakage demonstration: a random split can leak future-pattern information
# into training in a way the temporal split structurally cannot.
# --------------------------------------------------------------------------


def _concept_drift_dataset(n_early: int, n_late: int, seed: int) -> list[tuple[TrainingExample, str]]:
    """Synthetic dataset with a deliberate concept drift: in the "early" period,
    `semantic_score` predicts the label; in the "late" period, the relationship
    *inverts* — `semantic_score` becomes pure noise and `lexical_score` predicts
    instead. A model trained ONLY on early data (the honest, temporally-correct
    scenario) cannot know the late-period pattern; a model whose training set
    was contaminated with late-period examples (via a random split) can "see"
    it in advance, producing evaluation numbers that look better but reflect
    leaked information, not genuine early-only knowledge.

    Returns a list of (TrainingExample, period) pairs, period in {"early", "late"}.
    """
    rng = random.Random(seed)
    rows = []
    for _ in range(n_early):
        semantic_score = rng.random()
        label = semantic_score > 0.5
        rows.append(
            (
                TrainingExample(
                    lexical_score=rng.random(), semantic_score=semantic_score, centrality_score=rng.random(),
                    symbol_edited=label,
                ),
                "early",
            )
        )
    for _ in range(n_late):
        lexical_score = rng.random()
        label = lexical_score > 0.5
        rows.append(
            (
                TrainingExample(
                    lexical_score=lexical_score, semantic_score=rng.random(), centrality_score=rng.random(),
                    symbol_edited=label,
                ),
                "late",
            )
        )
    return rows


def _accuracy(ranker: Ranker, examples: list[TrainingExample]) -> float:
    correct = 0
    for e in examples:
        prob = ranker.predict(e.lexical_score, e.semantic_score, e.centrality_score)
        predicted = prob is not None and prob > 0.5
        if predicted == e.symbol_edited:
            correct += 1
    return correct / len(examples)


def test_random_split_leaks_relative_to_temporal_split():
    rows = _concept_drift_dataset(n_early=250, n_late=250, seed=99)

    early_examples = [ex for ex, period in rows if period == "early"]
    late_examples = [ex for ex, period in rows if period == "late"]

    # Temporally correct: train on early only, evaluate on late only — the
    # model has genuinely never seen the late-period pattern.
    temporal_ranker = Ranker()
    temporal_ranker.train(early_examples)
    temporal_eval_accuracy = _accuracy(temporal_ranker, late_examples)

    # Deliberately random split (mixing both periods into train and test) —
    # the leak: some late-period examples end up in training, letting the
    # model partially learn the late pattern before being "evaluated" on it.
    rng = random.Random(7)
    all_examples = [ex for ex, _period in rows]
    shuffled = all_examples[:]
    rng.shuffle(shuffled)
    random_train, random_test = shuffled[:400], shuffled[400:]

    random_ranker = Ranker()
    random_ranker.train(random_train)
    random_split_accuracy = _accuracy(random_ranker, random_test)

    assert random_split_accuracy > temporal_eval_accuracy, (
        f"expected the random split's accuracy ({random_split_accuracy:.2f}) to look better than the "
        f"temporally-honest split's ({temporal_eval_accuracy:.2f}) precisely because it leaked late-period "
        "examples into training — if this fails, the synthetic drift isn't demonstrating leakage"
    )
    # The temporally-honest model, having never seen the late pattern, should
    # perform close to chance (~0.5) on it — an honest reflection of "this
    # model genuinely doesn't know this pattern yet."
    assert temporal_eval_accuracy < 0.65
