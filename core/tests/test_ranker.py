"""Tests for retrieval/ranker.py (docs/phase-6-closing-the-loop.md §8 — Ranker)."""

import random

from loupe_core.retrieval.ranker import COLD_START_THRESHOLD, Ranker, TrainingExample


def _trained_ranker(seed: int = 42) -> Ranker:
    random.seed(seed)
    examples = [
        TrainingExample(
            lexical_score=random.random(), semantic_score=random.random(), centrality_score=random.random(),
            symbol_edited=random.random() > 0.5,
        )
        for _ in range(COLD_START_THRESHOLD)
    ]
    ranker = Ranker()
    ranker.train(examples)
    return ranker


def test_below_cold_start_threshold_ranker_refuses_to_train():
    examples = [TrainingExample(0.5, 0.5, 0.5, symbol_edited=True) for _ in range(COLD_START_THRESHOLD - 1)]
    ranker = Ranker()
    ranker.train(examples)

    assert ranker.is_trained is False
    assert ranker.coefficients is None


def test_below_cold_start_threshold_predict_returns_none_not_meaningless_prediction():
    """The fallback path (caller falling back to RRF) is only correct if predict()
    gives an unambiguous 'not trained' signal — verify that signal is actually there."""
    examples = [TrainingExample(0.9, 0.1, 0.1, symbol_edited=True) for _ in range(10)]
    ranker = Ranker()
    ranker.train(examples)

    result = ranker.predict(lexical_score=0.9, semantic_score=0.1, centrality_score=0.1)
    assert result is None, "predict() must return None (not a number) below the cold-start threshold"


def test_at_or_above_cold_start_threshold_ranker_trains():
    random.seed(42)
    examples = [
        TrainingExample(
            lexical_score=random.random(), semantic_score=random.random(), centrality_score=random.random(),
            symbol_edited=random.random() > 0.5,
        )
        for _ in range(COLD_START_THRESHOLD)
    ]
    ranker = Ranker()
    ranker.train(examples)

    assert ranker.is_trained is True
    assert ranker.coefficients is not None
    assert ranker.predict(0.5, 0.5, 0.5) is not None


def test_informative_feature_gets_a_meaningfully_larger_coefficient():
    """Synthetic dataset where semantic_score alone perfectly predicts the label;
    lexical_score/centrality_score are pure noise — a real correctness check on
    the learning itself, not just 'training completed without crashing'."""
    random.seed(7)
    examples = []
    for _ in range(400):
        semantic_score = random.random()
        label = semantic_score > 0.5
        examples.append(
            TrainingExample(
                lexical_score=random.random(),  # noise, uncorrelated with label
                semantic_score=semantic_score,
                centrality_score=random.random(),  # noise, uncorrelated with label
                symbol_edited=label,
            )
        )

    ranker = Ranker()
    ranker.train(examples)

    coefficients = ranker.coefficients
    assert coefficients is not None

    semantic_weight = abs(coefficients["semantic_score"])
    lexical_weight = abs(coefficients["lexical_score"])
    centrality_weight = abs(coefficients["centrality_score"])

    assert semantic_weight > lexical_weight * 3
    assert semantic_weight > centrality_weight * 3


def test_save_then_load_round_trips_trained_coefficients(tmp_path):
    ranker = _trained_ranker()
    path = tmp_path / "ranker.pkl"
    ranker.save(str(path))

    loaded = Ranker.load(str(path))

    assert loaded.is_trained is True
    assert loaded.coefficients == ranker.coefficients
    assert loaded.predict(0.5, 0.5, 0.5) == ranker.predict(0.5, 0.5, 0.5)


def test_load_missing_file_returns_untrained_ranker(tmp_path):
    ranker = Ranker.load(str(tmp_path / "does_not_exist.pkl"))

    assert ranker.is_trained is False
    assert ranker.coefficients is None
    assert ranker.predict(0.5, 0.5, 0.5) is None


def test_save_then_load_round_trips_untrained_ranker(tmp_path):
    ranker = Ranker()
    path = tmp_path / "cold_start.pkl"
    ranker.save(str(path))

    loaded = Ranker.load(str(path))

    assert loaded.is_trained is False
    assert loaded.coefficients is None
