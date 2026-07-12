"""Tests for bandit/thompson.py (docs/phase-6-closing-the-loop.md §8 — Bandit)."""

import random

from loupe_core.bandit.thompson import PendingDecision, ThompsonBandit


def test_basic_beta_update_increases_alpha_by_reward_count_beta_unchanged():
    bandit = ThompsonBandit()
    for i in range(5):
        bandit._pending[f"log-{i}"] = PendingDecision(
            retrieval_log_id=f"log-{i}", intent="debug", arm_chosen="rrf", top_candidate_symbol_id=f"sym-{i}"
        )
    initial_alpha, initial_beta = bandit.get_alpha_beta("debug", "rrf")

    for i in range(5):
        bandit.resolve_outcome(f"log-{i}", symbol_edited=True)

    alpha, beta = bandit.get_alpha_beta("debug", "rrf")
    assert alpha == initial_alpha + 5
    assert beta == initial_beta


def test_negative_reward_increases_beta_not_alpha():
    bandit = ThompsonBandit()
    bandit._pending["log-1"] = PendingDecision("log-1", "debug", "rrf", "sym-1")
    initial_alpha, initial_beta = bandit.get_alpha_beta("debug", "rrf")

    bandit.resolve_outcome("log-1", symbol_edited=False)

    alpha, beta = bandit.get_alpha_beta("debug", "rrf")
    assert alpha == initial_alpha
    assert beta == initial_beta + 1


def test_resolving_an_unknown_log_id_is_a_safe_no_op():
    bandit = ThompsonBandit()
    bandit.resolve_outcome("never-seen", symbol_edited=True)  # must not raise
    assert bandit.pending_count() == 0


def test_arms_for_different_intents_are_fully_independent():
    bandit = ThompsonBandit()
    bandit._pending["log-1"] = PendingDecision("log-1", "debug", "rrf", "sym-1")
    bandit.resolve_outcome("log-1", symbol_edited=True)

    debug_alpha, _ = bandit.get_alpha_beta("debug", "rrf")
    feature_alpha, feature_beta = bandit.get_alpha_beta("feature", "rrf")

    assert debug_alpha == 2.0  # 1 (prior) + 1 (reward)
    assert (feature_alpha, feature_beta) == (1.0, 1.0), "an unrelated intent's arm must be untouched"


def test_delayed_reward_decision_and_update_are_decoupled():
    """A decision is made; several unrelated decisions are made and resolved in
    between; only when the ORIGINAL decision's own outcome resolves does its
    arm update — proving decision-time and update-time aren't accidentally
    coupled to call order."""
    bandit = ThompsonBandit()
    bandit._pending["target-log"] = PendingDecision("target-log", "debug", "rrf", "sym-target")
    initial = bandit.get_alpha_beta("debug", "rrf")

    for i in range(5):
        log_id = f"other-{i}"
        bandit._pending[log_id] = PendingDecision(log_id, "feature", "learned_ranker", f"sym-{i}")
        bandit.resolve_outcome(log_id, symbol_edited=True)
        assert bandit.get_alpha_beta("debug", "rrf") == initial, "unrelated resolutions must not touch this arm"

    assert bandit.pending_count() == 1

    bandit.resolve_outcome("target-log", symbol_edited=True)

    alpha, beta = bandit.get_alpha_beta("debug", "rrf")
    assert alpha == initial[0] + 1
    assert beta == initial[1]
    assert bandit.pending_count() == 0


def test_convergence_favors_the_better_arm_over_many_rounds():
    """arm 'rrf' has true reward probability 0.9, 'learned_ranker' has 0.1 — over
    enough rounds Thompson sampling must select the better arm significantly
    more than half the time (a concrete, checkable convergence threshold)."""
    bandit = ThompsonBandit(rng=random.Random(123))
    reward_rng = random.Random(456)
    true_prob = {"rrf": 0.9, "learned_ranker": 0.1}

    selections = []
    for i in range(300):
        log_id = f"log-{i}"
        chosen = bandit.select_arm("debug", log_id, top_candidate_symbol_id=f"sym-{i}")
        selections.append(chosen)
        reward = reward_rng.random() < true_prob[chosen]
        bandit.resolve_outcome(log_id, symbol_edited=reward)

    last_50 = selections[-50:]
    rrf_fraction = last_50.count("rrf") / len(last_50)
    assert rrf_fraction > 0.7, f"expected the better arm to dominate late selections, got {rrf_fraction:.2f}"
