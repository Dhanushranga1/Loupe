"""Tests for governor/knapsack.py (docs/phase-3-resource-allocation.md §8).

`knapsack_exact` is checked against a brute-force exhaustive search over all
2^n subsets on a small hand-built set — a genuine ground-truth check, not
just "it ran."
"""

import itertools

from loupe_core.governor.knapsack import KnapsackCandidate, knapsack_exact, knapsack_greedy

CANDIDATES = [
    KnapsackCandidate("a", relevance_score=0.9, token_cost=10),
    KnapsackCandidate("b", relevance_score=0.8, token_cost=8),
    KnapsackCandidate("c", relevance_score=0.7, token_cost=15),
    KnapsackCandidate("d", relevance_score=0.6, token_cost=5),
    KnapsackCandidate("e", relevance_score=0.5, token_cost=12),
    KnapsackCandidate("f", relevance_score=0.4, token_cost=3),
    KnapsackCandidate("g", relevance_score=0.3, token_cost=7),
    KnapsackCandidate("h", relevance_score=0.2, token_cost=20),
]
BUDGET = 30


def _brute_force_optimal(candidates: list[KnapsackCandidate], budget: int) -> float:
    best = 0.0
    for r in range(len(candidates) + 1):
        for subset in itertools.combinations(candidates, r):
            cost = sum(c.token_cost for c in subset)
            if cost <= budget:
                value = sum(c.relevance_score for c in subset)
                best = max(best, value)
    return best


def _total_relevance(candidates: list[KnapsackCandidate], selected_ids: list[str]) -> float:
    by_id = {c.symbol_id: c for c in candidates}
    return sum(by_id[sid].relevance_score for sid in selected_ids)


def _total_cost(candidates: list[KnapsackCandidate], selected_ids: list[str]) -> int:
    by_id = {c.symbol_id: c for c in candidates}
    return sum(by_id[sid].token_cost for sid in selected_ids)


def test_knapsack_exact_matches_brute_force_optimal():
    optimal_value = _brute_force_optimal(CANDIDATES, BUDGET)
    selected = knapsack_exact(CANDIDATES, BUDGET)

    assert _total_cost(CANDIDATES, selected) <= BUDGET
    assert _total_relevance(CANDIDATES, selected) == optimal_value


def test_knapsack_exact_never_exceeds_budget_across_several_budgets():
    for budget in [0, 5, 10, 20, 30, 50, 100]:
        selected = knapsack_exact(CANDIDATES, budget)
        assert _total_cost(CANDIDATES, selected) <= budget


def test_knapsack_greedy_reaches_at_least_ninety_percent_of_optimal():
    optimal_value = _brute_force_optimal(CANDIDATES, BUDGET)
    greedy_selected = knapsack_greedy(CANDIDATES, BUDGET)
    greedy_value = _total_relevance(CANDIDATES, greedy_selected)

    assert greedy_value >= 0.9 * optimal_value


def test_knapsack_greedy_never_exceeds_budget_across_several_budgets():
    for budget in [0, 5, 10, 20, 30, 50, 100]:
        selected = knapsack_greedy(CANDIDATES, budget)
        assert _total_cost(CANDIDATES, selected) <= budget


def test_oversized_candidate_excluded_without_error():
    candidates = [
        KnapsackCandidate("small", relevance_score=0.5, token_cost=5),
        KnapsackCandidate("huge", relevance_score=0.99, token_cost=1000),
    ]
    exact_selected = knapsack_exact(candidates, budget=10)
    greedy_selected = knapsack_greedy(candidates, budget=10)

    assert "huge" not in exact_selected
    assert "huge" not in greedy_selected
    assert "small" in exact_selected
    assert "small" in greedy_selected


def test_greedy_ties_broken_by_relevance_score_descending():
    # Two candidates with identical ratio (relevance/cost) — the one with
    # higher relevance_score should be preferred when only one fits.
    candidates = [
        KnapsackCandidate("low", relevance_score=0.2, token_cost=10),
        KnapsackCandidate("high", relevance_score=0.4, token_cost=20),
    ]
    selected = knapsack_greedy(candidates, budget=20)
    assert selected == ["high"]
