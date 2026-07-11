"""0/1 knapsack symbol selection under a token budget (docs/phase-3-resource-allocation.md §5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnapsackCandidate:
    symbol_id: str
    relevance_score: float
    token_cost: int


def knapsack_exact(candidates: list[KnapsackCandidate], budget: int) -> list[str]:
    """Classic 0/1 knapsack via DP, indexed by integer token-budget units.

    Guaranteed optimal. Used only in tests as the ground truth `knapsack_greedy`
    is checked against — `O(n * budget)` is unnecessary in the live retrieval
    path when the fast approximation is good enough.
    """
    n = len(candidates)
    dp = [[0.0] * (budget + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        cost = candidates[i - 1].token_cost
        value = candidates[i - 1].relevance_score
        for b in range(budget + 1):
            dp[i][b] = dp[i - 1][b]
            if cost <= b:
                with_item = dp[i - 1][b - cost] + value
                if with_item > dp[i][b]:
                    dp[i][b] = with_item

    selected: list[str] = []
    remaining_budget = budget
    for i in range(n, 0, -1):
        if dp[i][remaining_budget] != dp[i - 1][remaining_budget]:
            candidate = candidates[i - 1]
            selected.append(candidate.symbol_id)
            remaining_budget -= candidate.token_cost
    selected.reverse()
    return selected


def knapsack_greedy(candidates: list[KnapsackCandidate], budget: int) -> list[str]:
    """Greedy by relevance/cost ratio — the production selector (§5).

    Sorted descending by ratio (ties broken by relevance_score descending),
    then walked once: an oversized candidate is skipped, not fatal — a
    smaller candidate later in the list may still fit.
    """

    def sort_key(c: KnapsackCandidate) -> tuple[float, float]:
        ratio = c.relevance_score / c.token_cost if c.token_cost > 0 else float("inf")
        return (-ratio, -c.relevance_score)

    ordered = sorted(candidates, key=sort_key)
    selected: list[str] = []
    running_total = 0
    for candidate in ordered:
        if running_total + candidate.token_cost <= budget:
            selected.append(candidate.symbol_id)
            running_total += candidate.token_cost
    return selected
