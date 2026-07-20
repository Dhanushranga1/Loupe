"""Session-level context tracking and governed symbol selection.

Implements docs/phase-3-resource-allocation.md §7 — this is where "already
resident costs zero tokens" gets formalized, which is what keeps a long
session cheap over time.

Two resolved spec loosenesses worth recording:
- §7's `request_symbols` signature shows a `mode` parameter, but the 5-step
  algorithm never references it anywhere — omitted here as unspecified
  rather than threaded through for no defined purpose.
- "free room... then retry the knapsack pass once" doesn't pin down exactly
  how much room counts as "enough." This implementation evicts the lowest-
  scoring residents one at a time until the *cheapest* currently-unselected
  new candidate would fit, then retries `knapsack_greedy` exactly once —
  a concrete, documented, testable interpretation of an intentionally loose
  paragraph.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .eviction import EvictionCache
from .knapsack import KnapsackCandidate, knapsack_greedy

HARD_CEILING = 20000
DEFAULT_BUDGET = 6000


@dataclass
class ResidentSymbol:
    relevance_score: float
    turns_since_ref: int
    token_cost: int


@dataclass
class SelectionResult:
    included: list[str] = field(default_factory=list)
    newly_sent: list[str] = field(default_factory=list)
    evicted: list[str] = field(default_factory=list)
    denied: list[str] = field(default_factory=list)


@dataclass
class SessionState:
    session_id: str
    token_budget_total: int = DEFAULT_BUDGET
    token_used: int = 0
    symbols_in_context: dict[str, ResidentSymbol] = field(default_factory=dict)
    eviction: EvictionCache = field(default_factory=EvictionCache, repr=False)
    # Phase 14 §4 (docs/PhaseX/phase-14-adaptive-context-compression.md):
    # ancestor symbol_ids whose shared context (signature + docstring only,
    # never the body) has already been sent and charged this session — a
    # sibling method requested afterward is charged the marginal cost only,
    # via governor/budget.py's symbol_extraction_marginal_cost.
    shared_context_charged: set[str] = field(default_factory=set)


def request_symbols(session: SessionState, candidates: list[KnapsackCandidate]) -> SelectionResult:
    """Decide which candidates are available this turn, governed by the session's budget."""
    result = SelectionResult()

    # Step 1: decay every current resident by one turn.
    session.eviction.decay_step()
    for symbol_id, resident in session.symbols_in_context.items():
        resident.relevance_score = session.eviction.current_scores[symbol_id]
        resident.turns_since_ref = session.eviction.turns_since_ref[symbol_id]

    already_resident = [c for c in candidates if session.eviction.is_resident(c.symbol_id)]
    new_candidates = [c for c in candidates if not session.eviction.is_resident(c.symbol_id)]

    # Step 2: already-resident candidates cost nothing new this turn.
    for candidate in already_resident:
        session.eviction.add_or_refresh(candidate.symbol_id, candidate.relevance_score)
        resident = session.symbols_in_context[candidate.symbol_id]
        resident.relevance_score = candidate.relevance_score
        resident.turns_since_ref = 0
        result.included.append(candidate.symbol_id)

    # Step 4: hard-ceiling violators are denied outright — no eviction attempted for them.
    fittable_new = []
    for candidate in new_candidates:
        if candidate.token_cost > HARD_CEILING:
            result.denied.append(candidate.symbol_id)
        else:
            fittable_new.append(candidate)

    # Step 3: fit what we can now; if something new doesn't fit, evict the
    # lowest-value residents to make room, then retry the knapsack pass once.
    remaining_budget = session.token_budget_total - session.token_used
    selected_ids = set(knapsack_greedy(fittable_new, remaining_budget))
    unselected = [c for c in fittable_new if c.symbol_id not in selected_ids]

    if unselected:
        cheapest_unselected_cost = min(c.token_cost for c in unselected)
        while remaining_budget < cheapest_unselected_cost:
            evicted_id = session.eviction.evict_lowest()
            if evicted_id is None:
                break
            freed_cost = session.symbols_in_context.pop(evicted_id).token_cost
            session.token_used -= freed_cost
            result.evicted.append(evicted_id)
            remaining_budget = session.token_budget_total - session.token_used
        selected_ids = set(knapsack_greedy(fittable_new, remaining_budget))

    for candidate in fittable_new:
        if candidate.symbol_id in selected_ids:
            session.eviction.add_or_refresh(candidate.symbol_id, candidate.relevance_score)
            session.symbols_in_context[candidate.symbol_id] = ResidentSymbol(
                relevance_score=candidate.relevance_score, turns_since_ref=0, token_cost=candidate.token_cost
            )
            session.token_used += candidate.token_cost
            result.included.append(candidate.symbol_id)
            result.newly_sent.append(candidate.symbol_id)
        else:
            result.denied.append(candidate.symbol_id)

    return result
