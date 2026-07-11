"""Tests for governor/session.py (docs/phase-3-resource-allocation.md §8)."""

from loupe_core.governor.knapsack import KnapsackCandidate
from loupe_core.governor.session import HARD_CEILING, SessionState, request_symbols


def test_repeated_request_across_turns_costs_nothing_the_second_time():
    session = SessionState(session_id="s1", token_budget_total=100)

    turn1 = request_symbols(session, [KnapsackCandidate("a", 0.9, 30)])
    assert turn1.newly_sent == ["a"]
    assert turn1.included == ["a"]
    assert session.token_used == 30

    turn2 = request_symbols(session, [KnapsackCandidate("a", 0.9, 30)])
    assert turn2.newly_sent == [], "an already-resident symbol must not be re-charged"
    assert turn2.included == ["a"]
    assert session.token_used == 30, "token_used must not double-count a resident symbol"


def test_symbol_exceeding_hard_ceiling_is_denied_with_no_eviction_attempted():
    session = SessionState(session_id="s1", token_budget_total=100)
    request_symbols(session, [KnapsackCandidate("a", 0.9, 40), KnapsackCandidate("b", 0.1, 40)])

    result = request_symbols(session, [KnapsackCandidate("huge", 0.99, HARD_CEILING + 1)])

    assert result.denied == ["huge"]
    assert result.evicted == [], "a hard-ceiling violator must never trigger eviction on its behalf"
    assert set(session.symbols_in_context) == {"a", "b"}, "existing residents must be untouched"


def test_new_high_value_candidate_evicts_lowest_scoring_resident_to_fit():
    session = SessionState(session_id="s1", token_budget_total=100)
    request_symbols(session, [KnapsackCandidate("a", 0.9, 40), KnapsackCandidate("b", 0.1, 40)])
    assert session.token_used == 80

    result = request_symbols(session, [KnapsackCandidate("c", 0.95, 30)])

    assert result.evicted == ["b"], "the lower-relevance resident must be evicted, not 'a'"
    assert result.included == ["c"]
    assert result.newly_sent == ["c"]
    assert set(session.symbols_in_context) == {"a", "c"}
    assert session.token_used == 70


def test_evicted_symbol_is_billed_again_as_new_on_a_later_request():
    session = SessionState(session_id="s1", token_budget_total=100)
    request_symbols(session, [KnapsackCandidate("a", 0.9, 40), KnapsackCandidate("b", 0.1, 40)])
    request_symbols(session, [KnapsackCandidate("c", 0.95, 30)])  # evicts b

    result = request_symbols(session, [KnapsackCandidate("b", 0.1, 40)])

    assert "b" in result.newly_sent, "a previously-evicted symbol must be billed again, not treated as free"


def test_candidate_alone_exceeding_remaining_budget_and_unevictable_is_denied():
    session = SessionState(session_id="s1", token_budget_total=50)
    request_symbols(session, [KnapsackCandidate("only", 0.9, 50)])  # fills the whole budget, sole resident

    # "big" needs more room than exists even after evicting everything (evicting
    # "only" wouldn't help since "big" alone still can't fit in a 50-token budget).
    result = request_symbols(session, [KnapsackCandidate("big", 0.99, 60)])

    assert "big" in result.denied
    assert result.evicted == ["only"], "eviction is attempted (unlike the hard-ceiling case) even though it can't help"


def test_full_multi_turn_session_simulation_matches_hand_computed_state():
    """5-turn scripted session: new requests, a repeat, and a forced eviction."""
    session = SessionState(session_id="sim", token_budget_total=100)

    # Turn 1: a(0.9, cost 40) and b(0.2, cost 40) both fit. used=80.
    r1 = request_symbols(session, [KnapsackCandidate("a", 0.9, 40), KnapsackCandidate("b", 0.2, 40)])
    assert r1.included == ["a", "b"] and r1.newly_sent == ["a", "b"] and r1.evicted == [] and r1.denied == []
    assert session.token_used == 80

    # Turn 2: repeat 'a' only (b not requested, but stays resident and just decays).
    r2 = request_symbols(session, [KnapsackCandidate("a", 0.9, 40)])
    assert r2.included == ["a"] and r2.newly_sent == []
    assert session.token_used == 80
    assert session.symbols_in_context["a"].turns_since_ref == 0  # refreshed this turn
    assert session.symbols_in_context["b"].turns_since_ref == 1  # decayed, not requested
    assert session.symbols_in_context["b"].relevance_score == 0.2 * 0.85

    # Turn 3: new candidate c(0.95, cost 30) needs room; b (lower score, now
    # decayed to 0.2*0.85=0.17) is evicted to fit it. remaining before eviction
    # = 100-80=20 < 30, so eviction triggers; after evicting b, remaining=60.
    r3 = request_symbols(session, [KnapsackCandidate("c", 0.95, 30)])
    assert r3.evicted == ["b"]
    assert r3.included == ["c"] and r3.newly_sent == ["c"]
    assert session.token_used == 40 + 30  # a(40) + c(30), b's 40 freed
    assert set(session.symbols_in_context) == {"a", "c"}

    # Turn 4: request b again (evicted at turn 3, so billed as new). Entering
    # this turn: a has decayed twice since its turn-2 refresh (0.9*0.85^2 =
    # 0.65025, turns_since_ref=2), c has decayed once since turn 3
    # (0.95*0.85 = 0.8075, turns_since_ref=1). used=70, remaining=30, b
    # costs 40 — doesn't fit. Neither a nor c was requested this turn, so
    # neither is protected; a's current score (0.65025) is now the lowest
    # (below c's 0.8075), so a — not c — is evicted to make room.
    r4 = request_symbols(session, [KnapsackCandidate("b", 0.2, 40)])
    assert r4.evicted == ["a"]
    assert r4.included == ["b"] and r4.newly_sent == ["b"]
    assert set(session.symbols_in_context) == {"b", "c"}
    assert session.token_used == 30 + 40  # c(30) + b(40), a's 40 freed
    assert session.token_used <= session.token_budget_total

    # Turn 5: nothing new requested, just let existing residents decay further.
    r5 = request_symbols(session, [])
    assert r5.included == [] and r5.newly_sent == [] and r5.denied == []
    assert session.token_used == 70  # unchanged — decay affects scores, not token accounting
    assert session.symbols_in_context["b"].turns_since_ref == 1
    assert session.symbols_in_context["c"].turns_since_ref == 2


def test_token_used_never_exceeds_budget_total_across_many_turns():
    session = SessionState(session_id="stress", token_budget_total=200)
    for i in range(20):
        request_symbols(session, [KnapsackCandidate(f"sym-{i}", 0.5 + (i % 5) / 10, 15 + (i % 7))])
        assert session.token_used <= session.token_budget_total
