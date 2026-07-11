#!/usr/bin/env python
"""Manual inspection tool: run a scripted turn sequence and print the governor's behavior.

Usage: python scripts/simulate_session.py

Prints the SelectionResult and full resident state after each turn — useful
for eyeballing decay and eviction behavior interactively (docs/phase-3-
resource-allocation.md §9, task 10). The script below is the same scenario
exercised by test_session.py's multi-turn simulation test.
"""

from __future__ import annotations

from loupe_core.governor.knapsack import KnapsackCandidate
from loupe_core.governor.session import SessionState, request_symbols

# Each turn: a list of (symbol_id, relevance_score, token_cost) requests.
SCRIPT: list[list[tuple[str, float, int]]] = [
    [("a", 0.9, 40), ("b", 0.2, 40)],  # turn 1: both fit
    [("a", 0.9, 40)],  # turn 2: repeat a, b just decays untouched
    [("c", 0.95, 30)],  # turn 3: new high-value symbol forces an eviction
    [("b", 0.2, 40)],  # turn 4: re-request the evicted b — billed as new again
    [],  # turn 5: nothing requested, just let residents decay
]


def print_resident_state(session: SessionState) -> None:
    if not session.symbols_in_context:
        print("    (no residents)")
        return
    for symbol_id, resident in sorted(session.symbols_in_context.items()):
        print(
            f"    {symbol_id:10s} score={resident.relevance_score:.4f}  "
            f"turns_since_ref={resident.turns_since_ref}  cost={resident.token_cost}"
        )


def main() -> int:
    session = SessionState(session_id="sim", token_budget_total=100)
    print(f"Session {session.session_id!r} — budget {session.token_budget_total} tokens\n")

    for turn_index, requests in enumerate(SCRIPT, start=1):
        candidates = [KnapsackCandidate(sid, score, cost) for sid, score, cost in requests]
        result = request_symbols(session, candidates)

        print(f"Turn {turn_index}: requested {[c.symbol_id for c in candidates] or '(none)'}")
        print(f"  included:   {result.included}")
        print(f"  newly_sent: {result.newly_sent}")
        print(f"  evicted:    {result.evicted}")
        print(f"  denied:     {result.denied}")
        print(f"  token_used: {session.token_used}/{session.token_budget_total}")
        print("  residents:")
        print_resident_state(session)
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
