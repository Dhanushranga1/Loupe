"""Thompson sampling bandit over retrieval strategies, per query intent (docs/phase-6-closing-the-loop.md §6).

A separate, independent Beta-distribution bandit per intent bucket — the
simplified, discrete-context version of a contextual bandit (§1's scoping).
Decision-time and reward-update-time are explicitly decoupled via a
`PendingDecision` record, since the reward isn't known until the outcome
backfill job (`eval/backfill.py`) resolves it, potentially minutes after the
arm was already chosen and used — the part most introductory bandit
explanations skip over entirely.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

ARMS = ("rrf", "learned_ranker")


@dataclass
class _BetaArm:
    alpha: float = 1.0
    beta: float = 1.0


@dataclass
class PendingDecision:
    retrieval_log_id: str
    intent: str
    arm_chosen: str
    top_candidate_symbol_id: str


class ThompsonBandit:
    """One independent Beta(1,1)-per-arm bandit per intent category.

    Uniform (1, 1) prior — no assumed advantage for either strategy at the
    start. Before the ranker clears its cold-start threshold, `learned_ranker`
    simply loses every real comparison (it isn't actually usable yet), so the
    bandit converges to `rrf` naturally rather than needing special-casing
    for "only one real arm exists yet."
    """

    def __init__(self, rng: random.Random | None = None) -> None:
        self._rng = rng or random.Random()
        self._arms: dict[tuple[str, str], _BetaArm] = {}
        self._pending: dict[str, PendingDecision] = {}

    def _arm(self, intent: str, arm_name: str) -> _BetaArm:
        key = (intent, arm_name)
        if key not in self._arms:
            self._arms[key] = _BetaArm()
        return self._arms[key]

    def get_alpha_beta(self, intent: str, arm_name: str) -> tuple[float, float]:
        arm = self._arm(intent, arm_name)
        return arm.alpha, arm.beta

    def select_arm(self, intent: str, retrieval_log_id: str, top_candidate_symbol_id: str) -> str:
        """Sample each arm's Beta distribution, pick the higher sample, record a PendingDecision.

        The reward for this decision is NOT applied here — see `resolve_outcome`.
        """
        samples = {arm_name: self._rng.betavariate(*self.get_alpha_beta(intent, arm_name)) for arm_name in ARMS}
        chosen = max(samples, key=lambda name: samples[name])

        self._pending[retrieval_log_id] = PendingDecision(
            retrieval_log_id=retrieval_log_id,
            intent=intent,
            arm_chosen=chosen,
            top_candidate_symbol_id=top_candidate_symbol_id,
        )
        return chosen

    def resolve_outcome(self, retrieval_log_id: str, symbol_edited: bool) -> None:
        """Apply the delayed Beta update for a decision, once its outcome is known.

        A `retrieval_log_id` with no matching `PendingDecision` (already
        resolved, or never tracked) is a no-op. A decision whose outcome
        never resolves is simply never applied — it neither helps nor hurts
        either arm, matching the backfill job's rule that an unresolved
        outcome carries no information.
        """
        decision = self._pending.pop(retrieval_log_id, None)
        if decision is None:
            return
        reward = 1 if symbol_edited else 0
        arm = self._arm(decision.intent, decision.arm_chosen)
        arm.alpha += reward
        arm.beta += 1 - reward

    def pending_count(self) -> int:
        """Number of decisions awaiting resolution — for test/inspection use."""
        return len(self._pending)
