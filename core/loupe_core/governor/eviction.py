"""Priority-eviction cache: lazy-deletion heap over decaying symbol scores.

Implements docs/phase-3-resource-allocation.md §6. `heapq` has no
decrease-key operation, so a score change is handled by pushing a *new*
`(score, symbol_id)` tuple rather than mutating an existing one — the old
tuple is left in the heap as a stale entry, discarded lazily the next time
it's popped and found not to match `current_scores`.
"""

from __future__ import annotations

import heapq

DEFAULT_DECAY_FACTOR = 0.85


class EvictionCache:
    """Tracks resident symbols' decaying relevance and evicts the lowest-value one on demand."""

    def __init__(self, decay_factor: float = DEFAULT_DECAY_FACTOR) -> None:
        self.decay_factor = decay_factor
        self._base_relevance: dict[str, float] = {}
        self.current_scores: dict[str, float] = {}
        self.turns_since_ref: dict[str, int] = {}
        self._heap: list[tuple[float, str]] = []
        self._protected_this_turn: set[str] = set()

    def is_resident(self, symbol_id: str) -> bool:
        return symbol_id in self.current_scores

    def add_or_refresh(self, symbol_id: str, relevance_score: float) -> None:
        """A symbol enters residency, or an already-resident one is referenced again this turn.

        Resets `turns_since_ref` to 0 and the score to the full, undecayed
        `relevance_score`. Protected from eviction for the remainder of this
        turn (until the next `decay_step`), even if its score is technically
        the lowest — you cannot evict something you're in the middle of serving.
        """
        self._base_relevance[symbol_id] = relevance_score
        self.turns_since_ref[symbol_id] = 0
        self.current_scores[symbol_id] = relevance_score
        heapq.heappush(self._heap, (relevance_score, symbol_id))
        self._protected_this_turn.add(symbol_id)

    def decay_step(self) -> None:
        """Start-of-turn: age every resident by one turn and decay its score.

        Also clears last turn's eviction protection — nothing is protected
        at the start of a fresh turn until it's added/refreshed again.
        """
        self._protected_this_turn.clear()
        for symbol_id in list(self.current_scores):
            self.turns_since_ref[symbol_id] += 1
            new_score = self._base_relevance[symbol_id] * (self.decay_factor ** self.turns_since_ref[symbol_id])
            self.current_scores[symbol_id] = new_score
            heapq.heappush(self._heap, (new_score, symbol_id))

    def evict_lowest(self) -> str | None:
        """Remove and return the lowest-current-score, non-protected resident.

        Returns None if every remaining resident is protected (nothing
        evictable this turn) or there are no residents at all.
        """
        skipped_protected: list[tuple[float, str]] = []
        evicted_id: str | None = None

        while self._heap:
            score, symbol_id = heapq.heappop(self._heap)
            if symbol_id not in self.current_scores or self.current_scores[symbol_id] != score:
                continue  # stale entry: this symbol's score has since changed (or it's gone) — discard
            if symbol_id in self._protected_this_turn:
                skipped_protected.append((score, symbol_id))
                continue
            del self.current_scores[symbol_id]
            del self.turns_since_ref[symbol_id]
            del self._base_relevance[symbol_id]
            evicted_id = symbol_id
            break

        for entry in skipped_protected:
            heapq.heappush(self._heap, entry)

        return evicted_id
