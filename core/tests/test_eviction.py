"""Tests for governor/eviction.py (docs/phase-3-resource-allocation.md §8)."""

from loupe_core.governor.eviction import EvictionCache


def test_decay_step_produces_exactly_relevance_times_decay_factor():
    cache = EvictionCache(decay_factor=0.85)
    cache.add_or_refresh("a", relevance_score=0.5)
    cache.decay_step()
    assert cache.current_scores["a"] == 0.5 * 0.85


def test_refresh_resets_turns_since_ref_and_restores_full_score_after_two_decays():
    cache = EvictionCache(decay_factor=0.85)
    cache.add_or_refresh("a", relevance_score=0.5)
    cache.decay_step()  # turn 1: score = 0.5 * 0.85
    cache.decay_step()  # turn 2: score = 0.5 * 0.85^2
    assert cache.turns_since_ref["a"] == 2
    assert cache.current_scores["a"] != 0.5

    cache.add_or_refresh("a", relevance_score=0.5)  # referenced again
    assert cache.turns_since_ref["a"] == 0
    assert cache.current_scores["a"] == 0.5


def test_eviction_picks_lowest_current_score_not_a_stale_heap_entry():
    """Deliberately constructs a stale low heap entry that a naive (non-lazy-
    deletion-aware) implementation would evict, even though the resident's
    CURRENT score has since risen well above another resident's."""
    cache = EvictionCache()
    cache.add_or_refresh("x", relevance_score=0.05)  # starts very low
    cache.add_or_refresh("y", relevance_score=0.5)  # starts higher than x

    # x is referenced again this same session with a much higher relevance —
    # its current score is now 0.9, but the original (0.05, "x") heap entry
    # is still sitting in the heap, stale.
    cache.add_or_refresh("x", relevance_score=0.9)
    assert cache.current_scores["x"] == 0.9

    # y is genuinely the lowest current score now (0.5 < 0.9). Simulate a
    # later turn so y is no longer protected and is evictable.
    cache.decay_step()
    evicted = cache.evict_lowest()

    assert evicted == "y", "the stale (0.05, 'x') entry must not cause x to be evicted"
    assert cache.is_resident("x")


def test_symbol_added_this_turn_is_never_evicted_within_the_same_turn():
    cache = EvictionCache()
    cache.add_or_refresh("old_resident", relevance_score=0.5)
    cache.decay_step()  # turn boundary: old_resident ages, protection clears

    # brand new symbol, added THIS turn, with a lower score than old_resident's
    # current (decayed) score — it would be "correct" by score alone, but
    # must be protected since it was just added this turn.
    cache.add_or_refresh("new_symbol", relevance_score=0.01)
    assert cache.current_scores["old_resident"] == 0.5 * 0.85 > cache.current_scores["new_symbol"]

    evicted = cache.evict_lowest()

    assert evicted == "old_resident", "the just-added lower-scored symbol must be protected this turn"
    assert cache.is_resident("new_symbol")


def test_evict_lowest_returns_none_when_only_resident_is_protected():
    cache = EvictionCache()
    cache.add_or_refresh("only", relevance_score=0.1)
    assert cache.evict_lowest() is None
    assert cache.is_resident("only")


def test_evict_lowest_returns_none_on_empty_cache():
    cache = EvictionCache()
    assert cache.evict_lowest() is None


def test_repeated_eviction_drains_residents_in_ascending_score_order():
    cache = EvictionCache()
    cache.add_or_refresh("a", relevance_score=0.3)
    cache.add_or_refresh("b", relevance_score=0.1)
    cache.add_or_refresh("c", relevance_score=0.2)
    cache.decay_step()  # clear protection so all three are evictable

    order = [cache.evict_lowest(), cache.evict_lowest(), cache.evict_lowest()]
    assert order == ["b", "c", "a"]
    assert cache.evict_lowest() is None
