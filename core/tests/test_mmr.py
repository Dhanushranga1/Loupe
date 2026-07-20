"""Tests for retrieval/mmr.py (docs/PhaseX/loupe-retrieval-upgrades.md §4).

Synthetic embeddings throughout, matching test_fusion.py's own precedent
(test_centrality_breaks_ties_..., test_fuse_never_introduces_...) — these
tests are about the *combination logic*, not real model embedding quality,
so hand-constructed vectors give exact, controllable "near-duplicate" vs.
"distinct" geometry that a real model wouldn't reliably reproduce on demand.
"""

from loupe_core.retrieval.mmr import mmr_select


def _unit(*components: float) -> list[float]:
    return list(components)


def test_mmr_diversifies_away_from_near_duplicate_getters():
    """§4's own fixture: 5 near-duplicate getters scoring similarly, plus 2
    genuinely distinct relevant symbols, final_top_k too small to fit all 7 —
    selection must include both distinct symbols, not exhaust every slot on
    near-duplicates alone.
    """
    # 5 getters clustered tightly around the same direction in embedding space.
    getters = {f"getter_{i}": _unit(1.0, 0.01 * i, 0.0) for i in range(5)}
    # 2 distinct symbols pointing in orthogonal directions from the getters and each other.
    distinct = {"distinct_a": _unit(0.0, 1.0, 0.0), "distinct_b": _unit(0.0, 0.0, 1.0)}
    embeddings = {**getters, **distinct}

    # Getters score slightly higher on raw relevance than the two distinct symbols —
    # a plain top-k-by-relevance sort would fill every slot with getters alone.
    candidates = [(gid, 0.9) for gid in getters] + [(sid, 0.85) for sid in distinct]

    selected = mmr_select(candidates, embeddings, final_top_k=4, lambda_param=0.7)
    selected_ids = {sid for sid, _ in selected}

    assert "distinct_a" in selected_ids
    assert "distinct_b" in selected_ids
    assert len(selected_ids & set(getters)) < 5, "must not exhaust all slots on near-duplicate getters"


def test_mmr_with_lambda_1_reduces_to_plain_relevance_sort():
    """§4's own backward-compatibility criterion: lambda_param=1.0 (diversity
    penalty fully off) must produce the identical ordering a plain
    top-k-by-relevance sort would — MMR is a strict generalization, not a
    behavior change when diversity isn't wanted.
    """
    embeddings = {
        "a": _unit(1.0, 0.0),
        "b": _unit(0.0, 1.0),
        "c": _unit(1.0, 0.0),  # identical direction to "a" — would be penalized under normal lambda
        "d": _unit(-1.0, 0.0),
    }
    candidates = [("a", 0.4), ("b", 0.9), ("c", 0.95), ("d", 0.1)]

    selected = mmr_select(candidates, embeddings, final_top_k=4, lambda_param=1.0)
    selected_ids = [sid for sid, _ in selected]

    plain_sort = sorted(candidates, key=lambda pair: (-pair[1], pair[0]))
    assert selected_ids == [sid for sid, _ in plain_sort]


def test_mmr_respects_final_top_k():
    embeddings = {"a": _unit(1.0, 0.0), "b": _unit(0.0, 1.0), "c": _unit(-1.0, 0.0)}
    candidates = [("a", 0.9), ("b", 0.8), ("c", 0.7)]

    selected = mmr_select(candidates, embeddings, final_top_k=2)

    assert len(selected) == 2


def test_mmr_ties_break_deterministically_by_symbol_id():
    embeddings = {"z": _unit(1.0, 0.0), "a": _unit(1.0, 0.0)}
    candidates = [("z", 0.5), ("a", 0.5)]

    selected = mmr_select(candidates, embeddings, final_top_k=2, lambda_param=1.0)

    assert [sid for sid, _ in selected] == ["a", "z"]


def test_mmr_missing_embedding_treated_as_zero_redundancy_not_a_crash():
    embeddings = {"a": _unit(1.0, 0.0)}  # "b" deliberately has no cached embedding
    candidates = [("a", 0.9), ("b", 0.8)]

    selected = mmr_select(candidates, embeddings, final_top_k=2)

    assert {sid for sid, _ in selected} == {"a", "b"}


def test_mmr_empty_candidates_returns_empty():
    assert mmr_select([], {}, final_top_k=5) == []
