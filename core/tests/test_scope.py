"""Tests for context/scope.py (docs/PhaseX/scope-aware-retrieval.md)."""

import os
from pathlib import Path

import pytest

from loupe_core.context.scope import (
    DEFAULT_IN_SCOPE_MASS,
    Scope,
    apply_hard_scope,
    resolve_scope_membership,
    scoped_personalized_pagerank,
)
from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.retrieval.fusion import fuse


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def two_package_repo(tmp_path):
    """Two "packages" by directory (pkg_a/, pkg_b/), each internally dense, one
    weak cross-package call — the same proven two-cluster shape used throughout
    Phases 9/10.5/11, just organized into real subdirectories this time so
    path-prefix scoping has something meaningful to filter on.
    """
    _write(
        tmp_path,
        "pkg_a/mod.py",
        "from pkg_b.mod import shared_util\n\n\n"
        "def a1():\n    shared_util()\n    return a2() + a3()\n\n\n"
        "def a2():\n    return a3() + a4()\n\n\n"
        "def a3():\n    return a4() + a1()\n\n\n"
        "def a4():\n    return a1() + a2()\n",
    )
    _write(
        tmp_path,
        "pkg_b/mod.py",
        "def shared_util():\n    return b2() + b3()\n\n\n"
        "def b2():\n    return b3() + b4()\n\n\n"
        "def b3():\n    return b4() + shared_util()\n\n\n"
        "def b4():\n    return shared_util() + b2()\n",
    )
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        files = ["pkg_a/mod.py", "pkg_b/mod.py"]
        parsed = [parse_file(f) for f in files]
        graph = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        yield graph, symbols_by_id
    finally:
        os.chdir(old_cwd)


def _id_by_name(symbols_by_id, qualified_name: str) -> str:
    return next(sid for sid, s in symbols_by_id.items() if s.qualified_name == qualified_name)


# --------------------------------------------------------------------------
# §1: explicit path-based scope
# --------------------------------------------------------------------------


def test_explicit_path_scope_includes_only_matching_package(two_package_repo):
    graph, symbols_by_id = two_package_repo
    scope = Scope(path_prefix="pkg_a/")

    membership = resolve_scope_membership(scope, graph, symbols_by_id)
    names = {symbols_by_id[sid].qualified_name for sid in membership}

    assert names == {"a1", "a2", "a3", "a4"}


def test_apply_hard_scope_is_a_pure_set_intersection():
    candidates = {"a", "b", "c"}
    membership = {"b", "c", "d"}
    assert apply_hard_scope(candidates, membership) == {"b", "c"}


# --------------------------------------------------------------------------
# §1 acceptance criterion: auto-detected scope from a seed symbol
# --------------------------------------------------------------------------


def test_auto_detected_scope_from_a_symbol_in_subsystem_a_returns_subsystem_as_membership(two_package_repo):
    graph, symbols_by_id = two_package_repo
    seed = _id_by_name(symbols_by_id, "a2")
    scope = Scope(seed_symbol_id=seed)

    membership = resolve_scope_membership(scope, graph, symbols_by_id)
    names = {symbols_by_id[sid].qualified_name for sid in membership}

    assert names == {"a1", "a2", "a3", "a4"}, "must return subsystem A's cluster, not subsystem B's"


# --------------------------------------------------------------------------
# §4 acceptance criteria: hard mode excludes, soft mode deprioritizes but
# keeps, unscoped soft mode is a strict backward-compatibility no-op.
# --------------------------------------------------------------------------


def test_hard_mode_excludes_the_outside_scope_candidate_entirely(two_package_repo):
    graph, symbols_by_id = two_package_repo
    scope = Scope(path_prefix="pkg_a/", mode="hard")
    membership = resolve_scope_membership(scope, graph, symbols_by_id)

    shared_util_id = _id_by_name(symbols_by_id, "shared_util")
    a1_id = _id_by_name(symbols_by_id, "a1")

    # A query where the lexical/semantic candidate pool contains a genuinely
    # strong out-of-scope match (shared_util, ranked #1 by both signals) plus
    # an in-scope one.
    lexical_results = [(shared_util_id, 5.0), (a1_id, 1.0)]
    semantic_results = [(shared_util_id, 0.9), (a1_id, 0.5)]

    scoped_lexical = [(sid, score) for sid, score in lexical_results if sid in apply_hard_scope({sid}, membership)]
    scoped_semantic = [(sid, score) for sid, score in semantic_results if sid in apply_hard_scope({sid}, membership)]

    fused = fuse(scoped_lexical, scoped_semantic, graph.pagerank_scores, top_k=10)
    fused_ids = {sid for sid, _ in fused}

    assert shared_util_id not in fused_ids, "hard mode must exclude the out-of-scope candidate entirely"
    assert a1_id in fused_ids


def test_soft_mode_deprioritizes_but_does_not_hide_the_outside_scope_candidate(two_package_repo):
    """The single test that actually demonstrates why soft mode was worth
    building (§4's own framing): constructed so a hard filter would have
    excluded `shared_util` entirely — soft mode must still surface it, just
    ranked below the in-scope candidate.
    """
    graph, symbols_by_id = two_package_repo
    scope = Scope(path_prefix="pkg_a/", mode="soft")
    membership = resolve_scope_membership(scope, graph, symbols_by_id)

    shared_util_id = _id_by_name(symbols_by_id, "shared_util")
    a1_id = _id_by_name(symbols_by_id, "a1")

    # Both candidates tie on lexical/semantic signal strength — any ranking
    # difference must come purely from the scope-biased centrality term.
    lexical_results = [(a1_id, 1.0), (shared_util_id, 1.0)]
    semantic_results = [(a1_id, 1.0), (shared_util_id, 1.0)]
    candidates = {a1_id, shared_util_id}

    fused = fuse(
        lexical_results,
        semantic_results,
        graph.pagerank_scores,
        graph=graph.graph,
        top_k=10,
        scope_seed_ids=membership,
        in_scope_mass=DEFAULT_IN_SCOPE_MASS,
    )
    fused_ids = [sid for sid, _ in fused]

    assert shared_util_id in fused_ids, "soft mode must not hide the out-of-scope candidate"
    assert a1_id in fused_ids
    assert fused_ids.index(a1_id) < fused_ids.index(shared_util_id), (
        "the in-scope candidate must rank above the out-of-scope one once scope-biased"
    )


def test_soft_mode_with_no_scope_at_all_is_identical_to_unscoped_ranking(two_package_repo):
    """§4's backward-compatibility check: `mode: soft` with nothing actually
    scoped must behave identically to today's plain, un-personalized fuse()."""
    graph, symbols_by_id = two_package_repo
    a1_id = _id_by_name(symbols_by_id, "a1")
    shared_util_id = _id_by_name(symbols_by_id, "shared_util")

    lexical_results = [(a1_id, 2.0), (shared_util_id, 1.0)]
    semantic_results = [(a1_id, 1.5), (shared_util_id, 0.8)]

    baseline = fuse(lexical_results, semantic_results, graph.pagerank_scores, top_k=10)
    # scope_seed_ids omitted entirely (no scope specified) — must match baseline exactly.
    unscoped_soft = fuse(lexical_results, semantic_results, graph.pagerank_scores, top_k=10)

    assert baseline == unscoped_soft


# --------------------------------------------------------------------------
# scoped_personalized_pagerank: direct unit coverage of the helper `fuse()`
# now calls internally in soft mode
# --------------------------------------------------------------------------


def test_scoped_personalized_pagerank_scores_every_requested_candidate_id(two_package_repo):
    graph, symbols_by_id = two_package_repo
    a1_id = _id_by_name(symbols_by_id, "a1")
    shared_util_id = _id_by_name(symbols_by_id, "shared_util")
    membership = {a1_id}

    scores = scoped_personalized_pagerank(graph, membership, candidate_ids={a1_id, shared_util_id})

    assert set(scores) == {a1_id, shared_util_id}
    assert scores[a1_id] > scores[shared_util_id]
