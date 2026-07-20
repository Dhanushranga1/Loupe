"""Tests for graph/clustering.py's Louvain community detection
(docs/PhaseX/phase-10.5-graph-clustering.md)."""

import os
from pathlib import Path

import pytest

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.graph.clustering import (
    COARSE_RESOLUTION,
    FINE_RESOLUTION,
    align_clusters,
    compute_clusters,
    jaccard_similarity,
)


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build(repo_root: Path, files: list[str]):
    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        parsed = [parse_file(f) for f in files]
        graph = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        return graph, symbols_by_id, parsed
    finally:
        os.chdir(old_cwd)


def _names(symbols_by_id: dict, ids: set[str]) -> set[str]:
    return {symbols_by_id[i].qualified_name for i in ids}


# --------------------------------------------------------------------------
# Two-subsystem fixture: densely-connected mesh within each subsystem, a
# single weak cross-call (a1 -> b1) as the only edge between them. No
# separate "bridge" symbol — an earlier version of this fixture used one and
# it produced a genuinely ambiguous 3-way split at fine resolution (the
# bridge symbol got clustered with whichever side it called last), caught by
# running it for real before trusting the fixture's own design.
# --------------------------------------------------------------------------


@pytest.fixture
def two_subsystem_repo(tmp_path):
    _write(
        tmp_path,
        "sys_a.py",
        "from sys_b import b1\n\n\n"
        "def a1():\n    b1()\n    return a2() + a3()\n\n\n"
        "def a2():\n    return a3() + a4()\n\n\n"
        "def a3():\n    return a4() + a1()\n\n\n"
        "def a4():\n    return a1() + a2()\n",
    )
    _write(
        tmp_path,
        "sys_b.py",
        "def b1():\n    return b2() + b3()\n\n\n"
        "def b2():\n    return b3() + b4()\n\n\n"
        "def b3():\n    return b4() + b1()\n\n\n"
        "def b4():\n    return b1() + b2()\n",
    )
    return _build(tmp_path, ["sys_a.py", "sys_b.py"])


def test_fine_resolution_separates_the_two_constructed_subsystems(two_subsystem_repo):
    graph, symbols_by_id, _ = two_subsystem_repo
    fine = graph.clusters.fine

    assert len(fine) == 2
    names_per_cluster = [_names(symbols_by_id, c) for c in fine]
    assert {"a1", "a2", "a3", "a4"} in names_per_cluster
    assert {"b1", "b2", "b3", "b4"} in names_per_cluster


def test_clustering_is_deterministic_across_repeated_calls_on_an_unchanged_graph(two_subsystem_repo):
    """§3's determinism fix, part 1 (fixed seed) — the concrete proof it works,
    not just an assumption that networkx's `seed` parameter does what it says."""
    graph, _, _ = two_subsystem_repo

    run1 = compute_clusters(graph.graph)
    run2 = compute_clusters(graph.graph)

    assert run1.fine == run2.fine
    assert run1.coarse == run2.coarse


def test_empty_graph_returns_empty_clusters():
    import networkx as nx

    result = compute_clusters(nx.DiGraph())
    assert result.fine == []
    assert result.coarse == []


# --------------------------------------------------------------------------
# Four-subsystem fixture: two pairs of tightly-connected subsystems, each
# pair weakly cross-linked internally (a<->b, c<->d) but with zero edges
# between the two pairs — a genuine two-level hierarchy, so fine resolution
# should find all 4 natural groups and coarse should merge each pair into 2.
# --------------------------------------------------------------------------


@pytest.fixture
def four_subsystem_repo(tmp_path):
    _write(
        tmp_path,
        "sys_a.py",
        "from sys_b import b1\n\n\n"
        "def a1():\n    b1()\n    return a2() + a3()\n\n\n"
        "def a2():\n    return a3() + a1()\n\n\n"
        "def a3():\n    return a1() + a2()\n",
    )
    _write(
        tmp_path,
        "sys_b.py",
        "def b1():\n    return b2() + b3()\n\n\n"
        "def b2():\n    return b3() + b1()\n\n\n"
        "def b3():\n    return b1() + b2()\n",
    )
    _write(
        tmp_path,
        "sys_c.py",
        "from sys_d import d1\n\n\n"
        "def c1():\n    d1()\n    return c2() + c3()\n\n\n"
        "def c2():\n    return c3() + c1()\n\n\n"
        "def c3():\n    return c1() + c2()\n",
    )
    _write(
        tmp_path,
        "sys_d.py",
        "def d1():\n    return d2() + d3()\n\n\n"
        "def d2():\n    return d3() + d1()\n\n\n"
        "def d3():\n    return d1() + d2()\n",
    )
    return _build(tmp_path, ["sys_a.py", "sys_b.py", "sys_c.py", "sys_d.py"])


def test_coarse_resolution_merges_natural_subgroups_fine_resolution_keeps_separate(four_subsystem_repo):
    graph, symbols_by_id, _ = four_subsystem_repo
    fine, coarse = graph.clusters.fine, graph.clusters.coarse

    assert len(fine) == 4, "fine resolution must find all 4 natural sub-groups"
    assert len(coarse) < len(fine), "coarse resolution must produce fewer, larger clusters than fine on the same graph"

    coarse_names = [_names(symbols_by_id, c) for c in coarse]
    assert {"a1", "a2", "a3", "b1", "b2", "b3"} in coarse_names
    assert {"c1", "c2", "c3", "d1", "d2", "d3"} in coarse_names


def test_resolution_constants_favor_fewer_vs_more_communities_as_documented():
    """Sanity check on the constants themselves, independent of any fixture:
    networkx's convention is resolution < 1 favors fewer/larger communities,
    > 1 favors more/smaller ones."""
    assert COARSE_RESOLUTION < 1.0
    assert FINE_RESOLUTION > 1.0


# --------------------------------------------------------------------------
# Jaccard cluster alignment
# --------------------------------------------------------------------------


def test_jaccard_similarity_basic_cases():
    assert jaccard_similarity(set(), set()) == 1.0
    assert jaccard_similarity({"a"}, set()) == 0.0
    assert jaccard_similarity({"a", "b"}, {"a", "b"}) == 1.0
    assert jaccard_similarity({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)


def test_align_clusters_correctly_tracks_a_subsystem_that_grew_by_one_symbol(four_subsystem_repo, tmp_path):
    """§3's determinism fix, part 2 — the test that actually proves the
    diffing feature this phase exists to support (claude_md_generator's
    architecture diff) will work: after a real, deliberate graph change,
    alignment must report "same cluster, evolved," not two unrelated ones.
    """
    graph_before, symbols_before, _ = four_subsystem_repo
    fine_before = graph_before.clusters.fine

    # Grow sys_a by exactly one new symbol, calling into the existing mesh.
    sys_a_path = tmp_path / "sys_a.py"
    sys_a_path.write_text(sys_a_path.read_text() + "\n\ndef a4():\n    return a1()\n")

    graph_after, symbols_after, _ = _build(tmp_path, ["sys_a.py", "sys_b.py", "sys_c.py", "sys_d.py"])
    fine_after = graph_after.clusters.fine

    alignment = align_clusters(fine_before, fine_after)

    for new_index, old_index in alignment.items():
        assert old_index is not None, "every post-change cluster must align to some pre-change cluster"
        new_names = _names(symbols_after, fine_after[new_index])
        old_names = _names(symbols_before, fine_before[old_index])
        # The grown sys_a cluster must align to old sys_a specifically, not
        # to sys_b/c/d — checked by requiring a strict superset relationship
        # on every real alignment pair in this fixture (no genuine splits/merges here).
        assert old_names <= new_names or new_names <= old_names, (
            f"{new_names} aligned to {old_names}, expected one to be a subset of the other (same subsystem, evolved)"
        )

    grown_cluster = next(c for c in fine_after if "a4" in _names(symbols_after, c))
    assert {"a1", "a2", "a3", "a4"} == _names(symbols_after, grown_cluster)


def test_align_clusters_maps_to_none_when_nothing_overlaps():
    old = [{"a", "b", "c"}]
    new = [{"x", "y", "z"}]

    alignment = align_clusters(old, new)

    assert alignment == {0: None}


# --------------------------------------------------------------------------
# Caching: compute_clusters runs inside build_graph only, never per call
# --------------------------------------------------------------------------


def test_clustering_is_computed_exactly_once_per_build_graph_call_not_recomputed_on_repeated_reads(tmp_path, monkeypatch):
    """§4: clustering must be computed once per reindex (i.e. once per
    `build_graph` call — the only place this project ever invokes it, never
    per MCP tool call) and cached alongside the rest of `LoupeGraph`, not
    recomputed for a subsequent read against an unchanged index. Spied
    directly on the clustering function itself, patched in *before*
    `build_graph` runs so the count is meaningful — the same pattern used
    throughout this project for cache-correctness tests (e.g.
    test_semantic.py's EncodeSpy).
    """
    import loupe_core.graph.builder as builder_module

    _write(tmp_path, "a.py", "def f():\n    return g()\n\n\ndef g():\n    return 1\n")

    call_count = 0
    real_compute_clusters = builder_module.compute_clusters

    def _spy(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return real_compute_clusters(*args, **kwargs)

    monkeypatch.setattr(builder_module, "compute_clusters", _spy)

    graph, _, _ = _build(tmp_path, ["a.py"])
    assert call_count == 1, "build_graph must compute clustering exactly once"

    # Simulate repeated reads against the already-built index — reading
    # `graph.clusters` multiple times, the way multiple MCP tool calls would.
    for _ in range(5):
        _ = graph.clusters.fine
        _ = graph.clusters.coarse

    assert call_count == 1, "reading an already-built LoupeGraph's clusters must never re-invoke compute_clusters"
