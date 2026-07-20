"""Tests for graph/centrality.py's compute_personalized_pagerank
(docs/PhaseX/loupe-retrieval-upgrades.md §2 — Personalized PageRank)."""

import os
from pathlib import Path

import networkx as nx
import pytest

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.graph.centrality import compute_personalized_pagerank


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def two_cluster_repo(tmp_path):
    """Cluster A: a hub with 3 callers. Cluster B: a hub with 8 callers (so cluster B's
    hub has the higher *static* PageRank) — connected only by one weak cross-cluster
    call, so a query seeded inside cluster A shouldn't drag much probability mass to B."""
    _write(
        tmp_path,
        "cluster_a.py",
        "def hub_a():\n    return 1\n\n\n"
        + "\n\n".join(f"def a_caller_{i}():\n    return hub_a()" for i in range(1, 4)),
    )
    _write(
        tmp_path,
        "cluster_b.py",
        "def hub_b():\n    return 2\n\n\n"
        + "\n\n".join(f"def b_caller_{i}():\n    return hub_b()" for i in range(1, 9)),
    )
    _write(
        tmp_path,
        "bridge.py",
        "from cluster_a import a_caller_1\nfrom cluster_b import b_caller_1\n\n\n"
        "def weak_bridge():\n    a_caller_1()\n    b_caller_1()\n",
    )

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        files = ["cluster_a.py", "cluster_b.py", "bridge.py"]
        parsed = [parse_file(f) for f in files]
        g = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        yield g, symbols_by_id
    finally:
        os.chdir(old_cwd)


def _id_by_name(symbols_by_id, qualified_name: str) -> str:
    return next(s.id for s in symbols_by_id.values() if s.qualified_name == qualified_name)


def test_static_pagerank_favors_the_more_connected_cluster(two_cluster_repo):
    """Sanity check on the fixture's own construction, not the function under test:
    cluster B's hub (8 callers) must have higher static PageRank than cluster A's (3)."""
    g, symbols_by_id = two_cluster_repo
    hub_a = _id_by_name(symbols_by_id, "hub_a")
    hub_b = _id_by_name(symbols_by_id, "hub_b")
    assert g.pagerank_scores[hub_b] > g.pagerank_scores[hub_a]


def test_personalized_pagerank_seeded_in_cluster_a_favors_hub_a_over_hub_b(two_cluster_repo):
    """The concrete, checkable proof personalization is actually changing behavior:
    seeding from inside cluster A must flip the ranking static PageRank gives,
    scoring cluster A's local hub above cluster B's globally-more-connected one."""
    g, symbols_by_id = two_cluster_repo
    hub_a = _id_by_name(symbols_by_id, "hub_a")
    hub_b = _id_by_name(symbols_by_id, "hub_b")
    seed = _id_by_name(symbols_by_id, "a_caller_2")

    result = compute_personalized_pagerank(g.graph, {seed, hub_a, hub_b}, g.pagerank_scores, depth=3)

    assert result[hub_a] > result[hub_b]


def test_candidate_not_present_in_the_graph_falls_back_to_its_static_score_not_zero(two_cluster_repo):
    g, symbols_by_id = two_cluster_repo
    seed = _id_by_name(symbols_by_id, "a_caller_2")
    phantom_id = "not-a-real-graph-node"
    static_scores = dict(g.pagerank_scores)
    static_scores[phantom_id] = 0.0777

    result = compute_personalized_pagerank(g.graph, {seed, phantom_id}, static_scores, depth=3)

    assert result[phantom_id] == 0.0777


def test_empty_seed_set_returns_empty_result():
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    assert compute_personalized_pagerank(graph, set(), {}) == {}


def test_no_seeds_present_in_graph_falls_back_to_static_for_all():
    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    static_scores = {"phantom-1": 0.1, "phantom-2": 0.2}

    result = compute_personalized_pagerank(graph, {"phantom-1", "phantom-2"}, static_scores)

    assert result == static_scores


def test_non_convergence_falls_back_to_static_scores_for_every_seed(monkeypatch):
    """The safety net for the rare case even the loosened tolerance/iteration budget
    doesn't converge on some unusual subgraph topology — verified by forcing the
    real failure mode (PowerIterationFailedConvergence), not just trusting it can't happen."""
    import loupe_core.graph.centrality as centrality_module

    def _always_fails(*args, **kwargs):
        raise nx.PowerIterationFailedConvergence(20)

    monkeypatch.setattr(centrality_module.nx, "pagerank", _always_fails)

    graph = nx.DiGraph()
    graph.add_edge("a", "b")
    static_scores = {"a": 0.4, "b": 0.6}

    result = compute_personalized_pagerank(graph, {"a", "b"}, static_scores)

    assert result == static_scores
