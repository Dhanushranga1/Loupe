"""Scope-aware retrieval: explicit path-based hard filtering and Louvain +
personalized-PageRank-based soft boundary biasing
(docs/PhaseX/scope-aware-retrieval.md).

Two scope *sources* (§1): an explicit file-path prefix, or auto-detection
from a starting symbol via Phase 10.5's fine-resolution Louvain clusters.
Two scope *modes* (§3), independent of source: `hard` (a genuine
set-intersection filter, applied to candidate generation) and `soft`
(personalized-PageRank restart-mass biasing, applied to the centrality
ranking term only — nothing outside scope is ever made invisible).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from loupe_core.graph.builder import LoupeGraph
from loupe_core.graph.centrality import compute_personalized_pagerank
from loupe_core.parsing.schema import Symbol

ScopeMode = Literal["hard", "soft"]
DEFAULT_SCOPE_MODE: ScopeMode = "soft"

# §2: "a reasonable starting split: roughly 90% of restart probability mass
# inside scope, 10% distributed across the rest" — used as specified; no
# empirical reason found yet to deviate the way some of Phase 9's literal
# spec constants needed to (see graph/centrality.py's PERSONALIZED_TOL and
# retrieval/router.py's NEAREST_CENTROID_THRESHOLD for those corrections).
DEFAULT_IN_SCOPE_MASS = 0.9


@dataclass(frozen=True)
class Scope:
    """Exactly one of `path_prefix`/`seed_symbol_id` is expected to be set —
    the two scope-*source* options from §1. `mode` controls hard-filter vs.
    soft-bias regardless of which source produced the membership set.
    """

    path_prefix: str | None = None
    seed_symbol_id: str | None = None
    mode: ScopeMode = DEFAULT_SCOPE_MODE


def resolve_scope_membership(scope: Scope, graph: LoupeGraph, symbols_by_id: dict[str, Symbol]) -> set[str]:
    """The set of symbol_ids inside `scope`'s boundary.

    Explicit path prefix: every symbol whose `file_path` starts with it —
    the deterministic option, right when isolation needs to be a genuine
    guarantee (§3).

    Auto-detected (§1): the Louvain *fine*-resolution cluster containing
    `seed_symbol_id` — fine, not coarse, since this needs "a tight, precise
    boundary around one starting symbol," exactly Phase 10.5's own stated
    reason for computing two resolution levels. A seed not found in any
    cluster (e.g. an isolated node with no edges) falls back to a trivial
    one-symbol scope rather than raising — the caller asked to scope around
    a real symbol, and a singleton scope is a degenerate-but-valid answer to
    that, not an error.
    """
    if scope.path_prefix is not None:
        return {sid for sid, symbol in symbols_by_id.items() if symbol.file_path.startswith(scope.path_prefix)}

    if scope.seed_symbol_id is not None:
        for cluster in graph.clusters.fine:
            if scope.seed_symbol_id in cluster:
                return set(cluster)
        return {scope.seed_symbol_id} if scope.seed_symbol_id in symbols_by_id else set()

    return set()


def apply_hard_scope(candidate_ids: set[str], membership: set[str]) -> set[str]:
    """§1's explicit hard filter — a genuine set intersection. Callers apply
    this to candidate generation itself (before Phase 2's ranking pipeline
    runs), not as a post-hoc filter on already-ranked results — see
    `mcp_tools.py`'s scope wiring.
    """
    return candidate_ids & membership


def scoped_personalized_pagerank(
    graph: LoupeGraph,
    membership: set[str],
    candidate_ids: set[str],
    in_scope_mass: float = DEFAULT_IN_SCOPE_MASS,
) -> dict[str, float]:
    """§2's soft-boundary retrieval: personalized PageRank seeded from
    `membership` (the scope), scored for `candidate_ids` (the query's own RRF
    candidates) — the exact same algorithm Phase 9 built for query-aware
    centrality, reused with a different restart distribution and the
    `in_scope_mass` split this use case needs and Phase 9's own didn't.
    """
    return compute_personalized_pagerank(
        graph.graph, membership, graph.pagerank_scores, in_scope_mass=in_scope_mass, score_ids=candidate_ids
    )
