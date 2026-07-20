"""E5 — Duplicate code detection (docs/PhaseX/zero-cost-static-analysis-pack.md).

No new computation: symbol embeddings already exist for semantic search
(Phase 2, stored in `sqlite-vec`) — this asks a different question of the
same vector store (all-pairs similarity, not one query against many).
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from loupe_core.graph.builder import EdgeType
from loupe_core.parsing.schema import Symbol
from loupe_core.retrieval.semantic import SemanticIndex

# "A high cosine-similarity threshold" — the spec's own phrase, without a
# number. Checked empirically against the real bge-small-en-v1.5 model
# before trusting a round default: two genuinely near-identical functions
# (copy-pasted, only names changed, identical docstring) scored 0.946, while
# an unrelated function scored 0.44-0.47 against either — a wide, clean gap.
# A commonly-cited round number like 0.95 would sit *above* a real
# near-duplicate pair's actual score, since `embed_text_for_symbol`'s
# "docstring + signature" text still differs by the renamed identifiers in
# the signature even when the docstring is identical. 0.90 sits
# comfortably below the observed near-duplicate score and far above the
# observed unrelated-pair ceiling — documented, revisit-eligible like every
# other tuned constant in this project.
DEFAULT_SIMILARITY_THRESHOLD = 0.90

# Bounded, not a full self-join: duplicates are rare in practice, so a
# handful of each symbol's nearest neighbors is enough to catch every real
# one without an O(n^2) all-pairs comparison over the whole repo.
NEIGHBOR_SCAN_SIZE = 10


@dataclass(frozen=True)
class DuplicateFinding:
    symbol_id_a: str
    symbol_id_b: str
    similarity: float


def _has_direct_call_or_inherit_relationship(graph: nx.DiGraph, a: str, b: str) -> bool:
    """A direct (depth-1) CALLS or INHERITS edge in either direction — "one
    calls the other" (a legitimate wrapper) is exactly the relationship the
    spec's own exclusion criterion names. IMPORTS/TESTS edges don't count:
    the spec says "call/inherit relationship," not "any relationship."
    """
    for u, v in ((a, b), (b, a)):
        if graph.has_edge(u, v) and graph[u][v].get("edge_type") in (EdgeType.CALLS, EdgeType.INHERITS):
            return True
    return False


def find_duplicates(
    semantic_index: SemanticIndex,
    symbols: list[Symbol],
    graph: nx.DiGraph,
    threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
) -> list[DuplicateFinding]:
    """Flags any two *unrelated* symbols — different files, no direct
    call/inherit edge between them per Phase 1's graph — whose embeddings
    exceed `threshold` cosine similarity. Copy-pasted code with only
    variable names changed is the headline case this catches; a real
    wrapper (`def f(x): return g(x)`) is deliberately excluded, since a
    call edge already explains the similarity honestly.
    """
    symbols_by_id = {s.id: s for s in symbols}
    seen_pairs: set[frozenset[str]] = set()
    findings: list[DuplicateFinding] = []

    for symbol in symbols:
        embedding = semantic_index.get_embedding(symbol.id)
        if embedding is None:
            continue

        neighbors = semantic_index.query_by_vector(embedding, top_k=NEIGHBOR_SCAN_SIZE)
        for other_id, similarity in neighbors:
            if other_id == symbol.id or similarity < threshold:
                continue

            pair_key = frozenset({symbol.id, other_id})
            if pair_key in seen_pairs:
                continue

            other = symbols_by_id.get(other_id)
            if other is None:
                continue
            if symbol.file_path == other.file_path:
                continue
            if _has_direct_call_or_inherit_relationship(graph, symbol.id, other_id):
                continue

            seen_pairs.add(pair_key)
            a_id, b_id = sorted((symbol.id, other_id))
            findings.append(DuplicateFinding(symbol_id_a=a_id, symbol_id_b=b_id, similarity=similarity))

    return sorted(findings, key=lambda f: (-f.similarity, f.symbol_id_a, f.symbol_id_b))
