"""Rule-based query-intent router (docs/phase-6-closing-the-loop.md §5), extended
with a nearest-centroid semantic fallback (docs/PhaseX/loupe-retrieval-upgrades.md
§5) for paraphrases the regex rules can't see (e.g. "things stopped working after
the last deploy" — no "error"/"crash"/"broken", but obviously debug-intent).

Layered, not replaced: the cheap regex fast path below runs first and wins
immediately on any match, keeping the common, unambiguous case cheap (no
embedding call). Only a query the regex path can't confidently classify
(falls through to "general") pays for the semantic fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Literal

import networkx as nx

from loupe_core.retrieval.mmr import cosine_similarity

Intent = Literal["debug", "feature", "refactor", "general"]

# Checked in this order — first match wins; "general" is the explicit default.
_DEBUG_KEYWORDS = ["error", "exception", "traceback", "crash", "fails", "broken", "bug"]
_FEATURE_KEYWORDS = ["add", "implement", "create", "new", "support for"]
_REFACTOR_KEYWORDS = ["refactor", "rename", "clean up", "restructure", "simplify"]

# A bare identifier-shaped token: snake_case, a dotted/qualified reference
# (Class.method), or a bare filename — the common shapes a debugging query
# anchors to a known error site with.
_SYMBOL_REFERENCE_PATTERN = re.compile(
    r"\b([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)+|[a-z]+_[a-z_]+|[a-zA-Z][a-zA-Z0-9_]*\.py)\b"
)


def classify_intent(query: str) -> Intent:
    """Keyword classification into one of 4 categories (§5)."""
    lowered = query.lower()
    if any(kw in lowered for kw in _DEBUG_KEYWORDS):
        return "debug"
    if any(kw in lowered for kw in _FEATURE_KEYWORDS):
        return "feature"
    if any(kw in lowered for kw in _REFACTOR_KEYWORDS):
        return "refactor"
    return "general"


def detect_symbol_reference(query: str) -> str | None:
    """A bare symbol/file-shaped token in `query`, or None (§5's debug candidate-seeding trigger)."""
    match = _SYMBOL_REFERENCE_PATTERN.search(query)
    return match.group(1) if match else None


def seed_debug_candidates(
    query: str,
    resolve_reference: Callable[[str], str | None],
    graph: nx.DiGraph,
    depth: int = 1,
) -> set[str]:
    """Debug-intent candidate seeding (§5): expand_dependencies of a referenced symbol.

    `resolve_reference` maps a raw text reference (e.g. "utils.py" or
    "validate_email") to a symbol_id, or None if it can't be resolved — kept
    as an injected callback so this module doesn't need a hard dependency on
    any particular symbol-lookup structure. Returns an empty set (never
    raises) whenever there's nothing to seed — no intent match, no reference
    found, or a reference that doesn't resolve to anything real (§8: "does
    not crash attempting to find a reference that isn't there").
    """
    if classify_intent(query) != "debug":
        return set()

    reference = detect_symbol_reference(query)
    if reference is None:
        return set()

    symbol_id = resolve_reference(reference)
    if symbol_id is None or symbol_id not in graph:
        return set()

    from loupe_core.graph.traversal import expand_dependencies

    return expand_dependencies(graph, symbol_id, depth=depth, direction="outgoing")


# --------------------------------------------------------------------------
# Phase 9 §5 — nearest-centroid semantic fallback
# --------------------------------------------------------------------------

# The retrieval-upgrades spec's own literal suggestion ("threshold (0.3), a
# starting constant") — checked empirically against the real bge-small-en-v1.5
# model before trusting it, the same discipline applied to personalized
# PageRank's max_iter/tol in graph/centrality.py. It doesn't work: bge-small's
# embedding space is anisotropic enough that cosine similarity against any of
# these 4 centroids sits in a 0.4-0.7+ baseline for essentially any input,
# including total gibberish ("asdkjaslkdj random gibberish text zzz" scored
# 0.72 against "debug") — at 0.3, the "general" fallback would never fire for
# any query. Genuine paraphrase-quality matches (real held-out examples of
# each intent, not the centroid's own training queries) scored 0.74-0.81 with
# a real model, distinctly above noise-level scores, but noise itself doesn't
# have a single clean floor — some gibberish/generic queries scored as high as
# 0.72, overlapping the low end of genuine matches. No single global threshold
# perfectly separates the two given this model's behavior; 0.65 is chosen to
# accept every genuine paraphrase measured while rejecting most low-signal
# input, documented as an empirically-necessary correction with a known
# residual false-positive rate on adversarial/generic phrasing, not a
# guaranteed-perfect classifier — revisit with real usage data, same as every
# other tuned constant in this project (RRF's k=60, MMR's lambda=0.7).
NEAREST_CENTROID_THRESHOLD = 0.65

# 5-10 representative example queries per category (§5) — the centroid for
# each intent is these examples' mean embedding. Deliberately hand-written to
# read like real, varied task descriptions rather than keyword-stuffed
# near-duplicates of _DEBUG_KEYWORDS etc. above, since the whole point of this
# fallback is covering phrasing the regex rules don't recognize.
_INTENT_EXAMPLE_QUERIES: dict[Intent, list[str]] = {
    "debug": [
        "why is this throwing an exception",
        "things stopped working after the last deploy",
        "getting a stack trace when I call this endpoint",
        "this used to work but now it's broken",
        "investigate why the request keeps failing",
        "something crashed in production overnight",
        "users are reporting unexpected behavior here",
    ],
    "feature": [
        "add support for exporting to CSV",
        "implement a new endpoint for user profiles",
        "build a way to filter results by date",
        "create a background job for sending emails",
        "we need to support multiple currencies",
        "wire up a new integration with the billing provider",
    ],
    "refactor": [
        "clean up this messy function",
        "simplify the branching logic here",
        "restructure this module into smaller pieces",
        "extract this duplicated logic into a shared helper",
        "reduce the complexity of this class",
        "make this code easier to follow",
    ],
    "general": [
        "how does this part of the system work",
        "explain what this function does",
        "walk me through the data flow here",
        "what are the responsibilities of this class",
        "summarize this module for me",
        "where is this value used elsewhere in the codebase",
    ],
}


@dataclass
class IntentCentroids:
    """One mean-embedding vector per `Intent` category, computed once and cached
    (§5) — only invalidated by editing `_INTENT_EXAMPLE_QUERIES` itself, a rare,
    deliberate config change, not something tied to a repo's index state.
    """

    vectors: dict[Intent, list[float]] = field(default_factory=dict)


_cached_centroids: IntentCentroids | None = None


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    dim = len(vectors[0])
    return [sum(v[i] for v in vectors) / len(vectors) for i in range(dim)]


def compute_intent_centroids(model: object) -> IntentCentroids:
    """Embeds every example query and averages per category. `model` is injectable
    (a real `SentenceTransformer` in production, a spy/stand-in in tests) — same
    discipline as `retrieval/semantic.py`'s `SemanticIndex(model=...)`.
    """
    vectors: dict[Intent, list[float]] = {}
    for intent, examples in _INTENT_EXAMPLE_QUERIES.items():
        embeddings = model.encode(examples, normalize_embeddings=True)
        vectors[intent] = _mean_vector([list(e) for e in embeddings])
    return IntentCentroids(vectors=vectors)


def get_cached_intent_centroids(model: object) -> IntentCentroids:
    """Process-wide cache — computed once per process, not once per query."""
    global _cached_centroids
    if _cached_centroids is None:
        _cached_centroids = compute_intent_centroids(model)
    return _cached_centroids


def classify_intent_semantic(
    query: str,
    model: object | None = None,
    centroids: IntentCentroids | None = None,
) -> Intent:
    """Layered classification (§5): `classify_intent`'s regex fast path runs first
    and wins immediately on any non-"general" match — no embedding call needed for
    the easy, unambiguous cases. Only when the regex path falls through to
    "general" does this embed the query and compare it against the 4 precomputed
    intent centroids, classifying as the closest one if its cosine similarity
    clears `NEAREST_CENTROID_THRESHOLD`, else staying "general" — the identical
    fallback behavior Phase 6 already has, just reached via a different path.

    A query that *would* match the regex fast path is classified identically
    whether forced through this function or called via `classify_intent`
    directly — the fast-path result returns immediately, before any embedding
    or centroid comparison happens (§5's own consistency acceptance criterion).
    """
    fast_path_result = classify_intent(query)
    if fast_path_result != "general":
        return fast_path_result

    from loupe_core.retrieval.semantic import get_default_model

    embed_model = model if model is not None else get_default_model()
    centroid_set = centroids if centroids is not None else get_cached_intent_centroids(embed_model)
    query_embedding = list(embed_model.encode([query], normalize_embeddings=True)[0])

    best_intent: Intent = "general"
    best_similarity = float("-inf")
    for intent, centroid in centroid_set.vectors.items():
        similarity = cosine_similarity(query_embedding, centroid)
        if similarity > best_similarity:
            best_intent, best_similarity = intent, similarity

    return best_intent if best_similarity >= NEAREST_CENTROID_THRESHOLD else "general"


# --------------------------------------------------------------------------
# Phase 14 §1 — starting zoom-level selection
# --------------------------------------------------------------------------

StartingLevel = Literal["L0_L1", "L3_L4"]


def select_starting_level(query: str) -> StartingLevel:
    """The router's job grows from "pick a retrieval strategy" to "pick a
    retrieval strategy *and* a starting zoom level" (§1). A `debug`/`feature`/
    `refactor`-intent query, or *any* query naming a specific symbol
    (`detect_symbol_reference`, already used for §5's debug candidate
    seeding), starts at L3/L4 — today's existing `list_symbols`/
    `search_symbols`/`get_symbol` default, unchanged. Only a `general`-intent
    query with no detected symbol reference at all — the "how does auth work
    across this repo" shape — starts at the L0/L1 `architecture://overview`
    resource instead.

    Reuses `classify_intent`'s fast path only, not the semantic fallback —
    picking a *starting* zoom level is a coarse, cheap decision by design; a
    wrong starting point costs one extra round-trip at worst; it isn't worth
    an embedding call the way genuine debug-candidate seeding is.
    """
    if classify_intent(query) != "general":
        return "L3_L4"
    if detect_symbol_reference(query) is not None:
        return "L3_L4"
    return "L0_L1"
