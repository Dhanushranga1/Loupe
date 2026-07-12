"""Rule-based query-intent router (docs/phase-6-closing-the-loop.md §5).

Deliberately rule-based, not ML — building a labeled intent-classification
dataset would be disproportionate effort for four categories with reasonably
distinguishable surface patterns (the same reasoning as choosing logistic
regression over a heavier model for the ranker).
"""

from __future__ import annotations

import re
from typing import Callable, Literal

import networkx as nx

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
