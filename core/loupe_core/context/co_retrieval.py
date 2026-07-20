"""Co-retrieval suggestions: pairwise `get_symbol` co-occurrence mining
(docs/PhaseX/phase-14-adaptive-context-compression.md §3).

A deliberately scoped-down version of full association-rule mining
(market-basket analysis, "customers who bought X also bought Y") — the
useful unit here is pairs, not larger itemsets, so full Apriori-style
frequent-itemset mining is unnecessary complexity.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

# A pair must co-occur at least this many times before ever being surfaced —
# avoids suggestions built on one-off historical coincidence.
MIN_SUPPORT = 5


@dataclass(frozen=True)
class CoRetrievalSuggestion:
    symbol_id: str
    confidence: float
    support: int  # the real co-occurrence count, kept for transparency/debugging


def sessions_from_retrieval_logs(logs_dir: Path) -> list[set[str]]:
    """One set of `get_symbol`-requested symbol_ids per session, read from
    Phase 4's real `RetrievalLog` JSONL files. `query_text` on a `get_symbol`
    entry is already the requested `symbol_id` — `telemetry.py`'s
    `log_tool_call` wrapper extracts it from the `symbol_id` kwarg via the
    same `query_text = kwargs.get("query") or kwargs.get("path_or_glob") or
    kwargs.get("symbol_id")` line every other tool's logging already reuses;
    no new field or extraction logic needed here.
    """
    sessions: list[set[str]] = []
    for log_path in sorted(logs_dir.glob("*.jsonl")):
        symbol_ids: set[str] = set()
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("tool_name") == "get_symbol" and entry.get("query_text"):
                    symbol_ids.add(entry["query_text"])
        if symbol_ids:
            sessions.append(symbol_ids)
    return sessions


def mine_co_retrieval_suggestions(sessions: list[set[str]]) -> dict[str, list[CoRetrievalSuggestion]]:
    """`sessions` is one set of `get_symbol`-requested symbol_ids per session —
    kept separate from log-file parsing (`sessions_from_retrieval_logs`) so
    this pure algorithm is testable without real JSONL files on disk.

    Confidence for `(A, B)`: `count(A and B together) / count(A alone)` — the
    standard market-basket metric. Directional, not symmetric: B's
    suggestion strength when A is requested need not equal A's when B is,
    since `count(A alone)` and `count(B alone)` can differ. A pair below
    `MIN_SUPPORT` co-occurrences is never surfaced in either direction,
    regardless of how high its confidence would otherwise be.
    """
    solo_counts: Counter[str] = Counter()
    pair_counts: Counter[tuple[str, str]] = Counter()

    for session in sessions:
        for symbol_id in session:
            solo_counts[symbol_id] += 1
        for a, b in combinations(sorted(session), 2):
            pair_counts[(a, b)] += 1

    suggestions: dict[str, list[CoRetrievalSuggestion]] = defaultdict(list)
    for (a, b), count in pair_counts.items():
        if count < MIN_SUPPORT:
            continue
        suggestions[a].append(CoRetrievalSuggestion(symbol_id=b, confidence=count / solo_counts[a], support=count))
        suggestions[b].append(CoRetrievalSuggestion(symbol_id=a, confidence=count / solo_counts[b], support=count))

    for symbol_id, suggested in suggestions.items():
        suggested.sort(key=lambda s: (-s.confidence, s.symbol_id))

    return dict(suggestions)
