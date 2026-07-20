"""Code churn: a recency-decayed measure of how recently and frequently each
symbol has actually been touched in real git history
(docs/PhaseX/phase-14-adaptive-context-compression.md §2).

Centrality (Phase 1) captures *structural* importance — how connected
something is. Churn captures *temporal* relevance — what's actually being
worked on lately, a named, established software-engineering metric (Nagappan
and Ball's defect-prediction work, later popularized as "hotspot analysis").

Reuses Phase 5's exact diff-hunk-to-symbol overlap detection
(`eval/mine_history.py`'s `diff_hunks_between_commits`/`hunk_overlaps_symbol`)
— not reimplemented.
"""

from __future__ import annotations

import time

import git

from loupe_core.eval.mine_history import diff_hunks_between_commits, hunk_overlaps_symbol
from loupe_core.parsing.schema import Symbol

CHURN_WINDOW_DAYS = 90
SECONDS_PER_DAY = 86400

# ~30-day half-life (0.97^30 ≈ 0.40, 0.97^23 ≈ 0.5) — a deliberately
# different constant from the governor's per-turn 0.85 (eviction.py's
# DEFAULT_DECAY_FACTOR): git commit cadence and conversation turns are
# entirely different timescales and shouldn't share a tuning constant just
# because both happen to be "decay."
CHURN_DECAY_FACTOR = 0.97


def compute_churn_scores(repo: git.Repo, symbols: list[Symbol], now: float | None = None) -> dict[str, float]:
    """Every contributing commit within `CHURN_WINDOW_DAYS` of `now` (default:
    real current time) adds `CHURN_DECAY_FACTOR ** days_since_commit` to a
    touched symbol's score — a symbol edited five times last week scores far
    higher than one edited once, three months ago, even at identical
    structural centrality. Every symbol gets an entry, `0.0` if untouched
    within the window (never absent — callers shouldn't have to guess whether
    a missing key means "zero churn" or "not computed").
    """
    if now is None:
        now = time.time()
    cutoff = now - CHURN_WINDOW_DAYS * SECONDS_PER_DAY

    symbols_by_file: dict[str, list[Symbol]] = {}
    for s in symbols:
        symbols_by_file.setdefault(s.file_path, []).append(s)

    scores: dict[str, float] = {s.id: 0.0 for s in symbols}

    for commit in repo.iter_commits():
        commit_time = commit.committed_date
        if commit_time < cutoff:
            break  # commits walk newest-first — everything earlier is outside the window too
        if not commit.parents:
            continue  # root commit: no parent to diff against

        parent = commit.parents[0]
        days_since = max(0.0, (now - commit_time) / SECONDS_PER_DAY)
        weight = CHURN_DECAY_FACTOR**days_since

        for rel_path in commit.stats.files:
            file_symbols = symbols_by_file.get(rel_path)
            if not file_symbols:
                continue
            hunks = diff_hunks_between_commits(repo, parent, commit, rel_path)
            for old_start, old_count, _new_start, _new_count in hunks:
                for symbol in file_symbols:
                    if hunk_overlaps_symbol(old_start, old_count, symbol):
                        scores[symbol.id] += weight

    return scores
