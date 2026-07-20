"""Tests for retrieval/churn.py (docs/PhaseX/phase-14-adaptive-context-compression.md §2).

Uses a real git repo with precisely-dated commits (GitPython's `commit_date`/
`author_date` parameters), not the real system clock — churn is fundamentally
about *when* commits happened relative to `now`, so tests need exact control
over both.
"""

import datetime

import git
import pytest

from loupe_core.graph.builder import parse_file
from loupe_core.retrieval.churn import CHURN_DECAY_FACTOR, CHURN_WINDOW_DAYS, compute_churn_scores

REFERENCE_NOW = datetime.datetime(2026, 1, 1, tzinfo=datetime.timezone.utc)


def _commit(repo: git.Repo, message: str, days_ago: int) -> git.Commit:
    date = REFERENCE_NOW - datetime.timedelta(days=days_ago)
    return repo.index.commit(message, author_date=date, commit_date=date)


@pytest.fixture
def churn_repo(tmp_path):
    repo = git.Repo.init(tmp_path)
    with repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@example.com")

    code_path = tmp_path / "code.py"
    code_path.write_text(
        "def hot_function():\n    return 1\n\n\ndef cold_function():\n    return 2\n"
    )
    repo.index.add(["code.py"])
    _commit(repo, "initial commit", days_ago=200)  # both symbols created, well outside the window

    # hot_function is edited repeatedly, recently; cold_function is never touched again.
    for i, days_ago in enumerate([45, 20, 5]):
        code_path.write_text(
            f"def hot_function():\n    return {i + 10}\n\n\ndef cold_function():\n    return 2\n"
        )
        repo.index.add(["code.py"])
        _commit(repo, f"tweak hot_function #{i}", days_ago=days_ago)

    import os

    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        parsed = parse_file("code.py")
    finally:
        os.chdir(old_cwd)

    return repo, parsed.symbols


def _by_name(symbols, name: str):
    return next(s for s in symbols if s.qualified_name == name)


def test_recently_and_frequently_modified_symbol_scores_higher_than_untouched_one(churn_repo):
    """§2's own acceptance criterion: a deliberately recently-and-frequently
    modified symbol versus an old, untouched-in-months symbol of similar
    structural centrality (both are trivial top-level functions here) —
    churn must rank the active one higher."""
    repo, symbols = churn_repo
    hot = _by_name(symbols, "hot_function")
    cold = _by_name(symbols, "cold_function")

    scores = compute_churn_scores(repo, symbols, now=REFERENCE_NOW.timestamp())

    assert scores[hot.id] > scores[cold.id]
    assert scores[cold.id] == 0.0, "cold_function's only touch (200 days ago) is well outside the 90-day window"


def test_every_symbol_gets_a_score_never_a_missing_key(churn_repo):
    repo, symbols = churn_repo
    scores = compute_churn_scores(repo, symbols, now=REFERENCE_NOW.timestamp())
    assert set(scores) == {s.id for s in symbols}


def test_commits_outside_the_window_contribute_nothing(churn_repo):
    repo, symbols = churn_repo
    hot = _by_name(symbols, "hot_function")

    # "now" set so far in the future that even the most recent real commit
    # (5 days before REFERENCE_NOW) falls outside a 90-day window from here.
    far_future = REFERENCE_NOW + datetime.timedelta(days=200)
    scores = compute_churn_scores(repo, symbols, now=far_future.timestamp())

    assert scores[hot.id] == 0.0


def test_decay_factor_gives_roughly_a_30_day_half_life():
    """Sanity check on the documented constant, not the function under test —
    catches a wrong constant before it silently produces wrong rankings."""
    assert CHURN_DECAY_FACTOR**23 == pytest.approx(0.5, abs=0.02)


def test_window_is_90_days_as_documented():
    assert CHURN_WINDOW_DAYS == 90
