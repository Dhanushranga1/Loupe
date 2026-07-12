"""Tests for eval/mine_history.py (docs/phase-5-evaluation.md §8 — Mining).

Uses the programmatically-built fixture git repo (§7) — a real repo with a
deliberate mix of good and bad commits, not another set of plain files.
"""

import sys
from pathlib import Path

import git
import pytest

sys.path.insert(0, str(Path(__file__).parent / "fixtures" / "phase5"))
from build_fixture_repo import build_fixture_repo  # noqa: E402

from loupe_core.eval.mine_history import extract_symbols_from_git_blob, mine_history


@pytest.fixture(scope="module")
def fixture_repo(tmp_path_factory):
    repo_root = tmp_path_factory.mktemp("phase5_repo")
    shas = build_fixture_repo(repo_root)
    return repo_root, shas


@pytest.fixture(scope="module")
def mined_tasks(fixture_repo):
    repo_root, _ = fixture_repo
    return mine_history(str(repo_root), max_commits=50)


@pytest.fixture(scope="module")
def tasks_by_sha(mined_tasks):
    return {t.commit_sha: t for t in mined_tasks}


def test_root_commit_is_excluded(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    assert shas["root"] not in tasks_by_sha


def test_merge_commit_is_excluded(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    assert shas["bad_merge"] not in tasks_by_sha


def test_short_subject_commit_is_excluded(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    assert shas["bad_short_subject"] not in tasks_by_sha


def test_stoplist_subject_commit_is_excluded(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    assert shas["bad_stoplist"] not in tasks_by_sha


def test_whitespace_only_commit_is_excluded(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    assert shas["bad_whitespace_only"] not in tasks_by_sha


def test_too_many_files_commit_is_excluded(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    assert shas["bad_too_many_files"] not in tasks_by_sha


def test_good_commits_are_all_included(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    for label in ["good_thousands_separator", "good_email_length_check", "good_new_function", "good_log_email"]:
        assert shas[label] in tasks_by_sha, f"{label} should have qualified for mining"


def test_task_description_is_commit_subject_line_only(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    task = tasks_by_sha[shas["good_thousands_separator"]]
    assert task.task_description == "Add thousands separator to currency formatting"


def test_hand_verified_good_commit_ground_truth_symbol_ids(fixture_repo, tasks_by_sha):
    """format_currency's body is the only thing changed — ground truth must be exactly its id."""
    repo_root, shas = fixture_repo
    task = tasks_by_sha[shas["good_thousands_separator"]]

    repo = git.Repo(repo_root)
    parent = repo.commit(shas["good_thousands_separator"]).parents[0]
    pre_fix_source = (parent.tree / "utils.py").data_stream.read()
    pre_fix_symbols = extract_symbols_from_git_blob("utils.py", pre_fix_source)
    expected_id = next(s.id for s in pre_fix_symbols if s.qualified_name == "format_currency")

    assert task.ground_truth_symbol_ids == [expected_id]
    assert task.ground_truth_files == ["utils.py"]
    assert task.new_symbols_added is False


def test_nested_class_and_method_both_overlap_a_method_body_change(fixture_repo, tasks_by_sha):
    """A one-line change inside create_order legitimately overlaps both the
    method's own range and its enclosing class's range — not a bug, the
    literal 'any symbol whose range overlaps' rule naturally includes both."""
    repo_root, shas = fixture_repo
    task = tasks_by_sha[shas["good_log_email"]]

    repo = git.Repo(repo_root)
    parent = repo.commit(shas["good_log_email"]).parents[0]
    pre_fix_source = (parent.tree / "services.py").data_stream.read()
    pre_fix_symbols = extract_symbols_from_git_blob("services.py", pre_fix_source)

    class_id = next(s.id for s in pre_fix_symbols if s.qualified_name == "OrderService")
    method_id = next(s.id for s in pre_fix_symbols if s.qualified_name == "OrderService.create_order")
    log_method_id = next(s.id for s in pre_fix_symbols if s.qualified_name == "OrderService.log")

    assert set(task.ground_truth_symbol_ids) == {class_id, method_id}
    assert log_method_id not in task.ground_truth_symbol_ids


def test_new_function_commit_sets_flag_and_does_not_crash(fixture_repo, tasks_by_sha):
    _, shas = fixture_repo
    task = tasks_by_sha[shas["good_new_function"]]
    assert task.new_symbols_added is True
    assert task.ground_truth_files == ["utils.py"]
    # the new function itself has no pre-fix version, so it must not appear
    # in ground_truth_symbol_ids — but format_currency/validate_email weren't
    # touched by this commit either, so ground truth is correctly empty here.
    assert task.ground_truth_symbol_ids == []


def test_mine_history_respects_max_commits(fixture_repo):
    repo_root, _ = fixture_repo
    tasks = mine_history(str(repo_root), max_commits=2)
    assert len(tasks) == 2
