"""Mine git commit history into (task, ground-truth-symbols) benchmark tasks.

Implements docs/phase-5-evaluation.md §3. No manual labeling: a commit's
subject line is a human-written task description, and the symbols whose
line ranges overlap its diff are the ground truth for what would need to
be retrieved to accomplish it.

One necessary precondition beyond the spec's explicit exclusion list: a
**root commit** (no parent) structurally cannot be diffed against a parent
at all, so it's skipped by the same `len(commit.parents) != 1` check used
for merge commits — not a new exclusion rule being invented, just the
minimum requirement for "diff against the parent" to mean anything.
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import git

from loupe_core.graph.builder import ParsedFile
from loupe_core.parsing.extractor import extract_symbols
from loupe_core.parsing.languages import get_parser
from loupe_core.parsing.schema import Symbol

MAX_FILES_CHANGED = 15
MIN_SUBJECT_LENGTH = 15
STOPLIST_PATTERN = re.compile(r"^(merge|bump|release|chore\(deps\)|wip|typo|fix$)$", re.IGNORECASE)
DEFAULT_MAX_COMMITS = 50

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")


@dataclass
class BenchmarkTask:
    commit_sha: str
    task_description: str
    ground_truth_symbol_ids: list[str]
    ground_truth_files: list[str]
    new_symbols_added: bool
    committed_at: float = 0.0  # unix timestamp; used by Phase 6's temporal_split (§7)


@contextmanager
def _chdir(path: Path):
    old = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def parsed_file_from_git_blob(rel_path: str, source_bytes: bytes) -> ParsedFile:
    """Parse historical blob content into a full ParsedFile, preserving `rel_path` as the
    file_path/id-hash input — so ids stay comparable across separate extractions of the
    same logical path at different points in history (mining here, and the harness's
    ephemeral per-task snapshot later both go through this)."""
    with tempfile.TemporaryDirectory() as scratch:
        scratch_path = Path(scratch)
        target = scratch_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source_bytes)
        with _chdir(scratch_path):
            tree = get_parser("python").parse(source_bytes)
            symbols = extract_symbols(rel_path)
            return ParsedFile(file_path=rel_path, tree=tree, source_bytes=source_bytes, symbols=symbols)


def extract_symbols_from_git_blob(rel_path: str, source_bytes: bytes) -> list[Symbol]:
    """Extract symbols from historical blob content — see `parsed_file_from_git_blob`."""
    return parsed_file_from_git_blob(rel_path, source_bytes).symbols


def _read_blob_at(commit: git.Commit, rel_path: str) -> bytes | None:
    try:
        blob = commit.tree / rel_path
    except KeyError:
        return None
    return blob.data_stream.read()


def _parse_hunk_headers(diff_text: str) -> list[tuple[int, int, int, int]]:
    """(old_start, old_count, new_start, new_count) per `@@ ... @@` header in a unified=0 diff."""
    hunks = []
    for line in diff_text.splitlines():
        m = _HUNK_HEADER.match(line)
        if m:
            old_start = int(m.group(1))
            old_count = int(m.group(2)) if m.group(2) is not None else 1
            new_start = int(m.group(3))
            new_count = int(m.group(4)) if m.group(4) is not None else 1
            hunks.append((old_start, old_count, new_start, new_count))
    return hunks


def _diff_hunks(repo: git.Repo, parent: git.Commit, commit: git.Commit, rel_path: str) -> list[tuple[int, int, int, int]]:
    """(old_start, old_count, new_start, new_count) per hunk, unified=0 for exact ranges."""
    diff_text = repo.git.diff(parent.hexsha, commit.hexsha, "--unified=0", "--", rel_path)
    return _parse_hunk_headers(diff_text)


def diff_hunks_between_contents(old_content: bytes, new_content: bytes) -> list[tuple[int, int, int, int]]:
    """Same (old_start, old_count, new_start, new_count) hunks as `_diff_hunks`, but between
    two arbitrary byte contents rather than two git commits — used by Phase 6's outcome
    backfill to compare a symbol's state when retrieved vs. its state now, which need not
    correspond to two git commits at all (e.g. an uncommitted mid-session edit)."""
    with tempfile.NamedTemporaryFile(suffix=".py") as f_old, tempfile.NamedTemporaryFile(suffix=".py") as f_new:
        f_old.write(old_content)
        f_old.flush()
        f_new.write(new_content)
        f_new.flush()
        result = subprocess.run(
            ["git", "diff", "--no-index", "--unified=0", f_old.name, f_new.name],
            capture_output=True,
            text=True,
        )
        return _parse_hunk_headers(result.stdout)


def hunk_overlaps_symbol(old_start: int, old_count: int, symbol: Symbol) -> bool:
    if old_count > 0:
        hunk_end = old_start + old_count - 1
        return not (hunk_end < symbol.line_start or old_start > symbol.line_end)
    # Pure insertion (nothing removed/modified on the old side): counts as
    # touching a symbol only if the insertion point falls strictly inside its
    # body (extending it) — an insertion between symbols is a new addition,
    # not a change to either neighbor.
    return symbol.line_start <= old_start < symbol.line_end


def _is_whitespace_only_change(repo: git.Repo, commit: git.Commit, parent: git.Commit) -> bool:
    raw_diff = repo.git.diff(parent.hexsha, commit.hexsha)
    if not raw_diff.strip():
        return False
    ignore_ws_diff = repo.git.diff(parent.hexsha, commit.hexsha, "--ignore-all-space")
    return not ignore_ws_diff.strip()


def _qualifies(commit: git.Commit) -> bool:
    if len(commit.parents) != 1:
        return False  # merge commit (2+ parents) or root commit (0 parents)

    subject = commit.message.splitlines()[0].strip()
    if len(subject) < MIN_SUBJECT_LENGTH:
        return False
    if STOPLIST_PATTERN.match(subject):
        return False

    changed_files = list(commit.stats.files.keys())
    if len(changed_files) > MAX_FILES_CHANGED:
        return False

    return True


def _build_task(repo: git.Repo, commit: git.Commit) -> BenchmarkTask:
    parent = commit.parents[0]
    subject = commit.message.splitlines()[0].strip()
    changed_files = [f for f in commit.stats.files if f.endswith(".py")]

    ground_truth_symbol_ids: set[str] = set()
    ground_truth_files: list[str] = []
    new_symbols_added = False

    for rel_path in changed_files:
        ground_truth_files.append(rel_path)

        pre_fix_source = _read_blob_at(parent, rel_path)
        if pre_fix_source is None:
            new_symbols_added = True  # the whole file is new; nothing pre-existing to match
            continue

        pre_fix_symbols = extract_symbols_from_git_blob(rel_path, pre_fix_source)
        hunks = _diff_hunks(repo, parent, commit, rel_path)

        for old_start, old_count, _new_start, _new_count in hunks:
            matched = [s for s in pre_fix_symbols if hunk_overlaps_symbol(old_start, old_count, s)]
            for symbol in matched:
                ground_truth_symbol_ids.add(symbol.id)
            if not matched:
                new_symbols_added = True

    return BenchmarkTask(
        commit_sha=commit.hexsha,
        task_description=subject,
        ground_truth_symbol_ids=sorted(ground_truth_symbol_ids),
        ground_truth_files=ground_truth_files,
        new_symbols_added=new_symbols_added,
        committed_at=float(commit.committed_date),
    )


def temporal_split(tasks: list[BenchmarkTask], cutoff: float) -> tuple[list[BenchmarkTask], list[BenchmarkTask]]:
    """Split into (training, evaluation) by commit date relative to `cutoff` (docs/phase-6-closing-the-loop.md §7).

    Training tasks come strictly before `cutoff`; evaluation tasks come from
    `cutoff` onward. A random split would leak information — patterns from a
    "future" task could influence a model then evaluated on an earlier task
    that helped shape it. This mirrors how the system is actually used
    (trained on past usage, evaluated on new incoming tasks it hasn't seen),
    not just a technical leakage-avoidance rule.
    """
    training = [t for t in tasks if t.committed_at < cutoff]
    evaluation = [t for t in tasks if t.committed_at >= cutoff]
    return training, evaluation


def mine_history(repo_path: str, max_commits: int = DEFAULT_MAX_COMMITS) -> list[BenchmarkTask]:
    """Mine up to `max_commits` qualifying commits from `repo_path`'s history into BenchmarkTasks."""
    repo = git.Repo(repo_path)
    tasks: list[BenchmarkTask] = []

    for commit in repo.iter_commits("HEAD"):
        if len(tasks) >= max_commits:
            break
        if not _qualifies(commit):
            continue
        parent = commit.parents[0]
        if _is_whitespace_only_change(repo, commit, parent):
            continue
        tasks.append(_build_task(repo, commit))

    return tasks
