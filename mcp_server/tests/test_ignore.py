"""Tests for app/ignore.py — the shared exclude-path logic used by both the initial
full index (bootstrap.py) and the incremental file watcher (indexer_worker.py)."""

from loupe_mcp_server.ignore import is_path_ignored, load_loupeignore_patterns


def test_default_names_excluded_at_any_depth_with_no_config():
    assert is_path_ignored(".venv/lib/foo.py", [])
    assert is_path_ignored("backend/.venv/lib/foo.py", [])
    assert is_path_ignored("frontend/node_modules/pkg/index.py", [])
    assert not is_path_ignored("app/main.py", [])


def test_a_directory_name_that_only_resembles_a_default_is_not_excluded_without_a_pattern():
    """The real bug this module exists to fix: 'backend/.venv-py314-backup/foo.py' is not
    literally '.venv' as a path segment, so it must NOT be silently excluded by the
    built-in default list alone — only exact names are free; anything else needs a
    pattern, same as any other real exclude mechanism."""
    assert not is_path_ignored("backend/.venv-py314-backup/lib/foo.py", [])


def test_a_wildcard_pattern_catches_the_lookalike_directory_at_any_depth():
    assert is_path_ignored("backend/.venv-py314-backup/lib/foo.py", [".venv*"])
    assert is_path_ignored(".venv-py314-backup/lib/foo.py", [".venv*"])


def test_an_exact_extra_pattern_also_works():
    assert is_path_ignored("backend/.venv-py314-backup/lib/foo.py", [".venv-py314-backup"])


def test_trailing_slash_on_a_pattern_is_tolerated():
    assert is_path_ignored("backend/.venv-py314-backup/lib/foo.py", [".venv-py314-backup/"])


def test_unrelated_pattern_does_not_over_match():
    assert not is_path_ignored("app/main.py", [".venv*", "*-backup"])


def test_load_loupeignore_patterns_reads_real_lines_skips_comments_and_blanks(tmp_path):
    (tmp_path / ".loupeignore").write_text("# comment\n\n.venv-py314-backup/\nbuild_artifacts/\n")

    patterns = load_loupeignore_patterns(tmp_path)

    assert patterns == [".venv-py314-backup/", "build_artifacts/"]


def test_load_loupeignore_patterns_on_missing_file_returns_empty(tmp_path):
    assert load_loupeignore_patterns(tmp_path) == []
