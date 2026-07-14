"""Tests for conventions/mining.py (docs/loupe-extensions.md E4 — Auto-Derived Conventions)."""

import os

import pytest

from loupe_core.conventions.mining import mine_conventions, mine_docstrings, mine_error_handling, mine_imports
from loupe_core.graph.builder import parse_file

FIXTURES_DIR = "tests/fixtures/e4"


@pytest.fixture
def handlers(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures" / "e4"
    shutil.copy(fixtures / "handlers.py", tmp_path / "handlers.py")
    monkeypatch.chdir(tmp_path)
    return parse_file("handlers.py")


@pytest.fixture
def imports_mixed(tmp_path, monkeypatch):
    import shutil
    from pathlib import Path

    fixtures = Path(__file__).parent / "fixtures" / "e4"
    shutil.copy(fixtures / "imports_mixed.py", tmp_path / "imports_mixed.py")
    monkeypatch.chdir(tmp_path)
    return parse_file("imports_mixed.py")


def test_majority_error_handling_pattern_and_exactly_the_outlier_flagged(handlers):
    """4 functions share one logger call pattern, 1 deliberately differs — the exact
    acceptance-criteria shape from loupe-extensions.md's E4 section."""
    report = mine_error_handling([handlers])

    assert report.majority_pattern == "except ValueError: logging.error"
    assert report.violation_count == 1

    name_by_id = {s.id: s.qualified_name for s in handlers.symbols}
    violating_names = {name_by_id[sid] for sid in report.violating_symbol_ids}
    assert violating_names == {"process_outlier"}


def test_functions_without_a_try_except_are_not_counted_either_way(handlers, tmp_path):
    (tmp_path / "handlers.py").write_text(
        (tmp_path / "handlers.py").read_text() + "\n\ndef no_error_handling_here():\n    \"\"\"Just a plain function.\"\"\"\n    return 1\n"
    )
    parsed = parse_file("handlers.py")
    report = mine_error_handling([parsed])

    name_by_id = {s.id: s.qualified_name for s in parsed.symbols}
    violating_names = {name_by_id[sid] for sid in report.violating_symbol_ids}
    assert "no_error_handling_here" not in violating_names


def test_no_try_except_anywhere_returns_no_majority_pattern():
    report = mine_error_handling([])
    assert report.majority_pattern is None
    assert report.violation_count == 0


def test_consistent_google_style_docstrings_correctly_classified(handlers):
    """The second acceptance-criteria case: a fixture with consistent Google-style
    docstrings must be classified as such, not just 'has a docstring'."""
    report = mine_docstrings([handlers])

    assert report.dominant_style == "google"
    assert report.coverage_pct == 100.0


def test_numpy_style_docstring_detected_via_parameters_and_dashes_line():
    from loupe_core.conventions.mining import _docstring_style

    numpy_doc = "Summary line.\n\nParameters\n----------\nx : int\n    A value.\n"
    assert _docstring_style(numpy_doc) == "numpy"


def test_docstring_with_neither_marker_is_plain():
    from loupe_core.conventions.mining import _docstring_style

    assert _docstring_style("Just a plain one-line docstring.") == "plain"


def test_private_symbols_excluded_from_docstring_coverage(handlers, tmp_path):
    (tmp_path / "handlers.py").write_text(
        (tmp_path / "handlers.py").read_text() + "\n\ndef _private_helper():\n    return 1\n"
    )
    parsed = parse_file("handlers.py")
    report_with_private = mine_docstrings([parsed])
    report_without = mine_docstrings([handlers])

    # adding an undocumented *private* function must not drag coverage down
    assert report_with_private.coverage_pct == report_without.coverage_pct == 100.0


def test_relative_imports_correctly_counted_and_dominant(imports_mixed):
    report = mine_imports([imports_mixed])

    assert report.relative_count == 3
    assert report.absolute_count == 0
    assert report.dominant_style == "relative"


def test_absolute_imports_correctly_counted_and_dominant(handlers):
    report = mine_imports([handlers])

    assert report.absolute_count == 1
    assert report.relative_count == 0
    assert report.dominant_style == "absolute"


def test_mine_conventions_combines_all_three_categories(handlers, imports_mixed):
    report = mine_conventions([handlers, imports_mixed])

    assert report.error_handling.majority_pattern == "except ValueError: logging.error"
    assert report.docstrings.dominant_style == "google"
    assert report.imports.dominant_style == "relative"  # 3 relative vs. 1 absolute (handlers.py's `import logging`)
