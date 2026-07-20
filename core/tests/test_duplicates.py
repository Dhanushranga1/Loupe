"""Tests for analysis/duplicates.py (docs/PhaseX/zero-cost-static-analysis-pack.md E5).

Uses the real embedding model (session-scoped) — the actual similarity
numbers this module's threshold was calibrated against, not fabricated data.
"""

import os
from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from loupe_core.analysis.duplicates import DEFAULT_SIMILARITY_THRESHOLD, find_duplicates
from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME, SemanticIndex


@pytest.fixture(scope="session")
def real_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def _build(repo_root: Path, files: list[str], model):
    old_cwd = os.getcwd()
    os.chdir(repo_root)
    try:
        parsed = [parse_file(f) for f in files]
        symbols = [s for pf in parsed for s in pf.symbols]
        graph = build_graph(parsed)
        semantic_index = SemanticIndex(model=model)
        semantic_index.index(symbols)
        return semantic_index, symbols, graph.graph
    finally:
        os.chdir(old_cwd)


def _by_name(symbols, name: str):
    return next(s for s in symbols if s.qualified_name == name)


@pytest.fixture
def near_duplicate_repo(tmp_path, real_model):
    _write(
        tmp_path,
        "math_utils.py",
        'def calculate_average(numbers):\n'
        '    """Compute the average of a list of numbers."""\n'
        '    total = sum(numbers)\n'
        '    count = len(numbers)\n'
        '    return total / count\n',
    )
    _write(
        tmp_path,
        "stats_helpers.py",
        'def calculate_mean(values):\n'
        '    """Compute the average of a list of numbers."""\n'
        '    total = sum(values)\n'
        '    count = len(values)\n'
        '    return total / count\n',
    )
    return _build(tmp_path, ["math_utils.py", "stats_helpers.py"], real_model)


def test_copy_pasted_near_identical_functions_are_flagged(near_duplicate_repo):
    """§6's own acceptance criterion: two near-identical functions
    (copy-pasted with only variable names changed) in different files are
    correctly flagged."""
    semantic_index, symbols, graph = near_duplicate_repo
    average = _by_name(symbols, "calculate_average")
    mean = _by_name(symbols, "calculate_mean")

    findings = find_duplicates(semantic_index, symbols, graph)
    flagged_pairs = {frozenset({f.symbol_id_a, f.symbol_id_b}) for f in findings}

    assert frozenset({average.id, mean.id}) in flagged_pairs


def test_legitimate_wrapper_is_excluded_despite_high_similarity(tmp_path, real_model):
    """§6's own acceptance criterion: two functions similar because one
    calls the other (a legitimate wrapper, not a duplicate) are correctly
    excluded, since Phase 1's graph shows a real relationship between them.
    """
    _write(
        tmp_path,
        "processing.py",
        'def process_data(x):\n'
        '    """Compute the average of a list of numbers."""\n'
        '    total = sum(x)\n'
        '    count = len(x)\n'
        '    return total / count\n\n\n'
        'def calculate_mean(x):\n'
        '    """Compute the average of a list of numbers."""\n'
        '    return process_data(x)\n',
    )
    semantic_index, symbols, graph = _build(tmp_path, ["processing.py"], real_model)
    process_data = _by_name(symbols, "process_data")
    calculate_mean = _by_name(symbols, "calculate_mean")

    findings = find_duplicates(semantic_index, symbols, graph)
    flagged_pairs = {frozenset({f.symbol_id_a, f.symbol_id_b}) for f in findings}

    assert frozenset({process_data.id, calculate_mean.id}) not in flagged_pairs


def test_same_file_near_duplicates_are_not_flagged(tmp_path, real_model):
    """The spec's own exclusion is "different files" — two near-identical
    helpers in the *same* file aren't the copy-paste-across-the-repo
    problem this check targets."""
    _write(
        tmp_path,
        "same_file.py",
        'def calculate_average(numbers):\n'
        '    """Compute the average of a list of numbers."""\n'
        '    total = sum(numbers)\n'
        '    count = len(numbers)\n'
        '    return total / count\n\n\n'
        'def calculate_mean(values):\n'
        '    """Compute the average of a list of numbers."""\n'
        '    total = sum(values)\n'
        '    count = len(values)\n'
        '    return total / count\n',
    )
    semantic_index, symbols, graph = _build(tmp_path, ["same_file.py"], real_model)

    findings = find_duplicates(semantic_index, symbols, graph)

    assert findings == []


def test_unrelated_functions_are_not_flagged(tmp_path, real_model):
    _write(
        tmp_path,
        "a.py",
        'def calculate_average(numbers):\n'
        '    """Compute the average of a list of numbers."""\n'
        '    return sum(numbers) / len(numbers)\n',
    )
    _write(
        tmp_path,
        "b.py",
        'def greet(name):\n    """Greet a person by name with a friendly message."""\n    return f"Hello, {name}!"\n',
    )
    semantic_index, symbols, graph = _build(tmp_path, ["a.py", "b.py"], real_model)

    findings = find_duplicates(semantic_index, symbols, graph)

    assert findings == []


def test_default_threshold_is_the_calibrated_constant():
    assert DEFAULT_SIMILARITY_THRESHOLD == 0.90
