"""Tests for app/bootstrap.py (docs/phase-4-systems.md §8 — Bootstrap)."""

import shutil
from pathlib import Path

import pytest

from app import bootstrap as bootstrap_module
from app.bootstrap import bootstrap
from app.config import load_config
from loupe_core.graph.builder import EdgeType

PHASE1_FIXTURES = Path(__file__).parent.parent.parent / "core" / "tests" / "fixtures" / "phase1"
PHASE1_FILES = ["utils.py", "models.py", "services.py", "handlers.py", "circular_a.py", "circular_b.py"]
PHASE1_SYMBOL_COUNT = 16  # utils(2) + models(4) + services(3) + handlers(5) + circular_a(1) + circular_b(1)

E2_FIXTURES = Path(__file__).parent.parent.parent / "core" / "tests" / "fixtures" / "e2"
E2_FILES = ["utils.py", "test_utils.py"]


@pytest.fixture
def mock_repo(tmp_path, monkeypatch):
    for f in PHASE1_FILES:
        shutil.copy(PHASE1_FIXTURES / f, tmp_path / f)
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def e2_repo(tmp_path, monkeypatch):
    for f in E2_FILES:
        shutil.copy(E2_FIXTURES / f, tmp_path / f)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _spy_on_extract_symbols(monkeypatch):
    call_count = 0
    real_extract = bootstrap_module.extract_symbols

    def spy(file_path):
        nonlocal call_count
        call_count += 1
        return real_extract(file_path)

    monkeypatch.setattr(bootstrap_module, "extract_symbols", spy)
    return lambda: call_count


def test_fresh_bootstrap_creates_full_loupe_structure_and_correct_symbol_count(mock_repo):
    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    index = bootstrap(mock_repo, config)

    loupe_dir = mock_repo / ".loupe"
    assert (loupe_dir / "schema_version").exists()
    assert (loupe_dir / "cache").is_dir()
    assert (loupe_dir / "logs" / "retrieval").is_dir()
    assert (loupe_dir / "logs" / "sessions").is_dir()
    assert (loupe_dir / "eval").is_dir()
    assert len(index.symbols) == PHASE1_SYMBOL_COUNT


def test_second_run_on_unchanged_files_does_not_reinvoke_extractor(mock_repo, monkeypatch):
    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    bootstrap(mock_repo, config)  # first run: full index, populates the caches

    call_count_fn = _spy_on_extract_symbols(monkeypatch)
    index = bootstrap(mock_repo, config)  # second run: nothing changed on disk

    assert call_count_fn() == 0, "unchanged files must not be re-run through the extractor"
    assert len(index.symbols) == PHASE1_SYMBOL_COUNT, "reused cached symbols must still be complete"


def test_editing_one_file_reextracts_only_that_file(mock_repo, monkeypatch):
    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    bootstrap(mock_repo, config)

    (mock_repo / "utils.py").write_text(
        "def format_currency(amount: float) -> str:\n"
        '    """Format a numeric amount as a display-ready currency string."""\n'
        "    return f'${amount:.2f} USD'\n"  # body changed\n"
        "\n\n"
        "def validate_email(email: str) -> bool:\n"
        '    """Return True if the given string looks like a valid email address."""\n'
        '    return "@" in email and "." in email\n'
    )

    reextracted = []
    real_extract = bootstrap_module.extract_symbols

    def spy(file_path):
        reextracted.append(file_path)
        return real_extract(file_path)

    monkeypatch.setattr(bootstrap_module, "extract_symbols", spy)
    bootstrap(mock_repo, config)

    assert reextracted == ["utils.py"], "only the changed file should be re-run through the extractor"


def test_bumping_schema_version_forces_full_reindex(mock_repo, monkeypatch):
    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    bootstrap(mock_repo, config)

    monkeypatch.setattr(bootstrap_module, "INDEX_SCHEMA_VERSION", bootstrap_module.INDEX_SCHEMA_VERSION + 1)
    call_count_fn = _spy_on_extract_symbols(monkeypatch)
    index = bootstrap(mock_repo, config)

    assert call_count_fn() == len(PHASE1_FILES), "a schema version bump must force a full reindex of every file"
    assert len(index.symbols) == PHASE1_SYMBOL_COUNT


def test_bootstrapped_index_graph_and_lexical_search_are_wired_correctly(mock_repo):
    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    index = bootstrap(mock_repo, config)

    # sanity: graph resolution and lexical search both work against the bootstrapped index
    assert index.graph.graph.number_of_edges() > 0

    results = index.lexical_index.query("format currency", top_k=3)
    ranked_names = [index.symbol_by_id(symbol_id).qualified_name for symbol_id, _ in results]
    assert "format_currency" in ranked_names


def test_bootstrap_wires_real_tests_edges_into_the_live_graph(e2_repo):
    """E2's link_tests() is only useful if bootstrap() actually calls it — verified end-to-end
    against a real repo with a real test file, not just at the core function's own unit-test level."""
    config = load_config(e2_repo, global_config_path=e2_repo / "no-global.yaml")
    index = bootstrap(e2_repo, config)

    format_currency = next(s for s in index.symbols if s.qualified_name == "format_currency")
    test_edges = [
        (u, v)
        for u, v, data in index.graph.graph.in_edges(format_currency.id, data=True)
        if data.get("edge_type") == EdgeType.TESTS
    ]
    tester_names = {index.symbol_by_id(u).qualified_name for u, _ in test_edges}
    assert tester_names == {"test_format_currency", "check_currency_formatting"}
