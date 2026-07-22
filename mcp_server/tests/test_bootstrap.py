"""Tests for loupe_mcp_server/bootstrap.py (docs/phase-4-systems.md §8 — Bootstrap)."""

import dataclasses
import shutil
import sqlite3
from pathlib import Path

import pytest

from loupe_mcp_server import bootstrap as bootstrap_module
from loupe_mcp_server.bootstrap import bootstrap, update_index
from loupe_mcp_server.compute_profiles import resolve_embedding_dim
from loupe_mcp_server.config import load_config
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


# --------------------------------------------------------------------------
# Regression: full index must actually honor .loupeignore / manifest
# exclude_paths, not just the built-in default names (found while indexing
# a real repo with a "backend/.venv-py314-backup/" directory — a stale venv
# whose name doesn't exactly match ".venv", so it silently got fully parsed
# and embedded: ~4,000 unwanted files against 69 real ones).
# --------------------------------------------------------------------------


def _write_rogue_venv_lookalike(repo_root: Path, rel_dir: str, package_count: int = 3) -> None:
    """A directory shaped like a stray venv backup — real, parseable .py files with
    real symbols, at a name the built-in default list does NOT exactly match."""
    for i in range(package_count):
        pkg_dir = repo_root / rel_dir / f"fake_pkg_{i}"
        pkg_dir.mkdir(parents=True, exist_ok=True)
        (pkg_dir / "__init__.py").write_text(f"def vendored_function_{i}():\n    return {i}\n")


def test_full_index_respects_loupeignore_for_a_directory_the_default_list_does_not_catch(mock_repo):
    _write_rogue_venv_lookalike(mock_repo, "backend/.venv-py314-backup")
    (mock_repo / ".loupeignore").write_text(".venv-py314-backup/\n")

    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    index = bootstrap(mock_repo, config)

    assert len(index.symbols) == PHASE1_SYMBOL_COUNT, "the rogue lookalike directory must be excluded via .loupeignore"
    assert not any("fake_pkg" in s.file_path for s in index.symbols)


def test_full_index_respects_manifest_exclude_paths_for_the_same_case(mock_repo):
    _write_rogue_venv_lookalike(mock_repo, "backend/.venv-py314-backup")
    (mock_repo / "loupe.manifest.yaml").write_text(
        "schema_version: 1\nlanguages: [python]\nindex:\n  exclude_paths: ['.venv-py314-backup']\n"
    )

    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    assert config.index.exclude_paths == [".venv-py314-backup"]
    index = bootstrap(mock_repo, config)

    assert len(index.symbols) == PHASE1_SYMBOL_COUNT
    assert not any("fake_pkg" in s.file_path for s in index.symbols)


def test_full_index_without_any_exclude_config_does_not_silently_exclude_a_lookalike_directory(mock_repo):
    """The flip side, verified directly: a name that only resembles a default
    (".venv-py314-backup" vs ".venv") is NOT free — same as any real name, it needs
    an explicit pattern. This is what makes the two tests above real regression
    tests and not just "everything gets excluded no matter what" false positives."""
    _write_rogue_venv_lookalike(mock_repo, "backend/.venv-py314-backup")

    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    index = bootstrap(mock_repo, config)

    assert len(index.symbols) == PHASE1_SYMBOL_COUNT + 3
    assert any("fake_pkg" in s.file_path for s in index.symbols)


class _FixedDimFakeModel:
    """Deterministic, zero-real-embedding stand-in for a compute-profile's
    model — only the *dimension* matters for these tests, and downloading
    real bge-base/bge-large weights just to assert a sqlite table's column
    width would be a slow, pointless dependency on network access."""

    def __init__(self, dim: int) -> None:
        self._dim = dim

    def encode(self, texts, **kwargs):
        return [[0.1] * self._dim for _ in texts]


def _connect_vec(vectors_db_path: Path) -> sqlite3.Connection:
    import sqlite_vec

    conn = sqlite3.connect(str(vectors_db_path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


def _vec_symbols_column_width(vectors_db_path: Path) -> str:
    conn = _connect_vec(vectors_db_path)
    try:
        row = conn.execute("SELECT sql FROM sqlite_master WHERE name = 'vec_symbols'").fetchone()
        return row[0]
    finally:
        conn.close()


def _vec_symbols_row_count(vectors_db_path: Path) -> int:
    conn = _connect_vec(vectors_db_path)
    try:
        return conn.execute("SELECT COUNT(*) FROM vec_symbols").fetchone()[0]
    finally:
        conn.close()


def test_changing_compute_profile_forces_full_reindex_and_recreates_vector_store_at_new_dimension(
    mock_repo, monkeypatch
):
    config = load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml")
    assert config.compute_profile == "cpu_small"
    small_dim = resolve_embedding_dim("cpu_small")
    bootstrap(mock_repo, config, embedding_model=_FixedDimFakeModel(small_dim))

    vectors_db_path = mock_repo / ".loupe" / "vectors.db"
    assert f"FLOAT[{small_dim}]" in _vec_symbols_column_width(vectors_db_path)
    assert _vec_symbols_row_count(vectors_db_path) == PHASE1_SYMBOL_COUNT

    gpu_config = dataclasses.replace(config, compute_profile="gpu_large")
    large_dim = resolve_embedding_dim("gpu_large")
    assert large_dim != small_dim

    call_count_fn = _spy_on_extract_symbols(monkeypatch)
    index = bootstrap(mock_repo, gpu_config, embedding_model=_FixedDimFakeModel(large_dim))

    assert call_count_fn() == len(PHASE1_FILES), "a compute_profile change must force a full reindex of every file"
    assert index.embedding_dim == large_dim

    # The table schema itself changed to the new width — proof the old
    # narrower table was actually recreated, not just logically ignored.
    assert f"FLOAT[{large_dim}]" in _vec_symbols_column_width(vectors_db_path)
    # Exactly the current symbol count, not double — proof old- and
    # new-profile vectors were never mixed in the same table; the old file
    # was discarded outright rather than reused/appended to.
    assert _vec_symbols_row_count(vectors_db_path) == PHASE1_SYMBOL_COUNT


def test_update_index_reuses_the_bootstrapped_compute_profiles_dim_and_model(mock_repo):
    """Regression test for a real bug found while wiring this up: `update_index`
    used to build a brand-new `SemanticIndex` with neither `dim=` nor `model=`
    passed through, silently falling back to the base 384-dim default — which
    would have reopened the on-disk vector table at the wrong width for any
    project running a non-default compute profile, on the very first
    incremental reindex after the initial full one.
    """
    config = dataclasses.replace(load_config(mock_repo, global_config_path=mock_repo / "no-global.yaml"))
    gpu_config = dataclasses.replace(config, compute_profile="gpu_large")
    large_dim = resolve_embedding_dim("gpu_large")
    index = bootstrap(mock_repo, gpu_config, embedding_model=_FixedDimFakeModel(large_dim))
    assert index.embedding_dim == large_dim

    (mock_repo / "utils.py").write_text(
        "def format_currency(amount: float) -> str:\n"
        '    """Reformatted body — just needs to change the content hash."""\n'
        "    return f'${amount:.2f} USD'\n"
    )

    updated = update_index(index, {"utils.py"})

    assert updated.embedding_dim == large_dim
    vectors_db_path = mock_repo / ".loupe" / "vectors.db"
    assert f"FLOAT[{large_dim}]" in _vec_symbols_column_width(vectors_db_path)
