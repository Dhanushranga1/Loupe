"""Tests for app/indexer_worker.py (docs/phase-4-systems.md §8 — Live updates).

Real wall-clock waits, not mocked time — the debounce window (300ms) and
check interval (500ms) are small enough that real-time testing stays fast
(~1-2s per test) while still exercising the actual watchdog Observer thread.
"""

import asyncio
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from loupe_mcp_server.main import create_app

PHASE1_FIXTURES = Path(__file__).parent.parent.parent / "core" / "tests" / "fixtures" / "phase1"
PHASE1_FILES = ["utils.py", "models.py", "services.py", "handlers.py", "circular_a.py", "circular_b.py"]

SETTLE_WAIT_SECONDS = 1.2  # comfortably past debounce_window(0.3s) + check_interval(0.5s)


@pytest.fixture
def repo(tmp_path):
    for f in PHASE1_FILES:
        shutil.copy(PHASE1_FIXTURES / f, tmp_path / f)
    return tmp_path


def test_editing_a_file_is_reflected_after_debounce_without_restart(repo):
    app = create_app(repo_root=repo)
    with TestClient(app) as client:
        before = client.get("/list_symbols", params={"path_or_glob": "utils.py"}).json()
        assert {s["qualified_name"] for s in before} == {"format_currency", "validate_email"}

        (repo / "utils.py").write_text(
            "def format_currency(amount: float) -> str:\n"
            '    """Format a numeric amount as a display-ready currency string."""\n'
            "    return f'${amount:.2f}'\n"
            "\n\n"
            "def validate_email(email: str) -> bool:\n"
            '    """Return True if the given string looks like a valid email address."""\n'
            '    return "@" in email and "." in email\n'
            "\n\n"
            "def new_helper() -> int:\n"
            '    """A brand-new function added after the server started."""\n'
            "    return 42\n"
        )

        time.sleep(SETTLE_WAIT_SECONDS)

        after = client.get("/list_symbols", params={"path_or_glob": "utils.py"}).json()
        assert {s["qualified_name"] for s in after} == {"format_currency", "validate_email", "new_helper"}


def test_two_rapid_writes_within_debounce_window_trigger_exactly_one_reparse(repo):
    app = create_app(repo_root=repo)
    with TestClient(app) as client:
        worker = app.state.indexer_worker
        assert worker.reparse_count == 0

        (repo / "utils.py").write_text("def format_currency(amount):\n    return str(amount)\n")
        time.sleep(0.05)  # well within the 300ms debounce window
        (repo / "utils.py").write_text("def format_currency(amount):\n    return f'${amount}'\n")

        time.sleep(SETTLE_WAIT_SECONDS)

        assert worker.reparse_count == 1, "two writes inside one debounce window must settle into a single re-parse"


def test_semantic_search_works_after_a_real_incremental_reindex(repo):
    """A real bug found dogfooding a live `loupe serve` process, not caught by
    any test before this one: `update_index` runs inside `asyncio.to_thread`
    (a threadpool thread) and rebuilds `SemanticIndex`'s sqlite connections
    there; a later `/search_symbols` request is handled on the main event
    loop thread — a different thread — and sqlite3's default same-thread
    check raised "SQLite objects created in a thread can only be used in
    that same thread" the moment a live server actually went through this
    exact sequence. Fixed with `check_same_thread=False` on both
    `VectorStore` and `EmbeddingCache` (see their own module comments); this
    test exercises the real end-to-end path the bug actually manifested at,
    not just the lower-level connection behavior test_vector_store.py and
    test_semantic.py also cover in isolation.
    """
    app = create_app(repo_root=repo)
    with TestClient(app) as client:
        # A real incremental reindex, exactly like the other tests in this
        # file — runs update_index inside asyncio.to_thread on a threadpool
        # thread, swapping in a freshly-built SemanticIndex.
        (repo / "utils.py").write_text(
            "def format_currency(amount: float) -> str:\n"
            '    """Format a numeric amount as a display-ready currency string."""\n'
            "    return f'${amount:.2f}'\n"
        )
        time.sleep(SETTLE_WAIT_SECONDS)

        # Handled on the main event loop thread — a different thread than
        # whichever threadpool worker built the index above.
        response = client.get("/search_symbols", params={"query": "format currency", "top_k": 3})

        assert response.status_code == 200
        names = {s["qualified_name"] for s in response.json()}
        assert "format_currency" in names
