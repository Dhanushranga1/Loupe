"""Tests for ledger/build_ledger.py (docs/target-project-build-ledger.md)."""

import shutil
from pathlib import Path

import pytest

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.graph.test_linkage import link_tests
from loupe_core.ledger.build_ledger import BuildLedgerStore, LedgerStatus

E1_FIXTURES = Path(__file__).parent / "fixtures" / "e1"
E1_FILES = ["utils.py", "models.py", "services.py"]

_CHANGED_UTILS_PY = (
    'def format_currency(amount: float) -> str:\n'
    '    """Changed body -- must change the content hash."""\n'
    '    return f"USD {amount:.2f}"\n\n\n'
    'def validate_email(email: str) -> bool:\n'
    '    return "@" in email and "." in email\n\n\n'
    'def unused_utility() -> None:\n'
    '    return None\n'
)


def _id_by_name(symbols_by_id, qualified_name: str) -> str:
    return next(s.id for s in symbols_by_id.values() if s.qualified_name == qualified_name)


def _parse_and_build(files):
    parsed = [parse_file(f) for f in files]
    g = build_graph(parsed)
    symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
    return g, symbols_by_id


@pytest.fixture
def e1_repo(tmp_path, monkeypatch):
    for f in E1_FILES:
        shutil.copy(E1_FIXTURES / f, tmp_path / f)
    monkeypatch.chdir(tmp_path)
    g, symbols_by_id = _parse_and_build(E1_FILES)
    return g, symbols_by_id, tmp_path


def test_set_status_creates_entry_and_auto_populates_depends_on(e1_repo):
    g, symbols_by_id, tmp_path = e1_repo
    order_total_id = _id_by_name(symbols_by_id, "Order.total")
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")

    store = BuildLedgerStore(tmp_path / "ledger.json")
    entry = store.set_status(order_total_id, LedgerStatus.PLANNED, graph=g.graph, symbols_by_id=symbols_by_id)

    assert entry.depends_on == [format_currency_id]
    assert entry.status == LedgerStatus.PLANNED


def test_set_status_reuses_previously_computed_depends_on_on_update(e1_repo):
    g, symbols_by_id, tmp_path = e1_repo
    order_total_id = _id_by_name(symbols_by_id, "Order.total")
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")

    store = BuildLedgerStore(tmp_path / "ledger.json")
    store.set_status(order_total_id, LedgerStatus.PLANNED, graph=g.graph, symbols_by_id=symbols_by_id)
    entry = store.set_status(order_total_id, LedgerStatus.IN_PROGRESS, graph=g.graph, symbols_by_id=symbols_by_id)

    assert entry.status == LedgerStatus.IN_PROGRESS
    assert entry.depends_on == [format_currency_id]


def test_depends_on_pointing_outside_the_ledger_is_still_stored_as_real_graph_data(e1_repo):
    """§7's second criterion: format_currency isn't itself tracked, but
    Order.total's depends_on still names it — the store records real graph
    data regardless of ledger membership; filtering that down to a
    "blocking" concern is a presentation-layer job (the CLI/tool), not the
    store's."""
    g, symbols_by_id, tmp_path = e1_repo
    order_total_id = _id_by_name(symbols_by_id, "Order.total")
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")

    store = BuildLedgerStore(tmp_path / "ledger.json")
    entry = store.set_status(order_total_id, LedgerStatus.PLANNED, graph=g.graph, symbols_by_id=symbols_by_id)

    assert format_currency_id in entry.depends_on
    tracked_ids = {e.symbol_id for e in store.list(graph=g.graph, symbols_by_id=symbols_by_id, pagerank_scores=g.pagerank_scores)}
    assert format_currency_id not in tracked_ids


def test_persists_across_a_fresh_store_instance(e1_repo):
    g, symbols_by_id, tmp_path = e1_repo
    order_total_id = _id_by_name(symbols_by_id, "Order.total")
    ledger_path = tmp_path / "ledger.json"

    BuildLedgerStore(ledger_path).set_status(order_total_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)

    reloaded = BuildLedgerStore(ledger_path)
    entries = reloaded.list(graph=g.graph, symbols_by_id=symbols_by_id, pagerank_scores=g.pagerank_scores)
    assert len(entries) == 1
    assert entries[0].symbol_id == order_total_id
    assert entries[0].status == LedgerStatus.DONE


def test_symbol_id_is_stable_across_a_reparse_of_unchanged_source(e1_repo):
    """§7's last criterion: the ledger survives a reindex because symbol_id
    is stable across a reparse of unchanged content (Phase 0's own
    guarantee) — verified directly for this mechanism, not just trusted
    transitively from Phase 0's own tests."""
    g, symbols_by_id, tmp_path = e1_repo
    order_total_id = _id_by_name(symbols_by_id, "Order.total")
    store = BuildLedgerStore(tmp_path / "ledger.json")
    store.set_status(order_total_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)

    new_g, new_symbols_by_id = _parse_and_build(E1_FILES)  # nothing edited on disk

    entry = store.get(order_total_id, graph=new_g.graph, symbols_by_id=new_symbols_by_id, pagerank_scores=new_g.pagerank_scores)
    assert entry is not None
    assert entry.status == LedgerStatus.DONE, "unchanged content must not be flagged stale"


def test_direct_staleness_detected_lazily_on_read_after_source_changes(e1_repo):
    g, symbols_by_id, tmp_path = e1_repo
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")
    store = BuildLedgerStore(tmp_path / "ledger.json")
    store.set_status(format_currency_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)

    (tmp_path / "utils.py").write_text(_CHANGED_UTILS_PY)
    new_g, new_symbols_by_id = _parse_and_build(E1_FILES)

    entry = store.get(
        format_currency_id, graph=new_g.graph, symbols_by_id=new_symbols_by_id, pagerank_scores=new_g.pagerank_scores
    )
    assert entry.status == LedgerStatus.STALE


def test_transitive_staleness_flags_a_done_entry_whose_dependency_changed(e1_repo):
    """§7's transitive criterion, on the exact chain E1's own acceptance
    tests use: format_currency <- Order.total <- Order.describe. Order.total
    is deliberately left untracked — only format_currency and Order.describe
    are DONE ledger entries."""
    g, symbols_by_id, tmp_path = e1_repo
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")
    order_total_id = _id_by_name(symbols_by_id, "Order.total")
    order_describe_id = _id_by_name(symbols_by_id, "Order.describe")

    store = BuildLedgerStore(tmp_path / "ledger.json")
    store.set_status(format_currency_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)
    store.set_status(order_describe_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)

    (tmp_path / "utils.py").write_text(_CHANGED_UTILS_PY)
    new_g, new_symbols_by_id = _parse_and_build(E1_FILES)

    entries = store.list(graph=new_g.graph, symbols_by_id=new_symbols_by_id, pagerank_scores=new_g.pagerank_scores)
    by_id = {e.symbol_id: e for e in entries}

    assert by_id[format_currency_id].status == LedgerStatus.STALE, "direct: its own content_hash changed"
    assert by_id[order_describe_id].status == LedgerStatus.STALE, "transitive: its dependency's dependency changed"
    assert order_total_id not in by_id, "never tracked, so never created or flagged"


def test_stale_does_not_silently_revert_without_an_explicit_status_change(e1_repo):
    g, symbols_by_id, tmp_path = e1_repo
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")
    store = BuildLedgerStore(tmp_path / "ledger.json")
    store.set_status(format_currency_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)

    (tmp_path / "utils.py").write_text(_CHANGED_UTILS_PY)
    new_g, new_symbols_by_id = _parse_and_build(E1_FILES)

    first_read = store.get(
        format_currency_id, graph=new_g.graph, symbols_by_id=new_symbols_by_id, pagerank_scores=new_g.pagerank_scores
    )
    second_read = store.get(
        format_currency_id, graph=new_g.graph, symbols_by_id=new_symbols_by_id, pagerank_scores=new_g.pagerank_scores
    )
    assert first_read.status == LedgerStatus.STALE
    assert second_read.status == LedgerStatus.STALE


def test_marking_done_again_after_stale_clears_it(e1_repo):
    g, symbols_by_id, tmp_path = e1_repo
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")
    store = BuildLedgerStore(tmp_path / "ledger.json")
    store.set_status(format_currency_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)

    (tmp_path / "utils.py").write_text(_CHANGED_UTILS_PY)
    new_g, new_symbols_by_id = _parse_and_build(E1_FILES)
    store.get(format_currency_id, graph=new_g.graph, symbols_by_id=new_symbols_by_id, pagerank_scores=new_g.pagerank_scores)

    re_marked = store.set_status(
        format_currency_id, LedgerStatus.DONE, graph=new_g.graph, symbols_by_id=new_symbols_by_id
    )
    assert re_marked.status == LedgerStatus.DONE
    still_done = store.get(
        format_currency_id, graph=new_g.graph, symbols_by_id=new_symbols_by_id, pagerank_scores=new_g.pagerank_scores
    )
    assert still_done.status == LedgerStatus.DONE


def test_list_filters_by_status(e1_repo):
    g, symbols_by_id, tmp_path = e1_repo
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")
    order_total_id = _id_by_name(symbols_by_id, "Order.total")
    store = BuildLedgerStore(tmp_path / "ledger.json")
    store.set_status(format_currency_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)
    store.set_status(order_total_id, LedgerStatus.PLANNED, graph=g.graph, symbols_by_id=symbols_by_id)

    done_only = store.list(
        graph=g.graph, symbols_by_id=symbols_by_id, pagerank_scores=g.pagerank_scores, filter_status=LedgerStatus.DONE
    )
    assert [e.symbol_id for e in done_only] == [format_currency_id]


def test_has_test_true_only_when_a_test_edge_actually_links_to_the_symbol(tmp_path, monkeypatch):
    (tmp_path / "utils.py").write_text(
        'def format_currency(amount: float) -> str:\n    return f"${amount:.2f}"\n\n\n'
        "def untested_function() -> None:\n    return None\n"
    )
    (tmp_path / "test_utils.py").write_text(
        "from utils import format_currency\n\n\n"
        'def test_format_currency():\n    assert format_currency(1.0) == "$1.00"\n'
    )
    monkeypatch.chdir(tmp_path)
    files = ["utils.py", "test_utils.py"]
    g, symbols_by_id = _parse_and_build(files)
    link_tests(g.graph, symbols_by_id)

    format_currency_id = _id_by_name(symbols_by_id, "format_currency")
    untested_id = _id_by_name(symbols_by_id, "untested_function")

    store = BuildLedgerStore(tmp_path / "ledger.json")
    tested_entry = store.set_status(format_currency_id, LedgerStatus.PLANNED, graph=g.graph, symbols_by_id=symbols_by_id)
    untested_entry = store.set_status(untested_id, LedgerStatus.PLANNED, graph=g.graph, symbols_by_id=symbols_by_id)

    assert tested_entry.has_test is True
    assert untested_entry.has_test is False
