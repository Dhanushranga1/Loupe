"""Tests for parsing/incremental.py's FileIndexCache (docs/phase-0-foundations.md §5/§7)."""

from loupe_core.parsing.extractor import extract_symbols
from loupe_core.parsing.incremental import FileIndexCache


def test_unindexed_file_is_stale(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    cache = FileIndexCache()
    assert cache.is_stale(str(f)) is True


def test_unchanged_file_is_not_stale_after_update(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    cache = FileIndexCache()
    cache.update(str(f), extract_symbols(str(f)))
    assert cache.is_stale(str(f)) is False


def test_changed_file_is_stale_after_update(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    cache = FileIndexCache()
    cache.update(str(f), extract_symbols(str(f)))

    f.write_text("def f():\n    return 2\n")
    assert cache.is_stale(str(f)) is True


def test_update_records_symbol_ids(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    cache = FileIndexCache()
    symbols = extract_symbols(str(f))
    cache.update(str(f), symbols)
    assert set(cache.symbol_ids_for(str(f))) == {s.id for s in symbols}


def test_unindexed_file_has_no_symbol_ids():
    cache = FileIndexCache()
    assert cache.symbol_ids_for("never/seen.py") == []


def test_json_round_trip_preserves_staleness_state(tmp_path):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    cache = FileIndexCache()
    cache.update(str(f), extract_symbols(str(f)))

    cache_path = tmp_path / "cache.json"
    cache.save(str(cache_path))
    reloaded = FileIndexCache.load(str(cache_path))

    assert reloaded.is_stale(str(f)) is False
    assert reloaded.symbol_ids_for(str(f)) == cache.symbol_ids_for(str(f))

    f.write_text("def f():\n    return 2\n")
    assert reloaded.is_stale(str(f)) is True
