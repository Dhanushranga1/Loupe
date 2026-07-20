"""Tests for storage/vector_store.py (docs/phase-2-retrieval.md §5)."""

import math

from loupe_core.storage.vector_store import VectorStore

ID_A = "a" * 16
ID_B = "b" * 16
ID_C = "c" * 16


def test_identical_vector_has_similarity_one():
    store = VectorStore(dim=4)
    store.upsert(ID_A, [1.0, 0.0, 0.0, 0.0])
    results = dict(store.query([1.0, 0.0, 0.0, 0.0], top_k=1))
    assert math.isclose(results[ID_A], 1.0, abs_tol=1e-6)


def test_orthogonal_vector_has_similarity_near_zero():
    store = VectorStore(dim=4)
    store.upsert(ID_A, [1.0, 0.0, 0.0, 0.0])
    store.upsert(ID_B, [0.0, 1.0, 0.0, 0.0])
    results = dict(store.query([1.0, 0.0, 0.0, 0.0], top_k=2))
    assert math.isclose(results[ID_B], 0.0, abs_tol=1e-5)


def test_ranking_orders_most_similar_first():
    store = VectorStore(dim=4)
    store.upsert(ID_A, [1.0, 0.0, 0.0, 0.0])
    store.upsert(ID_B, [0.0, 1.0, 0.0, 0.0])
    store.upsert(ID_C, [0.9, math.sqrt(1 - 0.81), 0.0, 0.0])

    ranked_ids = [symbol_id for symbol_id, _ in store.query([1.0, 0.0, 0.0, 0.0], top_k=3)]
    assert ranked_ids == [ID_A, ID_C, ID_B]


def test_delete_removes_symbol_from_results():
    store = VectorStore(dim=4)
    store.upsert(ID_A, [1.0, 0.0, 0.0, 0.0])
    store.upsert(ID_B, [0.0, 1.0, 0.0, 0.0])
    store.delete(ID_B)
    ranked_ids = [symbol_id for symbol_id, _ in store.query([1.0, 0.0, 0.0, 0.0], top_k=10)]
    assert ranked_ids == [ID_A]


def test_upsert_replaces_existing_embedding():
    store = VectorStore(dim=4)
    store.upsert(ID_A, [1.0, 0.0, 0.0, 0.0])
    store.upsert(ID_A, [0.0, 1.0, 0.0, 0.0])  # same id, new vector
    results = dict(store.query([0.0, 1.0, 0.0, 0.0], top_k=10))
    assert math.isclose(results[ID_A], 1.0, abs_tol=1e-6)
    assert len(store.query([0.0, 1.0, 0.0, 0.0], top_k=10)) == 1, "must not leave a stale duplicate row"


def test_sync_removes_symbols_not_in_current_set():
    store = VectorStore(dim=4)
    store.upsert(ID_A, [1.0, 0.0, 0.0, 0.0])
    store.upsert(ID_B, [0.0, 1.0, 0.0, 0.0])
    store.upsert(ID_C, [0.0, 0.0, 1.0, 0.0])

    store.sync({ID_A, ID_C})

    ranked_ids = {symbol_id for symbol_id, _ in store.query([1.0, 0.0, 0.0, 0.0], top_k=10)}
    assert ranked_ids == {ID_A, ID_C}


def test_empty_store_returns_no_results():
    store = VectorStore(dim=4)
    assert store.query([1.0, 0.0, 0.0, 0.0], top_k=5) == []


def test_store_created_in_one_thread_is_queryable_from_another():
    """A real bug found dogfooding a live `loupe serve` process, not in any
    test: mcp_server's IndexerWorker rebuilds the index (and this VectorStore)
    inside `asyncio.to_thread`, a threadpool thread distinct from whichever
    thread later serves a request against the swapped-in index — the exact
    "SQLite objects created in a thread can only be used in that same
    thread" error, reproduced directly here without needing a live server.
    """
    import queue
    import threading

    store_queue: queue.Queue[VectorStore] = queue.Queue()

    def _build_store() -> None:
        store = VectorStore(dim=4)
        store.upsert(ID_A, [1.0, 0.0, 0.0, 0.0])
        store_queue.put(store)

    builder_thread = threading.Thread(target=_build_store)
    builder_thread.start()
    builder_thread.join()

    store = store_queue.get()
    # Queried from *this* (the test's own) thread — a different thread than
    # the one that constructed it.
    results = dict(store.query([1.0, 0.0, 0.0, 0.0], top_k=1))
    assert math.isclose(results[ID_A], 1.0, abs_tol=1e-6)
