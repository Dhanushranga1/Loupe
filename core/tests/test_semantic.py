"""Tests for retrieval/semantic.py's embedding cache behavior (docs/phase-2-retrieval.md §8).

Uses the real bge-small-en-v1.5 model wrapped in a call-counting spy — not a
fake/stand-in model. Embeddings are genuinely computed; the spy only adds
call-count bookkeeping on top, which is what phase-2-retrieval.md §8 itself
asks for ("verified with a mock/spy on model.encode"). The model loads once
per test session (~10-15s) via a session-scoped fixture, not once per test.
"""

from pathlib import Path

import pytest
from sentence_transformers import SentenceTransformer

from loupe_core.graph.builder import parse_file
from loupe_core.retrieval.semantic import EMBEDDING_MODEL_NAME, SemanticIndex, embed_text_for_symbol


class EncodeSpy:
    """Delegates to a real model's encode() while counting calls/volume — a spy, not a fake."""

    def __init__(self, real_model: SentenceTransformer):
        self._real_model = real_model
        self.encode_call_count = 0
        self.total_texts_encoded = 0

    def encode(self, texts, **kwargs):
        self.encode_call_count += 1
        self.total_texts_encoded += len(texts)
        return self._real_model.encode(texts, **kwargs)


@pytest.fixture(scope="session")
def real_model():
    return SentenceTransformer(EMBEDDING_MODEL_NAME)


def test_reindexing_unchanged_corpus_makes_zero_further_encode_calls(tmp_path, real_model):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    '''doc'''\n    return 1\n\n\ndef g():\n    return 2\n")
    pf = parse_file(str(f))

    spy = EncodeSpy(real_model)
    index = SemanticIndex(model=spy)
    index.index(pf.symbols)
    assert spy.encode_call_count == 1  # first index: everything is new

    index.index(pf.symbols)  # nothing changed on disk
    assert spy.encode_call_count == 1, "re-indexing unchanged symbols must not call the model again"


def test_changing_one_symbol_reembeds_only_that_symbol(tmp_path, real_model):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    pf = parse_file(str(f))

    spy = EncodeSpy(real_model)
    index = SemanticIndex(model=spy)
    index.index(pf.symbols)
    assert spy.total_texts_encoded == 2

    f.write_text("def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    pf2 = parse_file(str(f))
    index.index(pf2.symbols)

    assert spy.total_texts_encoded == 3, "only the one changed symbol should be re-embedded"


def test_removing_a_symbol_deletes_its_cache_row(tmp_path, real_model):
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    pf = parse_file(str(f))
    f_id = next(s.id for s in pf.symbols if s.qualified_name == "f")
    g_id = next(s.id for s in pf.symbols if s.qualified_name == "g")

    spy = EncodeSpy(real_model)
    index = SemanticIndex(model=spy)
    index.index(pf.symbols)
    assert index.is_cached(f_id)
    assert index.is_cached(g_id)

    f.write_text("def g():\n    return 2\n")  # f() removed entirely
    pf2 = parse_file(str(f))
    index.index(pf2.symbols)

    assert not index.is_cached(f_id), "removed symbol's cache row must be deleted, not left stale"
    assert index.is_cached(g_id)


def test_get_embedding_returns_the_cached_vector_for_an_indexed_symbol_none_otherwise(tmp_path, real_model):
    """Phase 9's MMR selection (retrieval-upgrades §4) needs raw embedding vectors
    directly, not just KNN query hits — the real second consumer `get_embedding`
    was added for."""
    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")
    pf = parse_file(str(f))
    f_id = next(s.id for s in pf.symbols if s.qualified_name == "f")

    index = SemanticIndex(model=real_model)
    assert index.get_embedding(f_id) is None, "never-indexed symbol has no cached embedding"

    index.index(pf.symbols)
    embedding = index.get_embedding(f_id)
    assert embedding is not None
    assert len(embedding) == 384


def test_batched_not_one_by_one(tmp_path, real_model):
    f = tmp_path / "a.py"
    f.write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n\n\ndef c():\n    return 3\n")
    pf = parse_file(str(f))

    spy = EncodeSpy(real_model)
    index = SemanticIndex(model=spy)
    index.index(pf.symbols)

    assert spy.encode_call_count == 1, "all new symbols must be embedded in one batched call"
    assert spy.total_texts_encoded == 3


def test_embedding_cache_created_in_one_thread_is_usable_from_another(tmp_path, real_model):
    """A real bug found dogfooding a live `loupe serve` process, not in any
    test: mcp_server's IndexerWorker rebuilds the index (and this cache)
    inside `asyncio.to_thread`, a threadpool thread distinct from whichever
    thread later serves a request against the swapped-in index — see
    test_vector_store.py's identical test for the full incident writeup.
    """
    import queue
    import threading

    from loupe_core.retrieval.semantic import EmbeddingCache

    f = tmp_path / "a.py"
    f.write_text("def f():\n    return 1\n")
    pf = parse_file(str(f))
    symbol = pf.symbols[0]

    cache_queue: queue.Queue[EmbeddingCache] = queue.Queue()

    def _build_cache() -> None:
        cache = EmbeddingCache()
        cache.put(symbol.id, symbol.content_hash, [0.1, 0.2, 0.3])
        cache_queue.put(cache)

    builder_thread = threading.Thread(target=_build_cache)
    builder_thread.start()
    builder_thread.join()

    cache = cache_queue.get()
    # Read from *this* (the test's own) thread — a different thread than the
    # one that constructed it.
    content_hash, embedding = cache.get(symbol.id)
    assert content_hash == symbol.content_hash
    assert embedding == pytest.approx([0.1, 0.2, 0.3], abs=1e-6)  # packed/unpacked as float32, not exact


def test_semantic_query_finds_paraphrased_match(real_model):
    """A real-model sanity check: semantic search should find a paraphrase, not just keyword overlap."""
    from loupe_core.parsing.schema import Symbol, SymbolKind

    symbol = Symbol(
        id="a" * 16, kind=SymbolKind.FUNCTION, name="format_currency", qualified_name="format_currency",
        file_path="utils.py", byte_start=0, byte_end=1, line_start=1, line_end=1,
        signature="def format_currency(amount: float) -> str:",
        docstring="Format a numeric amount as a display-ready currency string.",
    )
    index = SemanticIndex(model=real_model)
    index.index([symbol])

    results = index.query("turn a number into a displayable price", top_k=1)
    assert results[0][0] == symbol.id
    assert results[0][1] > 0.5, "a genuine paraphrase should score well above an unrelated pair"


def test_docstring_present_embeds_docstring_plus_signature():
    from loupe_core.parsing.schema import Symbol, SymbolKind

    symbol = Symbol(
        id="a" * 16, kind=SymbolKind.FUNCTION, name="f", qualified_name="f", file_path="a.py",
        byte_start=0, byte_end=1, line_start=1, line_end=1,
        signature="def f():", docstring="Does a thing.",
    )
    assert embed_text_for_symbol(symbol) == "Does a thing.\ndef f():"


def test_no_docstring_embeds_qualified_name_plus_signature():
    from loupe_core.parsing.schema import Symbol, SymbolKind

    symbol = Symbol(
        id="a" * 16, kind=SymbolKind.METHOD, name="f", qualified_name="Foo.f", file_path="a.py",
        byte_start=0, byte_end=1, line_start=1, line_end=1,
        signature="def f(self):", docstring=None,
    )
    assert embed_text_for_symbol(symbol) == "Foo.f\ndef f(self):"
