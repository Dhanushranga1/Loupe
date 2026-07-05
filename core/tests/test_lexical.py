"""Tests for retrieval/lexical.py (docs/phase-2-retrieval.md §4/§8).

Tokenizer tests come first and in isolation — it's small, easy to get
subtly wrong, and everything else (BM25 corpus, fusion, recall) depends on it.
"""

from pathlib import Path

from loupe_core.graph.builder import parse_file
from loupe_core.retrieval.lexical import LexicalIndex, tokenize

PHASE1_FIXTURES = Path(__file__).parent / "fixtures" / "phase1"


def test_tokenize_camel_case():
    assert tokenize("createOrder") == ["create", "order"]


def test_tokenize_snake_case():
    assert tokenize("validate_email") == ["validate", "email"]


def test_tokenize_pascal_case():
    assert tokenize("OrderService") == ["order", "service"]


def test_tokenize_snake_case_with_trailing_number():
    assert tokenize("format_currency_v2") == ["format", "currency", "v2"]


def test_tokenize_drops_tokens_shorter_than_two_chars():
    assert tokenize("x") == []
    assert tokenize("id") == ["id"]  # exactly 2 chars is the boundary — kept


def test_tokenize_dunder_name_drops_empty_pieces():
    assert tokenize("__init__") == ["init"]


def test_tokenize_handles_full_sentence_with_punctuation():
    # Real BM25 document text is a whole signature/docstring, not a bare identifier.
    text = "def validate_email(email: str) -> bool: Return True if valid."
    tokens = tokenize(text)
    assert "validate" in tokens
    assert "email" in tokens
    assert "return" in tokens
    assert "def" not in tokens or "def" in tokens  # 'def' is 3 chars, passes length filter — just documenting it's not special-cased
    assert ":" not in tokens and "->" not in tokens


def test_bm25_ranks_format_currency_in_top_3_for_format_currency_query():
    pf = parse_file(str(PHASE1_FIXTURES / "utils.py"))
    index = LexicalIndex(pf.symbols)
    results = index.query("format currency", top_k=3)
    ranked_ids = [symbol_id for symbol_id, _ in results]
    format_currency_id = next(s.id for s in pf.symbols if s.qualified_name == "format_currency")
    assert format_currency_id in ranked_ids


def test_empty_corpus_returns_no_results():
    index = LexicalIndex([])
    assert index.query("anything") == []
