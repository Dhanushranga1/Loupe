"""Tests for governor/budget.py (docs/phase-3-resource-allocation.md §8).

Hardcoded expected counts below were computed directly against the real
cl100k_base encoding (raw count × 1.1 safety margin, rounded up) — no
mocking needed, tiktoken is deterministic.
"""

from pathlib import Path

from loupe_core.governor.budget import estimate_tokens, symbol_discovery_cost, symbol_extraction_cost
from loupe_core.graph.builder import parse_file

PHASE1_FIXTURES = Path(__file__).parent / "fixtures" / "phase1"


def test_estimate_tokens_deterministic_across_repeated_calls():
    text = "def format_currency(amount: float) -> str:"
    assert estimate_tokens(text) == estimate_tokens(text)


def test_estimate_tokens_hardcoded_expected_counts():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hello") == 2
    assert estimate_tokens("a") == 2
    assert estimate_tokens("def format_currency(amount: float) -> str:") == 11


def test_discovery_cost_uses_signature_and_first_docstring_line_only():
    from loupe_core.parsing.schema import Symbol, SymbolKind

    symbol = Symbol(
        id="a" * 16, kind=SymbolKind.FUNCTION, name="f", qualified_name="f", file_path="a.py",
        byte_start=0, byte_end=1, line_start=1, line_end=1,
        signature="def f():",
        docstring="First line of the docstring.\nSecond line, should be excluded from discovery text.",
    )
    expected_text = "def f():\nFirst line of the docstring."
    assert symbol_discovery_cost(symbol) == estimate_tokens(expected_text)


def test_discovery_cost_falls_back_to_signature_alone_without_docstring():
    from loupe_core.parsing.schema import Symbol, SymbolKind

    symbol = Symbol(
        id="a" * 16, kind=SymbolKind.FUNCTION, name="f", qualified_name="f", file_path="a.py",
        byte_start=0, byte_end=1, line_start=1, line_end=1,
        signature="def f():", docstring=None,
    )
    assert symbol_discovery_cost(symbol) == estimate_tokens("def f():")


def test_discovery_cost_meaningfully_smaller_than_extraction_cost(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(
        "def create_order(email: str, amount: float) -> str:\n"
        "    \"\"\"Validate, create, and log a new order; return it as a JSON string.\"\"\"\n"
        "    validate_email(email)\n"
        "    log(f'creating order for {email}')\n"
        "    order = Order(email, amount)\n"
        "    return json.dumps({'email': email, 'amount': amount})\n"
    )
    pf = parse_file(str(f))
    symbol = pf.symbols[0]

    discovery = symbol_discovery_cost(symbol)
    extraction = symbol_extraction_cost(symbol, pf.source_bytes)

    assert extraction > discovery
    assert extraction >= discovery * 2, "extraction (full body) should be substantially larger, not just marginally"


def test_extraction_cost_slices_exact_byte_range():
    pf = parse_file(str(PHASE1_FIXTURES / "utils.py"))
    symbol = next(s for s in pf.symbols if s.qualified_name == "format_currency")

    cost = symbol_extraction_cost(symbol, pf.source_bytes)
    expected_text = pf.source_bytes[symbol.byte_start : symbol.byte_end].decode("utf-8")

    assert cost == estimate_tokens(expected_text)
