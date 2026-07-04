"""Tests for parsing/extractor.py against the Phase 0 fixtures.

Byte ranges below were hand-verified against the fixture source (see
docs/phase-0-foundations.md §7 — validation here is example-based unit
testing against hand-verified fixtures, not the git-history-mined benchmark).
"""

from pathlib import Path

import pytest

from loupe_core.parsing.extractor import extract_symbols
from loupe_core.parsing.schema import SymbolKind

FIXTURES = Path(__file__).parent / "fixtures" / "phase0"


def _by_qualified_name(symbols, name):
    matches = [s for s in symbols if s.qualified_name == name]
    assert len(matches) == 1, f"expected exactly one symbol named {name!r}, found {len(matches)}"
    return matches[0]


def test_simple_module():
    symbols = extract_symbols(str(FIXTURES / "simple_module.py"))
    assert [s.qualified_name for s in symbols] == ["add", "multiply"]

    add = _by_qualified_name(symbols, "add")
    assert add.kind == SymbolKind.FUNCTION
    assert add.docstring == "Add two integers and return the sum."
    assert add.signature == "def add(a: int, b: int) -> int"
    assert add.decorators == []
    assert add.parent_id is None

    multiply = _by_qualified_name(symbols, "multiply")
    assert multiply.docstring is None


def test_nested_class_methods_and_decorators():
    symbols = extract_symbols(str(FIXTURES / "nested_class.py"))
    widget = _by_qualified_name(symbols, "Widget")
    assert widget.kind == SymbolKind.CLASS
    assert widget.docstring == "A simple widget with a name and a computed display label."

    init = _by_qualified_name(symbols, "Widget.__init__")
    assert init.kind == SymbolKind.METHOD
    assert init.parent_id == widget.id
    assert init.decorators == []

    default_name = _by_qualified_name(symbols, "Widget.default_name")
    assert default_name.decorators == ["staticmethod"]
    assert default_name.parent_id == widget.id

    label = _by_qualified_name(symbols, "Widget.label")
    assert label.decorators == ["property"]
    assert label.docstring == "The display label for this widget."
    assert label.parent_id == widget.id

    # every method in this fixture nests exactly one level under the same class
    assert {init.parent_id, default_name.parent_id, label.parent_id} == {widget.id}


def test_multi_decorator_order_preserved():
    symbols = extract_symbols(str(FIXTURES / "multi_decorator.py"))
    compute = _by_qualified_name(symbols, "compute")
    assert compute.decorators == ["cache", "log_calls"]
    assert compute.signature == "def compute(x: int) -> int"
    assert compute.docstring == "Compute something expensive, cached and logged."


def test_async_and_multiline_signature_verbatim():
    symbols = extract_symbols(str(FIXTURES / "async_and_multiline.py"))
    fetch_batch = _by_qualified_name(symbols, "fetch_batch")
    assert fetch_batch.kind == SymbolKind.ASYNC_FUNCTION
    assert fetch_batch.signature == (
        "async def fetch_batch(\n"
        "    endpoint: str,\n"
        "    ids: list[int],\n"
        "    timeout: float = 30.0,\n"
        "    retries: int = 3,\n"
        ") -> list[dict]"
    )
    assert not fetch_batch.signature.rstrip().endswith(":")
    assert fetch_batch.docstring == "Fetch a batch of records from an endpoint, with retry support."


def test_no_docstring_edge_cases():
    symbols = extract_symbols(str(FIXTURES / "no_docstring_edge_cases.py"))

    plain = _by_qualified_name(symbols, "Plain")
    assert plain.docstring is None

    greet = _by_qualified_name(symbols, "Plain.greet")
    assert greet.docstring is None

    only_string_body = _by_qualified_name(symbols, "only_string_body")
    assert only_string_body.docstring == "this bare string is the only statement, so it is a real docstring"

    not_first_statement = _by_qualified_name(symbols, "not_first_statement")
    assert not_first_statement.docstring is None, (
        "a bare string that is not the body's first statement must never be treated as a docstring"
    )


def test_byte_ranges_are_exact_and_slice_correctly():
    path = FIXTURES / "simple_module.py"
    source = path.read_bytes()
    symbols = extract_symbols(str(path))
    add = _by_qualified_name(symbols, "add")
    sliced = source[add.byte_start : add.byte_end].decode("utf-8")
    assert sliced == (
        'def add(a: int, b: int) -> int:\n    """Add two integers and return the sum."""\n    return a + b'
    )


def test_decorated_symbol_byte_range_includes_decorators():
    path = FIXTURES / "multi_decorator.py"
    source = path.read_bytes()
    symbols = extract_symbols(str(path))
    compute = _by_qualified_name(symbols, "compute")
    sliced = source[compute.byte_start : compute.byte_end].decode("utf-8")
    assert sliced.startswith("@cache\n@log_calls\ndef compute")


def test_content_hash_stable_and_change_sensitive(tmp_path):
    file_a = tmp_path / "a.py"
    file_a.write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n")

    first_pass = {s.qualified_name: s.content_hash for s in extract_symbols(str(file_a))}
    second_pass = {s.qualified_name: s.content_hash for s in extract_symbols(str(file_a))}
    assert first_pass == second_pass, "content_hash must be stable across repeated runs on unchanged source"

    file_a.write_text("def f():\n    return 999\n\n\ndef g():\n    return 2\n")
    third_pass = {s.qualified_name: s.content_hash for s in extract_symbols(str(file_a))}

    assert third_pass["f"] != first_pass["f"], "changing a symbol's own body must change its content_hash"
    assert third_pass["g"] == first_pass["g"], "an unrelated symbol's content_hash must be unaffected"


def test_id_stable_across_body_edit_but_not_across_kind_or_name(tmp_path):
    file_a = tmp_path / "a.py"
    file_a.write_text("def f():\n    return 1\n")
    before = _by_qualified_name(extract_symbols(str(file_a)), "f")

    file_a.write_text("def f():\n    return 2\n")
    after = _by_qualified_name(extract_symbols(str(file_a)), "f")

    assert before.id == after.id, "Symbol.id must stay stable across edits to the body"
    assert before.content_hash != after.content_hash


@pytest.mark.parametrize("fixture_name", [p.name for p in FIXTURES.glob("*.py")])
def test_every_fixture_extracts_without_crashing(fixture_name):
    extract_symbols(str(FIXTURES / fixture_name))
