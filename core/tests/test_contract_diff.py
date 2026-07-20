"""Tests for adapters/fastapi/contract_diff.py (docs/PhaseX/zero-cost-static-analysis-pack.md E9)."""

import os
from pathlib import Path

import pytest

from loupe_core.adapters.fastapi.contract_diff import (
    DEFAULT_STATUS_CODE,
    diff_contracts,
    extract_route_contracts,
)
from loupe_core.graph.builder import parse_file


def _parse(tmp_path: Path, source: str):
    f = tmp_path / "routes.py"
    f.write_text(source)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return [parse_file("routes.py")]
    finally:
        os.chdir(old_cwd)


BEFORE_DIRECT = """
class ItemOut:
    id: int
    name: str


@app.get('/items/{id}', response_model=ItemOut)
def get_item(id: int):
    ...
"""


def test_extracts_response_model_and_required_fields(tmp_path):
    parsed = _parse(tmp_path, BEFORE_DIRECT)
    contracts = extract_route_contracts(parsed)

    contract = contracts["get_item"]
    assert contract.method == "get"
    assert contract.response_model_name == "ItemOut"
    assert contract.required_fields == {"id", "name"}
    assert contract.status_code == DEFAULT_STATUS_CODE


def test_required_field_removed_from_response_model_is_flagged_breaking(tmp_path):
    """§6's own acceptance criterion: a required field removed from a
    response model between two commits is flagged as a breaking change."""
    before = _parse(tmp_path, BEFORE_DIRECT)
    old_contracts = extract_route_contracts(before)

    after_source = BEFORE_DIRECT.replace("    name: str\n", "")  # name field removed
    after = _parse(tmp_path, after_source)
    new_contracts = extract_route_contracts(after)

    changes = diff_contracts(old_contracts, new_contracts)

    assert any(c.qualified_name == "get_item" and "name" in c.description for c in changes)


def test_purely_additive_optional_field_is_not_flagged_breaking(tmp_path):
    """§6's own acceptance criterion: a purely additive change (a new
    optional field) is correctly not flagged as breaking."""
    before = _parse(tmp_path, BEFORE_DIRECT)
    old_contracts = extract_route_contracts(before)

    after_source = BEFORE_DIRECT.replace(
        "    name: str\n", "    name: str\n    description: str = None\n"
    )
    after = _parse(tmp_path, after_source)
    new_contracts = extract_route_contracts(after)

    changes = diff_contracts(old_contracts, new_contracts)

    assert changes == []


def test_status_code_change_is_flagged(tmp_path):
    before = _parse(tmp_path, BEFORE_DIRECT)
    old_contracts = extract_route_contracts(before)

    after_source = BEFORE_DIRECT.replace(
        "@app.get('/items/{id}', response_model=ItemOut)",
        "@app.get('/items/{id}', response_model=ItemOut, status_code=201)",
    )
    after = _parse(tmp_path, after_source)
    new_contracts = extract_route_contracts(after)

    changes = diff_contracts(old_contracts, new_contracts)

    assert any(c.qualified_name == "get_item" and "status code changed: 200 -> 201" in c.description for c in changes)


def test_route_removed_entirely_is_flagged(tmp_path):
    before = _parse(tmp_path, BEFORE_DIRECT)
    old_contracts = extract_route_contracts(before)

    after = _parse(tmp_path, "class ItemOut:\n    id: int\n    name: str\n")  # route deleted
    new_contracts = extract_route_contracts(after)

    changes = diff_contracts(old_contracts, new_contracts)

    assert any(c.qualified_name == "get_item" and c.description == "route removed" for c in changes)


def test_new_route_added_is_not_flagged_breaking(tmp_path):
    before = _parse(tmp_path, BEFORE_DIRECT)
    old_contracts = extract_route_contracts(before)

    after_source = BEFORE_DIRECT + "\n\n@app.post('/items', response_model=ItemOut)\ndef create_item():\n    ...\n"
    after = _parse(tmp_path, after_source)
    new_contracts = extract_route_contracts(after)

    changes = diff_contracts(old_contracts, new_contracts)

    assert changes == []


def test_non_route_functions_are_not_extracted(tmp_path):
    parsed = _parse(tmp_path, BEFORE_DIRECT + "\n\ndef helper():\n    return 1\n")
    contracts = extract_route_contracts(parsed)
    assert "helper" not in contracts
