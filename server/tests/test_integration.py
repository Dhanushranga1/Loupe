"""Full HTTP integration tests against a real running server (docs/phase-4-systems.md §8/§10 task 9).

Uses `TestClient` (real ASGI + lifespan startup, real HTTP request parsing)
rather than calling Python functions directly — wiring bugs (decorator
stacking, FastAPI signature inspection, MCP session-header propagation) are
exactly what unit tests of the underlying phases wouldn't catch.
"""

import json
import shutil
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import create_app

PHASE1_FIXTURES = Path(__file__).parent.parent.parent / "core" / "tests" / "fixtures" / "phase1"
PHASE1_FILES = ["utils.py", "models.py", "services.py", "handlers.py", "circular_a.py", "circular_b.py"]


@pytest.fixture(scope="module")
def mock_repo(tmp_path_factory):
    repo = tmp_path_factory.mktemp("integration_repo")
    for f in PHASE1_FILES:
        shutil.copy(PHASE1_FIXTURES / f, repo / f)
    return repo


@pytest.fixture(scope="module")
def client(mock_repo):
    app = create_app(repo_root=mock_repo)
    with TestClient(app) as c:
        yield c


def _mcp_initialize(client: TestClient) -> str:
    response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream"},
        json={
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "test", "version": "1.0"}},
        },
    )
    assert response.status_code == 200
    session_id = response.headers["mcp-session-id"]
    client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "method": "notifications/initialized"},
    )
    return session_id


def _mcp_tool_call(client: TestClient, session_id: str, tool_name: str, arguments: dict, request_id: int = 2) -> dict:
    response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": request_id, "method": "tools/call", "params": {"name": tool_name, "arguments": arguments}},
    )
    assert response.status_code == 200
    body = response.json()
    assert "error" not in body, body
    text = body["result"]["content"][0]["text"]
    return json.loads(text)


# --------------------------------------------------------------------------
# Plain HTTP path (proves the decorator stacking + FastAPI signature
# inspection actually works, independent of the MCP protocol layer)
# --------------------------------------------------------------------------


def test_list_symbols_plain_http_route(client):
    response = client.get("/list_symbols", params={"path_or_glob": "utils.py"})
    assert response.status_code == 200
    names = {s["qualified_name"] for s in response.json()}
    assert names == {"format_currency", "validate_email"}


def test_list_symbols_two_identical_calls_are_byte_identical(client):
    r1 = client.get("/list_symbols", params={"path_or_glob": "*.py"})
    r2 = client.get("/list_symbols", params={"path_or_glob": "*.py"})
    assert r1.content == r2.content


def test_version_endpoint_exposes_both_schema_versions(client):
    response = client.get("/loupe/version")
    assert response.status_code == 200
    body = response.json()
    assert "index_schema_version" in body
    assert "mcp_tool_schema_version" in body


# --------------------------------------------------------------------------
# Real MCP protocol path
# --------------------------------------------------------------------------


def test_mcp_handshake_and_tools_list(client):
    session_id = _mcp_initialize(client)
    response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tool_names = {t["name"] for t in response.json()["result"]["tools"]}
    assert tool_names == {"list_symbols", "search_symbols", "get_symbol", "expand_dependencies"}


def test_mcp_search_symbols_tool_call(client):
    session_id = _mcp_initialize(client)
    result = _mcp_tool_call(client, session_id, "search_symbols", {"query": "validate an email address", "top_k": 3})
    assert result[0]["qualified_name"] == "validate_email"


def test_mcp_get_symbol_tracks_residency_across_calls_with_the_same_session(client):
    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    symbol_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")

    first = _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": symbol_id}, request_id=3)
    assert first["already_resident"] is False

    second = _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": symbol_id}, request_id=4)
    assert second["already_resident"] is True


def test_mcp_expand_dependencies_on_circular_fixture_terminates_through_http(client):
    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "circular_a.py"}, request_id=2)
    helper_a_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "helper_a")

    result = _mcp_tool_call(
        client, session_id, "expand_dependencies", {"symbol_id": helper_a_id, "depth": 5, "direction": "both"}, request_id=3
    )
    assert {s["qualified_name"] for s in result} == {"helper_b"}


def test_two_mcp_sessions_have_independent_governor_state(client):
    session_a = _mcp_initialize(client)
    session_b = _mcp_initialize(client)

    list_result = _mcp_tool_call(client, session_a, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    symbol_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")

    _mcp_tool_call(client, session_a, "get_symbol", {"symbol_id": symbol_id}, request_id=3)

    # session B has never requested this symbol — must not see it as resident
    result_b = _mcp_tool_call(client, session_b, "get_symbol", {"symbol_id": symbol_id}, request_id=3)
    assert result_b["already_resident"] is False


# --------------------------------------------------------------------------
# Governor edge cases, through the real HTTP/MCP path
# --------------------------------------------------------------------------


def test_get_symbol_exceeding_hard_ceiling_is_denied_through_http(client, monkeypatch):
    import app.mcp_tools as mcp_tools_module
    from loupe_core.governor.session import HARD_CEILING

    monkeypatch.setattr(mcp_tools_module, "symbol_extraction_cost", lambda s, b: HARD_CEILING + 1)

    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    symbol_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")

    result = _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": symbol_id}, request_id=3)
    assert result["status"] == "denied"
    assert result["reason"] == "exceeds_hard_ceiling"


def test_tiny_budget_session_evicts_first_symbol_for_a_second_through_http(client, monkeypatch):
    import app.mcp_tools as mcp_tools_module

    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    format_currency_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")
    validate_email_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "validate_email")

    # Force a tiny effective budget for this pair: each symbol costs slightly
    # more than half the default 6000-token budget, so the second forces the
    # first out rather than both fitting side by side.
    costs = {format_currency_id: 3500, validate_email_id: 3500}
    monkeypatch.setattr(mcp_tools_module, "symbol_extraction_cost", lambda s, b: costs.get(s.id, 100))

    first = _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": format_currency_id}, request_id=3)
    assert first["already_resident"] is False

    second = _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": validate_email_id}, request_id=4)
    assert second["already_resident"] is False

    # format_currency should have been evicted to make room; requesting it
    # again must be billed as new, not free.
    third = _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": format_currency_id}, request_id=5)
    assert third["already_resident"] is False, "the evicted symbol must be billed again, not treated as still resident"


# --------------------------------------------------------------------------
# Telemetry: every tool call produces exactly one appended JSONL line
# --------------------------------------------------------------------------


def test_every_tool_call_produces_one_telemetry_line(client, mock_repo):
    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    symbol_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")
    _mcp_tool_call(client, session_id, "search_symbols", {"query": "format currency"}, request_id=3)
    _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": symbol_id}, request_id=4)
    _mcp_tool_call(client, session_id, "expand_dependencies", {"symbol_id": symbol_id, "depth": 1}, request_id=5)

    log_path = mock_repo / ".loupe" / "logs" / "retrieval" / f"{session_id}.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 4

    tool_names = set()
    for line in lines:
        entry = json.loads(line)
        assert entry["outcome"] is None
        assert entry["latency_ms"] >= 0
        assert entry["output_size_bytes"] >= 0
        assert entry["session_id"] == session_id
        tool_names.add(entry["tool_name"])
    assert tool_names == {"list_symbols", "search_symbols", "get_symbol", "expand_dependencies"}
