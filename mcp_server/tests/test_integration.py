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

from loupe_mcp_server.main import create_app

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
    assert tool_names == {
        "list_symbols",
        "search_symbols",
        "get_symbol",
        "expand_dependencies",
        "analyze_impact",
        "submit_feedback",
        "find_code_smells",
        "session_notes",
    }


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
    assert {s["qualified_name"] for s in result["results"]} == {"helper_b"}
    assert result["total_count"] == 1


def test_mcp_find_code_smells_detects_the_real_circular_fixture_through_http(client):
    """The same circular_a/circular_b fixture expand_dependencies already exercises above
    is a real, unrelated-to-Phase-7 instance of the circular_dependency smell — a genuine
    end-to-end check, not a purpose-built fixture reused twice for the same claim."""
    session_id = _mcp_initialize(client)
    result = _mcp_tool_call(
        client, session_id, "find_code_smells", {"category": "circular_dependency"}, request_id=2
    )
    names = {f["qualified_name"] for f in result["findings"]}
    assert {"helper_a", "helper_b"} <= names


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
    import loupe_core.governor.budget as budget_module
    import loupe_mcp_server.mcp_tools as mcp_tools_module
    from loupe_core.governor.session import HARD_CEILING

    # Phase 14 §4: the real charge goes through symbol_extraction_marginal_cost
    # (governor/budget.py), which calls that module's own symbol_extraction_cost
    # internally — patching mcp_tools.py's imported reference alone (used only
    # for the L5 decomposition-threshold check) no longer affects the real
    # charge, so both call sites are patched here.
    monkeypatch.setattr(mcp_tools_module, "symbol_extraction_cost", lambda s, b: HARD_CEILING + 1)
    monkeypatch.setattr(budget_module, "symbol_extraction_cost", lambda s, b: HARD_CEILING + 1)

    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    symbol_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")

    result = _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": symbol_id}, request_id=3)
    assert result["status"] == "denied"
    assert result["reason"] == "exceeds_hard_ceiling"


def test_tiny_budget_session_evicts_first_symbol_for_a_second_through_http(client, monkeypatch):
    import loupe_core.governor.budget as budget_module
    import loupe_mcp_server.mcp_tools as mcp_tools_module

    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    format_currency_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")
    validate_email_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "validate_email")

    # Force a tiny effective budget for this pair: each symbol costs slightly
    # more than half the default 6000-token budget, so the second forces the
    # first out rather than both fitting side by side.
    costs = {format_currency_id: 3500, validate_email_id: 3500}
    # See the hard-ceiling test above for why both references are patched.
    monkeypatch.setattr(mcp_tools_module, "symbol_extraction_cost", lambda s, b: costs.get(s.id, 100))
    monkeypatch.setattr(budget_module, "symbol_extraction_cost", lambda s, b: costs.get(s.id, 100))

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
    _mcp_tool_call(client, session_id, "analyze_impact", {"symbol_id": symbol_id, "depth": 2}, request_id=6)
    _mcp_tool_call(
        client, session_id, "submit_feedback", {"retrieval_log_id": "log-x", "rating": "helpful"}, request_id=7
    )
    _mcp_tool_call(client, session_id, "find_code_smells", {}, request_id=8)
    _mcp_tool_call(client, session_id, "session_notes", {"action": "write", "content": "a note", "importance": 3}, request_id=9)

    log_path = mock_repo / ".loupe" / "logs" / "retrieval" / f"{session_id}.jsonl"
    assert log_path.exists()
    lines = log_path.read_text().strip().splitlines()
    assert len(lines) == 8

    tool_names = set()
    for line in lines:
        entry = json.loads(line)
        assert entry["outcome"] is None
        assert entry["latency_ms"] >= 0
        assert entry["output_size_bytes"] >= 0
        assert entry["session_id"] == session_id
        tool_names.add(entry["tool_name"])
    assert tool_names == {
        "list_symbols",
        "search_symbols",
        "get_symbol",
        "expand_dependencies",
        "analyze_impact",
        "submit_feedback",
        "find_code_smells",
        "session_notes",
    }


def test_search_symbols_telemetry_logs_cross_encoder_latency_other_tools_dont(client, mock_repo):
    """Phase 9 §3's own acceptance criterion: "measure, don't assume" cross-encoder
    latency — checked here as a real, populated telemetry field on the one tool
    that runs reranking, and confirmed absent (null) everywhere else, not just
    present in the dataclass but silently unpopulated.
    """
    session_id = _mcp_initialize(client)
    list_result = _mcp_tool_call(client, session_id, "list_symbols", {"path_or_glob": "utils.py"}, request_id=2)
    symbol_id = next(s["symbol_id"] for s in list_result if s["qualified_name"] == "format_currency")
    _mcp_tool_call(client, session_id, "search_symbols", {"query": "format currency"}, request_id=3)
    _mcp_tool_call(client, session_id, "get_symbol", {"symbol_id": symbol_id}, request_id=4)

    log_path = mock_repo / ".loupe" / "logs" / "retrieval" / f"{session_id}.jsonl"
    entries = [json.loads(line) for line in log_path.read_text().strip().splitlines()]

    by_tool = {entry["tool_name"]: entry for entry in entries}
    assert by_tool["search_symbols"]["cross_encoder_latency_ms"] is not None
    assert by_tool["search_symbols"]["cross_encoder_latency_ms"] >= 0.0
    assert by_tool["list_symbols"]["cross_encoder_latency_ms"] is None
    assert by_tool["get_symbol"]["cross_encoder_latency_ms"] is None


# --------------------------------------------------------------------------
# E3: dashboard feedback — plain HTTP, deliberately not an MCP tool call
# --------------------------------------------------------------------------


def test_dashboard_feedback_is_a_plain_http_endpoint_not_an_mcp_tool(client, mock_repo):
    """The dashboard-equivalent API call the E3 acceptance criteria describe: a plain
    POST, no MCP session/protocol envelope at all — and it must not appear in tools/list."""
    session_id = _mcp_initialize(client)
    tools_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tool_names = {t["name"] for t in tools_response.json()["result"]["tools"]}
    assert "submit_dashboard_feedback" not in tool_names

    response = client.post(
        "/feedback", json={"retrieval_log_id": "log-abc", "rating": "helpful", "note": "found the right symbol"}
    )
    assert response.status_code == 200
    assert response.json() == {"status": "recorded"}

    feedback_path = mock_repo / ".loupe" / "logs" / "feedback" / "feedback.jsonl"
    assert feedback_path.exists()
    entry = json.loads(feedback_path.read_text().strip().splitlines()[-1])
    assert entry["retrieval_log_id"] == "log-abc"
    assert entry["rating"] == "helpful"
    assert entry["source"] == "dashboard"


# --------------------------------------------------------------------------
# E4: conventions://summary — a real MCP Resource, not a Tool
# --------------------------------------------------------------------------


def test_conventions_summary_is_a_resource_not_a_tool(client):
    session_id = _mcp_initialize(client)

    tools_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tool_names = {t["name"] for t in tools_response.json()["result"]["tools"]}
    assert "conventions_summary" not in tool_names

    resources_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
    )
    resources = resources_response.json()["result"]["resources"]
    assert any(r["uri"] == "conventions://summary" for r in resources)

    read_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={
            "jsonrpc": "2.0",
            "id": 4,
            "method": "resources/read",
            "params": {"uri": "conventions://summary"},
        },
    )
    contents = read_response.json()["result"]["contents"]
    assert len(contents) == 1
    report = json.loads(contents[0]["text"])
    assert set(report.keys()) == {"error_handling", "docstrings", "imports"}
    assert set(report["docstrings"].keys()) >= {"coverage_pct", "dominant_style"}
    assert set(report["imports"].keys()) >= {"dominant_style"}


def test_architecture_overview_is_a_resource_not_a_tool(client):
    """Phase 14 §1's L0/L1 LOD levels — a second, independent resource
    alongside conventions://summary, proving register_resources' single
    list_resources/read_resource handler pair genuinely serves both
    (a second, naively-independent registration would have silently
    overwritten conventions://summary's handler instead)."""
    session_id = _mcp_initialize(client)

    tools_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tool_names = {t["name"] for t in tools_response.json()["result"]["tools"]}
    assert "architecture_overview" not in tool_names

    resources_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 3, "method": "resources/list", "params": {}},
    )
    resource_uris = {r["uri"] for r in resources_response.json()["result"]["resources"]}
    assert resource_uris == {"conventions://summary", "architecture://overview", "static-analysis://summary"}

    read_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 4, "method": "resources/read", "params": {"uri": "architecture://overview"}},
    )
    contents = read_response.json()["result"]["contents"]
    assert len(contents) == 1
    overview = json.loads(contents[0]["text"])
    assert "repo_summary" in overview
    assert "clusters" in overview
    assert isinstance(overview["clusters"], list)


def test_static_analysis_summary_is_a_resource_not_a_tool(client):
    """The zero-cost static analysis pack (E5-E9, docs/PhaseX/zero-cost-static-analysis-pack.md)."""
    session_id = _mcp_initialize(client)

    tools_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tool_names = {t["name"] for t in tools_response.json()["result"]["tools"]}
    assert "static_analysis_summary" not in tool_names

    read_response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 3, "method": "resources/read", "params": {"uri": "static-analysis://summary"}},
    )
    contents = read_response.json()["result"]["contents"]
    assert len(contents) == 1
    summary = json.loads(contents[0]["text"])

    assert "total_count" in summary["dead_code"]
    assert "total_count" in summary["duplicates"]
    assert summary["config_drift"] is None  # mock_repo has no .env.example
    assert summary["migration_drift"] is None  # mock_repo has no alembic/versions
    assert "loupe check --since" in summary["api_contract_diff"]


# --------------------------------------------------------------------------
# Lens dashboard: plain REST endpoints, not MCP tools
# --------------------------------------------------------------------------


def test_dashboard_endpoints_are_not_mcp_tools(client):
    session_id = _mcp_initialize(client)
    response = client.post(
        "/mcp",
        headers={"Accept": "application/json, text/event-stream", "Mcp-Session-Id": session_id},
        json={"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}},
    )
    tool_names = {t["name"] for t in response.json()["result"]["tools"]}
    assert tool_names.isdisjoint({"dashboard_status", "dashboard_graph", "dashboard_conventions", "dashboard_telemetry", "dashboard_feedback"})


def test_dashboard_status_reports_real_index_state(client):
    response = client.get("/dashboard/status")
    assert response.status_code == 200
    body = response.json()
    assert body["symbol_count"] == 16
    assert body["file_count"] == 6
    assert body["languages"] == ["python"]


def test_dashboard_graph_returns_real_nodes_and_edges(client):
    response = client.get("/dashboard/graph")
    assert response.status_code == 200
    body = response.json()
    names = {n["name"] for n in body["nodes"]}
    assert "format_currency" in names
    assert len(body["edges"]) > 0
    assert all(set(e.keys()) == {"source", "target", "type"} for e in body["edges"])


def test_dashboard_conventions_returns_the_same_shape_as_the_mcp_resource(client):
    response = client.get("/dashboard/conventions")
    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {"error_handling", "docstrings", "imports"}


def test_dashboard_telemetry_reflects_real_tool_calls(client):
    session_id = _mcp_initialize(client)
    _mcp_tool_call(client, session_id, "search_symbols", {"query": "format currency"}, request_id=2)

    response = client.get("/dashboard/telemetry")
    assert response.status_code == 200
    entries = response.json()
    assert any(e["tool_name"] == "search_symbols" and e["session_id"] == session_id for e in entries)


def test_dashboard_feedback_reflects_submissions_via_post_feedback(client):
    client.post("/feedback", json={"retrieval_log_id": "dash-log-1", "rating": "helpful"})

    response = client.get("/dashboard/feedback")
    assert response.status_code == 200
    entries = response.json()
    assert any(e["retrieval_log_id"] == "dash-log-1" and e["rating"] == "helpful" for e in entries)


def test_dashboard_endpoints_allow_the_vite_dev_origin_via_cors(client):
    response = client.get("/dashboard/status", headers={"Origin": "http://localhost:5173"})
    assert response.headers.get("access-control-allow-origin") == "http://localhost:5173"


# --------------------------------------------------------------------------
# session_notes (docs/PhaseX/session-notes.md) — one tool, four actions
# --------------------------------------------------------------------------


def test_session_notes_write_then_read_recent_through_real_mcp_protocol(client):
    session_id = _mcp_initialize(client)

    write_result = _mcp_tool_call(
        client, session_id, "session_notes", {"action": "write", "content": "the bug is in token refresh", "importance": 5}
    )
    assert len(write_result["notes"]) == 1
    assert write_result["notes"][0]["content"] == "the bug is in token refresh"

    recent = _mcp_tool_call(client, session_id, "session_notes", {"action": "read_recent"})
    contents = [n["content"] for n in recent["notes"]]
    assert "the bug is in token refresh" in contents


def test_session_notes_list_returns_every_note_ever_written(client):
    session_id = _mcp_initialize(client)
    _mcp_tool_call(client, session_id, "session_notes", {"action": "write", "content": "note one", "importance": 2})
    _mcp_tool_call(client, session_id, "session_notes", {"action": "write", "content": "note two", "importance": 2})

    listed = _mcp_tool_call(client, session_id, "session_notes", {"action": "list"})
    assert {n["content"] for n in listed["notes"]} == {"note one", "note two"}


def test_session_notes_read_relevant_surfaces_the_matching_note(client):
    session_id = _mcp_initialize(client)
    _mcp_tool_call(
        client, session_id, "session_notes", {"action": "write", "content": "the retry decorator has an off-by-one bug", "importance": 5}
    )
    _mcp_tool_call(client, session_id, "session_notes", {"action": "write", "content": "checked the login page styling", "importance": 1})

    result = _mcp_tool_call(client, session_id, "session_notes", {"action": "read_relevant", "query": "bug in retry logic", "top_k": 1})
    assert "retry" in result["notes"][0]["content"]


def test_session_notes_across_two_sessions_never_cross_contaminate(client):
    session_a = _mcp_initialize(client)
    session_b = _mcp_initialize(client)

    _mcp_tool_call(client, session_a, "session_notes", {"action": "write", "content": "session A's note", "importance": 3})
    _mcp_tool_call(client, session_b, "session_notes", {"action": "write", "content": "session B's note", "importance": 3})

    a_notes = _mcp_tool_call(client, session_a, "session_notes", {"action": "list"})
    b_notes = _mcp_tool_call(client, session_b, "session_notes", {"action": "list"})

    assert {n["content"] for n in a_notes["notes"]} == {"session A's note"}
    assert {n["content"] for n in b_notes["notes"]} == {"session B's note"}
