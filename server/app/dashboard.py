"""Plain REST endpoints for the Lens dashboard (`lens/`, a separate React/Vite app).

Deliberately outside the MCP tool surface — same reasoning as E3's plain
`POST /feedback`: a human loading a browser tab never enters Claude's
context, so none of this costs anything against the MCP tool-count budget.
Nothing here is registered with `FastApiMCP`'s `include_operations`.

Read-only except for `/dashboard/feedback`'s underlying data, which is
written through the existing `POST /feedback` endpoint (`main.py`, E3) —
this module only ever reads `FeedbackStore`, never writes to it, so there's
exactly one write path for feedback, not two slightly-different ones.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone

from fastapi import APIRouter, Request
from pydantic import BaseModel

from loupe_core.conventions.mining import mine_conventions

from .bootstrap import LoupeIndex
from .feedback import FeedbackStore

router = APIRouter(prefix="/dashboard")


# --------------------------------------------------------------------------
# /dashboard/status
# --------------------------------------------------------------------------


class DashboardStatusResponse(BaseModel):
    repo_root: str
    symbol_count: int
    file_count: int
    unresolved_reference_count: int
    languages: list[str]
    last_indexed: str | None


@router.get("/status")
async def dashboard_status(request: Request) -> DashboardStatusResponse:
    index: LoupeIndex = request.app.state.index
    schema_path = index.loupe_dir / "schema_version"
    last_indexed = (
        datetime.fromtimestamp(schema_path.stat().st_mtime, tz=timezone.utc).isoformat() if schema_path.exists() else None
    )
    return DashboardStatusResponse(
        repo_root=str(index.repo_root),
        symbol_count=len(index.symbols),
        file_count=len(index.parsed_files),
        unresolved_reference_count=len(index.graph.unresolved),
        languages=sorted({s.language for s in index.symbols}) if index.symbols else [],
        last_indexed=last_indexed,
    )


# --------------------------------------------------------------------------
# /dashboard/graph
# --------------------------------------------------------------------------


class GraphNode(BaseModel):
    id: str
    name: str
    kind: str
    file_path: str
    module: str
    line_start: int
    line_end: int
    pagerank: float


class GraphEdge(BaseModel):
    source: str
    target: str
    type: str


class DashboardGraphResponse(BaseModel):
    nodes: list[GraphNode]
    edges: list[GraphEdge]


def _module_of(file_path: str) -> str:
    """First path segment, or the bare filename for a top-level file — a repo-agnostic
    grouping (unlike a hardcoded Loupe-specific mapping), good enough for a legend/filter."""
    parts = file_path.split("/")
    return parts[0] if len(parts) > 1 else file_path


@router.get("/graph")
async def dashboard_graph(request: Request) -> DashboardGraphResponse:
    index: LoupeIndex = request.app.state.index
    symbols_by_id = {s.id: s for s in index.symbols}

    nodes = [
        GraphNode(
            id=s.id,
            name=s.qualified_name,
            kind=s.kind.value,
            file_path=s.file_path,
            module=_module_of(s.file_path),
            line_start=s.line_start,
            line_end=s.line_end,
            pagerank=index.graph.pagerank_scores.get(s.id, 0.0),
        )
        for s in symbols_by_id.values()
    ]
    edges = [
        GraphEdge(source=u, target=v, type=data["edge_type"].value)
        for u, v, data in index.graph.graph.edges(data=True)
        if u in symbols_by_id and v in symbols_by_id
    ]
    return DashboardGraphResponse(nodes=nodes, edges=edges)


# --------------------------------------------------------------------------
# /dashboard/conventions
# --------------------------------------------------------------------------


@router.get("/conventions")
async def dashboard_conventions(request: Request) -> dict:
    # Deliberately re-uses the exact same core function E4's MCP Resource
    # calls (mine_conventions) rather than round-tripping through the MCP
    # protocol from the dashboard — a browser tab has no reason to speak
    # MCP JSON-RPC just to read a report a plain REST call can return.
    index: LoupeIndex = request.app.state.index
    report = mine_conventions(list(index.parsed_files.values()))
    return asdict(report)


# --------------------------------------------------------------------------
# /dashboard/telemetry
# --------------------------------------------------------------------------


@router.get("/telemetry")
async def dashboard_telemetry(request: Request, limit: int = 100) -> list[dict]:
    index: LoupeIndex = request.app.state.index
    logs_dir = index.loupe_dir / "logs" / "retrieval"
    entries: list[dict] = []
    if logs_dir.exists():
        for log_file in logs_dir.glob("*.jsonl"):
            with open(log_file) as f:
                for line in f:
                    entries.append(json.loads(line))
    entries.sort(key=lambda e: e["timestamp"], reverse=True)
    return entries[:limit]


# --------------------------------------------------------------------------
# /dashboard/feedback
# --------------------------------------------------------------------------


@router.get("/feedback")
async def dashboard_feedback(request: Request) -> list[dict]:
    store: FeedbackStore = request.app.state.feedback_store
    entries = list(store.all_by_log_id().values())
    entries.sort(key=lambda e: e.submitted_at, reverse=True)
    return [
        {
            "retrieval_log_id": e.retrieval_log_id,
            "rating": e.rating,
            "note": e.note,
            "submitted_at": e.submitted_at,
            "source": e.source,
        }
        for e in entries
    ]
