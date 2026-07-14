"""FastAPI application entrypoint (docs/phase-4-systems.md §7).

Bootstraps the index on startup, mounts the four MCP tools over HTTP via
`fastapi-mcp`, and runs the session TTL sweep as a background task. Transport
is HTTP, not stdio — a real, inspectable FastAPI process (§2's stated
decision), default port 8765.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi_mcp import FastApiMCP

from . import mcp_tools
from .bootstrap import bootstrap
from .config import DEFAULT_PORT, INDEX_SCHEMA_VERSION, MCP_TOOL_SCHEMA_VERSION, load_config
from .conventions import register_conventions_resource
from .feedback import FeedbackRequest, FeedbackStore
from .indexer_worker import IndexerWorker
from .session_manager import SessionManager
from .telemetry import TelemetryWriter

TTL_SWEEP_INTERVAL_SECONDS = 60  # once per minute (§5)


def create_app(repo_root: Path | None = None) -> FastAPI:
    resolved_repo_root = (repo_root or Path.cwd()).resolve()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # bootstrap()/extract_symbols() resolve relative paths against CWD —
        # see bootstrap.py's module docstring for why this is required.
        os.chdir(resolved_repo_root)
        config = load_config(resolved_repo_root)
        app.state.repo_root = resolved_repo_root
        app.state.config = config
        app.state.index = bootstrap(resolved_repo_root, config)
        app.state.session_manager = SessionManager()
        app.state.telemetry = TelemetryWriter(app.state.index.loupe_dir / "logs" / "retrieval")
        app.state.feedback_store = FeedbackStore(app.state.index.loupe_dir / "logs" / "feedback")

        indexer_worker = IndexerWorker(app, resolved_repo_root)
        indexer_worker.start()
        app.state.indexer_worker = indexer_worker

        sweep_task = asyncio.create_task(_ttl_sweep_loop(app))
        try:
            yield
        finally:
            sweep_task.cancel()
            try:
                await sweep_task
            except asyncio.CancelledError:
                pass
            await indexer_worker.stop()

    app = FastAPI(title="Loupe", description="AST-aware context orchestration for Claude", lifespan=lifespan)
    app.include_router(mcp_tools.router)

    @app.get("/loupe/version", operation_id="loupe_version")
    async def loupe_version() -> dict[str, int]:
        # Addendum item (c): index schema version (.loupe/'s on-disk format)
        # and the MCP tool-contract version are deliberately distinct and
        # independently bumped. Exposed as a plain endpoint rather than
        # overloading fastapi-mcp's own `initialize` handshake response,
        # whose internals belong to a third-party library we don't own.
        return {"index_schema_version": INDEX_SCHEMA_VERSION, "mcp_tool_schema_version": MCP_TOOL_SCHEMA_VERSION}

    @app.post("/feedback", operation_id="submit_dashboard_feedback")
    async def submit_dashboard_feedback(feedback: FeedbackRequest, request: Request) -> dict[str, str]:
        # E3's primary path (docs/loupe-extensions.md): a human clicking a
        # button in the Lens dashboard hits this plain HTTP endpoint
        # directly — deliberately not an MCP tool, so it costs nothing
        # against the tool-count budget (mcp_tools.py's submit_feedback is
        # the separate, optional, MCP-visible path for Claude itself).
        store: FeedbackStore = request.app.state.feedback_store
        store.submit(feedback.retrieval_log_id, feedback.rating, feedback.note, source="dashboard")
        return {"status": "recorded"}

    mcp = FastApiMCP(
        app,
        name="loupe",
        description="AST-aware context orchestration for Claude — surgical symbol retrieval over a codebase",
        headers=["mcp-session-id"],
        # Only the documented tools should ever reach Claude as callable MCP
        # tools (addendum's explicit tool-count ceiling) — plain HTTP
        # introspection/write endpoints like /loupe/version and POST
        # /feedback must not silently become a tool just by living in the
        # same FastAPI app. E1 adds analyze_impact (5th) and E3 adds the
        # optional submit_feedback (6th) — both accounted for by
        # docs/loupe-extensions.md's own running tool-budget check.
        include_operations=[
            "list_symbols",
            "search_symbols",
            "get_symbol",
            "expand_dependencies",
            "analyze_impact",
            "submit_feedback",
        ],
    )
    # E4 (docs/loupe-extensions.md): registered directly on the MCP SDK
    # `Server` FastApiMCP builds internally, not on `mcp_tools.router` — see
    # conventions.py's module docstring for why this is a Resource, not a
    # route that would need an include_operations entry.
    register_conventions_resource(mcp.server, lambda: app.state.index.parsed_files.values())

    mcp.mount_http()

    return app


async def _ttl_sweep_loop(app: FastAPI) -> None:
    while True:
        await asyncio.sleep(TTL_SWEEP_INTERVAL_SECONDS)
        app.state.session_manager.sweep_expired()


app = create_app()
