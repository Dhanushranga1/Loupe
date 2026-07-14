"""E4's `conventions://summary` MCP Resource (docs/loupe-extensions.md E4).

Registered directly on `FastApiMCP`'s underlying `mcp.server.lowlevel.Server`
(`FastApiMCP.server`, a public attribute), not routed through
`mcp_tools.router` like every other tool in this codebase — a deliberate
choice, and the entire point of E4's design decision. MCP's Resource
primitive (URI-addressed, readable data) is what a periodic whole-repo
report is; MCP's Tool primitive (callable actions, what `include_operations`
governs) is not, and registering a resource costs *zero* against the
tool-count budget every other extension in this project has had to account
for. `FastApiMCP` itself has no resource support of its own — it only
converts FastAPI routes into tools — so this bypasses it entirely and talks
to the real MCP SDK `Server` object it already builds internally.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from mcp import types
from mcp.server.lowlevel import Server

from loupe_core.conventions.mining import mine_conventions

CONVENTIONS_SUMMARY_URI = "conventions://summary"


def conventions_summary_json(parsed_files) -> str:
    report = mine_conventions(list(parsed_files))
    return json.dumps(asdict(report), indent=2)


def register_conventions_resource(mcp_server: Server, get_parsed_files) -> None:
    """`get_parsed_files` is a zero-arg callable (not a snapshot) so a resource
    read always reflects the *current* index, including after incremental
    reindexing — the same "always current, never stale" bar `list_symbols`
    and friends already hold themselves to via `request.app.state.index`."""

    @mcp_server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=CONVENTIONS_SUMMARY_URI,
                name="conventions_summary",
                description="Auto-derived repo-wide coding conventions: error-handling, docstring style, import style.",
                mimeType="application/json",
            )
        ]

    @mcp_server.read_resource()
    async def read_resource(uri) -> str:
        if str(uri) != CONVENTIONS_SUMMARY_URI:
            raise ValueError(f"unknown resource: {uri}")
        return conventions_summary_json(get_parsed_files())
