"""MCP Resources — all URI-addressed, periodically-read whole-repo data in one
place: `conventions://summary` (E4, docs/loupe-extensions.md),
`architecture://overview` (Phase 14 §1, docs/PhaseX/phase-14-adaptive-context-compression.md),
and `static-analysis://summary` (E5-E9, docs/PhaseX/zero-cost-static-analysis-pack.md).

Registered directly on `FastApiMCP`'s underlying `mcp.server.lowlevel.Server`
(`FastApiMCP.server`, a public attribute), not routed through
`mcp_tools.router` like every tool in this codebase — MCP's Resource
primitive (URI-addressed, readable data) is what a periodic whole-repo
report is; MCP's Tool primitive (callable actions, what `include_operations`
governs) is not, and a resource costs *zero* against the tool-count budget.
`FastApiMCP` itself has no resource support of its own — it only converts
FastAPI routes into tools — so this bypasses it entirely.

Both resources are registered by ONE function, not two independent
`register_*_resource` calls: the low-level MCP SDK's `list_resources`/
`read_resource` decorators are singleton per-server (`Server.request_handlers`
is a plain dict keyed by request type — a second registration silently
*overwrites* the first, verified against the SDK's own source before
assuming otherwise). `architecture://overview` is a second, genuinely
independent resource, not a variant of the first, so the original
single-resource `conventions.py` module was merged into this one rather than
duplicated — the second real consumer this project's own conventions call
for combining, not two near-identical files that would silently break each
other at runtime.
"""

from __future__ import annotations

import json
from dataclasses import asdict

from mcp import types
from mcp.server.lowlevel import Server

from loupe_core.adapters.fastapi.config_drift import find_config_drift
from loupe_core.adapters.fastapi.migration_drift import find_migration_drift
from loupe_core.analysis.dead_code import find_dead_code
from loupe_core.analysis.duplicates import find_duplicates
from loupe_core.context.claude_md_generator import compute_architecture_overview
from loupe_core.conventions.mining import mine_conventions

CONVENTIONS_SUMMARY_URI = "conventions://summary"
ARCHITECTURE_OVERVIEW_URI = "architecture://overview"
STATIC_ANALYSIS_SUMMARY_URI = "static-analysis://summary"

# Same output-size discipline `mcp_tools.py`'s DEFAULT_MAX_AFFECTED already
# established (found and fixed three separate times — analyze_impact,
# expand_dependencies, find_code_smells — before this resource existed to
# risk a fourth): core check functions return the full, correct,
# untruncated list; this presentation layer caps what's actually returned,
# preserving a real count so truncation stays visible, not silent.
STATIC_ANALYSIS_FINDING_CAP = 30

ENV_EXAMPLE_FILENAME = ".env.example"
ALEMBIC_VERSIONS_DIR = "alembic/versions"


def conventions_summary_json(parsed_files) -> str:
    report = mine_conventions(list(parsed_files))
    return json.dumps(asdict(report), indent=2)


def architecture_overview_json(parsed_files, graph, symbols_by_id) -> str:
    report = mine_conventions(list(parsed_files))
    overview = compute_architecture_overview(report, graph, symbols_by_id)
    return json.dumps(overview, indent=2)


def static_analysis_summary_json(parsed_files, loupe_graph, symbols, semantic_index, repo_root) -> str:
    """`loupe_graph` is the `LoupeGraph` wrapper (matching `architecture_overview_json`'s
    own convention for the `get_graph()` callable) — the raw `networkx.DiGraph`
    E5/E6's check functions actually need is `loupe_graph.graph`.
    """
    parsed_files = list(parsed_files)
    symbols_by_id = {s.id: s for s in symbols}
    graph = loupe_graph.graph

    dead_code = find_dead_code(graph, symbols_by_id)
    duplicates = find_duplicates(semantic_index, symbols, graph)

    payload: dict = {
        "dead_code": {
            "total_count": len(dead_code),
            "findings": [asdict(f) for f in dead_code[:STATIC_ANALYSIS_FINDING_CAP]],
        },
        "duplicates": {
            "total_count": len(duplicates),
            "findings": [asdict(f) for f in duplicates[:STATIC_ANALYSIS_FINDING_CAP]],
        },
    }

    env_example_path = repo_root / ENV_EXAMPLE_FILENAME
    if env_example_path.exists():
        drift = find_config_drift(parsed_files, env_example_path.read_text())
        payload["config_drift"] = {"total_count": len(drift), "findings": [asdict(f) for f in drift[:STATIC_ANALYSIS_FINDING_CAP]]}
    else:
        payload["config_drift"] = None

    migrations_dir = repo_root / ALEMBIC_VERSIONS_DIR
    if migrations_dir.exists():
        migration_contents = [p.read_text() for p in sorted(migrations_dir.glob("*.py"))]
        drift = find_migration_drift(parsed_files, migration_contents)
        payload["migration_drift"] = {
            "total_count": len(drift),
            "findings": [asdict(f) for f in drift[:STATIC_ANALYSIS_FINDING_CAP]],
        }
    else:
        payload["migration_drift"] = None

    payload["api_contract_diff"] = (
        "not included — E9 needs two points in time to compare; use `loupe check --since <git-ref>` instead"
    )

    return json.dumps(payload, indent=2)


def register_resources(
    mcp_server: Server, get_parsed_files, get_graph, get_symbols_by_id, get_semantic_index, get_repo_root
) -> None:
    """`get_*` are zero-arg callables (not snapshots) so a resource read always
    reflects the *current* index, including after incremental reindexing —
    the same "always current, never stale" bar `list_symbols` and friends
    already hold themselves to via `request.app.state.index`.
    """

    @mcp_server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri=CONVENTIONS_SUMMARY_URI,
                name="conventions_summary",
                description="Auto-derived repo-wide coding conventions: error-handling, docstring style, import style.",
                mimeType="application/json",
            ),
            types.Resource(
                uri=ARCHITECTURE_OVERVIEW_URI,
                name="architecture_overview",
                description="A one-paragraph repo summary plus one-line descriptions of each major architectural "
                "cluster (Phase 10.5 Louvain coarse clusters) — the LOD hierarchy's L0/L1 zoom levels.",
                mimeType="application/json",
            ),
            types.Resource(
                uri=STATIC_ANALYSIS_SUMMARY_URI,
                name="static_analysis_summary",
                description="The zero-cost static analysis pack: dead code, duplicate code, config/env-var drift, "
                "and ORM migration drift. API contract diffing (E9) needs `loupe check --since <git-ref>` instead.",
                mimeType="application/json",
            ),
        ]

    @mcp_server.read_resource()
    async def read_resource(uri) -> str:
        if str(uri) == CONVENTIONS_SUMMARY_URI:
            return conventions_summary_json(get_parsed_files())
        if str(uri) == ARCHITECTURE_OVERVIEW_URI:
            return architecture_overview_json(get_parsed_files(), get_graph(), get_symbols_by_id())
        if str(uri) == STATIC_ANALYSIS_SUMMARY_URI:
            symbols_by_id = get_symbols_by_id()
            return static_analysis_summary_json(
                get_parsed_files(), get_graph(), list(symbols_by_id.values()), get_semantic_index(), get_repo_root()
            )
        raise ValueError(f"unknown resource: {uri}")
