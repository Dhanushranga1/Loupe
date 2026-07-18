"""The context engineering layer (docs/PhaseX/phase-11-context-engineering.md,
claude-md-generator.md, session-notes.md, scope-aware-retrieval.md).

Package established now, as part of the mcp_server/cli restructure
(docs/PhaseX/MASTER_ROADMAP.md), ahead of its actual modules — Phase 11
itself hasn't been built yet. Lives in `loupe_core` rather than in
`mcp_server` or `cli` deliberately: `claude_md_generator`'s and
`session_notes`' actual generation/ranking logic (structured-data-to-template
rendering, decay-ranked scoring, MMR deduplication) is pure computation with
no FastAPI/MCP dependency, needed by both the CLI (`loupe generate-context`)
and the live MCP server (the `session_notes` tool, `architecture://overview`
in Phase 14) — putting it here keeps that logic framework-free and usable
from either caller, the same boundary every other `loupe_core` subpackage
already holds itself to.

Planned modules, once Phase 11 is actually specced into build-ready work:
- `claude_md_generator.py` — turns E4's conventions + Phase 1's centrality
  into a knapsack-budgeted CLAUDE.md (see claude-md-generator.md).
- `session_notes.py` — decay-ranked, MMR-deduplicated scratchpad storage
  (see session-notes.md).
- `scope.py` — path-based and Louvain-cluster-based candidate scoping, hard
  or soft (PageRank-biased) (see scope-aware-retrieval.md).
"""
