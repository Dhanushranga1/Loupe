"""The context engineering layer (docs/PhaseX/claude-md-generator.md,
scope-aware-retrieval.md, session-notes.md — each supersedes the simpler,
combined phase-11-context-engineering.md doc).

Lives in `loupe_core` rather than in `mcp_server` or `cli` deliberately:
`claude_md_generator`'s and `session_notes`' generation/ranking logic
(knapsack budgeting, decay-ranked scoring, MMR deduplication) is pure
computation with no FastAPI/MCP dependency, needed by both the CLI
(`loupe generate-context`) and the live MCP server (the `session_notes`
tool) — putting it here keeps that logic framework-free and usable from
either caller, the same boundary every other `loupe_core` subpackage holds.

Modules (docs/progress/phase-11/checklist.md — status: complete):
- `claude_md_generator.py` — knapsack-budgeted, Louvain-cluster-aware,
  content-hash-freshness-checked CLAUDE.md generation with structured diffing.
- `scope.py` — explicit path-based hard filtering and Louvain-cluster
  auto-detected, personalized-PageRank-biased soft-boundary retrieval.
- `session_notes.py` — decay-ranked (Phase 3's EvictionCache, reused
  directly), MMR-deduplicated (retrieval-upgrades' MMR, reused directly)
  scratchpad storage, two-tier (full log + managed active set).
"""
