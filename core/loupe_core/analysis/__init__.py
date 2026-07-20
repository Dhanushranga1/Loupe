"""The zero-cost static analysis pack (E5-E9, docs/PhaseX/zero-cost-static-analysis-pack.md).

Every check here reuses data already computed by an earlier phase — no new
symbol extraction machinery for the graph-based checks — and none of it
touches Claude's context budget unless explicitly asked for: exposed as CLI
output (`loupe check`) and an MCP Resource, never proactively injected into
context, the same treatment E4's conventions report already got.

- `dead_code.py` (E6) — zero-incoming-edge symbols, reusing E1's traversal.
- `duplicates.py` (E5) — all-pairs embedding similarity among unrelated symbols.
- `contract_diff.py` (E9) — structured route-contract diffing between commits.
- `config_drift.py` (E7, lives in `adapters/fastapi/` — Settings/.env.example
  are target-project-standard concepts, not generic ones).
- `migration_drift.py` (E8, same reasoning, `adapters/fastapi/`).
"""
