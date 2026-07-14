# Lens

A dashboard for [Loupe](../docs/loupe-project-guide.md) — visualizes the symbol graph, retrieval telemetry, auto-derived
conventions, and human feedback that a running `loupe serve` process already collects locally.

Lens never calls any Claude model and needs no API key. It talks to Loupe's plain REST `/dashboard/*` endpoints
(`server/app/dashboard.py`) over plain HTTP — nothing here goes through MCP. Claude Code stays the MCP client, using
whatever authentication you already have set up for it, entirely separately from this app.

## Running it

```bash
# 1. Start the Loupe server against the repo you want to inspect:
cd ../server && .venv/bin/loupe serve /path/to/your/repo

# 2. In another terminal, start Lens:
cd lens && npm install && npm run dev
```

Then open http://localhost:5173. By default Lens expects the Loupe server at `http://127.0.0.1:8765` — override with
a `VITE_LOUPE_SERVER_URL` env var if you're running it elsewhere.

## Pages

- **Overview** — index stats (symbol/file counts, unresolved references, last indexed).
- **Symbol Graph** — the live call graph, rendered on canvas: drag to pan, scroll to zoom, click a node to see its
  callers/callees, search by name, filter by module.
- **Telemetry** — recent MCP tool calls, with inline Helpful / Not helpful buttons per row (this is the dashboard
  button `docs/loupe-extensions.md`'s E3 section describes as feedback's primary path).
- **Conventions** — the auto-derived error-handling, docstring-style, and import-style report (E4), mirrored here as
  plain REST rather than the MCP resource Claude reads.
- **Feedback** — submission history across both paths (this dashboard, and Claude's own optional `submit_feedback`
  tool).

## Stack

Vite + React 19 + TypeScript + Tailwind v4, `HashRouter` (no server-side route config needed), no state library —
each page owns its own `fetch` via a small `useApi` hook (`src/hooks/useApi.ts`).
