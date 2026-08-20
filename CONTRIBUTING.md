# Contributing to Loupe

Thanks for looking at Loupe. This is a young, actively-developed project — issues, PRs, and honest bug reports are all welcome.

## Setup

Follow [`docs/SETUP.md`](docs/SETUP.md) — each package (`core/`, `mcp_server/`, `cli/`, `scaffold/`) has its own venv and installs independently.

## Running the tests

```bash
cd core        && .venv/bin/python -m pytest
cd mcp_server   && .venv/bin/python -m pytest
cd cli          && .venv/bin/python -m pytest
cd scaffold     && .venv/bin/python -m pytest
```

All four suites use real models (no mocked embeddings) — the first run downloads them from HuggingFace, which takes a minute.

Before opening a PR, run the suites for whichever package(s) you touched, and `loupe doctor .` against this repo itself if you changed anything config-related.

## Project layout

A monorepo, each package independently installable — see the [README's own layout section](README.md#project-layout) for the full breakdown. In short: `core/` is framework-free (parsing, graph, retrieval, governor), `mcp_server/` exposes it over FastAPI/MCP, `cli/` is the `loupe` command, `scaffold/` is the separate `loupe-new` project generator.

## Code conventions this project holds itself to

- **Pure function / thin wrapper split.** MCP tools are a pure `*_impl` function plus a thin `@router` HTTP wrapper that only handles request parsing. Keep new tools testable the same way.
- **Spec before code.** Non-trivial features get a short design doc first (see `docs/PhaseX/` and `docs/progress/*/steps/` for the pattern) — what's being built, why, and what design decisions were made. This isn't bureaucracy for its own sake; it's what makes the "why" of this codebase legible later.
- **Never guess.** The call resolver won't guess an ambiguous call; config loading won't silently assume a default that could be wrong. If something is genuinely ambiguous, surface it rather than picking a plausible-looking answer.
- **Honest reporting over flattering numbers.** If a benchmark result, a test, or a validation run comes back worse than hoped, that gets written up plainly — see `docs/PhaseX/retrieval-token-efficiency-fixes.md` for what this looks like in practice.
- **Dogfood real changes.** Where practical, verify a change against a real repo (this one, or an unrelated one), not just fixtures.

## Reporting a bug

Open a GitHub issue. Include your `loupe.manifest.yaml`, the output of `loupe doctor .`, and (if it's a crash) the full traceback. If it's config-related, `loupe doctor` catching it automatically is itself a useful signal — worth checking before you file.

## License

MIT — see [`LICENSE`](LICENSE). By contributing, you agree your contributions are licensed under the same terms.
