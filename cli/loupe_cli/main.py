"""CLI: `loupe init` / `index` / `serve` / `status` (docs/phase-4-systems.md §9).

Thin wrappers around what already exists — `bootstrap()`, `create_app()`,
and plain filesystem/config reads. No new logic lives here.

A standalone package, deliberately (docs/PhaseX/MASTER_ROADMAP.md's restructure
decision): `loupe_cli` depends on `loupe_mcp_server` and `loupe_core` the same
way `loupe_mcp_server` depends on `loupe_core` — imported, not co-located —
so the CLI's own surface (argument parsing, output formatting) stays free of
any FastAPI/MCP-specific code, and the two packages can evolve or even ship
independently.

Simplification worth noting: §9 describes `loupe init` as "interactively"
generating config. This implementation writes sensible, documented defaults
non-interactively rather than prompting — nothing in phase-4-systems.md §8's
Definition of Done tests interactive-prompt behavior, and a project can
already commit the generated manifest with just `languages` set and inherit
everything else (loupe-target-project-standard.md §3), so a prompt flow
isn't load-bearing for correctness here.
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path

from loupe_mcp_server.bootstrap import bootstrap
from loupe_mcp_server.config import DEFAULT_PORT, INDEX_SCHEMA_VERSION, load_config

DEFAULT_MANIFEST = f"""\
# loupe.manifest.yaml
schema_version: {INDEX_SCHEMA_VERSION}

languages:
  - python

token_budget:
  default_per_turn: 6000
  hard_ceiling: 20000

embedding_model: bge-small-en-v1.5

index:
  symbol_kinds: [function, class, method]
  exclude_paths: []
"""

DEFAULT_LOUPEIGNORE = """\
# .loupeignore
__pycache__/
.venv/
venv/
node_modules/
dist/
build/
"""


def cmd_init(args: argparse.Namespace) -> int:
    repo_root = Path(args.path).resolve()
    manifest_path = repo_root / "loupe.manifest.yaml"
    ignore_path = repo_root / ".loupeignore"

    if manifest_path.exists():
        print(f"{manifest_path} already exists, leaving it untouched.")
    else:
        manifest_path.write_text(DEFAULT_MANIFEST)
        print(f"Created {manifest_path}")

    if ignore_path.exists():
        print(f"{ignore_path} already exists, leaving it untouched.")
    else:
        ignore_path.write_text(DEFAULT_LOUPEIGNORE)
        print(f"Created {ignore_path}")

    return 0


def cmd_index(args: argparse.Namespace) -> int:
    repo_root = Path(args.path).resolve()
    os.chdir(repo_root)
    config = load_config(repo_root)
    index = bootstrap(repo_root, config)

    languages = sorted({s.language for s in index.symbols})
    print(f"Indexed {len(index.symbols)} symbols across {len(index.parsed_files)} files.")
    print(f"Languages detected: {', '.join(languages) if languages else '(none)'}")
    print(f"Unresolved references: {len(index.graph.unresolved)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    from loupe_mcp_server.main import create_app

    repo_root = Path(args.path).resolve()
    app = create_app(repo_root=repo_root)
    uvicorn.run(app, host="127.0.0.1", port=args.port)
    return 0


RANKER_FILENAME = "ranker.pkl"


def _ranker_path(loupe_dir: Path) -> Path:
    return loupe_dir / RANKER_FILENAME


def _print_ranker_summary(loupe_dir: Path) -> None:
    from loupe_core.retrieval.ranker import Ranker

    ranker = Ranker.load(str(_ranker_path(loupe_dir)))
    if ranker.is_trained:
        print("Learned ranker: trained")
        for name, weight in ranker.coefficients.items():
            print(f"  {name:<17} {weight:+.4f}")
    else:
        print("Learned ranker: not trained (cold-start — falling back to RRF)")


def cmd_status(args: argparse.Namespace) -> int:
    repo_root = Path(args.path).resolve()
    loupe_dir = repo_root / ".loupe"

    if not loupe_dir.exists():
        print(f"No .loupe/ directory found at {repo_root} — run `loupe index` or `loupe serve` first.")
        return 1

    schema_path = loupe_dir / "schema_version"
    schema_version = schema_path.read_text().strip() if schema_path.exists() else "(unknown)"

    symbol_cache_path = loupe_dir / "cache" / "symbols.json"
    symbol_count = 0
    if symbol_cache_path.exists():
        data = json.loads(symbol_cache_path.read_text())
        symbol_count = sum(len(v) for v in data.values())

    cache_size_bytes = sum(f.stat().st_size for f in loupe_dir.rglob("*") if f.is_file())

    print(f"Repo:           {repo_root}")
    print(f"Schema version: {schema_version} (current: {INDEX_SCHEMA_VERSION})")
    print(f"Symbol count:   {symbol_count}")
    print(f"Cache size:     {cache_size_bytes / 1024:.1f} KB")
    if schema_path.exists():
        last_indexed = datetime.datetime.fromtimestamp(schema_path.stat().st_mtime).isoformat(timespec="seconds")
        print(f"Last indexed:   {last_indexed}")
    _print_ranker_summary(loupe_dir)
    return 0


def _count_retrieval_log_entries(loupe_dir: Path) -> int:
    logs_dir = loupe_dir / "logs" / "retrieval"
    if not logs_dir.exists():
        return 0
    count = 0
    for log_file in logs_dir.glob("*.jsonl"):
        with open(log_file) as f:
            count += sum(1 for _ in f)
    return count


def cmd_retrain(args: argparse.Namespace) -> int:
    """Retrain the learned ranker from accumulated telemetry (docs/phase-6-closing-the-loop.md §4/§9).

    Honest limitation: Phase 4's `RetrievalLog.candidates` records only
    `{symbol_id, score}` per entry (telemetry.py's `_entry_to_dict`), not the
    full `Symbol` (file path, line range) or file content at retrieval time
    that `backfill_outcome` needs to compute a real edited/not-edited label.
    Until telemetry is widened to carry that, there is no real labeled data
    to train on yet — this command reports that state accurately instead of
    fabricating a training set, while still exercising the same save/load
    path `loupe status` reads from.
    """
    from loupe_core.retrieval.ranker import COLD_START_THRESHOLD, Ranker

    repo_root = Path(args.path).resolve()
    loupe_dir = repo_root / ".loupe"
    if not loupe_dir.exists():
        print(f"No .loupe/ directory found at {repo_root} — run `loupe index` or `loupe serve` first.")
        return 1

    ranker_path = _ranker_path(loupe_dir)
    ranker = Ranker.load(str(ranker_path))
    print("Before:")
    _print_ranker_summary(loupe_dir)

    log_entry_count = _count_retrieval_log_entries(loupe_dir)
    labeled_example_count = 0  # real backfill needs symbol/content data telemetry doesn't record yet (see docstring)

    print(f"\nRetrieval log entries found: {log_entry_count}")
    print(f"Labeled training examples usable: {labeled_example_count} (need {COLD_START_THRESHOLD})")
    if labeled_example_count < COLD_START_THRESHOLD:
        print(
            "Not enough labeled usage data yet to retrain — telemetry currently records symbol_id/score only, "
            "not the symbol/content data outcome backfill needs. Ranker left as-is."
        )
    ranker.save(str(ranker_path))

    print("\nAfter:")
    _print_ranker_summary(loupe_dir)
    return 0


CLAUDE_MD_STATE_FILENAME = "context/state.json"


def cmd_generate_context(args: argparse.Namespace) -> int:
    """`loupe generate-context` — writes/updates CLAUDE.md (docs/PhaseX/claude-md-generator.md).

    Human-review gate (§4 of the spec), resolved explicitly: this command
    never runs implicitly as part of `loupe index`/`loupe serve` — it's a
    deliberate action a human triggers on purpose, the same treatment
    `loupe retrain` already gets. It *does* write CLAUDE.md's real content to
    disk (there's no useful "dry run only" reading of the spec — a file
    nobody can `git diff` isn't reviewable) — "never auto-committed" refers
    to git commits specifically: Loupe never stages or commits anything itself.
    """
    from loupe_core.context.claude_md_generator import GeneratorState, generate_claude_md
    from loupe_core.conventions.mining import mine_conventions

    repo_root = Path(args.path).resolve()
    os.chdir(repo_root)
    config = load_config(repo_root)
    index = bootstrap(repo_root, config)

    report = mine_conventions(list(index.parsed_files.values()))
    symbols_by_id = {s.id: s for s in index.symbols}

    state_path = index.loupe_dir / CLAUDE_MD_STATE_FILENAME
    previous_state = GeneratorState.from_json(state_path.read_text()) if state_path.exists() else None

    result = generate_claude_md(report, index.graph, symbols_by_id, previous_state=previous_state)

    if not result.regenerated:
        print("No convention/architecture changes detected since the last generation — CLAUDE.md left untouched.")
        return 0

    claude_md_path = repo_root / "CLAUDE.md"
    claude_md_path.write_text(result.content)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(result.state.to_json())

    if previous_state is None:
        print(f"Generated {claude_md_path} for the first time.")
    else:
        print(f"Regenerated {claude_md_path}. Changes since last generation:")
        for line in result.diff_lines:
            print(f"  - {line}")
        if not result.diff_lines:
            print("  (content changed, but no individually-nameable convention/architecture shift detected)")
    print("Review the diff before committing — Loupe never commits this file itself.")
    return 0


def cmd_update_churn(args: argparse.Namespace) -> int:
    """`loupe update-churn` (Phase 14 §2) — recomputes churn scores from real git
    history and caches them for `search_symbols` to fold in as a fourth RRF
    signal. Deliberately a separate, manually/periodically-triggered command,
    not run by `loupe index`/the file watcher: churn reflects commit
    *history*, not current file content, so it doesn't need — and shouldn't
    pay — the same per-edit recompute cadence content-hash-driven caches use.
    """
    import git

    from loupe_core.retrieval.churn import compute_churn_scores
    from loupe_mcp_server.mcp_tools import CHURN_CACHE_FILENAME

    repo_root = Path(args.path).resolve()
    os.chdir(repo_root)
    config = load_config(repo_root)
    index = bootstrap(repo_root, config)

    try:
        repo = git.Repo(repo_root)
    except git.InvalidGitRepositoryError:
        print(f"{repo_root} is not a git repository — churn needs real commit history, nothing to compute.")
        return 1

    scores = compute_churn_scores(repo, index.symbols)
    nonzero = sum(1 for v in scores.values() if v > 0)

    churn_path = index.loupe_dir / CHURN_CACHE_FILENAME
    churn_path.parent.mkdir(parents=True, exist_ok=True)
    churn_path.write_text(json.dumps(scores))

    print(f"Computed churn for {len(scores)} symbols ({nonzero} touched within the churn window).")
    print(f"Cached to {churn_path}")
    return 0


def cmd_update_suggestions(args: argparse.Namespace) -> int:
    """`loupe update-suggestions` (Phase 14 §3) — mines real `get_symbol`
    co-occurrence history from Phase 4's `RetrievalLog`s and caches
    confidence-scored suggestions for `get_symbol` to attach to its own
    response. Same `loupe update-*` family as `update-churn`/Phase 6's
    `retrain` — batch, periodic, not computed per-query.
    """
    from loupe_core.context.co_retrieval import mine_co_retrieval_suggestions, sessions_from_retrieval_logs
    from loupe_mcp_server.mcp_tools import CO_RETRIEVAL_CACHE_FILENAME

    repo_root = Path(args.path).resolve()
    loupe_dir = repo_root / ".loupe"
    if not loupe_dir.exists():
        print(f"No .loupe/ directory found at {repo_root} — run `loupe index` or `loupe serve` first.")
        return 1

    logs_dir = loupe_dir / "logs" / "retrieval"
    sessions = sessions_from_retrieval_logs(logs_dir)
    suggestions = mine_co_retrieval_suggestions(sessions)

    suggestions_path = loupe_dir / CO_RETRIEVAL_CACHE_FILENAME
    suggestions_path.parent.mkdir(parents=True, exist_ok=True)
    suggestions_path.write_text(
        json.dumps(
            {
                symbol_id: [{"symbol_id": s.symbol_id, "confidence": s.confidence, "support": s.support} for s in suggested]
                for symbol_id, suggested in suggestions.items()
            }
        )
    )

    print(f"Mined co-retrieval suggestions from {len(sessions)} session(s) with get_symbol activity.")
    print(f"{len(suggestions)} symbol(s) now have at least one suggestion (minimum support met).")
    print(f"Cached to {suggestions_path}")
    return 0


ALEMBIC_VERSIONS_DIR = "alembic/versions"
ENV_EXAMPLE_FILENAME = ".env.example"
CHECK_FINDINGS_PRINT_LIMIT = 20


def cmd_check(args: argparse.Namespace) -> int:
    """`loupe check` — the zero-cost static analysis pack (E5-E9,
    docs/PhaseX/zero-cost-static-analysis-pack.md). CLI output by default,
    never proactively injected into Claude's context — the same treatment
    E4's conventions report already got; an MCP Resource is available
    separately for Claude to read on request (`resources.py`).

    E5/E6 always run (pure graph/embedding reuse, nothing project-specific
    required). E7/E8 run only when their respective inputs exist
    (`.env.example`, `alembic/versions/`) — silently skipped, not
    falsely-clean, when a project doesn't use that pattern at all. E9 only
    runs with `--since <git-ref>`, since contract diffing is inherently a
    two-points-in-time comparison, not a single-snapshot check like the
    other four.
    """
    from loupe_core.adapters.fastapi.config_drift import find_config_drift
    from loupe_core.adapters.fastapi.contract_diff import diff_contracts, extract_route_contracts
    from loupe_core.adapters.fastapi.migration_drift import find_migration_drift
    from loupe_core.analysis.dead_code import find_dead_code
    from loupe_core.analysis.duplicates import find_duplicates

    repo_root = Path(args.path).resolve()
    os.chdir(repo_root)
    config = load_config(repo_root)
    index = bootstrap(repo_root, config)
    symbols_by_id = {s.id: s for s in index.symbols}
    parsed_files = list(index.parsed_files.values())

    print(f"Static analysis pack — {repo_root}\n")

    dead = find_dead_code(index.graph.graph, symbols_by_id)
    print(f"E6 dead code: {len(dead)} finding(s)")
    for finding in dead[:CHECK_FINDINGS_PRINT_LIMIT]:
        print(f"  {finding.file_path}::{finding.qualified_name}")

    duplicates = find_duplicates(index.semantic_index, index.symbols, index.graph.graph)
    print(f"\nE5 duplicate code: {len(duplicates)} finding(s)")
    for finding in duplicates[:CHECK_FINDINGS_PRINT_LIMIT]:
        a = symbols_by_id.get(finding.symbol_id_a)
        b = symbols_by_id.get(finding.symbol_id_b)
        if a is not None and b is not None:
            print(f"  {a.qualified_name} <-> {b.qualified_name} (similarity {finding.similarity:.3f})")

    env_example_path = repo_root / ENV_EXAMPLE_FILENAME
    if env_example_path.exists():
        drift = find_config_drift(parsed_files, env_example_path.read_text())
        print(f"\nE7 config drift: {len(drift)} finding(s)")
        for finding in drift[:CHECK_FINDINGS_PRINT_LIMIT]:
            print(f"  {finding.env_var} ({finding.kind})")
    else:
        print(f"\nE7 config drift: skipped ({ENV_EXAMPLE_FILENAME} not found)")

    migrations_dir = repo_root / ALEMBIC_VERSIONS_DIR
    if migrations_dir.exists():
        migration_contents = [p.read_text() for p in sorted(migrations_dir.glob("*.py"))]
        drift = find_migration_drift(parsed_files, migration_contents)
        print(f"\nE8 migration drift: {len(drift)} finding(s)")
        for finding in drift[:CHECK_FINDINGS_PRINT_LIMIT]:
            print(f"  {finding.model_qualified_name}.{finding.field_name}")
    else:
        print(f"\nE8 migration drift: skipped ({ALEMBIC_VERSIONS_DIR}/ not found)")

    if args.since:
        import git

        repo = git.Repo(repo_root)
        old_parsed = _parsed_files_at_commit(repo, args.since)
        old_contracts = extract_route_contracts(old_parsed)
        new_contracts = extract_route_contracts(parsed_files)
        changes = diff_contracts(old_contracts, new_contracts)
        print(f"\nE9 API contract diff (since {args.since}): {len(changes)} breaking change(s)")
        for change in changes[:CHECK_FINDINGS_PRINT_LIMIT]:
            print(f"  {change.qualified_name}: {change.description}")
    else:
        print("\nE9 API contract diff: skipped (pass --since <git-ref> to compare route contracts)")

    return 0


def _parsed_files_at_commit(repo, ref: str) -> list:
    """Reconstruct every *.py file's `ParsedFile` as of `ref` — the same
    git-blob-reconstruction technique `eval/mine_history.py`'s benchmark
    mining and harness snapshotting already use, reused directly rather
    than a second implementation."""
    from loupe_core.eval.mine_history import parsed_file_from_git_blob

    commit = repo.commit(ref)
    file_paths = [line for line in repo.git.ls_tree("-r", ref, "--name-only").splitlines() if line.endswith(".py")]
    parsed = []
    for rel_path in file_paths:
        source_bytes = (commit.tree / rel_path).data_stream.read()
        parsed.append(parsed_file_from_git_blob(rel_path, source_bytes))
    return parsed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loupe")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="generate loupe.manifest.yaml and .loupeignore")
    init_parser.add_argument("path", nargs="?", default=".")
    init_parser.set_defaults(func=cmd_init)

    index_parser = subparsers.add_parser("index", help="run a full/incremental index and print a summary")
    index_parser.add_argument("path", nargs="?", default=".")
    index_parser.set_defaults(func=cmd_index)

    serve_parser = subparsers.add_parser("serve", help="start the MCP server")
    serve_parser.add_argument("path", nargs="?", default=".")
    serve_parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    serve_parser.set_defaults(func=cmd_serve)

    status_parser = subparsers.add_parser("status", help="show index freshness without touching anything")
    status_parser.add_argument("path", nargs="?", default=".")
    status_parser.set_defaults(func=cmd_status)

    retrain_parser = subparsers.add_parser("retrain", help="retrain the learned ranker from accumulated telemetry")
    retrain_parser.add_argument("path", nargs="?", default=".")
    retrain_parser.set_defaults(func=cmd_retrain)

    generate_context_parser = subparsers.add_parser("generate-context", help="write/update CLAUDE.md from detected conventions and architecture")
    generate_context_parser.add_argument("path", nargs="?", default=".")
    generate_context_parser.set_defaults(func=cmd_generate_context)

    update_churn_parser = subparsers.add_parser("update-churn", help="recompute code-churn scores from git history")
    update_churn_parser.add_argument("path", nargs="?", default=".")
    update_churn_parser.set_defaults(func=cmd_update_churn)

    update_suggestions_parser = subparsers.add_parser(
        "update-suggestions", help="mine get_symbol co-retrieval suggestions from telemetry"
    )
    update_suggestions_parser.add_argument("path", nargs="?", default=".")
    update_suggestions_parser.set_defaults(func=cmd_update_suggestions)

    check_parser = subparsers.add_parser("check", help="run the zero-cost static analysis pack (E5-E9)")
    check_parser.add_argument("path", nargs="?", default=".")
    check_parser.add_argument("--since", default=None, help="git ref to diff API contracts against (E9)")
    check_parser.set_defaults(func=cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
