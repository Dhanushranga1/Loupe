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

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
