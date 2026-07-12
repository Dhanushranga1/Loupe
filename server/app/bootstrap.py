"""Bootstrap flow: first-run `.loupe/` creation, schema-triggered reindex, incremental catch-up.

Implements docs/phase-4-systems.md §7, matching docs/loupe-target-project-
standard.md §8's `.loupe/` layout exactly.

Assumes the process's working directory is the repo root. `extract_symbols`
reads a file via `Path(file_path).read_bytes()` and stores that same string
as part of `Symbol.id`'s hash input — for ids to be relative and portable
across machines (per phase-0-foundations.md §3's design notes), the paths
passed here must be relative, and relative paths only resolve correctly
against the right CWD. `loupe serve`/`loupe index` are expected to `chdir`
to the resolved repo root before calling `bootstrap()` (see `app/cli.py`).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from loupe_core.graph.builder import LoupeGraph, ParsedFile, build_graph
from loupe_core.parsing.extractor import extract_symbols
from loupe_core.parsing.incremental import FileIndexCache
from loupe_core.parsing.languages import get_parser
from loupe_core.parsing.schema import Symbol, SymbolKind
from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.semantic import SemanticIndex

from .config import INDEX_SCHEMA_VERSION, LoupeConfig

LOUPE_SUBDIRS = ["cache", "logs/retrieval", "logs/sessions", "eval"]
DEFAULT_IGNORED_DIR_NAMES = {"__pycache__", ".venv", "venv", "node_modules", "dist", "build", ".git", ".loupe"}


@dataclass
class LoupeIndex:
    """Everything the MCP tools need to answer a query — the current servable index state."""

    repo_root: Path
    loupe_dir: Path
    parsed_files: dict[str, ParsedFile]
    graph: LoupeGraph
    lexical_index: LexicalIndex
    semantic_index: SemanticIndex
    file_cache: FileIndexCache

    @property
    def symbols(self) -> list[Symbol]:
        return [s for pf in self.parsed_files.values() for s in pf.symbols]

    def symbol_by_id(self, symbol_id: str) -> Symbol | None:
        return next((s for s in self.symbols if s.id == symbol_id), None)


def _ensure_loupe_dirs(loupe_dir: Path) -> None:
    for sub in LOUPE_SUBDIRS:
        (loupe_dir / sub).mkdir(parents=True, exist_ok=True)


def _discover_python_files(repo_root: Path) -> list[str]:
    """Relative, forward-slash-normalized paths of every *.py file under repo_root."""
    paths = []
    for p in repo_root.rglob("*.py"):
        rel = p.relative_to(repo_root)
        if any(part in DEFAULT_IGNORED_DIR_NAMES for part in rel.parts):
            continue
        paths.append(rel.as_posix())
    return sorted(paths)


def _symbol_to_dict(s: Symbol) -> dict:
    d = asdict(s)
    d["kind"] = s.kind.value
    return d


def _symbol_from_dict(d: dict) -> Symbol:
    d = dict(d)
    d["kind"] = SymbolKind(d["kind"])
    return Symbol(**d)


def _load_symbol_cache(path: Path) -> dict[str, list[Symbol]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    return {file_path: [_symbol_from_dict(s) for s in symbols] for file_path, symbols in data.items()}


def _save_symbol_cache(path: Path, symbols_by_file: dict[str, list[Symbol]]) -> None:
    data = {file_path: [_symbol_to_dict(s) for s in symbols] for file_path, symbols in symbols_by_file.items()}
    path.write_text(json.dumps(data))


def bootstrap(repo_root: Path, config: LoupeConfig, embedding_model: object | None = None) -> LoupeIndex:
    """Resolve `.loupe/` state and produce a fresh, fully-consistent `LoupeIndex`.

    - Missing `.loupe/schema_version` (first run) or a stale version -> full reindex.
    - Current version -> incremental catch-up: only files `FileIndexCache` flags
      as stale get re-run through Phase 0's extractor; everything else reuses
      its previously cached `Symbol` list, so the extractor is genuinely not
      re-invoked for unchanged files (not just "cheap to re-invoke").
    - The graph (Phase 1), lexical index, and semantic index are always fully
      rebuilt from whatever the current complete symbol set is — both are
      already specified as full-rebuild-every-time in their own phases.
    """
    repo_root = repo_root.resolve()
    loupe_dir = repo_root / ".loupe"
    schema_path = loupe_dir / "schema_version"
    file_cache_path = loupe_dir / "cache" / "file_index_cache.json"
    symbol_cache_path = loupe_dir / "cache" / "symbols.json"

    is_first_run = not schema_path.exists()
    current_schema_version: int | None = None
    if not is_first_run:
        try:
            current_schema_version = int(schema_path.read_text().strip())
        except ValueError:
            current_schema_version = None
    full_reindex = is_first_run or current_schema_version != INDEX_SCHEMA_VERSION

    _ensure_loupe_dirs(loupe_dir)

    file_cache = (
        FileIndexCache.load(str(file_cache_path))
        if not full_reindex and file_cache_path.exists()
        else FileIndexCache()
    )
    cached_symbols = {} if full_reindex else _load_symbol_cache(symbol_cache_path)

    parsed_files: dict[str, ParsedFile] = {}
    symbols_by_file: dict[str, list[Symbol]] = {}

    for rel_path in _discover_python_files(repo_root):
        source_bytes = Path(rel_path).read_bytes()
        tree = get_parser("python").parse(source_bytes)

        reusable = not full_reindex and not file_cache.is_stale(rel_path) and rel_path in cached_symbols
        if reusable:
            symbols = cached_symbols[rel_path]
        else:
            symbols = extract_symbols(rel_path)
            file_cache.update(rel_path, symbols)

        symbols_by_file[rel_path] = symbols
        parsed_files[rel_path] = ParsedFile(file_path=rel_path, tree=tree, source_bytes=source_bytes, symbols=symbols)

    graph = build_graph(list(parsed_files.values()))
    all_symbols = [s for symbols in symbols_by_file.values() for s in symbols]

    lexical_index = LexicalIndex(all_symbols)
    semantic_index = SemanticIndex(
        cache_db_path=str(loupe_dir / "cache" / "embeddings.db"),
        vector_db_path=str(loupe_dir / "vectors.db"),
        model=embedding_model,
    )
    semantic_index.index(all_symbols)

    schema_path.write_text(str(INDEX_SCHEMA_VERSION))
    file_cache.save(str(file_cache_path))
    _save_symbol_cache(symbol_cache_path, symbols_by_file)

    return LoupeIndex(
        repo_root=repo_root,
        loupe_dir=loupe_dir,
        parsed_files=parsed_files,
        graph=graph,
        lexical_index=lexical_index,
        semantic_index=semantic_index,
        file_cache=file_cache,
    )


def update_index(index: LoupeIndex, changed_rel_paths: set[str]) -> LoupeIndex:
    """Incrementally refresh an existing `LoupeIndex` for a set of settled file changes.

    Implements docs/phase-4-systems.md §4's per-settled-change pipeline.
    Only `changed_rel_paths` are re-run through Phase 0's extractor (or
    dropped entirely if the file no longer exists) — the graph and lexical
    index are always fully rebuilt from the resulting complete symbol set,
    matching their own phases' stated full-rebuild-every-time design; the
    semantic index's existing content-hash cache means "full rebuild" stays
    cheap, since only symbols whose hash actually changed get re-embedded.
    Assumes CWD is still the repo root (same process, same convention as
    `bootstrap()` — see this module's docstring).

    Deliberately opens a *fresh* `SemanticIndex` against the same on-disk
    cache/vector db files rather than reusing `index.semantic_index` — this
    function is designed to run inside `asyncio.to_thread` (§4's stated
    concurrency model), a different OS thread than the one that constructed
    the original object's `sqlite3` connections, and those connections
    cannot cross threads (a real `sqlite3.ProgrammingError`, caught by
    `test_indexer_worker.py`, not a theoretical concern). New connections to
    the same files are cheap, and the on-disk content-hash cache means this
    still doesn't trigger real re-embedding for anything unchanged.
    """
    parsed_files = dict(index.parsed_files)

    for rel_path in changed_rel_paths:
        path = Path(rel_path)
        if not path.exists():
            parsed_files.pop(rel_path, None)
            continue
        source_bytes = path.read_bytes()
        tree = get_parser("python").parse(source_bytes)
        symbols = extract_symbols(rel_path)
        index.file_cache.update(rel_path, symbols)
        parsed_files[rel_path] = ParsedFile(file_path=rel_path, tree=tree, source_bytes=source_bytes, symbols=symbols)

    graph = build_graph(list(parsed_files.values()))
    all_symbols = [s for pf in parsed_files.values() for s in pf.symbols]

    lexical_index = LexicalIndex(all_symbols)
    semantic_index = SemanticIndex(
        cache_db_path=str(index.loupe_dir / "cache" / "embeddings.db"),
        vector_db_path=str(index.loupe_dir / "vectors.db"),
    )
    semantic_index.index(all_symbols)

    index.file_cache.save(str(index.loupe_dir / "cache" / "file_index_cache.json"))
    _save_symbol_cache(
        index.loupe_dir / "cache" / "symbols.json", {rel: pf.symbols for rel, pf in parsed_files.items()}
    )

    return LoupeIndex(
        repo_root=index.repo_root,
        loupe_dir=index.loupe_dir,
        parsed_files=parsed_files,
        graph=graph,
        lexical_index=lexical_index,
        semantic_index=semantic_index,
        file_cache=index.file_cache,
    )
