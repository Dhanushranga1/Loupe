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
from loupe_core.graph.test_linkage import link_tests
from loupe_core.parsing.extractor import extract_symbols
from loupe_core.parsing.incremental import FileIndexCache
from loupe_core.parsing.languages import get_parser
from loupe_core.parsing.schema import Symbol, SymbolKind
from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.semantic import EMBEDDING_DIM, SemanticIndex

from .compute_profiles import resolve_embedding_dim, resolve_embedding_model
from .config import INDEX_SCHEMA_VERSION, LoupeConfig
from .ignore import is_path_ignored, load_loupeignore_patterns

LOUPE_SUBDIRS = ["cache", "logs/retrieval", "logs/sessions", "logs/feedback", "eval", "context"]


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
    embedding_dim: int = EMBEDDING_DIM
    embedding_model: object | None = None

    @property
    def symbols(self) -> list[Symbol]:
        return [s for pf in self.parsed_files.values() for s in pf.symbols]

    def symbol_by_id(self, symbol_id: str) -> Symbol | None:
        return next((s for s in self.symbols if s.id == symbol_id), None)


def _ensure_loupe_dirs(loupe_dir: Path) -> None:
    for sub in LOUPE_SUBDIRS:
        (loupe_dir / sub).mkdir(parents=True, exist_ok=True)


def _discover_python_files(repo_root: Path, ignore_patterns: list[str]) -> list[str]:
    """Relative, forward-slash-normalized paths of every *.py file under repo_root.

    `ignore_patterns` is `.loupeignore` lines plus the manifest's
    `index.exclude_paths` — previously this function only ever checked the
    built-in default names (see `ignore.py`'s module docstring for the real
    bug that gap caused).
    """
    paths = []
    for p in repo_root.rglob("*.py"):
        rel = p.relative_to(repo_root)
        if is_path_ignored(rel.as_posix(), ignore_patterns):
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
    - A changed `compute_profile` (docs/PhaseX/compute-profiles.md §3) is
      treated exactly like a stale schema version: full reindex, plus the
      old vector store and embedding cache are deleted outright rather than
      reused — their contents are for a *different embedding model*
      entirely, not just stale content a content-hash check could recognize
      and skip.
    """
    repo_root = repo_root.resolve()
    loupe_dir = repo_root / ".loupe"
    schema_path = loupe_dir / "schema_version"
    compute_profile_path = loupe_dir / "compute_profile"
    file_cache_path = loupe_dir / "cache" / "file_index_cache.json"
    symbol_cache_path = loupe_dir / "cache" / "symbols.json"

    is_first_run = not schema_path.exists()
    current_schema_version: int | None = None
    if not is_first_run:
        try:
            current_schema_version = int(schema_path.read_text().strip())
        except ValueError:
            current_schema_version = None

    previous_compute_profile = compute_profile_path.read_text().strip() if compute_profile_path.exists() else None
    compute_profile_changed = previous_compute_profile is not None and previous_compute_profile != config.compute_profile

    full_reindex = is_first_run or current_schema_version != INDEX_SCHEMA_VERSION or compute_profile_changed

    _ensure_loupe_dirs(loupe_dir)

    if compute_profile_changed:
        # §2's dimensionality constraint: embeddings from different models
        # aren't comparable and usually aren't even the same width — the old
        # vector store table and embedding cache must be recreated fresh at
        # the new model's dimension, not left mismatched or silently mixed.
        (loupe_dir / "vectors.db").unlink(missing_ok=True)
        (loupe_dir / "cache" / "embeddings.db").unlink(missing_ok=True)

    file_cache = (
        FileIndexCache.load(str(file_cache_path))
        if not full_reindex and file_cache_path.exists()
        else FileIndexCache()
    )
    cached_symbols = {} if full_reindex else _load_symbol_cache(symbol_cache_path)

    parsed_files: dict[str, ParsedFile] = {}
    symbols_by_file: dict[str, list[Symbol]] = {}

    ignore_patterns = load_loupeignore_patterns(repo_root) + config.index.exclude_paths
    for rel_path in _discover_python_files(repo_root, ignore_patterns):
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
    link_tests(graph.graph, {s.id: s for s in all_symbols})

    lexical_index = LexicalIndex(all_symbols)

    # `embedding_model` stays None in production (real callers never pass it)
    # — only tests inject a spy/stub model directly. In that real-production
    # case, resolve and load the compute-profile-selected model by name;
    # §4's "profile sets defaults, explicit values win" rule is exactly
    # `resolve_embedding_model`'s own job. Loaded via `get_default_model`'s
    # own per-name cache (not a fresh `SentenceTransformer(...)` here) so a
    # process that calls `bootstrap()` more than once for the same resolved
    # model name — e.g. a test exercising first-run then incremental-catch-up
    # bootstrap calls back to back — doesn't reload real model weights twice.
    resolved_dim = resolve_embedding_dim(config.compute_profile)
    if embedding_model is None:
        from loupe_core.retrieval.semantic import get_default_model

        resolved_model_name = resolve_embedding_model(config.compute_profile, config.embedding_model)
        embedding_model = get_default_model(resolved_model_name)

    semantic_index = SemanticIndex(
        dim=resolved_dim,
        cache_db_path=str(loupe_dir / "cache" / "embeddings.db"),
        vector_db_path=str(loupe_dir / "vectors.db"),
        model=embedding_model,
    )
    semantic_index.index(all_symbols)

    schema_path.write_text(str(INDEX_SCHEMA_VERSION))
    compute_profile_path.write_text(config.compute_profile)
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
        embedding_dim=resolved_dim,
        embedding_model=embedding_model,
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

    Opens a *fresh* `SemanticIndex` against the same on-disk cache/vector db
    files rather than reusing `index.semantic_index` — cheap (the on-disk
    content-hash cache means this still doesn't trigger real re-embedding
    for anything unchanged) and keeps this function's own output an
    independent object from whatever `index` it was called with, not a
    mutation of shared state. Reuses `index.embedding_dim`/`index.embedding_model`
    (resolved once, at `bootstrap()` time, from the active `compute_profile`)
    rather than re-resolving from config or falling back to bare defaults —
    an earlier version of this function built `SemanticIndex` with neither,
    which silently reopened the on-disk vector table at the wrong dimension
    for any non-default compute profile on the very first incremental
    reindex after a full one.

    Note this alone does *not* make cross-thread access safe: this function
    runs inside `asyncio.to_thread` (§4's stated concurrency model), so the
    connections it creates are built on a threadpool thread, while a later
    HTTP request reading them runs on the main event-loop thread — a real
    `sqlite3.ProgrammingError` ("SQLite objects created in a thread can only
    be used in that same thread"), found live on a real `loupe serve`
    process, not by any test, until `storage/vector_store.py`'s
    `VectorStore` and `retrieval/semantic.py`'s `EmbeddingCache` both added
    `check_same_thread=False` to their own `sqlite3.connect()` calls — see
    those modules' own comments for the full explanation. That fix, not
    "always build fresh connections," is what actually makes cross-thread
    access safe; opening fresh connections here is retained for the
    unrelated, still-good reason of output independence described above.
    Regression-tested end-to-end at `test_indexer_worker.py::test_semantic_search_works_after_a_real_incremental_reindex`.
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
    link_tests(graph.graph, {s.id: s for s in all_symbols})

    lexical_index = LexicalIndex(all_symbols)
    semantic_index = SemanticIndex(
        dim=index.embedding_dim,
        cache_db_path=str(index.loupe_dir / "cache" / "embeddings.db"),
        vector_db_path=str(index.loupe_dir / "vectors.db"),
        model=index.embedding_model,
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
        embedding_dim=index.embedding_dim,
        embedding_model=index.embedding_model,
    )
