"""Background file watcher with debounced incremental re-indexing (docs/phase-4-systems.md §4).

File watching via `watchdog.Observer`, respecting `.loupeignore` (falls back
to a small built-in default ignore list if absent, per loupe-target-project-
standard.md §4). Debounce, exact constants from the spec:
- `debounce_window = 300ms` — a file must have had no new change events for
  this long before it's considered settled and ready to process.
- `check_interval = 500ms` — how often the pending-changes set is scanned.

Concurrency: the actual re-index work (extractor + graph rebuild + semantic
indexing) runs via `asyncio.to_thread`, keeping the event loop free to keep
serving MCP tool calls against the previous index snapshot while a reindex
is in progress — there is no blocking wait on an in-flight reindex.
"""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

from fastapi import FastAPI
from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .bootstrap import update_index
from .ignore import is_path_ignored, load_loupeignore_patterns

DEBOUNCE_WINDOW_SECONDS = 0.3
CHECK_INTERVAL_SECONDS = 0.5


class _ChangeCollector(FileSystemEventHandler):
    """Records the last-event time per changed `*.py` file.

    watchdog delivers events from its own OS-event thread; only ever setting
    dict entries here (never read-modify-write) keeps this safe enough
    without an explicit lock — CPython dict item assignment is atomic.
    """

    def __init__(self, repo_root: Path, ignore_patterns: list[str]) -> None:
        self.repo_root = repo_root
        self.ignore_patterns = ignore_patterns
        self.pending: dict[str, float] = {}

    def _record(self, raw_path: str) -> None:
        path = Path(raw_path)
        if path.suffix != ".py":
            return
        try:
            rel_path = str(path.relative_to(self.repo_root))
        except ValueError:
            return
        if is_path_ignored(rel_path, self.ignore_patterns):
            return
        self.pending[rel_path] = time.monotonic()

    def on_created(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._record(event.src_path)

    def on_modified(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._record(event.src_path)

    def on_moved(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._record(event.dest_path)
            self._record(event.src_path)

    def on_deleted(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._record(event.src_path)


class IndexerWorker:
    """Owns the watchdog Observer + debounce loop; swaps `app.state.index` on settled changes."""

    def __init__(self, app: FastAPI, repo_root: Path, extra_exclude_paths: list[str] | None = None) -> None:
        self.app = app
        self.repo_root = repo_root
        ignore_patterns = load_loupeignore_patterns(repo_root) + list(extra_exclude_paths or [])
        self.collector = _ChangeCollector(repo_root, ignore_patterns)
        self._observer = Observer()
        self._task: asyncio.Task | None = None
        self.reparse_count = 0  # exposed for tests: how many settled-change batches were processed

    def start(self) -> None:
        self._observer.schedule(self.collector, str(self.repo_root), recursive=True)
        self._observer.start()
        self._task = asyncio.create_task(self._debounce_loop())

    async def stop(self) -> None:
        self._observer.stop()
        self._observer.join(timeout=2)
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _debounce_loop(self) -> None:
        while True:
            await asyncio.sleep(CHECK_INTERVAL_SECONDS)
            await self._process_settled_changes()

    async def _process_settled_changes(self) -> None:
        now = time.monotonic()
        settled = {
            path for path, last_event in list(self.collector.pending.items()) if now - last_event >= DEBOUNCE_WINDOW_SECONDS
        }
        if not settled:
            return
        for path in settled:
            self.collector.pending.pop(path, None)

        self.reparse_count += 1
        current_index = self.app.state.index
        self.app.state.index = await asyncio.to_thread(update_index, current_index, settled)
