"""Shared exclude-path logic for both the initial full index (`bootstrap.py`) and
the incremental file watcher (`indexer_worker.py`).

Before this module existed, `bootstrap.py`'s full-index path had its own
private `DEFAULT_IGNORED_DIR_NAMES` copy and never read `.loupeignore` or the
manifest's `index.exclude_paths` at all — only the incremental watcher did.
That gap is a real bug (found by indexing a real, large monorepo where a
stray `backend/.venv-py314-backup/` directory — not one of the hardcoded
exact names — got fully parsed and embedded: ~4,000 third-party files
crawled for a repo whose actual backend source was 69 files). One shared
implementation, used by both the full index and the incremental watcher, is
what makes "add a line to `.loupeignore`" actually work regardless of which
code path runs.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

DEFAULT_IGNORED_DIR_NAMES = {"__pycache__", ".venv", "venv", "node_modules", "dist", "build", ".git", ".loupe"}


def load_loupeignore_patterns(repo_root: Path) -> list[str]:
    path = repo_root / ".loupeignore"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text().splitlines() if line.strip() and not line.startswith("#")]


def is_path_ignored(rel_path: str, ignore_patterns: list[str]) -> bool:
    """True if `rel_path` (POSIX-style, relative to repo root) should be excluded.

    Two independent checks, either is sufficient:
    - Any path segment exactly matches one of the built-in default names, at
      any depth (".venv", "node_modules", etc.) — zero-config coverage of the
      common cases.
    - Any path segment, or the full relative path, matches a `.loupeignore`/
      manifest `exclude_paths` glob pattern (trailing "/" stripped). Matched
      per-segment, not just against the whole path, so a pattern like
      ".venv*" also catches a *nested* "backend/.venv-py314-backup/" — not
      only a repo-root ".venv-py314-backup/" — which is exactly the gap that
      let the real bug above through even for directories a user had tried
      to exclude by (exact-match) name.
    """
    parts = Path(rel_path).parts
    if any(part in DEFAULT_IGNORED_DIR_NAMES for part in parts):
        return True
    for raw_pattern in ignore_patterns:
        pattern = raw_pattern.rstrip("/")
        if not pattern:
            continue
        if any(fnmatch.fnmatch(part, pattern) for part in parts):
            return True
        if fnmatch.fnmatch(rel_path, pattern) or fnmatch.fnmatch(rel_path, f"{pattern}/*"):
            return True
    return False
