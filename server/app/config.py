"""Global + project config layering (docs/loupe-target-project-standard.md §3/§6).

Two layers, resolved project-over-global: `~/.config/loupe/global.yaml` for
personal defaults across every project, `<repo>/loupe.manifest.yaml` for this
project's overrides. A brand-new project with zero Loupe-specific setup still
works — it just inherits defaults — and only needs to state what's actually
different about it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Two distinct version numbers, deliberately not conflated (addendum item c):
# INDEX_SCHEMA_VERSION governs .loupe/'s on-disk format (bump -> full reindex).
# MCP_TOOL_SCHEMA_VERSION governs the four tools' input/output contracts
# (bump -> a client needs updating, independent of whether the index changed).
INDEX_SCHEMA_VERSION = 1
MCP_TOOL_SCHEMA_VERSION = 1

DEFAULT_TOKEN_BUDGET = 6000
DEFAULT_HARD_CEILING = 20000
DEFAULT_EMBEDDING_MODEL = "bge-small-en-v1.5"
DEFAULT_PORT = 8765
DEFAULT_SYMBOL_KINDS = ["function", "class", "method"]

GLOBAL_CONFIG_PATH = Path.home() / ".config" / "loupe" / "global.yaml"


@dataclass
class TokenBudgetConfig:
    default_per_turn: int = DEFAULT_TOKEN_BUDGET
    hard_ceiling: int = DEFAULT_HARD_CEILING


@dataclass
class IndexConfig:
    symbol_kinds: list[str] = field(default_factory=lambda: list(DEFAULT_SYMBOL_KINDS))
    exclude_paths: list[str] = field(default_factory=list)


@dataclass
class LoupeConfig:
    repo_root: Path
    languages: list[str] = field(default_factory=lambda: ["python"])
    token_budget: TokenBudgetConfig = field(default_factory=TokenBudgetConfig)
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    index: IndexConfig = field(default_factory=IndexConfig)
    packages: list[dict[str, str]] = field(default_factory=list)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge `override` onto `base` — nested dicts merge key-by-key, not wholesale replace."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(repo_root: Path, global_config_path: Path = GLOBAL_CONFIG_PATH) -> LoupeConfig:
    """Load and merge global + project (`loupe.manifest.yaml`) config, project winning conflicts."""
    merged = _merge(_load_yaml(global_config_path), _load_yaml(repo_root / "loupe.manifest.yaml"))

    token_budget_data = merged.get("token_budget", {})
    index_data = merged.get("index", {})

    return LoupeConfig(
        repo_root=repo_root,
        languages=merged.get("languages", ["python"]),
        token_budget=TokenBudgetConfig(
            default_per_turn=token_budget_data.get("default_per_turn", DEFAULT_TOKEN_BUDGET),
            hard_ceiling=token_budget_data.get("hard_ceiling", DEFAULT_HARD_CEILING),
        ),
        embedding_model=merged.get("embedding_model", DEFAULT_EMBEDDING_MODEL),
        index=IndexConfig(
            symbol_kinds=index_data.get("symbol_kinds", list(DEFAULT_SYMBOL_KINDS)),
            exclude_paths=index_data.get("exclude_paths", []),
        ),
        packages=merged.get("packages", []),
    )
