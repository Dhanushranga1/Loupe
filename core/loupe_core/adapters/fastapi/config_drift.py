"""E7 — Config/env-var drift detection (docs/PhaseX/zero-cost-static-analysis-pack.md).

Heuristic, not framework awareness — the same treatment
`adapters/fastapi/routes.py` already gives HTTP routes: a class named
exactly `Settings` is treated as the project's pydantic-settings class, the
standard, near-universal convention for this pattern, not confirmed via
base-class type inference (this project does no type inference anywhere).

Name mapping, decided explicitly: a Settings field `api_key` is expected to
correspond to env var `API_KEY` — a plain uppercase transform, matching
pydantic-settings' own default (case-insensitive, no explicit alias)
behavior. A field using an explicit `Field(alias=...)`/`env=` override
would not be caught correctly here — an honest scope limit, the same kind
E6 documents for framework-registered entry points.
"""

from __future__ import annotations

from dataclasses import dataclass

from loupe_core.graph.builder import ParsedFile
from loupe_core.parsing.ast_utils import class_field_annotations, symbol_nodes
from loupe_core.parsing.schema import SymbolKind

SETTINGS_CLASS_NAME = "Settings"


@dataclass(frozen=True)
class ConfigDriftFinding:
    env_var: str
    kind: str  # "missing_from_env_example" | "stale_in_env_example"


def _settings_field_env_names(parsed_files: list[ParsedFile]) -> set[str]:
    names: set[str] = set()
    for pf in parsed_files:
        for node, symbol in symbol_nodes(pf):
            if symbol.kind == SymbolKind.CLASS and symbol.name == SETTINGS_CLASS_NAME:
                names.update(field.name.upper() for field in class_field_annotations(node, pf.source_bytes))
    return names


def _env_example_var_names(env_example_text: str) -> set[str]:
    names: set[str] = set()
    for line in env_example_text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        var_name = stripped.split("=", 1)[0].strip()
        if var_name:
            names.add(var_name)
    return names


def find_config_drift(parsed_files: list[ParsedFile], env_example_text: str) -> list[ConfigDriftFinding]:
    """Structural set comparison, both directions (§6): a `Settings` field
    with no `.env.example` entry (`missing_from_env_example`), and a stale
    `.env.example` entry for a since-removed `Settings` field
    (`stale_in_env_example`) — drift in either direction is real drift,
    not just "missing docs."
    """
    settings_vars = _settings_field_env_names(parsed_files)
    env_vars = _env_example_var_names(env_example_text)

    findings = [ConfigDriftFinding(env_var=var, kind="missing_from_env_example") for var in sorted(settings_vars - env_vars)]
    findings += [ConfigDriftFinding(env_var=var, kind="stale_in_env_example") for var in sorted(env_vars - settings_vars)]
    return findings
