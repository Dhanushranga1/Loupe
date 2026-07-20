"""E8 — Migration drift detection (docs/PhaseX/zero-cost-static-analysis-pack.md).

Applicable when a project has SQLAlchemy/SQLModel models and Alembic
migrations (Scaffold's `migrations_alembic` brick). Reuses
`class_field_annotations` for model fields — the same shared
field-extraction helper E7/E9 use.

Two heuristics, pattern-based rather than real type/DB inspection, matching
`adapters/fastapi/routes.py`'s own documented philosophy ("a decorator-shape
heuristic, not framework awareness"):

1. **Model detection:** a class is treated as an ORM table model if its body
   declares `__tablename__` — the near-universal structural marker for both
   SQLAlchemy declarative models and SQLModel table models, requiring no
   base-class type resolution.
2. **Migration parsing:** not a real Alembic op interpreter. `op.add_column`/
   `Column(...)` calls are matched by their literal column-name string
   argument, scanned across *every* migration file in the given directory —
   not just the most recent one, since a model's real schema is the
   cumulative result of its whole migration history, not any single file.

Honest limitation, not hidden: column names are tracked globally across all
migrations, not scoped per table (no `__tablename__`-to-migration
resolution) — two different tables sharing a column name could mask a real
gap. Acceptable imprecision for a zero-cost, no-DB-connection check; a
`.env.example`-style false-negative risk, the same category of scope limit
E7's alias-mapping gap already accepts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from loupe_core.graph.builder import ParsedFile
from loupe_core.parsing.ast_utils import class_field_annotations, symbol_nodes
from loupe_core.parsing.schema import SymbolKind

_COLUMN_NAME_PATTERN = re.compile(r"Column\(\s*[\"'](\w+)[\"']")


@dataclass(frozen=True)
class MigrationDriftFinding:
    model_qualified_name: str
    field_name: str


def _has_tablename_marker(class_node, source_bytes: bytes) -> bool:
    body = class_node.child_by_field_name("body")
    if body is None:
        return False
    for stmt in body.children:
        if stmt.type != "expression_statement" or not stmt.children:
            continue
        assignment = stmt.children[0]
        if assignment.type != "assignment":
            continue
        left = assignment.child_by_field_name("left")
        if left is not None and left.type == "identifier":
            name = source_bytes[left.start_byte : left.end_byte].decode("utf-8")
            if name == "__tablename__":
                return True
    return False


def extract_model_fields(parsed_files: list[ParsedFile]) -> dict[str, set[str]]:
    """`{model_qualified_name: {field_name, ...}}` for every class with a
    `__tablename__` marker."""
    models: dict[str, set[str]] = {}
    for pf in parsed_files:
        for node, symbol in symbol_nodes(pf):
            if symbol.kind != SymbolKind.CLASS or not _has_tablename_marker(node, pf.source_bytes):
                continue
            fields = class_field_annotations(node, pf.source_bytes)
            models[symbol.qualified_name] = {f.name for f in fields}
    return models


def extract_migrated_column_names(migration_file_contents: list[str]) -> set[str]:
    """Every column name referenced via `Column("name", ...)` (covers both
    `sa.Column(...)` inside `op.add_column`/`op.create_table` and a bare
    `Column(...)` import style) across *all* given migration file contents.
    """
    names: set[str] = set()
    for content in migration_file_contents:
        names.update(_COLUMN_NAME_PATTERN.findall(content))
    return names


def find_migration_drift(parsed_files: list[ParsedFile], migration_file_contents: list[str]) -> list[MigrationDriftFinding]:
    """A model field with no corresponding migration (§6): its name never
    appears in any migration file's `Column(...)` calls across the whole
    migration history.
    """
    models = extract_model_fields(parsed_files)
    migrated_columns = extract_migrated_column_names(migration_file_contents)

    findings: list[MigrationDriftFinding] = []
    for model_name, fields in sorted(models.items()):
        for field in sorted(fields - migrated_columns):
            findings.append(MigrationDriftFinding(model_qualified_name=model_name, field_name=field))
    return findings
