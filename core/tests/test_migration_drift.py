"""Tests for adapters/fastapi/migration_drift.py (docs/PhaseX/zero-cost-static-analysis-pack.md E8)."""

import os
from pathlib import Path

from loupe_core.adapters.fastapi.migration_drift import extract_model_fields, find_migration_drift
from loupe_core.graph.builder import parse_file

MODEL_SOURCE = 'class Item:\n    __tablename__ = "items"\n    id: int\n    name: str\n'

MIGRATION_CONTENT = """
def upgrade():
    op.create_table(
        'items',
        sa.Column('id', sa.Integer(), primary_key=True),
        sa.Column('name', sa.String(), nullable=False),
    )
"""


def _parse(tmp_path: Path, source: str):
    f = tmp_path / "models.py"
    f.write_text(source)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return [parse_file("models.py")]
    finally:
        os.chdir(old_cwd)


def test_extract_model_fields_requires_tablename_marker(tmp_path):
    parsed = _parse(tmp_path, MODEL_SOURCE + '\n\nclass NotAModel:\n    x: int\n')
    models = extract_model_fields(parsed)

    assert models == {"Item": {"id", "name"}}


def test_model_field_added_without_migration_is_flagged(tmp_path):
    """§6's own acceptance criterion: a model field added without a
    corresponding Alembic migration is flagged on a fixture constructed for
    exactly this case."""
    source = MODEL_SOURCE.replace("    name: str\n", "    name: str\n    price: float\n")
    parsed = _parse(tmp_path, source)

    findings = find_migration_drift(parsed, [MIGRATION_CONTENT])

    assert any(f.model_qualified_name == "Item" and f.field_name == "price" for f in findings)


def test_model_fully_matching_its_migration_produces_no_false_positive(tmp_path):
    """§6's own acceptance criterion: a model whose fields fully match its
    latest migration produces no false positive."""
    parsed = _parse(tmp_path, MODEL_SOURCE)

    findings = find_migration_drift(parsed, [MIGRATION_CONTENT])

    assert findings == []


def test_column_added_in_an_earlier_migration_not_just_the_latest_still_counts(tmp_path):
    """Migration history is cumulative — a column created in an *earlier*
    migration file (not the most recent one) must still count, since the
    model's real schema is the sum of its whole migration history."""
    source = MODEL_SOURCE.replace("    name: str\n", "    name: str\n    price: float\n")
    parsed = _parse(tmp_path, source)

    earlier_migration = "def upgrade():\n    op.add_column('items', sa.Column('price', sa.Float()))\n"

    findings = find_migration_drift(parsed, [MIGRATION_CONTENT, earlier_migration])

    assert findings == []


def test_class_without_tablename_marker_is_never_checked(tmp_path):
    parsed = _parse(tmp_path, "class PlainClass:\n    id: int\n    undocumented_field: str\n")
    findings = find_migration_drift(parsed, [MIGRATION_CONTENT])
    assert findings == []
