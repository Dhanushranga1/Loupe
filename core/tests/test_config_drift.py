"""Tests for adapters/fastapi/config_drift.py (docs/PhaseX/zero-cost-static-analysis-pack.md E7)."""

import os
from pathlib import Path

from loupe_core.adapters.fastapi.config_drift import ConfigDriftFinding, find_config_drift
from loupe_core.graph.builder import parse_file


def _parse(tmp_path: Path, source: str):
    f = tmp_path / "config.py"
    f.write_text(source)
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        return [parse_file("config.py")]
    finally:
        os.chdir(old_cwd)


SETTINGS_SOURCE = "class Settings:\n    api_key: str\n    debug: bool = False\n"


def test_settings_field_with_no_env_example_entry_is_flagged(tmp_path):
    """§6's own acceptance criterion: a Settings field with no corresponding
    .env.example entry is flagged."""
    parsed = _parse(tmp_path, SETTINGS_SOURCE)
    findings = find_config_drift(parsed, env_example_text="DEBUG=false\n")

    missing = {f.env_var for f in findings if f.kind == "missing_from_env_example"}
    assert "API_KEY" in missing


def test_stale_env_example_entry_for_removed_field_is_flagged(tmp_path):
    """§6's own acceptance criterion: a stale .env.example entry for a
    since-removed Settings field is also flagged — drift in both
    directions matters, not just missing docs."""
    parsed = _parse(tmp_path, SETTINGS_SOURCE)
    findings = find_config_drift(parsed, env_example_text="API_KEY=\nDEBUG=false\nOLD_REMOVED_VAR=\n")

    stale = {f.env_var for f in findings if f.kind == "stale_in_env_example"}
    assert "OLD_REMOVED_VAR" in stale


def test_fully_synced_settings_and_env_example_produce_no_findings(tmp_path):
    parsed = _parse(tmp_path, SETTINGS_SOURCE)
    findings = find_config_drift(parsed, env_example_text="API_KEY=\nDEBUG=false\n")
    assert findings == []


def test_env_example_comments_and_blank_lines_are_ignored(tmp_path):
    parsed = _parse(tmp_path, SETTINGS_SOURCE)
    env_example = "# API credentials\nAPI_KEY=\n\nDEBUG=false\n# trailing comment\n"
    findings = find_config_drift(parsed, env_example_text=env_example)
    assert findings == []


def test_non_settings_classes_are_ignored(tmp_path):
    parsed = _parse(tmp_path, "class NotSettings:\n    something: str\n")
    findings = find_config_drift(parsed, env_example_text="")
    assert findings == []


def test_field_name_mapped_to_uppercase_env_var(tmp_path):
    parsed = _parse(tmp_path, "class Settings:\n    database_url: str\n")
    findings = find_config_drift(parsed, env_example_text="")
    assert findings == [ConfigDriftFinding(env_var="DATABASE_URL", kind="missing_from_env_example")]
