"""Tests for experimental_gate.py (docs/PhaseX/experimental-gate-and-hyde.md, Part 1)."""

import json

import pytest

from loupe_mcp_server.config import ExperimentalConfig, LoupeConfig
from loupe_mcp_server.experimental_gate import (
    UnknownExperimentalFeature,
    is_experimental_feature_enabled,
    log_experimental_usage,
    modeled_cost_for,
    total_logged_tokens,
)


def _config(tmp_path, llm_assist: bool, features: dict) -> LoupeConfig:
    return LoupeConfig(repo_root=tmp_path, experimental=ExperimentalConfig(llm_assist=llm_assist, features=features))


def test_feature_disabled_when_master_switch_and_feature_flag_are_both_off(tmp_path):
    config = _config(tmp_path, llm_assist=False, features={})
    assert is_experimental_feature_enabled(config, "hyde_query_rewrite") is False


def test_feature_disabled_when_master_switch_off_even_if_feature_flag_on(tmp_path):
    """§1's whole point: the category switch gates everything under it, so a
    stray per-feature `true` alone can never turn on real spend."""
    config = _config(tmp_path, llm_assist=False, features={"hyde_query_rewrite": True})
    assert is_experimental_feature_enabled(config, "hyde_query_rewrite") is False


def test_feature_disabled_when_master_switch_on_but_feature_flag_off(tmp_path):
    config = _config(tmp_path, llm_assist=True, features={"hyde_query_rewrite": False})
    assert is_experimental_feature_enabled(config, "hyde_query_rewrite") is False


def test_feature_enabled_only_when_both_levels_are_on(tmp_path):
    config = _config(tmp_path, llm_assist=True, features={"hyde_query_rewrite": True})
    assert is_experimental_feature_enabled(config, "hyde_query_rewrite") is True


def test_unset_feature_flag_defaults_to_disabled(tmp_path):
    config = _config(tmp_path, llm_assist=True, features={})
    assert is_experimental_feature_enabled(config, "hyde_query_rewrite") is False


def test_modeled_cost_for_known_feature_is_a_positive_token_estimate():
    modeled = modeled_cost_for("hyde_query_rewrite")
    assert modeled.estimated_tokens > 0
    assert "hyde" in modeled.description.lower() or "hypothetical" in modeled.description.lower()


def test_modeled_cost_for_unknown_feature_raises():
    with pytest.raises(UnknownExperimentalFeature):
        modeled_cost_for("not_a_real_feature")


def test_log_experimental_usage_writes_to_its_own_feature_file_not_retrieval_logs(tmp_path):
    loupe_dir = tmp_path / ".loupe"
    log_experimental_usage(loupe_dir, "hyde_query_rewrite", 123, cost_estimate_type="measured", query="foo")

    log_path = loupe_dir / "logs" / "experimental" / "hyde_query_rewrite.jsonl"
    assert log_path.exists()
    assert not (loupe_dir / "logs" / "retrieval").exists()

    entry = json.loads(log_path.read_text().splitlines()[0])
    assert entry["feature"] == "hyde_query_rewrite"
    assert entry["tokens"] == 123
    assert entry["cost_estimate_type"] == "measured"
    assert entry["detail"] == {"query": "foo"}


def test_log_experimental_usage_appends_and_keeps_features_in_separate_files(tmp_path):
    loupe_dir = tmp_path / ".loupe"
    log_experimental_usage(loupe_dir, "hyde_query_rewrite", 100)
    log_experimental_usage(loupe_dir, "hyde_query_rewrite", 50)
    log_experimental_usage(loupe_dir, "other_feature", 999)

    hyde_lines = (loupe_dir / "logs" / "experimental" / "hyde_query_rewrite.jsonl").read_text().splitlines()
    assert len(hyde_lines) == 2

    other_lines = (loupe_dir / "logs" / "experimental" / "other_feature.jsonl").read_text().splitlines()
    assert len(other_lines) == 1


def test_total_logged_tokens_sums_every_entry_for_that_feature(tmp_path):
    loupe_dir = tmp_path / ".loupe"
    log_experimental_usage(loupe_dir, "hyde_query_rewrite", 100)
    log_experimental_usage(loupe_dir, "hyde_query_rewrite", 50)

    assert total_logged_tokens(loupe_dir, "hyde_query_rewrite") == 150


def test_total_logged_tokens_is_zero_for_a_feature_never_used(tmp_path):
    assert total_logged_tokens(tmp_path / ".loupe", "hyde_query_rewrite") == 0
