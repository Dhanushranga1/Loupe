"""Tests for app/config.py (docs/loupe-target-project-standard.md §3/§6)."""

from pathlib import Path

from loupe_mcp_server.config import DEFAULT_HARD_CEILING, DEFAULT_TOKEN_BUDGET, load_config


def test_missing_manifest_and_global_config_uses_documented_defaults(tmp_path):
    cfg = load_config(tmp_path, global_config_path=tmp_path / "nonexistent-global.yaml")
    assert cfg.languages == ["python"]
    assert cfg.token_budget.default_per_turn == DEFAULT_TOKEN_BUDGET
    assert cfg.token_budget.hard_ceiling == DEFAULT_HARD_CEILING
    assert cfg.index.symbol_kinds == ["function", "class", "method"]


def test_project_manifest_overrides_defaults(tmp_path):
    (tmp_path / "loupe.manifest.yaml").write_text(
        "languages: [python, typescript]\n"
        "token_budget:\n"
        "  default_per_turn: 4000\n"
        "embedding_model: custom-model\n"
    )
    cfg = load_config(tmp_path, global_config_path=tmp_path / "nonexistent-global.yaml")
    assert cfg.languages == ["python", "typescript"]
    assert cfg.token_budget.default_per_turn == 4000
    assert cfg.token_budget.hard_ceiling == DEFAULT_HARD_CEILING, "unset fields must still fall back to defaults"
    assert cfg.embedding_model == "custom-model"


def test_project_manifest_wins_over_global_config(tmp_path):
    global_path = tmp_path / "global.yaml"
    global_path.write_text("token_budget:\n  default_per_turn: 3000\n  hard_ceiling: 15000\n")

    project_dir = tmp_path / "repo"
    project_dir.mkdir()
    (project_dir / "loupe.manifest.yaml").write_text("token_budget:\n  default_per_turn: 9000\n")

    cfg = load_config(project_dir, global_config_path=global_path)
    assert cfg.token_budget.default_per_turn == 9000, "project overrides global on conflict"
    assert cfg.token_budget.hard_ceiling == 15000, "global value inherited where project doesn't override"


def test_a_project_with_only_languages_set_inherits_everything_else(tmp_path):
    (tmp_path / "loupe.manifest.yaml").write_text("languages: [python]\n")
    cfg = load_config(tmp_path, global_config_path=tmp_path / "nonexistent-global.yaml")
    assert cfg.embedding_model == "auto"  # compute-profiles.md: resolved from compute_profile, not a fixed name
    assert cfg.compute_profile == "cpu_small"
    assert cfg.token_budget.default_per_turn == DEFAULT_TOKEN_BUDGET


def test_compute_profile_and_explicit_overrides_are_parsed(tmp_path):
    (tmp_path / "loupe.manifest.yaml").write_text(
        "compute_profile: gpu_large\nembedding_model: some-custom-model\ncross_encoder_model: auto\n"
    )
    cfg = load_config(tmp_path, global_config_path=tmp_path / "nonexistent-global.yaml")
    assert cfg.compute_profile == "gpu_large"
    assert cfg.embedding_model == "some-custom-model"
    assert cfg.cross_encoder_model == "auto"


def test_manifest_exclude_paths_and_packages(tmp_path):
    (tmp_path / "loupe.manifest.yaml").write_text(
        "index:\n"
        "  exclude_paths: ['**/migrations/**']\n"
        "packages:\n"
        "  - name: api\n"
        "    root: services/api\n"
    )
    cfg = load_config(tmp_path, global_config_path=tmp_path / "nonexistent-global.yaml")
    assert cfg.index.exclude_paths == ["**/migrations/**"]
    assert cfg.packages == [{"name": "api", "root": "services/api"}]
