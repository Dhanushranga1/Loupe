"""Tests for loupe_mcp_server/doctor.py (docs/progress/loupe-doctor/steps/01-doctor-command.md).

Pure unit tests against a hand-built LoupeConfig -- run_doctor_checks takes
an already-loaded config, so most checks need no real manifest file or index
on disk at all.
"""

from pathlib import Path

from loupe_mcp_server.config import ExperimentalConfig, IndexConfig, LoupeConfig, TokenBudgetConfig
from loupe_mcp_server.doctor import run_doctor_checks
from loupe_mcp_server.config import INDEX_SCHEMA_VERSION


def _config(repo_root: Path, **overrides) -> LoupeConfig:
    defaults = dict(
        repo_root=repo_root,
        compute_profile="cpu_small",
        embedding_model="auto",
        cross_encoder_model="auto",
    )
    defaults.update(overrides)
    return LoupeConfig(
        token_budget=TokenBudgetConfig(),
        index=IndexConfig(),
        experimental=ExperimentalConfig(),
        **defaults,
    )


def _write_index(loupe_dir: Path, schema_version: str = str(INDEX_SCHEMA_VERSION)) -> None:
    loupe_dir.mkdir(parents=True, exist_ok=True)
    (loupe_dir / "schema_version").write_text(schema_version)


def test_embedding_model_missing_org_prefix_is_a_warning(tmp_path):
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, embedding_model="bge-small-en-v1.5")

    findings = run_doctor_checks(tmp_path, config)

    embedding_findings = [f for f in findings if f.check == "embedding_model"]
    assert len(embedding_findings) == 1
    assert embedding_findings[0].level == "warning"
    assert "bge-small-en-v1.5" in embedding_findings[0].message


def test_embedding_model_auto_is_not_flagged(tmp_path):
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, embedding_model="auto")

    findings = run_doctor_checks(tmp_path, config)

    assert not [f for f in findings if f.check == "embedding_model"]


def test_embedding_model_with_org_prefix_is_not_flagged(tmp_path):
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, embedding_model="BAAI/bge-small-en-v1.5")

    findings = run_doctor_checks(tmp_path, config)

    assert not [f for f in findings if f.check == "embedding_model"]


def test_unknown_compute_profile_is_an_error(tmp_path):
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, compute_profile="nonexistent_profile")

    findings = run_doctor_checks(tmp_path, config)

    profile_findings = [f for f in findings if f.check == "compute_profile"]
    assert len(profile_findings) == 1
    assert profile_findings[0].level == "error"


def test_known_compute_profile_is_ok(tmp_path):
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, compute_profile="cpu_medium")

    findings = run_doctor_checks(tmp_path, config)

    profile_findings = [f for f in findings if f.check == "compute_profile"]
    assert len(profile_findings) == 1
    assert profile_findings[0].level == "ok"


def test_gpu_large_without_a_detected_gpu_is_a_warning(tmp_path, monkeypatch):
    import loupe_mcp_server.doctor as doctor_module

    monkeypatch.setattr(doctor_module, "detect_gpu", lambda: False)
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, compute_profile="gpu_large")

    findings = run_doctor_checks(tmp_path, config)

    gpu_findings = [f for f in findings if f.check == "gpu_profile"]
    assert len(gpu_findings) == 1
    assert gpu_findings[0].level == "warning"


def test_gpu_large_with_a_detected_gpu_is_not_flagged(tmp_path, monkeypatch):
    import loupe_mcp_server.doctor as doctor_module

    monkeypatch.setattr(doctor_module, "detect_gpu", lambda: True)
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, compute_profile="gpu_large")

    findings = run_doctor_checks(tmp_path, config)

    assert not [f for f in findings if f.check == "gpu_profile"]


def test_cpu_small_profile_never_triggers_the_gpu_check_regardless_of_hardware(tmp_path, monkeypatch):
    import loupe_mcp_server.doctor as doctor_module

    monkeypatch.setattr(doctor_module, "detect_gpu", lambda: False)
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path, compute_profile="cpu_small")

    findings = run_doctor_checks(tmp_path, config)

    assert not [f for f in findings if f.check == "gpu_profile"]


def test_stale_index_schema_is_a_warning(tmp_path):
    _write_index(tmp_path / ".loupe", schema_version="0")
    config = _config(tmp_path)

    findings = run_doctor_checks(tmp_path, config)

    schema_findings = [f for f in findings if f.check == "index_schema"]
    assert len(schema_findings) == 1
    assert schema_findings[0].level == "warning"


def test_current_index_schema_is_ok(tmp_path):
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path)

    findings = run_doctor_checks(tmp_path, config)

    schema_findings = [f for f in findings if f.check == "index_schema"]
    assert len(schema_findings) == 1
    assert schema_findings[0].level == "ok"


def test_ranker_status_reports_cold_start_when_no_ranker_file_exists(tmp_path):
    _write_index(tmp_path / ".loupe")
    config = _config(tmp_path)

    findings = run_doctor_checks(tmp_path, config)

    ranker_findings = [f for f in findings if f.check == "ranker"]
    assert len(ranker_findings) == 1
    assert ranker_findings[0].level == "ok"
    assert "cold-start" in ranker_findings[0].message
