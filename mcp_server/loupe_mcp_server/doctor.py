"""`loupe doctor`: health-check for a repo's Loupe setup (docs/progress/loupe-doctor/steps/01-doctor-command.md).

Detection, not enforcement — same boundary `find_code_smells`/`cmd_check`
already draw: every check here only ever reports, never fixes or blocks.
Direct response to a real bug found by hand this session, not a
hypothetical: this repo's own `loupe.manifest.yaml` had an `embedding_model`
missing its HuggingFace org prefix, which broke fresh indexing with a
misleading 401 instead of a clear error. Check 2 below is that bug's
class, not just that one instance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from .compute_profiles import AUTO, COMPUTE_PROFILES, detect_gpu
from .config import INDEX_SCHEMA_VERSION, LoupeConfig

Level = Literal["ok", "warning", "error"]


@dataclass(frozen=True)
class DoctorFinding:
    level: Level
    check: str
    message: str


def _check_model_override_looks_plausible(check: str, field_name: str, value: str) -> DoctorFinding | None:
    """A real HuggingFace model id has an org/model shape (`BAAI/bge-small-en-v1.5`).
    Not a network call to verify the model actually exists — that's what
    indexing itself will tell you, expensively; this is a cheap, offline,
    catch-the-obvious-typo check for the exact shape of bug this command
    was built in response to."""
    if value == AUTO:
        return None
    if "/" not in value:
        return DoctorFinding(
            level="warning",
            check=check,
            message=(
                f"{field_name}: {value!r} doesn't look like a real HuggingFace model id "
                f"(missing an org/ prefix, e.g. 'BAAI/{value}'). Set to \"auto\" to use the "
                f"compute profile's own default instead."
            ),
        )
    return None


def _check_compute_profile_known(config: LoupeConfig) -> DoctorFinding:
    if config.compute_profile not in COMPUTE_PROFILES:
        known = ", ".join(sorted(COMPUTE_PROFILES))
        return DoctorFinding(
            level="error",
            check="compute_profile",
            message=f"compute_profile: {config.compute_profile!r} is not a known profile (known: {known}).",
        )
    return DoctorFinding(level="ok", check="compute_profile", message=config.compute_profile)


def _check_gpu_profile_has_gpu(config: LoupeConfig) -> DoctorFinding | None:
    if config.compute_profile == "gpu_large" and not detect_gpu():
        return DoctorFinding(
            level="warning",
            check="gpu_profile",
            message="compute_profile is gpu_large but no GPU was detected on this machine — indexing will be slow.",
        )
    return None


def _check_index_schema_current(loupe_dir: Path) -> DoctorFinding:
    schema_path = loupe_dir / "schema_version"
    current = schema_path.read_text().strip() if schema_path.exists() else None
    if current is None:
        return DoctorFinding(level="error", check="index_schema", message="no schema_version file found in .loupe/")
    if current != str(INDEX_SCHEMA_VERSION):
        return DoctorFinding(
            level="warning",
            check="index_schema",
            message=f"index schema is {current}, current is {INDEX_SCHEMA_VERSION} — run `loupe index` to reindex.",
        )
    return DoctorFinding(level="ok", check="index_schema", message=f"index schema up to date ({current})")


def _check_ranker_status(loupe_dir: Path) -> DoctorFinding:
    from loupe_core.retrieval.ranker import Ranker

    ranker_path = loupe_dir / "ranker.pkl"
    ranker = Ranker.load(str(ranker_path))
    if ranker.is_trained:
        return DoctorFinding(level="ok", check="ranker", message="learned ranker: trained")
    return DoctorFinding(level="ok", check="ranker", message="learned ranker: not trained (cold-start — using RRF)")


def run_doctor_checks(repo_root: Path, config: LoupeConfig) -> list[DoctorFinding]:
    """Pure aggregator: takes an already-loaded config (matching `cmd_check`'s
    own pattern of loading config once in the CLI layer), so this is directly
    unit-testable against a hand-built `LoupeConfig` without a real manifest
    file on disk for most checks."""
    loupe_dir = repo_root / ".loupe"
    findings: list[DoctorFinding] = []

    embedding_finding = _check_model_override_looks_plausible(
        "embedding_model", "embedding_model", config.embedding_model
    )
    if embedding_finding is not None:
        findings.append(embedding_finding)

    cross_encoder_finding = _check_model_override_looks_plausible(
        "cross_encoder_model", "cross_encoder_model", config.cross_encoder_model
    )
    if cross_encoder_finding is not None:
        findings.append(cross_encoder_finding)

    findings.append(_check_compute_profile_known(config))

    gpu_finding = _check_gpu_profile_has_gpu(config)
    if gpu_finding is not None:
        findings.append(gpu_finding)

    findings.append(_check_index_schema_current(loupe_dir))
    findings.append(_check_ranker_status(loupe_dir))

    return findings
