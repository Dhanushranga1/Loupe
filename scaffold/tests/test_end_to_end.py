"""End-to-end acceptance test (docs/loupe-scaffold.md §5's "End-to-end" bullets,
§6 build-order step 3's "proven full path"): generate the medium-complexity
combination for real, onto a real temp directory, and prove it actually
starts — not by inspecting template output, but by running a fresh Python
subprocess (this venv's own interpreter, which has every generated
dependency installed — see pyproject.toml's dev extras) that imports
`app.main` exactly the way `uvicorn app.main:app` would at boot, plus
running the generated project's own generated test suite against it.
"""

import subprocess
import sys
from pathlib import Path

from loupe_scaffold.generate import write_project

MEDIUM_COMPLEXITY_ANSWERS = {
    "project_name": "Demo API",
    "one_line_purpose": "a demo service",
    "database": "postgresql",
    "orm_choice": "sqlalchemy_async",
    "migrations_needed": "no",
    "auth_strategy": "jwt_password",
    "background_work": "none",
    "deployment_target": "docker_compose",
    "testing_depth": "basic",
    "observability_level": "minimal",
}


def test_generated_project_has_a_valid_manifest_and_loupeignore(tmp_path):
    write_project(tmp_path, MEDIUM_COMPLEXITY_ANSWERS)

    assert (tmp_path / "loupe.manifest.yaml").exists()
    assert (tmp_path / ".loupeignore").exists()


def test_generated_project_imports_without_error_no_live_database_needed(tmp_path):
    """§5's core end-to-end criterion: the app actually starts. SQLAlchemy's
    `create_async_engine()` doesn't open a real connection until first use,
    so this succeeds with zero live Postgres — the same reason `uvicorn
    app.main:app` itself wouldn't need one just to boot.
    """
    write_project(tmp_path, MEDIUM_COMPLEXITY_ANSWERS)

    result = subprocess.run(
        [sys.executable, "-c", "import app.main; assert app.main.app.title == 'Demo API'"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_generated_projects_own_test_suite_passes(tmp_path):
    """The generated `tests/test_health.py` (from the `testing_basic` brick)
    must itself pass when run for real, using FastAPI's TestClient — an
    in-process ASGI call, no live server or network needed."""
    write_project(tmp_path, MEDIUM_COMPLEXITY_ANSWERS)

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "-q"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 passed" in result.stdout


def test_generated_project_readme_does_not_reference_unselected_features(tmp_path):
    write_project(tmp_path, MEDIUM_COMPLEXITY_ANSWERS)

    readme = (tmp_path / "README.md").read_text()
    assert "Celery" not in readme
    assert "Kubernetes" not in readme
    assert "OAuth" not in readme


def test_generating_twice_into_a_fresh_directory_produces_byte_identical_output(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    write_project(first_dir, MEDIUM_COMPLEXITY_ANSWERS)
    write_project(second_dir, MEDIUM_COMPLEXITY_ANSWERS)

    first_files = {p.relative_to(first_dir): p.read_bytes() for p in first_dir.rglob("*") if p.is_file()}
    second_files = {p.relative_to(second_dir): p.read_bytes() for p in second_dir.rglob("*") if p.is_file()}
    assert first_files == second_files
