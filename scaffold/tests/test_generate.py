"""Tests for generate.py — the elicitation -> compose -> manifest/README pipeline
(docs/loupe-scaffold.md §4 steps 7-8, §5's compose/doc-generation acceptance criteria)."""

import tomllib

import yaml

from loupe_scaffold.generate import build_context, generate_project_files, generate_readme

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


def test_build_context_derives_a_kebab_case_package_name_from_the_project_name():
    context = build_context({"project_name": "Demo API!!"})
    assert context["package_name"] == "demo-api"


def test_generated_pyproject_toml_is_valid_toml_with_every_bricks_dependency():
    files = generate_project_files(MEDIUM_COMPLEXITY_ANSWERS)
    parsed = tomllib.loads(files["pyproject.toml"])

    assert parsed["project"]["name"] == "demo-api"
    deps = parsed["project"]["dependencies"]
    assert any(d.startswith("fastapi") for d in deps)
    assert any(d.startswith("sqlalchemy") for d in deps)
    assert any(d.startswith("python-jose") for d in deps)


def test_generated_docker_compose_yml_is_valid_yaml_with_app_and_db_services():
    files = generate_project_files(MEDIUM_COMPLEXITY_ANSWERS)
    parsed = yaml.safe_load(files["docker-compose.yml"])

    assert set(parsed["services"]) == {"app", "db"}


def test_generated_manifest_and_loupeignore_are_present_and_indexable_looking():
    files = generate_project_files(MEDIUM_COMPLEXITY_ANSWERS)

    assert "schema_version:" in files["loupe.manifest.yaml"]
    assert "languages:" in files["loupe.manifest.yaml"]
    assert files[".loupeignore"].strip() != ""


def test_readme_names_the_actual_selected_stack_and_omits_what_was_not_selected():
    """§5's end-to-end acceptance criterion, checked directly."""
    context = build_context(MEDIUM_COMPLEXITY_ANSWERS)
    from loupe_scaffold.bricks import BRICK_REGISTRY
    from loupe_scaffold.compose import select_bricks

    bricks = select_bricks(MEDIUM_COMPLEXITY_ANSWERS, BRICK_REGISTRY)
    readme = generate_readme(context, bricks)

    assert "PostgreSQL" in readme
    assert "JWT authentication" in readme
    assert "Docker Compose" in readme
    assert "Celery" not in readme
    assert "OAuth" not in readme
    assert "Kubernetes" not in readme


def test_readme_with_no_optional_bricks_selected_still_generates_something_sane():
    minimal_answers = {
        "project_name": "Bare App",
        "one_line_purpose": "",
        "database": "none",
        "auth_strategy": "none",
        "background_work": "none",
        "deployment_target": "bare_uvicorn",
        "testing_depth": "basic",
        "observability_level": "minimal",
    }
    context = build_context(minimal_answers)
    from loupe_scaffold.bricks import BRICK_REGISTRY
    from loupe_scaffold.compose import select_bricks

    bricks = select_bricks(minimal_answers, BRICK_REGISTRY)
    readme = generate_readme(context, bricks)

    assert "Bare App" in readme
    assert "PostgreSQL" not in readme
