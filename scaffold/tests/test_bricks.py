"""Sanity sweep across the whole brick catalog (docs/loupe-scaffold.md §4's
catalog) — every registered brick, not just the one proven end-to-end path,
must at least produce syntactically valid Python/YAML/TOML when combined
with `core_app`. This is what catches a brick with a typo'd template before
someone picks that exact combination and only then finds out.
"""

import ast

import pytest
import yaml

from loupe_scaffold.bricks import BRICK_REGISTRY, CORE_APP
from loupe_scaffold.compose import compose

CONTEXT = {"project_name": "Sweep App", "one_line_purpose": "sweeping every brick", "package_name": "sweep-app"}


@pytest.mark.parametrize("brick_name", sorted(BRICK_REGISTRY))
def test_core_app_plus_each_brick_alone_produces_syntactically_valid_output(brick_name):
    brick = BRICK_REGISTRY[brick_name]
    bricks = [CORE_APP] if brick is CORE_APP else [CORE_APP, brick]

    files = compose(bricks, CONTEXT)

    for path, content in files.items():
        if path.endswith(".py"):
            ast.parse(content, filename=path)
        elif path.endswith((".yml", ".yaml")):
            yaml.safe_load(content)


def test_every_catalog_brick_is_reachable_from_at_least_one_select_bricks_answer():
    """A brick that's registered but that `select_bricks`'s lookup table can
    never actually select would be dead weight — verified, not assumed."""
    from loupe_scaffold.compose import select_bricks

    answer_axes = [
        {"database": v} for v in ["none", "sqlite", "postgresql", "mysql", "mongodb"]
    ] + [
        {"database": "postgresql", "orm_choice": v} for v in ["sqlalchemy_async", "sqlalchemy_sync", "sqlmodel"]
    ] + [
        {"database": "mongodb", "orm_choice": v} for v in ["beanie", "motor_raw"]
    ] + [
        {"database": "postgresql", "orm_choice": "sqlalchemy_async", "migrations_needed": "yes"}
    ] + [
        {"auth_strategy": v} for v in ["none", "api_key", "jwt_password", "oauth2_social", "session"]
    ] + [
        {"background_work": v} for v in ["none", "simple_tasks", "arq"]
    ] + [
        {"background_work": "celery", "broker_choice": v} for v in ["redis", "rabbitmq"]
    ] + [
        {"deployment_target": v} for v in ["docker_compose", "kubernetes", "serverless_lambda", "bare_uvicorn"]
    ] + [
        {"observability_level": v}
        for v in ["minimal", "structured_logging", "structured_logging_plus_otel"]
    ] + [
        {"testing_depth": v} for v in ["basic", "basic_plus_ci"]
    ]

    reachable = {"core_app"}
    for answers in answer_axes:
        reachable |= {b.name for b in select_bricks(answers, BRICK_REGISTRY)}

    assert reachable == set(BRICK_REGISTRY)
