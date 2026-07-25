"""Tests for the compose engine (docs/loupe-scaffold.md §5 — Compose engine acceptance
criteria), against two minimal placeholder bricks per §6 build-order step 2 — proving the
*mechanism* works before any real brick content exists.
"""

import pytest

from loupe_scaffold.compose import Brick, ExclusiveFileCollisionError, check_exclusive_collisions, compose, select_bricks

CONTEXT = {"package_name": "demo_app", "one_line_purpose": "a demo project", "project_name": "Demo App"}


def _placeholder_bricks():
    brick_a = Brick(
        name="brick_a",
        exclusive_files={"app/a.py": "# owned by brick_a\n"},
        shared_contributions={"app/main.py": "# fragment from brick_a"},
        dependencies=["fastapi", "shared-dep"],
        env_vars={"A_VAR": "1"},
    )
    brick_b = Brick(
        name="brick_b",
        exclusive_files={"app/b.py": "# owned by brick_b\n"},
        shared_contributions={"app/main.py": "# fragment from brick_b"},
        dependencies=["uvicorn", "shared-dep"],
        env_vars={"B_VAR": "2"},
    )
    return brick_a, brick_b


def test_two_bricks_each_contributing_dependencies_merge_deduplicated_with_nothing_dropped():
    brick_a, brick_b = _placeholder_bricks()
    files = compose([brick_a, brick_b], CONTEXT)

    assert '"fastapi",' in files["pyproject.toml"]
    assert '"uvicorn",' in files["pyproject.toml"]
    assert files["pyproject.toml"].count('"shared-dep",') == 1, "a dependency named by both bricks must appear once"


def test_two_bricks_claiming_the_same_exclusive_path_raises_a_named_error():
    brick_a = Brick(name="brick_a", exclusive_files={"app/main.py": "a"})
    brick_b = Brick(name="brick_b", exclusive_files={"app/main.py": "b"})

    with pytest.raises(ExclusiveFileCollisionError) as exc_info:
        check_exclusive_collisions([brick_a, brick_b])

    assert exc_info.value.path == "app/main.py"
    assert {exc_info.value.brick_a, exc_info.value.brick_b} == {"brick_a", "brick_b"}


def test_compose_raises_the_same_collision_error_not_a_silent_overwrite():
    brick_a = Brick(name="brick_a", exclusive_files={"app/main.py": "a"})
    brick_b = Brick(name="brick_b", exclusive_files={"app/main.py": "b"})

    with pytest.raises(ExclusiveFileCollisionError):
        compose([brick_a, brick_b], CONTEXT)


def test_shared_file_composition_is_deterministic_across_repeated_runs():
    brick_a, brick_b = _placeholder_bricks()

    first = compose([brick_b, brick_a], CONTEXT)  # deliberately passed out of alphabetical order
    second = compose([brick_a, brick_b], CONTEXT)

    assert first == second, "identical bricks + context must produce byte-identical output regardless of input order"


def test_shared_file_fragments_are_concatenated_in_alphabetical_order_by_brick_name():
    brick_a, brick_b = _placeholder_bricks()
    files = compose([brick_b, brick_a], CONTEXT)

    assert files["app/main.py"].index("brick_a") < files["app/main.py"].index("brick_b")


def test_exclusive_files_are_rendered_directly_and_do_not_collide_across_bricks():
    brick_a, brick_b = _placeholder_bricks()
    files = compose([brick_a, brick_b], CONTEXT)

    assert files["app/a.py"] == "# owned by brick_a\n"
    assert files["app/b.py"] == "# owned by brick_b\n"


def test_env_vars_from_every_brick_are_merged_into_one_env_example():
    brick_a, brick_b = _placeholder_bricks()
    files = compose([brick_a, brick_b], CONTEXT)

    assert "A_VAR=1" in files[".env.example"]
    assert "B_VAR=2" in files[".env.example"]


def test_select_bricks_uses_an_explicit_lookup_table_not_inference():
    registry = {b.name: b for b in _placeholder_bricks()} | {
        "core_app": Brick(name="core_app"),
        "db_postgresql": Brick(name="db_postgresql"),
    }
    selected = select_bricks({"database": "postgresql"}, registry)

    assert {b.name for b in selected} == {"core_app", "db_postgresql"}


def test_shared_bases_pin_a_brick_first_even_when_it_sorts_after_alphabetically():
    """The real refinement found while building bricks against this
    mechanism: a header-owning brick must lead the concatenation regardless
    of its name's alphabetical position relative to the other contributors."""
    header_brick = Brick(name="z_header_brick", shared_contributions={"app/config.py": "HEADER"}, shared_bases=frozenset({"app/config.py"}))
    body_brick = Brick(name="a_body_brick", shared_contributions={"app/config.py": "BODY"})

    files = compose([body_brick, header_brick], CONTEXT)

    assert files["app/config.py"].index("HEADER") < files["app/config.py"].index("BODY")


def test_config_py_gets_a_get_settings_closer_appended_after_all_contributions():
    header_brick = Brick(name="core_app", shared_contributions={"app/config.py": "class Settings:\n    x: int = 1"}, shared_bases=frozenset({"app/config.py"}))

    files = compose([header_brick], CONTEXT)

    assert "def get_settings() -> Settings:" in files["app/config.py"]
    assert files["app/config.py"].index("x: int = 1") < files["app/config.py"].index("def get_settings")


def test_select_bricks_silently_skips_a_mapped_name_not_yet_in_the_registry():
    """§2's explicit scope note: the catalog can grow incrementally — a
    mapped brick name that doesn't exist yet must not raise, only be absent
    from the activated set."""
    registry = {"core_app": Brick(name="core_app")}
    selected = select_bricks({"database": "mongodb"}, registry)

    assert {b.name for b in selected} == {"core_app"}
