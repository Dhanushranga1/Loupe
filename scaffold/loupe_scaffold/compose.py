"""The brick/compose engine (docs/loupe-scaffold.md §4, §6 build-order step 2).

A `Brick` is an independent, self-contained unit of project structure.
`select_bricks` maps elicitation answers to the exact list of activated
bricks via a small, explicit lookup table (§4 step 1 — never inferred
logic); `compose` turns that list into a finished set of `{path: content}`
files (§4 steps 2-6). Doc generation (README) and manifest/`.loupeignore`
generation are deliberately *not* here — §4 steps 7-8 treat those as
generated directly by the caller, not bricks, and §6's build order puts
doc generation last, once there's real, varied output to describe.

Two categories of file, handled by genuinely different mechanisms, matching
§4's own distinction:
- **Exclusive files**: one brick owns the path outright, rendered directly.
  Two bricks claiming the same path is a build-time error, not a silent
  overwrite (checked before anything is rendered).
- **Shared/composed files** (`main.py`, `config.py`, `docker-compose.yml`):
  every contributing brick's fragment is concatenated in a fixed,
  deterministic order *before* the combined source is rendered as a single
  Jinja2 pass — concatenate-then-render, matching §4 step 4's literal
  wording, not render-then-concatenate.

  Ordering within that concatenation is alphabetical by brick name, with
  one real refinement found while actually building bricks against this
  mechanism (not in the original spec text, but required for the output to
  be valid code at all): most of these files have exactly one brick that
  defines the file's *header* — `core_app` opens `class Settings` in
  `config.py` and creates `app = FastAPI()` in `main.py`; `deploy_docker_compose`
  opens `services:` in `docker-compose.yml` — and every other contributing
  brick's fragment is body content that's only valid *after* that header
  (an indented Settings field, an `app.include_router(...)` call, a nested
  service block). Plain alphabetical order puts several of those body
  fragments before their own header (`auth_jwt` < `core_app`), which breaks
  syntactically. `Brick.shared_bases` names the paths a brick's fragment
  must lead for; the sort key is `(is_base, brick_name)` rather than just
  `brick_name`, so a base fragment always sorts first for its own path,
  alphabetical among the rest.

`pyproject.toml`'s dependency list and `.env.example` are handled
differently still (§4 steps 5-6): each brick's `dependencies`/`env_vars`
are plain structured data, not a text fragment — text-fragment
concatenation can't produce valid TOML when multiple bricks each want to
contribute to one `dependencies = [...]` array, so those two files are
assembled directly from the merged, deduplicated structured data instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import jinja2

from loupe_core.adapters.fastapi.convention_categories import ConventionCategory


@dataclass(frozen=True)
class Brick:
    name: str
    exclusive_files: dict[str, str] = field(default_factory=dict)
    shared_contributions: dict[str, str] = field(default_factory=dict)
    dependencies: list[str] = field(default_factory=list)
    env_vars: dict[str, str] = field(default_factory=dict)
    category: ConventionCategory | None = None
    # Paths (a subset of this brick's own `shared_contributions` keys) whose
    # fragment must be concatenated first among that path's contributors —
    # see this module's docstring for why plain alphabetical order alone
    # isn't enough for files with a header/body structure.
    shared_bases: frozenset[str] = frozenset()


class ExclusiveFileCollisionError(Exception):
    def __init__(self, path: str, brick_a: str, brick_b: str) -> None:
        self.path = path
        self.brick_a = brick_a
        self.brick_b = brick_b
        super().__init__(f"both {brick_a!r} and {brick_b!r} claim exclusive file {path!r}")


def _render(template_source: str, context: dict[str, Any]) -> str:
    return jinja2.Template(template_source, keep_trailing_newline=True).render(**context)


def check_exclusive_collisions(bricks: list[Brick]) -> None:
    """§4 step 2: fail loudly, naming both bricks, before anything renders."""
    owner_by_path: dict[str, str] = {}
    for brick in bricks:
        for path in brick.exclusive_files:
            if path in owner_by_path:
                raise ExclusiveFileCollisionError(path, owner_by_path[path], brick.name)
            owner_by_path[path] = brick.name


def _render_pyproject(context: dict[str, Any], dependencies: list[str]) -> str:
    package_name = context["package_name"]
    lines = [
        "[build-system]",
        'requires = ["hatchling"]',
        'build-backend = "hatchling.build"',
        "",
        "[project]",
        f'name = "{package_name}"',
        'version = "0.1.0"',
        f'description = "{context["one_line_purpose"]}"',
        'requires-python = ">=3.11"',
        "dependencies = [",
    ]
    lines += [f'    "{dep}",' for dep in dependencies]
    lines += ["]", ""]
    return "\n".join(lines)


def compose(bricks: list[Brick], context: dict[str, Any]) -> dict[str, str]:
    """§4 steps 2-6: `bricks` must already be the exact activated set (from
    `select_bricks`) — this function doesn't decide which bricks apply, only
    how to merge the ones it's given. Deterministic: identical `bricks` +
    `context` always produce byte-identical output (§5's compose-engine
    acceptance criterion), since every merge step below sorts explicitly
    rather than relying on input order or set/dict iteration order.
    """
    check_exclusive_collisions(bricks)
    sorted_bricks = sorted(bricks, key=lambda b: b.name)

    files: dict[str, str] = {}

    # §4 step 3: exclusive files, rendered directly.
    for brick in sorted_bricks:
        for path, template_source in brick.exclusive_files.items():
            files[path] = _render(template_source, context)

    # §4 step 4: shared files — concatenate every contributing brick's raw
    # fragment (base fragment first if one claims this path, then
    # alphabetical by brick name among the rest — see module docstring),
    # *then* render once as a single combined template.
    shared_paths = sorted({path for b in bricks for path in b.shared_contributions})
    for path in shared_paths:
        contributing = sorted(
            (b for b in bricks if path in b.shared_contributions),
            key=lambda b: (path not in b.shared_bases, b.name),
        )
        combined_source = "\n".join(b.shared_contributions[path] for b in contributing)
        files[path] = _render(combined_source, context)

    # `app/config.py` needs one more thing no single brick's fragment can
    # provide: a closer that comes *after* every brick's field
    # contributions, however many there are. Same pragmatic special-casing
    # already applied to pyproject.toml/.env.example below — a small fixed
    # wrapper the compose engine itself owns, not a brick.
    if "app/config.py" in files:
        files["app/config.py"] += "\n\n@lru_cache\ndef get_settings() -> Settings:\n    return Settings()\n"

    # §4 step 5: merged, deduplicated pyproject.toml dependency list.
    all_dependencies = sorted({dep for b in bricks for dep in b.dependencies})
    files["pyproject.toml"] = _render_pyproject(context, all_dependencies)

    # §4 step 6: merged .env.example — later bricks (alphabetically) win on
    # a genuine key collision, which should never happen in practice since
    # each brick should own its own env var namespace, but a silent
    # last-write-wins is still safer here than raising over something as
    # low-stakes as a duplicated example env var.
    merged_env: dict[str, str] = {}
    for brick in sorted_bricks:
        merged_env.update(brick.env_vars)
    files[".env.example"] = "".join(f"{key}={_render(value, context)}\n" for key, value in merged_env.items())

    return files


def select_bricks(answers: dict[str, Any], registry: dict[str, Brick]) -> list[Brick]:
    """§4 step 1: a small, explicit lookup table from elicitation answers to
    brick names — never inferred. Any mapped name absent from `registry`
    (a brick not built yet) is silently skipped rather than raising, so the
    catalog can grow incrementally without every combination needing to
    exist first (docs/loupe-scaffold.md §2's explicit scope note).
    """
    names = ["core_app"]

    db = answers.get("database")
    names.append({"sqlite": "db_sqlite", "postgresql": "db_postgresql", "mysql": "db_mysql", "mongodb": "db_mongodb"}.get(db, ""))

    orm = answers.get("orm_choice")
    names.append(
        {
            "sqlalchemy_async": "orm_sqlalchemy_async",
            "sqlalchemy_sync": "orm_sqlalchemy_sync",
            "sqlmodel": "orm_sqlmodel",
            "beanie": "orm_beanie",
            "motor_raw": "orm_motor_raw",
        }.get(orm, "")
    )

    if answers.get("migrations_needed") == "yes":
        names.append("migrations_alembic")

    auth = answers.get("auth_strategy")
    names.append(
        {
            "none": "auth_none",
            "api_key": "auth_api_key",
            "jwt_password": "auth_jwt",
            "oauth2_social": "auth_oauth2_social",
            "session": "auth_session",
        }.get(auth, "")
    )

    background = answers.get("background_work")
    if background == "celery":
        names.append("background_celery_redis" if answers.get("broker_choice") == "redis" else "background_celery_rabbitmq")
    else:
        names.append({"none": "background_none", "simple_tasks": "background_simple_tasks", "arq": "background_arq"}.get(background, ""))

    deploy = answers.get("deployment_target")
    names.append(
        {
            "docker_compose": "deploy_docker_compose",
            "kubernetes": "deploy_kubernetes",
            "serverless_lambda": "deploy_serverless_lambda",
            "bare_uvicorn": "deploy_bare",
        }.get(deploy, "")
    )

    observability = answers.get("observability_level")
    names.append(
        {
            "minimal": "observability_minimal",
            "structured_logging": "observability_structured_logging",
            "structured_logging_plus_otel": "observability_otel",
        }.get(observability, "")
    )

    testing = answers.get("testing_depth")
    names.append({"basic": "testing_basic", "basic_plus_ci": "testing_ci"}.get(testing, ""))

    return [registry[name] for name in names if name and name in registry]
