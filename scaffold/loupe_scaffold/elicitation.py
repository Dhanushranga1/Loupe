"""The elicitation engine (docs/loupe-scaffold.md §3, §6 build-order step 1).

Pure logic, deliberately: no terminal I/O, no bricks, no compose engine —
just "given the answers so far, which question comes next, and with which
options." This is what §6 calls out as worth getting fully right before any
brick exists, and it's the whole reason `run_elicitation` takes an
`answer_fn` callback rather than calling `input()` itself: a real CLI can
pass a function that prompts a terminal, and a test can pass a function
that looks answers up in a plain dict — the engine's own logic (which
questions fire, in what order) is identical and independently testable
either way.

Explicitly NOT built yet, per docs/loupe-scaffold.md §6's stated order and
the project's own sequencing decision (see docs/progress/README.md): the
brick/compose system. It needs the FastAPI adapter's convention-category
taxonomy (`core/loupe_core/adapters/fastapi/convention_categories.py`),
which doesn't exist anywhere in this repo yet — building bricks against a
taxonomy that isn't there would mean inventing one now and likely redoing
it later. The elicitation engine has no such dependency (confirmed by
docs/loupe-scaffold.md §1's own correction: only the brick/compose system
needs the shared taxonomy), so it's what's safe to build standalone.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Union


@dataclass
class FixedQuestion:
    question_id: str
    options: list[str] | None  # None = free text


@dataclass
class ConditionalQuestion:
    """Fires only if `depends_on` matches the answers collected so far.

    `depends_on` maps a previously-answered question_id to either a single
    required value or a list of acceptable values — every key must match
    (AND across keys) for the question to fire. A key that hasn't been
    answered at all (because *its own* trigger never fired) fails closed:
    the question is never asked, not asked with a missing/None comparison.
    """

    question_id: str
    depends_on: dict[str, Union[str, list[str]]]
    options: list[str]
    multi_select: bool = False


# Fixed core questions (docs/loupe-scaffold.md §3) — always asked, in this order.
FIXED_QUESTIONS: list[FixedQuestion] = [
    FixedQuestion("project_name", options=None),
    FixedQuestion("one_line_purpose", options=None),
    FixedQuestion("database", options=["none", "sqlite", "postgresql", "mysql", "mongodb"]),
    FixedQuestion("auth_strategy", options=["none", "api_key", "jwt_password", "oauth2_social", "session"]),
    FixedQuestion("background_work", options=["none", "simple_tasks", "celery", "arq"]),
    FixedQuestion("deployment_target", options=["docker_compose", "kubernetes", "serverless_lambda", "bare_uvicorn"]),
    FixedQuestion("testing_depth", options=["basic", "basic_plus_ci"]),
    FixedQuestion("observability_level", options=["minimal", "structured_logging", "structured_logging_plus_otel"]),
]

# Adaptive extras (docs/loupe-scaffold.md §3). Declaration order matters: a
# question whose trigger depends on another conditional question's answer
# (migrations_needed depends on orm_choice) must be declared after it, so a
# single left-to-right pass resolves triggers correctly without needing a
# fixed-point/re-scan loop.
CONDITIONAL_QUESTIONS: list[ConditionalQuestion] = [
    # database in [postgresql, mysql] -> sqlalchemy_async/sync/sqlmodel.
    # database == mongodb -> beanie/motor_raw (below). Mutually exclusive by
    # construction (database can only hold one value), and the engine skips
    # a question_id it's already asked, so only one of these two ever fires.
    ConditionalQuestion(
        question_id="orm_choice",
        depends_on={"database": ["postgresql", "mysql"]},
        options=["sqlalchemy_async", "sqlalchemy_sync", "sqlmodel"],
    ),
    ConditionalQuestion(
        question_id="orm_choice",
        depends_on={"database": ["mongodb"]},
        options=["beanie", "motor_raw"],
    ),
    # "database != none and orm_choice supports it": expressed as depending
    # on orm_choice itself, restricted to the SQL-style choices that support
    # Alembic-style migrations — beanie/motor_raw (Mongo) don't, and sqlite
    # never triggers orm_choice at all above, so both correctly never reach
    # this question (orm_choice is simply absent from the answers so far).
    ConditionalQuestion(
        question_id="migrations_needed",
        depends_on={"orm_choice": ["sqlalchemy_async", "sqlalchemy_sync", "sqlmodel"]},
        options=["yes", "no"],
    ),
    ConditionalQuestion(
        question_id="oauth_providers",
        depends_on={"auth_strategy": "oauth2_social"},
        options=["google", "github", "microsoft"],
        multi_select=True,
    ),
    ConditionalQuestion(
        question_id="broker_choice",
        depends_on={"background_work": "celery"},
        options=["redis", "rabbitmq"],
    ),
    ConditionalQuestion(
        question_id="helm_chart_needed",
        depends_on={"deployment_target": "kubernetes"},
        options=["yes", "no"],
    ),
    ConditionalQuestion(
        question_id="otel_exporter",
        depends_on={"observability_level": "structured_logging_plus_otel"},
        options=["console", "otlp_collector"],
    ),
]


def _condition_matches(depends_on: dict[str, Union[str, list[str]]], answers: dict[str, Any]) -> bool:
    for question_id, expected in depends_on.items():
        if question_id not in answers:
            return False  # its own trigger never fired -> fail closed, don't guess
        actual = answers[question_id]
        expected_values = expected if isinstance(expected, list) else [expected]
        actual_values = actual if isinstance(actual, list) else [actual]
        if not any(v in expected_values for v in actual_values):
            return False
    return True


Question = Union[FixedQuestion, ConditionalQuestion]


def run_elicitation(answer_fn: Callable[[Question], Any]) -> dict[str, Any]:
    """Run the full fixed + adaptive interview, calling `answer_fn` once per
    question actually asked (in the order it was asked), and return
    `{question_id: answer}`. A question never triggered is simply never in
    the returned dict at all — not present with a null/"n/a" value.
    """
    answers: dict[str, Any] = {}

    for question in FIXED_QUESTIONS:
        answers[question.question_id] = answer_fn(question)

    for question in CONDITIONAL_QUESTIONS:
        if question.question_id in answers:
            continue  # e.g. orm_choice's other branch already fired
        if _condition_matches(question.depends_on, answers):
            answers[question.question_id] = answer_fn(question)

    return answers
