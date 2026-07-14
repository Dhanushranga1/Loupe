"""Tests for the elicitation engine (docs/loupe-scaffold.md §5 — Elicitation acceptance criteria)."""

from loupe_scaffold.elicitation import ConditionalQuestion, FixedQuestion, run_elicitation


def _canned(overrides: dict, asked: list[str] | None = None):
    """Returns a canned answer per question_id from `overrides` (falling back to that
    question's first option, or a placeholder string for free-text questions), and —
    if `asked` is passed — records every question_id actually asked, in order."""

    def answer_fn(question):
        if asked is not None:
            asked.append(question.question_id)
        if question.question_id in overrides:
            return overrides[question.question_id]
        if isinstance(question, FixedQuestion) and question.options is None:
            return "placeholder text"
        return question.options[0]

    return answer_fn


def test_database_none_asks_zero_database_related_questions():
    """§5's first elicitation acceptance criterion, verified directly."""
    asked: list[str] = []
    run_elicitation(_canned({"database": "none"}, asked))

    database_related = {"orm_choice", "migrations_needed"}
    assert database_related.isdisjoint(asked)


def test_mongodb_offers_beanie_and_motor_raw_not_sqlalchemy():
    """§5's second criterion: options genuinely differ per database, not filtered
    from one master list — sqlalchemy_async must never even be offered for mongodb."""
    captured_options: list[str] = []

    def answer_fn(question):
        if question.question_id == "orm_choice":
            captured_options.extend(question.options)
        if isinstance(question, FixedQuestion) and question.options is None:
            return "placeholder"
        if question.question_id == "database":
            return "mongodb"
        return question.options[0]

    run_elicitation(answer_fn)

    assert set(captured_options) == {"beanie", "motor_raw"}
    assert "sqlalchemy_async" not in captured_options


def test_postgresql_offers_sqlalchemy_options_not_mongo_ones():
    captured_options: list[str] = []

    def answer_fn(question):
        if question.question_id == "orm_choice":
            captured_options.extend(question.options)
        if isinstance(question, FixedQuestion) and question.options is None:
            return "placeholder"
        if question.question_id == "database":
            return "postgresql"
        return question.options[0]

    run_elicitation(answer_fn)

    assert set(captured_options) == {"sqlalchemy_async", "sqlalchemy_sync", "sqlmodel"}
    assert "beanie" not in captured_options


def test_auth_none_skips_oauth_providers_entirely():
    """§5's third criterion."""
    asked: list[str] = []
    run_elicitation(_canned({"auth_strategy": "none"}, asked))

    assert "oauth_providers" not in asked


def test_oauth2_social_triggers_oauth_providers_as_multi_select():
    asked_questions = {}

    def answer_fn(question):
        asked_questions[question.question_id] = question
        if isinstance(question, FixedQuestion) and question.options is None:
            return "placeholder"
        if question.question_id == "auth_strategy":
            return "oauth2_social"
        if question.question_id == "oauth_providers":
            return ["google", "github"]  # multi-select answer
        return question.options[0]

    answers = run_elicitation(answer_fn)

    assert isinstance(asked_questions["oauth_providers"], ConditionalQuestion)
    assert asked_questions["oauth_providers"].multi_select is True
    assert answers["oauth_providers"] == ["google", "github"]


def test_sqlite_never_triggers_orm_choice_or_migrations_needed():
    """sqlite isn't in orm_choice's trigger list at all (docs/loupe-scaffold.md §3) —
    unlike postgresql/mysql/mongodb, it never asks an ORM question."""
    asked: list[str] = []
    run_elicitation(_canned({"database": "sqlite"}, asked))

    assert "orm_choice" not in asked
    assert "migrations_needed" not in asked


def test_postgresql_with_sqlalchemy_async_triggers_migrations_needed():
    asked: list[str] = []
    run_elicitation(_canned({"database": "postgresql", "orm_choice": "sqlalchemy_async"}, asked))

    assert "migrations_needed" in asked


def test_mongodb_with_beanie_does_not_trigger_migrations_needed():
    """beanie/motor_raw don't use Alembic-style migrations — the ORM 'supports it' gate."""
    asked: list[str] = []
    run_elicitation(_canned({"database": "mongodb", "orm_choice": "beanie"}, asked))

    assert "migrations_needed" not in asked


def test_celery_triggers_broker_choice_other_background_work_does_not():
    asked_celery: list[str] = []
    run_elicitation(_canned({"background_work": "celery"}, asked_celery))
    assert "broker_choice" in asked_celery

    asked_none: list[str] = []
    run_elicitation(_canned({"background_work": "none"}, asked_none))
    assert "broker_choice" not in asked_none


def test_kubernetes_triggers_helm_chart_needed_other_targets_do_not():
    asked_k8s: list[str] = []
    run_elicitation(_canned({"deployment_target": "kubernetes"}, asked_k8s))
    assert "helm_chart_needed" in asked_k8s

    asked_docker: list[str] = []
    run_elicitation(_canned({"deployment_target": "docker_compose"}, asked_docker))
    assert "helm_chart_needed" not in asked_docker


def test_otel_exporter_only_triggered_by_the_full_otel_observability_level():
    asked_otel: list[str] = []
    run_elicitation(_canned({"observability_level": "structured_logging_plus_otel"}, asked_otel))
    assert "otel_exporter" in asked_otel

    asked_structured: list[str] = []
    run_elicitation(_canned({"observability_level": "structured_logging"}, asked_structured))
    assert "otel_exporter" not in asked_structured


def test_untriggered_question_is_absent_from_answers_not_present_with_a_null_value():
    answers = run_elicitation(_canned({"database": "none"}))

    assert "orm_choice" not in answers
    assert answers.get("orm_choice", "sentinel-not-set") == "sentinel-not-set"


def test_fixed_questions_are_always_all_asked_in_declared_order():
    asked: list[str] = []
    run_elicitation(_canned({}, asked))

    fixed_ids = [
        "project_name",
        "one_line_purpose",
        "database",
        "auth_strategy",
        "background_work",
        "deployment_target",
        "testing_depth",
        "observability_level",
    ]
    assert asked[: len(fixed_ids)] == fixed_ids


def test_a_maximal_answer_set_triggers_every_conditional_question_exactly_once():
    overrides = {
        "database": "postgresql",
        "orm_choice": "sqlalchemy_async",
        "auth_strategy": "oauth2_social",
        "oauth_providers": ["google"],
        "background_work": "celery",
        "deployment_target": "kubernetes",
        "observability_level": "structured_logging_plus_otel",
    }
    asked: list[str] = []
    answers = run_elicitation(_canned(overrides, asked))

    conditional_ids = {
        "orm_choice",
        "migrations_needed",
        "oauth_providers",
        "broker_choice",
        "helm_chart_needed",
        "otel_exporter",
    }
    assert conditional_ids.issubset(answers.keys())
    # each conditional question_id appears at most once in the asked order,
    # even though orm_choice is declared twice (postgresql/mysql vs mongodb)
    assert len(asked) == len(set(asked))
