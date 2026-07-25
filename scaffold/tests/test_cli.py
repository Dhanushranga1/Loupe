"""Tests for cli.py's `loupe new` entrypoint — scripted `input()`, no real terminal."""

from loupe_scaffold.cli import main

MEDIUM_COMPLEXITY_INPUTS = iter(
    [
        "Demo API",  # project_name
        "a demo service",  # one_line_purpose
        "postgresql",  # database
        "jwt_password",  # auth_strategy
        "none",  # background_work
        "docker_compose",  # deployment_target
        "basic",  # testing_depth
        "minimal",  # observability_level
        "sqlalchemy_async",  # orm_choice (conditional, fires for postgresql)
        "no",  # migrations_needed (conditional)
    ]
)


def test_loupe_new_runs_the_full_interview_and_writes_a_real_project(tmp_path, monkeypatch):
    output_dir = tmp_path / "generated"
    monkeypatch.setattr("builtins.input", lambda prompt="": next(MEDIUM_COMPLEXITY_INPUTS))

    exit_code = main([str(output_dir)])

    assert exit_code == 0
    assert (output_dir / "app" / "main.py").exists()
    assert (output_dir / "loupe.manifest.yaml").exists()


def test_loupe_new_refuses_to_generate_into_a_nonempty_directory(tmp_path, capsys):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    (output_dir / "already_here.txt").write_text("hi")

    exit_code = main([str(output_dir)])

    assert exit_code == 1
    assert "not empty" in capsys.readouterr().out
    assert not (output_dir / "app").exists()


def test_loupe_new_free_text_prompt_returns_the_typed_value(tmp_path, monkeypatch):
    inputs = iter(["My App", "", "none", "none", "none", "bare_uvicorn", "basic", "minimal"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    output_dir = tmp_path / "generated"
    main([str(output_dir)])

    assert "My App" in (output_dir / "README.md").read_text()


def test_loupe_new_invalid_option_falls_back_to_the_first_option(tmp_path, monkeypatch):
    inputs = iter(["My App", "", "not_a_real_database_choice", "none", "none", "bare_uvicorn", "basic", "minimal"])
    monkeypatch.setattr("builtins.input", lambda prompt="": next(inputs))

    output_dir = tmp_path / "generated"
    exit_code = main([str(output_dir)])

    assert exit_code == 0  # falls back to "none" (database's first option), doesn't crash
