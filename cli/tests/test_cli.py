"""Tests for loupe_cli/main.py (docs/phase-4-systems.md §9)."""

import shutil
from pathlib import Path

from loupe_cli.main import main

PHASE1_FIXTURES = Path(__file__).parent.parent.parent / "core" / "tests" / "fixtures" / "phase1"
PHASE1_FILES = ["utils.py", "models.py", "services.py", "handlers.py", "circular_a.py", "circular_b.py"]


def _make_repo(tmp_path) -> Path:
    for f in PHASE1_FILES:
        shutil.copy(PHASE1_FIXTURES / f, tmp_path / f)
    return tmp_path


def test_status_before_init_reports_no_loupe_dir(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    exit_code = main(["status", str(repo)])
    assert exit_code == 1
    assert "No .loupe/" in capsys.readouterr().out


def test_init_creates_manifest_and_ignore_file(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)
    repo = _make_repo(tmp_path)
    exit_code = main(["init", str(repo)])
    assert exit_code == 0
    assert (repo / "loupe.manifest.yaml").exists()
    assert (repo / ".loupeignore").exists()
    assert "Created" in capsys.readouterr().out


def test_init_is_idempotent_and_does_not_overwrite(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)
    repo = _make_repo(tmp_path)
    main(["init", str(repo)])
    (repo / "loupe.manifest.yaml").write_text("languages: [python]\ncustom_marker: true\n")

    main(["init", str(repo)])

    assert "custom_marker" in (repo / "loupe.manifest.yaml").read_text()
    assert "already exists" in capsys.readouterr().out


def test_init_with_no_gpu_defaults_to_cpu_small_without_prompting(tmp_path, monkeypatch):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)

    def _fail_if_prompted(prompt=""):
        raise AssertionError("should not prompt when no GPU is detected")

    monkeypatch.setattr("builtins.input", _fail_if_prompted)
    repo = _make_repo(tmp_path)

    main(["init", str(repo)])

    assert "compute_profile: cpu_small" in (repo / "loupe.manifest.yaml").read_text()


def test_init_with_gpu_prompts_and_defaults_to_cpu_small_on_decline(tmp_path, monkeypatch):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    repo = _make_repo(tmp_path)

    main(["init", str(repo)])

    assert "compute_profile: cpu_small" in (repo / "loupe.manifest.yaml").read_text()


def test_init_with_gpu_prompts_and_uses_gpu_large_on_accept(tmp_path, monkeypatch):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: True)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    repo = _make_repo(tmp_path)

    main(["init", str(repo)])

    assert "compute_profile: gpu_large" in (repo / "loupe.manifest.yaml").read_text()


def test_init_explicit_gpu_large_with_no_gpu_warns_and_falls_back_on_decline(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    repo = _make_repo(tmp_path)

    main(["init", str(repo), "--compute-profile", "gpu_large"])

    output = capsys.readouterr().out
    assert "No GPU detected" in output
    assert "compute_profile: cpu_small" in (repo / "loupe.manifest.yaml").read_text()


def test_init_explicit_gpu_large_with_no_gpu_proceeds_on_confirm(tmp_path, monkeypatch):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    repo = _make_repo(tmp_path)

    main(["init", str(repo), "--compute-profile", "gpu_large"])

    assert "compute_profile: gpu_large" in (repo / "loupe.manifest.yaml").read_text()


def test_init_explicit_profile_with_gpu_present_skips_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: True)

    def _fail_if_prompted(prompt=""):
        raise AssertionError("explicit --compute-profile should not prompt")

    monkeypatch.setattr("builtins.input", _fail_if_prompted)
    repo = _make_repo(tmp_path)

    main(["init", str(repo), "--compute-profile", "cpu_medium"])

    assert "compute_profile: cpu_medium" in (repo / "loupe.manifest.yaml").read_text()


def test_index_prints_symbol_count_and_unresolved_count(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    exit_code = main(["index", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Indexed 16 symbols" in output
    assert "Languages detected: python" in output
    assert "Unresolved references:" in output


def test_status_after_index_reports_real_state(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])
    capsys.readouterr()  # discard index output

    exit_code = main(["status", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Symbol count:   16" in output
    assert "Schema version: 1" in output


def test_status_reports_cold_start_ranker_before_any_training(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])
    capsys.readouterr()

    main(["status", str(repo)])
    output = capsys.readouterr().out
    assert "Learned ranker: not trained (cold-start" in output


def test_doctor_without_an_index_returns_exit_code_1(tmp_path, capsys):
    repo = _make_repo(tmp_path)

    exit_code = main(["doctor", str(repo)])

    assert exit_code == 1
    output = capsys.readouterr().out
    assert "run `loupe index`" in output


def test_doctor_on_a_clean_indexed_repo_reports_no_problems(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])
    capsys.readouterr()

    exit_code = main(["doctor", str(repo)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "No problems found." in output


def test_doctor_flags_a_manifest_embedding_model_missing_its_org_prefix(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])
    capsys.readouterr()

    (repo / "loupe.manifest.yaml").write_text(
        "schema_version: 1\nlanguages: [python]\nembedding_model: bge-small-en-v1.5\n"
    )

    exit_code = main(["doctor", str(repo)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "[WARN " in output
    assert "bge-small-en-v1.5" in output


def test_status_surfaces_coefficients_of_a_previously_saved_trained_ranker(tmp_path, capsys):
    import random

    from loupe_core.retrieval.ranker import COLD_START_THRESHOLD, Ranker, TrainingExample

    repo = _make_repo(tmp_path)
    main(["index", str(repo)])
    capsys.readouterr()

    random.seed(11)
    examples = [
        TrainingExample(
            lexical_score=random.random(), semantic_score=random.random(), centrality_score=random.random(),
            symbol_edited=random.random() > 0.5,
        )
        for _ in range(COLD_START_THRESHOLD)
    ]
    ranker = Ranker()
    ranker.train(examples)
    ranker.save(str(repo / ".loupe" / "ranker.pkl"))

    main(["status", str(repo)])
    output = capsys.readouterr().out
    assert "Learned ranker: trained" in output
    for name in ("lexical_score", "semantic_score", "centrality_score"):
        assert name in output


def test_retrain_before_loupe_dir_exists_reports_error(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    exit_code = main(["retrain", str(repo)])
    assert exit_code == 1
    assert "No .loupe/" in capsys.readouterr().out


def test_retrain_with_insufficient_data_leaves_ranker_untrained_but_saves_state(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])
    capsys.readouterr()

    exit_code = main(["retrain", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Not enough labeled usage data yet" in output
    assert (repo / ".loupe" / "ranker.pkl").exists()


def test_generate_context_writes_claude_md_on_first_run(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    exit_code = main(["generate-context", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Generated" in output
    assert "for the first time" in output
    assert (repo / "CLAUDE.md").exists()
    assert (repo / ".loupe" / "context" / "state.json").exists()
    assert (repo / "CLAUDE.md").read_text().startswith("# CLAUDE.md")


def test_generate_context_second_run_with_no_changes_is_a_no_op(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["generate-context", str(repo)])
    first_content = (repo / "CLAUDE.md").read_text()
    capsys.readouterr()

    exit_code = main(["generate-context", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "No convention/architecture changes detected" in output
    assert (repo / "CLAUDE.md").read_text() == first_content


def test_generate_context_reports_a_structured_diff_after_a_real_change(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["generate-context", str(repo)])
    capsys.readouterr()

    # Deliberately shift utils.py's docstring/error-handling shape so at least
    # one convention fact's underlying data actually changes.
    utils_path = repo / "utils.py"
    utils_path.write_text(
        utils_path.read_text() + "\n\ndef newly_added_helper():\n    try:\n        return 1\n    except Exception:\n        print('boom')\n"
    )

    exit_code = main(["generate-context", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Regenerated" in output
    assert "Changes since last generation" in output


def _make_git_repo(tmp_path) -> Path:
    import git

    repo = _make_repo(tmp_path)
    git_repo = git.Repo.init(repo)
    with git_repo.config_writer() as cw:
        cw.set_value("user", "name", "Test")
        cw.set_value("user", "email", "test@example.com")
    git_repo.index.add(PHASE1_FILES)
    git_repo.index.commit("initial commit")
    return repo


def test_update_churn_reports_not_a_git_repo_when_there_is_no_git_history(tmp_path, capsys):
    repo = _make_repo(tmp_path)  # deliberately not git-initialized
    exit_code = main(["update-churn", str(repo)])
    assert exit_code == 1
    assert "not a git repository" in capsys.readouterr().out


def test_update_churn_computes_and_caches_scores_for_a_real_git_repo(tmp_path, capsys):
    repo = _make_git_repo(tmp_path)

    exit_code = main(["update-churn", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Computed churn for" in output
    assert (repo / ".loupe" / "cache" / "churn.json").exists()

    import json

    scores = json.loads((repo / ".loupe" / "cache" / "churn.json").read_text())
    assert len(scores) == 16  # matches the "Indexed 16 symbols" figure the index tests already assert


def test_update_suggestions_reports_no_loupe_dir_before_index(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    exit_code = main(["update-suggestions", str(repo)])
    assert exit_code == 1
    assert "No .loupe/" in capsys.readouterr().out


def test_update_suggestions_mines_real_retrieval_log_history(tmp_path, capsys):
    import json

    repo = _make_repo(tmp_path)
    main(["index", str(repo)])
    capsys.readouterr()

    logs_dir = repo / ".loupe" / "logs" / "retrieval"
    logs_dir.mkdir(parents=True, exist_ok=True)
    for i in range(6):  # 6 separate sessions, each co-requesting symbol-a and symbol-b
        with open(logs_dir / f"session-{i}.jsonl", "w") as f:
            f.write(json.dumps({"tool_name": "get_symbol", "query_text": "symbol-a"}) + "\n")
            f.write(json.dumps({"tool_name": "get_symbol", "query_text": "symbol-b"}) + "\n")

    exit_code = main(["update-suggestions", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "Mined co-retrieval suggestions from 6 session" in output

    suggestions_path = repo / ".loupe" / "cache" / "co_retrieval.json"
    assert suggestions_path.exists()
    suggestions = json.loads(suggestions_path.read_text())
    assert suggestions["symbol-a"][0]["symbol_id"] == "symbol-b"


def test_check_runs_e5_and_e6_and_skips_e7_e8_e9_when_their_inputs_are_absent(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    exit_code = main(["check", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "E6 dead code:" in output
    assert "E5 duplicate code:" in output
    assert "E7 config drift: skipped (.env.example not found)" in output
    assert "E8 migration drift: skipped (alembic/versions/ not found)" in output
    assert "E9 API contract diff: skipped" in output


def test_check_runs_e7_when_env_example_present(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    (repo / "config.py").write_text("class Settings:\n    api_key: str\n")
    (repo / ".env.example").write_text("")

    exit_code = main(["check", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "E7 config drift: 1 finding(s)" in output
    assert "API_KEY (missing_from_env_example)" in output


def test_check_runs_e8_when_alembic_versions_present(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    (repo / "models.py").write_text('class Item:\n    __tablename__ = "items"\n    id: int\n    price: float\n')
    versions_dir = repo / "alembic" / "versions"
    versions_dir.mkdir(parents=True)
    (versions_dir / "0001_initial.py").write_text(
        "def upgrade():\n    op.create_table('items', sa.Column('id', sa.Integer()))\n"
    )

    exit_code = main(["check", str(repo)])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "E8 migration drift: 1 finding(s)" in output
    assert "Item.price" in output


def test_check_runs_e9_with_since_flag_against_a_real_git_repo(tmp_path, capsys):
    repo = _make_git_repo(tmp_path)

    routes_path = repo / "api_routes.py"
    routes_path.write_text(
        "class ItemOut:\n    id: int\n    name: str\n\n\n"
        "@app.get('/items', response_model=ItemOut)\n"
        "def list_items():\n    ...\n"
    )
    import git

    git_repo = git.Repo(repo)
    git_repo.index.add(["api_routes.py"])
    first_commit = git_repo.index.commit("add route")

    routes_path.write_text(
        "class ItemOut:\n    id: int\n\n\n"  # 'name' field removed -- a real breaking change
        "@app.get('/items', response_model=ItemOut)\n"
        "def list_items():\n    ...\n"
    )
    git_repo.index.add(["api_routes.py"])
    git_repo.index.commit("remove name field")

    exit_code = main(["check", str(repo), "--since", first_commit.hexsha])
    assert exit_code == 0

    output = capsys.readouterr().out
    assert "E9 API contract diff (since" in output
    assert "list_items" in output
    assert "name" in output


def test_enable_experimental_blocks_on_confirmation_and_persists_flags_on_accept(tmp_path, monkeypatch, capsys):
    repo = _make_repo(tmp_path)
    main(["init", str(repo), "--compute-profile", "cpu_small"])
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")

    exit_code = main(["enable-experimental", "hyde_query_rewrite", str(repo)])

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Estimated cost per call" in output
    assert "Enabled hyde_query_rewrite" in output

    from loupe_mcp_server.config import load_config
    from loupe_mcp_server.experimental_gate import is_experimental_feature_enabled

    config = load_config(repo, global_config_path=repo / "no-global.yaml")
    assert is_experimental_feature_enabled(config, "hyde_query_rewrite") is True


def test_enable_experimental_does_not_persist_flags_on_decline(tmp_path, monkeypatch, capsys):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)
    main(["init", str(repo)])
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")

    exit_code = main(["enable-experimental", "hyde_query_rewrite", str(repo)])

    assert exit_code == 1
    assert "Not enabled" in capsys.readouterr().out

    from loupe_mcp_server.config import load_config
    from loupe_mcp_server.experimental_gate import is_experimental_feature_enabled

    config = load_config(repo, global_config_path=repo / "no-global.yaml")
    assert is_experimental_feature_enabled(config, "hyde_query_rewrite") is False


def test_enable_experimental_is_idempotent_and_skips_prompt_when_already_enabled(tmp_path, monkeypatch, capsys):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)
    main(["init", str(repo)])
    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    main(["enable-experimental", "hyde_query_rewrite", str(repo)])

    def _fail_if_prompted(prompt=""):
        raise AssertionError("should not re-prompt once already enabled")

    monkeypatch.setattr("builtins.input", _fail_if_prompted)
    exit_code = main(["enable-experimental", "hyde_query_rewrite", str(repo)])

    assert exit_code == 0
    assert "already enabled" in capsys.readouterr().out


def test_enable_experimental_rejects_unknown_feature_without_prompting(tmp_path, monkeypatch, capsys):
    repo = _make_repo(tmp_path)
    monkeypatch.setattr("loupe_cli.main.detect_gpu", lambda: False)
    main(["init", str(repo)])

    def _fail_if_prompted(prompt=""):
        raise AssertionError("should not prompt for an unknown feature")

    monkeypatch.setattr("builtins.input", _fail_if_prompted)
    exit_code = main(["enable-experimental", "not_a_real_feature", str(repo)])

    assert exit_code == 1
    assert "Unknown experimental feature" in capsys.readouterr().out


def test_build_status_with_no_tracked_entries_reports_none(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])

    exit_code = main(["build-status", str(repo)])

    assert exit_code == 0
    assert "No tracked entries yet" in capsys.readouterr().out


def test_build_status_reports_counts_by_status(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])

    from loupe_core.graph.builder import build_graph, parse_file
    from loupe_core.ledger.build_ledger import BuildLedgerStore, LedgerStatus

    import os

    old_cwd = os.getcwd()
    os.chdir(repo)
    try:
        parsed = [parse_file(f) for f in PHASE1_FILES]
    finally:
        os.chdir(old_cwd)
    g = build_graph(parsed)
    symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
    format_currency_id = next(s.id for s in symbols_by_id.values() if s.qualified_name == "format_currency")
    validate_email_id = next(s.id for s in symbols_by_id.values() if s.qualified_name == "validate_email")

    store = BuildLedgerStore(repo / ".loupe" / "build" / "ledger.json")
    store.set_status(format_currency_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)
    store.set_status(validate_email_id, LedgerStatus.PLANNED, graph=g.graph, symbols_by_id=symbols_by_id)

    exit_code = main(["build-status", str(repo)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "done: 1" in output
    assert "planned: 1" in output


def test_build_status_names_stale_entries_explicitly(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["index", str(repo)])

    from loupe_core.graph.builder import build_graph, parse_file
    from loupe_core.ledger.build_ledger import BuildLedgerStore, LedgerStatus

    import os

    old_cwd = os.getcwd()
    os.chdir(repo)
    try:
        parsed = [parse_file(f) for f in PHASE1_FILES]
    finally:
        os.chdir(old_cwd)
    g = build_graph(parsed)
    symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
    format_currency_id = next(s.id for s in symbols_by_id.values() if s.qualified_name == "format_currency")

    store = BuildLedgerStore(repo / ".loupe" / "build" / "ledger.json")
    store.set_status(format_currency_id, LedgerStatus.DONE, graph=g.graph, symbols_by_id=symbols_by_id)

    (repo / "utils.py").write_text(
        'def format_currency(amount: float) -> str:\n'
        '    """Changed."""\n'
        '    return f"USD {amount:.2f}"\n\n\n'
        'def validate_email(email: str) -> bool:\n'
        '    return "@" in email and "." in email\n\n\n'
        'def unused_utility() -> None:\n'
        '    return None\n'
    )

    exit_code = main(["build-status", str(repo)])

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "stale: 1" in output
    assert "format_currency" in output
