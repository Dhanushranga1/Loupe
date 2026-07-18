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


def test_init_creates_manifest_and_ignore_file(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    exit_code = main(["init", str(repo)])
    assert exit_code == 0
    assert (repo / "loupe.manifest.yaml").exists()
    assert (repo / ".loupeignore").exists()
    assert "Created" in capsys.readouterr().out


def test_init_is_idempotent_and_does_not_overwrite(tmp_path, capsys):
    repo = _make_repo(tmp_path)
    main(["init", str(repo)])
    (repo / "loupe.manifest.yaml").write_text("languages: [python]\ncustom_marker: true\n")

    main(["init", str(repo)])

    assert "custom_marker" in (repo / "loupe.manifest.yaml").read_text()
    assert "already exists" in capsys.readouterr().out


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
