"""Tests for context/claude_md_generator.py (docs/PhaseX/claude-md-generator.md)."""

import os
import shutil
from pathlib import Path

import pytest

from loupe_core.context.claude_md_generator import (
    CONFIDENCE_THRESHOLD,
    ArchitectureEntry,
    ConventionFact,
    GeneratorState,
    compute_architecture_entries,
    compute_architecture_overview,
    compute_convention_facts,
    compute_input_hash,
    generate_claude_md,
    render_document,
)
from loupe_core.conventions.mining import (
    ConventionsReport,
    DocstringConvention,
    ErrorHandlingConvention,
    ImportConvention,
    mine_conventions,
)
from loupe_core.graph.builder import build_graph, parse_file

E4_FIXTURES = Path(__file__).parent / "fixtures" / "e4"


@pytest.fixture
def handlers_parsed(tmp_path, monkeypatch):
    shutil.copy(E4_FIXTURES / "handlers.py", tmp_path / "handlers.py")
    monkeypatch.chdir(tmp_path)
    return parse_file("handlers.py")


def _synthetic_report(
    error_confidence_pct: int = 100, error_total: int = 10, docstring_style: str = "none", import_style: str = "none"
) -> ConventionsReport:
    violation_count = error_total - round(error_total * error_confidence_pct / 100)
    return ConventionsReport(
        error_handling=ErrorHandlingConvention(
            majority_pattern="except ValueError: logging.error",
            violation_count=violation_count,
            total_count=error_total,
        ),
        docstrings=DocstringConvention(coverage_pct=100.0, dominant_style="none", documented_count=0, dominant_style_count=0)
        if docstring_style == "none"
        else DocstringConvention(
            coverage_pct=100.0, dominant_style=docstring_style, documented_count=10, dominant_style_count=10
        ),
        imports=ImportConvention(dominant_style="absolute", relative_count=0, absolute_count=0)
        if import_style == "none"
        else ImportConvention(dominant_style=import_style, relative_count=0, absolute_count=10),
    )


# --------------------------------------------------------------------------
# §5 acceptance criterion: majority error-handling pattern correctly stated
# --------------------------------------------------------------------------


def test_generated_document_states_the_real_majority_error_handling_pattern(handlers_parsed):
    report = mine_conventions([handlers_parsed])
    graph = build_graph([handlers_parsed])
    symbols_by_id = {s.id: s for s in handlers_parsed.symbols}

    result = generate_claude_md(report, graph, symbols_by_id)

    assert result.regenerated
    assert "Error handling: except ValueError: logging.error" in result.content


# --------------------------------------------------------------------------
# §5 acceptance criterion: below-threshold convention excluded, not hedged
# --------------------------------------------------------------------------


def test_convention_below_confidence_threshold_is_excluded_entirely():
    low_confidence_report = _synthetic_report(error_confidence_pct=50, error_total=10)  # well under CONFIDENCE_THRESHOLD
    facts = compute_convention_facts(low_confidence_report)
    error_fact = next(f for f in facts if f.kind == "error_handling")
    assert error_fact.confidence < CONFIDENCE_THRESHOLD

    import networkx as nx

    from loupe_core.graph.builder import LoupeGraph

    result = generate_claude_md(low_confidence_report, LoupeGraph(graph=nx.DiGraph()), {})

    assert "Error handling" not in result.content
    assert "except ValueError" not in result.content


def test_convention_at_or_above_confidence_threshold_is_included():
    high_confidence_report = _synthetic_report(error_confidence_pct=90, error_total=10)
    import networkx as nx

    from loupe_core.graph.builder import LoupeGraph

    result = generate_claude_md(high_confidence_report, LoupeGraph(graph=nx.DiGraph()), {})

    assert "Error handling: except ValueError: logging.error" in result.content


# --------------------------------------------------------------------------
# §5 acceptance criterion: knapsack budgeting prioritizes higher-value facts
# --------------------------------------------------------------------------


def test_knapsack_selection_prioritizes_higher_value_over_insertion_order():
    """Fixture: 3 architecture entries with deliberately unequal PageRank scores
    and near-identical token cost, budget only large enough for 2 of the 3 — the
    naive (insertion-order / alphabetical) selection would visibly differ from
    the value-driven knapsack choice.
    """
    import networkx as nx

    from loupe_core.governor.budget import estimate_tokens
    from loupe_core.graph.builder import LoupeGraph

    architecture = [
        ArchitectureEntry(cluster_index=0, symbol_id="z", qualified_name="zz_low_value", file_path="z.py", pagerank_score=0.01),
        ArchitectureEntry(cluster_index=1, symbol_id="a", qualified_name="aa_high_value", file_path="a.py", pagerank_score=0.9),
        ArchitectureEntry(cluster_index=2, symbol_id="m", qualified_name="mm_mid_value", file_path="m.py", pagerank_score=0.5),
    ]
    # Each label costs roughly the same number of tokens (similar string lengths),
    # so ratio-ordering is dominated by pagerank_score, not cost differences.
    budget = estimate_tokens(f"`{architecture[1].qualified_name}` ({architecture[1].file_path})") + estimate_tokens(
        f"`{architecture[2].qualified_name}` ({architecture[2].file_path})"
    )

    from loupe_core.context.claude_md_generator import _select_within_budget

    selected_facts, selected_architecture = _select_within_budget([], architecture, budget)
    selected_names = {e.qualified_name for e in selected_architecture}

    assert selected_names == {"aa_high_value", "mm_mid_value"}, (
        "the two highest-value entries must be selected regardless of their insertion order"
    )


# --------------------------------------------------------------------------
# §5 acceptance criterion: community-detection-based architecture selection
# spans genuinely separate subsystems, not just the globally-densest one
# --------------------------------------------------------------------------


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


@pytest.fixture
def two_subsystem_repo(tmp_path):
    _write(
        tmp_path,
        "sys_a.py",
        "from sys_b import b1\n\n\n"
        "def a1():\n    b1()\n    return a2() + a3()\n\n\n"
        "def a2():\n    return a3() + a4()\n\n\n"
        "def a3():\n    return a4() + a1()\n\n\n"
        "def a4():\n    return a1() + a2()\n",
    )
    _write(
        tmp_path,
        "sys_b.py",
        "def b1():\n    return b2() + b3()\n\n\n"
        "def b2():\n    return b3() + b4()\n\n\n"
        "def b3():\n    return b4() + b1()\n\n\n"
        "def b4():\n    return b1() + b2()\n",
    )
    old_cwd = os.getcwd()
    os.chdir(tmp_path)
    try:
        parsed = [parse_file("sys_a.py"), parse_file("sys_b.py")]
        graph = build_graph(parsed)
        symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
        yield graph, symbols_by_id
    finally:
        os.chdir(old_cwd)


def test_architecture_entries_include_a_representative_from_each_cluster(two_subsystem_repo):
    graph, symbols_by_id = two_subsystem_repo

    entries = compute_architecture_entries(graph, symbols_by_id)
    prefixes = {e.qualified_name[0] for e in entries}  # "a1".."a4" vs "b1".."b4"

    assert len(entries) == 2, "one representative hub per coarse cluster, not flat top-N"
    assert prefixes == {"a", "b"}, "both subsystems must be represented, not just the globally denser one"


# --------------------------------------------------------------------------
# architecture://overview (Phase 14 §1's L0/L1 LOD levels)
# --------------------------------------------------------------------------


def _empty_report() -> ConventionsReport:
    return ConventionsReport(
        error_handling=ErrorHandlingConvention(majority_pattern=None),
        docstrings=DocstringConvention(coverage_pct=0.0, dominant_style="none"),
        imports=ImportConvention(dominant_style="absolute", relative_count=0, absolute_count=0),
    )


def test_architecture_overview_names_both_constructed_subsystems(two_subsystem_repo):
    graph, symbols_by_id = two_subsystem_repo

    overview = compute_architecture_overview(_empty_report(), graph, symbols_by_id)

    assert len(overview["clusters"]) == 2
    hub_names = {c["hub_qualified_name"] for c in overview["clusters"]}
    assert {n[0] for n in hub_names} == {"a", "b"}, "must name both subsystems, not just the higher-centrality one"
    assert str(len(symbols_by_id)) in overview["repo_summary"]


def test_architecture_overview_repo_summary_includes_convention_facts(handlers_parsed):
    report = mine_conventions([handlers_parsed])
    graph = build_graph([handlers_parsed])
    symbols_by_id = {s.id: s for s in handlers_parsed.symbols}

    overview = compute_architecture_overview(report, graph, symbols_by_id)

    assert "Error handling: except ValueError: logging.error" in overview["repo_summary"]


def test_architecture_overview_cluster_member_count_matches_real_cluster_size(two_subsystem_repo):
    graph, symbols_by_id = two_subsystem_repo

    overview = compute_architecture_overview(_empty_report(), graph, symbols_by_id)

    for cluster_summary in overview["clusters"]:
        real_cluster = graph.clusters.coarse[cluster_summary["cluster_index"]]
        assert cluster_summary["member_count"] == len(real_cluster)


# --------------------------------------------------------------------------
# §5 acceptance criterion: no-op regeneration when nothing changed
# --------------------------------------------------------------------------


def test_regenerating_with_no_changes_produces_no_content(handlers_parsed):
    report = mine_conventions([handlers_parsed])
    graph = build_graph([handlers_parsed])
    symbols_by_id = {s.id: s for s in handlers_parsed.symbols}

    first = generate_claude_md(report, graph, symbols_by_id)
    assert first.regenerated
    second = generate_claude_md(report, graph, symbols_by_id, previous_state=first.state)

    assert second.regenerated is False
    assert second.content is None


# --------------------------------------------------------------------------
# §5 acceptance criterion: structured diff names the specific change
# --------------------------------------------------------------------------


def test_regenerating_after_a_convention_shift_produces_a_structured_diff_not_a_text_diff():
    import networkx as nx

    from loupe_core.graph.builder import LoupeGraph

    empty_graph = LoupeGraph(graph=nx.DiGraph())
    before_report = _synthetic_report(error_confidence_pct=100, error_total=10)
    after_report = ConventionsReport(
        error_handling=ErrorHandlingConvention(
            majority_pattern="except Exception: print", violation_count=1, total_count=10
        ),
        docstrings=before_report.docstrings,
        imports=before_report.imports,
    )

    before = generate_claude_md(before_report, empty_graph, {})
    after = generate_claude_md(after_report, empty_graph, {}, previous_state=before.state)

    assert after.regenerated
    assert len(after.diff_lines) == 1
    assert "except ValueError: logging.error" in after.diff_lines[0]
    assert "except Exception: print" in after.diff_lines[0]
    assert "convention shifted" in after.diff_lines[0]


def test_new_architectural_cluster_reported_by_diff(two_subsystem_repo):
    graph, symbols_by_id = two_subsystem_repo
    empty_report = ConventionsReport(
        error_handling=ErrorHandlingConvention(majority_pattern=None),
        docstrings=DocstringConvention(coverage_pct=0.0, dominant_style="none"),
        imports=ImportConvention(dominant_style="absolute", relative_count=0, absolute_count=0),
    )

    import networkx as nx

    from loupe_core.graph.builder import LoupeGraph

    empty_graph = LoupeGraph(graph=nx.DiGraph())
    before = generate_claude_md(empty_report, empty_graph, {})  # no architecture at all yet
    after = generate_claude_md(empty_report, graph, symbols_by_id, previous_state=before.state)

    assert after.regenerated
    new_cluster_lines = [line for line in after.diff_lines if "New architectural cluster detected" in line]
    assert len(new_cluster_lines) == 2  # both sys_a and sys_b clusters are new relative to "no architecture yet"


# --------------------------------------------------------------------------
# Small building-block tests
# --------------------------------------------------------------------------


def test_render_document_omits_empty_sections():
    doc = render_document([], [])
    assert "## Conventions" not in doc
    assert "## Architecture" not in doc
    assert doc.startswith("# CLAUDE.md")


def test_compute_input_hash_is_deterministic_and_order_independent():
    facts_a = [ConventionFact("error_handling", "x", 0.9), ConventionFact("imports", "y", 0.8)]
    facts_b = [ConventionFact("imports", "y", 0.8), ConventionFact("error_handling", "x", 0.9)]
    assert compute_input_hash(facts_a, []) == compute_input_hash(facts_b, [])


def test_generator_state_round_trips_through_json():
    state = GeneratorState(
        input_hash="abc123",
        facts=[ConventionFact("error_handling", "x", 0.9)],
        architecture=[ArchitectureEntry(0, "sid", "qn", "f.py", 0.5)],
        clusters=[["sid", "other"]],
    )
    restored = GeneratorState.from_json(state.to_json())
    assert restored.input_hash == state.input_hash
    assert restored.facts == state.facts
    assert restored.architecture == state.architecture
    assert restored.clusters == state.clusters
