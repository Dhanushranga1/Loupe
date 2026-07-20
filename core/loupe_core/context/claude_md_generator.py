"""CLAUDE.md generator: static-context budgeting via knapsack, community-detection-
informed architecture summarization, and structured diffing between regenerations
(docs/PhaseX/claude-md-generator.md).

`CLAUDE.md` is itself static context loaded into every session — the same
token-budget discipline Loupe applies to dynamic retrieval (Phase 3) applies
here, so this module reuses Phase 3's knapsack selector directly rather than
dumping every detected fact into an unbounded document.

Framework-free by design, like every other `loupe_core` module: this file
only computes; `cli/loupe_cli/main.py`'s `cmd_generate_context` owns all file
I/O (reading/writing `CLAUDE.md` and the persisted `GeneratorState`) and the
human-review gate (never auto-committed).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

from loupe_core.conventions.mining import ConventionsReport
from loupe_core.governor.budget import estimate_tokens
from loupe_core.governor.knapsack import KnapsackCandidate, knapsack_greedy
from loupe_core.graph.builder import LoupeGraph
from loupe_core.graph.clustering import align_clusters
from loupe_core.parsing.schema import Symbol

DEFAULT_DOCUMENT_BUDGET = 1500

# The spec names two illustrative endpoints ("a 95%-consistent majority
# pattern is worth including; a 55%-consistent one is noise") without a exact
# cutoff. 0.7 sits clearly inside "worth including" and clearly above "noise"
# for both named examples — a documented, revisit-eligible constant in the
# same spirit as RRF's k=60 or MMR's lambda=0.7, not a claim of a uniquely
# correct threshold.
CONFIDENCE_THRESHOLD = 0.7


# --------------------------------------------------------------------------
# Convention facts: value = E4's own confidence in the majority pattern
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ConventionFact:
    kind: str  # "error_handling" | "docstrings" | "imports" — a stable diff key across generations
    text: str
    confidence: float


def _error_handling_confidence(convention) -> float:
    if convention.total_count == 0:
        return 0.0
    return (convention.total_count - convention.violation_count) / convention.total_count


def _docstring_confidence(convention) -> float:
    if convention.documented_count == 0:
        return 0.0
    return convention.dominant_style_count / convention.documented_count


def _import_confidence(convention) -> float:
    total = convention.relative_count + convention.absolute_count
    if total == 0:
        return 0.0
    return max(convention.relative_count, convention.absolute_count) / total


def compute_convention_facts(report: ConventionsReport) -> list[ConventionFact]:
    """One fact per convention category actually detected (majority pattern exists,
    at least one documented symbol, at least one import) — each carrying E4's own
    confidence, unfiltered here (filtering by `CONFIDENCE_THRESHOLD` is a separate
    step, kept distinct so it stays independently testable).
    """
    facts: list[ConventionFact] = []

    if report.error_handling.majority_pattern is not None:
        facts.append(
            ConventionFact(
                kind="error_handling",
                text=f"Error handling: {report.error_handling.majority_pattern}",
                confidence=_error_handling_confidence(report.error_handling),
            )
        )

    if report.docstrings.dominant_style != "none":
        facts.append(
            ConventionFact(
                kind="docstrings",
                text=f"Docstring style: {report.docstrings.dominant_style} "
                f"({report.docstrings.coverage_pct:.0f}% coverage)",
                confidence=_docstring_confidence(report.docstrings),
            )
        )

    if report.imports.relative_count + report.imports.absolute_count > 0:
        facts.append(
            ConventionFact(
                kind="imports",
                text=f"Import style: {report.imports.dominant_style}",
                confidence=_import_confidence(report.imports),
            )
        )

    return facts


# --------------------------------------------------------------------------
# Architecture entries: one representative hub per Louvain community, not
# flat top-N PageRank (§2) — reuses Phase 10.5's coarse clustering directly.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ArchitectureEntry:
    cluster_index: int  # index into graph.clusters.coarse — the diff key across generations
    symbol_id: str
    qualified_name: str
    file_path: str
    pagerank_score: float


def compute_architecture_entries(graph: LoupeGraph, symbols_by_id: dict[str, Symbol]) -> list[ArchitectureEntry]:
    """The highest-PageRank symbol *within* each coarse cluster — guarantees the
    summary spans the repo's real subsystem boundaries instead of over-representing
    whichever subsystem happens to be globally most interconnected.
    """
    entries: list[ArchitectureEntry] = []
    for cluster_index, cluster in enumerate(graph.clusters.coarse):
        if not cluster:
            continue
        hub_id = max(cluster, key=lambda sid: (graph.pagerank_scores.get(sid, 0.0), sid))
        symbol = symbols_by_id.get(hub_id)
        if symbol is None:
            continue
        entries.append(
            ArchitectureEntry(
                cluster_index=cluster_index,
                symbol_id=hub_id,
                qualified_name=symbol.qualified_name,
                file_path=symbol.file_path,
                pagerank_score=graph.pagerank_scores.get(hub_id, 0.0),
            )
        )
    return entries


# --------------------------------------------------------------------------
# architecture://overview (docs/PhaseX/phase-14-adaptive-context-compression.md
# §1's L0/L1 LOD levels) — reuses this module's own convention/architecture
# computation exactly, the same underlying data `loupe generate-context`
# renders into CLAUDE.md, just a different delivery mechanism (a live MCP
# Resource read, not a committed file). Deliberately not knapsack-budgeted or
# freshness-cached like the CLAUDE.md path: a resource read is already cheap
# (conventions plus one hub per coarse cluster, no document-length constraint
# to enforce) and always reflects the current index — the same "always
# current" bar `conventions://summary` already holds itself to.
# --------------------------------------------------------------------------


def compute_architecture_overview(
    conventions_report: ConventionsReport, graph: LoupeGraph, symbols_by_id: dict[str, Symbol]
) -> dict:
    # Same confidence-threshold filtering as the CLAUDE.md path (§1's own
    # framing: "the exact same underlying computation") — a live resource
    # shouldn't state a less-trustworthy convention than the committed
    # document would.
    facts = [f for f in compute_convention_facts(conventions_report) if f.confidence >= CONFIDENCE_THRESHOLD]
    architecture = compute_architecture_entries(graph, symbols_by_id)

    summary_sentences = [f"This repository has {len(symbols_by_id)} symbols across {len(architecture)} architectural clusters."]
    summary_sentences += [f"{fact.text}." for fact in facts]

    clusters = []
    for entry in architecture:
        clusters.append(
            {
                "cluster_index": entry.cluster_index,
                "hub_symbol_id": entry.symbol_id,
                "hub_qualified_name": entry.qualified_name,
                "hub_file_path": entry.file_path,
                "member_count": len(graph.clusters.coarse[entry.cluster_index]),
            }
        )

    return {"repo_summary": " ".join(summary_sentences), "clusters": clusters}


# --------------------------------------------------------------------------
# Knapsack selection over both candidate types combined (§1)
# --------------------------------------------------------------------------


def _select_within_budget(
    facts: list[ConventionFact], architecture: list[ArchitectureEntry], document_budget: int
) -> tuple[list[ConventionFact], list[ArchitectureEntry]]:
    candidates: list[KnapsackCandidate] = []
    for fact in facts:
        candidates.append(KnapsackCandidate(f"fact:{fact.kind}", fact.confidence, estimate_tokens(fact.text)))
    for entry in architecture:
        label = f"`{entry.qualified_name}` ({entry.file_path})"
        candidates.append(KnapsackCandidate(f"arch:{entry.cluster_index}", entry.pagerank_score, estimate_tokens(label)))

    selected_ids = set(knapsack_greedy(candidates, document_budget))
    selected_facts = [f for f in facts if f"fact:{f.kind}" in selected_ids]
    selected_architecture = [e for e in architecture if f"arch:{e.cluster_index}" in selected_ids]
    return selected_facts, selected_architecture


# --------------------------------------------------------------------------
# Rendering — structured-data-to-template, plain Python string composition.
#
# The spec suggests Jinja2 ("the same underlying approach Scaffold already
# uses for its bricks"), but Scaffold's brick/compose system was never
# actually built (docs/progress/README.md — elicitation-only, brick system
# blocked), so there is no real precedent to match, and this document's
# structure is three fixed sections with no loops/conditionals complex enough
# to need a template engine. Checked, not assumed: adding a new formal
# dependency for a handful of f-strings would be exactly the kind of
# unjustified addition this project's own conventions warn against.
# --------------------------------------------------------------------------


def render_document(facts: list[ConventionFact], architecture: list[ArchitectureEntry]) -> str:
    lines = ["# CLAUDE.md", "", "<!-- Generated by `loupe generate-context`. Review before committing. -->", ""]

    if facts:
        lines += ["## Conventions", ""]
        lines += [f"- {fact.text}" for fact in facts]
        lines.append("")

    if architecture:
        lines += ["## Architecture / Core Modules", ""]
        lines += [f"- `{entry.qualified_name}` ({entry.file_path})" for entry in architecture]
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


# --------------------------------------------------------------------------
# Freshness: content-hash-based invalidation (§4), same mechanism as Phase 2's
# embedding cache — regenerate only when the underlying facts actually changed.
# --------------------------------------------------------------------------


def _canonical_json(facts: list[ConventionFact], architecture: list[ArchitectureEntry]) -> str:
    payload = {
        "facts": sorted(
            ({"kind": f.kind, "text": f.text, "confidence": round(f.confidence, 6)} for f in facts),
            key=lambda d: d["kind"],
        ),
        "architecture": sorted(
            (
                {"cluster_index": e.cluster_index, "qualified_name": e.qualified_name, "file_path": e.file_path}
                for e in architecture
            ),
            key=lambda d: d["cluster_index"],
        ),
    }
    return json.dumps(payload, sort_keys=True)


def compute_input_hash(facts: list[ConventionFact], architecture: list[ArchitectureEntry]) -> str:
    return hashlib.sha256(_canonical_json(facts, architecture).encode("utf-8")).hexdigest()


# --------------------------------------------------------------------------
# Persisted state — what the CLI layer round-trips through JSON so the next
# `loupe generate-context` run has something to diff/compare against.
# --------------------------------------------------------------------------


@dataclass
class GeneratorState:
    input_hash: str
    facts: list[ConventionFact] = field(default_factory=list)
    architecture: list[ArchitectureEntry] = field(default_factory=list)
    # Every coarse cluster's full symbol-id membership from the run that
    # produced `architecture` — needed (not just the hub list) so a later
    # run can Jaccard-align clusters across two genuinely different
    # partitions, exactly the mechanism Phase 10.5 built for this purpose.
    clusters: list[list[str]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "input_hash": self.input_hash,
                "facts": [{"kind": f.kind, "text": f.text, "confidence": f.confidence} for f in self.facts],
                "architecture": [
                    {
                        "cluster_index": e.cluster_index,
                        "symbol_id": e.symbol_id,
                        "qualified_name": e.qualified_name,
                        "file_path": e.file_path,
                        "pagerank_score": e.pagerank_score,
                    }
                    for e in self.architecture
                ],
                "clusters": self.clusters,
            }
        )

    @classmethod
    def from_json(cls, raw: str) -> "GeneratorState":
        data = json.loads(raw)
        return cls(
            input_hash=data["input_hash"],
            facts=[ConventionFact(**f) for f in data["facts"]],
            architecture=[ArchitectureEntry(**e) for e in data["architecture"]],
            clusters=data["clusters"],
        )


# --------------------------------------------------------------------------
# Structured diffing (§3) — compares underlying data between generations,
# not rendered text, so the summary names the specific change instead of
# being buried in re-wrapped prose.
# --------------------------------------------------------------------------


def _diff_conventions(old_facts: list[ConventionFact], new_facts: list[ConventionFact]) -> list[str]:
    old_by_kind = {f.kind: f for f in old_facts}
    new_by_kind = {f.kind: f for f in new_facts}
    lines: list[str] = []

    for kind in sorted(set(old_by_kind) | set(new_by_kind)):
        old, new = old_by_kind.get(kind), new_by_kind.get(kind)
        if old == new:
            continue
        if old is None:
            lines.append(f"New convention detected: {new.text} (confidence {new.confidence:.0%}).")
        elif new is None:
            lines.append(f"Convention no longer detected: {old.text}.")
        elif old.text != new.text:
            lines.append(
                f"{kind} convention shifted: {old.text!r} (confidence {old.confidence:.0%}) "
                f"-> {new.text!r} (confidence {new.confidence:.0%})."
            )
        else:  # same text, confidence moved enough to matter
            lines.append(f"{kind} convention confidence changed: {old.confidence:.0%} -> {new.confidence:.0%}.")

    return lines


def _diff_architecture(
    old_clusters: list[list[str]], old_architecture: list[ArchitectureEntry], new_graph: LoupeGraph, new_architecture: list[ArchitectureEntry]
) -> list[str]:
    old_hub_by_cluster = {e.cluster_index: e.qualified_name for e in old_architecture}
    new_hub_by_cluster = {e.cluster_index: e.qualified_name for e in new_architecture}

    old_sets = [set(c) for c in old_clusters]
    new_sets = list(new_graph.clusters.coarse)
    alignment = align_clusters(old_sets, new_sets)

    lines: list[str] = []
    for new_index in sorted(new_hub_by_cluster):
        old_index = alignment.get(new_index)
        new_hub = new_hub_by_cluster[new_index]
        if old_index is None:
            lines.append(f"New architectural cluster detected, centered on `{new_hub}`.")
            continue
        old_hub = old_hub_by_cluster.get(old_index)
        if old_hub is not None and old_hub != new_hub:
            lines.append(f"Architectural cluster's representative hub changed: `{old_hub}` -> `{new_hub}`.")

    return lines


# --------------------------------------------------------------------------
# Top-level entry point
# --------------------------------------------------------------------------


@dataclass
class GenerationResult:
    content: str | None  # None when nothing changed since `previous_state` (freshness short-circuit)
    regenerated: bool
    diff_lines: list[str]
    state: GeneratorState  # always the *current* state, persist regardless of `regenerated`


def generate_claude_md(
    conventions_report: ConventionsReport,
    graph: LoupeGraph,
    symbols_by_id: dict[str, Symbol],
    previous_state: GeneratorState | None = None,
    document_budget: int = DEFAULT_DOCUMENT_BUDGET,
) -> GenerationResult:
    all_facts = compute_convention_facts(conventions_report)
    confident_facts = [f for f in all_facts if f.confidence >= CONFIDENCE_THRESHOLD]
    all_architecture = compute_architecture_entries(graph, symbols_by_id)

    input_hash = compute_input_hash(confident_facts, all_architecture)
    current_clusters = [sorted(c) for c in graph.clusters.coarse]

    if previous_state is not None and previous_state.input_hash == input_hash:
        return GenerationResult(content=None, regenerated=False, diff_lines=[], state=previous_state)

    selected_facts, selected_architecture = _select_within_budget(confident_facts, all_architecture, document_budget)
    content = render_document(selected_facts, selected_architecture)

    diff_lines: list[str] = []
    if previous_state is not None:
        diff_lines = _diff_conventions(previous_state.facts, confident_facts) + _diff_architecture(
            previous_state.clusters, previous_state.architecture, graph, all_architecture
        )

    new_state = GeneratorState(
        input_hash=input_hash, facts=confident_facts, architecture=all_architecture, clusters=current_clusters
    )
    return GenerationResult(content=content, regenerated=True, diff_lines=diff_lines, state=new_state)
