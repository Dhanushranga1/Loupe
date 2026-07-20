"""Evaluation harness: three retrieval strategies x two conditions (docs/phase-5-evaluation.md §5-6).

All three strategies operate on the repo's state as of each task's *parent*
commit, reconstructed via `git show`-equivalent blob reads — never a real
checkout. An ephemeral, in-memory index is built per task; nothing is
written to `.loupe/` during evaluation.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path

import git

from loupe_core.eval.metrics import chunk_containment, recall_at_k, token_cost
from loupe_core.eval.mine_history import BenchmarkTask, mine_history, parsed_file_from_git_blob
from loupe_core.governor.budget import symbol_extraction_cost
from loupe_core.governor.knapsack import KnapsackCandidate, knapsack_greedy
from loupe_core.governor.session import DEFAULT_BUDGET
from loupe_core.graph.builder import LoupeGraph, ParsedFile, build_graph
from loupe_core.graph.centrality import compute_personalized_pagerank
from loupe_core.parsing.schema import Symbol
from loupe_core.retrieval.fusion import CANDIDATE_POOL_SIZE, FINAL_TOP_K, fuse
from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.semantic import SemanticIndex

# Strategy B's fixed chunk window — the naive chunking Loupe itself avoids (§5).
CHUNK_TOKEN_SIZE = 500
# Not pinned by the spec for strategy B; matches Loupe's own final_top_k for
# a fair, like-for-like comparison rather than an arbitrary different number.
VECTOR_RAG_TOP_K = FINAL_TOP_K
NAIVE_TOP_FILES = 5


@dataclass
class RepoSnapshot:
    """All *.py files' content + derived indexes as of one commit — no working-tree checkout."""

    commit_sha: str
    files: dict[str, bytes]
    parsed_files: dict[str, ParsedFile]
    symbols: list[Symbol]
    graph: LoupeGraph
    lexical_index: LexicalIndex
    semantic_index: SemanticIndex


@dataclass
class StrategyResult:
    retrieved_symbol_ids: list[str]
    retrieved_content: list[str]
    chunk_containments: list[float] = field(default_factory=list)


def build_repo_snapshot(repo: git.Repo, commit_sha: str, embedding_model: object | None = None) -> RepoSnapshot:
    """Reconstruct the repo's *.py file set + symbols/graph/indexes as of `commit_sha`."""
    commit = repo.commit(commit_sha)
    file_paths = [line for line in repo.git.ls_tree("-r", commit_sha, "--name-only").splitlines() if line.endswith(".py")]

    files: dict[str, bytes] = {}
    parsed_files: dict[str, ParsedFile] = {}
    for rel_path in file_paths:
        source_bytes = (commit.tree / rel_path).data_stream.read()
        files[rel_path] = source_bytes
        parsed_files[rel_path] = parsed_file_from_git_blob(rel_path, source_bytes)

    all_symbols = [s for pf in parsed_files.values() for s in pf.symbols]
    graph = build_graph(list(parsed_files.values()))
    lexical_index = LexicalIndex(all_symbols)
    semantic_index = SemanticIndex(model=embedding_model)
    semantic_index.index(all_symbols)

    return RepoSnapshot(
        commit_sha=commit_sha,
        files=files,
        parsed_files=parsed_files,
        symbols=all_symbols,
        graph=graph,
        lexical_index=lexical_index,
        semantic_index=semantic_index,
    )


# --------------------------------------------------------------------------
# Strategy A — naive whole-file loading (§5.A)
# --------------------------------------------------------------------------


def _significant_words(text: str) -> list[str]:
    """Deliberately unsophisticated: no stemming, no dedup, no stopword list (§5's stated framing)."""
    return [w for w in re.findall(r"[a-zA-Z]+", text.lower()) if len(w) >= 3]


def strategy_a_naive_whole_file(
    snapshot: RepoSnapshot, task_description: str, ground_truth_files: list[str] | None = None
) -> StrategyResult:
    if ground_truth_files is not None:
        candidate_files = [f for f in ground_truth_files if f in snapshot.files]
    else:
        words = _significant_words(task_description)
        scored = []
        for rel_path, source_bytes in snapshot.files.items():
            content_lower = source_bytes.decode("utf-8", errors="ignore").lower()
            score = sum(content_lower.count(w) for w in words)
            scored.append((rel_path, score))
        scored.sort(key=lambda pair: -pair[1])
        candidate_files = [path for path, _ in scored[:NAIVE_TOP_FILES]]

    retrieved_content = [snapshot.files[f].decode("utf-8", errors="ignore") for f in candidate_files]
    retrieved_symbol_ids = [s.id for f in candidate_files for s in snapshot.parsed_files[f].symbols]

    return StrategyResult(retrieved_symbol_ids=retrieved_symbol_ids, retrieved_content=retrieved_content)


# --------------------------------------------------------------------------
# Strategy B — fixed-chunk vector-RAG baseline (§5.B)
# --------------------------------------------------------------------------


def _chunk_file_by_tokens(source_bytes: bytes, chunk_token_size: int = CHUNK_TOKEN_SIZE) -> list[tuple[str, tuple[int, int]]]:
    """Fixed ~chunk_token_size-token windows, split on line boundaries; returns (text, (line_start, line_end))."""
    from loupe_core.governor.budget import estimate_tokens

    lines = source_bytes.decode("utf-8", errors="ignore").split("\n")
    chunks: list[tuple[str, tuple[int, int]]] = []
    current_lines: list[str] = []
    current_start_line = 1
    current_token_count = 0

    for i, line in enumerate(lines, start=1):
        line_tokens = estimate_tokens(line)
        if current_lines and current_token_count + line_tokens > chunk_token_size:
            chunks.append(("\n".join(current_lines), (current_start_line, i - 1)))
            current_lines = []
            current_start_line = i
            current_token_count = 0
        current_lines.append(line)
        current_token_count += line_tokens

    if current_lines:
        chunks.append(("\n".join(current_lines), (current_start_line, len(lines))))

    return chunks


def strategy_b_vector_rag(
    snapshot: RepoSnapshot,
    task_description: str,
    ground_truth_files: list[str] | None = None,
    top_k: int = VECTOR_RAG_TOP_K,
    embedding_model: object | None = None,
) -> StrategyResult:
    candidate_files = [f for f in (ground_truth_files or snapshot.files) if f in snapshot.files]

    all_chunks: list[tuple[str, str, tuple[int, int]]] = []  # (rel_path, text, line_range)
    for rel_path in candidate_files:
        for chunk_text, line_range in _chunk_file_by_tokens(snapshot.files[rel_path]):
            all_chunks.append((rel_path, chunk_text, line_range))

    if not all_chunks:
        return StrategyResult(retrieved_symbol_ids=[], retrieved_content=[])

    from loupe_core.retrieval.semantic import get_default_model

    model = embedding_model if embedding_model is not None else get_default_model()
    chunk_texts = [text for _, text, _ in all_chunks]
    chunk_embeddings = model.encode(chunk_texts, normalize_embeddings=True)
    query_embedding = model.encode([task_description], normalize_embeddings=True)[0]

    import numpy as np

    similarities = np.array(chunk_embeddings) @ np.array(query_embedding)
    ranked_indices = np.argsort(-similarities)[:top_k]

    retrieved_content = []
    # (rel_path, line_range, similarity) per retrieved chunk, in similarity-descending order.
    retrieved_chunks: list[tuple[str, tuple[int, int], float]] = []
    for idx in ranked_indices:
        rel_path, text, line_range = all_chunks[idx]
        retrieved_content.append(text)
        retrieved_chunks.append((rel_path, line_range, float(similarities[idx])))

    # For each touched symbol, track its best (highest-similarity) covering
    # chunk — this is what makes the shared recall_at_k's [:k] slice
    # meaningful for strategy B too, ranking by confidence rather than
    # arbitrary file/definition order (the same rank-order pitfall fixed in
    # strategy C just above, and in Phase 2's fusion.py before that).
    best_similarity_by_symbol: dict[str, float] = {}
    containment_by_symbol: dict[str, float] = {}
    for symbol in snapshot.symbols:
        for rel_path, line_range, similarity in retrieved_chunks:
            if symbol.file_path != rel_path:
                continue
            containment = chunk_containment(line_range, (symbol.line_start, symbol.line_end))
            if containment > 0:
                if similarity > best_similarity_by_symbol.get(symbol.id, float("-inf")):
                    best_similarity_by_symbol[symbol.id] = similarity
                containment_by_symbol[symbol.id] = max(containment_by_symbol.get(symbol.id, 0.0), containment)

    retrieved_symbol_ids = sorted(best_similarity_by_symbol, key=lambda sid: (-best_similarity_by_symbol[sid], sid))
    chunk_containments = [containment_by_symbol[sid] for sid in retrieved_symbol_ids]

    return StrategyResult(
        retrieved_symbol_ids=retrieved_symbol_ids, retrieved_content=retrieved_content, chunk_containments=chunk_containments
    )


# --------------------------------------------------------------------------
# Strategy C — Loupe (§5.C)
# --------------------------------------------------------------------------


def strategy_c_loupe_oracle(snapshot: RepoSnapshot, ground_truth_symbol_ids: list[str]) -> StrategyResult:
    """Oracle-file condition: extract exactly the ground-truth symbols — no retrieval/governor step."""
    symbols_by_id = {s.id: s for s in snapshot.symbols}
    retrieved_ids, retrieved_content = [], []
    for symbol_id in ground_truth_symbol_ids:
        symbol = symbols_by_id.get(symbol_id)
        if symbol is None:
            continue
        pf = snapshot.parsed_files[symbol.file_path]
        retrieved_content.append(pf.source_bytes[symbol.byte_start : symbol.byte_end].decode("utf-8"))
        retrieved_ids.append(symbol_id)
    return StrategyResult(retrieved_symbol_ids=retrieved_ids, retrieved_content=retrieved_content)


def strategy_c_loupe_end_to_end(
    snapshot: RepoSnapshot,
    task_description: str,
    token_budget: int = DEFAULT_BUDGET,
    use_personalized_pagerank: bool = True,
    repo: git.Repo | None = None,
    use_churn: bool = False,
) -> StrategyResult:
    """`use_personalized_pagerank` defaults to True — retrieval-upgrades §1's real,
    intended pipeline — but stays a parameter (not hardcoded) so
    `run_personalized_pagerank_ablation` below can call this same strategy function
    both ways rather than maintaining a second, parallel implementation.

    `use_churn`/`repo` (Phase 14 §2's own ablation, same pattern): `repo` is
    needed because `RepoSnapshot` itself carries no live `git.Repo` handle,
    only file content as of one commit. Churn is computed with
    `now=<this snapshot's own commit time>`, not real current time — using
    real "now" would leak information about commits *after* the benchmark
    task's cutoff into a historical evaluation, silently inflating recall.
    """
    churn_scores = None
    if use_churn and repo is not None:
        from loupe_core.retrieval.churn import compute_churn_scores

        commit = repo.commit(snapshot.commit_sha)
        churn_scores = compute_churn_scores(repo, snapshot.symbols, now=commit.committed_date)

    fused = fuse(
        snapshot.lexical_index.query(task_description, top_k=CANDIDATE_POOL_SIZE),
        snapshot.semantic_index.query(task_description, top_k=CANDIDATE_POOL_SIZE),
        snapshot.graph.pagerank_scores,
        graph=snapshot.graph.graph if use_personalized_pagerank else None,
        top_k=FINAL_TOP_K,
        churn_scores=churn_scores,
    )
    symbols_by_id = {s.id: s for s in snapshot.symbols}

    candidates: list[KnapsackCandidate] = []
    for symbol_id, score in fused:
        symbol = symbols_by_id.get(symbol_id)
        if symbol is None:
            continue
        cost = symbol_extraction_cost(symbol, snapshot.parsed_files[symbol.file_path].source_bytes)
        candidates.append(KnapsackCandidate(symbol_id, score, cost))

    selected_ids = set(knapsack_greedy(candidates, token_budget))

    # Preserve the original RRF rank order (best-first) rather than iterating
    # the unordered `selected_ids` set directly — the same hash-randomization
    # pitfall found and fixed in Phase 2's fusion.py. Without this, recall_at_k's
    # [:k] slice truncates an effectively arbitrary order, not a ranked one.
    retrieved_ids, retrieved_content = [], []
    for symbol_id, _score in fused:
        if symbol_id not in selected_ids:
            continue
        symbol = symbols_by_id[symbol_id]
        pf = snapshot.parsed_files[symbol.file_path]
        retrieved_content.append(pf.source_bytes[symbol.byte_start : symbol.byte_end].decode("utf-8"))
        retrieved_ids.append(symbol_id)

    return StrategyResult(retrieved_symbol_ids=retrieved_ids, retrieved_content=retrieved_content)


# --------------------------------------------------------------------------
# The two comparison conditions (§6)
# --------------------------------------------------------------------------


def run_oracle_condition(repo: git.Repo, tasks: list[BenchmarkTask], embedding_model: object | None = None) -> dict:
    """Isolates chunking granularity: all strategies get the correct files directly."""
    per_strategy_tokens: dict[str, list[int]] = {"naive": [], "vector_rag": [], "loupe": []}

    for task in tasks:
        snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=embedding_model)
        a = strategy_a_naive_whole_file(snapshot, task.task_description, ground_truth_files=task.ground_truth_files)
        c = strategy_c_loupe_oracle(snapshot, task.ground_truth_symbol_ids)

        per_strategy_tokens["naive"].append(token_cost(a.retrieved_content))
        per_strategy_tokens["loupe"].append(token_cost(c.retrieved_content))

    return {
        "naive_total_tokens": sum(per_strategy_tokens["naive"]),
        "loupe_total_tokens": sum(per_strategy_tokens["loupe"]),
        "ratio": (
            sum(per_strategy_tokens["naive"]) / sum(per_strategy_tokens["loupe"])
            if sum(per_strategy_tokens["loupe"]) > 0
            else None
        ),
        "per_task_tokens": per_strategy_tokens,
    }


def _aggregate(values: list[float]) -> dict[str, float | None]:
    scored = [v for v in values if v is not None]
    if not scored:
        return {"mean": None, "median": None, "n": 0}
    return {"mean": statistics.mean(scored), "median": statistics.median(scored), "n": len(scored)}


def run_end_to_end_condition(repo: git.Repo, tasks: list[BenchmarkTask], embedding_model: object | None = None) -> dict:
    """No file hints: all strategies must find relevant content from task_description alone."""
    results: dict[str, dict[str, list]] = {
        s: {"recall_5": [], "recall_10": [], "tokens": []} for s in ("naive", "vector_rag", "loupe")
    }

    for task in tasks:
        parent_sha = task.commit_sha + "^"
        snapshot = build_repo_snapshot(repo, parent_sha, embedding_model=embedding_model)
        ground_truth = set(task.ground_truth_symbol_ids)

        a = strategy_a_naive_whole_file(snapshot, task.task_description)
        b = strategy_b_vector_rag(snapshot, task.task_description, embedding_model=embedding_model)
        c = strategy_c_loupe_end_to_end(snapshot, task.task_description)

        for name, result in (("naive", a), ("vector_rag", b), ("loupe", c)):
            results[name]["recall_5"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=5))
            results[name]["recall_10"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=10))
            results[name]["tokens"].append(token_cost(result.retrieved_content))

    return {
        name: {
            "recall_5": _aggregate(data["recall_5"]),
            "recall_10": _aggregate(data["recall_10"]),
            "tokens": _aggregate(data["tokens"]),
        }
        for name, data in results.items()
    }


# --------------------------------------------------------------------------
# Personalized PageRank ablation (docs/PhaseX/loupe-retrieval-upgrades.md §2,
# acceptance criterion 1: "reusing Phase 5's harness directly, compare
# recall@k with static vs. personalized PageRank as RRF's centrality term").
# Reuses `strategy_c_loupe_end_to_end` both ways via its own
# `use_personalized_pagerank` flag rather than a second strategy function.
# --------------------------------------------------------------------------


def run_personalized_pagerank_ablation(repo: git.Repo, tasks: list[BenchmarkTask], embedding_model: object | None = None) -> dict:
    """Same end-to-end condition as `run_end_to_end_condition`, but comparing Loupe
    against itself with static vs. personalized PageRank as RRF's centrality term —
    isolates the one variable this ablation exists to answer, everything else
    (lexical/semantic candidates, knapsack budget) held identical per task.
    """
    results: dict[str, dict[str, list]] = {
        "static_pagerank": {"recall_5": [], "recall_10": [], "tokens": []},
        "personalized_pagerank": {"recall_5": [], "recall_10": [], "tokens": []},
    }

    for task in tasks:
        snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=embedding_model)
        ground_truth = set(task.ground_truth_symbol_ids)

        static = strategy_c_loupe_end_to_end(snapshot, task.task_description, use_personalized_pagerank=False)
        personalized = strategy_c_loupe_end_to_end(snapshot, task.task_description, use_personalized_pagerank=True)

        for name, result in (("static_pagerank", static), ("personalized_pagerank", personalized)):
            results[name]["recall_5"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=5))
            results[name]["recall_10"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=10))
            results[name]["tokens"].append(token_cost(result.retrieved_content))

    return {
        name: {
            "recall_5": _aggregate(data["recall_5"]),
            "recall_10": _aggregate(data["recall_10"]),
            "tokens": _aggregate(data["tokens"]),
        }
        for name, data in results.items()
    }


# --------------------------------------------------------------------------
# Churn ablation (docs/PhaseX/phase-14-adaptive-context-compression.md §2's
# own acceptance criterion: "reusing Phase 5's harness exactly as
# personalized PageRank's was, run the benchmark with and without churn as
# an RRF signal, compare recall@k"). Reuses `strategy_c_loupe_end_to_end`
# both ways via its own `use_churn` flag, the same pattern as
# `run_personalized_pagerank_ablation` above.
# --------------------------------------------------------------------------


def run_churn_ablation(repo: git.Repo, tasks: list[BenchmarkTask], embedding_model: object | None = None) -> dict:
    results: dict[str, dict[str, list]] = {
        "without_churn": {"recall_5": [], "recall_10": [], "tokens": []},
        "with_churn": {"recall_5": [], "recall_10": [], "tokens": []},
    }

    for task in tasks:
        snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=embedding_model)
        ground_truth = set(task.ground_truth_symbol_ids)

        without_churn = strategy_c_loupe_end_to_end(snapshot, task.task_description, repo=repo, use_churn=False)
        with_churn = strategy_c_loupe_end_to_end(snapshot, task.task_description, repo=repo, use_churn=True)

        for name, result in (("without_churn", without_churn), ("with_churn", with_churn)):
            results[name]["recall_5"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=5))
            results[name]["recall_10"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=10))
            results[name]["tokens"].append(token_cost(result.retrieved_content))

    return {
        name: {
            "recall_5": _aggregate(data["recall_5"]),
            "recall_10": _aggregate(data["recall_10"]),
            "tokens": _aggregate(data["tokens"]),
        }
        for name, data in results.items()
    }


# --------------------------------------------------------------------------
# Top-level harness runner
# --------------------------------------------------------------------------


def run_harness(
    repo_path: str, max_commits: int = 50, embedding_model: object | None = None, results_dir: str | Path = "benchmarks/results"
) -> dict:
    """Mine `repo_path`'s history, run both conditions, write a timestamped results JSON."""
    import datetime
    import json

    repo = git.Repo(repo_path)
    tasks = mine_history(repo_path, max_commits=max_commits)

    oracle = run_oracle_condition(repo, tasks, embedding_model=embedding_model)
    end_to_end = run_end_to_end_condition(repo, tasks, embedding_model=embedding_model)

    report = {
        "repo_path": str(repo_path),
        "task_count": len(tasks),
        "oracle_condition": oracle,
        "end_to_end_condition": end_to_end,
    }

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_file = results_path / f"{timestamp}.json"
    out_file.write_text(json.dumps(report, indent=2))
    report["_output_path"] = str(out_file)

    return report


# --------------------------------------------------------------------------
# Phase 6 addition (docs/phase-6-closing-the-loop.md §7): the learned-ranker
# strategy and its comparison runner. Appended below, nothing above this
# point is modified — the explicit proof that Phase 5's harness/metrics are
# reused as-is, not rebuilt, per §7's stated requirement.
# --------------------------------------------------------------------------

RRF_K = 60  # matches retrieval/fusion.py's constant — duplicated here rather
# than importing a private constant, to avoid any dependency on fusion.py's
# internals beyond its public `fuse()` function.


def strategy_c_loupe_learned_ranker(
    snapshot: RepoSnapshot, task_description: str, ranker: "object", token_budget: int = DEFAULT_BUDGET
) -> StrategyResult:
    """Same as `strategy_c_loupe_end_to_end`, except candidates are ranked by the
    trained ranker's P(edited | features) instead of RRF's fused score — falling
    back to identical RRF-ranked behavior whenever `ranker.is_trained` is False
    (cold-start), so this strategy is safe to run before 200 labeled examples exist.

    Features are the same per-signal RRF contributions (`1/(k+rank)`, 0 if a
    signal didn't surface the candidate) that `retrieval/fusion.py`'s `fuse()`
    sums together — the ranker is learning a replacement *combination rule*
    for those same three numbers (§4), not a differently-scaled input.
    """
    lexical_results = snapshot.lexical_index.query(task_description, top_k=CANDIDATE_POOL_SIZE)
    semantic_results = snapshot.semantic_index.query(task_description, top_k=CANDIDATE_POOL_SIZE)
    fused = fuse(
        lexical_results, semantic_results, snapshot.graph.pagerank_scores, graph=snapshot.graph.graph, top_k=FINAL_TOP_K
    )

    if not ranker.is_trained:
        ranked = fused
    else:
        lexical_rank = {sid: i + 1 for i, (sid, _) in enumerate(lexical_results)}
        semantic_rank = {sid: i + 1 for i, (sid, _) in enumerate(semantic_results)}
        candidate_ids = {sid for sid, _ in fused}
        # Mirrors fuse()'s own centrality term exactly (personalized, seeded from this
        # same candidate pool) — the ranker's features must match what fuse() actually
        # ranked by, not a stale static score fuse() itself no longer defaults to.
        personalized_pagerank = compute_personalized_pagerank(
            snapshot.graph.graph, candidate_ids, snapshot.graph.pagerank_scores
        )
        centrality_sorted = sorted(candidate_ids, key=lambda sid: (-personalized_pagerank.get(sid, 0.0), sid))
        centrality_rank = {sid: i + 1 for i, sid in enumerate(centrality_sorted)}

        def _signal(rank_map: dict[str, int], sid: str) -> float:
            rank = rank_map.get(sid)
            return 1.0 / (RRF_K + rank) if rank is not None else 0.0

        scored = [
            (
                sid,
                ranker.predict(_signal(lexical_rank, sid), _signal(semantic_rank, sid), _signal(centrality_rank, sid)),
            )
            for sid in candidate_ids
        ]
        ranked = sorted(scored, key=lambda pair: (-pair[1], pair[0]))[:FINAL_TOP_K]

    symbols_by_id = {s.id: s for s in snapshot.symbols}
    candidates: list[KnapsackCandidate] = []
    for symbol_id, score in ranked:
        symbol = symbols_by_id.get(symbol_id)
        if symbol is None:
            continue
        cost = symbol_extraction_cost(symbol, snapshot.parsed_files[symbol.file_path].source_bytes)
        candidates.append(KnapsackCandidate(symbol_id, score, cost))

    selected_ids = set(knapsack_greedy(candidates, token_budget))

    retrieved_ids, retrieved_content = [], []
    for symbol_id, _score in ranked:  # preserve rank order — see strategy_c_loupe_end_to_end's own note on this
        if symbol_id not in selected_ids:
            continue
        symbol = symbols_by_id[symbol_id]
        pf = snapshot.parsed_files[symbol.file_path]
        retrieved_content.append(pf.source_bytes[symbol.byte_start : symbol.byte_end].decode("utf-8"))
        retrieved_ids.append(symbol_id)

    return StrategyResult(retrieved_symbol_ids=retrieved_ids, retrieved_content=retrieved_content)


def run_learned_ranker_comparison(
    repo: git.Repo, tasks: list[BenchmarkTask], ranker: "object", embedding_model: object | None = None
) -> dict:
    """Compares 'loupe_rrf' (the existing, untouched `strategy_c_loupe_end_to_end`)
    against 'loupe_learned_ranker' (the new strategy above), same end-to-end
    condition, same aggregation (`_aggregate`, defined above, reused not rebuilt).
    """
    results: dict[str, dict[str, list]] = {
        "loupe_rrf": {"recall_5": [], "recall_10": [], "tokens": []},
        "loupe_learned_ranker": {"recall_5": [], "recall_10": [], "tokens": []},
    }

    for task in tasks:
        snapshot = build_repo_snapshot(repo, task.commit_sha + "^", embedding_model=embedding_model)
        ground_truth = set(task.ground_truth_symbol_ids)

        rrf_result = strategy_c_loupe_end_to_end(snapshot, task.task_description)
        learned_result = strategy_c_loupe_learned_ranker(snapshot, task.task_description, ranker)

        for name, result in (("loupe_rrf", rrf_result), ("loupe_learned_ranker", learned_result)):
            results[name]["recall_5"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=5))
            results[name]["recall_10"].append(recall_at_k(result.retrieved_symbol_ids, ground_truth, k=10))
            results[name]["tokens"].append(token_cost(result.retrieved_content))

    return {
        name: {
            "recall_5": _aggregate(data["recall_5"]),
            "recall_10": _aggregate(data["recall_10"]),
            "tokens": _aggregate(data["tokens"]),
        }
        for name, data in results.items()
    }
