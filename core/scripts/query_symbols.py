#!/usr/bin/env python
"""Manual inspection tool: run a query and print the per-signal ranking breakdown.

Usage: python scripts/query_symbols.py <repo_dir> "<query>"

This per-signal breakdown (lexical rank, semantic rank, centrality/pagerank,
fused score) is what makes debugging a bad ranking possible later — see
docs/phase-2-retrieval.md §8. Indexes every *.py file under <repo_dir>.
"""

from __future__ import annotations

import sys
from pathlib import Path

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.graph.centrality import compute_personalized_pagerank
from loupe_core.retrieval.fusion import CANDIDATE_POOL_SIZE, FINAL_TOP_K, fuse
from loupe_core.retrieval.lexical import LexicalIndex
from loupe_core.retrieval.semantic import SemanticIndex


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f'usage: {argv[0]} <repo_dir> "<query>"', file=sys.stderr)
        return 2

    repo_dir, query = Path(argv[1]), argv[2]
    files = sorted(repo_dir.rglob("*.py"))
    if not files:
        print(f"No .py files found under {repo_dir}", file=sys.stderr)
        return 1

    print(f"Indexing {len(files)} file(s) under {repo_dir}...", file=sys.stderr)
    parsed = [parse_file(str(f)) for f in files]
    symbols = [s for pf in parsed for s in pf.symbols]
    id_to_symbol = {s.id: s for s in symbols}

    loupe_graph = build_graph(parsed)
    lexical_index = LexicalIndex(symbols)
    semantic_index = SemanticIndex()
    semantic_index.index(symbols)

    lexical_results = lexical_index.query(query, top_k=CANDIDATE_POOL_SIZE)
    semantic_results = semantic_index.query(query, top_k=CANDIDATE_POOL_SIZE)
    fused = fuse(lexical_results, semantic_results, loupe_graph.pagerank_scores, graph=loupe_graph.graph, top_k=FINAL_TOP_K)

    lexical_rank = {sid: i + 1 for i, (sid, _) in enumerate(lexical_results)}
    semantic_rank = {sid: i + 1 for i, (sid, _) in enumerate(semantic_results)}
    candidate_ids = set(lexical_rank) | set(semantic_rank)
    # Personalized, not static — matches what `fuse(..., graph=...)` actually ranked by.
    personalized_pagerank = compute_personalized_pagerank(loupe_graph.graph, candidate_ids, loupe_graph.pagerank_scores)

    print(f"\nQuery: {query!r}\n")
    header = f"{'#':<3} {'symbol':<50} {'fused':>8} {'lex_rank':>9} {'sem_rank':>9} {'p_pagerank':>10}"
    print(header)
    print("-" * len(header))
    for rank, (symbol_id, fused_score) in enumerate(fused, 1):
        symbol = id_to_symbol[symbol_id]
        label = f"{symbol.file_path}::{symbol.qualified_name}"
        lr = lexical_rank.get(symbol_id, "-")
        sr = semantic_rank.get(symbol_id, "-")
        pr = personalized_pagerank.get(symbol_id, 0.0)
        print(f"{rank:<3} {label:<50} {fused_score:>8.4f} {str(lr):>9} {str(sr):>9} {pr:>10.4f}")

    if not fused:
        print("(no candidates found by either signal)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
