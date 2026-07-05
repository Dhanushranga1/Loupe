"""Shared pytest fixtures for the Phase 1 mock-project graph."""

from pathlib import Path

import pytest

from loupe_core.graph.builder import build_graph, parse_file

PHASE1_FIXTURES = Path(__file__).parent / "fixtures" / "phase1"
PHASE1_FILES = ["utils.py", "models.py", "services.py", "handlers.py", "circular_a.py", "circular_b.py"]


@pytest.fixture(scope="module")
def phase1_parsed():
    return [parse_file(str(PHASE1_FIXTURES / f)) for f in PHASE1_FILES]


@pytest.fixture(scope="module")
def loupe_graph(phase1_parsed):
    return build_graph(phase1_parsed)


@pytest.fixture(scope="module")
def symbol_by_qn(phase1_parsed):
    lookup = {}
    for pf in phase1_parsed:
        for s in pf.symbols:
            lookup[s.qualified_name] = s
    return lookup
