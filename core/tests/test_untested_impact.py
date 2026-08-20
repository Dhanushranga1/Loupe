"""Tests for graph/untested_impact.py -- composes E1 (blast-radius/impact) with E2 (test linkage)."""

from __future__ import annotations

import pytest

from loupe_core.graph.builder import build_graph, parse_file
from loupe_core.graph.test_linkage import link_tests
from loupe_core.graph.untested_impact import find_untested_high_impact_symbols

_UTILS_PY = (
    'def format_currency(amount: float) -> str:\n'
    '    """Format a numeric amount as a display-ready currency string."""\n'
    '    return f"${amount:.2f}"\n\n\n'
    'def validate_email(email: str) -> bool:\n'
    '    """Return True if the given string looks like a valid email address."""\n'
    '    return "@" in email and "." in email\n\n\n'
    'def unused_utility() -> None:\n'
    '    """Nothing in this fixture calls this -- the deliberate zero-caller leaf case."""\n'
    '    return None\n'
)

_MODELS_PY = (
    "from utils import format_currency\n\n\n"
    "class Order:\n"
    '    """A single customer order."""\n\n'
    "    def __init__(self, email: str, amount: float) -> None:\n"
    "        self.email = email\n"
    "        self.amount = amount\n\n"
    "    def total(self) -> str:\n"
    '        """format_currency\'s one direct caller inside this fixture\'s model layer."""\n'
    "        return format_currency(self.amount)\n\n"
    "    def describe(self) -> str:\n"
    '        """Calls self.total(), making it a transitive caller of format_currency."""\n'
    '        return f"Order for {self.email}: {self.total()}"\n'
)

# validate_email gets *two* independent direct callers (check_signup_email,
# OrderService.create_order) so its impact_size (2) is distinct from, and
# higher than, Order.total's (1) -- the fixture needs two different nonzero
# impact sizes among untested symbols to prove descending sort actually sorts.
_SERVICES_PY = (
    "from models import Order\n"
    "from utils import format_currency, validate_email\n\n\n"
    "def format_receipt_amount(amount: float) -> str:\n"
    '    """A second, independent direct caller of format_currency."""\n'
    "    return format_currency(amount)\n\n\n"
    "def check_signup_email(email: str) -> bool:\n"
    "    return validate_email(email)\n\n\n"
    "class OrderService:\n"
    "    def create_order(self, email: str, amount: float) -> Order:\n"
    "        validate_email(email)\n"
    "        return Order(email, amount)\n"
)

# The only test file in the fixture -- links format_currency (CONFIRMED: naming
# match on `test_format_currency` plus a real CALLS edge) and nothing else, so
# validate_email and Order.total both stay untested despite format_currency
# having the largest blast radius of the three.
_TEST_UTILS_PY = (
    "from utils import format_currency\n\n\n"
    "def test_format_currency():\n"
    '    assert format_currency(1.5) == "$1.50"\n'
)

FILES = ["utils.py", "models.py", "services.py", "test_utils.py"]


@pytest.fixture
def repo(tmp_path, monkeypatch):
    (tmp_path / "utils.py").write_text(_UTILS_PY)
    (tmp_path / "models.py").write_text(_MODELS_PY)
    (tmp_path / "services.py").write_text(_SERVICES_PY)
    (tmp_path / "test_utils.py").write_text(_TEST_UTILS_PY)
    monkeypatch.chdir(tmp_path)

    parsed = [parse_file(f) for f in FILES]
    g = build_graph(parsed)
    symbols_by_id = {s.id: s for pf in parsed for s in pf.symbols}
    link_tests(g.graph, symbols_by_id)
    return g, symbols_by_id


def _id_by_name(symbols_by_id, qualified_name: str) -> str:
    return next(s.id for s in symbols_by_id.values() if s.qualified_name == qualified_name)


def test_excludes_tested_symbol_even_though_it_has_the_largest_blast_radius(repo):
    g, symbols_by_id = repo
    format_currency_id = _id_by_name(symbols_by_id, "format_currency")

    results = find_untested_high_impact_symbols(g.graph, symbols_by_id, g.pagerank_scores)

    assert format_currency_id not in {sid for sid, _ in results}


def test_returns_untested_symbols_sorted_by_impact_size_descending(repo):
    """validate_email (2 independent direct callers) must outrank Order.total (1) in the
    output. Checked by relative position/value rather than exact list equality: the fixture's
    `OrderService.create_order` also resolves a constructor call to the `Order` class itself
    (untested, impact_size 1) -- a real, correct side effect of reusing E1's call resolution
    as-is, not something this test needs to special-case away."""
    g, symbols_by_id = repo
    validate_email_id = _id_by_name(symbols_by_id, "validate_email")
    order_total_id = _id_by_name(symbols_by_id, "Order.total")

    results = find_untested_high_impact_symbols(g.graph, symbols_by_id, g.pagerank_scores)
    by_id = dict(results)

    assert by_id[validate_email_id] == 2
    assert by_id[order_total_id] == 1
    # descending order actually holds across the whole result, not just these two entries
    impact_sizes = [impact for _sid, impact in results]
    assert impact_sizes == sorted(impact_sizes, reverse=True)
    assert results.index((validate_email_id, 2)) < results.index((order_total_id, 1))


def test_min_impact_filters_out_lower_impact_untested_symbols(repo):
    g, symbols_by_id = repo
    validate_email_id = _id_by_name(symbols_by_id, "validate_email")

    results = find_untested_high_impact_symbols(g.graph, symbols_by_id, g.pagerank_scores, min_impact=2)

    assert results == [(validate_email_id, 2)]


def test_zero_impact_untested_leaf_is_excluded_by_default_min_impact(repo):
    g, symbols_by_id = repo
    unused_utility_id = _id_by_name(symbols_by_id, "unused_utility")

    results = find_untested_high_impact_symbols(g.graph, symbols_by_id, g.pagerank_scores)

    assert unused_utility_id not in {sid for sid, _ in results}
