from utils import format_currency


class Base:
    """Common base for domain model classes."""


class Order(Base):
    """A single customer order."""

    def __init__(self, email: str, amount: float) -> None:
        """Create an order for the given customer email and amount."""
        self.email = email
        self.amount = amount

    def total(self) -> str:
        """Return this order's amount as a formatted currency string."""
        return format_currency(self.amount)

    def describe(self) -> str:
        """Return a short human-readable description including the formatted total.

        Calls `self.total()`, which itself calls `format_currency` — this is
        the real, resolvable 2-hop incoming chain `format_currency` <-
        `Order.total` <- `Order.describe` used by the E1 impact-analysis
        acceptance tests. It deliberately goes through `self.<name>`, the one
        cross-symbol call form Phase 1's resolver actually resolves without
        guessing (docs/phase-1-graph-theory.md's "no type inference" rule) —
        a chain through a *local variable* holding an `Order` instance (as
        `docs/loupe-extensions.md`'s E1 illustrative example describes) would
        hit that same no-type-inference rule and stay unresolved, so it can't
        demonstrate this chain; this method can.
        """
        return f"Order for {self.email}: {self.total()}"
