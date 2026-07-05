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
