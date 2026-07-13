import json

from models import Order
from utils import format_currency, validate_email


def format_receipt_amount(amount: float) -> str:
    """A second, independent direct caller of format_currency — makes format_currency's
    centrality unambiguously higher than its neighbors for the hub-threshold test."""
    return format_currency(amount)


class OrderService:
    """Creates and logs customer orders."""

    def log(self, message: str) -> None:
        """Record a message somewhere (stubbed for the fixture)."""
        print(message)

    def create_order(self, email: str, amount: float) -> str:
        """Validate, create, and log a new order; return it as a JSON string."""
        validate_email(email)
        self.log(f"creating order for {email}")
        order = Order(email, amount)
        return json.dumps({"email": email, "amount": amount})
