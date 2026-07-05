class OrderHandler:
    """Handles order-related requests."""

    def process(self) -> str:
        """Process an order-related request."""
        return "order processed"


class UserHandler:
    """Handles user-related requests."""

    def process(self) -> str:
        """Process a user-related request."""
        return "user processed"


def dispatch(x) -> str:
    """Dispatch via a bare, ambiguous `process` — must not resolve to either handler's method."""
    return process(x)
