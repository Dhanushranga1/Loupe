def format_currency(amount: float) -> str:
    """Format a numeric amount as a display-ready currency string."""
    return f"${amount:.2f}"


def validate_email(email: str) -> bool:
    """Return True if the given string looks like a valid email address."""
    return "@" in email and "." in email


def unused_utility() -> None:
    """Nothing in this fixture calls this — the deliberate zero-caller leaf case."""
    return None
