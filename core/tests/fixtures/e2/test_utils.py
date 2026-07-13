from utils import format_currency, validate_email


def test_format_currency():
    """Naming ('test_format_currency' -> 'format_currency') AND the call heuristic
    (this body actually calls format_currency) agree -> confirmed."""
    assert format_currency(19.999) == "$20.00"


def test_validate_email():
    """Naming matches ('test_validate_email' -> 'validate_email'), but the real
    validate_email is mocked out below rather than called -> naming_only."""

    def fake_validate(email: str) -> bool:
        return True

    assert fake_validate("someone@example.com") is True


def check_currency_formatting():
    """Doesn't follow the test_<name> naming convention, but does call
    format_currency directly from within this test file -> call_only."""
    return format_currency(5.0) == "$5.00"
