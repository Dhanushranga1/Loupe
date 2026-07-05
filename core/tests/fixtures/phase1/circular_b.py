from circular_a import helper_a


def helper_b(n: int) -> int:
    """Recurse into circular_a to demonstrate cycle-safe traversal."""
    if n <= 0:
        return 0
    return helper_a(n - 1)
