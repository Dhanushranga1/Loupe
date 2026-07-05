from circular_b import helper_b


def helper_a(n: int) -> int:
    """Recurse into circular_b to demonstrate cycle-safe traversal."""
    if n <= 0:
        return 0
    return helper_b(n - 1)
