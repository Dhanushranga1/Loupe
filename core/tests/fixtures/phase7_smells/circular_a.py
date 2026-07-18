from circular_b import helper_b


def helper_a(n: int) -> int:
    """Recurses into circular_b — the two together form a real call cycle."""
    if n <= 0:
        return 0
    return helper_b(n - 1)
