from circular_a import helper_a


def helper_b(n: int) -> int:
    """Recurses into circular_a — the two together form a real call cycle."""
    if n <= 0:
        return 0
    return helper_a(n - 1)
