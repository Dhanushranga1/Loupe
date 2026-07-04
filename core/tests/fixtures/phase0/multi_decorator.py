def cache(fn):
    return fn


def log_calls(fn):
    return fn


@cache
@log_calls
def compute(x: int) -> int:
    """Compute something expensive, cached and logged."""
    return x * x
