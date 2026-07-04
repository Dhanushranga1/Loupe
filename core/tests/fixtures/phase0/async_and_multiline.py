async def fetch_batch(
    endpoint: str,
    ids: list[int],
    timeout: float = 30.0,
    retries: int = 3,
) -> list[dict]:
    """Fetch a batch of records from an endpoint, with retry support."""
    return []
