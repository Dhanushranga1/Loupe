class ItemIn:
    """Request schema for creating an item."""

    name: str
    price: float


class ItemOut:
    """Response schema for a single item."""

    id: int
    name: str


class OrderIn:
    """Request schema for creating an order."""

    items: list
