from fastapi import APIRouter

from models import ItemIn, ItemOut

router = APIRouter(prefix="/items")


@router.post("/smelly")
async def create_item_smelly(payload: dict) -> ItemOut:
    """A bare dict parameter instead of a Pydantic schema — the smell."""
    return ItemOut()


@router.post("/clean")
async def create_item_clean(payload: ItemIn) -> ItemOut:
    """A real Pydantic-shaped schema parameter — clean."""
    return ItemOut()
