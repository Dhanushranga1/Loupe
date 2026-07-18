from fastapi import APIRouter

from models import ItemOut

router = APIRouter(prefix="/items")


@router.get("/smelly")
async def list_items_smelly():
    """No response_model on the decorator and no return type annotation — the smell."""
    return []


@router.get("/clean", response_model=ItemOut)
async def get_item_clean(item_id: int) -> ItemOut:
    """Both a response_model and a return type annotation present — clean."""
    return ItemOut()
