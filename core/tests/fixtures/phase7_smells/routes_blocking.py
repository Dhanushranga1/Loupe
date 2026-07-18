import time

from fastapi import APIRouter

router = APIRouter(prefix="/slow")


@router.get("/smelly")
async def slow_handler_smelly():
    """A blocking call inside an async def handler — the smell. Blocks the whole
    event loop, not just this one request."""
    time.sleep(1)
    return {"ok": True}


@router.get("/clean")
def slow_handler_clean_sync():
    """The exact same blocking call, but from a synchronous handler — not a smell.
    Sync handlers are allowed to block; FastAPI runs them in a threadpool."""
    time.sleep(1)
    return {"ok": True}
