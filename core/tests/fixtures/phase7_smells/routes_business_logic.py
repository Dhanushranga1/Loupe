from fastapi import APIRouter

from models import OrderIn
from validators import validate_a, validate_b
from notifications import log_event, notify, persist
from service_layer import create_order

router = APIRouter(prefix="/orders")


@router.post("/smelly")
async def create_order_smelly(payload: dict):
    """Complex control flow (multiple branches, a loop, a try/except) combined with
    several distinct direct calls — the actual business logic inline in the route
    handler instead of delegated to a service layer. The smell."""
    if payload.get("a"):
        for x in range(3):
            if x > 1:
                validate_a(x)
            else:
                validate_b(x)
    elif payload.get("b"):
        try:
            log_event(payload)
        except ValueError:
            notify(payload)
    else:
        log_event(payload)
        notify(payload)
        persist(payload)
    return {"ok": True}


@router.post("/clean")
async def create_order_clean(payload: OrderIn):
    """One call, no branching — delegates to the service layer. Clean."""
    return create_order(payload)
