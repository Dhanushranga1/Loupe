"""FastAPI route-handler detection — shared by E1's `analyze_impact`
(`affected_route_count`) and Phase 7's `find_code_smells`, both of which need
to answer the same question: "is this symbol an HTTP route handler?"

Extracted here (originally lived only in `graph/impact.py`) once a second
consumer needed the identical logic — the same "don't duplicate, share"
correction Phase 10.5's own doc made about Louvain clustering.

A decorator-shape heuristic, not framework awareness: a decorator matching
`<name>.<http_method>(...)` (e.g. `router.get("/x")`, `app.post("/y")`)
almost certainly is a FastAPI/Flask-style route in practice, but nothing
here confirms the base is really an `APIRouter` instance — that would be
real type inference, deliberately out of scope everywhere else in this
project too.
"""

from __future__ import annotations

import re

from loupe_core.parsing.schema import Symbol

HTTP_METHODS = {"get", "post", "put", "delete", "patch", "options", "head"}
_ROUTE_DECORATOR_PATTERN = re.compile(r"^\w+\.(" + "|".join(HTTP_METHODS) + r")\(")


def looks_like_http_route(symbol: Symbol) -> bool:
    """True if any of `symbol`'s decorators is shaped like `<name>.<http_method>(...)`."""
    return any(_ROUTE_DECORATOR_PATTERN.match(d) for d in symbol.decorators)
