"""The shared convention-category taxonomy (docs/loupe-scaffold.md §1).

Not a data source — a fixed vocabulary of "things worth having a convention
about at all" in a FastAPI project. Two tools organize around this same
list, filling it in from opposite directions: E4
(`loupe_core/conventions/mining.py`) is *descriptive* — it measures
whichever pattern a real repo actually follows, majority-vote style, even
if unconventional. Scaffold is *prescriptive* — there's no existing repo to
measure, so its bricks ship an opinionated default per category instead.
What's shared is the taxonomy, never the values; keeping it in one place is
what stops the two tools' category lists from silently drifting apart.

E4 currently mines exactly three of these categories (deliberately narrow
scope per `mining.py`'s own docstring) — `ERROR_HANDLING`, `DOCSTRING_STYLE`,
`IMPORT_STYLE`. `CONFIG_MANAGEMENT` and `DEPENDENCY_INJECTION` exist here
for Scaffold's use (every brick that touches settings or DI wiring is
categorized under one of these) even though nothing mines them from a real
repo yet.
"""

from __future__ import annotations

from enum import Enum


class ConventionCategory(str, Enum):
    ERROR_HANDLING = "error_handling"
    DOCSTRING_STYLE = "docstring_style"
    IMPORT_STYLE = "import_style"
    CONFIG_MANAGEMENT = "config_management"
    DEPENDENCY_INJECTION = "dependency_injection"
