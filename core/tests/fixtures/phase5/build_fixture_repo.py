"""Programmatically builds a real git repository for Phase 5 mining tests.

See docs/phase-5-evaluation.md §7 — unlike every prior phase's fixtures
(plain files), this phase needs an actual git repo with real commit history.
Built fresh into a given directory on every call (tests pass a `tmp_path`).
"""

from __future__ import annotations

from pathlib import Path

import git

UTILS_V0 = '''def format_currency(amount: float) -> str:
    """Format a numeric amount as a display-ready currency string."""
    return f"${amount:.2f}"


def validate_email(email: str) -> bool:
    """Return True if the given string looks like a valid email address."""
    return "@" in email and "." in email
'''

UTILS_V1_THOUSANDS_SEPARATOR = '''def format_currency(amount: float) -> str:
    """Format a numeric amount as a display-ready currency string."""
    return f"${amount:,.2f}"


def validate_email(email: str) -> bool:
    """Return True if the given string looks like a valid email address."""
    return "@" in email and "." in email
'''

UTILS_V2_EMAIL_LENGTH_CHECK = '''def format_currency(amount: float) -> str:
    """Format a numeric amount as a display-ready currency string."""
    return f"${amount:,.2f}"


def validate_email(email: str) -> bool:
    """Return True if the given string looks like a valid email address."""
    return "@" in email and "." in email and len(email) <= 254
'''

UTILS_V3_NEW_FUNCTION = '''def format_currency(amount: float) -> str:
    """Format a numeric amount as a display-ready currency string."""
    return f"${amount:,.2f}"


def validate_email(email: str) -> bool:
    """Return True if the given string looks like a valid email address."""
    return "@" in email and "." in email and len(email) <= 254


def apply_discount(amount: float, percent: float) -> float:
    """Apply a percentage discount to an amount and return the new total."""
    return amount * (1 - percent / 100)
'''

MODELS_V0 = '''from utils import format_currency


class Base:
    """Common base for domain model classes."""


class Order(Base):
    """A single customer order."""

    def __init__(self, email: str, amount: float) -> None:
        """Create an order for the given customer email and amount."""
        self.email = email
        self.amount = amount

    def total(self) -> str:
        """Return this order's amount as a formatted currency string."""
        return format_currency(self.amount)
'''

SERVICES_V0 = '''import json

from models import Order
from utils import validate_email


class OrderService:
    """Creates and logs customer orders."""

    def log(self, message: str) -> None:
        """Record a message somewhere (stubbed for the fixture)."""
        print(message)

    def create_order(self, email: str, amount: float) -> str:
        """Validate, create, and log a new order; return it as a JSON string."""
        validate_email(email)
        self.log(f"creating order for {email}")
        order = Order(email, amount)
        return json.dumps({"email": email, "amount": amount})
'''

SERVICES_V1_LOG_EMAIL = '''import json

from models import Order
from utils import validate_email


class OrderService:
    """Creates and logs customer orders."""

    def log(self, message: str) -> None:
        """Record a message somewhere (stubbed for the fixture)."""
        print(message)

    def create_order(self, email: str, amount: float) -> str:
        """Validate, create, and log a new order; return it as a JSON string."""
        validate_email(email)
        self.log(f"creating order for customer {email} - amount {amount}")
        order = Order(email, amount)
        return json.dumps({"email": email, "amount": amount})
'''

# Whitespace-only variant of services.py: trailing spaces added to two lines,
# no tokens changed — a genuine "git diff --ignore-all-space is empty" case.
SERVICES_V2_WHITESPACE_ONLY = SERVICES_V1_LOG_EMAIL.replace(
    'print(message)', 'print(message)   '
).replace(
    'order = Order(email, amount)', 'order = Order(email, amount)   '
)

HANDLERS_V0 = '''class OrderHandler:
    """Handles order-related requests."""

    def process(self) -> str:
        """Process an order-related request."""
        return "order processed"


class UserHandler:
    """Handles user-related requests."""

    def process(self) -> str:
        """Process a user-related request."""
        return "user processed"


def dispatch(x) -> str:
    """Dispatch via a bare, ambiguous `process` — must not resolve to either handler's method."""
    return process(x)
'''

HANDLERS_V1_TRIVIAL_RENAME = HANDLERS_V0.replace(
    'def dispatch(x) -> str:', 'def dispatch(request) -> str:'
).replace('return process(x)', 'return process(request)')


def _write(repo_root: Path, rel_path: str, content: str) -> None:
    path = repo_root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)


def build_fixture_repo(repo_root: Path) -> dict[str, str]:
    """Build the repo; return {label: commit_sha} for every commit created."""
    repo = git.Repo.init(repo_root, initial_branch="main")
    author = git.Actor("Fixture Author", "fixture@example.com")
    shas: dict[str, str] = {}

    # Commits are otherwise created back-to-back in the same process, which
    # can land them in the same literal second — deliberately spacing them
    # a day apart (rather than relying on wall-clock speed) is what makes
    # commit-date-based logic (Phase 6's temporal_split) testable at all.
    base_date = 1_700_000_000  # arbitrary, fixed reference point
    day_seconds = 24 * 60 * 60
    commit_counter = {"n": 0}

    def commit(message: str, add_paths: list[str] | None = None) -> str:
        if add_paths:
            repo.index.add(add_paths)
        else:
            repo.git.add(A=True)
        commit_date = base_date + commit_counter["n"] * day_seconds
        commit_counter["n"] += 1
        date_str = f"{commit_date} +0000"
        c = repo.index.commit(message, author=author, committer=author, author_date=date_str, commit_date=date_str)
        return c.hexsha

    # 1. Root commit — establishes the baseline. No parent, so mine_history
    #    must skip it (nothing to diff against) rather than crash on it.
    _write(repo_root, "utils.py", UTILS_V0)
    _write(repo_root, "models.py", MODELS_V0)
    _write(repo_root, "services.py", SERVICES_V0)
    _write(repo_root, "handlers.py", HANDLERS_V0)
    shas["root"] = commit("Initial project structure")

    # 2. Good commit — single-symbol body edit, single file.
    _write(repo_root, "utils.py", UTILS_V1_THOUSANDS_SEPARATOR)
    shas["good_thousands_separator"] = commit("Add thousands separator to currency formatting")

    # 3. Good commit — another single-symbol body edit, single file.
    _write(repo_root, "utils.py", UTILS_V2_EMAIL_LENGTH_CHECK)
    shas["good_email_length_check"] = commit("Add stricter length limit to email validation")

    # 4. Good commit, and the "new function" case — apply_discount has no
    #    pre-fix version; new_symbols_added must be True for this one.
    _write(repo_root, "utils.py", UTILS_V3_NEW_FUNCTION)
    shas["good_new_function"] = commit("Add percentage discount helper function")

    # 5. Good commit — body edit in a different file (services.py).
    _write(repo_root, "services.py", SERVICES_V1_LOG_EMAIL)
    shas["good_log_email"] = commit("Include amount in order creation log message")

    # 6. Bad commit — short subject, but does NOT match the stoplist regex
    #    (tests the length-based exclusion in isolation from the stoplist).
    _write(repo_root, "handlers.py", HANDLERS_V1_TRIVIAL_RENAME)
    shas["bad_short_subject"] = commit("update code")

    # 7. Bad commit — subject matches the stoplist regex exactly.
    _write(repo_root, "models.py", MODELS_V0 + "\n# trivial trailing comment\n")
    shas["bad_stoplist"] = commit("wip")

    # 8. Bad commit — whitespace-only change (raw diff non-empty, --ignore-all-space diff empty).
    _write(repo_root, "services.py", SERVICES_V2_WHITESPACE_ONLY)
    shas["bad_whitespace_only"] = commit("Reformat services.py spacing")

    # 9. Bad commit — touches more than 15 files (mass one-line changes).
    many_file_paths = []
    for i in range(16):
        rel = f"generated/module_{i:02d}.py"
        _write(repo_root, rel, f"def placeholder_{i}():\n    return {i}\n")
        many_file_paths.append(rel)
    shas["bad_too_many_files"] = commit(
        "Regenerate placeholder modules across the codebase", add_paths=many_file_paths
    )

    # 10. Bad commit — a real merge commit (two parents, --no-ff).
    repo.git.checkout("-b", "feature/branch-commit")
    _write(repo_root, "handlers.py", HANDLERS_V1_TRIVIAL_RENAME + "\n# branch-only comment\n")
    shas["branch_commit"] = commit("Add a branch-only comment to handlers")
    repo.git.checkout("main")
    repo.git.merge("feature/branch-commit", "--no-ff", "-m", "Merge branch 'feature/branch-commit'")
    shas["bad_merge"] = repo.head.commit.hexsha

    return shas
