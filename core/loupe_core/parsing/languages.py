"""Tree-sitter grammar registry — one entry per supported language.

Phase 0 supports Python only (see docs/phase-0-foundations.md §1). Additional
languages register here later; nothing else in `parsing/` should import a
grammar package directly.
"""

from __future__ import annotations

import tree_sitter as ts
import tree_sitter_python as tspython

_LANGUAGES: dict[str, ts.Language] = {
    "python": ts.Language(tspython.language()),
}


def get_language(name: str) -> ts.Language:
    """Look up a registered tree-sitter Language by name (e.g. "python")."""
    try:
        return _LANGUAGES[name]
    except KeyError:
        raise ValueError(
            f"Unsupported language: {name!r}. Registered: {sorted(_LANGUAGES)}"
        ) from None


def get_parser(name: str) -> ts.Parser:
    """Construct a fresh Parser bound to the named language's grammar."""
    return ts.Parser(get_language(name))
