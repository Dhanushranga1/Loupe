"""Token cost estimation (docs/phase-3-resource-allocation.md §4).

Uses tiktoken's cl100k_base encoding as a fast, deterministic, local
stand-in for Claude's real tokenizer — a reasonable approximation for a
budgeting decision, not a claim of exact token-count accuracy (§2).
"""

from __future__ import annotations

import math

import tiktoken

from loupe_core.parsing.schema import Symbol

SAFETY_MULTIPLIER = 1.1

_encoding = tiktoken.get_encoding("cl100k_base")


def estimate_tokens(text: str) -> int:
    """cl100k_base token count, ×1.1 safety margin, rounded up.

    Deterministic and side-effect-free: the safety margin biases every
    estimate conservative, so the governor never thinks it has more room
    than it actually does.
    """
    raw_count = len(_encoding.encode(text))
    return math.ceil(raw_count * SAFETY_MULTIPLIER)


def _discovery_text(symbol: Symbol) -> str:
    """`signature + "\\n" + first docstring line`, or just `signature` if none (§3)."""
    if not symbol.docstring:
        return symbol.signature
    lines = symbol.docstring.splitlines()
    first_line = lines[0] if lines else ""
    return f"{symbol.signature}\n{first_line}"


def symbol_discovery_cost(symbol: Symbol) -> int:
    """Cost of showing just the signature + first docstring line (pass-1 discovery)."""
    return estimate_tokens(_discovery_text(symbol))


def symbol_extraction_cost(symbol: Symbol, source_bytes: bytes) -> int:
    """Cost of the full source slice `[byte_start, byte_end)` (pass-2 extraction)."""
    text = source_bytes[symbol.byte_start : symbol.byte_end].decode("utf-8")
    return estimate_tokens(text)
