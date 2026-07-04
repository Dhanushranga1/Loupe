#!/usr/bin/env python
"""Manual inspection tool: pretty-print every symbol `extract_symbols` finds in a file.

Usage: python scripts/inspect_symbols.py <path/to/file.py>

Exists for eyeballing extractor behavior on real, messier files beyond the
clean-by-construction fixtures (see docs/phase-0-foundations.md §7).
"""

from __future__ import annotations

import sys

from loupe_core.parsing.extractor import extract_symbols


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(f"usage: {argv[0]} <path/to/file.py>", file=sys.stderr)
        return 2

    path = argv[1]
    symbols = extract_symbols(path)
    if not symbols:
        print(f"No symbols found in {path}")
        return 0

    for symbol in symbols:
        docstring_first_line = (symbol.docstring or "").splitlines()[:1]
        print(f"{symbol.kind.value:15s} {symbol.qualified_name}")
        print(f"  id            {symbol.id}")
        print(f"  parent_id     {symbol.parent_id}")
        print(f"  bytes         [{symbol.byte_start}, {symbol.byte_end})")
        print(f"  lines         [{symbol.line_start}, {symbol.line_end}]")
        print(f"  decorators    {symbol.decorators}")
        print(f"  signature     {symbol.signature}")
        print(f"  docstring     {docstring_first_line[0] if docstring_first_line else None!r}")
        print(f"  content_hash  {symbol.content_hash[:12]}...")
        print()

    print(f"{len(symbols)} symbol(s) found in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
