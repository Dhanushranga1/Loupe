"""`loupe new` — the one-time, interactive CLI entrypoint
(docs/loupe-scaffold.md's own framing: never something Claude Code calls
mid-session, so this is deliberately the only place in this package that
touches a real terminal).

Thin by design: all the actual logic (which questions fire, which bricks
activate, how they compose) lives in `elicitation.py`/`compose.py`/
`generate.py` and is already tested against a scripted `answer_fn` — this
module's only job is supplying the *real* one, backed by `input()`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .elicitation import ConditionalQuestion, FixedQuestion, Question
from .generate import run_and_write


def _prompt(question: Question) -> object:
    if isinstance(question, FixedQuestion) and question.options is None:
        return input(f"{question.question_id}: ").strip()

    options = question.options
    multi = isinstance(question, ConditionalQuestion) and question.multi_select
    prompt_suffix = " (comma-separated)" if multi else ""
    raw = input(f"{question.question_id} [{'/'.join(options)}]{prompt_suffix}: ").strip()

    if multi:
        chosen = [v.strip() for v in raw.split(",") if v.strip()]
        return [v for v in chosen if v in options] or [options[0]]

    return raw if raw in options else options[0]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="loupe-new")
    parser.add_argument("output_dir", help="directory to generate the new project into (created if missing)")
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists() and any(output_dir.iterdir()):
        print(f"{output_dir} already exists and is not empty — refusing to generate into it.")
        return 1

    answers = run_and_write(output_dir, _prompt)
    print(f"Generated {answers.get('project_name', 'project')} in {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
