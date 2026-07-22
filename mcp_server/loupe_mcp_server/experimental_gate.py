"""The experimental gate (docs/PhaseX/experimental-gate-and-hyde.md, Part 1).

Generic, reusable infrastructure for any feature that spends real, paid LLM
tokens independent of Claude Code's own usage — HyDE is the first such
feature, not the only one this is built for (§1 names
`legacy_docstring_backfill` as the next obvious candidate).

Two-level control (§1): nothing under `experimental.features` runs unless
`experimental.llm_assist` — the category-level switch — is also true. A
project that never touches this manifest section is provably running zero
experimental spend.

Separate telemetry (§4): every gated-feature invocation logs to
`.loupe/logs/experimental/<feature>.jsonl`, deliberately not mixed into the
regular `RetrievalLog` (`telemetry.py`) — this is a categorically different
kind of spend (a deliberate, optional LLM call) from the normal governed
retrieval flow, and needs to be answerable on its own: "how much has this
optional feature actually cost me."
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from loupe_core.governor.budget import estimate_tokens

from .config import LoupeConfig

# §3's "modeled first, measured once real usage exists" cost estimates: a
# representative prompt/completion for each feature, run through Phase 3's
# own token estimator, not a real logged average (no feature has usage yet).
# Adding a new gated feature means adding one entry here.


@dataclass(frozen=True)
class ModeledCost:
    description: str
    representative_prompt: str
    typical_completion_tokens: int

    @property
    def estimated_tokens(self) -> int:
        return estimate_tokens(self.representative_prompt) + self.typical_completion_tokens


HYDE_PROMPT_TEMPLATE = (
    "Write a short, plausible, hypothetical answer or code snippet that "
    "would appear in this codebase in response to the following query.\n\n"
    "Query: {query}"
)
_HYDE_REPRESENTATIVE_QUERY = "how does the retry logic handle a timed-out request?"

KNOWN_FEATURES: dict[str, ModeledCost] = {
    "hyde_query_rewrite": ModeledCost(
        description="One generative-LLM call per search_symbols query to write a hypothetical "
        "answer, embedded and fused into retrieval as a fourth RRF signal.",
        representative_prompt=HYDE_PROMPT_TEMPLATE.format(query=_HYDE_REPRESENTATIVE_QUERY),
        typical_completion_tokens=200,
    ),
}


class UnknownExperimentalFeature(ValueError):
    pass


def modeled_cost_for(feature: str) -> ModeledCost:
    if feature not in KNOWN_FEATURES:
        raise UnknownExperimentalFeature(
            f"{feature!r} is not a known experimental feature (known: {sorted(KNOWN_FEATURES)})"
        )
    return KNOWN_FEATURES[feature]


def is_experimental_feature_enabled(config: LoupeConfig, feature: str) -> bool:
    """§1's two-level check: the category switch AND the specific feature flag."""
    return config.experimental.llm_assist and config.experimental.features.get(feature, False)


@dataclass
class ExperimentalUsageEntry:
    feature: str
    tokens: int
    cost_estimate_type: str  # "modeled" | "measured"
    detail: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_json_line(self) -> str:
        return json.dumps(
            {
                "timestamp": self.timestamp,
                "feature": self.feature,
                "tokens": self.tokens,
                "cost_estimate_type": self.cost_estimate_type,
                "detail": self.detail,
            }
        )


def log_experimental_usage(
    loupe_dir: Path, feature: str, tokens: int, cost_estimate_type: str = "measured", **detail
) -> None:
    """Append one usage entry to `.loupe/logs/experimental/<feature>.jsonl`."""
    log_dir = loupe_dir / "logs" / "experimental"
    log_dir.mkdir(parents=True, exist_ok=True)
    entry = ExperimentalUsageEntry(feature=feature, tokens=tokens, cost_estimate_type=cost_estimate_type, detail=detail)
    with open(log_dir / f"{feature}.jsonl", "a") as f:
        f.write(entry.to_json_line() + "\n")


def total_logged_tokens(loupe_dir: Path, feature: str) -> int:
    """Sum of every logged `tokens` value for `feature` — the trustworthy,
    after-the-fact ledger §4 asks for, not an approximation."""
    log_path = loupe_dir / "logs" / "experimental" / f"{feature}.jsonl"
    if not log_path.exists():
        return 0
    total = 0
    for line in log_path.read_text().splitlines():
        if line.strip():
            total += json.loads(line)["tokens"]
    return total
