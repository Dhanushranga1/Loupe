"""Learned re-ranker (docs/phase-6-closing-the-loop.md §4).

Learns a *replacement combination rule* for RRF's same three signals — not a
fundamentally different retrieval mechanism, so this model competes on equal
footing with the static baseline it's meant to improve on. Refuses to
operate below a cold-start threshold rather than producing confidently-wrong
predictions from too little data (a well-known ML pitfall this phase
explicitly designs against, not an accident to catch later).
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from sklearn.linear_model import LogisticRegression

COLD_START_THRESHOLD = 200
FEATURE_NAMES = ["lexical_score", "semantic_score", "centrality_score"]


@dataclass
class TrainingExample:
    lexical_score: float
    semantic_score: float
    centrality_score: float
    symbol_edited: bool


class Ranker:
    """Wraps a `LogisticRegression` model with an explicit cold-start guard.

    RRF is never deleted — it's the fallback both for cold-start and as the
    baseline Phase 5's harness compares the trained model against.
    """

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self._coefficients: dict[str, float] | None = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(self, examples: list[TrainingExample]) -> None:
        """Fit on `examples`, or explicitly refuse below `COLD_START_THRESHOLD`.

        Refusing is not a crash and not a silently-meaningless fit — `is_trained`
        stays False and `predict` keeps returning None, so callers have a clear,
        checkable "not yet trained" state to fall back to RRF from.
        """
        if len(examples) < COLD_START_THRESHOLD:
            self._model = None
            self._coefficients = None
            return

        features = [[e.lexical_score, e.semantic_score, e.centrality_score] for e in examples]
        labels = [int(e.symbol_edited) for e in examples]

        model = LogisticRegression()
        model.fit(features, labels)

        self._model = model
        self._coefficients = dict(zip(FEATURE_NAMES, (float(c) for c in model.coef_[0])))

    def predict(self, lexical_score: float, semantic_score: float, centrality_score: float) -> float | None:
        """P(edited | features), or None if not yet trained (cold-start — caller must fall back to RRF)."""
        if self._model is None:
            return None
        proba = self._model.predict_proba([[lexical_score, semantic_score, centrality_score]])
        return float(proba[0][1])

    @property
    def coefficients(self) -> dict[str, float] | None:
        """The 3 learned coefficients keyed by feature name, or None if not trained.

        Interpretability as a feature, not a side effect (§4): which signal
        ends up weighted most heavily is itself informative about this
        specific repository's retrieval behavior.
        """
        return self._coefficients

    def save(self, path: str) -> None:
        """Persist this ranker's state (per-repo, local — never shared across projects, §1)."""
        Path(path).write_bytes(pickle.dumps({"model": self._model, "coefficients": self._coefficients}))

    @classmethod
    def load(cls, path: str) -> Ranker:
        """Load a previously-saved ranker; returns an untrained (cold-start) instance if the file doesn't exist."""
        ranker = cls()
        file_path = Path(path)
        if not file_path.exists():
            return ranker
        state = pickle.loads(file_path.read_bytes())
        ranker._model = state["model"]
        ranker._coefficients = state["coefficients"]
        return ranker
