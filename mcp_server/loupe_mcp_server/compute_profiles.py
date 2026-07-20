"""Compute profiles: letting projects choose their model tier
(docs/PhaseX/compute-profiles.md).

Three named tiers, not a free-form model picker (§1) — each pairs an
embedding model with a cross-encoder of roughly matching quality, so
choosing a tier is one decision, not two independently-reasoned-about ones.
`embedding_model`/`cross_encoder_model` can still be set explicitly in
`loupe.manifest.yaml` to override just one piece of a profile (§4's own
"profile sets defaults, explicit values win" rule).
"""

from __future__ import annotations

from dataclasses import dataclass

AUTO = "auto"


@dataclass(frozen=True)
class ComputeProfile:
    embedding_model: str
    cross_encoder_model: str
    embedding_dim: int


COMPUTE_PROFILES: dict[str, ComputeProfile] = {
    "cpu_small": ComputeProfile(
        embedding_model="BAAI/bge-small-en-v1.5",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_dim=384,
    ),
    "cpu_medium": ComputeProfile(
        embedding_model="BAAI/bge-base-en-v1.5",
        cross_encoder_model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        embedding_dim=768,
    ),
    "gpu_large": ComputeProfile(
        embedding_model="BAAI/bge-large-en-v1.5",
        cross_encoder_model="BAAI/bge-reranker-large",
        embedding_dim=1024,
    ),
}

DEFAULT_COMPUTE_PROFILE = "cpu_small"


def resolve_embedding_model(compute_profile: str, override: str | None) -> str:
    """§4's "profile sets defaults, explicit values win" rule."""
    if override is not None and override != AUTO:
        return override
    return COMPUTE_PROFILES[compute_profile].embedding_model


def resolve_cross_encoder_model(compute_profile: str, override: str | None) -> str:
    if override is not None and override != AUTO:
        return override
    return COMPUTE_PROFILES[compute_profile].cross_encoder_model


def resolve_embedding_dim(compute_profile: str) -> int:
    """Not independently overridable — a profile's embedding dimension is a
    direct, fixed consequence of its embedding model, not a separate knob;
    an explicit `embedding_model` override changes what model loads, but the
    *profile*, not the override, still determines the vector store's
    expected dimension here. A mismatched explicit override would surface as
    a real dimension error from `sentence-transformers` itself at encode
    time — an honest limitation, not silently handled.
    """
    return COMPUTE_PROFILES[compute_profile].embedding_dim


def detect_gpu() -> bool:
    """A simple, standard check (§6 task 2) — `torch` is already a
    transitive dependency of `sentence-transformers`, so this needs no new
    dependency of its own."""
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False
