"""Tests for compute_profiles.py (docs/PhaseX/compute-profiles.md)."""

from loupe_mcp_server.compute_profiles import (
    COMPUTE_PROFILES,
    DEFAULT_COMPUTE_PROFILE,
    detect_gpu,
    resolve_cross_encoder_model,
    resolve_embedding_dim,
    resolve_embedding_model,
)


def test_three_named_tiers_exist():
    assert set(COMPUTE_PROFILES) == {"cpu_small", "cpu_medium", "gpu_large"}


def test_default_profile_is_cpu_small():
    assert DEFAULT_COMPUTE_PROFILE == "cpu_small"


def test_each_tier_pairs_an_embedding_model_with_a_cross_encoder_and_a_dimension():
    for profile in COMPUTE_PROFILES.values():
        assert profile.embedding_model
        assert profile.cross_encoder_model
        assert profile.embedding_dim > 0


def test_resolve_embedding_model_uses_profile_default_when_override_is_auto():
    assert resolve_embedding_model("cpu_small", "auto") == COMPUTE_PROFILES["cpu_small"].embedding_model


def test_resolve_embedding_model_uses_profile_default_when_override_is_none():
    assert resolve_embedding_model("gpu_large", None) == COMPUTE_PROFILES["gpu_large"].embedding_model


def test_resolve_embedding_model_explicit_override_wins():
    """§5's own acceptance criterion: setting an explicit embedding_model
    alongside a compute_profile correctly uses the override, not the
    profile's default — "profile sets defaults, explicit values win"."""
    assert resolve_embedding_model("cpu_small", "some-other-model") == "some-other-model"


def test_resolve_cross_encoder_model_explicit_override_wins():
    assert resolve_cross_encoder_model("cpu_small", "some-other-cross-encoder") == "some-other-cross-encoder"


def test_resolve_cross_encoder_model_uses_profile_default_when_auto():
    assert resolve_cross_encoder_model("cpu_medium", "auto") == COMPUTE_PROFILES["cpu_medium"].cross_encoder_model


def test_resolve_embedding_dim_matches_the_profile_table():
    assert resolve_embedding_dim("cpu_small") == 384
    assert resolve_embedding_dim("cpu_medium") == 768
    assert resolve_embedding_dim("gpu_large") == 1024


def test_detect_gpu_returns_a_bool_and_does_not_raise():
    # Real hardware-dependent result — just confirming it runs cleanly and
    # returns the documented type, not asserting a specific machine's GPU state.
    assert isinstance(detect_gpu(), bool)
