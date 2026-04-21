"""Tests for primekv.tuning."""

from __future__ import annotations

from primekv.cache import PrimeKVCache, DEFAULT_POLICIES
from primekv.classifier import Tier
from primekv.tuning import (
    TuningProfile,
    auto_tune,
    build_cache_from_profile,
    describe_profile,
    estimate_bytes,
)


def test_auto_tune_short_prompt_no_budget_picks_relaxed():
    profile = auto_tune(prompt_length=64)
    assert profile.name == "relaxed"
    # Relaxed keeps SUPPORTING at FP16.
    assert profile.policies[Tier.SUPPORTING].precision == "fp16"


def test_auto_tune_medium_prompt_no_budget_picks_balanced():
    profile = auto_tune(prompt_length=1024)
    assert profile.name == "balanced"
    assert profile.policies[Tier.SUPPORTING].precision == DEFAULT_POLICIES[Tier.SUPPORTING].precision


def test_auto_tune_long_prompt_no_budget_picks_tight():
    profile = auto_tune(prompt_length=4096)
    assert profile.name == "tight"
    assert profile.policies[Tier.SEMANTIC].precision == "int8"
    assert Tier.SUPPORTING in profile.max_entries_per_tier


def test_auto_tune_with_generous_budget_picks_relaxed():
    profile = auto_tune(
        prompt_length=128,
        memory_budget_mb=10_000,
        num_layers=4,
        num_heads=4,
        head_dim=32,
    )
    assert profile.name == "relaxed"
    assert profile.estimated_bytes is not None
    assert profile.estimated_bytes <= 10_000 * 1024 * 1024


def test_auto_tune_with_tight_budget_selects_compressive_profile():
    # Very small budget forces us to a tighter profile.
    profile = auto_tune(
        prompt_length=4096,
        memory_budget_mb=1.0,
        num_layers=12,
        num_heads=12,
        head_dim=64,
    )
    assert profile.name in ("tight", "squeeze")
    # Either it fits, or it's flagged as over-budget.
    assert "budget" in profile.rationale.lower() or "over" in profile.rationale.lower()


def test_estimate_bytes_is_monotonic_in_prompt_length():
    profile = auto_tune(prompt_length=1024)
    a = estimate_bytes(512, profile, num_layers=4, num_heads=4, head_dim=32)
    b = estimate_bytes(4096, profile, num_layers=4, num_heads=4, head_dim=32)
    assert b > a


def test_describe_profile_contains_each_tier():
    profile = auto_tune(prompt_length=128)
    text = describe_profile(profile)
    for tier in Tier:
        assert tier.name in text


def test_build_cache_from_profile_returns_primekv_cache():
    profile = auto_tune(prompt_length=1024)
    cache = build_cache_from_profile(profile, num_layers=2)
    assert isinstance(cache, PrimeKVCache)
    assert cache.num_layers == 2
    # Policies were applied.
    for tier, policy in profile.policies.items():
        assert cache.policies[tier].precision == policy.precision


def test_tuning_profile_holds_shape():
    p = TuningProfile(name="x", policies={})
    assert p.name == "x"
    assert p.anchor_prefix_len == 4
    assert p.semantic_stride == 3
