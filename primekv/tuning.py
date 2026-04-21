"""Dynamic tuning: adapt tier policies to memory pressure and prompt length.

The default :data:`primekv.cache.DEFAULT_POLICIES` are a single fixed
point: ANCHOR pinned FP16, SEMANTIC FP16, SUPPORTING INT4, FILLER
INT4+summary. That's fine for a demo; it's wrong when the prompt is
short (over-compression wastes quality) or when memory is tight
(under-compression OOMs).

This module computes a :class:`TuningProfile` from a target prompt
length and a memory budget, then builds a ``PrimeKVCache`` that
honors it. The policy is intentionally simple — no learned
controller, no closed-loop feedback. Think of it as the first
derivative: given an operating point, pick the most appropriate
discrete policy from a shortlist.

Entry points:

* :func:`auto_tune` — pure function, returns a :class:`TuningProfile`.
* :func:`build_cache_from_profile` — constructs a ``PrimeKVCache``.
* :func:`describe_profile` — human-readable summary for logs.

Memory accounting uses a per-token byte estimate. Given
``num_layers``, ``num_heads``, ``head_dim``, one token's KV in FP16 is
``2 * num_layers * num_heads * head_dim * 2`` bytes. INT8 is half of
FP16; INT4 is a quarter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from primekv.cache import DEFAULT_POLICIES, PrimeKVCache, TierPolicy
from primekv.classifier import BaseClassifier, RuleBasedClassifier, Tier


PRECISION_BYTES_PER_ELEM = {"fp16": 2.0, "int8": 1.0, "int4": 0.5}


@dataclass
class TuningProfile:
    """An operating point for the tiered cache.

    Attributes:
        name: short label for logging ("relaxed", "balanced", "tight").
        policies: per-tier ``TierPolicy`` overrides. Tiers missing
            here fall back to :data:`DEFAULT_POLICIES`.
        max_entries_per_tier: per-tier capacity cap. A tier capped at
            0 is effectively off (all tokens drop through eviction).
        anchor_prefix_len: classifier anchor length. Short prompts
            want a small anchor; long prompts often benefit from more.
        semantic_stride: classifier stride for SEMANTIC positions.
            Smaller stride → more SEMANTIC tokens → more FP16 memory.
        rationale: one-sentence explanation of why this profile was
            chosen. Useful for logs and the Gradio UI.
        estimated_bytes: rough payload size if the profile is applied
            to a prompt of the target length at the target model shape.
    """

    name: str
    policies: dict[Tier, TierPolicy]
    max_entries_per_tier: dict[Tier, int] = field(default_factory=dict)
    anchor_prefix_len: int = 4
    semantic_stride: int = 3
    rationale: str = ""
    estimated_bytes: Optional[int] = None


def _bytes_per_token(num_layers: int, num_heads: int, head_dim: int, precision: str) -> float:
    per_elem = PRECISION_BYTES_PER_ELEM[precision]
    # K and V, both of shape (num_heads, head_dim), per layer.
    return 2 * num_layers * num_heads * head_dim * per_elem


def estimate_bytes(
    prompt_length: int,
    profile: TuningProfile,
    num_layers: int,
    num_heads: int,
    head_dim: int,
    tier_fractions: Optional[dict[Tier, float]] = None,
) -> int:
    """Estimate the cache payload size for this profile on a prompt of
    ``prompt_length`` tokens.

    ``tier_fractions`` is the expected share of tokens per tier. If not
    given, we fall back to a reasonable default:
    anchor 5% / semantic 30% / supporting 45% / filler 20%.
    """
    if tier_fractions is None:
        tier_fractions = {
            Tier.ANCHOR: 0.05,
            Tier.SEMANTIC: 0.30,
            Tier.SUPPORTING: 0.45,
            Tier.FILLER: 0.20,
        }
    total = 0.0
    for tier in Tier:
        pol = profile.policies.get(tier, DEFAULT_POLICIES[tier])
        n_tokens_tier = prompt_length * tier_fractions.get(tier, 0.25)
        cap = profile.max_entries_per_tier.get(tier)
        if cap is not None:
            n_tokens_tier = min(n_tokens_tier, cap)
        # FILLER with summary policy folds to one vector per layer —
        # constant cost regardless of count.
        if pol.summary:
            total += _bytes_per_token(num_layers, num_heads, head_dim, pol.precision)
            continue
        total += n_tokens_tier * _bytes_per_token(num_layers, num_heads, head_dim, pol.precision)
    return int(total)


# ---------------------------------------------------------------------------
# Canonical profiles
# ---------------------------------------------------------------------------


def _policy(precision: str, pinned: bool = False, evictable: bool = True, summary: bool = False) -> TierPolicy:
    return TierPolicy(
        precision=precision,
        location="hbm",
        pinned=pinned,
        evictable=evictable,
        summary=summary,
    )


def _profile_relaxed() -> TuningProfile:
    """Short prompt, budget plentiful — quality-first.

    Everything above FILLER is FP16. FILLER is INT8, not summarized
    (we can afford to keep it around).
    """
    return TuningProfile(
        name="relaxed",
        policies={
            Tier.ANCHOR: _policy("fp16", pinned=True, evictable=False),
            Tier.SEMANTIC: _policy("fp16"),
            Tier.SUPPORTING: _policy("fp16"),
            Tier.FILLER: _policy("int8"),
        },
        max_entries_per_tier={},
        anchor_prefix_len=4,
        semantic_stride=3,
        rationale="Budget allows full precision for SEMANTIC and SUPPORTING; no eviction.",
    )


def _profile_balanced() -> TuningProfile:
    """The default operating point."""
    return TuningProfile(
        name="balanced",
        policies=dict(DEFAULT_POLICIES),
        max_entries_per_tier={},
        anchor_prefix_len=4,
        semantic_stride=3,
        rationale="Default PrimeKV policies: FP16 for SEMANTIC, INT4 for SUPPORTING/FILLER.",
    )


def _profile_tight(prompt_length: int) -> TuningProfile:
    """Medium pressure — cap SUPPORTING, aggressively quantize.

    SUPPORTING is INT4 with a capacity cap proportional to prompt
    length. FILLER is summarized.
    """
    return TuningProfile(
        name="tight",
        policies={
            Tier.ANCHOR: _policy("fp16", pinned=True, evictable=False),
            Tier.SEMANTIC: _policy("int8"),
            Tier.SUPPORTING: _policy("int4"),
            Tier.FILLER: _policy("int4", summary=True),
        },
        max_entries_per_tier={
            Tier.SUPPORTING: max(32, prompt_length // 4),
        },
        anchor_prefix_len=4,
        semantic_stride=2,  # more SEMANTIC tokens to offset INT8
        rationale="Memory pressure: quantize SEMANTIC to INT8, cap SUPPORTING, summarize FILLER.",
    )


def _profile_squeeze(prompt_length: int) -> TuningProfile:
    """Severe pressure — only ANCHOR and a tight SEMANTIC stay.

    Used when the budget is so tight we'd rather lose SUPPORTING
    entirely than trigger OOM.
    """
    return TuningProfile(
        name="squeeze",
        policies={
            Tier.ANCHOR: _policy("fp16", pinned=True, evictable=False),
            Tier.SEMANTIC: _policy("int4"),
            Tier.SUPPORTING: _policy("int4"),
            Tier.FILLER: _policy("int4", summary=True),
        },
        max_entries_per_tier={
            Tier.SEMANTIC: max(16, prompt_length // 8),
            Tier.SUPPORTING: max(16, prompt_length // 16),
        },
        anchor_prefix_len=8,  # longer sink helps at heavy compression
        semantic_stride=4,
        rationale="Severe memory pressure: aggressive INT4, tight per-tier caps.",
    )


# ---------------------------------------------------------------------------
# auto_tune
# ---------------------------------------------------------------------------


def auto_tune(
    prompt_length: int,
    memory_budget_mb: Optional[float] = None,
    num_layers: int = 12,
    num_heads: int = 12,
    head_dim: int = 64,
    tier_fractions: Optional[dict[Tier, float]] = None,
) -> TuningProfile:
    """Pick an operating point for ``(prompt_length, memory_budget_mb)``.

    If ``memory_budget_mb`` is None, choose by prompt length alone:

    * length < 256        → ``relaxed``
    * 256 <= length < 2048 → ``balanced``
    * length >= 2048       → ``tight``

    If a budget is provided, pick the most relaxed profile whose
    estimated payload fits the budget.
    """
    candidates: list[TuningProfile]
    if memory_budget_mb is None:
        if prompt_length < 256:
            chosen = _profile_relaxed()
        elif prompt_length < 2048:
            chosen = _profile_balanced()
        else:
            chosen = _profile_tight(prompt_length)
        chosen.estimated_bytes = estimate_bytes(
            prompt_length, chosen, num_layers, num_heads, head_dim, tier_fractions
        )
        return chosen

    budget_bytes = int(memory_budget_mb * 1024 * 1024)
    candidates = [
        _profile_relaxed(),
        _profile_balanced(),
        _profile_tight(prompt_length),
        _profile_squeeze(prompt_length),
    ]
    for profile in candidates:
        est = estimate_bytes(
            prompt_length, profile, num_layers, num_heads, head_dim, tier_fractions
        )
        profile.estimated_bytes = est
        if est <= budget_bytes:
            profile.rationale += f" (fits {est / (1024*1024):.1f}MB <= {memory_budget_mb:.1f}MB budget)"
            return profile

    # Nothing fits — return the tightest profile anyway, flagged.
    tightest = candidates[-1]
    tightest.rationale += (
        f" (OVER BUDGET: {tightest.estimated_bytes / (1024*1024):.1f}MB "
        f"> {memory_budget_mb:.1f}MB)"
    )
    return tightest


def describe_profile(profile: TuningProfile) -> str:
    """Return a compact multiline string for logs / UI."""
    lines = [f"profile={profile.name}: {profile.rationale}"]
    for tier in Tier:
        pol = profile.policies.get(tier, DEFAULT_POLICIES[tier])
        cap = profile.max_entries_per_tier.get(tier, "unlimited")
        flags = []
        if pol.pinned:
            flags.append("pinned")
        if pol.summary:
            flags.append("summary")
        if not pol.evictable:
            flags.append("not-evictable")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        lines.append(f"  {tier.name:10s} {pol.precision:4s} cap={cap}{flag_str}")
    if profile.estimated_bytes is not None:
        lines.append(f"  est_bytes={profile.estimated_bytes / (1024*1024):.1f} MB")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cache construction
# ---------------------------------------------------------------------------


def build_cache_from_profile(
    profile: TuningProfile,
    num_layers: int,
    classifier: Optional[BaseClassifier] = None,
    device: str = "cpu",
    enable_dynamic_reclassification: bool = True,
) -> PrimeKVCache:
    """Instantiate a :class:`PrimeKVCache` from a profile.

    If ``classifier`` is None, build a ``RuleBasedClassifier`` using
    the profile's ``anchor_prefix_len`` and ``semantic_stride``. Pass
    your own classifier to use :class:`SpaCyClassifier` or a learned
    head.
    """
    if classifier is None:
        classifier = RuleBasedClassifier(
            anchor_prefix_len=profile.anchor_prefix_len,
            semantic_stride=profile.semantic_stride,
        )
    return PrimeKVCache(
        num_layers=num_layers,
        classifier=classifier,
        policies=profile.policies,
        max_entries_per_tier=profile.max_entries_per_tier or None,
        device=device,
        enable_dynamic_reclassification=enable_dynamic_reclassification,
    )
