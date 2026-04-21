"""Tiered KV cache manager.

``PrimeKVCache`` is the main entry point. It stores one :class:`CacheEntry`
per (layer, token) slot, where each entry has a :class:`Tier` label and a
storage state determined by that tier's policy.

Supported policies (see :class:`TierPolicy`):

* ``precision``: one of ``"fp16"``, ``"int8"``, ``"int4"``.
* ``location``: ``"hbm"`` or ``"cpu"``.
* ``pinned``: if True, this entry is never evicted.
* ``evictable``: if False, eviction skips this entry.
* ``summary``: if True, entries of this tier are merged into a single
  learned summary vector instead of being stored verbatim.

All hits, misses, evictions, promotions, and demotions are counted on
:class:`CacheStats` and also emitted through the ``logging`` module under
the ``primekv.cache`` logger.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from typing import Optional

import torch

from primekv.classifier import BaseClassifier, Tier, TierAssignment, NUM_TIERS
from primekv.quantize import (
    QuantTensor,
    dequantize_int4,
    dequantize_int8,
    quantize_int4,
    quantize_int8,
)

log = logging.getLogger("primekv.cache")


# ---------------------------------------------------------------------------
# Policy / stats
# ---------------------------------------------------------------------------


@dataclass
class TierPolicy:
    precision: str = "fp16"  # fp16 | int8 | int4
    location: str = "hbm"    # hbm | cpu
    pinned: bool = False
    evictable: bool = True
    summary: bool = False


DEFAULT_POLICIES: dict[Tier, TierPolicy] = {
    Tier.ANCHOR: TierPolicy(precision="fp16", location="hbm", pinned=True, evictable=False),
    Tier.SEMANTIC: TierPolicy(precision="fp16", location="hbm", evictable=True),
    Tier.SUPPORTING: TierPolicy(precision="int4", location="hbm", evictable=True),
    Tier.FILLER: TierPolicy(precision="int4", location="hbm", evictable=True, summary=True),
}


@dataclass
class CacheStats:
    """Aggregate counters for one cache's lifetime.

    All fields are plain ``int`` for easy serialization.
    """

    hits: Counter = field(default_factory=Counter)
    misses: Counter = field(default_factory=Counter)
    evictions: Counter = field(default_factory=Counter)
    promotions: Counter = field(default_factory=Counter)
    demotions: Counter = field(default_factory=Counter)
    tokens_per_tier: Counter = field(default_factory=Counter)

    def to_dict(self) -> dict:
        return {
            "hits": dict(self.hits),
            "misses": dict(self.misses),
            "evictions": dict(self.evictions),
            "promotions": dict(self.promotions),
            "demotions": dict(self.demotions),
            "tokens_per_tier": dict(self.tokens_per_tier),
        }


# ---------------------------------------------------------------------------
# Cache entry
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    """A single cached (K, V) pair for one token at one layer.

    Exactly one of ``k_fp`` / ``k_q`` is populated (same for V). The
    ``tier`` and ``policy`` fields are kept alongside the data so the
    cache can reclassify entries without re-running the classifier.
    """

    layer: int
    position: int
    tier: Tier
    policy: TierPolicy
    k_fp: Optional[torch.Tensor] = None
    v_fp: Optional[torch.Tensor] = None
    k_q: Optional[QuantTensor] = None
    v_q: Optional[QuantTensor] = None
    last_used: int = 0
    ema_attn: float = 0.0  # rolling attention score, for promotion/demotion

    @property
    def device(self) -> torch.device:
        if self.k_fp is not None:
            return self.k_fp.device
        if self.k_q is not None:
            return self.k_q.data.device
        raise RuntimeError("entry has no storage")


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------


class PrimeKVCache:
    """Tier-aware KV cache.

    Args:
        num_layers: number of transformer layers in the base model.
        classifier: any :class:`BaseClassifier` used at prefill time.
        policies: per-tier policy overrides. Missing tiers fall back to
            ``DEFAULT_POLICIES``.
        max_entries_per_tier: optional hard cap per tier. ``None`` means
            unlimited; a full cache triggers LRU eviction within the tier.
        summary_dim: dimensionality of the learned per-layer summary
            vector used for Tier 3. Defaults to the model's head dim.
        device: torch device for HBM-resident entries.
        promotion_threshold: if an entry's EMA attention exceeds this,
            promote it by one tier (less compressed).
        demotion_threshold: symmetric, for demotion.
    """

    def __init__(
        self,
        num_layers: int,
        classifier: BaseClassifier,
        policies: Optional[dict[Tier, TierPolicy]] = None,
        max_entries_per_tier: Optional[dict[Tier, int]] = None,
        summary_dim: Optional[int] = None,
        device: str | torch.device = "cpu",
        promotion_threshold: float = 0.25,
        demotion_threshold: float = 0.01,
        enable_dynamic_reclassification: bool = True,
    ) -> None:
        self.num_layers = num_layers
        self.classifier = classifier
        self.policies = {**DEFAULT_POLICIES, **(policies or {})}
        self.max_entries_per_tier = max_entries_per_tier or {}
        self.summary_dim = summary_dim
        self.device = torch.device(device)
        self.promotion_threshold = promotion_threshold
        self.demotion_threshold = demotion_threshold
        self.enable_dynamic_reclassification = enable_dynamic_reclassification

        # entries[layer][position] -> CacheEntry
        self._entries: list[dict[int, CacheEntry]] = [dict() for _ in range(num_layers)]
        # Per-tier LRU of positions (we share positions across layers).
        self._lru: dict[Tier, list[int]] = {t: [] for t in Tier}
        # Learned summary vectors (K, V) per layer, populated lazily.
        self._summaries_k: list[Optional[torch.Tensor]] = [None] * num_layers
        self._summaries_v: list[Optional[torch.Tensor]] = [None] * num_layers

        self.stats = CacheStats()
        self._step: int = 0
        self._tier_of_position: dict[int, Tier] = {}

    # ------------------------------------------------------------------ prefill

    def classify_prefill(
        self,
        input_ids: Optional[torch.Tensor] = None,
        hidden_states: Optional[torch.Tensor] = None,
        **classifier_kwargs,
    ) -> TierAssignment:
        """Run the classifier once at prefill and remember the assignment.

        Extra kwargs are forwarded to the classifier — this is how
        :class:`SpaCyClassifier` receives ``text`` and ``tokenizer``.
        """
        assignment = self.classifier.classify(
            input_ids=input_ids, hidden_states=hidden_states, **classifier_kwargs
        )
        for pos, t in enumerate(assignment.tiers.tolist()):
            self._tier_of_position[pos] = Tier(t)
            self.stats.tokens_per_tier[Tier(t)] += 1
        log.info("prefill classified %d tokens: %s", len(assignment.tiers), assignment.counts())
        return assignment

    # ------------------------------------------------------------------ insert

    def put(
        self,
        layer: int,
        position: int,
        k: torch.Tensor,
        v: torch.Tensor,
        tier: Optional[Tier] = None,
    ) -> CacheEntry:
        """Insert or overwrite a (K, V) pair at ``(layer, position)``.

        If ``tier`` is None, the tier recorded at prefill is used.
        """
        if tier is None:
            tier = self._tier_of_position.get(position, Tier.SUPPORTING)
        policy = self.policies[tier]

        entry = CacheEntry(
            layer=layer,
            position=position,
            tier=tier,
            policy=policy,
            last_used=self._step,
        )
        self._store(entry, k, v)

        # Tier 3 with summary policy: fold into summary vector instead of
        # keeping the raw entry.
        if policy.summary:
            self._fold_into_summary(layer, k, v)
            log.debug("layer=%d pos=%d tier=%s folded into summary", layer, position, tier.name)
            return entry

        self._entries[layer][position] = entry
        if position not in self._lru[tier]:
            self._lru[tier].append(position)
        self._maybe_evict(tier)
        log.debug(
            "put layer=%d pos=%d tier=%s precision=%s loc=%s",
            layer,
            position,
            tier.name,
            policy.precision,
            policy.location,
        )
        return entry

    def _store(self, entry: CacheEntry, k: torch.Tensor, v: torch.Tensor) -> None:
        target_device = self.device if entry.policy.location == "hbm" else torch.device("cpu")
        k = k.to(target_device)
        v = v.to(target_device)
        if entry.policy.precision == "fp16":
            entry.k_fp = k.to(torch.float16) if k.dtype != torch.float16 else k
            entry.v_fp = v.to(torch.float16) if v.dtype != torch.float16 else v
        elif entry.policy.precision == "int8":
            entry.k_q = quantize_int8(k.to(torch.float32))
            entry.v_q = quantize_int8(v.to(torch.float32))
        elif entry.policy.precision == "int4":
            entry.k_q = quantize_int4(k.to(torch.float32))
            entry.v_q = quantize_int4(v.to(torch.float32))
        else:
            raise ValueError(f"unknown precision: {entry.policy.precision}")

    # ------------------------------------------------------------------ fetch

    def get(self, layer: int, position: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        """Return the (K, V) for ``(layer, position)`` as FP16 tensors, or None."""
        entry = self._entries[layer].get(position)
        if entry is None:
            tier = self._tier_of_position.get(position, Tier.SUPPORTING)
            self.stats.misses[tier] += 1
            log.debug("miss layer=%d pos=%d tier=%s", layer, position, tier.name)
            return None

        self.stats.hits[entry.tier] += 1
        entry.last_used = self._step
        self._touch_lru(entry.tier, position)

        k, v = self._materialize(entry)
        if entry.policy.location == "cpu":
            # Prefetch on demand: move back to HBM for this decode step.
            k = k.to(self.device)
            v = v.to(self.device)
        return k, v

    def _materialize(self, entry: CacheEntry) -> tuple[torch.Tensor, torch.Tensor]:
        if entry.policy.precision == "fp16":
            assert entry.k_fp is not None and entry.v_fp is not None
            return entry.k_fp, entry.v_fp
        if entry.policy.precision == "int8":
            assert entry.k_q is not None and entry.v_q is not None
            return dequantize_int8(entry.k_q).to(torch.float16), dequantize_int8(entry.v_q).to(torch.float16)
        if entry.policy.precision == "int4":
            assert entry.k_q is not None and entry.v_q is not None
            return dequantize_int4(entry.k_q).to(torch.float16), dequantize_int4(entry.v_q).to(torch.float16)
        raise ValueError(entry.policy.precision)

    # ------------------------------------------------------------------ LRU / eviction

    def _touch_lru(self, tier: Tier, position: int) -> None:
        lru = self._lru[tier]
        if position in lru:
            lru.remove(position)
        lru.append(position)

    def _maybe_evict(self, tier: Tier) -> None:
        cap = self.max_entries_per_tier.get(tier)
        if cap is None:
            return
        policy = self.policies[tier]
        if policy.pinned or not policy.evictable:
            return
        lru = self._lru[tier]
        while len(lru) > cap:
            victim_pos = lru.pop(0)
            for layer in range(self.num_layers):
                if victim_pos in self._entries[layer]:
                    del self._entries[layer][victim_pos]
            self.stats.evictions[tier] += 1
            log.debug("evict tier=%s pos=%d", tier.name, victim_pos)

    # ------------------------------------------------------------------ reclassify

    def observe_attention(self, position: int, score: float, alpha: float = 0.1) -> None:
        """Update the EMA attention signal for all layers at ``position``.

        Call this during decode from the attention module. Triggers a
        tier change when the EMA crosses a threshold and dynamic
        reclassification is enabled.
        """
        if not self.enable_dynamic_reclassification:
            return
        promoted = False
        demoted = False
        for layer in range(self.num_layers):
            entry = self._entries[layer].get(position)
            if entry is None:
                continue
            entry.ema_attn = (1 - alpha) * entry.ema_attn + alpha * score
            if entry.ema_attn >= self.promotion_threshold and entry.tier > Tier.ANCHOR:
                self._change_tier(entry, Tier(int(entry.tier) - 1))
                promoted = True
            elif entry.ema_attn <= self.demotion_threshold and entry.tier < Tier.FILLER:
                if not entry.policy.pinned:
                    self._change_tier(entry, Tier(int(entry.tier) + 1))
                    demoted = True
        if promoted:
            self.stats.promotions[Tier.SEMANTIC] += 1
        if demoted:
            self.stats.demotions[Tier.SUPPORTING] += 1

    def _change_tier(self, entry: CacheEntry, new_tier: Tier) -> None:
        log.debug("reclassify pos=%d %s -> %s", entry.position, entry.tier.name, new_tier.name)
        old_tier = entry.tier
        k, v = self._materialize(entry)
        entry.k_fp = None
        entry.v_fp = None
        entry.k_q = None
        entry.v_q = None
        entry.tier = new_tier
        entry.policy = self.policies[new_tier]
        self._store(entry, k, v)
        if entry.position in self._lru[old_tier]:
            self._lru[old_tier].remove(entry.position)
        if entry.position not in self._lru[new_tier]:
            self._lru[new_tier].append(entry.position)

    # ------------------------------------------------------------------ summary

    def _fold_into_summary(self, layer: int, k: torch.Tensor, v: torch.Tensor) -> None:
        # Learned summary = running mean. A real implementation would use
        # a tiny learned projection — this is enough to plumb the data flow.
        k_s = self._summaries_k[layer]
        v_s = self._summaries_v[layer]
        k = k.to(self.device).to(torch.float16)
        v = v.to(self.device).to(torch.float16)
        self._summaries_k[layer] = k if k_s is None else 0.5 * (k_s + k)
        self._summaries_v[layer] = v if v_s is None else 0.5 * (v_s + v)

    def summary_for(self, layer: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        k = self._summaries_k[layer]
        v = self._summaries_v[layer]
        if k is None or v is None:
            return None
        return k, v

    # ------------------------------------------------------------------ misc

    def step(self) -> None:
        self._step += 1

    def positions_in_tier(self, tier: Tier) -> list[int]:
        return list(self._lru[tier])

    def memory_bytes(self) -> int:
        """Rough payload size across all cached entries (KV only)."""
        total = 0
        for layer_entries in self._entries:
            for entry in layer_entries.values():
                if entry.k_fp is not None:
                    total += entry.k_fp.numel() * entry.k_fp.element_size()
                if entry.v_fp is not None:
                    total += entry.v_fp.numel() * entry.v_fp.element_size()
                if entry.k_q is not None:
                    total += entry.k_q.nbytes()
                if entry.v_q is not None:
                    total += entry.v_q.nbytes()
        return total

    def reset(self) -> None:
        self._entries = [dict() for _ in range(self.num_layers)]
        self._lru = {t: [] for t in Tier}
        self._summaries_k = [None] * self.num_layers
        self._summaries_v = [None] * self.num_layers
        self._tier_of_position.clear()
        self.stats = CacheStats()
        self._step = 0
