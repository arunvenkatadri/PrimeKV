"""Tier-aware attention.

This is a *reference* attention implementation, not a fused kernel. It
takes a query tensor plus a :class:`PrimeKVCache` and produces an output
by attending to:

* Tier 0 + Tier 1 entries at full resolution.
* Tier 2 entries only if a cheap relevance estimate exceeds a threshold
  (prefetching from CPU when needed).
* Tier 3: skipped, or attention over the per-layer summary vectors if
  ``use_summary=True``.

Everything is materialized into Python lists and then batched with a
single ``torch.matmul``. Good enough for correctness checks and
ablations; bad for throughput.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import torch

from primekv.cache import PrimeKVCache
from primekv.classifier import Tier

log = logging.getLogger("primekv.attention")


@dataclass
class AttentionConfig:
    relevance_threshold: float = 0.05
    use_summary: bool = True
    skip_tier3: bool = True
    approx_score_dim: int = 16  # subspace size used for cheap relevance est.


def _approx_relevance(q: torch.Tensor, k: torch.Tensor, dim: int) -> float:
    """Cheap query/key similarity on the first ``dim`` channels."""
    q_s = q[..., :dim].to(torch.float32)
    k_s = k[..., :dim].to(torch.float32)
    return float(torch.matmul(q_s, k_s.transpose(-1, -2)).amax().item())


def tiered_attention(
    query: torch.Tensor,
    layer: int,
    cache: PrimeKVCache,
    config: Optional[AttentionConfig] = None,
) -> torch.Tensor:
    """Compute tier-aware attention output for a single decode step.

    Args:
        query: FloatTensor of shape ``(num_heads, head_dim)`` — the current
            step's query for one layer.
        layer: layer index into the cache.
        cache: :class:`PrimeKVCache` instance.
        config: attention configuration.

    Returns:
        FloatTensor of shape ``(num_heads, head_dim)``.
    """
    cfg = config or AttentionConfig()

    collected_k: list[torch.Tensor] = []
    collected_v: list[torch.Tensor] = []
    collected_positions: list[int] = []

    # Tier 0 and 1: always attend.
    for tier in (Tier.ANCHOR, Tier.SEMANTIC):
        for pos in cache.positions_in_tier(tier):
            kv = cache.get(layer, pos)
            if kv is None:
                continue
            k, v = kv
            collected_k.append(k)
            collected_v.append(v)
            collected_positions.append(pos)

    # Tier 2: cheap relevance gate.
    for pos in cache.positions_in_tier(Tier.SUPPORTING):
        kv = cache.get(layer, pos)
        if kv is None:
            continue
        k, v = kv
        rel = _approx_relevance(query, k, dim=cfg.approx_score_dim)
        if rel >= cfg.relevance_threshold:
            collected_k.append(k)
            collected_v.append(v)
            collected_positions.append(pos)
        else:
            log.debug("skip tier2 pos=%d rel=%.4f", pos, rel)

    # Tier 3: optionally attend to the learned summary vector.
    if not cfg.skip_tier3:
        for pos in cache.positions_in_tier(Tier.FILLER):
            kv = cache.get(layer, pos)
            if kv is None:
                continue
            k, v = kv
            collected_k.append(k)
            collected_v.append(v)
            collected_positions.append(pos)
    if cfg.use_summary:
        summary = cache.summary_for(layer)
        if summary is not None:
            k_s, v_s = summary
            collected_k.append(k_s)
            collected_v.append(v_s)
            collected_positions.append(-1)  # sentinel

    if not collected_k:
        return torch.zeros_like(query)

    dtype = query.dtype
    K = torch.stack([k.to(dtype) for k in collected_k], dim=-2)  # (num_heads, N, head_dim)
    V = torch.stack([v.to(dtype) for v in collected_v], dim=-2)
    q = query.unsqueeze(-2)  # (num_heads, 1, head_dim)

    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, K.transpose(-1, -2)) * scale  # (num_heads, 1, N)
    probs = torch.softmax(scores, dim=-1)
    out = torch.matmul(probs, V).squeeze(-2)  # (num_heads, head_dim)

    # Feed attention signal back to the cache so it can promote/demote.
    avg_probs = probs.mean(dim=0).squeeze(0)  # (N,)
    for i, pos in enumerate(collected_positions):
        if pos < 0:
            continue
        cache.observe_attention(pos, float(avg_probs[i].item()))

    return out
