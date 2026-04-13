"""Evaluation utilities.

Covers:

* Perplexity on a token stream.
* KV cache compression ratio.
* Hit / miss / eviction rate summaries.
* GPU memory tracking helpers.
* Simple latency timing context managers.

All helpers are plain functions; nothing here is ``nn.Module``.
"""

from __future__ import annotations

import math
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Iterator, Optional

import torch

from primekv.cache import CacheStats, PrimeKVCache
from primekv.classifier import Tier


# ---------------------------------------------------------------------------
# Perplexity
# ---------------------------------------------------------------------------


@torch.no_grad()
def perplexity_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Compute perplexity from raw logits.

    Args:
        logits: ``(seq_len, vocab_size)``.
        targets: ``(seq_len,)``, the next-token labels.
    """
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    nll = -log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return float(math.exp(nll.mean().item()))


@torch.no_grad()
def streaming_perplexity(model, input_ids: torch.Tensor) -> float:
    """Run the model over ``input_ids`` and return perplexity.

    Assumes a HuggingFace CausalLM. Useful for baseline / regression
    checks — not optimized.
    """
    device = next(model.parameters()).device
    input_ids = input_ids.to(device)
    out = model(input_ids=input_ids, labels=input_ids)
    return float(math.exp(out.loss.item()))


# ---------------------------------------------------------------------------
# Compression / hit rates
# ---------------------------------------------------------------------------


def compression_ratio(
    full_bytes: int, compressed_bytes: int
) -> float:
    """``full / compressed``, so higher = more compression."""
    if compressed_bytes == 0:
        return float("inf")
    return full_bytes / compressed_bytes


def summarize_stats(stats: CacheStats) -> dict:
    """Produce a flat, loggable dict from a :class:`CacheStats`."""
    out = stats.to_dict()
    total_hits = sum(out["hits"].values())
    total_misses = sum(out["misses"].values())
    total_access = total_hits + total_misses
    out["hit_rate"] = total_hits / total_access if total_access else 0.0
    out["total_evictions"] = sum(out["evictions"].values())
    out["total_promotions"] = sum(out["promotions"].values())
    out["total_demotions"] = sum(out["demotions"].values())
    return out


def tier_distribution(cache: PrimeKVCache) -> dict[str, int]:
    """How many tokens are currently resident per tier."""
    return {t.name: len(cache.positions_in_tier(t)) for t in Tier}


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


def gpu_memory_snapshot(device: Optional[torch.device] = None) -> dict[str, int]:
    """Return a snapshot of torch CUDA memory stats (bytes)."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "max_allocated": 0}
    d = device or torch.cuda.current_device()
    return {
        "allocated": torch.cuda.memory_allocated(d),
        "reserved": torch.cuda.memory_reserved(d),
        "max_allocated": torch.cuda.max_memory_allocated(d),
    }


def reset_peak_memory(device: Optional[torch.device] = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


@dataclass
class LatencyRecord:
    label: str
    seconds: float


@dataclass
class LatencyLog:
    records: list[LatencyRecord] = field(default_factory=list)

    def add(self, label: str, seconds: float) -> None:
        self.records.append(LatencyRecord(label=label, seconds=seconds))

    def totals(self) -> dict[str, float]:
        totals: dict[str, float] = {}
        for r in self.records:
            totals[r.label] = totals.get(r.label, 0.0) + r.seconds
        return totals


@contextmanager
def timed(label: str, log: LatencyLog) -> Iterator[None]:
    """Context manager that records wall-clock duration into ``log``."""
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        log.add(label, time.perf_counter() - start)
