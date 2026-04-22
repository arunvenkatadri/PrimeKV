"""Comparison harness for KV cache strategies.

This module gives every cache backend (PrimeKV and every baseline) a
single shared interface — :class:`CacheProtocol` — and a single way to
evaluate them against the same workload: :func:`run_comparison`.

Usage:

    from primekv.eval import Workload, run_comparison
    from primekv.baselines import FullCache
    from primekv.cache import PrimeKVCache
    from primekv.classifier import RuleBasedClassifier

    tok, model = ...  # HuggingFace model + tokenizer
    caches = {
        "full":    FullCache(num_layers=12),
        "primekv": PrimeKVCache(num_layers=12, classifier=RuleBasedClassifier()),
    }
    report = run_comparison(caches, Workload(prompt="..."), model, tok)
    print(report.to_markdown())

The harness runs each cache through the *same* model + prompt + decode
loop, so the resulting :class:`CompareReport` is directly comparable
across backends.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Optional, Protocol, runtime_checkable

import torch

log = logging.getLogger("primekv.eval")


# ---------------------------------------------------------------------------
# Shared cache interface
# ---------------------------------------------------------------------------


@runtime_checkable
class CacheProtocol(Protocol):
    """The minimal interface every cache backend must implement.

    Notes:
        * ``put`` may accept extra backend-specific kwargs (e.g. PrimeKV's
          ``tier=``); the harness only passes positional args so this
          stays compatible.
        * ``get`` must return FP16 (or FP32) tensors of shape
          ``(num_heads, head_dim)``. Quantized backends should dequantize
          on read.
    """

    def put(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> Any: ...

    def get(
        self, layer: int, position: int
    ) -> Optional[tuple[torch.Tensor, torch.Tensor]]: ...

    def memory_bytes(self) -> int: ...

    def reset(self) -> None: ...


# ---------------------------------------------------------------------------
# Workload / result dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Workload:
    """One evaluation scenario.

    Attributes:
        prompt: The prompt string fed to the model at prefill.
        decode_tokens: How many tokens to generate with the (lossy)
            reconstructed KV cache.
        max_length: Tokenizer truncation length for the prompt.
        name: Human-readable label, only used for logging / reports.
        seed: Optional RNG seed. When set, :func:`run_with_cache` seeds
            torch and numpy before the decode loop, making sampling
            runs reproducible. Has no effect on argmax decoding.
        sample_top_k: If >0, decode with top-k sampling at temperature
            ``sample_temperature`` instead of argmax. Needed to make
            ``seed`` produce different outputs across runs.
        sample_temperature: Softmax temperature for sampling. Ignored
            when ``sample_top_k == 0``.
    """

    prompt: str
    decode_tokens: int = 32
    max_length: int = 256
    name: str = "default"
    seed: Optional[int] = None
    sample_top_k: int = 0
    sample_temperature: float = 1.0


@dataclass
class CacheResult:
    """Metrics for a single cache run.

    All fields are plain Python scalars so the dataclass can be dumped
    to JSON via ``asdict`` without custom encoders.
    """

    name: str
    memory_bytes: int
    compression_ratio: float
    perplexity: Optional[float]
    prefill_ms: float
    decode_ms: float
    tokens_per_second: float
    generated: Optional[str] = None
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        """Return a flat dict suitable for tabular rendering."""
        return {
            "cache": self.name,
            "memory_MB": round(self.memory_bytes / (1024 * 1024), 3),
            "ratio": round(self.compression_ratio, 2),
            "ppl": round(self.perplexity, 3) if self.perplexity is not None else None,
            "prefill_ms": round(self.prefill_ms, 2),
            "decode_ms": round(self.decode_ms, 2),
            "tok_per_s": round(self.tokens_per_second, 1),
        }


@dataclass
class CompareReport:
    """All cache results from one :func:`run_comparison` call."""

    workload: Workload
    results: list[CacheResult]
    model_name: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "workload": asdict(self.workload),
            "model_name": self.model_name,
            "results": [asdict(r) for r in self.results],
        }

    def as_rows(self) -> list[dict]:
        return [r.as_row() for r in self.results]

    def to_markdown(self) -> str:
        """Render as a GitHub-flavored markdown table."""
        rows = self.as_rows()
        if not rows:
            return "_(no results)_"
        cols = list(rows[0].keys())
        header = "| " + " | ".join(cols) + " |"
        sep = "|" + "|".join(["---"] * len(cols)) + "|"
        lines = [header, sep]
        for r in rows:
            lines.append(
                "| " + " | ".join("" if r[c] is None else str(r[c]) for c in cols) + " |"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_comparison(
    caches: dict[str, CacheProtocol],
    workload: Workload,
    model,
    tokenizer,
    device: str = "cpu",
    baseline_name: str = "full",
) -> CompareReport:
    """Run every cache in ``caches`` through ``workload`` and report.

    The same model and tokenizer are reused across caches; only the
    cache substitution changes. Compression ratios are computed relative
    to whichever cache is named ``baseline_name`` (default ``"full"``),
    falling back to the first cache in the dict if that name is missing.

    The heavy lifting — prefill, cache population, reconstruction,
    decode, perplexity — lives in :mod:`primekv.adapters.gpt2` so the
    eval module stays small and testable without a model.
    """
    from primekv.adapters.gpt2 import run_with_cache  # local import: avoids cycles

    model_name = getattr(getattr(model, "config", None), "name_or_path", None)

    results: list[CacheResult] = []
    for name, cache in caches.items():
        cache.reset()
        log.info("running cache=%s", name)
        result = run_with_cache(name, cache, model, tokenizer, workload, device=device)
        results.append(result)

    # Fix up compression ratios now that all sizes are known.
    baseline_bytes: Optional[int] = None
    for r in results:
        if r.name == baseline_name:
            baseline_bytes = r.memory_bytes
            break
    if baseline_bytes is None and results:
        baseline_bytes = results[0].memory_bytes
    base = baseline_bytes or 1
    for r in results:
        r.compression_ratio = base / max(r.memory_bytes, 1)

    return CompareReport(workload=workload, results=results, model_name=model_name)
