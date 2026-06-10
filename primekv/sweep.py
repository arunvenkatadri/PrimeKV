"""Parameter sweep harness.

Three canonical sweeps implemented here:

* :func:`sweep_pareto` — for each cache backend, vary its primary
  capacity knob and trace the (compression_ratio, perplexity) tradeoff.
  This is the headline plot for the paper.

* :func:`sweep_vs_length` — fix capacity and vary prompt length.
  Shows when each method breaks as context grows.

* :func:`sweep_ablate_primekv` — fix everything else and vary one
  PrimeKV hyperparameter at a time (anchor length, semantic stride,
  SUPPORTING cap, dynamic reclassification).

Every sweep returns a :class:`SweepReport` that can be serialized to
JSON, CSV, or rendered with :func:`plot_report` into a PNG.
"""

from __future__ import annotations

import csv
import io
import logging
import statistics
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import torch

from primekv.baselines import (
    FullCache,
    H2OCache,
    H2OQuantCache,
    StreamingLLMCache,
    StreamingQuantCache,
    ThreeZoneCache,
    UniformQuantCache,
)
from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier, Tier
from primekv.eval import Workload, run_comparison

log = logging.getLogger("primekv.sweep")


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class SweepPoint:
    """One (cache, setting, metric) row."""

    cache: str
    sweep_axis: str          # e.g. "capacity", "prompt_length"
    sweep_value: float       # numeric value of the axis for this row
    memory_bytes: int
    compression_ratio: float
    perplexity: Optional[float]
    prefill_ms: float
    decode_ms: float
    tokens_per_second: float
    extra: dict = field(default_factory=dict)


@dataclass
class SweepReport:
    """A collection of :class:`SweepPoint` rows from one sweep."""

    mode: str                 # "pareto" | "vs_length" | "ablate"
    axis_label: str           # human-readable x-axis
    points: list[SweepPoint] = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "mode": self.mode,
            "axis_label": self.axis_label,
            "meta": self.meta,
            "points": [asdict(p) for p in self.points],
        }

    def to_csv(self) -> str:
        buf = io.StringIO()
        if not self.points:
            return ""
        fieldnames = [
            "cache",
            "sweep_axis",
            "sweep_value",
            "memory_bytes",
            "compression_ratio",
            "perplexity",
            "prefill_ms",
            "decode_ms",
            "tokens_per_second",
        ]
        w = csv.DictWriter(buf, fieldnames=fieldnames)
        w.writeheader()
        for p in self.points:
            row = {k: getattr(p, k) for k in fieldnames}
            w.writerow(row)
        return buf.getvalue()

    def by_cache(self) -> dict[str, list[SweepPoint]]:
        groups: dict[str, list[SweepPoint]] = {}
        for p in self.points:
            groups.setdefault(p.cache, []).append(p)
        for cache_name in groups:
            groups[cache_name].sort(key=lambda x: x.sweep_value)
        return groups


# ---------------------------------------------------------------------------
# Cache factories (keyed by name so sweeps can build caches by label)
# ---------------------------------------------------------------------------


def _build_primekv(
    num_layers: int,
    supporting_cap: int,
    tier2_precision: str = "int4",
    **kwargs,
) -> PrimeKVCache:
    from primekv.cache import DEFAULT_POLICIES, TierPolicy

    policies = dict(DEFAULT_POLICIES)
    # Allow the caller to override Tier 2's precision (FP16 / INT8 / INT4).
    # This is what makes the 2D eviction × quantization sweep possible.
    policies[Tier.SUPPORTING] = TierPolicy(
        precision=tier2_precision,
        location="hbm",
        evictable=True,
    )
    return PrimeKVCache(
        num_layers=num_layers,
        classifier=RuleBasedClassifier(
            anchor_prefix_len=kwargs.get("anchor_prefix_len", 16),
            semantic_stride=kwargs.get("semantic_stride", 3),
        ),
        policies=policies,
        max_entries_per_tier={Tier.SUPPORTING: int(supporting_cap)},
        enable_dynamic_reclassification=kwargs.get(
            "enable_dynamic_reclassification", True
        ),
    )


_PRECISION_TO_BITS = {"fp16": None, "int8": 8, "int4": 4}


def _cache_for(name: str, num_layers: int, capacity: int, **kwargs) -> Any:
    if name == "full":
        return FullCache(num_layers)
    if name == "uniform_int8":
        return UniformQuantCache(num_layers, bits=8)
    if name == "uniform_int4":
        return UniformQuantCache(num_layers, bits=4)
    if name == "h2o":
        return H2OCache(num_layers, capacity=int(capacity))
    if name == "h2o_int8":
        return H2OQuantCache(num_layers, capacity=int(capacity), bits=8)
    if name == "h2o_int4":
        return H2OQuantCache(num_layers, capacity=int(capacity), bits=4)
    if name == "streamingllm":
        return StreamingLLMCache(
            num_layers, num_sinks=kwargs.get("num_sinks", 4), window=int(capacity)
        )
    if name == "streamingllm_int8":
        return StreamingQuantCache(
            num_layers, num_sinks=kwargs.get("num_sinks", 4), window=int(capacity), bits=8
        )
    if name == "streamingllm_int4":
        return StreamingQuantCache(
            num_layers, num_sinks=kwargs.get("num_sinks", 4), window=int(capacity), bits=4
        )
    if name == "primekv":
        return _build_primekv(num_layers, supporting_cap=int(capacity), **kwargs)
    if name == "three_zone":
        # ``capacity`` is interpreted as the rolling FP16 window size.
        # Middle pool defaults to int4 with no capacity cap unless overridden.
        return ThreeZoneCache(
            num_layers,
            num_anchor=int(kwargs.get("num_anchor", 16)),
            recent_window=int(capacity),
            middle_capacity=kwargs.get("middle_capacity"),
            middle_bits=kwargs.get("middle_bits", 4),
        )
    raise ValueError(f"unknown cache name: {name}")


# Caches whose compression is *not* capacity-driven: they always appear
# as a single point regardless of the sweep axis.
FIXED_COMPRESSION_CACHES = ("full", "uniform_int8", "uniform_int4")

# Caches whose "primary capacity knob" is the sweep axis.
CAPACITY_CACHES = ("h2o", "streamingllm", "primekv")


# ---------------------------------------------------------------------------
# Sweeps
# ---------------------------------------------------------------------------


def _num_layers(model) -> int:
    return getattr(
        model.config,
        "num_hidden_layers",
        getattr(model.config, "n_layer", 12),
    )


def _run_once(
    cache_name: str,
    cache,
    workload: Workload,
    model,
    tokenizer,
    device: str,
) -> dict:
    """Run one cache through one workload, return its metrics dict.

    Thin wrapper around :func:`run_comparison` with a single cache so we
    get the same compression-ratio normalization (vs ``full``).
    """
    report = run_comparison(
        {cache_name: cache}, workload, model, tokenizer, device=device
    )
    result = report.results[0]
    return {
        "memory_bytes": result.memory_bytes,
        "compression_ratio": result.compression_ratio,
        "perplexity": result.perplexity,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "tokens_per_second": result.tokens_per_second,
    }


def _make_workload(
    prompt: str,
    decode_tokens: int,
    max_length: int,
    seed: Optional[int],
    sample_top_k: int,
    sample_temperature: float,
) -> Workload:
    return Workload(
        prompt=prompt,
        decode_tokens=decode_tokens,
        max_length=max_length,
        seed=seed,
        sample_top_k=sample_top_k,
        sample_temperature=sample_temperature,
    )


def _with_full_baseline(
    cache_name: str,
    cache,
    full_cache,
    workload: Workload,
    model,
    tokenizer,
    device: str,
) -> dict:
    """Run {full, cache} together so compression_ratio is normalized against full."""
    report = run_comparison(
        {"full": full_cache, cache_name: cache},
        workload,
        model,
        tokenizer,
        device=device,
    )
    result = next(r for r in report.results if r.name == cache_name)
    return {
        "memory_bytes": result.memory_bytes,
        "compression_ratio": result.compression_ratio,
        "perplexity": result.perplexity,
        "prefill_ms": result.prefill_ms,
        "decode_ms": result.decode_ms,
        "tokens_per_second": result.tokens_per_second,
    }


def sweep_pareto(
    model,
    tokenizer,
    prompt: str,
    capacities: list[int],
    caches: Optional[list[str]] = None,
    decode_tokens: int = 16,
    max_length: int = 512,
    device: str = "cpu",
    progress: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = None,
    sample_top_k: int = 0,
    sample_temperature: float = 1.0,
) -> SweepReport:
    """Pareto quality/compression curve for every cache in ``caches``.

    Capacity-driven caches (H2O, StreamingLLM, PrimeKV) get one point
    per capacity value. Fixed-compression caches (full, uniform_int8,
    uniform_int4) get a single point at the smallest capacity value
    (their ratio doesn't depend on capacity).
    """
    caches = caches or list(FIXED_COMPRESSION_CACHES) + list(CAPACITY_CACHES)
    num_layers = _num_layers(model)
    workload = _make_workload(
        prompt, decode_tokens, max_length, seed, sample_top_k, sample_temperature
    )

    report = SweepReport(
        mode="pareto",
        axis_label="capacity",
        meta={
            "capacities": capacities,
            "decode_tokens": decode_tokens,
            "max_length": max_length,
            "device": device,
            "seed": seed,
            "sample_top_k": sample_top_k,
            "sample_temperature": sample_temperature,
        },
    )

    # The baseline "full" cache is needed to normalize compression ratios.
    full_cache = FullCache(num_layers)

    for cache_name in caches:
        if cache_name in FIXED_COMPRESSION_CACHES:
            if progress:
                progress(f"running {cache_name} (fixed)")
            cache = _cache_for(cache_name, num_layers, capacity=0)
            full_cache.reset()
            metrics = _with_full_baseline(
                cache_name, cache, full_cache, workload, model, tokenizer, device
            )
            report.points.append(
                SweepPoint(
                    cache=cache_name,
                    sweep_axis="capacity",
                    sweep_value=float("inf"),
                    **metrics,
                )
            )
        else:
            for cap in capacities:
                if progress:
                    progress(f"running {cache_name} cap={cap}")
                cache = _cache_for(cache_name, num_layers, capacity=cap)
                full_cache.reset()
                metrics = _with_full_baseline(
                    cache_name, cache, full_cache, workload, model, tokenizer, device
                )
                report.points.append(
                    SweepPoint(
                        cache=cache_name,
                        sweep_axis="capacity",
                        sweep_value=float(cap),
                        **metrics,
                    )
                )

    return report


def sweep_vs_length(
    model,
    tokenizer,
    prompt: str,
    lengths: list[int],
    caches: Optional[list[str]] = None,
    capacity: int = 16,
    decode_tokens: int = 16,
    device: str = "cpu",
    progress: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = None,
    sample_top_k: int = 0,
    sample_temperature: float = 1.0,
) -> SweepReport:
    """Sweep prompt length at fixed capacity.

    ``prompt`` is truncated by the tokenizer to each length in
    ``lengths``. The resulting tokens become the workload for that row.
    """
    caches = caches or list(FIXED_COMPRESSION_CACHES) + list(CAPACITY_CACHES)
    num_layers = _num_layers(model)

    report = SweepReport(
        mode="vs_length",
        axis_label="prompt_length",
        meta={
            "lengths": lengths,
            "capacity": capacity,
            "decode_tokens": decode_tokens,
            "device": device,
            "seed": seed,
            "sample_top_k": sample_top_k,
            "sample_temperature": sample_temperature,
        },
    )
    full_cache = FullCache(num_layers)

    for length in lengths:
        workload = _make_workload(
            prompt, decode_tokens, int(length), seed, sample_top_k, sample_temperature
        )
        for cache_name in caches:
            if progress:
                progress(f"running {cache_name} len={length}")
            cache = _cache_for(cache_name, num_layers, capacity=capacity)
            full_cache.reset()
            metrics = _with_full_baseline(
                cache_name, cache, full_cache, workload, model, tokenizer, device
            )
            report.points.append(
                SweepPoint(
                    cache=cache_name,
                    sweep_axis="prompt_length",
                    sweep_value=float(length),
                    **metrics,
                )
            )
    return report


def sweep_ablate_primekv(
    model,
    tokenizer,
    prompt: str,
    axis: str,
    values: list,
    capacity: int = 16,
    decode_tokens: int = 16,
    max_length: int = 512,
    device: str = "cpu",
    progress: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = None,
    sample_top_k: int = 0,
    sample_temperature: float = 1.0,
) -> SweepReport:
    """PrimeKV ablation. ``axis`` is one of:

    * ``"anchor_prefix_len"``
    * ``"semantic_stride"``
    * ``"supporting_cap"``
    * ``"enable_dynamic_reclassification"`` (values: ``[False, True]``)
    """
    if axis not in {
        "anchor_prefix_len",
        "semantic_stride",
        "supporting_cap",
        "enable_dynamic_reclassification",
    }:
        raise ValueError(f"unknown ablation axis: {axis}")

    num_layers = _num_layers(model)
    workload = _make_workload(
        prompt, decode_tokens, max_length, seed, sample_top_k, sample_temperature
    )
    report = SweepReport(
        mode="ablate",
        axis_label=axis,
        meta={
            "capacity": capacity,
            "decode_tokens": decode_tokens,
            "max_length": max_length,
            "values": values,
            "seed": seed,
            "sample_top_k": sample_top_k,
            "sample_temperature": sample_temperature,
        },
    )
    full_cache = FullCache(num_layers)

    for val in values:
        if progress:
            progress(f"running primekv {axis}={val}")
        kwargs = {}
        cap = capacity
        if axis == "supporting_cap":
            cap = int(val)
        else:
            kwargs[axis] = val
        cache = _build_primekv(num_layers, supporting_cap=cap, **kwargs)
        full_cache.reset()
        metrics = _with_full_baseline(
            "primekv", cache, full_cache, workload, model, tokenizer, device
        )
        report.points.append(
            SweepPoint(
                cache="primekv",
                sweep_axis=axis,
                sweep_value=float(val) if isinstance(val, (int, float)) else float(bool(val)),
                **metrics,
                extra={"raw_value": val},
            )
        )
    return report


# ---------------------------------------------------------------------------
# 2D sweep: eviction × quantization
# ---------------------------------------------------------------------------


# Caches capable of eviction (varying cap makes sense).
_EVICTION_CAPABLE = {"h2o", "streamingllm", "primekv"}
# Caches capable of quantization (varying precision makes sense). Note:
# ``h2o`` and ``streamingllm`` are now in this set via the composed
# H2OQuantCache / StreamingQuantCache baselines, so the 2D sweep puts
# every capacity-driven method on the same 2D surface.
_QUANT_CAPABLE = {"uniform_quant", "h2o", "streamingllm", "primekv"}
# Fixed singletons.
_FIXED_SINGLETON = {"full"}


def sweep_2d_tradeoff(
    model,
    tokenizer,
    prompt: str,
    eviction_caps: list[int],
    precisions: list[str],
    caches: Optional[list[str]] = None,
    decode_tokens: int = 8,
    max_length: int = 256,
    device: str = "cpu",
    progress: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = None,
    sample_top_k: int = 0,
    sample_temperature: float = 1.0,
) -> SweepReport:
    """Sweep eviction × quantization simultaneously.

    Each cache fills in the cells it's capable of reaching:

    * ``full`` → one cell (no eviction, no quantization)
    * ``uniform_int8`` / ``uniform_int4`` → one cell each (keep all, quantize)
    * ``h2o``, ``streamingllm`` → **full grid** (eviction × precision),
      via :class:`H2OQuantCache` / :class:`StreamingQuantCache`. This
      makes the comparison against PrimeKV two-lever-vs-two-lever.
    * ``primekv`` → full grid (vary both eviction and Tier-2 precision)

    Every capacity-driven baseline now occupies the full 2D plane, so
    PrimeKV no longer gets a free "only method with two levers" win.

    ``precisions`` entries must be in ``{"fp16", "int8", "int4"}``.
    """
    if caches is None:
        caches = ["full", "uniform_int8", "uniform_int4", "h2o", "streamingllm", "primekv"]
    for p in precisions:
        if p not in {"fp16", "int8", "int4"}:
            raise ValueError(f"unknown precision: {p}")

    num_layers = _num_layers(model)
    workload = _make_workload(
        prompt, decode_tokens, max_length, seed, sample_top_k, sample_temperature
    )
    report = SweepReport(
        mode="2d_tradeoff",
        axis_label="compression_ratio",
        meta={
            "eviction_caps": eviction_caps,
            "precisions": precisions,
            "decode_tokens": decode_tokens,
            "max_length": max_length,
            "device": device,
            "seed": seed,
            "sample_top_k": sample_top_k,
            "sample_temperature": sample_temperature,
        },
    )
    full_cache = FullCache(num_layers)

    def _emit(cache_name: str, cap_value: float, precision: str, cache) -> None:
        if progress:
            progress(f"running {cache_name} cap={cap_value} prec={precision}")
        full_cache.reset()
        metrics = _with_full_baseline(
            cache_name, cache, full_cache, workload, model, tokenizer, device
        )
        report.points.append(
            SweepPoint(
                cache=cache_name,
                sweep_axis="2d",
                sweep_value=metrics["compression_ratio"],
                **metrics,
                extra={"cap": cap_value, "precision": precision},
            )
        )

    for cache_name in caches:
        if cache_name == "full":
            _emit("full", float("inf"), "fp16", FullCache(num_layers))
            continue
        if cache_name == "uniform_int8":
            _emit("uniform_int8", float("inf"), "int8", UniformQuantCache(num_layers, bits=8))
            continue
        if cache_name == "uniform_int4":
            _emit("uniform_int4", float("inf"), "int4", UniformQuantCache(num_layers, bits=4))
            continue
        if cache_name == "h2o":
            for cap in eviction_caps:
                for prec in precisions:
                    bits = _PRECISION_TO_BITS[prec]
                    if bits is None:
                        cache = H2OCache(num_layers, capacity=int(cap))
                    else:
                        cache = H2OQuantCache(num_layers, capacity=int(cap), bits=bits)
                    _emit("h2o", float(cap), prec, cache)
            continue
        if cache_name == "streamingllm":
            for cap in eviction_caps:
                for prec in precisions:
                    bits = _PRECISION_TO_BITS[prec]
                    if bits is None:
                        cache = StreamingLLMCache(num_layers, num_sinks=4, window=int(cap))
                    else:
                        cache = StreamingQuantCache(
                            num_layers, num_sinks=4, window=int(cap), bits=bits
                        )
                    _emit("streamingllm", float(cap), prec, cache)
            continue
        if cache_name == "primekv":
            for cap in eviction_caps:
                for prec in precisions:
                    _emit(
                        "primekv",
                        float(cap),
                        prec,
                        _build_primekv(num_layers, supporting_cap=cap, tier2_precision=prec),
                    )
            continue
        raise ValueError(f"unknown cache name: {cache_name}")

    return report


# ---------------------------------------------------------------------------
# Long-context sweep
# ---------------------------------------------------------------------------


# Default length grid for long-context runs. Spaced logarithmically so
# we can see where each method breaks. The top of the range needs a
# model whose ``max_position_embeddings`` actually supports it
# (GPT-2 caps at 1024 — use Qwen2.5-3B or similar).
DEFAULT_LONG_LENGTHS = [512, 1024, 2048, 4096, 8192, 16384]


def sweep_long_context(
    model,
    tokenizer,
    prompt: str,
    lengths: Optional[list[int]] = None,
    caches: Optional[list[str]] = None,
    capacity_fraction: float = 0.1,
    min_capacity: int = 32,
    precisions: Optional[list[str]] = None,
    decode_tokens: int = 16,
    device: str = "cpu",
    progress: Optional[Callable[[str], None]] = None,
    seed: Optional[int] = None,
    sample_top_k: int = 0,
    sample_temperature: float = 1.0,
) -> SweepReport:
    """Perplexity vs context length at fixed *relative* capacity.

    The key difference from :func:`sweep_vs_length`: capacity scales
    with prompt length so every cell represents the same *fraction* of
    tokens retained (``capacity_fraction``, min ``min_capacity``). That
    is the honest long-context question — "what's the quality cost of
    keeping 10% of a 16k-token cache?" — not "what happens at a fixed
    16-token budget as context grows?".

    Runs composed baselines (``h2o_int4``, ``streamingllm_int4``) along
    with PrimeKV by default so the comparison is two-lever-vs-two-lever
    at long context.
    """
    if lengths is None:
        lengths = list(DEFAULT_LONG_LENGTHS)
    if caches is None:
        # Full is too expensive at 16k on a single GPU; uniform_int4 is
        # cheap and acts as the "no eviction, pure quant" reference.
        caches = [
            "uniform_int4",
            "h2o_int4",
            "streamingllm_int4",
            "primekv",
        ]
    if precisions is None:
        precisions = ["int4"]

    num_layers = _num_layers(model)
    report = SweepReport(
        mode="long_context",
        axis_label="prompt_length",
        meta={
            "lengths": lengths,
            "capacity_fraction": capacity_fraction,
            "min_capacity": min_capacity,
            "decode_tokens": decode_tokens,
            "precisions": precisions,
            "device": device,
            "seed": seed,
            "sample_top_k": sample_top_k,
            "sample_temperature": sample_temperature,
        },
    )
    full_cache = FullCache(num_layers)

    for length in lengths:
        capacity = max(int(min_capacity), int(length * capacity_fraction))
        workload = _make_workload(
            prompt, decode_tokens, int(length), seed, sample_top_k, sample_temperature
        )
        for cache_name in caches:
            for prec in precisions:
                # Build the cache for this (name, precision) cell.
                if cache_name in FIXED_COMPRESSION_CACHES:
                    if prec != "int4" and cache_name == "uniform_int4":
                        # ``uniform_int4`` is pinned at int4 by name — skip other precisions.
                        continue
                    cache = _cache_for(cache_name, num_layers, capacity=0)
                elif cache_name == "primekv":
                    cache = _build_primekv(
                        num_layers, supporting_cap=capacity, tier2_precision=prec
                    )
                elif cache_name in ("h2o", "streamingllm"):
                    # FP16 variant (no quantization composed).
                    cache = _cache_for(cache_name, num_layers, capacity=capacity)
                elif cache_name in ("h2o_int8", "h2o_int4", "streamingllm_int8", "streamingllm_int4"):
                    # Pre-composed variant — precision is fixed by the name.
                    cache = _cache_for(cache_name, num_layers, capacity=capacity)
                else:
                    raise ValueError(f"unknown cache name for long-context sweep: {cache_name}")

                if progress:
                    progress(f"running {cache_name} ({prec}) len={length} cap={capacity}")
                full_cache.reset()
                metrics = _with_full_baseline(
                    cache_name, cache, full_cache, workload, model, tokenizer, device
                )
                report.points.append(
                    SweepPoint(
                        cache=cache_name,
                        sweep_axis="prompt_length",
                        sweep_value=float(length),
                        **metrics,
                        extra={"capacity": capacity, "precision": prec},
                    )
                )

    return report


# ---------------------------------------------------------------------------
# Seed aggregation
# ---------------------------------------------------------------------------


def _cell_key(p: SweepPoint) -> tuple:
    """Identifier that should be stable across seeded repeats of the same sweep."""
    extras = tuple(sorted(
        (k, v) for k, v in p.extra.items()
        if k not in {"stats", "tier_distribution"}
    ))
    return (p.cache, p.sweep_axis, p.sweep_value, extras)


def _mean_std(xs: list[float]) -> tuple[float, float]:
    xs = [x for x in xs if x is not None]
    if not xs:
        return (float("nan"), float("nan"))
    if len(xs) == 1:
        return (float(xs[0]), 0.0)
    return (statistics.fmean(xs), statistics.pstdev(xs))


def aggregate_reports(reports: list[SweepReport]) -> SweepReport:
    """Merge N :class:`SweepReport` s (e.g. from N seeded runs) into one.

    Points with the same ``(cache, sweep_axis, sweep_value, extra)``
    are collapsed; each numeric metric is replaced by its mean across
    runs, with ``<metric>_std`` written into ``extra``. The ``meta``
    dict records the seed list.

    Typical use::

        seeds = [0, 1, 2, 3, 4]
        reports = [sweep_pareto(..., seed=s, sample_top_k=50) for s in seeds]
        agg = aggregate_reports(reports)
    """
    if not reports:
        raise ValueError("aggregate_reports needs at least one report")

    mode = reports[0].mode
    axis = reports[0].axis_label
    for r in reports[1:]:
        if r.mode != mode or r.axis_label != axis:
            raise ValueError("reports have inconsistent mode/axis")

    grouped: dict[tuple, list[SweepPoint]] = {}
    for r in reports:
        for p in r.points:
            grouped.setdefault(_cell_key(p), []).append(p)

    metric_names = (
        "memory_bytes",
        "compression_ratio",
        "perplexity",
        "prefill_ms",
        "decode_ms",
        "tokens_per_second",
    )

    merged: list[SweepPoint] = []
    for key, pts in grouped.items():
        first = pts[0]
        extra = dict(first.extra)
        extra["n_runs"] = len(pts)
        agg_metrics: dict[str, float] = {}
        for m in metric_names:
            vals = [getattr(p, m) for p in pts]
            mean, std = _mean_std(vals)
            agg_metrics[m] = mean
            extra[f"{m}_std"] = std
        merged.append(
            SweepPoint(
                cache=first.cache,
                sweep_axis=first.sweep_axis,
                sweep_value=first.sweep_value,
                memory_bytes=int(agg_metrics["memory_bytes"]) if agg_metrics["memory_bytes"] == agg_metrics["memory_bytes"] else 0,
                compression_ratio=agg_metrics["compression_ratio"],
                perplexity=agg_metrics["perplexity"] if agg_metrics["perplexity"] == agg_metrics["perplexity"] else None,
                prefill_ms=agg_metrics["prefill_ms"],
                decode_ms=agg_metrics["decode_ms"],
                tokens_per_second=agg_metrics["tokens_per_second"],
                extra=extra,
            )
        )

    meta = dict(reports[0].meta)
    meta["seeds"] = [r.meta.get("seed") for r in reports]
    meta["n_reports"] = len(reports)

    return SweepReport(mode=mode, axis_label=axis, points=merged, meta=meta)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


_CACHE_COLORS = {
    "full": "#333333",
    "uniform_int8": "#8c564b",
    "uniform_int4": "#d62728",
    "h2o": "#1f77b4",
    "streamingllm": "#2ca02c",
    "primekv": "#ff7f0e",
}


def plot_report(report: SweepReport, output_path: Optional[str] = None):
    """Render ``report`` to a matplotlib Figure and optionally save it.

    Returns the figure so callers can embed it (e.g. Gradio).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    groups = report.by_cache()

    if report.mode == "pareto":
        for name, pts in groups.items():
            xs = [p.compression_ratio for p in pts]
            ys = [p.perplexity for p in pts if p.perplexity is not None]
            if not ys:
                continue
            color = _CACHE_COLORS.get(name, None)
            if len(pts) == 1:
                ax.scatter(xs, ys, s=80, label=name, color=color, zorder=3)
            else:
                ax.plot(xs, ys, marker="o", label=name, color=color)
        ax.set_xscale("log")
        ax.set_xlabel("compression ratio (higher = more compressed)")
        ax.set_ylabel("perplexity (lower = better)")
        ax.set_title("Pareto: quality vs compression")
    elif report.mode == "vs_length":
        for name, pts in groups.items():
            xs = [p.sweep_value for p in pts]
            ys = [p.perplexity for p in pts if p.perplexity is not None]
            if not ys:
                continue
            color = _CACHE_COLORS.get(name, None)
            ax.plot(xs, ys, marker="o", label=name, color=color)
        ax.set_xlabel("prompt length (tokens)")
        ax.set_ylabel("perplexity (lower = better)")
        ax.set_title(f"Quality vs prompt length (capacity={report.meta.get('capacity')})")
    elif report.mode == "ablate":
        for name, pts in groups.items():
            xs = [p.sweep_value for p in pts]
            ys = [p.perplexity for p in pts if p.perplexity is not None]
            if not ys:
                continue
            color = _CACHE_COLORS.get(name, None)
            ax.plot(xs, ys, marker="o", label=name, color=color)
        ax.set_xlabel(report.axis_label)
        ax.set_ylabel("perplexity (lower = better)")
        ax.set_title(f"PrimeKV ablation: {report.axis_label}")
    elif report.mode == "long_context":
        # One line per (cache, precision). Precision lives in extra so
        # different precision rollups of the same cache get their own
        # line. Eviction capacity scales with length, so we plot ppl
        # against absolute prompt length on a log axis.
        precision_styles = {"fp16": "-", "int8": "--", "int4": ":"}
        seen_labels = set()
        for name, pts in groups.items():
            color = _CACHE_COLORS.get(name, None)
            # Partition points by precision so lines don't cross.
            buckets: dict[str, list[SweepPoint]] = {}
            for p in pts:
                buckets.setdefault(str(p.extra.get("precision", "fp16")), []).append(p)
            for prec, prec_pts in buckets.items():
                prec_pts.sort(key=lambda x: x.sweep_value)
                xs = [p.sweep_value for p in prec_pts]
                ys = [p.perplexity for p in prec_pts]
                if all(y is None for y in ys):
                    continue
                label = f"{name} ({prec})"
                if label in seen_labels:
                    continue
                seen_labels.add(label)
                ax.plot(
                    xs, ys,
                    marker="o",
                    linestyle=precision_styles.get(prec, "-"),
                    color=color,
                    label=label,
                )
        ax.set_xscale("log")
        ax.set_xlabel("prompt length (tokens, log scale)")
        ax.set_ylabel("perplexity (lower = better)")
        ax.set_title(
            "Long-context: quality vs context length\n"
            f"(capacity = {report.meta.get('capacity_fraction')} × length, "
            f"min {report.meta.get('min_capacity')})"
        )
    elif report.mode == "2d_tradeoff":
        # Scatter of every (cache, config) point. PrimeKV shows up as a
        # cloud because it explores the 2D eviction × quantization surface.
        # Precision is encoded as marker shape; cache as color.
        precision_markers = {"fp16": "o", "int8": "s", "int4": "^"}
        for name, pts in groups.items():
            color = _CACHE_COLORS.get(name, None)
            for prec, marker in precision_markers.items():
                prec_pts = [p for p in pts if p.extra.get("precision") == prec]
                if not prec_pts:
                    continue
                xs = [p.compression_ratio for p in prec_pts]
                ys = [p.perplexity for p in prec_pts if p.perplexity is not None]
                if not ys:
                    continue
                label = f"{name} ({prec})"
                size = 120 if len(prec_pts) == 1 else 60
                ax.scatter(
                    xs, ys, marker=marker, s=size, label=label,
                    color=color, edgecolors="black", linewidths=0.5, alpha=0.85,
                )
        ax.set_xscale("log")
        ax.set_xlabel("compression ratio (higher = more compressed)")
        ax.set_ylabel("perplexity (lower = better)")
        ax.set_title(
            "2D tradeoff: eviction × quantization\n"
            "(○ FP16, ■ INT8, ▲ INT4 — PrimeKV is the only method filling the plane)"
        )
    else:
        raise ValueError(f"unknown mode: {report.mode}")

    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()

    if output_path is not None:
        fig.savefig(output_path, dpi=120)

    return fig
