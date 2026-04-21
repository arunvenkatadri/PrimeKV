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
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional

import torch

from primekv.baselines import (
    FullCache,
    H2OCache,
    StreamingLLMCache,
    UniformQuantCache,
)
from primekv.cache import PrimeKVCache
from primekv.classifier import BaseClassifier, RuleBasedClassifier, Tier
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
    classifier_factory: Optional[Callable[[], "BaseClassifier"]] = None,
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
    if classifier_factory is not None:
        classifier = classifier_factory()
    else:
        classifier = RuleBasedClassifier(
            anchor_prefix_len=kwargs.get("anchor_prefix_len", 16),
            semantic_stride=kwargs.get("semantic_stride", 3),
        )
    return PrimeKVCache(
        num_layers=num_layers,
        classifier=classifier,
        policies=policies,
        max_entries_per_tier={Tier.SUPPORTING: int(supporting_cap)},
        enable_dynamic_reclassification=kwargs.get(
            "enable_dynamic_reclassification", True
        ),
    )


def _cache_for(name: str, num_layers: int, capacity: int, **kwargs) -> Any:
    if name == "full":
        return FullCache(num_layers)
    if name == "uniform_int8":
        return UniformQuantCache(num_layers, bits=8)
    if name == "uniform_int4":
        return UniformQuantCache(num_layers, bits=4)
    if name == "h2o":
        return H2OCache(num_layers, capacity=int(capacity))
    if name == "streamingllm":
        return StreamingLLMCache(
            num_layers, num_sinks=kwargs.get("num_sinks", 4), window=int(capacity)
        )
    if name == "primekv":
        return _build_primekv(num_layers, supporting_cap=int(capacity), **kwargs)
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
) -> SweepReport:
    """Pareto quality/compression curve for every cache in ``caches``.

    Capacity-driven caches (H2O, StreamingLLM, PrimeKV) get one point
    per capacity value. Fixed-compression caches (full, uniform_int8,
    uniform_int4) get a single point at the smallest capacity value
    (their ratio doesn't depend on capacity).
    """
    caches = caches or list(FIXED_COMPRESSION_CACHES) + list(CAPACITY_CACHES)
    num_layers = _num_layers(model)
    workload = Workload(prompt=prompt, decode_tokens=decode_tokens, max_length=max_length)

    report = SweepReport(
        mode="pareto",
        axis_label="capacity",
        meta={
            "capacities": capacities,
            "decode_tokens": decode_tokens,
            "max_length": max_length,
            "device": device,
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
        },
    )
    full_cache = FullCache(num_layers)

    for length in lengths:
        workload = Workload(
            prompt=prompt,
            decode_tokens=decode_tokens,
            max_length=int(length),
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
    workload = Workload(prompt=prompt, decode_tokens=decode_tokens, max_length=max_length)
    report = SweepReport(
        mode="ablate",
        axis_label=axis,
        meta={
            "capacity": capacity,
            "decode_tokens": decode_tokens,
            "max_length": max_length,
            "values": values,
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
# Caches capable of quantization (varying precision makes sense).
_QUANT_CAPABLE = {"uniform_quant", "primekv"}
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
    primekv_classifier_factory: Optional[Callable[[], "BaseClassifier"]] = None,
) -> SweepReport:
    """Sweep eviction × quantization simultaneously.

    Each cache fills in the cells it's capable of reaching:

    * ``full`` → one cell (no eviction, no quantization)
    * ``uniform_int8`` / ``uniform_int4`` → one cell each (keep all, quantize)
    * ``h2o``, ``streamingllm`` → column (vary eviction, FP16 only)
    * ``primekv`` → full grid (vary both eviction and Tier-2 precision)

    This is the figure that shows PrimeKV's **two-lever** control surface
    vs. single-lever baselines.

    ``precisions`` entries must be in ``{"fp16", "int8", "int4"}``.
    """
    if caches is None:
        caches = ["full", "uniform_int8", "uniform_int4", "h2o", "streamingllm", "primekv"]
    for p in precisions:
        if p not in {"fp16", "int8", "int4"}:
            raise ValueError(f"unknown precision: {p}")

    num_layers = _num_layers(model)
    workload = Workload(prompt=prompt, decode_tokens=decode_tokens, max_length=max_length)
    report = SweepReport(
        mode="2d_tradeoff",
        axis_label="compression_ratio",
        meta={
            "eviction_caps": eviction_caps,
            "precisions": precisions,
            "decode_tokens": decode_tokens,
            "max_length": max_length,
            "device": device,
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
                _emit("h2o", float(cap), "fp16", H2OCache(num_layers, capacity=int(cap)))
            continue
        if cache_name == "streamingllm":
            for cap in eviction_caps:
                _emit(
                    "streamingllm",
                    float(cap),
                    "fp16",
                    StreamingLLMCache(num_layers, num_sinks=4, window=int(cap)),
                )
            continue
        if cache_name == "primekv":
            for cap in eviction_caps:
                for prec in precisions:
                    _emit(
                        "primekv",
                        float(cap),
                        prec,
                        _build_primekv(
                            num_layers,
                            supporting_cap=cap,
                            tier2_precision=prec,
                            classifier_factory=primekv_classifier_factory,
                        ),
                    )
            continue
        raise ValueError(f"unknown cache name: {cache_name}")

    return report


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
