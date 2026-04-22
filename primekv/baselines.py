"""Baseline KV cache strategies for comparison.

Each baseline implements the same minimal interface:

    class Baseline:
        def put(self, layer, position, k, v): ...
        def get(self, layer, position) -> (k, v) | None: ...
        def memory_bytes(self) -> int: ...
        def reset(self) -> None: ...

This lets benchmarks swap PrimeKV and any baseline behind one loop.

Implemented baselines:

* :class:`FullCache` — vanilla FP16 cache, no compression.
* :class:`H2OCache` — Heavy Hitter Oracle. Evicts by cumulative
  attention once ``capacity`` is exceeded.
* :class:`StreamingLLMCache` — Attention sinks + sliding window.
* :class:`UniformQuantCache` — everything in INT4 or INT8.
* :class:`H2OQuantCache` / :class:`StreamingQuantCache` — the two
  eviction baselines composed with INT8 or INT4 storage on retained
  entries. Use these when benchmarking against PrimeKV's two-lever
  design (eviction × precision) — without them the comparison is
  one-lever-vs-two-lever and unfair.

These are intentionally simplified reference implementations. They are
good enough for quality regressions and ordering, not for reproducing
the exact numbers in the original papers.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Optional, Union

import torch

from primekv.quantize import (
    QuantTensor,
    dequantize_int4,
    dequantize_int8,
    quantize_int4,
    quantize_int8,
)


# ---------------------------------------------------------------------------
# Full cache
# ---------------------------------------------------------------------------


class FullCache:
    """Vanilla FP16 cache — the upper bound on quality, lower bound on savings."""

    def __init__(self, num_layers: int) -> None:
        self.num_layers = num_layers
        self._entries: list[dict[int, tuple[torch.Tensor, torch.Tensor]]] = [
            dict() for _ in range(num_layers)
        ]

    def put(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        self._entries[layer][position] = (
            k.to(torch.float16),
            v.to(torch.float16),
        )

    def get(self, layer: int, position: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return self._entries[layer].get(position)

    def memory_bytes(self) -> int:
        total = 0
        for layer_entries in self._entries:
            for k, v in layer_entries.values():
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total

    def reset(self) -> None:
        self._entries = [dict() for _ in range(self.num_layers)]


# ---------------------------------------------------------------------------
# H2O
# ---------------------------------------------------------------------------


@dataclass
class _H2OEntry:
    k: torch.Tensor
    v: torch.Tensor
    attn_sum: float = 0.0


class H2OCache:
    """Heavy Hitter Oracle.

    Keeps a fixed number of entries per layer. When full, evicts the
    entry with the lowest cumulative attention (``attn_sum``). The caller
    is expected to feed attention back via :meth:`observe_attention`.
    """

    def __init__(self, num_layers: int, capacity: int) -> None:
        self.num_layers = num_layers
        self.capacity = capacity
        self._entries: list[dict[int, _H2OEntry]] = [dict() for _ in range(num_layers)]

    def put(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        layer_map = self._entries[layer]
        layer_map[position] = _H2OEntry(
            k=k.to(torch.float16), v=v.to(torch.float16), attn_sum=0.0
        )
        if len(layer_map) > self.capacity:
            victim = min(layer_map.items(), key=lambda kv: kv[1].attn_sum)[0]
            del layer_map[victim]

    def get(self, layer: int, position: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        entry = self._entries[layer].get(position)
        if entry is None:
            return None
        return entry.k, entry.v

    def observe_attention(self, layer: int, position: int, score: float) -> None:
        entry = self._entries[layer].get(position)
        if entry is not None:
            entry.attn_sum += score

    def memory_bytes(self) -> int:
        total = 0
        for layer_entries in self._entries:
            for entry in layer_entries.values():
                total += entry.k.numel() * entry.k.element_size()
                total += entry.v.numel() * entry.v.element_size()
        return total

    def reset(self) -> None:
        self._entries = [dict() for _ in range(self.num_layers)]


# ---------------------------------------------------------------------------
# StreamingLLM
# ---------------------------------------------------------------------------


class StreamingLLMCache:
    """Attention sinks + sliding window.

    Always keeps the first ``num_sinks`` positions plus the most recent
    ``window`` positions. Everything else is dropped.
    """

    def __init__(self, num_layers: int, num_sinks: int = 4, window: int = 512) -> None:
        self.num_layers = num_layers
        self.num_sinks = num_sinks
        self.window = window
        self._entries: list[OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]] = [
            OrderedDict() for _ in range(num_layers)
        ]

    def put(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        layer_map = self._entries[layer]
        layer_map[position] = (k.to(torch.float16), v.to(torch.float16))
        self._prune(layer_map)

    def _prune(self, layer_map: "OrderedDict[int, tuple[torch.Tensor, torch.Tensor]]") -> None:
        keep_sinks = {p for p in layer_map if p < self.num_sinks}
        non_sink = [p for p in layer_map if p not in keep_sinks]
        if len(non_sink) > self.window:
            to_remove = non_sink[: len(non_sink) - self.window]
            for p in to_remove:
                del layer_map[p]

    def get(self, layer: int, position: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        return self._entries[layer].get(position)

    def memory_bytes(self) -> int:
        total = 0
        for layer_entries in self._entries:
            for k, v in layer_entries.values():
                total += k.numel() * k.element_size()
                total += v.numel() * v.element_size()
        return total

    def reset(self) -> None:
        self._entries = [OrderedDict() for _ in range(self.num_layers)]


# ---------------------------------------------------------------------------
# Uniform quant
# ---------------------------------------------------------------------------


@dataclass
class _QEntry:
    k_q: object
    v_q: object


class UniformQuantCache:
    """Quantize *every* cached entry to INT8 or INT4.

    This is the simplest quantization baseline — no tiering, no policy.
    Good reference point for "what does uniform compression cost?".
    """

    def __init__(self, num_layers: int, bits: int = 4) -> None:
        if bits not in (4, 8):
            raise ValueError("bits must be 4 or 8")
        self.num_layers = num_layers
        self.bits = bits
        self._entries: list[dict[int, _QEntry]] = [dict() for _ in range(num_layers)]

    def put(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        k = k.to(torch.float32)
        v = v.to(torch.float32)
        if self.bits == 8:
            self._entries[layer][position] = _QEntry(
                k_q=quantize_int8(k), v_q=quantize_int8(v)
            )
        else:
            self._entries[layer][position] = _QEntry(
                k_q=quantize_int4(k), v_q=quantize_int4(v)
            )

    def get(self, layer: int, position: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        entry = self._entries[layer].get(position)
        if entry is None:
            return None
        if self.bits == 8:
            return dequantize_int8(entry.k_q).to(torch.float16), dequantize_int8(entry.v_q).to(torch.float16)
        return dequantize_int4(entry.k_q).to(torch.float16), dequantize_int4(entry.v_q).to(torch.float16)

    def memory_bytes(self) -> int:
        total = 0
        for layer_entries in self._entries:
            for entry in layer_entries.values():
                total += entry.k_q.nbytes()
                total += entry.v_q.nbytes()
        return total

    def reset(self) -> None:
        self._entries = [dict() for _ in range(self.num_layers)]


# ---------------------------------------------------------------------------
# Shared helpers for composed (eviction + quant) baselines
# ---------------------------------------------------------------------------


def _store_value(
    k: torch.Tensor, v: torch.Tensor, bits: Optional[int]
) -> Union[tuple[torch.Tensor, torch.Tensor], tuple[QuantTensor, QuantTensor]]:
    """Return either FP16 tensors or a quantized pair, depending on ``bits``."""
    if bits is None:
        return (
            k.to(torch.float16) if k.dtype != torch.float16 else k,
            v.to(torch.float16) if v.dtype != torch.float16 else v,
        )
    if bits == 8:
        return quantize_int8(k.to(torch.float32)), quantize_int8(v.to(torch.float32))
    if bits == 4:
        return quantize_int4(k.to(torch.float32)), quantize_int4(v.to(torch.float32))
    raise ValueError(f"bits must be None, 4, or 8 (got {bits})")


def _load_value(
    k_store, v_store, bits: Optional[int]
) -> tuple[torch.Tensor, torch.Tensor]:
    if bits is None:
        return k_store, v_store
    if bits == 8:
        return dequantize_int8(k_store).to(torch.float16), dequantize_int8(v_store).to(torch.float16)
    if bits == 4:
        return dequantize_int4(k_store).to(torch.float16), dequantize_int4(v_store).to(torch.float16)
    raise ValueError(f"bits must be None, 4, or 8 (got {bits})")


def _storage_bytes(k_store, v_store, bits: Optional[int]) -> int:
    if bits is None:
        return k_store.numel() * k_store.element_size() + v_store.numel() * v_store.element_size()
    return k_store.nbytes() + v_store.nbytes()


# ---------------------------------------------------------------------------
# H2O + quantization (composed baseline)
# ---------------------------------------------------------------------------


@dataclass
class _H2OQEntry:
    k_store: object
    v_store: object
    attn_sum: float = 0.0


class H2OQuantCache:
    """Heavy Hitter Oracle eviction composed with INT8/INT4 storage.

    Drops the lowest-attention entry when over capacity (same policy as
    :class:`H2OCache`) and stores surviving entries at reduced precision.
    Needed for a fair comparison against PrimeKV's two-lever design.
    """

    def __init__(self, num_layers: int, capacity: int, bits: Optional[int] = 4) -> None:
        if bits not in (None, 4, 8):
            raise ValueError("bits must be None, 4, or 8")
        self.num_layers = num_layers
        self.capacity = capacity
        self.bits = bits
        self._entries: list[dict[int, _H2OQEntry]] = [dict() for _ in range(num_layers)]

    def put(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        k_store, v_store = _store_value(k, v, self.bits)
        layer_map = self._entries[layer]
        layer_map[position] = _H2OQEntry(k_store=k_store, v_store=v_store, attn_sum=0.0)
        if len(layer_map) > self.capacity:
            victim = min(layer_map.items(), key=lambda kv: kv[1].attn_sum)[0]
            del layer_map[victim]

    def get(self, layer: int, position: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        entry = self._entries[layer].get(position)
        if entry is None:
            return None
        return _load_value(entry.k_store, entry.v_store, self.bits)

    def observe_attention(self, layer: int, position: int, score: float) -> None:
        entry = self._entries[layer].get(position)
        if entry is not None:
            entry.attn_sum += score

    def memory_bytes(self) -> int:
        total = 0
        for layer_entries in self._entries:
            for entry in layer_entries.values():
                total += _storage_bytes(entry.k_store, entry.v_store, self.bits)
        return total

    def reset(self) -> None:
        self._entries = [dict() for _ in range(self.num_layers)]


# ---------------------------------------------------------------------------
# StreamingLLM + quantization (composed baseline)
# ---------------------------------------------------------------------------


class StreamingQuantCache:
    """StreamingLLM eviction composed with INT8/INT4 storage.

    Keeps the first ``num_sinks`` positions plus the most recent
    ``window`` non-sink positions and quantizes what it keeps.
    """

    def __init__(
        self,
        num_layers: int,
        num_sinks: int = 4,
        window: int = 512,
        bits: Optional[int] = 4,
    ) -> None:
        if bits not in (None, 4, 8):
            raise ValueError("bits must be None, 4, or 8")
        self.num_layers = num_layers
        self.num_sinks = num_sinks
        self.window = window
        self.bits = bits
        self._entries: list[OrderedDict[int, tuple]] = [
            OrderedDict() for _ in range(num_layers)
        ]

    def put(self, layer: int, position: int, k: torch.Tensor, v: torch.Tensor) -> None:
        layer_map = self._entries[layer]
        layer_map[position] = _store_value(k, v, self.bits)
        self._prune(layer_map)

    def _prune(self, layer_map: "OrderedDict[int, tuple]") -> None:
        keep_sinks = {p for p in layer_map if p < self.num_sinks}
        non_sink = [p for p in layer_map if p not in keep_sinks]
        if len(non_sink) > self.window:
            to_remove = non_sink[: len(non_sink) - self.window]
            for p in to_remove:
                del layer_map[p]

    def get(self, layer: int, position: int) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
        entry = self._entries[layer].get(position)
        if entry is None:
            return None
        k_store, v_store = entry
        return _load_value(k_store, v_store, self.bits)

    def memory_bytes(self) -> int:
        total = 0
        for layer_entries in self._entries:
            for k_store, v_store in layer_entries.values():
                total += _storage_bytes(k_store, v_store, self.bits)
        return total

    def reset(self) -> None:
        self._entries = [OrderedDict() for _ in range(self.num_layers)]
