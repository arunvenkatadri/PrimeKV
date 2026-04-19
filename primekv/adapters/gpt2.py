"""GPT-2 (and GPT-2-like) cache adapter.

The research trick we use here — so we can plug *any* cache backend
into HuggingFace without monkey-patching attention kernels — is:

1. Run prefill with the model's normal KV cache. This gives us the
   "ground truth" ``past_key_values`` tuple of shape
   ``(num_layers, 2, batch, num_heads, seq_len, head_dim)``.
2. Push every (layer, position) K/V into our tested cache. Depending
   on the cache, this may quantize, evict, or fold entries.
3. Reconstruct a ``past_key_values``-shaped tuple by pulling each
   entry back out of the cache. Evicted / missing entries are
   zero-filled (which is the honest quality penalty — other methods
   are welcome to do something smarter).
4. Run the decode loop with the reconstructed, lossy past.
5. Measure perplexity by running the full model over the concatenation
   of prompt + generated tokens.

This is explicitly *not* fast. It touches every position in a Python
loop and runs two forward passes. It's designed to give us honest
comparison numbers across cache backends — not to be a serving
system.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Optional

import torch

from primekv.eval import CacheProtocol, CacheResult, Workload

log = logging.getLogger("primekv.adapters.gpt2")


# ---------------------------------------------------------------------------
# HF cache interop
# ---------------------------------------------------------------------------


def _to_legacy_tuple(past: Any) -> tuple:
    """Normalize HF's past_key_values to a tuple of ``(K, V)`` pairs.

    HuggingFace has changed the DynamicCache format across versions:

    * **transformers < 4.36**: plain ``tuple[tuple[K, V], ...]``.
    * **transformers 4.36–5.4**: ``DynamicCache`` with ``.key_cache``
      and ``.value_cache`` list attributes.
    * **transformers >= 5.5**: ``DynamicCache`` with ``.layers``; iterating
      yields 3-tuples ``(keys, values, sliding_window)``.

    We handle all three and always return ``tuple[tuple[K, V], ...]``.
    """
    if past is None:
        return tuple()

    # transformers >= 5.5: .layers[i].keys / .layers[i].values
    if hasattr(past, "layers"):
        return tuple((layer.keys, layer.values) for layer in past.layers)

    # transformers 4.36–5.4: .key_cache / .value_cache lists
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return tuple(zip(past.key_cache, past.value_cache))

    # Older explicit conversion method.
    if hasattr(past, "to_legacy_cache"):
        return past.to_legacy_cache()

    # Already a tuple / list — but entries might be 3-tuples from
    # DynamicCache.__iter__. Strip the third element if present.
    if past and isinstance(past[0], (tuple, list)) and len(past[0]) > 2:
        return tuple((entry[0], entry[1]) for entry in past)

    return past


def reconstruct_past_kv(
    cache: CacheProtocol,
    reference_past: tuple,
    seq_len: int,
) -> tuple:
    """Rebuild an HF-shaped ``past_key_values`` tuple from ``cache``.

    For each ``(layer, position)`` we look up the cache; misses are
    zero-filled so the output shape always matches ``reference_past``.
    This lets the downstream model consume our (lossy) past directly
    without knowing anything about PrimeKV.

    Args:
        cache: any :class:`CacheProtocol` implementation.
        reference_past: the ``past_key_values`` the model produced at
            prefill. Used only for shape / dtype / device.
        seq_len: prompt length in tokens.
    """
    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, (ref_k, ref_v) in enumerate(reference_past):
        # ref shapes: (batch, num_heads, seq_len, head_dim).
        new_k = torch.zeros_like(ref_k)
        new_v = torch.zeros_like(ref_v)
        for pos in range(seq_len):
            entry = cache.get(layer, pos)
            if entry is None:
                continue
            k, v = entry
            # k, v shapes from the cache: (num_heads, head_dim).
            new_k[0, :, pos, :] = k.to(new_k.dtype).to(new_k.device)
            new_v[0, :, pos, :] = v.to(new_v.dtype).to(new_v.device)
        rebuilt.append((new_k, new_v))
    return tuple(rebuilt)


# ---------------------------------------------------------------------------
# End-to-end runner
# ---------------------------------------------------------------------------


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


@torch.no_grad()
def run_with_cache(
    name: str,
    cache: CacheProtocol,
    model,
    tokenizer,
    workload: Workload,
    device: str = "cpu",
) -> CacheResult:
    """Run one workload end-to-end through ``cache`` and return metrics."""
    enc = tokenizer(
        workload.prompt,
        return_tensors="pt",
        truncation=True,
        max_length=workload.max_length,
    )
    input_ids = enc["input_ids"].to(device)
    seq_len = int(input_ids.shape[-1])

    # --- Prefill (full HF cache, then mirror into our cache) ----------
    _sync(device)
    t0 = time.perf_counter()
    out = model(input_ids=input_ids, use_cache=True)
    reference_past = _to_legacy_tuple(out.past_key_values)

    if hasattr(cache, "classify_prefill"):
        # PrimeKV needs the classifier to run before puts so tiers are known.
        cache.classify_prefill(input_ids=input_ids[0])

    for layer, (k, v) in enumerate(reference_past):
        # k, v: (1, num_heads, seq_len, head_dim).
        layer_k = k[0]
        layer_v = v[0]
        for pos in range(seq_len):
            cache.put(layer, pos, layer_k[:, pos, :], layer_v[:, pos, :])
    _sync(device)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    # --- Reconstruct a lossy past and run decode ----------------------
    lossy_past = reconstruct_past_kv(cache, reference_past, seq_len)

    generated = input_ids.clone()
    past = lossy_past
    decode_tokens = max(1, int(workload.decode_tokens))

    _sync(device)
    t1 = time.perf_counter()
    for _ in range(decode_tokens):
        last = generated[:, -1:]
        step = model(input_ids=last, past_key_values=past, use_cache=True)
        past = _to_legacy_tuple(step.past_key_values)
        next_id = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=-1)
    _sync(device)
    decode_ms = (time.perf_counter() - t1) * 1000.0
    tokens_per_s = decode_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0

    # --- Perplexity on the full prompt + generated sequence -----------
    # Cheap, honest quality signal: how plausible is the continuation
    # under the actual model? A cache that produces garbage decodes
    # will be penalized here.
    try:
        with torch.no_grad():
            ppl_out = model(input_ids=generated, labels=generated)
        ppl = float(math.exp(ppl_out.loss.item()))
    except Exception as e:  # pragma: no cover
        log.warning("ppl compute failed for cache=%s: %s", name, e)
        ppl = None

    # --- Extras (per-tier stats for PrimeKV, etc.) --------------------
    extra: dict = {}
    if hasattr(cache, "stats"):
        from primekv.metrics import summarize_stats

        extra["stats"] = summarize_stats(cache.stats)
    if hasattr(cache, "positions_in_tier"):
        from primekv.classifier import Tier

        extra["tier_distribution"] = {
            t.name: len(cache.positions_in_tier(t)) for t in Tier
        }

    try:
        decoded = tokenizer.decode(generated[0], skip_special_tokens=True)
    except Exception:  # pragma: no cover
        decoded = None

    return CacheResult(
        name=name,
        memory_bytes=int(cache.memory_bytes()),
        compression_ratio=1.0,  # filled in by run_comparison()
        perplexity=ppl,
        prefill_ms=prefill_ms,
        decode_ms=decode_ms,
        tokens_per_second=tokens_per_s,
        generated=decoded,
        extra=extra,
    )
