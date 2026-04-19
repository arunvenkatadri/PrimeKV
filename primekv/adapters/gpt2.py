"""GPT-2 (and GPT-2-like) cache adapter.

The research trick we use here — so we can plug *any* cache backend
into HuggingFace without monkey-patching attention kernels — is:

1. Run prefill with the model's normal KV cache. Extract the
   ground-truth K/V tensors.
2. Push every (layer, position) K/V into our tested cache. Depending
   on the cache, this may quantize, evict, or fold entries.
3. Reconstruct a ``past_key_values``-shaped tuple by pulling each
   entry back out of the cache. Evicted / missing entries are
   zero-filled (which is the honest quality penalty).
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
# HF cache interop — version-proof K/V extraction
# ---------------------------------------------------------------------------


def _extract_kv_pairs(past: Any) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Extract per-layer ``(K, V)`` tensor pairs from any HF cache format.

    HuggingFace has changed the DynamicCache format multiple times.
    Rather than checking version numbers, we probe the object for
    every known access pattern and use the first one that works.

    Returns a list of ``(K, V)`` where each tensor is shaped
    ``(batch, num_heads, seq_len, head_dim)``.

    Raises ``ValueError`` with a diagnostic message if no known
    format matches.
    """
    if past is None:
        return []

    # --- Plain tuple/list of (K, V) pairs (transformers < 4.36) -------
    if isinstance(past, (tuple, list)) and past:
        first = past[0]
        if isinstance(first, (tuple, list)):
            # Could be 2-tuples or 3-tuples; take first two elements.
            return [(entry[0], entry[1]) for entry in past]
        if isinstance(first, torch.Tensor):
            # Shouldn't happen for past_key_values, but handle it.
            raise ValueError("past appears to be a flat list of tensors")

    # --- DynamicCache with .key_cache / .value_cache (4.36–5.4) -------
    try:
        kc = getattr(past, "key_cache", None)
        vc = getattr(past, "value_cache", None)
        if kc is not None and vc is not None and len(kc) > 0:
            return list(zip(kc, vc))
    except Exception:
        pass

    # --- DynamicCache with .layers (transformers >= 5.5) --------------
    try:
        layers = getattr(past, "layers", None)
        if layers is not None and len(layers) > 0:
            pairs = []
            for layer_obj in layers:
                # Try common attribute names.
                k = getattr(layer_obj, "keys", None)
                if k is None:
                    k = getattr(layer_obj, "key", None)
                v = getattr(layer_obj, "values", None)
                if v is None:
                    v = getattr(layer_obj, "value", None)
                if k is not None and v is not None:
                    pairs.append((k, v))
                else:
                    # Try indexing: layer_obj[0], layer_obj[1].
                    pairs.append((layer_obj[0], layer_obj[1]))
            if pairs:
                return pairs
    except Exception:
        pass

    # --- Iterate the object (handles __iter__ yielding tuples) --------
    try:
        pairs = []
        for item in past:
            if isinstance(item, (tuple, list)) and len(item) >= 2:
                pairs.append((item[0], item[1]))
            elif hasattr(item, "keys") and hasattr(item, "values"):
                pairs.append((item.keys, item.values))
            elif hasattr(item, "key") and hasattr(item, "value"):
                pairs.append((item.key, item.value))
        if pairs:
            return pairs
    except TypeError:
        pass

    # --- to_legacy_cache (some transitional versions) -----------------
    try:
        legacy = past.to_legacy_cache()
        return [(entry[0], entry[1]) for entry in legacy]
    except Exception:
        pass

    # --- __getitem__ fallback -----------------------------------------
    try:
        n = len(past)
        return [(past[i][0], past[i][1]) for i in range(n)]
    except Exception:
        pass

    attrs = [a for a in dir(past) if not a.startswith("_")]
    raise ValueError(
        f"Cannot extract K/V pairs from {type(past).__name__}. "
        f"Public attrs: {attrs}"
    )


def reconstruct_past_kv(
    cache: CacheProtocol,
    kv_pairs: list[tuple[torch.Tensor, torch.Tensor]],
    seq_len: int,
) -> tuple:
    """Rebuild an HF-compatible ``past_key_values`` tuple from ``cache``.

    For each ``(layer, position)`` we look up the cache; misses are
    zero-filled so the output shape always matches the reference.
    Returns a plain tuple of (K, V) pairs which HF models accept as
    legacy cache format.
    """
    rebuilt: list[tuple[torch.Tensor, torch.Tensor]] = []
    for layer, (ref_k, ref_v) in enumerate(kv_pairs):
        new_k = torch.zeros_like(ref_k)
        new_v = torch.zeros_like(ref_v)
        for pos in range(seq_len):
            entry = cache.get(layer, pos)
            if entry is None:
                continue
            k, v = entry
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
    kv_pairs = _extract_kv_pairs(out.past_key_values)

    if hasattr(cache, "classify_prefill"):
        cache.classify_prefill(input_ids=input_ids[0])

    for layer, (k, v) in enumerate(kv_pairs):
        # k, v: (1, num_heads, seq_len, head_dim).
        layer_k = k[0]
        layer_v = v[0]
        for pos in range(seq_len):
            cache.put(layer, pos, layer_k[:, pos, :], layer_v[:, pos, :])
    _sync(device)
    prefill_ms = (time.perf_counter() - t0) * 1000.0

    # --- Reconstruct a lossy past and run decode ----------------------
    lossy_past = reconstruct_past_kv(cache, kv_pairs, seq_len)

    generated = input_ids.clone()
    past = lossy_past
    decode_tokens = max(1, int(workload.decode_tokens))

    _sync(device)
    t1 = time.perf_counter()
    for _ in range(decode_tokens):
        last = generated[:, -1:]
        step = model(input_ids=last, past_key_values=past, use_cache=True)
        # During decode, pass the model's own cache back as-is.
        # Don't convert — the model knows its own format.
        past = step.past_key_values
        next_id = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        generated = torch.cat([generated, next_id], dim=-1)
    _sync(device)
    decode_ms = (time.perf_counter() - t1) * 1000.0
    tokens_per_s = decode_tokens / (decode_ms / 1000.0) if decode_ms > 0 else 0.0

    # --- Perplexity on the full prompt + generated sequence -----------
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
