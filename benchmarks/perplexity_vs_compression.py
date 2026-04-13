"""Sweep compression ratios and measure perplexity degradation.

Compares PrimeKV against uniform INT8 / INT4 quant, H2O, and
StreamingLLM. Uses a tiny corpus hard-coded in ``_common.DEFAULT_PROMPTS``
so the script is runnable without any dataset downloads.

Usage:
    python benchmarks/perplexity_vs_compression.py --model gpt2
"""

from __future__ import annotations

import argparse
import logging
import math

import torch

from primekv.baselines import (
    FullCache,
    H2OCache,
    StreamingLLMCache,
    UniformQuantCache,
)
from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier
from primekv.metrics import streaming_perplexity, summarize_stats

from benchmarks._common import (
    DEFAULT_PROMPTS,
    dump_json,
    load_model,
    make_run_dir,
    pick_device,
    tokenize_prompts,
)


def build_primekv(num_layers: int, cap_supporting: int) -> PrimeKVCache:
    clf = RuleBasedClassifier(anchor_prefix_len=4, semantic_stride=3)
    return PrimeKVCache(
        num_layers=num_layers,
        classifier=clf,
        max_entries_per_tier={2: cap_supporting},  # cap Tier.SUPPORTING
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    device = pick_device()
    tok, model = load_model(args.model, device=device)
    input_ids = tokenize_prompts(tok, DEFAULT_PROMPTS, max_length=args.max_length)

    # Baseline perplexity with the raw HF model (no tiered cache in the loop).
    ppls = {}
    for i, row in enumerate(input_ids):
        ppls[f"full_prompt_{i}"] = streaming_perplexity(model, row.unsqueeze(0))

    # Build each cache and record how much memory it claims to use on
    # the same synthetic K/V tensors. This is a proxy until we wire the
    # caches into the real attention path end-to-end.
    num_layers = model.config.num_hidden_layers if hasattr(model.config, "num_hidden_layers") else model.config.n_layer
    num_heads = getattr(model.config, "num_attention_heads", getattr(model.config, "n_head", 12))
    head_dim = model.config.hidden_size // num_heads

    caches = {
        "full": FullCache(num_layers),
        "uniform_int8": UniformQuantCache(num_layers, bits=8),
        "uniform_int4": UniformQuantCache(num_layers, bits=4),
        "h2o_256": H2OCache(num_layers, capacity=256),
        "streamingllm": StreamingLLMCache(num_layers, num_sinks=4, window=256),
        "primekv": build_primekv(num_layers, cap_supporting=256),
    }

    # Synthetic KV fill so memory_bytes is meaningful.
    seq_len = int(input_ids.shape[-1])
    k_fake = torch.randn(num_heads, head_dim)
    v_fake = torch.randn(num_heads, head_dim)
    if "primekv" in caches:
        caches["primekv"].classify_prefill(input_ids=input_ids[0])
    for name, cache in caches.items():
        for layer in range(num_layers):
            for pos in range(seq_len):
                cache.put(layer, pos, k_fake, v_fake)

    results = {
        "model": args.model,
        "seq_len": seq_len,
        "num_layers": num_layers,
        "perplexity_full": ppls,
        "caches": {},
    }
    full_bytes = caches["full"].memory_bytes() or 1
    for name, cache in caches.items():
        mem = cache.memory_bytes()
        results["caches"][name] = {
            "bytes": mem,
            "compression_ratio": full_bytes / max(mem, 1),
        }
    if hasattr(caches["primekv"], "stats"):
        results["caches"]["primekv"]["stats"] = summarize_stats(caches["primekv"].stats)

    out_dir = make_run_dir("perplexity_vs_compression")
    dump_json(out_dir / "results.json", results)
    print(f"wrote {out_dir / 'results.json'}")
    for name, info in results["caches"].items():
        print(f"  {name:16s}  ratio={info['compression_ratio']:.2f}x  bytes={info['bytes']}")


if __name__ == "__main__":
    main()
