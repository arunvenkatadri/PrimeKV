"""Measure actual GPU (or CPU estimated) memory at various sequence lengths.

For each sequence length, we fill every cache baseline with a synthetic
KV tensor per (layer, position) and record ``memory_bytes`` plus
``torch.cuda.memory_allocated`` when on GPU. The resulting JSON is easy
to plot.

Usage:
    python benchmarks/memory_usage.py --model gpt2 --seq-lens 512 1024 2048
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from primekv.baselines import (
    FullCache,
    H2OCache,
    StreamingLLMCache,
    UniformQuantCache,
)
from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier
from primekv.metrics import gpu_memory_snapshot, reset_peak_memory

from benchmarks._common import dump_json, load_model, make_run_dir, pick_device


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[256, 512, 1024])
    args = ap.parse_args()

    device = pick_device()
    tok, model = load_model(args.model, device=device)
    num_layers = getattr(model.config, "num_hidden_layers", model.config.n_layer)
    num_heads = getattr(model.config, "num_attention_heads", model.config.n_head)
    head_dim = model.config.hidden_size // num_heads

    results = {"model": args.model, "device": device, "num_layers": num_layers, "per_seq": {}}

    for seq_len in args.seq_lens:
        reset_peak_memory()
        caches = {
            "full": FullCache(num_layers),
            "uniform_int8": UniformQuantCache(num_layers, bits=8),
            "uniform_int4": UniformQuantCache(num_layers, bits=4),
            "h2o": H2OCache(num_layers, capacity=seq_len // 2),
            "streamingllm": StreamingLLMCache(num_layers, num_sinks=4, window=seq_len // 2),
            "primekv": PrimeKVCache(
                num_layers=num_layers,
                classifier=RuleBasedClassifier(anchor_prefix_len=4, semantic_stride=3),
                max_entries_per_tier={2: seq_len // 4},
            ),
        }
        caches["primekv"].classify_prefill(
            input_ids=torch.zeros(seq_len, dtype=torch.long)
        )

        k_fake = torch.randn(num_heads, head_dim, device=device)
        v_fake = torch.randn(num_heads, head_dim, device=device)

        per_cache: dict[str, dict] = {}
        for name, cache in caches.items():
            for layer in range(num_layers):
                for pos in range(seq_len):
                    cache.put(layer, pos, k_fake, v_fake)
            per_cache[name] = {
                "cache_bytes": cache.memory_bytes(),
                "gpu": gpu_memory_snapshot(),
            }
        results["per_seq"][str(seq_len)] = per_cache

    out_dir = make_run_dir("memory_usage")
    dump_json(out_dir / "results.json", results)
    print(f"wrote {out_dir / 'results.json'}")
    for seq_len, entries in results["per_seq"].items():
        print(f"seq_len={seq_len}")
        for name, info in entries.items():
            print(f"  {name:14s}  cache={info['cache_bytes']}")


if __name__ == "__main__":
    main()
