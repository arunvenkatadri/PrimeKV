"""Unified cache comparison CLI.

Runs every selected cache through the same prompt + decode loop and
prints a markdown table of metrics. The underlying work happens in
:func:`primekv.eval.run_comparison`.

Usage:
    python benchmarks/compare.py --model gpt2 --decode-tokens 32
    python benchmarks/compare.py --caches full primekv --prompt "Hello world"
"""

from __future__ import annotations

import argparse
import logging

from primekv.baselines import (
    FullCache,
    H2OCache,
    StreamingLLMCache,
    UniformQuantCache,
)
from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier, Tier
from primekv.eval import Workload, run_comparison

from benchmarks._common import dump_json, load_model, make_run_dir, pick_device


CACHE_FACTORIES = {
    "full": lambda nl, _args: FullCache(nl),
    "uniform_int8": lambda nl, _args: UniformQuantCache(nl, bits=8),
    "uniform_int4": lambda nl, _args: UniformQuantCache(nl, bits=4),
    "h2o": lambda nl, args: H2OCache(nl, capacity=args.h2o_capacity),
    "streamingllm": lambda nl, args: StreamingLLMCache(
        nl, num_sinks=args.stream_sinks, window=args.stream_window
    ),
    "primekv": lambda nl, args: PrimeKVCache(
        num_layers=nl,
        classifier=RuleBasedClassifier(
            anchor_prefix_len=args.primekv_anchor,
            semantic_stride=args.primekv_stride,
        ),
        max_entries_per_tier={Tier.SUPPORTING: args.primekv_supporting_cap},
    ),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare KV cache strategies on one workload.")
    ap.add_argument("--model", default="gpt2")
    ap.add_argument(
        "--prompt",
        default="The quick brown fox jumps over the lazy dog. " * 4,
        help="Prompt text to run through every cache.",
    )
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    ap.add_argument(
        "--caches",
        nargs="+",
        default=list(CACHE_FACTORIES.keys()),
        choices=list(CACHE_FACTORIES.keys()),
    )
    ap.add_argument("--verbose", action="store_true")

    # Per-baseline knobs.
    ap.add_argument("--h2o-capacity", type=int, default=256)
    ap.add_argument("--stream-sinks", type=int, default=4)
    ap.add_argument("--stream-window", type=int, default=256)
    ap.add_argument("--primekv-anchor", type=int, default=4)
    ap.add_argument("--primekv-stride", type=int, default=3)
    ap.add_argument("--primekv-supporting-cap", type=int, default=256)

    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)
    device = pick_device()
    tok, model = load_model(args.model, device=device)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    num_layers = getattr(
        model.config,
        "num_hidden_layers",
        getattr(model.config, "n_layer", 12),
    )

    caches = {
        name: CACHE_FACTORIES[name](num_layers, args) for name in args.caches
    }
    workload = Workload(
        prompt=args.prompt,
        decode_tokens=args.decode_tokens,
        max_length=args.max_length,
    )

    report = run_comparison(caches, workload, model, tok, device=device)

    print(report.to_markdown())

    out_dir = make_run_dir("compare")
    dump_json(out_dir / "report.json", report.to_dict())
    (out_dir / "report.md").write_text(report.to_markdown())
    print(f"\nwrote {out_dir}/report.{{json,md}}")


if __name__ == "__main__":
    main()
