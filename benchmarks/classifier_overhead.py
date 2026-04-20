"""How much does tier classification add to prefill time?

Sweeps sequence lengths and times:

1. ``RuleBasedClassifier.classify`` — the current default.
2. ``MLPClassifier.classify`` — a learned head over random hidden states.

Usage:
    python benchmarks/classifier_overhead.py --model gpt2
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

from primekv.classifier import MLPClassifier, RuleBasedClassifier

from benchmarks._common import dump_json, load_model, make_run_dir, pick_device


def time_it(fn, iters: int = 20) -> float:
    # Warmup.
    for _ in range(3):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return (time.perf_counter() - start) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--seq-lens", type=int, nargs="+", default=[128, 512, 1024, 2048])
    args = ap.parse_args()

    device = pick_device()
    tok, model = load_model(args.model, device=device)
    d_model = model.config.hidden_size

    rule = RuleBasedClassifier(anchor_prefix_len=4, semantic_stride=3)
    mlp = MLPClassifier(d_model=d_model).to(device).eval()

    results = {"model": args.model, "device": device, "d_model": d_model, "per_seq": {}}

    for seq_len in args.seq_lens:
        input_ids = torch.zeros(seq_len, dtype=torch.long, device=device)
        hidden = torch.randn(seq_len, d_model, device=device)

        rule_s = time_it(lambda: rule.classify(input_ids=input_ids))
        with torch.no_grad():
            mlp_s = time_it(lambda: mlp.classify(hidden_states=hidden))

        results["per_seq"][str(seq_len)] = {
            "rule_seconds": rule_s,
            "mlp_seconds": mlp_s,
        }
        print(f"seq_len={seq_len:5d}  rule={rule_s*1000:.3f} ms  mlp={mlp_s*1000:.3f} ms")

    out_dir = make_run_dir("classifier_overhead")
    dump_json(out_dir / "results.json", results)
    print(f"wrote {out_dir / 'results.json'}")


if __name__ == "__main__":
    main()
