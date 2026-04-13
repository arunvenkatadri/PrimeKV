"""Latency benchmark: TTFT and decode tokens/sec.

This measures the raw HuggingFace model without any cache surgery,
then *adds* PrimeKV's classification + put step at prefill time so we
can measure the overhead PrimeKV would introduce if wired in. It does
not claim to measure actual decode speedups — those require integrating
PrimeKV into the model's attention, which is a later milestone.

Usage:
    python benchmarks/latency.py --model gpt2 --decode-tokens 64
"""

from __future__ import annotations

import argparse

import torch

from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier
from primekv.metrics import LatencyLog, timed

from benchmarks._common import (
    DEFAULT_PROMPTS,
    dump_json,
    load_model,
    make_run_dir,
    pick_device,
    tokenize_prompts,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt2")
    ap.add_argument("--decode-tokens", type=int, default=32)
    ap.add_argument("--max-length", type=int, default=128)
    args = ap.parse_args()

    device = pick_device()
    tok, model = load_model(args.model, device=device)

    input_ids = tokenize_prompts(tok, DEFAULT_PROMPTS[:1], max_length=args.max_length).to(device)

    num_layers = getattr(model.config, "num_hidden_layers", model.config.n_layer)
    num_heads = getattr(model.config, "num_attention_heads", model.config.n_head)
    head_dim = model.config.hidden_size // num_heads

    cache = PrimeKVCache(
        num_layers=num_layers,
        classifier=RuleBasedClassifier(anchor_prefix_len=4, semantic_stride=3),
        device=device,
    )

    latlog = LatencyLog()

    with timed("ttft_prefill", latlog):
        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)

    with timed("primekv_classify_prefill", latlog):
        cache.classify_prefill(input_ids=input_ids[0])

    with timed("primekv_put_prefill", latlog):
        # Simulate putting every prefill position into the cache.
        k = torch.randn(num_heads, head_dim, device=device)
        v = torch.randn(num_heads, head_dim, device=device)
        for layer in range(num_layers):
            for pos in range(int(input_ids.shape[-1])):
                cache.put(layer, pos, k, v)

    # Vanilla decode loop (no cache surgery; this is the "what we start
    # from" number). We just call the model autoregressively.
    generated = input_ids.clone()
    past = out.past_key_values
    with timed("decode_loop", latlog):
        with torch.no_grad():
            for _ in range(args.decode_tokens):
                last = generated[:, -1:]
                step = model(input_ids=last, past_key_values=past, use_cache=True)
                past = step.past_key_values
                next_id = step.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=-1)

    totals = latlog.totals()
    decode_s = totals.get("decode_loop", 0.0)
    tokens_per_s = args.decode_tokens / decode_s if decode_s else 0.0

    results = {
        "model": args.model,
        "device": device,
        "decode_tokens": args.decode_tokens,
        "latency_seconds": totals,
        "tokens_per_second": tokens_per_s,
    }
    out_dir = make_run_dir("latency")
    dump_json(out_dir / "results.json", results)
    print(f"wrote {out_dir / 'results.json'}")
    for k_, v_ in totals.items():
        print(f"  {k_:28s}  {v_*1000:.2f} ms")
    print(f"  tokens/sec                    {tokens_per_s:.2f}")


if __name__ == "__main__":
    main()
