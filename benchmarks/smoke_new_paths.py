"""CPU smoke test for the new sweep harness paths.

Runs miniature versions of everything added in Sections 8 and 9 of the
notebook against GPT-2 on CPU:

  1. sweep_2d_tradeoff with composed baselines (h2o/streaming fill the plane)
  2. sweep_long_context (small grid, GPT-2's 1024-token ceiling)
  3. seeded sweep_pareto + aggregate_reports
  4. run_reasoning_suite with the filtered pass rate
  5. Section 9.1/9.2 diagnostics (attention curve + classifier agreement)
  6. Section 9.3 lossy-prefix prefill viability

This is a bug hunt, not an experiment: GPT-2 at short context tells us
little about the research questions, but it exercises every new code
path end-to-end before anyone burns a GPU session on them.

Usage:
    python benchmarks/smoke_new_paths.py            # real GPT-2 (needs HF access)
    python benchmarks/smoke_new_paths.py --fake     # synthetic model, fully offline

``--fake`` uses the same synthetic HF-shaped model as the unit tests.
It exercises every new code path except the attention-curve diagnostic
(which needs real attention weights), so it's the right mode for
sandboxes without huggingface.co access.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import torch

PROMPT = (
    "The Eiffel Tower is a wrought-iron lattice tower on the Champ de Mars in "
    "Paris, France. It is named after the engineer Gustave Eiffel, whose company "
    "designed and built the tower from 1887 to 1889. Locally nicknamed La dame "
    "de fer, it was constructed as the centrepiece of the 1889 World's Fair and "
    "to crown the centennial anniversary of the French Revolution. Although "
    "initially criticised by some of France's leading artists and intellectuals "
    "for its design, it has since become a global cultural icon of France and "
    "one of the most recognisable structures in the world. The tower is 330 "
    "metres tall, about the same height as an 81-storey building, and is the "
    "tallest structure in Paris. Its base is square, measuring 125 metres on "
    "each side."
)


def banner(msg: str) -> None:
    print(f"\n{'=' * 70}\n{msg}\n{'=' * 70}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fake", action="store_true",
                    help="use the synthetic test model (no HF access needed)")
    args = ap.parse_args()

    t_start = time.time()
    failures: list[str] = []

    if args.fake:
        print("loading synthetic fake model (offline mode) ...")
        from tests.test_adapters import _FakeCausalLM, _FakeTokenizer
        model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=8, vocab=64)
        tok = _FakeTokenizer(vocab=64)
        tok.pad_token = tok.eos_token
    else:
        print("loading gpt2 ...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained("gpt2").eval()
    n_layers = model.config.n_layer

    # ------------------------------------------------------------------ 1
    banner("1. sweep_2d_tradeoff: composed baselines fill the plane")
    try:
        from primekv.sweep import sweep_2d_tradeoff

        report = sweep_2d_tradeoff(
            model=model, tokenizer=tok, prompt=PROMPT,
            eviction_caps=[8, 16], precisions=["fp16", "int4"],
            decode_tokens=4, max_length=96, device="cpu",
        )
        by_cache: dict[str, int] = {}
        for p in report.points:
            by_cache[p.cache] = by_cache.get(p.cache, 0) + 1
        print(f"points per cache: {by_cache}")
        assert by_cache["h2o"] == 4, "h2o should fill 2 caps x 2 precisions"
        assert by_cache["streamingllm"] == 4
        assert by_cache["primekv"] == 4
        h2o_int4 = [p for p in report.points
                    if p.cache == "h2o" and p.extra["precision"] == "int4"]
        h2o_fp16 = [p for p in report.points
                    if p.cache == "h2o" and p.extra["precision"] == "fp16"]
        for q, f in zip(sorted(h2o_int4, key=lambda p: p.extra["cap"]),
                        sorted(h2o_fp16, key=lambda p: p.extra["cap"])):
            assert q.memory_bytes < f.memory_bytes, "int4 cell must be smaller than fp16 cell"
        print("OK: composed h2o int4 cells are smaller than fp16 cells at equal capacity")
    except Exception as e:
        failures.append(f"2d_tradeoff: {e}")
        print(f"FAILED: {e}")

    # ------------------------------------------------------------------ 2
    banner("2. sweep_long_context (GPT-2 ceiling: lengths 128/256/512)")
    try:
        from primekv.sweep import sweep_long_context

        report = sweep_long_context(
            model=model, tokenizer=tok, prompt=PROMPT * 4,
            lengths=[128, 256], capacity_fraction=0.25, min_capacity=8,
            precisions=["int4"],
            caches=["uniform_int4", "h2o_int4", "streamingllm_int4", "primekv"],
            decode_tokens=4, device="cpu",
        )
        assert len(report.points) == 8, f"expected 8 points, got {len(report.points)}"
        for p in report.points:
            assert p.extra["capacity"] == max(8, int(p.sweep_value * 0.25))
            print(f"  {p.cache:18s} len={int(p.sweep_value):4d} cap={p.extra['capacity']:3d} "
                  f"ppl={p.perplexity:.3f} mem={p.memory_bytes/1024:.0f}KB")
        print("OK: capacity scales with length")
    except Exception as e:
        failures.append(f"long_context: {e}")
        print(f"FAILED: {e}")

    # ------------------------------------------------------------------ 3
    banner("3. seeded sweep_pareto + aggregate_reports")
    try:
        from primekv.sweep import sweep_pareto, aggregate_reports

        reports = []
        for s in (0, 1):
            reports.append(sweep_pareto(
                model=model, tokenizer=tok, prompt=PROMPT,
                capacities=[16], caches=["full", "h2o_int4", "primekv"],
                decode_tokens=4, max_length=96, device="cpu",
                seed=s, sample_top_k=50,
            ))
        agg = aggregate_reports(reports)
        assert all(p.extra["n_runs"] == 2 for p in agg.points)
        ppl_stds = [p.extra["perplexity_std"] for p in agg.points]
        print(f"cells: {len(agg.points)}, ppl stds: {[f'{s:.4f}' for s in ppl_stds]}")
        if all(s == 0.0 for s in ppl_stds):
            print("note: all stds zero — sampling may not have diverged on this tiny workload")
        print("OK: aggregation runs, n_runs recorded, stds present")
    except Exception as e:
        failures.append(f"seeded_pareto: {e}")
        print(f"FAILED: {e}")

    # ------------------------------------------------------------------ 4
    banner("4. reasoning suite + filtered pass rate")
    try:
        from primekv.reasoning import default_reasoning_suite, run_reasoning_suite
        from primekv.baselines import FullCache, H2OQuantCache, UniformQuantCache
        from primekv.cache import PrimeKVCache
        from primekv.classifier import RuleBasedClassifier, Tier

        factories = {
            "full": lambda: FullCache(n_layers),
            "uniform_int4": lambda: UniformQuantCache(n_layers, bits=4),
            "h2o_int4": lambda: H2OQuantCache(n_layers, capacity=24, bits=4),
            "primekv": lambda: PrimeKVCache(
                n_layers,
                classifier=RuleBasedClassifier(anchor_prefix_len=8, semantic_stride=3),
                max_entries_per_tier={Tier.SUPPORTING: 24},
            ),
        }
        caches = {k: f() for k, f in factories.items()}
        suite = default_reasoning_suite()
        rep = run_reasoning_suite(
            caches=caches, tests=suite, model=model, tokenizer=tok,
            seeds=[None], device="cpu", cache_factories=factories,
        )
        print(f"{'cache':15s} {'raw':>6s} {'filtered':>9s}")
        summary = rep.summary()
        for cache_name, row in summary.items():
            raw = row["raw_pass_rate"]
            filt = row["filtered_pass_rate"]
            filt_str = f"{filt:.0%}" if filt == filt else "n/a"
            print(f"{cache_name:15s} {raw:>6.0%} {filt_str:>9s}")
        print(f"valid tests (full passes): {summary['full']['n_valid_tests']}/{len(suite)}")
        full_filtered = summary["full"]["filtered_pass_rate"]
        if full_filtered == full_filtered:  # not NaN
            assert full_filtered == 1.0, "full must score 100% on its own filtered metric"
        print("OK: harness runs end-to-end; filtered metric behaves")
    except Exception as e:
        failures.append(f"reasoning: {e}")
        print(f"FAILED: {e}")

    # ------------------------------------------------------------------ 5
    banner("5. diagnostics 9.1/9.2: attention curve + classifier agreement")
    if args.fake:
        print("SKIPPED: needs real attention weights (run without --fake or in Colab)")
    else:
      try:
        ids = tok(PROMPT * 3, return_tensors="pt", truncation=True, max_length=256)["input_ids"]
        seq_len = int(ids.shape[-1])
        with torch.no_grad():
            out = model(input_ids=ids, output_attentions=True, use_cache=False)
        attn_per_pos = torch.zeros(seq_len)
        for layer_attn in out.attentions:
            a = layer_attn[0].float().mean(dim=0)
            col_sums = a.sum(dim=0)
            counts = torch.arange(seq_len, 0, -1, dtype=torch.float32)
            attn_per_pos += col_sums / counts
        attn_per_pos = (attn_per_pos / len(out.attentions)).numpy()

        sink = attn_per_pos[:8].mean()
        middle = attn_per_pos[seq_len // 4: 3 * seq_len // 4].mean()
        recent = attn_per_pos[-32:].mean()
        print(f"sink={sink:.4e}  middle={middle:.4e}  recent={recent:.4e}")
        print(f"sink/middle = {sink/middle:.1f}x   recent/middle = {recent/middle:.1f}x")

        from primekv.classifier import RuleBasedClassifier
        clf = RuleBasedClassifier(anchor_prefix_len=16, semantic_stride=3)
        tiers = clf.classify(input_ids=ids[0]).tiers.numpy()
        tier_names = {0: "ANCHOR", 1: "SEMANTIC", 2: "SUPPORTING", 3: "FILLER"}
        print("mean attention received per tier:")
        means = {}
        for t, name in tier_names.items():
            mask = tiers == t
            if mask.any():
                means[name] = float(attn_per_pos[mask].mean())
                print(f"  {name:11s} {means[name]:.4e}  (n={int(mask.sum())})")
        ordered = means.get("ANCHOR", 0) > means.get("SUPPORTING", float("inf"))
        print(f"classifier/attention agreement (ANCHOR > SUPPORTING): {ordered}")
        print("OK: diagnostics run; numbers above are the preliminary GPT-2 answers")
      except Exception as e:
        failures.append(f"diagnostics: {e}")
        print(f"FAILED: {e}")

    # ------------------------------------------------------------------ 6
    banner("6. diagnostic 9.3: lossy-prefix prefill viability")
    try:
        from primekv.adapters.gpt2 import _extract_kv_pairs, reconstruct_past_kv, _wrap_as_hf_cache
        from primekv.baselines import FullCache, UniformQuantCache, H2OQuantCache

        ids = tok(PROMPT * 3, return_tensors="pt", truncation=True, max_length=384)["input_ids"]
        split = int(ids.shape[-1]) // 2
        prefix_ids, suffix_ids = ids[:, :split], ids[:, split:]
        prefix_len = int(prefix_ids.shape[-1])

        def suffix_ppl(prefix_cache) -> float:
            prefix_cache.reset()
            with torch.no_grad():
                out_pre = model(input_ids=prefix_ids, use_cache=True)
                kv = _extract_kv_pairs(out_pre.past_key_values)
                for layer, (k, v) in enumerate(kv):
                    for pos in range(prefix_len):
                        prefix_cache.put(layer, pos, k[0, :, pos, :], v[0, :, pos, :])
                lossy = reconstruct_past_kv(prefix_cache, kv, prefix_len)
                past = _wrap_as_hf_cache(list(lossy))
                out_suf = model(input_ids=suffix_ids, past_key_values=past,
                                use_cache=False, labels=suffix_ids)
            return float(math.exp(out_suf.loss.item()))

        keep = max(8, prefix_len // 4)
        variants = {
            "full_prefix": FullCache(n_layers),
            "int4_prefix": UniformQuantCache(n_layers, bits=4),
            "h2o_int4_prefix": H2OQuantCache(n_layers, capacity=keep, bits=4),
        }
        results = {name: suffix_ppl(c) for name, c in variants.items()}
        base = results["full_prefix"]
        for name, ppl in results.items():
            print(f"  {name:18s} suffix_ppl={ppl:8.3f}  delta={ppl - base:+.3f}")
        print("OK: lossy-prefix prefill path runs (deltas above are the GPT-2 preview)")
    except Exception as e:
        failures.append(f"lossy_prefill: {e}")
        print(f"FAILED: {e}")

    # ------------------------------------------------------------------ done
    banner(f"SMOKE TEST {'FAILED' if failures else 'PASSED'} "
           f"({time.time() - t_start:.0f}s)")
    for f in failures:
        print(f"  - {f}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
