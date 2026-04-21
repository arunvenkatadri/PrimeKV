# Autoresearch program

You are an agent optimizing a PrimeKV tiered KV cache configuration.

## Your task

Edit `autoresearch/experiment.py` to produce a `build_candidate(num_layers,
device) -> PrimeKVCache` that scores higher than the current best.

## What "score" means

The evaluator in `autoresearch/prepare.py` computes:

```
score = reasoning_pass_rate - lambda_ppl * perplexity_delta
```

- `reasoning_pass_rate`: fraction of `primekv.reasoning_eval.default_tests()`
  passed by your candidate (constraint persistence, entity recall, fact
  chains). **This is the dominant signal.**
- `perplexity_delta`: `ppl_candidate - ppl_full` on one reference prompt.
  Positive means your candidate is worse than an uncompressed cache.
- `lambda_ppl`: 0.01 by default.

A candidate that keeps every constraint but has mediocre perplexity still
wins. A candidate that ships a tiny cache but drops constraints loses.

## Levers you can pull

In `experiment.py`, you may:

- Change `TierPolicy` per tier: `precision` ∈ {fp16, int8, int4},
  `pinned`, `evictable`, `summary`.
- Change `max_entries_per_tier` — the per-tier LRU cap. `None` is unlimited.
- Swap the classifier:
  - `RuleBasedClassifier(anchor_prefix_len, semantic_stride, ...)`
  - `SpaCyClassifier(...)` — requires spacy installed + a model downloaded.
    Prefers text-aware classification; falls back to rule-based otherwise.
  - Build a custom `BaseClassifier` subclass inline if you like.
- Toggle `enable_dynamic_reclassification`.
- Adjust `promotion_threshold` / `demotion_threshold` (on the `PrimeKVCache`
  constructor).

## Rules

- Keep the function signature: `build_candidate(num_layers: int, device: str = "cpu") -> PrimeKVCache`.
- Return a `PrimeKVCache`, not a baseline.
- Only import from `primekv.*` and stdlib.
- No file I/O, no network calls, no randomness — the evaluator must be
  deterministic for fair comparison.
- Stay under ~80 lines of code in `build_candidate`; readability matters.

## Respond with

Reply with the FULL contents of the new `experiment.py`. No fences, no
commentary — just the file.
