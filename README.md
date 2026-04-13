# PrimeKV

**Priority-Managed Inference KV Cache**

PrimeKV is a research prototype of a KV cache management system for LLM
inference that classifies cached key-value entries by **structural role**
rather than by attention magnitude, and then applies per-class retention,
precision, and eviction policies.

> Status: early research scaffold. Interfaces and numbers are unstable; do not
> use this for anything that needs to work.

## The core idea

Existing KV cache management methods (H2O, StreamingLLM, Quest, ...) treat
importance as a scalar derived from historical attention weights. PrimeKV
treats importance as a **classification problem** — tokens belong to
functional tiers with distinct policies:

| Tier | Name       | Examples                                      | Storage policy                                   |
|------|------------|-----------------------------------------------|--------------------------------------------------|
| 0    | Anchor     | System prompt, attention sinks, format tokens | Pinned in HBM at FP16. Never evicted.            |
| 1    | Semantic   | Entities, facts, constraints, reasoning       | HBM at FP16 or INT8. LRU eviction under pressure.|
| 2    | Supporting | Elaboration, examples, transitions            | INT4 in HBM or CPU-offloaded, prefetch on demand.|
| 3    | Filler     | Articles, connectives, boilerplate            | Evicted, or merged into a learned summary vector.|

Classification happens at **prefill time**, using a lightweight classifier
head over the model's own embeddings / early-layer hidden states. The key
insight: because tier membership is determined by structural role (not just
cumulative attention), PrimeKV can compress far more aggressively than
reactive, attention-score-driven methods. A constraint token mentioned once
early in the prompt may have low cumulative attention but is structurally
critical, and PrimeKV can keep it around.

PrimeKV is inspired by the three-zone context compression architecture in
[Clawboss], applied to the inference-time KV cache.

```
         prefill tokens
              |
              v
      +-----------------+
      |   Classifier    |  (rule-based or learned MLP head)
      +-----------------+
              |
    +---------+---------+---------+---------+
    |         |         |         |         |
    v         v         v         v         v
  Tier 0    Tier 1    Tier 2    Tier 3    (summary)
  pinned    HBM/INT8  INT4/CPU  evicted
   FP16                offload
```

## Repository layout

```
primekv/          core package
  classifier.py   tier classifier (rule-based + MLP)
  cache.py        tiered KV cache manager
  attention.py    tier-aware attention
  quantize.py     INT8 / INT4 K and V (de)quantization
  metrics.py      perplexity, hit rate, mem, latency
  baselines.py    full / H2O / StreamingLLM / uniform quant
tests/            unit + smoke tests
benchmarks/       perplexity / memory / latency / classifier_overhead
paper/            outline and notes
```

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Requires PyTorch and `transformers`. We start with GPT-2 (124M) for fast
iteration; interfaces are model-agnostic so you can swap in Llama / Mistral
later.

## Quick start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from primekv.cache import PrimeKVCache
from primekv.classifier import RuleBasedClassifier

tok = AutoTokenizer.from_pretrained("gpt2")
model = AutoModelForCausalLM.from_pretrained("gpt2")

clf = RuleBasedClassifier(anchor_prefix_len=16)
cache = PrimeKVCache(num_layers=model.config.n_layer, classifier=clf)

# See benchmarks/ for full integration examples.
```

## Running benchmarks

Individual scripts (each writes to `runs/<timestamp>-<name>/`):

```bash
python benchmarks/perplexity_vs_compression.py --model gpt2
python benchmarks/memory_usage.py --model gpt2 --seq-lens 512 1024 2048
python benchmarks/latency.py --model gpt2
python benchmarks/classifier_overhead.py --model gpt2
```

Unified comparison driver (runs every cache through the same prompt
and prints a markdown table):

```bash
python benchmarks/compare.py --model gpt2 --decode-tokens 32
python benchmarks/compare.py --caches full primekv --prompt "Hello."
```

## Web interface

A Gradio app is included for interactive experimentation:

```bash
pip install -e ".[web]"
python webui/app.py             # http://127.0.0.1:7860
```

Enter a prompt, pick which caches to compare, tune per-backend
hyperparameters, and hit **Run**. The UI shows the comparison table,
PrimeKV's tier distribution, and sample decodes from each cache.

## Testing

All unit tests run on CPU — no GPU required. GPT-2 (124M) is used as
the default smoke model for benchmarks and works fine on CPU.

```bash
pytest tests/                   # 33 tests, ~3s on CPU
```

The adapter is tested against a synthetic HF-shaped fake model so no
network access is required to run the suite.

## Design principles

- **Ablatable everything.** Tiers, classifier, per-tier policies, and
  reclassification can all be toggled independently from a single config.
- **Logging everywhere.** Hits, misses, evictions, promotions, and demotions
  are all recorded for post-hoc analysis.
- **Start simple.** The first classifier is literally rule-based (positions
  `[0, N)` are Tier 0, etc.) before we train anything.
- **No production frameworks yet.** Vanilla PyTorch + HuggingFace. vLLM /
  TensorRT-LLM integration is explicitly out of scope for v0.

## Citing

See `paper/outline.md` for the current paper draft outline.

[Clawboss]: #  "internal: Clawboss context compression architecture"
