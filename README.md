# PrimeKV - Research Project

**Priority-Managed Inference KV Cache**

PrimeKV is a research prototype of a KV cache management system for LLM
inference that classifies cached key-value entries by **structural role**
rather than by attention magnitude, and then applies per-class retention,
precision, and eviction policies.

> **Status:** early research scaffold, public for feedback. Method does not
> yet beat attention-based baselines on modern instruction-tuned models;
> see "Current results" below for the honest state. Interfaces, numbers,
> and the classifier are all expected to change.

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

PrimeKV is inspired by a three-zone context compression architecture from
prior work, applied to the inference-time KV cache.

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

## Current results (honest)

These are preliminary. Everything runs end-to-end but the method has **not** been shown to beat attention-based baselines on a real instruction-tuned model yet.

### GPT-2 (124M), 256-token prompt, aggressive capacity

| cache | compression | perplexity |
|-------|-------------|-----------|
| full | 1.0x | 7.3 |
| uniform_int4 | 4.0x | 7.3 |
| h2o | 20.9x | 17.0 |
| streamingllm | 13.9x | 17.6 |
| **primekv** | **2.5x** | **11.7** |

On GPT-2, PrimeKV beats attention-based eviction (H2O, StreamingLLM) at the quality end of the tradeoff — meaningfully closer to full-cache quality at moderate compression.

### Qwen2.5-3B, 3500-token prompt, 2D sweep across eviction × quantization

All methods land within 0.14 perplexity of each other at up to 40x compression. Modern instruction-tuned models are robust enough that none of the tested compression methods produce meaningful quality differences at this context length. **PrimeKV does not win on this benchmark**; uniform INT4 quantization achieves the lowest perplexity at 4x compression.

This is a real finding, not a failure. It tells us:
1. The rule-based (positional) classifier is not encoding genuine structural role.
2. Method differentiation requires longer contexts (8k+) or weaker models.
3. The paper's core claim is still open — it needs a trained classifier to be fairly evaluated.

## Limitations

- **Rule-based classifier.** The current `RuleBasedClassifier` uses positional heuristics (anchor prefix + semantic stride). This is a placeholder; the method's thesis depends on having a classifier that encodes real structural role (POS-tagging, NER, or a trained MLP head).
- **GPT-2-first validation.** The adapter works on modern models (tested with Qwen2.5-3B) but most sweeps were developed against GPT-2. Longer contexts (8k+) have not been tested.
- **No fair "H2O with pinning" baseline.** A stronger comparison would be H2O with the first N tokens pinned (removing PrimeKV's anchor-pinning advantage). We don't have that baseline yet.
- **Perplexity only.** No downstream task evaluation (LongBench, RULER, etc.).
- **Production tooling integration.** vLLM / TensorRT-LLM integration is explicitly out of scope for v0.

## Planned next steps

- POS/NER-based classifier (spaCy labels → tier assignments)
- Trained MLP classifier with weak labels from teacher attention
- LongBench and RULER evaluation suites
- Fair baselines: H2O+pinning, StreamingLLM+quantization
- 7B+ models at 8k-32k context (where compression actually bites)
- Optional: autoresearch loop for hyperparameter optimization

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

