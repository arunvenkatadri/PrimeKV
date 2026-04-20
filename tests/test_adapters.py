"""Tests for primekv/adapters/gpt2.py.

We avoid downloading a real HF model here — the reconstruction logic
is tested against a synthetic ``reference_past`` tuple, and the
end-to-end ``run_with_cache`` path is tested against a tiny fake model
that mimics the HF ``AutoModelForCausalLM`` interface just enough.
"""

from dataclasses import dataclass

import math
import torch

from primekv.adapters.gpt2 import reconstruct_past_kv, run_with_cache
from primekv.baselines import FullCache
from primekv.eval import Workload


# ---------------------------------------------------------------------------
# reconstruct_past_kv
# ---------------------------------------------------------------------------


def _make_reference_past(num_layers: int, num_heads: int, seq_len: int, head_dim: int):
    return tuple(
        (
            torch.randn(1, num_heads, seq_len, head_dim),
            torch.randn(1, num_heads, seq_len, head_dim),
        )
        for _ in range(num_layers)
    )


def test_reconstruct_populates_hit_positions_zeros_misses():
    num_layers, num_heads, seq_len, head_dim = 2, 4, 5, 8
    ref = _make_reference_past(num_layers, num_heads, seq_len, head_dim)

    cache = FullCache(num_layers=num_layers)
    # Only populate positions 1 and 3 — the rest should zero-fill.
    for layer in range(num_layers):
        for pos in (1, 3):
            cache.put(layer, pos, ref[layer][0][0, :, pos, :], ref[layer][1][0, :, pos, :])

    rebuilt = reconstruct_past_kv(cache, ref, seq_len)
    assert len(rebuilt) == num_layers
    for layer in range(num_layers):
        rk, rv = rebuilt[layer]
        assert rk.shape == ref[layer][0].shape
        assert rv.shape == ref[layer][1].shape
        # Hit positions match (up to fp16 rounding).
        for pos in (1, 3):
            assert torch.allclose(
                rk[0, :, pos, :].float(),
                ref[layer][0][0, :, pos, :].float(),
                atol=1e-2,
            )
        # Miss positions are zero.
        for pos in (0, 2, 4):
            assert torch.all(rebuilt[layer][0][0, :, pos, :] == 0)
            assert torch.all(rebuilt[layer][1][0, :, pos, :] == 0)


# ---------------------------------------------------------------------------
# Fake model for run_with_cache
# ---------------------------------------------------------------------------


def _normalize_past(past):
    """Reduce any HF cache format to a list of (K, V) pairs for the fake model."""
    if past is None:
        return None
    if isinstance(past, (tuple, list)):
        return list(past)
    # DynamicCache: try common access patterns.
    if hasattr(past, "key_cache") and hasattr(past, "value_cache"):
        return list(zip(past.key_cache, past.value_cache))
    if hasattr(past, "layers"):
        return [(getattr(l, "keys", None), getattr(l, "values", None)) for l in past.layers]
    if hasattr(past, "to_legacy_cache"):
        return list(past.to_legacy_cache())
    return None


@dataclass
class _FakeConfig:
    n_layer: int = 2
    num_attention_heads: int = 2
    hidden_size: int = 8
    name_or_path: str = "fake-gpt2"


class _FakeOutput:
    def __init__(self, logits, past_key_values, loss=None):
        self.logits = logits
        self.past_key_values = past_key_values
        self.loss = loss


class _FakeCausalLM:
    """Minimal HF-compatible model for smoke-testing the adapter.

    * Supports ``model(input_ids=..., use_cache=True)`` -> past_key_values.
    * Supports ``model(input_ids=..., past_key_values=..., use_cache=True)``.
    * Supports ``model(input_ids=..., labels=...)`` -> loss.
    * The "weights" are deterministic tensors so tests are reproducible.
    """

    def __init__(self, num_layers=2, num_heads=2, head_dim=4, vocab=17):
        self.config = _FakeConfig(
            n_layer=num_layers,
            num_attention_heads=num_heads,
            hidden_size=num_heads * head_dim,
        )
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.vocab = vocab

    def to(self, device):
        return self

    def eval(self):
        return self

    def parameters(self):  # pragma: no cover — not used in these tests
        return iter([])

    def __call__(self, input_ids=None, past_key_values=None, labels=None, use_cache=False):
        batch = input_ids.shape[0]
        seq_len = input_ids.shape[-1]

        # Deterministic "logits" derived from input_ids.
        logits = torch.zeros(batch, seq_len, self.vocab)
        for b in range(batch):
            for t in range(seq_len):
                logits[b, t, int(input_ids[b, t].item()) % self.vocab] = 5.0

        loss = None
        if labels is not None:
            # Simple dummy cross-entropy-ish loss for the perplexity path.
            loss = torch.tensor(0.5)

        past = None
        if use_cache:
            # Normalize past_key_values: may arrive as a tuple OR a
            # DynamicCache wrapper (the adapter wraps when modern
            # transformers is installed).
            prev_pairs = _normalize_past(past_key_values)
            new_past = []
            for layer in range(self.num_layers):
                k_new = torch.randn(batch, self.num_heads, seq_len, self.head_dim)
                v_new = torch.randn(batch, self.num_heads, seq_len, self.head_dim)
                if prev_pairs is not None and len(prev_pairs) > layer:
                    prev_k, prev_v = prev_pairs[layer]
                    k_new = torch.cat([prev_k, k_new], dim=-2)
                    v_new = torch.cat([prev_v, v_new], dim=-2)
                new_past.append((k_new, v_new))
            past = tuple(new_past)

        return _FakeOutput(logits=logits, past_key_values=past, loss=loss)


class _FakeTokenizer:
    def __init__(self, vocab=17):
        self.vocab = vocab
        self.pad_token = None
        self.eos_token = "<eos>"

    def __call__(self, prompt, return_tensors=None, truncation=None, max_length=None):
        # Just use the first min(len(prompt), max_length) characters as ids.
        n = min(len(prompt), max_length or len(prompt))
        ids = torch.tensor([[(ord(c) % self.vocab) for c in prompt[:n]]], dtype=torch.long)
        return {"input_ids": ids}

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr((int(i.item()) % 26) + ord("a")) for i in ids)


def test_run_with_cache_end_to_end_full_cache():
    model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17)
    tok = _FakeTokenizer(vocab=17)
    cache = FullCache(num_layers=2)

    result = run_with_cache(
        "full",
        cache,
        model,
        tok,
        Workload(prompt="hello world", decode_tokens=4, max_length=8),
        device="cpu",
    )
    assert result.name == "full"
    assert result.memory_bytes > 0
    assert result.prefill_ms >= 0
    assert result.decode_ms >= 0
    assert result.perplexity is not None
    assert math.isfinite(result.perplexity)
    assert result.generated is not None
    assert result.tokens_per_second > 0


def test_run_comparison_on_fake_model_produces_ordered_results():
    from primekv.baselines import UniformQuantCache
    from primekv.eval import run_comparison

    model = _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17)
    tok = _FakeTokenizer(vocab=17)
    caches = {
        "full": FullCache(2),
        "uniform_int4": UniformQuantCache(2, bits=4),
    }
    report = run_comparison(
        caches,
        Workload(prompt="hello world", decode_tokens=2, max_length=8),
        model,
        tok,
        device="cpu",
    )
    assert len(report.results) == 2
    names = [r.name for r in report.results]
    assert names == ["full", "uniform_int4"]
    # Compression ratio of "full" relative to itself should be 1.0.
    full_r = next(r for r in report.results if r.name == "full")
    assert full_r.compression_ratio == 1.0
    int4_r = next(r for r in report.results if r.name == "uniform_int4")
    assert int4_r.compression_ratio > 1.0  # int4 is smaller than fp16 full
    md = report.to_markdown()
    assert "full" in md and "uniform_int4" in md
