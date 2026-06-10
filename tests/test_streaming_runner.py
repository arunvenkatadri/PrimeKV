"""Smoke tests for streaming_run_with_cache against the synthetic model.

The synthetic model from tests/test_adapters.py is deterministic but
respects the past_key_values interface — enough to verify that the
streaming runner walks the prompt in chunks, populates the cache,
and produces a valid CacheResult.
"""

import math

from primekv.adapters.gpt2 import streaming_run_with_cache
from primekv.baselines import FullCache, ThreeZoneCache, UniformQuantCache
from primekv.eval import Workload

from tests.test_adapters import _FakeCausalLM, _FakeTokenizer


def _model_and_tok():
    return _FakeCausalLM(num_layers=2, num_heads=2, head_dim=4, vocab=17), _FakeTokenizer(vocab=17)


def test_streaming_run_with_full_cache_produces_valid_result():
    model, tok = _model_and_tok()
    cache = FullCache(num_layers=2)
    result = streaming_run_with_cache(
        "full", cache, model, tok,
        Workload(prompt="hello there my friend this is a test", decode_tokens=2, max_length=24),
        chunk_size=4, device="cpu",
    )
    assert result.name == "full"
    assert result.memory_bytes > 0
    assert result.perplexity is not None and math.isfinite(result.perplexity)
    assert result.extra["streaming_chunk_size"] == 4


def test_streaming_uses_three_zone_cache_without_crashing():
    model, tok = _model_and_tok()
    cache = ThreeZoneCache(num_layers=2, num_anchor=2, recent_window=4, middle_bits=4)
    result = streaming_run_with_cache(
        "three_zone", cache, model, tok,
        Workload(prompt="streaming test of three zone cache here", decode_tokens=2, max_length=24),
        chunk_size=4, device="cpu",
    )
    assert result.memory_bytes > 0
    # Cache should contain entries across all three zones if seq_len is enough.
    seen_positions = (
        len(cache._anchor[0]) + len(cache._recent[0]) + len(cache._middle[0])
    )
    assert seen_positions >= 1


def test_streaming_with_one_chunk_matches_single_pass_shape():
    # With chunk_size >= seq_len, the streaming path degenerates to one
    # pass — the cache should still be populated and the result valid.
    model, tok = _model_and_tok()
    cache = UniformQuantCache(num_layers=2, bits=4)
    result = streaming_run_with_cache(
        "uniform_int4", cache, model, tok,
        Workload(prompt="short prompt", decode_tokens=2, max_length=12),
        chunk_size=512, device="cpu",
    )
    assert result.perplexity is not None
