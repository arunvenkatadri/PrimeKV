import torch

from primekv.baselines import (
    FullCache,
    H2OCache,
    H2OQuantCache,
    StreamingLLMCache,
    StreamingQuantCache,
    UniformQuantCache,
)


def test_full_cache_roundtrip():
    c = FullCache(num_layers=2)
    k = torch.randn(4, 8)
    v = torch.randn(4, 8)
    c.put(0, 0, k, v)
    gk, gv = c.get(0, 0)
    assert gk.shape == k.shape
    assert gv.shape == v.shape
    assert c.memory_bytes() > 0


def test_h2o_evicts_lowest_attention():
    c = H2OCache(num_layers=1, capacity=2)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    for pos in range(3):
        c.put(0, pos, k, v)
        c.observe_attention(0, pos, score=float(pos))  # 0 has lowest
    # Position 0 should be evicted.
    assert c.get(0, 0) is None
    assert c.get(0, 1) is not None
    assert c.get(0, 2) is not None


def test_streamingllm_keeps_sinks_and_window():
    c = StreamingLLMCache(num_layers=1, num_sinks=2, window=2)
    k = torch.randn(2, 4)
    v = torch.randn(2, 4)
    for pos in range(6):
        c.put(0, pos, k, v)
    # Sinks 0, 1 stay, plus last 2 positions (4, 5).
    assert c.get(0, 0) is not None
    assert c.get(0, 1) is not None
    assert c.get(0, 4) is not None
    assert c.get(0, 5) is not None
    assert c.get(0, 2) is None
    assert c.get(0, 3) is None


def test_uniform_int4_smaller_than_int8():
    c8 = UniformQuantCache(num_layers=1, bits=8)
    c4 = UniformQuantCache(num_layers=1, bits=4)
    k = torch.randn(4, 16)
    v = torch.randn(4, 16)
    c8.put(0, 0, k, v)
    c4.put(0, 0, k, v)
    assert c4.memory_bytes() < c8.memory_bytes()


def test_h2o_quant_evicts_lowest_attention_and_dequantizes():
    c = H2OQuantCache(num_layers=1, capacity=2, bits=4)
    k = torch.randn(2, 16)
    v = torch.randn(2, 16)
    for pos in range(3):
        c.put(0, pos, k, v)
        c.observe_attention(0, pos, score=float(pos))
    # Eviction: lowest attention (pos 0) gone; others survive.
    assert c.get(0, 0) is None
    gk, gv = c.get(0, 1)
    assert gk.shape == k.shape and gv.shape == v.shape


def test_h2o_quant_smaller_than_h2o_fp16():
    fp = H2OCache(num_layers=1, capacity=8)
    q4 = H2OQuantCache(num_layers=1, capacity=8, bits=4)
    k = torch.randn(4, 32)
    v = torch.randn(4, 32)
    for pos in range(6):
        fp.put(0, pos, k, v)
        q4.put(0, pos, k, v)
    # Composed baseline must actually compress (int4 ≈ 1/4 of fp16).
    assert q4.memory_bytes() < fp.memory_bytes()


def test_streaming_quant_keeps_sinks_and_window():
    c = StreamingQuantCache(num_layers=1, num_sinks=2, window=2, bits=4)
    k = torch.randn(2, 16)
    v = torch.randn(2, 16)
    for pos in range(6):
        c.put(0, pos, k, v)
    # Sinks (0,1) and last two non-sinks (4,5) survive; 2,3 dropped.
    for p in (0, 1, 4, 5):
        assert c.get(0, p) is not None
    for p in (2, 3):
        assert c.get(0, p) is None
