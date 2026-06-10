"""Tests for ThreeZoneCache: anchor + rolling FP16 + compressed middle."""

import torch

from primekv.baselines import ThreeZoneCache


def test_anchor_zone_is_pinned():
    c = ThreeZoneCache(num_layers=1, num_anchor=4, recent_window=4, middle_bits=4)
    k = torch.randn(2, 8)
    v = torch.randn(2, 8)
    # Insert positions far beyond the recent window.
    for pos in range(20):
        c.put(0, pos, k, v)
    # All anchor positions must still be retrievable, exactly equal (FP16).
    for pos in range(4):
        gk, gv = c.get(0, pos)
        assert gk.shape == k.shape
        # Anchor stays fp16 — values unchanged through quantization.
        assert torch.allclose(gk, k.to(torch.float16))


def test_rolling_window_keeps_last_w_non_anchor():
    c = ThreeZoneCache(num_layers=1, num_anchor=2, recent_window=3, middle_bits=4)
    k = torch.randn(2, 8)
    v = torch.randn(2, 8)
    for pos in range(10):
        c.put(0, pos, k, v)
    # Recent window = 3, max_pos = 9, cutoff = 6 — recent pool keeps {7,8,9}.
    assert 7 in c._recent[0] and 8 in c._recent[0] and 9 in c._recent[0]
    # Older non-anchor positions have moved to middle.
    for pos in (2, 3, 4, 5, 6):
        assert pos in c._middle[0]


def test_middle_storage_is_smaller_than_fp16():
    fp_only = ThreeZoneCache(num_layers=1, num_anchor=2, recent_window=2, middle_bits=None)
    int4 = ThreeZoneCache(num_layers=1, num_anchor=2, recent_window=2, middle_bits=4)
    k = torch.randn(4, 16)
    v = torch.randn(4, 16)
    for pos in range(10):
        fp_only.put(0, pos, k, v)
        int4.put(0, pos, k, v)
    # Middle is the larger zone (8 positions). int4 must compress it.
    assert int4.memory_bytes() < fp_only.memory_bytes()


def test_middle_capacity_evicts_oldest_first():
    c = ThreeZoneCache(num_layers=1, num_anchor=0, recent_window=2,
                       middle_capacity=3, middle_bits=4)
    k = torch.randn(2, 8)
    v = torch.randn(2, 8)
    for pos in range(10):
        c.put(0, pos, k, v)
    # Recent window = 2: keeps {8,9}. Middle cap = 3: keeps the three
    # most-recently-aged-out, which are {5,6,7}. Anything older is gone.
    assert 8 in c._recent[0] and 9 in c._recent[0]
    assert sorted(c._middle[0].keys()) == [5, 6, 7]
    assert c.get(0, 4) is None
    assert c.get(0, 0) is None


def test_reset_clears_all_pools():
    c = ThreeZoneCache(num_layers=2, num_anchor=2, recent_window=2)
    k = torch.randn(2, 8); v = torch.randn(2, 8)
    for pos in range(8):
        c.put(0, pos, k, v)
        c.put(1, pos, k, v)
    c.reset()
    assert c._max_pos == -1
    for layer in range(2):
        assert not c._anchor[layer]
        assert not c._recent[layer]
        assert not c._middle[layer]
    assert c.memory_bytes() == 0
