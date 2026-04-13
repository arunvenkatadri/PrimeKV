import torch

from primekv.quantize import (
    dequantize_int4,
    dequantize_int8,
    quantize_int4,
    quantize_int8,
)


def test_int8_roundtrip_shape_and_close():
    x = torch.randn(4, 8, 16)
    q = quantize_int8(x)
    assert q.bits == 8
    assert q.data.dtype == torch.int8
    assert q.data.shape == x.shape
    y = dequantize_int8(q)
    assert y.shape == x.shape
    # Per-channel absmax INT8 should be tight.
    assert (y - x).abs().mean().item() < 0.05


def test_int4_roundtrip_shape_and_close():
    x = torch.randn(4, 8, 16)
    q = quantize_int4(x)
    assert q.bits == 4
    y = dequantize_int4(q)
    assert y.shape == x.shape
    # INT4 is lossier — allow more slack.
    assert (y - x).abs().mean().item() < 0.3


def test_int4_packed_size():
    x = torch.randn(2, 32)  # 64 values -> 32 bytes packed
    q = quantize_int4(x)
    assert q.nbytes() == 32


def test_per_tensor_quant_runs():
    x = torch.randn(8, 16)
    q8 = quantize_int8(x, per_channel_dim=None)
    assert dequantize_int8(q8).shape == x.shape
    q4 = quantize_int4(x, per_channel_dim=None)
    assert dequantize_int4(q4).shape == x.shape
