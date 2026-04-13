"""KV cache (de)quantization utilities.

Supports INT8 and INT4 with either absmax (per-tensor) or per-channel
scales along the head dimension. Quantization is applied to K and V
tensors shaped ``(num_heads, seq_len, head_dim)`` or
``(num_tokens, head_dim)``.

These routines are intentionally simple and slow — we're measuring
quality, not kernel throughput. Swap in a CUDA kernel later if needed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class QuantTensor:
    """Quantized tensor plus the metadata needed to dequantize it.

    ``data`` is stored as ``int8`` (INT4 is packed two-per-byte into
    ``int8`` storage — see :func:`quantize_int4`).
    """

    data: torch.Tensor
    scale: torch.Tensor
    zero_point: torch.Tensor
    bits: int
    orig_shape: Tuple[int, ...]
    per_channel_dim: int | None

    def nbytes(self) -> int:
        # Quantized payload bytes only (ignores scale/zp bookkeeping).
        # For INT4 ``data`` is already the packed int8 buffer, so its
        # ``numel`` is the byte count.
        if self.bits in (4, 8):
            return self.data.numel()
        raise ValueError(f"unknown bits: {self.bits}")


def _compute_scale(
    x: torch.Tensor, qmax: int, per_channel_dim: int | None
) -> Tuple[torch.Tensor, torch.Tensor]:
    if per_channel_dim is None:
        amax = x.abs().amax().clamp_min(1e-8)
        scale = amax / qmax
        zp = torch.zeros((), dtype=x.dtype, device=x.device)
        return scale, zp

    # Collapse everything except ``per_channel_dim`` and take absmax.
    dims = [d for d in range(x.dim()) if d != per_channel_dim]
    amax = x.abs().amax(dim=dims, keepdim=True).clamp_min(1e-8)
    scale = amax / qmax
    zp = torch.zeros_like(scale)
    return scale, zp


def quantize_int8(
    x: torch.Tensor, per_channel_dim: int | None = -1
) -> QuantTensor:
    """Symmetric absmax INT8 quantization.

    Args:
        x: FloatTensor.
        per_channel_dim: If given, compute one scale per slice along this
            dimension. Defaults to the last dim (e.g. ``head_dim``).
    """
    if per_channel_dim is not None and per_channel_dim < 0:
        per_channel_dim = x.dim() + per_channel_dim
    scale, zp = _compute_scale(x, qmax=127, per_channel_dim=per_channel_dim)
    q = torch.round(x / scale).clamp(-128, 127).to(torch.int8)
    return QuantTensor(
        data=q,
        scale=scale.to(torch.float32),
        zero_point=zp.to(torch.float32),
        bits=8,
        orig_shape=tuple(x.shape),
        per_channel_dim=per_channel_dim,
    )


def dequantize_int8(q: QuantTensor) -> torch.Tensor:
    if q.bits != 8:
        raise ValueError("quant tensor is not int8")
    return q.data.to(q.scale.dtype) * q.scale


def quantize_int4(
    x: torch.Tensor, per_channel_dim: int | None = -1
) -> QuantTensor:
    """Symmetric absmax INT4 quantization with two-per-byte packing.

    Packing layout: two INT4 values per ``int8`` byte, low nibble first.
    The last dim of the stored tensor is halved (rounded up).
    """
    if per_channel_dim is not None and per_channel_dim < 0:
        per_channel_dim = x.dim() + per_channel_dim
    scale, zp = _compute_scale(x, qmax=7, per_channel_dim=per_channel_dim)
    q = torch.round(x / scale).clamp(-8, 7).to(torch.int8)

    # Pack the last dimension.
    flat = q.reshape(-1)
    if flat.numel() % 2 == 1:
        flat = torch.cat([flat, torch.zeros(1, dtype=torch.int8, device=flat.device)])
    low = flat[0::2] & 0x0F
    high = flat[1::2] & 0x0F
    packed = (low | (high << 4)).to(torch.int8)

    return QuantTensor(
        data=packed,
        scale=scale.to(torch.float32),
        zero_point=zp.to(torch.float32),
        bits=4,
        orig_shape=tuple(x.shape),
        per_channel_dim=per_channel_dim,
    )


def dequantize_int4(q: QuantTensor) -> torch.Tensor:
    if q.bits != 4:
        raise ValueError("quant tensor is not int4")
    packed = q.data
    low = (packed & 0x0F).to(torch.int8)
    high = ((packed >> 4) & 0x0F).to(torch.int8)
    # Sign extend 4-bit values.
    low = torch.where(low >= 8, low - 16, low)
    high = torch.where(high >= 8, high - 16, high)

    unpacked = torch.empty(low.numel() * 2, dtype=torch.int8, device=packed.device)
    unpacked[0::2] = low
    unpacked[1::2] = high

    numel = 1
    for d in q.orig_shape:
        numel *= d
    unpacked = unpacked[:numel].reshape(q.orig_shape)
    return unpacked.to(q.scale.dtype) * q.scale
