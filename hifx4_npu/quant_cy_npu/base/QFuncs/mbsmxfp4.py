import torch
from torch import Tensor

from ..QType import QType


FP32_EXP_MASK = 0x7F800000
FP32_MANTISSA_TOP8_MASK = 0x007F8000
FP32_MANTISSA_TOP8_SHIFT = 15
E2M1_MAX = 6.0
MACRO_GROUP = 128
INNER_GROUP = 32
INNER_PER_MACRO = 4
E8M0_MIN_VALUE = 2.0 ** -127


def _e0m8_macro_factor(amax: Tensor) -> Tensor:
    safe = torch.where(amax > 0, amax, torch.ones_like(amax)).to(torch.float32)
    recip = (E2M1_MAX / safe).contiguous()
    bits = recip.view(torch.int32)
    top8 = (bits & FP32_MANTISSA_TOP8_MASK) >> FP32_MANTISSA_TOP8_SHIFT
    factor = 1.0 + top8.to(torch.float32) / 256.0
    return torch.where(amax > 0, factor, torch.ones_like(factor))


def _floor_e8m0_scale(x: Tensor) -> Tensor:
    safe = torch.clamp(x.to(torch.float32), min=E8M0_MIN_VALUE).contiguous()
    bits = safe.view(torch.int32)
    lower = (bits & FP32_EXP_MASK).view(torch.float32)
    return torch.maximum(lower, torch.full_like(lower, E8M0_MIN_VALUE))


def _quantize_e2m1(x: Tensor) -> Tensor:
    abs_x = x.abs().clamp(max=E2M1_MAX)
    quant_abs = torch.zeros_like(abs_x)
    quant_abs = torch.where(abs_x >= 0.25, torch.full_like(abs_x, 0.5), quant_abs)
    quant_abs = torch.where(abs_x >= 0.75, torch.full_like(abs_x, 1.0), quant_abs)
    quant_abs = torch.where(abs_x >= 1.25, torch.full_like(abs_x, 1.5), quant_abs)
    quant_abs = torch.where(abs_x >= 1.75, torch.full_like(abs_x, 2.0), quant_abs)
    quant_abs = torch.where(abs_x >= 2.5, torch.full_like(abs_x, 3.0), quant_abs)
    quant_abs = torch.where(abs_x >= 3.5, torch.full_like(abs_x, 4.0), quant_abs)
    quant_abs = torch.where(abs_x >= 5.0, torch.full_like(abs_x, 6.0), quant_abs)
    return quant_abs * torch.sign(x)


@torch.no_grad()
def quant_mbsmxfp4(x: Tensor, Q: QType, qdim: int) -> Tensor:
    del Q
    x_shape = x.shape
    x_grouped = x.unflatten(qdim, (-1, INNER_PER_MACRO, INNER_GROUP))

    amax128 = x_grouped.abs().amax(dim=(-1, -2), keepdim=True)
    factor = _e0m8_macro_factor(amax128)
    normalized = x_grouped * factor

    amax32 = normalized.abs().amax(dim=-1, keepdim=True)
    scale = _floor_e8m0_scale(amax32 / E2M1_MAX)
    q = _quantize_e2m1(normalized / scale) * scale
    out = q / factor
    return out.reshape(x_shape)
