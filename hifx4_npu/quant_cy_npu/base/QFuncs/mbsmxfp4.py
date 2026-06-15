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
_A2_DIV_FAMILY10_TOWARD_ZERO = frozenset([8, 10, 78, 124])
_A2_DIV_FAMILY10_AWAY_FROM_ZERO = frozenset([232])
_A2_DIV_FAMILY15_TOWARD_ZERO = frozenset([8, 29, 46, 91, 102, 104, 108, 122, 140])


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


def _a2_mbs_div_fp32_sim(prediv: Tensor, factor: Tensor) -> Tensor:
    def _mask_from_codes(codes: Tensor, values) -> Tensor:
        mask = torch.zeros_like(codes, dtype=torch.bool)
        for v in values:
            mask |= codes == int(v)
        return mask

    ref = prediv / factor
    toward_zero = torch.nextafter(ref, torch.zeros_like(ref))
    away_dir = torch.where(ref >= 0, torch.full_like(ref, torch.inf), torch.full_like(ref, -torch.inf))
    away_from_zero = torch.nextafter(ref, away_dir)

    abs_prediv = prediv.abs().contiguous()
    abs_bits = abs_prediv.view(torch.int32)
    mantissa = abs_bits & 0x007FFFFF
    is_zero = abs_prediv == 0
    family15 = (~is_zero) & (mantissa == 0x00400000)
    family10 = (~is_zero) & (mantissa == 0x00000000)

    factor_k = torch.round((factor.to(torch.float64) - 1.0) * 256.0).to(torch.int64)

    out = ref
    family10_toward = family10 & _mask_from_codes(factor_k, _A2_DIV_FAMILY10_TOWARD_ZERO)
    family10_away = family10 & _mask_from_codes(factor_k, _A2_DIV_FAMILY10_AWAY_FROM_ZERO)
    family15_toward = family15 & _mask_from_codes(factor_k, _A2_DIV_FAMILY15_TOWARD_ZERO)

    out = torch.where(family10_toward | family15_toward, toward_zero, out)
    out = torch.where(family10_away, away_from_zero, out)
    return out


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
    out = _a2_mbs_div_fp32_sim(q, factor)
    return out.reshape(x_shape)
