import torch 
from ..QType import QType 
from torch import Tensor 


@torch.no_grad()
def quant_hifx(x: Tensor, Q: QType, qdim: int): 
    # print('HIFxV14')
    # ---- This code is modified from quant_py ----
    # reshape x 
    x = x.unflatten(qdim, (-1, 8, 2, 4))
    special_mask = ~torch.isfinite(x)
    x_finite = torch.where(special_mask, torch.zeros_like(x), x)
    x_unsigned = torch.abs(x_finite)
    sign = torch.sign(x_finite)

    assert Q.exp_bits==0
    # compute initial shared exp 
    max_lv3 = torch.max(x_unsigned, dim=qdim, keepdim=True)[0]
    max_lv2 = torch.max(max_lv3, dim=qdim-1, keepdim=True)[0]
    max_lv1 = torch.max(max_lv2, dim=qdim-2, keepdim=True)[0]
    zero_group = max_lv1 == 0
    
    div7 = torch.ones_like(max_lv1) / 7.0
    div7 = div7.to(torch.bfloat16).to(x.dtype)
    scale_factor = max_lv1 * div7
    safe_scale = torch.where(scale_factor > 0, scale_factor, torch.ones_like(scale_factor))
    # scale_factor = (scale_factor).to(torch.bfloat16).to(x.dtype).clip(min=2 ** (-48), max=49152) 
    ## change to tobf16(rint)
    e_sf = torch.floor(torch.log2(safe_scale))
    mant_sf = safe_scale / 2**e_sf * 2**7
    scale_factor = torch.round(mant_sf) / 2**7 * 2**e_sf

    
    # scale_factor to e6m2
    e_sf = torch.floor(torch.log2(scale_factor))
    scale_factor = torch.round(scale_factor * torch.exp2(2-e_sf)) * torch.exp2(e_sf-2)
    scale_factor = torch.where(zero_group, torch.full_like(scale_factor, 2.0 ** (-48)), scale_factor)
    
    rec_sf = (1.0 / scale_factor).to(torch.bfloat16).to(x.dtype)
    scale_lv2 = (max_lv2 * rec_sf)
    scale_lv2 = torch.exp2((scale_lv2.clip(0, 4) / 4).floor())
    scale_lv3 = torch.exp2(((max_lv3 * rec_sf / scale_lv2).clip(0, 2) / 2).floor())
    
    mant = x_unsigned / scale_lv2 / scale_lv3 * rec_sf
    mant = torch.floor(mant * 2**(Q.man_bits - 1) + 0.5) / 2**(Q.man_bits - 1)
    mant[mant>=2] = 2 - 2**(-Q.man_bits+1)
    
    # out = torch.ones_like(x) * scale_factor 
    out = sign * mant * scale_lv2 * scale_lv3 * scale_factor
    out = torch.where(torch.broadcast_to(zero_group, out.shape), torch.zeros_like(out), out)
    out = torch.where(torch.isnan(x), torch.full_like(out, torch.nan), out)
    out = torch.where(torch.isposinf(x), torch.full_like(out, torch.inf), out)
    out = torch.where(torch.isneginf(x), torch.full_like(out, -torch.inf), out)

    out = out.flatten(qdim-3, qdim)
    return out 
