import numpy as np
import torch
import torch_npu

from quant_cy_npu import QType, quant_dequant_float


np.random.seed(42)
torch.manual_seed(42)

M = 512
N = 512

x = (0.2 * np.random.randn(M, N) + np.random.uniform(-0.03, 0.04, (M, N))).astype(np.float32)
print(x.shape)

qtype_str = "mbsmxfp4"
print("Qtype string: %s " % (qtype_str))
quant_type = QType(qtype_str).dim(1)

def run_case(dtype, label):
    x_torch = torch.from_numpy(x).to(dtype).npu()
    y1 = quant_dequant_float(x_torch.cpu(), quant_type, force_py=True).cpu().to(torch.float32).numpy()
    y2 = quant_dequant_float(x_torch, quant_type, force_py=False).cpu().to(torch.float32).numpy()
    diff = np.abs(y1 - y2)
    max_diff = np.max(diff)
    print("ABS diff max (%s, torch <-> kernel):" % label, max_diff)
    assert max_diff < 1e-4

    print("Testing zero values (%s)" % label)
    y1 = quant_dequant_float(x_torch.cpu() * 0, quant_type, force_py=True).cpu().to(torch.float32).numpy()
    y2 = quant_dequant_float(x_torch * 0, quant_type, force_py=False).cpu().to(torch.float32).numpy()
    diff = np.abs(y1 - y2)
    max_diff = np.max(diff)
    print("ABS diff max (zero values, %s, torch <-> kernel):" % label, max_diff)
    assert max_diff < 1e-4

run_case(torch.float32, "float32")
run_case(torch.bfloat16, "bfloat16")
