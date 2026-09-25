# PyTorch's scaled_dot_product_attention at the shapes of run.jl, per backend.
import torch
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend

def bench(f, n=30):
    for _ in range(5):
        f()
    torch.cuda.synchronize()
    times = []
    for _ in range(n):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(); f(); end.record(); torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    times.sort()
    return times[len(times) // 2]

for bh, s in ((64, 4096), (32, 8192), (128, 2048)):
    q, k, v = (torch.randn(bh // 16, 16, s, 128, dtype=torch.bfloat16, device="cuda") for _ in range(3))
    flops = 4.0 * bh * s * s * 128
    for backend in (SDPBackend.FLASH_ATTENTION, SDPBackend.CUDNN_ATTENTION):
        try:
            with sdpa_kernel(backend):
                ms = bench(lambda: F.scaled_dot_product_attention(q, k, v))
            print(f"batch·heads {bh}, seq {s}: {backend.name} {flops / ms / 1e9:.1f} TFLOP/s ({ms:.3f} ms)")
        except RuntimeError as e:
            print(f"batch·heads {bh}, seq {s}: {backend.name} unavailable: {str(e)[:80]}")
