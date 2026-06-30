"""
012 Raw Power Benchmark
Measures actual matrix multiply throughput on your GPU (the core op
in all neural network training) and compares against published
datacenter GPU specs.
"""

import torch
import time

device = torch.device("cuda")
name = torch.cuda.get_device_name(0)
print(f"GPU: {name}\n")

def benchmark_matmul(size, dtype, iters=50):
    a = torch.randn(size, size, device=device, dtype=dtype)
    b = torch.randn(size, size, device=device, dtype=dtype)
    torch.cuda.synchronize()

    # warmup
    for _ in range(5):
        c = a @ b
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(iters):
        c = a @ b
    torch.cuda.synchronize()
    dt = time.time() - t0

    flops_per_matmul = 2 * size**3
    total_flops = flops_per_matmul * iters
    tflops = total_flops / dt / 1e12
    return tflops

print("Running matmul benchmark (4096x4096)...\n")

results = {}
for dtype_name, dtype in [("FP32", torch.float32), ("FP16", torch.float16)]:
    tflops = benchmark_matmul(4096, dtype)
    results[dtype_name] = tflops
    print(f"  Your RTX 5060 — {dtype_name}: {tflops:.1f} TFLOPS")

print("\n" + "="*65)
print("  COMPARISON — Raw Compute (TFLOPS)")
print("="*65)

published = {
    "RTX 5060 (yours, measured)" : results["FP16"],
    "RTX 4090 (consumer flagship)": 330,
    "A100 80GB (datacenter)"      : 312,
    "H100 (datacenter, current)"  : 989,
    "8x A100 cluster"             : 312 * 8,
    "200x A100 cluster (MS-scale)": 312 * 200,
}

max_val = max(published.values())
for name, tflops in published.items():
    bar_len = int(tflops / max_val * 40)
    bar = "#" * bar_len
    ratio = tflops / results["FP16"]
    print(f"  {name:<30} {tflops:>8.0f} TFLOPS  {bar}")

print(f"\n  Your 23-minute fine-tune used:  {results['FP16']:.1f} TFLOPS x 23 min")
print(f"  A 200x A100 cluster has:         {312*200:.0f} TFLOPS  ({312*200/results['FP16']:.0f}x more raw power)")
print(f"\n  But MiniLM pretraining needs that scale because it's learning")
print(f"  language from scratch across billions of examples — a task")
print(f"  that genuinely requires weeks of distributed compute.")
print(f"  Fine-tuning is small enough that 1 GPU finishes in minutes")
print(f"  regardless of which consumer card you use.")
