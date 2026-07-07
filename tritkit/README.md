# tritkit

**A focused, honest ternary-QAT toolkit for tiny edge models.**

Ternarize a small CNN, train it quantization-aware, and export a bit-packed
model that runs multiply-free. `tritkit` does one thing well — `α`-scaled
ternary/binary weights for microcontroller-class vision — and tells you exactly
where that works and where it doesn't.

## Why it's different

Every quantization library sells you the win and buries the boundary. `tritkit`
ships the **ledger** (see Benchmarks) and bakes the boundary into the code:

- **Works:** coarse / redundant tasks on tiny CNNs, trained with QAT. Ternary ≈ float.
- **Fails (and it says so):** fine identity/verification, and PTQ of any kind
  (measured: transformer 95%→50%, face verification AUC 0.997→0.58). PTQ is
  deliberately not a one-liner here.
- **The `α` scale is built into every layer** — a ternary weight is `α·{−1,0,+1}`,
  never raw `{−1,0,+1}` (the missing scale is what silently collapses naive TNNs).

## Install

```bash
pip install ./tritkit      # from the repo, or build a wheel: python -m build tritkit
```

## Quickstart

```python
import tritkit as tk

model = MyTinyCNN()                                     # any small CNN
tk.ternarize(model, keep_first_last=True)               # α-scaled ternary layers
tk.qat_fit(model, train_loader, epochs=20, warmup=4)    # QAT (float warmup → quantized)
tk.profile(model)                                       # params / size / FLOPs / latency
tk.save_packed(model, "model.tt")                       # bit-packed, deployable
```

Reload and run — lossless, and multiply-free with the reference kernel:

```python
fresh = MyTinyCNN(); tk.ternarize(fresh, keep_first_last=True)
tk.load_packed(fresh, "model.tt")                       # reproduces predictions exactly

from tritkit.kernel import run_tiny_cnn
logits, stats = run_tiny_cnn("model.tt", x_numpy)       # inference w/o PyTorch, no float multiplies
# stats -> {mults_avoided, add_sub, skipped, skip_frac}
```

## Benchmarks — the honest ledger

Two entries, produced by the public API (`bench/edge_detect.py`, `bench/classify.py`):

| entry | task | float | ternary | ternary cost | ternary size |
|---|---|---|---|---|---|
| #1 | coarse: "is it a vehicle?" (2-way) | 94.84% | 94.46% | **−0.38pp** | 12.0 KB (12.8×) |
| #2 | finer: CIFAR-10 (10-way) | 76.59% | 74.83% | **+1.76pp** | 23.7 KB (12.6×) |

The gradient is the point: **ternary ≈ float on coarse tasks; the cost grows as
the task gets finer.** And the zero level earns its keep with granularity — on
#1 ternary ties binary; on #2 ternary (−1.8pp) clearly beats binary (−4.8pp).
Pick the alphabet to the task, not by assumption.

## Deployment: real size, and now multiply-free compute

- `save_packed` bit-packs ternary at 5 trits/byte (1.6 bits/weight) + one `α` per
  layer; `load_packed` reproduces predictions **exactly** (100% match, 0 logit diff).
- `tritkit.kernel.run_tiny_cnn` runs the packed model **without PyTorch and without
  float multiplies** in the ternary layers, validated bit-exact against the trained
  model. On the bench model it **eliminates 35.4M float multiplies** and **skips 35%
  of MACs** (zero weights) — the energy story, counted, not asserted.

## Status & roadmap

- **v0.3 (now):** `quant · layers · convert · qat · profile · export · kernel` +
  two benchmarks. α-scaled core, lossless bit-packed export, multiply-free
  reference kernel, pip-installable.
- **v0.3.1:** raw-bytes container (drop `torch.save` overhead → hit the 1.6-bit target).
- **v0.4:** port the reference kernel to C/CUDA/MCU (XNOR-popcount or LUT) — turns
  the counted multiply-free ops into real wall-clock/energy on hardware.
- **later:** threshold/temperature annealing for harder tasks; keyword-spotting benchmark.

## Scope, one more time

Edge / tiny CNNs, coarse tasks, QAT. Not a general "compress any model" library,
and explicitly **not** the LLM/transformer path (that needs from-scratch ternary
training à la BitNet — out of scope here).
