# tritkit

**A focused, honest ternary-QAT toolkit for tiny edge models.**

Ternarize a small CNN, train it quantization-aware, and export a bit-packed
model that runs. `tritkit` does one thing well — `α`-scaled ternary/binary
weights for microcontroller-class vision — and tells you exactly where that
works and where it doesn't.

## Why it's different

Every quantization library sells you the win and buries the boundary. `tritkit`
ships the **ledger**:

- **Works:** coarse / redundant tasks on tiny CNNs, trained with QAT. Ternary ≈ float.
- **Fails (and it says so):** fine identity/verification, and PTQ of any kind
  (measured: transformer 95%→50%, face verification AUC 0.997→0.58). PTQ is
  deliberately not a one-liner here.
- **The `α` scale is built into every layer** — a ternary weight is `α·{−1,0,+1}`,
  never raw `{−1,0,+1}` (the missing scale is what silently collapses naive TNNs).

## Quickstart

```python
import tritkit as tk

model  = MyTinyCNN()                                    # any small CNN
tk.ternarize(model, keep_first_last=True)               # α-scaled ternary layers
tk.qat_fit(model, train_loader, epochs=20, warmup=4)    # QAT (float warmup → quantized)
tk.profile(model)                                       # params / size / FLOPs / latency
tk.save_packed(model, "model.tt")                       # bit-packed, deployable
```

Reload and run (lossless):

```python
fresh = MyTinyCNN(); tk.ternarize(fresh, keep_first_last=True)
tk.load_packed(fresh, "model.tt")                       # reproduces predictions exactly
```

## Benchmark #1 — coarse edge detection (CIFAR animal/vehicle)

Produced by the public API (`bench/edge_detect.py`), majority baseline 60%:

| variant | accuracy | size (target) | vs float | latency (CPU) |
|---|---|---|---|---|
| float32 | 94.84% | 153.1 KB | 1.0× | 1.23 ms |
| **ternary** | **94.46%** | 12.0 KB | 12.8× | 1.60 ms |
| binary | 94.37% | 9.3 KB | 16.5× | 0.95 ms |

Ternary holds within **0.38pp** of float at **12.8× smaller**. Honest notes:
on coarse tasks the alphabet barely matters (binary is competitive); and the
ternary *latency* is currently a cost, not a win, because there is **no native
ternary kernel yet** — the export delivers real **disk size** (lossless, 100%
prediction match on reload), not runtime speed.

## Status & roadmap

- **v0.2 (now):** `quant · layers · convert · qat · profile · export` + benchmark #1.
  Correct `α`-scaled core, lossless bit-packed export (8× on-disk today).
- **v0.2.1:** raw-bytes container (drop `torch.save` overhead → hit the 1.6-bit target).
- **v0.3:** native ternary inference kernel (XNOR/popcount or LUT) — turns the
  size win into a *speed* win on real hardware.
- **later:** threshold/temperature annealing for harder tasks; keyword-spotting benchmark.

## Scope, one more time

Edge / tiny CNNs, coarse tasks, QAT. Not a general "compress any model" library,
and explicitly **not** the LLM/transformer path (that needs from-scratch ternary
training à la BitNet — out of scope here).
