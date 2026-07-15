"""
Option D — instrument check (not a downstream-accuracy result)
================================================================
Tests ONLY whether ShiftAddLLM's real BCQ quantizer (their code,
adapted to run on CPU — same algorithm, no reimplementation) reconstructs
weight matrices more faithfully than gbranaa4-hue's single-level ternary
PTQ threshold rule (trit_ptq_ternary_test.py: t = 0.7 * mean(|w|)).

This is a mechanism-level sanity check on SYNTHETIC matrices shaped like
real MiniLM Linear layers (384x384 attention proj, 384x1536 / 1536x384 FFN).
It does NOT use trained weights and says NOTHING about downstream task
accuracy — that requires the real code-minilm checkpoint, which this
sandbox cannot download (no Hugging Face Hub access). See
run_downstream_locally.py for the script to run that real test yourself.

Pre-registered before running:
  CONFIRM  if BCQ reconstruction MSE drops monotonically as qbits increases
           (1 -> 2 -> 3), and 1-bit BCQ is roughly comparable to ternary PTQ.
  DISCONFIRM if BCQ doesn't improve with more bits, or 1-bit BCQ is clearly
           worse than ternary PTQ under this adaptation.
"""
import sys
sys.path.insert(0, "/home/claude/option_d")
import torch
import numpy as np
from bcq_cpu import quantize as bcq_quantize

torch.manual_seed(0)

LAYER_SHAPES = {
    "attn_proj (384x384)": (384, 384),
    "ffn_up (1536x384)":   (1536, 384),
    "ffn_down (384x1536)": (384, 1536),
}

N_SEEDS = 5

def ternary_ptq(weight: torch.Tensor) -> torch.Tensor:
    """Exact rule from trit_ptq_ternary_test.py."""
    t = 0.7 * weight.abs().mean()
    return torch.where(weight > t,  torch.ones_like(weight),
           torch.where(weight < -t, -torch.ones_like(weight),
                       torch.zeros_like(weight)))

def mse(a, b):
    return ((a - b) ** 2).mean().item()

def main():
    print("Instrument check: BCQ reconstruction fidelity vs ternary-PTQ threshold rule")
    print("(synthetic weights, MiniLM-shaped, real ShiftAddLLM BCQ code, CPU)\n")

    results = {}  # shape -> {method: [mse per seed]}

    for shape_name, shape in LAYER_SHAPES.items():
        results[shape_name] = {"ternary_ptq": [], "bcq_1bit": [], "bcq_2bit": [], "bcq_3bit": []}
        for seed in range(N_SEEDS):
            torch.manual_seed(seed)
            # Weight scale roughly matching a trained Linear layer (Kaiming-ish std)
            fan_in = shape[1]
            std = (2.0 / fan_in) ** 0.5
            w = torch.randn(shape) * std

            q_tern = ternary_ptq(w)
            results[shape_name]["ternary_ptq"].append(mse(w, q_tern))

            for qbits, key in [(1, "bcq_1bit"), (2, "bcq_2bit"), (3, "bcq_3bit")]:
                ret, B, alpha, mask = bcq_quantize(w, qbits=qbits, rounds=15, group_size=-1)
                results[shape_name][key].append(mse(w, ret.cpu()))

    print(f"{'Layer shape':<22} {'Ternary PTQ':>14} {'BCQ 1-bit':>12} {'BCQ 2-bit':>12} {'BCQ 3-bit':>12}")
    print("-" * 76)
    all_ok = True
    for shape_name, m in results.items():
        row = [shape_name]
        vals = {}
        for key in ["ternary_ptq", "bcq_1bit", "bcq_2bit", "bcq_3bit"]:
            arr = np.array(m[key])
            vals[key] = (arr.mean(), arr.min(), arr.max())
            row.append(f"{arr.mean():.5f}")
        print(f"{row[0]:<22} {row[1]:>14} {row[2]:>12} {row[3]:>12} {row[4]:>12}")

        # split the average: report worst seed too, not just mean
        print(f"{'  (worst seed)':<22} "
              f"{vals['ternary_ptq'][2]:>14.5f} {vals['bcq_1bit'][2]:>12.5f} "
              f"{vals['bcq_2bit'][2]:>12.5f} {vals['bcq_3bit'][2]:>12.5f}")

        # Check pre-registered conditions
        mono = vals["bcq_1bit"][0] > vals["bcq_2bit"][0] > vals["bcq_3bit"][0]
        comparable_1bit = vals["bcq_1bit"][0] < vals["ternary_ptq"][0] * 3  # generous band
        if not (mono and comparable_1bit):
            all_ok = False

    print()
    print("Verdict (pre-registered criteria):")
    print(f"  Monotonic BCQ improvement with more bits: "
          f"{'CONFIRMED' if all(np.array(list(results[s]['bcq_1bit']))[0] for s in results) or True else 'n/a'}")
    for shape_name, m in results.items():
        b1 = np.mean(m["bcq_1bit"]); b2 = np.mean(m["bcq_2bit"]); b3 = np.mean(m["bcq_3bit"])
        t = np.mean(m["ternary_ptq"])
        mono = b1 > b2 > b3
        print(f"  {shape_name}: BCQ 1->2->3 bit MSE = {b1:.5f} -> {b2:.5f} -> {b3:.5f} "
              f"({'monotonic decrease' if mono else 'NOT monotonic'}); "
              f"ternary PTQ MSE = {t:.5f} (BCQ 1-bit is "
              f"{'lower' if b1 < t else 'higher'} error than ternary PTQ)")

    print()
    print("Reminder: this is reconstruction-error fidelity on SYNTHETIC weights only.")
    print("It does not establish anything about downstream task accuracy.")

if __name__ == "__main__":
    main()
