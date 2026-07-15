#!/usr/bin/env python3
"""
FAST REAL SIGNAL -- does ternary compression hurt a real face embedder?

Post-training quantization (PTQ, no fine-tuning) of a PRETRAINED face model,
measured on real LFW verification pairs. This is the conservative lower bound:
QAT would recover more than PTQ; if ternary-PTQ already holds up, QAT is safe;
if it collapses, that quantifies how much QAT has to recover.

Model : InceptionResnetV1 (facenet-pytorch, pretrained VGGFace2), 28.9M params.
        A real, strong face embedder. NOT MobileFaceNet (the 1M-param edge
        target profiled in mobilefacenet_ternary_test.py) -- this measures the
        "does ternary damage face discrimination" question on an available
        pretrained model; the SIZE story is MobileFaceNet's.
Data  : LFW test pairs via sklearn (figshare mirror), funneled, color.
Quant : this repo's ternary rule (|w|<0.7*mean|w| -> 0) with the standard TWN
        per-layer scale (alpha = mean|w| over kept weights) for a FAIR PTQ --
        the unit-magnitude QAT form needs BatchNorm to re-fit, which PTQ can't
        do. Binary = BWN sign*mean|w|. First conv + final linear kept float
        (standard low-bit practice).

HONESTY: sklearn's funneled LFW is NOT MTCNN-aligned to the model's training,
so ABSOLUTE accuracy sits below the model's true ~99% LFW. The float-vs-quant
DELTA at identical preprocessing is the valid signal, and that's what we read.
"""
import copy
import numpy as np
import torch
import torch.nn.functional as F
from facenet_pytorch import InceptionResnetV1
from sklearn.datasets import fetch_lfw_pairs
from mobilefacenet_ternary_test import score_pairs  # the validated LFW scorer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ternary_scaled(w):
    t = 0.7 * w.abs().mean()
    q = torch.where(w > t, torch.ones_like(w),
        torch.where(w < -t, -torch.ones_like(w), torch.zeros_like(w)))
    nz = q != 0
    alpha = w[nz].abs().mean() if nz.any() else w.new_tensor(1.0)  # TWN scale
    return alpha * q


def binary_scaled(w):
    return w.abs().mean() * torch.sign(w)  # BWN scale


def zero_frac_ternary(w):
    t = 0.7 * w.abs().mean()
    q = torch.where(w.abs() > t, torch.ones_like(w), torch.zeros_like(w))
    return 1.0 - q.mean().item()


def quantize(model, fn, keep_first_last=True):
    m = copy.deepcopy(model)
    layers = [mod for mod in m.modules() if isinstance(mod, (torch.nn.Conv2d, torch.nn.Linear))]
    keep = {id(layers[0]), id(layers[-1])} if keep_first_last else set()
    zfs = []
    with torch.no_grad():
        for mod in layers:
            if id(mod) in keep:
                continue
            if fn is ternary_scaled:
                zfs.append(zero_frac_ternary(mod.weight.data))
            mod.weight.data = fn(mod.weight.data)
    return m, (np.mean(zfs) if zfs else 0.0)


def prep(imgs):
    """(N,H,W,3) in [0,1] -> facenet input: 160x160, (x*255-127.5)/128."""
    x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(160, 160), mode="bilinear", align_corners=False)
    return (x * 255.0 - 127.5) / 128.0


@torch.no_grad()
def embed(model, x, bs=128):
    out = []
    for i in range(0, len(x), bs):
        out.append(model(x[i:i + bs].to(device)).cpu())
    return torch.cat(out)


@torch.no_grad()
def lfw_roc(model, A, B, target):
    model = model.to(device).eval()
    ea = F.normalize(embed(model, A)); eb = F.normalize(embed(model, B))
    sims = (ea * eb).sum(1).numpy()
    model.cpu(); torch.cuda.empty_cache() if device.type == "cuda" else None
    return score_pairs(sims, target)


def main():
    print("=" * 76)
    print("Ternary-PTQ vs float vs binary-PTQ on a REAL face embedder (LFW)")
    print("=" * 76)
    lfw = fetch_lfw_pairs(subset="test", color=True, resize=1.0, funneled=True)
    A = prep(lfw.pairs[:, 0]); B = prep(lfw.pairs[:, 1]); y = lfw.target.astype(int)
    print(f"LFW: {len(y)} pairs ({int(y.sum())} same / {int((y==0).sum())} diff), "
          f"input {tuple(A.shape[1:])}")

    base = InceptionResnetV1(pretrained="vggface2").eval()
    fmb = sum(p.numel() for p in base.parameters()) * 4 / 1e6
    print(f"Model: InceptionResnetV1 (vggface2), {sum(p.numel() for p in base.parameters()):,} "
          f"params, {fmb:.1f} MB float32\n")

    print(f"  {'scheme':<26}{'LFW acc':>12}{'AUC':>8}{'zero%':>8}  dacc vs float")
    print("  " + "-" * 64)
    acc0, std0, auc0 = lfw_roc(base, A, B, y)
    print(f"  {'float32 (baseline)':<26}{acc0:>10.2f}%{auc0:>8.3f}{'-':>8}{'  -':>14}")

    runs = [
        ("ternary  keep 1st/last", ternary_scaled, True),
        ("ternary  ALL layers", ternary_scaled, False),
        ("binary   keep 1st/last", binary_scaled, True),
        ("binary   ALL layers", binary_scaled, False),
    ]
    results = {"float": (acc0, auc0)}
    for name, fn, keep in runs:
        m, zf = quantize(base, fn, keep)
        acc, std, auc = lfw_roc(m, A, B, y)
        results[name] = (acc, auc)
        zt = f"{zf*100:>6.1f}%" if fn is ternary_scaled else f"{'0':>6}%"
        print(f"  {name:<26}{acc:>10.2f}%{auc:>8.3f}{zt:>8}{acc-acc0:>+13.2f}pp")

    # ── read the result ──
    print("\n" + "=" * 76)
    print("READ")
    print("=" * 76)
    t_keep = results["ternary  keep 1st/last"][0]
    b_keep = results["binary   keep 1st/last"][0]
    drop_t = acc0 - t_keep
    print(f"  float {acc0:.1f}%  ->  ternary-PTQ (keep 1st/last) {t_keep:.1f}%  "
          f"(drop {drop_t:+.1f}pp)")
    print(f"  ternary {t_keep:.1f}%  vs  binary {b_keep:.1f}%  "
          f"(d {t_keep-b_keep:+.1f}pp -- does the zero level help under PTQ?)")
    if drop_t < 3:
        print("  => ternary-PTQ barely dents a real face embedder: identity discrimination")
        print("     SURVIVES compression even WITHOUT retraining. QAT would only help more.")
    elif drop_t < 15:
        print("  => ternary-PTQ takes a real but recoverable hit; QAT is the fix (this is a")
        print("     conservative lower bound -- PTQ never fine-tunes to absorb the rounding).")
    else:
        print("  => ternary-PTQ substantially degrades identity discrimination -> PTQ is NOT")
        print("     enough for faces; QAT (fine-tune under quantization) is REQUIRED. Quantified.")
    print("\n[scope] PTQ only (no fine-tune); InceptionResnetV1 not MobileFaceNet; sklearn")
    print("        funneled LFW is not MTCNN-aligned so ABSOLUTE acc is below the model's")
    print("        true ~99% -- the float-vs-quant DELTA at fixed preprocessing is the signal.")


if __name__ == "__main__":
    main()
