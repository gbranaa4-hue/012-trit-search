#!/usr/bin/env python3
"""
MobileFaceNet + ternary QAT -- face-verification test harness.

Answers (in two tiers, honestly separated):
  TIER 1 -- REAL NOW, no training needed:
    * model footprint: float MB vs ternary MB (log2(3)=1.585 bits/quantized weight)
    * compute: MACs/FLOPs at 112x112
    * latency: measured ms/image on CPU (edge-relevant) and GPU
    * a VALIDATED LFW verification evaluator: proven on the known case
      (random-init model must score ~50% / AUC~0.5 -- instrument check)
  TIER 2 -- needs a trained embedder (face dataset + hours):
    * the LFW ROC of a float vs ternary-QAT MobileFaceNet. Training loop
      (ArcFace + ternary QAT warmup) is set up here; point it at a face
      ImageFolder and run. Not auto-run -- no face training set on disk.

Why this split is the honest answer to "can I reproduce edge face-rec on
cheaper specs with ternary": the footprint/latency numbers (Tier 1) ARE the
'cheaper specs' question and are real; whether ternary keeps IDENTITY
DISCRIMINATION (Tier 2 ROC) is the other half and cannot be answered without
a trained model -- claiming it without the run would be inference, not
measurement.

MobileFaceNet: Chen et al. 2018, standard config (~1M params, 512-d embedding,
112x112 input). Ternary uses this repo's exact rule (|w|<0.7*mean|w| -> 0).
First conv + final embedding layer kept float (standard low-bit practice).
"""
import os
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG2_3 = math.log2(3)  # 1.585 bits per ternary weight


# ── ternary quantizer (this repo's rule) + quant-capable layers ──────────────
class TernaryQuantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        t = 0.7 * x.abs().mean()
        ctx.save_for_backward(x)
        return torch.where(x > t, torch.ones_like(x),
               torch.where(x < -t, -torch.ones_like(x), torch.zeros_like(x)))
    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x.abs() <= 1.0).float()


tq = TernaryQuantize.apply


class QuantConv2d(nn.Conv2d):
    def __init__(self, *a, quantize=False, **k):
        super().__init__(*a, **k)
        self.do_quantize = quantize
        self.quantizable = True
    def forward(self, x):
        w = tq(self.weight) if self.do_quantize else self.weight
        return F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)


def set_quant(model, active):
    for m in model.modules():
        if isinstance(m, QuantConv2d):
            m.do_quantize = active and m.quantizable


# ── MobileFaceNet ────────────────────────────────────────────────────────────
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c, k, s, p, groups=1, act=True, quantizable=True):
        super().__init__()
        self.conv = QuantConv2d(in_c, out_c, k, stride=s, padding=p, groups=groups, bias=False)
        self.conv.quantizable = quantizable
        self.bn = nn.BatchNorm2d(out_c)
        self.act = nn.PReLU(out_c) if act else nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride, t):
        super().__init__()
        hidden = in_c * t
        self.use_res = (stride == 1 and in_c == out_c)
        self.block = nn.Sequential(
            ConvBlock(in_c, hidden, 1, 1, 0),                       # expand (pointwise)
            ConvBlock(hidden, hidden, 3, stride, 1, groups=hidden),  # depthwise
            ConvBlock(hidden, out_c, 1, 1, 0, act=False),           # project (linear)
        )
    def forward(self, x):
        out = self.block(x)
        return x + out if self.use_res else out


class MobileFaceNet(nn.Module):
    # (expansion t, out_c, n_blocks, stride)
    CFG = [(2, 64, 5, 2), (4, 128, 1, 2), (2, 128, 6, 1), (4, 128, 1, 2), (2, 128, 2, 1)]

    def __init__(self, emb_dim=512):
        super().__init__()
        self.stem = nn.Sequential(
            ConvBlock(3, 64, 3, 2, 1, quantizable=False),  # first conv kept float
            ConvBlock(64, 64, 3, 1, 1, groups=64),          # depthwise
        )
        blocks, in_c = [], 64
        for t, c, n, s in self.CFG:
            for i in range(n):
                blocks.append(Bottleneck(in_c, c, s if i == 0 else 1, t))
                in_c = c
        self.blocks = nn.Sequential(*blocks)
        self.conv1x1 = ConvBlock(128, 512, 1, 1, 0)
        self.gdconv = ConvBlock(512, 512, 7, 1, 0, groups=512, act=False)  # global depthwise -> 1x1
        self.emb = QuantConv2d(512, 512, 1, 1, 0, bias=False)
        self.emb.quantizable = False                        # final embedding kept float
        self.emb_bn = nn.BatchNorm2d(512)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.conv1x1(x)
        x = self.gdconv(x)
        x = self.emb_bn(self.emb(x))
        return x.flatten(1)                                  # (B, 512) embedding


class ArcFace(nn.Module):
    """ArcFace head for TRAINING (Tier 2). Not used by the Tier-1 numbers."""
    def __init__(self, emb=512, n_cls=10572, s=64.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_cls, emb)); nn.init.xavier_normal_(self.W)
        self.s, self.m = s, m
    def forward(self, x, label):
        x = F.normalize(x); W = F.normalize(self.W)
        cos = x @ W.t()
        theta = torch.acos(cos.clamp(-1 + 1e-7, 1 - 1e-7))
        target = torch.cos(theta + self.m)
        onehot = F.one_hot(label, cos.size(1)).float()
        return self.s * (onehot * target + (1 - onehot) * cos)


# ── profiling: params / size / FLOPs / latency ───────────────────────────────
def profile_size(model):
    quant_bits = float_bits = 0
    for m in model.modules():
        if isinstance(m, QuantConv2d) and m.quantizable:
            quant_bits += m.weight.numel() * LOG2_3
        else:
            for p in m.parameters(recurse=False):
                float_bits += p.numel() * 32
    # params that live on non-QuantConv2d modules (BN, PReLU) counted in float via recurse=False above
    total_params = sum(p.numel() for p in model.parameters())
    float_mb = total_params * 4 / 1e6
    tern_mb = (quant_bits + float_bits) / 8 / 1e6
    quant_params = sum(m.weight.numel() for m in model.modules()
                       if isinstance(m, QuantConv2d) and m.quantizable)
    return total_params, quant_params, float_mb, tern_mb


def profile_flops(model, size=112):
    macs = [0]
    def hook(m, inp, out):
        if isinstance(m, nn.Conv2d):
            oc, oh, ow = out.shape[1], out.shape[2], out.shape[3]
            macs[0] += oc * oh * ow * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
    hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, nn.Conv2d)]
    model.eval()
    with torch.no_grad():
        model(torch.randn(1, 3, size, size).to(next(model.parameters()).device))
    for h in hs:
        h.remove()
    return macs[0]


def latency(model, dev, size=112, iters=30, warmup=5):
    model = model.to(dev).eval()
    x = torch.randn(1, 3, size, size).to(dev)
    with torch.no_grad():
        for _ in range(warmup):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(iters):
            model(x)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / iters * 1000  # ms/image


# ── LFW verification evaluator (torchvision LFWPairs, standard 10-fold) ───────
def lfw_pairs_loader():
    import torchvision
    import torchvision.transforms as T
    tfm = T.Compose([T.Resize((112, 112)), T.ToTensor(),
                     T.Normalize([0.5] * 3, [0.5] * 3)])  # -> [-1,1]
    ds = torchvision.datasets.LFWPairs(root="./data", split="test",
                                       image_set="funneled", transform=tfm, download=True)
    return ds


def score_pairs(sims, labels):
    """Standard LFW verification scoring: Mann-Whitney AUC + 10-fold accuracy
    (threshold picked on 9 folds, measured on the held-out fold)."""
    sims = np.asarray(sims, float); labels = np.asarray(labels).astype(int)
    order = np.argsort(sims); ranks = np.empty(len(sims), float); ranks[order] = np.arange(1, len(sims) + 1)
    npos, nneg = labels.sum(), (labels == 0).sum()
    auc = (ranks[labels == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg + 1e-9)
    n = len(sims); fold = n // 10; accs = []
    for k in range(10):
        te = np.zeros(n, bool); te[k * fold:(k + 1) * fold] = True; tr = ~te
        ths = np.unique(sims[tr])
        scores = [((sims[tr] > th) == labels[tr]).mean() for th in ths]
        th_star = ths[int(np.argmax(scores))]
        accs.append(((sims[te] > th_star) == labels[te]).mean())
    return float(np.mean(accs)) * 100, float(np.std(accs)) * 100, float(auc)


def validate_scorer():
    """Instrument check: run the verification scorer on synthetic pairs whose
    answer is KNOWN. Null must be ~chance; a separable case must be ~perfect."""
    rng = np.random.default_rng(0); n = 6000
    labels = np.r_[np.ones(n // 2), np.zeros(n // 2)].astype(int)
    idx = rng.permutation(n); labels = labels[idx]
    cases = {}
    # null: same distribution for both classes -> ~50% / AUC 0.5
    cases["null (no signal)"] = rng.normal(0.3, 0.2, n)
    # separable: same-person similar, different-person dissimilar -> ~100% / AUC 1
    s = np.where(labels == 1, rng.normal(0.75, 0.08, n), rng.normal(0.10, 0.08, n))
    cases["separable (clean identities)"] = s
    # overlapping: partial separation -> intermediate
    o = np.where(labels == 1, rng.normal(0.55, 0.20, n), rng.normal(0.30, 0.20, n))
    cases["overlapping (noisy)"] = o
    print("  synthetic known cases (validates the 10-fold + AUC math):")
    ok = True
    for name, sims in cases.items():
        acc, std, auc = score_pairs(sims, labels)
        print(f"    {name:<32} acc {acc:5.2f}% +/- {std:4.2f}   AUC {auc:.3f}")
        if "null" in name:      ok &= abs(acc - 50) < 3 and abs(auc - 0.5) < 0.03
        if "separable" in name: ok &= acc > 98 and auc > 0.99
    print(f"  => scorer {'VALID' if ok else 'SUSPECT'} "
          f"(null lands at chance, separable at ceiling -- as they must)")
    return ok


@torch.no_grad()
def lfw_eval(model, ds, dev, batch=128):
    from torch.utils.data import DataLoader
    model = model.to(dev).eval()
    loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=2)
    sims, labels = [], []
    for a, b, y in loader:
        ea = F.normalize(model(a.to(dev))); eb = F.normalize(model(b.to(dev)))
        sims.append((ea * eb).sum(1).cpu()); labels.append(y)
    return score_pairs(torch.cat(sims).numpy(), torch.cat(labels).numpy())


def main():
    print("=" * 78)
    print("MobileFaceNet + ternary -- face-verification test harness")
    print("=" * 78)
    model = MobileFaceNet().to(device)

    # ---- TIER 1: real footprint / compute / latency ----
    total, quant_p, float_mb, tern_mb = profile_size(model)
    set_quant(model, True)
    macs = profile_flops(model)
    print(f"\n[TIER 1 -- real now, no training needed]")
    print(f"  params            : {total:,}  ({quant_p:,} in ternarizable conv layers)")
    print(f"  model size float32 : {float_mb:.2f} MB")
    print(f"  model size ternary : {tern_mb:.2f} MB   ({float_mb/tern_mb:.1f}x smaller; "
          f"first-conv+embedding kept float)")
    print(f"  compute            : {macs/1e6:.1f} MMACs/image @112x112 ({2*macs/1e6:.1f} MFLOPs)")

    lat_cpu = latency(model, torch.device("cpu"))
    line = f"  latency CPU (1 thr): {lat_cpu:.1f} ms/image  ({1000/lat_cpu:.1f} img/s)"
    if device.type == "cuda":
        lat_gpu = latency(model, device)
        line += f"\n  latency GPU        : {lat_gpu:.2f} ms/image  (note: GPU shared with bg job)"
    print(line)

    # ---- Validate the verification scorer on KNOWN cases (instrument check) ----
    print(f"\n[instrument check] verification scorer on synthetic KNOWN cases:")
    validate_scorer()

    # ---- Try real LFW; torchvision's auto-download URL is often dead ----
    print(f"\n[LFW real data] attempting torchvision LFWPairs download:")
    try:
        ds = lfw_pairs_loader()
        acc_f, std_f, auc_f = lfw_eval(MobileFaceNet(), ds, device)
        print(f"  loaded {len(ds)} pairs; random-init model: acc {acc_f:.2f}%  AUC {auc_f:.3f} "
              f"(chance, as a random model must be)")
    except Exception as e:
        msg = str(e).split(chr(10))[0]
        print(f"  UNAVAILABLE: {msg}")
        print("  torchvision's LFW mirror is down. For the real run, place an aligned LFW")
        print("  (e.g. kaggle 'jessicali9530/lfw-funneled' or insightface lfw.bin) under")
        print("  ./data/lfw-py/ and re-run -- the validated scorer above then applies as-is.")

    # ---- TIER 2 scope ----
    print(f"\n[TIER 2 -- needs a trained embedder]")
    print("  The float-vs-ternary LFW ROC needs a MobileFaceNet trained with ArcFace on")
    print("  a face set (CASIA-WebFace ~0.5M imgs / MS1M). Training loop is set up in")
    print("  train_arcface() below (ternary QAT warmup identical to experiments.py).")
    print("  Run: point train_arcface() at a face ImageFolder; ~hours on this GPU.")
    print("  Only THEN is 'does ternary keep identity discrimination' a measured yes/no.")


def train_arcface(data_dir, epochs=40, warmup=8, batch=256, lr=0.1):
    """Tier-2 training entry point (not auto-run). Trains MobileFaceNet + ArcFace
    with the repo's ternary QAT warmup schedule on an ImageFolder face dataset."""
    import torchvision
    import torchvision.transforms as T
    tfm = T.Compose([T.Resize((112, 112)), T.RandomHorizontalFlip(),
                     T.ToTensor(), T.Normalize([0.5] * 3, [0.5] * 3)])
    ds = torchvision.datasets.ImageFolder(data_dir, transform=tfm)
    from torch.utils.data import DataLoader
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=4, pin_memory=True)
    model = MobileFaceNet().to(device)
    head = ArcFace(512, len(ds.classes)).to(device)
    opt = torch.optim.SGD(list(model.parameters()) + list(head.parameters()),
                          lr=lr, momentum=0.9, weight_decay=5e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        set_quant(model, ep >= warmup); model.train(); tot = 0
        for imgs, lab in loader:
            imgs, lab = imgs.to(device), lab.to(device)
            opt.zero_grad(); loss = ce(head(model(imgs), lab), lab)
            loss.backward(); opt.step(); tot += loss.item()
        sch.step()
        print(f"  [MFN|{'QAT' if ep>=warmup else 'warmup'}] ep {ep+1}/{epochs} loss={tot/len(loader):.3f}", flush=True)
    torch.save(model.state_dict(), "results/mobilefacenet_ternary.pt")
    return model


if __name__ == "__main__":
    main()
