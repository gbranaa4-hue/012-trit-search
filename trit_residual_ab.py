"""
Residual-ternary ("keep the shadow") training A/B -- 3-arm, component-isolated.

trit_residual_quant_test.py already showed keeping the residual halves
RECONSTRUCTION error. That's necessary, not sufficient -- reconstruction
error is not task accuracy. This trains the real triadic encoder under three
quantizers and measures held-out retrieval accuracy, to see whether the
reconstruction win becomes an actual accuracy win.

Three arms (everything else identical -- same architecture, data, seed,
gradient estimator -- only the weight quantizer differs, so any accuracy
gap is attributable to the quantizer alone):

  A  bare1      {-1,0,+1}, no scale         <- the current method (baseline)
  B  scale1     a*{-1,0,+1}, optimal scale  <- isolates the scale's contribution
  C  residual2  a1*t1 + a2*t2 (two trits)   <- B + the shadow trit

A->B isolates the scale; B->C isolates the second ("shadow") trit specifically.

The encoder itself is reused UNCHANGED from trit_triadic_encoder.py -- the
only thing swapped is the module-level quantizer `tq`, which its
TernaryLinear looks up at call time. So this is a genuine A/B on the real
model, not a reimplementation that might differ subtly.

Usage:
    python trit_residual_ab.py [epochs]     default 400 (a first-signal run,
                                            not the final 2000-epoch verdict)
"""
import sys
import time
import random

import torch
import torch.nn.functional as F

import trit_triadic_encoder as tte


# ── configurable quantizers (drop-in replacements for tte.tq) ────────────

def _ternary_hard(x, frac=0.7):
    t = frac * x.abs().mean()
    return torch.where(x > t, torch.ones_like(x),
           torch.where(x < -t, -torch.ones_like(x), torch.zeros_like(x)))

def _optimal_scale(w, t):
    denom = (t * t).sum()
    return (w * t).sum() / denom if denom > 0 else torch.zeros((), device=w.device)

def _ste(w, approx):
    """Straight-through: forward returns `approx`, backward passes gradient
    straight to `w` (identity). Used identically across all three arms so
    the gradient estimator is NOT a confound -- only the forward quant differs."""
    return w + (approx - w).detach()

def make_quantizer(mode):
    def q(w):
        if mode == "bare1":
            approx = _ternary_hard(w)                       # {-1,0,+1}, no scale
        elif mode == "scale1":
            t1 = _ternary_hard(w)
            approx = _optimal_scale(w, t1) * t1             # a*{-1,0,+1}
        elif mode == "residual2":
            t1 = _ternary_hard(w)
            a1 = _optimal_scale(w, t1)
            approx1 = a1 * t1
            resid = w - approx1                            # the shadow
            t2 = _ternary_hard(resid)
            a2 = _optimal_scale(resid, t2)
            approx = approx1 + a2 * t2
        else:
            raise ValueError(mode)
        return _ste(w, approx)
    return q


# ── data (reuse the encoder's real pipeline), fixed split shared by all arms ──

def build_data():
    pairs = tte.collect_local_pairs()
    random.Random(0).shuffle(pairs)
    n_val = max(20, len(pairs) // 10)
    val, train = pairs[:n_val], pairs[n_val:]
    all_texts = [a for a, b in pairs] + [b for a, b in pairs]
    vocab = tte.Vocab(all_texts)
    return train, val, vocab


@torch.no_grad()
def retrieval_accuracy(model, vocab, val, max_len, device):
    """Top-1 retrieval: for each held-out (anchor, positive), does the
    positive rank as the most similar among all val positives? A fair
    relative metric across arms (higher = better embeddings)."""
    model.eval()
    anchors = torch.tensor([vocab.encode(a, max_len) for a, b in val], device=device)
    positives = torch.tensor([vocab.encode(b, max_len) for a, b in val], device=device)
    a_emb = model(anchors)
    p_emb = model(positives)
    sims = a_emb @ p_emb.T                     # (N, N)
    pred = sims.argmax(dim=1)
    correct = (pred == torch.arange(len(val), device=device)).float().mean().item()
    return correct


def train_arm(mode, train, val, vocab, epochs, device):
    tte.tq = make_quantizer(mode)              # swap the quantizer the model uses
    torch.manual_seed(0); random.seed(0)       # identical init + batch order per arm

    model = tte.TritSentenceEncoder(
        vocab_size=len(vocab), n_embd=tte.CFG["n_embd"], n_head=tte.CFG["n_head"],
        n_layer=tte.CFG["n_layer"], max_len=tte.CFG["max_len"],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tte.CFG["lr"], weight_decay=0.01)
    max_len = tte.CFG["max_len"]

    t0 = time.time()
    for step in range(epochs):
        tte.set_quant(model, step >= tte.CFG["quant_warmup"])
        model.train()
        batch = random.sample(train, min(tte.CFG["batch_size"], len(train)))
        anchors = torch.tensor([vocab.encode(a, max_len) for a, b in batch], device=device)
        positives = torch.tensor([vocab.encode(b, max_len) for a, b in batch], device=device)
        loss = tte.contrastive_loss(model(anchors), model(positives))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()

    tte.set_quant(model, True)
    acc = retrieval_accuracy(model, vocab, val, max_len, device)
    return acc, loss.item(), time.time() - t0


def main():
    epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    # optional 2nd arg: comma-separated arms to run (default all three).
    # e.g. "scale1,residual2" to isolate the shadow trit with the scale held on.
    arms = sys.argv[2].split(",") if len(sys.argv) > 2 else ["bare1", "scale1", "residual2"]
    device = torch.device("cpu")   # be explicit; this machine has no guaranteed CUDA
    print(f"Residual-ternary A/B -- {epochs} epochs/arm on {device}, arms={arms}")
    print("(reduced-epoch first-signal run; not the final 2000-epoch verdict)\n")

    train, val, vocab = build_data()
    print(f"{len(train)} train pairs, {len(val)} val pairs, vocab {len(vocab)}\n")

    results = {}
    for mode in arms:
        acc, final_loss, dt = train_arm(mode, train, val, vocab, epochs, device)
        results[mode] = acc
        print(f"  {mode:10s}  retrieval_acc={acc*100:5.1f}%   final_loss={final_loss:.4f}   {dt:.0f}s")

    labels = {"bare1": "bare1     (no scale)",
              "scale1": "scale1    (+ optimal scale)",
              "residual2": "residual2 (+ shadow trit, 2x storage)"}
    print("\n" + "=" * 60)
    print("  RESULT (higher retrieval accuracy = better embeddings)")
    print("=" * 60)
    for mode in arms:
        print(f"  {labels[mode]:38s}: {results[mode]*100:5.1f}%")
    print()
    if "bare1" in results and "scale1" in results:
        print(f"  scale contribution   (scale1 - bare1)   : {(results['scale1']-results['bare1'])*100:+.1f}pp")
    if "scale1" in results and "residual2" in results:
        print(f"  shadow contribution  (residual2 - scale1): {(results['residual2']-results['scale1'])*100:+.1f}pp")
    print("\n  Honest read: small val sets are noisy. Treat this as")
    print("  signal/no-signal. This run tests whether MORE DATA lets the")
    print("  shadow trit generalize (C>B) or whether it's a genuine tie --")
    print("  the earlier 400-epoch run overfit on ~1400 pairs.")


if __name__ == "__main__":
    main()
