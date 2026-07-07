"""tritkit.qat -- quantization-aware training loop.

QAT is the DEFAULT and only recommended path: train with the quantizer simulated
in the forward pass (after a float warmup) so the network learns to survive
rounding. PTQ -- quantizing a trained float model without retraining -- collapses
on hard/fine tasks (measured in this repo: transformer 95%->50%, faces 0.997->0.58
AUC), so it is deliberately not offered as a one-liner here.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .convert import set_quant


def _kd_loss(student_logits, teacher_logits, T=4.0):
    """Soft-target knowledge distillation (teacher -> ternary student)."""
    s = F.log_softmax(student_logits / T, dim=1)
    t = F.softmax(teacher_logits / T, dim=1)
    return F.kl_div(s, t, reduction="batchmean") * (T * T)


def qat_fit(model, loader, epochs=20, warmup=4, lr=1e-3, weight_decay=1e-4,
            teacher=None, kd_weight=1.0, device=None, log=True):
    """Warmup in float for `warmup` epochs, then train with quantization on.
    Optional `teacher` model adds knowledge distillation (helps the weak student)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if teacher is not None:
        teacher.to(device).eval()
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ce = nn.CrossEntropyLoss()
    for ep in range(epochs):
        set_quant(model, ep >= warmup)
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = ce(out, y)
            if teacher is not None:
                with torch.no_grad():
                    t_out = teacher(x)
                loss = loss + kd_weight * _kd_loss(out, t_out)
            loss.backward()
            opt.step()
        sch.step()
        if log and ((ep + 1) % max(1, epochs // 5) == 0 or ep == 0):
            phase = "QAT" if ep >= warmup else "warmup"
            print(f"  [tritkit|{phase}] epoch {ep+1}/{epochs}", flush=True)
    return model
