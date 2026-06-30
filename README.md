# 012 Ternary

A research and tooling project exploring ternary computing ({-1, 0, +1}) — from a hardware RTL design, through a software CPU emulator, to a working, shipping product: **OBSERVE**, a local semantic code search desktop app.

This repo contains real, measured, reproducible results — both positive and negative — not just claims. See [paper/resonance_domain.md](paper/resonance_domain.md) for an example of a fully falsifiable test series (5 honest experiments, 1 confirmed win, 4 disconfirmed hypotheses, with a precise explanation for the boundary). See [paper/cross_substrate_symmetry_findings.md](paper/cross_substrate_symmetry_findings.md) for a synthesis of this project's own symmetry/resonance results against an independent acoustic-MEMS-plate study — four substrates, four different answers to "does breaking symmetry help computation?" See [paper/npc_consensus_findings.md](paper/npc_consensus_findings.md) for the consensus-gate primitive applied, independently of any neural net, to a real game's (Tribe's) fight-or-flee NPC logic — a clean, statistically robust win (+1.8pp accuracy, 30/30 seeds) — and [paper/order_acceptance_findings.md](paper/order_acceptance_findings.md) for the same primitive tested on a *different* decision shape in the same game, where it loses just as decisively (-4.6pp, 30/30 seeds). The resulting scoping rule (weighted combination wins under calibrated evidence, voting wins under uncalibrated/contaminated evidence) was then checked against two independent published fields and held both times: [Spikeling-Project/research/POPULATION_CODING_FINDINGS.md](https://github.com/gbranaa4-hue/Spikeling/blob/main/research/POPULATION_CODING_FINDINGS.md) (population-coding theory + robust statistics, LIF neurons) and [paper/tmr_findings.md](paper/tmr_findings.md) (classical Triple Modular Redundancy / fault-tolerant systems engineering).

**Full documentation:** [DOCS.md](DOCS.md) (architecture + key results) and [FILES.md](FILES.md) (every file, what it does, how to run it).

---

## What's actually proven here

| Claim | Status | Where |
|---|---|---|
| Fine-tuned MiniLM beats baseline on code search | ✅ Measured: 96% vs 92% (hard benchmark), 92% vs 76.7% (real OSS code) | `trit_benchmark.py`, `trit_oss_test.py` |
| Ternary weight compression (20x) works, but costs accuracy | ✅ Measured: 7.35pp lost, vs INT8's 0.19pp gain at 4x compression | `precision_loss_test.py`, `int8_vs_ternary_test.py` |
| Triadic structure (not just ternary weights) causes CIFAR-10 rotation robustness | ✅ Measured ablation in `experiments.py` | `experiments.py` |
| "Resonating cell" hybrid improves decision-making generally | ❌ Disproven — wins only when the signal is genuinely time-varying (1 of 5 tests) | `paper/resonance_domain.md` |
| Triadic architecture beats pretrained models on text embedding from scratch | ❌ Disproven — scored 60% vs MiniLM's 96% | `trit_triadic_encoder.py` |
| Ternary CPU's native CONSENSUS instruction is more efficient than software fallback | ✅ Measured: 1.4x fewer instructions (modest, not the "10x" sometimes claimed informally) | `trit_emulator_tests.py` |

---

## Quick start

```bash
pip install -r requirements.txt          # full project
# or: pip install -r requirements_app.txt   # just the OBSERVE search app

python trit_app.py                       # launch OBSERVE desktop search
python experiments.py                    # CIFAR-10 ternary/triadic ablation study
python trit_emulator.py                  # balanced-ternary CPU emulator demo
```

GPU users: install `torch`/`torchvision` matching your CUDA version first (see comment in `requirements.txt`).

---

## Structure

```
012-ternary/
├── trit_app.py              OBSERVE — desktop semantic code search (the shipping product)
├── trit_search.py           Core search engine (CLI/HTTP API)
├── trit_embed_train.py      Fine-tunes MiniLM on code for search
├── experiments.py           Core ternary/triadic learning system (CIFAR-10 ablations)
├── trit_emulator.py         Balanced-ternary CPU emulator
├── trit_assembler.py        Text assembly syntax for the emulator
├── trit_*.py                LLM fine-tuning, language model, memory store, etc.
├── trit_resonant_*.py       The 5-test resonance hypothesis series (see paper/)
├── hardware/                SystemVerilog RTL — consensus gate, ternary ALU, FPGA target
├── paper/                   Research write-ups (resonance_domain.md, 012_paper.md)
├── DOCS.md                  Full architecture + results documentation
└── FILES.md                 Every file, what it does, how to run it
```

## Hardware validation

```bash
iverilog -g2012 -o sim hardware/trit_pkg.sv hardware/trit_register.sv \
    hardware/trit_not.sv hardware/trit_add.sv \
    hardware/consensus_gate.sv hardware/testbench.sv
vvp sim
# 36/36 testbench cases pass in simulation
```

FPGA synthesis (`hardware/vivado_synth.tcl`) targets Xilinx Ultrascale+; this has not been run on real silicon — see DOCS.md for what's simulated vs hardware-verified.

## License

MIT — see [LICENSE](LICENSE).
