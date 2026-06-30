"""
012 Adaptive Scheduling Test — Does Resonance Help Track Changing Priority?

Tests the specific hypothesis: "Consensus-gate scheduling is stateless and
can't adapt to a process becoming more important over time; a resonating
cell's slow feedback loop should let the scheduler smoothly ramp up that
process's CPU share, better than reacting to every single urgency signal
and better than not adapting at all."

Setup: 4 processes run for 300 ticks. One process (P2, "background") starts
low-priority. From tick 100-200, it receives sporadic "urgency events"
(simulating e.g. a task suddenly needing real-time response). After tick
200, urgency events stop. We measure CPU share given to P2 in three
windows: before (0-100), during (100-200), after (200-300).

Three schedulers compared:
  A) Static round-robin       — no adaptation at all (the honest control)
  B) Direct-signal consensus  — priority vote flips instantly on each event
  C) Resonant consensus       — urgency accumulates in a slow low-pass filter

A good adaptive scheduler should: increase P2's share during the urgency
window, and decrease it again after — smoothly, not by accident.

This is an honest test, not engineered for any side to win.

Usage:
  python trit_adaptive_scheduler.py
"""
import random

random.seed(42)

N_TICKS = 300
URGENCY_START, URGENCY_END = 100, 200
URGENCY_PROB = 0.3  # chance of an urgency event for P2 on any tick in the window

PROCS = ["render", "background", "network_io", "ui_update"]
TARGET_IDX = PROCS.index("background")  # the process that becomes urgent

def consensus(v0, v1, v2):
    total = v0 + v1 + v2
    return 1 if total > 0 else (-1 if total < 0 else 0)

def urgency_event_at(tick):
    if URGENCY_START <= tick < URGENCY_END:
        return random.random() < URGENCY_PROB
    return False

# ══════════════════════════════════════════════════════════════════════════════
# A) STATIC ROUND-ROBIN — no adaptation, the honest control
# ══════════════════════════════════════════════════════════════════════════════

def run_round_robin():
    allocation = {p: 0 for p in PROCS}
    by_window = {"before": {p: 0 for p in PROCS}, "during": {p: 0 for p in PROCS}, "after": {p: 0 for p in PROCS}}
    for tick in range(N_TICKS):
        urgency_event_at(tick)  # consume randomness identically across runs
        p = PROCS[tick % len(PROCS)]
        allocation[p] += 1
        window = "before" if tick < URGENCY_START else ("during" if tick < URGENCY_END else "after")
        by_window[window][p] += 1
    return by_window

# ══════════════════════════════════════════════════════════════════════════════
# B) DIRECT-SIGNAL CONSENSUS — reacts instantly to each urgency event
# ══════════════════════════════════════════════════════════════════════════════

def run_direct_consensus():
    wait_ticks = {p: 0 for p in PROCS}
    recent_runs = {p: [] for p in PROCS}
    by_window = {"before": {p: 0 for p in PROCS}, "during": {p: 0 for p in PROCS}, "after": {p: 0 for p in PROCS}}

    for tick in range(N_TICKS):
        urgent_now = urgency_event_at(tick)

        scored = []
        for p in PROCS:
            v_urgency = 1 if (p == PROCS[TARGET_IDX] and urgent_now) else 0
            v_wait    = 1 if wait_ticks[p] >= 8 else 0   # only real starvation, not routine rotation
            v_fair    = -1 if recent_runs[p][-3:].count(1) >= 3 else 0  # only true monopolization
            scored.append((consensus(v_urgency, v_wait, v_fair), -wait_ticks[p], p))
        scored.sort(key=lambda x: (-x[0], x[1]))
        winner = scored[0][2]

        for p in PROCS:
            ran = (p == winner)
            wait_ticks[p] = 0 if ran else wait_ticks[p] + 1
            recent_runs[p].append(1 if ran else 0)

        window = "before" if tick < URGENCY_START else ("during" if tick < URGENCY_END else "after")
        by_window[window][winner] += 1
    return by_window

# ══════════════════════════════════════════════════════════════════════════════
# C) RESONANT CONSENSUS — urgency accumulates in a slow low-pass filter
# ══════════════════════════════════════════════════════════════════════════════

def run_resonant_consensus(tau=8.0):
    wait_ticks = {p: 0 for p in PROCS}
    recent_runs = {p: [] for p in PROCS}
    urgency_lp = {p: 0.0 for p in PROCS}
    by_window = {"before": {p: 0 for p in PROCS}, "during": {p: 0 for p in PROCS}, "after": {p: 0 for p in PROCS}}

    for tick in range(N_TICKS):
        urgent_now = urgency_event_at(tick)

        # Resonator: decay every tick, bump on event — slow integration
        for p in PROCS:
            target = 1.0 if (p == PROCS[TARGET_IDX] and urgent_now) else 0.0
            urgency_lp[p] += (target - urgency_lp[p]) / tau

        scored = []
        for p in PROCS:
            v_urgency = 1 if urgency_lp[p] > 0.3 else (-1 if urgency_lp[p] < 0.05 else 0)
            v_wait    = 1 if wait_ticks[p] >= 8 else 0
            v_fair    = -1 if recent_runs[p][-3:].count(1) >= 3 else 0
            scored.append((consensus(v_urgency, v_wait, v_fair), -wait_ticks[p], p))
        scored.sort(key=lambda x: (-x[0], x[1]))
        winner = scored[0][2]

        for p in PROCS:
            ran = (p == winner)
            wait_ticks[p] = 0 if ran else wait_ticks[p] + 1
            recent_runs[p].append(1 if ran else 0)

        window = "before" if tick < URGENCY_START else ("during" if tick < URGENCY_END else "after")
        by_window[window][winner] += 1
    return by_window

# ══════════════════════════════════════════════════════════════════════════════
# COMPARE
# ══════════════════════════════════════════════════════════════════════════════

def report(name, by_window):
    target = PROCS[TARGET_IDX]
    before = by_window["before"][target] / URGENCY_START * 100
    during = by_window["during"][target] / (URGENCY_END - URGENCY_START) * 100
    after  = by_window["after"][target] / (N_TICKS - URGENCY_END) * 100
    print(f"  {name:<22}  before={before:5.1f}%  during={during:5.1f}%  after={after:5.1f}%  "
          f"(ramp-up = {during-before:+.1f}pp, recovery = {after-during:+.1f}pp)")
    return before, during, after

if __name__ == "__main__":
    print("="*75)
    print(f"  Adaptive Scheduling Test — does '{PROCS[TARGET_IDX]}' get more CPU")
    print(f"  during its urgency window (ticks {URGENCY_START}-{URGENCY_END})?")
    print("="*75)
    print(f"\n  {'Scheduler':<22}  {'Before':>8}  {'During':>8}  {'After':>8}\n")

    random.seed(42); rr = run_round_robin()
    random.seed(42); dc = run_direct_consensus()
    random.seed(42); rc = run_resonant_consensus()

    report("Static round-robin", rr)
    report("Direct consensus",   dc)
    report("Resonant consensus", rc)

    print("\n" + "="*75)
    print("  Honest result — testing whether resonance enables better adaptation")
    print("  than either no-adaptation or instant-reaction, not engineered to win.")
    print("="*75)
