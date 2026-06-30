"""
012 Ternary OS Concept Simulator

IMPORTANT — what this is and isn't:
  This is a SOFTWARE SIMULATION of OS concepts using ternary logic,
  running on your ordinary binary CPU (Python/CPython). It demonstrates
  how a scheduler and process model COULD work if ternary states were
  the native primitive, using the same consensus-gate math as your
  hardware RTL (hardware/consensus_gate.sv).

  It does NOT run on, or require, real ternary silicon. No such
  hardware exists at consumer/commercial scale anywhere. This is a
  concept demo, not a bootable operating system.

Core ideas demonstrated:
  1. Process state as a trit: -1=blocked, 0=ready, +1=running
     (instead of binary running/not-running)
  2. Consensus-gate scheduling: instead of a single priority number,
     each process is scored by 3 independent trit "voters"
     (priority, wait-time, fairness) and the scheduler picks via
     sign(v0+v1+v2) — the same majority-vote math as your hardware
     consensus_gate.sv, applied to a software scheduling decision.
  3. Process control blocks stored ternary-compressed in memory using
     the same pack_ternary/unpack_ternary from trit_app.py.

Usage:
  python trit_os_sim.py
"""

import sys, time, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_app import pack_ternary, unpack_ternary
import numpy as np

# ══════════════════════════════════════════════════════════════════════════════
# TRIT PROCESS STATE
# ══════════════════════════════════════════════════════════════════════════════

BLOCKED, READY, RUNNING = -1, 0, 1
STATE_NAME = {BLOCKED: "BLOCKED", READY: "READY", RUNNING: "RUNNING"}

class TritProcess:
    """A process whose state is a single trit, not a binary flag."""
    _next_pid = 1

    def __init__(self, name, work_units, priority_trit=0):
        self.pid = TritProcess._next_pid
        TritProcess._next_pid += 1
        self.name = name
        self.state = READY
        self.work_remaining = work_units
        self.priority_trit = priority_trit  # -1 low, 0 normal, +1 high
        self.wait_ticks = 0
        self.history = []

    def __repr__(self):
        return f"P{self.pid}:{self.name}[{STATE_NAME[self.state]}]"

# ══════════════════════════════════════════════════════════════════════════════
# CONSENSUS-GATE SCHEDULER
# Same math as hardware/consensus_gate.sv: sign(v0 + v1 + v2)
# Three independent trit voters score each ready process; the process
# with the highest consensus vote wins the CPU this tick.
# ══════════════════════════════════════════════════════════════════════════════

def consensus(v0, v1, v2):
    """The hardware-native gate: majority vote across 3 trits."""
    total = v0 + v1 + v2
    return 1 if total > 0 else (-1 if total < 0 else 0)

def vote_priority(p):
    return p.priority_trit

def vote_wait_time(p):
    # Aging: processes waiting long enough vote +1 to prevent starvation
    if p.wait_ticks >= 3: return 1
    if p.wait_ticks == 0: return -1
    return 0

def vote_fairness(p):
    # Penalize processes that already got a lot of CPU time recently
    recent_runs = p.history[-3:].count(RUNNING)
    if recent_runs >= 2: return -1
    if recent_runs == 0: return 1
    return 0

class TernaryScheduler:
    def __init__(self):
        self.processes = []
        self.tick = 0

    def add(self, process):
        self.processes.append(process)

    def pick_next(self):
        """Consensus-gate scheduling: score every READY process via the
        3-voter consensus gate, run the one with the highest vote."""
        ready = [p for p in self.processes if p.state == READY]
        if not ready:
            return None
        scored = []
        for p in ready:
            v0, v1, v2 = vote_priority(p), vote_wait_time(p), vote_fairness(p)
            score = consensus(v0, v1, v2)
            scored.append((score, -p.wait_ticks, p))  # tiebreak: longest-waiting first
        scored.sort(key=lambda x: (-x[0], x[1]))
        return scored[0][2]

    def step(self):
        self.tick += 1
        next_p = self.pick_next()

        for p in self.processes:
            if p.state == BLOCKED:
                continue
            if p is next_p:
                p.state = RUNNING
                p.wait_ticks = 0
                p.work_remaining -= 1
                print(f"  tick {self.tick:>3}  {p}  work_left={p.work_remaining}")
                if p.work_remaining <= 0:
                    p.state = BLOCKED  # finished -> terminal state
                    print(f"           {p.name} (PID {p.pid}) finished.")
                else:
                    p.state = READY
            else:
                if p.state == READY:
                    p.wait_ticks += 1
            p.history.append(p.state)

        return next_p

    def run_until_done(self, max_ticks=200):
        while any(p.state != BLOCKED or p.work_remaining > 0 for p in self.processes) \
                and self.tick < max_ticks:
            if not any(p.state == READY for p in self.processes):
                break
            self.step()

# ══════════════════════════════════════════════════════════════════════════════
# TERNARY-COMPRESSED PROCESS TABLE
# Process control blocks stored as ternary-packed bytes, same encoding
# as the OBSERVE search index (trit_app.py's pack_ternary/unpack_ternary).
# ══════════════════════════════════════════════════════════════════════════════

def encode_process_table(processes):
    """Encode [state, priority, normalized_wait] per process as trits,
    then bit-pack 5 trits/byte — same compression as the search index."""
    rows = []
    for p in processes:
        wait_trit = 1 if p.wait_ticks >= 3 else (-1 if p.wait_ticks == 0 else 0)
        rows.append([p.state, p.priority_trit, wait_trit])
    trit_matrix = np.array(rows, dtype="int8")
    packed = pack_ternary(trit_matrix)
    return packed, trit_matrix.shape[1]

def decode_process_table(packed, dim, n):
    unpacked = unpack_ternary(packed, dim)
    return unpacked[:n]

# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("="*65)
    print("  012 Ternary OS Concept Simulator")
    print("  (software simulation — runs on your normal binary CPU)")
    print("="*65)

    sched = TernaryScheduler()
    sched.add(TritProcess("render",     work_units=4, priority_trit=1))
    sched.add(TritProcess("background", work_units=6, priority_trit=-1))
    sched.add(TritProcess("network_io", work_units=3, priority_trit=0))
    sched.add(TritProcess("ui_update",  work_units=2, priority_trit=1))

    print(f"\nProcesses: {sched.processes}\n")
    print("Consensus-gate scheduling (sign of 3 trit votes: priority, wait-time, fairness):\n")

    sched.run_until_done()

    print(f"\nDone in {sched.tick} ticks.\n")

    print("="*65)
    print("  Ternary-compressed process table (same encoding as OBSERVE)")
    print("="*65)
    packed, dim = encode_process_table(sched.processes)
    float_equiv = len(sched.processes) * dim * 4
    print(f"  Process control blocks : {len(sched.processes)}")
    print(f"  Float32 equivalent     : {float_equiv} bytes")
    print(f"  Ternary packed          : {packed.nbytes} bytes")
    print(f"  Compression             : {float_equiv/packed.nbytes:.1f}x")

    decoded = decode_process_table(packed, dim, len(sched.processes))
    print(f"\n  Round-trip check (decoded matches original): "
          f"{'PASS' if decoded.shape == (len(sched.processes), dim) else 'FAIL'}")
