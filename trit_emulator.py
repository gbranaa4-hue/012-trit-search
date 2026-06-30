"""
012 Ternary CPU Emulator

A software simulation of a balanced-ternary CPU, running on your ordinary
binary machine. Lets you write and run programs using balanced ternary
arithmetic and a native CONSENSUS instruction (the same sign(a+b+c) gate
as hardware/consensus_gate.sv), without needing real ternary silicon.

IMPORTANT — what this is and isn't (same honesty note as trit_os_sim.py):
  This is an emulator/teaching tool. It does not run on, require, or
  prove anything about real ternary hardware. No ternary transistors
  exist at consumer/commercial manufacturing scale. This lets you
  experiment with the *programming model* of ternary computing today.

Architecture:
  - Balanced ternary words: 9 trits, range -9841..+9841 (3^9 states)
  - 8 general-purpose registers (R0-R7)
  - Flat memory array
  - Instruction set: LOAD, STORE, ADD, SUB, NEG, CONSENSUS, JMP, JZ, JNZ,
    PRINT, HALT
  - CONSENSUS is a native instruction (not built from ADD+compare) —
    it's the hardware-native sign(a+b+c) majority vote, computed
    per-trit across three registers' balanced-ternary digits.

Usage:
  python trit_emulator.py            Run built-in demo programs
"""

# ══════════════════════════════════════════════════════════════════════════════
# BALANCED TERNARY WORD
# ══════════════════════════════════════════════════════════════════════════════

WORD_TRITS = 9  # 3^9 = 19,683 states, range -9841..+9841

def int_to_trits(n, width=WORD_TRITS):
    """Convert an integer to balanced ternary digits (-1,0,+1), least significant first."""
    trits = []
    for _ in range(width):
        n, r = divmod(n + 1, 3)
        r -= 1
        trits.append(r)
        n -= 0  # divmod already adjusted
    return trits

def trits_to_int(trits):
    """Convert balanced ternary digits back to an integer."""
    n = 0
    for i, t in enumerate(reversed(trits)):
        n = n * 3 + t
    return n

def clamp_word(n, width=WORD_TRITS):
    """Wrap an integer into the representable balanced-ternary range."""
    trits = int_to_trits(n, width)
    return trits_to_int(trits)

# ══════════════════════════════════════════════════════════════════════════════
# CPU
# ══════════════════════════════════════════════════════════════════════════════

class TernaryCPU:
    def __init__(self, mem_size=256):
        self.regs = {f"R{i}": 0 for i in range(8)}
        self.mem = [0] * mem_size
        self.pc = 0
        self.program = []
        self.halted = False
        self.output = []

    def load_program(self, instructions):
        self.program = instructions
        self.pc = 0
        self.halted = False

    def step(self):
        if self.pc >= len(self.program) or self.halted:
            self.halted = True
            return False

        instr = self.program[self.pc]
        op, *args = instr
        advance = True

        if op == "LOAD":
            dst, val = args
            self.regs[dst] = clamp_word(val)

        elif op == "LOADM":
            dst, addr = args
            self.regs[dst] = self.mem[addr]

        elif op == "STORE":
            addr, src = args
            self.mem[addr] = self.regs[src]

        elif op == "ADD":
            dst, a, b = args
            self.regs[dst] = clamp_word(self.regs[a] + self.regs[b])

        elif op == "SUB":
            dst, a, b = args
            self.regs[dst] = clamp_word(self.regs[a] - self.regs[b])

        elif op == "NEG":
            dst, a = args
            self.regs[dst] = clamp_word(-self.regs[a])

        elif op == "SGN":
            # Sign of a register's value: +1/0/-1. The natural building
            # block for "manually" reconstructing majority-vote logic
            # without the native CONSENSUS instruction.
            dst, a = args
            v = self.regs[a]
            self.regs[dst] = 1 if v > 0 else (-1 if v < 0 else 0)

        elif op == "CONSENSUS":
            # Native hardware-equivalent instruction: per-trit sign(a+b+c)
            # majority vote across three registers' balanced-ternary digits.
            dst, a, b, c = args
            ta = int_to_trits(self.regs[a])
            tb = int_to_trits(self.regs[b])
            tc = int_to_trits(self.regs[c])
            result_trits = []
            for x, y, z in zip(ta, tb, tc):
                s = x + y + z
                result_trits.append(1 if s > 0 else (-1 if s < 0 else 0))
            self.regs[dst] = trits_to_int(result_trits)

        elif op == "JMP":
            target, = args
            self.pc = target
            advance = False

        elif op == "JZ":
            reg, target = args
            if self.regs[reg] == 0:
                self.pc = target
                advance = False

        elif op == "JNZ":
            reg, target = args
            if self.regs[reg] != 0:
                self.pc = target
                advance = False

        elif op == "PRINT":
            reg, = args
            self.output.append(self.regs[reg])
            print(f"    PRINT {reg} = {self.regs[reg]}")

        elif op == "HALT":
            self.halted = True
            advance = False

        else:
            raise ValueError(f"Unknown instruction: {op}")

        if advance:
            self.pc += 1
        return not self.halted

    def run(self, max_steps=10000):
        steps = 0
        while not self.halted and steps < max_steps:
            self.step()
            steps += 1
        return steps

# ══════════════════════════════════════════════════════════════════════════════
# DEMO PROGRAMS
# ══════════════════════════════════════════════════════════════════════════════

def demo_consensus_vote():
    """Three sensors report -1/0/+1 votes; CONSENSUS picks the majority."""
    print("Demo 1: Consensus vote among three sensor readings")
    print("  Sensor A = +1 (yes), Sensor B = +1 (yes), Sensor C = -1 (no)")
    print("  Expected: majority says +1 (yes)\n")

    cpu = TernaryCPU()
    cpu.load_program([
        ("LOAD", "R0", 1),    # sensor A = +1
        ("LOAD", "R1", 1),    # sensor B = +1
        ("LOAD", "R2", -1),   # sensor C = -1
        ("CONSENSUS", "R3", "R0", "R1", "R2"),
        ("PRINT", "R3"),
        ("HALT",),
    ])
    cpu.run()
    print()

def demo_balanced_arithmetic():
    """Balanced ternary arithmetic: no separate sign bit needed."""
    print("Demo 2: Balanced ternary arithmetic (negative numbers, no sign bit)")
    print("  R0 = 47, R1 = -12, R2 = R0 + R1 (expect 35)")
    print("  R3 = R0 - R1 (expect 59)\n")

    cpu = TernaryCPU()
    cpu.load_program([
        ("LOAD", "R0", 47),
        ("LOAD", "R1", -12),
        ("ADD", "R2", "R0", "R1"),
        ("SUB", "R3", "R0", "R1"),
        ("PRINT", "R2"),
        ("PRINT", "R3"),
        ("HALT",),
    ])
    cpu.run()
    print()

def demo_loop_with_consensus():
    """A loop that runs CONSENSUS repeatedly, counting down via SUB+JNZ."""
    print("Demo 3: Loop — run CONSENSUS 3 times, counting down with balanced ternary")
    print("  Counter starts at 3, decrements each iteration\n")

    cpu = TernaryCPU()
    cpu.load_program([
        ("LOAD", "R0", 3),     # counter
        ("LOAD", "R1", 1),     # decrement amount
        ("LOAD", "R2", 1), ("LOAD", "R3", -1), ("LOAD", "R4", 1),  # vote inputs
        # loop start (index 5):
        ("CONSENSUS", "R5", "R2", "R3", "R4"),
        ("PRINT", "R5"),
        ("SUB", "R0", "R0", "R1"),
        ("JNZ", "R0", 5),
        ("HALT",),
    ])
    steps = cpu.run()
    print(f"  Completed in {steps} CPU steps\n")

def demo_word_range():
    """Show the representable range of a 9-trit balanced ternary word."""
    print(f"Demo 4: Word range — {WORD_TRITS} trits = 3^{WORD_TRITS} = "
          f"{3**WORD_TRITS:,} states, range -{3**WORD_TRITS//2}..+{3**WORD_TRITS//2}")
    print(f"  Storage: {WORD_TRITS} trits = {WORD_TRITS} x log2(3) = "
          f"{WORD_TRITS*1.585:.1f} bits  (vs {((3**WORD_TRITS).bit_length())} bits needed in binary)\n")

if __name__ == "__main__":
    print("="*65)
    print("  012 Ternary CPU Emulator")
    print("  (software simulation — runs on your normal binary CPU)")
    print("="*65)
    print()
    demo_word_range()
    demo_consensus_vote()
    demo_balanced_arithmetic()
    demo_loop_with_consensus()
    print("="*65)
    print("  All demos completed.")
    print("="*65)
