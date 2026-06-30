"""
012 TritChip Assembler

A text-based assembly syntax for trit_emulator.py's TernaryCPU, so
programs can be written as readable assembly instead of Python tuples.

Syntax (one instruction per line, # for comments, LABEL: for jump targets):

  LOAD R0, 5
  LOAD R1, -2
  ADD  R2, R0, R1
  CONSENSUS R3, R0, R1, R2
  PRINT R3
  loop:
    SUB R0, R0, R1
    JNZ R0, loop
  HALT

Usage:
  python trit_assembler.py program.tasm     Assemble and run a file
  python trit_assembler.py                  Run the built-in demo program
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from trit_emulator import TernaryCPU

OPCODE_ARITY = {
    "LOAD": 2, "LOADM": 2, "STORE": 2,
    "ADD": 3, "SUB": 3, "NEG": 2, "SGN": 2,
    "CONSENSUS": 4,
    "JMP": 1, "JZ": 2, "JNZ": 2,
    "PRINT": 1, "HALT": 0,
}

class AssemblerError(Exception):
    pass

def assemble(source_text):
    """
    Two-pass assembler:
      Pass 1: strip comments/blank lines, resolve label: lines to addresses
      Pass 2: parse each instruction, substitute label references with addresses
    Returns a list of instruction tuples ready for TernaryCPU.load_program().
    """
    raw_lines = []
    for lineno, line in enumerate(source_text.splitlines(), 1):
        line = line.split("#", 1)[0].strip()
        if line:
            raw_lines.append((lineno, line))

    # Pass 1: find labels
    labels = {}
    instr_lines = []
    for lineno, line in raw_lines:
        if line.endswith(":"):
            label = line[:-1].strip()
            labels[label] = len(instr_lines)
        else:
            instr_lines.append((lineno, line))

    # Pass 2: parse instructions
    program = []
    for lineno, line in instr_lines:
        parts = line.replace(",", " ").split()
        op = parts[0].upper()
        operands = parts[1:]

        if op not in OPCODE_ARITY:
            raise AssemblerError(f"Line {lineno}: unknown instruction '{op}'")

        expected = OPCODE_ARITY[op]
        if len(operands) != expected:
            raise AssemblerError(
                f"Line {lineno}: '{op}' expects {expected} operand(s), got {len(operands)}")

        parsed = []
        for tok in operands:
            if tok.upper().startswith("R") and tok[1:].isdigit():
                parsed.append(tok.upper())            # register
            elif tok in labels:
                parsed.append(labels[tok])             # label -> address
            else:
                try:
                    parsed.append(int(tok))             # immediate integer
                except ValueError:
                    raise AssemblerError(f"Line {lineno}: cannot parse operand '{tok}'")

        program.append((op, *parsed))

    return program

def assemble_and_run(source_text, max_steps=10000, verbose=True):
    program = assemble(source_text)
    cpu = TernaryCPU()
    cpu.load_program(program)
    steps = cpu.run(max_steps=max_steps)
    if verbose:
        print(f"  Assembled {len(program)} instructions, ran {steps} CPU steps")
    return cpu

DEMO_PROGRAM = """
# Demo: compute consensus vote, then count down printing it each time
LOAD R0, 3        # counter
LOAD R1, 1        # decrement amount
LOAD R2, 1
LOAD R3, -1
LOAD R4, 1
loop:
  CONSENSUS R5, R2, R3, R4
  PRINT R5
  SUB R0, R0, R1
  JNZ R0, loop
HALT
"""

if __name__ == "__main__":
    if len(sys.argv) > 1:
        with open(sys.argv[1]) as f:
            source = f.read()
        print(f"Assembling and running: {sys.argv[1]}\n")
    else:
        source = DEMO_PROGRAM
        print("No file given — running built-in demo program:\n")
        print(source)

    try:
        cpu = assemble_and_run(source)
        print("\nFinal register state:")
        for r, v in cpu.regs.items():
            if v != 0:
                print(f"  {r} = {v}")
    except AssemblerError as e:
        print(f"Assembler error: {e}")
        sys.exit(1)
