"""
A real, working fault-localization debugger.
Same scoring idea you built by hand, now automatically tracking which
functions run during each test, instead of hand-typed fake data.
"""

# ── The "codebase" being tested — one function has a real bug ────────────────

touched = []   # global list: every tracked function appends its own name here when called

def track(func):
    """A decorator: wraps a function so that every time it's called for
    real, its name gets appended to the global `touched` list automatically.
    This replaces hand-typing test1 = ["take_damage", "fail"] — the tool
    now OBSERVES which functions actually ran, instead of being told."""
    def wrapper(*args):
        touched.append(func.__name__)
        return func(*args)
    return wrapper

@track
def add(a, b):
    return a + b

@track
def subtract(a, b):
    return a - b

@track
def multiply(a, b):
    return a * b + 1   # <-- REAL BUG: shouldn't have + 1

@track
def divide(a, b):
    return a / b

# ── Test cases — each one calls some functions and checks the result ─────────

def test_add():
    assert add(2, 3) == 5

def test_subtract():
    assert subtract(5, 2) == 3

def test_multiply_small():
    assert multiply(2, 3) == 6      # will FAIL because of the bug

def test_multiply_big():
    assert multiply(4, 5) == 20     # will also FAIL because of the bug

def test_divide():
    assert divide(10, 2) == 5

all_tests = [test_add, test_subtract, test_multiply_small, test_multiply_big, test_divide]

# ── The scoring engine — same algorithm you built by hand ────────────────────

fail_counts = {}
pass_counts = {}

def ensure_tracked(name):
    if name not in fail_counts:
        fail_counts[name] = 0
        pass_counts[name] = 0

for test in all_tests:
    touched.clear()             # reset before each test — only track THIS test's calls
    try:
        test()
        result = "pass"
    except AssertionError:
        result = "fail"

    functions_this_test = set(touched)   # dedupe — a function might run more than once
    for name in functions_this_test:
        ensure_tracked(name)
        if result == "fail":
            fail_counts[name] = fail_counts[name] + 1
        else:
            pass_counts[name] = pass_counts[name] + 1

    print(f"{test.__name__:<20} -> {result}   (touched: {sorted(functions_this_test)})")

# ── Print suspiciousness ranking, most suspicious first ──────────────────────

print("\nSuspiciousness ranking:")
scores = {}
for name in fail_counts:
    scores[name] = fail_counts[name] / (fail_counts[name] + pass_counts[name])

for name in sorted(scores, key=lambda n: -scores[n]):
    print(f"  {name:<12} score={scores[name]:.2f}  (fails={fail_counts[name]}, passes={pass_counts[name]})")
