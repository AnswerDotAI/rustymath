"How fast rustymath's Python API runs. Each case in `speed_python.txt` is timed as `speed.rs` times the Rust slice forms, and checked against its budget."
import gc, os, sys, time
from pathlib import Path
import numpy as np, pytest, rustymath as rm

N, RUNS = 65_536, 5

def arrays():
    "The arrays that the cases name. Each case's main input holds `N` items."
    rng = np.random.default_rng(0)
    u = lambda *shape: rng.uniform(-1, 1, shape)
    x, y = u(N), u(N)
    return dict(np=np, rm=rm, xn=x, x=rm.array(x), y=rm.array(y), xf=rm.array(x.astype(np.float32)), yf=rm.array(y.astype(np.float32)),
        m=rm.array(u(256, 256)), a=rm.array(u(256, 256)), b=rm.array(u(256, 256)), row=rm.array(u(256)), idx=rm.array(rng.permutation(N)),
        img=rm.array(u(256, 256)), k=rm.array(u(3, 3)), kern=rm.array(u(16)), frames=rm.array(u(1024, 64)),
        r=rm.array(rng.integers(0, 256, N)), c=rm.array(rng.integers(0, 256, N)), mask=rm.array(u(N) > 0))

def budgets():
    "Each case in `speed_python.txt` as `(section, code, budget)`, with the native budget that `rm.timing().column` picks for this machine, as written, or `None` on a machine with no budgets."
    section, cases, column = None, [], rm.timing().column
    for line in Path(__file__).with_name('speed_python.txt').read_text().splitlines():
        if not line or line.startswith('#'): continue
        if line.startswith('['): section = line.strip('[]')
        else:
            code, *native = line.split('\t')
            cases.append((section, code, None if column is None else native[column]))
    return cases

def fastest(run):
    "The fastest of `RUNS` runs of `run`, in seconds. Each result is freed after its run is timed."
    best = float('inf')
    for _ in range(RUNS):
        start = time.perf_counter()
        result = run()
        best = min(best, time.perf_counter() - start)
        del result
    return best

def cost(code, namespace, copy):
    "The time that `code` takes for each of `N` items, in copies that take `copy` seconds an item, and in seconds."
    case = compile(code, code, 'eval')
    seconds = fastest(lambda: eval(case, namespace)) / N
    return seconds / copy, seconds

@pytest.fixture
def timing():
    "Hold the timing lock exclusively through `rm.timing`, as `speed.rs` does, with the garbage collector off. Gives the hold, whose `pauses`, `copy` and `line` the test uses. As a fixture, its wait for the lock falls outside the test's timeout."
    with rm.timing() as hold:
        gc.disable()
        try: yield hold
        finally: gc.enable()

def over_budget(cases, namespace, timing, show):
    "Time `cases` in copies of the time that `timing.copy` gives for their section's item type, then time each one over its budget again after each of `timing.pauses`, in seconds, printing the earlier and new costs. Gives the lines of the cases still over budget. With `show`, prints each item type's copy line and every case's line, in the formats `basedpl.bpltests.logged_costs` reads."
    units = {s: timing.copy(np.dtype(s).itemsize, N) for s in dict.fromkeys(s for s, _, _ in cases)}
    if show:
        for _, line in units.values(): print(line, file=sys.stderr)
    def attempt(case):
        section, code, budget = case
        copies, seconds = cost(code, namespace, units[section][0])
        return timing.line(section, code, copies, budget or '-', seconds), budget is not None and copies > float(budget)
    over = []
    for case in cases:
        line, failed = attempt(case)
        if show: print(line, file=sys.stderr)
        if failed: over.append((case, line))
    for pause in timing.pauses:
        if not over: break
        for _, line in over: print(f'{line}: over, timed again after {pause:g} s', file=sys.stderr)
        time.sleep(pause)
        again = []
        for case, _ in over:
            line, failed = attempt(case)
            print(line, file=sys.stderr)
            if failed: again.append((case, line))
        over = again
    return [line for _, line in over]

def test_speed(capsys, timing):
    "Every case costs no more than its budget, on the CPU that the budgets were set for (`timing.checks_budgets`). On any other CPU, it prints every cost, then each case still over its budget, marked `over`, and fails on none. `RUSTYMATH_COSTS` prints every cost."
    namespace, cases = arrays(), budgets()
    with capsys.disabled():
        over = over_budget(cases, namespace, timing, not timing.checks_budgets or 'RUSTYMATH_COSTS' in os.environ)
        if timing.checks_budgets: assert not over, '\n'.join(over)
        for line in over: print(f'{line}: over', file=sys.stderr)
