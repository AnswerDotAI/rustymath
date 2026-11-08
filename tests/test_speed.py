"How fast rustymath's Python API runs. Each case in `speed_python.txt` is timed as `speed.rs` times the Rust slice forms, and checked against its budget and against any instruction count that the file records for this machine's column. `RUSTYMATH_COSTS` asks for a calibration run, as `speed.rs` describes."
import gc, os, sys, time
from contextlib import contextmanager
from pathlib import Path
import numpy as np, rustymath as rm

N, RUNS = 65_536, 5
CALIBRATING = 'RUSTYMATH_COSTS' in os.environ

def arrays():
    "The arrays that the cases name. Each case's main input holds `N` items."
    rng = np.random.default_rng(0)
    u = lambda *shape: rng.uniform(-1, 1, shape)
    x, y = u(N), u(N)
    xs = np.sort(x)
    return dict(np=np, rm=rm, xn=x, x=rm.array(x), y=rm.array(y), xf=rm.array(x.astype(np.float32)), yf=rm.array(y.astype(np.float32)),
        xs=rm.array(xs), xsf=rm.array(xs.astype(np.float32)),
        m=rm.array(u(256, 256)), a=rm.array(u(256, 256)), b=rm.array(u(256, 256)), row=rm.array(u(256)), idx=rm.array(rng.permutation(N)),
        img=rm.array(u(256, 256)), k=rm.array(u(3, 3)), kern=rm.array(u(16)), frames=rm.array(u(1024, 64)),
        r=rm.array(rng.integers(0, 256, N)), c=rm.array(rng.integers(0, 256, N)), mask=rm.array(u(N) > 0))

def fastest(run):
    "The fastest of `RUNS` runs of `run`, in seconds. Each result is freed after its run is timed."
    best = float('inf')
    for _ in range(RUNS):
        start = time.perf_counter()
        result = run()
        best = min(best, time.perf_counter() - start)
        del result
    return best

@contextmanager
def held(hold):
    "Keep `hold`, a hold on the timing lock from `rm.timing` or `rm.try_timing`, with the garbage collector off."
    with hold:
        gc.disable()
        try: yield hold
        finally: gc.enable()

def attempt(case, namespace, units, timing):
    "The line of `case`, a `(section, code, budget, counts)` from `timing.budgets`, its cost in copies of the copy time that `units` gives for its section's item type, and the instructions that one run retires for each of `N` items, or `None` where they can't be counted. The counted run comes before the timed runs."
    section, code, budget, _ = case
    expr = compile(code, code, 'eval')
    _, instructions = timing.instructions(lambda: eval(expr, namespace))
    best, peak = timing.peak_held(lambda: fastest(lambda: eval(expr, namespace)))
    copies, per_item = best / N / units[section][0], None if instructions is None else instructions / N
    footprint = timing.footprint(np.dtype(section).itemsize, N, peak)
    return timing.line(section, code, copies, '-' if budget is None else str(budget), best / N, footprint, (per_item, None)), copies, per_item

def retimed(over, namespace, units, timing):
    "Time each of `over`'s `(case, line)` pairs again, printing its new line, and give those still over budget, with their new lines."
    again = []
    for case, _ in over:
        line, copies, _ = attempt(case, namespace, units, timing)
        print(line, file=sys.stderr)
        if case[2].over(copies): again.append((case, line))
    return again

def test_speed(capsys):
    "Every case costs no more than its budget, and retires no more instructions than its recorded count allows, on the CPU that the budgets were set for (`timing.checks_budgets`). On any other CPU, it prints every cost, then each case still over, marked `over`, and fails on none. An ordinary run holds the timing lock shared, on one thread, beside builds and other tests. It then times each case over its budget again if it can hold the lock exclusively at once, and otherwise marks the case `unconfirmed` and doesn't fail it. A calibration run, which `RUSTYMATH_COSTS` asks for, holds the lock exclusively, prints every cost and times each case over its budget again after each of `timing.pauses`. Each case whose gap's hold has outlived the gap is printed, marked `stale`, and each whose instructions fall below its record divided by the margin is marked `fewer instructions`."
    timing = rm.timing(shared=not CALIBRATING)
    namespace, cases = arrays(), timing.budgets(Path(__file__).with_name('speed_python.txt').read_text())
    show, failures = not timing.checks_budgets or CALIBRATING, []
    with capsys.disabled():
        with held(timing):
            units = {s: timing.copy(np.dtype(s).itemsize, N) for s in dict.fromkeys(case[0] for case in cases)}
            if show:
                for _, line in units.values(): print(line, file=sys.stderr)
            measured = [(case, *attempt(case, namespace, units, timing)) for case in cases]
            for (_, _, budget, counts), line, copies, instructions in measured:
                if show: print(line, file=sys.stderr)
                if budget is not None and budget.stale(copies): print(f'{line}: stale', file=sys.stderr)
                record = counts[timing.measures.index('instructions')]
                if record is None or instructions is None: continue
                if record.over(instructions, timing.count_margin): failures.append(f'{line}: instructions over the {record.value} recorded')
                elif instructions * timing.count_margin < record.value: print(f'{line}: fewer instructions than the {record.value} recorded', file=sys.stderr)
            over = [(case, line) for case, line, copies, _ in measured if case[2] is not None and case[2].over(copies)]
            for pause in timing.pauses if CALIBRATING else ():
                if not over: break
                for _, line in over: print(f'{line}: over, timed again after {pause:g} s', file=sys.stderr)
                time.sleep(pause)
                timing.warm()
                over = retimed(over, namespace, units, timing)
        if over and not CALIBRATING:
            if (quiet := rm.try_timing()) is None:
                for _, line in over: print(f'{line}: unconfirmed, because builds or tests hold the timing lock', file=sys.stderr)
                over = []
            else:
                with held(quiet):
                    timing.warm()
                    for _, line in over: print(f'{line}: over, timed again with the timing lock held exclusively', file=sys.stderr)
                    over = retimed(over, namespace, units, timing)
        failures += [line for _, line in over]
        if timing.checks_budgets: assert not failures, '\n'.join(failures)
        for line in failures: print(f'{line}: over', file=sys.stderr)
