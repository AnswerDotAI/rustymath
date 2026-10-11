"""Writes rustymath's reference results with mpmath. Run it from the repository's root as `python testdata/references.py`.

It adds the correctly rounded f64 result to each line of `core-math.txt`, and writes `mpmath.txt`: samples of the special functions,
which Rust's standard library lacks, with their correctly rounded f64 results, and the IEEE results of their special inputs. An input
that is zero, infinite or NaN takes NumPy's result, which follows C99's Annex F."""
import math, random
from pathlib import Path
import mpmath as mp
import numpy as np

HERE = Path(__file__).parent
mp.mp.prec = 256
inf, nan = math.inf, math.nan

def f64(v):
    "`v` rounded to the nearest f64. Overflow gives an infinity, and a complex or undefined `v` gives NaN."
    if isinstance(v, mp.mpc):
        if v.imag: return nan
        v = v.real
    if mp.isnan(v): return nan
    if mp.isinf(v): return math.copysign(inf, v)
    if not v: return 0.0
    q = max(mp.frexp(v)[1] - 53, -1074)
    n = int(mp.nint(mp.ldexp(v, -q)))
    if not n: return math.copysign(0.0, v)
    try: return math.ldexp(n, q)
    except OverflowError: return math.copysign(inf, v)

def show(x):
    "`x` as Rust parses it back exactly."
    return 'NaN' if math.isnan(x) else ('inf' if x > 0 else '-inf') if math.isinf(x) else repr(x)

def special(*xs): return any(x == 0 or not math.isfinite(x) for x in xs)

def guarded(f):
    "`f`, giving NaN where mpmath raises: at poles and outside a function's domain."
    def g(*xs):
        try: return f(*xs)
        except (ValueError, ZeroDivisionError): return nan
    return g

def cbrt(x): return mp.sign(x) * mp.cbrt(abs(x))

# Each function of core-math.txt with mpmath's version and NumPy's, which gives the IEEE result for special inputs.
ELEMENTARY = dict(
    exp=(mp.exp, np.exp), expm1=(mp.expm1, np.expm1), log=(mp.log, np.log), log1p=(mp.log1p, np.log1p),
    log2=(lambda x: mp.log(x, 2), np.log2), log10=(mp.log10, np.log10), sin=(mp.sin, np.sin), cos=(mp.cos, np.cos),
    tan=(mp.tan, np.tan), asin=(mp.asin, np.arcsin), acos=(mp.acos, np.arccos), atan=(mp.atan, np.arctan), sinh=(mp.sinh, np.sinh),
    cosh=(mp.cosh, np.cosh), tanh=(mp.tanh, np.tanh), asinh=(mp.asinh, np.arcsinh), acosh=(mp.acosh, np.arccosh),
    atanh=(mp.atanh, np.arctanh), cbrt=(cbrt, np.cbrt), pow=(mp.power, np.power), hypot=(mp.hypot, np.hypot),
)

def result(name, xs):
    exact, ieee = ELEMENTARY[name]
    if special(*xs):
        with np.errstate(all='ignore'): return float(ieee(*xs))
    return f64(guarded(exact)(*map(mp.mpf, xs)))

def core_math():
    "Rewrites each data line of core-math.txt as its name, its inputs and their correctly rounded result."
    path = HERE/'core-math.txt'
    out = []
    for line in path.read_text().splitlines():
        if line.startswith('#'): out.append(line); continue
        name, *fields = line.split()
        xs = [float(f) for f in fields[:2 if name in ('pow', 'hypot') else 1]]
        out.append(' '.join([name, *map(show, xs), show(result(name, xs))]))
    path.write_text('\n'.join(out) + '\n')

def erfcinv(q):
    "erfc⁻¹(q), as erf⁻¹(1 - q) with enough precision that 1 - q keeps q's digits."
    with mp.workprec(256 + max(0, -mp.frexp(q)[1])): return mp.erfinv(1 - q)

def lgamma(x):
    "ln |Γ(x)|, which is +∞ at the poles."
    return mp.inf if x <= 0 and x == mp.floor(x) else mp.loggamma(x).real

# Each special function of one argument: mpmath's version, its sample ranges, and extra inputs, such as the breakpoints of FreeBSD's
# erf and the half-integers, where Γ has closed forms.
halves = [k * 0.5 for k in range(1, 81)] + [0.25 - k * 0.5 for k in range(1, 81)]
UNARY = dict(
    erf=(mp.erf, [(-1, 1), (-6, 6)], [0.84375, 1.25, 2.857142857142857, 6.0]),
    erfc=(mp.erfc, [(-1, 1), (-6, 27), (4, 5)], [0.84375, 1.25, 2.857142857142857, 26.5]),
    erfinv=(mp.erfinv, [(-0.5, 0.5), (-1, 1)], [0.5, -0.5, 0.999999]),
    erfcinv=(erfcinv, [(0, 2), (0, 1e-10)], [0.5, 1.5, 1e-300, 2 - 1e-16]),
    tgamma=(mp.gamma, [(0.5, 10), (-10, 0.5), (10, 171.62437695630272), (-185, -10)], halves),
    lgamma=(lgamma, [(0, 3), (-10, 10), (10, 1e6), (-200, -10)], halves),
)
# The IEEE results of special inputs, and of inputs at poles and outside the domain.
SPECIAL = dict(
    erf=[(0.0, 0.0), (-0.0, -0.0), (inf, 1.0), (-inf, -1.0), (nan, nan)],
    erfc=[(0.0, 1.0), (-0.0, 1.0), (inf, 0.0), (-inf, 2.0), (nan, nan)],
    erfinv=[(0.0, 0.0), (-0.0, -0.0), (1.0, inf), (-1.0, -inf), (2.0, nan), (inf, nan), (nan, nan)],
    erfcinv=[(0.0, inf), (1.0, 0.0), (2.0, -inf), (-1.0, nan), (3.0, nan), (inf, nan), (nan, nan)],
    tgamma=[(0.0, inf), (-0.0, -inf), (inf, inf), (-inf, nan), (-1.0, nan), (nan, nan)],
    lgamma=[(0.0, inf), (-0.0, inf), (inf, inf), (-inf, inf), (1.0, 0.0), (2.0, 0.0), (-1.0, inf), (nan, nan)],
)
# The regularized incomplete functions, with ranges for their parameters and argument. Every input is a float at both widths.
gamma_sets = [((0.05, 5), (0, 10)), ((5, 50), (0, 100)), ((50, 500), (40, 600))]
beta_sets = [((0.1, 5), (0.1, 5), (0, 1)), ((1, 50), (1, 50), (0, 1)), ((20, 500), (20, 500), (0, 1))]
INCOMPLETE = dict(
    gammainc=(lambda a, x: mp.gammainc(a, 0, x, regularized=True), gamma_sets),
    gammaincc=(lambda a, x: mp.gammainc(a, x, mp.inf, regularized=True), gamma_sets),
    betainc=(lambda a, b, x: mp.betainc(a, b, 0, x, regularized=True), beta_sets),
)
SAMPLES = 512

def narrow(x): return float(np.float32(x))

def inputs(rng, ranges, extra):
    "`SAMPLES` uniform inputs from each range, half of them floats at both widths, `SAMPLES` f32 inputs over every exponent, and `extra`."
    xs = []
    for lo, hi in ranges:
        xs += [rng.uniform(lo, hi) for _ in range(SAMPLES // 2)] + [narrow(rng.uniform(lo, hi)) for _ in range(SAMPLES // 2)]
    spread = (float(np.uint32(rng.getrandbits(32)).view(np.float32)) for _ in range(4 * SAMPLES))
    return xs + [x for x in spread if math.isfinite(x)][:SAMPLES] + extra

def special_functions():
    "Writes mpmath.txt."
    rows = ['# Written by references.py: special-function inputs with their correctly rounded f64 results, from mpmath at 256 bits, and the',
            '# IEEE results of special inputs. Each line is a function name, its inputs and its result.']
    for name, (f, ranges, extra) in UNARY.items():
        rng = random.Random(name)
        rows += [f'{name} {show(x)} {show(y)}' for x, y in SPECIAL[name]]
        rows += [f'{name} {show(x)} {show(f64(guarded(f)(mp.mpf(x))))}' for x in inputs(rng, ranges, extra) if not special(x)]
    for name, (f, sets) in INCOMPLETE.items():
        rng = random.Random(name)
        for ranges in sets:
            for _ in range(SAMPLES):
                xs = [narrow(rng.uniform(lo, hi)) for lo, hi in ranges]
                if not special(*xs): rows.append(' '.join([name, *map(show, xs), show(f64(guarded(f)(*map(mp.mpf, xs))))]))
    (HERE/'mpmath.txt').write_text('\n'.join(rows) + '\n')

if __name__ == '__main__':
    core_math()
    special_functions()
