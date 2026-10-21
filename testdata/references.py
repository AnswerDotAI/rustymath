"""Writes rustymath's reference results with mpmath. Run it from the repository's root as `python testdata/references.py`.

It adds the correctly rounded f64 result to each line of `core-math.txt`. It writes `mpmath.txt`: samples of the special functions,
which Rust's standard library lacks, with their correctly rounded f64 results, and the IEEE results of their special inputs. An input
that is zero, infinite or NaN takes NumPy's result, which follows C99's Annex F. It also writes `complex.txt`: samples of the complex
functions, with the real and imaginary parts of each result correctly rounded to f64."""
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
    digamma=(mp.digamma, [(0, 3), (1.4, 1.5), (-20, 10), (10, 1e6), (-200, -20)], halves + [1.4616321449683622]),
    lambertw=(mp.lambertw, [(-0.3678794411714423, 0), (-0.3678794411714423, -0.3678), (0, 10), (10, 1e300)], [-0.3678794411714423, -0.25, 1.0, math.e]),
    sqrt1mx2=(lambda x: mp.sqrt(1 - x * x), [(-1, 1), (0.99, 1), (-1, -0.99)], [1.0, -1.0, 0.5, 0.9999999999999999]),
    sqrt1px2=(lambda x: mp.sqrt(1 + x * x), [(-2, 2), (-1e-8, 1e-8), (1e150, 1e300)], [1.0, -1.0, 1e154, 1.7976931348623157e308]),
    sqrtx2m1=(lambda x: mp.sign(x) * mp.sqrt(x * x - 1), [(1, 2), (-2, -1), (1, 1e300), (1, 1.001)], [1.0, -1.0, 1e154, 1.7976931348623157e308]),
)
# The IEEE results of special inputs, and of inputs at poles and outside the domain.
SPECIAL = dict(
    erf=[(0.0, 0.0), (-0.0, -0.0), (inf, 1.0), (-inf, -1.0), (nan, nan)],
    erfc=[(0.0, 1.0), (-0.0, 1.0), (inf, 0.0), (-inf, 2.0), (nan, nan)],
    erfinv=[(0.0, 0.0), (-0.0, -0.0), (1.0, inf), (-1.0, -inf), (2.0, nan), (inf, nan), (nan, nan)],
    erfcinv=[(0.0, inf), (1.0, 0.0), (2.0, -inf), (-1.0, nan), (3.0, nan), (inf, nan), (nan, nan)],
    tgamma=[(0.0, inf), (-0.0, -inf), (inf, inf), (-inf, nan), (-1.0, nan), (nan, nan)],
    lgamma=[(0.0, inf), (-0.0, inf), (inf, inf), (-inf, inf), (1.0, 0.0), (2.0, 0.0), (-1.0, inf), (nan, nan)],
    digamma=[(0.0, -inf), (-0.0, inf), (inf, inf), (-inf, nan), (-1.0, nan), (nan, nan)],
    lambertw=[(0.0, 0.0), (-0.0, -0.0), (inf, inf), (-inf, nan), (nan, nan), (-0.36787944117144233, -1.0), (-1.0, nan)],
    sqrt1mx2=[(0.0, 1.0), (-0.0, 1.0), (inf, nan), (-inf, nan), (nan, nan)],
    sqrt1px2=[(0.0, 1.0), (-0.0, 1.0), (inf, inf), (-inf, inf), (nan, nan)],
    sqrtx2m1=[(0.0, nan), (-0.0, nan), (inf, inf), (-inf, -inf), (nan, nan)],
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

def narrow(x):
    "`x` rounded to the nearest f32, or `x` itself where rounding would overflow, or would turn a nonzero `x` into 0."
    with np.errstate(over='ignore', under='ignore'): y = float(np.float32(x))
    return y if math.isfinite(y) and (y or not x) else x

def inputs(rng, ranges, extra):
    "`SAMPLES` uniform inputs from each range, half of them floats at both widths where they fit f32, `SAMPLES` f32 inputs over every exponent, and `extra`."
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

# Each complex function: mpmath's version, and the boxes its samples come from. A box gives, for each argument in turn, a range for
# its real part and one for its imaginary part, or `('e', lo, hi)` for magnitudes from 10^lo to 10^hi, spread evenly over the
# exponents, at every angle.
square = lambda r: ((-r, r), (-r, r))
spread = (('e', -300, 300),)
near = lambda c, r: ((c.real - r, c.real + r), (c.imag - r, c.imag + r))
COMPLEX = dict(
    exp=(mp.exp, [square(2), ((-700, 700), (-50, 50)), ((-5, 5), (-1e6, 1e6))]),
    log=(mp.log, [square(2), spread, near(1, 1e-3), near(1j, 1e-3)]),
    sqrt=(mp.sqrt, [square(2), spread, near(-1, 1e-3)]),
    sqrt1mx2=(lambda z: mp.sqrt(1 - z * z), [square(2), spread, near(1, 1e-3), near(-1, 1e-3)]),
    sqrt1px2=(lambda z: mp.sqrt(1 + z * z), [square(2), spread, near(1j, 1e-3), near(-1j, 1e-3)]),
    sqrtx2m1=(lambda z: mp.sqrt(z - 1) * mp.sqrt(z + 1), [square(2), spread, near(1, 1e-3), near(-1, 1e-3)]),
    sin=(mp.sin, [square(2), ((-100, 100), (-20, 20)), ((-1e6, 1e6), (-1, 1))]),
    cos=(mp.cos, [square(2), ((-100, 100), (-20, 20)), ((-1e6, 1e6), (-1, 1))]),
    tan=(mp.tan, [square(2), ((-100, 100), (-30, 30))]),
    sinh=(mp.sinh, [square(2), ((-20, 20), (-100, 100)), ((-1, 1), (-1e6, 1e6))]),
    cosh=(mp.cosh, [square(2), ((-20, 20), (-100, 100)), ((-1, 1), (-1e6, 1e6))]),
    tanh=(mp.tanh, [square(2), ((-30, 30), (-100, 100))]),
    asin=(mp.asin, [square(2), spread, near(1, 1e-3), near(-1, 1e-3)]),
    acos=(mp.acos, [square(2), spread, near(1, 1e-3), near(-1, 1e-3)]),
    atan=(mp.atan, [square(2), spread, near(1j, 1e-3), near(-1j, 1e-3)]),
    asinh=(mp.asinh, [square(2), spread, near(1j, 1e-3), near(-1j, 1e-3)]),
    acosh=(mp.acosh, [square(2), spread, near(1, 1e-3), near(-1, 1e-3)]),
    atanh=(mp.atanh, [square(2), spread, near(1, 1e-3), near(-1, 1e-3)]),
    lgamma=(mp.loggamma, [square(10), ((0.5, 50), (-50, 50)), ((-50, 0.5), (-20, 20)), ((-10, 10), (-1e4, 1e4))]),
    gamma=(mp.gamma, [square(10), ((0.5, 30), (-20, 20))]),
    pow=(mp.power, [square(3) + square(3), spread + ((-2, 2), (-2, 2))]),
    binom=(mp.binomial, [square(10) + square(10)]),
    lambertw=(mp.lambertw, [square(5), spread, near(-1 / math.e, 1e-3), near(-1 / math.e, 0.3), ((-50, -0.3), (-1e-3, 1e-3))]),
)
COMPLEX_SAMPLES = 128

def args(rng, box):
    "`COMPLEX_SAMPLES` lists of complex arguments from `box`. In every other list, both parts of each argument round to f32, where they fit."
    out = []
    for k in range(COMPLEX_SAMPLES):
        zs, i = [], 0
        while i < len(box):
            if box[i][0] == 'e':
                _, lo, hi = box[i]
                t = rng.uniform(-math.pi, math.pi)
                zs.append(10 ** rng.uniform(lo, hi) * complex(math.cos(t), math.sin(t)))
                i += 1
            else:
                zs.append(complex(rng.uniform(*box[i]), rng.uniform(*box[i + 1])))
                i += 2
        if k % 2: zs = [complex(narrow(z.real), narrow(z.imag)) for z in zs]
        out.append(zs)
    return out

def complex_parts(v):
    "The real and imaginary parts of `v`, each rounded to the nearest f64."
    v = mp.mpc(v)
    return f64(v.real), f64(v.imag)

def complex_exact(f, zs):
    "`f` of the complex arguments `zs`, with 3 extra bits for each binary digit of their exponents, which mpmath's formulas can cancel."
    zs = [mp.mpc(z) for z in zs]
    extra = max((abs(int(mp.mag(z))) for z in zs if z), default=0)
    with mp.workprec(256 + 3 * extra): return complex_parts(f(*zs))

def complex_functions():
    "Writes complex.txt."
    rows = ['# Written by references.py: complex-function inputs with their results from mpmath at 256 bits or more, real and imaginary',
            "# parts each correctly rounded to f64. Each line is a function name, each argument's real and imaginary parts, and the result's."]
    for name, (f, boxes) in COMPLEX.items():
        rng = random.Random(name)
        for box in boxes:
            for zs in args(rng, box):
                try: r = complex_exact(f, zs)
                except (ValueError, ZeroDivisionError): r = (nan, nan)
                rows.append(' '.join([name, *(show(p) for z in zs for p in (z.real, z.imag)), *map(show, r)]))
    (HERE/'complex.txt').write_text('\n'.join(rows) + '\n')

if __name__ == '__main__':
    core_math()
    special_functions()
    complex_functions()
