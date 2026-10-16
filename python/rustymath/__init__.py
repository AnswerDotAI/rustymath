"""Maths in plain Rust, from fast f16, f32 and f64 functions to exact rationals.

Modules:

- `rustymath.optimize`: Minimizing a function of several variables, and solving equations, nonlinear or linear, with SciPy's names. The searches run in Rust and call the Python functions at each point."""

import sys
from . import _core, linalg, polynomial, exact, integer, search, optimize
from .linalg import matmul
from .integer import encode, decode
from .search import Unique, unique, index, isin, argsort, lexsort, searchsorted, bins, bincount, counts, group_indices
from ._core import (__version__, CT, Array, absolute, add, all, angle, any, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, array, betainc,
    betaincinv, binom, cbrt, ceil, cis, cos, cosh, cumprod, cumsum, distribution, divide, equal, erf, erfc, erfcinv, erfinv, exp, expm1, factorial, floor,
    gamma, gammainc, gammaincc, gammaincinv, gammaln, gcd, greater, greater_equal, hypot, lambertw, lcm, less, less_equal, log, log10, log1p, log2, logb,
    loggamma, logical_and, logical_or, logical_scan, logical_xor, logical_xnor, max, maximum, min, minimum, minmax, multiply, not_equal, number_theory, power,
    prod, remainder, root, set_threads, sign, sin, sinh, sqrt, sqrt1mx2, sqrt1px2, sqrtx2m1, subtract, sum, tan, tanh, threads, thread_setting)

for _m in (distribution, number_theory): sys.modules[f'{__name__}.{_m.__name__}'] = _m

