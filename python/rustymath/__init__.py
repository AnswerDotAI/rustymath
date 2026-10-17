"""Maths in plain Rust, from fast f16, f32 and f64 functions to exact rationals.

Modules:

- `rustymath.optimize`: Minimizing a function of several variables, and solving equations, nonlinear or linear, with SciPy's names. The searches run in Rust and call the Python functions at each point."""

import sys
from . import _core, linalg, polynomial, exact, integer, search, optimize, random
from .linalg import matmul
from .integer import encode, decode
from .search import (Unique, unique, index, isin, argsort, lexsort, searchsorted, bins, bincount, counts, group_indices, sort, partition, argpartition,
    argmin, argmax, median, quantile, percentile, union1d, intersect1d, setdiff1d)
from ._core import (__version__, CT, Array, absolute, add, all, angle, any, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, array, betainc,
    betaincinv, binom, cbrt, ceil, cis, cos, cosh, cumprod, cumsum, distribution, divide, equal, erf, erfc, erfcinv, erfinv, exp, expm1, factorial, floor,
    gamma, gammainc, gammaincc, gammaincinv, gammaln, gcd, greater, greater_equal, hypot, lambertw, lcm, less, less_equal, log, log10, log1p, log2, logb,
    loggamma, logical_and, logical_or, logical_scan, logical_xor, logical_xnor, max, maximum, min, minimum, minmax, multiply, not_equal, number_theory, power,
    prod, remainder, root, set_threads, sign, sin, sinh, sqrt, sqrt1mx2, sqrt1px2, sqrtx2m1, subtract, sum, tan, tanh, threads, thread_setting)
from .create import arange, finfo, full, full_like, iinfo, issubdtype, linspace, logspace, ones, ones_like, result_type, zeros, zeros_like
from ._core import clip, conj, imag, isfinite, isinf, isnan, real, rint, trunc
from ._core import bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift, floor_divide, round, degrees, radians
from ._core import average, corrcoef, cov, histogram, mean, norm, std, var
from ._core import (argwhere, concatenate, expand_dims, flatten, flip, nonzero, pad, ravel, repeat, reshape, roll, split, squeeze, stack, swapaxes, take,
    transpose, where)
from .structure import diag, diagonal, eye, identity, tril, triu, trace, diff, ptp, broadcast_to
from ._naming import _binary_name
Generator, default_rng = random.Generator, random.default_rng

def accumulate(f, x, axis=None, start=None):
    "Left-to-right running results of a native binary function, along the flattened input or one axis by position/name. A scalar start precedes each lane without adding an output item. Comparisons retain the input's computing dtype, reading each later result as 0 or 1; ordinary arithmetic promotion and overflow rules apply"
    return _core._accumulate(_binary_name(f), x, axis, start)

for _m in (distribution, number_theory): sys.modules[f'{__name__}.{_m.__name__}'] = _m

