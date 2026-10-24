"""Maths in plain Rust, from fast f16, f32 and f64 functions to exact rationals.

Rustymath provides fast elementary and special functions, statistics, searching and sorting, random sampling, linear algebra, optimization, polynomials and exact arithmetic in Rust and Python. It is the numerical core of [BasedPL](https://answerdotai.github.io/basedpl/).

The Python package needs no NumPy. Its eager operations use the same broadcasting, selection and tolerant-comparison rules as the shared Rust core. For richer array notation, operators and fusion, use BasedPL.

Float functions target four units in the last place (ulps), with [documented exceptions](accuracy.qmd). Rustymath prioritizes speed over last-bit agreement and reproducibility.

[Arrays and indexing](https://answerdotai.github.io/rustymath/arrays.html) covers construction, dtypes, selection, reshaping, axis names, keys and interoperability.

[Maths and statistics](https://answerdotai.github.io/rustymath/math.html) covers elementary and special functions, reductions, scans, statistics and tolerant comparison.

[Searching, sorting and selection](https://answerdotai.github.io/rustymath/search.html) distinguishes tolerant matching from exact ordering.

[Random numbers and distributions](https://answerdotai.github.io/rustymath/random.html) covers sampling and probability functions.

[Linear algebra](https://answerdotai.github.io/rustymath/linalg.html) covers matrix products, reusable factorizations, decompositions and least squares.

[Optimization and iterative solves](https://answerdotai.github.io/rustymath/optimize.html) also covers matrix-free linear systems.

[Polynomials, exact arithmetic and number theory](https://answerdotai.github.io/rustymath/exact.html) includes rational matrix solves and calculations with `fractions.Fraction`.

The Rust page's [Accuracy and speed](https://answerdotai.github.io/rustymath/rust.html#accuracy-and-speed) section compares each function's speed with Rust's standard library. It also explains how the 4-ulp bound holds for `f16`, with and without fused multiply-add, and for `pow` and `root`.

Modules:

- `rustymath.optimize`: Minimizing a function of several variables, and solving equations, nonlinear or linear, with SciPy's names. The searches run in Rust and call the Python functions at each point."""

import sys
from . import _core, linalg, polynomial, exact, integer, search, optimize, random
from .linalg import matmul
from .integer import encode, decode
from .search import (Unique, unique, index, isin, argsort, lexsort, searchsorted, bins, bincount, counts, group_indices, sort, partition, argpartition,
    argmin, argmax, median, quantile, percentile, union1d, intersect1d, setdiff1d, find)
from ._core import (__version__, Array, absolute, add, all, angle, any, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, array, betainc,
    betaincinv, binom, binom_dk, binom_dn, cbrt, ceil, cis, cos, cosh, cumprod, cumsum, digamma, distribution, divide, equal, erf, erfc, erfcinv, erfinv, exp, expm1, factorial, floor,
    gamma, gammainc, gammaincc, gammaincinv, gammaln, gcd, greater, greater_equal, hypot, lambertw, lcm, less, less_equal, log, log10, log1p, log2, logb,
    loggamma, logical_and, logical_or, logical_scan, logical_xor, logical_xnor, max, maximum, min, minimum, minmax, multiply, not_equal, number_theory, power,
    prod, remainder, root, set_threads, sign, sin, sinh, sqrt, sqrt1mx2, sqrt1px2, sqrtx2m1, subtract, sum, tan, tanh, threads, thread_setting, timing)
from .create import arange, finfo, full, full_like, iinfo, issubdtype, linspace, logspace, ones, ones_like, result_type, zeros, zeros_like
from ._core import clip, conj, imag, isfinite, isinf, isnan, real, rint, trunc
from ._core import bitwise_and, bitwise_or, bitwise_xor, bitwise_not, left_shift, right_shift, floor_divide, round, degrees, radians
from ._core import average, corrcoef, cov, histogram, mean, norm, std, var
from ._core import (argwhere, concatenate, expand_dims, flatten, flip, nonzero, pad, ravel, repeat, reshape, roll, split, squeeze, stack, swapaxes, take,
    transpose, where)
from .structure import diag, diagonal, eye, identity, tril, triu, trace, diff, ptp, broadcast_to
from .signal import correlate, convolve, overlap_add
from ._naming import _binary_function
Generator, default_rng = random.Generator, random.default_rng

def accumulate(f, x, axis=None, start=None):
    "Left-to-right running results of a native binary function, along the flattened input or one axis by position/name. A scalar start precedes each lane without adding an output item. Comparisons retain the input's computing dtype, reading each later result as 0 or 1; ordinary arithmetic promotion and overflow rules apply"
    return _core._accumulate(_binary_function(f), x, axis, start)

def sliding(f, x, width, axis=-1):
    "Folds of each window of `width` neighbouring items along one axis, by position or name, by a native binary function with an associative reduction: `add`, `multiply`, `maximum`, `minimum` or a logical one. A lane of n items gives n - width + 1 folds, each keyed as the item that starts its window. A fold costs the same for each item whatever the width"
    return _core._sliding(_binary_function(f), x, width, axis)

def sliding_argmax(x, width, axis=-1):
    "The position along one axis, by position or name, of the first maximum of each window of `width` neighbouring items, as int64, keyed as `sliding` keys its folds. A window compares its items as `sliding(maximum, ...)` folds them, so it skips NaN, where `argmax` counts NaN as the largest"
    return _core._sliding_pick(x, width, axis, True)

def sliding_argmin(x, width, axis=-1):
    "The position of the first minimum of each window, as `sliding_argmax` gives the first maximum"
    return _core._sliding_pick(x, width, axis, False)

def at(f, a, indices, b):
    "A copy of `a` with `b`, broadcast over the selection that `indices` makes as brackets take it, combined into each selected item by the native binary function `f`. An item selected more than once combines every time, as NumPy's `ufunc.at` combines in place. `f` is `add`, `subtract`, `multiply`, `divide`, `maximum` or `minimum`."
    return _core._at(_binary_function(f), a, indices, b)
for _m in (distribution, number_theory): sys.modules[f'{__name__}.{_m.__name__}'] = _m
from . import mathish

