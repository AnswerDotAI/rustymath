"""Fast maths functions for f16, f32 and f64 in plain Rust.

Modules:

- `rustymath.linalg`: rustymath's linear algebra, under the names of NumPy's similar functions. Each function takes a stack of matrices in the last two axes of its arguments, and broadcasts the axes before them as NumPy's generalized ufuncs do. float32, float64, complex64 and complex128 matrices compute at their own width. float16 matrices compute in float32, with each result rounded to float16. Integers and Booleans compute as float64, except in `matmul` and `semiring`, where integers compute in int64."""

import sys
from typing import NamedTuple
import numpy as np
from . import _core, linalg, polynomial
from .linalg import matmul
from .polynomial import _dtype
from ._core import (__version__, CT, absolute, add, angle, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, betainc, betaincinv, binom,
    cbrt, cos, cosh, distribution, erf, erfc, erfcinv, erfinv, exp, expm1, gamma, gammainc, gammaincc, gammaincinv, gammaln, hypot,
    lambertw, log, log10, log1p, log2, loggamma, maximum, minimum, multiply, number_theory, power, root, sign, sin, sinh, sqrt, sqrt1mx2,
    sqrt1px2, sqrtx2m1, tan, tanh)

abs = absolute

for _m in (distribution, number_theory): sys.modules[f'{__name__}.{_m.__name__}'] = _m

def _check(ct):
    if not 0 <= ct < 1: raise ValueError(f"ct must be at least 0 and below 1, not {ct}")
    return ct

def _comparison(name, relation):
    "A wrapper for the tolerant comparison ufunc `name` that checks `ct`"
    ufunc = getattr(_core, name)
    def compare(
        x1, x2, # Arrays to compare
        ct=CT, # Relative tolerance: two items are equal when `|x1-x2|` is at most `ct` times the larger magnitude
        **kwargs # Passed to the ufunc, such as `out=`
    ): return ufunc(x1, x2, _check(ct), **kwargs)
    compare.__name__ = compare.__qualname__ = name
    compare.__doc__ = f"Whether `x1` {relation} `x2`, with floats equal within tolerance `ct` as APL compares them, and integers compared exactly"
    return compare

equal = _comparison('equal', "equals")
not_equal = _comparison('not_equal', "doesn't equal")
less = _comparison('less', "is below and doesn't equal")
less_equal = _comparison('less_equal', "is below or equals")
greater = _comparison('greater', "is above and doesn't equal")
greater_equal = _comparison('greater_equal', "is above or equals")

def floor(
    x, # Array to round down
    ct=CT, # Relative tolerance: an item within `ct` of the next whole number rounds to it
    **kwargs # Passed to the ufunc, such as `out=`
):
    "The largest whole number at most `x`, within tolerance `ct`"
    return _core.floor(x, _check(ct), **kwargs)

def ceil(
    x, # Array to round up
    ct=CT, # Relative tolerance: an item within `ct` of the whole number below rounds to it
    **kwargs # Passed to the ufunc, such as `out=`
):
    "The smallest whole number at least `x`, within tolerance `ct`"
    return _core.ceil(x, _check(ct), **kwargs)

def remainder(
    x1, x2, # Dividends and divisors
    ct=CT, # Relative tolerance for floats: a quotient within `ct` of a whole number counts as whole, giving 0
    **kwargs # Passed to the ufunc, such as `out=`
):
    "The remainder of `x1` divided by `x2`, with the sign of `x2`, as `numpy.remainder` gives it, within tolerance `ct`. A zero `x2` gives `x1`, by APL's rule."
    return _core.remainder(x1, x2, _check(ct), **kwargs)

mod = remainder

def gcd(
    x1, x2, # Arrays of integers, floats or complex numbers
    ct=CT, # Relative tolerance for each remainder step of floats and complex numbers
    **kwargs # Passed to the ufunc, such as `out=`
):
    "The greatest common divisor of `x1` and `x2`, never negative for integers and floats, within tolerance `ct`"
    return _core.gcd(x1, x2, _check(ct), **kwargs)

def lcm(
    x1, x2, # Arrays of integers or floats
    ct=CT, # Relative tolerance for each remainder step of floats
    **kwargs # Passed to the ufunc, such as `out=`
):
    "The least common multiple of `x1` and `x2`, with the sign of `x1 x2`, within tolerance `ct`"
    return _core.lcm(x1, x2, _check(ct), **kwargs)

def _searched(*arrays):
    "`arrays` as flat contiguous vectors of one dtype: the Boolean or integer dtype that NumPy would compute them in, or else its float dtype, at least float16"
    dt = np.result_type(*(a if isinstance(a, (bool, int, float)) else np.asarray(a) for a in arrays))
    if dt.kind not in 'biu': dt = _dtype(*arrays)
    if dt.kind not in 'biuf': raise TypeError(f"searches take Booleans, integers or real numbers, not {dt}")
    return [np.ascontiguousarray(np.ravel(a), dt) for a in arrays]

class Unique(NamedTuple):
    values: np.ndarray
    firsts: np.ndarray
    numbers: np.ndarray
    counts: np.ndarray

def unique(
    ar, # Array of any shape, flattened first
    *, ct=CT, # Relative tolerance: each run of floats that match within `ct` counts as one value. Booleans and integers match exactly
):
    "The unique values of `ar` in order of first occurrence, each the first item of its class, with each value's first position, each item's class number and each class's size"
    ar = np.ravel(ar)
    [x] = _searched(ar)
    firsts, numbers, counts = _core.unique(x, _check(ct))
    return Unique(ar[firsts], firsts, numbers, counts)

def index(
    x, # Array to search, flattened first
    y, # A number, or an array of any shape, of values to find
    *, ct=CT, # Relative tolerance for floats: an item matches a value when they differ by at most `ct` times the larger magnitude
):
    "Where in the flattened `x` each item of `y` first matches, or the size of `x` where nothing does, shaped like `y`. Booleans and integers match exactly"
    fx, fy = _searched(x, y)
    return _core.index_of(fx, fy, _check(ct)).reshape(np.shape(y))[()]

def isin(
    element, # A number, or an array of any shape
    test_elements, # Values to test against, flattened first
    *, ct=CT, # Relative tolerance, as for `index`
):
    "Whether each item of `element` matches an item of `test_elements` within tolerance `ct`, shaped like `element`"
    return np.asarray(index(test_elements, element, ct=ct) < np.size(test_elements))
