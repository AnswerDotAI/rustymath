"""Fast maths functions as NumPy ufuncs, plus APL's tolerant comparison, special functions, polynomials, probability distributions and
number theory. The ufuncs take float16, float32 and float64 arrays, complex64 and complex128 arrays where the function has complex
values, and int64 arrays where NumPy's version does.

Functions that NumPy or SciPy also has keep their names, so code switches by changing an import. Every result is within 4 units in the
last place. `add`, `multiply`, `fmax` and `fmin` sum, multiply and compare as NumPy's do, and their `reduce` and `accumulate` run
rustymath's vectorized sums, products, maximums and minimums. `polynomial` has the functions of `numpy.polynomial.polynomial`,
`distribution` has the distributions, and `number_theory` works on Python integers of any size."""
import sys
import numpy as np
from . import _core, polynomial
from .polynomial import _dtype
from ._core import (__version__, CT, absolute, add, angle, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, betainc, betaincinv, binom,
    cbrt, cos, cosh, distribution, erf, erfc, erfcinv, erfinv, exp, expm1, fmax, fmin, gamma, gammainc, gammaincc, gammaincinv, gammaln,
    hypot, lambertw, log, log10, log1p, log2, loggamma, multiply, number_theory, power, root, sign, sin, sinh, sqrt, sqrt1mx2, sqrt1px2, sqrtx2m1,
    tan, tanh)

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
    compare.__doc__ = f"Whether `x1` {relation} `x2`, with items equal within tolerance `ct` as APL compares them"
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
    "The remainder of `x1` divided by `x2`, with the sign of `x2`, as `numpy.remainder` gives it, within tolerance `ct`"
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
    "The least common multiple of `x1` and `x2`, never negative, as `numpy.lcm` gives it, within tolerance `ct`"
    return _core.lcm(x1, x2, _check(ct), **kwargs)

def _floats(*arrays):
    "`arrays` as flat contiguous vectors of the float dtype that NumPy would compute them in, at least float16"
    dt = _dtype(*arrays)
    if dt.kind != 'f': raise TypeError(f"tolerant searches take real numbers, not {dt}")
    return [np.ascontiguousarray(np.ravel(a), dt) for a in arrays]

def unique(
    ar, # Array of any shape, flattened first
    return_index=False, # Also return where each unique value first occurs in the flattened `ar`?
    return_inverse=False, # Also return, shaped like `ar`, where each item's value is in the result?
    return_counts=False, # Also return how many items of `ar` each unique value stands for?
    *, ct=CT, # Relative tolerance: each run of floats that match within `ct` counts as one value
):
    "The sorted unique values of `ar` within tolerance `ct`, each the first item of its class, with the extras `numpy.unique` gives"
    ar = np.asarray(ar)
    flat, ct = ar.ravel(), _check(ct)
    [x] = _floats(flat)
    classes = _core.classes(x, ct) if return_inverse or return_counts else None
    first = np.flatnonzero(_core.firsts(x, ct) if classes is None else classes == np.arange(len(classes)))
    first = first[np.argsort(x[first], kind='stable')]
    res = [flat[first]]
    if return_index: res.append(first)
    if classes is not None:
        rank = np.empty(len(flat), np.intp)
        rank[first] = np.arange(len(first))
        inverse = rank[classes]
        if return_inverse: res.append(inverse.reshape(ar.shape))
        if return_counts: res.append(np.bincount(inverse, minlength=len(first)))
    return res[0] if len(res) == 1 else tuple(res)

def index(
    x, # Array to search, flattened first
    y, # A number, or an array of any shape, of values to find
    *, ct=CT, # Relative tolerance: an item matches a value when they differ by at most `ct` times the larger magnitude
):
    "Where in the flattened `x` each item of `y` first matches within tolerance `ct`, or the size of `x` where nothing does, shaped like `y`"
    fx, fy = _floats(x, y)
    return _core.index_of(fx, fy, _check(ct)).reshape(np.shape(y))[()]

def isin(
    element, # A number, or an array of any shape
    test_elements, # Values to test against, flattened first
    *, invert=False, # Return whether each item matches none of `test_elements`?
    ct=CT, # Relative tolerance, as for `index`
):
    "Whether each item of `element` matches an item of `test_elements` within tolerance `ct`, shaped like `element`, as `numpy.isin` gives it"
    return np.asarray((index(test_elements, element, ct=ct) < np.size(test_elements)) != invert)
