"""Maths in plain Rust, from fast f16, f32 and f64 functions to exact rationals."""

import sys
import numpy as np
from . import _core, linalg, polynomial, exact, integer, search, optimize
from .linalg import matmul
from .integer import encode, decode
from .search import Unique, unique, index, isin, argsort, lexsort, searchsorted, bins, bincount, counts, group_indices
from ._checks import _check, _named
from ._core import (__version__, CT, absolute, add, angle, arccos, arccosh, arcsin, arcsinh, arctan, arctan2, arctanh, betainc, betaincinv, binom,
    cbrt, cis, cos, cosh, distribution, divide, erf, erfc, erfcinv, erfinv, exp, expm1, factorial, gamma, gammainc, gammaincc, gammaincinv, gammaln, hypot,
    lambertw, log, log10, log1p, log2, logb, loggamma, logical_and, logical_or, logical_xor, logical_xnor, maximum, minimum, minmax, multiply,
    number_theory, power, root, set_threads, sign, sin, sinh, sqrt, sqrt1mx2, sqrt1px2, sqrtx2m1, subtract, tan, tanh, threads, thread_setting)

for _m in (distribution, number_theory): sys.modules[f'{__name__}.{_m.__name__}'] = _m

def _tolerant(name, doc):
    "A wrapper that checks `ct` and passes it to the tolerant ufunc `name` after the other arguments"
    ufunc = getattr(_core, name)
    def unary(
        x, # Array
        ct=CT, # Relative tolerance, at least 0 and below 1
        **kwargs # Passed to the ufunc, such as `out=`
    ): return ufunc(x, _check(ct), **kwargs)
    def binary(
        x1, x2, # Arrays
        ct=CT, # Relative tolerance, at least 0 and below 1
        **kwargs # Passed to the ufunc, such as `out=`
    ): return ufunc(x1, x2, _check(ct), **kwargs)
    return _named(unary if ufunc.nin == 2 else binary, name, doc)

def _comparison(name, relation):
    "A `_tolerant` wrapper for the comparison ufunc `name`. Its docstring says it tests whether `x1` `relation` `x2`"
    return _tolerant(name, f"Whether `x1` {relation} `x2`. Floats are equal when `|x1-x2|` is at most `ct` times the larger magnitude, as APL compares them. Integers compare exactly")

equal = _comparison('equal', "equals")
not_equal = _comparison('not_equal', "doesn't equal")
less = _comparison('less', "is below and doesn't equal")
less_equal = _comparison('less_equal', "is below or equals")
greater = _comparison('greater', "is above and doesn't equal")
greater_equal = _comparison('greater_equal', "is above or equals")

floor = _tolerant('floor', "The largest whole number at most `x`. An item within relative tolerance `ct` of the next whole number rounds to it")
ceil = _tolerant('ceil', "The smallest whole number at least `x`. An item within relative tolerance `ct` of the whole number below rounds to it")
remainder = _tolerant('remainder', "The remainder of `x1` divided by `x2`, with the sign of `x2`, as `numpy.remainder` gives it. For floats, a quotient within relative tolerance `ct` of a whole number counts as whole, giving 0. A zero `x2` gives `x1`, by APL's rule.")
gcd = _tolerant('gcd', "The greatest common divisor of integers, floats or complex numbers `x1` and `x2`, never negative for integers and floats. Each remainder step of floats and complex numbers uses relative tolerance `ct`")
lcm = _tolerant('lcm', "The least common multiple of integers or floats `x1` and `x2`, with the sign of `x1 x2`. Each remainder step of floats uses relative tolerance `ct`")

def logical_scan(x, table, axis=-1, start=None):
    "The running results along an axis of the Boolean function given by table[running_result][next_item]"
    return _core.logical_scan(np.asarray(x, dtype=bool), np.asarray(table, dtype=bool).tolist(), axis, start)

