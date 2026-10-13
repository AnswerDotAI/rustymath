"""Exact rational arithmetic on integers and fractions.Fraction. Arrays have dtype=object; scalar results are Fractions. Elementwise functions broadcast. Matrix functions take one 2-D matrix, not stacks, and solve accepts a vector or matrix RHS."""
import numpy as np
from ._core import exact as _exact
from .linalg import Echelon
from ._checks import _named, _radix

def _array(a): return np.asarray(a, dtype=object)
def _values(a): return a.ravel().tolist()
def _result(values, shape): return _array(values).reshape(shape)[()]

def _vector(a):
    a = _array(a)
    if a.ndim != 1: raise ValueError("expected a one-dimensional array")
    return a

def _scalar(a):
    a = _array(a)
    if a.ndim: raise ValueError("expected a scalar")
    return a.item()

def _matrix(a, square=False):
    a = _array(a)
    if a.ndim != 2: raise ValueError("expected a two-dimensional matrix")
    if square and a.shape[0] != a.shape[1]: raise ValueError("expected a square matrix")
    return a

def _binary(f, doc):
    "An elementwise function of two arrays, broadcast together, that calls `f` on their flat values"
    def g(x, y):
        x, y = np.broadcast_arrays(_array(x), _array(y))
        return _result(f(_values(x), _values(y)), x.shape)
    return _named(g, f.__name__, doc)

gcd = _binary(_exact.gcd, "The nonnegative rational gcd of x and y")
lcm = _binary(_exact.lcm, "The rational lcm, with the sign of x*y")
remainder = _binary(_exact.remainder, "x modulo y, with y's sign; a zero y gives x")
binom = _binary(_exact.binom, "The exact binomial coefficient x choose y, for rational x and whole y fitting int64")

def solve(a, b):
    "The least-norm least-squares solution of a @ x = b, exactly over the rationals"
    a, b = _matrix(a), _array(b)
    m, n = a.shape
    if b.ndim not in (1, 2) or b.shape[0] != m: raise ValueError("b must be a vector or matrix with one row per row of a")
    k = 1 if b.ndim == 1 else b.shape[1]
    return _result(_exact.solve(_values(a), _values(b), m, n, k), (n,) if b.ndim == 1 else (n, k))

def pinv(a):
    "The exact pseudoinverse of a; its inverse when square and nonsingular"
    a = _matrix(a)
    return _result(_exact.pinv(_values(a), *a.shape), a.shape[::-1])

def _square(f, doc):
    "A function of one square matrix that calls `f` on its flat values and size"
    def g(a):
        a = _matrix(a, square=True)
        return _result(f(_values(a), a.shape[0]), ())
    return _named(g, f.__name__, doc)

det = _square(_exact.det, "The exact determinant of a square matrix")
permanent = _square(_exact.permanent, "The permanent of a square matrix, in exponential time; Ctrl-C cancels it")

def echelon(a, columns=None):
    "Reduced row echelon form, omitting zero rows, with pivot columns and the signed product of unscaled pivots"
    a = _matrix(a)
    rows, pivots, determinant = _exact.echelon(_values(a), *a.shape, a.shape[1] if columns is None else columns)
    return Echelon(_result(rows, (len(pivots), a.shape[1])), np.asarray(pivots, dtype=np.intp), determinant[0])

def polyval(x, c):
    "The polynomial c at x; coefficients are one-dimensional, in ascending order"
    x, c = _array(x), _vector(c)
    return _result(_exact.polyval(_values(x), _values(c)), x.shape)

def polyfromroots(roots, m=1):
    "The coefficients of m times the product of (x - root)"
    return _array(_exact.polyfromroots(_values(_vector(roots)), _scalar(m)))

def polyder(c):
    "The derivative's coefficients; a constant's derivative is empty"
    return _array(_exact.polyder(_values(_vector(c))))

def polyvalfromroots(x, roots, m=1):
    "m times the product of (x - root), at each x"
    x, roots = _array(x), _vector(roots)
    return _result(_exact.polyvalfromroots(_values(x), _values(roots), _scalar(m)), x.shape)

def decode(bases, digits):
    "Mixed-radix values of digits along their last axis; bases is a nonempty vector and its first item is unused"
    bases, digits = _array(bases), _array(digits)
    _radix(bases, digits)
    return _result(_exact.decode(_values(bases), _values(digits)), digits.shape[:-1])
