"""Polynomials of one variable, under the names of NumPy's similar functions in `numpy.polynomial.polynomial`. A polynomial is the array
of its coefficients in ascending order: `c` stands for `c[0] + c[1]*x + c[2]*x**2 + ...`."""
import numpy as np
from . import _core

def _dtype(*arrays):
    "The float or complex dtype that `arrays` compute in, at least float16"
    return np.result_type(*(a if isinstance(a, (int, float, complex)) else np.asarray(a) for a in arrays), np.float16)

def _contiguous(a, dt):
    "`a` as a contiguous array of `dt`, or of complex128 if `dt` is complex, as `_core` takes it"
    return np.ascontiguousarray(a, np.complex128 if dt.kind == 'c' else dt)

def polyval(
    x, # A number, or an array of any shape
    c, # Real coefficients in ascending order
):
    "The polynomial `c` at each item of `x`, by Horner's rule"
    c = np.asarray(c)
    dt = _dtype(x, c)
    y = np.array(x, dt, order='C')
    _core.polyval(y, _contiguous(c, dt))
    return y[()]

def polyroots(
    c, # Real or complex coefficients in ascending order
):
    "The roots of `c` as complex numbers, sorted by real part, then imaginary part, from the eigenvalues of its companion matrix. With real coefficients, a real root has an imaginary part of exactly 0. Where the eigenvalue solver doesn't converge, each root is NaN and NumPy reports an invalid value"
    c = np.asarray(c)
    return _core.polyroots(np.ascontiguousarray(c, np.complex128)).astype(np.result_type(_dtype(c), np.complex64))

def polyfromroots(
    roots, # Real or complex roots
):
    "The monic polynomial whose roots are `roots`"
    dt = _dtype(roots)
    return _core.polyfromroots(_contiguous(roots, dt)).astype(dt)

def polyder(
    c, # Real or complex coefficients in ascending order
):
    "The derivative of `c`. A constant's derivative is empty, which is the zero polynomial"
    dt = _dtype(c)
    return _core.polyder(_contiguous(c, dt)).astype(dt)

def polyfit(
    x, # The points' x coordinates
    y, # The points' y coordinates, one for each x
    deg, # The degree of the polynomial
):
    "The coefficients, in ascending order, of the polynomial of degree `deg` that fits the points by least squares. Where fewer than `deg + 1` distinct points fix it, it's the fit whose coefficients have the least norm"
    if np.shape(x) != np.shape(y): raise ValueError("polyfit needs one y for each x")
    dt = _dtype(x, y)
    return _core.polyfit(_contiguous(x, dt), _contiguous(y, dt), deg)
