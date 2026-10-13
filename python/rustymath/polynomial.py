"""Polynomials of one variable, under the names of NumPy's similar functions in `numpy.polynomial.polynomial`. A polynomial is the array of its coefficients in ascending order: `c` stands for `c[0] + c[1]*x + c[2]*x**2 + ...`."""
import numpy as np
from . import _core
from ._checks import _radix

def _dtype(*arrays):
    "The float or complex dtype that `arrays` compute in, at least float16"
    return np.result_type(*(a if isinstance(a, (int, float, complex)) else np.asarray(a) for a in arrays), np.float16)

def _working(dt):
    "The dtype for generic field operations, widening float16 to float32"
    return np.dtype(np.float32) if dt == np.float16 else dt

def _coefficient(m, dt):
    "The scalar leading coefficient in the working dtype"
    if np.ndim(m): raise ValueError("the leading coefficient must be scalar")
    return np.asarray(m, dt)

def polyval(
    x, # A number, or an array of any shape
    c, # Real or complex coefficients in ascending order
):
    "The polynomial `c` at each item of `x`, by Horner's rule"
    c = np.asarray(c)
    dt = _dtype(x, c)
    y = np.array(x, dt, order='C')
    _core.polyval(y, np.ascontiguousarray(c, dt))
    return y[()]

def polyvalfromroots(
    x, # A number, or an array of any shape
    roots, # Real or complex roots
    m=1, # The leading coefficient
):
    "`m` times the product of `x - r` over the roots `r`, without forming the coefficients"
    dt = _dtype(x, roots, m)
    wd = _working(dt)
    y = np.array(x, wd, order='C')
    _core.polyvalfromroots(y, np.ascontiguousarray(roots, wd), _coefficient(m, wd))
    return y.astype(dt, copy=False)[()]

def polyroots(
    c, # Real or complex coefficients in ascending order
):
    "The roots of `c` as complex numbers, sorted by real part, then imaginary part, from the eigenvalues of its companion matrix. With real coefficients, a real root has an imaginary part of exactly 0. Where the eigenvalue solver doesn't converge, each root is NaN and NumPy reports an invalid value"
    c = np.asarray(c)
    return _core.polyroots(np.ascontiguousarray(c, np.complex128)).astype(np.result_type(_dtype(c), np.complex64))

def polyfromroots(
    roots, # Real or complex roots
    m=1, # The leading coefficient
):
    "The polynomial with roots `roots` and leading coefficient `m`"
    dt = _dtype(roots, m)
    wd = _working(dt)
    return _core.polyfromroots(np.ascontiguousarray(roots, wd), _coefficient(m, wd)).astype(dt, copy=False)

def polyder(
    c, # Real or complex coefficients in ascending order
):
    "The derivative of `c`. A constant's derivative is empty, which is the zero polynomial"
    dt = _dtype(c)
    return _core.polyder(np.ascontiguousarray(c, _working(dt))).astype(dt, copy=False)

def decode(
    bases, # A nonempty vector of real or complex bases; its first item is unused
    digits, # An array whose last dimension has one digit per base
):
    "The value of each last-axis digit vector in the mixed radix `bases`, by Horner's rule"
    bases, digits = np.asarray(bases), np.asarray(digits)
    _radix(bases, digits)
    dt = _dtype(bases, digits)
    wd = _working(dt)
    return _core.polydecode(np.ascontiguousarray(bases, wd), np.ascontiguousarray(digits, wd).ravel()).astype(dt, copy=False).reshape(digits.shape[:-1])[()]

def polyfit(
    x, # The points' x coordinates
    y, # The points' y coordinates, one for each x
    deg, # The degree of the polynomial
):
    "The coefficients, in ascending order, of the polynomial of degree `deg` that fits the points by least squares. Where fewer than `deg + 1` distinct points fix it, it's the fit whose coefficients have the least norm"
    if np.shape(x) != np.shape(y): raise ValueError("polyfit needs one y for each x")
    dt = _dtype(x, y)
    if dt.kind == 'c': raise TypeError("polyfit requires real points")
    return _core.polyfit(np.ascontiguousarray(x, dt), np.ascontiguousarray(y, dt), deg)
