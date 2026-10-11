"""Polynomials of one variable, under the names and argument orders of `numpy.polynomial.polynomial`, so code switches by changing an
import. A polynomial is the array of its coefficients in ascending order: `c` stands for `c[0] + c[1]*x + c[2]*x**2 + ...`."""
import numpy as np
from . import _core

def _dtype(*arrays):
    "The float or complex dtype that NumPy's polynomial functions compute `arrays` in, at least float16"
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
    "The roots of `c`, sorted, as the eigenvalues of its companion matrix. They're real when every root is real"
    c = np.asarray(c)
    real = np.finfo(_dtype(c)).dtype
    r = _core.polyroots(np.ascontiguousarray(c, np.complex128))
    if r is None: raise np.linalg.LinAlgError("the eigenvalue solver didn't converge")
    r = np.sort(r)
    return r.real.astype(real) if not r.imag.any() else r.astype(np.result_type(real, np.complex64))

def polyfromroots(
    roots, # Real or complex roots
):
    "The monic polynomial whose roots are `roots`"
    dt = _dtype(roots)
    return _core.polyfromroots(_contiguous(roots, dt)).astype(dt)

def polyder(
    c, # Real or complex coefficients in ascending order
    m=1, # How many times to differentiate
    scl=1, # A factor that each differentiation also multiplies by, for a change of variable
):
    "The `m`th derivative of `c`. As in NumPy, a derivative with no terms left is `[0]`"
    dt = _dtype(c)
    d = _contiguous(c, dt)
    for _ in range(m): d = _core.polyder(d)
    d = (d * scl**m).astype(dt)
    return d if len(d) else np.zeros(1, dt)

def polyfit(
    x, # The points' x coordinates
    y, # The points' y coordinates, one for each x
    deg, # The degree of the polynomial
):
    "The coefficients, in ascending order, of the polynomial of degree `deg` that fits the points by least squares"
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or x.shape != y.shape: raise ValueError("polyfit needs x and y as vectors of one length")
    if len(x) <= deg: raise ValueError(f"polyfit needs more than {deg} points for degree {deg}")
    dt = _dtype(x, y)
    return _core.polyfit(_contiguous(x, dt), _contiguous(y, dt), deg)
