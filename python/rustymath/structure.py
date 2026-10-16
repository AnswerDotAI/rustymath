"""Conveniences over shared selection and compact array construction. Axis selectors are independent; broadcasting aligns from the front."""
from operator import index
from . import _core
from ._core import diagonal, broadcast_to

def diag(x, k=0):
    "Construct a matrix from a vector, or extract a matrix's diagonal"
    return _core._diag(x, k)

def eye(n, m=None, k=0, dtype='f64'):
    "An n-by-m matrix with ones on diagonal k"
    return _core._diag(1, k, (n, n if m is None else m), dtype)

def identity(n, dtype='f64'):
    "An n-by-n identity matrix"
    return eye(n, dtype=dtype)

def tril(x, k=0):
    "Keep the lower triangle of each matrix in the last two axes"
    return _core._triangle(x, k, False)

def triu(x, k=0):
    "Keep the upper triangle of each matrix in the last two axes"
    return _core._triangle(x, k, True)

def trace(x, offset=0, axis1=-2, axis2=-1):
    "Sum a diagonal of two axes; remaining axes keep their metadata"
    return _core.sum(diagonal(x, offset, axis1, axis2), axis=-1)

def diff(x, n=1, axis=-1):
    "Adjacent differences along an axis, repeated n times; Booleans give XOR"
    n = index(n)
    if n < 0: raise ValueError('difference order must be nonnegative')
    if not n: return _core.array(x)
    for _ in range(n): x = _core._difference(x, axis)
    return x

def ptp(x, axis=None):
    "Maximum minus minimum over the selected axes"
    lo, hi = _core.minmax(x, axis)
    return _core.subtract(hi, lo)
