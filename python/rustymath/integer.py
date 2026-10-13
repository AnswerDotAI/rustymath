"""Checked int64 mixed-radix conversions. For rational or complex bases, use exact.decode or polynomial.decode."""
import numpy as np
from . import _core
from ._checks import _radix

def _integers(a):
    a = np.asarray(a)
    if a.size and a.dtype.kind not in 'biu': raise TypeError("mixed-radix integer conversion takes integers")
    if a.dtype == np.uint64 and np.any(a > np.iinfo(np.int64).max): raise OverflowError("an input does not fit int64")
    return np.ascontiguousarray(a, dtype=np.int64)

def _bases(bases):
    bases = np.asarray(bases)
    if bases.ndim != 1: raise ValueError("bases must be a vector")
    return _integers(bases)

def encode(bases, values):
    "The mixed-radix digits of each integer, most significant first along a new last axis; a zero base takes the remaining value"
    bases, values = _bases(bases), np.asarray(values)
    return _core.encode(bases, _integers(values).ravel()).reshape((*values.shape, bases.size))

def decode(bases, digits):
    "The int64 value of each last-axis digit vector; the first base is unused. Overflow raises OverflowError"
    bases, digits = _bases(bases), np.asarray(digits)
    _radix(bases, digits)
    return _core.decode(bases, _integers(digits).ravel()).reshape(digits.shape[:-1])[()]
