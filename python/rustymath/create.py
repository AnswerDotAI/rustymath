"""Compact array constructors and dtype facts from the shared Rust core."""
from collections import namedtuple
from . import _core
from ._core import arange, linspace, logspace, issubdtype, result_type

FloatInfo = namedtuple('FloatInfo', 'bits eps tiny min max')
IntInfo = namedtuple('IntInfo', 'bits min max')

def full(shape, value, dtype=None):
    "An array of `shape` filled with one value"
    return _core._full(shape, value, dtype)

def zeros(shape, dtype='float64'):
    "An array filled with the dtype's zero, or spaces for characters"
    return _core._full(shape, None, dtype)

def ones(shape, dtype='float64'):
    "An array filled with ones"
    return _core._full(shape, 1, dtype)

def full_like(x, value, dtype=None):
    "Fill an array with x's shape, keys and names; its dtype defaults to x's"
    return _core._full(None, value, dtype, like=x)

def zeros_like(x, dtype=None):
    "An array of zeros or spaces, retaining x's layout and default dtype"
    return _core._full(None, None, dtype, like=x)

def ones_like(x, dtype=None):
    "An array of ones, retaining x's layout and default dtype"
    return _core._full(None, 1, dtype, like=x)

def finfo(dtype):
    "Precision and finite range of a float dtype, or a complex dtype's components"
    return FloatInfo(*_core._finfo(dtype))

def iinfo(dtype):
    "Finite range of an integer dtype"
    return IntInfo(*_core._iinfo(dtype))
