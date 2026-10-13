"""Numeric search, stable sorting and grouping. Sorts and insertion searches compare exactly: signed zeros compare equal, and all NaNs compare equal after positive infinity. Classification uses the same tolerant matching as `unique`."""
from typing import NamedTuple
import numpy as np
from . import _core
from ._core import CT
from ._checks import _check
from .polynomial import _dtype

def _searched(*arrays):
    "`arrays` as flat contiguous vectors of one Boolean, integer or real dtype"
    dt = np.result_type(*(a if isinstance(a, (bool, int, float)) else np.asarray(a) for a in arrays))
    if dt.kind not in 'biu': dt = _dtype(*arrays)
    if dt.kind not in 'biuf': raise TypeError(f"searches take Booleans, integers or real numbers, not {dt}")
    return [np.ascontiguousarray(np.ravel(a), dt) for a in arrays]

class Unique(NamedTuple):
    values: np.ndarray
    firsts: np.ndarray
    numbers: np.ndarray
    counts: np.ndarray

def unique(ar, *, ct=CT):
    "The unique values of flattened `ar` in first-occurrence order, with first positions, class numbers and class sizes"
    ar = np.ravel(ar)
    [x] = _searched(ar)
    firsts, numbers, counts = _core.unique(x, _check(ct))
    return Unique(ar[firsts], firsts, numbers, counts)

def index(x, y, *, ct=CT):
    "Where in flattened `x` each item of `y` first matches, or the size of `x` where none does, shaped like `y`"
    fx, fy = _searched(x, y)
    return _core.index_of(fx, fy, _check(ct)).reshape(np.shape(y))[()]

def isin(element, test_elements, *, ct=CT):
    "Whether each item of `element` matches an item of flattened `test_elements`, shaped like `element`"
    return np.asarray(index(test_elements, element, ct=ct) < np.size(test_elements))

def counts(x, *, ct=CT):
    "The size of each class of flattened `x`, in first-occurrence order, without returning each item's class number"
    [x] = _searched(x)
    return _core.class_counts(x, _check(ct))

def group_indices(x, *, ct=CT):
    "A list of flat-position arrays, one per class in first-occurrence order; positions within each class ascend"
    [x] = _searched(x)
    return _core.group_indices(x, _check(ct))

def argsort(a, axis=-1):
    "Stable indices sorting a Boolean, integer or real array along `axis`; `axis=None` sorts the flattened array"
    if axis is None: a, axis = np.ravel(a), -1
    return lexsort([a], axis)

def lexsort(keys, axis=-1):
    "Stable indices sorting equally shaped numeric `keys` along `axis`, with the last key primary, as in NumPy"
    keys = [np.asarray(k) for k in keys]
    if not keys or any(k.shape != keys[0].shape for k in keys): raise ValueError("lexsort needs one or more equally shaped keys")
    keys = [np.moveaxis(k, axis, -1) for k in keys]
    vectors = [_searched(k)[0] for k in keys]
    shape = keys[0].shape
    return np.moveaxis(_core.sort_indices(vectors, shape[-1]).reshape(shape), -1, axis)

def searchsorted(a, v, side='left'):
    "Insertion positions of `v` in ascending vector `a`, comparing exactly; `side` chooses before or after equal values"
    if np.ndim(a) != 1: raise ValueError("searchsorted needs a one-dimensional sorted array")
    if side not in ('left', 'right'): raise ValueError("side must be 'left' or 'right'")
    a, values = _searched(a, v)
    return _core.searchsorted(a, values, side == 'right').reshape(np.shape(v))[()]

def bins(boundaries, values):
    "The number of ascending `boundaries` at or below each value, as `searchsorted(..., side='right')` gives it"
    return searchsorted(boundaries, values, side='right')

def bincount(x, minlength=0):
    "Counts a vector of nonnegative integers, indexed from zero, padded to `minlength`; weighted counts are not supported"
    x = np.asarray(x)
    if x.ndim != 1 or x.dtype.kind not in 'biu': raise TypeError("bincount needs a vector of integers")
    if minlength < 0: raise ValueError("minlength must be nonnegative")
    return _core.bincount(np.ascontiguousarray(x), minlength)
