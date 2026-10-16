"""Numeric search, stable sorting and grouping. Sorts and insertion searches compare exactly: signed zeros compare equal, and every NaN comes after positive infinity. Classification matches floats within the tolerance `ct`, and Booleans, integers and characters exactly. With no `axis`, `argsort` and `lexsort` sort the flattened array."""
from typing import NamedTuple
from . import _core
from ._core import Array, index, isin, sort, argsort, partition, argpartition, argmin, argmax, median, quantile
from ._core import lexsort, searchsorted, bincount, counts, group_indices
from ._naming import _named

class Unique(NamedTuple):
    values: Array
    firsts: Array
    numbers: Array
    counts: Array

unique = _named(lambda ar, *, ct=None: Unique(*_core.unique(ar, ct=ct)), 'unique', _core.unique.__doc__)

def bins(boundaries, values):
    "The number of ascending `boundaries` at or below each value, as `searchsorted(..., side='right')` gives it"
    return searchsorted(boundaries, values, side='right')

def percentile(x, q, axis=None):
    "Quantiles with probabilities in percent rather than fractions"
    return quantile(x, _core.divide(q, 100), axis)

def union1d(x, y, *, ct=None):
    "Unique flattened values in x then y, in first-occurrence order"
    return _core._set(x, y, 'union', ct)

def intersect1d(x, y, *, ct=None):
    "Unique flattened values of x that match y, in first-occurrence order"
    return _core._set(x, y, 'intersection', ct)

def setdiff1d(x, y, *, ct=None):
    "Unique flattened values of x absent from y, in first-occurrence order"
    return _core._set(x, y, 'difference', ct)
