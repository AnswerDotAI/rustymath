"""Numeric and character search, stable sorting and grouping. Sorts and insertion searches compare exactly: signed zeros compare equal, and every NaN comes after positive infinity. Matching and classification compare exactly; the Mathish namespace compares f64/complex128 tolerantly. Sorts use the last axis by default. Explicit `axis=None` flattens the input."""
from typing import NamedTuple
from . import _core
from ._core import Array, index, isin, sort, argsort, partition, argpartition, argmin, argmax, median, quantile
from ._core import find, lexsort, searchsorted, bincount, counts, group_indices
from ._naming import _named

class Unique(NamedTuple):
    values: Array
    firsts: Array
    numbers: Array
    counts: Array

unique = _named(lambda ar: Unique(*_core.unique(ar)), 'unique', _core.unique.__doc__)

def bins(boundaries, values):
    "The number of ascending `boundaries` at or below each value, as `searchsorted(..., side='right')` gives it"
    return searchsorted(boundaries, values, side='right')

def percentile(x, q, axis=None):
    "Quantiles with probabilities in percent rather than fractions"
    return quantile(x, _core.divide(q, 100), axis)

def union1d(x, y):
    "Unique flattened values in x then y, in first-occurrence order"
    return _core._set(x, y, 'union')

def intersect1d(x, y):
    "Unique flattened values of x that match y, in first-occurrence order"
    return _core._set(x, y, 'intersection')

def setdiff1d(x, y):
    "Unique flattened values of x absent from y, in first-occurrence order"
    return _core._set(x, y, 'difference')
