"""Numeric search, stable sorting and grouping. Sorts and insertion searches compare exactly: signed zeros compare equal, and every NaN comes after positive infinity. Classification matches floats within the tolerance `ct`, and Booleans, integers and characters exactly. With no `axis`, `argsort` and `lexsort` sort the flattened array."""
from typing import NamedTuple
from . import _core
from ._core import Array, index, isin, argsort, lexsort, searchsorted, bincount, counts, group_indices
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
