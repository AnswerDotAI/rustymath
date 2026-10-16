"""Exact rational arithmetic on integers and fractions.Fraction. Arguments are integers, Fractions, nested lists of them, or arrays of integers. Results are Fractions, in nested lists for arrays. Elementwise functions pair their arguments' axes from the front, as rustymath broadcasts. Matrix functions take one matrix, not stacks."""
from ._core import exact as _exact
from .linalg import Echelon

gcd, lcm, remainder, binom = _exact.gcd, _exact.lcm, _exact.remainder, _exact.binom
solve, pinv, det, permanent = _exact.solve, _exact.pinv, _exact.det, _exact.permanent
polyval, polyfromroots, polyder, polyvalfromroots, decode = _exact.polyval, _exact.polyfromroots, _exact.polyder, _exact.polyvalfromroots, _exact.decode

def echelon(a, columns=None):
    "Reduced row echelon form, omitting zero rows, with the pivot columns and the signed product of the pivots before scaling"
    return Echelon(*_exact.echelon(a, columns))
