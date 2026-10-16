"""Checked int64 mixed-radix conversions. For rational or complex bases, use exact.decode or polynomial.decode."""
from ._core import integer as _integer

encode, decode = _integer.encode, _integer.decode
