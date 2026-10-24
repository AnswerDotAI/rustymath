"rustymath with tolerant f64/complex128 comparisons and mathematical zero division. Other operations are unchanged."
from . import _core
from . import search as _search
from ._naming import _named

_native = _core._mathish
globals().update((name, getattr(_native, name)) for name in dir(_native) if not name.startswith('_'))
unique = _named(lambda ar: _search.Unique(*_native.unique(ar)), 'unique', _search.unique.__doc__)

union1d = _named(lambda x,y: _native._set(x, y, 'union'), 'union1d', _search.union1d.__doc__)
intersect1d = _named(lambda x,y: _native._set(x, y, 'intersection'), 'intersect1d', _search.intersect1d.__doc__)
setdiff1d = _named(lambda x,y: _native._set(x, y, 'difference'), 'setdiff1d', _search.setdiff1d.__doc__)

def __getattr__(name):
    import rustymath
    return getattr(rustymath, name)
