"Function naming that several of rustymath's Python modules share."
from . import _core

def _named(f, name, doc):
    "`f`, with `name` as its name and `doc` as its docstring"
    f.__name__ = f.__qualname__ = name
    f.__doc__ = doc
    return f

def _binary_name(f):
    "The name of a native binary function, rejecting Python callbacks and lookalikes"
    name = getattr(f, '__name__', None)
    if name not in _core._binary_names or f is not getattr(_core, name):
        raise ValueError(f"expected a rustymath native binary function, not {f!r}")
    return name
