"Function naming that several of rustymath's Python modules share."
from . import _core

def _named(f, name, doc):
    "`f`, with `name` as its name and `doc` as its docstring"
    f.__name__ = f.__qualname__ = name
    f.__doc__ = doc
    return f

def _binary_function(f):
    "A native binary function's name and Mathish choice, rejecting Python callbacks and lookalikes"
    name = getattr(f, '__name__', None)
    if name not in _core._binary_names:
        raise ValueError(f"expected a rustymath native binary function, not {f!r}")
    if f is getattr(_core, name): return name, False
    if f is getattr(_core._mathish, name, None): return name, True
    raise ValueError(f"expected a rustymath native binary function, not {f!r}")
