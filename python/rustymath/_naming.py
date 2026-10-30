"Function naming that several of rustymath's Python modules share."
from . import _core

def _named(f, name, doc):
    "`f`, with `name` as its name and `doc` as its docstring"
    f.__name__ = f.__qualname__ = name
    f.__doc__ = doc
    return f

def _native(f, names, kind):
    "A native function's name and Mathish choice, where `names` holds its name, rejecting Python callbacks and lookalikes"
    name = getattr(f, '__name__', None)
    if name in names:
        if f is getattr(_core, name): return name, False
        if f is getattr(_core._mathish, name, None): return name, True
    raise ValueError(f"expected a rustymath native {kind} function, not {f!r}")

def _binary_function(f):
    "A native binary function's name and Mathish choice"
    return _native(f, _core._binary_names, 'binary')

def _elementwise_function(f):
    "A native elementwise function's name and Mathish choice"
    return _native(f, _core._elementwise_names, 'elementwise')
