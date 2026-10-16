"Function naming that several of rustymath's Python modules share."

def _named(f, name, doc):
    "`f`, with `name` as its name and `doc` as its docstring"
    f.__name__ = f.__qualname__ = name
    f.__doc__ = doc
    return f
