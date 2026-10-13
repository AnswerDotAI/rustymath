"Argument checks and function naming that several of rustymath's Python modules share."

def _check(ct):
    "`ct`, after checking that it's at least 0 and below 1"
    if not 0 <= ct < 1: raise ValueError(f"ct must be at least 0 and below 1, not {ct}")
    return ct

def _named(f, name, doc):
    "`f`, with `name` as its name and `doc` as its docstring"
    f.__name__ = f.__qualname__ = name
    f.__doc__ = doc
    return f

def _radix(bases, digits):
    "Raise `ValueError` unless `bases` is a nonempty vector and the last axis of `digits` has one digit for each base"
    if bases.ndim != 1 or not bases.size or not digits.ndim or digits.shape[-1] != bases.size:
        raise ValueError("decode needs a nonempty vector of bases and one digit for each base")
