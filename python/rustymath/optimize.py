"""Minimizing a function of several variables, and finding a zero of a function of one, with SciPy's names. The searches run in Rust and call the Python function at each point.

`minimize` runs Brent's method, Nelder–Mead, Powell's method of conjugate directions, L-BFGS or differential evolution. Without a `method`, it runs Brent's method for one variable, L-BFGS for a function with `jac` and no bounds, Powell's method within bounds, and Nelder–Mead otherwise. A float32 or float16 `x0` searches in float32, and any other start searches in float64. Every method evaluates `fun` only within the bounds and returns the best point it evaluated. A NaN value counts as infinity. An exception from `fun` stops the search and reaches the caller. Without a `seed` option, differential evolution uses the same seed on every call."""
from typing import NamedTuple
import numpy as np
from ._core import optimize as _o

class OptimizeResult(NamedTuple):
    "What `minimize` found: the best point it evaluated, the function's value there, the number of evaluations, and whether the method's stopping test passed before the evaluations ran out"
    x: np.ndarray
    fun: float
    nfev: int
    success: bool

class RootResults(NamedTuple):
    "What `root_scalar` found: the end of the last bracket nearer zero, the number of evaluations, and whether the bracket shrank within tolerance"
    root: float
    function_calls: int
    converged: bool

def minimize(
    fun, # Function of a 1-D array that returns a float
    x0, # Start point, flattened
    method=None, # 'brent', 'nelder-mead', 'powell', 'l-bfgs' or 'differential-evolution', in any case; None picks one
    jac=None, # Function of a 1-D array that returns the gradient of `fun`
    bounds=None, # A (min, max) pair for each variable, with None for no limit
    tol=None, # Relative change at which the method stops; None for the square root of the search's machine epsilon
    options=None, # dict that can hold `maxfev`, the most evaluations; `popsize`, differential evolution's members for each variable; and `seed`, its generator's seed
    vectorized=False # Whether `fun` takes points as the columns of a 2-D array and returns one value for each, as SciPy's `differential_evolution` calls it
):
    "The minimum of `fun` from `x0`, as an `OptimizeResult`"
    x0 = np.ravel(x0)
    x0 = np.ascontiguousarray(x0, np.float32 if x0.dtype in (np.float16, np.float32) else np.float64)
    limits = None
    if bounds is not None:
        pairs = np.array([(-np.inf if lo is None else lo, np.inf if hi is None else hi) for lo, hi in bounds], x0.dtype).reshape(-1, 2)
        limits = np.ascontiguousarray(pairs[:, 0]), np.ascontiguousarray(pairs[:, 1])
    options = dict(options or {})
    maxfev, popsize, seed = options.pop('maxfev', None), options.pop('popsize', None), options.pop('seed', 0)
    if options: raise TypeError(f"minimize has no option {', '.join(options)}")
    population = None if popsize is None else popsize * x0.size
    return OptimizeResult(*_o.minimize(fun, x0, method, jac, limits, tol, maxfev, population, seed, vectorized))

def root_scalar(
    f, # Function of a float that returns a float
    bracket, # Two points where the signs of `f` differ
    rtol=None, # Relative width at which the bracket stops shrinking; None for the square root of float64's machine epsilon
    maxfev=None # The most evaluations of `f`; None for 1000
):
    "A zero of `f` within `bracket`, by Brent's method, as a `RootResults`. Ends whose signs agree give the end nearer zero, with `converged` False"
    a, b = bracket
    return RootResults(*_o.root_scalar(f, a, b, rtol, maxfev))
