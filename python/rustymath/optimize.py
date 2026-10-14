"""Minimizing a function of several variables, and solving equations, nonlinear or linear, with SciPy's names. The searches run in Rust and call the Python functions at each point.

`minimize` runs Brent's method, Nelder–Mead, Powell's method of conjugate directions, L-BFGS or differential evolution. Without a `method`, it runs Brent's method for one variable, L-BFGS for a function with `jac` and no bounds, Powell's method within bounds, and Nelder–Mead otherwise. Every method evaluates `fun` only within the bounds and returns the best point it evaluated. A NaN value counts as infinity. Without a `seed` option, differential evolution uses the same seed on every call.

`root_scalar` finds a zero of a function of one variable by Brent's method, and `least_squares` the least-squares solution of a system of equations by Levenberg–Marquardt, which is a zero for a square system.

`cg`, `minres` and `gmres` solve a square linear system by conjugate gradients, MINRES and restarted GMRES, and `lsmr` gives a least-squares solution through MINRES on the normal equations, which gives LSMR's iterates. They take `A` as an array, or as anything with a `shape` and a `matvec` method, plus `rmatvec` for `lsmr`, such as SciPy's `LinearOperator`. Each takes a preconditioner `M`, an approximation to the inverse of `A`, or of `A.T @ A` for `lsmr`.

A float32 or float16 start, or `b` for the linear solvers, computes in float32, and anything else in float64. An exception from a function stops the search and reaches the caller."""
from typing import NamedTuple
import numpy as np
from ._core import optimize as _o
from ._checks import _named

class OptimizeResult(NamedTuple):
    "What `minimize` found: the best point it evaluated, the function's value there, the number of evaluations, and whether the method's stopping test passed before the evaluations ran out"
    x: np.ndarray
    fun: float
    nfev: int
    success: bool

class RootResults(NamedTuple):
    "What `root_scalar` found: the point nearest zero it evaluated, the number of evaluations, and whether the bracket shrank within tolerance"
    root: float
    function_calls: int
    converged: bool

class LeastSquaresResult(NamedTuple):
    "What `least_squares` found: the best point it evaluated, half the sum of the squared residuals there, the number of evaluations, and whether the method's stopping test passed before the evaluations ran out"
    x: np.ndarray
    cost: float
    nfev: int
    success: bool

class LinearResult(NamedTuple):
    "What a linear solver found: its last iterate, the norm of `b - A @ x` there, the number of products with `A` and its transpose, and whether the residual fell within tolerance"
    x: np.ndarray
    normr: float
    nmatvec: int
    success: bool

def _floats(x):
    "`x` flattened, as a contiguous float32 vector where it holds float16 or float32 numbers, and float64 otherwise"
    x = np.ravel(x)
    return np.ascontiguousarray(x, np.float32 if x.dtype in (np.float16, np.float32) else np.float64)

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
    x0 = _floats(x0)
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
    bracket=None, # Two points where the signs of `f` differ
    x0=None, # Start point, from which the search steps outward until the signs of `f` differ, where there's no `bracket`
    rtol=None, # Relative width at which the bracket stops shrinking; None for the square root of float64's machine epsilon
    maxfev=None # The most evaluations of `f`; None for 2000
):
    "A zero of `f` by Brent's method, as a `RootResults`. Where the signs of `f` never differ, it gives the point nearest zero it evaluated, with `converged` False"
    if bracket is None and x0 is None: raise TypeError("root_scalar needs a bracket or x0")
    ends = None if bracket is None else tuple(sorted(bracket))
    return RootResults(*_o.root_scalar(f, ends[0] if x0 is None else x0, ends, rtol, maxfev))

def least_squares(
    fun, # Function of a 1-D array that returns the residuals
    x0, # Start point, flattened
    jac=None, # Function of a 1-D array that returns the Jacobian, with a row for each residual; None for forward differences
    tol=None, # Relative change at which the method stops; None for the square root of the search's machine epsilon
    max_nfev=None # The most evaluations of `fun`; None for 1000 for each variable, plus 1000
):
    "The least-squares solution of `fun(x) = 0` from `x0` by Levenberg–Marquardt, as a `LeastSquaresResult`"
    x0 = _floats(x0)
    x, norm, nfev, success = _o.least_squares(fun, x0, jac, tol, max_nfev)
    return LeastSquaresResult(x, norm * norm / 2, nfev, success)

def _operator(A):
    "`A`, as an array unless it has a `shape`, with its product with a vector, and its transpose's where it has one"
    A = A if hasattr(A, 'shape') else np.asarray(A)
    if hasattr(A, 'matvec'): return A, A.matvec, getattr(A, 'rmatvec', None)
    return A, (lambda v: A @ v), (lambda u: A.T @ u)

def _krylov(name, method, transpose, doc):
    "The linear solver `name`, which runs `method`, on the normal equations where `transpose`"
    def solve(
        A, # Matrix as an array, or anything with a `shape` and a `matvec` method, plus `rmatvec` for least squares
        b, # Right-hand side, flattened
        x0=None, # Start point; None for zeros
        rtol=None, # The residual's norm, relative to the right-hand side's, at which the solver stops; None for the square root of its machine epsilon
        maxiter=None, # The most iterations, as SciPy counts them: restart cycles of up to 20 steps for `gmres`, and steps for the others, where `lsmr`'s each take a product with `A` and one with its transpose; None for 20 products for each unknown, plus 20
        M=None # Preconditioner, an approximation to the inverse of `A`, or of `A.T @ A` for `lsmr`, as an array or anything with `matvec`. It must be symmetric positive definite for all but `gmres`, which applies it on the right, where SciPy's applies it on the left
    ):
        A, matvec, rmatvec = _operator(A)
        if transpose and rmatvec is None: raise TypeError(f"{name} needs `A.rmatvec`")
        msolve = None if M is None else _operator(M)[1]
        b = _floats(b)
        x0 = np.zeros(A.shape[1], b.dtype) if x0 is None else np.ascontiguousarray(np.ravel(x0), b.dtype)
        return LinearResult(*_o.linear(matvec, b, x0, method, rmatvec if transpose else None, msolve, rtol, maxiter))
    return _named(solve, name, doc)

cg = _krylov('cg', 'cg', False, "The solution of `A @ x = b` for a symmetric positive definite `A` by conjugate gradients, as a `LinearResult`")
minres = _krylov('minres', 'minres', False, "The solution of `A @ x = b` for a symmetric `A` by MINRES, as a `LinearResult`")
gmres = _krylov('gmres', 'gmres', False, "The solution of `A @ x = b` for a square `A` by GMRES, restarted every 20 steps, as a `LinearResult`")
lsmr = _krylov('lsmr', 'minres', True, "The least-squares solution of `A @ x = b` by MINRES on the normal equations, which gives LSMR's iterates, as a `LinearResult`")
