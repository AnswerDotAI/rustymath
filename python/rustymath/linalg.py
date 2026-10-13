"""rustymath's linear algebra, under the names of NumPy's similar functions. Each function takes a stack of matrices in the last two axes of its arguments, and broadcasts the axes before them as NumPy's generalized ufuncs do. float32, float64, complex64 and complex128 matrices compute at their own width. float16 matrices compute in float32, with each result rounded to float16. Integers and Booleans compute as float64, except in `matmul` and `semiring`, where integers compute in int64.

`solve` and `pinv` give a result for every matrix. A square matrix with no negligible LU pivot solves and inverts through LU, with no cutoff, as NumPy's `solve` and `inv` do. Any other matrix gets `lstsq`'s least-squares solution with the least norm, or the pseudoinverse with `lstsq`'s cutoff. A matrix that `cholesky` can't factor, or whose eigenvalues don't converge, gives NaN results and NumPy's "invalid value" warning, as a failing item of any rustymath ufunc does. R in a QR factorization has a real diagonal. `matmul` and `semiring` multiply matrices, including through APL's inner products, such as the min-plus product of shortest paths."""
from typing import NamedTuple
import numpy as np
from ._core import linalg as _g
from ._checks import _named

class Svd(NamedTuple):
    "Thin SVD factors, reusable for least-norm solves and pseudoinverses"
    u: np.ndarray
    s: np.ndarray
    v: np.ndarray

    def solve(self, b):
        "The least-norm least-squares solution for `b`, using these factors and `lstsq`'s cutoff"
        return _solve(_g.svd_solve, *_inexact(*self, b))

    def pseudoinverse(self):
        "The pseudoinverse, using these factors and `lstsq`'s cutoff"
        return _g.svd_pinv(*self)

class Eigen(NamedTuple):
    values: np.ndarray
    vectors: np.ndarray

class Lu(NamedTuple):
    "Packed LU factors and row positions; `l` and `u` materialize the separate triangular matrices"
    factors: np.ndarray
    p: np.ndarray

    @property
    def l(self): return np.tril(self.factors, -1) + np.eye(self.factors.shape[-1], dtype=self.factors.dtype)

    @property
    def u(self): return np.triu(self.factors)

    def solve(self, b):
        "Solve for `b` using these factors, without refactoring or applying a singular-value cutoff"
        f, b = _inexact(self.factors, b)
        return _solve(_g.lu_solve, f, self.p, b)

    def inverse(self):
        "The inverse from these factors, without refactoring"
        return _g.lu_inverse(*self)

    def determinant(self):
        "The determinant from these factors"
        return _g.lu_det(*self)

    def log_determinant(self):
        "The determinant's sign and natural log magnitude, from these factors"
        return _g.lu_slogdet(*self)

class Echelon(NamedTuple):
    rows: np.ndarray
    pivots: np.ndarray
    determinant: object

def _inexact(*arrays):
    "`arrays` as arrays, as float64 where all are integers or Booleans, which NumPy would otherwise send to a narrower float loop"
    arrays = [np.asarray(a) for a in arrays]
    return [a.astype(np.float64) for a in arrays] if np.result_type(*arrays).kind in 'biu' else arrays

def _solve(f, *arrays):
    "Call a matrix-RHS gufunc, temporarily giving a vector RHS one column"
    b = arrays[-1]
    return f(*arrays[:-1], b[:, None])[..., 0] if b.ndim == 1 else f(*arrays)

def _shaped(a, wide, tall):
    "The generalized ufunc `wide` where `a`'s matrices have no more rows than columns, and otherwise `tall`"
    m, n = a.shape[-2:]
    return wide if m <= n else tall

def _matrices(
    name, # The function's name
    gufunc, # The generalized ufunc, or the `(wide, tall)` pair that `_shaped` chooses between
    doc, # The function's docstring
    result=None, # A type to build from the gufunc's outputs, such as `Lu`
):
    "A function of one stack of matrices that calls `gufunc`, with integers and Booleans as float64"
    def f(
        a, # Matrices
    ):
        [a] = _inexact(a)
        r = (_shaped(a, *gufunc) if isinstance(gufunc, tuple) else gufunc)(a)
        return result(*r) if result else r
    return _named(f, name, doc)

def _product_inputs(x1, x2):
    "`x1` and `x2` as arrays of the type a product computes in: Booleans for two Boolean arrays, int64 for other integers and Booleans, and otherwise their own types"
    x1, x2 = np.asarray(x1), np.asarray(x2)
    if x1.dtype == bool and x2.dtype == bool: return x1, x2
    if np.result_type(x1, x2).kind in 'biu': return x1.astype(np.int64, copy=False), x2.astype(np.int64, copy=False)
    return x1, x2

def matmul(
    x1, x2, # Matrices, or vectors, which act as one row on the left and one column on the right
    /, out=None, # Array to write the result to
    **kwargs # Passed to the generalized ufunc, such as `axes=`
):
    "The matrix product of `x1` and `x2`, broadcast as `numpy.matmul` does. Booleans give whether some `k` has both `x1[i,k]` and `x2[k,j]`. Other integers compute in int64, which wraps on overflow"
    return _g.matmul(*_product_inputs(x1, x2), out=out, **kwargs)

_PRODUCTS = {('add', 'multiply'): _g.matmul, ('maximum', 'add'): _g.maximum_add, ('minimum', 'add'): _g.minimum_add,
    ('logical_or', 'logical_and'): _g.matmul, ('logical_and', 'logical_or'): _g.logical_and_or,
    ('add', 'equal'): _g.add_equal, ('add', 'not_equal'): _g.add_not_equal}

def semiring(
    f, # The ufunc that sums: `add`, `maximum`, `minimum`, `logical_or` or `logical_and`, from NumPy or rustymath
    g, # The ufunc that multiplies: `multiply`, `add`, `logical_and` or `logical_or` from NumPy or rustymath, or NumPy's `equal` or `not_equal`
):
    "The matrix product whose sums use `f` and whose products use `g`, as APL's inner product `f.g` takes them. Pairs are (`add`, `multiply`), (`maximum`, `add`), (`minimum`, `add`), (`logical_or`, `logical_and`), (`logical_and`, `logical_or`), and the int64 counts (`add`, `equal`) and (`add`, `not_equal`). The counts take NumPy's exact `equal` and `not_equal`, not rustymath's tolerant comparisons. Only (`add`, `multiply`) takes complex matrices. Integers compute in wrapping int64"
    names = tuple(getattr(u, '__name__', None) for u in (f, g))
    if names not in _PRODUCTS or not all(isinstance(u, np.ufunc) for u in (f, g)): raise ValueError(f"semiring takes ufuncs forming one of the pairs {list(_PRODUCTS)}, not {f!r} and {g!r}")
    gufunc, logical = _PRODUCTS[names], names[0].startswith('logical_')
    def product(
        x1, x2, # Matrices, or vectors, which act as one row on the left and one column on the right
        /, out=None, # Array to write the result to
        **kwargs # Passed to the generalized ufunc, such as `axes=`
    ):
        x1, x2 = (np.asarray(x1, bool), np.asarray(x2, bool)) if logical else _product_inputs(x1, x2)
        if x1.dtype.kind == 'c' and gufunc is not _g.matmul: raise TypeError(f"{product.__name__} takes real matrices, not {x1.dtype}")
        return gufunc(x1, x2, out=out, **kwargs)
    return _named(product, '_'.join(names), f"The matrix product whose item `[i, j]` is the `{names[0]}` over `k` of `{names[1]}(x1[i, k], x2[k, j])`")

def solve(
    a, # Matrices
    b, # Right-hand sides: one vector, or matrices whose columns are right-hand sides
):
    "The solution `x` of `a @ x == b` for each matrix in `a`, through LU with no cutoff where `a` is square with no negligible pivot. Any other `a` gives the least-squares solution with the least norm"
    return _solve(_g.solve, *_inexact(a, b))

def lstsq(
    a, # Matrices
    b, # Right-hand sides: one vector, or matrices whose columns are right-hand sides
):
    "The least-squares solution `x` of `a @ x == b` with the least norm, and `a`'s singular values in descending order, for each matrix in `a`"
    a, b = _inexact(a, b)
    f = _shaped(a, _g.lstsq_m, _g.lstsq_n)
    if b.ndim > 1: return f(a, b)
    x, s = f(a, b[:, None])
    return x[..., 0], s

pinv = _matrices('pinv', _g.pinv, "The pseudoinverse of each matrix in `a`. A square matrix with no negligible LU pivot gives its inverse through LU, with no cutoff. Any other gives the pseudoinverse with `lstsq`'s cutoff")
det = _matrices('det', _g.det, "The determinant of each square matrix in `a`, from its LU factorization")
slogdet = _matrices('slogdet', _g.slogdet, "The sign of each square matrix's determinant, and the natural logarithm of its magnitude, which can't overflow. A singular matrix gives sign 0 and logarithm -inf")
lu = _matrices('lu', _g.lu, "Reusable packed LU factors of each square matrix in `a`, with partial pivoting. Row `i` of `f.l @ f.u` is row `f.p[i]` of `a`; `f.solve(b)` reuses the packed factors", Lu)

def permanent(a):
    "The permanent of each square matrix, by Ryser's exponential-time formula (fewer than 64 rows)"
    [a] = _inexact(a)
    if a.shape[-1] >= 64: raise ValueError("the permanent takes fewer than 64 rows")
    return _g.permanent(a)

def echelon(a, columns=None):
    "Reduced row echelon form of one matrix: nonzero rows, pivot columns and signed pivot product. Reduces only `columns` leading columns when supplied. Pivots on the first nonzero item, without a tolerance; use decompositions for numerically stable floating-point solves"
    [a] = _inexact(a)
    return Echelon(*_g.echelon(a, columns))

cholesky = _matrices('cholesky', _g.cholesky, "The lower triangular `l` with `l @ l.conj().mT == a` for each Hermitian positive definite matrix in `a`, reading only `a`'s lower triangle")

def qr(
    a, # Matrices
    complete=False, # Make `q` square, and `r` as tall as `a`?
):
    "The QR factorization `(q, r)` of each matrix in `a`, with `r`'s diagonal real and never negative"
    [a] = _inexact(a)
    return _g.qr_complete(a) if complete else _shaped(a, _g.qr_m, _g.qr_n)(a)

def svd(
    a, # Matrices
    rank=None, # Keep only the `rank` largest singular values and their vectors, from a randomized approximation; None: all of them
):
    "The thin SVD of each matrix in `a`: `u`, the singular values `s` in descending order, and `v`, with `a == u @ (s[..., None] * v.conj().mT)`. With `rank`, `u` and `v` have `rank` columns, or fewer for a smaller matrix, from Halko, Martinsson and Tropp's randomized range finder. Its error depends on how fast the singular values beyond the `rank`th decay"
    [a] = _inexact(a)
    if rank is None: return Svd(*_shaped(a, _g.svd_m, _g.svd_n)(a))
    *batch, m, n = a.shape
    k = min(rank, m, n)
    return Svd(*_g.truncated_svd(a, out=(np.empty((*batch, m, k), a.dtype), np.empty((*batch, k), a.real.dtype), np.empty((*batch, n, k), a.dtype))))

svdvals = _matrices('svdvals', (_g.svdvals_m, _g.svdvals_n), "The singular values of each matrix in `a`, in descending order")
matrix_rank = _matrices('matrix_rank', _g.rank, "The rank of each matrix in `a`: how many singular values exceed machine epsilon times the larger side times the largest, as `lstsq` counts them")
eig = _matrices('eig', _g.eig, "The eigenvalues of each square matrix in `a`, in the order of its Schur form, and its unit eigenvectors in columns, both complex. A real matrix's complex pair comes as the one with a positive imaginary part, then its conjugate", Eigen)
eigvals = _matrices('eigvals', _g.eigvals, "The complex eigenvalues of each square matrix in `a`, in the order `eig` gives them")
eigh = _matrices('eigh', _g.eigh, "The eigenvalues of each Hermitian matrix in `a` in ascending order, and its unit eigenvectors in columns, reading only `a`'s lower triangle", Eigen)
eigvalsh = _matrices('eigvalsh', _g.eigvalsh, "The eigenvalues of each Hermitian matrix in `a` in ascending order, reading only `a`'s lower triangle")
