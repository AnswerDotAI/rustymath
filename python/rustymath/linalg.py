"""rustymath's linear algebra, under the names of NumPy's similar functions. Each function takes a stack of matrices in the last two axes
of its arguments, and broadcasts the axes before them as NumPy's generalized ufuncs do. float32, float64, complex64 and complex128
matrices compute at their own width. float16 matrices compute in float32, with each result rounded to float16. Integers and Booleans
compute as float64, except in `matmul` and `semiring`, where integers compute in int64.

`solve` and `pinv` give a result for every matrix: the least-squares solution with the least norm, and the pseudoinverse, through LU
where a matrix is square and regular. A matrix that `cholesky` can't factor, or whose eigenvalues don't converge, gives NaN results and
NumPy's "invalid value" warning, as a failing item of any rustymath ufunc does. Each column of U in an SVD, and each eigenvector, has its
largest component real and positive. R in a QR factorization has a real diagonal with no negative item. `matmul` and `semiring`
multiply matrices, including through APL's inner products, such as the min-plus product of shortest paths."""
from typing import NamedTuple
import numpy as np
from ._core import linalg as _g

class Svd(NamedTuple):
    u: np.ndarray
    s: np.ndarray
    v: np.ndarray

class Eigen(NamedTuple):
    values: np.ndarray
    vectors: np.ndarray

class Lu(NamedTuple):
    l: np.ndarray
    u: np.ndarray
    p: np.ndarray

def _inexact(*arrays):
    "`arrays` as arrays, as float64 where all are integers or Booleans, which NumPy would otherwise send to a narrower float loop"
    arrays = [np.asarray(a) for a in arrays]
    return [a.astype(np.float64) for a in arrays] if np.result_type(*arrays).kind in 'biu' else arrays

def _shaped(a, wide, tall):
    "The generalized ufunc `wide` where `a`'s matrices have no more rows than columns, and otherwise `tall`"
    m, n = a.shape[-2:]
    return wide if m <= n else tall

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
    ('logical_or', 'logical_and'): _g.matmul, ('add', 'equal'): _g.add_equal, ('add', 'not_equal'): _g.add_not_equal}

def semiring(
    f, # The ufunc that sums: `add`, `maximum`, `minimum` or `logical_or`, from NumPy or rustymath
    g, # The ufunc that multiplies: `multiply`, `add`, `logical_and`, `equal` or `not_equal`
):
    "The matrix product whose sums use `f` and whose products use `g`, as APL's inner product `f.g` takes them. The pairs are (`add`, `multiply`), (`maximum`, `add`), (`minimum`, `add`) and (`logical_or`, `logical_and`), and the counts (`add`, `equal`) and (`add`, `not_equal`), which give int64. Only (`add`, `multiply`) takes complex matrices. Integers compute in int64, which wraps on overflow"
    names = tuple(u.__name__ if isinstance(u, np.ufunc) else None for u in (f, g))
    if names not in _PRODUCTS: raise ValueError(f"semiring takes one of the pairs {list(_PRODUCTS)}, not {names}")
    gufunc, logical = _PRODUCTS[names], names[0] == 'logical_or'
    def product(
        x1, x2, # Matrices, or vectors, which act as one row on the left and one column on the right
        /, out=None, # Array to write the result to
        **kwargs # Passed to the generalized ufunc, such as `axes=`
    ):
        x1, x2 = (np.asarray(x1, bool), np.asarray(x2, bool)) if logical else _product_inputs(x1, x2)
        if x1.dtype.kind == 'c' and gufunc is not _g.matmul: raise TypeError(f"{product.__name__} takes real matrices, not {x1.dtype}")
        return gufunc(x1, x2, out=out, **kwargs)
    product.__name__ = product.__qualname__ = '_'.join(names)
    product.__doc__ = f"The matrix product whose item `[i, j]` is the `{names[0]}` over `k` of `{names[1]}(x1[i, k], x2[k, j])`"
    return product

def solve(
    a, # Matrices
    b, # Right-hand sides: one vector, or matrices whose columns are right-hand sides
):
    "The solution `x` of `a @ x == b` for each matrix in `a`: the least-squares solution with the least norm, through LU where `a` is square and regular"
    a, b = _inexact(a, b)
    return _g.solve(a, b[:, None])[..., 0] if b.ndim == 1 else _g.solve(a, b)

def lstsq(
    a, # Matrices
    b, # Right-hand sides: one vector, or matrices whose columns are right-hand sides
):
    "The least-squares solution `x` of `a @ x == b` with the least norm, and `a`'s singular values in descending order, for each matrix in `a`"
    a, b = _inexact(a, b)
    if b.ndim > 1: return _shaped(a, _g.lstsq_m, _g.lstsq_n)(a, b)
    x, s = _shaped(a, _g.lstsq_m, _g.lstsq_n)(a, b[:, None])
    return x[..., 0], s

def pinv(
    a, # Matrices
):
    "The pseudoinverse of each matrix in `a`, which is its inverse, through LU, where it's square and regular"
    [a] = _inexact(a)
    return _g.pinv(a)

inv = pinv

def det(
    a, # Square matrices
):
    "The determinant of each matrix in `a`, from its LU factorization"
    [a] = _inexact(a)
    return _g.det(a)

def slogdet(
    a, # Square matrices
):
    "The sign of each matrix's determinant, and the natural logarithm of its magnitude, which can't overflow. A singular matrix gives sign 0 and logarithm -inf"
    [a] = _inexact(a)
    return _g.slogdet(a)

def lu(
    a, # Square matrices
):
    "The LU factorization of each matrix in `a`, with partial pivoting: row `i` of `l @ u` is row `p[i]` of the matrix"
    [a] = _inexact(a)
    return Lu(*_g.lu(a))

def cholesky(
    a, # Hermitian positive definite matrices
):
    "The lower triangular `l` with `l @ l.conj().mT == a` for each matrix in `a`, reading only `a`'s lower triangle"
    [a] = _inexact(a)
    return _g.cholesky(a)

def qr(
    a, # Matrices
    complete=False, # Make `q` square, and `r` as tall as `a`?
):
    "The QR factorization `(q, r)` of each matrix in `a`, with `r`'s diagonal real and never negative"
    [a] = _inexact(a)
    return _g.qr_complete(a) if complete else _shaped(a, _g.qr_m, _g.qr_n)(a)

def svd(
    a, # Matrices
):
    "The thin SVD of each matrix in `a`: `u`, the singular values `s` in descending order, and `v`, with `a == u @ (s[..., None] * v.conj().mT)`"
    [a] = _inexact(a)
    return Svd(*_shaped(a, _g.svd_m, _g.svd_n)(a))

def svdvals(
    a, # Matrices
):
    "The singular values of each matrix in `a`, in descending order"
    [a] = _inexact(a)
    return _shaped(a, _g.svdvals_m, _g.svdvals_n)(a)

def matrix_rank(
    a, # Matrices
):
    "The rank of each matrix in `a`: how many singular values exceed machine epsilon times the larger side times the largest, as `lstsq` counts them"
    [a] = _inexact(a)
    return _g.rank(a)

def eig(
    a, # Square matrices
):
    "The eigenvalues of each matrix in `a`, in the order of its Schur form, and its unit eigenvectors in columns, both complex. A real matrix's complex pair comes as the one with a positive imaginary part, then its conjugate"
    [a] = _inexact(a)
    return Eigen(*_g.eig(a))

def eigvals(
    a, # Square matrices
):
    "The complex eigenvalues of each matrix in `a`, in the order `eig` gives them"
    [a] = _inexact(a)
    return _g.eigvals(a)

def eigh(
    a, # Hermitian matrices
):
    "The eigenvalues of each matrix in `a` in ascending order, and its unit eigenvectors in columns, reading only `a`'s lower triangle"
    [a] = _inexact(a)
    return Eigen(*_g.eigh(a))

def eigvalsh(
    a, # Hermitian matrices
):
    "The eigenvalues of each matrix in `a` in ascending order, reading only `a`'s lower triangle"
    [a] = _inexact(a)
    return _g.eigvalsh(a)
