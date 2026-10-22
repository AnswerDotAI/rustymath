"""rustymath's linear algebra, under the names of NumPy's similar functions. Functions take stacks of matrices in the last two axes and return rustymath arrays. Only a one-dimensional right-hand side is a vector. Higher-rank right-hand sides contain matrix cells; a stack of vectors needs a singleton final column axis. The axes before those form each argument's frame. Frames pair as rustymath broadcasts: axes with the same name first, then the rest from the front, where NumPy pairs them from the back. Where both have keys, a matrix's rows and its right-hand side's rows, and the axes that a product contracts, pair by key, and keys that differ raise `ValueError`. Results keep the keys and names of the argument axes they index, such as a solution's rows, which index the matrix's columns. Factors from a decomposition have none. float32, float64, complex64 and complex128 matrices compute at their own width. float16 matrices compute in float32, with each result rounded to float16. Booleans, uint8 and int16 compute as float32, and wider integers as float64, except in `matmul` and `inner`, where integers compute in int64 and raise `OverflowError` for a result that doesn't fit.

`solve` and `pinv` give a result for every matrix. A square matrix with no negligible LU pivot solves and inverts through LU, with no cutoff, as NumPy's `solve` and `inv` do. Any other matrix gets `lstsq`'s least-squares solution with the least norm, or the pseudoinverse with `lstsq`'s cutoff. A matrix that `cholesky` can't factor, or whose eigenvalues don't converge, gives NaN results, with no warning. R in a QR factorization has a real diagonal. `matmul` and `inner` multiply matrices, including through APL's inner products, such as the min-plus product of shortest paths."""
from typing import NamedTuple
from ._core import Array, linalg as _g
from ._naming import _binary_name, _named

class Svd(NamedTuple):
    "Thin SVD factors, reusable for least-norm solves and pseudoinverses"
    u: Array
    s: Array
    v: Array

    def solve(self, b):
        "The least-norm least-squares solution for `b`, using these factors and `lstsq`'s cutoff"
        return _g.svd_solve(*self, b)

    def pseudoinverse(self):
        "The pseudoinverse, using these factors and `lstsq`'s cutoff"
        return _g.svd_pinv(*self)

class Eigen(NamedTuple):
    values: Array
    vectors: Array

class Lu(NamedTuple):
    "Packed LU factors in rustymath's order, where row `j` holds column `j`, and row positions. `l` and `u` give the separate triangular matrices"
    factors: Array
    p: Array

    @property
    def l(self): return _g.lu_lower(self.factors)

    @property
    def u(self): return _g.lu_upper(self.factors)

    def solve(self, b):
        "Solve for `b` using these factors, without factoring again or applying a cutoff"
        return _g.lu_solve(*self, b)

    def inverse(self):
        "The inverse from these factors, without factoring again"
        return _g.lu_inverse(*self)

    def determinant(self):
        "The determinant from these factors"
        return _g.lu_det(*self)

    def log_determinant(self):
        "The determinant's sign and the natural logarithm of its magnitude, from these factors"
        return _g.lu_slogdet(*self)

class Echelon(NamedTuple):
    rows: Array
    pivots: Array
    determinant: object

solve, lstsq, pinv, det, slogdet, cholesky, qr = _g.solve, _g.lstsq, _g.pinv, _g.det, _g.slogdet, _g.cholesky, _g.qr
svdvals, matrix_rank, eigvals, eigvalsh, permanent, matmul, norm, cond = _g.svdvals, _g.matrix_rank, _g.eigvals, _g.eigvalsh, _g.permanent, _g.matmul, _g.norm, _g.cond
lu = _named(lambda a: Lu(*_g.lu(a)), 'lu', _g.lu.__doc__)
eig = _named(lambda a: Eigen(*_g.eig(a)), 'eig', _g.eig.__doc__)
eigh = _named(lambda a: Eigen(*_g.eigh(a)), 'eigh', _g.eigh.__doc__)
svd = _named(lambda a, rank=None: Svd(*_g.svd(a, rank)), 'svd', _g.svd.__doc__)
echelon = _named(lambda a, columns=None: Echelon(*_g.echelon(a, columns)), 'echelon', _g.echelon.__doc__)

def inner(
    f, # Native binary function that folds the products, from the right
    g, # Native binary function that forms the products
):
    "The inner product `f.g`: apply `g` over the contracted axis and fold with `f` from the right. Recognized pairs use specialized kernels; others use bounded scratch. Comparisons use the product's `ct`, as the elementwise functions do"
    names = tuple(map(_binary_name, (f, g)))
    name = '_'.join(names)
    def product(
        x1, x2, # Matrices, or vectors, which act as one row on the left and one column on the right
        /,
        *, ct=None, # Tolerance for a comparing `f` or `g`; None for the process's tolerance (`set_tolerance`)
    ): return _g.product(*names, x1, x2, name, ct=ct)
    return _named(product, name, f"The matrix product whose item `[i, j]` is the `{names[0]}` over `k` of `{names[1]}(x1[i, k], x2[k, j])`")

