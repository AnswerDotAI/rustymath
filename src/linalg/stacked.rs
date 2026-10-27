//! Matrix functions on stacks of matrices, which [`Stacked`] runs. Each takes stacks of matrices in its arguments' last two axes, and only a one-dimensional right-hand side is a vector. The axes before those form each argument's frame. Frames pair as rustymath broadcasts, through [`Frames`]: axes with the same name first, then the rest from the front. A result has the frame's axes, then each cell's, and a cell axis keeps the keys and name of the argument axis it indexes. The cells of a stack split across rustymath's threads. A matrix that `cholesky`, `eigh`, `eig` or their value-only forms can't factor gives NaN results.
use num_complex::Complex;
use num_traits::Float as _;
use crate::agreement::{Frames, Mapping};
use crate::array::{Array, ArrayView};
use crate::buffer::{Buffer, DType, Element, FloatWidth, Items, Width};
use crate::error::{Error, Kind};
use crate::gather::Steps;
use crate::layout::{Lanes, Layout};
use crate::parallel::stack;
use super::{exact, Lu, LuWork, QrWork, Scalar, Svd};

/// The int64 dtype of row positions and ranks.
const I64: DType = DType::Integer(Width::I64);

/// The documentation of each matrix function, by its name. [`Stacked`]'s methods and rustymath's Python functions of the same names both read their docs here.
#[macro_export]
macro_rules! stacked_doc {
    (solve) => { "The solution `x` of `a @ x == b` for each matrix `a` and right-hand side `b`, a vector or matrices. A square `a` with no negligible LU pivot solves through LU, with no cutoff. Any other `a` gives the least-squares solution with the least norm." };
    (lstsq) => { "The least-squares solution `x` of `a @ x == b` with the least norm, and `a`'s singular values in descending order, for each matrix `a` and right-hand side `b`, a vector or matrices." };
    (pinv) => { "The pseudoinverse of each matrix. A square matrix with no negligible LU pivot gives its inverse through LU, with no cutoff. Any other gives the pseudoinverse with `lstsq`'s cutoff." };
    (det) => { "The determinant of each square matrix, from its LU factorization." };
    (inverse) => { "The inverse of each square matrix and its determinant, from one LU factorization, or `None` where any matrix has a negligible LU pivot: where `solve` and `pinv` fall back to least squares." };
    (slogdet) => { "The sign of each square matrix's determinant, and the natural logarithm of its magnitude, which can't overflow. A singular matrix gives sign 0 and logarithm -inf." };
    (lu) => { "The packed LU factors of each square matrix, with partial pivoting, and the positions of its rows. The factors are in rustymath's order: row `j` holds column `j` of the packed matrix. Row `i` of `l @ u` is row `p[i]` of the matrix." };
    (lu_solve) => { "`x` with `a @ x == b`, from the packed LU factors `factors` and row positions `p` of each matrix `a`, as `lu` gives them, without factoring again or applying a cutoff." };
    (lu_inverse) => { "The inverse of each matrix, from its packed LU factors and row positions, as `lu` gives them." };
    (lu_det) => { "The determinant of each matrix, from its packed LU factors and row positions, as `lu` gives them." };
    (lu_slogdet) => { "The sign of each matrix's determinant and the natural logarithm of its magnitude, from its packed LU factors and row positions, as `lu` gives them." };
    (lu_lower) => { "The unit lower triangle of each matrix's packed LU factors, as `lu` gives them." };
    (lu_upper) => { "The upper triangle of each matrix's packed LU factors, as `lu` gives them." };
    (cholesky) => { "The lower triangular factor L satisfying L L-adjoint = A for each Hermitian positive definite matrix A, reading only its lower triangle." };
    (qr) => { "The QR factorization `(q, r)` of each matrix, with `r`'s diagonal real. With `complete`, `q` is square and `r` as tall as the matrix." };
    (svd) => { "The thin SVD of each matrix: `u`, the singular values `s` in descending order, and `v`, with reconstruction U diag(s) V-adjoint. With `rank`, only the `rank` largest singular values and their vectors, or fewer for a smaller matrix, from Halko, Martinsson and Tropp's randomized range finder." };
    (svd_solve) => { "The least-norm least-squares solution `x` of `a @ x == b`, from the thin SVD factors `u`, `s` and `v` of each matrix `a`, as `svd` gives them, using `lstsq`'s cutoff. A complex `s` is a Domain error." };
    (svd_pinv) => { "The pseudoinverse of each matrix, from its thin SVD factors `u`, `s` and `v`, as `svd` gives them, using `lstsq`'s cutoff. A complex `s` is a Domain error." };
    (svdvals) => { "The singular values of each matrix, in descending order." };
    (matrix_rank) => { "The rank of each matrix: how many singular values exceed machine epsilon times the larger side times the largest, as `lstsq` counts them." };
    (norm) => { "The norm of each matrix that `ord` names, as NumPy's `linalg.norm` names them: the largest sum of a column's magnitudes (`1`), the largest singular value (`2`), the largest sum of a row's magnitudes (`inf`), or the square root of the sum of the squared magnitudes (`\"fro\"`, the default in Python)." };
    (cond) => { "The condition number of each matrix in the norm `p`, which `norm` names as its `ord`: the matrix's norm times its inverse's, as NumPy's `linalg.cond` defines it. The 2-norm, the default in Python, is the largest singular value over the smallest, for a matrix of any shape. Any other norm needs square matrices, and gives infinity for a matrix with a negligible LU pivot." };
    (eig) => { "The eigenvalues of each square matrix, in the order of its Schur form, and its unit eigenvectors in columns, both complex. A real matrix's complex pair comes as the one with a positive imaginary part, then its conjugate." };
    (eigvals) => { "The complex eigenvalues of each square matrix, in the order `eig` gives them." };
    (eigh) => { "The eigenvalues of each Hermitian matrix in ascending order, and its unit eigenvectors in columns, reading only the matrix's lower triangle." };
    (eigvalsh) => { "The eigenvalues of each Hermitian matrix in ascending order, reading only the matrix's lower triangle." };
    (permanent) => { "The permanent of each square matrix, by Ryser's exponential-time formula. A Domain error, Python's `ValueError`, for 64 rows or more." };
    (echelon) => { "The reduced row echelon form of one matrix: its nonzero rows, its pivot columns and the signed product of its pivots. With `columns`, it reduces only that many leading columns, or all of them where the matrix has fewer. It pivots on the first nonzero item, with no tolerance, so the decompositions give numerically stable solves of floats." };
}

/// The matrix functions of stacks. Each call computes in the dtype that rustymath's ladder gives its arguments, where exact numbers compute as floats of `float`, which the caller chooses by its own policy, as rustymath's Python API and basedpl each do. Each call checks for an interrupt between rounds of cells, and an interrupt's error stops it.
pub struct Stacked { float: FloatWidth }

impl Stacked {
    /// Matrix functions where exact numbers compute as floats of `float`, the caller's policy.
    pub fn new(float: FloatWidth) -> Self { Self { float } }

    /// The `per` items of each cell of `frames`, which `f` writes into the cell's slot of the output, with scratch `S` that a thread's cells share.
    fn slots<B: Clone + Default + Send + 'static, S: Default>(&self, frames: &Frames, per: usize, f: impl Fn(usize, &mut [B], &mut S) + Sync) -> Result<Vec<B>, Error> {
        Ok(stack::<(Vec<B>,), S>(frames.count, [per], &|i, (out,), s| { f(i, out, s); true })?.0.0)
    }

    /// The stack `a` that the matrix function named `function` takes alone.
    fn matrices<'a>(&self, function: &str, a: ArrayView<'a>) -> Result<Matrices<'a>, Error> {
        let (compute, store) = self.dtypes(function, &[a])?;
        let layout = a.layout();
        Ok(Matrices { compute, store, layout, size: matrix(function, layout)?, frames: Frames::new(&[(layout, 2)])?, items: a.cast(compute)? })
    }

    /// The dtypes that a call of `function` on `args` computes in and gives its results at. It computes in the dtype that rustymath's ladder ([`DType::computes_in`]) gives the arguments' common dtype, where exact numbers compute as floats of the caller's width. Results take the common dtype where it's a float or complex dtype, and the computing dtype otherwise. A Domain error for characters.
    fn dtypes(&self, function: &str, args: &[ArrayView<'_>]) -> Result<(DType, DType), Error> {
        let missing = |d: DType| Error::new(Kind::Domain, format!("{function} doesn't take {}", d.name()));
        let mut common = args[0].dtype();
        for a in &args[1..] { common = common.common(a.dtype()).ok_or_else(|| missing(DType::Character))?; }
        let compute = common.computes_in(Width::I64, self.float, |d| matches!(d, DType::Float(FloatWidth::F32 | FloatWidth::F64) | DType::Complex(_))).ok_or_else(|| missing(common))?;
        Ok((compute, if matches!(common, DType::Float(_) | DType::Complex(_)) { common } else { compute }))
    }

    #[doc = crate::stacked_doc!(solve)]
    pub fn solve(&self, a: ArrayView<'_>, b: ArrayView<'_>) -> Result<Array, Error> {
        let (compute, store) = self.dtypes("solve", &[a, b])?;
        let (la, lb) = (a.layout(), b.layout());
        let (ra, rb) = (la.shape().len(), lb.shape().len());
        let [m, n] = matrix("solve", la)?;
        let (core, k, vector) = rhs("solve", lb, m)?;
        let frames = Frames::new(&[(la, 2), (lb, core)])?;
        let (ai, bi) = (a.cast(compute)?, paired(b.cast(compute)?, lb, rb - core, la, ra - 2, "solve")?);
        crate::with_inexact!(compute, T => {
            let (a, b) = (ai.of::<T>(), bi.of::<T>());
            let x = self.slots(&frames, n * k, |i, out: &mut [T], w: &mut LuWork<T>| out.copy_from_slice(&w.solve(&a[frames.items(0, i, m * n)], &b[frames.items(1, i, m * k)], [m, n], k)))?;
            Ok(array(x, &frames.layout, solution(la.axes(ra - 1..ra), lb, vector), store)?)
        })
    }

    #[doc = crate::stacked_doc!(lstsq)]
    pub fn lstsq(&self, a: ArrayView<'_>, b: ArrayView<'_>) -> Result<(Array, Array), Error> {
        let (compute, store) = self.dtypes("lstsq", &[a, b])?;
        let (la, lb) = (a.layout(), b.layout());
        let (ra, rb) = (la.shape().len(), lb.shape().len());
        let [m, n] = matrix("lstsq", la)?;
        let (core, k, vector) = rhs("lstsq", lb, m)?;
        let frames = Frames::new(&[(la, 2), (lb, core)])?;
        let (ai, bi) = (a.cast(compute)?, paired(b.cast(compute)?, lb, rb - core, la, ra - 2, "lstsq")?);
        crate::with_inexact!(compute, T => {
            let (a, b) = (ai.of::<T>(), bi.of::<T>());
            let ((x, sv), _) = stack::<(Vec<T>, Vec<<T as Scalar>::Real>), ()>(frames.count, [n * k, m.min(n)], &|i, (x, sv), _| {
                let (cx, cs) = super::lstsq(&a[frames.items(0, i, m * n)], &b[frames.items(1, i, m * k)], [m, n], k);
                x.copy_from_slice(&cx);
                sv.copy_from_slice(&cs);
                true
            })?;
            Ok((array(x, &frames.layout, solution(la.axes(ra - 1..ra), lb, vector), store)?, array(sv, &frames.layout, [m.min(n)], store.real())?))
        })
    }

    #[doc = crate::stacked_doc!(pinv)]
    pub fn pinv(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, store, layout: la, size: [m, n], frames, items: ai } = self.matrices("pinv", a)?;
        let ra = la.shape().len();
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let x = self.slots(&frames, m * n, |i, out: &mut [T], w: &mut LuWork<T>| w.pinv_into(&a[frames.items(0, i, m * n)], [m, n], out))?;
            Ok(array(x, &frames.layout, la.axes([ra - 1, ra - 2].into_iter()), store)?)
        })
    }

    #[doc = crate::stacked_doc!(det)]
    pub fn det(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("det", a)?;
        let n = square("det", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let d = self.slots(&frames, 1, |i, out: &mut [T], w: &mut LuWork<T>| out[0] = w.factor(&a[frames.items(0, i, n * n)], n).determinant())?;
            Ok(array(d, &frames.layout, Layout::EMPTY, store)?)
        })
    }

    #[doc = crate::stacked_doc!(inverse)]
    pub fn inverse(&self, a: ArrayView<'_>) -> Result<Option<(Array, Array)>, Error> {
        let Matrices { compute, store, layout: la, size, frames, items: ai } = self.matrices("inverse", a)?;
        let (ra, n) = (la.shape().len(), square("inverse", size)?);
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let ((x, d), ok) = stack::<(Vec<T>, Vec<T>), LuWork<T>>(frames.count, [n * n, 1], &|i, (x, d), w| {
                w.inverse_into(&a[frames.items(0, i, n * n)], [n, n], x).map(|det| d[0] = det).is_some()
            })?;
            Ok(if ok { Some((array(x, &frames.layout, la.axes([ra - 1, ra - 2].into_iter()), store)?, array(d, &frames.layout, Layout::EMPTY, store)?)) } else { None })
        })
    }

    #[doc = crate::stacked_doc!(slogdet)]
    pub fn slogdet(&self, a: ArrayView<'_>) -> Result<(Array, Array), Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("slogdet", a)?;
        let n = square("slogdet", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let ((sign, log), _) = stack::<(Vec<T>, Vec<<T as Scalar>::Real>), LuWork<T>>(frames.count, [1, 1], &|i, (sign, log), w| {
                (sign[0], log[0]) = w.factor(&a[frames.items(0, i, n * n)], n).log_determinant();
                true
            })?;
            Ok((array(sign, &frames.layout, Layout::EMPTY, store)?, array(log, &frames.layout, Layout::EMPTY, store.real())?))
        })
    }

    #[doc = crate::stacked_doc!(lu)]
    pub fn lu(&self, a: ArrayView<'_>) -> Result<(Array, Array), Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("lu", a)?;
        let n = square("lu", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let ((factors, p), _) = stack::<(Vec<T>, Vec<i64>), LuWork<T>>(frames.count, [n * n, n], &|i, (factors, p), w| {
                let f = w.factor(&a[frames.items(0, i, n * n)], n);
                factors.copy_from_slice(&f.factors);
                for (p, &r) in p.iter_mut().zip(f.perm.iter()) { *p = r as i64; }
                true
            })?;
            Ok((array(factors, &frames.layout, [n, n], store)?, array(p, &frames.layout, [n], I64)?))
        })
    }

    #[doc = crate::stacked_doc!(lu_solve)]
    pub fn lu_solve(&self, factors: ArrayView<'_>, p: ArrayView<'_>, b: ArrayView<'_>) -> Result<Array, Error> {
        let (compute, store) = self.dtypes("lu_solve", &[factors, b])?;
        let (lf, lb) = (factors.layout(), b.layout());
        let (n, positions) = factored("lu_solve", lf, p)?;
        let (core, k, vector) = rhs("lu_solve", lb, n)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1), (lb, core)])?;
        let (fi, bi) = (factors.cast(compute)?, b.cast(compute)?);
        crate::with_inexact!(compute, T => {
            let (a, b, p) = (fi.of::<T>(), bi.of::<T>(), &positions);
            let x = self.slots(&frames, n * k, |i, out: &mut [T], _: &mut ()| {
                out.copy_from_slice(&Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).solve(&b[frames.items(2, i, n * k)], k))
            })?;
            Ok(array(x, &frames.layout, solution(Layout::from([n]), lb, vector), store)?)
        })
    }

    #[doc = crate::stacked_doc!(lu_inverse)]
    pub fn lu_inverse(&self, factors: ArrayView<'_>, p: ArrayView<'_>) -> Result<Array, Error> {
        let (compute, store) = self.dtypes("lu_inverse", &[factors])?;
        let lf = factors.layout();
        let (n, positions) = factored("lu_inverse", lf, p)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1)])?;
        let fi = factors.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let (a, p) = (fi.of::<T>(), &positions);
            let x = self.slots(&frames, n * n, |i, out: &mut [T], _: &mut ()| out.copy_from_slice(&Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).inverse()))?;
            Ok(array(x, &frames.layout, [n, n], store)?)
        })
    }

    #[doc = crate::stacked_doc!(lu_det)]
    pub fn lu_det(&self, factors: ArrayView<'_>, p: ArrayView<'_>) -> Result<Array, Error> {
        let (compute, store) = self.dtypes("lu_det", &[factors])?;
        let lf = factors.layout();
        let (n, positions) = factored("lu_det", lf, p)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1)])?;
        let fi = factors.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let (a, p) = (fi.of::<T>(), &positions);
            let d = self.slots(&frames, 1, |i, out: &mut [T], _: &mut ()| out[0] = Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).determinant())?;
            Ok(array(d, &frames.layout, Layout::EMPTY, store)?)
        })
    }

    #[doc = crate::stacked_doc!(lu_slogdet)]
    pub fn lu_slogdet(&self, factors: ArrayView<'_>, p: ArrayView<'_>) -> Result<(Array, Array), Error> {
        let (compute, store) = self.dtypes("lu_slogdet", &[factors])?;
        let lf = factors.layout();
        let (n, positions) = factored("lu_slogdet", lf, p)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1)])?;
        let fi = factors.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let (a, p) = (fi.of::<T>(), &positions);
            let ((sign, log), _) = stack::<(Vec<T>, Vec<<T as Scalar>::Real>), ()>(frames.count, [1, 1], &|i, (sign, log), _| {
                (sign[0], log[0]) = Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).log_determinant();
                true
            })?;
            Ok((array(sign, &frames.layout, Layout::EMPTY, store)?, array(log, &frames.layout, Layout::EMPTY, store.real())?))
        })
    }

    #[doc = crate::stacked_doc!(lu_lower)]
    pub fn lu_lower(&self, factors: ArrayView<'_>) -> Result<Array, Error> { self.triangle("lu_lower", factors, true) }

    #[doc = crate::stacked_doc!(lu_upper)]
    pub fn lu_upper(&self, factors: ArrayView<'_>) -> Result<Array, Error> { self.triangle("lu_upper", factors, false) }

    /// The unit lower triangle `l` or the upper triangle `u` of each matrix's packed LU factors.
    fn triangle(&self, function: &str, factors: ArrayView<'_>, lower: bool) -> Result<Array, Error> {
        let Matrices { compute, store, size, frames, items: fi, .. } = self.matrices(function, factors)?;
        let n = square(function, size)?;
        crate::with_inexact!(compute, T => {
            let a = fi.of::<T>();
            let x = self.slots(&frames, n * n, |i, out: &mut [T], _: &mut ()| {
                let f = Lu::new(n, &a[frames.items(0, i, n * n)], &[][..]);
                out.copy_from_slice(&if lower { f.lower() } else { f.upper() })
            })?;
            Ok(array(x, &frames.layout, [n, n], store)?)
        })
    }

    #[doc = crate::stacked_doc!(cholesky)]
    pub fn cholesky(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("cholesky", a)?;
        let n = square("cholesky", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let l = self.slots(&frames, n * n, |i, out: &mut [T], _: &mut ()| match super::cholesky(&a[frames.items(0, i, n * n)], n) {
                Some(l) => out.copy_from_slice(&l),
                None => out.fill(T::from_real(<T as Scalar>::Real::nan())),
            })?;
            Ok(array(l, &frames.layout, [n, n], store)?)
        })
    }

    #[doc = crate::stacked_doc!(qr)]
    pub fn qr(&self, a: ArrayView<'_>, complete: bool) -> Result<(Array, Array), Error> {
        let Matrices { compute, store, size: [m, n], frames, items: ai, .. } = self.matrices("qr", a)?;
        let rows = if complete { m } else { m.min(n) };
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let qr = |i: usize, (q, r): (&mut [T], &mut [T]), w: &mut QrWork<T>| { w.qr_into(&a[frames.items(0, i, m * n)], [m, n], complete, q, r); true };
            let ((q, r), _) = stack::<(Vec<T>, Vec<T>), _>(frames.count, [m * rows, rows * n], &qr)?;
            Ok((array(q, &frames.layout, [m, rows], store)?, array(r, &frames.layout, [rows, n], store)?))
        })
    }

    #[doc = crate::stacked_doc!(svd)]
    pub fn svd(&self, a: ArrayView<'_>, rank: Option<usize>) -> Result<(Array, Array, Array), Error> {
        let Matrices { compute, store, size: [m, n], frames, items: ai, .. } = self.matrices("svd", a)?;
        let k = rank.map_or(m.min(n), |r| r.min(m).min(n));
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let ((u, sv, v), _) = stack::<(Vec<T>, Vec<<T as Scalar>::Real>, Vec<T>), ()>(frames.count, [m * k, k, n * k], &|i, (u, sv, v), _| {
                let x = &a[frames.items(0, i, m * n)];
                let f = if rank.is_some() { super::truncated_svd(x, [m, n], k) } else { super::svd(x, [m, n]) };
                u.copy_from_slice(&f.u);
                sv.copy_from_slice(&f.s);
                v.copy_from_slice(&f.v);
                true
            })?;
            Ok((array(u, &frames.layout, [m, k], store)?, array(sv, &frames.layout, [k], store.real())?, array(v, &frames.layout, [n, k], store)?))
        })
    }

    #[doc = crate::stacked_doc!(svd_solve)]
    pub fn svd_solve(&self, u: ArrayView<'_>, s: ArrayView<'_>, v: ArrayView<'_>, b: ArrayView<'_>) -> Result<Array, Error> {
        let (compute, store) = self.dtypes("svd_solve", &[u, s, v, b])?;
        let (lu, ls, lv, lb) = (u.layout(), s.layout(), v.layout(), b.layout());
        let ([m, n], p) = factors("svd_solve", u, s, v)?;
        let (core, k, vector) = rhs("svd_solve", lb, m)?;
        let (ru, rv, rb) = (lu.shape().len(), lv.shape().len(), lb.shape().len());
        let frames = Frames::new(&[(lu, 2), (ls, 1), (lv, 2), (lb, core)])?;
        let (ui, si, vi, bi) = (u.cast(compute)?, s.cast(compute.real())?, v.cast(compute)?, paired(b.cast(compute)?, lb, rb - core, lu, ru - 2, "svd_solve")?);
        crate::with_inexact!(compute, T => {
            let (u, sv, v, b) = (ui.of::<T>(), si.of::<<T as Scalar>::Real>(), vi.of::<T>(), bi.of::<T>());
            let x = self.slots(&frames, n * k, |i, out: &mut [T], _: &mut ()| {
                out.copy_from_slice(&Svd::<T, _, _>::new([m, n], &u[frames.items(0, i, m * p)], &sv[frames.items(1, i, p)], &v[frames.items(2, i, n * p)]).solve(&b[frames.items(3, i, m * k)], k))
            })?;
            Ok(array(x, &frames.layout, solution(lv.axes(rv - 2..rv - 1), lb, vector), store)?)
        })
    }

    #[doc = crate::stacked_doc!(svd_pinv)]
    pub fn svd_pinv(&self, u: ArrayView<'_>, s: ArrayView<'_>, v: ArrayView<'_>) -> Result<Array, Error> {
        let (compute, store) = self.dtypes("svd_pinv", &[u, s, v])?;
        let (lu, ls, lv) = (u.layout(), s.layout(), v.layout());
        let ([m, n], p) = factors("svd_pinv", u, s, v)?;
        let (ru, rv) = (lu.shape().len(), lv.shape().len());
        let frames = Frames::new(&[(lu, 2), (ls, 1), (lv, 2)])?;
        let (ui, si, vi) = (u.cast(compute)?, s.cast(compute.real())?, v.cast(compute)?);
        crate::with_inexact!(compute, T => {
            let (u, sv, v) = (ui.of::<T>(), si.of::<<T as Scalar>::Real>(), vi.of::<T>());
            let x = self.slots(&frames, n * m, |i, out: &mut [T], _: &mut ()| {
                out.copy_from_slice(&Svd::<T, _, _>::new([m, n], &u[frames.items(0, i, m * p)], &sv[frames.items(1, i, p)], &v[frames.items(2, i, n * p)]).pseudoinverse())
            })?;
            Ok(array(x, &frames.layout, lv.axes(rv - 2..rv - 1).concat(&lu.axes(ru - 2..ru - 1)), store)?)
        })
    }

    #[doc = crate::stacked_doc!(svdvals)]
    pub fn svdvals(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, store, size: [m, n], frames, items: ai, .. } = self.matrices("svdvals", a)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let sv = self.slots(&frames, m.min(n), |i, out: &mut [<T as Scalar>::Real], _: &mut ()| out.copy_from_slice(&super::singular_values(&a[frames.items(0, i, m * n)], [m, n])))?;
            Ok(array(sv, &frames.layout, [m.min(n)], store.real())?)
        })
    }

    #[doc = crate::stacked_doc!(matrix_rank)]
    pub fn matrix_rank(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, size: [m, n], frames, items: ai, .. } = self.matrices("matrix_rank", a)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let r = self.slots(&frames, 1, |i, out: &mut [i64], _: &mut ()| out[0] = super::rank(&a[frames.items(0, i, m * n)], [m, n]) as i64)?;
            Ok(array(r, &frames.layout, Layout::EMPTY, I64)?)
        })
    }

    #[doc = crate::stacked_doc!(norm)]
    pub fn norm(&self, a: ArrayView<'_>, ord: super::Norm) -> Result<Array, Error> {
        let Matrices { compute, store, size: [m, n], frames, items: ai, .. } = self.matrices("norm", a)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let r = self.slots(&frames, 1, |i, out: &mut [<T as Scalar>::Real], _: &mut ()| out[0] = super::norm(&a[frames.items(0, i, m * n)], [m, n], ord))?;
            Ok(array(r, &frames.layout, Layout::EMPTY, store.real())?)
        })
    }

    #[doc = crate::stacked_doc!(cond)]
    pub fn cond(&self, a: ArrayView<'_>, p: super::Norm) -> Result<Array, Error> {
        let Matrices { compute, store, size: [m, n], frames, items: ai, .. } = self.matrices("cond", a)?;
        if p != super::Norm::Two { square("cond", [m, n])?; }
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let r = self.slots(&frames, 1, |i, out: &mut [<T as Scalar>::Real], _: &mut ()| out[0] = super::cond(&a[frames.items(0, i, m * n)], [m, n], p))?;
            Ok(array(r, &frames.layout, Layout::EMPTY, store.real())?)
        })
    }

    #[doc = crate::stacked_doc!(eig)]
    pub fn eig(&self, a: ArrayView<'_>) -> Result<(Array, Array), Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("eig", a)?;
        let n = square("eig", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let nan = Complex::new(<T as Scalar>::Real::nan(), <T as Scalar>::Real::nan());
            let ((w, v), _) = stack::<(Vec<Complex<<T as Scalar>::Real>>, Vec<Complex<<T as Scalar>::Real>>), ()>(frames.count, [n, n * n], &|i, (w, v), _| {
                match super::eig(&a[frames.items(0, i, n * n)], n) {
                    Some(f) => { w.copy_from_slice(&f.values); v.copy_from_slice(&f.vectors); }
                    None => { w.fill(nan); v.fill(nan); }
                }
                true
            })?;
            Ok((array(w, &frames.layout, [n], store.complex())?, array(v, &frames.layout, [n, n], store.complex())?))
        })
    }

    #[doc = crate::stacked_doc!(eigvals)]
    pub fn eigvals(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("eigvals", a)?;
        let n = square("eigvals", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let nan = Complex::new(<T as Scalar>::Real::nan(), <T as Scalar>::Real::nan());
            let w = self.slots(&frames, n, |i, out: &mut [Complex<<T as Scalar>::Real>], _: &mut ()| match super::eigenvalues(&a[frames.items(0, i, n * n)], n) {
                Some(w) => out.copy_from_slice(&w),
                None => out.fill(nan),
            })?;
            Ok(array(w, &frames.layout, [n], store.complex())?)
        })
    }

    #[doc = crate::stacked_doc!(eigh)]
    pub fn eigh(&self, a: ArrayView<'_>) -> Result<(Array, Array), Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("eigh", a)?;
        let n = square("eigh", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let ((w, v), _) = stack::<(Vec<<T as Scalar>::Real>, Vec<T>), ()>(frames.count, [n, n * n], &|i, (w, v), _| {
                match super::eigh(&a[frames.items(0, i, n * n)], n) {
                    Some(f) => { w.copy_from_slice(&f.values); v.copy_from_slice(&f.vectors); }
                    None => { w.fill(<T as Scalar>::Real::nan()); v.fill(T::from_real(<T as Scalar>::Real::nan())); }
                }
                true
            })?;
            Ok((array(w, &frames.layout, [n], store.real())?, array(v, &frames.layout, [n, n], store)?))
        })
    }

    #[doc = crate::stacked_doc!(eigvalsh)]
    pub fn eigvalsh(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("eigvalsh", a)?;
        let n = square("eigvalsh", size)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let w = self.slots(&frames, n, |i, out: &mut [<T as Scalar>::Real], _: &mut ()| match super::hermitian_eigenvalues(&a[frames.items(0, i, n * n)], n) {
                Some(w) => out.copy_from_slice(&w),
                None => out.fill(<T as Scalar>::Real::nan()),
            })?;
            Ok(array(w, &frames.layout, [n], store.real())?)
        })
    }

    #[doc = crate::stacked_doc!(permanent)]
    pub fn permanent(&self, a: ArrayView<'_>) -> Result<Array, Error> {
        let Matrices { compute, store, size, frames, items: ai, .. } = self.matrices("permanent", a)?;
        let n = square("permanent", size)?;
        if n >= 64 { return Err(Error::new(Kind::Domain, format!("permanent takes fewer than 64 rows, not {n}")).into()); }
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let x = cells(&frames, |i, _: &mut ()| exact::permanent(&a[frames.items(0, i, n * n)], n))?.into_iter().collect::<Result<Vec<_>, _>>()?;
            Ok(array(x, &frames.layout, Layout::EMPTY, store)?)
        })
    }

    #[doc = crate::stacked_doc!(echelon)]
    pub fn echelon(&self, a: ArrayView<'_>, columns: Option<usize>) -> Result<(Array, Array, Array), Error> {
        let (compute, store) = self.dtypes("echelon", &[a])?;
        let &[m, n] = a.layout().shape() else { return Err(Error::new(Kind::Rank, "echelon takes one matrix").into()) };
        let columns = columns.unwrap_or(n);
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let f = exact::echelon(ai.of::<T>().to_vec(), [m, n], columns)?;
            let r = f.pivots.len();
            let pivots = f.pivots.into_iter().map(|c| c as i64).collect();
            Ok((array(f.rows, &Layout::EMPTY, [r, n], store)?, array(pivots, &Layout::EMPTY, [r], I64)?, array(vec![f.determinant], &Layout::EMPTY, Layout::EMPTY, store)?))
        })
    }
}

/// `f` of each cell of `frames`, in order. The cells split across rustymath's threads, with scratch `S` that a thread's cells share. It checks for an interrupt between rounds of cells, and an interrupt's error stops the call.
pub fn cells<R: Send, S: Default>(frames: &Frames, f: impl Fn(usize, &mut S) -> R + Sync) -> Result<Vec<R>, Error> { crate::parallel::each(frames.count, f) }

/// One stack of matrices that a matrix function takes alone, as [`Stacked::matrices`] reads it: its compute and store dtypes, its layout, its matrices' rows and columns, its frames, and its items in the compute dtype.
struct Matrices<'a> { compute: DType, store: DType, layout: &'a Layout, size: [usize; 2], frames: Frames, items: Items<'a> }

/// The rows and columns of the matrices in `layout`. A Rank error for fewer than two axes.
fn matrix(function: &str, layout: &Layout) -> Result<[usize; 2], Error> {
    match *layout.shape() {
        [.., m, n] => Ok([m, n]),
        _ => Err(Error::new(Kind::Rank, format!("{function} needs matrices, with at least two axes"))),
    }
}

/// The size of `function`'s square matrices. A Length error for a matrix that isn't square.
fn square(function: &str, [m, n]: [usize; 2]) -> Result<usize, Error> {
    if m == n { Ok(n) } else { Err(Error::new(Kind::Length, format!("{function} needs square matrices, not {m} by {n}"))) }
}

/// The cell axes of a right-hand side with `layout`, its number of columns, and whether it's one vector, beside matrices of `rows` rows. A Length error where its rows differ.
fn rhs(function: &str, layout: &Layout, rows: usize) -> Result<(usize, usize, bool), Error> {
    let (core, r, k) = match *layout.shape() {
        [] => return Err(Error::new(Kind::Rank, format!("{function} needs a right-hand side with at least one axis"))),
        [r] => (1, r, 1),
        [.., r, k] => (2, r, k),
    };
    if r != rows { return Err(Error::new(Kind::Length, format!("{function}'s right-hand side has {r} rows, not {rows}"))); }
    Ok((core, k, core == 1))
}

/// The cell axes of a solution: `rows`, then the right-hand side's columns unless it's a vector.
fn solution(rows: Layout, rhs: &Layout, vector: bool) -> Layout {
    let r = rhs.shape().len();
    if vector { rows } else { rows.concat(&rhs.axes(r - 1..r)) }
}

/// `items`, of an argument of `layout`, read so that its axis `axis` pairs by key with axis `along` of `with`, as [`Mapping::contract`] pairs them. Axes without keys pair by position. A Length error where both have keys and they differ.
pub fn paired<'a>(items: Items<'a>, layout: &Layout, axis: usize, with: &Layout, along: usize, function: &str) -> Result<Items<'a>, Error> {
    let contracted = Mapping::contract(with, along, layout, axis).map_err(|_| Error::new(Kind::Length, format!("{function} needs the same keys on the axes it pairs")))?;
    let Mapping::Steps { steps, .. } = contracted else { return Ok(items) };
    let [outer, len, inner] = Lanes::new(layout.shape(), axis)?.dims();
    let table = Vec::from(steps).into_iter().next().expect("a contraction maps one axis").scaled(inner);
    let plan = [Steps::along(outer, len * inner), table, Steps::along(inner, 1)];
    Ok(Items::Owned(Buffer::walked(items.view(), 0, &plan, outer * plan[1].len() * inner)))
}

/// The size of each of `function`'s square factor matrices, laid out by `factors`, and the row positions in `p`, as [`Stacked::lu`] gives them. Positions read as index positions do ([`signed_positions`](crate::structure::signed_positions)), so a float must be a whole number. A Length error where `p`'s cells don't hold that many positions. An Index error for a position outside 0 to the matrices' size.
fn factored(function: &str, factors: &Layout, p: ArrayView<'_>) -> Result<(usize, Vec<usize>), Error> {
    let n = square(function, matrix(function, factors)?)?;
    if p.layout().shape().last() != Some(&n) { return Err(Error::new(Kind::Length, format!("{function} needs {n} row positions for each matrix"))); }
    let items = p.cast(p.dtype())?;
    let positions = crate::structure::signed_positions(items.view())?.into_iter().map(|r| usize::try_from(r?).ok().filter(|&r| r < n)).collect::<Option<Vec<_>>>();
    Ok((n, positions.ok_or_else(|| Error::new(Kind::Index, format!("{function} needs row positions from 0 to {}", n.saturating_sub(1))))?))
}

/// The rows and columns of each of `function`'s matrices, from their thin SVD factors `u`, `s` and `v`, as [`Stacked::svd`] gives them, and how many singular values each has. A Length error where the factors' shapes don't fit together, and a Domain error for complex singular values, which `svd` never gives.
fn factors(function: &str, u: ArrayView<'_>, s: ArrayView<'_>, v: ArrayView<'_>) -> Result<([usize; 2], usize), Error> {
    if matches!(s.dtype(), DType::Complex(_)) { return Err(Error::new(Kind::Domain, format!("{function} needs real singular values"))); }
    let ([m, p], [n, q]) = (matrix(function, u.layout())?, matrix(function, v.layout())?);
    if p != q || s.layout().shape().last() != Some(&p) { return Err(Error::new(Kind::Length, format!("{function} needs u and v with as many columns as s has values"))); }
    Ok(([m, n], p))
}

/// An array of `buffer`'s items, the cells of `frame` in order, each with the axes `cell`, at the dtype `store`.
pub fn stored(buffer: Buffer, frame: &Layout, cell: impl Into<Layout>, store: DType) -> Result<Array, Error> {
    let buffer = if buffer.dtype() == store { buffer } else { Buffer::cast(buffer.view(), store)? };
    Array::new(frame.concat(&cell.into()), buffer)
}

/// [`stored`] for `items`.
fn array<T: Element>(items: Vec<T>, frame: &Layout, cell: impl Into<Layout>, store: DType) -> Result<Array, Error> { stored(T::buffer(items), frame, cell, store) }
