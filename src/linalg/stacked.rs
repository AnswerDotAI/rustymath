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
use super::{exact, Lu, LuWork, Scalar, Svd};

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
    (eig) => { "The eigenvalues of each square matrix, in the order of its Schur form, and its unit eigenvectors in columns, both complex. A real matrix's complex pair comes as the one with a positive imaginary part, then its conjugate." };
    (eigvals) => { "The complex eigenvalues of each square matrix, in the order `eig` gives them." };
    (eigh) => { "The eigenvalues of each Hermitian matrix in ascending order, and its unit eigenvectors in columns, reading only the matrix's lower triangle." };
    (eigvalsh) => { "The eigenvalues of each Hermitian matrix in ascending order, reading only the matrix's lower triangle." };
    (permanent) => { "The permanent of each square matrix, by Ryser's exponential-time formula. A Domain error, Python's `ValueError`, for 64 rows or more." };
    (echelon) => { "The reduced row echelon form of one matrix: its nonzero rows, its pivot columns and the signed product of its pivots. With `columns`, it reduces only that many leading columns, or all of them where the matrix has fewer. It pivots on the first nonzero item, with no tolerance, so the decompositions give numerically stable solves of floats." };
}

/// The matrix functions of stacks. Each call computes in the dtype that rustymath's ladder gives its arguments, where exact numbers compute as floats of `float`, which the caller chooses by its own policy, as rustymath's Python API and basedpl each do. `check` runs between rounds of cells, and its error stops the call.
pub struct Stacked<C> { float: FloatWidth, check: C }

impl<C, E> Stacked<C> where C: FnMut() -> Result<(), E>, E: From<Error> {
    /// Matrix functions where exact numbers compute as floats of `float`, the caller's policy.
    pub fn new(float: FloatWidth, check: C) -> Self { Self { float, check } }

    /// `f` of each cell of `frames`, in order, as [`cells`] runs it, checking with `check`.
    pub fn run<R: Send, S: Default>(&mut self, frames: &Frames, f: impl Fn(usize, &mut S) -> R + Sync) -> Result<Vec<R>, E> {
        cells(frames, &mut self.check, f)
    }

    /// The `per` items of each cell of `frames`, which `f` writes into the cell's slot of the output, with scratch `S` that a thread's cells share.
    fn slots<B: Default + Send, S: Default>(&mut self, frames: &Frames, per: usize, f: impl Fn(usize, &mut [B], &mut S) + Sync) -> Result<Vec<B>, E> {
        stack(frames.count, per, &f, &mut self.check)
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
    pub fn solve(&mut self, a: ArrayView<'_>, b: ArrayView<'_>) -> Result<Array, E> {
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
    pub fn lstsq(&mut self, a: ArrayView<'_>, b: ArrayView<'_>) -> Result<(Array, Array), E> {
        let (compute, store) = self.dtypes("lstsq", &[a, b])?;
        let (la, lb) = (a.layout(), b.layout());
        let (ra, rb) = (la.shape().len(), lb.shape().len());
        let [m, n] = matrix("lstsq", la)?;
        let (core, k, vector) = rhs("lstsq", lb, m)?;
        let frames = Frames::new(&[(la, 2), (lb, core)])?;
        let (ai, bi) = (a.cast(compute)?, paired(b.cast(compute)?, lb, rb - core, la, ra - 2, "lstsq")?);
        crate::with_inexact!(compute, T => {
            let (a, b) = (ai.of::<T>(), bi.of::<T>());
            let (x, sv): (Vec<_>, Vec<_>) = self.run(&frames, |i, _: &mut ()| super::lstsq(&a[frames.items(0, i, m * n)], &b[frames.items(1, i, m * k)], [m, n], k))?.into_iter().unzip();
            Ok((array(x.concat(), &frames.layout, solution(la.axes(ra - 1..ra), lb, vector), store)?, array(sv.concat(), &frames.layout, [m.min(n)], store.real())?))
        })
    }

    #[doc = crate::stacked_doc!(pinv)]
    pub fn pinv(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("pinv", &[a])?;
        let la = a.layout();
        let ra = la.shape().len();
        let [m, n] = matrix("pinv", la)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let x = self.slots(&frames, m * n, |i, out: &mut [T], w: &mut LuWork<T>| w.pinv_into(&a[frames.items(0, i, m * n)], [m, n], out))?;
            Ok(array(x, &frames.layout, la.axes([ra - 1, ra - 2].into_iter()), store)?)
        })
    }

    #[doc = crate::stacked_doc!(det)]
    pub fn det(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("det", &[a])?;
        let la = a.layout();
        let n = square("det", matrix("det", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let d = self.run(&frames, |i, w: &mut LuWork<T>| w.factor(&a[frames.items(0, i, n * n)], n).determinant())?;
            Ok(array(d, &frames.layout, Layout::EMPTY, store)?)
        })
    }

    #[doc = crate::stacked_doc!(inverse)]
    pub fn inverse(&mut self, a: ArrayView<'_>) -> Result<Option<(Array, Array)>, E> {
        let (compute, store) = self.dtypes("inverse", &[a])?;
        let la = a.layout();
        let ra = la.shape().len();
        let n = square("inverse", matrix("inverse", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let cells = self.run(&frames, |i, w: &mut LuWork<T>| { let mut x = vec![T::default(); n * n]; w.inverse_into(&a[frames.items(0, i, n * n)], [n, n], &mut x).map(|d| (x, d)) })?;
            Ok(match cells.into_iter().collect::<Option<Vec<_>>>() {
                None => None,
                Some(cells) => { let (x, d): (Vec<_>, Vec<_>) = cells.into_iter().unzip(); Some((array(x.concat(), &frames.layout, la.axes([ra - 1, ra - 2].into_iter()), store)?, array(d, &frames.layout, Layout::EMPTY, store)?)) }
            })
        })
    }

    #[doc = crate::stacked_doc!(slogdet)]
    pub fn slogdet(&mut self, a: ArrayView<'_>) -> Result<(Array, Array), E> {
        let (compute, store) = self.dtypes("slogdet", &[a])?;
        let la = a.layout();
        let n = square("slogdet", matrix("slogdet", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let (sign, log): (Vec<_>, Vec<_>) = self.run(&frames, |i, w: &mut LuWork<T>| w.factor(&a[frames.items(0, i, n * n)], n).log_determinant())?.into_iter().unzip();
            Ok((array(sign, &frames.layout, Layout::EMPTY, store)?, array(log, &frames.layout, Layout::EMPTY, store.real())?))
        })
    }

    #[doc = crate::stacked_doc!(lu)]
    pub fn lu(&mut self, a: ArrayView<'_>) -> Result<(Array, Array), E> {
        let (compute, store) = self.dtypes("lu", &[a])?;
        let la = a.layout();
        let n = square("lu", matrix("lu", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let (factors, p): (Vec<_>, Vec<_>) = self.run(&frames, |i, w: &mut LuWork<T>| {
                let f = w.factor(&a[frames.items(0, i, n * n)], n);
                (f.factors.to_vec(), f.perm.iter().map(|&r| r as i64).collect::<Vec<_>>())
            })?.into_iter().unzip();
            Ok((array(factors.concat(), &frames.layout, [n, n], store)?, array(p.concat(), &frames.layout, [n], I64)?))
        })
    }

    #[doc = crate::stacked_doc!(lu_solve)]
    pub fn lu_solve(&mut self, factors: ArrayView<'_>, p: ArrayView<'_>, b: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("lu_solve", &[factors, b])?;
        let (lf, lb) = (factors.layout(), b.layout());
        let (n, positions) = factored("lu_solve", lf, p)?;
        let (core, k, vector) = rhs("lu_solve", lb, n)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1), (lb, core)])?;
        let (fi, bi) = (factors.cast(compute)?, b.cast(compute)?);
        crate::with_inexact!(compute, T => {
            let (a, b, p) = (fi.of::<T>(), bi.of::<T>(), &positions);
            let x = self.run(&frames, |i, _: &mut ()| {
                Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).solve(&b[frames.items(2, i, n * k)], k)
            })?;
            Ok(array(x.concat(), &frames.layout, solution(Layout::from([n]), lb, vector), store)?)
        })
    }

    #[doc = crate::stacked_doc!(lu_inverse)]
    pub fn lu_inverse(&mut self, factors: ArrayView<'_>, p: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("lu_inverse", &[factors])?;
        let lf = factors.layout();
        let (n, positions) = factored("lu_inverse", lf, p)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1)])?;
        let fi = factors.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let (a, p) = (fi.of::<T>(), &positions);
            let x = self.run(&frames, |i, _: &mut ()| Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).inverse())?;
            Ok(array(x.concat(), &frames.layout, [n, n], store)?)
        })
    }

    #[doc = crate::stacked_doc!(lu_det)]
    pub fn lu_det(&mut self, factors: ArrayView<'_>, p: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("lu_det", &[factors])?;
        let lf = factors.layout();
        let (n, positions) = factored("lu_det", lf, p)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1)])?;
        let fi = factors.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let (a, p) = (fi.of::<T>(), &positions);
            let d = self.run(&frames, |i, _: &mut ()| Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).determinant())?;
            Ok(array(d, &frames.layout, Layout::EMPTY, store)?)
        })
    }

    #[doc = crate::stacked_doc!(lu_slogdet)]
    pub fn lu_slogdet(&mut self, factors: ArrayView<'_>, p: ArrayView<'_>) -> Result<(Array, Array), E> {
        let (compute, store) = self.dtypes("lu_slogdet", &[factors])?;
        let lf = factors.layout();
        let (n, positions) = factored("lu_slogdet", lf, p)?;
        let frames = Frames::new(&[(lf, 2), (p.layout(), 1)])?;
        let fi = factors.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let (a, p) = (fi.of::<T>(), &positions);
            let (sign, log): (Vec<_>, Vec<_>) = self.run(&frames, |i, _: &mut ()| {
                Lu::new(n, &a[frames.items(0, i, n * n)], &p[frames.items(1, i, n)]).log_determinant()
            })?.into_iter().unzip();
            Ok((array(sign, &frames.layout, Layout::EMPTY, store)?, array(log, &frames.layout, Layout::EMPTY, store.real())?))
        })
    }

    #[doc = crate::stacked_doc!(lu_lower)]
    pub fn lu_lower(&mut self, factors: ArrayView<'_>) -> Result<Array, E> { self.triangle("lu_lower", factors, true) }

    #[doc = crate::stacked_doc!(lu_upper)]
    pub fn lu_upper(&mut self, factors: ArrayView<'_>) -> Result<Array, E> { self.triangle("lu_upper", factors, false) }

    /// The unit lower triangle `l` or the upper triangle `u` of each matrix's packed LU factors.
    fn triangle(&mut self, function: &str, factors: ArrayView<'_>, lower: bool) -> Result<Array, E> {
        let (compute, store) = self.dtypes(function, &[factors])?;
        let lf = factors.layout();
        let n = square(function, matrix(function, lf)?)?;
        let frames = Frames::new(&[(lf, 2)])?;
        let fi = factors.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = fi.of::<T>();
            let x = self.run(&frames, |i, _: &mut ()| {
                let f = Lu::new(n, &a[frames.items(0, i, n * n)], &[][..]);
                if lower { f.lower() } else { f.upper() }
            })?;
            Ok(array(x.concat(), &frames.layout, [n, n], store)?)
        })
    }

    #[doc = crate::stacked_doc!(cholesky)]
    pub fn cholesky(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("cholesky", &[a])?;
        let la = a.layout();
        let n = square("cholesky", matrix("cholesky", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let l = self.run(&frames, |i, _: &mut ()| super::cholesky(&a[frames.items(0, i, n * n)], n).unwrap_or_else(|| nans(n * n)))?;
            Ok(array(l.concat(), &frames.layout, [n, n], store)?)
        })
    }

    #[doc = crate::stacked_doc!(qr)]
    pub fn qr(&mut self, a: ArrayView<'_>, complete: bool) -> Result<(Array, Array), E> {
        let (compute, store) = self.dtypes("qr", &[a])?;
        let la = a.layout();
        let [m, n] = matrix("qr", la)?;
        let rows = if complete { m } else { m.min(n) };
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let (q, r): (Vec<_>, Vec<_>) = self.run(&frames, |i, _: &mut ()| {
                let x = &a[frames.items(0, i, m * n)];
                if complete { super::qr_complete(x, [m, n]) } else { super::qr(x, [m, n]) }
            })?.into_iter().unzip();
            Ok((array(q.concat(), &frames.layout, [m, rows], store)?, array(r.concat(), &frames.layout, [rows, n], store)?))
        })
    }

    #[doc = crate::stacked_doc!(svd)]
    pub fn svd(&mut self, a: ArrayView<'_>, rank: Option<usize>) -> Result<(Array, Array, Array), E> {
        let (compute, store) = self.dtypes("svd", &[a])?;
        let la = a.layout();
        let [m, n] = matrix("svd", la)?;
        let k = rank.map_or(m.min(n), |r| r.min(m).min(n));
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let cells = self.run(&frames, |i, _: &mut ()| {
                let x = &a[frames.items(0, i, m * n)];
                let f = if rank.is_some() { super::truncated_svd(x, [m, n], k) } else { super::svd(x, [m, n]) };
                (f.u, f.s, f.v)
            })?;
            let (mut u, mut sv, mut v) = (Vec::new(), Vec::new(), Vec::new());
            for (cu, cs, cv) in cells { u.extend(cu); sv.extend(cs); v.extend(cv); }
            Ok((array(u, &frames.layout, [m, k], store)?, array(sv, &frames.layout, [k], store.real())?, array(v, &frames.layout, [n, k], store)?))
        })
    }

    #[doc = crate::stacked_doc!(svd_solve)]
    pub fn svd_solve(&mut self, u: ArrayView<'_>, s: ArrayView<'_>, v: ArrayView<'_>, b: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("svd_solve", &[u, s, v, b])?;
        let (lu, ls, lv, lb) = (u.layout(), s.layout(), v.layout(), b.layout());
        let ([m, n], p) = factors("svd_solve", u, s, v)?;
        let (core, k, vector) = rhs("svd_solve", lb, m)?;
        let (ru, rv, rb) = (lu.shape().len(), lv.shape().len(), lb.shape().len());
        let frames = Frames::new(&[(lu, 2), (ls, 1), (lv, 2), (lb, core)])?;
        let (ui, si, vi, bi) = (u.cast(compute)?, s.cast(compute.real())?, v.cast(compute)?, paired(b.cast(compute)?, lb, rb - core, lu, ru - 2, "svd_solve")?);
        crate::with_inexact!(compute, T => {
            let (u, sv, v, b) = (ui.of::<T>(), si.of::<<T as Scalar>::Real>(), vi.of::<T>(), bi.of::<T>());
            let x = self.run(&frames, |i, _: &mut ()| {
                Svd::<T, _, _>::new([m, n], &u[frames.items(0, i, m * p)], &sv[frames.items(1, i, p)], &v[frames.items(2, i, n * p)]).solve(&b[frames.items(3, i, m * k)], k)
            })?;
            Ok(array(x.concat(), &frames.layout, solution(lv.axes(rv - 2..rv - 1), lb, vector), store)?)
        })
    }

    #[doc = crate::stacked_doc!(svd_pinv)]
    pub fn svd_pinv(&mut self, u: ArrayView<'_>, s: ArrayView<'_>, v: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("svd_pinv", &[u, s, v])?;
        let (lu, ls, lv) = (u.layout(), s.layout(), v.layout());
        let ([m, n], p) = factors("svd_pinv", u, s, v)?;
        let (ru, rv) = (lu.shape().len(), lv.shape().len());
        let frames = Frames::new(&[(lu, 2), (ls, 1), (lv, 2)])?;
        let (ui, si, vi) = (u.cast(compute)?, s.cast(compute.real())?, v.cast(compute)?);
        crate::with_inexact!(compute, T => {
            let (u, sv, v) = (ui.of::<T>(), si.of::<<T as Scalar>::Real>(), vi.of::<T>());
            let x = self.run(&frames, |i, _: &mut ()| {
                Svd::<T, _, _>::new([m, n], &u[frames.items(0, i, m * p)], &sv[frames.items(1, i, p)], &v[frames.items(2, i, n * p)]).pseudoinverse()
            })?;
            Ok(array(x.concat(), &frames.layout, lv.axes(rv - 2..rv - 1).concat(&lu.axes(ru - 2..ru - 1)), store)?)
        })
    }

    #[doc = crate::stacked_doc!(svdvals)]
    pub fn svdvals(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("svdvals", &[a])?;
        let la = a.layout();
        let [m, n] = matrix("svdvals", la)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let sv = self.run(&frames, |i, _: &mut ()| super::singular_values(&a[frames.items(0, i, m * n)], [m, n]))?;
            Ok(array(sv.concat(), &frames.layout, [m.min(n)], store.real())?)
        })
    }

    #[doc = crate::stacked_doc!(matrix_rank)]
    pub fn matrix_rank(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, _) = self.dtypes("matrix_rank", &[a])?;
        let la = a.layout();
        let [m, n] = matrix("matrix_rank", la)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let r = self.run(&frames, |i, _: &mut ()| super::rank(&a[frames.items(0, i, m * n)], [m, n]) as i64)?;
            Ok(array(r, &frames.layout, Layout::EMPTY, I64)?)
        })
    }

    #[doc = crate::stacked_doc!(eig)]
    pub fn eig(&mut self, a: ArrayView<'_>) -> Result<(Array, Array), E> {
        let (compute, store) = self.dtypes("eig", &[a])?;
        let la = a.layout();
        let n = square("eig", matrix("eig", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let nan = Complex::new(<T as Scalar>::Real::nan(), <T as Scalar>::Real::nan());
            let (w, v): (Vec<_>, Vec<_>) = self.run(&frames, |i, _: &mut ()| {
                super::eig(&a[frames.items(0, i, n * n)], n).map_or_else(|| (vec![nan; n], vec![nan; n * n]), |f| (f.values, f.vectors))
            })?.into_iter().unzip();
            Ok((array(w.concat(), &frames.layout, [n], store.complex())?, array(v.concat(), &frames.layout, [n, n], store.complex())?))
        })
    }

    #[doc = crate::stacked_doc!(eigvals)]
    pub fn eigvals(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("eigvals", &[a])?;
        let la = a.layout();
        let n = square("eigvals", matrix("eigvals", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let nan = Complex::new(<T as Scalar>::Real::nan(), <T as Scalar>::Real::nan());
            let w = self.run(&frames, |i, _: &mut ()| super::eigenvalues(&a[frames.items(0, i, n * n)], n).unwrap_or_else(|| vec![nan; n]))?;
            Ok(array(w.concat(), &frames.layout, [n], store.complex())?)
        })
    }

    #[doc = crate::stacked_doc!(eigh)]
    pub fn eigh(&mut self, a: ArrayView<'_>) -> Result<(Array, Array), E> {
        let (compute, store) = self.dtypes("eigh", &[a])?;
        let la = a.layout();
        let n = square("eigh", matrix("eigh", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let (w, v): (Vec<_>, Vec<_>) = self.run(&frames, |i, _: &mut ()| {
                super::eigh(&a[frames.items(0, i, n * n)], n).map_or_else(|| (vec![<T as Scalar>::Real::nan(); n], nans::<T>(n * n)), |f| (f.values, f.vectors))
            })?.into_iter().unzip();
            Ok((array(w.concat(), &frames.layout, [n], store.real())?, array(v.concat(), &frames.layout, [n, n], store)?))
        })
    }

    #[doc = crate::stacked_doc!(eigvalsh)]
    pub fn eigvalsh(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("eigvalsh", &[a])?;
        let la = a.layout();
        let n = square("eigvalsh", matrix("eigvalsh", la)?)?;
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let w = self.run(&frames, |i, _: &mut ()| super::hermitian_eigenvalues(&a[frames.items(0, i, n * n)], n).unwrap_or_else(|| vec![<T as Scalar>::Real::nan(); n]))?;
            Ok(array(w.concat(), &frames.layout, [n], store.real())?)
        })
    }

    #[doc = crate::stacked_doc!(permanent)]
    pub fn permanent(&mut self, a: ArrayView<'_>) -> Result<Array, E> {
        let (compute, store) = self.dtypes("permanent", &[a])?;
        let la = a.layout();
        let n = square("permanent", matrix("permanent", la)?)?;
        if n >= 64 { return Err(Error::new(Kind::Domain, format!("permanent takes fewer than 64 rows, not {n}")).into()); }
        let frames = Frames::new(&[(la, 2)])?;
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let a = ai.of::<T>();
            let x = self.run(&frames, |i, _: &mut ()| exact::permanent(&a[frames.items(0, i, n * n)], n))?;
            Ok(array(x, &frames.layout, Layout::EMPTY, store)?)
        })
    }

    #[doc = crate::stacked_doc!(echelon)]
    pub fn echelon(&mut self, a: ArrayView<'_>, columns: Option<usize>) -> Result<(Array, Array, Array), E> {
        let (compute, store) = self.dtypes("echelon", &[a])?;
        let &[m, n] = a.layout().shape() else { return Err(Error::new(Kind::Rank, "echelon takes one matrix").into()) };
        let columns = columns.unwrap_or(n);
        let ai = a.cast(compute)?;
        crate::with_inexact!(compute, T => {
            let f = exact::echelon_with(ai.of::<T>().to_vec(), [m, n], columns, &mut self.check)?;
            let r = f.pivots.len();
            let pivots = f.pivots.into_iter().map(|c| c as i64).collect();
            Ok((array(f.rows, &Layout::EMPTY, [r, n], store)?, array(pivots, &Layout::EMPTY, [r], I64)?, array(vec![f.determinant], &Layout::EMPTY, Layout::EMPTY, store)?))
        })
    }
}

/// `f` of each cell of `frames`, in order. The cells split across rustymath's threads, with scratch `S` that a thread's cells share. `check` runs between rounds of cells, and its error stops the call.
pub fn cells<R: Send, S: Default, E>(frames: &Frames, check: impl FnMut() -> Result<(), E>, f: impl Fn(usize, &mut S) -> R + Sync) -> Result<Vec<R>, E> {
    let cells = stack(frames.count, 1, &|i: usize, out: &mut [Option<R>], s: &mut S| out[0] = Some(f(i, s)), check)?;
    Ok(cells.into_iter().map(|c| c.expect("`stack` writes every cell")).collect())
}

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
    let table = steps.into_iter().next().expect("a contraction maps one axis").scaled(inner);
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

/// `n` NaNs, the results of a matrix that fails.
fn nans<T: Scalar>(n: usize) -> Vec<T> { vec![T::from_real(T::Real::nan()); n] }

/// An array of `buffer`'s items, the cells of `frame` in order, each with the axes `cell`, at the dtype `store`.
pub fn stored(buffer: Buffer, frame: &Layout, cell: impl Into<Layout>, store: DType) -> Result<Array, Error> {
    let buffer = if buffer.dtype() == store { buffer } else { Buffer::cast(buffer.view(), store)? };
    Array::new(frame.concat(&cell.into()), buffer)
}

/// [`stored`] for `items`.
fn array<T: Element>(items: Vec<T>, frame: &Layout, cell: impl Into<Layout>, store: DType) -> Result<Array, Error> { stored(T::buffer(items), frame, cell, store) }
