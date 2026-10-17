//! rustymath's `linalg` module as Python functions, which `python/rustymath/linalg.py` wraps. Each takes stacks of matrices in its arguments' last two axes, or a right-hand side vector in its last axis. The axes before those form each argument's frame. Frames pair as rustymath broadcasts, through [`Frames`]: axes with the same name first, then the rest from the front. The cells of a stack split across rustymath's threads, and Ctrl-C stops a call between cells.
//!
//! float32, float64, complex64 and complex128 matrices compute at their own width. float16 matrices compute in float32, with results rounded to float16. Booleans and integers compute in the float that `DType::inexact` gives them, float32 for Booleans, `u8` and `i16` and float64 for `i32` and `i64`, except in the matrix products, where integers compute in int64 and raise `OverflowError` for a result that doesn't fit. Results that are always real, such as singular values, take the real dtype of the matrices' width, and a general matrix's eigenvalues the complex dtype. A matrix that `cholesky`, `eigh`, `eig` or their value-only forms can't factor gives NaN results.
use rustymath::num_complex::Complex;
use num_traits::{Float as _, Zero as _};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustymath::agreement::{Frames, Mapping};
use rustymath::array::Array as Core;
use rustymath::buffer::{Buffer, DType, Element, FloatWidth, Items, View, Width};
use rustymath::error::{Error, Kind};
use rustymath::layout::{Lanes, Layout};
use rustymath::linalg::{self, AndOr, Lu, LuWork, MaxPlus, MinPlus, OrAnd, PlusEqual, PlusNotEqual, PlusTimes, Scalar, Svd};
use rustymath::parallel::stack;
use rustymath::operation::{Kernels, ProductKernel};
use rustymath_buffer::py_err;

use crate::array::give;
use crate::convert::{Call, convert};
use crate::functions::{missing, native_binary};

/// A matrix function's data arguments at one dtype, the dtype that it computes in, and the dtype that its results take.
struct Stack<'py> { call: Call<'py>, compute: DType, store: DType }

impl<'py> Stack<'py> {
    /// Converts `args` for `function`, which computes in the dtype that `DType::inexact` gives and stores its results in the other. Characters raise.
    fn new(function: &str, args: &[&Bound<'py, PyAny>]) -> PyResult<Self> {
        let call = convert(args)?;
        let dtype = call.args[0].view().dtype();
        let (compute, store) = dtype.inexact().ok_or_else(|| missing(function, dtype))?;
        Ok(Self { call, compute, store })
    }

    /// Argument `k`'s layout.
    fn layout(&self, k: usize) -> Layout { self.call.args[k].layout() }

    /// Argument `k`'s items at the computing dtype.
    fn items(&self, k: usize) -> PyResult<Items<'_>> { self.call.args[k].at(self.compute) }
}

/// The frames of arguments with these layouts, each with its number of axes in a cell.
fn framed(args: &[(&Layout, usize)]) -> PyResult<Frames> { Frames::new(args).map_err(py_err) }

/// The rows and columns of the matrices in `layout`. A Rank error for fewer than two axes.
fn matrix(function: &str, layout: &Layout) -> PyResult<[usize; 2]> {
    match *layout.shape() {
        [.., m, n] => Ok([m, n]),
        _ => Err(py_err(Error::new(Kind::Rank, format!("{function} needs matrices, with at least two axes")))),
    }
}

/// The size of `function`'s square matrices. A Length error for a matrix that isn't square.
fn square(function: &str, [m, n]: [usize; 2]) -> PyResult<usize> {
    if m == n { Ok(n) } else { Err(py_err(Error::new(Kind::Length, format!("{function} needs square matrices, not {m} by {n}")))) }
}

/// The cell axes of a right-hand side with `layout`, its number of columns, and whether it's one vector, beside matrices of `rows` rows. A Length error where its rows differ.
fn rhs(function: &str, layout: &Layout, rows: usize) -> PyResult<(usize, usize, bool)> {
    let (core, r, k) = match *layout.shape() {
        [] => return Err(py_err(Error::new(Kind::Rank, format!("{function} needs a right-hand side with at least one axis")))),
        [r] => (1, r, 1),
        [.., r, k] => (2, r, k),
    };
    if r != rows { return Err(py_err(Error::new(Kind::Length, format!("{function}'s right-hand side has {r} rows, not {rows}")))); }
    Ok((core, k, core == 1))
}

/// The cell axes of a solution: `rows`, then the right-hand side's columns unless it's a vector.
fn solution(rows: Layout, rhs: &Layout, vector: bool) -> Layout {
    let r = rhs.shape().len();
    if vector { rows } else { rows.concat(&rhs.axes(r - 1..r)) }
}

/// `items`, of an argument of `layout`, read so that its axis `axis` pairs by key with axis `along` of `with`, as [`Mapping::contract`] pairs them. Axes without keys pair by position. A Length error where both have keys and they differ.
fn paired<'a>(items: Items<'a>, layout: &Layout, axis: usize, with: &Layout, along: usize, function: &str) -> PyResult<Items<'a>> {
    let contracted = Mapping::contract(with, along, layout, axis).map_err(|_| py_err(Error::new(Kind::Length, format!("{function} needs the same keys on the axes it pairs"))))?;
    let Mapping::Positions(p) = contracted else { return Ok(items) };
    let lanes = Lanes::new(layout.shape(), axis).map_err(py_err)?;
    let [outer, _, inner] = lanes.dims();
    let offsets: Vec<i64> = (0..outer).flat_map(|i| p.iter().flat_map(move |&j| (0..inner).map(move |k| lanes.offset(i, j, k) as i64))).collect();
    Ok(Items::Owned(Buffer::gathered(items.view(), &offsets)))
}

/// The `per` items of cell `at` in `x`.
fn at<T>(x: &[T], at: usize, per: usize) -> &[T] { &x[at * per..][..per] }

/// `n` NaNs, the results of a matrix that fails.
fn nans<T: Scalar>(n: usize) -> Vec<T> { vec![T::from_real(T::Real::nan()); n] }

/// `f` of each cell of `frames`, in order. The cells split across rustymath's threads, with scratch `S` that a thread's cells share. Ctrl-C stops the call between cells.
fn run<R: Send, S: Default>(py: Python<'_>, frames: &Frames, f: impl Fn(usize, &mut S) -> R + Sync) -> PyResult<Vec<R>> {
    let cells = stack(frames.count, 1, &|i: usize, out: &mut [Option<R>], s: &mut S| out[0] = Some(f(i, s)), || py.check_signals())?;
    Ok(cells.into_iter().map(|c| c.expect("`stack` writes every cell")).collect())
}

/// An array of `buffer`'s items, the cells of `frame` in order, each with the axes `cell`, at the dtype `store`. A cell axis keeps the keys and name of the argument axis it indexes, where it has one.
fn stored(buffer: Buffer, frame: &Layout, cell: impl Into<Layout>, store: DType) -> PyResult<Core> {
    let buffer = if buffer.dtype() == store { buffer } else { Buffer::cast(buffer.view(), store).map_err(py_err)? };
    Core::new(frame.concat(&cell.into()), buffer).map_err(py_err)
}

/// [`stored`] for `items`.
fn array<T: Element>(items: Vec<T>, frame: &Layout, cell: impl Into<Layout>, store: DType) -> PyResult<Core> { stored(T::buffer(items), frame, cell, store) }

/// `core` as a Python result: a Python number where it has no axes.
fn result(py: Python<'_>, core: Core) -> PyResult<Bound<'_, PyAny>> {
    let scalar = core.layout().shape().is_empty();
    give(py, core, scalar)
}

/// The int64 dtype of row positions, ranks and counts.
const I64: DType = DType::Integer(Width::I64);

/// The solution `x` of `a @ x == b` for each matrix `a` and right-hand side `b`, a vector or matrices. A square `a` with no negligible LU pivot solves through LU, with no cutoff. Any other `a` gives the least-squares solution with the least norm.
#[pyfunction]
fn solve<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("solve", &[a, b])?;
    let (la, lb) = (s.layout(0), s.layout(1));
    let (ra, rb) = (la.shape().len(), lb.shape().len());
    let [m, n] = matrix("solve", &la)?;
    let (core, k, vector) = rhs("solve", &lb, m)?;
    let frames = framed(&[(&la, 2), (&lb, core)])?;
    rustymath::with_inexact!(s.compute, T => {
        let (ai, bi) = (s.items(0)?, paired(s.items(1)?, &lb, rb - core, &la, ra - 2, "solve")?);
        let (a, b) = (ai.of::<T>(), bi.of::<T>());
        let x = run(py, &frames, |i, w: &mut LuWork<T>| w.solve(at(a, frames.cell(0, i), m * n), at(b, frames.cell(1, i), m * k), [m, n], k))?;
        result(py, array(x.concat(), &frames.layout, solution(la.axes(ra - 1..ra), &lb, vector), s.store)?)
    })
}

/// The least-squares solution `x` of `a @ x == b` with the least norm, and `a`'s singular values in descending order, for each matrix `a` and right-hand side `b`, a vector or matrices.
#[pyfunction]
fn lstsq<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("lstsq", &[a, b])?;
    let (la, lb) = (s.layout(0), s.layout(1));
    let (ra, rb) = (la.shape().len(), lb.shape().len());
    let [m, n] = matrix("lstsq", &la)?;
    let (core, k, vector) = rhs("lstsq", &lb, m)?;
    let frames = framed(&[(&la, 2), (&lb, core)])?;
    rustymath::with_inexact!(s.compute, T => {
        let (ai, bi) = (s.items(0)?, paired(s.items(1)?, &lb, rb - core, &la, ra - 2, "lstsq")?);
        let (a, b) = (ai.of::<T>(), bi.of::<T>());
        let (x, sv): (Vec<_>, Vec<_>) = run(py, &frames, |i, _: &mut ()| linalg::lstsq(at(a, frames.cell(0, i), m * n), at(b, frames.cell(1, i), m * k), [m, n], k))?.into_iter().unzip();
        Ok((result(py, array(x.concat(), &frames.layout, solution(la.axes(ra - 1..ra), &lb, vector), s.store)?)?,
            result(py, array(sv.concat(), &frames.layout, [m.min(n)], s.store.real())?)?))
    })
}

/// The pseudoinverse of each matrix. A square matrix with no negligible LU pivot gives its inverse through LU, with no cutoff. Any other gives the pseudoinverse with `lstsq`'s cutoff.
#[pyfunction]
fn pinv<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("pinv", &[a])?;
    let la = s.layout(0);
    let ra = la.shape().len();
    let [m, n] = matrix("pinv", &la)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let x = run(py, &frames, |i, w: &mut LuWork<T>| {
            let mut out = vec![T::zero(); m * n];
            w.pinv_into(at(a, frames.cell(0, i), m * n), [m, n], &mut out);
            out
        })?;
        result(py, array(x.concat(), &frames.layout, la.axes([ra - 1, ra - 2].into_iter()), s.store)?)
    })
}

/// The determinant of each square matrix, from its LU factorization.
#[pyfunction]
fn det<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("det", &[a])?;
    let la = s.layout(0);
    let n = square("det", matrix("det", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let d = run(py, &frames, |i, w: &mut LuWork<T>| w.factor(at(a, frames.cell(0, i), n * n), n).determinant())?;
        result(py, array(d, &frames.layout, Layout::EMPTY, s.store)?)
    })
}

/// The sign of each square matrix's determinant, and the natural logarithm of its magnitude, which can't overflow. A singular matrix gives sign 0 and logarithm -inf.
#[pyfunction]
fn slogdet<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("slogdet", &[a])?;
    let la = s.layout(0);
    let n = square("slogdet", matrix("slogdet", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let (sign, log): (Vec<_>, Vec<_>) = run(py, &frames, |i, w: &mut LuWork<T>| w.factor(at(a, frames.cell(0, i), n * n), n).log_determinant())?.into_iter().unzip();
        Ok((result(py, array(sign, &frames.layout, Layout::EMPTY, s.store)?)?, result(py, array(log, &frames.layout, Layout::EMPTY, s.store.real())?)?))
    })
}

/// The packed LU factors of each square matrix, with partial pivoting, and the positions of its rows. The factors are in rustymath's order: row `j` holds column `j` of the packed matrix. Row `i` of `l @ u` is row `p[i]` of the matrix.
#[pyfunction]
fn lu<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("lu", &[a])?;
    let la = s.layout(0);
    let n = square("lu", matrix("lu", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let (factors, p): (Vec<_>, Vec<_>) = run(py, &frames, |i, w: &mut LuWork<T>| {
            let f = w.factor(at(a, frames.cell(0, i), n * n), n);
            (f.factors.to_vec(), f.perm.iter().map(|&r| r as i64).collect::<Vec<_>>())
        })?.into_iter().unzip();
        Ok((result(py, array(factors.concat(), &frames.layout, [n, n], s.store)?)?, result(py, array(p.concat(), &frames.layout, [n], I64)?)?))
    })
}

/// LU factors as [`lu`] gives them, and their row positions: the factors' stack, with any further arguments at their dtype, the size of each matrix, both layouts, and the positions.
struct Factored<'py> { stack: Stack<'py>, n: usize, factors: Layout, positions: Layout, p: Vec<usize> }

impl<'py> Factored<'py> {
    /// The factors `f` and row positions `p` for `function`, with `rest` at the factors' dtype. Row positions that aren't integers raise. An Index error for a row position outside 0 to the matrices' size.
    fn new(function: &str, f: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, rest: &[&Bound<'py, PyAny>]) -> PyResult<Self> {
        let stack = Stack::new(function, &[&[f], rest].concat())?;
        let factors = stack.layout(0);
        let n = square(function, matrix(function, &factors)?)?;
        let call = convert(&[p])?;
        let positions = call.args[0].layout();
        if positions.shape().last() != Some(&n) {
            return Err(py_err(Error::new(Kind::Length, format!("{function} needs {n} row positions for each matrix"))));
        }
        let p = call.args[0].items::<i64>()?.iter().map(|&r| usize::try_from(r).ok().filter(|&r| r < n)).collect::<Option<Vec<_>>>()
            .ok_or_else(|| py_err(Error::new(Kind::Index, format!("{function} needs row positions from 0 to {}", n.saturating_sub(1)))))?;
        Ok(Self { stack, n, factors, positions, p })
    }
}

/// `x` with `a @ x == b`, from the packed LU factors `factors` and row positions `p` of each matrix `a`, as `lu` gives them, without factoring again or applying a cutoff.
#[pyfunction]
fn lu_solve<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let f = Factored::new("lu_solve", factors, p, &[b])?;
    let (n, lb) = (f.n, f.stack.layout(1));
    let (core, k, vector) = rhs("lu_solve", &lb, n)?;
    let frames = framed(&[(&f.factors, 2), (&f.positions, 1), (&lb, core)])?;
    rustymath::with_inexact!(f.stack.compute, T => {
        let (fi, bi) = (f.stack.items(0)?, f.stack.items(1)?);
        let (a, b, p) = (fi.of::<T>(), bi.of::<T>(), &f.p);
        let x = run(py, &frames, |i, _: &mut ()| {
            Lu::new(n, at(a, frames.cell(0, i), n * n), at(p, frames.cell(1, i), n)).solve(at(b, frames.cell(2, i), n * k), k)
        })?;
        result(py, array(x.concat(), &frames.layout, solution(Layout::from([n]), &lb, vector), f.stack.store)?)
    })
}

/// The inverse of each matrix, from its packed LU factors and row positions, as `lu` gives them.
#[pyfunction]
fn lu_inverse<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let f = Factored::new("lu_inverse", factors, p, &[])?;
    let n = f.n;
    let frames = framed(&[(&f.factors, 2), (&f.positions, 1)])?;
    rustymath::with_inexact!(f.stack.compute, T => {
        let fi = f.stack.items(0)?;
        let (a, p) = (fi.of::<T>(), &f.p);
        let x = run(py, &frames, |i, _: &mut ()| Lu::new(n, at(a, frames.cell(0, i), n * n), at(p, frames.cell(1, i), n)).inverse())?;
        result(py, array(x.concat(), &frames.layout, [n, n], f.stack.store)?)
    })
}

/// The determinant of each matrix, from its packed LU factors and row positions, as `lu` gives them.
#[pyfunction]
fn lu_det<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let f = Factored::new("lu_det", factors, p, &[])?;
    let n = f.n;
    let frames = framed(&[(&f.factors, 2), (&f.positions, 1)])?;
    rustymath::with_inexact!(f.stack.compute, T => {
        let fi = f.stack.items(0)?;
        let (a, p) = (fi.of::<T>(), &f.p);
        let d = run(py, &frames, |i, _: &mut ()| Lu::new(n, at(a, frames.cell(0, i), n * n), at(p, frames.cell(1, i), n)).determinant())?;
        result(py, array(d, &frames.layout, Layout::EMPTY, f.stack.store)?)
    })
}

/// The sign of each matrix's determinant and the natural logarithm of its magnitude, from its packed LU factors and row positions, as `lu` gives them.
#[pyfunction]
fn lu_slogdet<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let f = Factored::new("lu_slogdet", factors, p, &[])?;
    let n = f.n;
    let frames = framed(&[(&f.factors, 2), (&f.positions, 1)])?;
    rustymath::with_inexact!(f.stack.compute, T => {
        let fi = f.stack.items(0)?;
        let (a, p) = (fi.of::<T>(), &f.p);
        let (sign, log): (Vec<_>, Vec<_>) = run(py, &frames, |i, _: &mut ()| {
            Lu::new(n, at(a, frames.cell(0, i), n * n), at(p, frames.cell(1, i), n)).log_determinant()
        })?.into_iter().unzip();
        Ok((result(py, array(sign, &frames.layout, Layout::EMPTY, f.stack.store)?)?, result(py, array(log, &frames.layout, Layout::EMPTY, f.stack.store.real())?)?))
    })
}

/// The unit lower triangle `l` or the upper triangle `u` of each matrix's packed LU factors, as `lu` gives them.
fn triangle<'py>(py: Python<'py>, function: &str, factors: &Bound<'py, PyAny>, lower: bool) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new(function, &[factors])?;
    let lf = s.layout(0);
    let n = square(function, matrix(function, &lf)?)?;
    let frames = framed(&[(&lf, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let fi = s.items(0)?;
        let a = fi.of::<T>();
        let x = run(py, &frames, |i, _: &mut ()| {
            let f = Lu::new(n, at(a, frames.cell(0, i), n * n), &[][..]);
            if lower { f.lower() } else { f.upper() }
        })?;
        result(py, array(x.concat(), &frames.layout, [n, n], s.store)?)
    })
}

/// The unit lower triangle of each matrix's packed LU factors, as `lu` gives them.
#[pyfunction]
fn lu_lower<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { triangle(py, "lu_lower", factors, true) }

/// The upper triangle of each matrix's packed LU factors, as `lu` gives them.
#[pyfunction]
fn lu_upper<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { triangle(py, "lu_upper", factors, false) }

/// The lower triangular `l` with `l @ l.conj().mT == a` for each Hermitian positive definite matrix `a`, reading only `a`'s lower triangle.
#[pyfunction]
fn cholesky<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("cholesky", &[a])?;
    let la = s.layout(0);
    let n = square("cholesky", matrix("cholesky", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let l = run(py, &frames, |i, _: &mut ()| linalg::cholesky(at(a, frames.cell(0, i), n * n), n).unwrap_or_else(|| nans(n * n)))?;
        result(py, array(l.concat(), &frames.layout, [n, n], s.store)?)
    })
}

/// The QR factorization `(q, r)` of each matrix, with `r`'s diagonal real and never negative. With `complete`, `q` is square and `r` as tall as the matrix.
#[pyfunction]
#[pyo3(signature = (a, complete=false))]
fn qr<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, complete: bool) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("qr", &[a])?;
    let la = s.layout(0);
    let [m, n] = matrix("qr", &la)?;
    let rows = if complete { m } else { m.min(n) };
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let (q, r): (Vec<_>, Vec<_>) = run(py, &frames, |i, _: &mut ()| {
            let x = at(a, frames.cell(0, i), m * n);
            if complete { linalg::qr_complete(x, [m, n]) } else { linalg::qr(x, [m, n]) }
        })?.into_iter().unzip();
        Ok((result(py, array(q.concat(), &frames.layout, [m, rows], s.store)?)?, result(py, array(r.concat(), &frames.layout, [rows, n], s.store)?)?))
    })
}

/// The thin SVD of each matrix: `u`, the singular values `s` in descending order, and `v`, with `a == u @ (s[..., None] * v.conj().mT)`. With `rank`, only the `rank` largest singular values and their vectors, or fewer for a smaller matrix, from Halko, Martinsson and Tropp's randomized range finder.
#[pyfunction]
#[pyo3(signature = (a, rank=None))]
fn svd<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, rank: Option<usize>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("svd", &[a])?;
    let la = s.layout(0);
    let [m, n] = matrix("svd", &la)?;
    let k = rank.map_or(m.min(n), |r| r.min(m).min(n));
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let cells = run(py, &frames, |i, _: &mut ()| {
            let x = at(a, frames.cell(0, i), m * n);
            let f = if rank.is_some() { linalg::truncated_svd(x, [m, n], k) } else { linalg::svd(x, [m, n]) };
            (f.u, f.s, f.v)
        })?;
        let (mut u, mut sv, mut v) = (Vec::new(), Vec::new(), Vec::new());
        for (cu, cs, cv) in cells { u.extend(cu); sv.extend(cs); v.extend(cv); }
        Ok((result(py, array(u, &frames.layout, [m, k], s.store)?)?, result(py, array(sv, &frames.layout, [k], s.store.real())?)?,
            result(py, array(v, &frames.layout, [n, k], s.store)?)?))
    })
}

/// Thin SVD factors as [`svd`] gives them: the stack of `u`, `v` and any further arguments, the singular values, the size of each matrix and of its values, and the layouts.
struct Factors<'py> { stack: Stack<'py>, values: Call<'py>, size: [usize; 2], p: usize, layouts: [Layout; 3] }

impl<'py> Factors<'py> {
    /// The factors `u`, `s` and `v` for `function`, with `rest` at their dtype. A Length error where their shapes don't fit together.
    fn new(function: &str, u: &Bound<'py, PyAny>, s: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>, rest: &[&Bound<'py, PyAny>]) -> PyResult<Self> {
        let stack = Stack::new(function, &[&[u, v], rest].concat())?;
        let (lu, lv) = (stack.layout(0), stack.layout(1));
        let ([m, p], [n, q]) = (matrix(function, &lu)?, matrix(function, &lv)?);
        let values = convert(&[s])?;
        let ls = values.args[0].layout();
        if p != q || ls.shape().last() != Some(&p) {
            return Err(py_err(Error::new(Kind::Length, format!("{function} needs u and v with as many columns as s has values"))));
        }
        Ok(Self { stack, values, size: [m, n], p, layouts: [lu, ls, lv] })
    }

    /// The singular values at the real dtype of the factors' computing width, or a narrower real dtype that widens to it. A dtype that would narrow raises.
    fn values(&self) -> PyResult<Items<'_>> { self.values.args[0].at(self.stack.compute.real()) }
}

/// The least-norm least-squares solution `x` of `a @ x == b`, from the thin SVD factors `u`, `s` and `v` of each matrix `a`, as `svd` gives them, using `lstsq`'s cutoff.
#[pyfunction]
fn svd_solve<'py>(py: Python<'py>, u: &Bound<'py, PyAny>, s: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let f = Factors::new("svd_solve", u, s, v, &[b])?;
    let ([m, n], p, lb) = (f.size, f.p, f.stack.layout(2));
    let (core, k, vector) = rhs("svd_solve", &lb, m)?;
    let [lu, ls, lv] = &f.layouts;
    let (ru, rv, rb) = (lu.shape().len(), lv.shape().len(), lb.shape().len());
    let frames = framed(&[(lu, 2), (ls, 1), (lv, 2), (&lb, core)])?;
    rustymath::with_inexact!(f.stack.compute, T => {
        let (ui, si, vi, bi) = (f.stack.items(0)?, f.values()?, f.stack.items(1)?, paired(f.stack.items(2)?, &lb, rb - core, lu, ru - 2, "svd_solve")?);
        let (u, sv, v, b) = (ui.of::<T>(), si.of::<<T as Scalar>::Real>(), vi.of::<T>(), bi.of::<T>());
        let x = run(py, &frames, |i, _: &mut ()| {
            Svd::<T, _, _>::new([m, n], at(u, frames.cell(0, i), m * p), at(sv, frames.cell(1, i), p), at(v, frames.cell(2, i), n * p)).solve(at(b, frames.cell(3, i), m * k), k)
        })?;
        result(py, array(x.concat(), &frames.layout, solution(lv.axes(rv - 2..rv - 1), &lb, vector), f.stack.store)?)
    })
}

/// The pseudoinverse of each matrix, from its thin SVD factors `u`, `s` and `v`, as `svd` gives them, using `lstsq`'s cutoff.
#[pyfunction]
fn svd_pinv<'py>(py: Python<'py>, u: &Bound<'py, PyAny>, s: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let f = Factors::new("svd_pinv", u, s, v, &[])?;
    let ([m, n], p) = (f.size, f.p);
    let [lu, ls, lv] = &f.layouts;
    let (ru, rv) = (lu.shape().len(), lv.shape().len());
    let frames = framed(&[(lu, 2), (ls, 1), (lv, 2)])?;
    rustymath::with_inexact!(f.stack.compute, T => {
        let (ui, si, vi) = (f.stack.items(0)?, f.values()?, f.stack.items(1)?);
        let (u, sv, v) = (ui.of::<T>(), si.of::<<T as Scalar>::Real>(), vi.of::<T>());
        let x = run(py, &frames, |i, _: &mut ()| {
            Svd::<T, _, _>::new([m, n], at(u, frames.cell(0, i), m * p), at(sv, frames.cell(1, i), p), at(v, frames.cell(2, i), n * p)).pseudoinverse()
        })?;
        result(py, array(x.concat(), &frames.layout, lv.axes(rv - 2..rv - 1).concat(&lu.axes(ru - 2..ru - 1)), f.stack.store)?)
    })
}

/// The singular values of each matrix, in descending order.
#[pyfunction]
fn svdvals<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("svdvals", &[a])?;
    let la = s.layout(0);
    let [m, n] = matrix("svdvals", &la)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let sv = run(py, &frames, |i, _: &mut ()| linalg::singular_values(at(a, frames.cell(0, i), m * n), [m, n]))?;
        result(py, array(sv.concat(), &frames.layout, [m.min(n)], s.store.real())?)
    })
}

/// The rank of each matrix: how many singular values exceed machine epsilon times the larger side times the largest, as `lstsq` counts them.
#[pyfunction]
fn matrix_rank<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("matrix_rank", &[a])?;
    let la = s.layout(0);
    let [m, n] = matrix("matrix_rank", &la)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let r = run(py, &frames, |i, _: &mut ()| linalg::rank(at(a, frames.cell(0, i), m * n), [m, n]) as i64)?;
        result(py, array(r, &frames.layout, Layout::EMPTY, I64)?)
    })
}

/// The eigenvalues of each square matrix, in the order of its Schur form, and its unit eigenvectors in columns, both complex. A real matrix's complex pair comes as the one with a positive imaginary part, then its conjugate.
#[pyfunction]
fn eig<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("eig", &[a])?;
    let la = s.layout(0);
    let n = square("eig", matrix("eig", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let nan = Complex::new(<T as Scalar>::Real::nan(), <T as Scalar>::Real::nan());
        let (w, v): (Vec<_>, Vec<_>) = run(py, &frames, |i, _: &mut ()| {
            linalg::eig(at(a, frames.cell(0, i), n * n), n).map_or_else(|| (vec![nan; n], vec![nan; n * n]), |f| (f.values, f.vectors))
        })?.into_iter().unzip();
        Ok((result(py, array(w.concat(), &frames.layout, [n], s.store.complex())?)?, result(py, array(v.concat(), &frames.layout, [n, n], s.store.complex())?)?))
    })
}

/// The complex eigenvalues of each square matrix, in the order `eig` gives them.
#[pyfunction]
fn eigvals<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("eigvals", &[a])?;
    let la = s.layout(0);
    let n = square("eigvals", matrix("eigvals", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let nan = Complex::new(<T as Scalar>::Real::nan(), <T as Scalar>::Real::nan());
        let w = run(py, &frames, |i, _: &mut ()| linalg::eigenvalues(at(a, frames.cell(0, i), n * n), n).unwrap_or_else(|| vec![nan; n]))?;
        result(py, array(w.concat(), &frames.layout, [n], s.store.complex())?)
    })
}

/// The eigenvalues of each Hermitian matrix in ascending order, and its unit eigenvectors in columns, reading only the matrix's lower triangle.
#[pyfunction]
fn eigh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("eigh", &[a])?;
    let la = s.layout(0);
    let n = square("eigh", matrix("eigh", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let (w, v): (Vec<_>, Vec<_>) = run(py, &frames, |i, _: &mut ()| {
            linalg::eigh(at(a, frames.cell(0, i), n * n), n).map_or_else(|| (nans::<T>(n).into_iter().map(|x| x.re()).collect(), nans::<T>(n * n)), |f| (f.values, f.vectors))
        })?.into_iter().unzip();
        Ok((result(py, array(w.concat(), &frames.layout, [n], s.store.real())?)?, result(py, array(v.concat(), &frames.layout, [n, n], s.store)?)?))
    })
}

/// The eigenvalues of each Hermitian matrix in ascending order, reading only the matrix's lower triangle.
#[pyfunction]
fn eigvalsh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("eigvalsh", &[a])?;
    let la = s.layout(0);
    let n = square("eigvalsh", matrix("eigvalsh", &la)?)?;
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let w = run(py, &frames, |i, _: &mut ()| linalg::hermitian_eigenvalues(at(a, frames.cell(0, i), n * n), n).unwrap_or_else(|| vec![<T as Scalar>::Real::nan(); n]))?;
        result(py, array(w.concat(), &frames.layout, [n], s.store.real())?)
    })
}

/// The permanent of each square matrix, by Ryser's exponential-time formula. A ValueError for 64 rows or more.
#[pyfunction]
fn permanent<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let s = Stack::new("permanent", &[a])?;
    let la = s.layout(0);
    let n = square("permanent", matrix("permanent", &la)?)?;
    if n >= 64 { return Err(PyValueError::new_err(format!("permanent takes fewer than 64 rows, not {n}"))); }
    let frames = framed(&[(&la, 2)])?;
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let a = ai.of::<T>();
        let x = run(py, &frames, |i, _: &mut ()| linalg::exact::permanent(at(a, frames.cell(0, i), n * n), n))?;
        result(py, array(x, &frames.layout, Layout::EMPTY, s.store)?)
    })
}

/// The reduced row echelon form of one matrix: its nonzero rows, its pivot columns and the signed product of its pivots. With `columns`, it reduces only that many leading columns. It pivots on the first nonzero item, with no tolerance, so the decompositions give numerically stable solves of floats.
#[pyfunction]
#[pyo3(signature = (a, columns=None))]
fn echelon<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, columns: Option<usize>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let s = Stack::new("echelon", &[a])?;
    let la = s.layout(0);
    let &[m, n] = la.shape() else { return Err(py_err(Error::new(Kind::Rank, "echelon takes one matrix"))) };
    let columns = columns.unwrap_or(n);
    if columns > n { return Err(PyValueError::new_err(format!("echelon can reduce at most the matrix's {n} columns, not {columns}"))); }
    rustymath::with_inexact!(s.compute, T => {
        let ai = s.items(0)?;
        let f = linalg::exact::echelon_with(ai.of::<T>().to_vec(), [m, n], columns, || py.check_signals())?;
        let r = f.pivots.len();
        let pivots = f.pivots.into_iter().map(|c| c as i64).collect();
        Ok((result(py, array(f.rows, &Layout::EMPTY, [r, n], s.store)?)?, result(py, array(pivots, &Layout::EMPTY, [r], I64)?)?,
            result(py, array(vec![f.determinant], &Layout::EMPTY, Layout::EMPTY, s.store)?)?))
    })
}

/// A matrix product's arguments: their items, with the right one's contracted axis in the left one's key order, their frames and sizes `[n, k, m]`, the result's cell axes, and the name its errors use.
struct Product<'a, 'py> { py: Python<'py>, args: [View<'a>; 2], frames: &'a Frames, size: [usize; 3], cell: &'a Layout, name: &'a str }

impl<'py> Product<'_, 'py> {
    /// The products through `P`, at the dtype that its `kernels` compute in. Integer results, counts included, are int64. float16 matrices give float16 results, and other results keep their kernels' dtype. An Overflow error for an integer result that doesn't fit int64.
    fn run<P: Default>(&self, kernels: Kernels<ProductKernel<P>>) -> PyResult<Bound<'py, PyAny>> {
        let [x, y] = self.args;
        let mut stopped = None;
        let products = kernels.product(&P::default(), [x, y], self.frames, self.size, &mut || self.py.check_signals().map_err(|e| stopped = Some(e)).is_ok());
        let (buffer, ok) = match (products, stopped) {
            (Some(p), _) => p,
            (None, Some(e)) => return Err(e),
            (None, None) => return Err(missing(self.name, x.dtype())),
        };
        if !ok { return Err(py_err(Error::new(Kind::Overflow, format!("an integer result of {} doesn't fit int64", self.name)))); }
        self.finish(buffer)
    }

    fn finish(&self, buffer: Buffer) -> PyResult<Bound<'py, PyAny>> {
        let x = self.args[0];
        let half = DType::Float(FloatWidth::F16);
        let store = match buffer.dtype() { DType::Integer(_) => I64, DType::Float(_) if x.dtype() == half => half, d => d };
        result(self.py, stored(buffer, &self.frames.layout, self.cell.clone(), store)?)
    }

    fn generic(&self, f: &str, g: &str) -> PyResult<Bound<'py, PyAny>> {
        let (f, g) = (native_binary(f)?, native_binary(g)?);
        let dtype = self.args[0].dtype();
        let compute = match dtype { DType::Integer(_) => I64, DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), d => d };
        let wide = if dtype != compute {
            Some([Buffer::cast(self.args[0], compute).map_err(py_err)?, Buffer::cast(self.args[1], compute).map_err(py_err)?])
        } else { None };
        let [x, y] = wide.as_ref().map_or(self.args, |a| a.each_ref().map(Buffer::view));
        let [n, k, m] = self.size;
        if k == 0 {
            let empty = rustymath::agreement::Agreement { layout: Layout::EMPTY, len: 0, left: Mapping::Single, right: Mapping::Single };
            let terms = (g.apply)(x, y, &empty)?;
            return self.finish((f.fold)(terms.view(), [self.frames.count * n, 0, m], None)?);
        }
        let (frames, size) = (self.frames, self.size);
        let parts = run(self.py, frames, |i, _: &mut ()| {
            let (a, b) = (frames.cell(0, i), frames.cell(1, i));
            linalg::inner_with(x.slice(a * n * k..(a + 1) * n * k), y.slice(b * k * m..(b + 1) * k * m), size, g.apply, f.fold)
        })?;
        let mut parts = parts.into_iter();
        let mut out = parts.next().unwrap_or_else(|| linalg::inner_with(x.slice(0..0), y.slice(0..0), [0, k, 0], g.apply, f.fold))?;
        for part in parts {
            let part = part?;
            assert!(out.extend_from(part.view(), 0..part.len()), "native inner-product cells have one dtype");
        }
        self.finish(out)
    }
}

/// The inner product of native binary functions named `f` and `g`, on vectors, matrices or broadcast stacks. Recognized pairs use specialized kernels; other pairs apply `g` in bounded blocks and fold `f` from the right. `name` names the product in errors.
#[pyfunction]
#[pyo3(signature = (f, g, x1, x2, name=None))]
pub fn product<'py>(py: Python<'py>, f: &str, g: &str, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>, name: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    let name = name.unwrap_or("inner");
    let call = convert(&[x1, x2])?;
    let dtype = call.args[0].view().dtype();
    let (l1, l2) = (call.args[0].layout(), call.args[1].layout());
    let side = |l: &Layout| match l.shape().len() {
        0 => Err(py_err(Error::new(Kind::Rank, format!("{name} needs matrices or vectors, with at least one axis")))),
        1 => Ok(1),
        _ => Ok(2),
    };
    let (c1, c2) = (side(&l1)?, side(&l2)?);
    let (s1, s2, r1, r2) = (l1.shape(), l2.shape(), l1.shape().len(), l2.shape().len());
    let (n, k) = if c1 == 1 { (1, s1[0]) } else { (s1[r1 - 2], s1[r1 - 1]) };
    let (j, m) = if c2 == 1 { (s2[0], 1) } else { (s2[r2 - 2], s2[r2 - 1]) };
    if k != j { return Err(py_err(Error::new(Kind::Length, format!("{name} needs as many columns on the left as rows on the right, not {k} and {j}")))); }
    let frames = framed(&[(&l1, c1), (&l2, c2)])?;
    let x = call.args[0].at(dtype)?;
    let y = paired(call.args[1].at(dtype)?, &l2, r2 - c2, &l1, r1 - 1, name)?;
    let cell = l1.axes(r1 - c1..r1).contracted(&l2.axes(r2 - c2..r2));
    let p = Product { py, args: [x.view(), y.view()], frames: &frames, size: [n, k, m], cell: &cell, name };
    match (f, g) {
        ("add", "multiply") if dtype == DType::Boolean => p.run(rustymath::kernels!(Product OrAnd)),
        ("add", "multiply") => p.run(rustymath::kernels!(Product PlusTimes)),
        ("maximum", "add") => p.run(rustymath::kernels!(Product MaxPlus)),
        ("minimum", "add") => p.run(rustymath::kernels!(Product MinPlus)),
        ("logical_or", "logical_and") => p.run(rustymath::kernels!(Product OrAnd)),
        ("logical_and", "logical_or") => p.run(rustymath::kernels!(Product AndOr)),
        ("add", "equal") => p.run(rustymath::kernels!(Product PlusEqual)),
        ("add", "not_equal") => p.run(rustymath::kernels!(Product PlusNotEqual)),
        _ => p.generic(f, g),
    }
}

/// The matrix product of `x1` and `x2`. A vector on the left acts as one row, and a vector on the right as one column, and the result leaves that axis out. Booleans give whether some `k` has both `x1[i, k]` and `x2[k, j]`. Integers compute in int64, and raise `OverflowError` for a result that doesn't fit.
#[pyfunction]
pub fn matmul<'py>(py: Python<'py>, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { product(py, "add", "multiply", x1, x2, Some("matmul")) }

/// Adds the linear algebra functions to the submodule `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(solve, m)?, wrap_pyfunction!(lstsq, m)?, wrap_pyfunction!(pinv, m)?, wrap_pyfunction!(det, m)?, wrap_pyfunction!(slogdet, m)?,
        wrap_pyfunction!(lu, m)?, wrap_pyfunction!(lu_solve, m)?, wrap_pyfunction!(lu_inverse, m)?, wrap_pyfunction!(lu_det, m)?, wrap_pyfunction!(lu_slogdet, m)?,
        wrap_pyfunction!(lu_lower, m)?, wrap_pyfunction!(lu_upper, m)?, wrap_pyfunction!(cholesky, m)?, wrap_pyfunction!(qr, m)?, wrap_pyfunction!(svd, m)?,
        wrap_pyfunction!(svd_solve, m)?, wrap_pyfunction!(svd_pinv, m)?, wrap_pyfunction!(svdvals, m)?, wrap_pyfunction!(matrix_rank, m)?, wrap_pyfunction!(eig, m)?,
        wrap_pyfunction!(eigvals, m)?, wrap_pyfunction!(eigh, m)?, wrap_pyfunction!(eigvalsh, m)?, wrap_pyfunction!(permanent, m)?, wrap_pyfunction!(echelon, m)?,
        wrap_pyfunction!(product, m)?, wrap_pyfunction!(matmul, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}

