//! Exact arithmetic on Python integers and `fractions.Fraction`. An argument is an `int`, a `Fraction`, nested lists or tuples of them, or a buffer of integers or Booleans, such as a NumPy array. Results are Fractions, in nested lists for arrays. Elementwise functions pair their arguments' axes as rustymath broadcasts them: axes with the same name first, then the rest from the front. Matrix functions take one matrix, not stacks. Long operations poll for Ctrl-C.
use crate::convert::{Arg, released, signals};
use crate::integer::radix;
use num_rational::BigRational;
use num_traits::{One, ToPrimitive};
use pyo3::{exceptions::{PyOverflowError, PyTypeError, PyValueError}, prelude::*, sync::PyOnceLock, types::{PyComplex, PyFloat, PyInt, PyType}};
use rustymath::agreement::Agreement;
use rustymath::error::{Error, Kind};
use rustymath::layout::Layout;
use rustymath::{exact, linalg::exact as matrix, polynomial};
use rustymath_buffer::{py_err, read, walk::{Collect, walked}};

type R = BigRational;

/// Python's `fractions.Fraction`.
fn fraction(py: Python<'_>) -> PyResult<&Bound<'_, PyType>> {
    static FRACTION: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    FRACTION.import(py, "fractions", "Fraction")
}

fn inexact() -> PyErr { PyTypeError::new_err("exact arithmetic takes integers and fractions.Fraction, not floats") }

/// One part of nested exact values: one value, or the integers of a buffer.
struct Rationals(Vec<R>);

/// The collector for exact values.
struct Exact;

impl Collect for Exact {
    type Piece = Rationals;
    fn int(&mut self, n: i128) -> Rationals { Rationals(vec![R::from_integer(n.into())]) }
    fn piece(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<Option<(Rationals, Vec<usize>)>> {
        if let Some(r) = read(obj)? {
            if !r.view().is_empty() && !r.view().dtype().is_integral() { return Err(inexact()); }
            let shape = r.shape().to_vec();
            return Ok(Some((Rationals(Arg::from(r).items::<i64>()?.iter().map(|&n| R::from_integer(n.into())).collect()), shape)));
        }
        let value = if obj.is_instance_of::<PyInt>() {
            R::from_integer(obj.extract()?)
        } else if obj.is_instance(fraction(obj.py())?)? {
            R::new(obj.getattr("numerator")?.extract()?, obj.getattr("denominator")?.extract()?)
        } else if obj.is_instance_of::<PyFloat>() || obj.is_instance_of::<PyComplex>() {
            return Err(inexact());
        } else {
            return Ok(None);
        };
        Ok(Some((Rationals(vec![value]), vec![])))
    }
}

/// The layout and values of `x`.
fn values(x: &Bound<'_, PyAny>) -> PyResult<(Layout, Vec<R>)> {
    let (layout, pieces) = walked(x, &mut Exact)?;
    Ok((layout, pieces.into_iter().flat_map(|p| p.0).collect()))
}

/// The shape and values of `x`, after checking that it has `axes` axes, or raising `expected`.
fn shaped(x: &Bound<'_, PyAny>, axes: usize, expected: &str) -> PyResult<(Vec<usize>, Vec<R>)> {
    let (layout, items) = values(x)?;
    if layout.shape().len() != axes { return Err(py_err(Error::new(Kind::Rank, expected.to_owned()))); }
    Ok((layout.shape().to_vec(), items))
}

fn vector(x: &Bound<'_, PyAny>) -> PyResult<Vec<R>> { Ok(shaped(x, 1, "expected a vector")?.1) }

fn matrix_of(a: &Bound<'_, PyAny>) -> PyResult<([usize; 2], Vec<R>)> {
    let (shape, a) = shaped(a, 2, "expected a matrix")?;
    Ok(([shape[0], shape[1]], a))
}

fn square(a: &Bound<'_, PyAny>) -> PyResult<(usize, Vec<R>)> {
    let ([m, n], a) = matrix_of(a)?;
    if m != n { return Err(py_err(Error::new(Kind::Length, "expected a square matrix"))); }
    Ok((n, a))
}

/// The multiplier `m`, 1 where it isn't given.
fn multiplier(m: Option<&Bound<'_, PyAny>>) -> PyResult<R> {
    let Some(m) = m else { return Ok(R::one()) };
    Ok(shaped(m, 0, "m must be a scalar")?.1.pop().expect("a scalar has one value"))
}

/// `items` as Fractions, nested in lists by `shape`, or one Fraction where `shape` is empty.
fn output<'py>(py: Python<'py>, shape: &[usize], items: Vec<R>) -> PyResult<Bound<'py, PyAny>> {
    let fraction = fraction(py)?;
    let items = each(items, || py.check_signals(), |x| fraction.call1((x.numer().clone(), x.denom().clone())))?;
    rustymath_buffer::nested(py, shape, &|i| Ok(items[i].clone()))
}

/// `f` on each of `items`, calling `check` every 1024 items, whose error stops the loop.
fn each<T, U>(items: impl IntoIterator<Item = T>, mut check: impl FnMut() -> PyResult<()>, mut f: impl FnMut(T) -> PyResult<U>) -> PyResult<Vec<U>> {
    items.into_iter().enumerate().map(|(i, x)| {
        if i % 1024 == 0 { check()?; }
        f(x)
    }).collect()
}

/// `f` on each pair of items of `x` and `y`, broadcast together, with the GIL released.
fn binary<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, f: impl Fn(&R, &R) -> PyResult<R> + Sync) -> PyResult<Bound<'py, PyAny>> {
    let ((lx, x), (ly, y)) = (values(x)?, values(y)?);
    let agreement = Agreement::new(&lx, &ly).map_err(py_err)?;
    let fill = || py_err(Error::new(Kind::Domain, "exact arithmetic has no fill for a missing key"));
    let out = released(py, || each(0..agreement.len, || Ok(signals()?), |i| f(&x[agreement.left.get(i).ok_or_else(fill)?], &y[agreement.right.get(i).ok_or_else(fill)?])))?;
    output(py, agreement.layout.shape(), out)
}

/// `f` on each item of `x`, in `x`'s shape, with the GIL released.
fn pointwise<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, f: impl Fn(&R) -> PyResult<R> + Sync) -> PyResult<Bound<'py, PyAny>> {
    let (layout, x) = values(x)?;
    let out = released(py, || each(&x, || Ok(signals()?), &f))?;
    output(py, layout.shape(), out)
}

/// The nonnegative rational gcd of `x` and `y`.
#[pyfunction]
fn gcd<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { binary(py, x, y, |x, y| Ok(exact::gcd(x, y))) }

/// The rational lcm of `x` and `y`, with the sign of `x*y`.
#[pyfunction]
fn lcm<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { binary(py, x, y, |x, y| Ok(exact::lcm(x, y))) }

/// `x` modulo `y`, with `y`'s sign. A zero `y` gives `x`.
#[pyfunction]
fn remainder<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { binary(py, x, y, |x, y| Ok(exact::residue(y, x))) }

/// The binomial coefficient `n` choose `k`, for a rational `n` and a whole `k` that fits `int64`.
#[pyfunction]
fn binom<'py>(py: Python<'py>, n: &Bound<'py, PyAny>, k: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    binary(py, n, k, |n, k| {
        if !k.is_integer() { return Err(PyValueError::new_err("binom needs a whole k")); }
        let k = k.to_integer().to_i64().ok_or_else(|| PyOverflowError::new_err("binom needs k to fit int64"))?;
        exact::binom_with(n, k, || Ok(signals()?))
    })
}

/// The least-norm least-squares solution of `a @ x = b`, for a matrix `a` and a vector or matrix `b` with a row for each row of `a`.
#[pyfunction]
fn solve<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ([m, n], a) = matrix_of(a)?;
    let (layout, b) = values(b)?;
    let (k, shape) = match *layout.shape() {
        [r] if r == m => (1, vec![n]),
        [r, k] if r == m => (k, vec![n, k]),
        _ => return Err(py_err(Error::new(Kind::Length, "b must be a vector or a matrix with a row for each row of a"))),
    };
    output(py, &shape, released(py, || matrix::solve_with(&a, &b, [m, n], k, signals))?)
}

/// The pseudoinverse of the matrix `a`, which is its inverse where `a` is square and nonsingular.
#[pyfunction]
fn pinv<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ([m, n], a) = matrix_of(a)?;
    output(py, &[n, m], released(py, || matrix::pinv_with(&a, [m, n], signals))?)
}

/// The determinant of a square matrix.
#[pyfunction]
fn det<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (n, a) = square(a)?;
    output(py, &[], vec![released(py, || matrix::determinant_with(&a, n, signals))?])
}

/// The permanent of a square matrix of fewer than 64 rows, in exponential time. Ctrl-C cancels it.
#[pyfunction]
fn permanent<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (n, a) = square(a)?;
    if n >= 64 { return Err(PyValueError::new_err("permanent needs fewer than 64 rows")); }
    output(py, &[], vec![released(py, || matrix::permanent_with(&a, n, signals))?])
}

/// The reduced row echelon form of the matrix `a`, with its zero rows left out, the pivot columns, and the signed product of the pivots before scaling. Pivots come from the first `columns` columns, or from every column where `columns` isn't given or the matrix has fewer.
#[pyfunction]
#[pyo3(signature = (a, columns=None))]
fn echelon<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, columns: Option<usize>) -> PyResult<(Bound<'py, PyAny>, Vec<usize>, Bound<'py, PyAny>)> {
    let ([m, n], a) = matrix_of(a)?;
    let columns = columns.unwrap_or(n);
    let e = released(py, || matrix::echelon_with(a, [m, n], columns, signals))?;
    Ok((output(py, &[e.pivots.len(), n], e.rows)?, e.pivots, output(py, &[], vec![e.determinant])?))
}

/// The polynomial with coefficients `c`, a vector in ascending order, at each item of `x`.
#[pyfunction]
fn polyval<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, c: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let c = vector(c)?;
    pointwise(py, x, |x| polynomial::value_with(&c, x, || Ok(signals()?)))
}

/// The coefficients of `m` times the product of `x - root` over the vector `roots`.
#[pyfunction]
#[pyo3(signature = (roots, m=None))]
fn polyfromroots<'py>(py: Python<'py>, roots: &Bound<'py, PyAny>, m: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let (m, roots) = (multiplier(m)?, vector(roots)?);
    let c = released(py, || polynomial::from_roots_with(m, &roots, signals))?;
    output(py, &[c.len()], c)
}

/// The coefficients of the derivative of the polynomial with coefficients `c`. A constant's derivative has none.
#[pyfunction]
fn polyder<'py>(py: Python<'py>, c: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let c = vector(c)?;
    let d = released(py, || polynomial::derivative_with(&c, signals))?;
    output(py, &[d.len()], d)
}

/// `m` times the product of `x - root` over the vector `roots`, at each item of `x`.
#[pyfunction]
#[pyo3(signature = (x, roots, m=None))]
fn polyvalfromroots<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, roots: &Bound<'py, PyAny>, m: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let (roots, m) = (vector(roots)?, multiplier(m)?);
    pointwise(py, x, |x| polynomial::factored_with(&m, &roots, x, || Ok(signals()?)))
}

/// The mixed-radix value of each vector of digits along the last axis of `digits`. `bases` is a nonempty vector with one base for each digit, and its first base is unused.
#[pyfunction]
fn decode<'py>(py: Python<'py>, bases: &Bound<'py, PyAny>, digits: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ((lb, bases), (ld, digits)) = (values(bases)?, values(digits)?);
    radix(lb.shape(), ld.shape())?;
    let out = released(py, || -> PyResult<Vec<R>> {
        let mut out = Vec::with_capacity(digits.len() / bases.len());
        for group in digits.chunks(bases.len().saturating_mul(1024)) {
            signals()?;
            out.extend(polynomial::decode(&bases, group));
        }
        Ok(out)
    })?;
    let shape = ld.shape();
    output(py, &shape[..shape.len() - 1], out)
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(gcd, m)?, wrap_pyfunction!(lcm, m)?, wrap_pyfunction!(remainder, m)?, wrap_pyfunction!(binom, m)?, wrap_pyfunction!(solve, m)?,
        wrap_pyfunction!(pinv, m)?, wrap_pyfunction!(det, m)?, wrap_pyfunction!(permanent, m)?, wrap_pyfunction!(echelon, m)?, wrap_pyfunction!(polyval, m)?,
        wrap_pyfunction!(polyfromroots, m)?, wrap_pyfunction!(polyder, m)?, wrap_pyfunction!(polyvalfromroots, m)?, wrap_pyfunction!(decode, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}

