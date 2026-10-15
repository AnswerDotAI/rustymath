//! Exact arithmetic on Python integers and `fractions.Fraction`. The Python wrapper owns array shapes; these functions convert flat inputs once, compute with BigRational, and return flat Fraction lists. Long operations poll for Ctrl-C.
use num_bigint::BigInt;
use num_rational::BigRational;
use num_traits::ToPrimitive;
use pyo3::{exceptions::{PyOverflowError, PyTypeError, PyValueError}, prelude::*, types::PyList};
use rustymath::{exact, linalg::exact as matrix, polynomial};

type R = BigRational;

/// Python's `fractions.Fraction`.
fn fraction(py: Python<'_>) -> PyResult<Bound<'_, PyAny>> { py.import("fractions")?.getattr("Fraction") }

fn rational(x: &Bound<'_, PyAny>, fraction: &Bound<'_, PyAny>) -> PyResult<R> {
    if let Ok(n) = x.extract::<BigInt>() { return Ok(R::from_integer(n)); }
    if x.is_instance(fraction)? { return Ok(R::new(x.getattr("numerator")?.extract()?, x.getattr("denominator")?.extract()?)); }
    Err(PyTypeError::new_err("exact arithmetic takes integers and fractions.Fraction, not floats"))
}

fn read(x: &Bound<'_, PyAny>) -> PyResult<Vec<R>> {
    let py = x.py();
    let fraction = fraction(py)?;
    x.try_iter()?.enumerate().map(|(i, x)| {
        if i % 1024 == 0 { py.check_signals()?; }
        rational(&x?, &fraction)
    }).collect()
}

fn output<'py>(py: Python<'py>, xs: Vec<R>) -> PyResult<Bound<'py, PyList>> {
    let fraction = fraction(py)?;
    let xs = xs.into_iter().enumerate().map(|(i, x)| {
        if i % 1024 == 0 { py.check_signals()?; }
        fraction.call1((x.numer().clone(), x.denom().clone()))
    }).collect::<PyResult<Vec<_>>>()?;
    PyList::new(py, xs)
}

fn binary<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, mut f: impl FnMut(&R, &R) -> PyResult<R>) -> PyResult<Bound<'py, PyList>> {
    let (x, y) = (read(x)?, read(y)?);
    if x.len() != y.len() { return Err(PyValueError::new_err("inputs must have the same number of items")); }
    output(py, x.iter().zip(&y).enumerate().map(|(i, (x, y))| {
        if i % 1024 == 0 { py.check_signals()?; }
        f(x, y)
    }).collect::<PyResult<_>>()?)
}

#[pyfunction]
fn gcd<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> { binary(py, x, y, |x, y| Ok(exact::gcd(x, y))) }

#[pyfunction]
fn lcm<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> { binary(py, x, y, |x, y| Ok(exact::lcm(x, y))) }

#[pyfunction]
fn remainder<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    binary(py, x, y, |x, y| Ok(exact::residue(y, x)))
}

#[pyfunction]
fn binom<'py>(py: Python<'py>, n: &Bound<'py, PyAny>, k: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    binary(py, n, k, |n, k| {
        if !k.is_integer() { return Err(PyValueError::new_err("binom needs a whole k")); }
        let k = k.to_integer().to_i64().ok_or_else(|| PyOverflowError::new_err("binom needs k to fit in int64"))?;
        exact::binom_with(n, k, || py.check_signals())
    })
}

fn read_matrix(a: &Bound<'_, PyAny>, m: usize, n: usize) -> PyResult<Vec<R>> {
    let a = read(a)?;
    if a.len() != m * n { return Err(PyValueError::new_err("matrix shape does not match its items")); }
    Ok(a)
}

#[pyfunction]
fn solve<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>, m: usize, n: usize, k: usize) -> PyResult<Bound<'py, PyList>> {
    output(py, matrix::solve_with(&read_matrix(a, m, n)?, &read_matrix(b, m, k)?, [m, n], k, || py.check_signals())?)
}

#[pyfunction]
fn pinv<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, m: usize, n: usize) -> PyResult<Bound<'py, PyList>> {
    output(py, matrix::pinv_with(&read_matrix(a, m, n)?, [m, n], || py.check_signals())?)
}

#[pyfunction]
fn det<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, n: usize) -> PyResult<Bound<'py, PyList>> {
    output(py, vec![matrix::determinant_with(&read_matrix(a, n, n)?, n, || py.check_signals())?])
}

#[pyfunction]
fn permanent<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, n: usize) -> PyResult<Bound<'py, PyList>> {
    if n >= 64 { return Err(PyValueError::new_err("permanent needs fewer than 64 rows")); }
    output(py, vec![matrix::permanent_with(&read_matrix(a, n, n)?, n, || py.check_signals())?])
}

#[pyfunction]
fn echelon<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, m: usize, n: usize, columns: usize) -> PyResult<(Bound<'py, PyList>, Vec<usize>, Bound<'py, PyList>)> {
    if columns > n { return Err(PyValueError::new_err("columns cannot exceed the matrix width")); }
    let e = matrix::echelon_with(read_matrix(a, m, n)?, [m, n], columns, || py.check_signals())?;
    Ok((output(py, e.rows)?, e.pivots, output(py, vec![e.determinant])?))
}

#[pyfunction]
fn polyval<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, c: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let (x, c) = (read(x)?, read(c)?);
    output(py, x.iter().enumerate().map(|(i, x)| {
        if i % 1024 == 0 { py.check_signals()?; }
        polynomial::value_with(&c, x, || py.check_signals())
    }).collect::<PyResult<_>>()?)
}

#[pyfunction]
fn polyfromroots<'py>(py: Python<'py>, roots: &Bound<'py, PyAny>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let fraction = fraction(py)?;
    output(py, polynomial::from_roots_with(rational(m, &fraction)?, &read(roots)?, || py.check_signals())?)
}

#[pyfunction]
fn polyder<'py>(py: Python<'py>, c: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    output(py, polynomial::derivative_with(&read(c)?, || py.check_signals())?)
}

#[pyfunction]
fn polyvalfromroots<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, roots: &Bound<'py, PyAny>, m: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let fraction = fraction(py)?;
    let (x, roots, m) = (read(x)?, read(roots)?, rational(m, &fraction)?);
    output(py, x.iter().enumerate().map(|(i, x)| {
        if i % 1024 == 0 { py.check_signals()?; }
        polynomial::factored_with(&m, &roots, x, || py.check_signals())
    }).collect::<PyResult<_>>()?)
}

#[pyfunction]
fn decode<'py>(py: Python<'py>, bases: &Bound<'py, PyAny>, digits: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyList>> {
    let (bases, digits) = (read(bases)?, read(digits)?);
    if bases.is_empty() || digits.len() % bases.len() != 0 { return Err(PyValueError::new_err("decode needs one digit for each base")); }
    let mut out = Vec::with_capacity(digits.len() / bases.len());
    for group in digits.chunks(bases.len().saturating_mul(1024)) {
        py.check_signals()?;
        out.extend(polynomial::decode(&bases, group));
    }
    output(py, out)
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(gcd, m)?, wrap_pyfunction!(lcm, m)?, wrap_pyfunction!(remainder, m)?, wrap_pyfunction!(binom, m)?, wrap_pyfunction!(solve, m)?,
        wrap_pyfunction!(pinv, m)?, wrap_pyfunction!(det, m)?, wrap_pyfunction!(permanent, m)?, wrap_pyfunction!(echelon, m)?, wrap_pyfunction!(polyval, m)?,
        wrap_pyfunction!(polyfromroots, m)?, wrap_pyfunction!(polyder, m)?, wrap_pyfunction!(polyvalfromroots, m)?, wrap_pyfunction!(decode, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}
