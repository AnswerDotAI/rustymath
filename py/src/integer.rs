//! Checked mixed-radix conversion of integers. Booleans and integers of any dtype compute as `i64`.
use crate::{array::{core, give}, convert::{one, released}};
use pyo3::{exceptions::PyTypeError, prelude::*};
use rustymath::error::{Error, Kind};
use rustymath_buffer::py_err;

/// The shape and `i64` items of `x`, which must hold Booleans or integers unless it's empty.
fn integers(x: &Bound<'_, PyAny>) -> PyResult<(Vec<usize>, Vec<i64>)> {
    let x = one(x)?;
    if !x.view().is_empty() && !x.view().dtype().is_integral() { return Err(PyTypeError::new_err("mixed-radix integer conversion takes integers")); }
    Ok((x.layout().shape().to_vec(), x.items()?.into_owned()))
}

/// Checks that `bases` is a nonempty vector, and that the last axis of `digits` has one digit for each base. `polynomial.decode` and `exact.decode` check the same.
pub(crate) fn radix(bases: &[usize], digits: &[usize]) -> PyResult<()> {
    match (bases, digits.last()) {
        (&[n], Some(&d)) if n > 0 && d == n => Ok(()),
        _ => Err(py_err(Error::new(Kind::Length, "decode needs a nonempty vector of bases and one digit for each base"))),
    }
}

/// The mixed-radix digits of each integer in `values`, most significant first, along a new last axis. A zero base takes the whole remaining value. A quotient that doesn't fit `int64` raises `OverflowError`.
#[pyfunction]
fn encode<'py>(py: Python<'py>, bases: &Bound<'py, PyAny>, values: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ((b, bases), (mut shape, values)) = (integers(bases)?, integers(values)?);
    if b.len() != 1 { return Err(py_err(Error::new(Kind::Rank, "bases must be a vector"))); }
    let digits = released(py, || rustymath::integer::encode(&bases, &values))
        .ok_or_else(|| py_err(Error::new(Kind::Overflow, "an encoding quotient doesn't fit int64")))?;
    shape.push(bases.len());
    give(py, core(digits, &shape)?, false)
}

/// The `int64` value of each vector of digits along the last axis of `digits`. The first base is unused. A value that doesn't fit `int64` raises `OverflowError`. A vector of digits gives a Python integer.
#[pyfunction]
fn decode<'py>(py: Python<'py>, bases: &Bound<'py, PyAny>, digits: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let ((b, bases), (mut shape, digits)) = (integers(bases)?, integers(digits)?);
    radix(&b, &shape)?;
    let values =
        released(py, || rustymath::integer::decode(&bases, &digits)).ok_or_else(|| py_err(Error::new(Kind::Overflow, "a decoded value doesn't fit int64")))?;
    shape.pop();
    give(py, core(values, &shape)?, true)
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> { for f in [wrap_pyfunction!(encode, m)?, wrap_pyfunction!(decode, m)?] { m.add_function(f)?; } Ok(()) }
