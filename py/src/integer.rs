//! Checked mixed-radix conversion on int64 arrays.
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::{exceptions::PyOverflowError, prelude::*};

#[pyfunction]
fn encode<'py>(py: Python<'py>, bases: PyReadonlyArray1<'py, i64>, values: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let out = rustymath::integer::encode(bases.as_slice()?, values.as_slice()?)
        .ok_or_else(|| PyOverflowError::new_err("an encoding quotient does not fit int64"))?;
    Ok(PyArray1::from_vec(py, out))
}

#[pyfunction]
fn decode<'py>(py: Python<'py>, bases: PyReadonlyArray1<'py, i64>, digits: PyReadonlyArray1<'py, i64>) -> PyResult<Bound<'py, PyArray1<i64>>> {
    let (bases, digits) = (bases.as_slice()?, digits.as_slice()?);
    let out = rustymath::integer::decode(bases, digits).ok_or_else(|| PyOverflowError::new_err("a decoded value does not fit int64"))?;
    Ok(PyArray1::from_vec(py, out))
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(encode, m)?, wrap_pyfunction!(decode, m)?] { m.add_function(f)?; }
    Ok(())
}
