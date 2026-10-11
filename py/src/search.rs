//! rustymath's tolerant searches on vectors of float16, float32 or float64. The wrappers in `python/rustymath/__init__.py` give each
//! argument a contiguous vector of one of those dtypes and check `ct`. Each search checks for Ctrl-C as it goes, and stops with
//! `KeyboardInterrupt`.
use half::f16;
use numpy::{PyArray1, PyArrayDescrMethods, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rustymath::{tolerant, with_tolerance};

use crate::tolerance;

/// `positions` as an array of NumPy's index type.
fn indices(py: Python<'_>, positions: Vec<usize>) -> Bound<'_, PyArray1<isize>> { PyArray1::from_iter(py, positions.into_iter().map(|i| i as isize)) }

/// The position of the earliest item of each item's class within tolerance `ct`.
#[pyfunction]
pub fn classes<'py>(py: Python<'py>, x: &Bound<'py, PyUntypedArray>, ct: f64) -> PyResult<Bound<'py, PyArray1<isize>>> {
    floats!(x, F => {
        let x = x.extract::<PyReadonlyArray1<F>>()?;
        Ok(indices(py, with_tolerance!(tolerance(ct), |t| tolerant::classes_with(t, x.as_slice()?, || py.check_signals()))?))
    })
}

/// Whether each item of `x` is the first of its class within tolerance `ct`.
#[pyfunction]
pub fn firsts<'py>(py: Python<'py>, x: &Bound<'py, PyUntypedArray>, ct: f64) -> PyResult<Bound<'py, PyArray1<bool>>> {
    floats!(x, F => {
        let x = x.extract::<PyReadonlyArray1<F>>()?;
        Ok(PyArray1::from_vec(py, with_tolerance!(tolerance(ct), |t| tolerant::firsts(t, x.as_slice()?))))
    })
}

/// The position in `x` of the first item that matches each item of `y` within tolerance `ct`, or `len(x)` where none does.
#[pyfunction]
pub fn index_of<'py>(py: Python<'py>, x: &Bound<'py, PyUntypedArray>, y: &Bound<'py, PyUntypedArray>, ct: f64) -> PyResult<Bound<'py, PyArray1<isize>>> {
    floats!(x, F => {
        let (x, y) = (x.extract::<PyReadonlyArray1<F>>()?, y.extract::<PyReadonlyArray1<F>>()?);
        Ok(indices(py, with_tolerance!(tolerance(ct), |t| tolerant::index_of_with(t, x.as_slice()?, y.as_slice()?, || py.check_signals()))?))
    })
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(classes, m)?, wrap_pyfunction!(firsts, m)?, wrap_pyfunction!(index_of, m)?] { m.add_function(f)?; }
    Ok(())
}
