//! rustymath's searches on vectors of one dtype, which the wrappers in `python/rustymath/__init__.py` give each argument. Booleans and
//! integers match exactly, through `search`. float16, float32 and float64 match within tolerance `ct`, through `tolerant`. Each float
//! search checks for Ctrl-C as it goes, stopping with `KeyboardInterrupt`.
use half::f16;
use numpy::{PyArray1, PyArrayDescrMethods, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rustymath::{search, tolerant, with_tolerance};

use crate::tolerance;

/// Runs `$keys` with `$t` the Rust type of `$a`'s Boolean or integer dtype, or `$floats` with `$t` the type of its float dtype.
macro_rules! items {
    ($a:expr, $t:ident => keys: $keys:expr, floats: $floats:expr) => {{
        let dtype = $a.dtype();
        match (dtype.kind(), dtype.itemsize()) {
            (b'b', _) => { type $t = bool; $keys }
            (b'i', 1) => { type $t = i8; $keys }
            (b'i', 2) => { type $t = i16; $keys }
            (b'i', 4) => { type $t = i32; $keys }
            (b'i', _) => { type $t = i64; $keys }
            (b'u', 1) => { type $t = u8; $keys }
            (b'u', 2) => { type $t = u16; $keys }
            (b'u', 4) => { type $t = u32; $keys }
            (b'u', _) => { type $t = u64; $keys }
            (_, 2) => { type $t = f16; $floats }
            (_, 4) => { type $t = f32; $floats }
            _ => { type $t = f64; $floats }
        }
    }};
}

/// `positions` as an array of NumPy's index type.
fn indices(py: Python<'_>, positions: impl IntoIterator<Item = usize>) -> Bound<'_, PyArray1<isize>> {
    PyArray1::from_iter(py, positions.into_iter().map(|i| i as isize))
}

/// The classes of `x`, numbered from 0 in order of first position, as `search::numbered` gives them: each class's first position, each
/// item's class number, and each class's size.
#[pyfunction]
pub fn unique<'py>(
    py: Python<'py>, x: &Bound<'py, PyUntypedArray>, ct: f64,
) -> PyResult<(Bound<'py, PyArray1<isize>>, Bound<'py, PyArray1<isize>>, Bound<'py, PyArray1<isize>>)> {
    let classes = items!(x, T => keys: search::classes(x.extract::<PyReadonlyArray1<T>>()?.as_slice()?),
        floats: with_tolerance!(tolerance(ct), |t| tolerant::classes_with(t, x.extract::<PyReadonlyArray1<T>>()?.as_slice()?, || py.check_signals()))?);
    let search::Numbered { firsts, numbers, counts } = search::numbered(&classes);
    Ok((indices(py, firsts), indices(py, numbers), indices(py, counts)))
}

/// The position in `x` of the first item that matches each item of `y`, or `len(x)` where none does.
#[pyfunction]
pub fn index_of<'py>(py: Python<'py>, x: &Bound<'py, PyUntypedArray>, y: &Bound<'py, PyUntypedArray>, ct: f64) -> PyResult<Bound<'py, PyArray1<isize>>> {
    items!(x, T => keys: {
        let (x, y) = (x.extract::<PyReadonlyArray1<T>>()?, y.extract::<PyReadonlyArray1<T>>()?);
        Ok(indices(py, search::index_of(x.as_slice()?, y.as_slice()?).into_iter().map(|p| p as usize)))
    }, floats: {
        let (x, y) = (x.extract::<PyReadonlyArray1<T>>()?, y.extract::<PyReadonlyArray1<T>>()?);
        let found = with_tolerance!(tolerance(ct), |t| tolerant::index_of_with(t, x.as_slice()?, y.as_slice()?, || py.check_signals()))?;
        Ok(indices(py, found.into_iter().map(|p| p as usize)))
    })
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(unique, m)?, wrap_pyfunction!(index_of, m)?] { m.add_function(f)?; }
    Ok(())
}
