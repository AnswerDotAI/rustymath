//! rustymath's searches on contiguous vectors, which the wrappers in `python/rustymath/search.py` give each argument. Booleans and
//! integers match exactly, through `search`. float16, float32 and float64 match within tolerance `ct`, through `tolerant`. Each float
//! search checks for Ctrl-C as it goes, stopping with `KeyboardInterrupt`.
use half::f16;
use numpy::{PyArray1, PyArrayDescrMethods, PyReadonlyArray1, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rustymath::{search::{self, OrderKey}, tolerant, with_tolerance, Float};

use crate::tolerance;

/// Runs `$keys` with `$t` the Rust type of `$a`'s Boolean or integer dtype. A float dtype runs `$floats` with `$t` the type of that
/// float, or `$other`, which takes no type.
macro_rules! items {
    (@match $a:expr, $t:ident, $keys:expr, $($floats:tt)+) => {{
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
            $($floats)+
        }
    }};
    ($a:expr, $t:ident => keys: $keys:expr, floats: $floats:expr) => {
        items!(@match $a, $t, $keys, (_, 2) => { type $t = f16; $floats } (_, 4) => { type $t = f32; $floats } _ => { type $t = f64; $floats })
    };
    ($a:expr, $t:ident => keys: $keys:expr, other: $other:expr) => { items!(@match $a, $t, $keys, _ => $other) };
}

/// `positions` as an array of NumPy's index type.
fn indices(py: Python<'_>, positions: impl IntoIterator<Item = usize>) -> Bound<'_, PyArray1<isize>> {
    PyArray1::from_iter(py, positions.into_iter().map(|i| i as isize))
}

/// The first position of each item's class, with the same matching policy for every classification operation.
fn classes(py: Python<'_>, x: &Bound<'_, PyUntypedArray>, ct: f64) -> PyResult<Vec<u32>> {
    items!(x, T => keys: Ok(search::classes(x.extract::<PyReadonlyArray1<T>>()?.as_slice()?)),
        floats: with_tolerance!(tolerance(ct), |t| tolerant::classes_with(t, x.extract::<PyReadonlyArray1<T>>()?.as_slice()?, || py.check_signals())))
}

/// The classes of `x`, numbered from 0 in order of first position, as `search::numbered` gives them: each class's first position, each
/// item's class number, and each class's size.
#[pyfunction]
pub fn unique<'py>(
    py: Python<'py>, x: &Bound<'py, PyUntypedArray>, ct: f64,
) -> PyResult<(Bound<'py, PyArray1<isize>>, Bound<'py, PyArray1<isize>>, Bound<'py, PyArray1<isize>>)> {
    let classes = classes(py, x, ct)?;
    let search::Numbered { firsts, numbers, counts } = search::numbered(&classes);
    Ok((indices(py, firsts), indices(py, numbers), indices(py, counts)))
}

/// Counts each class without constructing a result for every input item when a table suffices.
#[pyfunction]
pub fn class_counts<'py>(py: Python<'py>, x: &Bound<'py, PyUntypedArray>, ct: f64) -> PyResult<Bound<'py, PyArray1<isize>>> {
    let counts = items!(x, T => keys: search::class_counts(x.extract::<PyReadonlyArray1<T>>()?.as_slice()?),
        other: search::numbered(&classes(py, x, ct)?).counts);
    Ok(indices(py, counts))
}

/// Groups flat positions, with each class and its positions in order of first occurrence.
#[pyfunction]
pub fn group_indices<'py>(py: Python<'py>, x: &Bound<'py, PyUntypedArray>, ct: f64) -> PyResult<Vec<Bound<'py, PyArray1<isize>>>> {
    let fallback = || -> PyResult<search::Grouped> {
        let search::Numbered { firsts, numbers, .. } = search::numbered(&classes(py, x, ct)?);
        Ok(search::Grouped::new(firsts, &numbers))
    };
    let grouped = items!(x, T => keys: match search::grouped_keys(x.extract::<PyReadonlyArray1<T>>()?.as_slice()?) {
        Some(g) => g, None => fallback()?,
    }, other: fallback()?);
    Ok(grouped.each().map(|positions| indices(py, positions.iter().map(|&p| p as usize))).collect())
}

/// Unsigned keys in numeric order. Float keys equate signed zeros and NaNs, with NaNs after infinity. Each lexsort column keeps its
/// own dtype; normalizing its keys avoids promoting mixed integer columns through floats.
fn ordered(x: &Bound<'_, PyUntypedArray>) -> PyResult<Vec<u64>> {
    items!(x, T => keys: Ok(x.extract::<PyReadonlyArray1<T>>()?.as_slice()?.iter().map(|&v| v.order_key().into()).collect()),
        floats: Ok(x.extract::<PyReadonlyArray1<T>>()?.as_slice()?.iter().map(|&v| v.order_key() as u64).collect()))
}

/// Stable orders of last-axis lanes. `keys` are in NumPy's order, with the primary key last.
#[pyfunction]
pub fn sort_indices<'py>(py: Python<'py>, keys: Vec<Bound<'py, PyUntypedArray>>, n: usize) -> PyResult<Bound<'py, PyArray1<isize>>> {
    let keys = keys.iter().map(ordered).collect::<PyResult<Vec<_>>>()?;
    let mut result = Vec::new();
    if n != 0 {
        for start in (0..keys[0].len()).step_by(n) {
            let columns = keys.iter().rev().map(|k| k[start..start + n].to_vec()).collect();
            result.extend(search::sort_rows(n, columns).into_iter().map(|p| p as usize));
        }
    }
    Ok(indices(py, result))
}

/// Insertion positions in an ascending vector. Complementing and reversing the keys turns strict-less into at-or-below.
#[pyfunction]
pub fn searchsorted<'py>(py: Python<'py>, a: &Bound<'py, PyUntypedArray>, v: &Bound<'py, PyUntypedArray>, right: bool) -> PyResult<Bound<'py, PyArray1<isize>>> {
    let (mut a, mut v) = (ordered(a)?, ordered(v)?);
    let n = a.len();
    if !right {
        a.reverse();
        for k in a.iter_mut().chain(&mut v) { *k = !*k; }
    }
    let found = search::bins(&a, &v);
    Ok(indices(py, found.into_iter().map(|p| if right { p as usize } else { n - p as usize })))
}

/// Counts nonnegative integer keys into a dense zero-based table, padded to `minlength`.
#[pyfunction]
pub fn bincount<'py>(py: Python<'py>, x: &Bound<'py, PyUntypedArray>, minlength: usize) -> PyResult<Bound<'py, PyArray1<isize>>> {
    items!(x, T => keys: {
        use search::Key;
        let x = x.extract::<PyReadonlyArray1<T>>()?;
        let x = x.as_slice()?;
        if x.iter().any(|v| v.key() < 0) { return Err(pyo3::exceptions::PyValueError::new_err("bincount needs nonnegative integers fitting int64")); }
        let len = x.iter().map(|v| v.key() as usize + 1).max().unwrap_or(0).max(minlength);
        let mut counts = vec![0usize; len];
        if let Some((min, table)) = search::counts(x) { counts[min as usize..min as usize + table.len()].copy_from_slice(&table); }
        else { for v in x { counts[v.key() as usize] += 1; } }
        Ok(indices(py, counts))
    }, other: Err(pyo3::exceptions::PyTypeError::new_err("bincount needs integers")))
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
    for f in [wrap_pyfunction!(unique, m)?, wrap_pyfunction!(index_of, m)?, wrap_pyfunction!(class_counts, m)?, wrap_pyfunction!(group_indices, m)?,
        wrap_pyfunction!(sort_indices, m)?, wrap_pyfunction!(searchsorted, m)?, wrap_pyfunction!(bincount, m)?] { m.add_function(f)?; }
    Ok(())
}
