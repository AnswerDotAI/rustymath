//! rustymath's searches, sorts and classifications, on arguments that `convert` reads. Booleans, integers and characters match exactly. Floats match within the tolerance `ct`, and each search of floats checks for Ctrl-C as it goes, stopping with `KeyboardInterrupt`. Sorts and insertion searches compare exactly: signed zeros compare equal, and every NaN comes after positive infinity. Complex numbers raise. Positions come back as `i64` arrays.
use crate::array::{core, give};
use crate::convert::{Arg, Call, convert, one};
use crate::functions::{Axis, along_axis, missing, tolerance};
use pyo3::prelude::*;
use rustymath::array::Array as Core;
use rustymath::buffer::{Buffer, Element, View};
use rustymath::error::{Error, Kind};
use rustymath::layout::{Layout, generated_len};
use rustymath::search::{self, Key, Numbered};
use rustymath_buffer::py_err;

/// `positions` as an `i64` array of `layout`.
fn positions(layout: Layout, positions: impl IntoIterator<Item = usize>) -> PyResult<Core> {
    Core::new(layout, i64::buffer(positions.into_iter().map(|p| p as i64).collect())).map_err(py_err)
}

/// `positions` as an `i64` vector.
fn vector(positions: impl IntoIterator<Item = usize>) -> PyResult<Core> {
    let items: Vec<i64> = positions.into_iter().map(|p| p as i64).collect();
    let n = items.len();
    core(items, &[n])
}

/// The classes of flattened `ar` in order of first position: their values, each one's first position, each item's class number, and each class's size.
#[pyfunction]
#[pyo3(signature = (ar, *, ct=None))]
pub fn unique<'py>(py: Python<'py>, ar: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<[Bound<'py, PyAny>; 4]> {
    let x = one(ar)?;
    let view = x.view();
    let classes = search::view_classes(view, tolerance(ct)?, || py.check_signals()).ok_or_else(|| missing("unique", view.dtype()))??;
    let Numbered { firsts, numbers, counts } = search::numbered(&classes);
    let firsts: Vec<i64> = firsts.into_iter().map(|p| p as i64).collect();
    let n = firsts.len();
    let values = Core::new(Layout::from([n].as_slice()), Buffer::gathered(view, &firsts)).map_err(py_err)?;
    Ok([give(py, values, false)?, give(py, core(firsts, &[n])?, false)?, give(py, vector(numbers)?, false)?, give(py, vector(counts)?, false)?])
}

/// `x` and `y` converted, and for each item of `y`, the first position in flattened `x` of an item that matches it within the tolerance `ct`, or the size of `x` where none does.
fn first_matches<'py>(py: Python<'py>, function: &str, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<(Call<'py>, Vec<u32>)> {
    let call = convert(&[x, y])?;
    let (xs, ys) = (call.args[0].view(), call.args[1].view());
    let found = search::view_index_of(xs, ys, tolerance(ct)?, || py.check_signals()).ok_or_else(|| missing(function, xs.dtype()))??;
    Ok((call, found))
}

/// Where in flattened `x` each item of `y` first matches, or the size of `x` where none does, shaped like `y`.
#[pyfunction]
#[pyo3(signature = (x, y, *, ct=None))]
pub fn index<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
    let (call, found) = first_matches(py, "index", x, y, ct)?;
    let ys = &call.args[1];
    give(py, positions(ys.layout(), found.into_iter().map(|p| p as usize))?, ys.scalar)
}

/// Whether each item of `element` matches an item of flattened `test_elements`, shaped like `element`.
#[pyfunction]
#[pyo3(signature = (element, test_elements, *, ct=None))]
pub fn isin<'py>(py: Python<'py>, element: &Bound<'py, PyAny>, test_elements: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
    let (call, found) = first_matches(py, "isin", test_elements, element, ct)?;
    let (n, ys) = (call.args[0].view().len(), &call.args[1]);
    give(py, Core::new(ys.layout(), bool::buffer(found.into_iter().map(|p| (p as usize) < n).collect())).map_err(py_err)?, ys.scalar)
}

/// The size of each class of flattened `x`, in order of first position.
#[pyfunction]
#[pyo3(signature = (x, *, ct=None))]
pub fn counts<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let view = x.view();
    let counts = search::view_class_counts(view, tolerance(ct)?, || py.check_signals()).ok_or_else(|| missing("counts", view.dtype()))??;
    give(py, vector(counts)?, false)
}

/// The flat positions of each class of `x`, one vector for each class in order of first position, each in ascending order.
#[pyfunction]
#[pyo3(signature = (x, *, ct=None))]
pub fn group_indices<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let x = one(x)?;
    let view = x.view();
    let grouped = search::view_grouped(view, tolerance(ct)?, || py.check_signals()).ok_or_else(|| missing("group_indices", view.dtype()))??;
    grouped.each().map(|these| give(py, vector(these.iter().map(|&p| p as usize))?, false)).collect()
}

/// Stable positions that sort equally shaped `keys` along `axis`, with the last key primary, or along the flattened keys with no axis. A rank-0 key sorts as one item, as in NumPy.
fn sorted<'py>(py: Python<'py>, function: &str, keys: &[Arg<'_>], axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    let layout = keys.first().map(Arg::layout).filter(|l| keys.iter().all(|k| k.layout().shape() == l.shape())).ok_or_else(|| py_err(Error::new(Kind::Length, format!("{function} needs one or more equally shaped keys"))))?;
    let words = keys.iter().rev().map(|k| search::view_order_keys(k.view()).ok_or_else(|| missing(function, k.view().dtype()))).collect::<PyResult<Vec<_>>>()?;
    let total = words[0].len();
    let layout = if layout.shape().is_empty() { Layout::from([total].as_slice()) } else { layout };
    let ([outer, n, inner], layout) = along_axis(layout, total, axis)?;
    let mut order = vec![0; total];
    if n > 0 {
        for o in 0..outer {
            for i in 0..inner {
                let at = |j: usize| (o * n + j) * inner + i;
                let columns = words.iter().map(|w| (0..n).map(|j| w[at(j)]).collect()).collect();
                for (j, p) in search::sort_rows(n, columns).into_iter().enumerate() { order[at(j)] = p as usize; }
            }
        }
    }
    give(py, positions(layout, order)?, false)
}

/// Stable positions that sort `a` along `axis`, the last axis by default, as in NumPy. `axis=None` sorts along the flattened `a`.
#[pyfunction]
#[pyo3(signature = (a, axis=Some(Axis::Position(-1))), text_signature = "(a, axis=-1)")]
pub fn argsort<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    sorted(py, "argsort", &[one(a)?], axis)
}

/// Stable positions that sort equally shaped `keys` along `axis`, the last axis by default, with the last key primary, as in NumPy. `axis=None` sorts along the flattened keys. Each key keeps its own dtype.
#[pyfunction]
#[pyo3(signature = (keys, axis=Some(Axis::Position(-1))), text_signature = "(keys, axis=-1)")]
pub fn lexsort<'py>(py: Python<'py>, keys: Vec<Bound<'py, PyAny>>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    sorted(py, "lexsort", &keys.iter().map(one).collect::<PyResult<Vec<_>>>()?, axis)
}

/// Where each item of `v` goes in the ascending vector `a`, comparing exactly, shaped like `v`: before items equal to it for `side='left'`, and after them for `side='right'`.
#[pyfunction]
#[pyo3(signature = (a, v, side="left"))]
pub fn searchsorted<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>, side: &str) -> PyResult<Bound<'py, PyAny>> {
    let right = match side { "left" => false, "right" => true, _ => return Err(py_err(Error::new(Kind::Domain, "side must be 'left' or 'right'"))) };
    let call = convert(&[a, v])?;
    let (xs, ys) = (call.args[0].view(), call.args[1].view());
    if call.args[0].layout().shape().len() != 1 { return Err(py_err(Error::new(Kind::Rank, "searchsorted needs a one-dimensional sorted array"))); }
    let found = if right { search::view_bins(xs, ys) } else { search::view_bins_below(xs, ys) }.ok_or_else(|| missing("searchsorted", xs.dtype()))?;
    give(py, positions(call.args[1].layout(), found.into_iter().map(|p| p as usize))?, call.args[1].scalar)
}

/// The number of times each key in `x` appears, indexed from zero and padded to `minlength`. A Domain error for a negative key.
fn counted<K: Key>(x: &[K], minlength: usize) -> PyResult<Vec<usize>> {
    if x.iter().any(|v| v.key() < 0) { return Err(py_err(Error::new(Kind::Domain, "bincount needs nonnegative integers"))); }
    let len = x.iter().map(|v| v.key() as usize + 1).max().unwrap_or(0).max(minlength);
    let mut counts = vec![0; generated_len(&[len]).map_err(py_err)?];
    match search::counts(x) {
        Some((min, table)) => counts[min as usize..min as usize + table.len()].copy_from_slice(&table),
        None => for v in x { counts[v.key() as usize] += 1; },
    }
    Ok(counts)
}

/// Counts a vector of nonnegative integers, indexed from zero, padded to `minlength`. Booleans count as 0 and 1.
#[pyfunction]
#[pyo3(signature = (x, minlength=0))]
pub fn bincount<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, minlength: i64) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    if x.layout().shape().len() != 1 { return Err(py_err(Error::new(Kind::Rank, "bincount needs a vector of integers"))); }
    let minlength = usize::try_from(minlength).map_err(|_| py_err(Error::new(Kind::Domain, "minlength must be nonnegative")))?;
    let counts = match x.view() {
        View::Integers(ints) => rustymath::with_ints!(ints, |x| counted(x, minlength))?,
        View::Booleans(x) => counted(x, minlength)?,
        view => return Err(missing("bincount", view.dtype())),
    };
    give(py, vector(counts)?, false)
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(unique, m)?, wrap_pyfunction!(index, m)?, wrap_pyfunction!(isin, m)?, wrap_pyfunction!(counts, m)?, wrap_pyfunction!(group_indices, m)?,
        wrap_pyfunction!(argsort, m)?, wrap_pyfunction!(lexsort, m)?, wrap_pyfunction!(searchsorted, m)?, wrap_pyfunction!(bincount, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}
