//! rustymath's searches, sorts and classifications, on arguments that `convert` reads. Booleans, integers and characters match exactly. Floats match within the tolerance `ct`, or the process's tolerance (`set_tolerance`) where `ct` is `None`, which is 0, exact, unless set. Tolerant classification and item matching check for Ctrl-C, stopping with `KeyboardInterrupt`. Sorts and insertion searches compare exactly: signed zeros compare equal, and every NaN comes after positive infinity. Complex numbers raise. Positions come back as `i64` arrays.
use crate::array::{core, give, give_items};
use crate::convert::{Arg, Call, convert, one, released, signals};
use crate::functions::{Axis, along_axis, missing, tolerance};
use pyo3::prelude::*;
use rustymath::array::Array as Core;
use rustymath::buffer::{Buffer, Element, View};
use rustymath::error::{Error, Kind};
use rustymath::layout::{Layout, generated_len};
use rustymath::search::{self, Key, Numbered};
use rustymath_buffer::py_err;

pub const LAST_AXIS: Option<Axis> = Some(Axis::Position(-1));
pub const FIRST_AXIS: Option<Axis> = Some(Axis::Position(0));

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
    let items = x.row_major()?;
    let (view, t) = (items.view(), tolerance(ct)?);
    let classes = released(py, || search::view_classes(view, t, signals)).ok_or_else(|| missing("unique", view.dtype()))??;
    let Numbered { firsts, numbers, counts } = search::numbered(&classes);
    let firsts: Vec<i64> = firsts.into_iter().map(|p| p as i64).collect();
    let n = firsts.len();
    let values = Core::new(Layout::from([n].as_slice()), Buffer::gathered(view, &firsts)).map_err(py_err)?;
    Ok([give(py, values, false)?, give(py, core(firsts, &[n])?, false)?, give(py, vector(numbers)?, false)?, give(py, vector(counts)?, false)?])
}

/// `x` and `y` converted, and for each item of `y`, the first position in flattened `x` of an item that matches it within the tolerance `ct`, or the size of `x` where none does.
fn first_matches<'py>(py: Python<'py>, function: &str, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<(Call, Vec<u32>)> {
    let call = convert(&[x, y])?;
    let found = {
        let (xs, ys, t) = (call.args[0].at(call.args[0].view().dtype())?, call.args[1].at(call.args[1].view().dtype())?, tolerance(ct)?);
        released(py, || search::view_index_of(xs.view(), ys.view(), t, signals)).ok_or_else(|| missing(function, xs.view().dtype()))??
    };
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
    give_items(py, ys.layout(), bool::buffer(found.into_iter().map(|p| (p as usize) < n).collect()), ys.scalar)
}

pub fn contains(py: Python<'_>, a: &Bound<'_, PyAny>, query: &Bound<'_, PyAny>) -> PyResult<bool> {
    let (x, y) = (one(a)?, one(query)?);
    let wanted = if query.is_instance_of::<pyo3::types::PyString>() && query.len()? == 1 { Layout::EMPTY } else { y.layout() };
    let (xs, ys) = (x.row_major()?, y.row_major()?);
    let (lx, t) = (x.layout(), tolerance(None)?);
    Ok(released(py, || search::view_member(&lx, xs.view(), &wanted, ys.view(), t, signals))
        .map_err(|e| if e.kind == Kind::Rank { pyo3::exceptions::PyTypeError::new_err(e.message.to_string()) } else { py_err(e) })??)
}

/// A Boolean mask shaped like `target`, marking starts where `pattern` fits and matches. A lower-rank pattern has leading singleton axes; a higher-rank pattern never matches. Floats use `ct`.
#[pyfunction]
#[pyo3(signature = (pattern, target, *, ct=None))]
pub fn find<'py>(py: Python<'py>, pattern: &Bound<'py, PyAny>, target: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[pattern, target])?;
    let (pattern, target) = (&call.args[0], &call.args[1]);
    let (xs, ys) = (pattern.row_major()?, target.row_major()?);
    let (lp, lt, t) = (pattern.layout(), target.layout(), tolerance(ct)?);
    let found = released(py, || search::view_find(xs.view(), lp.shape(), ys.view(), lt.shape(), t)).ok_or_else(|| missing("find", xs.view().dtype()))?;
    give_items(py, lt, bool::buffer(found), target.scalar)
}

#[pyfunction]
#[pyo3(signature = (x, y, how, ct=None))]
fn _set<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, how: &str, ct: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
    let how = match how { "union" => search::Set::Union, "intersection" => search::Set::Intersection, "difference" => search::Set::Difference,
        _ => return Err(py_err(Error::new(Kind::Domain, "unknown set operation"))), };
    let call = convert(&[x, y])?;
    let (xs, ys) = (call.args[0].at(call.args[0].view().dtype())?, call.args[1].at(call.args[1].view().dtype())?);
    let (x, y, t) = (xs.view(), ys.view(), tolerance(ct)?);
    generated_len(&[x.len().saturating_add(y.len())]).map_err(py_err)?;
    let values = released(py, || search::view_set(x, y, how, t, signals)).ok_or_else(|| missing("set operation", x.dtype()))??;
    let n = values.len();
    give_items(py, [n].into(), values, false)
}

/// The size of each class of flattened `x`, in order of first position.
#[pyfunction]
#[pyo3(signature = (x, *, ct=None))]
pub fn counts<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let items = x.row_major()?;
    let (view, t) = (items.view(), tolerance(ct)?);
    let counts = released(py, || search::view_class_counts(view, t, signals)).ok_or_else(|| missing("counts", view.dtype()))??;
    give(py, vector(counts)?, false)
}

/// The flat positions of each class of `x`, one vector for each class in order of first position, each in ascending order.
#[pyfunction]
#[pyo3(signature = (x, *, ct=None))]
pub fn group_indices<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, ct: Option<f64>) -> PyResult<Vec<Bound<'py, PyAny>>> {
    let x = one(x)?;
    let items = x.row_major()?;
    let (view, t) = (items.view(), tolerance(ct)?);
    let grouped = released(py, || search::view_grouped(view, t, signals)).ok_or_else(|| missing("group_indices", view.dtype()))??;
    grouped.groups.each().map(|these| give(py, vector(these.iter().map(|&p| p as usize))?, false)).collect()
}

/// Stable positions that sort equally shaped `keys` along `axis`, with the last key primary, or along the flattened keys with no axis. The sorted axis keeps its name and drops its keys. A rank-0 key sorts as one item, as in NumPy.
fn sorted<'py>(py: Python<'py>, function: &str, keys: &[Arg], axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    let layout = keys.first().map(Arg::layout).filter(|l| keys.iter().all(|k| k.layout().shape() == l.shape())).ok_or_else(|| py_err(Error::new(Kind::Length, format!("{function} needs one or more equally shaped keys"))))?;
    let keys = keys.iter().map(|k| k.row_major()).collect::<PyResult<Vec<_>>>()?;
    for k in &keys { if matches!(k.view(), View::Complexes(_)) { return Err(missing(function, k.view().dtype())); } }
    let total = keys[0].view().len();
    let layout = if layout.shape().is_empty() { Layout::from([total].as_slice()) } else { layout };
    let (dims, mut layout, position) = along_axis(layout, total, axis)?;
    if let Some(a) = position { layout = layout.without_keys(a); }
    let columns: Vec<_> = keys.iter().rev().map(|k| k.view()).collect();
    let order = released(py, || search::view_lexgrade_along(&columns, dims, false)).ok_or_else(|| missing(function, keys[0].view().dtype()))?;
    give_items(py, layout, i64::buffer(order), false)
}

/// Stable positions that sort `a` along `axis`, the last axis by default, as in NumPy. `axis=None` sorts along the flattened `a`.
#[pyfunction]
#[pyo3(signature = (a, axis=LAST_AXIS), text_signature = "(a, axis=-1)")]
pub fn argsort<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    sorted(py, "argsort", &[one(a)?], axis)
}

#[derive(Clone, Copy)]
enum Ordering { Sort(bool), Partition(i64, bool), Extreme(bool) }

/// Runs one vector operation along the selected axis. Only the Python axis and rank arguments are decoded here.
fn ordered<'py>(py: Python<'py>, name: &str, a: &Bound<'py, PyAny>, axis: Option<Axis>, how: Ordering) -> PyResult<Bound<'py, PyAny>> {
    let x = one(a)?;
    let items = x.row_major()?;
    let view = items.view();
    if matches!(view, View::Complexes(_)) { return Err(missing(name, view.dtype())); }
    let layout = if x.layout().shape().is_empty() { Layout::from([1]) } else { x.layout() };
    let (dims @ [outer, n, inner], mut layout, position) = along_axis(layout, view.len(), axis)?;
    let rank = if let Ordering::Partition(k, _) = how {
        rustymath::layout::position(k, n).ok_or_else(|| py_err(Error::new(Kind::Index, "partition rank is outside the axis")))?
    } else { 0 };
    if matches!(how, Ordering::Extreme(_)) && n == 0 { return Err(py_err(Error::new(Kind::Length, "an empty axis has no minimum or maximum"))); }
    if let Some(a) = position.filter(|&a| layout.keys(a).is_some()) {
        let order = if outer * inner == 1 { released(py, || match how {
            Ordering::Sort(down) => search::view_grade(view, down),
            Ordering::Partition(_, false) => search::view_argpartition(view, rank), _ => None,
        })} else { None };
        if let Some(order) = order {
            let layout = layout.select(a, order.iter().map(|&p| Some(p as usize))).map_err(py_err)?;
            return give_items(py, layout, Buffer::gathered(view, &order), false);
        }
        layout = layout.without_keys(a);
    }
    let result = released(py, || match how {
        Ordering::Sort(down) => search::view_sort_along(view, dims, down).unwrap(),
        Ordering::Partition(_, indices) => search::view_partition_along(view, dims, rank, indices).unwrap(),
        Ordering::Extreme(down) => i64::buffer(search::view_argextremes(view, dims, down).unwrap()),
    });
    let layout = if matches!(how, Ordering::Extreme(_)) { position.map_or(Layout::EMPTY, |a| layout.axes((0..layout.shape().len()).filter(move |&i| i != a))) } else { layout };
    give_items(py, layout, result, matches!(how, Ordering::Extreme(_)) && position.is_none())
}

macro_rules! order_functions {
    ($($(#[doc = $doc:literal])* $name:ident($($arg:ident: $t:ty),*) = $how:expr, $axis:expr;)+) => {$(
        $(#[doc = $doc])*
        #[pyfunction]
        #[pyo3(signature = (a, $($arg,)* axis=$axis))]
        pub fn $name<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, $($arg: $t,)* axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
            ordered(py, stringify!($name), a, axis, $how)
        }
    )+};
}

/// Values sorted along `axis`, the last axis by default. `axis=None` sorts the flattened input. Equal values keep their order; NaNs sort last ascending and first descending.
#[pyfunction]
#[pyo3(signature = (a, axis=LAST_AXIS, *, descending=false))]
pub fn sort<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, axis: Option<Axis>, descending: bool) -> PyResult<Bound<'py, PyAny>> {
    ordered(py, "sort", a, axis, Ordering::Sort(descending))
}

order_functions! {
    /// Values partitioned around `kth`, with neither side sorted. Negative ranks count from the end.
    partition(kth: i64) = Ordering::Partition(kth, false), LAST_AXIS;
    /// Positions partitioned around `kth`, with neither side sorted. Equal values need not keep their original order.
    argpartition(kth: i64) = Ordering::Partition(kth, true), LAST_AXIS;
    /// The first position of the minimum, along `axis` or in the flattened array. Uses sorting's exact NaN-last ordering.
    argmin() = Ordering::Extreme(false), None;
    /// The first position of the maximum, along `axis` or in the flattened array. Uses sorting's exact NaN-last ordering.
    argmax() = Ordering::Extreme(true), None;
}

/// Quantiles along `axis` or in flattened `a`, by selecting the required ranks and interpolating linearly. A vector of probabilities adds a leading axis. NaN counts as the largest value, as in sorting.
#[pyfunction]
#[pyo3(signature = (a, q, axis=None))]
pub fn quantile<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, q: &Bound<'py, PyAny>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    let (x, q) = (one(a)?, one(q)?);
    if q.layout().shape().len() > 1 { return Err(py_err(Error::new(Kind::Rank, "quantile probabilities must be a scalar or vector"))); }
    let probabilities = q.items::<f64>()?;
    let layout = x.layout();
    let (dims, _, position) = along_axis(layout.clone(), x.view().len(), axis)?;
    let input = x.row_major()?;
    let items = released(py, || search::view_quantiles_along(input.view(), dims, &probabilities)).map_err(py_err)?.ok_or_else(|| missing("quantile", x.view().dtype()))?;
    let frame = position.map_or(Layout::EMPTY, |a| layout.axes((0..layout.shape().len()).filter(move |&i| i != a)));
    let layout = if q.layout().shape().is_empty() { frame } else { q.layout().concat(&frame) };
    let out = Core::new(layout, items).map_err(py_err)?;
    give(py, out, position.is_none() && q.layout().shape().is_empty())
}

/// The middle value, or the mean of the two middle values, without fully sorting the input. NaN counts as the largest value, as in sorting.
#[pyfunction]
#[pyo3(signature = (a, axis=None))]
pub fn median<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    use pyo3::IntoPyObject;
    quantile(py, a, &0.5f64.into_pyobject(py)?.into_any(), axis)
}

/// Stable positions that sort equally shaped `keys` along `axis`, the last axis by default, with the last key primary, as in NumPy. `axis=None` sorts along the flattened keys. Each key keeps its own dtype.
#[pyfunction]
#[pyo3(signature = (keys, axis=LAST_AXIS), text_signature = "(keys, axis=-1)")]
pub fn lexsort<'py>(py: Python<'py>, keys: Vec<Bound<'py, PyAny>>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    sorted(py, "lexsort", &keys.iter().map(one).collect::<PyResult<Vec<_>>>()?, axis)
}

/// Where each item of `v` goes in the ascending vector `a`, comparing exactly, shaped like `v`: before items equal to it for `side='left'`, and after them for `side='right'`.
#[pyfunction]
#[pyo3(signature = (a, v, side="left"))]
pub fn searchsorted<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>, side: &str) -> PyResult<Bound<'py, PyAny>> {
    let right = match side { "left" => false, "right" => true, _ => return Err(py_err(Error::new(Kind::Domain, "side must be 'left' or 'right'"))) };
    let call = convert(&[a, v])?;
    let (x, y) = (call.args[0].at(call.args[0].view().dtype())?, call.args[1].at(call.args[1].view().dtype())?);
    let (xs, ys) = (x.view(), y.view());
    if call.args[0].layout().shape().len() != 1 { return Err(py_err(Error::new(Kind::Rank, "searchsorted needs a one-dimensional sorted array"))); }
    let found = released(py, || if right { search::view_bins(xs, ys) } else { search::view_bins_below(xs, ys) }).ok_or_else(|| missing("searchsorted", xs.dtype()))?;
    give(py, positions(call.args[1].layout(), found.into_iter().map(|p| p as usize))?, call.args[1].scalar)
}

/// The number of times each key in `x` appears, indexed from zero and padded to `minlength`. A Domain error for a negative key.
fn counted<K: Key>(x: &[K], minlength: usize) -> PyResult<Vec<usize>> {
    let range = (!x.is_empty()).then(|| K::range(x));
    if range.is_some_and(|(min, _)| min < 0) { return Err(py_err(Error::new(Kind::Domain, "bincount needs nonnegative integers"))); }
    let len = range.map_or(0, |(_, max)| max as usize + 1).max(minlength);
    let mut counts = vec![0; generated_len(&[len]).map_err(py_err)?];
    if let Some((min, max)) = range { search::counts_into(x, min, &mut counts[min as usize..=max as usize]); }
    Ok(counts)
}

/// How many items of `x`, every item whatever its rank, equal each integer from zero, padded to `minlength`. Booleans count as 0 and 1, and a negative item is a Domain error.
#[pyfunction]
#[pyo3(signature = (x, minlength=0))]
pub fn bincount<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, minlength: i64) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let minlength = usize::try_from(minlength).map_err(|_| py_err(Error::new(Kind::Domain, "minlength must be nonnegative")))?;
    let input = x.row_major()?;
    let counts = released(py, || match input.view() {
        View::Integers(ints) => rustymath::with_ints!(ints, |x| counted(x, minlength)),
        View::Booleans(x) => counted(x, minlength),
        view => Err(missing("bincount", view.dtype())),
    })?;
    give(py, vector(counts)?, false)
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(unique, m)?, wrap_pyfunction!(index, m)?, wrap_pyfunction!(isin, m)?, wrap_pyfunction!(find, m)?,
        wrap_pyfunction!(counts, m)?, wrap_pyfunction!(group_indices, m)?,
        wrap_pyfunction!(argsort, m)?, wrap_pyfunction!(sort, m)?, wrap_pyfunction!(partition, m)?, wrap_pyfunction!(argpartition, m)?,
        wrap_pyfunction!(argmin, m)?, wrap_pyfunction!(argmax, m)?, wrap_pyfunction!(median, m)?, wrap_pyfunction!(quantile, m)?,
        wrap_pyfunction!(lexsort, m)?, wrap_pyfunction!(searchsorted, m)?, wrap_pyfunction!(bincount, m)?, wrap_pyfunction!(_set, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}
