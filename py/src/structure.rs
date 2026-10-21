//! Python syntax for the shared structural plans and independent-value views.
use crate::{array::{Array, core, give}, convert::{self, Arg, released}, functions::{Axis, Axes}};
use pyo3::{exceptions::{PyIndexError, PyTypeError, PyValueError}, prelude::*, types::PyTuple};
use rustymath::{array::{Array as Core, ArrayView}, buffer::{Buffer, DType}, error::Error, layout::{Layout, element_count, generated_len, position}, structure::{self, Plan, Selector}};
use rustymath_buffer::py_err;

fn at(n: i64, size: usize) -> PyResult<usize> {
    position(n, size).ok_or_else(|| PyIndexError::new_err(format!("position {n} is outside an axis of length {size}")))
}
fn array_selector(layout: &Layout, axis: usize, obj: &Bound<'_, PyAny>) -> PyResult<(Selector, bool)> {
    let indices = convert::one(obj)?;
    let items = indices.row_major()?;
    rustymath_buffer::selection::array_selector(layout, axis, indices.layout(), items.view(), false)
}
fn selector(layout: &Layout, axis: usize, obj: &Bound<'_, PyAny>) -> PyResult<Selector> {
    rustymath_buffer::selection::selector(layout, axis, obj, || array_selector(layout, axis, obj)).map(|(s, _)| s)
}
pub fn selection(layout: &Layout, index: &Bound<'_, PyAny>) -> PyResult<Plan> {
    rustymath_buffer::selection::selection(layout, index,
        |p| {
            if let Ok(a) = p.cast::<Array>() { return Ok(structure::sole_name(a.borrow().core().layout()).map(str::to_owned)); }
            Ok(if p.hasattr("__rustymath_array__")? { rustymath_buffer::read_array(p)?.and_then(|(_, layout)| structure::sole_name(&layout).map(str::to_owned)) } else { None })
        },
        array_selector,
    ).map(|(plan, _)| plan)
}

fn planned(py: Python<'_>, arg: Arg, build: impl FnOnce(&Layout) -> Result<Plan, Error>) -> PyResult<Array> {
    let plan = build(arg.view().layout()).map_err(py_err)?;
    Ok(Array::new(arg.selected(py, plan)?))
}
/// The layout that `axis` refers to, and the axis's position: `x`'s own layout, or the flattened layout and its one axis when `axis` is `None`.
fn along(x: &Arg, axis: Option<Axis>) -> PyResult<(Layout, usize)> {
    match axis { Some(a) => { let l = x.layout(); let a = a.of(&l)?; Ok((l, a)) }, None => Ok((Layout::from([x.view().len()]), 0)) }
}

/// Shares compact items under a new shape; one -1 infers its length. Reshaping drops axis metadata.
pub fn reshaped(py: Python<'_>, source: &Core, lengths: &[isize]) -> PyResult<Array> {
    let mut sizes = lengths.iter().map(|&n| if n == -1 { Ok(1) } else { usize::try_from(n).map_err(|_| PyValueError::new_err("shape lengths must be nonnegative, or one -1")) }).collect::<PyResult<Vec<_>>>()?;
    let free: Vec<_> = lengths.iter().enumerate().filter_map(|(a, &n)| (n == -1).then_some(a)).collect();
    if free.len() > 1 { return Err(PyValueError::new_err("only one length can be -1")); }
    if let Some(&a) = free.first() {
        let known = element_count(&sizes).map_err(py_err)?;
        if known == 0 || source.len() % known != 0 { return Err(PyValueError::new_err("no length for -1 fits the items")); }
        sizes[a] = source.len() / known;
    }
    generated_len(&sizes).map_err(py_err)?;
    Ok(Array::new(released(py, || source.reshape(sizes.into())).map_err(py_err)?))
}

#[pyfunction]
pub fn reshape(py: Python<'_>, x: &Bound<'_, PyAny>, shape: Vec<isize>) -> PyResult<Array> { reshaped(py, &convert::one(x)?.into_core()?, &shape) }
#[pyfunction]
pub fn ravel(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Array> { let x = convert::one(x)?.into_core()?; reshaped(py, &x, &[x.len() as isize]) }

/// Reverses the axis order by default. `axes` gives the source axis for each result axis.
#[pyfunction]
#[pyo3(signature = (x, axes=None))]
pub fn transpose(py: Python<'_>, x: &Bound<'_, PyAny>, axes: Option<Vec<Axis>>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.view().layout();
    let rank = layout.shape().len();
    let axes = match axes { Some(a) => a.iter().map(|a| a.of(&layout)).collect::<PyResult<Vec<_>>>()?, None => (0..rank).rev().collect() };
    if axes.len() != rank { return Err(PyValueError::new_err("transpose needs one source for each axis")); }
    let mut destinations = vec![usize::MAX; rank];
    for (to, &from) in axes.iter().enumerate() {
        if destinations[from] != usize::MAX { return Err(PyValueError::new_err("transpose axes must not repeat")); }
        destinations[from] = to;
    }
    planned(py, x, |layout| structure::transpose(layout, &destinations))
}

#[pyfunction]
pub fn swapaxes(py: Python<'_>, x: &Bound<'_, PyAny>, axis1: Axis, axis2: Axis) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.view().layout();
    let mut axes: Vec<_> = (0..layout.shape().len()).collect();
    axes.swap(axis1.of(&layout)?, axis2.of(&layout)?);
    planned(py, x, |layout| structure::transpose(layout, &axes))
}

/// Inserts a length-one axis. Negative positions count from the resulting rank.
#[pyfunction]
pub fn expand_dims(py: Python<'_>, x: &Bound<'_, PyAny>, axis: i64) -> PyResult<Array> {
    let x = convert::one(x)?;
    let at = at(axis, x.view().layout().shape().len() + 1)?;
    planned(py, x, |layout| structure::select_axis(layout, at, Selector::NewAxis))
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn squeeze(py: Python<'_>, x: &Bound<'_, PyAny>, axis: Option<Axes>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.view().layout();
    let axes = match axis {
        None => layout.shape().iter().enumerate().filter_map(|(a, &n)| (n == 1).then_some(a)).collect(),
        Some(Axes::One(a)) => vec![a.of(&layout)?],
        Some(Axes::Many(axes)) => axes.iter().map(|a| a.of(&layout)).collect::<PyResult<Vec<_>>>()?,
    };
    if axes.iter().any(|&a| layout.shape()[a] != 1) { return Err(PyValueError::new_err("squeeze axes must have length one")); }
    let selectors = layout.shape().iter().enumerate().map(|(a, &n)| if axes.contains(&a) { Selector::At(0) } else { Selector::all(n) }).collect::<Vec<_>>();
    planned(py, x, |layout| structure::select(layout, selectors, false))
}

/// Gathers independent positions on one axis, or on the flattened input without `axis`.
#[pyfunction]
#[pyo3(signature = (x, indices, axis=None))]
pub fn take(py: Python<'_>, x: &Bound<'_, PyAny>, indices: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (layout, axis) = along(&x, axis)?;
    let selector = selector(&layout, axis, indices)?;
    planned(py, x, |_| structure::select_axis(&layout, axis, selector))
}

#[pyfunction]
#[pyo3(signature = (x, shift, axis=None))]
pub fn roll(py: Python<'_>, x: &Bound<'_, PyAny>, shift: i64, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let flattened = axis.is_none();
    let (layout, a) = along(&x, axis)?;
    // Python's positive shift moves items right; the shared rotation starts at its count.
    let n = layout.shape()[a];
    let left = if n == 0 { 0 } else { (-(shift as i128)).rem_euclid(n as i128) as i64 };
    planned(py, x, |original| {
        let mut plan = structure::rotate(&layout, a, Some((&[], &[left])))?;
        if flattened { plan.layout = original.clone(); }
        Ok(plan)
    })
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn flip(py: Python<'_>, x: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.view().layout();
    let selected = axis.map(|a| a.of(&layout)).transpose()?;
    let selectors: Vec<_> = layout.shape().iter().enumerate().map(|(a, &n)| if selected.is_none_or(|s| s == a) {
        Selector::Range { start: n.saturating_sub(1), len: n, step: -1 }
    } else { Selector::all(n) }).collect();
    planned(py, x, |layout| structure::select(layout, selectors, false))
}

/// Joins arrays along `axis`, the first by default. `axis=None` joins the flattened arrays.
#[pyfunction]
#[pyo3(signature = (arrays, axis=crate::search::FIRST_AXIS), text_signature = "(arrays, axis=0)")]
pub fn concatenate(py: Python<'_>, arrays: Vec<Bound<'_, PyAny>>, axis: Option<Axis>) -> PyResult<Array> {
    let args: Vec<_> = arrays.iter().collect();
    let call = convert::convert(&args)?;
    let first = call.args.first().ok_or_else(|| PyValueError::new_err("concatenate needs at least one array"))?;
    let at = axis.map(|a| a.of(&first.layout())).transpose()?;
    let layouts: Vec<_> = call.args.iter().map(|a| if at.is_some() { a.layout() } else { Layout::from([a.view().len()]) }).collect();
    let pairs: Vec<_> = layouts.iter().zip(&call.args).map(|(l, a)| ArrayView::new(l, a.view().storage(), a.view().mapping())).collect();
    Ok(Array::new(released(py, || Core::concatenate(&pairs, at.unwrap_or(0))).map_err(py_err)?))
}

#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn stack(py: Python<'_>, arrays: Vec<Bound<'_, PyAny>>, axis: i64) -> PyResult<Array> {
    let args: Vec<_> = arrays.iter().collect();
    let call = convert::convert(&args)?;
    let first = call.args.first().ok_or_else(|| PyValueError::new_err("stack needs at least one array"))?;
    if call.args.iter().any(|a| a.layout().shape() != first.layout().shape()) { return Err(PyValueError::new_err("stack arrays must have one shape")); }
    let at = at(axis, first.layout().shape().len() + 1)?;
    let layouts = call.args.iter().map(|a| structure::select_axis(&a.layout(), at, Selector::NewAxis).map(|p| p.layout).map_err(py_err)).collect::<PyResult<Vec<_>>>()?;
    let pairs: Vec<_> = layouts.iter().zip(&call.args).map(|(l, a)| ArrayView::new(l, a.view().storage(), a.view().mapping())).collect();
    Ok(Array::new(released(py, || Core::concatenate(&pairs, at)).map_err(py_err)?))
}

/// Splits along `axis`, the first by default, into equal sections or at the supplied boundaries. `axis=None` splits the flattened input.
#[pyfunction]
#[pyo3(signature = (x, sections, axis=crate::search::FIRST_AXIS), text_signature = "(x, sections, axis=0)")]
pub fn split(py: Python<'_>, x: &Bound<'_, PyAny>, sections: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Vec<Array>> {
    let x = convert::one(x)?;
    let (layout, a) = along(&x, axis)?;
    let size = layout.shape()[a];
    let points = if let Ok(n) = sections.extract::<usize>() {
        if n == 0 || size % n != 0 { return Err(PyValueError::new_err("sections must divide the axis length")); }
        (1..n).map(|j| j * (size / n)).collect::<Vec<_>>()
    } else {
        sections.extract::<Vec<i64>>()?.into_iter().map(|n| if n < 0 { (size as i128 + n as i128).clamp(0, size as i128) as usize } else { (n as usize).min(size) }).collect()
    };
    let mut start = 0;
    points.into_iter().chain([size]).map(|end| {
        let plan = structure::select_axis(&layout, a, Selector::Range { start, len: end.saturating_sub(start), step: 1 }).map_err(py_err)?;
        start = end;
        Ok(Array::new(x.selected(py, plan)?))
    }).collect()
}

/// `x`'s cells along `axis`, or its items with no axis, each repeated by its count: one count for every cell, or one for each (`structure::repeat`). A negative count puts that many fills in its cell's place.
#[pyfunction]
#[pyo3(signature = (x, repeats, axis=None))]
pub fn repeat(py: Python<'_>, x: &Bound<'_, PyAny>, repeats: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (layout, a) = along(&x, axis)?;
    let counts = convert::one(repeats)?;
    if counts.layout().shape().len() > 1 { return Err(PyValueError::new_err("repeat counts must be a scalar or vector")); }
    let counts = counts.items::<i64>()?;
    planned(py, x, |_| structure::repeat(&layout, a, &counts))
}

/// Constant padding with the dtype's fill: zero, false or a space.
#[pyfunction]
pub fn pad(py: Python<'_>, x: &Bound<'_, PyAny>, pad_width: &Bound<'_, PyAny>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let rank = x.view().layout().shape().len();
    let widths = if let Ok(n) = pad_width.extract::<usize>() { vec![(n, n); rank] }
    else if let Ok(pair) = pad_width.extract::<(usize, usize)>() { vec![pair; rank] }
    else { pad_width.extract::<Vec<(usize, usize)>>()? };
    planned(py, x, |layout| structure::pad(layout, &widths))
}

/// The coordinates of `x`'s nonzero items, one row of axis positions for each item, and the number of rows.
fn nonzero_rows(py: Python<'_>, x: &Arg) -> PyResult<(usize, Vec<i64>)> {
    let (items, layout) = (x.row_major()?, x.layout());
    Ok(released(py, || {
        let positions = Buffer::nonzero(items.view());
        let rank = layout.shape().len();
        let mut rows = vec![0; positions.len() * rank];
        for (row, &i) in positions.iter().enumerate() {
            for (axis, coordinate) in rustymath::layout::coordinates(i, layout.shape()) { rows[row * rank + axis] = coordinate as i64; }
        }
        (positions.len(), rows)
    }))
}

/// Coordinates of non-fill items, as one index vector per axis. Fill is zero, false or a space for the item's dtype.
#[pyfunction]
pub fn nonzero<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let x = convert::one(x)?;
    let rank = x.layout().shape().len();
    let (n, rows) = nonzero_rows(py, &x)?;
    let axes = (0..rank).map(|a| core(rows.iter().skip(a).step_by(rank).copied().collect(), &[n]).map(Array::new)).collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, axes)
}

/// Coordinates of non-fill items, with one row per item and one column per axis.
#[pyfunction]
pub fn argwhere(py: Python<'_>, x: &Bound<'_, PyAny>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (n, rows) = nonzero_rows(py, &x)?;
    Ok(Array::new(core(rows, &[n, x.layout().shape().len()])?))
}

/// Choose between x and y with a Boolean condition, broadcasting all three by leading axes.
#[pyfunction(name = "where")]
pub fn where_<'py>(py: Python<'py>, condition: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let condition = convert::one(condition)?;
    if condition.view().dtype() != DType::Boolean { return Err(PyTypeError::new_err("where condition must be Boolean")); }
    let call = convert::convert(&[x, y])?;
    let [c, x, y] = [condition.view(), call.args[0].view(), call.args[1].view()];
    let result = released(py, || Core::choose(c, x, y)).map_err(py_err)?;
    give(py, result, condition.scalar && call.scalar())
}

/// A diagonal moved to the last result axis. Its axes default to the last two, the axes that the matrix functions take.
#[pyfunction]
#[pyo3(signature = (x, offset=0, axis1=Axis::Position(-2), axis2=Axis::Position(-1)))]
pub fn diagonal(py: Python<'_>, x: &Bound<'_, PyAny>, offset: i64, axis1: Axis, axis2: Axis) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.view().layout();
    let axes = [axis1.of(layout)?, axis2.of(layout)?];
    planned(py, x, |layout| structure::diagonal(layout, axes, offset))
}

/// Diagonal construction or extraction; `shape` allows a scalar to fill a rectangular matrix's diagonal.
#[pyfunction]
#[pyo3(signature = (x, k=0, shape=None, dtype=None))]
fn _diag(py: Python<'_>, x: &Bound<'_, PyAny>, k: i64, shape: Option<[usize; 2]>, dtype: Option<&str>) -> PyResult<Array> {
    let x = match dtype { Some(d) => convert::one_at(x, convert::dtype(d)?)?, None => convert::one(x)? };
    let shape = match (shape, x.layout().shape()) {
        (Some(shape), _) => shape,
        (None, &[len]) => {
            let n = len.checked_add(usize::try_from(k.unsigned_abs()).unwrap_or(usize::MAX)).ok_or_else(|| PyValueError::new_err("diagonal shape is too large"))?;
            [n, n]
        }
        (None, [_, _]) => return planned(py, x, |layout| structure::diagonal(layout, [0, 1], k)),
        _ => return Err(PyValueError::new_err("diag takes a vector or a matrix")),
    };
    let items = x.row_major()?;
    Ok(Array::new(released(py, || rustymath::create::diagonal(items.view(), shape, k)).map_err(py_err)?))
}

#[pyfunction]
fn _triangle(py: Python<'_>, x: &Bound<'_, PyAny>, k: i64, upper: bool) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (layout, items) = (x.layout(), x.row_major()?);
    Ok(Array::new(released(py, || rustymath::create::triangle(layout, items.view(), k, upper)).map_err(py_err)?))
}

/// A compact array of `shape`, broadcasting from the front rather than NumPy's trailing-axis rule.
#[pyfunction]
pub fn broadcast_to(py: Python<'_>, x: &Bound<'_, PyAny>, shape: &Bound<'_, PyAny>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (view, shape) = (x.view(), crate::create::shape(shape)?);
    Ok(Array::new(released(py, || Core::broadcasted(view, &shape)).map_err(py_err)?))
}

#[pyfunction]
#[pyo3(signature = (x, axis=Axis::Position(-1)))]
fn _difference(py: Python<'_>, x: &Bound<'_, PyAny>, axis: Axis) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (view, axis) = (x.view(), axis.of(&x.layout())?);
    Ok(Array::new(released(py, || Core::differences(view, axis)).map_err(py_err)?))
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(reshape, m)?, wrap_pyfunction!(ravel, m)?, wrap_pyfunction!(transpose, m)?, wrap_pyfunction!(swapaxes, m)?,
        wrap_pyfunction!(expand_dims, m)?, wrap_pyfunction!(squeeze, m)?, wrap_pyfunction!(take, m)?, wrap_pyfunction!(roll, m)?, wrap_pyfunction!(flip, m)?,
        wrap_pyfunction!(concatenate, m)?, wrap_pyfunction!(stack, m)?, wrap_pyfunction!(split, m)?, wrap_pyfunction!(repeat, m)?, wrap_pyfunction!(pad, m)?,
        wrap_pyfunction!(nonzero, m)?, wrap_pyfunction!(argwhere, m)?, wrap_pyfunction!(where_, m)?, wrap_pyfunction!(diagonal, m)?,
        wrap_pyfunction!(_diag, m)?, wrap_pyfunction!(_triangle, m)?, wrap_pyfunction!(broadcast_to, m)?, wrap_pyfunction!(_difference, m)?] { m.add_function(f)?; }
    m.add("flatten", m.getattr("ravel")?)
}
