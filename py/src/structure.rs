//! Python syntax for the shared structural plans. Results always own compact items.
use crate::{array::{Array, core, give}, convert::{self, Arg}, functions::{Axis, Axes}};
use pyo3::{exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError}, prelude::*, types::{PyBool, PySlice, PyString, PyTuple}};
use rustymath::{array::Array as Core, buffer::{Buffer, View}, layout::{Layout, element_count, generated_len, position}, structure::{self, Plan, Selector}};
use rustymath_buffer::py_err;

fn at(n: i64, size: usize) -> PyResult<usize> {
    position(n, size).ok_or_else(|| PyIndexError::new_err(format!("position {n} is outside an axis of length {size}")))
}
fn key(layout: &Layout, axis: usize, name: &str) -> PyResult<usize> {
    layout.keys(axis).and_then(|k| k.position(name)).ok_or_else(|| PyKeyError::new_err(name.to_owned()))
}

/// One Python selector, independent of the selectors on the other axes.
fn selector(layout: &Layout, axis: usize, obj: &Bound<'_, PyAny>) -> PyResult<Selector> {
    let size = *layout.shape().get(axis).ok_or_else(|| PyIndexError::new_err("too many axis selectors"))?;
    if let Ok(slice) = obj.cast::<PySlice>() {
        let s = slice.indices(size as isize)?;
        return Ok(Selector::Range { start: s.start.max(0) as usize, len: s.slicelength, step: s.step });
    }
    if let Ok(s) = obj.cast::<PyString>() { return Ok(Selector::At(key(layout, axis, s.to_str()?)?)); }
    if !obj.is_instance_of::<PyBool>() {
        if let Ok(i) = obj.extract::<i64>() { return Ok(Selector::At(at(i, size)?)); }
    }
    if let Ok(names) = obj.extract::<Vec<String>>() {
        let positions = names.iter().map(|n| key(layout, axis, n).map(Some)).collect::<PyResult<Vec<_>>>()?;
        return Ok(Selector::Indices { layout: Layout::from([positions.len()]), positions });
    }
    let indices = convert::one(obj)?;
    let (within, positions) = match indices.view() {
        View::Booleans(mask) => {
            if indices.layout().shape() != [size] { return Err(PyIndexError::new_err("a Boolean selector must be a vector matching its axis")); }
            let positions: Vec<_> = mask.iter().enumerate().filter_map(|(i, &b)| b.then_some(Some(i))).collect();
            (Layout::from([positions.len()]), positions)
        }
        View::Integers(items) => {
            let named = indices.layout().names().iter().any(Option::is_some);
            (indices.layout().clone(), structure::index_positions((0..items.len()).map(|i| items.get(i)), size, named).map_err(py_err)?)
        }
        _ if indices.view().is_empty() => (indices.layout().clone(), vec![]),
        _ => return Err(PyTypeError::new_err("an axis selector must be an integer, key, slice, Boolean mask or integer array")),
    };
    Ok(Selector::Indices { layout: within, positions })
}

/// Decodes brackets once; the same plan serves reads and assignment.
pub fn selection(layout: &Layout, index: &Bound<'_, PyAny>) -> PyResult<Plan> {
    let parts: Vec<_> = match index.cast::<PyTuple>() { Ok(t) => t.iter().collect(), Err(_) => vec![index.clone()] };
    let ellipsis = index.py().Ellipsis();
    let used = parts.iter().filter(|p| !p.is_none() && !p.is(&ellipsis)).count();
    if used > layout.shape().len() { return Err(PyIndexError::new_err("too many axis selectors")); }
    if parts.iter().filter(|p| p.is(&ellipsis)).count() > 1 { return Err(PyIndexError::new_err("only one ellipsis is allowed")); }
    let (mut axis, mut selectors) = (0, Vec::new());
    for p in parts {
        if p.is_none() { selectors.push(Selector::NewAxis); }
        else if p.is(&ellipsis) {
            for _ in 0..layout.shape().len() - used { selectors.push(Selector::all(layout.shape()[axis])); axis += 1; }
        } else { selectors.push(selector(layout, axis, &p)?); axis += 1; }
    }
    structure::select(layout, &selectors).map_err(py_err)
}

fn planned(arg: &Arg<'_>, plan: Plan) -> PyResult<Array> { Ok(Array::new(Core::gathered(arg.view(), &plan).map_err(py_err)?)) }
/// The layout that `axis` refers to, and the axis's position: `x`'s own layout, or the flattened layout and its one axis when `axis` is `None`.
fn along(x: &Arg<'_>, axis: Option<Axis>) -> PyResult<(Layout, usize)> {
    match axis { Some(a) => { let l = x.layout(); let a = a.of(&l)?; Ok((l, a)) }, None => Ok((Layout::from([x.view().len()]), 0)) }
}

/// Copies under a new shape; one -1 infers its length. Reshaping drops axis metadata.
pub fn reshaped(view: View<'_>, lengths: &[isize]) -> PyResult<Array> {
    let mut sizes = lengths.iter().map(|&n| if n == -1 { Ok(1) } else { usize::try_from(n).map_err(|_| PyValueError::new_err("shape lengths must be nonnegative, or one -1")) }).collect::<PyResult<Vec<_>>>()?;
    let free: Vec<_> = lengths.iter().enumerate().filter_map(|(a, &n)| (n == -1).then_some(a)).collect();
    if free.len() > 1 { return Err(PyValueError::new_err("only one length can be -1")); }
    if let Some(&a) = free.first() {
        let known = element_count(&sizes).map_err(py_err)?;
        if known == 0 || view.len() % known != 0 { return Err(PyValueError::new_err("no length for -1 fits the items")); }
        sizes[a] = view.len() / known;
    }
    generated_len(&sizes).map_err(py_err)?;
    Ok(Array::new(Core::new(sizes.into(), Buffer::copied(view, 0..view.len())).map_err(py_err)?))
}

#[pyfunction]
pub fn reshape(x: &Bound<'_, PyAny>, shape: Vec<isize>) -> PyResult<Array> { reshaped(convert::one(x)?.view(), &shape) }
#[pyfunction]
pub fn ravel(x: &Bound<'_, PyAny>) -> PyResult<Array> { let x = convert::one(x)?; reshaped(x.view(), &[x.view().len() as isize]) }

/// Reverses the axis order by default. `axes` gives the source axis for each result axis.
#[pyfunction]
#[pyo3(signature = (x, axes=None))]
pub fn transpose(x: &Bound<'_, PyAny>, axes: Option<Vec<Axis>>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.layout();
    let rank = layout.shape().len();
    let axes = match axes { Some(a) => a.iter().map(|a| a.of(&layout)).collect::<PyResult<Vec<_>>>()?, None => (0..rank).rev().collect() };
    if axes.len() != rank { return Err(PyValueError::new_err("transpose needs one source for each axis")); }
    let mut destinations = vec![usize::MAX; rank];
    for (to, &from) in axes.iter().enumerate() {
        if destinations[from] != usize::MAX { return Err(PyValueError::new_err("transpose axes must not repeat")); }
        destinations[from] = to;
    }
    planned(&x, structure::transpose(&layout, &destinations).map_err(py_err)?)
}

#[pyfunction]
pub fn swapaxes(x: &Bound<'_, PyAny>, axis1: Axis, axis2: Axis) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.layout();
    let mut axes: Vec<_> = (0..layout.shape().len()).collect();
    axes.swap(axis1.of(&layout)?, axis2.of(&layout)?);
    planned(&x, structure::transpose(&layout, &axes).map_err(py_err)?)
}

/// Inserts a length-one axis. Negative positions count from the resulting rank.
#[pyfunction]
pub fn expand_dims(x: &Bound<'_, PyAny>, axis: i64) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.layout();
    let at = at(axis, layout.shape().len() + 1)?;
    planned(&x, structure::select_axis(&layout, at, Selector::NewAxis).map_err(py_err)?)
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn squeeze(x: &Bound<'_, PyAny>, axis: Option<Axes>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.layout();
    let axes = match axis {
        None => layout.shape().iter().enumerate().filter_map(|(a, &n)| (n == 1).then_some(a)).collect(),
        Some(Axes::One(a)) => vec![a.of(&layout)?],
        Some(Axes::Many(axes)) => axes.iter().map(|a| a.of(&layout)).collect::<PyResult<Vec<_>>>()?,
    };
    if axes.iter().any(|&a| layout.shape()[a] != 1) { return Err(PyValueError::new_err("squeeze axes must have length one")); }
    let selectors = layout.shape().iter().enumerate().map(|(a, &n)| if axes.contains(&a) { Selector::At(0) } else { Selector::all(n) }).collect::<Vec<_>>();
    planned(&x, structure::select(&layout, &selectors).map_err(py_err)?)
}

/// Gathers independent positions on one axis, or on the flattened input without `axis`.
#[pyfunction]
#[pyo3(signature = (x, indices, axis=None))]
pub fn take(x: &Bound<'_, PyAny>, indices: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (layout, axis) = along(&x, axis)?;
    let selector = selector(&layout, axis, indices)?;
    planned(&x, structure::select_axis(&layout, axis, selector).map_err(py_err)?)
}

#[pyfunction]
#[pyo3(signature = (x, shift, axis=None))]
pub fn roll(x: &Bound<'_, PyAny>, shift: i64, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let flattened = axis.is_none();
    let (layout, a) = along(&x, axis)?;
    // Python's positive shift moves items right; the shared rotation starts at its count.
    let n = layout.shape()[a];
    let left = if n == 0 { 0 } else { (-(shift as i128)).rem_euclid(n as i128) as i64 };
    let mut plan = structure::rotate(&layout, a, Some((&[], &[left]))).map_err(py_err)?;
    if flattened { plan.layout = x.layout(); }
    planned(&x, plan)
}

#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub fn flip(x: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.layout();
    let selected = axis.map(|a| a.of(&layout)).transpose()?;
    let selectors: Vec<_> = layout.shape().iter().enumerate().map(|(a, &n)| if selected.is_none_or(|s| s == a) {
        Selector::Range { start: n.saturating_sub(1), len: n, step: -1 }
    } else { Selector::all(n) }).collect();
    planned(&x, structure::select(&layout, &selectors).map_err(py_err)?)
}

/// Joins arrays along `axis`, the first by default. `axis=None` joins the flattened arrays.
#[pyfunction]
#[pyo3(signature = (arrays, axis=crate::search::FIRST_AXIS), text_signature = "(arrays, axis=0)")]
pub fn concatenate(arrays: Vec<Bound<'_, PyAny>>, axis: Option<Axis>) -> PyResult<Array> {
    let args: Vec<_> = arrays.iter().collect();
    let call = convert::convert(&args)?;
    let first = call.args.first().ok_or_else(|| PyValueError::new_err("concatenate needs at least one array"))?;
    let at = axis.map(|a| a.of(&first.layout())).transpose()?;
    let layouts: Vec<_> = call.args.iter().map(|a| if at.is_some() { a.layout() } else { Layout::from([a.view().len()]) }).collect();
    let pairs: Vec<_> = layouts.iter().zip(&call.args).map(|(l, a)| (l, a.view())).collect();
    Ok(Array::new(Core::concatenate(&pairs, at.unwrap_or(0)).map_err(py_err)?))
}

#[pyfunction]
#[pyo3(signature = (arrays, axis=0))]
pub fn stack(arrays: Vec<Bound<'_, PyAny>>, axis: i64) -> PyResult<Array> {
    let args: Vec<_> = arrays.iter().collect();
    let call = convert::convert(&args)?;
    let first = call.args.first().ok_or_else(|| PyValueError::new_err("stack needs at least one array"))?;
    if call.args.iter().any(|a| a.layout().shape() != first.layout().shape()) { return Err(PyValueError::new_err("stack arrays must have one shape")); }
    let at = at(axis, first.layout().shape().len() + 1)?;
    let layouts = call.args.iter().map(|a| structure::select_axis(&a.layout(), at, Selector::NewAxis).map(|p| p.layout).map_err(py_err)).collect::<PyResult<Vec<_>>>()?;
    let pairs: Vec<_> = layouts.iter().zip(&call.args).map(|(l, a)| (l, a.view())).collect();
    Ok(Array::new(Core::concatenate(&pairs, at).map_err(py_err)?))
}

/// Splits along `axis`, the first by default, into equal sections or at the supplied boundaries. `axis=None` splits the flattened input. Each result is compact.
#[pyfunction]
#[pyo3(signature = (x, sections, axis=crate::search::FIRST_AXIS), text_signature = "(x, sections, axis=0)")]
pub fn split(x: &Bound<'_, PyAny>, sections: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Vec<Array>> {
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
        planned(&x, plan)
    }).collect()
}

#[pyfunction]
#[pyo3(signature = (x, repeats, axis=None))]
pub fn repeat(x: &Bound<'_, PyAny>, repeats: &Bound<'_, PyAny>, axis: Option<Axis>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (layout, a) = along(&x, axis)?;
    let counts = convert::one(repeats)?;
    if counts.layout().shape().len() > 1 { return Err(PyValueError::new_err("repeat counts must be a scalar or vector")); }
    let counts = counts.items::<i64>()?;
    planned(&x, structure::repeat(&layout, a, &counts).map_err(py_err)?)
}

/// Constant padding with the dtype's fill: zero, false or a space.
#[pyfunction]
pub fn pad(x: &Bound<'_, PyAny>, pad_width: &Bound<'_, PyAny>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let rank = x.layout().shape().len();
    let widths = if let Ok(n) = pad_width.extract::<usize>() { vec![(n, n); rank] }
    else if let Ok(pair) = pad_width.extract::<(usize, usize)>() { vec![pair; rank] }
    else { pad_width.extract::<Vec<(usize, usize)>>()? };
    planned(&x, structure::pad(&x.layout(), &widths).map_err(py_err)?)
}

/// The coordinates of `x`'s nonzero items, one row of axis positions for each item, and the number of rows.
fn nonzero_rows(x: &Arg<'_>) -> (usize, Vec<i64>) {
    let positions = Buffer::nonzero(x.view());
    let shape = x.layout().shape().to_vec();
    let stride = rustymath::layout::strides(&shape);
    (positions.len(), positions.iter().flat_map(|&i| shape.iter().zip(&stride).map(move |(&n, &s)| ((i / s) % n) as i64)).collect())
}

#[pyfunction]
pub fn nonzero<'py>(py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyTuple>> {
    let x = convert::one(x)?;
    let rank = x.layout().shape().len();
    let (n, rows) = nonzero_rows(&x);
    let axes = (0..rank).map(|a| core(rows.iter().skip(a).step_by(rank).copied().collect(), &[n]).map(Array::new)).collect::<PyResult<Vec<_>>>()?;
    PyTuple::new(py, axes)
}

#[pyfunction]
pub fn argwhere(x: &Bound<'_, PyAny>) -> PyResult<Array> {
    let x = convert::one(x)?;
    let (n, rows) = nonzero_rows(&x);
    Ok(Array::new(core(rows, &[n, x.layout().shape().len()])?))
}

#[pyfunction(name = "where")]
pub fn where_<'py>(py: Python<'py>, condition: &Bound<'py, PyAny>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let condition = convert::one(condition)?;
    let View::Booleans(mask) = condition.view() else { return Err(PyTypeError::new_err("where condition must be Boolean")) };
    let call = convert::convert(&[x, y])?;
    let args = [&call.args[0], &call.args[1]];
    let result = Core::choose((&condition.layout(), mask), (&args[0].layout(), args[0].view()), (&args[1].layout(), args[1].view())).map_err(py_err)?;
    give(py, result, condition.scalar && call.scalar())
}

/// A compact copy of a diagonal, moved to the last result axis. Its axes default to the last two, the axes that the matrix functions take.
#[pyfunction]
#[pyo3(signature = (x, offset=0, axis1=Axis::Position(-2), axis2=Axis::Position(-1)))]
pub fn diagonal(x: &Bound<'_, PyAny>, offset: i64, axis1: Axis, axis2: Axis) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.layout();
    planned(&x, structure::diagonal(&layout, [axis1.of(&layout)?, axis2.of(&layout)?], offset).map_err(py_err)?)
}

/// Diagonal construction or extraction; `shape` allows a scalar to fill a rectangular matrix's diagonal.
#[pyfunction]
#[pyo3(signature = (x, k=0, shape=None, dtype=None))]
fn _diag(x: &Bound<'_, PyAny>, k: i64, shape: Option<[usize; 2]>, dtype: Option<&str>) -> PyResult<Array> {
    let x = match dtype { Some(d) => convert::one_at(x, convert::dtype(d)?)?, None => convert::one(x)? };
    let layout = x.layout();
    if let Some(shape) = shape { return Ok(Array::new(rustymath::create::diagonal(x.view(), shape, k).map_err(py_err)?)); }
    match layout.shape() {
        &[len] => {
            let n = len.checked_add(usize::try_from(k.unsigned_abs()).unwrap_or(usize::MAX)).ok_or_else(|| PyValueError::new_err("diagonal shape is too large"))?;
            Ok(Array::new(rustymath::create::diagonal(x.view(), [n, n], k).map_err(py_err)?))
        }
        [_, _] => planned(&x, structure::diagonal(&layout, [0, 1], k).map_err(py_err)?),
        _ => Err(PyValueError::new_err("diag takes a vector or a matrix")),
    }
}

#[pyfunction]
fn _triangle(x: &Bound<'_, PyAny>, k: i64, upper: bool) -> PyResult<Array> {
    let x = convert::one(x)?;
    Ok(Array::new(rustymath::create::triangle(x.layout(), x.view(), k, upper).map_err(py_err)?))
}

/// A compact array of `shape`, broadcasting from the front rather than NumPy's trailing-axis rule.
#[pyfunction]
pub fn broadcast_to(x: &Bound<'_, PyAny>, shape: &Bound<'_, PyAny>) -> PyResult<Array> {
    let x = convert::one(x)?;
    Ok(Array::new(Core::broadcasted(&x.layout(), x.view(), &crate::create::shape(shape)?).map_err(py_err)?))
}

#[pyfunction]
#[pyo3(signature = (x, axis=Axis::Position(-1)))]
fn _difference(x: &Bound<'_, PyAny>, axis: Axis) -> PyResult<Array> {
    let x = convert::one(x)?;
    let layout = x.layout();
    Ok(Array::new(Core::differences(&layout, x.view(), axis.of(&layout)?).map_err(py_err)?))
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(reshape, m)?, wrap_pyfunction!(ravel, m)?, wrap_pyfunction!(transpose, m)?, wrap_pyfunction!(swapaxes, m)?,
        wrap_pyfunction!(expand_dims, m)?, wrap_pyfunction!(squeeze, m)?, wrap_pyfunction!(take, m)?, wrap_pyfunction!(roll, m)?, wrap_pyfunction!(flip, m)?,
        wrap_pyfunction!(concatenate, m)?, wrap_pyfunction!(stack, m)?, wrap_pyfunction!(split, m)?, wrap_pyfunction!(repeat, m)?, wrap_pyfunction!(pad, m)?,
        wrap_pyfunction!(nonzero, m)?, wrap_pyfunction!(argwhere, m)?, wrap_pyfunction!(where_, m)?, wrap_pyfunction!(diagonal, m)?,
        wrap_pyfunction!(_diag, m)?, wrap_pyfunction!(_triangle, m)?, wrap_pyfunction!(broadcast_to, m)?, wrap_pyfunction!(_difference, m)?] { m.add_function(f)?; }
    m.add("flatten", m.getattr("ravel")?)
}
