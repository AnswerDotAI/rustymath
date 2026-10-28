use crate::py_err;
use pyo3::{exceptions::{PyIndexError, PyKeyError, PyTypeError}, prelude::*, types::{PyBool, PyInt, PyList, PySlice, PyString, PyTuple}};
use rustymath::{buffer::View, gather::{IndexSource, Table}, layout::{Layout, position}, structure::{self, IndexPart, Plan, Selector}};
use std::sync::Arc;

pub fn key(layout: &Layout, axis: usize, name: &str) -> PyResult<usize> {
    layout.keys(axis).and_then(|k| k.position(name)).ok_or_else(|| PyKeyError::new_err(name.to_owned()))
}

/// Whether `selector` decodes `obj` itself, with no conversion: a slice, a key, an integer or a list of keys. Other selectors go to its `array` callback.
pub fn native(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_instance_of::<PySlice>() || obj.is_instance_of::<PyString>() || (obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>())
        || ((obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>()) && obj.extract::<Vec<String>>().is_ok())
}

/// One Python selector, independent of the selectors on the other axes.
pub fn selector(layout: &Layout, axis: usize, obj: &Bound<'_, PyAny>, array: impl FnOnce() -> PyResult<(Selector, bool)>) -> PyResult<(Selector, bool)> {
    let size = *layout.shape().get(axis).ok_or_else(|| PyIndexError::new_err("too many axis selectors"))?;
    if !native(obj) { return array(); }
    if let Ok(slice) = obj.cast::<PySlice>() {
        let py = obj.py();
        let bound = |name| -> PyResult<Bound<'_, PyAny>> {
            let value = slice.getattr(name)?;
            if let Ok(s) = value.cast::<PyString>() { Ok(key(layout, axis, s.to_str()?)?.into_pyobject(py)?.into_any()) } else { Ok(value) }
        };
        let (start, mut stop, step) = (bound("start")?, bound("stop")?, slice.getattr("step")?);
        if slice.getattr("stop")?.is_instance_of::<PyString>() {
            let reverse = !step.is_none() && step.extract::<isize>()? < 0;
            let at = stop.extract::<isize>()?;
            stop = if reverse && at == 0 { py.None().into_bound(py) } else { (at + if reverse { -1 } else { 1 }).into_pyobject(py)?.into_any() };
        }
        let s = py.get_type::<PySlice>().call1((start, stop, step))?.cast_into::<PySlice>()?.indices(size as isize)?;
        return Ok((Selector::Range { start: s.start.max(0) as usize, len: s.slicelength, step: s.step }, false));
    }
    if let Ok(s) = obj.cast::<PyString>() { return Ok((Selector::At(key(layout, axis, s.to_str()?)?), true)); }
    if obj.is_instance_of::<PyInt>() && !obj.is_instance_of::<PyBool>() {
        let i = obj.extract::<i64>()?;
        return Ok((Selector::At(position(i, size).ok_or_else(|| PyIndexError::new_err(format!("position {i} is outside an axis of length {size}")))?), true));
    }
    if (obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>()) && let Ok(names) = obj.extract::<Vec<String>>() {
        let positions = names.iter().map(|n| key(layout, axis, n)).collect::<PyResult<Vec<_>>>()?;
        return Ok((Selector::Indices { layout: Layout::from([positions.len()]), positions: Table::new(positions) }, false));
    }
    unreachable!("`native` admits only the kinds decoded above")
}

/// The selector that an array's items give: a character string picks a key, Booleans form a mask, and a table reads other numbers in place as positions.
pub fn array_selector(target: &Layout, axis: usize, layout: Layout, indices: Arc<dyn IndexSource>) -> PyResult<(Selector, bool)> {
    let size = *target.shape().get(axis).ok_or_else(|| PyIndexError::new_err("too many axis selectors"))?;
    let positions = match indices.view() {
        View::Characters(text) if layout.shape().len() <= 1 => return Ok((Selector::At(key(target, axis, &text.iter().collect::<String>())?), true)),
        View::Booleans(mask) => {
            if layout.shape() != [size] { return Err(PyIndexError::new_err("a Boolean selector must be a vector matching its axis")); }
            let positions: Vec<_> = mask.iter().enumerate().filter_map(|(i, &b)| b.then_some(i)).collect();
            return Ok((Selector::Indices { layout: Layout::from([positions.len()]), positions: Table::new(positions) }, false));
        }
        View::Integers(_) | View::Floats(_) => structure::index_table(indices.clone(), size).map_err(py_err)?,
        view if view.is_empty() => Table::new(vec![]),
        _ => return Err(PyTypeError::new_err("an axis selector must be an integer, key, slice, Boolean mask or integer or float array")),
    };
    Ok((Selector::Indices { layout, positions }, false))
}

/// Keeps structural markers in their written order until the applicable frame is known.
pub fn parts<'py>(index: &Bound<'py, PyAny>) -> PyResult<Vec<IndexPart<Bound<'py, PyAny>>>> {
    let parts: Vec<_> = match index.cast::<PyTuple>() { Ok(t) => t.iter().collect(), Err(_) => vec![index.clone()] };
    let ellipsis = index.py().Ellipsis();
    if parts.iter().filter(|p| p.is(&ellipsis)).count() > 1 { return Err(PyIndexError::new_err("only one ellipsis is allowed")); }
    parts.into_iter().map(|p| {
        Ok(if p.is_none() { IndexPart::NewAxis }
        else if p.is(&ellipsis) {
            IndexPart::Rest
        } else if p.is_instance_of::<PySlice>() && ["start", "stop", "step"].iter().all(|n| p.getattr(*n).is_ok_and(|v| v.is_none())) { IndexPart::All }
        else { IndexPart::Axis(p) })
    }).collect()
}

/// Decodes brackets once for reads and assignment. Named selectors address their axis through `structure::addressed`.
pub fn selection(
    layout: &Layout, index: &Bound<'_, PyAny>, name: impl FnMut(&Bound<'_, PyAny>) -> PyResult<Option<String>>,
    array: impl FnMut(&Layout, usize, &Bound<'_, PyAny>) -> PyResult<(Selector, bool)>) -> PyResult<(Plan, bool)> {
    select_parts(layout, parts(index)?, name, array)
}

pub fn select_parts(
    layout: &Layout, written: Vec<IndexPart<Bound<'_, PyAny>>>, mut name: impl FnMut(&Bound<'_, PyAny>) -> PyResult<Option<String>>,
    mut array: impl FnMut(&Layout, usize, &Bound<'_, PyAny>) -> PyResult<(Selector, bool)>) -> PyResult<(Plan, bool)> {
    let names = written.iter().map(|p| p.axis().map_or(Ok(None), &mut name)).collect::<PyResult<Vec<_>>>()?;
    let claims: Vec<_> = written.iter().zip(&names).map(|(p, n)| p.as_ref().map(|_| (n.as_deref(), 1))).collect();
    structure::plan_parts(layout, &claims, None, true, |i, axis| {
        let IndexPart::Axis(p) = &written[i] else { unreachable!("an axis refers to its written payload") };
        selector(layout, axis, p, || array(layout, axis, p))
    }, py_err)
}
