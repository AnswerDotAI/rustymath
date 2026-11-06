use crate::py_err;
use pyo3::{exceptions::{PyIndexError, PyKeyError, PyTypeError}, prelude::*, types::{PyInt, PyList, PySlice, PyString, PyTuple}};
use rustymath::{buffer::View, gather::{IndexSource, Table}, layout::{Layout, position}, structure::{self, IndexPart, Plan, Selector}};
use std::sync::Arc;

/// The position of key `name` on `axis` of `layout`, as a slice's bound reads it: a missing bound is a `KeyError`, since a slice needs a position on the axis.
fn key(layout: &Layout, axis: usize, name: &str) -> PyResult<usize> {
    layout.keys(axis).and_then(|k| k.position(name)).ok_or_else(|| PyKeyError::new_err(name.to_owned()))
}

/// Whether `selector` decodes `obj` itself, with no conversion: a slice, a key, an integer or a list of keys. Other selectors go to its `array` callback.
pub fn native(obj: &Bound<'_, PyAny>) -> bool { obj.is_instance_of::<PySlice>() || obj.is_instance_of::<PyInt>() || key_names(obj).is_some() }

/// The keys that `obj` names: one for a string, and one for each item of a list or tuple of strings. `None` for any other selector.
pub fn key_names(obj: &Bound<'_, PyAny>) -> Option<Vec<String>> {
    if let Ok(s) = obj.cast::<PyString>() { return s.to_str().ok().map(|s| vec![s.to_owned()]); }
    if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() { obj.extract().ok() } else { None }
}

/// One Python selector, independent of the selectors on the other axes. Keys read by `structure::key_selector`, so a missing key reads the fill, as in BPL.
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
    if let Some(names) = key_names(obj) {
        let one = obj.is_instance_of::<PyString>();
        let within = (!one).then(|| Layout::from([names.len()]));
        return Ok((structure::key_selector(layout, axis, names.iter().map(String::as_str), within).map_err(py_err)?, one));
    }
    if obj.is_instance_of::<PyInt>() {
        let i = obj.extract::<i64>()?;
        return Ok((Selector::At(position(i, size).ok_or_else(|| PyIndexError::new_err(format!("position {i} is outside an axis of length {size}")))?), true));
    }
    unreachable!("`native` admits only the kinds decoded above")
}

/// The selector that an array's items give: a character string picks a key, or the fill where the axis lacks it, and a table reads Booleans and other numbers in place as positions, as `⌷` does.
pub fn array_selector(target: &Layout, axis: usize, layout: Layout, indices: Arc<dyn IndexSource>) -> PyResult<(Selector, bool)> {
    let size = *target.shape().get(axis).ok_or_else(|| PyIndexError::new_err("too many axis selectors"))?;
    let positions = match indices.view() {
        View::Characters(text) if layout.shape().len() <= 1 => {
            return Ok((structure::key_selector(target, axis, [&*text.iter().collect::<String>()], None).map_err(py_err)?, true));
        }
        View::Booleans(_) | View::Integers(_) | View::Floats(_) => structure::index_table(indices.clone(), size, rustymath::tolerant::Tolerance::EXACT).map_err(py_err)?,
        view if view.is_empty() => Table::new(vec![]),
        _ => return Err(PyTypeError::new_err("an axis selector must be an integer, key, slice, or Boolean, integer or float array")),
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

/// Decodes brackets once for reads and assignments. Named selectors address their axis through `structure::addressed`.
pub fn selection(
    layout: &Layout, index: &Bound<'_, PyAny>, name: impl FnMut(&Bound<'_, PyAny>) -> PyResult<Option<String>>,
    array: impl FnMut(&Layout, usize, &Bound<'_, PyAny>) -> PyResult<(Selector, bool)>) -> PyResult<(Plan, bool)> {
    select_parts(layout, parts(index)?, name, array)
}

fn select_parts(
    layout: &Layout, written: Vec<IndexPart<Bound<'_, PyAny>>>, name: impl FnMut(&Bound<'_, PyAny>) -> PyResult<Option<String>>,
    mut array: impl FnMut(&Layout, usize, &Bound<'_, PyAny>) -> PyResult<(Selector, bool)>) -> PyResult<(Plan, bool)> {
    let names = axis_names(&written, name)?;
    structure::plan_parts(layout, &claims(&written, &names), None, true, |i, axis| {
        let IndexPart::Axis(p) = &written[i] else { unreachable!("an axis refers to its written payload") };
        selector(layout, axis, p, || array(layout, axis, p))
    }, py_err)
}

/// For each axis of `layout`, the keys that a write by `index` names with Python's own strings and lists of strings and the axis lacks (`structure::missing_keys`). The write adds them before it plans (`structure::extended`), as BPL's assignment does. `name` gives a part's one axis name, as for `selection`.
pub fn missing_keys(layout: &Layout, index: &Bound<'_, PyAny>, name: impl FnMut(&Bound<'_, PyAny>) -> PyResult<Option<String>>) -> PyResult<Vec<Vec<Arc<str>>>> {
    let written = parts(index)?;
    let names = axis_names(&written, name)?;
    let keys: Vec<_> = written.iter().map(|p| p.axis().and_then(key_names).unwrap_or_default()).collect();
    let routes = structure::addressed_parts(layout, &claims(&written, &names), None).map_err(py_err)?;
    structure::missing_keys(layout, routes.iter().filter_map(|(axis, p)| match p { IndexPart::Axis(i) => Some((*axis, keys[*i].iter().map(String::as_str))), _ => None }))
        .map_err(py_err)
}

/// The one axis name of each written axis part, where `name` gives one.
fn axis_names(written: &[IndexPart<Bound<'_, PyAny>>], mut name: impl FnMut(&Bound<'_, PyAny>) -> PyResult<Option<String>>) -> PyResult<Vec<Option<String>>> {
    written.iter().map(|p| p.axis().map_or(Ok(None), &mut name)).collect()
}

/// The claims that address the written parts' axes (`structure::addressed_parts`): each axis part's name, from `names`, and one axis.
fn claims<'a>(written: &[IndexPart<Bound<'_, PyAny>>], names: &'a [Option<String>]) -> Vec<IndexPart<(Option<&'a str>, usize)>> {
    written.iter().zip(names).map(|(p, n)| p.as_ref().map(|_| (n.as_deref(), 1))).collect()
}
