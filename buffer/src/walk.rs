//! Nested Python values walked into a layout and the pieces that fill it, in row-major order. Lists, tuples and ranges are axes. A collector gives the piece for any other object, and a `dict` it doesn't take is an axis keyed by its keys. A level whose items differ in shape gives each item to its collector whole: rustymath's arrays raise there, and basedpl nests the items.
use crate::{labels, py_err};
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyRange, PyTuple};
use rustymath::error::{Error, Kind};
use rustymath::layout::{Keys, Layout, element_count, generated_len};
use std::{cell::RefCell, sync::Arc};

/// What a walk of nested Python values collects.
pub trait Collect {
    type Piece;
    /// The piece for the item `n` of a `range`.
    fn int(&mut self, n: i128) -> Self::Piece;
    /// The piece for `obj`, with the shape of its items, which is empty for one value. `None` for an object that holds no values of this kind.
    fn piece(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<Option<(Self::Piece, Vec<usize>)>>;
    /// The piece for an item of a level whose items differ in shape, taken whole. Such a level raises unless the collector nests its items.
    fn element(&mut self, _obj: &Bound<'_, PyAny>) -> PyResult<Self::Piece> { Err(py_err(Error::new(Kind::Length, "ragged nested sequence"))) }
}

thread_local! {
    /// The containers that this thread's walks are inside, which a container holding itself would revisit.
    static OPEN: RefCell<Vec<usize>> = const { RefCell::new(Vec::new()) };
}

/// Runs `f` with `obj` open, raising `ValueError` where a walk inside `obj` reaches `obj` again. Collectors that convert a container's items themselves, as basedpl's records do, open it too.
pub fn within<R>(obj: &Bound<'_, PyAny>, f: impl FnOnce() -> PyResult<R>) -> PyResult<R> {
    let id = obj.as_ptr() as usize;
    if OPEN.with_borrow(|open| open.contains(&id)) { return Err(PyValueError::new_err("cyclic Python container")); }
    OPEN.with_borrow_mut(|open| open.push(id));
    let result = f();
    OPEN.with_borrow_mut(|open| open.pop());
    result
}

/// An axis's length, with the keys of the dict that gave it.
type Axis = (usize, Option<Vec<Option<Arc<str>>>>);

/// The nested values of `obj`: their layout, with the keys of any dicts, and the pieces that `collect` gives them.
pub fn walked<C: Collect>(obj: &Bound<'_, PyAny>, collect: &mut C) -> PyResult<(Layout, Vec<C::Piece>)> {
    let mut walk = Walk { collect, pieces: vec![], axes: vec![] };
    walk.walk(obj)?;
    let shape: Vec<usize> = walk.axes.iter().rev().map(|a| a.0).collect();
    element_count(&shape).map_err(py_err)?;
    let layout = Layout::from(shape.as_slice());
    if walk.axes.iter().all(|a| a.1.is_none()) { return Ok((layout, walk.pieces)); }
    let keys = walk.axes.into_iter().rev().map(|a| a.1.map(Keys::partial).transpose()).collect::<Result<Vec<_>, _>>().map_err(py_err)?;
    Ok((layout.with_keys(keys).map_err(py_err)?, walk.pieces))
}

/// A walk in progress, with its pieces so far, and the axes of the values it has walked, innermost first. A value's axes run from where they start to the end, so a container compares its items' axes in place, and adds its own axis last.
struct Walk<'c, C: Collect> { collect: &'c mut C, pieces: Vec<C::Piece>, axes: Vec<Axis> }

impl<C: Collect> Walk<'_, C> {
    /// Adds the pieces of `obj` and their axes, and gives where its axes start.
    fn walk(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<usize> {
        if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() { return self.items(obj, obj, None); }
        let start = self.axes.len();
        if obj.is_instance_of::<PyRange>() {
            let len = generated_len(&[obj.len()?]).map_err(py_err)?;
            let (first, step): (i128, i128) = (obj.getattr("start")?.extract()?, obj.getattr("step")?.extract()?);
            for i in 0..len as i128 {
                let piece = self.collect.int(first + i * step);
                self.pieces.push(piece);
            }
            self.axes.push((len, None));
            return Ok(start);
        }
        if let Some((piece, shape)) = self.collect.piece(obj)? {
            self.pieces.push(piece);
            self.axes.extend(shape.into_iter().rev().map(|n| (n, None)));
            return Ok(start);
        }
        if let Ok(d) = obj.cast::<PyDict>() { return self.items(obj, &d.values().into_any(), Some(labels(&d.keys())?)); }
        Err(PyTypeError::new_err(format!("can't convert {}", obj.get_type().name()?)))
    }
    /// Adds the list, tuple or dict `obj`, whose items are `values`, keyed by `keys` for a dict, and gives where its axes start. Where its items differ in shape, each gives one piece whole.
    fn items(&mut self, obj: &Bound<'_, PyAny>, values: &Bound<'_, PyAny>, keys: Option<Vec<Option<Arc<str>>>>) -> PyResult<usize> {
        within(obj, || {
            let (pieces, start) = (self.pieces.len(), self.axes.len());
            let (mut first, mut ragged) = (true, false);
            for v in values.try_iter()? {
                let item = self.walk(&v?)?;
                if std::mem::take(&mut first) { continue; }
                if self.axes[start..item] != self.axes[item..] { ragged = true; break; }
                self.axes.truncate(item);
            }
            if ragged {
                self.pieces.truncate(pieces);
                self.axes.truncate(start);
                for v in values.try_iter()? {
                    let piece = self.collect.element(&v?)?;
                    self.pieces.push(piece);
                }
            }
            self.axes.push((values.len()?, keys));
            Ok(start)
        })
    }
}
