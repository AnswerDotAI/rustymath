//! Data frames, read without importing a frame library. Any object with `columns` and `__array__`, as a pandas `DataFrame` has, reads as a matrix of rows by columns.
use crate::{labels, py_err};
use pyo3::{prelude::*, types::{PyDict, PyString}};
use rustymath::layout::Keys;
use std::sync::Arc;

/// A data frame's columns and labels, oriented as a matrix of rows by columns.
pub struct Frame<'py> {
    /// Each column's items, from its `to_numpy()`: a NumPy array, or a list for a column of Python objects, with `None` where a value is missing.
    pub columns: Vec<Bound<'py, PyAny>>,
    /// The number of rows, which a frame with no columns still has.
    pub rows: usize,
    /// Keys for the rows and for the columns. The rows take the index's labels only where every label is a string. The columns take [`labels`]'s rule, as a dict's keys do.
    pub keys: [Option<Arc<Keys>>; 2],
    /// The names of the rows' and the columns' axes: the index's and the column labels' `name`, where it's a string.
    pub names: [Option<Arc<str>>; 2],
}

/// The frame that `obj` holds, or `None` for an object that isn't a frame.
pub fn frame<'py>(obj: &Bound<'py, PyAny>) -> PyResult<Option<Frame<'py>>> {
    if !obj.hasattr("columns")? || !obj.hasattr("__array__")? { return Ok(None); }
    let labelled = obj.getattr("columns")?;
    let column_keys = labels(&labelled)?;
    let columns = labelled.try_iter()?.map(|label| column(&obj.get_item(label?)?)).collect::<PyResult<Vec<_>>>()?;
    let index = if obj.hasattr("index")? { Some(obj.getattr("index")?) } else { None };
    let rows = obj.len()?;
    let row_keys = match &index { Some(index) if rows > 0 => row_keys(index)?, _ => None };
    let column_keys = if column_keys.iter().all(Option::is_none) { None } else { Some(Keys::partial(column_keys).map_err(py_err)?) };
    Ok(Some(Frame { columns, rows, keys: [row_keys, column_keys], names: [index.as_ref().map(name).transpose()?.flatten(), name(&labelled)?] }))
}

/// A column's items: a NumPy array, or a list for a column of Python objects, which no buffer holds. pandas gives a missing value in an object column as its `NA`, which `na_value=None` replaces. A `to_numpy` that takes no `na_value` gives its plain items.
fn column<'py>(series: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let items = series.call_method0("to_numpy")?;
    if items.getattr("dtype")?.getattr("kind")?.extract::<String>()? != "O" { return Ok(items); }
    let options = PyDict::new(series.py());
    options.set_item("na_value", series.py().None())?;
    series.call_method("to_numpy", (), Some(&options)).unwrap_or(items).call_method0("tolist")
}

/// Keys for the rows where every label of `index` is a string. An index of positions, numbers or dates labels nothing that a key can hold, so the rows stay unkeyed.
fn row_keys(index: &Bound<'_, PyAny>) -> PyResult<Option<Arc<Keys>>> {
    let mut keys = Vec::new();
    for label in index.try_iter()? {
        match label?.cast::<PyString>() { Ok(s) => keys.push(Some(Arc::from(s.to_str()?))), Err(_) => return Ok(None) }
    }
    Keys::partial(keys).map(Some).map_err(py_err)
}

/// The `name` of an index or of a frame's column labels, where it's a string.
fn name(labels: &Bound<'_, PyAny>) -> PyResult<Option<Arc<str>>> {
    if !labels.hasattr("name")? { return Ok(None); }
    Ok(labels.getattr("name")?.cast::<PyString>().ok().map(|s| s.to_str().map(Arc::from)).transpose()?)
}
