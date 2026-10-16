//! The `rustymath.Array` class: a rustymath [`Array`](rustymath::array::Array) that readers of the buffer protocol, such as NumPy, can view without a copy.
use crate::functions;
use pyo3::exceptions::{PyIndexError, PyKeyError, PyTypeError, PyValueError};
use pyo3::types::{PyComplex, PyString, PyTuple};
use pyo3::{IntoPyObjectExt, ffi, prelude::*};
use pyo3::pyclass::CompareOp;
use rustymath::array::Array as Core;
use rustymath::buffer::{Buffer, DType, Element, FloatWidth, Source, View};
use rustymath::layout::{Layout, element_count, strides};
use rustymath_buffer::{export, geometry, item, name, named, nested, py_err};
use std::ffi::c_int;
use std::sync::Arc;

/// An array of one dtype, with a shape, and optional keys and names for its axes. NumPy and other readers of the buffer protocol can view and write its items without a copy.
#[pyclass(name = "Array", module = "rustymath")]
pub struct Array { core: Core, shape: Box<[isize]>, strides: Box<[isize]> }

impl Array {
    pub fn new(core: Core) -> Self {
        let (shape, strides) = geometry(core.dtype(), core.layout().shape());
        Self { core, shape, strides }
    }
    pub fn core(&self) -> &Core { &self.core }
}

/// `core` as a Python object: with `scalar`, a rank-0 result is a Python number, and anything else is a `rustymath.Array`. A function passes `scalar` where every argument was a Python scalar, and where its result covers the whole of its arguments, as a reduction of a whole array or one decoded vector of digits does.
pub fn give(py: Python<'_>, core: Core, scalar: bool) -> PyResult<Bound<'_, PyAny>> {
    if scalar && core.layout().shape().is_empty() { item(py, core.view(), 0) } else { Ok(Bound::new(py, Array::new(core))?.into_any()) }
}

/// `items` as a rustymath array of `shape`, with no keys or names.
pub fn core<T: Element>(items: Vec<T>, shape: &[usize]) -> PyResult<Core> { Core::new(shape.into(), T::buffer(items)).map_err(py_err) }

/// `NotImplemented`, which tells Python to try the other operand's method.
fn not_implemented(py: Python<'_>) -> Bound<'_, PyAny> { py.NotImplemented().into_bound(py) }

/// An operator's `result`, or `NotImplemented` where rustymath can't convert the other operand.
fn operand<'py>(py: Python<'py>, result: PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    match result { Err(e) if e.is_instance_of::<PyTypeError>(py) => Ok(not_implemented(py)), r => r }
}

/// The position on `axis` of `layout` that `key` gives: an integer, counting from the end when negative, or the name of a key.
fn position(layout: &Layout, axis: usize, key: &Bound<'_, PyAny>) -> PyResult<usize> {
    let n = layout.shape()[axis];
    if let Ok(name) = key.cast::<PyString>() {
        let name = name.to_str()?;
        return layout.keys(axis).and_then(|k| k.position(name)).ok_or_else(|| PyKeyError::new_err(name.to_owned()));
    }
    let k: i64 = key.extract()?;
    rustymath::layout::position(k, n).ok_or_else(|| PyIndexError::new_err(format!("position {k} is outside an axis of length {n}")))
}

/// Item `i` of `view` as the repr shows it: a 16- or 32-bit float, or each part of a complex number with 32-bit parts, as the shortest decimal that rounds back to it at its own width.
fn shown<'py>(py: Python<'py>, view: View<'_>, i: usize) -> PyResult<Bound<'py, PyAny>> {
    match view {
        View::Floats(v) => shortest(v.get(i), v.tag()).into_bound_py_any(py),
        View::Complexes(v) => { let (z, w) = (v.get(i), v.tag().into()); Ok(PyComplex::from_doubles(py, shortest(z.re, w), shortest(z.im, w)).into_any()) }
        _ => item(py, view, i),
    }
}

/// The shortest decimal that rounds to `x` at `width`, read as an `f64`.
fn shortest(x: f64, width: FloatWidth) -> f64 {
    if width == FloatWidth::F64 || !x.is_finite() { return x; }
    rustymath::with_float_width!(width, F => (0..17).map(|p| format!("{x:.p$e}").parse().expect("Rust reads the floats it writes"))
        .find(|&y: &f64| Source::<F>::read(y) == Source::<F>::read(x)).unwrap_or(x))
}

#[pymethods]
impl Array {
    #[getter]
    fn dtype(&self) -> &'static str { name(self.core.dtype()) }
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, self.core.layout().shape()) }
    #[getter]
    fn ndim(&self) -> usize { self.shape.len() }
    #[getter]
    fn size(&self) -> usize { self.core.len() }
    /// For each axis, `None`, or the name of each position, which is `None` where a position has no name.
    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let layout = self.core.layout();
        let axes = (0..self.shape.len()).map(|a| layout.keys(a).map(|k| PyTuple::new(py, k.names().iter().map(|n| n.as_deref()))).transpose()).collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, axes)
    }
    /// For each axis, its name or `None`. Assign one name or `None` for each axis to rename the axes, or `()` to remove every name.
    #[getter]
    fn names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, (0..self.shape.len()).map(|a| self.core.layout().name(a).map(|n| &**n))) }
    #[setter]
    fn set_names(&mut self, names: Vec<Option<String>>) -> PyResult<()> { self.core.set_names(names.into_iter().map(|n| n.map(Arc::from)).collect()).map_err(py_err) }
    fn __len__(&self) -> PyResult<usize> {
        self.core.layout().shape().first().copied().ok_or_else(|| PyTypeError::new_err("a rank-0 array has no length"))
    }
    /// The item at one position or key on each leading axis, given alone or as a tuple: a Python scalar when every axis is given, and otherwise the cell they select, as a new array.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let layout = self.core.layout();
        let keys: Vec<_> = match key.cast::<PyTuple>() { Ok(t) => t.iter().collect(), Err(_) => vec![key.clone()] };
        let rank = layout.shape().len();
        if keys.len() > rank { return Err(PyIndexError::new_err(format!("an array of rank {rank} takes at most {rank} positions"))); }
        let steps = strides(layout.shape());
        let start = keys.iter().enumerate().map(|(a, k)| Ok(position(layout, a, k)? * steps[a])).sum::<PyResult<usize>>()?;
        if keys.len() == rank { return item(py, self.core.view(), start); }
        let cell: usize = layout.shape()[keys.len()..].iter().product();
        give(py, Core::new(layout.axes(keys.len()..rank), Buffer::copied(self.core.view(), start..start + cell)).map_err(py_err)?, false)
    }
    /// The items as nested lists of Python scalars, or one Python scalar for a rank-0 array.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> { rustymath_buffer::tolist(py, self.core.view(), self.core.layout().shape()) }
    /// A copy at `dtype`, a name such as `"f64"`, keeping keys and names. Items convert as `Buffer::cast` converts them: floats to integers truncate toward zero, a value that doesn't fit raises `OverflowError`, and a complex number with an imaginary part, or any conversion to or from characters, raises `ValueError`.
    fn astype(&self, dtype: &str) -> PyResult<Array> {
        let to = named(dtype).ok_or_else(|| PyValueError::new_err(format!("unknown dtype {dtype:?}")))?;
        let buffer = Buffer::cast(self.core.view(), to).map_err(py_err)?;
        Ok(Array::new(Core::new(self.core.layout().clone(), buffer).map_err(py_err)?))
    }
    /// A copy under a new shape, given as integers or as one sequence. One length can be -1, which takes the length that fits the items. Keys and names don't carry over.
    #[pyo3(signature = (*shape))]
    fn reshape(&self, shape: &Bound<'_, PyTuple>) -> PyResult<Array> {
        let lengths: Vec<isize> = match shape.len() {
            1 if shape.get_item(0)?.extract::<isize>().is_err() => shape.get_item(0)?.extract()?,
            _ => shape.extract()?,
        };
        let n = self.core.len();
        let mut sizes = lengths.iter().map(|&l| if l == -1 { Ok(1) } else { usize::try_from(l).map_err(|_| PyValueError::new_err(format!("a length can't be {l}"))) }).collect::<PyResult<Vec<usize>>>()?;
        let mut free = lengths.iter().enumerate().filter(|&(_, &l)| l == -1).map(|(a, _)| a);
        if let Some(a) = free.next() {
            if free.next().is_some() { return Err(PyValueError::new_err("only one length can be -1")); }
            let known = element_count(&sizes).map_err(py_err)?;
            if known == 0 || n % known != 0 { return Err(PyValueError::new_err(format!("no length for -1 fits {n} items"))); }
            sizes[a] = n / known;
        }
        element_count(&sizes).map_err(py_err)?;
        Ok(Array::new(Core::new(Layout::from(sizes), Buffer::copied(self.core.view(), 0..n)).map_err(py_err)?))
    }
    fn __add__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::add(slf.py(), slf, y)) }
    fn __radd__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::add(slf.py(), x, slf)) }
    fn __sub__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::subtract(slf.py(), slf, y)) }
    fn __rsub__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::subtract(slf.py(), x, slf)) }
    fn __mul__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::multiply(slf.py(), slf, y)) }
    fn __rmul__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::multiply(slf.py(), x, slf)) }
    fn __truediv__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::divide(slf.py(), slf, y)) }
    fn __rtruediv__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::divide(slf.py(), x, slf)) }
    fn __matmul__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), crate::linalg::matmul(slf.py(), slf, y)) }
    fn __rmatmul__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), crate::linalg::matmul(slf.py(), x, slf)) }
    fn __mod__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::remainder(slf.py(), slf, y, None)) }
    fn __rmod__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(slf.py(), functions::remainder(slf.py(), x, slf, None)) }
    fn __pow__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>, modulo: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if !modulo.is_none() { return Ok(not_implemented(slf.py())) }
        operand(slf.py(), functions::power(slf.py(), slf, y))
    }
    fn __rpow__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>, modulo: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if !modulo.is_none() { return Ok(not_implemented(slf.py())) }
        operand(slf.py(), functions::power(slf.py(), x, slf))
    }
    fn __neg__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> { functions::subtract(slf.py(), &0.into_bound_py_any(slf.py())?, slf) }
    fn __abs__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> { functions::absolute(slf.py(), slf) }
    fn __richcmp__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>, op: CompareOp) -> PyResult<Bound<'py, PyAny>> {
        let compare = match op { CompareOp::Eq => functions::equal, CompareOp::Ne => functions::not_equal, CompareOp::Lt => functions::less, CompareOp::Le => functions::less_equal, CompareOp::Gt => functions::greater, CompareOp::Ge => functions::greater_equal };
        operand(slf.py(), compare(slf.py(), slf, y, None))
    }
    /// `rustymath.sum` of this array.
    #[pyo3(signature = (axis=None))]
    fn sum<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>) -> PyResult<Bound<'py, PyAny>> { functions::sum(slf.py(), slf, axis) }
    /// `rustymath.prod` of this array.
    #[pyo3(signature = (axis=None))]
    fn prod<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>) -> PyResult<Bound<'py, PyAny>> { functions::prod(slf.py(), slf, axis) }
    /// `rustymath.max` of this array.
    #[pyo3(signature = (axis=None))]
    fn max<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>) -> PyResult<Bound<'py, PyAny>> { functions::max(slf.py(), slf, axis) }
    /// `rustymath.min` of this array.
    #[pyo3(signature = (axis=None))]
    fn min<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>) -> PyResult<Bound<'py, PyAny>> { functions::min(slf.py(), slf, axis) }
    /// `rustymath.all` of this array.
    #[pyo3(signature = (axis=None))]
    fn all<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>) -> PyResult<Bound<'py, PyAny>> { functions::all(slf.py(), slf, axis) }
    /// `rustymath.any` of this array.
    #[pyo3(signature = (axis=None))]
    fn any<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>) -> PyResult<Bound<'py, PyAny>> { functions::any(slf.py(), slf, axis) }
    /// `rustymath.cumsum` of this array.
    #[pyo3(signature = (axis=None))]
    fn cumsum<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axis>) -> PyResult<Bound<'py, PyAny>> { functions::cumsum(slf.py(), slf, axis) }
    /// `rustymath.cumprod` of this array.
    #[pyo3(signature = (axis=None))]
    fn cumprod<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axis>) -> PyResult<Bound<'py, PyAny>> { functions::cumprod(slf.py(), slf, axis) }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut text = format!("Array({}, dtype={}", nested(py, self.core.layout().shape(), &|i| shown(py, self.core.view(), i))?.repr()?, self.dtype());
        let layout = self.core.layout();
        if layout.has_keys() { text += &format!(", keys={}", self.keys(py)?.repr()?); }
        if layout.names().iter().any(Option::is_some) { text += &format!(", names={}", self.names(py)?.repr()?); }
        Ok(text + ")")
    }
    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        let owner = slf.clone().into_any();
        let mut this = slf.borrow_mut();
        let (dtype, len, data) = (this.core.dtype(), this.core.len(), this.core.as_mut_ptr());
        // A NumPy view of another dtype could write a byte other than 0 or 1 into a `bool`, or a surrogate into a `char`, so those two export read-only.
        let readonly = matches!(dtype, DType::Boolean | DType::Character);
        // SAFETY: the object owns its items, shape and strides, and keeps them while `owner`, which the export holds, lives.
        unsafe { export(view, flags, &owner, dtype, data, len, &this.shape, &this.strides, readonly) }
    }
}

/// A new array holding a copy of `x`: a `rustymath.Array`, any object that exports a buffer, such as a NumPy array, or nested Python values.
#[pyfunction]
fn array(x: &Bound<'_, PyAny>) -> PyResult<Array> {
    Ok(Array::new(crate::convert::one(x)?.into_core()?))
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Array>()?;
    m.add_function(wrap_pyfunction!(array, m)?)
}
