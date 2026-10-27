//! The `rustymath.Array` class: a rustymath [`Array`](rustymath::array::Array) that readers of the buffer protocol, such as NumPy, can view without a copy.
use crate::functions;
use crate::convert::released;
use pyo3::exceptions::{PyTypeError, PyValueError};
use pyo3::types::{PyComplex, PyTuple};
use pyo3::{IntoPyObjectExt, ffi, prelude::*};
use pyo3::pyclass::CompareOp;
use rustymath::array::{Array as Core, ArrayView};
use rustymath::buffer::{Buffer, Element, FloatWidth, Items, View};
use rustymath::layout::Layout;
use rustymath_buffer::{item, lend, nested, py_err};
use std::ffi::c_int;
use std::sync::Arc;

/// An array of one dtype, with a shape, and optional keys and names for its axes. Buffer readers receive read-only snapshots of its items.
#[pyclass(name = "Array", module = "rustymath")]
pub struct Array { core: Core }

impl Array {
    pub fn new(core: Core) -> Self { Self { core } }
    pub fn core(&self) -> &Core { &self.core }
}

/// `core` as a Python object: with `scalar`, a rank-0 result is a Python number, and anything else is a `rustymath.Array`. A function passes `scalar` where every argument was a Python scalar, and where its result covers the whole of its arguments, as a reduction of a whole array or one decoded vector of digits does.
pub fn give(py: Python<'_>, core: Core, scalar: bool) -> PyResult<Bound<'_, PyAny>> {
    if scalar && core.layout().shape().is_empty() { item(py, core.strided().contiguous().expect("one strided item is contiguous"), 0) } else { Ok(Bound::new(py, Array::new(core))?.into_any()) }
}

/// The array of `buffer`'s items laid out by `layout`, given as [`give`] gives it.
pub fn give_items(py: Python<'_>, layout: Layout, buffer: Buffer, scalar: bool) -> PyResult<Bound<'_, PyAny>> { give(py, Core::new(layout, buffer).map_err(py_err)?, scalar) }

/// `items` as a rustymath array of `shape`, with no keys or names.
pub fn core<T: Element>(items: Vec<T>, shape: &[usize]) -> PyResult<Core> { Core::new(shape.into(), T::buffer(items)).map_err(py_err) }

/// `NotImplemented`, which tells Python to try the other operand's method.
fn not_implemented(py: Python<'_>) -> Bound<'_, PyAny> { py.NotImplemented().into_bound(py) }

const ARRAY_PRIORITY: f64 = 100.0;

/// Let richer arrays handle mixed operators before attempting eager conversion.
fn operand<'py>(other: &Bound<'py, PyAny>, call: impl FnOnce() -> PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let py = other.py();
    if other.getattr("__array_priority__").and_then(|v| v.extract::<f64>()).unwrap_or(0.0) > ARRAY_PRIORITY { return Ok(not_implemented(py)); }
    match call() { Err(e) if e.is_instance_of::<PyTypeError>(py) => Ok(not_implemented(py)), r => r }
}

/// Item `i` rounded for display, without changing the stored value or scalar conversions.
fn shown<'py>(py: Python<'py>, view: ArrayView<'_>, i: usize) -> PyResult<Bound<'py, PyAny>> {
    let i = view.mapping().get(i).expect("a strided view has no fills");
    match view.storage() {
        View::Floats(v) => display_float(v.get(i), v.tag()).into_bound_py_any(py),
        View::Complexes(v) => { let (z, w) = (v.get(i), v.tag().into()); Ok(PyComplex::from_doubles(py, display_float(z.re, w), display_float(z.im, w)).into_any()) }
        view => item(py, view, i),
    }
}

/// Round to the width's display precision, retaining Python's scalar repr and scientific notation.
fn display_float(x: f64, width: FloatWidth) -> f64 {
    if x == 0.0 || !x.is_finite() { return x; }
    let digits = match width { FloatWidth::F16 => 3, FloatWidth::F32 => 6, FloatWidth::F64 => 8 };
    let p = digits - 1;
    format!("{x:.p$e}").parse().expect("Rust reads the floats it writes")
}

/// Method aliases live beside their signatures; all computation stays in the function bindings.
macro_rules! array_methods {
    (folds [$($fold:ident),*] scans [$($scan:ident),*] parts [$($part:ident),*] ddof [$($ddof:ident),*] search_axes [$($search:ident),*]
        copies [$($copy:ident => $copy_fn:ident),*]
        operators [$($forward:ident, $reverse:ident => $operator:ident),*]
        calls [$($(#[$call_meta:meta])* $call:ident($($call_arg:ident: $call_type:ty),*) => $call_module:ident::$call_fn:ident;)*]
        arrays [$($(#[$array_meta:meta])* $array:ident($($array_arg:ident: $array_type:ty),*) => $array_fn:ident;)*]
        { $($body:tt)* }) => {
        // PyO3 emits unsafe wrapper items beside the impl; scope its generated glue's lint allowance here.
        #[allow(unsafe_op_in_unsafe_fn)]
        mod methods {
            use super::*;
            #[pymethods]
            impl Array {
            $($body)*
            $(#[pyo3(signature = (axis=None))]
            fn $fold<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>) -> PyResult<Bound<'py, PyAny>> { functions::$fold(slf.py(), slf, axis) })*
            $(#[pyo3(signature = (axis=None))]
            fn $scan<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axis>) -> PyResult<Bound<'py, PyAny>> { functions::$scan(slf.py(), slf, axis) })*
            $(#[getter]
            fn $part<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> { functions::$part(slf.py(), slf) })*
            $(#[pyo3(signature = (axis=None, ddof=0))]
            fn $ddof<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axes>, ddof: usize) -> PyResult<Bound<'py, PyAny>> { functions::$ddof(slf.py(), slf, axis, ddof) })*
            $(#[pyo3(signature = (axis=None))]
            fn $search<'py>(slf: &Bound<'py, Self>, axis: Option<functions::Axis>) -> PyResult<Bound<'py, PyAny>> { crate::search::$search(slf.py(), slf, axis) })*
            $(fn $copy(slf: &Bound<'_, Self>) -> PyResult<Array> { crate::structure::$copy_fn(slf.py(), slf.as_any()) })*
            $(fn $forward<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(y, || functions::$operator(slf.py(), slf, y)) }
            fn $reverse<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(x, || functions::$operator(slf.py(), x, slf)) })*
            $($(#[$call_meta])*
            fn $call<'py>(slf: &Bound<'py, Self>, $($call_arg: $call_type),*) -> PyResult<Bound<'py, PyAny>> { crate::$call_module::$call_fn(slf.py(), slf, $($call_arg),*) })*
            $($(#[$array_meta])*
            fn $array<'py>(slf: &Bound<'py, Self>, $($array_arg: $array_type),*) -> PyResult<Array> { crate::structure::$array_fn(slf.py(), slf.as_any(), $($array_arg),*) })*
            }
        }
    };
}

array_methods! { folds [sum, prod, max, min, all, any, mean] scans [cumsum, cumprod] parts [real, imag] ddof [var, std] search_axes [argmin, argmax, median]
    copies [ravel => ravel, flatten => ravel]
    operators [__add__, __radd__ => add, __sub__, __rsub__ => subtract, __mul__, __rmul__ => multiply, __truediv__, __rtruediv__ => divide,
        __and__, __rand__ => bitwise_and, __or__, __ror__ => bitwise_or, __xor__, __rxor__ => bitwise_xor,
        __lshift__, __rlshift__ => left_shift, __rshift__, __rrshift__ => right_shift]
    calls [
        conj() => functions::conj;
        #[pyo3(signature = (decimals=0))] round(decimals: i32) => functions::round;
        clip(lower: &Bound<'py, PyAny>, upper: &Bound<'py, PyAny>) => functions::clip;
        #[pyo3(signature = (axis=None, weights=None))] average(axis: Option<functions::Axes>, weights: Option<&Bound<'py, PyAny>>) => functions::average;
        #[pyo3(signature = (axis=None, ord=2.0))] norm(axis: Option<functions::Axes>, ord: f64) => functions::norm;
        #[pyo3(signature = (axis=crate::search::LAST_AXIS, descending=false))] sort(axis: Option<functions::Axis>, descending: bool) => search::sort;
        #[pyo3(signature = (kth, axis=crate::search::LAST_AXIS))] partition(kth: i64, axis: Option<functions::Axis>) => search::partition;
        #[pyo3(signature = (kth, axis=crate::search::LAST_AXIS))] argpartition(kth: i64, axis: Option<functions::Axis>) => search::argpartition;
        #[pyo3(signature = (q, axis=None))] quantile(q: &Bound<'py, PyAny>, axis: Option<functions::Axis>) => search::quantile;
    ]
    arrays [
        #[pyo3(signature = (axes=None))] transpose(axes: Option<Vec<functions::Axis>>) => transpose;
        swapaxes(axis1: functions::Axis, axis2: functions::Axis) => swapaxes;
        #[pyo3(signature = (axis=None))] squeeze(axis: Option<functions::Axes>) => squeeze;
        #[pyo3(signature = (indices, axis=None, mode="raise", fill=None))] take(indices: &Bound<'py, PyAny>, axis: Option<functions::Axis>, mode: &str, fill: Option<&Bound<'py, PyAny>>) => take;
    ] {
    #[classattr]
    #[allow(non_upper_case_globals)]
    const __array_priority__: f64 = ARRAY_PRIORITY;
    #[getter]
    fn dtype(&self) -> &'static str { self.core.dtype().name() }
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, self.core.layout().shape()) }
    #[getter]
    fn ndim(&self) -> usize { self.core.layout().shape().len() }
    #[getter]
    fn size(&self) -> usize { self.core.len() }
    /// For each axis, `None`, or a tuple of position keys with `None` for unlabelled positions. Assign the same representation to replace keys without reordering values, or `None` to clear them.
    #[getter]
    fn keys<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        let layout = self.core.layout();
        let axes = (0..self.ndim()).map(|a| layout.keys(a).map(|k| PyTuple::new(py, k.names().iter().map(|n| n.as_deref()))).transpose()).collect::<PyResult<Vec<_>>>()?;
        PyTuple::new(py, axes)
    }
    #[setter]
    pub(super) fn set_keys(&mut self, keys: Option<Vec<Option<Vec<Option<String>>>>>) -> PyResult<()> {
        let keys = rustymath_buffer::axis_keys(keys.unwrap_or_default())?;
        self.core.set_keys(keys).map_err(py_err)
    }
    /// For each axis, its name or `None`. Assign one name or `None` for each axis to rename the axes, or `None` to clear them.
    #[getter]
    fn names<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> { PyTuple::new(py, (0..self.ndim()).map(|a| self.core.layout().name(a).map(|n| &**n))) }
    #[setter]
    pub(super) fn set_names(&mut self, names: Option<Vec<Option<String>>>) -> PyResult<()> { self.core.set_names(names.unwrap_or_default().into_iter().map(|n| n.map(Arc::from)).collect()).map_err(py_err) }
    fn __rustymath_array__<'py>(&self, py: Python<'py>) -> PyResult<(Array, Bound<'py, PyTuple>, Bound<'py, PyTuple>)> {
        Ok((Array::new(self.core.clone()), self.keys(py)?, self.names(py)?))
    }
    fn __len__(&self) -> PyResult<usize> {
        self.core.layout().shape().first().copied().ok_or_else(|| PyTypeError::new_err("a rank-0 array has no length"))
    }
    fn __bool__(&self, py: Python<'_>) -> PyResult<bool> {
        if self.core.len() != 1 { return Err(PyValueError::new_err("an array with other than one item has ambiguous truth")); }
        let view = self.core.strided();
        item(py, view.storage(), view.mapping().get(0).expect("a strided view has no fills"))?.is_truthy()
    }
    /// Selects each axis by an integer, key, slice, mask or index array. Index arrays pair by broadcasting from the front, as arguments do, and their axes take the first array's place. `None` inserts an axis and `...` keeps omitted axes. Basic slices share storage, with independent value semantics. No remaining axes gives a Python scalar.
    fn __getitem__<'py>(&self, py: Python<'py>, key: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        let plan = crate::structure::selection(self.core.layout(), key)?;
        give(py, released(py, || self.core.selected(plan, None)).map_err(py_err)?, true)
    }
    fn __contains__(slf: &Bound<'_, Self>, query: &Bound<'_, PyAny>) -> PyResult<bool> { crate::search::contains(slf.py(), slf, query) }
    /// Writes through the same selection as indexing, broadcasting from the front. Repeated targets take the last assigned item. Other arrays and existing buffer exports keep their values.
    fn __setitem__(slf: &Bound<'_, Self>, key: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let plan = crate::structure::selection(slf.try_borrow()?.core.layout(), key)?;
        // Finish reading/copying first: `value` can be this array, or a buffer that views its items.
        let source = crate::convert::one(value)?.into_core()?;
        slf.try_borrow_mut()?.core.assign(&plan, &source).map_err(py_err)
    }
    /// The items as nested lists of Python scalars, or one Python scalar for a rank-0 array.
    fn tolist<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let view = self.core.strided();
        nested(py, view.layout().shape(), &|i| item(py, view.storage(), view.mapping().get(i).expect("a strided view has no fills")))
    }
    /// A copy at `dtype`, a name such as `"float64"`, keeping keys and names. Items convert as `Buffer::cast` converts them: floats to integers truncate toward zero, a value that doesn't fit raises `OverflowError`, and a complex number with an imaginary part, or any conversion to or from characters, raises `ValueError`.
    fn astype(&self, py: Python<'_>, dtype: &str) -> PyResult<Array> {
        let to = crate::convert::dtype(dtype)?;
        let buffer = released(py, || self.core.view().cast(to).map(|items| match items { Items::Borrowed(v) => Buffer::copied(v, 0..v.len()), Items::Owned(b) => b })).map_err(py_err)?;
        Ok(Array::new(Core::new(self.core.layout().clone(), buffer).map_err(py_err)?))
    }
    /// Compact items without retaining a larger base allocation. Small views otherwise keep their whole base alive. Keys and names are preserved.
    fn compact(&self, py: Python<'_>) -> Array { Array::new(released(py, || self.core.compact())) }
    /// A new shape, given as integers or as one sequence, sharing storage: a view keeps its mapping, which reads the same items in order. One length can be -1. Keys and names don't carry over.
    #[pyo3(signature = (*shape))]
    fn reshape(&self, py: Python<'_>, shape: &Bound<'_, PyTuple>) -> PyResult<Array> {
        let lengths: Vec<isize> = match shape.len() {
            1 if shape.get_item(0)?.extract::<isize>().is_err() => shape.get_item(0)?.extract()?,
            _ => shape.extract()?,
        };
        crate::structure::reshaped(py, &self.core, &lengths)
    }
    fn __matmul__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(y, || crate::linalg::matmul(slf.py(), slf, y)) }
    fn __rmatmul__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(x, || crate::linalg::matmul(slf.py(), x, slf)) }
    fn __mod__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(y, || functions::remainder(slf.py(), slf, y)) }
    fn __rmod__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(x, || functions::remainder(slf.py(), x, slf)) }
    fn __floordiv__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(y, || functions::floor_divide(slf.py(), slf, y)) }
    fn __rfloordiv__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { operand(x, || functions::floor_divide(slf.py(), x, slf)) }
    fn __pow__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>, modulo: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if !modulo.is_none() { return Ok(not_implemented(slf.py())) }
        operand(y, || functions::power(slf.py(), slf, y))
    }
    fn __rpow__<'py>(slf: &Bound<'py, Self>, x: &Bound<'py, PyAny>, modulo: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
        if !modulo.is_none() { return Ok(not_implemented(slf.py())) }
        operand(x, || functions::power(slf.py(), x, slf))
    }
    fn __neg__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> { functions::subtract(slf.py(), &0.into_bound_py_any(slf.py())?, slf) }
    fn __invert__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> { functions::bitwise_not(slf.py(), slf) }
    fn __abs__<'py>(slf: &Bound<'py, Self>) -> PyResult<Bound<'py, PyAny>> { functions::absolute(slf.py(), slf) }
    fn __richcmp__<'py>(slf: &Bound<'py, Self>, y: &Bound<'py, PyAny>, op: CompareOp) -> PyResult<Bound<'py, PyAny>> {
        let compare = match op { CompareOp::Eq => functions::equal, CompareOp::Ne => functions::not_equal, CompareOp::Lt => functions::less, CompareOp::Le => functions::less_equal, CompareOp::Gt => functions::greater, CompareOp::Ge => functions::greater_equal };
        operand(y, || compare(slf.py(), slf, y))
    }
    #[getter]
    #[pyo3(name = "T")]
    fn transposed(slf: &Bound<'_, Self>) -> PyResult<Array> { crate::structure::transpose(slf.py(), slf.as_any(), None) }
    fn __repr__(&self, py: Python<'_>) -> PyResult<String> {
        let mut text = format!("Array({}, dtype={}", nested(py, self.core.layout().shape(), &|i| shown(py, self.core.strided(), i))?.repr()?, self.dtype());
        let layout = self.core.layout();
        if layout.has_keys() { text += &format!(", keys={}", self.keys(py)?.repr()?); }
        if layout.names().iter().any(Option::is_some) { text += &format!(", names={}", self.names(py)?.repr()?); }
        Ok(text + ")")
    }
    unsafe fn __getbuffer__(slf: Bound<'_, Self>, view: *mut ffi::Py_buffer, flags: c_int) -> PyResult<()> {
        let core = slf.borrow().core.clone();
        let (offset, strides) = core.strided().mapping().affine(core.layout().shape()).expect("a strided view is affine");
        let (dtype, shape) = (core.dtype(), core.layout().shape().to_vec());
        // SAFETY: the offset and strides reach only items of `core`, which the export holds.
        unsafe { lend(view, flags, slf.py(), core, |c| c.strided().storage().as_ptr().add(offset * dtype.bytes()), dtype, &shape, Some(&strides)) }
    }
} }

/// A new array holding a copy of `x`: a `rustymath.Array`, any object that exports a buffer, such as a NumPy array, or nested Python values.
#[pyfunction]
#[pyo3(signature = (x, dtype=None, *, keys=None, names=None))]
fn array(x: &Bound<'_, PyAny>, dtype: Option<&str>, keys: Option<Vec<Option<Vec<Option<String>>>>>, names: Option<Vec<Option<String>>>) -> PyResult<Array> {
    let value = match dtype { Some(d) => crate::convert::one_at(x, crate::convert::dtype(d)?)?.into_core()?, None => crate::convert::one(x)?.into_core()? };
    let mut array = Array::new(value);
    if keys.is_some() { array.set_keys(keys)?; }
    if names.is_some() { array.set_names(names)?; }
    Ok(array)
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Array>()?;
    m.add_function(wrap_pyfunction!(array, m)?)
}
