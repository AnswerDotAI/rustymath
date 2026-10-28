//! Python array interchange and selectors, shared by rustymath and basedpl. [`read`] borrows native contiguous buffers and normalizes other supported buffers. [`read_array`] preserves sibling-array metadata. [`export`] fills a `Py_buffer`, and [`lend`] fills one read-only for items that the export itself keeps alive. [`labels`] and [`frame`] read dict keys and data frames, [`selection`] decodes Python indexing, [`walk`] walks nested Python values, and [`py_err`] converts array errors to Python exceptions.
use pyo3::exceptions::{PyBufferError, PyIndexError, PyKeyboardInterrupt, PyOverflowError, PyTimeoutError, PyTypeError, PyValueError};
use pyo3::{IntoPyObjectExt, ffi, prelude::*, types::{PyComplex, PyInt, PyList, PyString}};
use rustymath::buffer::{Buffer, ComplexWidth, DType, Element, FloatWidth, View, Width};
use rustymath::error::{Error, Kind};
use rustymath::{layout::element_count, with_dtype};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::{mem::size_of, ptr, slice, sync::{Arc, Mutex}};

pub mod frame;
pub mod selection;
pub mod walk;

/// Decodes the Python per-axis key representation; shared layout methods validate its shape.
pub fn axis_keys(keys: Vec<Option<Vec<Option<String>>>>) -> PyResult<Vec<Option<Arc<rustymath::layout::Keys>>>> {
    keys.into_iter().map(|k| k.map(|v| rustymath::layout::Keys::partial(v.into_iter().map(|s| s.map(Arc::from)).collect())).transpose())
        .collect::<Result<Vec<_>, _>>().map_err(py_err)
}

/// Keys from a Python dict's keys or a frame's column labels: a `str` is a key, and an `int` equal to its position leaves that position unkeyed. A `TypeError` for any other label.
pub fn labels(labels: &Bound<'_, PyAny>) -> PyResult<Vec<Option<Arc<str>>>> {
    labels.try_iter()?.enumerate().map(|(i, label)| {
        let label = label?;
        if let Ok(s) = label.cast::<PyString>() { return Ok(Some(Arc::from(s.to_str()?))); }
        if label.is_exact_instance_of::<PyInt>() && label.extract::<usize>().is_ok_and(|n| n == i) { return Ok(None); }
        Err(PyTypeError::new_err("keys must be strings, or ints equal to their positions"))
    }).collect()
}

/// The exception that a signal handler raised during an interrupt check, which [`py_err`] raises for the check's Interrupt error.
static RAISED: Mutex<Option<PyErr>> = Mutex::new(None);

/// The Interrupt error that stops a computation at `e`, the exception that a signal handler raised during an interrupt check. [`py_err`] raises `e` for it.
pub fn raised(e: PyErr) -> Error {
    *RAISED.lock().expect("no panic while holding the raised exception") = Some(e);
    Error::new(Kind::Interrupt, "interrupted by a signal")
}

/// The Python exception for a rustymath error: the exception that [`raised`] keeps, or `KeyboardInterrupt`, for Interrupt, `TimeoutError` for Timeout, `OverflowError` for Overflow, `IndexError` for Index, and `ValueError` for the other kinds.
pub fn py_err(e: Error) -> PyErr {
    let text = e.to_string();
    match e.kind {
        Kind::Interrupt => RAISED.lock().expect("no panic while holding the raised exception").take().unwrap_or_else(|| PyKeyboardInterrupt::new_err(text)),
        Kind::Timeout => PyTimeoutError::new_err(text), Kind::Overflow => PyOverflowError::new_err(text), Kind::Index => PyIndexError::new_err(text),
        _ => PyValueError::new_err(text)
    }
}

/// Item `i` of `view` as a Python `bool`, `int`, `float`, `complex` or one-character `str`.
pub fn item<'py>(py: Python<'py>, view: View<'_>, i: usize) -> PyResult<Bound<'py, PyAny>> {
    match view {
        View::Booleans(v) => v[i].into_bound_py_any(py), View::Integers(v) => v.get(i).into_bound_py_any(py), View::Floats(v) => v.get(i).into_bound_py_any(py),
        View::Complexes(v) => { let z = v.get(i); Ok(PyComplex::from_doubles(py, z.re, z.im).into_any()) } View::Characters(v) => v[i].into_bound_py_any(py)
    }
}

/// The items of `view` under `shape` as nested lists of Python scalars, or one scalar when `shape` is empty. It needs no NumPy, and reads every dtype, complex numbers and characters included.
pub fn tolist<'py>(py: Python<'py>, view: View<'_>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> { nested(py, shape, &|i| item(py, view, i)) }

/// The items that `item` gives for each position in row-major order, nested in lists by `shape`, or the item at position 0 when `shape` is empty.
pub fn nested<'py>(py: Python<'py>, shape: &[usize], item: &dyn Fn(usize) -> PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    nest(py, shape, 0, item)
}

fn nest<'py>(py: Python<'py>, shape: &[usize], start: usize, item: &dyn Fn(usize) -> PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let [n, rest @ ..] = shape else { return item(start) };
    let cell: usize = rest.iter().product();
    let items = (0..*n).map(|k| nest(py, rest, start + k * cell, item)).collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any())
}
/// Each dtype, with the PEP 3118 format code that [`export`] writes for it.
const DTYPES: [(DType, &CStr); 11] = [
    (DType::Boolean, c"?"), (DType::Integer(Width::U8), c"B"), (DType::Integer(Width::I16), c"h"), (DType::Integer(Width::I32), c"i"),
    (DType::Integer(Width::I64), c"q"), (DType::Float(FloatWidth::F16), c"e"), (DType::Float(FloatWidth::F32), c"f"), (DType::Float(FloatWidth::F64), c"d"),
    (DType::Complex(ComplexWidth::F32), c"Zf"), (DType::Complex(ComplexWidth::F64), c"Zd"), (DType::Character, c"w"),
];

/// Format codes that [`read`] also accepts. `l`, C's `long`, has a size that varies by platform, so [`source`] reads it by its item size.
const ALSO: [(&[u8], DType); 3] = [(b"c", DType::Integer(Width::U8)), (b"n", DType::Integer(Width::I64)), (b"1w", DType::Character)];

/// Integer formats with no dtype of their own, each with its item size and the type that [`read`] widens from. Signed bytes widen to int16, and wider unsigned integers to int64.
const WIDENED: [(&[u8], usize, Widen); 6] =
    [(b"b", 1, Widen::I8), (b"H", 2, Widen::U16), (b"I", 4, Widen::U32), (b"L", 4, Widen::U32), (b"L", 8, Widen::U64), (b"Q", 8, Widen::U64)];

/// The PEP 3118 format code that [`export`] writes for `dtype`.
fn format(dtype: DType) -> &'static CStr { DTYPES.iter().find(|e| e.0 == dtype).expect("every dtype has a format code").1 }

/// The size and alignment of an item of `dtype`.
fn layout(dtype: DType) -> (usize, usize) { with_dtype!(dtype, T => (size_of::<T>(), align_of::<T>())) }

/// An integer type that a buffer can hold and no dtype matches, which [`read`] widens as it copies.
#[derive(Clone, Copy)]
enum Widen { I8, U16, U32, U64 }

impl Widen {
    /// The dtype that holds this type's values: int16 for signed bytes, and int64 for the unsigned types.
    fn dtype(self) -> DType { DType::Integer(if matches!(self, Widen::I8) { Width::I16 } else { Width::I64 }) }
    /// The `n` items of `view`, held as this type, copied into a buffer of [`Self::dtype`]. A `u64` above `i64`'s range raises `OverflowError`.
    unsafe fn copied(self, view: &ffi::Py_buffer, n: usize, swapped: Option<usize>, py: Python<'_>) -> PyResult<Buffer> {
        unsafe {
            Ok(match self {
                Widen::I8 => i16::buffer(copied::<i8>(view, n, swapped, py)?.into_iter().map(i16::from).collect()),
                Widen::U16 => i64::buffer(copied::<u16>(view, n, swapped, py)?.into_iter().map(i64::from).collect()),
                Widen::U32 => i64::buffer(copied::<u32>(view, n, swapped, py)?.into_iter().map(i64::from).collect()), Widen::U64 => i64::buffer(
                    copied::<u64>(view, n, swapped, py)?.into_iter().map(i64::try_from).collect::<Result<Vec<_>, _>>()
                        .map_err(|_| PyOverflowError::new_err("an unsigned integer exceeds int64; convert explicitly"))?)
            })
        }
    }
}

/// How a buffer stores its items, read once from its PEP 3118 format and item size.
struct Source {
    /// The dtype the items read as.
    dtype: DType,
    /// For items in the other byte order from this machine's, the length of each run of bytes to reverse: the item size, or half of it for a complex number's two parts.
    swapped: Option<usize>,
    /// The type of integer items that no dtype matches, which reading widens.
    widen: Option<Widen>,
}

/// How a buffer with PEP 3118 format `format` and items of `item_size` bytes stores them. A format with no supported dtype, or whose dtype has another size, is DOMAIN.
fn source(format: &CStr, item_size: usize) -> Result<Source, Error> {
    let unsupported = || Error::new(Kind::Domain, format!("unsupported buffer format `{}`", format.to_string_lossy()));
    let (code, swap) = match format.to_bytes() {
        [order @ (b'@' | b'=' | b'<' | b'>' | b'!'), code @ ..] => (code, match order {
            b'<' => cfg!(target_endian = "big"), b'>' | b'!' => cfg!(target_endian = "little"), _ => false
        }),
        code => (code, false),
    };
    let swapped = swap.then_some(if code.starts_with(b"Z") { item_size / 2 } else { item_size });
    if let Some(&(_, _, widen)) = WIDENED.iter().find(|e| e.0 == code && e.1 == item_size) {
        return Ok(Source { dtype: widen.dtype(), swapped, widen: Some(widen) });
    }
    let dtype = match code {
        b"l" => DType::Integer(if item_size == 4 { Width::I32 } else { Width::I64 }),
        code => DTYPES.iter().map(|e| (e.1.to_bytes(), e.0)).chain(ALSO).find(|e| e.0 == code).ok_or_else(unsupported)?.1,
    };
    if layout(dtype).0 != item_size { return Err(unsupported()); }
    Ok(Source { dtype, swapped, widen: None })
}

/// A buffer that an exporter lends, released when this drops.
struct Held(Box<ffi::Py_buffer>);

impl Drop for Held { fn drop(&mut self) { Python::attach(|_| unsafe { ffi::PyBuffer_Release(&mut *self.0) }) } }

/// The items of an object that exports a buffer: the exporter's own, or a copy in C order.
pub struct Read { shape: Vec<usize>, len: usize, dtype: DType, items: Items }

enum Items { Lent(Held), Owned(Buffer) }

impl Read {
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn dtype(&self) -> DType { self.dtype }
    /// Whether reading made a compact copy that [`Self::into_buffer`] can move without copying again.
    pub fn is_owned(&self) -> bool { matches!(self.items, Items::Owned(_)) }
    pub fn view(&self) -> View<'_> {
        match &self.items {
            Items::Owned(b) => b.view(),
            // SAFETY: `read` lends only contiguous items, aligned for the dtype and valid for it, which `Held` keeps alive.
            Items::Lent(h) => with_dtype!(self.dtype, T => T::view(unsafe { items(h.0.buf, self.len) })),
        }
    }
    /// The items as an owned buffer. Only lent items are copied.
    pub fn into_buffer(self) -> Buffer { if let Items::Owned(b) = self.items { return b; } Buffer::copied(self.view(), 0..self.len) }
}

/// The `n` items of type `T` at `p`, borrowed.
unsafe fn items<'a, T>(p: *const c_void, n: usize) -> &'a [T] { if n == 0 { &[] } else { unsafe { slice::from_raw_parts(p.cast(), n) } } }

/// The `n` items of `view` in C order, copied by CPython into a new vector, with each run of `swapped` bytes reversed where it's given. Every bit pattern must be a valid `T`.
unsafe fn copied<T>(view: &ffi::Py_buffer, n: usize, swapped: Option<usize>, py: Python<'_>) -> PyResult<Vec<T>> {
    let mut out = Vec::<T>::with_capacity(n);
    if n > 0 && unsafe { ffi::PyBuffer_ToContiguous(out.as_mut_ptr().cast(), view, (n * size_of::<T>()) as ffi::Py_ssize_t, b'C' as c_char) } < 0 {
        return Err(PyErr::fetch(py));
    }
    unsafe { out.set_len(n) };
    if let Some(width) = swapped {
        let bytes = unsafe { slice::from_raw_parts_mut(out.as_mut_ptr().cast::<u8>(), n * size_of::<T>()) };
        for item in bytes.chunks_exact_mut(width) { item.reverse(); }
    }
    Ok(out)
}

/// A Domain error for a code in `codes` that isn't a Unicode scalar value.
fn characters(codes: &[u32]) -> PyResult<()> {
    match codes.iter().find(|&&c| char::from_u32(c).is_none()) {
        Some(c) => Err(py_err(Error::new(Kind::Domain, format!("{c:#x} isn't a Unicode character")))), None => Ok(())
    }
}

/// The `n` items of `view`, stored as `src` describes, as an owned buffer. A Boolean byte other than 0 or 1 reads as true.
unsafe fn owned(src: &Source, view: &ffi::Py_buffer, n: usize, py: Python<'_>) -> PyResult<Buffer> {
    let swapped = src.swapped;
    unsafe {
        if let Some(widen) = src.widen { return widen.copied(view, n, swapped, py); }
        Ok(match src.dtype {
            DType::Boolean => bool::buffer(copied::<u8>(view, n, swapped, py)?.into_iter().map(|b| b != 0).collect()),
            DType::Character => {
                let codes = copied::<u32>(view, n, swapped, py)?;
                characters(&codes)?;
                char::buffer(codes.into_iter().filter_map(char::from_u32).collect())
            }
            // Every bit pattern is a valid number, which the arms above leave.
            numbers => with_dtype!(numbers, T => T::buffer(copied::<T>(view, n, swapped, py)?)),
        })
    }
}

/// The items of `obj` if it exports a buffer, or `None` if it doesn't. Contiguous items at an address aligned for their dtype are borrowed. Others are copied into C order, as are Booleans with a byte other than 0 or 1, which read as true. A character that isn't a Unicode scalar value raises `ValueError`.
pub fn read(obj: &Bound<'_, PyAny>) -> PyResult<Option<Read>> {
    let py = obj.py();
    if unsafe { ffi::PyObject_CheckBuffer(obj.as_ptr()) } == 0 { return Ok(None); }
    let mut buffer = Box::new(ffi::Py_buffer::new());
    if unsafe { ffi::PyObject_GetBuffer(obj.as_ptr(), &mut *buffer, ffi::PyBUF_FULL_RO) } < 0 { return Err(PyErr::fetch(py)); }
    let held = Held(buffer);
    let v = &*held.0;
    let format = if v.format.is_null() { c"B" } else { unsafe { CStr::from_ptr(v.format) } };
    let src = source(format, v.itemsize as usize).map_err(py_err)?;
    let dtype = src.dtype;
    let shape: Vec<usize> = match v.ndim {
        0 => vec![], _ if v.shape.is_null() => vec![v.len as usize / v.itemsize as usize],
        ndim => unsafe { items::<ffi::Py_ssize_t>(v.shape.cast(), ndim as usize) }.iter().map(|&n| n as usize).collect()
    };
    let n = element_count(&shape).map_err(py_err)?;
    let lend = src.swapped.is_none() && src.widen.is_none() && unsafe { ffi::PyBuffer_IsContiguous(v, b'C' as c_char) } == 1
        && v.buf as usize % layout(dtype).1 == 0 && match dtype {
            DType::Boolean => unsafe { items::<u8>(v.buf, n) }.iter().all(|&b| b <= 1),
            DType::Character => characters(unsafe { items::<u32>(v.buf, n) }).map(|_| true)?, _ => true
        };
    let items = if lend { Items::Lent(held) } else { Items::Owned(unsafe { owned(&src, &held.0, n, py) }?) };
    Ok(Some(Read { shape, len: n, dtype, items }))
}

/// Reads an ordinary array, including its keys and names when its frontend supplies `__rustymath_array__`. No numerical conversion is implicit in that handoff.
pub fn read_array(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Read, rustymath::layout::Layout)>> {
    use rustymath::layout::Layout;
    use std::sync::Arc;
    if obj.hasattr("__rustymath_array__")? {
        let (source, keys, names): (Bound<'_, PyAny>, Vec<Option<Vec<Option<String>>>>, Vec<Option<String>>) =
            obj.call_method0("__rustymath_array__")?.extract()?;
        let read = read(&source)?.ok_or_else(|| PyTypeError::new_err("array interchange requires compact items"))?;
        let keys = axis_keys(keys)?;
        let layout =
            Layout::from(read.shape()).with_keys(keys).map_err(py_err)?.with_names(names.into_iter().map(|n| n.map(Arc::from)).collect()).map_err(py_err)?;
        return Ok(Some((read, layout)));
    }
    let Some(read) = read(obj)? else { return Ok(None) };
    let layout = Layout::from(read.shape());
    Ok(Some((read, layout)))
}

/// The `shape` and C-order byte strides that [`export`] takes, for items of `dtype` laid out in `shape`. Keep both alive for each export.
pub fn geometry(dtype: DType, shape: &[usize]) -> (Box<[isize]>, Box<[isize]>) {
    let strides: Vec<isize> = rustymath::layout::strides(shape).iter().map(|&s| s as isize).collect();
    strided_geometry(dtype, shape, &strides)
}

/// The `shape` and byte strides that [`export`] takes, for items of `dtype` reached through signed item `strides`, one for each axis of `shape`, as [`Mapping::affine`](rustymath::agreement::Mapping::affine) gives them. Keep both alive for each export.
fn strided_geometry(dtype: DType, shape: &[usize], strides: &[isize]) -> (Box<[isize]>, Box<[isize]>) {
    let size = layout(dtype).0 as isize;
    (shape.iter().map(|&n| n as isize).collect(), strides.iter().map(|&s| s * size).collect())
}

/// Fills `view` for a class's `__getbuffer__` with the `len` logical items of `dtype` under `shape` and signed byte `strides`. `data` points to the first logical item. Keep storage and geometry valid until release. Only the parts that `flags` requests are filled. Unsupported write or contiguity requests raise `BufferError`.
///
/// # Safety
/// `view` must be the `Py_buffer` that CPython passed to `__getbuffer__`, and every address reached by the geometry must hold a valid item of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn export(
    view: *mut ffi::Py_buffer, flags: c_int, owner: &Bound<'_, PyAny>, dtype: DType, data: *mut u8, len: usize, shape: &[isize], strides: &[isize],
    readonly: bool) -> PyResult<()> {
    if view.is_null() { return Err(PyBufferError::new_err("no Py_buffer to fill")); }
    if readonly && flags & ffi::PyBUF_WRITABLE != 0 { return Err(PyBufferError::new_err("the array is read-only")); }
    let size = layout(dtype).0;
    let nd = flags & ffi::PyBUF_ND == ffi::PyBUF_ND;
    let mut v = ffi::Py_buffer::new();
    v.buf = data.cast();
    v.len = (len * size) as ffi::Py_ssize_t;
    v.itemsize = size as ffi::Py_ssize_t;
    v.readonly = c_int::from(readonly);
    v.format = if flags & ffi::PyBUF_FORMAT != 0 { format(dtype).as_ptr().cast_mut() } else { ptr::null_mut() };
    v.ndim = shape.len() as c_int;
    v.shape = shape.as_ptr().cast_mut();
    v.strides = strides.as_ptr().cast_mut();
    let strided = flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES;
    for (flag, order) in [(ffi::PyBUF_C_CONTIGUOUS, b'C'), (ffi::PyBUF_F_CONTIGUOUS, b'F'), (ffi::PyBUF_ANY_CONTIGUOUS, b'A')] {
        if (flags & flag == flag || order == b'C' && !strided) && unsafe { ffi::PyBuffer_IsContiguous(&v, order as c_char) } == 0 {
            return Err(PyBufferError::new_err("the array doesn't have the requested contiguous layout"));
        }
    }
    v.obj = owner.clone().into_ptr();
    if !nd { v.ndim = 1; }
    if !nd || shape.is_empty() { v.shape = ptr::null_mut(); }
    if !strided || shape.is_empty() { v.strides = ptr::null_mut(); }
    v.suboffsets = ptr::null_mut();
    v.internal = ptr::null_mut();
    unsafe { *view = v; }
    Ok(())
}

/// What a [`lend`] export holds: the value that owns the items, and the export's shape and byte strides.
#[pyclass(frozen)]
struct Lent { _keep: Box<dyn std::any::Any + Send + Sync>, shape: Box<[isize]>, strides: Box<[isize]> }

/// Fills `view` for a class's `__getbuffer__`, read-only, with the items of `dtype` under `shape` that `keep` owns. The first is at `first(&keep)`, and the rest follow at signed item `strides` from it, or in C order where `strides` is `None`. The export holds `keep`, so the items stay valid until release, whatever then happens to the object that exported them.
///
/// # Safety
/// `view` must be the `Py_buffer` that CPython passed to `__getbuffer__`, and every address that the geometry reaches from `first(&keep)` must hold a valid item of `dtype` that `keep` owns.
#[allow(clippy::too_many_arguments)]
pub unsafe fn lend<K: Send + Sync + 'static>(
    view: *mut ffi::Py_buffer, flags: c_int, py: Python<'_>, keep: K, first: impl FnOnce(&K) -> *const u8, dtype: DType, shape: &[usize],
    strides: Option<&[isize]>) -> PyResult<()> {
    let keep = Box::new(keep);
    let data = first(&keep);
    let (dims, steps) = match strides { Some(s) => strided_geometry(dtype, shape, s), None => geometry(dtype, shape) };
    let lent = Bound::new(py, Lent { _keep: keep, shape: dims, strides: steps })?;
    let this = lent.get();
    unsafe { export(view, flags, lent.as_any(), dtype, data.cast_mut(), shape.iter().product(), &this.shape, &this.strides, true) }
}
