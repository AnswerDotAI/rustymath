//! The Python buffer protocol for rustymath's compact buffers, shared by the Python extensions of rustymath and basedpl. [`read`] reads any object that exports a buffer. It borrows contiguous aligned items, and copies others into C order. [`export`] fills a `Py_buffer` for a class's `__getbuffer__`. [`py_err`] turns a rustymath error into a Python exception.
use pyo3::exceptions::{PyBufferError, PyIndexError, PyOverflowError, PyValueError};
use pyo3::{IntoPyObjectExt, ffi, prelude::*, types::{PyComplex, PyList}};
use rustymath::buffer::{Buffer, ComplexWidth, DType, Element, FloatWidth, View, Width};
use rustymath::error::{Error, Kind};
use rustymath::{layout::element_count, with_dtype};
use std::ffi::{CStr, c_char, c_int, c_void};
use std::{mem::size_of, ptr, slice};

/// The Python exception for a rustymath error: `OverflowError` for Overflow, `IndexError` for Index, and `ValueError` for the other kinds.
pub fn py_err(e: Error) -> PyErr {
    let text = e.to_string();
    match e.kind { Kind::Overflow => PyOverflowError::new_err(text), Kind::Index => PyIndexError::new_err(text), _ => PyValueError::new_err(text) }
}

/// Item `i` of `view` as a Python `bool`, `int`, `float`, `complex` or one-character `str`.
pub fn item<'py>(py: Python<'py>, view: View<'_>, i: usize) -> PyResult<Bound<'py, PyAny>> {
    match view {
        View::Booleans(v) => v[i].into_bound_py_any(py),
        View::Integers(v) => v.get(i).into_bound_py_any(py),
        View::Floats(v) => v.get(i).into_bound_py_any(py),
        View::Complexes(v) => { let z = v.get(i); Ok(PyComplex::from_doubles(py, z.re, z.im).into_any()) }
        View::Characters(v) => v[i].into_bound_py_any(py),
    }
}

/// The items of `view` under `shape` as nested lists of Python scalars, or one scalar when `shape` is empty. It needs no NumPy, and reads every dtype, complex numbers and characters included.
pub fn tolist<'py>(py: Python<'py>, view: View<'_>, shape: &[usize]) -> PyResult<Bound<'py, PyAny>> { nested(py, shape, &|i| item(py, view, i)) }

/// The items that `item` gives for each position in row-major order, nested in lists by `shape`, or the item at position 0 when `shape` is empty.
pub fn nested<'py>(py: Python<'py>, shape: &[usize], item: &dyn Fn(usize) -> PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> { nest(py, shape, 0, item) }

fn nest<'py>(py: Python<'py>, shape: &[usize], start: usize, item: &dyn Fn(usize) -> PyResult<Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let [n, rest @ ..] = shape else { return item(start) };
    let cell: usize = rest.iter().product();
    let items = (0..*n).map(|k| nest(py, rest, start + k * cell, item)).collect::<PyResult<Vec<_>>>()?;
    Ok(PyList::new(py, items)?.into_any())
}
/// Each dtype, with the PEP 3118 format code that [`export`] writes for it and its name in Python.
const DTYPES: [(DType, &CStr, &str); 11] = [
    (DType::Boolean, c"?", "bool"), (DType::Integer(Width::U8), c"B", "u8"), (DType::Integer(Width::I16), c"h", "i16"),
    (DType::Integer(Width::I32), c"i", "i32"), (DType::Integer(Width::I64), c"q", "i64"), (DType::Float(FloatWidth::F16), c"e", "f16"),
    (DType::Float(FloatWidth::F32), c"f", "f32"), (DType::Float(FloatWidth::F64), c"d", "f64"), (DType::Complex(ComplexWidth::F32), c"Zf", "c64"),
    (DType::Complex(ComplexWidth::F64), c"Zd", "c128"), (DType::Character, c"w", "char"),
];

/// Format codes that [`read`] also accepts. `l`, C's `long`, has a size that varies by platform, so [`dtype`] reads it by its item size.
const ALSO: [(&[u8], DType); 3] = [(b"c", DType::Integer(Width::U8)), (b"n", DType::Integer(Width::I64)), (b"1w", DType::Character)];

fn entry(dtype: DType) -> &'static (DType, &'static CStr, &'static str) { DTYPES.iter().find(|e| e.0 == dtype).expect("every dtype has an entry") }

/// The name of `dtype` in Python, such as `"f32"`.
pub fn name(dtype: DType) -> &'static str { entry(dtype).2 }

/// The dtype that [`name`] calls `name`, or `None` for any other name.
pub fn named(name: &str) -> Option<DType> { DTYPES.iter().find(|e| e.2 == name).map(|e| e.0) }

/// The size and alignment of an item of `dtype`.
fn layout(dtype: DType) -> (usize, usize) { with_dtype!(dtype, T => (size_of::<T>(), align_of::<T>())) }

/// The dtype of a buffer whose items have the PEP 3118 format `format` and `item_size` bytes each. A Domain error for a format that no dtype has, or for non-native byte order.
pub fn dtype(format: &CStr, item_size: usize) -> Result<DType, Error> {
    let unsupported = || Error::new(Kind::Domain, format!("unsupported buffer format `{}`", format.to_string_lossy()));
    let code = match format.to_bytes() {
        [b'>' | b'!', ..] => return Err(Error::new(Kind::Domain, "rustymath reads only native byte order")),
        [b'@' | b'=' | b'<', code @ ..] => code,
        code => code,
    };
    let dtype = match code {
        b"l" => DType::Integer(if item_size == 4 { Width::I32 } else { Width::I64 }),
        code => DTYPES.iter().map(|e| (e.1.to_bytes(), e.0)).chain(ALSO).find(|e| e.0 == code).ok_or_else(unsupported)?.1,
    };
    if layout(dtype).0 != item_size { return Err(unsupported()); }
    Ok(dtype)
}

/// A buffer that an exporter lends, released when this drops.
struct Held(Box<ffi::Py_buffer>);

impl Drop for Held {
    fn drop(&mut self) { Python::attach(|_| unsafe { ffi::PyBuffer_Release(&mut *self.0) }) }
}

/// The items of an object that exports a buffer: the exporter's own, or a copy in C order.
pub struct Read { shape: Vec<usize>, len: usize, dtype: DType, items: Items }

enum Items { Lent(Held), Owned(Buffer) }

impl Read {
    pub fn shape(&self) -> &[usize] { &self.shape }
    pub fn dtype(&self) -> DType { self.dtype }
    pub fn view(&self) -> View<'_> {
        match &self.items {
            Items::Owned(b) => b.view(),
            // SAFETY: `read` lends only contiguous items, aligned for the dtype and valid for it, which `Held` keeps alive.
            Items::Lent(h) => with_dtype!(self.dtype, T => T::view(unsafe { items(h.0.buf, self.len) })),
        }
    }
    /// The items as an owned buffer. Only lent items are copied.
    pub fn into_buffer(self) -> Buffer {
        if let Items::Owned(b) = self.items { return b; }
        Buffer::copied(self.view(), 0..self.len)
    }
}

/// The `n` items of type `T` at `p`, borrowed.
unsafe fn items<'a, T>(p: *const c_void, n: usize) -> &'a [T] { if n == 0 { &[] } else { unsafe { slice::from_raw_parts(p.cast(), n) } } }

/// The `n` items of `view` in C order, copied by CPython into a new vector. Every bit pattern must be a valid `T`.
unsafe fn copied<T>(view: &ffi::Py_buffer, n: usize, py: Python<'_>) -> PyResult<Vec<T>> {
    let mut out = Vec::<T>::with_capacity(n);
    if n > 0 && unsafe { ffi::PyBuffer_ToContiguous(out.as_mut_ptr().cast(), view, (n * size_of::<T>()) as ffi::Py_ssize_t, b'C' as c_char) } < 0 {
        return Err(PyErr::fetch(py));
    }
    unsafe { out.set_len(n) };
    Ok(out)
}

/// A Domain error for a code in `codes` that isn't a Unicode scalar value.
fn characters(codes: &[u32]) -> PyResult<()> {
    match codes.iter().find(|&&c| char::from_u32(c).is_none()) {
        Some(c) => Err(py_err(Error::new(Kind::Domain, format!("{c:#x} isn't a Unicode character")))),
        None => Ok(()),
    }
}

/// The `n` items of `view` as an owned buffer of `dtype`. A Boolean byte other than 0 or 1 reads as true.
unsafe fn owned(dtype: DType, view: &ffi::Py_buffer, n: usize, py: Python<'_>) -> PyResult<Buffer> {
    unsafe {
        Ok(match dtype {
            DType::Boolean => bool::buffer(copied::<u8>(view, n, py)?.into_iter().map(|b| b != 0).collect()),
            DType::Character => {
                let codes = copied::<u32>(view, n, py)?;
                characters(&codes)?;
                char::buffer(codes.into_iter().filter_map(char::from_u32).collect())
            }
            // Every bit pattern is a valid number, which the arms above leave.
            numbers => with_dtype!(numbers, T => T::buffer(copied::<T>(view, n, py)?)),
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
    let dtype = dtype(format, v.itemsize as usize).map_err(py_err)?;
    let shape: Vec<usize> = match v.ndim {
        0 => vec![],
        _ if v.shape.is_null() => vec![v.len as usize / v.itemsize as usize],
        ndim => unsafe { items::<ffi::Py_ssize_t>(v.shape.cast(), ndim as usize) }.iter().map(|&n| n as usize).collect(),
    };
    let n = element_count(&shape).map_err(py_err)?;
    let lend = unsafe { ffi::PyBuffer_IsContiguous(v, b'C' as c_char) } == 1 && v.buf as usize % layout(dtype).1 == 0 && match dtype {
        DType::Boolean => unsafe { items::<u8>(v.buf, n) }.iter().all(|&b| b <= 1),
        DType::Character => characters(unsafe { items::<u32>(v.buf, n) }).map(|_| true)?,
        _ => true,
    };
    let items = if lend { Items::Lent(held) } else { Items::Owned(unsafe { owned(dtype, &held.0, n, py) }?) };
    Ok(Some(Read { shape, len: n, dtype, items }))
}

/// The `shape` and C-order byte strides that [`export`] takes, for items of `dtype` laid out in `shape`. A class keeps both while it can be exported.
pub fn geometry(dtype: DType, shape: &[usize]) -> (Box<[isize]>, Box<[isize]>) {
    let size = layout(dtype).0;
    (shape.iter().map(|&n| n as isize).collect(), rustymath::layout::strides(shape).iter().map(|&s| (s * size) as isize).collect())
}

/// Fills `view` for a class's `__getbuffer__` with the `len` items of `dtype` at `data`, under `shape` and the byte `strides`. `owner` must keep the items, `shape` and `strides` valid while it lives. Only the parts that `flags` requests are filled. A request to write to a read-only export, or for Fortran order of items in more than one dimension, raises `BufferError`.
///
/// # Safety
/// `view` must be the `Py_buffer` that CPython passed to `__getbuffer__`, and `data` must point to `len` items of `dtype`.
#[allow(clippy::too_many_arguments)]
pub unsafe fn export(
    view: *mut ffi::Py_buffer, flags: c_int, owner: &Bound<'_, PyAny>, dtype: DType, data: *mut u8, len: usize, shape: &[isize], strides: &[isize], readonly: bool,
) -> PyResult<()> {
    if view.is_null() { return Err(PyBufferError::new_err("no Py_buffer to fill")); }
    if readonly && flags & ffi::PyBUF_WRITABLE != 0 { return Err(PyBufferError::new_err("the array is read-only")); }
    if flags & ffi::PyBUF_F_CONTIGUOUS == ffi::PyBUF_F_CONTIGUOUS && shape.iter().filter(|&&n| n > 1).count() > 1 {
        return Err(PyBufferError::new_err("the array's items are in C order"));
    }
    let size = layout(dtype).0;
    let nd = flags & ffi::PyBUF_ND == ffi::PyBUF_ND;
    let v = unsafe { &mut *view };
    v.obj = owner.clone().into_ptr();
    v.buf = data.cast();
    v.len = (len * size) as ffi::Py_ssize_t;
    v.itemsize = size as ffi::Py_ssize_t;
    v.readonly = c_int::from(readonly);
    v.format = if flags & ffi::PyBUF_FORMAT != 0 { entry(dtype).1.as_ptr().cast_mut() } else { ptr::null_mut() };
    v.ndim = if nd { shape.len() as c_int } else { 1 };
    v.shape = if nd && !shape.is_empty() { shape.as_ptr().cast_mut() } else { ptr::null_mut() };
    v.strides = if flags & ffi::PyBUF_STRIDES == ffi::PyBUF_STRIDES && !shape.is_empty() { strides.as_ptr().cast_mut() } else { ptr::null_mut() };
    v.suboffsets = ptr::null_mut();
    v.internal = ptr::null_mut();
    Ok(())
}
