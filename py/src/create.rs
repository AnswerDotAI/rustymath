//! Familiar constructors over rustymath's compact buffers and sequence generators.
use crate::{array::{Array, core}, convert};
use pyo3::{exceptions::PyValueError, prelude::*, types::{PyInt, PyTuple}};
use rustymath::{array::Array as Core, buffer::{Buffer, DType, Element, Source}, layout::{Layout, generated_len}, with_dtype, with_float_width, with_width};
use rustymath_buffer::{name, py_err};

/// A shape given as one length or a sequence of lengths, checked against the shared allocation limits.
pub fn shape(size: &Bound<'_, PyAny>) -> PyResult<Vec<usize>> {
    let shape = size.extract::<usize>().map(|n| vec![n]).or_else(|_| size.extract::<Vec<usize>>())?;
    generated_len(&shape).map_err(py_err)?;
    Ok(shape)
}

fn layout(size: &Bound<'_, PyAny>) -> PyResult<Layout> { Ok(Layout::from(shape(size)?)) }

fn cast(core: Core, dtype: Option<&str>) -> PyResult<Array> {
    let Some(dtype) = dtype else { return Ok(Array::new(core)) };
    let (layout, buffer) = core.into_compact();
    let dtype = convert::dtype(dtype)?;
    let buffer = if buffer.dtype() == dtype { buffer } else { Buffer::cast(buffer.view(), dtype).map_err(py_err)? };
    Ok(Array::new(Core::new(layout, buffer).map_err(py_err)?))
}

/// One fill primitive. `like` supplies layout and default dtype; `None` fills with the dtype's zero or space.
#[pyfunction]
#[pyo3(signature = (shape, value, dtype=None, *, like=None))]
fn _full(shape: &Bound<'_, PyAny>, value: &Bound<'_, PyAny>, dtype: Option<&str>, like: Option<&Bound<'_, PyAny>>) -> PyResult<Array> {
    let like = like.map(convert::one).transpose()?;
    let layout = match &like { Some(x) => x.layout(), None => layout(shape)? };
    let dtype = dtype.map(convert::dtype).transpose()?.or_else(|| like.as_ref().map(|x| x.view().dtype()));
    if value.is_none() {
        let dtype = dtype.unwrap_or(DType::Float(rustymath::buffer::FloatWidth::F64));
        return Ok(Array::new(with_dtype!(dtype, T => rustymath::create::full(layout, T::view(&[T::FILL]))).map_err(py_err)?));
    }
    let value = match dtype { Some(dtype) => convert::one_at(value, dtype)?, None => convert::one(value)? };
    Ok(Array::new(rustymath::create::full(layout, value.at(value.view().dtype())?.view()).map_err(py_err)?))
}

/// Values separated by `step`, including `start` and excluding `stop`. With only one bound, `start` is zero.
#[pyfunction]
#[pyo3(signature = (start, stop=None, step=None, dtype=None))]
fn arange(start: &Bound<'_, PyAny>, stop: Option<&Bound<'_, PyAny>>, step: Option<&Bound<'_, PyAny>>, dtype: Option<&str>) -> PyResult<Array> {
    let integer = |x: &Bound<'_, PyAny>| x.is_instance_of::<PyInt>() || x.extract::<i64>().is_ok();
    let integers = integer(start) && stop.is_none_or(integer) && step.is_none_or(integer);
    let core = if integers {
        let (a, b) = match stop { Some(stop) => (start.extract()?, stop.extract()?), None => (0, start.extract()?) };
        let values = rustymath::create::arange_int(a, b, step.map(|x| x.extract()).transpose()?.unwrap_or(1)).map_err(py_err)?;
        let n = values.len(); core(values, &[n])?
    } else {
        let (a, b) = match stop { Some(stop) => (start.extract()?, stop.extract()?), None => (0.0, start.extract()?) };
        let values = rustymath::create::arange(a, b, step.map(|x| x.extract()).transpose()?.unwrap_or(1.0)).map_err(py_err)?;
        let n = values.len(); core(values, &[n])?
    };
    cast(core, dtype)
}

/// `num` evenly spaced values between the bounds, including `stop` when `endpoint` is true.
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true, dtype=None))]
fn linspace(start: f64, stop: f64, num: usize, endpoint: bool, dtype: Option<&str>) -> PyResult<Array> {
    cast(core(rustymath::create::linspace(start, stop, num, endpoint).map_err(py_err)?, &[num])?, dtype)
}

/// `base` raised to `num` evenly spaced exponents between the bounds.
#[pyfunction]
#[pyo3(signature = (start, stop, num=50, endpoint=true, base=10.0, dtype=None))]
fn logspace(start: f64, stop: f64, num: usize, endpoint: bool, base: f64, dtype: Option<&str>) -> PyResult<Array> {
    cast(core(rustymath::create::logspace(start, stop, num, endpoint, base).map_err(py_err)?, &[num])?, dtype)
}

/// The precision and finite range of a float dtype, or of a complex dtype's real parts.
#[pyfunction]
fn _finfo(dtype: &str) -> PyResult<(usize, f64, f64, f64, f64)> {
    let DType::Float(width) = convert::dtype(dtype)?.real() else { return Err(PyValueError::new_err("finfo needs a float or complex dtype")) };
    let (bits, eps, tiny, max) = with_float_width!(width, F => (std::mem::size_of::<F>() * 8, Source::<f64>::read(F::EPSILON),
        Source::<f64>::read(F::MIN_POSITIVE), Source::<f64>::read(F::MAX)));
    Ok((bits, eps, tiny, -max, max))
}

/// The finite range of an integer dtype.
#[pyfunction]
fn _iinfo(dtype: &str) -> PyResult<(usize, i64, i64)> {
    let DType::Integer(width) = convert::dtype(dtype)? else { return Err(PyValueError::new_err("iinfo needs an integer dtype")) };
    let (bits, min, max) = with_width!(width, I => (std::mem::size_of::<I>() * 8, i64::from(I::MIN), i64::from(I::MAX)));
    Ok((bits, min, max))
}

/// Whether a dtype belongs to `kind`: bool, integer, floating, complex, number or character, or is one exact dtype. Booleans count as numbers, as promotion treats them.
#[pyfunction]
fn issubdtype(dtype: &str, kind: &str) -> PyResult<bool> {
    let d = convert::dtype(dtype)?;
    Ok(match kind {
        "integer" => matches!(d, DType::Integer(_)), "floating" => matches!(d, DType::Float(_)), "complex" => matches!(d, DType::Complex(_)),
        "number" => d != DType::Character, "character" => d == DType::Character, _ => d == convert::dtype(kind)?,
    })
}

/// The dtype to which all the named dtypes promote through the shared numerical promotion table.
#[pyfunction]
#[pyo3(signature = (*dtypes))]
fn result_type(dtypes: &Bound<'_, PyTuple>) -> PyResult<&'static str> {
    let dtypes: Vec<String> = dtypes.extract()?;
    let mut dtypes = dtypes.iter().map(|d| convert::dtype(d));
    let mut dtype = dtypes.next().ok_or_else(|| PyValueError::new_err("result_type needs at least one dtype"))??;
    for other in dtypes { dtype = dtype.common(other?).ok_or_else(|| PyValueError::new_err("characters and numbers can't share a dtype"))?; }
    Ok(name(dtype))
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(_full, m)?, wrap_pyfunction!(arange, m)?, wrap_pyfunction!(linspace, m)?, wrap_pyfunction!(logspace, m)?,
        wrap_pyfunction!(_finfo, m)?, wrap_pyfunction!(_iinfo, m)?, wrap_pyfunction!(issubdtype, m)?, wrap_pyfunction!(result_type, m)?] {
        m.add_function(f)?;
    }
    Ok(())
}
