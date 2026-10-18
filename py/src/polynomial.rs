//! rustymath's `polynomial` module under the names of `numpy.polynomial.polynomial`. Each function converts its data arguments to one dtype, as `convert` does. Booleans, `u8` and `i16` compute as `f32`, wider integers as `f64`, and `float16` computes as `float32` and rounds its result back. Root finding computes in `complex128`, and gives `complex64` roots for `float16`, `float32` and `complex64` coefficients.
use crate::array::give;
use crate::convert::{convert, one};
use crate::functions::missing;
use crate::integer::radix;
use pyo3::prelude::*;
use rustymath::array::Array as Core;
use rustymath::buffer::{Buffer, DType, Element, FloatWidth};
use rustymath::error::{Error, Kind};
use rustymath::layout::Layout;
use rustymath::num_complex::Complex64;
use rustymath::polynomial;
use rustymath_buffer::py_err;

/// The dtype that `function` computes in for arguments of `dtype`, and the dtype of its result, as `DType::inexact` gives them.
fn working(function: &str, dtype: DType) -> PyResult<(DType, DType)> { dtype.inexact(FloatWidth::F64).ok_or_else(|| missing(function, dtype)) }

/// `data` as an array of `layout`, in the dtype `result`.
fn finished<T: Element>(layout: Layout, data: Vec<T>, result: DType) -> PyResult<Core> {
    let buffer = T::buffer(data);
    let buffer = if buffer.dtype() == result { buffer } else { Buffer::cast(buffer.view(), result).map_err(py_err)? };
    Core::new(layout, buffer).map_err(py_err)
}

/// A vector of `data`, in the dtype `result`.
fn vector_of<T: Element>(data: Vec<T>, result: DType) -> PyResult<Core> { finished(Layout::from([data.len()].as_slice()), data, result) }

/// A Rank error unless `layout` is a vector of `what`.
fn vector(layout: &Layout, what: &str) -> PyResult<()> {
    if layout.shape().len() == 1 { Ok(()) } else { Err(py_err(Error::new(Kind::Rank, format!("{what} must be a vector")))) }
}

/// A Rank error unless the leading coefficient, of `layout`, is scalar.
fn leading(layout: &Layout) -> PyResult<()> {
    if layout.shape().is_empty() { Ok(()) } else { Err(py_err(Error::new(Kind::Rank, "the leading coefficient must be scalar"))) }
}

/// The polynomial `c` at each item of `x`, by Horner's rule.
#[pyfunction]
pub fn polyval<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, c: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[x, c])?;
    let (xs, cs) = (&call.args[0], &call.args[1]);
    vector(&cs.layout(), "the coefficients")?;
    let (work, result) = working("polyval", xs.view().dtype())?;
    let core = match work {
        DType::Float(w) => rustymath::with_float_width!(w, F => {
            let (c, mut y) = (cs.items::<F>()?, xs.items::<F>()?.into_owned());
            polynomial::evaluate(&c, &mut y);
            finished(xs.layout(), y, result)?
        }),
        _ => rustymath::with_inexact!(work, T => {
            let c = cs.items::<T>()?;
            finished(xs.layout(), xs.items::<T>()?.iter().map(|x| polynomial::value(&c, x)).collect(), result)?
        }),
    };
    give(py, core, xs.scalar)
}

/// `m` times the product of `x - r` over the roots `r` at each item of `x`, without forming the coefficients.
#[pyfunction]
#[pyo3(signature = (x, roots, m=None))]
pub fn polyvalfromroots<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, roots: &Bound<'py, PyAny>, m: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let unit = 1i64.into_pyobject(py)?;
    let call = convert(&[x, roots, m.unwrap_or(unit.as_any())])?;
    let (xs, rs, ms) = (&call.args[0], &call.args[1], &call.args[2]);
    vector(&rs.layout(), "the roots")?;
    leading(&ms.layout())?;
    let (work, result) = working("polyvalfromroots", xs.view().dtype())?;
    let core = rustymath::with_inexact!(work, T => {
        let (roots, m) = (rs.items::<T>()?, ms.items::<T>()?[0]);
        finished(xs.layout(), xs.items::<T>()?.iter().map(|x| polynomial::factored(&m, &roots, x)).collect(), result)?
    });
    give(py, core, xs.scalar)
}

/// The roots of `c` as complex numbers, sorted by real part, then imaginary part, from the eigenvalues of its companion matrix. With real coefficients, a real root has an imaginary part of exactly 0. Where the eigenvalue solver doesn't converge, each root is NaN.
#[pyfunction]
pub fn polyroots<'py>(py: Python<'py>, c: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let cs = one(c)?;
    vector(&cs.layout(), "the coefficients")?;
    let dtype = cs.view().dtype();
    if dtype == DType::Character { return Err(missing("polyroots", dtype)); }
    let result = dtype.complex();
    let c = cs.items::<Complex64>()?;
    let roots = polynomial::roots(&c).unwrap_or_else(|| vec![Complex64::new(f64::NAN, f64::NAN); polynomial::trimmed(&c).len().saturating_sub(1)]);
    give(py, vector_of(roots, result)?, false)
}

/// The polynomial with roots `roots` and leading coefficient `m`.
#[pyfunction]
#[pyo3(signature = (roots, m=None))]
pub fn polyfromroots<'py>(py: Python<'py>, roots: &Bound<'py, PyAny>, m: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let unit = 1i64.into_pyobject(py)?;
    let call = convert(&[roots, m.unwrap_or(unit.as_any())])?;
    let (rs, ms) = (&call.args[0], &call.args[1]);
    vector(&rs.layout(), "the roots")?;
    leading(&ms.layout())?;
    let (work, result) = working("polyfromroots", rs.view().dtype())?;
    let core = rustymath::with_inexact!(work, T => vector_of(polynomial::from_roots(ms.items::<T>()?[0], &rs.items::<T>()?), result)?);
    give(py, core, false)
}

/// The derivative of `c`. A constant's derivative is empty, which is the zero polynomial.
#[pyfunction]
pub fn polyder<'py>(py: Python<'py>, c: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let cs = one(c)?;
    vector(&cs.layout(), "the coefficients")?;
    let (work, result) = working("polyder", cs.view().dtype())?;
    let core = rustymath::with_inexact!(work, T => vector_of(polynomial::derivative(&cs.items::<T>()?), result)?);
    give(py, core, false)
}

/// The value of each last-axis vector of `digits` in the mixed radix `bases`, by Horner's rule. The first base is unused.
#[pyfunction]
pub fn polydecode<'py>(py: Python<'py>, bases: &Bound<'py, PyAny>, digits: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[bases, digits])?;
    let (bs, ds) = (&call.args[0], &call.args[1]);
    let d = ds.layout();
    radix(bs.layout().shape(), d.shape())?;
    let (work, result) = working("decode", bs.view().dtype())?;
    let layout = d.axes(0..d.shape().len() - 1);
    let core = rustymath::with_inexact!(work, T => finished(layout, polynomial::decode(&bs.items::<T>()?, &ds.items::<T>()?), result)?);
    let reduced = core.layout().shape().is_empty();
    give(py, core, reduced)
}

/// The coefficients, in ascending order, of the polynomial of degree `deg` that fits the points by least squares. Where fewer than `deg + 1` distinct points fix it, it's the fit whose coefficients have the least norm.
#[pyfunction]
pub fn polyfit<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, y: &Bound<'py, PyAny>, deg: usize) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[x, y])?;
    let (xs, ys) = (&call.args[0], &call.args[1]);
    vector(&xs.layout(), "the x coordinates")?;
    if xs.layout().shape() != ys.layout().shape() { return Err(py_err(Error::new(Kind::Length, "polyfit needs one y for each x"))); }
    let (work, result) = working("polyfit", xs.view().dtype())?;
    let DType::Float(w) = work else { return Err(missing("polyfit", work)) };
    let core = rustymath::with_float_width!(w, F => vector_of(polynomial::fit(&xs.items::<F>()?, &ys.items::<F>()?, deg), result)?);
    give(py, core, false)
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(polyval, m)?, wrap_pyfunction!(polyvalfromroots, m)?, wrap_pyfunction!(polyroots, m)?, wrap_pyfunction!(polyfromroots, m)?,
        wrap_pyfunction!(polyder, m)?, wrap_pyfunction!(polydecode, m)?, wrap_pyfunction!(polyfit, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}
