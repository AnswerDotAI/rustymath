//! rustymath's `linalg` module as Python functions, which `python/rustymath/linalg.py` wraps. Each matrix function converts its arguments, runs the method of the same name of rustymath's [`Stacked`] with the GIL released, and gives its results. `Stacked` holds the rules for stacks of matrices: how their frames and keys pair, the shape errors and the threads. Ctrl-C stops a call between rounds of cells.
//!
//! Each matrix function's docstring is [`rustymath::stacked_doc`]'s entry for its name, which the `Stacked` method of the same name also reads. Exact matrices compute as floats of the width that [`numpy_float`] gives them. `python/rustymath/linalg.py`'s docstring states the dtype rules for Python users.
use pyo3::prelude::*;
use rustymath::array::{Array as Core, ArrayView};
use rustymath::buffer::{Buffer, DType, FloatWidth, Width};
use rustymath::error::{Error, Kind};
use rustymath::linalg::{self, Stacked};
use rustymath::operation::numpy_float;
use rustymath_buffer::py_err;

use crate::array::give;
use crate::convert::{Call, convert, one, released};
use crate::functions::{NativeFunction, native_binary};

/// The [`Stacked`] for `call`'s arguments, where exact numbers compute as floats of the width that rustymath's own policy gives them ([`numpy_float`]).
fn stacked(call: &Call) -> Stacked { Stacked::new(numpy_float(call.args[0].view().dtype())) }

/// The views of `call`'s first `N` arguments.
fn views<'c, const N: usize>(call: &'c Call) -> [ArrayView<'c>; N] { std::array::from_fn(|k| call.args[k].view()) }

/// `core` as a Python result: a Python number where it has no axes.
fn result(py: Python<'_>, core: Core) -> PyResult<Bound<'_, PyAny>> { let scalar = core.layout().shape().is_empty(); give(py, core, scalar) }

/// The int64 dtype of the matrix products' integer results.
const I64: DType = DType::Integer(Width::I64);

#[doc = rustymath::stacked_doc!(solve)]
#[pyfunction]
fn solve<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a, b])?;
    let (s, [a, b]) = (stacked(&call), views(&call));
    result(py, released(py, || s.solve(a, b)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(lstsq)]
#[pyfunction]
fn lstsq<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a, b])?;
    let (s, [a, b]) = (stacked(&call), views(&call));
    let (x, sv) = released(py, || s.lstsq(a, b)).map_err(py_err)?;
    Ok((result(py, x)?, result(py, sv)?))
}

#[doc = rustymath::stacked_doc!(pinv)]
#[pyfunction]
fn pinv<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.pinv(a)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(det)]
#[pyfunction]
fn det<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.det(a)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(slogdet)]
#[pyfunction]
fn slogdet<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    let (sign, log) = released(py, || s.slogdet(a)).map_err(py_err)?;
    Ok((result(py, sign)?, result(py, log)?))
}

#[doc = rustymath::stacked_doc!(lu)]
#[pyfunction]
fn lu<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    let (factors, p) = released(py, || s.lu(a)).map_err(py_err)?;
    Ok((result(py, factors)?, result(py, p)?))
}

#[doc = rustymath::stacked_doc!(lu_solve)]
#[pyfunction]
fn lu_solve<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (call, positions) = (convert(&[factors, b])?, one(p)?);
    let (s, [factors, b], p) = (stacked(&call), views(&call), positions.view());
    result(py, released(py, || s.lu_solve(factors, p, b)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(lu_inverse)]
#[pyfunction]
fn lu_inverse<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (call, positions) = (convert(&[factors])?, one(p)?);
    let (s, [factors], p) = (stacked(&call), views(&call), positions.view());
    result(py, released(py, || s.lu_inverse(factors, p)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(lu_det)]
#[pyfunction]
fn lu_det<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (call, positions) = (convert(&[factors])?, one(p)?);
    let (s, [factors], p) = (stacked(&call), views(&call), positions.view());
    result(py, released(py, || s.lu_det(factors, p)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(lu_slogdet)]
#[pyfunction]
fn lu_slogdet<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (call, positions) = (convert(&[factors])?, one(p)?);
    let (s, [factors], p) = (stacked(&call), views(&call), positions.view());
    let (sign, log) = released(py, || s.lu_slogdet(factors, p)).map_err(py_err)?;
    Ok((result(py, sign)?, result(py, log)?))
}

#[doc = rustymath::stacked_doc!(lu_lower)]
#[pyfunction]
fn lu_lower<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[factors])?;
    let (s, [factors]) = (stacked(&call), views(&call));
    result(py, released(py, || s.lu_lower(factors)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(lu_upper)]
#[pyfunction]
fn lu_upper<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[factors])?;
    let (s, [factors]) = (stacked(&call), views(&call));
    result(py, released(py, || s.lu_upper(factors)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(cholesky)]
#[pyfunction]
fn cholesky<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.cholesky(a)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(qr)]
#[pyfunction]
#[pyo3(signature = (a, complete=false))]
fn qr<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, complete: bool) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    let (q, r) = released(py, || s.qr(a, complete)).map_err(py_err)?;
    Ok((result(py, q)?, result(py, r)?))
}

#[doc = rustymath::stacked_doc!(svd)]
#[pyfunction]
#[pyo3(signature = (a, rank=None))]
fn svd<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, rank: Option<usize>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    let (u, sv, v) = released(py, || s.svd(a, rank)).map_err(py_err)?;
    Ok((result(py, u)?, result(py, sv)?, result(py, v)?))
}

#[doc = rustymath::stacked_doc!(svd_solve)]
#[pyfunction]
fn svd_solve<'py>(py: Python<'py>, u: &Bound<'py, PyAny>, s: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (call, values) = (convert(&[u, v, b])?, one(s)?);
    let (st, [u, v, b], s) = (stacked(&call), views(&call), values.view());
    result(py, released(py, || st.svd_solve(u, s, v, b)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(svd_pinv)]
#[pyfunction]
fn svd_pinv<'py>(py: Python<'py>, u: &Bound<'py, PyAny>, s: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (call, values) = (convert(&[u, v])?, one(s)?);
    let (st, [u, v], s) = (stacked(&call), views(&call), values.view());
    result(py, released(py, || st.svd_pinv(u, s, v)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(svdvals)]
#[pyfunction]
fn svdvals<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.svdvals(a)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(matrix_rank)]
#[pyfunction]
fn matrix_rank<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.matrix_rank(a)).map_err(py_err)?)
}

/// The matrix norm that NumPy's `ord` value names: 1, 2, inf or "fro", or `default` where it's None.
fn order(ord: Option<&Bound<'_, PyAny>>, default: linalg::Norm) -> PyResult<linalg::Norm> {
    let Some(ord) = ord else { return Ok(default) };
    if ord.extract::<String>().is_ok_and(|s| s == "fro") { return Ok(linalg::Norm::Frobenius); }
    match ord.extract::<f64>() {
        Ok(x) if x == 1.0 => Ok(linalg::Norm::One), Ok(x) if x == 2.0 => Ok(linalg::Norm::Two), Ok(x) if x == f64::INFINITY => Ok(linalg::Norm::Inf),
        _ => Err(pyo3::exceptions::PyValueError::new_err("a matrix norm's order must be 1, 2, inf or \"fro\""))
    }
}

#[doc = rustymath::stacked_doc!(norm)]
#[pyfunction]
#[pyo3(signature = (a, ord=None))]
fn norm<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, ord: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let ord = order(ord, linalg::Norm::Frobenius)?;
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.norm(a, ord)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(cond)]
#[pyfunction]
#[pyo3(signature = (a, p=None))]
fn cond<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, p: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let p = order(p, linalg::Norm::Two)?;
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.cond(a, p)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(eig)]
#[pyfunction]
fn eig<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    let (w, v) = released(py, || s.eig(a)).map_err(py_err)?;
    Ok((result(py, w)?, result(py, v)?))
}

#[doc = rustymath::stacked_doc!(eigvals)]
#[pyfunction]
fn eigvals<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.eigvals(a)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(eigh)]
#[pyfunction]
fn eigh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    let (w, v) = released(py, || s.eigh(a)).map_err(py_err)?;
    Ok((result(py, w)?, result(py, v)?))
}

#[doc = rustymath::stacked_doc!(eigvalsh)]
#[pyfunction]
fn eigvalsh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.eigvalsh(a)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(permanent)]
#[pyfunction]
fn permanent<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    result(py, released(py, || s.permanent(a)).map_err(py_err)?)
}

#[doc = rustymath::stacked_doc!(echelon)]
#[pyfunction]
#[pyo3(signature = (a, columns=None))]
fn echelon<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, columns: Option<usize>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let call = convert(&[a])?;
    let (s, [a]) = (stacked(&call), views(&call));
    let (rows, pivots, determinant) = released(py, || s.echelon(a, columns)).map_err(py_err)?;
    Ok((result(py, rows)?, result(py, pivots)?, result(py, determinant)?))
}

/// A matrix product's arguments: their items, with the right one's contracted axis in the left one's key order, their frames and sizes `[n, k, m]`, the result's cell axes, and the name its errors use.

/// The inner product of native binary functions `f` and `g`, each with its Mathish choice, on vectors, matrices or broadcast stacks. Recognized pairs use specialized kernels; other pairs apply `g` in bounded blocks and fold `f` from the right. `name` names the product in errors.
#[pyfunction]
#[pyo3(signature = (f, g, x1, x2, name=None))]
pub fn product<'py>(
    py: Python<'py>, f: NativeFunction<'_>, g: NativeFunction<'_>, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>, name: Option<&str>
) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[x1, x2])?;
    let views = [call.args[0].view(), call.args[1].view()];
    result(py, released(py, || inner(f, g, views, name.unwrap_or("inner")))?)
}

/// [`product`] of two arrays that share one dtype, as a core array.
pub fn inner(f: NativeFunction<'_>, g: NativeFunction<'_>, [left, right]: [ArrayView<'_>; 2], name: &str) -> PyResult<Core> {
    let named = |e: Error| py_err(Error::new(e.kind, format!("{name}: {}", e.message)));
    let plan = linalg::ProductPlan::matrices([left.layout(), right.layout()], 0).map_err(named)?;
    let dtype = left.dtype();
    let compute = match dtype { DType::Integer(_) => I64, DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), d => d };
    let map = (native_binary(g.0)?.kernel)(compute, g.1)?;
    let fold = (native_binary(f.0)?.kernel)(map.result_dtype(), f.1)?;
    let shape = plan.iteration.shape();
    let len = rustymath::layout::count(&shape[shape.len() - plan.contracted..]);
    if len == 0 && fold.fold().is_none() { return Err(named(Error::new(Kind::Length, "no identity for an empty contraction"))); }
    let native = map.product_dtype(&fold, dtype);
    let total = native.unwrap_or_else(|| fold.fold_dtype(map.result_dtype(), len, None));
    let store = match total { DType::Integer(_) => I64, DType::Float(_) if dtype == DType::Float(FloatWidth::F16) => dtype, d => d };
    let mut output = Buffer::with_capacity(store, plan.layout.count());
    rustymath::operation::product(&map, &fold, [left.read(), right.read()], &plan, Default::default(), &mut output).map_err(named)?;
    Core::new(plan.layout, output).map_err(named)
}

/// The matrix product of `x1` and `x2`. A vector on the left acts as one row, and a vector on the right as one column, and the result leaves that axis out. Booleans give whether some `k` has both `x1[i, k]` and `x2[k, j]`. Integers compute in int64 and wrap at its range, as BPL's `+.×` does.
#[pyfunction]
pub fn matmul<'py>(py: Python<'py>, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    product(py, ("add", false), ("multiply", false), x1, x2, Some("matmul"))
}

/// Adds the linear algebra functions to the submodule `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(solve, m)?, wrap_pyfunction!(lstsq, m)?, wrap_pyfunction!(pinv, m)?, wrap_pyfunction!(det, m)?, wrap_pyfunction!(slogdet, m)?,
        wrap_pyfunction!(lu, m)?, wrap_pyfunction!(lu_solve, m)?, wrap_pyfunction!(lu_inverse, m)?, wrap_pyfunction!(lu_det, m)?,
        wrap_pyfunction!(lu_slogdet, m)?, wrap_pyfunction!(lu_lower, m)?, wrap_pyfunction!(lu_upper, m)?, wrap_pyfunction!(cholesky, m)?,
        wrap_pyfunction!(qr, m)?, wrap_pyfunction!(svd, m)?, wrap_pyfunction!(svd_solve, m)?, wrap_pyfunction!(svd_pinv, m)?, wrap_pyfunction!(svdvals, m)?,
        wrap_pyfunction!(matrix_rank, m)?, wrap_pyfunction!(norm, m)?, wrap_pyfunction!(cond, m)?, wrap_pyfunction!(eig, m)?, wrap_pyfunction!(eigvals, m)?,
        wrap_pyfunction!(eigh, m)?, wrap_pyfunction!(eigvalsh, m)?, wrap_pyfunction!(permanent, m)?, wrap_pyfunction!(echelon, m)?,
        wrap_pyfunction!(product, m)?, wrap_pyfunction!(matmul, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}
