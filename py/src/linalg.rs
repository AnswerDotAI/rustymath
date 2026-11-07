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
use crate::convert::{convert, one, released};
use crate::functions::{NativeFunction, native_binary};

/// `f`'s result on the views of `args`, run with the GIL released through the [`Stacked`] whose exact numbers compute as floats of the width that rustymath's own policy gives them ([`numpy_float`]).
fn on_stacked<'py, const N: usize, R: Send>(
    py: Python<'py>, args: [&Bound<'py, PyAny>; N], f: impl for<'c> FnOnce(Stacked, [ArrayView<'c>; N]) -> Result<R, Error> + Send) -> PyResult<R> {
    let call = convert(&args)?;
    let (s, views) = (Stacked::new(numpy_float(call.args[0].view().dtype())), std::array::from_fn(|k| call.args[k].view()));
    released(py, || f(s, views)).map_err(py_err)
}

/// `core` as a Python result: a Python number where it has no axes.
fn result(py: Python<'_>, core: Core) -> PyResult<Bound<'_, PyAny>> { let scalar = core.layout().shape().is_empty(); give(py, core, scalar) }

/// The int64 dtype of the matrix products' integer results.
const I64: DType = DType::Integer(Width::I64);

#[doc = rustymath::stacked_doc!(solve)]
#[pyfunction]
fn solve<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    result(py, on_stacked(py, [a, b], |s, [a, b]| s.solve(a, b))?)
}

#[doc = rustymath::stacked_doc!(lstsq)]
#[pyfunction]
fn lstsq<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (x, sv) = on_stacked(py, [a, b], |s, [a, b]| s.lstsq(a, b))?;
    Ok((result(py, x)?, result(py, sv)?))
}

#[doc = rustymath::stacked_doc!(pinv)]
#[pyfunction]
fn pinv<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.pinv(a))?) }

#[doc = rustymath::stacked_doc!(det)]
#[pyfunction]
fn det<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.det(a))?) }

#[doc = rustymath::stacked_doc!(slogdet)]
#[pyfunction]
fn slogdet<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (sign, log) = on_stacked(py, [a], |s, [a]| s.slogdet(a))?;
    Ok((result(py, sign)?, result(py, log)?))
}

#[doc = rustymath::stacked_doc!(lu)]
#[pyfunction]
fn lu<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (factors, p) = on_stacked(py, [a], |s, [a]| s.lu(a))?;
    Ok((result(py, factors)?, result(py, p)?))
}

#[doc = rustymath::stacked_doc!(lu_solve)]
#[pyfunction]
fn lu_solve<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let positions = one(p)?;
    let p = positions.view();
    result(py, on_stacked(py, [factors, b], |s, [factors, b]| s.lu_solve(factors, p, b))?)
}

#[doc = rustymath::stacked_doc!(lu_inverse)]
#[pyfunction]
fn lu_inverse<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let positions = one(p)?;
    let p = positions.view();
    result(py, on_stacked(py, [factors], |s, [factors]| s.lu_inverse(factors, p))?)
}

#[doc = rustymath::stacked_doc!(lu_det)]
#[pyfunction]
fn lu_det<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let positions = one(p)?;
    let p = positions.view();
    result(py, on_stacked(py, [factors], |s, [factors]| s.lu_det(factors, p))?)
}

#[doc = rustymath::stacked_doc!(lu_slogdet)]
#[pyfunction]
fn lu_slogdet<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>, p: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let positions = one(p)?;
    let p = positions.view();
    let (sign, log) = on_stacked(py, [factors], |s, [factors]| s.lu_slogdet(factors, p))?;
    Ok((result(py, sign)?, result(py, log)?))
}

#[doc = rustymath::stacked_doc!(lu_lower)]
#[pyfunction]
fn lu_lower<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [factors], |s, [f]| s.lu_lower(f))?) }

#[doc = rustymath::stacked_doc!(lu_upper)]
#[pyfunction]
fn lu_upper<'py>(py: Python<'py>, factors: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [factors], |s, [f]| s.lu_upper(f))?) }

#[doc = rustymath::stacked_doc!(cholesky)]
#[pyfunction]
fn cholesky<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.cholesky(a))?) }

#[doc = rustymath::stacked_doc!(qr)]
#[pyfunction]
#[pyo3(signature = (a, complete=false))]
fn qr<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, complete: bool) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (q, r) = on_stacked(py, [a], |s, [a]| s.qr(a, complete))?;
    Ok((result(py, q)?, result(py, r)?))
}

#[doc = rustymath::stacked_doc!(svd)]
#[pyfunction]
#[pyo3(signature = (a, rank=None))]
fn svd<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, rank: Option<usize>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (u, sv, v) = on_stacked(py, [a], |s, [a]| s.svd(a, rank))?;
    Ok((result(py, u)?, result(py, sv)?, result(py, v)?))
}

#[doc = rustymath::stacked_doc!(svd_solve)]
#[pyfunction]
fn svd_solve<'py>(py: Python<'py>, u: &Bound<'py, PyAny>, s: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let values = one(s)?;
    let s = values.view();
    result(py, on_stacked(py, [u, v, b], |st, [u, v, b]| st.svd_solve(u, s, v, b))?)
}

#[doc = rustymath::stacked_doc!(svd_pinv)]
#[pyfunction]
fn svd_pinv<'py>(py: Python<'py>, u: &Bound<'py, PyAny>, s: &Bound<'py, PyAny>, v: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let values = one(s)?;
    let s = values.view();
    result(py, on_stacked(py, [u, v], |st, [u, v]| st.svd_pinv(u, s, v))?)
}

#[doc = rustymath::stacked_doc!(svdvals)]
#[pyfunction]
fn svdvals<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.svdvals(a))?) }

#[doc = rustymath::stacked_doc!(matrix_rank)]
#[pyfunction]
fn matrix_rank<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.matrix_rank(a))?) }

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
    result(py, on_stacked(py, [a], |s, [a]| s.norm(a, ord))?)
}

#[doc = rustymath::stacked_doc!(cond)]
#[pyfunction]
#[pyo3(signature = (a, p=None))]
fn cond<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, p: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let p = order(p, linalg::Norm::Two)?;
    result(py, on_stacked(py, [a], |s, [a]| s.cond(a, p))?)
}

#[doc = rustymath::stacked_doc!(eig)]
#[pyfunction]
fn eig<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (w, v) = on_stacked(py, [a], |s, [a]| s.eig(a))?;
    Ok((result(py, w)?, result(py, v)?))
}

#[doc = rustymath::stacked_doc!(eigvals)]
#[pyfunction]
fn eigvals<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.eigvals(a))?) }

#[doc = rustymath::stacked_doc!(eigh)]
#[pyfunction]
fn eigh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (w, v) = on_stacked(py, [a], |s, [a]| s.eigh(a))?;
    Ok((result(py, w)?, result(py, v)?))
}

#[doc = rustymath::stacked_doc!(eigvalsh)]
#[pyfunction]
fn eigvalsh<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.eigvalsh(a))?) }

#[doc = rustymath::stacked_doc!(permanent)]
#[pyfunction]
fn permanent<'py>(py: Python<'py>, a: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { result(py, on_stacked(py, [a], |s, [a]| s.permanent(a))?) }

#[doc = rustymath::stacked_doc!(echelon)]
#[pyfunction]
#[pyo3(signature = (a, columns=None))]
fn echelon<'py>(py: Python<'py>, a: &Bound<'py, PyAny>, columns: Option<usize>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let (rows, pivots, determinant) = on_stacked(py, [a], |s, [a]| s.echelon(a, columns))?;
    Ok((result(py, rows)?, result(py, pivots)?, result(py, determinant)?))
}

/// The inner product of native binary functions `f` and `g`, each with its Mathish choice, on vectors, matrices or broadcast stacks. Recognized pairs use specialized kernels; other pairs apply `g` in bounded blocks and fold `f` from the right. `name` names the product in errors.
#[pyfunction]
#[pyo3(signature = (f, g, x1, x2, name=None))]
pub fn product<'py>(
    py: Python<'py>, f: NativeFunction<'_>, g: NativeFunction<'_>, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>, name: Option<&str>
) -> PyResult<Bound<'py, PyAny>> { products(py, [x1, x2], |_| [f, g], name.unwrap_or("inner")) }

/// [`inner`] of `x1` and `x2`, converted to one dtype, by the formulas `f.g` that `pair` gives for that dtype, with the GIL released, as a Python result.
fn products<'py, 'f>(
    py: Python<'py>, [x1, x2]: [&Bound<'py, PyAny>; 2], pair: impl FnOnce(DType) -> [NativeFunction<'f>; 2], name: &str) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[x1, x2])?;
    let views = [call.args[0].view(), call.args[1].view()];
    let [f, g] = pair(views[0].dtype());
    result(py, released(py, || inner(f, g, views, name))?)
}

/// [`product`] of two arrays that share one dtype, as a core array.
pub fn inner(f: NativeFunction<'_>, g: NativeFunction<'_>, [left, right]: [ArrayView<'_>; 2], name: &str) -> PyResult<Core> {
    let named = |e: Error| py_err(Error::new(e.kind, format!("{name}: {}", e.message)));
    let plan = linalg::ProductPlan::matrices([left.layout(), right.layout()], 0).map_err(named)?;
    let dtype = left.dtype();
    let compute = match dtype { DType::Integer(_) => I64, DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), d => d };
    let map = (native_binary(g.0)?.kernel)(compute, g.1)?;
    let fold = (native_binary(f.0)?.kernel)(map.result_dtype(), f.1)?;
    if plan.contracted_len() == 0 && fold.fold().is_none() { return Err(named(Error::new(Kind::Length, "no identity for an empty contraction"))); }
    let total = map.product_dtype(&fold, &plan);
    let store = match total { DType::Integer(_) => I64, DType::Float(_) if dtype == DType::Float(FloatWidth::F16) => dtype, d => d };
    let mut output = Buffer::with_capacity(store, 0);
    rustymath::operation::product(&map, &fold, [left.read(), right.read()], &plan, crate::functions::settings(g.1), &mut output).map_err(named)?;
    Core::new(plan.layout, output).map_err(named)
}

/// The matrix product of `x1` and `x2`. A vector on the left acts as one row, and a vector on the right as one column, and the result leaves that axis out. Booleans give whether some `k` has both `x1[i, k]` and `x2[k, j]`. Integers compute in int64 and wrap at its range, as BPL's `+.×` does.
#[pyfunction]
pub fn matmul<'py>(py: Python<'py>, x1: &Bound<'py, PyAny>, x2: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    // NumPy's Boolean matrix product is `∨.∧`, where `+.×` on Booleans counts.
    let pair = |dtype: DType| if dtype == DType::Boolean { [("logical_or", false), ("logical_and", false)] } else { [("add", false), ("multiply", false)] };
    products(py, [x1, x2], pair, "matmul")
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
