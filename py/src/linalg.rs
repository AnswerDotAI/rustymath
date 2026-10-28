//! rustymath's `linalg` module as Python functions, which `python/rustymath/linalg.py` wraps. Each matrix function converts its arguments, runs the method of the same name of rustymath's [`Stacked`] with the GIL released, and gives its results. `Stacked` holds the rules for stacks of matrices: how their frames and keys pair, the shape errors and the threads. Ctrl-C stops a call between rounds of cells.
//!
//! Each matrix function's docstring is [`rustymath::stacked_doc`]'s entry for its name, which the `Stacked` method of the same name also reads. Exact matrices compute as floats of the width that [`numpy_float`] gives them. `python/rustymath/linalg.py`'s docstring states the dtype rules for Python users.
use pyo3::prelude::*;
use rustymath::agreement::{Frames, Mapping};
use rustymath::array::{Array as Core, ArrayView};
use rustymath::buffer::{Buffer, DType, Element, FloatWidth, View, Width};
use rustymath::error::{Error, Kind};
use rustymath::gather::Steps;
use rustymath::layout::Layout;
use rustymath::linalg::{self, Inner, Mapped, Orientation, PlusTimes, Stacked};
use rustymath::operation::numpy_float;
use rustymath_buffer::py_err;

use crate::array::give;
use crate::convert::{Call, convert, one, released};
use crate::functions::{NativeFunction, mathish_at, native_binary};

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
struct Operands<'a> { args: [View<'a>; 2], frames: &'a Frames, size: [usize; 3], cell: &'a Layout, name: &'a str }

impl Operands<'_> {
    /// `pair`'s products, at the dtype that it computes in (`linalg::Product::run`), or `None` where it has no kernel for the arguments' dtype. Integer results, counts included, are int64. float16 matrices give float16 results, and other results keep their kernels' dtype. An Overflow error for an integer result that doesn't fit int64.
    fn run(&self, pair: linalg::Product) -> PyResult<Option<Core>> {
        let [x, y] = self.args;
        let Some(products) = pair.run([x, y], self.frames, self.size) else { return Ok(None) };
        let (buffer, ok) = products.map_err(py_err)?;
        if !ok { return Err(py_err(Error::new(Kind::Overflow, format!("an integer result of {} doesn't fit int64", self.name)))); }
        self.finish(buffer).map(Some)
    }

    fn finish(&self, buffer: Buffer) -> PyResult<Core> {
        let x = self.args[0];
        let half = DType::Float(FloatWidth::F16);
        let store = match buffer.dtype() { DType::Integer(_) => I64, DType::Float(_) if x.dtype() == half => half, d => d };
        linalg::stored(buffer, &self.frames.layout, self.cell.clone(), store).map_err(py_err)
    }

    fn generic(&self, (f, fm): NativeFunction<'_>, (g, gm): NativeFunction<'_>) -> PyResult<Core> {
        let (f, g) = (native_binary(f)?, native_binary(g)?);
        let dtype = self.args[0].dtype();
        let compute = match dtype { DType::Integer(_) => I64, DType::Float(FloatWidth::F16) => DType::Float(FloatWidth::F32), d => d };
        let wide = if dtype != compute {
            Some([Buffer::cast(self.args[0], compute).map_err(py_err)?, Buffer::cast(self.args[1], compute).map_err(py_err)?])
        } else { None };
        let [x, y] = wide.as_ref().map_or(self.args, |a| a.each_ref().map(Buffer::view));
        let [n, k, m] = self.size;
        if k == 0 {
            let empty = rustymath::agreement::Agreement { layout: Layout::EMPTY, len: 0, left: Mapping::Single, right: Mapping::Single };
            let terms = (g.apply)(x, y, &empty, gm)?;
            return self.finish((f.fold)(terms.view(), [self.frames.count * n, 0, m], None, fm)?);
        }
        let (frames, size) = (self.frames, self.size);
        let parts = linalg::cells(frames, |i, _: &mut ()| {
            linalg::inner_with(
                x.slice(frames.items(0, i, n * k)), y.slice(frames.items(1, i, k * m)), size, |x, y, a| (g.apply)(x, y, a, gm), |x, d, s| (f.fold)(x, d, s, fm))
        }).map_err(py_err)?;
        let mut parts = parts.into_iter();
        let mut out = parts.next().unwrap_or_else(|| {
            linalg::inner_with(x.slice(0..0), y.slice(0..0), [0, k, 0], |x, y, a| (g.apply)(x, y, a, gm), |x, d, s| (f.fold)(x, d, s, fm))
        })?;
        for part in parts {
            let part = part?;
            assert!(out.extend_from(part.view(), 0..part.len()), "native inner-product cells have one dtype");
        }
        self.finish(out)
    }
}

/// A product operand's mapping as one entry for each axis of its frame and two for its matrix. A vector's matrix takes a length-1 axis: before its own axis on the left of a product, and after it on the right.
struct Cells { base: usize, frame: Vec<Steps>, matrix: Vec<Steps> }

impl Cells {
    /// The split of `view`'s mapping, where its last `side` axes form its matrix. `None` where the mapping doesn't split by axis.
    fn new(view: ArrayView<'_>, side: usize, left: bool) -> Option<Self> {
        let shape = view.layout().shape();
        let (base, mut frame) = view.mapping().entries(shape)?;
        let mut matrix = frame.split_off(shape.len() - side);
        if side == 1 { matrix.insert(usize::from(!left), Steps::along(1, 0)); }
        Some(Self { base, frame, matrix })
    }
    /// The mapping of the matrix at position `i` of the frame. `None` where the frame reads a fill there.
    fn at(&self, i: usize) -> Option<Mapping> { Some(Mapping::Steps { base: Steps::reach(self.base, &self.frame, i)?, steps: self.matrix.clone().into() }) }
}

/// `x +.× y` of 32- or 64-bit floats or complex numbers, where a view holds an operand out of order but gemm can read both operands where they lie, as strided or transposed matrices. `Inner::inner_mapped` multiplies them without copying either. `None` where both operands are contiguous, for other dtypes, and where an operand would need a copy, as reversed rows do. A copy and one gemm cost less than gemm over blocks read in turn.
fn mapped(views: [ArrayView<'_>; 2], sides: [usize; 2], frames: &Frames, size @ [n, k, m]: [usize; 3], cell: &Layout) -> PyResult<Option<Core>> {
    let dtype = views[0].dtype();
    if frames.count == 0 || views.iter().all(|v| v.contiguous().is_some()) || views[1].dtype() != dtype
        || !matches!(dtype, DType::Float(FloatWidth::F32 | FloatWidth::F64) | DType::Complex(_)) { return Ok(None); }
    let Some(cells) = [0, 1].map(|j| Cells::new(views[j], sides[j], j == 0)).into_iter().collect::<Option<Vec<_>>>() else { return Ok(None) };
    rustymath::with_inexact!(dtype, T => {
        let [x, y] = views.map(|v| T::of(v.storage()).expect("a view's storage holds its dtype"));
        let in_place = |data: &[T], cells: &Cells, shape| cells.at(0).is_some_and(|mapping| Mapped { data, mapping: &mapping }.in_place(Orientation::Normal, shape));
        if !in_place(x, &cells[0], [n, k]) || !in_place(y, &cells[1], [k, m]) { return Ok(None); }
        let parts = linalg::cells(frames, |i, _: &mut ()| {
            let (a, b) = (cells[0].at(frames.cell(0, i))?, cells[1].at(frames.cell(1, i))?);
            Some(PlusTimes.inner_mapped(Mapped { data: x, mapping: &a }, Mapped { data: y, mapping: &b }, [Orientation::Normal; 2], size))
        }).map_err(py_err)?;
        let Some(parts) = parts.into_iter().collect::<Option<Vec<_>>>() else { return Ok(None) };
        Ok(Some(linalg::stored(T::buffer(parts.concat()), &frames.layout, cell.clone(), dtype).map_err(py_err)?))
    })
}

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

/// The product kernel for the native functions named `$names`, from rustymath's table of facts ([`facts!`](rustymath::facts)): a `boolean_products` row first where `$booleans` holds, then a `products` row.
macro_rules! known_product {
    (($names:expr, $booleans:expr) folds $folds:tt booleans $truths:tt comparisons $comparisons:tt characters $characters:tt
        products { $($kernel:ident: [$($p:tt)*] [$(($pf:literal, $pg:literal)),*];)* } boolean_products { $($truth:ident: [$($tp:tt)*] [$(($tf:literal, $tg:literal)),*];)* }) => {
        match $names { $($(($tf, $tg) if $booleans => Some(linalg::Product::$truth),)*)* $($(($pf, $pg) => Some(linalg::Product::$kernel),)*)* _ => None }
    };
}

/// [`product`] of two arrays that share one dtype, as a core array. It runs without the GIL, checking for Ctrl-C.
pub fn inner(f: NativeFunction<'_>, g: NativeFunction<'_>, [x1, x2]: [ArrayView<'_>; 2], name: &str) -> PyResult<Core> {
    let dtype = x1.dtype();
    // Counts compare exactly through their kernels. Floats and complex numbers within a tolerance compare through the generic route's `equal`.
    let exact = !g.1 || !mathish_at(Some(dtype));
    let named = rustymath::facts!(known_product, (f.0, g.0), (dtype == DType::Boolean)).filter(|pair| exact || !pair.compares_exactly());
    let (l1, l2) = (x1.layout().clone(), x2.layout().clone());
    let side = |l: &Layout| match l.shape().len() {
        0 => Err(py_err(Error::new(Kind::Rank, format!("{name} needs matrices or vectors, with at least one axis")))), 1 => Ok(1), _ => Ok(2)
    };
    let (c1, c2) = (side(&l1)?, side(&l2)?);
    let (s1, s2, r1, r2) = (l1.shape(), l2.shape(), l1.shape().len(), l2.shape().len());
    let (n, k) = if c1 == 1 { (1, s1[0]) } else { (s1[r1 - 2], s1[r1 - 1]) };
    let (j, m) = if c2 == 1 { (s2[0], 1) } else { (s2[r2 - 2], s2[r2 - 1]) };
    if k != j {
        return Err(py_err(Error::new(Kind::Length, format!("{name} needs as many columns on the left as rows on the right, not {k} and {j}"))));
    }
    let frames = Frames::new(&[(&l1, c1), (&l2, c2)]).map_err(py_err)?;
    let cell = l1.axes(r1 - c1..r1).contracted(&l2.axes(r2 - c2..r2));
    let keyless = l1.keys(r1 - 1).is_none() && l2.keys(r2 - c2).is_none();
    if named == Some(linalg::Product::PlusTimes) && keyless && let Some(r) = mapped([x1, x2], [c1, c2], &frames, [n, k, m], &cell)? { return Ok(r); }
    let x = x1.cast(dtype).map_err(py_err)?;
    let y = linalg::paired(x2.cast(dtype).map_err(py_err)?, &l2, r2 - c2, &l1, r1 - 1, name).map_err(py_err)?;
    let p = Operands { args: [x.view(), y.view()], frames: &frames, size: [n, k, m], cell: &cell, name };
    let Some(pair) = named else { return p.generic(f, g) };
    match p.run(pair)? { Some(r) => Ok(r), None => p.generic(f, g) }
}

/// The matrix product of `x1` and `x2`. A vector on the left acts as one row, and a vector on the right as one column, and the result leaves that axis out. Booleans give whether some `k` has both `x1[i, k]` and `x2[k, j]`. Integers compute in int64, and raise `OverflowError` for a result that doesn't fit.
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
