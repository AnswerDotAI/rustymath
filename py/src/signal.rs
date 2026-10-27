//! rustymath's `signal` module as Python functions, which `python/rustymath/signal.py` wraps.
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustymath::array::Array as Core;
use rustymath::buffer::{Buffer, DType, Element, FloatWidth};
use rustymath::error::{Error, Kind};
use rustymath::layout::{count, Layout};
use rustymath::linalg::PlusTimes;
use rustymath::signal::{self, Mode};
use rustymath::structure::Plan;
use rustymath_buffer::py_err;

use crate::array::{give, give_items};
use crate::convert::{convert, one, released};
use crate::functions::{missing, Axis, NativeFunction};

/// `buffer`, computed at `compute`, as the dtype `store`.
fn stored(buffer: Buffer, compute: DType, store: DType) -> PyResult<Buffer> {
    if store == compute { Ok(buffer) } else { Buffer::cast(buffer.view(), store).map_err(py_err) }
}

/// The correlation of `x` with `kernel` by the native binary functions named `f` and `g`, along `x`'s first `slide` axes, or its convolution where `flip` is true. `stride` gives one stride for each sliding axis, or one for all of them. Float and complex sums of products read the windows a block at a time. Other pairs and dtypes build the windows as a matrix and take its inner product with the kernel, as `linalg.product` does.
#[pyfunction]
#[pyo3(signature = (f, g, x, kernel, mode, slide, stride, flip))]
fn _correlate<'py>(py: Python<'py>, f: NativeFunction<'_>, g: NativeFunction<'_>, x: &Bound<'py, PyAny>, kernel: &Bound<'py, PyAny>, mode: &str, slide: Option<usize>, stride: Vec<usize>, flip: bool) -> PyResult<Bound<'py, PyAny>> {
    let name = if flip { "convolve" } else { "correlate" };
    let mode = match mode { "valid" => Mode::Valid, "full" => Mode::Full, "same" => Mode::Same, _ => return Err(PyValueError::new_err(format!("mode must be 'valid', 'full' or 'same', not {mode:?}"))) };
    let call = convert(&[x, kernel])?;
    let (lx, lk) = (call.args[0].layout(), call.args[1].layout());
    let slide = slide.unwrap_or(lx.shape().len());
    let stride = if let [s] = *stride.as_slice() { vec![s; slide] } else { stride };
    let c = signal::correlation(lx.shape(), lk.shape(), slide, mode, &stride, flip).map_err(py_err)?;
    let dtype = call.args[0].view().dtype();
    if (f.0, g.0) == ("add", "multiply") && matches!(dtype, DType::Float(_) | DType::Complex(_)) {
        let (compute, store) = dtype.inexact(FloatWidth::F64).ok_or_else(|| missing(name, dtype))?;
        return rustymath::with_inexact!(compute, T => {
            let (xs, ks) = (call.args[0].items::<T>()?, call.args[1].items::<T>()?);
            let buffer = stored(T::buffer(released(py, || signal::correlate(&PlusTimes, &xs, &ks, &c))), compute, store)?;
            give_items(py, Layout::from(c.shape.clone()), buffer, false)
        });
    }
    let (xi, ki) = (call.args[0].at(dtype)?, call.args[1].at(dtype)?);
    let product = released(py, || -> PyResult<Core> {
        let padded = c.padding.as_ref().map(|p| Core::gathered(xi.view(), p, None)).transpose().map_err(py_err)?;
        let source = match &padded { Some(p) => p.view().contiguous().expect("a gathered array is contiguous"), None => xi.view() };
        let matrix = |view, plan: &Plan, rows, cols| Core::gathered(view, &Plan { layout: Layout::from(vec![rows, cols]), steps: plan.steps.clone() }, None).map_err(py_err);
        let [positions, cells, outputs] = c.size;
        let (windows, kernel) = (matrix(source, &c.windows, positions, cells)?, matrix(ki.view(), &c.kernel, cells, outputs)?);
        crate::linalg::inner(f, g, [windows.view(), kernel.view()], name)
    })?;
    give(py, product.reshape(Layout::from(c.shape.clone())).map_err(py_err)?, false)
}

/// Each frame along `axis` of `frames` added back where its window was read, with the frames' items along the next axis: the transpose of windows that move `step` items at a time. Booleans and integers compute in the float that `DType::inexact` gives them, as linear algebra does.
#[pyfunction]
#[pyo3(signature = (frames, step=1, axis=Axis::Position(-2)))]
fn overlap_add<'py>(py: Python<'py>, frames: &Bound<'py, PyAny>, step: usize, axis: Axis) -> PyResult<Bound<'py, PyAny>> {
    if step == 0 { return Err(PyValueError::new_err("step must be positive")); }
    let x = one(frames)?;
    let layout = x.layout();
    let (a, shape) = (axis.of(&layout)?, layout.shape());
    if a + 1 >= shape.len() { return Err(py_err(Error::new(Kind::Rank, "overlap_add needs each frame's items along the axis after the frames"))); }
    let (n, w) = (shape[a], shape[a + 1]);
    let dims = [count(&shape[..a]), n, 1, w, count(&shape[a + 2..])];
    let len = if n == 0 { 0 } else { (n - 1) * step + w };
    let result: Vec<usize> = shape[..a].iter().copied().chain([len]).chain(shape[a + 2..].iter().copied()).collect();
    let dtype = x.view().dtype();
    let (compute, store) = dtype.inexact(FloatWidth::F64).ok_or_else(|| missing("overlap_add", dtype))?;
    rustymath::with_inexact!(compute, T => {
        let xs = x.items::<T>()?;
        let buffer = stored(T::buffer(released(py, || signal::overlap_add(&xs, dims, step, len))), compute, store)?;
        give_items(py, Layout::from(result), buffer, false)
    })
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [wrap_pyfunction!(_correlate, m)?, wrap_pyfunction!(overlap_add, m)?] { m.add_function(f)?; }
    Ok(())
}
