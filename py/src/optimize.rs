//! rustymath's `optimize` module on Python functions. Each call to the function, for one point or a batch, first checks for
//! Ctrl-C. An exception from the function stops the search and comes back to the caller.
use numpy::{Element, PyArray1, PyArrayMethods, PyReadonlyArray1};
use pyo3::{exceptions::PyValueError, intern, prelude::*};
use rustymath::{optimize::{self as opt, Bounds, Method, Objective, Options}, Float, Native};
use std::marker::PhantomData;

/// A Python function to minimize, and its gradient if it has one. With `vectorized`, `fun` takes points as the columns of a 2-D
/// array and gives a value for each, as SciPy's `differential_evolution` calls it.
struct Callable<'py, F> {
    fun: Bound<'py, PyAny>,
    jac: Option<Bound<'py, PyAny>>,
    vectorized: bool,
    float: PhantomData<F>,
}

/// `values` as floats, after checking there's one for each slot of `out`.
fn read<F: Float>(values: Bound<'_, PyAny>, out: &mut [F], what: &str) -> PyResult<()> {
    let values: Vec<f64> = values.extract()?;
    if values.len() != out.len() { return Err(PyValueError::new_err(format!("{what} gave {} values where {} were needed", values.len(), out.len()))) }
    for (o, v) in out.iter_mut().zip(values) { *o = F::narrow(v) }
    Ok(())
}

impl<'py, F: Native + Element> Objective<F> for Callable<'py, F> {
    type Error = PyErr;
    fn value(&mut self, x: &[F]) -> PyResult<F> {
        let mut y = [F::zero()];
        self.values(x, &mut y)?;
        Ok(y[0])
    }
    fn values(&mut self, xs: &[F], out: &mut [F]) -> PyResult<()> {
        let py = self.fun.py();
        py.check_signals()?;
        let n = xs.len() / out.len();
        if self.vectorized {
            let columns = PyArray1::from_slice(py, xs).reshape([out.len(), n])?.getattr(intern!(py, "T"))?;
            return read(self.fun.call1((columns,))?, out, "fun")
        }
        for (o, x) in out.iter_mut().zip(xs.chunks(n)) {
            *o = F::narrow(self.fun.call1((PyArray1::from_slice(py, x),))?.extract::<f64>()?);
        }
        Ok(())
    }
    fn has_gradient(&self) -> bool { self.jac.is_some() }
    fn gradient(&mut self, x: &[F], g: &mut [F]) -> PyResult<()> {
        let jac = self.jac.as_ref().expect("minimize asks only an objective with a gradient for it");
        read(jac.call1((PyArray1::from_slice(jac.py(), x),))?, g, "jac")
    }
}

/// `minimize` at the float type `F` of `x0` and `limits`.
#[allow(clippy::too_many_arguments)]
fn minimize_at<'py, F: Native + Element>(
    fun: Bound<'py, PyAny>, jac: Option<Bound<'py, PyAny>>, vectorized: bool, x0: &Bound<'py, PyAny>, limits: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    method: Option<Method>, tol: Option<f64>, maxfev: Option<usize>, population: Option<usize>, seed: u64,
) -> PyResult<(Bound<'py, PyAny>, f64, usize, bool)> {
    let x0 = x0.extract::<PyReadonlyArray1<F>>()?;
    let limits = limits.map(|(l, u)| Ok::<_, PyErr>((l.extract::<PyReadonlyArray1<F>>()?, u.extract::<PyReadonlyArray1<F>>()?))).transpose()?;
    let bounds = match &limits { Some((l, u)) => Some(Bounds { lower: l.as_slice()?, upper: u.as_slice()? }), None => None };
    let options = Options { method, bounds, tolerance: tol.map(F::narrow), evaluations: maxfev, population, seed };
    let mut objective = Callable { fun, jac, vectorized, float: PhantomData };
    let start = x0.as_slice()?;
    options.check(start.len(), objective.has_gradient()).map_err(PyValueError::new_err)?;
    let found = opt::minimize(&mut objective, start, &options)?;
    Ok((PyArray1::from_vec(x0.py(), found.x).into_any(), found.value.wide(), found.evaluations, found.converged))
}

/// The minimum of `fun` from `x0`, as the point, its value, the number of evaluations and whether the method's stopping test passed.
/// `rustymath.optimize.minimize` converts SciPy's arguments to these: `x0` and the `(lower, upper)` pair in `limits` are contiguous
/// float32 or float64 vectors of one type, which the search computes in, and `population` counts differential evolution's members.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn minimize<'py>(
    fun: Bound<'py, PyAny>, x0: Bound<'py, PyAny>, method: Option<&str>, jac: Option<Bound<'py, PyAny>>,
    limits: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>, tol: Option<f64>, maxfev: Option<usize>, population: Option<usize>, seed: u64,
    vectorized: bool,
) -> PyResult<(Bound<'py, PyAny>, f64, usize, bool)> {
    let method = method.map(str::parse::<Method>).transpose().map_err(PyValueError::new_err)?;
    if x0.cast::<PyArray1<f32>>().is_ok() { minimize_at::<f32>(fun, jac, vectorized, &x0, limits, method, tol, maxfev, population, seed) }
    else { minimize_at::<f64>(fun, jac, vectorized, &x0, limits, method, tol, maxfev, population, seed) }
}

/// A zero of `f` between `a` and `b`, by Brent's method, as the point, the number of evaluations and whether the bracket shrank
/// within `rtol` before `maxfev` evaluations.
#[pyfunction]
#[pyo3(signature = (f, a, b, rtol=None, maxfev=None))]
fn root_scalar(py: Python<'_>, f: Bound<'_, PyAny>, a: f64, b: f64, rtol: Option<f64>, maxfev: Option<usize>) -> PyResult<(f64, usize, bool)> {
    let found = opt::root(|x: f64| { py.check_signals()?; f.call1((x,))?.extract::<f64>() }, [a, b], rtol, maxfev)?;
    Ok((found.x, found.evaluations, found.converged))
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimize, m)?)?;
    m.add_function(wrap_pyfunction!(root_scalar, m)?)
}
