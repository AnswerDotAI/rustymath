//! rustymath's `optimize` module on Python functions. Each call to a function, for one point or a batch, first checks for Ctrl-C.
//! An exception from the function stops the search and comes back to the caller.
use numpy::{Element, PyArray1, PyArrayDyn, PyArrayMethods, PyReadonlyArray1, PyReadonlyArrayDyn};
use pyo3::{exceptions::PyValueError, intern, prelude::*};
use rustymath::{linalg::Factor, optimize::{self as opt, Bounds, Mapping, Method, Objective, Options, Task}, Native};
use std::marker::PhantomData;

/// A Python function to minimize, and its gradient if it has one. With `vectorized`, `fun` takes points as the columns of a 2-D
/// array and gives a value for each, as SciPy's `differential_evolution` calls it.
struct Callable<'py, F> {
    fun: Bound<'py, PyAny>,
    jac: Option<Bound<'py, PyAny>>,
    vectorized: bool,
    float: PhantomData<F>,
}

/// What `f` gives for `arg`, after checking for Ctrl-C, as an array of floats `F` that one call of `numpy.asarray` makes. A result
/// that's already such an array isn't copied.
fn give<'py, F: Element>(f: &Bound<'py, PyAny>, arg: Bound<'py, PyAny>) -> PyResult<PyReadonlyArrayDyn<'py, F>> {
    let py = f.py();
    py.check_signals()?;
    let numpy = py.import(intern!(py, "numpy"))?;
    Ok(numpy.call_method1(intern!(py, "asarray"), (f.call1((arg,))?, F::get_dtype(py)))?.cast_into::<PyArrayDyn<F>>()?.readonly())
}

/// `values`, read into `out` in row-major order, after checking there's one for each slot of `out`.
fn read<F: Element + Copy>(values: PyReadonlyArrayDyn<'_, F>, out: &mut [F], what: &str) -> PyResult<()> {
    let values = values.as_array();
    if values.len() != out.len() { return Err(PyValueError::new_err(format!("{what} gave {} values where {} were needed", values.len(), out.len()))) }
    for (o, &v) in out.iter_mut().zip(values) { *o = v }
    Ok(())
}

/// What `f` gives for the vector `x`, read into `out`.
fn call<F: Native + Element>(f: &Bound<'_, PyAny>, x: &[F], out: &mut [F], what: &str) -> PyResult<()> {
    read(give(f, PyArray1::from_slice(f.py(), x).into_any())?, out, what)
}

impl<'py, F: Native + Element> Objective<F> for Callable<'py, F> {
    type Error = PyErr;
    /// The value at `x`. A vectorized `fun` gets `x` as a one-column 2-D array, as it gets every point.
    fn value(&mut self, x: &[F]) -> PyResult<F> {
        if self.vectorized {
            let mut y = [F::zero()];
            self.values(x, &mut y)?;
            return Ok(y[0]);
        }
        let py = self.fun.py();
        py.check_signals()?;
        Ok(F::narrow(self.fun.call1((PyArray1::from_slice(py, x),))?.extract::<f64>()?))
    }
    fn batch(&self) -> bool { self.vectorized }
    fn values(&mut self, xs: &[F], out: &mut [F]) -> PyResult<()> {
        let py = self.fun.py();
        let columns = PyArray1::from_slice(py, xs).reshape([out.len(), xs.len() / out.len()])?.getattr(intern!(py, "T"))?;
        read(give(&self.fun, columns)?, out, "fun")
    }
    fn has_gradient(&self) -> bool { self.jac.is_some() }
    fn gradient(&mut self, x: &[F], g: &mut [F]) -> PyResult<()> {
        call(self.jac.as_ref().expect("minimize asks only an objective with a gradient for it"), x, g, "jac")
    }
}

/// Python functions that give a vector for a vector: `fun` the values, `jac`, where given, the Jacobian flattened by rows,
/// `rmatvec`, where given, the products with a linear `fun`'s transpose, and `msolve`, where given, a preconditioner's products.
struct Vectors<'py, F> {
    fun: Bound<'py, PyAny>,
    jac: Option<Bound<'py, PyAny>>,
    rmatvec: Option<Bound<'py, PyAny>>,
    msolve: Option<Bound<'py, PyAny>>,
    float: PhantomData<F>,
}

impl<'py, F: Native + Element> Mapping<F> for Vectors<'py, F> {
    type Error = PyErr;
    fn value(&mut self, x: &[F], out: &mut [F]) -> PyResult<()> { call(&self.fun, x, out, "fun") }
    fn has_vjp(&self) -> bool { self.rmatvec.is_some() }
    fn vjp(&mut self, u: &[F], _: &[F], out: &mut [F]) -> PyResult<()> {
        call(self.rmatvec.as_ref().expect("solve_linear asks only a mapping with a transpose for it"), u, out, "rmatvec")
    }
    fn jacobian(&mut self, x: &[F], j: &mut [F]) -> PyResult<bool>
    where
        F: Native,
    {
        let Some(jac) = &self.jac else { return Ok(false) };
        call(jac, x, j, "jac")?;
        Ok(true)
    }
    fn has_preconditioner(&self) -> bool { self.msolve.is_some() }
    fn precondition(&mut self, v: &[F], out: &mut [F]) -> PyResult<()> {
        call(self.msolve.as_ref().expect("solve_linear asks only a mapping with a preconditioner for it"), v, out, "M")
    }
}

/// What a search found, as the binding returns it: the point, the value there, the number of evaluations and whether the method's
/// stopping test passed.
type Outcome<'py> = (Bound<'py, PyAny>, f64, usize, bool);

/// `found` as an [`Outcome`].
fn outcome<'py, F: Native + Element>(py: Python<'py>, found: opt::Found<Vec<F>, F>) -> Outcome<'py> {
    (PyArray1::from_vec(py, found.x).into_any(), found.value.wide(), found.evaluations, found.converged)
}

/// `minimize` at the float type `F` of `x0` and `limits`.
#[allow(clippy::too_many_arguments)]
fn minimize_at<'py, F: Native + Element>(
    fun: Bound<'py, PyAny>, jac: Option<Bound<'py, PyAny>>, vectorized: bool, x0: &Bound<'py, PyAny>, limits: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    method: Option<Method>, tol: Option<f64>, maxfev: Option<usize>, population: Option<usize>, seed: u64,
) -> PyResult<Outcome<'py>> {
    let x0 = x0.extract::<PyReadonlyArray1<F>>()?;
    let limits = limits.map(|(l, u)| Ok::<_, PyErr>((l.extract::<PyReadonlyArray1<F>>()?, u.extract::<PyReadonlyArray1<F>>()?))).transpose()?;
    let bounds = match &limits { Some((l, u)) => Some(Bounds { lower: l.as_slice()?, upper: u.as_slice()? }), None => None };
    let options = Options { method, bounds, tolerance: tol.map(F::narrow), evaluations: maxfev, population, seed };
    let mut objective = Callable { fun, jac, vectorized, float: PhantomData };
    let start = x0.as_slice()?;
    options.check(Task::Minimize { gradient: objective.has_gradient() }, start.len()).map_err(PyValueError::new_err)?;
    let found = opt::minimize(&mut objective, start, &options)?;
    Ok(outcome(x0.py(), found))
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
) -> PyResult<Outcome<'py>> {
    let method = method.map(str::parse::<Method>).transpose().map_err(PyValueError::new_err)?;
    if x0.cast::<PyArray1<f32>>().is_ok() { minimize_at::<f32>(fun, jac, vectorized, &x0, limits, method, tol, maxfev, population, seed) }
    else { minimize_at::<f64>(fun, jac, vectorized, &x0, limits, method, tol, maxfev, population, seed) }
}

/// A zero of `f` by Brent's method, as the point, the number of evaluations and whether the bracket shrank within `rtol` before
/// `maxfev` evaluations. With `bracket`, the search starts from its ends. Without it, the search steps outward from `x0`.
#[pyfunction]
#[pyo3(signature = (f, x0, bracket=None, rtol=None, maxfev=None))]
fn root_scalar(py: Python<'_>, f: Bound<'_, PyAny>, x0: f64, bracket: Option<(f64, f64)>, rtol: Option<f64>, maxfev: Option<usize>) -> PyResult<(f64, usize, bool)> {
    let ends = bracket.map(|(a, b)| ([a], [b]));
    let bounds = ends.as_ref().map(|(lower, upper)| Bounds { lower, upper });
    let options = Options { method: Some(Method::Brent), bounds, tolerance: rtol, evaluations: maxfev, ..Default::default() };
    options.check(Task::Solve { residuals: 1 }, 1).map_err(PyValueError::new_err)?;
    let mut g = |x: &[f64], out: &mut [f64]| -> PyResult<()> {
        py.check_signals()?;
        out[0] = f.call1((x[0],))?.extract()?;
        Ok(())
    };
    let mut first = [0.0];
    g(&[x0], &mut first)?;
    let found = opt::solve(&mut g, &[x0], &first, &options)?;
    Ok((found.x[0], found.evaluations, found.converged))
}

/// `least_squares` at the float type `F` of `x0`.
fn least_squares_at<'py, F: Native + Element + Factor>(
    fun: Bound<'py, PyAny>, x0: &Bound<'py, PyAny>, jac: Option<Bound<'py, PyAny>>, tol: Option<f64>, maxfev: Option<usize>,
) -> PyResult<Outcome<'py>> {
    let x0 = x0.extract::<PyReadonlyArray1<F>>()?;
    let options = Options { method: Some(Method::LevenbergMarquardt), tolerance: tol.map(F::narrow), evaluations: maxfev, ..Default::default() };
    let start = x0.as_slice()?;
    let first: Vec<F> = give::<F>(&fun, PyArray1::from_slice(x0.py(), start).into_any())?.as_array().iter().copied().collect();
    options.check(Task::Solve { residuals: first.len() }, start.len()).map_err(PyValueError::new_err)?;
    let found = opt::solve(&mut Vectors { fun, jac, rmatvec: None, msolve: None, float: PhantomData }, start, &first, &options)?;
    Ok(outcome(x0.py(), found))
}

/// The least-squares solution of `fun(x) = 0` from `x0` by Levenberg–Marquardt, as the point, the norm of the residuals there, the
/// number of evaluations and whether the method's stopping test passed. `fun`'s values at `x0`, which the search counts as its first
/// evaluation, give the number of residuals. `jac`, where given, gives the Jacobian, with a row for each residual. `x0` is a contiguous
/// float32 or float64 vector, whose type the search computes in.
#[pyfunction]
#[pyo3(signature = (fun, x0, jac=None, tol=None, maxfev=None))]
fn least_squares<'py>(
    fun: Bound<'py, PyAny>, x0: Bound<'py, PyAny>, jac: Option<Bound<'py, PyAny>>, tol: Option<f64>, maxfev: Option<usize>,
) -> PyResult<Outcome<'py>> {
    if x0.cast::<PyArray1<f32>>().is_ok() { least_squares_at::<f32>(fun, &x0, jac, tol, maxfev) }
    else { least_squares_at::<f64>(fun, &x0, jac, tol, maxfev) }
}

/// `linear` at the float type `F` of `b` and `x0`.
#[allow(clippy::too_many_arguments)]
fn linear_at<'py, F: Native + Element>(
    matvec: Bound<'py, PyAny>, b: &Bound<'py, PyAny>, x0: &Bound<'py, PyAny>, method: Method, rmatvec: Option<Bound<'py, PyAny>>,
    msolve: Option<Bound<'py, PyAny>>, rtol: Option<f64>, maxiter: Option<usize>,
) -> PyResult<Outcome<'py>> {
    let (b, x0) = (b.extract::<PyReadonlyArray1<F>>()?, x0.extract::<PyReadonlyArray1<F>>()?);
    let (b, start) = (b.as_slice()?, x0.as_slice()?);
    let mut a = Vectors { fun: matvec, jac: None, rmatvec, msolve, float: PhantomData };
    let (task, n) = (Task::Linear { equations: b.len(), adjoint: a.has_vjp() }, start.len());
    // SciPy's `maxiter` counts GMRES's restart cycles, and the other methods' steps. A step takes one product with A, and one with
    // Aᵀ too for least squares, which also takes one product for Aᵀb. A cycle includes its residual's product. The other methods take
    // the residual's once, before their first step.
    let normal = task.least_squares(n);
    let pass = 1 + usize::from(normal);
    let (per, before) = if method == Method::Gmres { ((opt::GMRES_RESTART.min(n) + 1) * pass, 0) } else { (pass, pass) };
    let evaluations = maxiter.map(|steps| steps * per + before + usize::from(normal));
    let options = Options { method: Some(method), tolerance: rtol.map(F::narrow), evaluations, ..Default::default() };
    options.check(task, n).map_err(PyValueError::new_err)?;
    let found = opt::solve_linear(&mut a, b, start, &options)?;
    Ok(outcome(x0.py(), found))
}

/// The solution of A x = `b` from `x0` by `method`, as the point, the norm of `b` − A x there, the number of products with A and its
/// transpose, and whether the residual fell within `rtol`. `matvec` gives A's products, and `rmatvec`, where given, its transpose's,
/// which makes the solution least squares. `msolve`, where given, gives the preconditioner's products. `b` and `x0` are contiguous
/// float32 or float64 vectors of one type, which the solver computes in. `maxiter` counts iterations as SciPy's solvers do.
#[pyfunction]
#[pyo3(signature = (matvec, b, x0, method, rmatvec=None, msolve=None, rtol=None, maxiter=None))]
#[allow(clippy::too_many_arguments)]
fn linear<'py>(
    matvec: Bound<'py, PyAny>, b: Bound<'py, PyAny>, x0: Bound<'py, PyAny>, method: &str, rmatvec: Option<Bound<'py, PyAny>>,
    msolve: Option<Bound<'py, PyAny>>, rtol: Option<f64>, maxiter: Option<usize>,
) -> PyResult<Outcome<'py>> {
    let method = method.parse::<Method>().map_err(PyValueError::new_err)?;
    if b.cast::<PyArray1<f32>>().is_ok() { linear_at::<f32>(matvec, &b, &x0, method, rmatvec, msolve, rtol, maxiter) }
    else { linear_at::<f64>(matvec, &b, &x0, method, rmatvec, msolve, rtol, maxiter) }
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimize, m)?)?;
    m.add_function(wrap_pyfunction!(root_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(least_squares, m)?)?;
    m.add_function(wrap_pyfunction!(linear, m)?)
}
