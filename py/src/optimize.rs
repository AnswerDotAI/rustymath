//! rustymath's `optimize` module on Python functions. Python functions receive their points as rustymath arrays, and what they return goes through the shared converter. Each call to a function, for one point or a batch, first checks for Ctrl-C. An exception from a function stops the search and comes back to the caller. A search computes in `f32` where its start, or `b` for the linear solvers, holds 16- or 32-bit floats, and in `f64` otherwise.
use crate::array::{Array, core};
use crate::convert::{Arg, convert, one};
use pyo3::{exceptions::{PyTypeError, PyValueError}, intern, prelude::*};
use rustymath::buffer::{DType, Element, FloatWidth};
use rustymath::error::{Error, Kind};
use rustymath::linalg::{Factor, Inner, Orientation, PlusTimes};
use rustymath::{Native, optimize::{self as opt, Bounds, Mapping, Method, Objective, Options, Task}};
use rustymath_buffer::py_err;
use std::marker::PhantomData;

/// Runs `$body` with `$f` as the float type that a search computes in for data of `$dtype`, as [`DType::inexact`] gives it: `f32` for 16- and 32-bit floats, and `f64` otherwise.
macro_rules! float {
    ($dtype:expr, $f:ident => $body:expr) => {
        if $dtype.inexact(FloatWidth::F64).map(|d| d.0) == Some(DType::Float(FloatWidth::F32)) { type $f = f32; $body } else { type $f = f64; $body }
    };
}

/// `x` as a rustymath array of `shape`, to pass to a Python function.
fn array<'py, F: Element>(py: Python<'py>, x: Vec<F>, shape: &[usize]) -> PyResult<Bound<'py, Array>> { Bound::new(py, Array::new(core(x, shape)?)) }

/// What `f` gives for `arg`, after checking for Ctrl-C.
fn apply<'py>(f: &Bound<'py, PyAny>, arg: Bound<'py, Array>) -> PyResult<Bound<'py, PyAny>> {
    f.py().check_signals()?;
    f.call1((arg,))
}

/// `values`, which a Python function gave, read into `out` in row-major order, after checking that there's one for each slot of `out`. They convert to the search's float type `F`, narrowing included.
fn read<F: Element>(values: &Bound<'_, PyAny>, out: &mut [F], what: &str) -> PyResult<()> {
    let values = one(values)?.cast::<F>()?;
    if values.len() != out.len() { return Err(PyValueError::new_err(format!("{what} gave {} values where {} were needed", values.len(), out.len()))); }
    out.copy_from_slice(&values);
    Ok(())
}

/// What `f` gives for the vector `x`, read into `out`.
fn call<F: Element>(f: &Bound<'_, PyAny>, x: &[F], out: &mut [F], what: &str) -> PyResult<()> { read(&apply(f, array(f.py(), x.to_vec(), &[x.len()])?)?, out, what) }

/// A Python function to minimize, and its gradient if it has one. With `vectorized`, `fun` takes points as the columns of a 2-D array and gives a value for each, as SciPy's `differential_evolution` calls it.
struct Callable<'py, F> { fun: Bound<'py, PyAny>, jac: Option<Bound<'py, PyAny>>, vectorized: bool, float: PhantomData<F> }

impl<F: Native + Element> Objective<F> for Callable<'_, F> {
    type Error = PyErr;
    /// The value at `x`. A vectorized `fun` gets `x` as a one-column 2-D array, as it gets every point.
    fn value(&mut self, x: &[F]) -> PyResult<F> {
        let mut y = [F::zero()];
        if self.vectorized {
            self.values(x, &mut y)?;
        } else {
            let value = apply(&self.fun, array(self.fun.py(), x.to_vec(), &[x.len()])?)?;
            match value.extract::<f64>() { Ok(v) => y[0] = F::narrow(v), Err(_) => read(&value, &mut y, "fun")? }
        }
        Ok(y[0])
    }
    fn batch(&self) -> bool { self.vectorized }
    fn values(&mut self, xs: &[F], out: &mut [F]) -> PyResult<()> {
        let (count, n) = (out.len(), xs.len() / out.len());
        let columns = (0..n * count).map(|ij| xs[ij % count * n + ij / count]).collect();
        read(&apply(&self.fun, array(self.fun.py(), columns, &[n, count])?)?, out, "fun")
    }
    fn has_gradient(&self) -> bool { self.jac.is_some() }
    fn gradient(&mut self, x: &[F], g: &mut [F]) -> PyResult<()> {
        call(self.jac.as_ref().expect("minimize asks only an objective with a gradient for it"), x, g, "jac")
    }
}

/// A function that gives a vector for a vector: a Python function, with its transpose's where given, or a matrix, whose products rustymath computes.
enum Product<'py, F> { Python { f: Bound<'py, PyAny>, t: Option<Bound<'py, PyAny>> }, Matrix { a: Vec<F>, shape: [usize; 2] } }

/// Whether `a` gives its products through a `matvec` method, as SciPy's `LinearOperator` does, rather than being a matrix.
fn operator(a: &Bound<'_, PyAny>) -> PyResult<bool> { a.hasattr(intern!(a.py(), "matvec")) }

impl<'py, F: Element> Product<'py, F> where PlusTimes: Inner<F, Out = F> {
    /// `a`'s `matvec` and `rmatvec` methods, where it has them.
    fn python(a: &Bound<'py, PyAny>) -> PyResult<Self> {
        let py = a.py();
        let t = if a.hasattr(intern!(py, "rmatvec"))? { Some(a.getattr(intern!(py, "rmatvec"))?) } else { None };
        Ok(Self::Python { f: a.getattr(intern!(py, "matvec"))?, t })
    }
    /// The matrix `a`.
    fn matrix(a: &Arg<'_>) -> PyResult<Self> {
        let &[m, n] = a.layout().shape() else { return Err(py_err(Error::new(Kind::Rank, "a matrix needs two axes"))) };
        Ok(Self::Matrix { a: a.items()?.into_owned(), shape: [m, n] })
    }
    fn has_transpose(&self) -> bool { !matches!(self, Self::Python { t: None, .. }) }
    /// The product with `x`, into `out`.
    fn apply(&self, x: &[F], out: &mut [F], what: &str) -> PyResult<()> {
        match self {
            Self::Python { f, .. } => call(f, x, out, what),
            &Self::Matrix { ref a, shape: [m, n] } => { out.copy_from_slice(&PlusTimes.inner_ops(a, x, [Orientation::Normal; 2], [m, n, 1])); Ok(()) }
        }
    }
    /// The transpose's product with `u`, into `out`.
    fn transposed(&self, u: &[F], out: &mut [F], what: &str) -> PyResult<()> {
        match self {
            Self::Python { t, .. } => call(t.as_ref().expect("solve_linear asks only a mapping with a transpose for it"), u, out, what),
            &Self::Matrix { ref a, shape: [m, n] } => { out.copy_from_slice(&PlusTimes.inner_ops(a, u, [Orientation::Transpose, Orientation::Normal], [n, m, 1])); Ok(()) }
        }
    }
}

/// A function that gives a vector for a vector, as a search uses it: `fun` its values, `jac`, where given, the Jacobian flattened by rows, `msolve`, where given, a preconditioner, and `adjoint` whether the search uses `fun`'s transpose.
struct Vectors<'py, F> { fun: Product<'py, F>, jac: Option<Bound<'py, PyAny>>, msolve: Option<Product<'py, F>>, adjoint: bool }

impl<F: Native + Element> Mapping<F> for Vectors<'_, F> where PlusTimes: Inner<F, Out = F> {
    type Error = PyErr;
    fn value(&mut self, x: &[F], out: &mut [F]) -> PyResult<()> { self.fun.apply(x, out, "fun") }
    fn has_vjp(&self) -> bool { self.adjoint }
    fn vjp(&mut self, u: &[F], _: &[F], out: &mut [F]) -> PyResult<()> { self.fun.transposed(u, out, "rmatvec") }
    fn jacobian(&mut self, x: &[F], j: &mut [F]) -> PyResult<bool> where F: Native {
        let Some(jac) = &self.jac else { return Ok(false) };
        call(jac, x, j, "jac")?;
        Ok(true)
    }
    fn has_preconditioner(&self) -> bool { self.msolve.is_some() }
    fn precondition(&mut self, v: &[F], out: &mut [F]) -> PyResult<()> {
        self.msolve.as_ref().expect("solve_linear asks only a mapping with a preconditioner for it").apply(v, out, "M")
    }
}

/// What a search found, as the binding returns it: the point, the value there, the number of evaluations and whether the method's stopping test passed.
type Outcome<'py> = (Bound<'py, Array>, f64, usize, bool);

/// `found` as an [`Outcome`].
fn outcome<F: Native + Element>(py: Python<'_>, found: opt::Found<Vec<F>, F>) -> PyResult<Outcome<'_>> {
    let n = found.x.len();
    Ok((array(py, found.x, &[n])?, found.value.wide(), found.evaluations, found.converged))
}

/// `minimize` at the float type `F`, with `data` the start point and, where there are bounds, the lower and upper bounds.
#[allow(clippy::too_many_arguments)]
fn minimize_at<'py, F: Native + Element>(
    fun: Bound<'py, PyAny>, jac: Option<Bound<'py, PyAny>>, vectorized: bool, data: &[Arg<'_>], method: Option<Method>, tol: Option<f64>, maxfev: Option<usize>,
    popsize: Option<usize>, seed: u64) -> PyResult<Outcome<'py>> {
    let (py, start) = (fun.py(), data[0].items::<F>()?);
    let limits = if let [_, lower, upper] = data { Some((lower.items::<F>()?, upper.items::<F>()?)) } else { None };
    let bounds = limits.as_ref().map(|(lower, upper)| Bounds { lower, upper });
    let options = Options { method, bounds, tolerance: tol.map(F::narrow), evaluations: maxfev, population: popsize.map(|p| p * start.len()), seed };
    let mut objective = Callable { fun, jac, vectorized, float: PhantomData };
    options.check(Task::Minimize { gradient: objective.has_gradient() }, start.len()).map_err(py_err)?;
    let found = opt::minimize(&mut objective, &start, &options)?;
    outcome(py, found)
}

/// The minimum of `fun` from `x0`, as the point, its value, the number of evaluations and whether the method's stopping test passed. `rustymath.optimize.minimize` converts SciPy's arguments to these: `limits` holds the lower and upper bounds, with infinities for no limit, which convert together with `x0`, and `popsize` counts differential evolution's members for each variable.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn minimize<'py>(
    fun: Bound<'py, PyAny>, x0: &Bound<'py, PyAny>, method: Option<&str>, jac: Option<Bound<'py, PyAny>>, limits: Option<(Bound<'py, PyAny>, Bound<'py, PyAny>)>,
    tol: Option<f64>, maxfev: Option<usize>, popsize: Option<usize>, seed: u64, vectorized: bool) -> PyResult<Outcome<'py>> {
    let method = method.map(str::parse::<Method>).transpose().map_err(py_err)?;
    let mut data = vec![x0];
    if let Some((lower, upper)) = &limits { data.extend([lower, upper]); }
    let data = convert(&data)?.args;
    float!(data[0].view().dtype(), F => minimize_at::<F>(fun, jac, vectorized, &data, method, tol, maxfev, popsize, seed))
}

/// A zero of `f` by Brent's method, as the point, the number of evaluations and whether the bracket shrank within `rtol` before `maxfev` evaluations. With `bracket`, the search starts from its ends. Without it, the search steps outward from `x0`.
#[pyfunction]
#[pyo3(signature = (f, x0, bracket=None, rtol=None, maxfev=None))]
fn root_scalar(
    py: Python<'_>, f: Bound<'_, PyAny>, x0: f64, bracket: Option<(f64, f64)>, rtol: Option<f64>, maxfev: Option<usize>) -> PyResult<(f64, usize, bool)> {
    let ends = bracket.map(|(a, b)| ([a], [b]));
    let bounds = ends.as_ref().map(|(lower, upper)| Bounds { lower, upper });
    let options = Options { method: Some(Method::Brent), bounds, tolerance: rtol, evaluations: maxfev, ..Default::default() };
    options.check(Task::Solve { residuals: 1 }, 1).map_err(py_err)?;
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

/// `least_squares` at the float type `F`.
fn least_squares_at<'py, F: Native + Element + Factor>(
    fun: Bound<'py, PyAny>, x0: &Arg<'_>, jac: Option<Bound<'py, PyAny>>, tol: Option<f64>, maxfev: Option<usize>) -> PyResult<Outcome<'py>> where PlusTimes: Inner<F, Out = F> {
    let (py, start) = (fun.py(), x0.items::<F>()?);
    let options = Options { method: Some(Method::LevenbergMarquardt), tolerance: tol.map(F::narrow), evaluations: maxfev, ..Default::default() };
    let first = one(&apply(&fun, array(py, start.to_vec(), &[start.len()])?)?)?.cast::<F>()?;
    options.check(Task::Solve { residuals: first.len() }, start.len()).map_err(py_err)?;
    let found = opt::solve(&mut Vectors { fun: Product::Python { f: fun, t: None }, jac, msolve: None, adjoint: false }, &start, &first, &options)?;
    outcome(py, found)
}

/// The least-squares solution of `fun(x) = 0` from `x0` by Levenberg–Marquardt, as the point, the norm of the residuals there, the number of evaluations and whether the method's stopping test passed. `fun`'s values at `x0`, which the search counts as its first evaluation, give the number of residuals. `jac`, where given, gives the Jacobian, with a row for each residual.
#[pyfunction]
#[pyo3(signature = (fun, x0, jac=None, tol=None, maxfev=None))]
fn least_squares<'py>(
    fun: Bound<'py, PyAny>, x0: &Bound<'py, PyAny>, jac: Option<Bound<'py, PyAny>>, tol: Option<f64>, maxfev: Option<usize>) -> PyResult<Outcome<'py>> {
    let x0 = one(x0)?;
    float!(x0.view().dtype(), F => least_squares_at::<F>(fun, &x0, jac, tol, maxfev))
}

/// `linear` at the float type `F`. `data` holds `b`, then `a` where it's a matrix, `x0` where `start` says it's given, and `m` where it's a matrix.
#[allow(clippy::too_many_arguments)]
fn linear_at<'py, F: Native + Element>(
    a: &Bound<'py, PyAny>, m: Option<&Bound<'py, PyAny>>, data: &[Arg<'_>], start: bool, method: Method, normal: bool, name: &str, rtol: Option<f64>,
    maxiter: Option<usize>) -> PyResult<Outcome<'py>> where PlusTimes: Inner<F, Out = F> {
    let (py, mut data) = (a.py(), data.iter());
    let b = data.next().expect("b comes first").items::<F>()?;
    let fun = if operator(a)? { Product::python(a)? } else { Product::matrix(data.next().expect("a matrix A follows b"))? };
    if normal && !fun.has_transpose() { return Err(PyTypeError::new_err(format!("{name} needs `A.rmatvec`"))); }
    let start = if start {
        data.next().expect("x0 follows A").items::<F>()?.into_owned()
    } else {
        vec![F::zero(); match &fun { Product::Matrix { shape, .. } => shape[1], Product::Python { .. } => a.getattr(intern!(py, "shape"))?.extract::<(usize, usize)>()?.1 }]
    };
    let n = start.len();
    if let Product::Matrix { shape, .. } = &fun && *shape != [b.len(), n] { return Err(py_err(Error::new(Kind::Length, "A needs a row for each item of b and a column for each unknown"))); }
    let msolve = match m {
        None => None,
        Some(m) if operator(m)? => Some(Product::python(m)?),
        Some(_) => Some(Product::matrix(data.next().expect("a matrix M comes last"))?),
    };
    if let Some(Product::Matrix { shape, .. }) = &msolve && *shape != [n, n] { return Err(py_err(Error::new(Kind::Length, "M needs a row and a column for each unknown"))); }
    let mut vectors = Vectors { fun, jac: None, msolve, adjoint: normal };
    let task = Task::Linear { equations: b.len(), adjoint: normal };
    // SciPy's `maxiter` counts GMRES's restart cycles, and the other methods' steps. A step takes one product with A, and one with Aᵀ too for least squares, which also takes one product for Aᵀb. A cycle includes its residual's product. The other methods take the residual's once, before their first step.
    let least = task.least_squares(n);
    let pass = 1 + usize::from(least);
    let (per, before) = if method == Method::Gmres { ((opt::GMRES_RESTART.min(n) + 1) * pass, 0) } else { (pass, pass) };
    let evaluations = maxiter.map(|steps| steps * per + before + usize::from(least));
    let options = Options { method: Some(method), tolerance: rtol.map(F::narrow), evaluations, ..Default::default() };
    options.check(task, n).map_err(py_err)?;
    let found = opt::solve_linear(&mut vectors, &b, &start, &options)?;
    outcome(py, found)
}

/// The solution of A x = `b` from `x0` by `method`, as the point, the norm of `b` − A x there, the number of products with A and its transpose, and whether the residual fell within `rtol`. `a` is a matrix, or anything with a `matvec` method, plus `rmatvec` where `normal` makes the solution least squares, through the normal equations. `x0` defaults to zeros. `m`, where given, is a preconditioner in either form. `b` and the matrices among `a`, `x0` and `m` convert together, so arrays of different dtypes promote to one, as `convert` promotes them. `name` names the solver in errors. `maxiter` counts iterations as SciPy's solvers do.
#[pyfunction]
#[pyo3(signature = (a, b, x0, method, normal, name, m=None, rtol=None, maxiter=None))]
#[allow(clippy::too_many_arguments)]
fn linear<'py>(
    a: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>, x0: Option<&Bound<'py, PyAny>>, method: &str, normal: bool, name: &str, m: Option<&Bound<'py, PyAny>>,
    rtol: Option<f64>, maxiter: Option<usize>) -> PyResult<Outcome<'py>> {
    let method = method.parse::<Method>().map_err(py_err)?;
    let mut data = vec![b];
    if !operator(a)? { data.push(a); }
    data.extend(x0);
    if let Some(m) = m && !operator(m)? { data.push(m); }
    let data = convert(&data)?.args;
    float!(data[0].view().dtype(), F => linear_at::<F>(a, m, &data, x0.is_some(), method, normal, name, rtol, maxiter))
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(minimize, m)?)?;
    m.add_function(wrap_pyfunction!(root_scalar, m)?)?;
    m.add_function(wrap_pyfunction!(least_squares, m)?)?;
    m.add_function(wrap_pyfunction!(linear, m)?)
}


