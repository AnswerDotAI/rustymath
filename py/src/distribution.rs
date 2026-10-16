//! rustymath's `distribution` module. Each constructor checks its parameters and gives a `Distribution`, whose methods take a number or an array of any shape.
use crate::array::{core, give};
use crate::functions::Apply;
use pyo3::{exceptions::PyValueError, prelude::*, types::PyFloat};
use rand::{SeedableRng, rngs::StdRng};
use rustymath::{distribution::{self as dist, standard::*}, kernels, layout::generated_len};
use rustymath_buffer::py_err;

/// A probability distribution, with a density, a cumulative distribution, a quantile function and a sampler.
#[pyclass(module = "rustymath.distribution", frozen)]
pub struct Distribution(dist::Distribution);

#[pymethods]
impl Distribution {
    /// The density at each item of `x`, or for a discrete distribution the probability of each.
    fn pdf<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { let k = dist::Pdf(self.0); kernels!(Unary of k).apply(py, "pdf", &k, &[x]) }

    /// The probability of a value at most each item of `x`.
    fn cdf<'py>(&self, py: Python<'py>, x: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { let k = dist::Cdf(self.0); kernels!(Unary of k).apply(py, "cdf", &k, &[x]) }

    /// The least value whose `cdf` is at least each item of `p`. A `p` outside 0 to 1 gives NaN.
    fn quantile<'py>(&self, py: Python<'py>, p: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> { let k = dist::Quantile(self.0); kernels!(Unary of k).apply(py, "quantile", &k, &[p]) }

    /// Values drawn from the distribution: one float with no `size`, or else an array of shape `size`. The same `seed` gives the same values.
    #[pyo3(signature = (size=None, seed=None))]
    fn sample<'py>(&self, py: Python<'py>, size: Option<&Bound<'py, PyAny>>, seed: Option<u64>) -> PyResult<Bound<'py, PyAny>> {
        let mut rng = seed.map_or_else(rand::make_rng::<StdRng>, StdRng::seed_from_u64);
        let Some(size) = size else { return Ok(PyFloat::new(py, self.0.sample(&mut rng)).into_any()) };
        let shape = size.extract::<usize>().map(|n| vec![n]).or_else(|_| size.extract::<Vec<usize>>())?;
        let mut out = vec![0.0; generated_len(&shape).map_err(py_err)?];
        self.0.samples(&mut rng, &mut out);
        give(py, core(out, &shape)?, false)
    }

    /// Whether the distribution's values are whole numbers.
    #[getter]
    fn is_discrete(&self) -> bool { self.0.is_discrete() }
}

/// Defines a constructor for each distribution, and `add`, which adds them and `Distribution` to a module. A parameter's default is one of rustymath's standard parameters.
macro_rules! constructors {
    ($($name:ident($($p:ident: $t:ty $(= $default:expr)?),+) $doc:literal;)+) => {
        $(
            #[doc = $doc]
            #[pyfunction]
            #[pyo3(signature = ($($p $(= $default)?),+))]
            pub fn $name($($p: $t),+) -> PyResult<Distribution> {
                dist::Distribution::$name($($p),+).map(Distribution).map_err(|e| PyValueError::new_err(e.0))
            }
        )+

        /// Adds `Distribution` and its constructors to `m`.
        pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add_class::<Distribution>()?;
            for f in [$(wrap_pyfunction!($name, m)?),+] { m.add_function(f)?; }
            Ok(())
        }
    };
}

constructors! {
    normal(mean: f64 = NORMAL[0], sd: f64 = NORMAL[1]) "The normal distribution with this mean and standard deviation.";
    uniform(min: f64 = UNIFORM[0], max: f64 = UNIFORM[1]) "The uniform distribution from `min` to `max`.";
    beta(a: f64, b: f64) "The beta distribution with shapes `a` and `b`.";
    cauchy(location: f64 = CAUCHY[0], scale: f64 = CAUCHY[1]) "The Cauchy distribution with this location and scale.";
    chi_squared(df: f64) "The chi-squared distribution with `df` degrees of freedom.";
    exponential(rate: f64 = EXPONENTIAL[0]) "The exponential distribution with this rate.";
    fisher(df1: f64, df2: f64) "The F distribution with `df1` and `df2` degrees of freedom.";
    gamma(shape: f64, scale: f64) "The gamma distribution with this shape and scale.";
    inverse_gamma(shape: f64, scale: f64) "The inverse gamma distribution with this shape and scale.";
    laplace(location: f64 = LAPLACE[0], scale: f64 = LAPLACE[1]) "The Laplace distribution with this location and scale.";
    logistic(location: f64 = LOGISTIC[0], scale: f64 = LOGISTIC[1]) "The logistic distribution with this location and scale.";
    log_normal(location: f64 = LOG_NORMAL[0], scale: f64 = LOG_NORMAL[1]) "The log-normal distribution whose logarithm has mean `location` and standard deviation `scale`.";
    student_t(df: f64) "Student's t distribution with `df` degrees of freedom.";
    weibull(shape: f64, scale: f64) "The Weibull distribution with this shape and scale.";
    binomial(n: u64, p: f64) "The binomial distribution of the successes in `n` trials, each with probability `p`.";
    poisson(rate: f64) "The Poisson distribution with this rate.";
}
