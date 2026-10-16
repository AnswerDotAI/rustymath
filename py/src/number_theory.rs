//! rustymath's `number_theory` module on Python integers of any size. Each function checks for Ctrl-C at least about once a millisecond, and stops with `KeyboardInterrupt`.
use num_bigint::BigUint;
use num_traits::Zero;
use crate::array::{core, give};
use pyo3::{exceptions::{PyOverflowError, PyValueError}, prelude::*};
use rustymath::number_theory as nt;

fn positive(n: &BigUint, name: &str) -> PyResult<()> {
    if n.is_zero() { Err(PyValueError::new_err(format!("{name} needs a positive integer"))) } else { Ok(()) }
}

/// Whether `n` is prime.
#[pyfunction]
pub fn is_prime(py: Python<'_>, n: BigUint) -> PyResult<bool> { nt::is_prime_with(&n, || py.check_signals()) }

/// The prime factors of `n` in increasing order, each with its multiplicity.
#[pyfunction]
pub fn factor(py: Python<'_>, n: BigUint) -> PyResult<Vec<(BigUint, u32)>> { positive(&n, "factor")?; nt::factor_with(&n, || py.check_signals()) }

/// The smallest prime above `n`.
#[pyfunction]
pub fn next_prime(py: Python<'_>, n: BigUint) -> PyResult<BigUint> {
    Ok(nt::next_prime_with(&n, || py.check_signals())?.expect("a BigUint has no largest value"))
}

/// The largest prime below `n`, or `None` if `n` is 2 or less.
#[pyfunction]
pub fn prev_prime(py: Python<'_>, n: BigUint) -> PyResult<Option<BigUint>> { nt::prev_prime_with(&n, || py.check_signals()) }

/// Euler's totient: how many integers from 1 to `n` are coprime to `n`.
#[pyfunction]
pub fn totient(py: Python<'_>, n: BigUint) -> PyResult<BigUint> { positive(&n, "totient")?; nt::totient_with(&n, || py.check_signals()) }

/// How many primes are below `n`.
#[pyfunction]
pub fn prime_count(py: Python<'_>, n: u64) -> PyResult<u64> { nt::prime_count_with(n, || py.check_signals()) }

/// The exact factorial of the nonnegative integer `n`, as a Python integer.
#[pyfunction]
pub fn factorial(py: Python<'_>, n: u64) -> PyResult<BigUint> { nt::factorial_with(n, || py.check_signals()) }

/// The primes from `start` up to but not including `stop`, as an `i64` array. With one argument, the primes below it, as `range` counts.
#[pyfunction]
#[pyo3(signature = (start, stop=None))]
pub fn primes(py: Python<'_>, start: u64, stop: Option<u64>) -> PyResult<Bound<'_, PyAny>> {
    let (start, stop) = stop.map_or((0, start), |stop| (start, stop));
    let mut found = nt::Primes::new(start);
    let mut out = Vec::new();
    while let Some(p) = found.next_with(|| py.check_signals())?.filter(|&p| p < stop) {
        out.push(i64::try_from(p).map_err(|_| PyOverflowError::new_err(format!("the prime {p} doesn't fit i64")))?);
    }
    let len = out.len();
    give(py, core(out, &[len])?, false)
}

/// Adds this module's functions to `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    for f in [
        wrap_pyfunction!(is_prime, m)?, wrap_pyfunction!(factor, m)?, wrap_pyfunction!(next_prime, m)?, wrap_pyfunction!(prev_prime, m)?,
        wrap_pyfunction!(totient, m)?, wrap_pyfunction!(prime_count, m)?, wrap_pyfunction!(primes, m)?, wrap_pyfunction!(factorial, m)?,
    ] { m.add_function(f)?; }
    Ok(())
}
