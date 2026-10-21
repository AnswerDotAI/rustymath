//! Primality and factorization on arbitrary-size nonnegative Python integers. Prime enumeration/counting and factorial take `u64` arguments; enumerated primes must fit `int64`. Long calls check for Ctrl-C and stop with `KeyboardInterrupt`.
use num_bigint::BigUint;
use num_traits::Zero;
use crate::array::{core, give};
use crate::convert::{released, signals};
use pyo3::{exceptions::{PyOverflowError, PyValueError}, prelude::*};
use rustymath::number_theory as nt;

fn positive(n: &BigUint, name: &str) -> PyResult<()> {
    if n.is_zero() { Err(PyValueError::new_err(format!("{name} needs a positive integer"))) } else { Ok(()) }
}

/// Whether `n` passes the Baillie–PSW primality test. No composite below 2**64 passes; larger results are probable primes, with no known counterexample.
#[pyfunction]
pub fn is_prime(py: Python<'_>, n: BigUint) -> PyResult<bool> { Ok(released(py, || nt::is_prime_with(&n, signals))?) }

/// The prime factors of `n` in increasing order, each with its multiplicity.
#[pyfunction]
pub fn factor(py: Python<'_>, n: BigUint) -> PyResult<Vec<(BigUint, u32)>> { positive(&n, "factor")?; Ok(released(py, || nt::factor_with(&n, signals))?) }

/// The smallest prime above `n`.
#[pyfunction]
pub fn next_prime(py: Python<'_>, n: BigUint) -> PyResult<BigUint> {
    Ok(released(py, || nt::next_prime_with(&n, signals))?.expect("a BigUint has no largest value"))
}

/// The largest prime below `n`, or `None` if `n` is 2 or less.
#[pyfunction]
pub fn prev_prime(py: Python<'_>, n: BigUint) -> PyResult<Option<BigUint>> { Ok(released(py, || nt::prev_prime_with(&n, signals))?) }

/// Euler's totient: how many integers from 1 to `n` are coprime to `n`.
#[pyfunction]
pub fn totient(py: Python<'_>, n: BigUint) -> PyResult<BigUint> { positive(&n, "totient")?; Ok(released(py, || nt::totient_with(&n, signals))?) }

/// How many primes are below `n`.
#[pyfunction]
pub fn prime_count(py: Python<'_>, n: u64) -> PyResult<u64> { Ok(released(py, || nt::prime_count_with(n, signals))?) }

/// The exact factorial of the nonnegative integer `n`, as a Python integer.
#[pyfunction]
pub fn factorial(py: Python<'_>, n: u64) -> PyResult<BigUint> { Ok(released(py, || nt::factorial_with(n, signals))?) }

/// The primes from `start` up to but not including `stop`, as an `i64` array. With one argument, the primes below it, as `range` counts.
#[pyfunction]
#[pyo3(signature = (start, stop=None))]
pub fn primes(py: Python<'_>, start: u64, stop: Option<u64>) -> PyResult<Bound<'_, PyAny>> {
    let (start, stop) = stop.map_or((0, start), |stop| (start, stop));
    let out = released(py, || -> PyResult<Vec<i64>> {
        let (mut found, mut out) = (nt::Primes::new(start), Vec::new());
        while let Some(p) = found.next_with(signals)?.filter(|&p| p < stop) {
            out.push(i64::try_from(p).map_err(|_| PyOverflowError::new_err(format!("the prime {p} doesn't fit int64")))?);
        }
        Ok(out)
    })?;
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
