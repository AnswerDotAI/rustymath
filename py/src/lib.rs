//! rustymath's Python extension. Each module's functions are plain Python functions, which convert their arguments through `convert` and return rustymath arrays through `array::give`.

use pyo3::prelude::*;
use rustymath::tolerant::{Tolerance, Tolerant};

mod array;
mod convert;
mod create;
mod distribution;
mod exact;
mod functions;
mod integer;
mod linalg;
mod number_theory;
mod optimize;
mod polynomial;
mod random;
mod search;
mod signal;
mod structure;
mod timing;

/// Sets the thread limit for calls made on this thread, returning its previous setting. 0 restores the default, 1 runs serially.
#[pyfunction]
fn set_threads(n: usize) -> usize { rustymath::parallel::set_threads(n) }
/// The thread limit explicitly set on this thread, or 0 for the default.
#[pyfunction]
fn thread_setting() -> usize { rustymath::parallel::thread_setting() }
/// The effective thread count for calls made on this thread.
#[pyfunction]
fn threads() -> usize { rustymath::parallel::threads() }

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    functions::add_to(m)?;
    for f in [wrap_pyfunction!(set_threads, m)?, wrap_pyfunction!(thread_setting, m)?, wrap_pyfunction!(threads, m)?] {
        m.add_function(f)?;
    }
    array::add(m)?;
    create::add(m)?;
    structure::add(m)?;
    polynomial::add(m)?;
    search::add(m)?;
    signal::add(m)?;
    timing::add(m)?;
    for (name, add) in [
        ("number_theory", number_theory::add as fn(&Bound<'_, PyModule>) -> PyResult<()>), ("distribution", distribution::add), ("linalg", linalg::add),
        ("exact", exact::add), ("integer", integer::add), ("optimize", optimize::add), ("random", random::add),
    ] {
        let sub = PyModule::new(m.py(), name)?;
        add(&sub)?;
        m.add_submodule(&sub)?;
    }
    m.add("CT", Tolerance::DEFAULT.value())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
