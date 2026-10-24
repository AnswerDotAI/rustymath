//! rustymath's Python extension. Each module's functions are plain Python functions, which convert their arguments through `convert` and return rustymath arrays through `array::give`.

use pyo3::prelude::*;

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

/// Sets the thread limit for calls in the whole process, returning its previous setting. 0 restores the default, 1 runs serially.
#[pyfunction]
fn set_threads(n: usize) -> usize { rustymath::parallel::set_threads(n) }
/// The thread limit that `set_threads` set for the process, or 0 for the default.
#[pyfunction]
fn thread_setting() -> usize { rustymath::parallel::thread_setting() }
/// The effective thread count for calls made on this thread: the process's limit, or 1 on a pool's worker thread.
#[pyfunction]
fn threads() -> usize { rustymath::parallel::threads() }
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    rustymath::interrupt::install(convert::signals);
    functions::add_to(m)?;
    for f in [
        wrap_pyfunction!(set_threads, m)?, wrap_pyfunction!(thread_setting, m)?, wrap_pyfunction!(threads, m)?,
    ] {
        m.add_function(f)?;
    }
    array::add(m)?;
    create::add(m)?;
    structure::add(m)?;
    polynomial::add(m)?;
    search::add(m)?;
    let mathish = PyModule::new(m.py(), "_mathish")?;
    functions::add_mathish(&mathish)?;
    search::add_mathish(&mathish)?;
    m.add_submodule(&mathish)?;
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
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
