//! The machine's timing lock for Python, as `rustymath::timing` describes.
use pyo3::prelude::*;

/// A hold on the machine's timing lock, taken exclusively when a `with` block starts and released when it ends, as `rustymath::timing::exclusive` takes it. While the block runs, the calling thread runs rustymath's calls on one thread, and on macOS it asks for user-interactive QoS. `pauses` gives the pauses, in seconds, before each retry of a case over its budget.
#[pyclass(module = "rustymath")]
struct Timing(Option<rustymath::timing::Hold>);

#[pymethods]
impl Timing {
    #[classattr]
    fn pauses() -> Vec<f64> { rustymath::timing::RETRY_PAUSES.to_vec() }
    /// Waits for the lock without holding the GIL, as long as `rustymath::timing::exclusive` waits.
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        slf.0 = Some(slf.py().detach(rustymath::timing::exclusive)?);
        Ok(slf)
    }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) { self.0 = None; }
}

/// A hold on the machine's timing lock, for `with rustymath.timing():` around a timing.
#[pyfunction]
fn timing() -> Timing { Timing(None) }

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Timing>()?;
    m.add_function(wrap_pyfunction!(timing, m)?)
}
