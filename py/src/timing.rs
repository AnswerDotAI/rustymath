//! The machine's timing lock, copy times and cost lines for Python, as `rustymath::timing` describes.
use pyo3::prelude::*;

/// A hold on the machine's timing lock, taken exclusively when a `with` block starts and released when it ends, as `rustymath::timing::exclusive` takes it. While the block runs, the calling thread runs rustymath's calls on one thread, and on macOS it asks for user-interactive QoS. `pauses` gives the pauses, in seconds, before each retry of a case over its budget, `column` which of a budget file's two native budgets this machine compares costs with, or `None` on a machine with no budgets, as `rustymath::timing::NATIVE_COLUMN` gives it, and `checks_budgets` whether a case still over its budget fails, as `rustymath::timing::checks_budgets` decides.
#[pyclass(module = "rustymath")]
struct Timing(Option<rustymath::timing::Hold>);

#[pymethods]
impl Timing {
    #[classattr]
    fn pauses() -> Vec<f64> { rustymath::timing::RETRY_PAUSES.to_vec() }
    #[classattr]
    fn column() -> Option<usize> { rustymath::timing::NATIVE_COLUMN }
    #[classattr]
    fn checks_budgets() -> bool { rustymath::timing::checks_budgets() }
    /// One copy's time for each of `items` items of `bytes` bytes, in seconds, and the line that the speed tests print for it, as `rustymath::timing::copy_time` and `copy_line` give them.
    #[staticmethod]
    fn copy(bytes: usize, items: usize) -> PyResult<(f64, String)> {
        let t = match bytes {
            4 => rustymath::timing::copy_time::<f32>(items), 8 => rustymath::timing::copy_time::<f64>(items),
            _ => return Err(pyo3::exceptions::PyValueError::new_err(format!("no copy time for {bytes}-byte items")))
        };
        Ok((t.seconds, rustymath::timing::copy_line(&t)))
    }
    /// A case's line in a speed test's log, from its time for each item in `seconds`, as `rustymath::timing::line` gives it.
    #[staticmethod]
    fn line(section: &str, code: &str, cost: f64, budget: &str, seconds: f64) -> String { rustymath::timing::line(section, code, cost, budget, seconds) }
    /// Waits for the lock without holding the GIL, as long as `rustymath::timing::exclusive` waits.
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> { slf.0 = Some(slf.py().detach(rustymath::timing::exclusive)?); Ok(slf) }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) { self.0 = None; }
}

/// A hold on the machine's timing lock, for `with rustymath.timing():` around a timing.
#[pyfunction]
fn timing() -> Timing { Timing(None) }

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Timing>()?; m.add_function(wrap_pyfunction!(timing, m)?) }
