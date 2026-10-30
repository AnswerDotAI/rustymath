//! The machine's timing lock, copy times and cost lines for Python, as `rustymath::timing` describes.
use pyo3::prelude::*;

/// `rustymath::timing`'s `$f` at the float type whose items take `$bytes` bytes, f32 or f64. Returns a ValueError from the calling function for any other size.
macro_rules! at_width {
    ($bytes:expr, $f:ident($($arg:expr),*)) => {
        match $bytes {
            4 => rustymath::timing::$f::<f32>($($arg),*), 8 => rustymath::timing::$f::<f64>($($arg),*),
            bytes => return Err(pyo3::exceptions::PyValueError::new_err(format!("no copy time for {bytes}-byte items"))),
        }
    };
}

/// A hold on the machine's timing lock, taken exclusively when a `with` block starts and released when it ends, as `rustymath::timing::exclusive` takes it. While the block runs, the calling thread runs rustymath's calls on one thread, and on macOS it asks for user-interactive QoS. `pauses` gives the pauses, in seconds, before each retry of a case over its budget, `column` which of a budget file's two native budgets this machine compares costs with, or `None` on a machine with no budgets, as `rustymath::timing::NATIVE_COLUMN` gives it, and `checks_budgets` whether a case still over its budget fails, as `rustymath::timing::checks_budgets` decides. `warm` keeps the thread busy after each pause. `budgets` reads a budget file, and `copy`, `peak_held`, `footprint` and `line` give what the speed test measures and prints.
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
    /// Keeps the calling thread busy for 0.1 s after a retry's pause, as `rustymath::timing::warm` does.
    #[staticmethod]
    fn warm() { rustymath::timing::warm() }
    /// One copy's time for each of `items` items of `bytes` bytes, in seconds, and the line that the speed tests print for it, as `rustymath::timing::copy_time` and `copy_line` give them.
    #[staticmethod]
    fn copy(bytes: usize, items: usize) -> PyResult<(f64, String)> {
        let t = at_width!(bytes, copy_time(items));
        Ok((t.seconds, rustymath::timing::copy_line(&t)))
    }
    /// What `run` gives when called, and the most bytes that it held at once, as `rustymath::memory::peak_held` counts them.
    #[staticmethod]
    fn peak_held(run: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, usize)> {
        let (result, held) = rustymath::memory::peak_held(|| run.call0());
        Ok((result?.unbind(), held))
    }
    /// A case's footprint in bytes and the time for each item of a copy of it, measured now, as `rustymath::timing::footprint` gives them for `items` items of `bytes` bytes and `held` bytes held at once, or `None` for a footprint within one core's cache.
    #[staticmethod]
    fn footprint(bytes: usize, items: usize, held: usize) -> PyResult<Option<(usize, f64)>> {
        Ok(at_width!(bytes, footprint(items, held)).map(|f| (f.bytes, f.seconds)))
    }
    /// A case's line in a speed test's log, from its time for each item in `seconds` and the `footprint` that `footprint` gives, as `rustymath::timing::line` gives it.
    #[staticmethod]
    #[pyo3(signature = (section, code, cost, budget, seconds, footprint=None))]
    fn line(section: &str, code: &str, cost: f64, budget: &str, seconds: f64, footprint: Option<(usize, f64)>) -> String {
        rustymath::timing::line(section, code, cost, budget, seconds, footprint.map(|(bytes, seconds)| rustymath::timing::Footprint { bytes, seconds }))
    }
    /// Each case of `text`, a budget file such as `tests/speed_python.txt`, as `(section, code, budget)`, where `budget` is its `Budget` in the column that this machine checks, or `None` on a machine with no budgets, as `rustymath::timing::budgeted` gives them.
    #[staticmethod]
    fn budgets(text: &str) -> PyResult<Vec<(String, String, Option<Budget>)>> {
        let cases = rustymath::timing::budgeted(text, rustymath::timing::NATIVE_COLUMN).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(cases.into_iter().map(|(section, code, budget)| (section.into(), code.into(), budget.map(Budget))).collect())
    }
    /// Waits for the lock without holding the GIL, as long as `rustymath::timing::exclusive` waits.
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> { slf.0 = Some(slf.py().detach(rustymath::timing::exclusive)?); Ok(slf) }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) { self.0 = None; }
}

/// A case's budget in the column that this machine checks, with any hold, as `rustymath::timing::Budget` gives it. `over(cost)` says whether a cost is over it, `stale(cost)` whether the hold of a gap has outlived the gap, and `str()` gives it as a speed test's line does.
#[pyclass(module = "rustymath", frozen)]
struct Budget(rustymath::timing::Budget);

#[pymethods]
impl Budget {
    fn over(&self, cost: f64) -> bool { self.0.over(cost) }
    fn stale(&self, cost: f64) -> bool { self.0.stale(cost) }
    fn __str__(&self) -> String { self.0.to_string() }
}

/// A hold on the machine's timing lock, for `with rustymath.timing():` around a timing.
#[pyfunction]
fn timing() -> Timing { Timing(None) }

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Timing>()?; m.add_class::<Budget>()?; m.add_function(wrap_pyfunction!(timing, m)?) }
