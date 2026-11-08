//! The machine's timing lock, copy times and cost lines for Python, as `rustymath_timing` describes.
use pyo3::prelude::*;

/// `rustymath_timing`'s `$f` at the float type whose items take `$bytes` bytes, f32 or f64. Returns a ValueError from the calling function for any other size.
macro_rules! at_width {
    ($bytes:expr, $f:ident($($arg:expr),*)) => {
        match $bytes {
            4 => rustymath_timing::$f::<f32>($($arg),*), 8 => rustymath_timing::$f::<f64>($($arg),*),
            bytes => return Err(pyo3::exceptions::PyValueError::new_err(format!("no copy time for {bytes}-byte items"))),
        }
    };
}

/// A hold on the timing lock that `timing` names, taken exclusively, or `shared` for a speed test's ordinary run, when a `with` block starts and released when it ends, as `rustymath_timing::hold_at` takes it. While the block runs, the calling thread runs rustymath's calls on one thread, and on macOS an exclusive hold asks for user-interactive QoS. `pauses` gives the pauses, in seconds, before each retry of a case over its budget in a calibration run, `column` which of a budget file's two native budgets this machine compares costs with, or `None` on a machine with no budgets, as `rustymath_timing::NATIVE_COLUMN` gives it, and `checks_budgets` whether a case still over its budget fails, as `rustymath_timing::checks_budgets` decides. `measures` names what a budget file counts for each item, and `count_margin` is the usual margin over a recorded count. `warm` keeps the thread busy after each pause. `budgets` reads a budget file, and `copy`, `peak_held`, `footprint`, `instructions` and `line` give what the speed test measures and prints.
#[pyclass(module = "rustymath")]
struct Timing { lock: String, shared: bool, hold: Option<rustymath_timing::Hold> }

#[pymethods]
impl Timing {
    #[classattr]
    fn pauses() -> Vec<f64> { rustymath_timing::RETRY_PAUSES.to_vec() }
    #[classattr]
    fn column() -> Option<usize> { rustymath_timing::NATIVE_COLUMN }
    #[classattr]
    fn checks_budgets() -> bool { rustymath_timing::checks_budgets() }
    #[classattr]
    fn measures() -> Vec<&'static str> { rustymath_timing::MEASURES.to_vec() }
    #[classattr]
    fn count_margin() -> f64 { rustymath_timing::COUNT_MARGIN }
    /// Keeps the calling thread busy for 0.1 s after a retry's pause, as `rustymath_timing::warm` does.
    #[staticmethod]
    fn warm() { rustymath_timing::warm() }
    /// One copy's time for each of `items` items of `bytes` bytes, in seconds, and the line that the speed tests print for it, as `rustymath_timing::copy_time` and `copy_line` give them.
    #[staticmethod]
    fn copy(bytes: usize, items: usize) -> PyResult<(f64, String)> {
        let t = at_width!(bytes, copy_time(items));
        Ok((t.seconds, rustymath_timing::copy_line(&t)))
    }
    /// What `run` gives when called, and the most bytes that it held at once, as `rustymath::memory::peak_held` counts them.
    #[staticmethod]
    fn peak_held(run: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, usize)> {
        let (result, held) = rustymath::memory::peak_held(|| run.call0());
        Ok((result?.unbind(), held))
    }
    /// A case's footprint in bytes and the time for each item of a copy of it, measured now, as `rustymath_timing::footprint` gives them for `items` items of `bytes` bytes and `held` bytes held at once, or `None` for a footprint within one core's cache.
    #[staticmethod]
    fn footprint(bytes: usize, items: usize, held: usize) -> PyResult<Option<(usize, f64)>> {
        Ok(at_width!(bytes, footprint(items, held)).map(|f| (f.bytes, f.seconds)))
    }
    /// A case's line in a speed test's log, from its time for each item in `seconds`, the `footprint` that `footprint` gives and its `counts` for each item in the order of `measures`, as `rustymath_timing::line` gives it.
    #[staticmethod]
    #[pyo3(signature = (section, code, cost, budget, seconds, footprint=None, counts=(None, None)))]
    fn line(section: &str, code: &str, cost: f64, budget: &str, seconds: f64, footprint: Option<(usize, f64)>, counts: (Option<f64>, Option<f64>)) -> String {
        let footprint = footprint.map(|(bytes, seconds)| rustymath_timing::Footprint { bytes, seconds });
        rustymath_timing::line(section, code, cost, budget, seconds, footprint, [counts.0, counts.1])
    }
    /// Each case of `text`, a budget file such as `tests/speed_python.txt`, as `(section, code, budget, counts)`. `budget` is its `Budget` in the column that this machine checks, or `None` on a machine with no budgets, as `rustymath_timing::budgeted` gives them. `counts` holds the `Count` that the case records in that column for each of `measures`, or `None`.
    #[staticmethod]
    fn budgets(text: &str) -> PyResult<Vec<(String, String, Option<Budget>, Vec<Option<Count>>)>> {
        let column = rustymath_timing::NATIVE_COLUMN;
        let cases = rustymath_timing::budgeted(text, column).map_err(pyo3::exceptions::PyValueError::new_err)?;
        Ok(cases.into_iter().map(|(case, budget)| {
            let counts = rustymath_timing::MEASURES.iter().map(|m| case.count(m, column).map(Count)).collect();
            (case.section.into(), case.code.into(), budget.map(Budget), counts)
        }).collect())
    }
    /// What `run` gives when called, and the instructions retired while it ran, as `rustymath_timing::instructions` counts them, or `None` where this machine can't count them.
    #[staticmethod]
    fn instructions(run: &Bound<'_, PyAny>) -> PyResult<(Py<PyAny>, Option<u64>)> {
        let (result, count) = rustymath_timing::instructions(|| run.call0());
        Ok((result?.unbind(), count))
    }
    /// Waits for the lock without holding the GIL, as long as `rustymath_timing::hold_at` waits, unless `try_timing` already holds it.
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyResult<PyRefMut<'_, Self>> {
        if slf.hold.is_none() {
            let (lock, shared) = (slf.lock.clone(), slf.shared);
            slf.hold = Some(slf.py().detach(|| rustymath_timing::hold_at(&lock, shared).map(rustymath_timing::Hold::on_one_thread))?);
        }
        Ok(slf)
    }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) { self.hold = None; }
}

/// A case's budget in the column that this machine checks, with any hold, as `rustymath_timing::Budget` gives it. `over(cost)` says whether a cost is over it, `stale(cost)` whether the hold of a gap has outlived the gap, and `str()` gives it as a speed test's line does.
#[pyclass(module = "rustymath", frozen)]
struct Budget(rustymath_timing::Budget);

#[pymethods]
impl Budget {
    fn over(&self, cost: f64) -> bool { self.0.over(cost) }
    fn stale(&self, cost: f64) -> bool { self.0.stale(cost) }
    fn __str__(&self) -> String { self.0.to_string() }
}

/// A count that a budget file records for each item of a case, in the column that this machine checks, as `rustymath_timing::Count` gives it. `value` is the count, and `over(count, margin)` says whether a count is more than `margin` times it allows, as `rustymath_timing::Count::over` decides.
#[pyclass(module = "rustymath", frozen)]
struct Count(rustymath_timing::Count);

#[pymethods]
impl Count {
    #[getter]
    fn value(&self) -> f64 { self.0.value }
    fn over(&self, count: f64, margin: f64) -> bool { self.0.over(count, margin) }
}

/// A hold on the machine's timing lock, for `with rustymath.timing():` around a timing, or on the lock whose file is `lock`, as a test of the protocol holds a file of its own. A `shared` hold is a speed test's ordinary run, which times beside builds and other tests.
#[pyfunction]
#[pyo3(signature = (lock=None, shared=false))]
fn timing(lock: Option<String>, shared: bool) -> Timing { Timing { lock: lock.unwrap_or_else(|| rustymath_timing::LOCK.into()), shared, hold: None } }

/// The machine's timing lock, held exclusively, where no build, test run or other timing holds it, or else `None` at once, as `rustymath_timing::try_exclusive` takes it. A `with` block around the hold keeps it, and the block's end releases it. A speed test's ordinary run uses it to confirm a case over its budget, after its shared hold ends.
#[pyfunction]
fn try_timing() -> PyResult<Option<Timing>> {
    Ok(rustymath_timing::try_exclusive()?.map(|hold| Timing { lock: rustymath_timing::LOCK.into(), shared: false, hold: Some(hold) }))
}

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Timing>()?;
    m.add_class::<Budget>()?;
    m.add_class::<Count>()?;
    m.add_function(wrap_pyfunction!(timing, m)?)?;
    m.add_function(wrap_pyfunction!(try_timing, m)?)
}
