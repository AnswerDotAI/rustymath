//! Python access to the shared producer-event capture.
use pyo3::{prelude::*, types::PyDict};
use rustymath::events::{Event, EventKind, Field, Report, Run, Sink};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

#[derive(Default)]
struct Store(Mutex<Report>);
impl Store { fn lock(&self) -> MutexGuard<'_, Report> { self.0.lock().unwrap_or_else(PoisonError::into_inner) } }
impl Sink for Store { fn record(&self, event: Event) { self.lock().record(event); } }

#[pyclass(name = "Explanation", module = "rustymath")]
struct Explanation { store: Arc<Store>, run: Option<Run> }
#[pymethods]
impl Explanation {
    fn __enter__(mut slf: PyRefMut<'_, Self>) -> PyRefMut<'_, Self> {
        slf.store.lock().records.clear();
        slf.run = Some(Run::start(slf.store.clone()));
        slf
    }
    fn __exit__(&mut self, _kind: &Bound<'_, PyAny>, _value: &Bound<'_, PyAny>, _traceback: &Bound<'_, PyAny>) {
        let outer = self.run.as_ref().and_then(Run::outer).cloned();
        self.run = None;
        if let Some(outer) = outer { for event in &self.store.lock().records { outer.record(event.clone()); } }
    }
    #[getter]
    fn routes(&self) -> Vec<&'static str> { self.store.lock().routes().filter(|r| r.kind == EventKind::Taken).map(|r| r.route.name()).collect() }
    #[getter]
    fn records(&self, py: Python<'_>) -> PyResult<Vec<Py<PyDict>>> {
        let report = self.store.lock().clone();
        report.records.iter().map(|event| {
            let record = PyDict::new(py);
            record.set_item("kind", event.name())?;
            for ((name, _), field) in event.table().1.iter().zip(event.fields()) {
                match field {
                    Field::Text(t) => record.set_item(name, t.as_ref())?, Field::Count(n) => record.set_item(name, n)?,
                    Field::Truth(t) => record.set_item(name, t)?
                }
            }
            Ok(record.unbind())
        }).collect()
    }
    fn __repr__(&self) -> String {
        let report = self.store.lock();
        if report.records.is_empty() { return "no events".into(); }
        report.records.iter().map(|event| format!("{} {:?}", event.name(), event)).collect::<Vec<_>>().join("\n")
    }
}

/// Capture producer events with `with rm.explain() as report:`. `routes` lists successes; `records` gives dictionaries of event fields. Capture includes workers and permits nesting on one thread.
#[pyfunction]
fn explain() -> Explanation { Explanation { store: Arc::default(), run: None } }

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> { m.add_class::<Explanation>()?; m.add_function(wrap_pyfunction!(explain, m)?) }
