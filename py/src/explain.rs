//! Python access to the shared route-event capture.
use pyo3::prelude::*;
use rustymath::events::{EventKind, Record, Report, Run, Sink};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

#[derive(Default)]
struct Store(Mutex<Report>);
impl Store { fn lock(&self) -> MutexGuard<'_, Report> { self.0.lock().unwrap_or_else(PoisonError::into_inner) } }
impl Sink for Store { fn record(&self, event: Record) { self.lock().record(event); } }

#[pyclass(name = "RouteEvent", module = "rustymath", get_all, frozen)]
struct RouteEvent { kind: &'static str, route: &'static str, reason: &'static str, calls: usize, items: usize }
#[pymethods]
impl RouteEvent {
    fn __repr__(&self) -> String { format!("{} {}{} ({} calls, {} items)", self.kind, self.route, if self.reason.is_empty() { String::new() } else { format!(": {}", self.reason) }, self.calls, self.items) }
}

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
    fn routes(&self) -> Vec<&'static str> { self.store.lock().records.iter().filter(|r| r.kind == EventKind::Taken).map(|r| r.route.name()).collect() }
    #[getter]
    fn records(&self) -> Vec<RouteEvent> { self.store.lock().records.iter().map(|r| RouteEvent { kind: r.kind.name(), route: r.route.name(), reason: r.reason, calls: r.calls, items: r.items }).collect() }
    /// One line for each record, as its `RouteEvent` displays, or `no route events` where the capture recorded none.
    fn __repr__(&self) -> String {
        let records = self.records();
        if records.is_empty() { "no route events".into() } else { records.iter().map(RouteEvent::__repr__).collect::<Vec<_>>().join("\n") }
    }
}

/// Capture route events with `with rm.explain() as report:`. `report.routes` lists routes taken; `report.records` gives their counts, declined optimizations and stopped traces. A report displays its records, one a line. Captures include every thread and must not overlap, apart from nesting on one thread.
#[pyfunction]
fn explain() -> Explanation { Explanation { store: Arc::default(), run: None } }

pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<RouteEvent>()?;
    m.add_class::<Explanation>()?;
    m.add_function(wrap_pyfunction!(explain, m)?)
}
