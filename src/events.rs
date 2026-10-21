//! The events that a frontend asks for while it runs code, as basedpl's `•explain` does. A run puts its sink in place for the whole process (`Run::start`), and code on every thread records into it, worker threads included. A process supports one run at a time, apart from runs nested on one thread: code that another thread runs during a run records into that run. rustymath records the faster routes that code considered and didn't take (`declined`). A frontend records its own events into its sink through `with`.
use std::{any::Any, sync::{atomic::{AtomicUsize, Ordering::Relaxed}, Arc, Mutex, MutexGuard, PoisonError}};

/// What a run records into. A frontend's sink also holds the frontend's own events, which it finds again through `with`.
pub trait Sink: Any + Send + Sync {
    /// Records that code declined `route` for `reason`, for `items` items.
    fn decline(&self, route: &'static str, reason: &'static str, items: usize);
}

/// The sink of the run under way, if any.
static CURRENT: Mutex<Option<Arc<dyn Sink>>> = Mutex::new(None);

/// The runs under way, nested ones included. Code outside every run pays one atomic load for each event it could record.
static RUNS: AtomicUsize = AtomicUsize::new(0);

/// Whether a run is under way.
#[inline]
pub fn running() -> bool { RUNS.load(Relaxed) > 0 }

/// A run under way, which records into its sink until it's dropped. Dropping it, as a panic in the run does, puts back the sink of the run it interrupted, if any, so a frontend that turns a panic into an error can carry on.
pub struct Run { outer: Option<Arc<dyn Sink>> }

impl Run {
    /// Starts a run that records into `sink`.
    pub fn start(sink: Arc<dyn Sink>) -> Self {
        RUNS.fetch_add(1, Relaxed);
        Self { outer: current().replace(sink) }
    }
    /// The sink of the run that this one interrupted, if any.
    pub fn outer(&self) -> Option<&Arc<dyn Sink>> { self.outer.as_ref() }
}

impl Drop for Run {
    fn drop(&mut self) {
        *current() = self.outer.take();
        RUNS.fetch_sub(1, Relaxed);
    }
}

fn current() -> MutexGuard<'static, Option<Arc<dyn Sink>>> { CURRENT.lock().unwrap_or_else(PoisonError::into_inner) }

/// `f` on the sink of the run under way, or `None` outside every run. `f` runs after the lock on the run's sink is released, so it can record.
pub fn with<T>(f: impl FnOnce(&Arc<dyn Sink>) -> T) -> Option<T> {
    if !running() { return None; }
    let sink = current().clone();
    sink.as_ref().map(f)
}

/// Records that code declined `route` for `reason`, for `items` items, into the run under way. Outside every run it costs one atomic load.
#[inline]
pub fn declined(route: &'static str, reason: &'static str, items: usize) {
    if running() { add(route, reason, items) }
}

#[cold]
#[inline(never)]
fn add(route: &'static str, reason: &'static str, items: usize) { with(|sink| sink.decline(route, reason, items)); }
