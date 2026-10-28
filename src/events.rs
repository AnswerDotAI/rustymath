//! The events that a frontend asks for while it runs code, as basedpl's `•explain` and rustymath's Python `explain` do. A run puts its sink in place for the whole process (`Run::start`), and code on every thread records into it, worker threads included. A process supports one run at a time, apart from runs nested on one thread: code that another thread runs during a run records into that run. Code in rustymath and its frontends records the routes it took (`taken`), the faster routes it considered and didn't take (`declined`) and stopped traces (`stopped`), each named by a [`Route`] that BPL and Python share, into one aggregate [`Report`]. The inventory also names the routes that handle items one at a time (`Route::reports_itemwise`), which basedpl records in `•explain`'s own rows. A frontend records its own events into its sink through `with`.
use std::{any::Any, sync::{Arc, Mutex, MutexGuard, PoisonError, atomic::{AtomicUsize, Ordering::Relaxed}}};

macro_rules! routes {
    ($($id:ident: $name:literal, $taken:literal, $declined:literal, $itemwise:literal;)+) => {
        /// A numerical or frontend route, with its diagnostic name.
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum Route { $($id),+ }
        impl Route {
            pub const ALL: &'static [Self] = &[$(Self::$id),+];
            pub const fn name(self) -> &'static str { match self { $(Self::$id => $name),+ } }
            pub const fn reports_taken(self) -> bool { match self { $(Self::$id => $taken),+ } }
            pub const fn reports_decline(self) -> bool { match self { $(Self::$id => $declined),+ } }
            pub const fn reports_itemwise(self) -> bool { match self { $(Self::$id => $itemwise),+ } }
        }
    };
}
routes! {
    Trace: "trace", true, false, false;
    Fusion: "fusion", true, true, false;
    Batching: "batching", true, true, false;
    SlidingFold: "sliding fold", true, false, false;
    MappedFold: "mapped fold", true, false, false;
    MappedProduct: "mapped product", true, false, false;
    VectorKernel: "vector kernel", true, false, false;
    MappedKernel: "mapped kernel", true, false, false;
    PartialGrade: "partial grade", true, true, false;
    PartialSort: "partial sort", true, true, false;
    Ranks: "ranks", true, true, false;
    GroupCounts: "group counts", true, false, false;
    GroupFolds: "group folds", true, true, false;
    Determinant: "determinant", true, true, false;
    Permanent: "permanent", true, true, false;
    MaskedSelect: "masked select", true, false, false;
    Contraction: "contraction", true, true, false;
    Program: "program", false, true, false;
    FusedFold: "fused fold", true, true, false;
    InPlace: "in place", true, true, false;
    ItemPairs: "item pairs", false, false, true;
    ItemLanes: "item lanes", false, false, true;
    NumberParts: "number parts", false, false, true;
    Accumulation: "accumulation", false, false, true;
    NestedLabels: "nested labels", false, false, true;
    LabelIds: "label ids", false, false, true;
    InverseScan: "inverse scan", false, false, true;
    KeyGroups: "key groups", false, false, true;
    BatchCells: "batch cells", false, false, true;
    BatchedItems: "batched items", false, false, true;
    DerivativeItems: "∂ items", false, false, true;
    TracedItems: "traced items", false, false, true;
}

/// A taken route, a declined optimization or a stopped trace.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind { Taken, Declined, Stopped }
impl EventKind {
    pub const fn name(self) -> &'static str { match self { Self::Taken => "taken", Self::Declined => "declined", Self::Stopped => "stopped" } }
}

/// Calls with one kind, route and reason, and the total items they concerned.
#[derive(Clone, Debug)]
pub struct Record { pub kind: EventKind, pub route: Route, pub reason: &'static str, pub calls: usize, pub items: usize }

/// A run's events, aggregated by kind, route and reason.
#[derive(Clone, Default)]
pub struct Report { pub records: Vec<Record> }
impl Report {
    pub fn record(&mut self, event: Record) {
        match self.records.iter_mut().find(|r| r.kind == event.kind && r.route == event.route && r.reason == event.reason) {
            Some(r) => {
                r.calls += event.calls;
                r.items += event.items;
            }
            None => self.records.push(event),
        }
    }
    pub fn extend(&mut self, other: &Self) { for event in &other.records { self.record(event.clone()); } }
}

/// What a run records into. A frontend's sink also holds the frontend's own events, which it finds again through `with`.
pub trait Sink: Any + Send + Sync { fn record(&self, event: Record); }

/// The sink of the run under way, if any.
static CURRENT: Mutex<Option<Arc<dyn Sink>>> = Mutex::new(None);
/// The runs under way, nested ones included. Code outside every run pays one atomic load for each event it could record.
static RUNS: AtomicUsize = AtomicUsize::new(0);

/// Whether a run is under way.
#[inline]
pub fn running() -> bool { RUNS.load(Relaxed) > 0 }

/// A run under way, which records into its sink until it's dropped, as do calls on other threads. Dropping it, as a panic in the run does, puts back the sink of the run it interrupted, if any, so a frontend that turns a panic into an error can carry on.
pub struct Run { outer: Option<Arc<dyn Sink>> }
impl Run {
    /// Starts a run that records into `sink`.
    pub fn start(sink: Arc<dyn Sink>) -> Self { RUNS.fetch_add(1, Relaxed); Self { outer: current().replace(sink) } }
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

/// Records that code took `route`, for `items` items, into the run under way. Outside every run it costs one atomic load.
#[inline]
pub fn taken(route: Route, items: usize) { if running() { add(EventKind::Taken, route, "", items); } }
/// Records that code declined `route` for `reason`, for `items` items, into the run under way. Outside every run it costs one atomic load.
#[inline]
pub fn declined(route: Route, reason: &'static str, items: usize) { if running() { add(EventKind::Declined, route, reason, items); } }
/// Records that a trace stopped for `reason`, for `items` items, into the run under way. Outside every run it costs one atomic load.
#[inline]
pub fn stopped(reason: &'static str, items: usize) { if running() { add(EventKind::Stopped, Route::Trace, reason, items); } }

#[cold]
#[inline(never)]
fn add(kind: EventKind, route: Route, reason: &'static str, items: usize) { with(|sink| sink.record(Record { kind, route, reason, calls: 1, items })); }
