//! Route events captured across every thread while a frontend's run is active, worker threads included. A process supports one capture at a time, apart from captures nested on one thread. BPL and Python share the route IDs and aggregate report.
use std::{any::Any, sync::{atomic::{AtomicUsize, Ordering::Relaxed}, Arc, Mutex, MutexGuard, PoisonError}};

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
    InPlace: "in place", false, true, false;
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

/// A taken route, an optional optimization declined, a trace stopped, or an itemwise fallback.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind { Taken, Declined, Stopped, Itemwise }
impl EventKind { pub const fn name(self) -> &'static str { match self { Self::Taken => "taken", Self::Declined => "declined", Self::Stopped => "stopped", Self::Itemwise => "itemwise" } } }

/// Calls with one kind, route and reason, and the total items they concerned.
#[derive(Clone, Debug)]
pub struct Record { pub kind: EventKind, pub route: Route, pub reason: &'static str, pub calls: usize, pub items: usize }

/// A run's events, aggregated by kind, route and reason.
#[derive(Clone, Default)]
pub struct Report { pub records: Vec<Record> }
impl Report {
    pub fn record(&mut self, event: Record) {
        match self.records.iter_mut().find(|r| r.kind == event.kind && r.route == event.route && r.reason == event.reason) {
            Some(r) => { r.calls += event.calls; r.items += event.items; }
            None => self.records.push(event),
        }
    }
    pub fn extend(&mut self, other: &Self) { for event in &other.records { self.record(event.clone()); } }
}

/// A frontend's run store. Its other events remain accessible through `with`.
pub trait Sink: Any + Send + Sync { fn record(&self, event: Record); }

static CURRENT: Mutex<Option<Arc<dyn Sink>>> = Mutex::new(None);
static RUNS: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn running() -> bool { RUNS.load(Relaxed) > 0 }

/// Records into a sink until dropped, restoring any enclosing run even when code unwinds. Calls on other threads record into the same process-wide sink.
pub struct Run { outer: Option<Arc<dyn Sink>> }
impl Run {
    pub fn start(sink: Arc<dyn Sink>) -> Self {
        RUNS.fetch_add(1, Relaxed);
        Self { outer: current().replace(sink) }
    }
    pub fn outer(&self) -> Option<&Arc<dyn Sink>> { self.outer.as_ref() }
}
impl Drop for Run {
    fn drop(&mut self) { *current() = self.outer.take(); RUNS.fetch_sub(1, Relaxed); }
}
fn current() -> MutexGuard<'static, Option<Arc<dyn Sink>>> { CURRENT.lock().unwrap_or_else(PoisonError::into_inner) }

/// Calls `f` on the active sink after releasing the process-wide lock.
pub fn with<T>(f: impl FnOnce(&Arc<dyn Sink>) -> T) -> Option<T> {
    if !running() { return None; }
    let sink = current().clone();
    sink.as_ref().map(f)
}

#[inline]
pub fn taken(route: Route, items: usize) { if running() { add(EventKind::Taken, route, "", items); } }
#[inline]
pub fn declined(route: Route, reason: &'static str, items: usize) { if running() { add(EventKind::Declined, route, reason, items); } }
#[inline]
pub fn stopped(reason: &'static str, items: usize) { if running() { add(EventKind::Stopped, Route::Trace, reason, items); } }
#[inline]
pub fn itemwise(route: Route, items: usize) { if running() { add(EventKind::Itemwise, route, "", items); } }

#[cold]
#[inline(never)]
fn add(kind: EventKind, route: Route, reason: &'static str, items: usize) { with(|sink| sink.record(Record { kind, route, reason, calls: 1, items })); }
