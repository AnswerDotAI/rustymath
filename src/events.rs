//! Producer events shared by BPL's `•explain` and rustymath's Python `explain`. Capture is process-wide, including workers, with nesting on one thread. Payloads own their diagnostic data and never retain runtime values.
use std::sync::{Arc, Mutex, MutexGuard, PoisonError, atomic::{AtomicUsize, Ordering::Relaxed}};

macro_rules! routes {
    ($($id:ident: $name:literal;)+) => {
        #[derive(Clone, Copy, Debug, Eq, PartialEq)]
        pub enum Route { $($id),+ }
        impl Route {
            pub const ALL: &'static [Self] = &[$(Self::$id),+];
            pub const REQUIRED: &'static [Self] = &[Self::Compiled, Self::FusedFold, Self::Trace];
            pub const fn name(self) -> &'static str { match self { $(Self::$id => $name),+ } }
        }
    };
}
routes! {
    Trace: "trace";
    Fusion: "fusion";
    Batching: "batching";
    SlidingFold: "sliding fold";
    MappedFold: "mapped fold";
    MappedProduct: "mapped product";
    VectorKernel: "vector kernel";
    MappedKernel: "mapped kernel";
    PartialGrade: "partial grade";
    PartialSort: "partial sort";
    Ranks: "ranks";
    Accuracy: "accuracy rule";
    KeyClasses: "key classes";
    Determinant: "determinant";
    Permanent: "permanent";
    MaskedSelect: "masked select";
    Contraction: "contraction";
    Program: "program";
    FusedFold: "fused fold";
    Compiled: "compiled";
    InPlace: "in place";
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind { Taken, Declined, Stopped }
impl EventKind {
    pub const fn name(self) -> &'static str { match self { Self::Taken => "taken", Self::Declined => "declined", Self::Stopped => "stopped" } }
}

#[derive(Clone, Debug)]
pub struct Record { pub kind: EventKind, pub route: Route, pub reason: &'static str, pub calls: usize, pub items: usize, pub site: Option<String> }

#[derive(Clone, Debug)]
pub enum Event {
    Route(Record), Fused { glyphs: String, items: usize, blocks: usize, runs: usize, copied: usize },
    Stored { glyphs: String, items: usize, reason: &'static str, partway: bool },
    Forced { function: String, line: usize, column: usize, items: usize, reason: &'static str, site: Option<String> },
    Program { event: &'static str, steps: usize, compiled: usize, blocks: usize, listing: String }, Fast { rule: &'static str, function: String },
    Fallback { reason: String, items: usize }, Accumulation { items: usize, size: usize, path: &'static str },
    Mixed { code: String, line: usize, column: usize, target: &'static str, written: &'static str, items: usize },
    Compile { glyphs: String, microseconds: usize }
}
#[derive(Clone, Copy)]
pub enum Column { Text, Count, Truth }
pub enum Field<'a> { Text(&'a str), Count(usize), Truth(bool) }
impl Field<'_> {
    pub fn text(&self) -> &str { match self { Self::Text(t) => t, _ => unreachable!("text column") } }
    pub fn count(&self) -> usize { match self { Self::Count(n) => *n, _ => unreachable!("count column") } }
    pub fn truth(&self) -> bool { match self { Self::Truth(t) => *t, _ => unreachable!("truth column") } }
}
pub const TABLES: &[(&str, &[(&str, Column)])] = {
    use Column::*;
    &[
        ("fused", &[("glyphs", Text), ("items", Count), ("blocks", Count), ("runs", Count), ("copied", Count)]),
        ("stored", &[("glyphs", Text), ("items", Count), ("reason", Text), ("partway", Truth)]),
        ("forced", &[("function", Text), ("line", Count), ("column", Count), ("items", Count), ("reason", Text), ("site", Text)]),
        ("programs", &[("event", Text), ("steps", Count), ("compiled", Count), ("blocks", Count), ("listing", Text)]),
        ("stopped", &[("reason", Text), ("calls", Count), ("items", Count), ("site", Text)]), ("fast", &[("rule", Text), ("function", Text)]),
        ("fallbacks", &[("reason", Text), ("items", Count)]), ("declined", &[("route", Text), ("reason", Text), ("calls", Count), ("items", Count)]),
        ("accumulations", &[("items", Count), ("size", Count), ("path", Text)]),
        ("mixed", &[("code", Text), ("line", Count), ("column", Count), ("target", Text), ("written", Text), ("items", Count)]),
        ("compiles", &[("glyphs", Text), ("microseconds", Count)]), ("routes", &[("route", Text), ("calls", Count), ("items", Count)]),
    ]
};
impl Event {
    pub const fn table(&self) -> usize {
        match self {
            Self::Fused { .. } => 0, Self::Stored { .. } => 1, Self::Forced { .. } => 2, Self::Program { .. } => 3,
            Self::Route(r) => match r.kind { EventKind::Stopped => 4, EventKind::Declined => 7, EventKind::Taken => 11 }, Self::Fast { .. } => 5,
            Self::Fallback { .. } => 6, Self::Accumulation { .. } => 8, Self::Mixed { .. } => 9, Self::Compile { .. } => 10
        }
    }
    pub const fn name(&self) -> &'static str { TABLES[self.table()].0 }
    pub fn route(&self) -> Option<&Record> { match self { Self::Route(r) => Some(r), _ => None } }
    pub const fn slow(&self) -> bool {
        matches!(self, Self::Stored { partway: true, .. } | Self::Forced { .. } | Self::Fallback { .. } | Self::Route(Record { kind: EventKind::Stopped, .. }))
    }
    pub fn fields(&self) -> Vec<Field<'_>> {
        use Field::*;
        match self {
            Self::Route(r) => match r.kind {
                EventKind::Taken => vec![Text(r.route.name()), Count(r.calls), Count(r.items)],
                EventKind::Declined => vec![Text(r.route.name()), Text(r.reason), Count(r.calls), Count(r.items)],
                EventKind::Stopped => vec![Text(r.reason), Count(r.calls), Count(r.items), Text(r.site.as_deref().unwrap_or_default())],
            },
            Self::Fused { glyphs, items, blocks, runs, copied } => vec![Text(glyphs), Count(*items), Count(*blocks), Count(*runs), Count(*copied)],
            Self::Stored { glyphs, items, reason, partway } => vec![Text(glyphs), Count(*items), Text(reason), Truth(*partway)],
            Self::Forced { function, line, column, items, reason, site } => {
                vec![Text(function), Count(*line), Count(*column), Count(*items), Text(reason), Text(site.as_deref().unwrap_or_default())]
            }
            Self::Program { event, steps, compiled, blocks, listing } => vec![Text(event), Count(*steps), Count(*compiled), Count(*blocks), Text(listing)],
            Self::Fast { rule, function } => vec![Text(rule), Text(function)],
            Self::Fallback { reason, items } => vec![Text(reason), Count(*items)],
            Self::Accumulation { items, size, path } => vec![Count(*items), Count(*size), Text(path)],
            Self::Mixed { code, line, column, target, written, items } => {
                vec![Text(code), Count(*line), Count(*column), Text(target), Text(written), Count(*items)]
            }
            Self::Compile { glyphs, microseconds } => vec![Text(glyphs), Count(*microseconds)],
        }
    }
}

#[derive(Clone, Default)]
pub struct Report { pub records: Vec<Event> }
impl Report {
    pub fn record(&mut self, event: Event) {
        if let Event::Route(r) = &event {
            if let Some(existing) = self.records.iter_mut().find_map(|event| match event {
                Event::Route(existing) if existing.kind == r.kind && existing.route == r.route && existing.reason == r.reason && existing.site == r.site => {
                    Some(existing)
                }
                _ => None,
            }) {
                existing.calls += r.calls;
                existing.items += r.items;
                return;
            }
        }
        self.records.push(event);
    }
    pub fn extend(&mut self, other: &Self) { for event in &other.records { self.record(event.clone()); } }
    pub fn routes(&self) -> impl Iterator<Item = &Record> { self.records.iter().filter_map(Event::route) }
}

pub trait Sink: Send + Sync {
    fn sites(&self) -> bool { false }
    fn record(&self, event: Event);
}

static CURRENT: Mutex<Option<Arc<dyn Sink>>> = Mutex::new(None);
static RUNS: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub fn running() -> bool { RUNS.load(Relaxed) > 0 }

/// Restores the outer capture when dropped, including during unwinding.
pub struct Run { outer: Option<Arc<dyn Sink>> }
impl Run {
    pub fn start(sink: Arc<dyn Sink>) -> Self { RUNS.fetch_add(1, Relaxed); Self { outer: current().replace(sink) } }
    pub fn outer(&self) -> Option<&Arc<dyn Sink>> { self.outer.as_ref() }
}
impl Drop for Run {
    fn drop(&mut self) {
        *current() = self.outer.take();
        RUNS.fetch_sub(1, Relaxed);
    }
}
fn current() -> MutexGuard<'static, Option<Arc<dyn Sink>>> { CURRENT.lock().unwrap_or_else(PoisonError::into_inner) }

/// Calls `f` after releasing the capture registry's lock.
#[inline]
pub fn with<T>(f: impl FnOnce(&Arc<dyn Sink>) -> T) -> Option<T> {
    if !running() { return None; }
    let sink = current().clone();
    sink.as_ref().map(f)
}

/// Builds an owned event outside capture locks. The argument requests optional sites.
#[inline]
pub fn emit(make: impl FnOnce(bool) -> Event) {
    with(|sink| {
        let event = make(sink.sites());
        sink.record(event);
    });
}

#[inline]
pub fn taken(route: Route, items: usize) { emit(|_| Event::Route(Record { kind: EventKind::Taken, route, reason: "", calls: 1, items, site: None })); }
#[inline]
pub fn declined(route: Route, reason: &'static str, items: usize) {
    emit(|_| Event::Route(Record { kind: EventKind::Declined, route, reason, calls: 1, items, site: None }));
}
#[inline]
pub fn stopped(reason: &'static str, items: usize, site: impl FnOnce() -> Option<String>) {
    emit(|sites| Event::Route(Record { kind: EventKind::Stopped, route: Route::Trace, reason, calls: 1, items, site: if sites { site() } else { None } }));
}
