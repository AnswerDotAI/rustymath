//! Diagnostic capture shared by mathematical kernels and their callers. Rows own diagnostic data and never retain runtime values. Capture includes worker threads and permits nesting on one thread.
use std::{borrow::Cow, sync::{Arc, Mutex, MutexGuard, PoisonError, atomic::{AtomicUsize, Ordering::Relaxed}}};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Route { name: &'static str, taken: bool, declined: bool, itemwise: bool }
impl Route {
    pub const fn new(name: &'static str, taken: bool, declined: bool, itemwise: bool) -> Self { Self { name, taken, declined, itemwise } }
    pub const fn name(self) -> &'static str { self.name }
    pub const fn reports_taken(self) -> bool { self.taken }
    pub const fn reports_decline(self) -> bool { self.declined }
    pub const fn reports_itemwise(self) -> bool { self.itemwise }
}
macro_rules! routes {
    ($($id:ident: $name:literal, $declined:literal;)+) => {
        #[allow(non_upper_case_globals)]
        impl Route {
            pub const ALL: &'static [Self] = &[$(Self::$id),+];
            $(pub const $id: Self = Self::new($name, true, $declined, false);)+
        }
    };
}
routes! {
    SlidingFold: "sliding fold", false;
    MappedFold: "mapped fold", false;
    AlternatedFold: "alternated fold", false;
    MappedProduct: "mapped product", false;
    VectorKernel: "vector kernel", false;
    MappedKernel: "mapped kernel", false;
    Contraction: "contraction", true;
    InPlace: "in place", true;
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EventKind { Taken, Declined, Stopped }
impl EventKind {
    pub const fn name(self) -> &'static str { match self { Self::Taken => "taken", Self::Declined => "declined", Self::Stopped => "stopped" } }
}

#[derive(Clone, Debug)]
pub struct Record { pub kind: EventKind, pub route: Route, pub reason: &'static str, pub calls: usize, pub items: usize, pub site: Option<String> }

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Column { Text, Count, Truth }
pub type Table = (&'static str, &'static [(&'static str, Column)]);
pub const ROUTES: Table = ("routes", &[("route", Column::Text), ("calls", Column::Count), ("items", Column::Count)]);
pub const DECLINED: Table = ("declined", &[("route", Column::Text), ("reason", Column::Text), ("calls", Column::Count), ("items", Column::Count)]);
pub const STOPPED: Table = ("stopped", &[("reason", Column::Text), ("calls", Column::Count), ("items", Column::Count), ("site", Column::Text)]);

#[derive(Clone, Debug)]
pub enum Field<'a> { Text(Cow<'a, str>), Count(usize), Truth(bool) }
impl Field<'_> {
    pub fn text(&self) -> &str { match self { Self::Text(t) => t, _ => unreachable!("text column") } }
    pub fn count(&self) -> usize { match self { Self::Count(n) => *n, _ => unreachable!("count column") } }
    pub fn truth(&self) -> bool { match self { Self::Truth(t) => *t, _ => unreachable!("truth column") } }
    fn borrowed(&self) -> Field<'_> {
        match self { Self::Text(t) => Field::Text(t.as_ref().into()), Self::Count(n) => Field::Count(*n), Self::Truth(t) => Field::Truth(*t) }
    }
}

#[derive(Clone, Debug)]
pub enum Event { Route(Record), Row { table: &'static Table, fields: Vec<Field<'static>>, label: Option<usize> } }
impl Event {
    pub fn table(&self) -> &'static Table {
        match self {
            Self::Route(r) => match r.kind { EventKind::Stopped => &STOPPED, EventKind::Declined => &DECLINED, EventKind::Taken => &ROUTES },
            Self::Row { table, .. } => table,
        }
    }
    pub fn name(&self) -> &'static str { self.table().0 }
    pub fn route(&self) -> Option<&Record> { match self { Self::Route(r) => Some(r), _ => None } }
    /// The producer's designated text label, if this event has one.
    pub fn slow_name(&self) -> Option<&str> {
        match self {
            Self::Route(Record { kind: EventKind::Stopped, reason, .. }) => Some(*reason),
            Self::Row { fields, label: Some(i), .. } => Some(fields[*i].text()), _ => None,
        }
    }
    pub fn fields(&self) -> Vec<Field<'_>> {
        use Field::*;
        match self {
            Self::Route(r) => match r.kind {
                EventKind::Taken => vec![Text(r.route.name().into()), Count(r.calls), Count(r.items)],
                EventKind::Declined => vec![Text(r.route.name().into()), Text(r.reason.into()), Count(r.calls), Count(r.items)],
                EventKind::Stopped => vec![Text(r.reason.into()), Count(r.calls), Count(r.items), Text(r.site.as_deref().unwrap_or_default().into())],
            },
            Self::Row { fields, .. } => fields.iter().map(Field::borrowed).collect(),
        }
    }
}

#[derive(Clone, Default)]
pub struct Report { pub records: Vec<Event> }
impl Report {
    pub fn record(&mut self, event: Event) {
        if let Event::Route(r) = &event {
            if let Some(existing) = self.records.iter_mut().find_map(|event| match event {
                Event::Route(existing) if existing.kind == r.kind && existing.route == r.route && existing.reason == r.reason && existing.site == r.site => Some(existing),
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
