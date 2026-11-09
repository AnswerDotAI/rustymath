//! Errors from rustymath's functions on arrays. Each error has a kind, and a message that says what went wrong.
use std::borrow::Cow;

/// What went wrong.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    /// Lengths that don't agree.
    Length,
    /// An argument of the wrong rank.
    Rank,
    /// A position outside an axis.
    Index,
    /// A result past an array limit, such as `layout::MAX_GENERATED_ELEMENTS`.
    Limit,
    /// An argument outside the function's domain.
    Domain,
    /// An integer result that doesn't fit its dtype.
    Overflow,
    /// A computation that the installed interrupt check stopped, as at a Ctrl-C ([`interrupt`](crate::interrupt)).
    Interrupt,
    /// A computation that the installed interrupt check stopped at a deadline.
    Timeout,
}

macro_rules! reporters {
    ($($kind:ident: $message:literal),+ $(,)?) => {
        impl Kind {
            pub fn reporter(self) -> extern "C" fn() {
                match self { $(Self::$kind => {
                    extern "C" fn report() { record(Error::new(Kind::$kind, $message)); }
                    report
                }),+ }
            }
        }
    };
}
reporters! {
    Length: "lengths do not agree", Rank: "argument has the wrong rank", Index: "read or write position is outside its source",
    Limit: "array limit exceeded", Domain: "argument is outside the function's domain", Overflow: "integer arithmetic overflowed",
    Interrupt: "computation interrupted", Timeout: "computation timed out",
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error { pub kind: Kind, pub message: Cow<'static, str> }

impl Error { pub fn new(kind: Kind, message: impl Into<Cow<'static, str>>) -> Self { Self { kind, message: message.into() } } }

impl std::fmt::Display for Error { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(&self.message) } }

impl std::error::Error for Error {}

/// The first recorded numerical error and its count. The parallel runtime installs the caller's active slot on workers; `checked` supplies one for a standalone operation.
#[derive(Clone, Default)]
pub struct ErrorSlot(std::sync::Arc<Errors>);

#[derive(Default)]
struct Errors { count: std::sync::atomic::AtomicU64, first: std::sync::Mutex<Option<Error>> }

thread_local! { static ACTIVE: std::cell::RefCell<Option<ErrorSlot>> = const { std::cell::RefCell::new(None) }; }

struct Restore(Option<ErrorSlot>);
impl Drop for Restore { fn drop(&mut self) { ACTIVE.with(|active| active.replace(self.0.take())); } }

impl ErrorSlot {
    pub fn new() -> Self { Self::default() }
    pub fn count(&self) -> u64 { self.0.count.load(std::sync::atomic::Ordering::Relaxed) }
    pub fn pending(&self) -> bool { self.0.first.lock().unwrap().is_some() }
    pub fn take(&self) -> Option<Error> { self.0.first.lock().unwrap().take() }
    /// Captured by the parallel runtime before it starts workers.
    pub fn current() -> Option<Self> { ACTIVE.with(|active| active.borrow().clone()) }
    pub fn within<R>(&self, run: impl FnOnce() -> R) -> R {
        let (mut run, mut result) = (Some(run), None);
        self.enter(&mut || result = run.take().map(|run| run()));
        result.expect("the run ran")
    }
    fn enter(&self, run: &mut dyn FnMut()) { let _restore = Restore(ACTIVE.with(|active| active.replace(Some(self.clone())))); run() }
}

/// Record in the active read's slot; standalone scalar maths has no slot.
pub fn record(error: Error) {
    ACTIVE.with(|active| {
        if let Some(slot) = active.borrow().as_ref() {
            slot.0.count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut first = slot.0.first.lock().unwrap();
            if first.is_none() { *first = Some(error); }
        }
    });
}

/// Checked eager entry for Rust and Python callers.
/// Returns direct failures and, when no slot is active, the first recorded failure. With an
/// active slot, recorded failures remain for its enclosing operation to collect.
pub fn checked<R>(run: impl FnOnce() -> Result<R, Error>) -> Result<R, Error> {
    let (mut run, mut result) = (Some(run), None);
    let recorded = recorded(&mut || result = run.take().map(|run| run()));
    let result = result.expect("the run ran")?;
    recorded.map_or(Ok(result), Err)
}
/// Runs `run` in the thread's spare error slot unless one is active, giving the first error recorded in it. Every caller of [`checked`] shares this one non-generic copy. The spare is cleared before each run: a run that unwound can leave its error in it.
fn recorded(run: &mut dyn FnMut()) -> Option<Error> {
    if ACTIVE.with(|active| active.borrow().is_some()) { run(); return None; }
    thread_local! { static SPARE: ErrorSlot = ErrorSlot::new(); }
    SPARE.with(|slot| {
        slot.take();
        slot.enter(run);
        slot.take()
    })
}
