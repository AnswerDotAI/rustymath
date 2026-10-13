//! Splitting work across threads. `pieces` cuts a call's items into pieces of the sizes its caller gives. Each result of an
//! elementwise call comes from its own items alone and doesn't depend on where the cuts fall or on the number of threads. The pieces
//! run on a Rayon pool with Rayon's default number of threads:
//! `RAYON_NUM_THREADS` where it's set, and otherwise one for each core. [`set_threads`] changes that number for the calls made on one
//! thread. A call made on one of a pool's threads, such as a function that another call's piece runs, doesn't split again. In the
//! browser build, which has no threads, every call runs its pieces in turn on the calling thread.
use crate::slice::CHUNK;
use std::{cell::Cell, mem::MaybeUninit, ops::Range, time::{Duration, Instant}};

thread_local! {
    /// The number of threads that calls made on this thread split across, or 0 for the default pool's number.
    static THREADS: Cell<usize> = const { Cell::new(0) };
}

/// Sets how many threads the calls made on this thread split their work across, and gives the number it replaces. 0 restores the
/// default. 1 runs every call on this thread, as the speed tests do for their budgets.
pub fn set_threads(n: usize) -> usize { THREADS.replace(n) }

/// How many threads the calls made on this thread split their work across.
pub fn threads() -> usize { requested().map_or(1, |n| if n == 0 { pool::get(0).current_num_threads() } else { n }) }

/// The pool that calls made on this thread split across, as its number of threads, 0 for the default. `None` where they run on this
/// thread.
fn requested() -> Option<usize> {
    let n = THREADS.get();
    (n != 1 && !pool::inside()).then_some(n)
}

/// Runs `piece` on each range that cuts `0..n` into pieces of `size` items, the last one shorter, and gives whether every call said
/// `true`. Every piece runs, whatever another one gives. The pieces run across this thread's [`threads`], or in order on this thread.
pub(crate) fn pieces(n: usize, size: usize, piece: &(dyn Fn(Range<usize>) -> bool + Sync)) -> bool {
    let (count, range) = (n.div_ceil(size), |p: usize| p * size..n.min((p + 1) * size));
    match requested() {
        Some(threads) if count > 1 => pool::run(threads, count, &|p| piece(range(p))),
        _ => (0..count).fold(true, |ok, p| piece(range(p)) & ok),
    }
}

/// Runs `write` on each piece of `out` that [`pieces`] cuts, and gives whether every call said `true`. `write(start, slots)` writes
/// `slots`, the results for the items from `start`.
pub(crate) fn pieces_of<B: Send>(out: &mut [MaybeUninit<B>], size: usize, write: &(dyn Fn(usize, &mut [MaybeUninit<B>]) -> bool + Sync)) -> bool {
    /// `out`'s slots, shared by the pieces.
    struct Shared<B>(*mut MaybeUninit<B>);
    // SAFETY: each piece reaches only the slots in its own range, so no two threads reach the same slot.
    unsafe impl<B: Send> Sync for Shared<B> {}
    impl<B> Shared<B> {
        /// The slots in `range`.
        /// SAFETY: no other caller may hold any of these slots at the same time.
        unsafe fn part(&self, range: Range<usize>) -> &mut [MaybeUninit<B>] { unsafe { std::slice::from_raw_parts_mut(self.0.add(range.start), range.len()) } }
    }
    let (shared, n) = (Shared(out.as_mut_ptr()), out.len());
    // SAFETY: `pieces` gives each call its own range within `0..n`.
    pieces(n, size, &|range| write(range.start, unsafe { shared.part(range) }))
}

/// How long the rest of an elementwise call would take on one thread, judged from its first `CHUNK`, before the call splits. Handing a
/// shorter call's pieces to the pool would cost more than it saves.
const WORTH_SPLITTING: Duration = Duration::from_micros(100);

/// Runs `write` over `out` for an elementwise call, and gives whether every call said `true`. `write(start, slots)` writes `slots`, the
/// results for the items from `start`, each from its own items alone. A call of up to two `CHUNK`s, or one whose calls run on this
/// thread, runs whole on this thread. A longer call times its first `CHUNK` on this thread. It splits the rest as [`pieces_of`] cuts
/// them, only where they would take longer than [`WORTH_SPLITTING`] on one thread. `write` is a `&dyn Fn`, and this function is out of
/// line, so each caller's loop compiles once, however the call runs.
#[inline(never)]
pub(crate) fn elementwise<B: Send>(out: &mut [MaybeUninit<B>], write: &(dyn Fn(usize, &mut [MaybeUninit<B>]) -> bool + Sync)) -> bool {
    if out.len() <= 2 * CHUNK || requested().is_none() { return write(0, out) }
    let (first, rest) = out.split_at_mut(CHUNK);
    let start = Instant::now();
    let ok = write(0, first);
    if start.elapsed().mul_f64(rest.len() as f64 / CHUNK as f64) < WORTH_SPLITTING { return ok & write(CHUNK, rest) }
    let size = rest.len().div_ceil(4 * threads()).next_multiple_of(CHUNK);
    ok & pieces_of(rest, size, &|start, slots| write(CHUNK + start, slots))
}

/// Runs `write` on each `CHUNK` of `slots` in turn, the results for the items from `start`, and gives whether every call said `true`. A
/// slice form that makes several passes over buffers of its own keeps them in cache this way.
#[inline]
pub(crate) fn in_chunks<B>(start: usize, slots: &mut [MaybeUninit<B>], write: impl Fn(usize, &mut [MaybeUninit<B>]) -> bool) -> bool {
    let (n, mut ok) = (slots.len(), true);
    for at in (0..n).step_by(CHUNK) { ok &= write(start + at, &mut slots[at..n.min(at + CHUNK)]) }
    ok
}

#[cfg(not(target_arch = "wasm32"))]
mod pool {
    use std::sync::{Arc, Mutex};

    /// The pools built in this process, each with its number of threads, 0 for the default, and the id of the process that built them.
    /// A child that `fork` creates has none of its parent's threads, so it builds pools of its own.
    static POOLS: Mutex<(u32, Vec<(usize, Arc<rayon::ThreadPool>)>)> = Mutex::new((0, Vec::new()));

    /// The pool of `n` threads, 0 for Rayon's default number, built the first time a call needs it.
    pub(super) fn get(n: usize) -> Arc<rayon::ThreadPool> {
        let mut pools = POOLS.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
        let id = std::process::id();
        if pools.0 != id {
            // Dropping a parent's pool would signal threads that this process doesn't have.
            std::mem::forget(std::mem::take(&mut pools.1));
            pools.0 = id;
        }
        if let Some((_, pool)) = pools.1.iter().find(|(k, _)| *k == n) { return pool.clone() }
        let pool = Arc::new(rayon::ThreadPoolBuilder::new().num_threads(n).build().expect("a thread pool"));
        pools.1.push((n, pool.clone()));
        pool
    }

    /// Whether this thread is one of a pool's.
    pub(super) fn inside() -> bool { rayon::current_thread_index().is_some() }

    /// Runs `f` on each of `count` pieces across the pool of `threads` threads, and gives whether every call said `true`.
    pub(super) fn run(threads: usize, count: usize, f: &(dyn Fn(usize) -> bool + Sync)) -> bool {
        use rayon::prelude::*;
        get(threads).install(|| (0..count).into_par_iter().map(f).reduce(|| true, |a, b| a & b))
    }
}

#[cfg(target_arch = "wasm32")]
mod pool {
    pub(super) struct Pool;
    impl Pool {
        pub(super) fn current_num_threads(&self) -> usize { 1 }
    }
    pub(super) fn get(_: usize) -> Pool { Pool }
    pub(super) fn inside() -> bool { true }
    pub(super) fn run(_: usize, count: usize, f: &dyn Fn(usize) -> bool) -> bool { (0..count).fold(true, |ok, p| f(p) & ok) }
}
