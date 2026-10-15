//! Splitting work across threads. [`pieces`] cuts a call's items into pieces of the sizes its caller gives, and [`ranges`] into as many pieces as the time of its first unit justifies. The pieces run on a Rayon pool with Rayon's default number of threads: `RAYON_NUM_THREADS` where it's set, and otherwise one for each core. [`set_threads`] changes that number for the calls made on one thread. A call made on one of a pool's threads, such as a function that another call's piece runs, doesn't split again. In the browser build, which has no threads, every call runs its pieces in turn on the calling thread.
use crate::slice::{Arg, CHUNK};
use std::{cell::Cell, mem::MaybeUninit, ops::Range, time::{Duration, Instant}};

thread_local! {
    /// The number of threads that calls made on this thread split across, or 0 for the default pool's number.
    static THREADS: Cell<usize> = const { Cell::new(0) };
}

/// Sets how many threads the calls made on this thread split their work across, and gives the number it replaces. 0 restores the default. 1 runs every call on this thread, as the speed tests do for their budgets.
pub fn set_threads(n: usize) -> usize { THREADS.replace(n) }

/// The number that [`set_threads`] last set on this thread. It starts at 0, the default.
pub fn thread_setting() -> usize { THREADS.get() }

/// How many threads the calls made on this thread split their work across.
pub fn threads() -> usize { requested().map_or(1, |n| if n == 0 { pool::get(0).current_num_threads() } else { n }) }

/// Whether calls made on this thread run serially: after `set_threads(1)`, or on a pool's worker thread.
pub(crate) fn serial() -> bool { requested().is_none() }

/// The pool that calls made on this thread split across, as its number of threads, 0 for the default. `None` where they run on this thread.
fn requested() -> Option<usize> { let n = THREADS.get(); (n != 1 && !pool::inside()).then_some(n) }

/// Runs `piece` on each range that cuts `0..n` into pieces of `size` items, the last one shorter, and gives whether every call said `true`. Every piece runs, whatever another one gives. The pieces run across at most `most` of this thread's [`threads`], on a smaller pool where `most` or the number of pieces is below the pool's threads, as [`sized`] chooses. Where either is 1, they run in order on this thread.
pub(crate) fn pieces(n: usize, size: usize, most: usize, piece: &(dyn Fn(Range<usize>) -> bool + Sync)) -> bool {
    let (count, range) = (n.div_ceil(size), |p: usize| p * size..n.min((p + 1) * size));
    if count.min(most) > 1 && let Some(threads) = requested() { return pool::run(sized(threads, count.min(most)), count, &|p| piece(range(p))); }
    (0..count).fold(true, |ok, p| piece(range(p)) & ok)
}

/// The pool that `count` pieces run on, where this thread's calls split across the pool of `threads` threads, 0 for the default. Where there are fewer pieces than that pool's threads, it's the pool of the largest power of two up to `count`. On an M5 Max, `-v` on a million items, cut into about 4 pieces, took 0.33 copies on a 4-thread pool and 0.5 to 0.8 on pools of 12 to 18 threads.
fn sized(threads: usize, count: usize) -> usize {
    let all = if threads == 0 { pool::get(0).current_num_threads() } else { threads };
    if count >= all { threads } else { 1 << count.ilog2() }
}

/// Runs `write` on each piece of `size` items that cuts `out`, the last one shorter, across at most `most` threads as [`pieces`] runs its ranges, and gives whether every call said `true`. `write(start, items)` writes `items`, the piece of `out` from `start`.
pub fn pieces_of<B: Send>(out: &mut [B], size: usize, most: usize, write: &(dyn Fn(usize, &mut [B]) -> bool + Sync)) -> bool {
    let (shared, n) = (Shared(out.as_mut_ptr()), out.len());
    // SAFETY: `pieces` gives each call its own range within `0..n`.
    pieces(n, size, most, &|range| write(range.start, unsafe { shared.part(range) }))
}

/// An output slice shared only through disjoint ranges.
struct Shared<B>(*mut B);
// SAFETY: each piece reaches only the items in its own range, so no two threads reach the same item.
unsafe impl<B: Send> Sync for Shared<B> {}
impl<B> Shared<B> {
    /// The items in `range`. SAFETY: the range must lie within the output, and no other caller may hold any of these items at the same time.
    unsafe fn part(&self, range: Range<usize>) -> &mut [B] { unsafe { std::slice::from_raw_parts_mut(self.0.add(range.start), range.len()) } }
}

/// The time on one thread that each piece of a [`ranges`] call is given at least. A call whose pieces would be shorter splits into fewer of them, because waking a thread for a shorter piece costs more than the piece saves. On an M5 Max, `v+w` on a million items, about 200 µs on one thread, took 0.51 copies split across 4 threads and 0.81 across all 18.
pub const PIECE_TIME: Duration = Duration::from_micros(30);

/// Runs `run` on disjoint ranges covering `0..n`, and gives whether every call said `true`. Every range runs, even after a failure. Each range contains whole units of `unit` items, except that the last unit may be shorter. A call of at most one unit, or one whose calls run on this thread, runs whole on this thread. A longer call times its first unit on this thread, then cuts the rest into as many pieces as that estimate gives [`PIECE_TIME`] each, at most four for each of this thread's [`threads`]. It runs them as [`pieces`] does, or runs the rest in one call on this thread where that gives fewer than two pieces. This non-generic, out-of-line scheduler keeps Rayon's machinery out of its callers. An empty call runs `0..0` once. Panics if `unit` is zero.
#[inline(never)]
pub fn ranges(n: usize, unit: usize, run: &(dyn Fn(Range<usize>) -> bool + Sync)) -> bool {
    assert!(unit > 0, "a work unit must contain at least one item");
    if n <= unit || serial() { return run(0..n); }
    let start = Instant::now();
    let ok = run(0..unit);
    let units = (n - unit).div_ceil(unit);
    let count = ((start.elapsed().as_secs_f64() * units as f64 / PIECE_TIME.as_secs_f64()) as usize).min(4 * threads()).min(units);
    if count < 2 { return ok & run(unit..n); }
    ok & pieces(n - unit, units.div_ceil(count) * unit, usize::MAX, &|range| run(unit + range.start..unit + range.end))
}

/// Runs `write(start, items)` over disjoint pieces of `out`, using [`ranges`] in units of `unit` items, and gives whether every call said `true`. An empty output calls `write` once. Panics if `unit` is zero.
#[inline(never)]
pub fn split<B: Send>(out: &mut [B], unit: usize, write: &(dyn Fn(usize, &mut [B]) -> bool + Sync)) -> bool {
    let (shared, n) = (Shared(out.as_mut_ptr()), out.len());
    // SAFETY: `ranges` gives each call its own range within `0..n`.
    ranges(n, unit, &|range| write(range.start, unsafe { shared.part(range) }))
}

/// The most cells that [`stack_with`] runs alone at its start, to time a cell.
const CELLS: usize = 8;

/// The pieces that each thread is given in a round of a [`stack_with`] call, so that a thread that finishes early takes more of the round.
const PIECES: usize = 4;

/// The time that each round of a [`stack_with`] call gives each thread, where its cells are that short. A stack stops within about a round of an error from `check`, or within one cell where a cell takes longer.
const ROUND_TIME: Duration = Duration::from_millis(20);

/// Runs `write(i, items, scratch)` on each cell `i` of a stack of `count` cells, and gives the output with whether every call said `true`. `items` is cell `i`'s part of the output: `per` items from `i * per`. Cells that run in turn on one thread share one `S` for their scratch buffers, such as an `LuWork`. `check` runs on this thread before the first cell and between rounds, and its error stops the call.
///
/// The first cells run alone on this thread, to time a cell: up to `CELLS` of them, or [`PIECE_TIME`] of work, because one cell alone reads high from a cold start. Each later round holds about `ROUND_TIME` of cells for each of this thread's [`threads`], at least one each. A round's cells split into about `PIECES` pieces for each thread, each of at least `PIECE_TIME` of work. A cell that runs on this thread can split its own work, and one on another thread doesn't. On all 18 threads of an M5 Max, splitting stacks of 8 to 16 matrices of 128² to 512² was 1.5 to 5 times as fast as running their cells in turn, each split across the threads. Three inverses of 1024² were 25% slower. The browser build has no clock to time a cell, and runs the cells in turn with `check` before each.
pub fn stack_with<B: Default + Send, S: Default, E>(
    count: usize, per: usize, write: &(dyn Fn(usize, &mut [B], &mut S) -> bool + Sync), mut check: impl FnMut() -> Result<(), E>) -> Result<(Vec<B>, bool), E> {
    let mut out: Vec<B> = (0..count * per).map(|_| B::default()).collect();
    let (threads, mut ok, mut scratch) = (threads(), true, S::default());
    if cfg!(target_arch = "wasm32") {
        for i in 0..count {
            check()?;
            ok &= write(i, &mut out[i * per..][..per], &mut scratch);
        }
        return Ok((out, ok));
    }
    check()?;
    let (start, mut timed) = (Instant::now(), 0);
    while timed < count && (timed == 0 || (timed < CELLS && start.elapsed() < PIECE_TIME)) {
        ok &= write(timed, &mut out[timed * per..][..per], &mut scratch);
        timed += 1;
    }
    let cell = start.elapsed().as_secs_f64().max(1e-9) / timed as f64;
    let round = ((ROUND_TIME.as_secs_f64() / cell) as usize).max(1).saturating_mul(threads);
    let least = (PIECE_TIME.as_secs_f64() / cell).ceil() as usize;
    let shared = Shared(out.as_mut_ptr());
    for first in (timed..count).step_by(round) {
        check()?;
        let cells = round.min(count - first);
        ok &= pieces(cells, cells.div_ceil(threads * PIECES).max(least).max(1), usize::MAX, &|range| {
            let mut scratch = S::default();
            // SAFETY: `pieces` gives each call its own range of cells, and each cell has its own `per` items.
            range.map(|c| first + c).fold(true, |ok, i| write(i, unsafe { shared.part(i * per..(i + 1) * per) }, &mut scratch) & ok)
        });
    }
    Ok((out, ok))
}

/// Appends `n` items to `out`, and gives whether every call of `write` said `true`. `write(range, slots)` fills `slots` with the items in `range`, in order. The calls split across threads as [`split`] splits a call, in units of `unit` items. A call that leaves any of its slots unfilled panics, and `out` keeps its length.
pub fn extended<B: Copy + Send>(out: &mut Vec<B>, n: usize, unit: usize, write: &(dyn Fn(Range<usize>, &mut Slots<'_, B>) -> bool + Sync)) -> bool {
    crate::slice::appended(out, n, |slots| split(slots, unit, &|start, slots| filled(slots, |slots| write(start..start + slots.slots.len(), slots))))
}

/// `write`'s result for `slots`, which it fills in order through one [`Slots`]. Panics where it leaves any unfilled.
fn filled<B, T>(slots: &mut [MaybeUninit<B>], write: impl FnOnce(&mut Slots<'_, B>) -> T) -> T {
    let mut slots = Slots { slots, written: 0 };
    let result = write(&mut slots);
    assert_eq!(slots.written, slots.slots.len(), "a writer fills every slot it gets");
    result
}

/// The slots of a vector that a writer fills in order: one call's part of an [`extended`] vector, or the items that [`Slots::append`] adds.
pub struct Slots<'a, B> { slots: &'a mut [MaybeUninit<B>], written: usize }

impl<B> Slots<'_, B> {
    /// Appends `n` items to `out` on this thread, and gives `write`'s result. `write` fills the slots in order. Where it leaves any unfilled, `append` panics and `out` keeps its length.
    pub fn append<T>(out: &mut Vec<B>, n: usize, write: impl FnOnce(&mut Slots<'_, B>) -> T) -> T {
        crate::slice::appended(out, n, |slots| filled(slots, write))
    }
    /// Writes the `n` results of [`pairs`](crate::pairs) after the items already written, on this thread, and gives whether every result succeeded. Panics where they don't fit.
    pub fn pairs<X: Copy, Y: Copy>(&mut self, x: Arg<'_, X>, y: Arg<'_, Y>, n: usize, f: impl Fn(X, Y) -> (B, bool)) -> bool {
        self.fill(n, |slots| crate::slice::pairs_into(x, y, n, slots, f))
    }
    /// `write`'s result for the next `n` slots, which it must fill. They then count as written. Panics where they don't fit.
    pub(crate) fn fill(&mut self, n: usize, write: impl FnOnce(&mut [MaybeUninit<B>]) -> bool) -> bool {
        let ok = write(&mut self.slots[self.written..self.written + n]);
        self.written += n;
        ok
    }
}

/// Runs `write` on each `CHUNK` of `slots` in turn, the results for the items from `start`, and gives whether every call said `true`. A slice form that makes several passes over buffers of its own keeps them in cache this way.
#[inline]
pub(crate) fn in_chunks<B>(start: usize, slots: &mut [MaybeUninit<B>], write: impl Fn(usize, &mut [MaybeUninit<B>]) -> bool) -> bool {
    let (n, mut ok) = (slots.len(), true);
    for at in (0..n).step_by(CHUNK) { ok &= write(start + at, &mut slots[at..n.min(at + CHUNK)]) }
    ok
}

#[cfg(not(target_arch = "wasm32"))]
mod pool {
    use std::sync::{Arc, Mutex};

    /// The pools built in this process, each with its number of threads, 0 for the default, and the id of the process that built them. A child that `fork` creates has none of its parent's threads, so it builds pools of its own.
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
        if let Some((_, pool)) = pools.1.iter().find(|(k, _)| *k == n) { return pool.clone(); }
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
    impl Pool { pub(super) fn current_num_threads(&self) -> usize { 1 } }
    pub(super) fn get(_: usize) -> Pool { Pool }
    pub(super) fn inside() -> bool { true }
    pub(super) fn run(_: usize, count: usize, f: &dyn Fn(usize) -> bool) -> bool { (0..count).fold(true, |ok, p| f(p) & ok) }
}
