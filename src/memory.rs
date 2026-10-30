//! The process's memory guard. A binary that registers [`Guard`] as its global allocator counts the bytes it holds in large allocations, of at least [`LARGE`] bytes each. The process's limit is half of physical memory, unless [`set_limit`] sets another. The guard refuses an allocation that would take its count past the limit. It writes one line to stderr naming the limit. Rust then aborts the process, before the machine runs out of memory. Before a function makes an array, [`generated_len`](crate::layout::generated_len) asks [`admit`], which gives a Limit error where the array would pass the limit. [`peak_held`] gives the most bytes that a call held at once, which the speed tests count in a case's footprint. On Linux, the guard also asks the kernel for transparent huge pages for each allocation of at least 4 MiB. The kernel gives them where `/sys/kernel/mm/transparent_hugepage/enabled` is `madvise` or `always`. Without huge pages, each 4 KB page of a new block pays a page fault on its first write. On some hosts those faults take several times as long as the elementwise work that fills the block. basedpl's `bpl`, Python extension, browser build, reference test and speed test register the guard, as do rustymath's Python extension and speed test. [`set_checks`] turns on test-only checks: of the size that each free of a large block passes, and, in basedpl's reference and speed tests, of [`held`] after each case. A binary that doesn't register it counts nothing. There, `admit` refuses only an array that alone would pass the limit.
use crate::error::{Error, Kind};
use std::{alloc::{GlobalAlloc, Layout, System}, any::TypeId, io::Write, ptr::null_mut, sync::atomic::{AtomicBool, AtomicUsize, Ordering::Relaxed}};

/// The smallest allocation that the guard counts, 64 KiB: an array of 65,536 bytes, or of 8,192 f64 items. A smaller one costs the guard one comparison.
pub const LARGE: usize = 64 << 10;
/// The smallest allocation that the guard asks Linux to back with huge pages, 4 MiB. Each block that size holds a whole 2 MiB huge page, wherever it starts.
#[cfg(target_os = "linux")]
const HUGE: usize = 4 << 20;
/// The bytes that [`admit`] counts for each item of an array: the size of an f64.
const ITEM_BYTES: usize = 8;

/// The bytes held in large allocations.
static HELD: AtomicUsize = AtomicUsize::new(0);
/// The most that `HELD` has reached since [`peak_held`] last started.
static PEAK: AtomicUsize = AtomicUsize::new(0);
/// The process's limit on `HELD`: 0 until `limit` first reads it or [`set_limit`] sets it.
static LIMIT: AtomicUsize = AtomicUsize::new(0);
/// Whether [`set_checks`] turned the test-only checks on.
static CHECKS: AtomicBool = AtomicBool::new(false);

/// The process's limit on `HELD`: what [`set_limit`] set, or else half of physical memory, or no limit where the platform doesn't report physical memory. The guard and [`admit`] both read it here. A process's first large array meets the same limit as its first large allocation.
fn limit() -> usize {
    match LIMIT.load(Relaxed) {
        0 => {
            let limit = physical_memory().map_or(usize::MAX, |bytes| bytes / 2);
            LIMIT.store(limit, Relaxed);
            limit
        }
        limit => limit,
    }
}

/// Sets the process's limit on bytes held in large allocations to `bytes`, in place of half of physical memory. A `bytes` of 0 restores that default.
pub fn set_limit(bytes: usize) { LIMIT.store(bytes, Relaxed); }

/// What `f` gives, and the most bytes that the process held at once in large allocations while `f` ran, beyond what it held when `f` started. Allocations on other threads count too. A binary that doesn't register the guard counts nothing, and gets 0.
pub fn peak_held<R>(f: impl FnOnce() -> R) -> (R, usize) {
    let start = HELD.load(Relaxed);
    PEAK.store(start, Relaxed);
    let result = f();
    (result, PEAK.load(Relaxed).saturating_sub(start))
}

/// The bytes that the process holds in large allocations now. A binary that doesn't register the guard counts nothing, and gets 0.
pub fn held() -> usize { HELD.load(Relaxed) }

/// Turns test-only checks on or off. With them on, a guard over `System` aborts with a backtrace where a free or reallocation of a large block passes more bytes than the block holds, and the test harnesses check that [`held`] returns to its start after each case.
pub fn set_checks(on: bool) { CHECKS.store(on, Relaxed); }

/// Whether [`set_checks`] turned the test-only checks on.
pub fn checks() -> bool { CHECKS.load(Relaxed) }

/// The global allocator that counts large allocations, refuses those that would pass the process's limit and, on Linux, asks for huge pages for the largest. It allocates through `A`: `System`, unless [`Guard::over`] names another global allocator, as the browser build names talc. Register it as `#[global_allocator] static GUARD: Guard = Guard::new();`.
#[derive(Default)]
pub struct Guard<A = System>(A);

impl Guard {
    /// A guard over `System`, for the process's limit, which [`set_limit`] can change.
    pub const fn new() -> Self { Self(System) }
}

impl<A: GlobalAlloc + 'static> Guard<A> {
    /// A guard that allocates through `allocator`. [`Guard::new`] gives one over `System`.
    pub const fn over(allocator: A) -> Self { Self(allocator) }

    /// Counts `bytes` more, or refuses them where they would take `HELD` past the limit.
    fn reserve(&self, bytes: usize) -> bool {
        let limit = limit();
        let held = HELD.fetch_add(bytes, Relaxed).saturating_add(bytes);
        if held <= limit {
            PEAK.fetch_max(held, Relaxed);
            return true;
        }
        HELD.fetch_sub(bytes, Relaxed);
        let _ =
            writeln!(std::io::stderr(), "memory guard: refused {} MiB, which would take large allocations past the limit of {} MiB", bytes >> 20, limit >> 20);
        false
    }

    /// Aborts with a backtrace where a free or reallocation of the large block at `ptr` passes more `bytes` than the system allocator gave it. It checks only a guard over `System`, whose blocks the platform can measure.
    #[cold]
    #[inline(never)]
    fn check_size(ptr: *mut u8, bytes: usize) {
        if TypeId::of::<A>() != TypeId::of::<System>() { return; }
        let Some(real) = real_size(ptr) else { return };
        if bytes <= real { return; }
        let _ = writeln!(std::io::stderr(), "memory guard: a free passed {bytes} bytes for a block of {real}\n{}", std::backtrace::Backtrace::force_capture());
        std::process::abort();
    }

    /// Counts `layout`, of at least `LARGE` bytes, then allocates it through the inner allocator and asks for huge pages for it, or gives null where the guard refuses it.
    #[inline(never)]
    unsafe fn alloc_large(&self, layout: Layout, zeroed: bool) -> *mut u8 {
        if !self.reserve(layout.size()) { return null_mut(); }
        let ptr = unsafe { if zeroed { self.0.alloc_zeroed(layout) } else { self.0.alloc(layout) } };
        if ptr.is_null() { HELD.fetch_sub(layout.size(), Relaxed); }
        else { advise_huge(ptr, layout.size()); }
        ptr
    }

    /// Reallocates through the inner allocator where either size is at least `LARGE`, counting the change and asking for huge pages for a block that doesn't shrink, or gives null where the guard refuses it.
    #[inline(never)]
    unsafe fn realloc_large(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let counted = |bytes: usize| if bytes < LARGE { 0 } else { bytes };
        let (old, new) = (counted(layout.size()), counted(new_size));
        if old > 0 && CHECKS.load(Relaxed) { Self::check_size(ptr, old); }
        if new > old && !self.reserve(new - old) { return null_mut(); }
        let moved = unsafe { self.0.realloc(ptr, layout, new_size) };
        if moved.is_null() {
            if new > old { HELD.fetch_sub(new - old, Relaxed); }
        } else if old > new { HELD.fetch_sub(old - new, Relaxed); }
        else { advise_huge(moved, new_size); }
        moved
    }
}

// SAFETY: each method passes its arguments to the inner allocator's. Besides that, it changes only the guard's count and, on Linux, which pages the kernel uses for a large block.
unsafe impl<A: GlobalAlloc + 'static> GlobalAlloc for Guard<A> {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() < LARGE { unsafe { self.0.alloc(layout) } } else { unsafe { self.alloc_large(layout, false) } }
    }
    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() < LARGE { unsafe { self.0.alloc_zeroed(layout) } } else { unsafe { self.alloc_large(layout, true) } }
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() >= LARGE {
            if CHECKS.load(Relaxed) { Self::check_size(ptr, layout.size()); }
            HELD.fetch_sub(layout.size(), Relaxed);
        }
        unsafe { self.0.dealloc(ptr, layout) }
    }
    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.size() < LARGE && new_size < LARGE { unsafe { self.0.realloc(ptr, layout, new_size) } }
        else { unsafe { self.realloc_large(ptr, layout, new_size) } }
    }
}

/// The machine's physical memory in bytes, where the platform reports it.
#[cfg(unix)]
fn physical_memory() -> Option<usize> {
    // SAFETY: `sysconf` reads a system value and has no preconditions.
    let (pages, size) = unsafe { (libc::sysconf(libc::_SC_PHYS_PAGES), libc::sysconf(libc::_SC_PAGESIZE)) };
    (pages > 0 && size > 0).then(|| (pages as usize).saturating_mul(size as usize))
}

#[cfg(not(unix))]
fn physical_memory() -> Option<usize> { None }

/// Asks Linux to back the whole pages of the `bytes` bytes at `ptr` with transparent huge pages, where the block holds at least `HUGE` bytes.
#[cfg(target_os = "linux")]
fn advise_huge(ptr: *mut u8, bytes: usize) {
    if bytes < HUGE { return; }
    // SAFETY: `sysconf` reads a system value and has no preconditions.
    let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
    let (start, end) = ((ptr as usize).next_multiple_of(page), (ptr as usize + bytes) / page * page);
    // SAFETY: the range lies inside the block, and `MADV_HUGEPAGE` changes only which pages the kernel uses for it.
    unsafe { libc::madvise(start as *mut libc::c_void, end - start, libc::MADV_HUGEPAGE) };
}

#[cfg(not(target_os = "linux"))]
fn advise_huge(_: *mut u8, _: usize) {}

/// The bytes that the system allocator gave the block at `ptr`, where the platform reports them.
#[cfg(target_vendor = "apple")]
fn real_size(ptr: *mut u8) -> Option<usize> {
    // SAFETY: `ptr` is a live block from the system allocator, which `malloc_size` reads.
    Some(unsafe { libc::malloc_size(ptr.cast()) })
}

#[cfg(all(target_os = "linux", target_env = "gnu"))]
fn real_size(ptr: *mut u8) -> Option<usize> {
    // SAFETY: `ptr` is a live block from the system allocator, which `malloc_usable_size` reads.
    Some(unsafe { libc::malloc_usable_size(ptr.cast()) })
}

#[cfg(not(any(target_vendor = "apple", all(target_os = "linux", target_env = "gnu"))))]
fn real_size(_: *mut u8) -> Option<usize> { None }

/// A Limit error where an array of `len` items would take large allocations past the process's limit. `admit` counts each item as an f64, 8 bytes. A Boolean or byte array near the limit can therefore get the error early. A complex or nested array can pass `admit` and still meet the guard's refusal. An array under [`LARGE`] bytes passes after one comparison.
#[inline]
pub fn admit(len: usize) -> Result<(), Error> { if len < LARGE / ITEM_BYTES { Ok(()) } else { admit_large(len) } }

fn admit_large(len: usize) -> Result<(), Error> {
    let limit = limit();
    if HELD.load(Relaxed).saturating_add(len.saturating_mul(ITEM_BYTES)) <= limit { return Ok(()); }
    Err(Error::new(Kind::Limit, format!("an array of {len} items would take the process past its memory limit of {} MiB", limit >> 20)))
}
