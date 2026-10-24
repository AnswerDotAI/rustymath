//! The process's memory guard. A binary that registers [`Guard`] as its global allocator counts the bytes it holds in large allocations, of at least [`LARGE`] bytes each. The guard refuses an allocation that would take that count past its limit, which is half of physical memory by default. It writes one line to stderr naming the limit. Rust then aborts the process, before the machine runs out of memory. Before a function makes an array, [`generated_len`](crate::layout::generated_len) asks [`admit`], which gives a Limit error where the array would pass the limit. basedpl's `bpl` and its Python extension register the guard. A binary that doesn't counts nothing and admits every array.
use crate::error::{Error, Kind};
use std::{alloc::{GlobalAlloc, Layout, System}, io::Write, ptr::null_mut, sync::atomic::{AtomicUsize, Ordering::Relaxed}};

/// The smallest allocation that the guard counts, 1 MiB. A smaller one costs the guard one comparison.
pub const LARGE: usize = 1 << 20;
/// The bytes that [`admit`] counts for each item of an array: the size of an f64.
const ITEM_BYTES: usize = 8;

/// The bytes held in large allocations.
static HELD: AtomicUsize = AtomicUsize::new(0);
/// The limit on `HELD`, which the guard sets at its first large allocation. 0 until then.
static LIMIT: AtomicUsize = AtomicUsize::new(0);

/// The global allocator that counts large allocations and refuses those that would pass its limit. Register it as `#[global_allocator] static GUARD: Guard = Guard::new();`.
#[derive(Default)]
pub struct Guard { limit: usize }

impl Guard {
    /// A guard whose limit is half of physical memory, or that never refuses where the platform doesn't report physical memory.
    pub const fn new() -> Self { Self { limit: 0 } }
    /// A guard whose limit is `bytes`.
    pub const fn with_limit(bytes: usize) -> Self { Self { limit: bytes } }

    fn limit(&self) -> usize {
        match LIMIT.load(Relaxed) {
            0 => {
                let limit = if self.limit > 0 { self.limit } else { physical_memory().map_or(usize::MAX, |bytes| bytes / 2) };
                LIMIT.store(limit, Relaxed);
                limit
            }
            limit => limit,
        }
    }

    /// Counts `bytes` more, or refuses them where they would take `HELD` past the limit.
    fn reserve(&self, bytes: usize) -> bool {
        let limit = self.limit();
        if HELD.fetch_add(bytes, Relaxed).saturating_add(bytes) <= limit { return true; }
        HELD.fetch_sub(bytes, Relaxed);
        let _ = writeln!(std::io::stderr(), "memory guard: refused {} MiB, which would take large allocations past the limit of {} MiB", bytes >> 20, limit >> 20);
        false
    }

    /// Allocates `layout`, of at least `LARGE` bytes, as `System` does, after counting it, or gives null where the guard refuses it.
    #[inline(never)]
    unsafe fn alloc_large(&self, layout: Layout, zeroed: bool) -> *mut u8 {
        if !self.reserve(layout.size()) { return null_mut(); }
        let ptr = unsafe { if zeroed { System.alloc_zeroed(layout) } else { System.alloc(layout) } };
        if ptr.is_null() { HELD.fetch_sub(layout.size(), Relaxed); }
        ptr
    }

    /// Reallocates as `System` does where either size is at least `LARGE`, counting the change, or gives null where the guard refuses it.
    #[inline(never)]
    unsafe fn realloc_large(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let counted = |bytes: usize| if bytes < LARGE { 0 } else { bytes };
        let (old, new) = (counted(layout.size()), counted(new_size));
        if new > old && !self.reserve(new - old) { return null_mut(); }
        let moved = unsafe { System.realloc(ptr, layout, new_size) };
        if moved.is_null() { if new > old { HELD.fetch_sub(new - old, Relaxed); } }
        else if old > new { HELD.fetch_sub(old - new, Relaxed); }
        moved
    }
}

// SAFETY: each method passes its arguments to `System`'s and changes only the guard's count.
unsafe impl GlobalAlloc for Guard {
    #[inline]
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        if layout.size() < LARGE { unsafe { System.alloc(layout) } } else { unsafe { self.alloc_large(layout, false) } }
    }
    #[inline]
    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if layout.size() < LARGE { unsafe { System.alloc_zeroed(layout) } } else { unsafe { self.alloc_large(layout, true) } }
    }
    #[inline]
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        if layout.size() >= LARGE { HELD.fetch_sub(layout.size(), Relaxed); }
        unsafe { System.dealloc(ptr, layout) }
    }
    #[inline]
    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        if layout.size() < LARGE && new_size < LARGE { unsafe { System.realloc(ptr, layout, new_size) } } else { unsafe { self.realloc_large(ptr, layout, new_size) } }
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

/// A Limit error where an array of `len` items would take large allocations past the registered guard's limit. `admit` counts each item as an f64, 8 bytes. A Boolean or byte array near the limit can therefore get the error early. A complex or nested array can pass `admit` and still meet the guard's refusal. An array under [`LARGE`] bytes passes after one comparison. Every array passes until a guard has made its first large allocation.
#[inline]
pub fn admit(len: usize) -> Result<(), Error> { if len < LARGE / ITEM_BYTES { Ok(()) } else { admit_large(len) } }

fn admit_large(len: usize) -> Result<(), Error> {
    let limit = LIMIT.load(Relaxed);
    if limit == 0 || HELD.load(Relaxed).saturating_add(len.saturating_mul(ITEM_BYTES)) <= limit { return Ok(()); }
    Err(Error::new(Kind::Limit, format!("an array of {len} items would take the process past its memory limit of {} MiB", limit >> 20)))
}
