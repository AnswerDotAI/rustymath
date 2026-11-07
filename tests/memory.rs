//! The registered memory guard's large-allocation count and checked frees, the Limit error that `generated_len` gives through `admit`, and refusal past the 64 MiB limit.
use rustymath::{error::Kind, layout::generated_len, memory::{Guard, LARGE, peak_held, set_checks, set_limit}};
use std::alloc::{GlobalAlloc, Layout};

#[global_allocator]
static GUARD: Guard = Guard::new();

const MIB: usize = 1 << 20;

/// Allocation, reallocation and refusal update the guard's count used by `generated_len`.
#[test]
fn tracks_and_limits_large_allocations() {
    set_checks(true);
    set_limit(64 * MIB);
    let fits = |items: usize| generated_len(&[items]).is_ok();
    let layout = |bytes| Layout::from_size_align(bytes, 1).unwrap();
    assert_eq!(generated_len(&[9 * MIB]).unwrap_err().kind, Kind::Limit);
    // SAFETY: allocations use valid nonzero layouts. Each non-null block is reallocated or freed with its current layout.
    unsafe {
        let held = GUARD.alloc(layout(40 * MIB));
        assert!(!held.is_null());
        assert!(fits(2 * MIB) && !fits(4 * MIB));
        assert_eq!(generated_len(&[4 * MIB]).unwrap_err().kind, Kind::Limit);
        GUARD.dealloc(held, layout(40 * MIB));
        assert!(fits(8 * MIB) && !fits(8 * MIB + 1));
        let grown = GUARD.alloc(layout(16));
        assert!(!grown.is_null());
        let grown = GUARD.realloc(grown, layout(16), 40 * MIB);
        assert!(!grown.is_null());
        assert!(fits(2 * MIB) && !fits(4 * MIB));
        let grown = GUARD.realloc(grown, layout(40 * MIB), LARGE / 2);
        assert!(!grown.is_null());
        assert!(fits(8 * MIB));
        let ((), peak) = peak_held(|| {
            let held = GUARD.alloc(layout(3 * MIB));
            assert!(!held.is_null());
            GUARD.dealloc(held, layout(3 * MIB));
        });
        assert_eq!(peak, 3 * MIB);
        GUARD.dealloc(grown, layout(LARGE / 2));
        assert!(GUARD.alloc(layout(80 * MIB)).is_null());
        assert!(fits(8 * MIB));
    }
}
