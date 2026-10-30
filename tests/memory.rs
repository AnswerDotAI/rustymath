//! The memory guard, registered here, with each test setting the process's limit to 64 MiB: its count of large allocations, the Limit error that `generated_len` gives through `admit`, its refusal past the limit, and, with the checks on, its abort on a free that passes the wrong size.
use rustymath::{error::Kind, layout::generated_len, memory::{Guard, LARGE, peak_held, set_checks, set_limit}};
use std::{alloc::{Layout, alloc, dealloc}, hint::black_box, process::Command};

#[global_allocator]
static GUARD: Guard = Guard::new();

const MIB: usize = 1 << 20;

/// `generated_len` admits an array of f64 items only where the bytes held in large allocations leave room for it, which shows the guard's count.
#[test]
fn counts_large_allocations() {
    set_limit(64 * MIB);
    let fits = |items: usize| generated_len(&[items]).is_ok();
    let held = vec![1u8; 40 * MIB];
    assert!(fits(2 * MIB) && !fits(4 * MIB));
    assert_eq!(generated_len(&[4 * MIB]).unwrap_err().kind, Kind::Limit);
    drop(held);
    assert!(fits(8 * MIB) && !fits(8 * MIB + 1));
    // A vector that grows past `LARGE` through `realloc` is counted as it grows, and released as it shrinks.
    let mut grown = vec![1u8; 16];
    grown.resize(40 * MIB, 1);
    assert!(fits(2 * MIB) && !fits(4 * MIB));
    grown.truncate(LARGE / 2);
    grown.shrink_to_fit();
    assert!(fits(8 * MIB));
    let ((), peak) = peak_held(|| drop(black_box(vec![1u8; 3 * MIB])));
    assert_eq!(peak, 3 * MIB);
    drop(black_box(grown));
}

/// Runs this test binary again on the test `name` alone, as the child that `RUSTYMATH_GUARD_CHILD` marks, and gives whether it succeeded and what it wrote to stderr.
fn child(name: &str) -> (bool, String) {
    let out = Command::new(std::env::current_exe().unwrap()).args(["--exact", name, "--nocapture"]).env("RUSTYMATH_GUARD_CHILD", "1").output().unwrap();
    (out.status.success(), String::from_utf8_lossy(&out.stderr).into_owned())
}

/// Whether this process is a child that [`child`] started.
fn in_child() -> bool { std::env::var_os("RUSTYMATH_GUARD_CHILD").is_some() }

/// In a fresh process, `generated_len` gives a Limit error for an array past the limit before any large allocation. The guard refuses an allocation past its limit and says so on stderr, and the process aborts.
#[test]
fn refuses_past_the_limit() {
    if in_child() {
        set_limit(64 * MIB);
        assert_eq!(generated_len(&[9 * MIB]).unwrap_err().kind, Kind::Limit);
        drop(black_box(vec![0u8; 80 * MIB]));
        return;
    }
    let (ok, stderr) = child("refuses_past_the_limit");
    assert!(!ok);
    assert!(stderr.contains("memory guard: refused 80 MiB, which would take large allocations past the limit of 64 MiB"), "{stderr}");
}

/// With the checks on, a free that passes more bytes than its block holds aborts the process, before the system allocator sees the size.
#[test]
#[cfg(any(target_vendor = "apple", all(target_os = "linux", target_env = "gnu")))]
fn checks_the_sizes_that_frees_pass() {
    if in_child() {
        set_checks(true);
        let size = |bytes| Layout::from_size_align(bytes, 8).unwrap();
        // SAFETY: the block is live, and the guard aborts on the wrong size before it reaches the system allocator.
        unsafe { dealloc(alloc(size(MIB)), size(4 * MIB)) };
        return;
    }
    let (ok, stderr) = child("checks_the_sizes_that_frees_pass");
    assert!(!ok);
    assert!(stderr.contains("memory guard: a free passed 4194304 bytes for a block of"), "{stderr}");
}
