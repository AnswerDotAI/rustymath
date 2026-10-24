//! The memory guard, registered here with a limit of 64 MiB: its count of large allocations, the Limit error that `generated_len` gives through `admit`, and its refusal past the limit.
use rustymath::{error::Kind, layout::generated_len, memory::Guard};
use std::{hint::black_box, process::Command};

#[global_allocator]
static GUARD: Guard = Guard::with_limit(64 << 20);

const MIB: usize = 1 << 20;

/// `generated_len` admits an array of f64 items only where the bytes held in large allocations leave room for it, which shows the guard's count. The guard reads its limit at its first large allocation.
#[test]
fn counts_large_allocations() {
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
    grown.truncate(MIB / 2);
    grown.shrink_to_fit();
    assert!(fits(8 * MIB));
    drop(black_box(grown));
}

/// The guard refuses an allocation past its limit and says so on stderr, and the process aborts. The allocation runs in a child process: this test binary, run again on this test alone.
#[test]
fn refuses_past_the_limit() {
    if std::env::var_os("RUSTYMATH_GUARD_CHILD").is_some() { drop(black_box(vec![0u8; 80 * MIB])); return; }
    let exe = std::env::current_exe().unwrap();
    let out = Command::new(exe).args(["--exact", "refuses_past_the_limit", "--nocapture"]).env("RUSTYMATH_GUARD_CHILD", "1").output().unwrap();
    assert!(!out.status.success());
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(stderr.contains("memory guard: refused 80 MiB, which would take large allocations past the limit of 64 MiB"), "{stderr}");
}
