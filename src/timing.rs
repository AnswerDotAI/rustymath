//! The machine's timing lock, which keeps timings apart from builds and test runs on one machine, and the retries that speed tests share. A timing takes the lock exclusively. A build or a test run takes it shared. It is an `flock` on the file `LOCK`. `fastws.cargo.timing_lock` takes the same lock from Python. The browser build has no timing lock.
use crate::parallel::set_threads;
use std::{fs::{File, TryLockError}, io, time::{Duration, Instant}};

/// The file whose lock keeps timings apart from builds and test runs.
pub const LOCK: &str = "/tmp/aai-ws-timing.lock";

/// A hold on the timing lock, released when dropped. The operating system also releases it when its process ends, even by a crash. An exclusive hold runs every call on its calling thread alone until it's dropped, because speed budgets are for one thread.
pub struct Hold {
    _file: File,
    /// The thread setting that an exclusive hold replaced, to restore when dropped.
    threads: Option<usize>,
}

impl Drop for Hold { fn drop(&mut self) { if let Some(n) = self.threads { set_threads(n); } } }

/// Takes the timing lock exclusively, for a timing. It waits up to 5 minutes for builds, test runs and other timings, then fails. After waiting 30 seconds it prints, once, the other processes that have the lock's file open. On macOS it also sets the calling thread's QoS class to user-interactive. macOS then favours the performance cores for that thread. QoS isn't CPU affinity.
pub fn exclusive() -> io::Result<Hold> { hold(false) }

/// Takes the timing lock shared, for a build or a test run. It waits up to 5 minutes for a timing, then goes ahead without the lock.
pub fn shared() -> io::Result<Hold> { hold(true) }

/// The pauses, in seconds, before each retry in `retried`. All of a case's runs together take a fraction of a millisecond, and one disturbance on a shared machine can last that long.
pub const RETRY_PAUSES: [f64; 2] = [0.1, 1.0];

/// Which of a budget file's two native budgets this machine's speed tests check: 0, for macOS on Apple silicon, where matrix products run on Accelerate, or 1, for Linux on CI's x86 runner and every other system.
pub const NATIVE_COLUMN: usize = if cfg!(target_os = "macos") { 0 } else { 1 };

/// The failures of a timing test that still fail after every retry. After each pause in `RETRY_PAUSES`, `retime` gets the failures and the pause, times them again, prints the retry and gives those that still fail. A speed test calls it after timing every case.
pub fn retried<T>(mut failures: Vec<T>, mut retime: impl FnMut(Vec<T>, f64) -> Vec<T>) -> Vec<T> {
    for pause in RETRY_PAUSES {
        if failures.is_empty() { break; }
        std::thread::sleep(Duration::from_secs_f64(pause));
        failures = retime(failures, pause);
    }
    failures
}

fn hold(shared: bool) -> io::Result<Hold> {
    #[cfg(target_vendor = "apple")]
    if !shared {
        unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
    }
    let failed = |e: io::Error| io::Error::new(e.kind(), format!("{LOCK}: {e}"));
    let held = |file| Ok(Hold { _file: file, threads: (!shared).then(|| set_threads(1)) });
    let file = File::options().create(true).append(true).open(LOCK).map_err(failed)?;
    let others = if shared { "a timing" } else { "a build or another timing" };
    let (start, mut waiting, mut reported) = (Instant::now(), false, false);
    loop {
        match if shared { file.try_lock_shared() } else { file.try_lock() } {
            Ok(()) => return held(file),
            Err(TryLockError::WouldBlock) => {}
            Err(TryLockError::Error(e)) => return Err(failed(e)),
        }
        if start.elapsed() > Duration::from_secs(300) {
            if !shared { return Err(io::Error::new(io::ErrorKind::TimedOut, format!("{others} has held {LOCK} for 5 minutes"))); }
            eprintln!("going ahead although {others} still holds {LOCK}");
            return held(file);
        }
        if !waiting { eprintln!("waiting for {others} to release {LOCK}"); }
        if !shared && !reported && start.elapsed() > Duration::from_secs(30) {
            eprintln!("{LOCK} is still taken after 30 seconds. Processes with it open: {}", users());
            reported = true;
        }
        waiting = true;
        std::thread::sleep(Duration::from_millis(250));
    }
}

/// The processes other than this one that have the lock's file open, as `pid command` pairs from `lsof`. On macOS, where `lsof` doesn't show `flock` holds, the list also includes any other process waiting for the lock.
fn users() -> String {
    let Ok(out) = std::process::Command::new("lsof").args(["+c", "0", "-F", "pc", LOCK]).output() else {
        return "unknown, because lsof didn't run".into();
    };
    let (text, me) = (String::from_utf8_lossy(&out.stdout), std::process::id().to_string());
    let (mut users, mut pid) = (Vec::new(), "");
    for line in text.lines() {
        match line.split_at_checked(1) { Some(("p", p)) => pid = p, Some(("c", c)) if pid != me => users.push(format!("{pid} {c}")), _ => {} }
    }
    if users.is_empty() { "none".into() } else { users.join(", ") }
}
