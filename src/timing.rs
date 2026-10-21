//! What speed tests share: the machine's timing lock, the timing loop, the copy that costs count in, the format of budget files, and the retries. rustymath's and basedpl's speed tests use all of it, and basedpl's browser build uses all but the lock.
//!
//! The lock keeps timings apart from builds and test runs on one machine. A timing takes it exclusively. A build or a test run takes it shared. It is an `flock` on the file `LOCK`. `fastws.cargo.timing_lock` takes the same lock from Python. The browser build has no timing lock.
#[cfg(not(target_arch = "wasm32"))]
pub use lock::{exclusive, shared, Hold, LOCK};
use std::{ops::Neg, time::Duration};
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
use std::time::Instant;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
use web_time::Instant;

/// How many calls `fastest` times.
pub const RUNS: usize = 5;

/// The time of the fastest of `RUNS` calls of `run`, in seconds. Call `k` is passed `k`, so it can read its own inputs.
pub fn fastest(mut run: impl FnMut(usize)) -> f64 {
    (0..RUNS).map(|k| {
        let start = Instant::now();
        run(k);
        start.elapsed()
    }).min().expect("RUNS is positive").as_secs_f64()
}

/// One copy, the unit that speed budgets count in: each item of `data`, negated, written into `out`. The loop reads and writes each item once. It calls no `memcpy` and allocates nothing, because their speed differs between platforms far more than a loop's does.
pub fn copy<T: Copy + Neg<Output = T>>(data: &[T], out: &mut [T]) { for (o, &v) in out.iter_mut().zip(data) { *o = -v } }

/// One section of a budget file: its name, the rest of its header line, such as basedpl's prefs, and its lines.
pub type Section<'a> = (&'a str, &'a str, Vec<&'a str>);

/// The sections of `text`, a budget file such as either project's `tests/speed.txt`. A line `[name]` or `[name] rest` starts each section. Each line under it holds a case's code, then its budgets, separated by tabs. Blank lines and lines that start with `#` are skipped.
pub fn sections(text: &str) -> Result<Vec<Section<'_>>, String> {
    let mut sections: Vec<Section> = Vec::new();
    for line in text.lines().filter(|l| !l.is_empty() && !l.starts_with('#')) {
        match header(line) {
            Some((name, rest)) => sections.push((name, rest, Vec::new())),
            None => sections.last_mut().ok_or_else(|| format!("{line:?} comes before the first section"))?.2.push(line),
        }
    }
    Ok(sections)
}

/// The name and the rest of a budget file's header line: `[name]` or `[name] rest`.
pub fn header(line: &str) -> Option<(&str, &str)> {
    line.strip_prefix('[').and_then(|l| l.split_once(']')).filter(|(name, rest)| !name.contains(' ') && (rest.is_empty() || rest.starts_with(' '))).map(|(name, rest)| (name, rest.trim()))
}

/// A case line's code: the text before its first tab.
pub fn code_of(line: &str) -> &str { line.split('\t').next().unwrap_or_default() }

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

#[cfg(not(target_arch = "wasm32"))]
mod lock {
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
}
