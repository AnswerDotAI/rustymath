//! What speed tests share: the machine's timing lock, the timing loop, the copy that costs count in with the copy times stored for each machine that runs them, the format of cost lines, the format of budget files, and the retries. rustymath's and basedpl's speed tests use all of it, and basedpl's browser build uses all but the lock.
//!
//! The lock keeps timings apart from builds and test runs on one machine. A timing takes it exclusively. A build or a test run takes it shared. It is an `flock` on the file `LOCK`. `fastws.cargo.timing_lock` takes the same lock from Python. The browser build has no timing lock.
#[cfg(not(target_arch = "wasm32"))]
pub use lock::{exclusive, shared, Hold, LOCK};
use std::{ops::Neg, time::Duration};
/// The clock that timings and `parallel`'s rounds read: the browser's in a browser build, where `std::time::Instant` panics.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) use std::time::Instant;
#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
pub(crate) use web_time::Instant;

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

/// One copy's time for each item, in seconds, for `items` items of `bytes` bytes each. `stored` says whether it's this machine's entry in [`COPIES`] or was measured.
#[derive(Clone, Copy, Debug)]
pub struct CopyTime { pub seconds: f64, pub items: usize, pub bytes: usize, pub stored: bool }

/// The most bytes that a copy's source and destination take together for [`copy_time`] to use a stored time. Below it, the copy's data stays in cache, and its time is a property of the CPU. Above it, the copy runs at the speed of memory, which a shared runner varies between runs, and cases of the same size vary with it. On CI's runner under V8, the 36-million-item copy and the cases at that size both varied up to 2.1 times between runs, and dividing by the copy measured in the same run cut the cases' spread from 2.24 times to 1.32. The speed tests' copies take at most 16 MB below it, or 576 MB above it.
const CACHED_BYTES: usize = 64 << 20;

/// The copy times of the machines that run the speed tests in native builds, for copies whose source and destination fit in `CACHED_BYTES`: the CPU's name as [`cpu`] reads it, an item's size in bytes, the number of items, and the time for each item in seconds. Each is at the slower end of what that machine's copy normally takes, which is the copy that the expected costs in the budget files assume. On CI's runner, a copy measured beside each case varied more than 3 times between cases, while every function's own time varied at most 1.18 times. A stored time keeps that noise out of every cost.
#[cfg(not(target_arch = "wasm32"))]
pub const COPIES: &[(&str, usize, usize, f64)] = &[
    // CI's Ubicloud runner, from 630 copies timed beside cases in 21 runs of rustymath's speed test, 15 to 21 October 2026: 0.13 to 0.16 ns for 8-byte items, and 0.33 to 0.45 ns when the copy read its source from the L3 cache that other machines share.
    ("AMD Ryzen 9 7950X3D 16-Core Processor", 8, 65_536, 0.16e-9),
    ("AMD Ryzen 9 7950X3D 16-Core Processor", 4, 65_536, 0.08e-9),
    // 16-byte items, complex f64: 0.285 ns, measured once on 22 October 2026. Each complex f64 function's time per item that day, divided by its cost in the earlier runs whose 8-byte copies came from cache, gives 0.24 to 0.29 ns.
    ("AMD Ryzen 9 7950X3D 16-Core Processor", 16, 65_536, 0.29e-9),
    // The development Mac, 22 October 2026, through basedpl's `time_cases` and rustymath's speed test: 0.115 to 0.13 ns for 8-byte items at each size, and 0.056 to 0.066 ns for 4-byte items.
    ("Apple M5 Max", 8, 65_536, 0.12e-9),
    ("Apple M5 Max", 8, 1_048_576, 0.125e-9),
    ("Apple M5 Max", 4, 65_536, 0.065e-9),
    // 16-byte items, complex f64, measured at 0.240 ns by rustymath's speed test on 22 October 2026.
    ("Apple M5 Max", 16, 65_536, 0.25e-9),
    // GitHub's runners in basedpl's `Release` workflow, from the copies that `bpl --costs` measured in its three runs of 22 October 2026: the slowest copy at each size that the cases' own times in that run bore out. The macOS runner's 1,048,576-item copy also measured 0.369 ns once, when its cases' times rose 1.28 times and its copy 1.78 times. The Linux ARM runner is the Neoverse-N2 that `lscpu` names.
    ("Apple M1 (Virtual)", 8, 65_536, 0.18e-9),
    ("Apple M1 (Virtual)", 8, 1_048_576, 0.25e-9),
    ("CPU implementer 0x41, part 0xd49", 8, 65_536, 0.18e-9),
    ("CPU implementer 0x41, part 0xd49", 8, 1_048_576, 0.26e-9),
    // The Linux x86-64 runner's CPU changes between runs, so each of these comes from one run. The EPYC 7763's 65,536-item copy measured 0.317 ns, but its cases ran as fast as the 9V74's, whose copy took 0.206 ns, so it takes 0.21 ns.
    ("AMD EPYC 7763 64-Core Processor", 8, 65_536, 0.21e-9),
    ("AMD EPYC 7763 64-Core Processor", 8, 1_048_576, 0.22e-9),
    ("AMD EPYC 9V45 96-Core Processor", 8, 65_536, 0.10e-9),
    ("AMD EPYC 9V45 96-Core Processor", 8, 1_048_576, 0.12e-9),
    ("AMD EPYC 9V74 80-Core Processor", 8, 65_536, 0.21e-9),
    ("AMD EPYC 9V74 80-Core Processor", 8, 1_048_576, 0.24e-9),
];

/// [`COPIES`] for the browser build, whose copy runs under the JavaScript engine that runs the build.
#[cfg(target_arch = "wasm32")]
pub const COPIES: &[(&str, usize, usize, f64)] = &[
    // CI's Ubicloud runner under Node's V8, from the copies that `tests/wasm-speed.mjs` measured in seven runs of 22 October 2026: 0.172 to 0.218 ns at 65,536 items, and 0.196 to 0.249 ns at 1,048,576.
    ("AMD Ryzen 9 7950X3D 16-Core Processor", 8, 65_536, 0.22e-9),
    ("AMD Ryzen 9 7950X3D 16-Core Processor", 8, 1_048_576, 0.25e-9),
];

/// This machine's CPU, as its operating system names it: `machdep.cpu.brand_string` on macOS. On Linux, it's the first `model name` in `/proc/cpuinfo`, or else the `CPU implementer` and `CPU part` codes that ARM machines give there in place of a model name, as `CPU implementer 0x41, part 0xd49`. In the browser build, it's the name passed to [`name_cpu`]. `None` without a name.
pub fn cpu() -> Option<&'static str> { CPU.get_or_init(read_cpu).as_deref() }
static CPU: std::sync::OnceLock<Option<String>> = std::sync::OnceLock::new();

/// Names this machine's CPU for [`cpu`] where the system gives no name, as in the browser build, whose tests pass Node's name for it. Has no effect once `cpu` has been read.
pub fn name_cpu(name: &str) { _ = CPU.set(read_cpu().or_else(|| Some(name.to_owned()))); }

#[cfg(target_os = "macos")]
fn read_cpu() -> Option<String> {
    let (name, mut len) = (c"machdep.cpu.brand_string", 0);
    // SAFETY: a null buffer asks for the length, and the second call writes at most `len` bytes into `text`.
    if unsafe { libc::sysctlbyname(name.as_ptr(), std::ptr::null_mut(), &mut len, std::ptr::null_mut(), 0) } != 0 { return None; }
    let mut text = vec![0u8; len];
    if unsafe { libc::sysctlbyname(name.as_ptr(), text.as_mut_ptr().cast(), &mut len, std::ptr::null_mut(), 0) } != 0 { return None; }
    text.truncate(len);
    String::from_utf8(text).ok().map(|s| s.trim_end_matches('\0').trim().to_owned())
}

#[cfg(target_os = "linux")]
fn read_cpu() -> Option<String> {
    let info = std::fs::read_to_string("/proc/cpuinfo").ok()?;
    let field = |name: &str| info.lines().find_map(|l| Some(l.strip_prefix(name)?.split_once(':')?.1.trim().to_owned()));
    field("model name").or_else(|| Some(format!("CPU implementer {}, part {}", field("CPU implementer")?, field("CPU part")?)))
}

#[cfg(not(any(target_os = "macos", target_os = "linux")))]
fn read_cpu() -> Option<String> { None }

/// One copy's time for each of `items` items of type `T`. A copy whose source and destination fit in `CACHED_BYTES` takes this machine's entry in [`COPIES`]. Any other copy, or one with no entry, is measured once, the first time its size is asked for. A copy that fits but has no entry prints a line on standard error, giving the time to add.
pub fn copy_time<T: Copy + Default + Neg<Output = T>>(items: usize) -> CopyTime {
    let bytes = size_of::<T>();
    let cached = 2 * items * bytes <= CACHED_BYTES;
    if cached && let Some(name) = cpu() && let Some(&(_, _, _, seconds)) = COPIES.iter().find(|&&(c, b, n, _)| c == name && b == bytes && n == items) {
        return CopyTime { seconds, items, bytes, stored: true };
    }
    static MEASURED: std::sync::Mutex<Vec<CopyTime>> = std::sync::Mutex::new(Vec::new());
    let mut measured = MEASURED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&t) = measured.iter().find(|t| (t.bytes, t.items) == (bytes, items)) { return t; }
    let t = CopyTime { seconds: measured_copy::<T>(items), items, bytes, stored: false };
    if cached { eprintln!("{}: add it to rustymath::timing::COPIES", copy_line(&t)); }
    measured.push(t);
    t
}

/// One copy's time for each item, measured on `items` items of type `T`: the fastest of `RUNS` copies of the same two buffers, which stay in cache where they fit. Copying both ways first maps every page, so no copy reads the shared page of zeros that a fresh allocation reads as.
fn measured_copy<T: Copy + Default + Neg<Output = T>>(items: usize) -> f64 {
    let (mut data, mut out) = (vec![T::default(); items], vec![T::default(); items]);
    copy(&data, &mut out);
    copy(&out, &mut data);
    fastest(|_| { copy(std::hint::black_box(&data), &mut out); std::hint::black_box(&mut out); }) / items.max(1) as f64
}

/// A line saying what a copy of `t`'s items costs for each item, and whether that time was stored for this machine or measured, as the speed tests print it before their costs.
pub fn copy_line(t: &CopyTime) -> String {
    let source = match (t.stored, cpu()) { (true, Some(c)) => format!("stored for {c}"), (false, Some(c)) => format!("measured on {c}"), (_, None) => "measured on a CPU with no name".into() };
    format!("copy of {} {}-byte items: {:.3} ns an item, {source}", t.items, t.bytes, t.seconds * 1e9)
}

/// A case's line in a speed test's log: its section, code, cost, budget and time for each item. `basedpl.bpltests.logged_costs` reads these lines.
pub fn line(section: &str, code: &str, cost: f64, budget: &str, seconds: f64) -> String {
    format!("{section} {code}: {cost:.2}, budget {budget}, {:.3} ns", seconds * 1e9)
}

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
