//! What speed tests share: the machine's timing lock, the timing loop, the copy that costs count in with the copy times stored for each machine that runs them, the copy of a case's footprint that its line shows, the format of cost lines, the format of budget files, the CPU that budgets were set for, and the retries. rustymath's and basedpl's speed tests use all of it, and basedpl's browser build uses all but the lock.
//!
//! The lock keeps timings apart from builds and test runs on one machine. A timing takes it exclusively. A build or a test run takes it shared. It is an `flock` on the file `LOCK`. A timing holds a second file, `GATE`, exclusively from before it takes `LOCK` until it releases it, and a shared hold passes `GATE` before it takes `LOCK`. A waiting timing then stops new builds and test runs from starting, and it runs once the holds already taken end. `fastws.timelock` takes the same locks from Python. The browser build has no timing lock.
#[cfg(not(target_arch = "wasm32"))]
pub use lock::{GATE, Hold, LOCK, exclusive, shared};
/// The clock that timings and `parallel`'s rounds read: the browser's in a browser build, where `std::time::Instant` panics.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
pub(crate) use std::time::Instant;
use std::{ops::Neg, time::Duration};
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

/// Whether a copy of `items` items of `bytes` bytes each, with its destination, fits in `CACHED_BYTES`.
fn cached(items: usize, bytes: usize) -> bool { 2 * items * bytes <= CACHED_BYTES }

/// The CPU of CI's Ubicloud runner, on which the Linux budgets and basedpl's wasm budgets were set.
const CI_CPU: &str = "AMD Ryzen 9 7950X3D 16-Core Processor";

/// The CPU of the development Mac, on which the macOS budgets were set.
const DEV_CPU: &str = "Apple M5 Max";

/// The copy times of the machines that run the speed tests in native builds, for copies whose source and destination fit in `CACHED_BYTES`: the CPU's name as [`cpu`] reads it, an item's size in bytes, the number of items, and the time for each item in seconds. Each is at the slower end of what that machine's copy normally takes, which is the copy that the expected costs in the budget files assume. On CI's runner, a copy measured beside each case varied more than 3 times between cases, while every function's own time varied at most 1.18 times. A stored time keeps that noise out of every cost.
#[cfg(not(target_arch = "wasm32"))]
pub const COPIES: &[(&str, usize, usize, f64)] = &[
    // CI's Ubicloud runner, from 630 copies timed beside cases in 21 runs of rustymath's speed test, 15 to 21 October 2026: 0.13 to 0.16 ns for 8-byte items, and 0.33 to 0.45 ns when the copy read its source from the L3 cache that other machines share.
    (CI_CPU, 8, 65_536, 0.16e-9),
    (CI_CPU, 4, 65_536, 0.08e-9),
    // 16-byte items, complex f64: 0.285 ns, measured once on 22 October 2026. Each complex f64 function's time per item that day, divided by its cost in the earlier runs whose 8-byte copies came from cache, gives 0.24 to 0.29 ns.
    (CI_CPU, 16, 65_536, 0.29e-9),
    // 1,048,576 8-byte items: 0.157 ns, measured once by basedpl's speed test on 22 October 2026. The browser build's copy at this size on the same runner varied 1.27 times across seven runs, so the entry allows for that.
    (CI_CPU, 8, 1_048_576, 0.18e-9),
    // The development Mac, 22 October 2026, through basedpl's `time_cases` and rustymath's speed test: 0.115 to 0.13 ns for 8-byte items at each size, and 0.056 to 0.066 ns for 4-byte items.
    (DEV_CPU, 8, 65_536, 0.12e-9),
    (DEV_CPU, 8, 1_048_576, 0.125e-9),
    (DEV_CPU, 4, 65_536, 0.065e-9),
    // 16-byte items, complex f64, measured at 0.240 ns by rustymath's speed test on 22 October 2026.
    (DEV_CPU, 16, 65_536, 0.25e-9),
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
    (CI_CPU, 8, 65_536, 0.22e-9),
    (CI_CPU, 8, 1_048_576, 0.25e-9),
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

/// One copy's time for each of `items` items of type `T`. A copy whose source and destination fit in `CACHED_BYTES` takes this machine's entry in [`COPIES`]. Any other copy, or one with no entry, is measured once, the first time its size is asked for. [`copy_line`] says when a measured copy belongs in [`COPIES`].
pub fn copy_time<T: Copy + Default + Neg<Output = T>>(items: usize) -> CopyTime {
    let bytes = size_of::<T>();
    if cached(items, bytes) && let Some(name) = cpu()
        && let Some(&(_, _, _, seconds)) = COPIES.iter().find(|&&(c, b, n, _)| c == name && b == bytes && n == items) {
            return CopyTime { seconds, items, bytes, stored: true };
        }
    static MEASURED: std::sync::Mutex<Vec<CopyTime>> = std::sync::Mutex::new(Vec::new());
    let mut measured = MEASURED.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(&t) = measured.iter().find(|t| (t.bytes, t.items) == (bytes, items)) { return t; }
    let t = CopyTime { seconds: measured_copy::<T>(items), items, bytes, stored: false };
    measured.push(t);
    t
}

/// One copy's time for each item, measured on `items` items of type `T`: the fastest of `RUNS` copies between the two parts of one block, which stay in cache where they fit. The destination starts a whole number of 4 KB pages after the source, wherever the allocator puts the block. A store then never shares the low 12 bits of its address with a load just ahead of it, which on x86 stalls the load (4K aliasing). Copying both ways first maps every page, so no copy reads the shared page of zeros that a fresh allocation reads as.
fn measured_copy<T: Copy + Default + Neg<Output = T>>(items: usize) -> f64 {
    let gap = items.next_multiple_of(4096 / size_of::<T>());
    let mut block = vec![T::default(); gap + items];
    let (data, out) = block.split_at_mut(gap);
    let (data, out) = (&mut data[..items], &mut out[..items]);
    copy(data, out);
    copy(out, data);
    fastest(|_| {
        copy(std::hint::black_box(&*data), out);
        std::hint::black_box(&mut *out);
    }) / items.max(1) as f64
}

/// The bytes of a case's footprint, its input and the most it holds at once, above which the speed tests measure a copy of it: 1 MiB, the L2 cache of each core on CI's runner. A larger footprint reaches the L3 cache and memory, which the runner shares with other VMs on its host. In basedpl's `Test` runs 56 to 62, cases with footprints of up to 1.5 MB held their times, while cases from about 2 MB up swung by up to 4 times between runs.
pub const CORE_CACHE_BYTES: usize = 1 << 20;

/// A case's footprint in bytes, and the time for each item of a copy whose source and destination take as many bytes together, measured beside the case.
#[derive(Clone, Copy, Debug)]
pub struct Footprint { pub bytes: usize, pub seconds: f64 }

/// A case's [`Footprint`], with its copy measured now: the bytes of `items` items of type `T` and the `held` bytes that the case held at once, as `memory::peak_held` counts them. `None` for a footprint of at most [`CORE_CACHE_BYTES`]. The speed tests measure it beside each case and print it in the case's line, so the logs show whether a case's time follows its copy's.
pub fn footprint<T: Copy + Default + Neg<Output = T>>(items: usize, held: usize) -> Option<Footprint> {
    let bytes = items * size_of::<T>() + held;
    (bytes > CORE_CACHE_BYTES).then(|| Footprint { bytes, seconds: measured_copy::<T>(bytes / (2 * size_of::<T>())) })
}

/// A line saying what a copy of `t`'s items costs for each item, and whether that time was stored for this machine or measured, as the speed tests print it before their costs. A measured copy that fits in `CACHED_BYTES` adds that it belongs in [`COPIES`]: the line reaches the log where the speed tests' own lines do.
pub fn copy_line(t: &CopyTime) -> String {
    let source = match (t.stored, cpu()) {
        (true, Some(c)) => format!("stored for {c}"),
        (false, Some(c)) if cached(t.items, t.bytes) => format!("measured on {c}: add it to rustymath::timing::COPIES"),
        (false, Some(c)) => format!("measured on {c}"), (_, None) => "measured on a CPU with no name".into()
    };
    format!("copy of {} {}-byte items: {:.3} ns an item, {source}", t.items, t.bytes, t.seconds * 1e9)
}

/// A case's line in a speed test's log: its section, code, cost, budget and time for each item, then, where it has one, its [`Footprint`]. `basedpl.bpltests.logged_costs` reads these lines.
pub fn line(section: &str, code: &str, cost: f64, budget: &str, seconds: f64, footprint: Option<Footprint>) -> String {
    let mut line = format!("{section} {code}: {cost:.2}, budget {budget}, {:.3} ns", seconds * 1e9);
    if let Some(f) = footprint { line += &format!(", footprint {:.1} MiB copied at {:.3} ns", f.bytes as f64 / (1 << 20) as f64, f.seconds * 1e9); }
    line
}

/// The budget columns, in the order that basedpl's speed-file footers and both projects' budget files give them: native macOS on Apple silicon, native Linux on CI's x86 runner, and basedpl's browser build on CI's runner, which only basedpl's files have.
pub const COLUMNS: [&str; 3] = ["macos", "linux", "wasm"];

/// A case's budget for a cost, by the rule in basedpl's DEV.md: in copies, the cost times 2 under 50 copies, 1.5 from 50 to 500 and 1.3 over 500, or 2 for an `estimate`, rounded up to two significant figures and at least 0.5; in the section `widths`, a 32/64-bit ratio plus 0.25, rounded up to two decimal places.
pub fn budget(section: &str, cost: f64, estimate: bool) -> f64 {
    // Rounds `x` up to `places` decimal places, or to tens, hundreds and so on for negative `places`. The tolerance keeps a product such as 2 × 0.27, which f64 holds a little above 0.54, from rounding up a step.
    let up = |x: f64, places: i32| {
        let scale = 10f64.powi(places.abs());
        if places >= 0 { (x * scale - 1e-9).ceil() / scale } else { (x / scale - 1e-9).ceil() * scale }
    };
    if section == "widths" { return up(cost + 0.25, 2); }
    let factor = if estimate || cost < 50.0 { 2.0 } else if cost <= 500.0 { 1.5 } else { 1.3 };
    let b = (factor * cost).max(0.5);
    up(b, 1 - b.log10().floor() as i32)
}

/// Why a column's budget is held above its rule budget: a gap in the code, which a `held` line names, or CI's busy hosts, which a `busy` line names. A busy neighbour on CI's runner slows cases whose data passes one core's cache, in some runs and not in others.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Cause { Gap, Busy }

impl Cause {
    /// The keyword of a line that holds a column for this cause: `held` or `busy`.
    pub fn name(self) -> &'static str { match self { Cause::Gap => "held", Cause::Busy => "busy" } }
}

/// One column of a case held above its rule budget, at the cost that the case's `held` or `busy` line gives. A cost that is an `estimate` keeps twice the estimate as its budget, as [`budget`] gives an estimate's.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Held { pub column: usize, pub cost: f64, pub cause: Cause, pub estimate: bool }

/// Adds to `holds` the columns that a `held` or `busy` line holds, from its text after its file's marker, and says whether `text` was such a line. The line gives its keyword, then each column's name from [`COLUMNS`] with the cost it is held at, then a colon and the reason, as in `held linux 15.6 wasm 18.9: the cause isn't known yet`. A `~` before a cost marks an estimate that no run has measured on that column's platform, as in `held linux ~107`. Text that starts with either keyword in any other form is an error, as is a column that `holds` already holds.
pub fn hold(text: &str, holds: &mut Vec<Held>) -> Result<bool, String> {
    let (cause, rest) = match text.split_once(' ') {
        Some((keyword, rest)) if keyword == Cause::Gap.name() => (Cause::Gap, rest),
        Some((keyword, rest)) if keyword == Cause::Busy.name() => (Cause::Busy, rest),
        _ => return Ok(false),
    };
    let fail = || format!("{text:?} should be a column, the cost it's held at, any more columns and costs, then a colon and the reason");
    let (columns, _) = rest.split_once(':').filter(|(_, reason)| !reason.trim().is_empty()).ok_or_else(fail)?;
    let fields: Vec<&str> = columns.split_whitespace().collect();
    if fields.is_empty() || fields.len() % 2 == 1 { return Err(fail()); }
    for pair in fields.chunks(2) {
        let column = COLUMNS.iter().position(|&c| c == pair[0]).ok_or_else(fail)?;
        let (estimate, cost) = pair[1].strip_prefix('~').map_or((false, pair[1]), |c| (true, c));
        let cost = cost.parse::<f64>().ok().filter(|c| c.is_finite() && *c > 0.0).ok_or_else(fail)?;
        if holds.iter().any(|h| h.column == column) { return Err(format!("{text:?}: the {} column is held twice", pair[0])); }
        holds.push(Held { column, cost, cause, estimate });
    }
    Ok(true)
}

/// The budget that a case meets in one column: its rule budget, from its footer or budget line, or, where a `held` or `busy` line holds the column, the budget for the cost it is held at, if that is larger.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Budget { pub value: f64, pub rule: f64, pub cause: Option<Cause> }

impl Budget {
    /// The budget in `column` of a case in `section` whose footer or budget line gives `budgets`, held where `holds` holds that column. `None` where `column` is `None` or the case has no budget there.
    pub fn of(section: &str, budgets: &[impl AsRef<str>], holds: &[Held], column: Option<usize>) -> Option<Self> {
        let column = column?;
        let rule: f64 = budgets.get(column)?.as_ref().parse().ok()?;
        Some(match holds.iter().find(|h| h.column == column) {
            Some(h) => Self { value: budget(section, h.cost, h.estimate).max(rule), rule, cause: Some(h.cause) },
            None => Self { value: rule, rule, cause: None },
        })
    }

    /// Whether `cost` is over the budget.
    pub fn over(&self, cost: f64) -> bool { cost > self.value }

    /// Whether a gap's hold has outlived the gap: `cost` is at most 0.9 times the rule budget, inside the margin of 10% under which basedpl's budget review holds a column. A busy host's hold is never stale, since a quiet host's run meets the rule budget anyway.
    pub fn stale(&self, cost: f64) -> bool { self.cause == Some(Cause::Gap) && cost <= 0.9 * self.rule }
}

/// The budget as a speed test's line gives it: `16`, or with the hold and the rule budget, `32 held from 16` or `5.4 busy from 2.5`.
impl std::fmt::Display for Budget {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self.cause {
            None => write!(f, "{}", self.value),
            Some(cause) => write!(f, "{} {} from {}", self.value, cause.name(), self.rule),
        }
    }
}

/// One case of a budget file: its section, its code, its budgets as written, the columns that its `held` and `busy` lines hold, and the byte span of its line.
pub struct Listed<'a> { pub section: &'a str, pub code: &'a str, pub budgets: Vec<&'a str>, pub holds: Vec<Held>, pub span: (usize, usize) }

/// The cases of `text`, a budget file such as rustymath's `tests/speed.txt`. A line `[name]`, or `[name] ` and more text, starts each section. Each line under it holds a case's code, then its budgets, separated by tabs. A line `# held ...` or `# busy ...` above a case's line holds columns of that case, as [`hold`] reads it. Blank lines and other lines that start with `#` are skipped.
pub fn cases(text: &str) -> Result<Vec<Listed<'_>>, String> {
    let (mut cases, mut section, mut holds, mut start) = (Vec::new(), None, Vec::new(), 0);
    for whole in text.split_inclusive('\n') {
        let line = whole.trim_end_matches(['\n', '\r']);
        let span = (start, start + line.len());
        start += whole.len();
        if line.starts_with('#') { hold(line.strip_prefix("# ").unwrap_or_default(), &mut holds)?; continue; }
        if line.is_empty() { continue; }
        let named = line.strip_prefix('[').and_then(|l| l.split_once(']')).filter(|(name, rest)| !name.contains(' ') && (rest.is_empty() || rest.starts_with(' ')));
        if let Some((name, _)) = named { section = Some(name); continue; }
        let section = section.ok_or_else(|| format!("{line:?} comes before the first section"))?;
        let mut fields = line.split('\t');
        let (code, budgets): (&str, Vec<&str>) = (fields.next().unwrap_or_default(), fields.collect());
        if budgets.iter().any(|b| b.parse::<f64>().map_or(true, |b| !b.is_finite() || b < 0.0)) { return Err(format!("{line:?} has a budget that isn't a number")); }
        cases.push(Listed { section, code, budgets, holds: std::mem::take(&mut holds), span });
    }
    if holds.is_empty() { Ok(cases) } else { Err("a held or busy line ends the file, with no case after it".into()) }
}

/// Each case of `text`, a budget file, as its section, its code and its budget in `column`, as [`Budget::of`] gives it, or with `None` for every case where `column` is `None`. A case with no budget in `column` is an error.
pub fn budgeted(text: &str, column: Option<usize>) -> Result<Vec<(&str, &str, Option<Budget>)>, String> {
    cases(text)?.into_iter().map(|c| match (column, Budget::of(c.section, &c.budgets, &c.holds, column)) {
        (Some(i), None) => Err(format!("{} {} has no {} budget", c.section, c.code, COLUMNS[i])),
        (_, budget) => Ok((c.section, c.code, budget)),
    }).collect()
}

/// The pauses, in seconds, before each retry in `retried`. All of a case's runs together take a fraction of a millisecond, and one disturbance on a shared machine can last that long. The last pause waits out a neighbour's longer burst: in basedpl's `Test` run 62 on CI's runner, cases over their budgets ran slower at each of the first two retries.
pub const RETRY_PAUSES: [f64; 3] = [0.1, 1.0, 10.0];

/// Keeps the calling thread busy for 0.1 s by reading the clock, so that a timing after a pause runs at the core's full speed. An idle core runs slower until it has been busy for a while. A timing's `RUNS` calls of a small case finish within microseconds, before the core speeds up: on the development Mac, cases of 65,536 items that cost 0.74-0.84 copies cost 2.28-3.06 straight after a 10 s pause (30 October 2026). [`retried`] calls it after each pause, and the Python and browser speed tests call it after theirs.
pub fn warm() {
    let start = Instant::now();
    while start.elapsed() < Duration::from_millis(100) {}
}

/// Which of a budget file's two native budgets this machine's speed tests compare costs with: 0, for macOS on Apple silicon, where matrix products run on Accelerate, or 1, for Linux on x86-64, as CI's runner is. Other machines, such as Linux on Arm, have no budgets: their speed tests report costs without comparing them. [`checks_budgets`] says whether a cost over its budget fails.
pub const NATIVE_COLUMN: Option<usize> = if cfg!(all(target_os = "macos", target_arch = "aarch64")) {
    Some(0)
} else if cfg!(all(target_os = "linux", target_arch = "x86_64")) { Some(1) }
else { None };

/// The CPU that this build's budgets were set for, as [`cpu`] names it: the development Mac's for macOS on Apple silicon, and CI's runner's for Linux on x86-64 and for the browser build. `None` for a build with no budgets, such as Linux on Arm.
pub const BUDGET_CPU: Option<&str> = match NATIVE_COLUMN {
    Some(0) => Some(DEV_CPU), Some(_) => Some(CI_CPU), None if cfg!(target_arch = "wasm32") => Some(CI_CPU), None => None
};

/// Whether this machine's speed tests fail on a case still over its budget after the retries: only where [`cpu`] is [`BUDGET_CPU`]. On any other CPU they print every cost, then each case still over its budget, marked `over`, and fail on none.
pub fn checks_budgets() -> bool { BUDGET_CPU.is_some_and(|c| cpu() == Some(c)) }

/// The failures of a timing test that still fail after every retry. After each pause in `RETRY_PAUSES`, and [`warm`] after it, `retime` gets the failures and the pause, times them again, prints the retry and gives those that still fail. A speed test calls it after timing every case.
pub fn retried<T>(mut failures: Vec<T>, mut retime: impl FnMut(Vec<T>, f64) -> Vec<T>) -> Vec<T> {
    for pause in RETRY_PAUSES {
        if failures.is_empty() { break; }
        std::thread::sleep(Duration::from_secs_f64(pause));
        warm();
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

    /// The file that a timing holds exclusively while it waits for `LOCK` and while it holds it. A shared hold locks it for a moment first, so it waits behind any timing.
    pub const GATE: &str = "/tmp/aai-ws-timing.gate";

    /// A hold on the timing lock, released when dropped. The operating system also releases it when its process ends, even by a crash. An exclusive hold runs every call on its calling thread alone until it's dropped, because speed budgets are for one thread.
    pub struct Hold {
        _file: File,
        /// `GATE`, which an exclusive hold keeps until it's dropped.
        _gate: Option<File>,
        /// The thread setting that an exclusive hold replaced, to restore when dropped.
        threads: Option<usize>,
    }

    impl Drop for Hold { fn drop(&mut self) { if let Some(n) = self.threads { set_threads(n); } } }

    /// Takes the timing lock exclusively, for a timing. It waits up to 5 minutes for builds, test runs and other timings, then fails. From the start of its wait it holds `GATE`, so no build or test run starts meanwhile. After waiting 30 seconds it prints, once, the other processes that have the lock's file open. On macOS it also sets the calling thread's QoS class to user-interactive. macOS then favours the performance cores for that thread. QoS isn't CPU affinity.
    pub fn exclusive() -> io::Result<Hold> { hold(false) }

    /// Takes the timing lock shared, for a build or a test run. It waits behind any timing that holds `GATE` or the lock, for up to 5 minutes, then goes ahead without the lock.
    pub fn shared() -> io::Result<Hold> { hold(true) }

    fn hold(shared: bool) -> io::Result<Hold> {
        #[cfg(target_vendor = "apple")]
        if !shared {
            unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
        }
        let open = |path: &str| File::options().create(true).append(true).open(path).map_err(|e| io::Error::new(e.kind(), format!("{path}: {e}")));
        let (gate, file) = (open(GATE)?, open(LOCK)?);
        let mut wait = Wait { start: Instant::now(), shared, waiting: false, reported: false, late: false };
        wait.take(&gate, GATE)?;
        wait.take(&file, LOCK)?;
        Ok(Hold { _file: file, _gate: (!shared).then_some(gate), threads: (!shared).then(|| set_threads(1)) })
    }

    /// One hold's wait for `GATE` and then `LOCK`, against one deadline of 5 minutes.
    struct Wait { start: Instant, shared: bool, waiting: bool, reported: bool, late: bool }

    impl Wait {
        /// Locks `file`, which is the file at `path`, as the hold needs it, waiting while others hold it. After 5 minutes a shared hold goes ahead without it, and an exclusive hold fails.
        fn take(&mut self, file: &File, path: &str) -> io::Result<()> {
            let others = if self.shared { "a timing" } else { "a build or another timing" };
            loop {
                if self.late { return Ok(()); }
                match if self.shared { file.try_lock_shared() } else { file.try_lock() } {
                    Ok(()) => return Ok(()), Err(TryLockError::WouldBlock) => {} Err(TryLockError::Error(e)) => return Err(io::Error::new(e.kind(), format!("{path}: {e}")))
                }
                if self.start.elapsed() > Duration::from_secs(300) {
                    if !self.shared { return Err(io::Error::new(io::ErrorKind::TimedOut, format!("{others} has held {LOCK} for 5 minutes"))); }
                    eprintln!("going ahead although {others} still holds {LOCK}");
                    self.late = true;
                    return Ok(());
                }
                if !self.waiting { eprintln!("waiting for {others} to release {LOCK}"); }
                if !self.shared && !self.reported && self.start.elapsed() > Duration::from_secs(30) {
                    eprintln!("{LOCK} is still taken after 30 seconds. Processes with it open: {}", users());
                    self.reported = true;
                }
                self.waiting = true;
                std::thread::sleep(Duration::from_millis(250));
            }
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
