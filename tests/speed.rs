//! How fast each function runs on 65,536 inputs, against the standard library's version. A function that is slower than the standard
//! library has usually stopped vectorizing. `TWICE` lists the functions allowed up to twice its time. The functions that the standard
//! library lacks have budgets in copies of the same number of floats instead, in `special`.
//! `RUSTYMATH_TIMES=1 cargo test --test speed -- --nocapture` prints the README's table.
use rustymath::*;
use std::{hint::black_box, time::Instant};

const N: usize = 65_536;
/// The functions that may take up to twice the standard library's time. Their f64 versions read tables, and NEON can't vectorize
/// table reads. The Mac's libm beats them at both widths.
const TWICE: [&str; 2] = ["pow", "root"];

/// The fastest of a few runs of `f`, which writes one output for each of `inputs`, in ns per item.
fn time_slice<A, T: Copy + Default>(inputs: &[A], f: impl Fn(&[A], &mut [T])) -> f64 {
    let mut out = vec![T::default(); inputs.len()];
    (0..5)
        .map(|_| {
            let start = Instant::now();
            f(black_box(inputs), &mut out);
            black_box(&mut out);
            start.elapsed().as_secs_f64() * 1e9 / inputs.len() as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// The fastest of a few runs of `f` on each of `inputs`, in ns per item.
fn time<A: Copy, T: Copy + Default>(inputs: &[A], f: impl Fn(A) -> T) -> f64 {
    time_slice(inputs, |xs, out| for (o, &x) in out.iter_mut().zip(xs) { *o = f(x); })
}

/// Horner's rule for one item, as a loop over the items would call it with the standard library's `mul_add`.
fn horner<F: Float>(c: &[F], x: F) -> F { c.iter().rev().fold(F::zero(), |y, &k| y.mul_add(x, k)) }

/// `N` inputs from `lo` to `hi`, from the fractional parts of multiples of `step`. Their order has no pattern that a branch predictor
/// could learn.
fn inputs(lo: f64, hi: f64, step: f64) -> Vec<f64> { (0..N).map(|i| lo + (hi - lo) * (i as f64 * step).fract()).collect() }
const GOLDEN: f64 = 0.618_033_988_749_894_9;
const SILVER: f64 = 0.414_213_562_373_095_1;
const BRONZE: f64 = 0.302_775_637_731_994_6;

/// `xs`, and the same inputs narrowed to f32 by `narrow`.
fn widths<T: Copy, U>(xs: Vec<T>, narrow: impl Fn(T) -> U) -> (Vec<T>, Vec<U>) {
    let narrowed = xs.iter().map(|&x| narrow(x)).collect();
    (xs, narrowed)
}

/// Pairs from the ranges `xs` and `ys`.
fn pairs(xs: (f64, f64), ys: (f64, f64)) -> Vec<(f64, f64)> { inputs(xs.0, xs.1, GOLDEN).into_iter().zip(inputs(ys.0, ys.1, SILVER)).collect() }

/// `sin`, `cos` and `tan` without the branch for large arguments, as a loop that vectorizes calls them.
fn sin<F: Float>(x: F) -> F { checked_sin(x).unwrap_or(x) }
fn cos<F: Float>(x: F) -> F { checked_cos(x).unwrap_or(x) }
fn tan<F: Float>(x: F) -> F { checked_tan(x).unwrap_or(x) }

/// The times of a function and of a reference version on inputs from `lo` to `hi`. `f` holds both at f64. `g` holds both at f32.
fn unary(
    name: &'static str,
    (lo, hi): (f64, f64),
    f: (impl Fn(f64) -> f64, impl Fn(f64) -> f64),
    g: (impl Fn(f32) -> f32, impl Fn(f32) -> f32),
) -> (&'static str, [f64; 4]) {
    let (x64, x32) = widths(inputs(lo, hi, GOLDEN), |x| x as f32);
    (name, [time(&x64, f.0), time(&x64, f.1), time(&x32, g.0), time(&x32, g.1)])
}

/// The times of `name` and of the standard library's `std`, at f64 and then f32, on inputs from `lo` to `hi`.
macro_rules! unary {
    ($($name:ident = $std:ident on $lo:literal to $hi:literal;)+) => {
        [$(unary(stringify!($name), ($lo, $hi), ($name::<f64>, f64::$std), ($name::<f32>, f32::$std))),+]
    };
}

/// The times of a function and of the standard library's version on pairs from the ranges `xs` and `ys`. `f` holds both at f64. `g`
/// holds both at f32.
fn binary(
    name: &'static str,
    (xs, ys): ((f64, f64), (f64, f64)),
    f: (impl Fn(f64, f64) -> f64, impl Fn(f64, f64) -> f64),
    g: (impl Fn(f32, f32) -> f32, impl Fn(f32, f32) -> f32),
) -> (&'static str, [f64; 4]) {
    let (p64, p32) = widths(pairs(xs, ys), |(x, y)| (x as f32, y as f32));
    (name, [time(&p64, |(x, y)| f.0(x, y)), time(&p64, |(x, y)| f.1(x, y)), time(&p32, |(x, y)| g.0(x, y)), time(&p32, |(x, y)| g.1(x, y))])
}

/// The time to copy `N` items of `T`, in ns per item: the unit of the special functions' budgets.
fn copy<T: Copy + Default>() -> f64 { time_slice(&vec![T::default(); N], |x, out: &mut [T]| out.copy_from_slice(x)) }

/// The time per item of `f` on the first `n` of `x64` and of `g` on the first `n` of `x32`, over `unit`'s time per item of a copy at
/// each width.
fn copies<A: Copy, B: Copy>(n: usize, unit: [f64; 2], (x64, x32): &(Vec<A>, Vec<B>), f: impl Fn(A) -> f64, g: impl Fn(B) -> f32) -> [f64; 2] {
    [time(&x64[..n], f) / unit[0], time(&x32[..n], g) / unit[1]]
}

/// Each function that the standard library lacks, with the most copies' worth of time it may take at f64 and at f32, then its cost.
/// A loop that stops vectorizing takes several times as long. The incomplete gamma and beta functions iterate, so they run on 1,024
/// inputs, and the others on 16,384.
fn special() -> [(&'static str, [f64; 2], [f64; 2]); 11] {
    let (many, few, unit) = (N / 4, N / 64, [copy::<f64>(), copy::<f32>()]);
    let x = |lo, hi| widths(inputs(lo, hi, GOLDEN), |x| x as f32);
    let p = |xs, ys| widths(pairs(xs, ys), |(x, y)| (x as f32, y as f32));
    let t = |xs, ys, (lo, hi): (f64, f64)| {
        let t64 = pairs(xs, ys).into_iter().zip(inputs(lo, hi, BRONZE)).map(|((x, y), z)| (x, y, z)).collect();
        widths(t64, |(x, y, z)| (x as f32, y as f32, z as f32))
    };
    let (gi, gp) = (p((0.5, 20.0), (0.0, 30.0)), p((0.5, 20.0), (0.001, 0.999)));
    let (bi, bp) = (t((0.5, 20.0), (0.5, 20.0), (0.0, 1.0)), t((0.5, 20.0), (0.5, 20.0), (0.001, 0.999)));
    [
        ("gamma", [460.0, 350.0], copies(many, unit, &x(0.1, 20.0), gamma::<f64>, gamma::<f32>)),
        ("lgamma", [240.0, 220.0], copies(many, unit, &x(0.1, 1000.0), lgamma::<f64>, lgamma::<f32>)),
        ("erf", [85.0, 48.0], copies(many, unit, &x(-5.0, 5.0), erf::<f64>, erf::<f32>)),
        ("erfc", [130.0, 150.0], copies(many, unit, &x(-5.0, 10.0), erfc::<f64>, erfc::<f32>)),
        ("erfinv", [410.0, 95.0], copies(many, unit, &x(-0.999, 0.999), erfinv::<f64>, erfinv::<f32>)),
        ("erfcinv", [410.0, 95.0], copies(many, unit, &x(0.001, 1.999), erfcinv::<f64>, erfcinv::<f32>)),
        ("gammainc", [1250.0, 2430.0], copies(few, unit, &gi, |(a, x)| gammainc(a, x), |(a, x)| gammainc(a, x))),
        ("gammaincc", [1250.0, 2430.0], copies(few, unit, &gi, |(a, x)| gammaincc(a, x), |(a, x)| gammaincc(a, x))),
        ("gammaincinv", [4900.0, 19900.0], copies(few, unit, &gp, |(a, p)| gammaincinv(a, p), |(a, p)| gammaincinv(a, p))),
        ("betainc", [3300.0, 17600.0], copies(few, unit, &bi, |(a, b, x)| betainc(a, b, x), |(a, b, x)| betainc(a, b, x))),
        ("betaincinv", [12000.0, 51000.0], copies(few, unit, &bp, |(a, b, p)| betaincinv(a, b, p), |(a, b, p)| betaincinv(a, b, p))),
    ]
}

#[test]
fn speed() {
    let mut rows = unary! {
        exp = exp on -20.0 to 20.0;
        expm1 = exp_m1 on -1.0 to 1.0;
        ln = ln on 0.001 to 1000.0;
        log1p = ln_1p on -0.5 to 1000.0;
        log2 = log2 on 0.001 to 1000.0;
        log10 = log10 on 0.001 to 1000.0;
        sin = sin on -10.0 to 10.0;
        cos = cos on -10.0 to 10.0;
        tan = tan on -10.0 to 10.0;
        asin = asin on -1.0 to 1.0;
        acos = acos on -1.0 to 1.0;
        atan = atan on -10.0 to 10.0;
        sinh = sinh on -10.0 to 10.0;
        cosh = cosh on -10.0 to 10.0;
        tanh = tanh on -10.0 to 10.0;
        asinh = asinh on -1000.0 to 1000.0;
        acosh = acosh on 1.0 to 1000.0;
        atanh = atanh on -0.999 to 0.999;
        cbrt = cbrt on -1000.0 to 1000.0;
    }
    .to_vec();
    rows.push(binary("pow", ((0.0, 4.0), (-30.0, 30.0)), (pow, f64::powf), (pow, f32::powf)));
    rows.push(binary("hypot", ((-1000.0, 1000.0), (-1000.0, 1000.0)), (hypot, f64::hypot), (hypot, f32::hypot)));
    rows.push(binary("root", ((5.0, 5.0), (0.0, 1000.0)), (root, |n: f64, y: f64| y.powf(1.0 / n)), (root, |n: f32, y: f32| y.powf(1.0 / n))));
    // A polynomial of degree 7 at each of the inputs, against `horner` for one input at a time. `black_box` hides the degree, which a
    // caller rarely knows when it compiles.
    let c64 = [0.5, -1.25, 0.75, 2.0, -0.125, 0.0625, -1.5, 0.25];
    let c32 = c64.map(|k| k as f32);
    let (c64, c32): (&[f64], &[f32]) = (black_box(&c64), black_box(&c32));
    let x64 = inputs(-2.0, 2.0, GOLDEN);
    let x32: Vec<f32> = x64.iter().map(|&x| x as f32).collect();
    rows.push((
        "polynomial::evaluate",
        [
            time_slice(&x64, |x, out| { out.copy_from_slice(x); polynomial::evaluate(c64, out) }),
            time(&x64, |x| horner(c64, x)),
            time_slice(&x32, |x, out| { out.copy_from_slice(x); polynomial::evaluate(c32, out) }),
            time(&x32, |x| horner(c32, x)),
        ],
    ));
    let special = special();
    if std::env::var_os("RUSTYMATH_TIMES").is_some() {
        println!("| Function | f64 | std f64 | f32 | std f32 |\n|---|---|---|---|---|");
        for (name, t) in rows.iter() { println!("| `{name}` | {:.2} | {:.2} | {:.2} | {:.2} |", t[0], t[1], t[2], t[3]); }
        println!("\n| Function | f64 copies | f32 copies |\n|---|---|---|");
        for (name, _, c) in special.iter() { println!("| `{name}` | {:.0} | {:.0} |", c[0], c[1]); }
    }
    let limit = |name| if TWICE.contains(&name) { 2.0 } else { 1.0 };
    let slower: Vec<_> =
        rows.iter().flat_map(|&(name, t)| [(name, "f64", t[0], t[1]), (name, "f32", t[2], t[3])]).filter(|r| r.2 >= r.3 * limit(r.0)).collect();
    let over: Vec<_> =
        special.iter().flat_map(|&(name, b, c)| [(name, "f64", c[0], b[0]), (name, "f32", c[1], b[1])]).filter(|r| r.2 > r.3).collect();
    assert!(slower.is_empty(), "slower than the standard library allows, in ns per item: {slower:?}");
    assert!(over.is_empty(), "over budget, in copies: {over:?}");
}
