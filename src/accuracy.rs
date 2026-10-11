//! The accuracy tests. Each function, with `mla` fused and unfused, must be within `BOUND` of the right result at every f16 input, at
//! samples of f32 and f64 inputs, and at published test cases. Each check prints its worst error.
//!
//! The elementary functions are checked against the platform's libm at f64, through Rust's standard library. Rounded to f32 or f16,
//! its result is correctly rounded at those widths, apart from rare double rounding. At f64 the libm has errors of its own, which
//! `LIBM` allows for. The hard cases in `testdata/core-math.txt`, and the special functions in `testdata/mpmath.txt`, carry correctly
//! rounded results that `testdata/references.py` computes with mpmath.
use half::f16;
use std::fmt::Debug;

/// Every function's error bound, in ulps, at every width.
const BOUND: u64 = 4;
/// The error, in ulps, that a check against the platform's libm at f64 allows on top of `BOUND`. glibc documents errors of up to 2
/// ulps for some of these functions, such as `sinh` and `tanh`.
const LIBM: u64 = 2;
/// The number of f64 inputs, or of f32 inputs from a range, that each check samples.
const SAMPLES: u64 = 1 << 14;

/// A float type whose errors the tests measure.
trait Measured: Copy + Debug {
    /// A key whose order is the order of the floats, with -0 just below 0. Adjacent floats have adjacent keys.
    fn key(self) -> u64;
    fn is_nan(self) -> bool;
}

/// The input of a function under test: one float, or several. An input with a zero, infinite or NaN argument is special, and must
/// give exactly IEEE's result.
trait Input: Copy + Debug {
    fn is_special(self) -> bool;
    /// The input from the fields of a line of test data.
    fn parse(fields: &[&str]) -> Self;
}

macro_rules! measured {
    ($($t:ty: $bits:ty),+) => {$(
        impl Measured for $t {
            fn key(self) -> u64 {
                let (bits, sign) = (self.to_bits(), 1 << (<$bits>::BITS - 1));
                (if bits & sign != 0 { !bits } else { bits | sign }) as u64
            }
            fn is_nan(self) -> bool { num_traits::Float::is_nan(self) }
        }
        impl Input for $t {
            fn is_special(self) -> bool { !num_traits::Float::is_normal(self) && !num_traits::Float::is_subnormal(self) }
            fn parse(fields: &[&str]) -> Self { fields[0].parse().unwrap() }
        }
        impl Input for ($t, $t) {
            fn is_special(self) -> bool { self.0.is_special() || self.1.is_special() }
            fn parse(fields: &[&str]) -> Self { (fields[0].parse().unwrap(), fields[1].parse().unwrap()) }
        }
        impl Input for ($t, $t, $t) {
            fn is_special(self) -> bool { self.0.is_special() || self.1.is_special() || self.2.is_special() }
            fn parse(fields: &[&str]) -> Self { (fields[0].parse().unwrap(), fields[1].parse().unwrap(), fields[2].parse().unwrap()) }
        }
    )+};
}
measured!(f16: u16, f32: u32, f64: u64);

/// The error of `got` in ulps, against the result `want` at `x`. A special input or a NaN must give exactly the right result, or the
/// error is `u64::MAX`.
fn error<A: Input, T: Measured>(x: A, got: T, want: T) -> u64 {
    let exact = |same: bool| if same { 0 } else { u64::MAX };
    if want.is_nan() || got.is_nan() { exact(want.is_nan() && got.is_nan()) }
    else if x.is_special() { exact(got.key() == want.key()) }
    else { got.key().abs_diff(want.key()) }
}

/// The worst error of `f` against `reference` at the inputs `input(i)` for each `i` below `n`, with the input that gives it.
fn worst<A: Input, T: Measured>(n: u64, input: impl Fn(u64) -> A, f: impl Fn(A) -> T, reference: impl Fn(A) -> T) -> (u64, A) {
    (0..n).map(|i| { let x = input(i); (error(x, f(x), reference(x)), x) }).max_by_key(|&(e, _)| e).unwrap()
}

/// The worst error of `f` over `cases` of inputs and right results, with the input that gives it, where there are any.
fn worst_case<A: Input, T: Measured>(cases: &[(A, T)], f: impl Fn(A) -> T) -> Option<(u64, A)> {
    cases.iter().map(|&(x, want)| (error(x, f(x), want), x)).max_by_key(|&(e, _)| e)
}

/// `check` for the worst error of `f` over `cases` of inputs and right results, where there are any.
fn check_cases<A: Input, T: Measured>(name: &str, cases: &[(A, T)], f: impl Fn(A) -> T) {
    if let Some(worst) = worst_case(cases, f) { check(name, worst); }
}

/// Every case in `cases` must come out exactly.
fn exactly<A: Input, T: Measured>(name: &str, cases: &[(A, T)], f: impl Fn(A) -> T) { within(name, 0, worst_case(cases, f).unwrap()) }

/// Asserts that the worst error `(e, x)` of the function called `name` is within `BOUND`, and prints it.
fn check<A: Debug>(name: &str, worst: (u64, A)) { within(name, BOUND, worst) }

/// `check` for a check against the platform's libm at f64, which allows `LIBM` more.
fn check_libm<A: Debug>(name: &str, worst: (u64, A)) { within(name, BOUND + LIBM, worst) }

fn within<A: Debug>(name: &str, bound: u64, (e, x): (u64, A)) {
    eprintln!("{name}: worst error {e}, at {x:?}");
    assert!(e <= bound, "{name} is {e} out at {x:?}, beyond the bound of {bound}");
}

/// A random 64-bit number for each `i`: splitmix64's output.
pub(crate) fn random(i: u64) -> u64 {
    let z = i.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
/// A random f64 from `lo` to `hi`.
fn uniform(i: u64, (lo, hi): (f64, f64)) -> f64 { lo + (hi - lo) * (random(i) >> 11) as f64 / (1u64 << 53) as f64 }
/// A random f64 of either sign whose bits are random, so that small magnitudes are as likely as large ones. It includes the
/// subnormals, the infinities and NaN.
fn scattered(i: u64) -> f64 { f64::from_bits(random(i)) }
/// The `i`th of 2^16 f32 inputs, whose top 16 bits are `i` and whose low bits are random. Every sign and exponent gets the same share
/// of them.
fn spread32(i: u64) -> f32 { f32::from_bits((i << 16) as u32 | (random(i) as u32 & 0xffff)) }

/// The files of inputs with their correctly rounded results, by source. Each file's header names its source and licence.
const VECTORS: [(&str, &str); 3] = [
    ("libc-test", include_str!("../testdata/libc-test.txt")),
    ("Arm", include_str!("../testdata/arm.txt")),
    ("compiler-builtins", include_str!("../testdata/compiler-builtins.txt")),
];
const CORE_MATH: &str = include_str!("../testdata/core-math.txt");
const MPMATH: &str = include_str!("../testdata/mpmath.txt");

/// The cases in the vector file `text` for the function called `name`: each input with its right result.
fn vectors<A: Input, T: std::str::FromStr<Err: Debug>>(text: &str, name: &str) -> Vec<(A, T)> {
    rows(text, name).map(|r| (A::parse(&r), r[r.len() - 1].parse().unwrap())).collect()
}
/// The fields after the name of each line of a test data file that belongs to the function called `name`.
fn rows<'a>(text: &'a str, name: &'a str) -> impl Iterator<Item = Vec<&'a str>> + 'a {
    text.lines().filter(|l| !l.starts_with('#')).map(|l| l.split_whitespace().collect::<Vec<_>>()).filter(move |r| r[0] == name).map(|r| r[1..].to_vec())
}
/// The `cases` whose input is a float at f32 too, at f32.
fn at32(cases: &[(f64, f64)]) -> Vec<(f32, f32)> {
    cases.iter().filter(|&&(x, _)| x as f32 as f64 == x || x.is_nan()).map(|&(x, y)| (x as f32, y as f32)).collect()
}

/// The function at f64 and f32 with `mla` fused, and the same unfused.
type Variants<W, N> = [(W, N, &'static str); 2];
macro_rules! variants {
    ($f:ident) => {
        [
            (crate::fused::$f::<f64> as fn(f64) -> f64, crate::fused::$f::<f32> as fn(f32) -> f32, "fused"),
            (crate::unfused::$f::<f64>, crate::unfused::$f::<f32>, "unfused"),
        ]
    };
    ($f:ident, 2) => {
        [
            (
                (|(x, y)| crate::fused::$f::<f64>(x, y)) as fn((f64, f64)) -> f64,
                (|(x, y)| crate::fused::$f::<f32>(x, y)) as fn((f32, f32)) -> f32,
                "fused",
            ),
            (|(x, y)| crate::unfused::$f::<f64>(x, y), |(x, y)| crate::unfused::$f::<f32>(x, y), "unfused"),
        ]
    };
}

/// The checks of a function of one argument, called `name` in the test data, against `libm`, the platform's version at f64: 2^16
/// f32 inputs spread over every exponent, `SAMPLES` f64 inputs with random bits, `SAMPLES` uniform inputs at each width from each of
/// `ranges`, `extra` inputs and every f16 input. It also checks the hard cases in `testdata/core-math.txt` and the cases in `VECTORS`.
fn unary(
    name: &str,
    variants: Variants<fn(f64) -> f64, fn(f32) -> f32>,
    f16: fn(f16) -> f16,
    libm: fn(f64) -> f64,
    ranges: &[(f64, f64)],
    extra: &[f64],
) {
    let narrow_libm = |x: f32| libm(x.into()) as f32;
    let hard: Vec<(f64, f64)> = vectors(CORE_MATH, name);
    for (wide, narrow, how) in variants {
        check(&format!("f32 {name}, {how}"), worst(1 << 16, spread32, narrow, narrow_libm));
        check_libm(&format!("f64 {name} at scattered inputs, {how}"), worst(SAMPLES, scattered, wide, libm));
        for &range in ranges {
            check_libm(&format!("f64 {name} in {range:?}, {how}"), worst(SAMPLES, |i| uniform(i, range), wide, libm));
            check(&format!("f32 {name} in {range:?}, {how}"), worst(SAMPLES, |i| uniform(i, range) as f32, narrow, narrow_libm));
        }
        if !extra.is_empty() { check_libm(&format!("f64 {name} at extra inputs, {how}"), worst(extra.len() as u64, |i| extra[i as usize], wide, libm)); }
        check_cases(&format!("CORE-MATH {name}, {how}"), &hard, wide);
        for (source, text) in VECTORS {
            check_cases(&format!("{source} {name}, {how}"), &vectors::<f64, f64>(text, name), wide);
            check_cases(&format!("{source} {name}f, {how}"), &vectors::<f32, f32>(text, &format!("{name}f")), narrow);
        }
    }
    check(&format!("f16 {name}"), worst(1 << 16, |i| f16::from_bits(i as u16), f16, |x| f16::from_f64(libm(x.into()))));
}

/// The checks of a function of two arguments, called `name` in the test data, against `libm`, the platform's version at f64: 2^16 f32
/// pairs and `SAMPLES` f64 pairs with random bits, `SAMPLES` uniform pairs at each width from each of `ranges`, `extra` pairs and 2^16
/// random f16 pairs. It also checks the hard cases in `testdata/core-math.txt` and the cases in `VECTORS`.
fn binary(
    name: &str,
    variants: Variants<fn((f64, f64)) -> f64, fn((f32, f32)) -> f32>,
    f16: fn(f16, f16) -> f16,
    libm: fn(f64, f64) -> f64,
    ranges: &[((f64, f64), (f64, f64))],
    extra: &[(f64, f64)],
) {
    let hard: Vec<((f64, f64), f64)> = vectors(CORE_MATH, name);
    let wide_libm = |(x, y)| libm(x, y);
    let narrow_libm = |(x, y): (f32, f32)| libm(x.into(), y.into()) as f32;
    let pair32 = |i: u64| { let r = random(i); (f32::from_bits(r as u32), f32::from_bits((r >> 32) as u32)) };
    for (wide, narrow, how) in variants {
        check(&format!("f32 {name} at scattered pairs, {how}"), worst(1 << 16, pair32, narrow, narrow_libm));
        check_libm(&format!("f64 {name} at scattered pairs, {how}"), worst(SAMPLES, |i| (scattered(2 * i), scattered(2 * i + 1)), wide, wide_libm));
        for &(xs, ys) in ranges {
            let pair = |i: u64| (uniform(2 * i, xs), uniform(2 * i + 1, ys));
            check_libm(&format!("f64 {name} in {xs:?} by {ys:?}, {how}"), worst(SAMPLES, pair, wide, wide_libm));
            let pair = |i: u64| (uniform(2 * i, xs) as f32, uniform(2 * i + 1, ys) as f32);
            check(&format!("f32 {name} in {xs:?} by {ys:?}, {how}"), worst(SAMPLES, pair, narrow, narrow_libm));
        }
        if !extra.is_empty() { check_libm(&format!("f64 {name} at extra pairs, {how}"), worst(extra.len() as u64, |i| extra[i as usize], wide, wide_libm)); }
        check_cases(&format!("CORE-MATH {name}, {how}"), &hard, wide);
        for (source, text) in VECTORS {
            check_cases(&format!("{source} {name}, {how}"), &vectors::<(f64, f64), f64>(text, name), wide);
            check_cases(&format!("{source} {name}f, {how}"), &vectors::<(f32, f32), f32>(text, &format!("{name}f")), narrow);
        }
    }
    let pair16 = |i: u64| { let r = random(i); (f16::from_bits(r as u16), f16::from_bits((r >> 16) as u16)) };
    let half_libm = |(x, y): (f16, f16)| f16::from_f64(libm(x.into(), y.into()));
    check(&format!("f16 {name} at scattered pairs"), worst(1 << 16, pair16, |(x, y)| f16(x, y), half_libm));
}

/// The platform libm's inverse hyperbolic functions. Rust's standard library computes these with formulas of its own, and its `acosh`
/// loses most of its accuracy near 1.
mod libm {
    unsafe extern "C" {
        pub safe fn asinh(x: f64) -> f64;
        pub safe fn acosh(x: f64) -> f64;
        pub safe fn atanh(x: f64) -> f64;
    }
}

/// The doubles below 2^23 closest to multiples of π/2, from `testdata/reduction.txt`, with their negatives.
fn reduction() -> Vec<f64> {
    let text = include_str!("../testdata/reduction.txt");
    text.lines().filter(|l| !l.starts_with('#')).map(|l| l.split_whitespace().next().unwrap().parse().unwrap()).flat_map(|x: f64| [x, -x]).collect()
}

/// Inputs to the circle functions that reduce by π: every magnitude up to 2^23 and up to 10, and the doubles nearest multiples of π/2.
const TRIG: &[(f64, f64)] = &[(-8388608.0, 8388608.0), (-10.0, 10.0)];

#[test]
fn exp_accuracy() { unary("exp", variants!(exp), crate::exp, f64::exp, &[(-760.0, 720.0)], &[]); }

#[test]
fn expm1_accuracy() { unary("expm1", variants!(expm1), crate::expm1, f64::exp_m1, &[(-50.0, 720.0), (-1.0, 1.0)], &[]); }

#[test]
fn ln_accuracy() { unary("log", variants!(ln), crate::ln, f64::ln, &[(0.5, 2.0)], &[]); }

#[test]
fn log1p_accuracy() { unary("log1p", variants!(log1p), crate::log1p, f64::ln_1p, &[(-1.0, 1.0), (-1e-6, 1e-6)], &[]); }

#[test]
fn log2_accuracy() { unary("log2", variants!(log2), crate::log2, f64::log2, &[(0.5, 2.0)], &[]); }

#[test]
fn log10_accuracy() {
    let powers: Vec<f64> = (-22..=22).map(|k| format!("1e{k}").parse().unwrap()).collect();
    unary("log10", variants!(log10), crate::log10, f64::log10, &[(0.5, 2.0)], &powers);
}

#[test]
fn sin_accuracy() { unary("sin", variants!(sin), crate::sin, f64::sin, TRIG, &reduction()); }

#[test]
fn cos_accuracy() { unary("cos", variants!(cos), crate::cos, f64::cos, TRIG, &reduction()); }

#[test]
fn tan_accuracy() { unary("tan", variants!(tan), crate::tan, f64::tan, TRIG, &reduction()); }

#[test]
fn asin_accuracy() { unary("asin", variants!(asin), crate::asin, f64::asin, &[(-1.0, 1.0)], &[]); }

#[test]
fn acos_accuracy() { unary("acos", variants!(acos), crate::acos, f64::acos, &[(-1.0, 1.0)], &[]); }

#[test]
fn atan_accuracy() { unary("atan", variants!(atan), crate::atan, f64::atan, &[(-4.0, 4.0)], &[]); }

#[test]
fn sinh_accuracy() { unary("sinh", variants!(sinh), crate::sinh, f64::sinh, &[(-1.0, 1.0), (-711.0, 711.0)], &[]); }

#[test]
fn cosh_accuracy() { unary("cosh", variants!(cosh), crate::cosh, f64::cosh, &[(-1.0, 1.0), (-711.0, 711.0)], &[]); }

#[test]
fn tanh_accuracy() { unary("tanh", variants!(tanh), crate::tanh, f64::tanh, &[(-1.0, 1.0), (-25.0, 25.0)], &[]); }

#[test]
fn asinh_accuracy() { unary("asinh", variants!(asinh), crate::asinh, |x| libm::asinh(x), &[(-1.0, 1.0), (-1e20, 1e20)], &[]); }

#[test]
fn acosh_accuracy() { unary("acosh", variants!(acosh), crate::acosh, |x| libm::acosh(x), &[(1.0, 2.0), (1.0, 1e20)], &[]); }

#[test]
fn atanh_accuracy() { unary("atanh", variants!(atanh), crate::atanh, |x| libm::atanh(x), &[(-1.0, 1.0)], &[]); }

#[test]
fn cbrt_accuracy() {
    let cubes: Vec<f64> = (-1000..=1000).map(|k: i32| (k as f64).powi(3)).collect();
    unary("cbrt", variants!(cbrt), crate::cbrt, f64::cbrt, &[(-1e300, 1e300), (-8.0, 8.0)], &cubes);
}

#[test]
fn pow_accuracy() {
    // Powers that are floats, and powers near overflow and underflow of a base near 1, where the error of ln |x| grows most.
    let exact: Vec<(f64, f64)> = (2..=12).flat_map(|b| (-30..=60).map(move |k| (b as f64, k as f64))).collect();
    let near = [((0.5, 2.0), (-1000.0, 1000.0)), ((1.2, 1.5), (1500.0, 2700.0)), ((0.0, 4.0), (-30.0, 30.0)), ((-8.0, 0.0), (-20.0, 20.0))];
    binary("pow", variants!(pow, 2), crate::pow, f64::powf, &near, &exact);
}

#[test]
fn hypot_accuracy() {
    let ranges = [((-1e300, 1e300), (-1e300, 1e300)), ((-1e-300, 1e-300), (-1e-300, 1e-300)), ((-4.0, 4.0), (-4.0, 4.0))];
    binary("hypot", variants!(hypot, 2), crate::hypot, f64::hypot, &ranges, &[]);
}

#[test]
fn root_accuracy() {
    // Degrees whose reciprocals are floats, against pow, and degree 3, against cbrt.
    for n in [2.0, -2.0, 4.0, -4.0, 8.0, 0.5, -0.5, 1.0, -1.0] {
        for (wide, narrow, how) in variants!(root, 2) {
            check_libm(&format!("f64 root {n} at scattered inputs, {how}"), worst(SAMPLES, |i| scattered(i).abs(), |y| wide((n, y)), |y| y.powf(1.0 / n)));
            check(&format!("f32 root {n}, {how}"), worst(1 << 16, |i| spread32(i).abs(), |y| narrow((n as f32, y)), |y| f64::from(y).powf(1.0 / n) as f32));
        }
    }
    let fifths: Vec<f64> = (-100..=100).map(|k: i32| (k as f64).powi(5)).collect();
    for (wide, _, how) in variants!(root, 2) {
        check_libm(&format!("f64 root 3, {how}"), worst(SAMPLES, scattered, |y| wide((3.0, y)), f64::cbrt));
        check(&format!("f64 root 5 of fifth powers, {how}"), worst(fifths.len() as u64, |i| fifths[i as usize], |y| wide((5.0, y)), |y| y.signum() * y.abs().powf(0.2).round()));
    }
}

/// Powers, cube roots and roots whose exact result is a float give it exactly, as BPL's `7*5` must. The cases are whole powers below
/// 2^53, or 2^24 at f32, of the first 3,000 bases and 1,000 random ones for each exponent, cubes of floats with up to 17 significant
/// bits at several scales, and whole roots of whole powers.
#[test]
fn exact_accuracy() {
    let powers = |bits: i32| -> Vec<((f64, f64), f64)> {
        let limit = 2f64.powi(bits);
        (2..bits)
            .flat_map(|e| {
                let top = limit.powf(1.0 / e as f64);
                let bases = (2..3002).map(f64::from).chain((0..1000).map(move |i| uniform(i, (2.0, top)).floor()));
                bases.filter(move |&b| b.powi(e) < limit).flat_map(move |b| [((b, e as f64), b.powi(e)), ((-b, e as f64), (-b).powi(e))])
            })
            .collect()
    };
    let (p64, p32) = (powers(53), powers(24));
    let p32: Vec<((f32, f32), f32)> = p32.iter().map(|&((b, e), y)| ((b as f32, e as f32), y as f32)).collect();
    let cubes = |c: u64, scales: &[i32]| -> Vec<(f64, f64)> {
        (1..=c).step_by(13).flat_map(|c| scales.iter().map(move |&k| c as f64 * 2f64.powi(k))).flat_map(|r| [(r * r * r, r), (-r * r * r, -r)]).collect()
    };
    let c64 = cubes(131072, &[-300, -3, 0, 60, 300]);
    let c32: Vec<(f32, f32)> = cubes(256, &[-30, 0, 30]).iter().map(|&(x, r)| (x as f32, r as f32)).collect();
    let roots: Vec<((f64, f64), f64)> = (4..=40)
        .flat_map(|n| (2..).map(f64::from).take_while(move |b| b.powi(n) < 2f64.powi(53)).map(move |b| ((n as f64, b.powi(n)), b)))
        .collect();
    for (wide, narrow, how) in variants!(pow, 2) {
        exactly(&format!("f64 pow of exact powers, {how}"), &p64, wide);
        exactly(&format!("f32 pow of exact powers, {how}"), &p32, narrow);
    }
    for (wide, narrow, how) in variants!(cbrt) {
        exactly(&format!("f64 cbrt of exact cubes, {how}"), &c64, wide);
        exactly(&format!("f32 cbrt of exact cubes, {how}"), &c32, narrow);
    }
    for (wide, _, how) in variants!(root, 2) { exactly(&format!("f64 root of exact powers, {how}"), &roots, wide); }
}

/// The checks of a special function, against its rows in `testdata/mpmath.txt` called `name`: at f64, at f32 where the input is a
/// float at both widths, and at every f16 input against rustymath's f64 result rounded. It also checks the cases in `VECTORS`.
fn special(name: &str, variants: Variants<fn(f64) -> f64, fn(f32) -> f32>, f16: fn(f16) -> f16, wide64: fn(f64) -> f64) {
    let cases: Vec<(f64, f64)> = vectors(MPMATH, name);
    let narrow_cases = at32(&cases);
    for (wide, narrow, how) in variants {
        check_cases(&format!("f64 {name}, {how}"), &cases, wide);
        check_cases(&format!("f32 {name}, {how}"), &narrow_cases, narrow);
        for (source, text) in VECTORS {
            check_cases(&format!("{source} {name}, {how}"), &vectors::<f64, f64>(text, name), wide);
            check_cases(&format!("{source} {name}f, {how}"), &vectors::<f32, f32>(text, &format!("{name}f")), narrow);
        }
    }
    check(&format!("f16 {name}"), worst(1 << 16, |i| f16::from_bits(i as u16), f16, |x| f16::from_f64(wide64(x.into()))));
}

#[test]
fn erf_accuracy() { special("erf", variants!(erf), crate::erf, crate::erf::<f64>); }

#[test]
fn erfc_accuracy() { special("erfc", variants!(erfc), crate::erfc, crate::erfc::<f64>); }

#[test]
fn erfinv_accuracy() { special("erfinv", variants!(erfinv), crate::erfinv, crate::erfinv::<f64>); }

#[test]
fn erfcinv_accuracy() { special("erfcinv", variants!(erfcinv), crate::erfcinv, crate::erfcinv::<f64>); }

/// The worst error of `f` at `cases` of inputs and reference results, in units of `eps` times the larger of |reference| and `floor`,
/// with the input that gives it. Infinities and NaN must match exactly.
fn scaled<A: Copy + Debug>(cases: impl IntoIterator<Item = (A, f64)>, f: impl Fn(A) -> f64, eps: f64, floor: f64) -> (u64, A) {
    let error = |got: f64, want: f64| {
        if !want.is_finite() || !got.is_finite() { if got == want || (got.is_nan() && want.is_nan()) { 0 } else { u64::MAX } }
        else { ((got - want).abs() / (eps * want.abs().max(floor))).ceil() as u64 }
    };
    cases.into_iter().map(|(x, want)| (error(f(x), want), x)).max_by_key(|&(e, _)| e).unwrap()
}

#[test]
/// `gamma`'s checks, with each error in units of epsilon times the result, which allows a little more than `BOUND` ulps for a result
/// just below a power of 2. The f16 check rounds rustymath's f64 result.
fn gamma_accuracy() {
    let cases: Vec<(f64, f64)> = vectors(MPMATH, "tgamma");
    for (wide, narrow, how) in variants!(gamma) {
        check(&format!("f64 tgamma, {how}"), scaled(cases.iter().copied(), wide, f64::EPSILON, 1e-300));
        check(&format!("f32 tgamma, {how}"), scaled(at32(&cases).into_iter().map(|(x, y)| (x, y.into())), |x| narrow(x).into(), f32::EPSILON.into(), f32::MIN_POSITIVE.into()));
    }
    check("f16 tgamma", worst(1 << 16, |i| f16::from_bits(i as u16), crate::gamma, |x| f16::from_f64(crate::gamma::<f64>(x.into()))));
}

/// `lgamma`'s checks, with each error in units of epsilon times the larger of the result and 2, because its relative error near its
/// zeros at 1 and 2 has no bound.
#[test]
fn lgamma_accuracy() {
    let cases: Vec<(f64, f64)> = vectors(MPMATH, "lgamma");
    for (wide, narrow, how) in variants!(lgamma) {
        check(&format!("f64 lgamma, {how}"), scaled(cases.iter().copied(), wide, f64::EPSILON, 2.0));
        check(&format!("f32 lgamma, {how}"), scaled(at32(&cases).into_iter().map(|(x, y)| (x, y.into())), |x| narrow(x).into(), f32::EPSILON.into(), 2.0));
    }
}

/// The error that the incomplete functions allow at f64, in units of f64's epsilon times the reference: their factors x^a e^-x and
/// x^a (1-x)^b come from `lgamma`, whose absolute error grows with a and b.
const ITERATIVE: u64 = 1 << 10;

/// Asserts that the worst f64 error `(e, x)` of the incomplete function called `name` is within `ITERATIVE`, and prints it.
fn iterative<A: Debug>(name: &str, worst: (u64, A)) { within(&format!("f64 {name}"), ITERATIVE, worst) }

/// The checks of an incomplete function against its rows in `testdata/mpmath.txt` called `name`, whose inputs are floats at both
/// widths: at f64 within `ITERATIVE`, and at f32 within `BOUND`, each in units of its width's epsilon times the reference.
fn incomplete<A: Input>(name: &str, wide: impl Fn(A) -> f64, narrow: impl Fn(A) -> f64) -> Vec<(A, f64)> {
    let cases: Vec<(A, f64)> = vectors(MPMATH, name);
    iterative(name, scaled(cases.iter().copied(), wide, f64::EPSILON, 1e-280));
    check(&format!("f32 {name}"), scaled(cases.iter().map(|&(x, y)| (x, y as f32 as f64)), narrow, f32::EPSILON.into(), 1e-30));
    cases
}

/// Whether `p` is a probability that an inverse can recover its argument from: far enough from 0 and 1 that rounding `p` moves the
/// argument by little.
fn central(p: f64) -> bool { (0.001..0.999).contains(&p) }

#[test]
fn incomplete_gamma_accuracy() {
    let cases = incomplete("gammainc", |(a, x)| crate::gammainc(a, x), |(a, x): (f64, f64)| crate::gammainc(a as f32, x as f32).into());
    incomplete("gammaincc", |(a, x)| crate::gammaincc(a, x), |(a, x): (f64, f64)| crate::gammaincc(a as f32, x as f32).into());
    // The inverse recovers x from a and P(a, x). At f32 it's checked against its own f64 result at the same inputs.
    let inverse: Vec<_> = cases.iter().filter(|&&(_, p)| central(p)).map(|&((a, x), p)| ((a, p), x)).collect();
    iterative("gammaincinv", scaled(inverse.iter().copied(), |(a, p)| crate::gammaincinv(a, p), f64::EPSILON, 1e-280));
    let narrow = inverse.iter().map(|&((a, p), _)| { let p = p as f32 as f64; ((a, p), crate::gammaincinv(a, p) as f32 as f64) });
    check("f32 gammaincinv", scaled(narrow, |(a, p)| crate::gammaincinv(a as f32, p as f32).into(), f32::EPSILON.into(), 1e-30));
}

#[test]
fn incomplete_beta_accuracy() {
    let cases = incomplete("betainc", |(a, b, x)| crate::betainc(a, b, x), |(a, b, x): (f64, f64, f64)| crate::betainc(a as f32, b as f32, x as f32).into());
    // The inverse recovers x from a, b and I(a, b, x). At f32 it's checked against its own f64 result at the same inputs.
    let inverse: Vec<_> = cases.iter().filter(|&&(_, p)| central(p)).map(|&((a, b, x), p)| ((a, b, p), x)).collect();
    iterative("betaincinv", scaled(inverse.iter().copied(), |(a, b, p)| crate::betaincinv(a, b, p), f64::EPSILON, 1e-280));
    let narrow = inverse.iter().map(|&((a, b, p), _)| { let p = p as f32 as f64; ((a, b, p), crate::betaincinv(a, b, p) as f32 as f64) });
    check("f32 betaincinv", scaled(narrow, |(a, b, p)| crate::betaincinv(a as f32, b as f32, p as f32).into(), f32::EPSILON.into(), 1e-30));
}
