//! The accuracy tests. Each real function, with `mla` fused and unfused, must be within `BOUND` of the right result at every f16 input, at samples of f32 and f64 inputs, and at published test cases. Each check prints its worst error.
//!
//! The elementary functions are checked against the platform's libm at f64, through Rust's standard library. Rounded to f32 or f16, its result is correctly rounded at those widths, apart from rare double rounding. At f64 the libm has errors of its own, which `LIBM` allows for. The hard cases in `testdata/core-math.txt`, and the special functions in `testdata/mpmath.txt`, carry correctly rounded results that `testdata/references.py` computes with mpmath.
pub mod support;
use crate::support::random;
use half::f16;
use std::fmt::Debug;

/// Every function's error bound, in ulps, at every width.
const BOUND: u64 = 4;
/// The error, in ulps, that a check against the platform's libm at f64 allows on top of `BOUND`. glibc documents errors of up to 2 ulps for some of these functions, such as `sinh` and `tanh`.
const LIBM: u64 = 2;
/// The number of f64 inputs, or of f32 inputs from a range, that each check samples.
const SAMPLES: u64 = 1 << 14;

/// A float type whose errors the tests measure.
trait Measured: Copy + Debug {
    /// A key whose order is the order of the floats, with -0 just below 0. Adjacent floats have adjacent keys.
    fn key(self) -> u64;
    fn is_nan(self) -> bool;
    fn is_zero(self) -> bool;
}

/// The input of a function under test: one float, or several. An input with a zero, infinite or NaN argument is special, and must give IEEE's result exactly, except that a zero may have either sign.
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
            fn is_zero(self) -> bool { num_traits::Zero::is_zero(&self) }
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

/// The error of `got` in ulps, against the result `want` at `x`. A special input or a NaN must give the right result exactly, with a zero of either sign, or the error is `u64::MAX`.
fn error<A: Input, T: Measured>(x: A, got: T, want: T) -> u64 {
    let exact = |same: bool| if same { 0 } else { u64::MAX };
    if want.is_nan() || got.is_nan() { exact(want.is_nan() && got.is_nan()) }
    else if x.is_special() { exact(got.key() == want.key() || got.is_zero() && want.is_zero()) }
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
fn check_cases<A: Input, T: Measured>(name: &str, cases: &[(A, T)], f: impl Fn(A) -> T) { if let Some(worst) = worst_case(cases, f) { check(name, worst); } }

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

/// A random f64 from `lo` to `hi`.
fn uniform(i: u64, (lo, hi): (f64, f64)) -> f64 { lo + (hi - lo) * (random(i) >> 11) as f64 / (1u64 << 53) as f64 }
/// A random f64 of either sign whose bits are random, so that small magnitudes are as likely as large ones. It includes the subnormals, the infinities and NaN.
fn scattered(i: u64) -> f64 { f64::from_bits(random(i)) }
/// The `i`th of 2^16 f32 inputs, whose top 16 bits are `i` and whose low bits are random. Every sign and exponent gets the same share of them.
fn spread32(i: u64) -> f32 { f32::from_bits((i << 16) as u32 | (random(i) as u32 & 0xffff)) }

/// The files of inputs with their correctly rounded results, by source. Each file's header names its source and licence.
const VECTORS: [(&str, &str); 3] = [
    ("libc-test", include_str!("../testdata/libc-test.txt")), ("Arm", include_str!("../testdata/arm.txt")),
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

/// A version of a function at f64 and at f32, named for the checks' messages.
type Variant<W, N> = (W, N, &'static str);
/// The function at f64 and f32 with `mla` fused, and the same unfused.
type Variants<W, N> = [Variant<W, N>; 2];
/// `Pow`'s and `Root`'s scalar forms, which take fast paths of their own, as a variant beside the free functions' `fused` and `unfused`.
const POW_ONE: Variant<fn((f64, f64)) -> f64, fn((f32, f32)) -> f32> =
    (|(x, y)| rustymath::Binary::one(&rustymath::Pow, x, y), |(x, y)| rustymath::Binary::one(&rustymath::Pow, x, y), "object");
const ROOT_ONE: Variant<fn((f64, f64)) -> f64, fn((f32, f32)) -> f32> =
    (|(n, y)| rustymath::Binary::one(&rustymath::Root, n, y), |(n, y)| rustymath::Binary::one(&rustymath::Root, n, y), "object");
macro_rules! variants {
    ($f:ident) => {
        [
            (rustymath::fused::$f::<f64> as fn(f64) -> f64, rustymath::fused::$f::<f32> as fn(f32) -> f32, "fused"),
            (rustymath::unfused::$f::<f64>, rustymath::unfused::$f::<f32>, "unfused"),
        ]
    };
    ($f:ident, 2) => {
        [
            (
                (|(x, y)| rustymath::fused::$f::<f64>(x, y)) as fn((f64, f64)) -> f64, (|(x, y)| rustymath::fused::$f::<f32>(x, y)) as fn((f32, f32)) -> f32,
                "fused",
            ), (|(x, y)| rustymath::unfused::$f::<f64>(x, y), |(x, y)| rustymath::unfused::$f::<f32>(x, y), "unfused"),
        ]
    };
}

/// The checks of a function of one argument, called `name` in the test data, against `libm`, the platform's version at f64: 2^16 f32 inputs spread over every exponent, `SAMPLES` f64 inputs with random bits, `SAMPLES` uniform inputs at each width from each of `ranges`, `extra` inputs and every f16 input. It also checks the hard cases in `testdata/core-math.txt` and the cases in `VECTORS`.
fn unary(name: &str, variants: Variants<fn(f64) -> f64, fn(f32) -> f32>, f16: fn(f16) -> f16, libm: fn(f64) -> f64, ranges: &[(f64, f64)], extra: &[f64]) {
    let narrow_libm = |x: f32| libm(x.into()) as f32;
    let hard: Vec<(f64, f64)> = vectors(CORE_MATH, name);
    for (wide, narrow, how) in variants {
        check(&format!("f32 {name}, {how}"), worst(1 << 16, spread32, narrow, narrow_libm));
        check_libm(&format!("f64 {name} at scattered inputs, {how}"), worst(SAMPLES, scattered, wide, libm));
        for &range in ranges {
            check_libm(&format!("f64 {name} in {range:?}, {how}"), worst(SAMPLES, |i| uniform(i, range), wide, libm));
            check(&format!("f32 {name} in {range:?}, {how}"), worst(SAMPLES, |i| uniform(i, range) as f32, narrow, narrow_libm));
        }
        if !extra.is_empty() {
            check_libm(&format!("f64 {name} at extra inputs, {how}"), worst(extra.len() as u64, |i| extra[i as usize], wide, libm));
        }
        check_cases(&format!("CORE-MATH {name}, {how}"), &hard, wide);
        for (source, text) in VECTORS {
            check_cases(&format!("{source} {name}, {how}"), &vectors::<f64, f64>(text, name), wide);
            check_cases(&format!("{source} {name}f, {how}"), &vectors::<f32, f32>(text, &format!("{name}f")), narrow);
        }
    }
    check(&format!("f16 {name}"), worst(1 << 16, |i| f16::from_bits(i as u16), f16, |x| f16::from_f64(libm(x.into()))));
}

/// The checks of a function of two arguments, called `name` in the test data, against `libm`, the platform's version at f64: 2^16 f32 pairs and `SAMPLES` f64 pairs with random bits, `SAMPLES` uniform pairs at each width from each of `ranges`, `extra` pairs and 2^16 random f16 pairs. It also checks the hard cases in `testdata/core-math.txt` and the cases in `VECTORS`.
fn binary(
    name: &str, variants: impl IntoIterator<Item = Variant<fn((f64, f64)) -> f64, fn((f32, f32)) -> f32>>, f16: fn(f16, f16) -> f16, libm: fn(f64, f64) -> f64,
    ranges: &[((f64, f64), (f64, f64))], extra: &[(f64, f64)]) {
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
        if !extra.is_empty() {
            check_libm(&format!("f64 {name} at extra pairs, {how}"), worst(extra.len() as u64, |i| extra[i as usize], wide, wide_libm));
        }
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

/// The platform libm's inverse hyperbolic functions. Rust's standard library computes these with formulas of its own, and its `acosh` loses most of its accuracy near 1.
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
fn exponentials_accuracy() {
    unary("exp", variants!(exp), rustymath::exp, f64::exp, &[(-760.0, 720.0)], &[]);
    unary("expm1", variants!(expm1), rustymath::expm1, f64::exp_m1, &[(-50.0, 720.0), (-1.0, 1.0)], &[]);
}

#[test]
fn logarithms_accuracy() {
    unary("log", variants!(ln), rustymath::ln, f64::ln, &[(0.5, 2.0)], &[]);
    unary("log1p", variants!(log1p), rustymath::log1p, f64::ln_1p, &[(-1.0, 1.0), (-1e-6, 1e-6)], &[]);
    unary("log2", variants!(log2), rustymath::log2, f64::log2, &[(0.5, 2.0)], &[]);
    let powers: Vec<f64> = (-22..=22).map(|k| format!("1e{k}").parse().unwrap()).collect();
    unary("log10", variants!(log10), rustymath::log10, f64::log10, &[(0.5, 2.0)], &powers);
}

/// `log2` gives k exactly at 2^k, and `log10` gives k at 10^k, or at the float nearest 10^k where 10^k isn't a float. Each does at every width, fused and unfused. So does `Logb` in base 2 and base 10: in its scalar form, and in its slice form with the base constant, the argument constant or neither.
#[test]
fn logarithms_exact() {
    use rustymath::{Arg, Binary, Float, Logb};
    use std::ops::RangeInclusive;
    fn exact<F: Float + Debug>(twos: RangeInclusive<i32>, tens: RangeInclusive<i32>) where Logb: Binary<F, Out = F> {
        // 2^k in two exact factors. `powi` alone gives 0 below 2^-1022, from the reciprocal of a power that overflows.
        let cases = twos.map(|k| (2.0, 2f64.powi(k / 2) * 2f64.powi(k - k / 2), k)).chain(tens.map(|k| (10.0, 10f64.powi(k), k)));
        for (b, x, k) in cases {
            let (b, x, k) = (F::narrow(b), F::narrow(x), F::narrow(k.into()));
            let alone = if b == F::narrow(2.0) { [rustymath::fused::log2(x), rustymath::unfused::log2(x)] }
            else { [rustymath::fused::log10(x), rustymath::unfused::log10(x)] };
            let mut out = Vec::new();
            assert!(Logb.many(Arg::One(b), Arg::Many(&[x]), 1, &mut out) & Logb.many(Arg::Many(&[b, F::narrow(3.0)]), Arg::One(x), 2, &mut out));
            assert!(Logb.many(Arg::Many(&[b]), Arg::Many(&[x]), 1, &mut out));
            assert_eq!([alone[0], alone[1], Logb.one(b, x), out[0], out[1], out[3]], [k; 6], "the logarithm of {x:?} to base {b:?}");
        }
    }
    exact::<f64>(-1074..=1023, -307..=22);
    exact::<f32>(-149..=127, -37..=10);
    exact::<f16>(-24..=15, -4..=4);
}

#[test]
fn trig_accuracy() {
    let extra = reduction();
    unary("sin", variants!(sin), rustymath::sin, f64::sin, TRIG, &extra);
    unary("cos", variants!(cos), rustymath::cos, f64::cos, TRIG, &extra);
    unary("tan", variants!(tan), rustymath::tan, f64::tan, TRIG, &extra);
}

#[test]
fn inverse_trig_accuracy() {
    unary("asin", variants!(asin), rustymath::asin, f64::asin, &[(-1.0, 1.0)], &[]);
    unary("acos", variants!(acos), rustymath::acos, f64::acos, &[(-1.0, 1.0)], &[]);
    unary("atan", variants!(atan), rustymath::atan, f64::atan, &[(-4.0, 4.0)], &[]);
}

#[test]
fn hyperbolic_accuracy() {
    unary("sinh", variants!(sinh), rustymath::sinh, f64::sinh, &[(-1.0, 1.0), (-711.0, 711.0)], &[]);
    unary("cosh", variants!(cosh), rustymath::cosh, f64::cosh, &[(-1.0, 1.0), (-711.0, 711.0)], &[]);
    unary("tanh", variants!(tanh), rustymath::tanh, f64::tanh, &[(-1.0, 1.0), (-25.0, 25.0)], &[]);
}

#[test]
fn inverse_hyperbolic_accuracy() {
    unary("asinh", variants!(asinh), rustymath::asinh, |x| libm::asinh(x), &[(-1.0, 1.0), (-1e20, 1e20)], &[]);
    unary("acosh", variants!(acosh), rustymath::acosh, |x| libm::acosh(x), &[(1.0, 2.0), (1.0, 1e20)], &[]);
    unary("atanh", variants!(atanh), rustymath::atanh, |x| libm::atanh(x), &[(-1.0, 1.0)], &[]);
}

#[test]
fn cbrt_accuracy() {
    let cubes: Vec<f64> = (-1000..=1000).map(|k: i32| (k as f64).powi(3)).collect();
    unary("cbrt", variants!(cbrt), rustymath::cbrt, f64::cbrt, &[(-1e300, 1e300), (-8.0, 8.0)], &cubes);
}

#[test]
fn pow_accuracy() {
    // Three kinds of points: powers that are floats, whole powers from 2 to 8 of bases that aren't whole, and powers near overflow and underflow of a base near 1. The second kind squares and multiplies. The third is where the error of ln |x| grows most.
    let wholes = [0.3, 1.7, 3.14159, 123.456, -2.71828, 1e10, 1e-10].into_iter().flat_map(|x| (2..=8).flat_map(move |k| [(x, k as f64), (x, -k as f64)]));
    let exact: Vec<(f64, f64)> = (2..=12).flat_map(|b| (-30..=60).map(move |k| (b as f64, k as f64))).chain(wholes).collect();
    let near = [((0.5, 2.0), (-1000.0, 1000.0)), ((1.2, 1.5), (1500.0, 2700.0)), ((0.0, 4.0), (-30.0, 30.0)), ((-8.0, 0.0), (-20.0, 20.0))];
    binary("pow", variants!(pow, 2).into_iter().chain([POW_ONE]), rustymath::pow, f64::powf, &near, &exact);
}

#[test]
fn polar_accuracy() {
    let ranges = [((-1e300, 1e300), (-1e300, 1e300)), ((-1e-300, 1e-300), (-1e-300, 1e-300)), ((-4.0, 4.0), (-4.0, 4.0))];
    binary("hypot", variants!(hypot, 2), rustymath::hypot, f64::hypot, &ranges, &[]);
    binary("atan2", variants!(atan2, 2), rustymath::atan2, f64::atan2, &ranges, &[]);
}

#[test]
fn root_accuracy() {
    // Degrees whose reciprocals are floats, against pow, and degree 3, against cbrt.
    for n in [2.0, -2.0, 4.0, -4.0, 8.0, 0.5, -0.5, 1.0, -1.0] {
        for (wide, narrow, how) in variants!(root, 2).into_iter().chain([ROOT_ONE]) {
            check_libm(&format!("f64 root {n} at scattered inputs, {how}"), worst(SAMPLES, |i| scattered(i).abs(), |y| wide((n, y)), |y| y.powf(1.0 / n)));
            check(&format!("f32 root {n}, {how}"), worst(1 << 16, |i| spread32(i).abs(), |y| narrow((n as f32, y)), |y| f64::from(y).powf(1.0 / n) as f32));
        }
    }
    let fifths: Vec<f64> = (-100..=100).map(|k: i32| (k as f64).powi(5)).collect();
    for (wide, _, how) in variants!(root, 2).into_iter().chain([ROOT_ONE]) {
        check_libm(&format!("f64 root 3, {how}"), worst(SAMPLES, scattered, |y| wide((3.0, y)), f64::cbrt));
        check(
            &format!("f64 root 5 of fifth powers, {how}"),
            worst(fifths.len() as u64, |i| fifths[i as usize], |y| wide((5.0, y)), |y| y.signum() * y.abs().powf(0.2).round()));
    }
}

/// Powers, cube roots and roots whose exact result is a float give it exactly, as BPL's `7*5` must. The cases are whole powers below 2^53, or 2^24 at f32, of the first 3,000 bases and 1,000 random ones for each exponent, cubes of floats with up to 17 significant bits at several scales, and whole roots of whole powers.
#[test]
fn exact_accuracy() {
    let powers = |bits: i32| -> Vec<((f64, f64), f64)> {
        let limit = 2f64.powi(bits);
        (2..bits).flat_map(|e| {
            let top = limit.powf(1.0 / e as f64);
            let bases = (2..3002).map(f64::from).chain((0..1000).map(move |i| uniform(i, (2.0, top)).floor()));
            bases.filter(move |&b| b.powi(e) < limit).flat_map(move |b| [((b, e as f64), b.powi(e)), ((-b, e as f64), (-b).powi(e))])
        }).collect()
    };
    let (p64, p32) = (powers(53), powers(24));
    let p32: Vec<((f32, f32), f32)> = p32.iter().map(|&((b, e), y)| ((b as f32, e as f32), y as f32)).collect();
    let cubes = |c: u64, scales: &[i32]| -> Vec<(f64, f64)> {
        (1..=c).step_by(13).flat_map(|c| scales.iter().map(move |&k| c as f64 * 2f64.powi(k))).flat_map(|r| [(r * r * r, r), (-r * r * r, -r)]).collect()
    };
    let c64 = cubes(131072, &[-300, -3, 0, 60, 300]);
    let c32: Vec<(f32, f32)> = cubes(256, &[-30, 0, 30]).iter().map(|&(x, r)| (x as f32, r as f32)).collect();
    let roots: Vec<((f64, f64), f64)> =
        (4..=40).flat_map(|n| (2..).map(f64::from).take_while(move |b| b.powi(n) < 2f64.powi(53)).map(move |b| ((n as f64, b.powi(n)), b))).collect();
    for (wide, narrow, how) in variants!(pow, 2).into_iter().chain([POW_ONE]) {
        exactly(&format!("f64 pow of exact powers, {how}"), &p64, wide);
        exactly(&format!("f32 pow of exact powers, {how}"), &p32, narrow);
    }
    for (wide, narrow, how) in variants!(cbrt) {
        exactly(&format!("f64 cbrt of exact cubes, {how}"), &c64, wide);
        exactly(&format!("f32 cbrt of exact cubes, {how}"), &c32, narrow);
    }
    for (wide, _, how) in variants!(root, 2).into_iter().chain([ROOT_ONE]) { exactly(&format!("f64 root of exact powers, {how}"), &roots, wide); }
}

/// The checks of a special function, against its rows in `testdata/mpmath.txt` called `name`: at f64, at f32 where the input is a float at both widths, and at every f16 input against rustymath's f64 result rounded. It also checks the cases in `VECTORS`.
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
fn error_functions_accuracy() {
    special("erf", variants!(erf), rustymath::erf, rustymath::erf::<f64>);
    special("erfc", variants!(erfc), rustymath::erfc, rustymath::erfc::<f64>);
    special("erfinv", variants!(erfinv), rustymath::erfinv, rustymath::erfinv::<f64>);
    special("erfcinv", variants!(erfcinv), rustymath::erfcinv, rustymath::erfcinv::<f64>);
}

/// `Erfc`'s slice form gives `erfc`'s bits at both widths. Its blocks of 256 items lie wholly below 0.84375 (with NaN), wholly from it up (with -∞ and magnitudes past 28), on both sides, and in a short block of zeros and ∞.
#[test]
fn erfc_blocks() {
    use rustymath::{Arg, Erfc, Unary};
    let mut xs = vec![f64::NAN];
    xs.extend((1..256).map(|i| -0.84 + 1.68 * f64::from(i) / 256.0));
    xs.push(f64::NEG_INFINITY);
    xs.extend((1..256).map(|i| (0.85 + f64::from(i) / 8.0) * if i % 2 == 0 { 1.0 } else { -1.0 }));
    xs.extend((0..256).map(|i| if i % 2 == 0 { f64::from(i) / 1024.0 } else { -0.9 - f64::from(i) / 16.0 }));
    xs.extend([0.0, -0.0, f64::INFINITY]);
    let xs32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
    let (mut out, mut out32) = (Vec::new(), Vec::new());
    assert!(Erfc.many(Arg::Many(&xs), xs.len(), &mut out) & Erfc.many(Arg::Many(&xs32), xs32.len(), &mut out32));
    let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
    assert!(out.iter().zip(&xs).all(|(&r, &x)| same(r, rustymath::erfc(x))), "{out:?}");
    assert!(out32.iter().zip(&xs32).all(|(&r, &x)| same(r.into(), rustymath::erfc(x).into())), "{out32:?}");
}

#[test]
fn lambertw_accuracy() { special("lambertw", variants!(lambertw), rustymath::lambertw, rustymath::lambertw::<f64>); }

#[test]
fn circle_roots_accuracy() {
    special("sqrt1mx2", variants!(sqrt1mx2), rustymath::sqrt1mx2, rustymath::sqrt1mx2::<f64>);
    special("sqrt1px2", variants!(sqrt1px2), rustymath::sqrt1px2, rustymath::sqrt1px2::<f64>);
    special("sqrtx2m1", variants!(sqrtx2m1), rustymath::sqrtx2m1, rustymath::sqrtx2m1::<f64>);
}

/// The worst error of `f` at `cases` of inputs and reference results, in units of `eps` times the larger of |reference| and `floor`, with the input that gives it. Infinities and NaN must match exactly.
fn scaled<A: Copy + Debug>(cases: impl IntoIterator<Item = (A, f64)>, f: impl Fn(A) -> f64, eps: f64, floor: f64) -> (u64, A) {
    let error = |got: f64, want: f64| {
        if !want.is_finite() || !got.is_finite() { if got == want || (got.is_nan() && want.is_nan()) { 0 }
        else { u64::MAX } }
        else { ((got - want).abs() / (eps * want.abs().max(floor))).ceil() as u64 }
    };
    cases.into_iter().map(|(x, want)| (error(f(x), want), x)).max_by_key(|&(e, _)| e).unwrap()
}

#[test]
/// `gamma`'s checks, with each error in units of epsilon times the result, which allows a little more than `BOUND` ulps for a result just below a power of 2. The f16 check rounds rustymath's f64 result.
fn gamma_accuracy() {
    let cases: Vec<(f64, f64)> = vectors(MPMATH, "tgamma");
    for (wide, narrow, how) in variants!(gamma) {
        check(&format!("f64 tgamma, {how}"), scaled(cases.iter().copied(), wide, f64::EPSILON, 1e-300));
        check(
            &format!("f32 tgamma, {how}"),
            scaled(at32(&cases).into_iter().map(|(x, y)| (x, y.into())), |x| narrow(x).into(), f32::EPSILON.into(), f32::MIN_POSITIVE.into()));
    }
    check("f16 tgamma", worst(1 << 16, |i| f16::from_bits(i as u16), rustymath::gamma, |x| f16::from_f64(rustymath::gamma::<f64>(x.into()))));
}

/// `lgamma`'s checks, with each error in units of epsilon times the larger of the result and 2, because its relative error near its zeros at 1 and 2 has no bound.
#[test]
fn lgamma_accuracy() {
    let cases: Vec<(f64, f64)> = vectors(MPMATH, "lgamma");
    for (wide, narrow, how) in variants!(lgamma) {
        check(&format!("f64 lgamma, {how}"), scaled(cases.iter().copied(), wide, f64::EPSILON, 2.0));
        check(&format!("f32 lgamma, {how}"), scaled(at32(&cases).into_iter().map(|(x, y)| (x, y.into())), |x| narrow(x).into(), f32::EPSILON.into(), 2.0));
    }
}

/// `digamma`'s checks, with each error in units of epsilon times the larger of the result and 1, because its relative error near its zeros has no bound. Below -20, where `digamma` reflects, ψ(1 - x) and π cot πx cancel near each zero, and an f64 error counts in units of epsilon times ln(1 - x), their size there. The f16 check rounds rustymath's f64 result.
#[test]
fn digamma_accuracy() {
    let cases: Vec<(f64, f64)> = vectors(MPMATH, "digamma");
    let (lifted, reflected): (Vec<_>, Vec<_>) = cases.iter().copied().partition(|&(x, _)| !(x < -20.0));
    for (wide, narrow, how) in variants!(digamma) {
        check(&format!("f64 digamma, {how}"), scaled(lifted.iter().copied(), wide, f64::EPSILON, 1.0));
        check(&format!("f64 reflected digamma, {how}"), reflected.iter().map(|&c| scaled([c], wide, f64::EPSILON, (1.0 - c.0).ln())).max_by_key(|w| w.0).unwrap());
        check(&format!("f32 digamma, {how}"), scaled(at32(&cases).into_iter().map(|(x, y)| (x, y.into())), |x| narrow(x).into(), f32::EPSILON.into(), 1.0));
    }
    check("f16 digamma", worst(1 << 16, |i| f16::from_bits(i as u16), rustymath::digamma, |x| f16::from_f64(rustymath::digamma::<f64>(x.into()))));
    // Near zero, ψ(x) rounds to -1/x, including where that reciprocal overflows.
    assert_eq!([rustymath::digamma(1e-309f64), rustymath::digamma(-1e-309)], [f64::NEG_INFINITY, f64::INFINITY]);
}

/// The error that the incomplete functions allow at f64, in units of f64's epsilon times the reference: their factors x^a e^-x and x^a (1-x)^b come from `lgamma`, whose absolute error grows with a and b.
const ITERATIVE: u64 = 1 << 10;

/// Asserts that the worst f64 error `(e, x)` of the incomplete function called `name` is within `ITERATIVE`, and prints it.
fn iterative<A: Debug>(name: &str, worst: (u64, A)) { within(&format!("f64 {name}"), ITERATIVE, worst) }

/// The checks of an incomplete function against its rows in `testdata/mpmath.txt` called `name`, whose inputs are floats at both widths: at f64 within `ITERATIVE`, and at f32 within `BOUND`, each in units of its width's epsilon times the reference.
fn incomplete<A: Input>(name: &str, wide: impl Fn(A) -> f64, narrow: impl Fn(A) -> f64) -> Vec<(A, f64)> {
    let cases: Vec<(A, f64)> = vectors(MPMATH, name);
    iterative(name, scaled(cases.iter().copied(), wide, f64::EPSILON, 1e-280));
    check(&format!("f32 {name}"), scaled(cases.iter().map(|&(x, y)| (x, y as f32 as f64)), narrow, f32::EPSILON.into(), 1e-30));
    cases
}

/// Whether `p` is a probability that an inverse can recover its argument from: far enough from 0 and 1 that rounding `p` moves the argument by little.
fn central(p: f64) -> bool { (0.001..0.999).contains(&p) }

#[test]
fn incomplete_gamma_accuracy() {
    let cases = incomplete("gammainc", |(a, x)| rustymath::gammainc(a, x), |(a, x): (f64, f64)| rustymath::gammainc(a as f32, x as f32).into());
    incomplete("gammaincc", |(a, x)| rustymath::gammaincc(a, x), |(a, x): (f64, f64)| rustymath::gammaincc(a as f32, x as f32).into());
    // The inverse recovers x from a and P(a, x). At f32 it's checked against its own f64 result at the same inputs.
    let inverse: Vec<_> = cases.iter().filter(|&&(_, p)| central(p)).map(|&((a, x), p)| ((a, p), x)).collect();
    iterative("gammaincinv", scaled(inverse.iter().copied(), |(a, p)| rustymath::gammaincinv(a, p), f64::EPSILON, 1e-280));
    let narrow = inverse.iter().map(|&((a, p), _)| {
        let p = p as f32 as f64;
        ((a, p), rustymath::gammaincinv(a, p) as f32 as f64)
    });
    check("f32 gammaincinv", scaled(narrow, |(a, p)| rustymath::gammaincinv(a as f32, p as f32).into(), f32::EPSILON.into(), 1e-30));
}

#[test]
fn incomplete_beta_accuracy() {
    let cases =
        incomplete("betainc", |(a, b, x)| rustymath::betainc(a, b, x), |(a, b, x): (f64, f64, f64)| rustymath::betainc(a as f32, b as f32, x as f32).into());
    // The inverse recovers x from a, b and I(a, b, x). At f32 it's checked against its own f64 result at the same inputs.
    let inverse: Vec<_> = cases.iter().filter(|&&(_, p)| central(p)).map(|&((a, b, x), p)| ((a, b, p), x)).collect();
    iterative("betaincinv", scaled(inverse.iter().copied(), |(a, b, p)| rustymath::betaincinv(a, b, p), f64::EPSILON, 1e-280));
    let narrow = inverse.iter().map(|&((a, b, p), _)| {
        let p = p as f32 as f64;
        ((a, b, p), rustymath::betaincinv(a, b, p) as f32 as f64)
    });
    check("f32 betaincinv", scaled(narrow, |(a, b, p)| rustymath::betaincinv(a as f32, b as f32, p as f32).into(), f32::EPSILON.into(), 1e-30));
}

mod complex {
    use num_complex::Complex;
    use rustymath::{
        Acos, Acosh, Arg, Asin, Asinh, Atan, Atanh, Binary, Binom, Cos, Cosh, Exp, Gamma, LambertW, Lgamma, Ln, Native, Pow, Sign, Sin, Sinh, Sqrt, Sqrt1mx2,
        Sqrt1px2, Sqrtx2m1, Tan, Tanh, Unary, divide,
    };
    use std::f64::consts::{FRAC_PI_2, PI};

    fn c<R>(re: R, im: R) -> Complex<R> { Complex::new(re, im) }

    /// The bound on each function's error, in ulps of the result's magnitude.
    const BOUND: f64 = 4.0;

    // Concrete widths keep the object bounds out of the test helpers.
    macro_rules! at_width {
        ($name:ident, $r:ty) => {
            mod $name {
                use super::*;
                type R = $r;

                pub(super) fn apply(name: &str, z: &[Complex<R>], sliced: bool) -> Option<Complex<R>> {
                    fn unary<O: Unary<Complex<R>, Out = Complex<R>>>(object: O, z: &[Complex<R>], sliced: bool) -> Complex<R> {
                        if !sliced {
                            return object.one(z[0]);
                        }
                        let mut out = Vec::new();
                        object.many(Arg::Many(z), z.len(), &mut out);
                        out[0]
                    }
                    fn binary<O: Binary<Complex<R>, Out = Complex<R>>>(object: O, z: &[Complex<R>], sliced: bool) -> Complex<R> {
                        if !sliced {
                            return object.one(z[0], z[1]);
                        }
                        let mut out = Vec::new();
                        object.many(Arg::Many(&z[..1]), Arg::Many(&z[1..2]), 1, &mut out);
                        out[0]
                    }
                    Some(match name {
                        "exp" => unary(Exp, z, sliced),
                        "log" => unary(Ln, z, sliced),
                        "sqrt" => unary(Sqrt, z, sliced),
                        "sin" => unary(Sin, z, sliced),
                        "cos" => unary(Cos, z, sliced),
                        "tan" => unary(Tan, z, sliced),
                        "sinh" => unary(Sinh, z, sliced),
                        "cosh" => unary(Cosh, z, sliced),
                        "tanh" => unary(Tanh, z, sliced),
                        "asin" => unary(Asin, z, sliced),
                        "acos" => unary(Acos, z, sliced),
                        "atan" => unary(Atan, z, sliced),
                        "asinh" => unary(Asinh, z, sliced),
                        "acosh" => unary(Acosh, z, sliced),
                        "atanh" => unary(Atanh, z, sliced),
                        "lgamma" => unary(Lgamma, z, sliced),
                        "gamma" => unary(Gamma, z, sliced),
                        "pow" => binary(Pow, z, sliced),
                        "binom" => binary(Binom, z, sliced),
                        "lambertw" => unary(LambertW, z, sliced),
                        "sqrt1mx2" => unary(Sqrt1mx2, z, sliced),
                        "sqrtx2m1" => unary(Sqrtx2m1, z, sliced),
                        "sqrt1px2" => unary(Sqrt1px2, z, sliced),
                        _ => return None,
                    })
                }
            }
        };
    }
    at_width!(single, f32);
    at_width!(double, f64);

    /// Each line of `complex.txt` as its function's name, its arguments and the correctly rounded result.
    fn cases() -> Vec<(String, Vec<Complex<f64>>, Complex<f64>)> {
        include_str!("../testdata/complex.txt").lines().filter(|l| !l.starts_with('#')).map(|l| {
            let mut fields = l.split_whitespace();
            let name = fields.next().unwrap().to_string();
            let v: Vec<f64> = fields.map(|f| f.parse().unwrap()).collect();
            let z = v.chunks(2).map(|p| c(p[0], p[1])).collect::<Vec<_>>();
            let (args, want) = z.split_at(z.len() - 1);
            (name, args.to_vec(), want[0])
        }).collect()
    }

    /// The magnitude that a function's error counts in ulps of, as the module doc gives it.
    fn scale(name: &str, args: &[Complex<f64>], want: Complex<f64>) -> f64 {
        let size = |z: Complex<f64>| z.re.hypot(z.im);
        let lg = |z: Complex<f64>| size(Lgamma.one(z));
        let exponentiated = |exponent: f64| size(want) * (exponent.max(2.0) + 1.0);
        let one = c(1.0, 0.0);
        match name {
            "lgamma" => size(want).max(2.0), "gamma" => exponentiated(lg(args[0])),
            "binom" => exponentiated(lg(args[0] + one) + lg(args[1] + one) + lg(args[0] - args[1] + one)),
            "pow" => exponentiated(size(args[1] * Ln.one(args[0]))), _ => size(want)
        }
    }

    /// The worst error of each function that `f` computes, in ulps of its scale, over the cases whose arguments and result are finite and normal at `R`.
    fn worst<R: Native>(eps: f64, narrow: impl Fn(f64) -> R, f: impl Fn(&str, &[Complex<R>]) -> Option<Complex<R>>) -> Vec<(String, f64, Vec<Complex<f64>>)> {
        let mut out: Vec<(String, f64, Vec<Complex<f64>>)> = Vec::new();
        let fits = |v: f64| v.is_finite() && narrow(v).wide() == v;
        let normal = |v: f64| v == 0.0 || (v.abs() >= R::min_positive_value().wide() && v.abs() <= R::max_value().wide());
        for (name, args, want) in cases() {
            if !args.iter().all(|z| fits(z.re) && fits(z.im)) || !(normal(want.re) && normal(want.im)) || !(want.re.is_finite() && want.im.is_finite()) {
                continue;
            }
            let Some(got) = f(&name, &args.iter().map(|z| c(narrow(z.re), narrow(z.im))).collect::<Vec<_>>()) else { continue };
            let d = (got.re.wide() - want.re).hypot(got.im.wide() - want.im);
            let s = scale(&name, &args, want);
            let err = if s == 0.0 { if d == 0.0 { 0.0 } else { f64::INFINITY } } else { d / (s * eps) };
            let err = if err.is_nan() { f64::INFINITY } else { err };
            match out.iter_mut().find(|(n, _, _)| *n == name) {
                Some(row) => { if err > row.1 { *row = (name, err, args) } } None => out.push((name, err, args))
            }
        }
        out
    }

    #[test]
    fn accuracy() {
        let rows = [
            ("f64 scalar", worst::<f64>(f64::EPSILON, |v| v, |name, z| double::apply(name, z, false))),
            ("f64 sliced", worst::<f64>(f64::EPSILON, |v| v, |name, z| double::apply(name, z, true))),
            ("f32 scalar", worst::<f32>(f32::EPSILON.into(), |v| v as f32, |name, z| single::apply(name, z, false))),
            ("f32 sliced", worst::<f32>(f32::EPSILON.into(), |v| v as f32, |name, z| single::apply(name, z, true))),
        ];
        let over: Vec<_> = rows.iter().flat_map(|(how, r)| r.iter().filter(|(_, e, _)| *e > BOUND).map(move |(n, e, a)| (*how, n, *e, a))).collect();
        if std::env::var("RUSTYMATH_COMPLEX").is_ok() { for (how, r) in &rows { for (n, e, a) in r { println!("{how} {n} {e:.2} {a:?}"); } } }
        assert!(over.is_empty(), "{over:#?}");
    }

    /// C99's values on the branch cuts, where the sign of a zero part chooses the side, and exact real or imaginary results on the axes. The composite square roots take the side that a zero's sign points to too. Slice forms run on several copies, so they reach their vector loops.
    #[test]
    fn cuts_and_axes() {
        let k = |re: f64, im: f64| c(re, im);
        let (a, b, s, h) = (1.3169578969248166, 0.5493061443340549, 3f64.sqrt(), 0.75f64.sqrt());
        let cases = [
            ("sqrt", k(-4.0, 0.0), k(0.0, 2.0)), ("sqrt", k(-4.0, -0.0), k(0.0, -2.0)), ("log", k(-1.0, 0.0), k(0.0, PI)), ("log", k(-1.0, -0.0), k(0.0, -PI)),
            ("asin", k(2.0, 0.0), k(FRAC_PI_2, a)), ("asin", k(2.0, -0.0), k(FRAC_PI_2, -a)), ("acos", k(2.0, 0.0), k(0.0, -a)),
            ("acos", k(2.0, -0.0), k(0.0, a)), ("acosh", k(-2.0, 0.0), k(a, PI)), ("acosh", k(-2.0, -0.0), k(a, -PI)), ("atanh", k(2.0, 0.0), k(b, FRAC_PI_2)),
            ("atanh", k(-2.0, 0.0), k(-b, FRAC_PI_2)), ("asinh", k(0.0, 2.0), k(a, FRAC_PI_2)), ("asinh", k(-0.0, 2.0), k(-a, FRAC_PI_2)),
            ("atan", k(0.0, 2.0), k(FRAC_PI_2, b)), ("atan", k(-0.0, 2.0), k(-FRAC_PI_2, b)), ("sqrt1mx2", k(2.0, 0.0), k(0.0, -s)),
            ("sqrt1mx2", k(-2.0, 0.0), k(0.0, s)), ("sqrt1mx2", k(2.0, -0.0), k(0.0, s)), ("sqrtx2m1", k(0.5, 0.0), k(0.0, h)),
            ("sqrtx2m1", k(-0.5, 0.0), k(0.0, h)), ("sqrtx2m1", k(0.5, -0.0), k(0.0, -h)), ("sqrt1px2", k(0.0, 2.0), k(0.0, s)),
            ("sqrt1px2", k(-0.0, 2.0), k(0.0, -s)),
        ];
        let close = |g: f64, w: f64, e: f64| (g - w).abs() <= 4.0 * e * w.abs().max(1.0) && g.is_sign_negative() == w.is_sign_negative();
        for (name, z, want) in cases {
            for sliced in [false, true] {
                let got = double::apply(name, &[z; 17], sliced).unwrap();
                assert!(close(got.re, want.re, f64::EPSILON) && close(got.im, want.im, f64::EPSILON), "{name}({z}) gave {got}, not {want}");
                let got = single::apply(name, &[c(z.re as f32, z.im as f32); 17], sliced).unwrap();
                let e = f32::EPSILON.into();
                assert!(close(got.re.into(), want.re, e) && close(got.im.into(), want.im, e), "f32 {name}({z}) gave {got}, not {want}");
            }
        }
        for x in [0.5, 3.0, -2.0, 700.0, 1e-300] {
            for name in ["exp", "sin", "cos", "sinh", "cosh", "tanh", "tan"] {
                assert_eq!(double::apply(name, &[k(x, 0.0)], false).unwrap().im, 0.0, "{name}({x})");
            }
            if x > 0.0 {
                for name in ["log", "sqrt"] { assert_eq!(double::apply(name, &[k(x, 0.0)], false).unwrap().im, 0.0, "{name}({x})"); }
            }
            for name in ["sin", "sinh", "tan", "tanh"] { assert_eq!(double::apply(name, &[k(0.0, x)], false).unwrap().re, 0.0, "{name}({x}i)"); }
        }
        assert_eq!(Pow.one(k(1.0, 1.0), k(0.0, 0.0)), k(1.0, 0.0));
        assert_eq!(Pow.one(k(-8.0, 0.0), k(3.0, 0.0)), k(-512.0, 0.0));
        assert_eq!(Pow.one(k(0.0, 0.0), k(2.0, 5.0)), k(0.0, 0.0));
        assert_eq!(Sign.one(k(f64::INFINITY, 3.0)), k(1.0, 0.0));
        assert_eq!(Sign.one(k(-0.0, 0.0)), k(-0.0, 0.0));
        assert!(Lgamma.one(k(-3.0, 0.0)).re.is_nan());
        assert_eq!(Binom.one(k(2.0, 0.0), k(3.0, 0.0)), k(0.0, 0.0));
    }

    /// A real part of z too small to change 1 ± z still changes the smaller part of each composite square root, by first order in that real part. Complex-step differentiation reads derivatives from such a part. At magnitudes near 64-bit floats' limits, a tiny part of z still gives the result's tiny part, without underflow.
    #[test]
    fn tiny_parts() {
        let (e, q) = (1e-30, 1.16f64.sqrt());
        let cases = [
            ("sqrt1mx2", c(e, 0.4), c(q, -e * 0.4 / q), true), ("sqrt1px2", c(0.4, e), c(q, e * 0.4 / q), true), ("sqrtx2m1", c(e, 0.4), c(e * 0.4 / q, q), true),
            ("sqrtx2m1", c(1e200, 1e-200), c(1e200, 1e-200), false), ("sqrt1mx2", c(1e200, 1e-200), c(1e-200, -1e200), false),
            ("sqrtx2m1", c(1e300, 1e-300), c(1e300, 1e-300), false),
        ];
        let close = |g: f64, w: f64, eps: f64| (g - w).abs() <= 4.0 * eps * w.abs();
        for (name, z, want, narrow) in cases {
            for sliced in [false, true] {
                let got = double::apply(name, &[z; 17], sliced).unwrap();
                assert!(close(got.re, want.re, f64::EPSILON) && close(got.im, want.im, f64::EPSILON), "{name}({z}) gave {got}, not {want}");
                if !narrow { continue; }
                let got = single::apply(name, &[c(z.re as f32, z.im as f32); 17], sliced).unwrap();
                let eps = f32::EPSILON.into();
                assert!(close(got.re.into(), want.re, eps) && close(got.im.into(), want.im, eps), "f32 {name}({z}) gave {got}, not {want}");
            }
        }
    }

    /// Division divides two reals as reals, so a real zero divisor gives ∞ with an imaginary part of 0. Any other zero divisor gives NaN.
    #[test]
    fn division() {
        let k = |re: f64, im: f64| c(re, im);
        let q = divide(k(3.0, 4.0), k(1.0, 2.0));
        assert!((q - k(2.2, -0.4)).norm() <= 4.0 * f64::EPSILON * 2.3, "{q}");
        assert_eq!(divide(k(5.0, 0.0), k(2.0, 0.0)), k(2.5, 0.0));
        assert_eq!(divide(k(1.0, 0.0), k(0.0, 0.0)), k(f64::INFINITY, 0.0));
        let z = divide(k(1.0, -2.0), k(0.0, 0.0));
        assert!(z.re.is_nan() && z.im.is_nan(), "{z}");
    }

    /// e^(iy) at the float nearest kπ/2 is exactly i^k, through the scalar and slice forms of `Exp` and of the `Pow`s that compute through it. A part the quarter turn makes 0 stays 0 where e^x overflows, and NaN in x still gives NaN.
    #[test]
    fn quarter_turns() {
        fn check<R: Native + std::fmt::Debug>(big: f64) where Exp: Unary<Complex<R>, Out = Complex<R>>, Pow: Binary<Complex<R>, Out = Complex<R>> {
            let q = R::narrow(FRAC_PI_2);
            let (zero, one) = (R::zero(), R::one());
            let turns = [c(one, zero), c(zero, one), c(-one, zero), c(zero, -one)];
            let sliced = |z: Complex<R>| {
                let mut out = Vec::new();
                Exp.many(Arg::Many(&[z]), 1, &mut out);
                out[0]
            };
            for k in -8i32..=8 {
                let (z, want) = (c(zero, R::narrow(k as f64) * q), turns[k.rem_euclid(4) as usize]);
                assert_eq!((Exp.one(z), sliced(z)), (want, want), "k = {k}");
            }
            let (above, nan) = (c(R::narrow(big), q), c(R::nan(), q));
            assert_eq!((Exp.one(above), sliced(above)), (c(zero, R::infinity()), c(zero, R::infinity())));
            assert!(Exp.one(nan).re.is_nan() && sliced(nan).re.is_nan());
            let (minus, half, three) = (c(-one, zero), c(R::narrow(0.5), zero), c(R::narrow(1.5), zero));
            for (z, w, want) in [(minus, half, turns[1]), (minus, three, turns[3]), (turns[1], c(one + one, zero), turns[2])] {
                let mut out = Vec::new();
                Pow.many(Arg::Many(&[z]), Arg::Many(&[w]), 1, &mut out);
                assert_eq!((Pow.one(z, w), out[0]), (want, want), "{z:?}^{w:?}");
            }
        }
        check::<f64>(800.0);
        check::<f32>(100.0);
    }
}

mod lambert {
    use num_complex::Complex;
    use rustymath::{LambertW, Unary, lambertw};

    #[test]
    fn special_values() {
        assert!(lambertw(0.0f64).to_bits() == 0 && lambertw(-0.0f64).to_bits() == (-0.0f64).to_bits());
        assert_eq!(lambertw(f64::INFINITY), f64::INFINITY);
        assert!([f64::NEG_INFINITY, f64::NAN, -1.0, -0.3678794411714425].iter().all(|&x| lambertw(x).is_nan()));
        // The floats nearest -1/e lie below it, within half an ulp.
        assert_eq!((lambertw(-0.36787944117144233f64), lambertw(-0.36787945f32)), (-1.0, -1.0));
        assert_eq!(lambertw(1.0f64), 0.5671432904097838);
        let c = |re: f64, im: f64| Complex::new(re, im);
        assert_eq!(LambertW.one(c(0.0, -0.0)).im.to_bits(), (-0.0f64).to_bits());
        // Each side of the cut along the real axis below -1/e, by the sign of the imaginary part's zero.
        let (above, below) = (LambertW.one(c(-2.0, 0.0)), LambertW.one(c(-2.0, -0.0)));
        assert!(above.im > 0.0 && below == above.conj());
        assert!(LambertW.one(c(f64::NAN, 1.0)).re.is_nan());
        assert_eq!(LambertW.one(c(f64::NEG_INFINITY, 0.0)), c(f64::INFINITY, std::f64::consts::PI));
    }
}

mod pow {
    use rustymath::{Binary, Pow};

    /// IEEE 754's special values for `pow`, through the free function and through `Pow`'s scalar form, which has its own fast path.
    #[test]
    fn special_values() {
        for p in [rustymath::pow::<f64> as fn(f64, f64) -> f64, |x, y| Pow.one(x, y)] {
            assert_eq!((p(f64::NAN, 0.0), p(1.0, f64::NAN), p(-2.0, 3.0), p(0.0, -1.0)), (1.0, 1.0, -8.0, f64::INFINITY));
            assert!(p(-2.0, 0.5).is_nan() && (p(2.0, 0.5) - std::f64::consts::SQRT_2).abs() < 1e-15);
        }
        assert!(rustymath::pow(-2.0f32, 3.0) == -8.0 && Pow.one(-2.0f32, 0.5).is_nan());
    }
}
