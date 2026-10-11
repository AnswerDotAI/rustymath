//! The accuracy tests. Each function is compared with a correctly rounded reference at every f16 and f32 input and at many f64
//! inputs, and with published test cases, with `mla` both fused and unfused. Each test prints its worst error.
use crate::{exp::exp_at, ln::ln_at, trig::{cos_at, sin_at}};
use half::f16;
use std::{fmt::Debug, str::FromStr, thread};

/// A float type whose errors the tests measure.
trait Measured: Copy + Debug + Send + Sync {
    /// A key whose order is the order of the floats, with -0 just below 0. Adjacent floats have adjacent keys.
    fn key(self) -> u64;
    fn is_nan(self) -> bool;
    /// Whether the float is zero, infinite or NaN, whose results must be exactly IEEE's.
    fn is_special(self) -> bool;
}
macro_rules! measured {
    ($($t:ty: $bits:ty),+) => {$(
        impl Measured for $t {
            fn key(self) -> u64 {
                let (bits, sign) = (self.to_bits(), 1 << (<$bits>::BITS - 1));
                (if bits & sign != 0 { !bits } else { bits | sign }) as u64
            }
            fn is_nan(self) -> bool { num_traits::Float::is_nan(self) }
            fn is_special(self) -> bool { !num_traits::Float::is_normal(self) && !num_traits::Float::is_subnormal(self) }
        }
    )+};
}
measured!(f16: u16, f32: u32, f64: u64);

/// The error of `got` in ulps, against the correctly rounded result `want` at `x`. A special input or a NaN must give exactly the
/// right result, or the error is `u64::MAX`.
fn error<T: Measured>(x: T, got: T, want: T) -> u64 {
    let exact = |same: bool| if same { 0 } else { u64::MAX };
    if want.is_nan() || got.is_nan() { exact(want.is_nan() && got.is_nan()) }
    else if x.is_special() { exact(got.key() == want.key()) }
    else { got.key().abs_diff(want.key()) }
}

/// The worst error of `f` against `reference` at the inputs `input(i)` for each `i` below `n`, with the input that gives it. The
/// inputs run on every core.
fn worst<T: Measured>(n: u64, input: impl Fn(u64) -> T + Sync, f: impl Fn(T) -> T + Sync, reference: impl Fn(T) -> T + Sync) -> (u64, T) {
    let threads = thread::available_parallelism().map_or(1, |n| n.get()) as u64;
    thread::scope(|scope| {
        let parts: Vec<_> = (0..threads)
            .map(|t| {
                let (input, f, reference) = (&input, &f, &reference);
                scope.spawn(move || {
                    let (start, end) = (n * t / threads, n * (t + 1) / threads);
                    (start..end).map(|i| { let x = input(i); (error(x, f(x), reference(x)), x) }).max_by_key(|&(e, _)| e)
                })
            })
            .collect();
        parts.into_iter().filter_map(|part| part.join().unwrap()).max_by_key(|&(e, _)| e).unwrap()
    })
}

/// The worst error of `f` over `cases` of inputs and correctly rounded results, with the input that gives it.
fn listed<T: Measured>(cases: &[(T, T)], f: impl Fn(T) -> T) -> (u64, T) {
    cases.iter().map(|&(x, want)| (error(x, f(x), want), x)).max_by_key(|&(e, _)| e).unwrap()
}

/// Asserts that the worst error `(e, x)` of the function called `name` is at most `bound` ulps, and prints it.
fn check<T: Measured>(name: &str, bound: u64, (e, x): (u64, T)) {
    eprintln!("{name}: worst error {e} ulp, at {x:?}");
    assert!(e <= bound, "{name} is {e} ulp out at {x:?}, beyond its bound of {bound}");
}

/// A random 64-bit number for each `i`: splitmix64's output.
pub(crate) fn random(i: u64) -> u64 {
    let z = i.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}
/// A random f64 from `lo` to `hi`.
fn uniform(i: u64, lo: f64, hi: f64) -> f64 { lo + (hi - lo) * (random(i) >> 11) as f64 / (1u64 << 53) as f64 }
/// A random f64 of either sign whose bits are random, so that small magnitudes are as likely as large ones. It includes the
/// subnormals, the infinities and NaN.
fn scattered(i: u64) -> f64 { f64::from_bits(random(i)) }

/// The f64 reference for `exp`. pxfm's `f_exp` is wrong near the bottom of the subnormal results, as at -708.6554, so results below
/// the normal floats come from the platform's `exp`.
fn exp64(x: f64) -> f64 { if x < -708.3 { x.exp() } else { pxfm::f_exp(x) } }

/// The cases in libc-test's vectors for the function called `name`: each input with its correctly rounded result.
fn libc_test<T: FromStr<Err: Debug>>(name: &str) -> Vec<(T, T)> {
    rows(include_str!("../testdata/libc-test.txt")).filter(|r| r[0] == name).map(|r| (r[1].parse().unwrap(), r[2].parse().unwrap())).collect()
}
/// CORE-MATH's inputs for the function called `name` whose correctly rounded results are hardest to find.
fn core_math(name: &str) -> Vec<f64> { rows(include_str!("../testdata/core-math.txt")).filter(|r| r[0] == name).map(|r| r[1].parse().unwrap()).collect() }
/// Each line of a test data file that isn't a comment, split at spaces.
fn rows(text: &str) -> impl Iterator<Item = Vec<&str>> { text.lines().filter(|l| !l.starts_with('#')).map(|l| l.split_whitespace().collect()) }

/// `f` at both widths, with `mla` fused and unfused.
macro_rules! variants {
    ($f:ident) => {
        [($f::<f64, true> as fn(f64) -> f64, $f::<f32, true> as fn(f32) -> f32, "fused"), ($f::<f64, false>, $f::<f32, false>, "unfused")]
    };
}

#[test]
fn exp_accuracy() {
    let hard = core_math("exp");
    for (wide, narrow, how) in variants!(exp_at) {
        check(&format!("f32 exp, {how}"), 2, worst(1 << 32, |i| f32::from_bits(i as u32), narrow, pxfm::f_expf));
        check(&format!("f64 exp at scattered inputs, {how}"), 1, worst(1 << 24, scattered, wide, exp64));
        check(&format!("f64 exp, {how}"), 1, worst(1 << 24, |i| uniform(i, -760.0, 720.0), wide, exp64));
        check(&format!("f64 exp at CORE-MATH's inputs, {how}"), 1, worst(hard.len() as u64, |i| hard[i as usize], wide, exp64));
        check(&format!("libc-test exp, {how}"), 1, listed(&libc_test("exp"), wide));
        check(&format!("libc-test expf, {how}"), 2, listed(&libc_test("expf"), narrow));
    }
    check("f16 exp", 1, worst(1 << 16, |i| f16::from_bits(i as u16), crate::exp, |x| f16::from_f64(pxfm::f_exp(x.into()))));
}

#[test]
fn ln_accuracy() {
    let hard = core_math("log");
    for (wide, narrow, how) in variants!(ln_at) {
        check(&format!("f32 ln, {how}"), 3, worst(1 << 32, |i| f32::from_bits(i as u32), narrow, pxfm::f_logf));
        check(&format!("f64 ln at scattered inputs, {how}"), 2, worst(1 << 24, scattered, wide, pxfm::f_log));
        check(&format!("f64 ln near 1, {how}"), 2, worst(1 << 24, |i| uniform(i, 0.5, 2.0), wide, pxfm::f_log));
        check(&format!("f64 ln at CORE-MATH's inputs, {how}"), 2, worst(hard.len() as u64, |i| hard[i as usize], wide, pxfm::f_log));
        check(&format!("libc-test log, {how}"), 2, listed(&libc_test("log"), wide));
        check(&format!("libc-test logf, {how}"), 3, listed(&libc_test("logf"), narrow));
    }
    check("f16 ln", 1, worst(1 << 16, |i| f16::from_bits(i as u16), crate::ln, |x| f16::from_f64(pxfm::f_log(x.into()))));
}

/// The accuracy checks that `sin` and `cos` share. `variants` are the function at both widths with `mla` fused and unfused, `f16` is
/// the public function at f16, and `wide_ref` and `narrow_ref` are the references. Inputs beyond 2^23 take the exact reduction.
fn trig_accuracy(
    name: &str,
    variants: [(fn(f64) -> f64, fn(f32) -> f32, &str); 2],
    f16: fn(f16) -> f16,
    wide_ref: fn(f64) -> f64,
    narrow_ref: fn(f32) -> f32,
    (wide_bound, narrow_bound): (u64, u64),
) {
    let hard = core_math(name);
    let near = reduction();
    for (wide, narrow, how) in variants {
        check(&format!("f32 {name}, {how}"), narrow_bound, worst(1 << 32, |i| f32::from_bits(i as u32), narrow, narrow_ref));
        check(&format!("f64 {name} at scattered inputs, {how}"), wide_bound, worst(1 << 24, scattered, wide, wide_ref));
        check(&format!("f64 {name} up to 2^23, {how}"), wide_bound, worst(1 << 24, |i| uniform(i, -8388608.0, 8388608.0), wide, wide_ref));
        check(&format!("f64 {name} up to 10, {how}"), wide_bound, worst(1 << 24, |i| uniform(i, -10.0, 10.0), wide, wide_ref));
        check(&format!("f64 {name} at CORE-MATH's inputs, {how}"), wide_bound, worst(hard.len() as u64, |i| hard[i as usize], wide, wide_ref));
        check(&format!("f64 {name} near multiples of pi/2, {how}"), wide_bound, worst(near.len() as u64, |i| near[i as usize], wide, wide_ref));
        check(&format!("libc-test {name}, {how}"), wide_bound, listed(&libc_test(name), wide));
        check(&format!("libc-test {name}f, {how}"), narrow_bound, listed(&libc_test(&format!("{name}f")), narrow));
    }
    check(&format!("f16 {name}"), 1, worst(1 << 16, |i| f16::from_bits(i as u16), f16, |x| f16::from_f64(wide_ref(x.into()))));
}

/// The doubles below 2^23 closest to multiples of π/2, from `testdata/reduction.txt`, with their negatives.
fn reduction() -> Vec<f64> { rows(include_str!("../testdata/reduction.txt")).flat_map(|r| { let x: f64 = r[0].parse().unwrap(); [x, -x] }).collect() }

#[test]
fn sin_accuracy() { trig_accuracy("sin", variants!(sin_at), crate::sin, pxfm::f_sin, pxfm::f_sinf, (4, 2)); }

#[test]
fn cos_accuracy() { trig_accuracy("cos", variants!(cos_at), crate::cos, pxfm::f_cos, pxfm::f_cosf, (4, 4)); }
