//! Tolerant comparison of floats, as APL defines it: `x` equals `y` within a relative tolerance `t` when `|x-y|` is at most `t` times
//! the larger of `|x|` and `|y|`. Dyalog APL calls `t` the comparison tolerance, `⎕CT`, and its default is 1e-14. A tolerance has no
//! effect when either number is zero.
//!
//! A tolerance is a value of a type that implements `Tolerant`: a `Tolerance`, or `Exactly`, whose methods compile to plain IEEE
//! comparisons and `floor`. A function written once over `Tolerant` compiles to a version for each. `with_tolerance!` picks one when
//! the program runs, outside any loop. The methods are generic over the float type and compute at its width. `TolerantComplex` gives
//! the complex versions. The function objects `Equal`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual`, `Matches`,
//! `Compare`, `Floor`, `Ceil`, `Residue`, `Gcd` and `Lcm` each hold a `Tolerance` and give the same results as the methods, one item
//! at a time or over slices. A slice form picks its loop with `with_tolerance!` once. A comparison's slice form compares many items
//! with one number through the range of floats equal to it (`Tolerant::range`), so each item takes one or two plain comparisons. All
//! but `Matches` and `Compare` also have integer forms, from the `integer` module, which ignore the tolerance. `Floor` and `Ceil` also
//! take Booleans.
//!
//! Tolerant equality isn't transitive: `x` can equal `y` and `y` equal `z` while `x` doesn't equal `z`. Hash maps and sorting need a
//! transitive `PartialEq`. This module therefore has no float type with a tolerant `==`. Each comparison is a method of the tolerance.
use crate::{
    convert::{float, int},
    float::{Float, Native},
    ok, Arg, BETWEEN_CHECKS,
    search::{self, Chains, Key, Links},
    slice::pairs_into,
};
use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use std::{cmp::Ordering, marker::PhantomData};

/// A relative tolerance, from 0 up to but not including 1. A tolerance of 0 compares exactly, as `Exactly` does, but with more work.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Tolerance(f64);

impl Tolerance {
    /// Exact comparison.
    pub const EXACT: Self = Self(0.0);
    /// Dyalog APL's default comparison tolerance, 1e-14. It is about 45 units in the last place of an `f64`.
    pub const DEFAULT: Self = Self(1e-14);
    /// The tolerance `t`, when it lies from 0 up to but not including 1. `None` for any other `t`, including NaN.
    #[inline]
    pub fn new(t: f64) -> Option<Self> { (0.0..1.0).contains(&t).then_some(Self(t)) }
    /// Whether this tolerance is 0.
    #[inline]
    pub fn is_exact(self) -> bool { self.0 == 0.0 }
}

/// Comparison within a tolerance. `Exactly` and `Tolerance` implement it.
pub trait Tolerant: Copy {
    /// The tolerance: 0 for `Exactly`.
    fn value(self) -> f64;
    /// Whether `x` equals `y`. NaN equals nothing, as in IEEE, and an infinity equals only itself.
    fn equal<F: Float>(self, x: F, y: F) -> bool;
    /// The floor of `y`: the whole number nearest to `y`, unless it lies above `y` and doesn't equal it. Then the whole number below.
    fn floor<F: Float>(self, y: F) -> F;
    /// The ceiling of `y`: the whole number nearest to `y`, unless it lies below `y` and doesn't equal it. Then the whole number above.
    #[inline]
    fn ceil<F: Float>(self, y: F) -> F { -self.floor(-y) }
    /// Whether `x` and `y` match: they are equal, or both are NaN. A search that must find NaN compares with `matches`.
    #[inline]
    fn matches<F: Float>(self, x: F, y: F) -> bool { self.equal(x, y) | (x.is_nan() & y.is_nan()) }
    /// Whether `x` is below `y` and doesn't equal it. Comparisons with NaN are false.
    #[inline]
    fn less<F: Float>(self, x: F, y: F) -> bool { (x < y) & !self.equal(x, y) }
    /// Whether `x` is below `y` or equals it. Comparisons with NaN are false.
    #[inline]
    fn less_equal<F: Float>(self, x: F, y: F) -> bool { (x <= y) | self.equal(x, y) }
    /// Whether `x` is above `y` and doesn't equal it. Comparisons with NaN are false.
    #[inline]
    fn greater<F: Float>(self, x: F, y: F) -> bool { self.less(y, x) }
    /// Whether `x` is above `y` or equals it. Comparisons with NaN are false.
    #[inline]
    fn greater_equal<F: Float>(self, x: F, y: F) -> bool { self.less_equal(y, x) }
    /// The order of `x` and `y`, which is `Equal` when they are equal. `None` when either is NaN.
    #[inline]
    fn compare<F: Float>(self, x: F, y: F) -> Option<Ordering> { if self.equal(x, y) { Some(Ordering::Equal) } else { x.partial_cmp(&y) } }
    /// `y` modulo `x`, with the sign of `x`: `y - x·floor(y/x)`. A quotient equal to a whole number leaves no remainder. `x` = 0 gives
    /// `y`.
    #[inline]
    fn residue<F: Float>(self, x: F, y: F) -> F {
        let q = y / x;
        let r = if self.equal(x * nearest(q), y) { F::zero() } else { y - x * self.floor(q) };
        if x == F::zero() { y } else { r }
    }
    /// The whole number nearest `y`, and whether `y` equals it within this tolerance.
    #[inline]
    fn nearest_whole<F: Float>(self, y: F) -> (F, bool) {
        let n = nearest(y);
        (n, (y - n).abs() <= F::narrow(self.value()) * larger(y.abs(), n.abs()))
    }
    /// The whole number equal to `y`, when there is one.
    #[inline]
    fn whole<F: Float>(self, y: F) -> Option<F> {
        let (n, equal) = self.nearest_whole(y);
        equal.then_some(n)
    }
    /// The floats equal to `c`, as `(lo, hi)`: `equal(x, c)` holds exactly when `lo <= x && x <= hi`. Comparing many floats with `c`
    /// then takes two plain comparisons for each, in a loop that vectorizes. `None` when `c` is NaN, which equals nothing.
    fn range<F: Float>(self, c: F) -> Option<(F, F)> {
        if c.is_nan() { return None; }
        if !c.is_finite() || c == F::zero() { return Some((c, c)); }
        let t = F::narrow(self.value());
        let (near, far) = (c * (F::one() - t), c / (F::one() - t));
        let (lo, hi) = if c > F::zero() { (near, far) } else { (far, near) };
        // The last float equal to `c` in the direction that `out` steps, found from the estimate `x`. `back` steps towards `c`.
        let end = |mut x: F, out: fn(F) -> F, back: fn(F) -> F| {
            while !self.equal(x, c) { x = back(x); }
            while self.equal(out(x), c) { x = out(x); }
            x
        };
        Some((end(lo, F::next_down, F::next_up), end(hi, F::next_up, F::next_down)))
    }
    /// The greatest common divisor of `x` and `y`: the larger magnitude divided by the numerator of the first convergent of the ratio of
    /// the magnitudes that equals the ratio. It computes in `f64` and rounds. NaN when either argument is NaN or infinite. Whole
    /// magnitudes whose product is below 1e14 take integer gcd, which gives the same result at any tolerance up to the default.
    fn gcd<F: Float>(self, x: F, y: F) -> F {
        let (x, y) = (x.wide(), y.wide());
        if !(x.is_finite() && y.is_finite()) { return F::nan(); }
        let (x, y) = (x.abs().max(y.abs()), x.abs().min(y.abs()));
        if y == 0.0 { return F::narrow(x); }
        let whole = x.fract() == 0.0 && y.fract() == 0.0 && x * y < 1e14 && self.value() <= Tolerance::DEFAULT.value();
        if let Some(g) = whole.then(|| crate::integer::gcd(int(x)?, int(y)?)).flatten() { return float(g); }
        let scale = BigRational::from_float(x).unwrap();
        let ratio = &scale / BigRational::from_float(y).unwrap();
        let tolerance = BigRational::from_float(self.value()).unwrap() * &ratio;
        let (mut n, mut d) = ratio.clone().into_raw();
        let (mut p0, mut p1) = (BigInt::from(0), BigInt::from(1));
        let (mut q0, mut q1) = (BigInt::from(1), BigInt::from(0));
        loop {
            let a = &n / &d;
            (p0, p1) = (p1.clone(), &a * &p1 + p0);
            (q0, q1) = (q1.clone(), &a * &q1 + q0);
            if (BigRational::new(p1.clone(), q1.clone()) - &ratio).abs() <= tolerance {
                return F::narrow((scale / BigRational::from_integer(p1)).to_f64().unwrap());
            }
            (n, d) = (d.clone(), n % d);
        }
    }
}

/// The larger of `a` and `b`. A NaN argument makes the caller's comparison false, whichever one this returns. wasm SIMD has an
/// instruction for this choice. It has none for `max`, which must return the number beside a NaN.
#[inline(always)]
fn larger<F: Float>(a: F, b: F) -> F { if a > b { a } else { b } }

/// The whole number nearest to `y`, with a tie going down. It's built on `floor`, an instruction on every target. In wasm, `round` is a
/// library call, and a call in a loop stops it vectorizing.
#[inline(always)]
fn nearest<F: Float>(y: F) -> F {
    let f = y.floor();
    if y - f > F::narrow(0.5) { f + F::one() } else { f }
}

/// Exact comparison, as a tolerance of 0 gives, through plain IEEE comparisons and `floor`.
#[derive(Clone, Copy, Debug)]
pub struct Exactly;

impl Tolerant for Exactly {
    #[inline]
    fn value(self) -> f64 { 0.0 }
    #[inline]
    fn equal<F: Float>(self, x: F, y: F) -> bool { x == y }
    #[inline]
    fn floor<F: Float>(self, y: F) -> F { y.floor() }
}

impl Tolerant for Tolerance {
    #[inline]
    fn value(self) -> f64 { self.0 }
    // A loop of comparisons vectorizes only when the test has no branch. The test therefore uses `&` and `|`, not `&&` and `||`.
    #[inline]
    fn equal<F: Float>(self, x: F, y: F) -> bool {
        let t = F::narrow(self.0);
        (x == y) | (x.is_finite() & y.is_finite() & ((x - y).abs() <= t * larger(x.abs(), y.abs())))
    }
    #[inline]
    fn floor<F: Float>(self, y: F) -> F {
        let (n, t) = (y.floor() + F::one(), F::narrow(self.0));
        if n - y <= t * larger(y.abs(), n.abs()) { n } else { n - F::one() }
    }
}

/// The complex versions of `Tolerant`'s comparisons, for every tolerance.
pub trait TolerantComplex: Tolerant {
    /// Whether `x` equals `y`: the magnitude of `x-y` is at most the tolerance times the larger magnitude, as Dyalog APL defines it. A
    /// number with an infinite or NaN part equals only itself, as IEEE compares it. A tolerance of 0 compares the parts exactly.
    #[inline]
    fn complex_equal<R: Native>(self, x: Complex<R>, y: Complex<R>) -> bool {
        let (equal, decided) = self.squared_equal(x, y);
        if decided { equal } else { self.scaled_equal(x, y) }
    }
    /// `complex_equal` through squared magnitudes, |x-y|² ≤ t² max(|x|², |y|²), with no branch or square root, and whether that test
    /// decides it. A tolerance of 0 compares the parts exactly, which always decides. Otherwise the test decides wherever the larger
    /// squared magnitude lies between 2^-800 and 2^1000 at f64, or 2^-80 and 2^120 at f32. In that range no square overflows, and t²
    /// times one doesn't underflow. A NaN part gives `false`, as IEEE compares it.
    #[inline(always)]
    fn squared_equal<R: Native>(self, x: Complex<R>, y: Complex<R>) -> (bool, bool) {
        let t = R::narrow(self.value());
        let (low, high) = if R::WIDE { (2f64.powi(-800), 2f64.powi(1000)) } else { (2f64.powi(-80), 2f64.powi(120)) };
        let size = x.norm_sqr().max(y.norm_sqr());
        let near = (x - y).norm_sqr() <= t * t * size;
        let exact = t == R::zero();
        ((exact & (x == y)) | (!exact & near), exact | ((size >= R::narrow(low)) & (size <= R::narrow(high))))
    }
    /// `complex_equal` from the magnitudes themselves, which `norm` computes without overflow or underflow, for the pairs that
    /// `squared_equal` leaves undecided.
    fn scaled_equal<R: Native>(self, x: Complex<R>, y: Complex<R>) -> bool {
        let finite = |z: Complex<R>| z.re.is_finite() & z.im.is_finite();
        if !finite(x) || !finite(y) { return x == y; }
        let t = R::narrow(self.value());
        (x - y).norm() <= (x * t).norm().max((y * t).norm())
    }
    /// McDonnell's complex floor: the floors of both parts, plus 1 on the part with the larger fraction when the fractions add up to 1.
    #[inline]
    fn complex_floor<R: Native>(self, y: Complex<R>) -> Complex<R> {
        let (a, b) = (self.floor(y.re), self.floor(y.im));
        let (x, z) = (y.re - a, y.im - b);
        if x + z < R::one() - R::narrow(self.value()) {
            Complex::new(a, b)
        } else if x <= z {
            Complex::new(a, b + R::one())
        } else {
            Complex::new(a + R::one(), b)
        }
    }
    #[inline]
    fn complex_ceil<R: Native>(self, y: Complex<R>) -> Complex<R> { -self.complex_floor(-y) }
    /// `y` modulo `x` for complex numbers. A quotient equal to a Gaussian integer leaves no remainder. `x` = 0 gives `y`.
    #[inline]
    fn complex_residue<R: Native>(self, x: Complex<R>, y: Complex<R>) -> Complex<R> {
        if x.is_zero() { return y; }
        let q = y / x;
        if self.complex_equal(x * Complex::new(nearest(q.re), nearest(q.im)), y) { Complex::zero() } else { y - x * self.complex_floor(q) }
    }
    /// The greatest common divisor of `x` and `y` as complex numbers, by Euclid's algorithm with the quotient rounded to the nearest
    /// Gaussian integer. It stops once the remainder is within the tolerance of the larger magnitude, and turns the result into the
    /// first quadrant, real part above 0. Two real numbers give `gcd`. A non-finite argument, or a remainder that stops shrinking, gives
    /// NaN.
    fn complex_gcd<R: Native>(self, x: Complex<R>, y: Complex<R>) -> Complex<R> {
        if x.im == R::zero() && y.im == R::zero() { return Complex::new(self.gcd(x.re, y.re), R::zero()); }
        let nan = Complex::new(R::nan(), R::nan());
        let scale = x.norm().max(y.norm());
        if !scale.is_finite() { return nan; }
        let original = if x.norm() >= y.norm() { x } else { y };
        let t = R::narrow(self.value()) * scale;
        let whole = |z: Complex<R>| Complex::new(z.re.round(), z.im.round());
        let small = |z: Complex<R>| Complex::new(if z.re.abs() <= t { R::zero() } else { z.re }, if z.im.abs() <= t { R::zero() } else { z.im });
        let (mut x, mut y) = (x, y);
        while y.norm() > t {
            let r = x - y * whole(x / y);
            if !(r.norm() < y.norm()) { return nan; }
            (x, y) = (y, r);
        }
        let mut x = small(x);
        if x.is_zero() { return x; }
        while x.re <= R::zero() || x.im < R::zero() { x = Complex::new(-x.im, x.re); }
        small(original / whole(original / x))
    }
}
impl<T: Tolerant> TolerantComplex for T {}

/// Runs `$body` with `$t` bound to `Exactly` when the `Tolerance` `$tolerance` is 0, and to the tolerance otherwise. Each binding
/// has its own compiled copy of `$body`. The choice happens once, outside any loop in `$body`.
#[macro_export]
macro_rules! with_tolerance {
    ($tolerance:expr, |$t:ident| $body:expr) => {{
        let tolerance: $crate::tolerant::Tolerance = $tolerance;
        if tolerance.is_exact() {
            let $t = $crate::tolerant::Exactly;
            $body
        } else {
            let $t = tolerance;
            $body
        }
    }};
}

/// Writes a tolerant function object's forms from one item body over a tolerance `$t`, at `real` or at a list of concrete types, as
/// `forms!` takes them. The object holds a `Tolerance`. Its slice form reads it once and runs the body's loop at `Exactly` where the
/// tolerance is 0.
macro_rules! tolerant {
    ($object:ident at real $(=> $o:ty)?: $($rest:tt)+) => { tolerant!(@forms $object, (real $(=> $o)?), $($rest)+); };
    ($object:ident at $($e:ty $(=> $o:ty)?),+: $($rest:tt)+) => { tolerant!(@forms $object, ($($e $(=> $o)?),+), $($rest)+); };
    (@forms $object:ident, ($($at:tt)+), |$t:ident, [$($x:ident),+]| $body:expr) => {
        forms!($object at $($at)+: |k, [$($x),+]| with_tolerance!(k.0, |$t| $body),
            many |k, [$($x),+], n, out| with_tolerance!(k.0, |$t| each!([$($x),+], n, out, |[$($x),+]| $body)));
    };
}

/// Writes a tolerant comparison's forms at `real`: its item body over a tolerance `$t`, as `tolerant!` writes them, and a slice form
/// that compares many items with one number through `against`. `$left` is the band an item must lie in when it is on the left of the
/// one number, and `$right` the band when it is on the right.
macro_rules! compared {
    ($object:ident: |$t:ident, [$x:ident, $y:ident]| $body:expr, $left:ident, $right:ident) => {
        forms!($object at real => bool: |k, [$x, $y]| with_tolerance!(k.0, |$t| $body),
            many |k, [$x, $y], n, out| match against(k.0, $x, $y, n, out, Band::$left, Band::$right) {
                Some(ok) => ok,
                None => with_tolerance!(k.0, |$t| each!([$x, $y], n, out, |[$x, $y]| $body)),
            });
    };
}

/// Where an item must lie against the range of floats equal to one number (`Tolerant::range`) for a comparison with that number to
/// hold. `NotAbove` and `NotBelow` include the range itself.
#[derive(Clone, Copy)]
enum Band { Inside, Outside, Below, NotAbove, Above, NotBelow }

/// Compares each item of the `Many` argument with the `One` argument `c`, through the range of floats equal to `c`, worked out once.
/// Each item then takes one or two plain comparisons, in a loop that vectorizes. `left` is the band an item on the left of `c` must
/// lie in, and `right` the band for an item on its right. `None`, writing nothing, unless exactly one argument is `One` and `c`
/// isn't NaN.
#[inline(always)]
fn against<F: Float>(t: Tolerance, x: Arg<'_, F>, y: Arg<'_, F>, n: usize, out: &mut [std::mem::MaybeUninit<bool>], left: Band, right: Band) -> Option<bool> {
    let (items, (lo, hi), band) = match (x, y) {
        (Arg::Many(_), Arg::One(c)) => (x, t.range(c)?, left),
        (Arg::One(c), Arg::Many(_)) => (y, t.range(c)?, right),
        _ => return None,
    };
    Some(match band {
        Band::Inside => mask(items, n, out, #[inline(always)] |v| (lo <= v) & (v <= hi)),
        Band::Outside => mask(items, n, out, #[inline(always)] |v| !((lo <= v) & (v <= hi))),
        Band::Below => mask(items, n, out, #[inline(always)] |v| v < lo),
        Band::NotAbove => mask(items, n, out, #[inline(always)] |v| v <= hi),
        Band::Above => mask(items, n, out, #[inline(always)] |v| v > hi),
        Band::NotBelow => mask(items, n, out, #[inline(always)] |v| v >= lo),
    })
}

/// Writes `f` on each of `n` items of `items` into `out`, through `pairs_into`.
#[inline(always)]
fn mask<F: Copy>(items: Arg<'_, F>, n: usize, out: &mut [std::mem::MaybeUninit<bool>], f: impl Fn(F) -> bool) -> bool {
    pairs_into(items, Arg::One(()), n, out, #[inline(always)] |v, ()| (f(v), true))
}

/// [`Tolerant::equal`], or [`TolerantComplex::complex_equal`] for complex items, at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equal(pub Tolerance);
compared!(Equal: |t, [x, y]| t.equal(x, y), Inside, Inside);
forms!(Equal at Complex<f32> => bool, Complex<f64> => bool: |k, [x, y]| with_tolerance!(k.0, |t| t.complex_equal(x, y)),
    many |k, [x, y], n, out| with_tolerance!(k.0, |t| complex_equals(t, [x, y], n, out, false)));

/// The negation of [`Equal`] at the tolerance it holds, as a function object, for real and complex items.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotEqual(pub Tolerance);
compared!(NotEqual: |t, [x, y]| !t.equal(x, y), Outside, Outside);
forms!(NotEqual at Complex<f32> => bool, Complex<f64> => bool: |k, [x, y]| with_tolerance!(k.0, |t| !t.complex_equal(x, y)),
    many |k, [x, y], n, out| with_tolerance!(k.0, |t| complex_equals(t, [x, y], n, out, true)));

/// `Equal`'s slice form on complex items, or with `negate` `NotEqual`'s. Each `CHUNK` of items splits into slices of real and
/// imaginary parts, as in `slice::split`, so `squared_equal` runs over plain slices with no branch and vectorizes. `scaled_equal` then
/// takes the pairs it leaves undecided, one at a time. No result fails.
#[inline(always)]
fn complex_equals<T: TolerantComplex, R: Native>(t: T, [x, y]: [Arg<'_, Complex<R>>; 2], n: usize, out: &mut [std::mem::MaybeUninit<bool>], negate: bool) -> bool {
    if pairs_into(x, y, n, out, #[inline(always)] |a, b| { let (equal, decided) = t.squared_equal(a, b); (equal != negate, decided) }) { return true; }
    let out = crate::slice::written(out);
    let at = |a: Arg<'_, Complex<R>>, i: usize| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
    for i in 0..n {
        let (a, b) = (at(x, i), at(y, i));
        if !t.squared_equal(a, b).1 { out[i] = t.scaled_equal(a, b) != negate }
    }
    true
}

/// [`Tolerant::less`] at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Less(pub Tolerance);
compared!(Less: |t, [x, y]| t.less(x, y), Below, Above);

/// [`Tolerant::less_equal`] at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LessEqual(pub Tolerance);
compared!(LessEqual: |t, [x, y]| t.less_equal(x, y), NotAbove, NotBelow);

/// [`Tolerant::greater`] at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Greater(pub Tolerance);
compared!(Greater: |t, [x, y]| t.greater(x, y), Above, Below);

/// [`Tolerant::greater_equal`] at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GreaterEqual(pub Tolerance);
compared!(GreaterEqual: |t, [x, y]| t.greater_equal(x, y), NotBelow, NotAbove);

/// [`Tolerant::matches`] at the tolerance it holds, as a function object: `equal`, except that NaN matches NaN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Matches(pub Tolerance);
compared!(Matches: |t, [x, y]| t.matches(x, y), Inside, Inside);

/// [`Tolerant::compare`] at the tolerance it holds, as a function object. NaN orders after every number and equal to NaN, as
/// [`Tolerant::matches`] has NaN match NaN.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Compare(pub Tolerance);
tolerant!(Compare at real => Ordering: |t, [x, y]| t.compare(x, y).unwrap_or(x.is_nan().cmp(&y.is_nan())));

/// [`Tolerant::floor`], or [`TolerantComplex::complex_floor`] for complex items, at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Floor(pub Tolerance);
tolerant!(Floor at real: |t, [y]| t.floor(y));
tolerant!(Floor at Complex<f32>, Complex<f64>: |t, [y]| t.complex_floor(y));

/// [`Tolerant::ceil`], or [`TolerantComplex::complex_ceil`] for complex items, at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Ceil(pub Tolerance);
tolerant!(Ceil at real: |t, [y]| t.ceil(y));
tolerant!(Ceil at Complex<f32>, Complex<f64>: |t, [y]| t.complex_ceil(y));

/// [`Tolerant::residue`], or [`TolerantComplex::complex_residue`] for complex items, at the tolerance it holds, as a function object:
/// `y` modulo `x`, for arguments `x` and `y` in that order. Its `i64` form is [`integer::residue`](crate::integer::residue), which
/// needs no tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Residue(pub Tolerance);
tolerant!(Residue at real: |t, [x, y]| t.residue(x, y));
tolerant!(Residue at Complex<f32>, Complex<f64>: |t, [x, y]| t.complex_residue(x, y));

/// [`Tolerant::gcd`], or [`TolerantComplex::complex_gcd`] for complex items, at the tolerance it holds, as a function object. A
/// non-finite argument gives NaN, which marks the item failed. Its `i64` form is [`integer::gcd`](crate::integer::gcd).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gcd(pub Tolerance);
tolerant!(Gcd at real: |t, [x, y]| t.gcd(x, y));
tolerant!(Gcd at Complex<f32>, Complex<f64>: |t, [x, y]| t.complex_gcd(x, y));

/// The least common multiple x y / gcd(x, y), with the sign of x y, at the tolerance it holds, as a function object. Where the gcd
/// is 0, so is the result. Its `i64` form is [`integer::lcm`](crate::integer::lcm).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lcm(pub Tolerance);
tolerant!(Lcm at real: |t, [x, y]| {
    let g = t.gcd(x, y);
    if g == 0.0 { g } else { x / g * y }
});
tolerant!(Lcm at Complex<f32>, Complex<f64>: |t, [x, y]| {
    let g = t.complex_gcd(x, y);
    if g == Complex::new(0.0, 0.0) { g } else { x / g * y }
});

/// For each item of `y`, the position of the first item of `x` that it matches within tolerance `t`, or the length of `x` where none
/// does. NaN matches NaN. Up to `search::SCANNED` items of `y` each take one pass over `x`, which compares every item with the range of
/// floats equal to it. When every item of both equals a whole number below `whole_limit(t)`, the search indexes those whole numbers
/// (`search::index_of`), then checks that each item of `y` matches the first item of `x` with its whole number. When one doesn't, as two
/// floats on either side of one whole number may not, the search takes the buckets below. Otherwise, with `Exactly`, the search indexes order
/// keys. With a `Tolerance`, it files the items of `x` in buckets of neighbouring floats (`Near`), and compares each item of `y` with
/// the items in the two buckets nearest it.
pub fn index_of<F: Float>(t: impl Tolerant, x: &[F], y: &[F]) -> Vec<u32> {
    let Ok(found) = index_of_with(t, x, y, ok);
    found
}

/// `index_of`, calling `check` before each block of `BETWEEN_CHECKS` items of `y`, and returning `check`'s error at once.
#[inline(never)]
pub fn index_of_with<F: Float, E>(t: impl Tolerant, x: &[F], y: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<u32>, E> {
    if y.len() <= search::SCANNED { return Ok(y.iter().map(|&v| scan(t, x, v) as u32).collect()); }
    if let Some((keys, exact)) = wholes(t, x, whole_limit(t)) && let Some((whole_y, _)) = wholes(t, y, whole_limit(t)) {
        // Two floats near one whole number each equal it, but needn't equal each other. Unless every item of `x` is a whole number,
        // each needle must be identical to its first candidate or within tolerance of it, or the buckets below search instead.
        // Floats near other whole numbers can't match.
        let found = search::index_of(&keys, &whole_y);
        if exact || found.iter().zip(y).all(|(&p, &v)| p as usize == x.len() || x[p as usize] == v || t.matches(x[p as usize], v)) { return Ok(found); }
    }
    if t.value() == 0.0 { return Ok(search::index_of(&order_keys(x), &order_keys(y))); }
    let near = Near::new(t, x, |j, i| x[j].order_key() == x[i].order_key());
    let mut found = Vec::with_capacity(y.len());
    for block in y.chunks(BETWEEN_CHECKS) {
        check()?;
        for &v in block {
            let mut first = x.len();
            for i in near.near(v) { if i < first && t.matches(x[i], v) { first = i; } }
            found.push(first as u32);
        }
    }
    Ok(found)
}

/// The position of the earliest item of each item's class within tolerance `t`. When every item equals a whole number below
/// `whole_limit(t)`, a class holds the items equal to one whole number. Otherwise, with `Exactly`, a class holds equal floats, and with
/// a `Tolerance`, a class is a run that `runs` gives. Two items of one class therefore needn't match each other.
pub fn classes<F: Float>(t: impl Tolerant, x: &[F]) -> Vec<u32> {
    let Ok(classes) = classes_with(t, x, ok);
    classes
}

/// `classes`, calling `check` after about every `BETWEEN_CHECKS` items, and returning `check`'s error at once.
#[inline(never)]
pub fn classes_with<F: Float, E>(t: impl Tolerant, x: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<u32>, E> {
    if let Some((keys, _)) = wholes(t, x, whole_limit(t)) { return Ok(search::classes(&keys)); }
    if t.value() == 0.0 { return Ok(search::classes(&order_keys(x))); }
    let (mut classes, mut since) = (vec![0; x.len()], 0);
    runs(t, x, |run| {
        since += run.len();
        if since >= BETWEEN_CHECKS { since = 0; check()?; }
        for &i in run { classes[i] = run[0] as u32; }
        Ok(())
    })?;
    Ok(classes)
}

/// Whether each item of `x` is the first of its class, with the classes that `classes` gives.
pub fn firsts<F: Float>(t: impl Tolerant, x: &[F]) -> Vec<bool> {
    let Ok(mask) = firsts_with(t, x, ok);
    mask
}

/// `firsts`, calling `check` as `classes_with` does, and returning `check`'s error at once.
#[inline(never)]
pub fn firsts_with<F: Float, E>(t: impl Tolerant, x: &[F], check: impl FnMut() -> Result<(), E>) -> Result<Vec<bool>, E> {
    if let Some((keys, _)) = wholes(t, x, whole_limit(t)) { return Ok(search::firsts(&keys)); }
    if t.value() == 0.0 { return Ok(search::firsts(&order_keys(x))); }
    Ok(classes_with(t, x, check)?.into_iter().enumerate().map(|(i, c)| c as usize == i).collect())
}

/// Calls `each` with the positions of each run of `x`, in ascending order of position. A run is a longest sequence of floats, taken in
/// ascending order, in which each matches the next within tolerance `t`. Every float that a float matches lies in one interval
/// (`Tolerant::range`), so a run holds every match of each of its floats. An error from `each` ends the walk, and `runs` returns it.
/// The floats sort by their order keys, which read back as floats in memory order.
#[inline(never)]
pub fn runs<F: Float, E>(t: impl Tolerant, x: &[F], mut each: impl FnMut(&[usize]) -> Result<(), E>) -> Result<(), E> {
    let (keys, order) = search::sorted(x.iter().map(|&v| v.order_key()).collect());
    let (mut run, mut start) = (Vec::new(), 0);
    for end in 1..=keys.len() {
        if end < keys.len() && t.matches(F::from_order_key(keys[end - 1]), F::from_order_key(keys[end])) { continue; }
        run.clear();
        run.extend(order[start..end].iter().map(|&i| i as usize));
        run.sort_unstable();
        each(&run)?;
        start = end;
    }
    Ok(())
}

/// Floats filed in buckets of neighbouring order keys, for tolerant search. Two floats within tolerance `t` of each other lie at most
/// `2t/((1-t)ε)` floats apart, where `ε` is the type's epsilon. A bucket holds a power of 2 of keys, at least four times that. A key in
/// the lower half of its bucket is therefore far enough from the upper edge that every float it matches lies in its own bucket or the
/// neighbour on the side of its half. At f64 with the default tolerance, a bucket holds 512 keys.
pub struct Near<F> { chains: Chains, shift: u32, width: PhantomData<F> }

impl<F: Float> Near<F> {
    /// Files each item of `x` in its bucket, except an item `i` for which `copies(j, i)` holds with an item `j` already in the bucket.
    /// A copy can't be a first match, because an earlier item matches whatever it matches.
    pub fn new(t: impl Tolerant, x: &[F], copies: impl Fn(usize, usize) -> bool) -> Self {
        let (t, epsilon) = (t.value(), F::epsilon().to_f64().unwrap_or(1.0));
        let apart = (4.0 * t / ((1.0 - t) * epsilon)).ceil() + 1.0;
        let shift = (2.0 * (apart + 1.0)).log2().ceil().min(64.0) as u32;
        let mut near = Self { chains: Chains::new(x.len()), shift, width: PhantomData };
        for (i, &v) in x.iter().enumerate() { near.chains.push_new(near.bucket(v).0, i, |j| copies(j, i)); }
        near
    }
    /// The bucket of `y`, and whether `y` lies in its upper half.
    #[inline]
    fn bucket(&self, y: F) -> (u64, bool) {
        let k = y.order_key().key() as u64;
        (k.checked_shr(self.shift).unwrap_or(0), self.shift > 0 && (k >> (self.shift - 1)) & 1 == 1)
    }
    /// The positions filed near `y`: those in its bucket, then those in the neighbour on the side of its half, each in order of position.
    #[inline]
    pub fn near(&self, y: F) -> Links<'_> {
        let (b, upper) = self.bucket(y);
        self.chains.chains(b, if upper { b.checked_add(1) } else { b.checked_sub(1) })
    }
}

/// The order key of each float.
fn order_keys<F: Float>(x: &[F]) -> Vec<F::OrderKey> { x.iter().map(|v| v.order_key()).collect() }

/// The magnitude below which each whole number equals no other whole number within tolerance `t`, nor do floats equal to different
/// whole numbers equal each other: 1/(4t), or 2^53 for a tolerance of 0. It is never above 2^53, below which an `f64` holds every whole
/// number.
pub fn whole_limit(t: impl Tolerant) -> f64 { (0.25 / t.value()).min(9_007_199_254_740_992.0) }

/// Each item of `x` as the whole number it equals within tolerance `t`, when every item equals one below `limit` in magnitude, and
/// whether every item is exactly its whole number. A search can then index these integers in place of the floats, with
/// `whole_limit(t)` as the limit. A `limit` of at most 2^63 keeps every whole number within `i64`. Each block of `WHOLES` items
/// is rounded to its nearest whole numbers in one loop that vectorizes. A block with an item that isn't exactly whole then tests
/// each item against the tolerance in a second loop. The first block that fails ends the conversion.
pub fn wholes<F: Float>(t: impl Tolerant, x: &[F], limit: f64) -> Option<(Vec<i64>, bool)> {
    const WHOLES: usize = 256;
    let (limit, mut keys, mut exact) = (F::narrow(limit), Vec::with_capacity(x.len()), true);
    for block in x.chunks(WHOLES) {
        let (fits, whole) = crate::slice::appended(&mut keys, block.len(), |slots| {
            let (mut fits, mut whole) = (true, true);
            for (slot, &y) in slots.iter_mut().zip(block) {
                let n = nearest(y);
                fits &= n.abs() < limit;
                whole &= n == y;
                slot.write(n.wide() as i64);
            }
            (fits, whole)
        });
        if !fits || !whole && !block.iter().fold(true, |equal, &y| equal & t.nearest_whole(y).1) { return None; }
        exact &= whole;
    }
    Some((keys, exact))
}

/// The position of the first item of `x` that matches `v` within tolerance `t`, or the length of `x`. One pass compares each item with
/// the range of floats equal to `v`. A NaN `v` matches the first NaN.
fn scan<F: Float>(t: impl Tolerant, x: &[F], v: F) -> usize {
    let found = match t.range(v) {
        Some((lo, hi)) => x.iter().position(|&h| (lo <= h) & (h <= hi)),
        None => x.iter().position(|h| h.is_nan()),
    };
    found.unwrap_or(x.len())
}

