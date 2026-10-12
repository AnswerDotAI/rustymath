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
    ok, pairs, Arg, BETWEEN_CHECKS,
    search::{self, Chains, Key, Links},
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
    /// The whole number equal to `y`, when there is one.
    #[inline]
    fn whole<F: Float>(self, y: F) -> Option<F> {
        let n = nearest(y);
        ((y - n).abs() <= F::narrow(self.value()) * larger(y.abs(), n.abs())).then_some(n)
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
    /// number with an infinite or NaN part equals only itself, as IEEE compares it. Two real numbers compare as `equal` compares them.
    #[inline]
    fn complex_equal<R: Native>(self, x: Complex<R>, y: Complex<R>) -> bool {
        if x.im == R::zero() && y.im == R::zero() { return self.equal(x.re, y.re); }
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
/// lie in, and `right` the band for an item on its right. `None`, appending nothing, unless exactly one argument is `One` and `c`
/// isn't NaN.
#[inline(always)]
fn against<F: Float>(t: Tolerance, x: Arg<'_, F>, y: Arg<'_, F>, n: usize, out: &mut Vec<bool>, left: Band, right: Band) -> Option<bool> {
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

/// Appends `f` on each of `n` items of `items` to `out`, through `pairs`.
#[inline(always)]
fn mask<F: Copy>(items: Arg<'_, F>, n: usize, out: &mut Vec<bool>, f: impl Fn(F) -> bool) -> bool {
    pairs(items, Arg::One(()), n, out, #[inline(always)] |v, ()| (f(v), true))
}

/// [`Tolerant::equal`], or [`TolerantComplex::complex_equal`] for complex items, at the tolerance it holds, as a function object.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Equal(pub Tolerance);
compared!(Equal: |t, [x, y]| t.equal(x, y), Inside, Inside);
tolerant!(Equal at Complex<f32> => bool, Complex<f64> => bool: |t, [x, y]| t.complex_equal(x, y));

/// The negation of [`Equal`] at the tolerance it holds, as a function object, for real and complex items.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct NotEqual(pub Tolerance);
compared!(NotEqual: |t, [x, y]| !t.equal(x, y), Outside, Outside);
tolerant!(NotEqual at Complex<f32> => bool, Complex<f64> => bool: |t, [x, y]| !t.complex_equal(x, y));

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
/// (`search::index_of`), so items equal to one whole number match each other. Otherwise, with `Exactly`, the search indexes order
/// keys. With a `Tolerance`, it files the items of `x` in buckets of neighbouring floats (`Near`), and compares each item of `y` with
/// the items in the two buckets nearest it.
pub fn index_of<F: Float>(t: impl Tolerant, x: &[F], y: &[F]) -> Vec<usize> {
    let Ok(found) = index_of_with(t, x, y, ok);
    found
}

/// `index_of`, calling `check` before each block of `BETWEEN_CHECKS` items of `y`, and returning `check`'s error at once.
#[inline(never)]
pub fn index_of_with<F: Float, E>(t: impl Tolerant, x: &[F], y: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<usize>, E> {
    if y.len() <= search::SCANNED { return Ok(y.iter().map(|&v| scan(t, x, v)).collect()); }
    if let Some(keys) = wholes(t, x) && let Some(y) = wholes(t, y) { return Ok(search::index_of(&keys, &y)); }
    if t.value() == 0.0 { return Ok(search::index_of(&order_keys(x), &order_keys(y))); }
    let near = Near::new(t, x, |j, i| x[j].order_key() == x[i].order_key());
    let mut found = Vec::with_capacity(y.len());
    for block in y.chunks(BETWEEN_CHECKS) {
        check()?;
        for &v in block {
            let mut first = x.len();
            for i in near.near(v) { if i < first && t.matches(x[i], v) { first = i; } }
            found.push(first);
        }
    }
    Ok(found)
}

/// The position of the earliest item of each item's class within tolerance `t`. When every item equals a whole number below
/// `whole_limit(t)`, a class holds the items equal to one whole number. Otherwise, with `Exactly`, a class holds equal floats, and with
/// a `Tolerance`, a class is a run that `runs` gives. Two items of one class therefore needn't match each other.
pub fn classes<F: Float>(t: impl Tolerant, x: &[F]) -> Vec<usize> {
    let Ok(classes) = classes_with(t, x, ok);
    classes
}

/// `classes`, calling `check` after about every `BETWEEN_CHECKS` items, and returning `check`'s error at once.
#[inline(never)]
pub fn classes_with<F: Float, E>(t: impl Tolerant, x: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<usize>, E> {
    if let Some(keys) = wholes(t, x) { return Ok(search::classes(&keys)); }
    if t.value() == 0.0 { return Ok(search::classes(&order_keys(x))); }
    let (mut classes, mut since) = (vec![0; x.len()], 0);
    runs(t, x, |run| {
        since += run.len();
        if since >= BETWEEN_CHECKS { since = 0; check()?; }
        for &i in run { classes[i] = run[0]; }
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
    if let Some(keys) = wholes(t, x) { return Ok(search::firsts(&keys)); }
    if t.value() == 0.0 { return Ok(search::firsts(&order_keys(x))); }
    Ok(classes_with(t, x, check)?.into_iter().enumerate().map(|(i, c)| c == i).collect())
}

/// Calls `each` with the positions of each run of `x`, in ascending order of position. A run is a longest sequence of floats, taken in
/// ascending order, in which each matches the next within tolerance `t`. Every float that a float matches lies in one interval
/// (`Tolerant::range`), so a run holds every match of each of its floats. An error from `each` ends the walk, and `runs` returns it.
/// The floats sort by their order keys, which read back as floats in memory order.
#[inline(never)]
pub fn runs<F: Float, E>(t: impl Tolerant, x: &[F], mut each: impl FnMut(&[usize]) -> Result<(), E>) -> Result<(), E> {
    let sorted = search::sorted(x.iter().zip(0..).map(|(&v, i)| (v.order_key(), i)).collect());
    let mut run = Vec::new();
    for pairs in sorted.chunk_by(|a, b| t.matches(F::from_order_key(a.0), F::from_order_key(b.0))) {
        run.clear();
        run.extend(pairs.iter().map(|&(_, i)| i as usize));
        run.sort_unstable();
        each(&run)?;
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

/// Each item of `x` as the whole number it equals within tolerance `t`, when every item equals one below `whole_limit(t)` in
/// magnitude. A search can then index these integers in place of the floats.
fn wholes<F: Float>(t: impl Tolerant, x: &[F]) -> Option<Vec<i64>> {
    let limit = whole_limit(t);
    x.iter().map(|&v| t.whole(v).map(F::wide).filter(|n| n.abs() < limit).map(|n| n as i64)).collect()
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Arg, Binary, Unary};
    use half::f16;
    use num_complex::Complex64;

    const T: Tolerance = Tolerance::DEFAULT;
    fn c(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }

    #[test]
    fn equal_and_match() {
        // The rounded pair 0.3 and 0.1+0.2 lies inside the default tolerance, and 1 and 1+3e-14 outside it.
        assert!(T.equal(0.3, 0.1 + 0.2) && T.equal(-0.3, -(0.1 + 0.2)) && !Exactly.equal(0.3, 0.1 + 0.2));
        assert!(!T.equal(1.0, 1.0 + 3e-14));
        assert_eq!([3.1, 3.0, -2.0, -3.0].map(|x| T.equal(3.0, x)), [false, true, false, false]);
        let wide = Tolerance::new(1e-10).unwrap();
        assert!(wide.equal(1.0, 1.000000000001) && !wide.equal(1.0, 1.0000001));
        // A tolerance has no effect beside zero.
        assert!(!T.equal(0.0, 1e-300) && T.equal(0.0, -0.0));
        assert!(T.equal(f64::INFINITY, f64::INFINITY) && !T.equal(f64::INFINITY, f64::MAX) && !T.equal(f64::NAN, f64::NAN));
        assert!(T.matches(f64::NAN, f64::NAN) && !T.matches(f64::NAN, 1.0));
        assert_eq!((T.compare(0.3, 0.1 + 0.2), T.compare(1.0, 2.0), T.compare(f64::NAN, 1.0)), (Some(Ordering::Equal), Some(Ordering::Less), None));
        // Below and above leave out what's equal within the tolerance, and comparisons with NaN are false.
        assert!(!T.less(0.3, 0.1 + 0.2) && T.less_equal(0.1 + 0.2, 0.3) && Exactly.less(0.3, 0.1 + 0.2) && !Exactly.greater(0.3, 0.1 + 0.2));
        assert!(!T.less(f64::NAN, 1.0) && !T.less_equal(f64::NAN, f64::NAN) && !T.greater_equal(1.0, f64::NAN));
        // At f32, a tolerance of 4 units in the last place.
        let t32 = Tolerance::new(4.0 * f32::EPSILON as f64).unwrap();
        assert!(t32.equal(1.0f32, 1.0 + 4.0 * f32::EPSILON) && !t32.equal(1.0f32, 1.0 + 5.0 * f32::EPSILON));
        assert_eq!((Tolerance::new(-1e-14), Tolerance::new(1.0), Tolerance::new(f64::NAN)), (None, None, None));
        // `with_tolerance!` runs its body at `Exactly` for a tolerance of 0.
        let count = |t: Tolerance| with_tolerance!(t, |t| [0.3, 0.1 + 0.2, 0.31].iter().filter(|&&x| t.equal(x, 0.3)).count());
        assert_eq!((count(T), count(Tolerance::EXACT)), (2, 1));
    }

    #[test]
    fn floor_ceil_and_whole() {
        assert_eq!([-2.3, 0.1, 100.0, 3.3, 0.5 + 0.4, 0.5 + 0.5, 0.5 + 0.6].map(|y| T.floor(y)), [-3.0, 0.0, 100.0, 3.0, 0.0, 1.0, 1.0]);
        assert_eq!((T.floor(2.9999999999999996), Exactly.floor(2.9999999999999996)), (3.0, 2.0));
        assert_eq!((T.ceil(1.0000000000000002), Exactly.ceil(1.0000000000000002)), (1.0, 2.0));
        assert_eq!((T.floor(-0.9999999999999999), T.floor(-0.999999999999)), (-1.0, -1.0));
        assert_eq!((T.floor(0.999999999999), T.floor(f64::INFINITY)), (0.0, f64::INFINITY));
        assert!(T.floor(f64::NAN).is_nan());
        assert_eq!((T.floor(f16::from_f32(2.5)), T.ceil(2.5f32)), (f16::from_f32(2.0), 3.0));
        assert_eq!((T.whole(2.9999999999999996), T.whole(2.5), T.whole(f64::INFINITY), Exactly.whole(4.0)), (Some(3.0), None, None, Some(4.0)));
    }

    #[test]
    fn residue() {
        assert_eq!([(3.0, -5.0), (3.0, 5.0), (-3.0, -4.0), (-3.0, 4.0)].map(|(x, y)| T.residue(x, y)), [1.0, 2.0, -1.0, -2.0]);
        let r = [3.12, -1.0, -0.6].map(|y| T.residue(0.5, y));
        assert!((r[0] - 0.12).abs() < 1e-15 && r[1] == 0.0 && (r[2] - 0.4).abs() < 1e-15);
        let r = [(-1.0, -5.25), (0.0, 0.0), (1.0, 2.41)].map(|(x, y)| T.residue(x, y));
        assert!(r[0] == -0.25 && r[1] == 0.0 && (r[2] - 0.41).abs() < 1e-15);
        // A quotient within tolerance of a whole number leaves no remainder, where exact comparison leaves a whole divisor.
        assert_eq!((T.residue(0.1, 0.3), T.residue(0.0, 2.5)), (0.0, 2.5));
        assert!((Exactly.residue(0.1, 0.3) - 0.1).abs() < 1e-15);
    }

    #[test]
    fn complex() {
        let a = c(2.0, 1e-14);
        assert!(T.complex_equal(a, c(2.0, 0.00000000000001)) && !T.complex_equal(a, c(2.0, 0.0000000000001)));
        assert!(T.complex_equal(c(0.3, 0.0), c(0.1 + 0.2, 0.0)) && !T.complex_equal(c(f64::NAN, 1.0), c(f64::NAN, 1.0)));
        assert_eq!([c(1.0, 3.2), c(3.3, 2.5), c(-3.3, -2.5)].map(|y| T.complex_floor(y)), [c(1.0, 3.0), c(3.0, 2.0), c(-3.0, -3.0)]);
        assert_eq!(T.complex_ceil(c(1.0, 3.2)), c(1.0, 4.0));
        let r = [c(2.0, 3.0), c(3.0, 4.0), c(5.0, 6.0)].map(|y| T.complex_residue(c(1.0, 2.0), y));
        assert_eq!(r, [c(1.0, 1.0), c(-1.0, 1.0), c(0.0, 1.0)]);
        assert_eq!(T.complex_residue(c(0.0, 0.0), c(2.0, 1.0)), c(2.0, 1.0));
        // 5 = (2+i)(2-i) and 3+4i = (2+i)², so their gcd is 2+i, the associate in the first quadrant. Halving both halves it.
        assert_eq!((T.complex_gcd(c(5.0, 0.0), c(3.0, 4.0)), T.complex_gcd(c(2.5, 0.0), c(1.5, 2.0))), (c(2.0, 1.0), c(1.0, 0.5)));
        assert_eq!((T.complex_gcd(c(0.0, 0.0), c(-4.0, 3.0)), T.complex_gcd(c(12.0, 0.0), c(18.0, 0.0))), (c(3.0, 4.0), c(6.0, 0.0)));
        assert!(T.complex_gcd(c(f64::INFINITY, 1.0), c(2.0, 1.0)).re.is_nan());
    }

    /// `range(c)` gives the last floats equal to `c` on each side.
    fn check_range<F: Float + std::fmt::Debug>(t: impl Tolerant, c: F) {
        let (lo, hi) = t.range(c).unwrap();
        assert!(lo <= c && c <= hi, "{c:?}");
        assert!(t.equal(lo, c) && t.equal(hi, c) && !t.equal(lo.next_down(), c) && !t.equal(hi.next_up(), c), "{c:?}: {lo:?} {hi:?}");
    }

    #[test]
    fn range() {
        let specials = [0.3, -0.3, 1.0, 1e-5, 5e-324, 2.2250738585072014e-308, 8796093022208.0, -1e20, f64::MAX, -f64::MAX];
        for c in specials { check_range(T, c); check_range(Exactly, c); }
        for c in (0..100_000).map(|i| f64::from_bits(crate::accuracy::random(i))).filter(|c| !c.is_nan()) { check_range(T, c); }
        let t32 = Tolerance::new(4.0 * f32::EPSILON as f64).unwrap();
        for c in [0.3f32, -7.5, 1e-40, f32::MAX] { check_range(t32, c); }
        let t16 = Tolerance::new(4.0 * f16::EPSILON.to_f64()).unwrap();
        for c in [0.3, -7.5, 65504.0, 1e-7].map(f16::from_f64) { check_range(t16, c); }
        assert_eq!((T.range(f64::NAN), T.range(0.0), T.range(f64::NEG_INFINITY)), (None, Some((0.0, 0.0)), Some((f64::NEG_INFINITY, f64::NEG_INFINITY))));
        // At f16 the default tolerance rounds to 0, so every float equals only itself.
        assert_eq!(T.range(f16::from_f32(0.3)), Some((f16::from_f32(0.3), f16::from_f32(0.3))));
    }

    /// Below a magnitude of 1/(8t), where no float equals two whole numbers, the tolerant floor of `y` is the ordinary floor of the
    /// greatest float equal to `y`. The test steps up to 60 floats either side of whole numbers.
    #[test]
    fn floor_is_the_floor_of_the_range() {
        for n in [1.0f64, 3.0, -3.0, 1e6, -1e6, 1e12, 4e12, -4e12] {
            let (mut down, mut up) = (n, n);
            for _ in 0..60 {
                for y in [down, up] { assert_eq!(T.floor(y), T.range(y).unwrap().1.floor(), "{y}"); }
                (down, up) = (down.next_down(), up.next_up());
            }
        }
    }

    #[test]
    fn gcd() {
        assert!((T.gcd(0.6, 0.4) - 0.2).abs() < 1e-15 && Exactly.gcd(0.6, 0.4) < 1e-15);
        assert_eq!((T.gcd(12.0, 18.0), T.gcd(-12.0, 18.0), T.gcd(0.0, 2.5), T.gcd(0.0, 0.0), T.gcd(12.0f32, 18.0)), (6.0, 6.0, 2.5, 0.0, 6.0));
        assert!(T.gcd(f64::NAN, 1.0).is_nan() && T.gcd(f64::NAN, f64::NAN).is_nan() && T.gcd(f64::INFINITY, 1.0).is_nan());
    }

    /// Checks each real comparison's slice form against its scalar form for many items against one number, in either order, with each
    /// item of `items` as the number.
    fn against_one<F: Float + std::fmt::Debug>(t: Tolerance, items: &[f64])
    where
        Equal: Binary<F, Out = bool>,
        NotEqual: Binary<F, Out = bool>,
        Less: Binary<F, Out = bool>,
        LessEqual: Binary<F, Out = bool>,
        Greater: Binary<F, Out = bool>,
        GreaterEqual: Binary<F, Out = bool>,
        Matches: Binary<F, Out = bool>,
    {
        fn check<F: Copy + std::fmt::Debug, K: Binary<F, Out = bool> + std::fmt::Debug>(k: K, x: &[F]) {
            for &c in x {
                let (mut left, mut right) = (Vec::new(), Vec::new());
                assert!(k.many(Arg::Many(x), Arg::One(c), x.len(), &mut left) & k.many(Arg::One(c), Arg::Many(x), x.len(), &mut right));
                assert_eq!(left, x.iter().map(|&v| k.one(v, c)).collect::<Vec<_>>(), "{k:?} with {c:?} on the right");
                assert_eq!(right, x.iter().map(|&v| k.one(c, v)).collect::<Vec<_>>(), "{k:?} with {c:?} on the left");
            }
        }
        let x: Vec<F> = items.iter().map(|&v| F::narrow(v)).collect();
        check(Equal(t), &x);
        check(NotEqual(t), &x);
        check(Less(t), &x);
        check(LessEqual(t), &x);
        check(Greater(t), &x);
        check(GreaterEqual(t), &x);
        check(Matches(t), &x);
    }
    /// Each tolerant object gives its method's result for each item, at the default tolerance and at 0, for slices beside slices and
    /// one item beside a slice, at f32, f64 and complex items. Only `Gcd`'s infinite argument fails.
    #[test]
    fn objects() {
        let (xs, ys) = (&[3.0, -3.0, 0.3, 0.0, f64::NAN, 2.0][..], &[-5.0, 5.0, 0.1 + 0.2, 7.0, 1.0, 2.9999999999999996][..]);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for t in [T, Tolerance::EXACT] {
            let (mut floors, mut residues, mut equal, mut order) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            assert!(Floor(t).many(Arg::Many(ys), 6, &mut floors) & Ceil(t).many(Arg::Many(ys), 6, &mut floors));
            assert!(Residue(t).many(Arg::Many(xs), Arg::Many(ys), 6, &mut residues) & Residue(t).many(Arg::One(3.0), Arg::Many(ys), 6, &mut residues));
            assert!(Equal(t).many(Arg::Many(xs), Arg::Many(ys), 6, &mut equal) & Matches(t).many(Arg::Many(xs), Arg::One(f64::NAN), 6, &mut equal));
            assert!(Compare(t).many(Arg::Many(xs), Arg::Many(ys), 6, &mut order));
            let floor: Vec<f64> = ys.iter().map(|&y| t.floor(y)).chain(ys.iter().map(|&y| t.ceil(y))).collect();
            let residue: Vec<f64> = (0..6).map(|i| t.residue(xs[i], ys[i])).chain(ys.iter().map(|&y| t.residue(3.0, y))).collect();
            let eq: Vec<bool> = (0..6).map(|i| t.equal(xs[i], ys[i])).chain(xs.iter().map(|&x| t.matches(x, f64::NAN))).collect();
            assert_eq!((bits(&floors), bits(&residues), equal), (bits(&floor), bits(&residue), eq));
            assert_eq!(order, (0..6).map(|i| t.compare(xs[i], ys[i]).unwrap_or(Ordering::Greater)).collect::<Vec<_>>());
            let check = |many: &dyn Fn(&mut Vec<bool>) -> bool, one: &dyn Fn(f64, f64) -> bool| {
                let mut got = Vec::new();
                assert!(many(&mut got));
                assert_eq!(got, (0..6).map(|i| one(xs[i], ys[i])).collect::<Vec<_>>());
            };
            check(&|o| NotEqual(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| !t.equal(x, y));
            check(&|o| Less(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.less(x, y));
            check(&|o| LessEqual(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.less_equal(x, y));
            check(&|o| Greater(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.greater(x, y));
            check(&|o| GreaterEqual(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.greater_equal(x, y));
            let (ys32, mut floors32) = (ys.iter().map(|&y| y as f32).collect::<Vec<_>>(), Vec::new());
            assert!(Floor(t).many(Arg::Many(&ys32), 6, &mut floors32));
            assert_eq!(floors32, ys32.iter().map(|&y| t.floor(y)).collect::<Vec<_>>());
            let (zs, mut complex) = (&[c(2.3, -1.7), c(0.5, 0.5), c(-1.0, 2.0)][..], Vec::new());
            assert!(Floor(t).many(Arg::Many(zs), 3, &mut complex) & Residue(t).many(Arg::One(c(1.0, 1.0)), Arg::Many(zs), 3, &mut complex));
            let want: Vec<Complex64> = zs.iter().map(|&z| t.complex_floor(z)).chain(zs.iter().map(|&z| t.complex_residue(c(1.0, 1.0), z))).collect();
            assert_eq!(complex, want);
        }
        // Many items against one number take the range of floats equal to it, in either order. Each test item serves as the number.
        let items = [3.0, -3.0, 0.3, 0.1 + 0.2, 0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0, 2.9999999999999996, 1e-310, 1.0 + 3e-14];
        for t in [T, Tolerance::EXACT] {
            against_one::<f64>(t, &items);
            against_one::<f32>(t, &items);
            against_one::<f16>(t, &items);
        }
        assert_eq!(Gcd(T).one(12.0, 18.0), 6.0);
        assert!(!Gcd(T).many(Arg::One(f64::INFINITY), Arg::Many(&[2.0][..]), 1, &mut Vec::new()));
        // Lcm has the sign of x y, gives 0 where the gcd is 0, and divides at the tolerance's gcd.
        assert_eq!([Lcm(T).one(-4.0, 6.0), Lcm(T).one(0.0, 5.0), Lcm(T).one(0.6, 0.4)], [-12.0, 0.0, 1.2000000000000002]);
        assert_eq!(Lcm(T).one(c(2.0, 0.0), c(0.0, 3.0)), c(0.0, 6.0));
    }

    /// Tolerant search finds the first match that a brute-force search finds, at every width and at small and large tolerances, among
    /// floats crowded near a few values, and among zeros, infinities and NaN, whether it scans for a few items or indexes many.
    #[test]
    fn slice_search() {
        fn check<F: Float + std::fmt::Debug>(t: Tolerance) {
            let base = [1.0, -1.0, 3.0, 1e-3, 0.0, 1000.0];
            let x: Vec<F> = (0..400u64)
                .map(|i| {
                    let r = crate::accuracy::random(i);
                    F::narrow(base[(r % 6) as usize] * (1.0 + (((r >> 8) % 9) as f64 - 4.0) * t.value() / 2.0))
                })
                .chain([F::nan(), F::infinity(), F::neg_infinity(), F::neg_zero()])
                .collect();
            let y: Vec<F> = x.iter().rev().copied().chain([F::narrow(7.0)]).collect();
            let brute: Vec<usize> = y.iter().map(|&v| x.iter().position(|&h| t.matches(h, v)).unwrap_or(x.len())).collect();
            assert_eq!(index_of(t, &x, &y), brute, "{t:?}");
            assert_eq!(index_of(t, &x, &y[..5]), brute[..5], "{t:?}, scanning");
        }
        for t in [T, Tolerance::new(1e-3).unwrap(), Tolerance::new(0.3).unwrap()] { check::<f64>(t); check::<f32>(t); check::<f16>(t); }
        let x = [0.3, 1.0, 0.1 + 0.2];
        assert_eq!((index_of(T, &x, &[0.1 + 0.2, 2.0]), index_of(Exactly, &x, &[0.1 + 0.2, 2.0])), (vec![0, 3], vec![2, 3]));
        // Floats that each equal a whole number index as integers. At a tolerance of 0.3, which makes 3 equal 4, `whole_limit` lets only
        // 0 qualify, so the search falls back to buckets.
        let wx: Vec<f64> = (0..300u64).map(|i| (crate::accuracy::random(i) % 50) as f64 - 25.0).collect();
        let wy: Vec<f64> = (0..40).map(|i| i as f64 - 30.0).collect();
        for t in [T, Tolerance::EXACT, Tolerance::new(0.3).unwrap()] {
            let brute: Vec<usize> = wy.iter().map(|&v| wx.iter().position(|&h| t.matches(h, v)).unwrap_or(wx.len())).collect();
            assert_eq!(index_of(t, &wx, &wy), brute, "{t:?}");
        }
        let ints: Vec<i64> = wx.iter().map(|&v| v as i64).collect();
        assert_eq!((classes(T, &wx), firsts(T, &wx)), (search::classes(&ints), search::firsts(&ints)));
    }
}
