//! Tolerant comparison of floats, as APL defines it: `x` equals `y` within a relative tolerance `t` when `|x-y|` is at most `t` times
//! the larger of `|x|` and `|y|`. Dyalog APL calls `t` the comparison tolerance, `⎕CT`, and its default is 1e-14. A tolerance has no
//! effect when either number is zero.
//!
//! A tolerance is a value of a type that implements `Tolerant`: a `Tolerance`, or `Exactly`, whose methods compile to plain IEEE
//! comparisons and `floor`. A function written once over `Tolerant` compiles to a version for each. `with_tolerance!` picks one when
//! the program runs, outside any loop. The methods are generic over the float type and compute at its width. `TolerantComplex` gives
//! the complex versions.
//!
//! Tolerant equality isn't transitive: `x` can equal `y` and `y` equal `z` while `x` doesn't equal `z`. Hash maps and sorting need a
//! transitive `PartialEq`. This module therefore has no float type with a tolerant `==`. Each comparison is a method of the tolerance.
//!
//! ```
//! use rustymath::tolerant::{Exactly, Tolerance, Tolerant};
//! let t = Tolerance::DEFAULT;
//! assert!(t.equal(0.1 + 0.2, 0.3));
//! assert!(!Exactly.equal(0.1 + 0.2, 0.3));
//! assert_eq!(t.floor(2.9999999999999996), 3.0);
//! assert_eq!(Exactly.floor(2.9999999999999996), 2.0);
//! ```
use crate::{
    float::{sealed::Core, Float, Native},
    ok, BETWEEN_CHECKS,
    search::{self, Chains, Key},
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
    /// the magnitudes that equals the ratio. It computes in `f64` and rounds. NaN when either argument is NaN or infinite.
    fn gcd<F: Float>(self, x: F, y: F) -> F {
        let (x, y) = (x.total().wide(), y.total().wide());
        if !(x.is_finite() && y.is_finite()) { return F::nan(); }
        let (x, y) = (x.abs().max(y.abs()), x.abs().min(y.abs()));
        if y == 0.0 { return F::narrow(x); }
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
}
impl<T: Tolerant> TolerantComplex for T {}

/// Runs `$body` with `$t` bound to `Exactly` when the `Tolerance` `$tolerance` is 0, and to the tolerance otherwise. Each binding
/// has its own compiled copy of `$body`. The choice happens once, outside any loop in `$body`.
///
/// ```
/// use rustymath::{tolerant::{Tolerance, Tolerant}, with_tolerance};
/// let xs = [0.3, 0.1 + 0.2, 0.31];
/// let count = |t: Tolerance| with_tolerance!(t, |t| xs.iter().filter(|&&x| t.equal(x, 0.3)).count());
/// assert_eq!((count(Tolerance::DEFAULT), count(Tolerance::EXACT)), (2, 1));
/// ```
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

/// For each item of `y`, the position of the first item of `x` that it matches within tolerance `t`, or the length of `x` where none
/// does. NaN matches NaN. With `Exactly`, the search indexes order keys (`search::index_of`). With a `Tolerance`, it files the items of
/// `x` in buckets of neighbouring floats (`Near`), and compares each item of `y` with the items in the two buckets nearest it.
///
/// ```
/// use rustymath::tolerant::{index_of, Exactly, Tolerance};
/// let x = [0.3, 1.0, 0.1 + 0.2];
/// assert_eq!(index_of(Tolerance::DEFAULT, &x, &[0.1 + 0.2, 2.0]), [0, 3]);
/// assert_eq!(index_of(Exactly, &x, &[0.1 + 0.2, 2.0]), [2, 3]);
/// ```
pub fn index_of<F: Float>(t: impl Tolerant, x: &[F], y: &[F]) -> Vec<usize> {
    let Ok(found) = index_of_with(t, x, y, ok);
    found
}

/// `index_of`, calling `check` before each block of `BETWEEN_CHECKS` items of `y`, and returning `check`'s error at once.
pub fn index_of_with<F: Float, E>(t: impl Tolerant, x: &[F], y: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<usize>, E> {
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

/// The position of the earliest item of each item's class within tolerance `t`. With `Exactly`, a class holds equal floats. With a
/// `Tolerance`, a class is a run that `runs` gives, so two items of one class needn't match each other.
pub fn classes<F: Float>(t: impl Tolerant, x: &[F]) -> Vec<usize> {
    let Ok(classes) = classes_with(t, x, ok);
    classes
}

/// `classes`, calling `check` after about every `BETWEEN_CHECKS` items, and returning `check`'s error at once.
pub fn classes_with<F: Float, E>(t: impl Tolerant, x: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<usize>, E> {
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
    if t.value() == 0.0 { return search::firsts(&order_keys(x)); }
    classes(t, x).into_iter().enumerate().map(|(i, c)| c == i).collect()
}

/// Calls `each` with the positions of each run of `x`, in ascending order of position. A run is a longest sequence of floats, taken in
/// ascending order, in which each matches the next within tolerance `t`. Every float that a float matches lies in one interval
/// (`Tolerant::range`), so a run holds every match of each of its floats. An error from `each` ends the walk, and `runs` returns it.
/// The floats sort by their order keys, which read back as floats in memory order.
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
    pub fn near(&self, y: F) -> impl Iterator<Item = usize> + '_ {
        let (b, upper) = self.bucket(y);
        [Some(b), if upper { b.checked_add(1) } else { b.checked_sub(1) }].into_iter().flatten().flat_map(|b| self.chains.chain(b))
    }
}

/// The order key of each float.
fn order_keys<F: Float>(x: &[F]) -> Vec<F::OrderKey> { x.iter().map(|v| v.order_key()).collect() }

#[cfg(test)]
mod tests {
    use super::*;
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
        // At f32, a tolerance of 4 units in the last place.
        let t32 = Tolerance::new(4.0 * f32::EPSILON as f64).unwrap();
        assert!(t32.equal(1.0f32, 1.0 + 4.0 * f32::EPSILON) && !t32.equal(1.0f32, 1.0 + 5.0 * f32::EPSILON));
        assert_eq!((Tolerance::new(-1e-14), Tolerance::new(1.0), Tolerance::new(f64::NAN)), (None, None, None));
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

    /// Tolerant search finds the first match that a brute-force search finds, at every width and at small and large tolerances, among
    /// floats crowded near a few values, and among zeros, infinities and NaN.
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
        }
        for t in [T, Tolerance::new(1e-3).unwrap(), Tolerance::new(0.3).unwrap()] { check::<f64>(t); check::<f32>(t); check::<f16>(t); }
    }
}
