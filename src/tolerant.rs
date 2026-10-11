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
use crate::float::{sealed::Core, Float, Native};
use num_bigint::BigInt;
use num_complex::Complex;
use num_rational::BigRational;
use num_traits::{Signed, ToPrimitive, Zero};
use std::cmp::Ordering;

/// A relative tolerance, from 0 up to but not including 1. A tolerance of 0 compares exactly, as `Exactly` does, but with more work.
#[derive(Clone, Copy, Debug, PartialEq, PartialOrd)]
pub struct Tolerance(f64);

impl Tolerance {
    /// Exact comparison.
    pub const EXACT: Self = Self(0.0);
    /// Dyalog APL's default comparison tolerance, 1e-14. It is about 45 units in the last place of an `f64`.
    pub const DEFAULT: Self = Self(1e-14);
    /// The tolerance `t`, when it lies from 0 up to but not including 1. `None` for any other `t`, including NaN.
    pub fn new(t: f64) -> Option<Self> { (0.0..1.0).contains(&t).then_some(Self(t)) }
    /// Whether this tolerance is 0.
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
        let r = if self.equal(x * q.round(), y) { F::zero() } else { y - x * self.floor(q) };
        if x == F::zero() { y } else { r }
    }
    /// The whole number equal to `y`, when there is one.
    #[inline]
    fn whole<F: Float>(self, y: F) -> Option<F> {
        let n = y.round();
        ((y - n).abs() <= F::narrow(self.value()) * y.abs().max(n.abs())).then_some(n)
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
        (x == y) | (x.is_finite() & y.is_finite() & ((x - y).abs() <= t * x.abs().max(y.abs())))
    }
    #[inline]
    fn floor<F: Float>(self, y: F) -> F {
        let (n, t) = (y.round(), F::narrow(self.0));
        if (n > y) & (n - y > t * y.abs().max(n.abs())) { n - F::one() } else { n }
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
        if self.complex_equal(x * Complex::new(q.re.round(), q.im.round()), y) { Complex::zero() } else { y - x * self.complex_floor(q) }
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
}
