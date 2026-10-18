//! Tolerant comparison of floats: `x` equals `y` within a relative tolerance `t` when `|x-y|` is less than `t` times the larger of `|x|` and `|y|`. APL's definition says at most, which differs only where `|x-y|` lands exactly on that bound. Dyalog APL calls `t` the comparison tolerance, `⎕CT`, and its default is 1e-14. A tolerance has no effect when either number is zero.
//!
//! A tolerance is a value of a type that implements `Tolerant`: a `Tolerance`, or `Exactly`, whose methods compile to plain IEEE comparisons and `floor`. A function written once over `Tolerant` compiles to a version for each. `with_tolerance!` picks one when the program runs, outside any loop. The methods are generic over the float type and compute at its width. `TolerantComplex` gives the complex versions. The function objects `Equal`, `NotEqual`, `Less`, `LessEqual`, `Greater`, `GreaterEqual`, `Matches`, `Compare`, `Floor`, `Ceil`, `Residue`, `FloorDivide`, `Gcd` and `Lcm` each hold a `Tolerance` and give the same results as the methods, one item at a time or over slices. A slice form picks its loop with `with_tolerance!` once. A comparison's slice form compares many items with one number through the range of floats equal to it (`Tolerant::range`), so each item takes one or two plain comparisons. All but `Matches` and `Compare` also have integer forms, from the `integer` module, which ignore the tolerance. `Floor` and `Ceil` also take Booleans.
//!
//! Tolerant equality isn't transitive: `x` can equal `y` and `y` equal `z` while `x` doesn't equal `z`. Hash maps and sorting need a transitive `PartialEq`. This module therefore has no float type with a tolerant `==`. Each comparison is a method of the tolerance.
use crate::{Arg, BETWEEN_CHECKS, convert::{float, int}, float::{Float, Native}, ok, search::{self, Chains, Key, Links}, slice::{Total, pairs_into}};
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
pub trait Tolerant: Copy + Send + Sync {
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
    /// The floor of `x/y`, and whether the quotient equals a whole number within this tolerance. Such a quotient gives that whole number, where `residue` leaves no remainder.
    #[inline]
    fn floor_quotient<F: Float>(self, x: F, y: F) -> (F, bool) {
        let q = x / y;
        let n = nearest(q);
        if self.equal(y * n, x) { (n, true) } else { (self.floor(q), false) }
    }
    /// `y` modulo `x`, with the sign of `x`: `y - x·floor(y/x)`, with the floor that `floor_quotient` gives. A quotient equal to a whole number leaves no remainder. `x` = 0 gives `y`.
    #[inline]
    fn residue<F: Float>(self, x: F, y: F) -> F {
        let (f, whole) = self.floor_quotient(y, x);
        let r = if whole { F::zero() } else { y - x * f };
        if x == F::zero() { y } else { r }
    }
    /// The whole number nearest `y`, and whether `y` equals it within this tolerance.
    #[inline]
    fn nearest_whole<F: Float>(self, y: F) -> (F, bool) { let n = nearest(y); (n, (y - n).abs() <= F::narrow(self.value()) * larger(y.abs(), n.abs())) }
    /// The whole number equal to `y`, when there is one.
    #[inline]
    fn whole<F: Float>(self, y: F) -> Option<F> { let (n, equal) = self.nearest_whole(y); equal.then_some(n) }
    /// The floats equal to `c`, as `(lo, hi)`: `equal(x, c)` holds exactly when `lo <= x && x <= hi`. Comparing many floats with `c` then takes two plain comparisons for each, in a loop that vectorizes. `None` when `c` is NaN, which equals nothing.
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
    /// The greatest common divisor of `x` and `y`: the larger magnitude divided by the numerator of the first convergent of the ratio of the magnitudes that equals the ratio. It computes in `f64` and rounds. NaN when either argument is NaN or infinite. Whole magnitudes whose product is below 1e14 take integer gcd, which gives the same result at any tolerance up to the default.
    fn gcd<F: Float>(self, x: F, y: F) -> F {
        let (x, y) = (x.wide(), y.wide());
        if !(x.is_finite() && y.is_finite()) { return F::nan(); }
        let (x, y) = (x.abs().max(y.abs()), x.abs().min(y.abs()));
        if y == 0.0 { return F::narrow(x); }
        let whole = x.fract() == 0.0 && y.fract() == 0.0 && x * y < 1e14 && self.value() <= Tolerance::DEFAULT.value();
        if let Some(g) = whole.then(|| crate::integer::gcd(int(x)?, int(y)?)).flatten() { return float(g); }
        let (t, (nx, dy)) = (self.value(), ratio_parts(x, y));
        if let Some(g) = fixed_gcd(x, y, nx, dy, t) { return F::narrow(g); }
        // The same walk on big integers, where the test `|p·d − q·n| ≤ t·n·q` is exact for `t = m·2^e`.
        let (n, d) = (BigInt::from(nx.0) << nx.1, BigInt::from(dy.0) << dy.1);
        let (m, e, _) = num_traits::Float::integer_decode(t);
        let p = convergents(n.clone(), d.clone(), |p, q| {
            let (gap, bound) = ((p * &d - q * &n).abs(), BigInt::from(m) * &n * q);
            let within = gap.is_zero() || m != 0 && if e >= 0 { gap <= bound << e as usize } else { gap << (-e) as usize <= bound };
            within.then(|| p.clone())
        });
        F::narrow((BigRational::from_float(x).unwrap() / BigRational::from_integer(p)).to_f64().unwrap())
    }
}

/// `x / y`, for finite positive `x` and `y`, as `n / d` with `n = mx·2^sx` and `d = my·2^sy`: odd mantissas, with the larger exponent's excess as a shift on its side.
fn ratio_parts(x: f64, y: f64) -> ((u64, u32), (u64, u32)) {
    let odd = |v: f64| { let (m, e, _) = num_traits::Float::integer_decode(v); (m >> m.trailing_zeros(), i32::from(e) + m.trailing_zeros() as i32) };
    let ((mx, ex), (my, ey)) = (odd(x), odd(y));
    if ex >= ey { ((mx, (ex - ey) as u32), (my, 0)) } else { ((mx, 0), (my, (ey - ex) as u32)) }
}

/// Walks the convergents `p/q` of the continued fraction of `n/d`, for `n, d > 0`, and gives `done`'s answer for the first convergent where it gives one. The last convergent is `n/d` itself, so `done` must answer there at the latest.
fn convergents<I: num_traits::Num + Clone, R>(n: I, d: I, mut done: impl FnMut(&I, &I) -> Option<R>) -> R {
    let (mut a, mut b) = (n, d);
    let (mut p0, mut p1, mut q0, mut q1) = (I::zero(), I::one(), I::one(), I::zero());
    loop {
        let k = a.clone() / b.clone();
        let p = k.clone() * p1.clone() + p0;
        p0 = std::mem::replace(&mut p1, p);
        let q = k * q1.clone() + q0;
        q0 = std::mem::replace(&mut q1, q);
        if let Some(answer) = done(&p1, &q1) { return answer; }
        let r = a % b.clone();
        (a, b) = (b, r);
    }
}

/// [`Tolerant::gcd`]'s walk at tolerance `t`, in fixed width, for finite `x ≥ y > 0` whose ratio has the parts that `ratio_parts` gives. It applies where both parts fit in 64 bits. Every convergent is then below 2^64, and every product below 2^128. `None` where the parts don't fit, where a step's test is too close to decide in floats, or where the result needs a division by an integer above 2^53. The exact walk decides those, so both give the same result.
fn fixed_gcd(x: f64, y: f64, (mx, sx): (u64, u32), (my, sy): (u64, u32), t: f64) -> Option<f64> {
    let shifted = |m: u64, s: u32| (m.leading_zeros() >= s).then(|| m << s);
    let (n, d) = (shifted(mx, sx)?, shifted(my, sy)?);
    convergents(n, d, |&p, &q| {
        // `|p/q − n/d| ≤ t·n/d`, multiplied through by `q·d`. The float estimates decide unless they lie within 1e-15 of each other.
        let gap = (u128::from(p) * u128::from(d)).abs_diff(u128::from(q) * u128::from(n));
        let bound = t * (u128::from(n) * u128::from(q)) as f64;
        let within = if gap == 0 { true } else if bound < 0.5 { false } else {
            let g = gap as f64;
            if g <= bound * (1.0 - 1e-15) { true } else if g >= bound * (1.0 + 1e-15) { false } else { return Some(None) }
        };
        // The result is `x / p`, which IEEE division rounds as the exact walk does while `p` is below 2^53. An exact end has `x / p = y / q`, with the smaller `q`.
        within.then(|| if p <= 1 << 53 { Some(x / p as f64) } else if gap == 0 && q <= 1 << 53 { Some(y / q as f64) } else { None })
    })
}

/// The larger of `a` and `b`. A NaN argument makes the caller's comparison false, whichever one this returns. wasm SIMD has an instruction for this choice. It has none for `max`, which must return the number beside a NaN.
#[inline(always)]
fn larger<F: Float>(a: F, b: F) -> F { if a > b { a } else { b } }

/// The whole number nearest to `y`, with a tie going down. It's built on `floor`, an instruction on every target. In wasm, `round` is a library call, and a call in a loop stops it vectorizing.
#[inline(always)]
fn nearest<F: Float>(y: F) -> F { let f = y.floor(); if y - f > F::narrow(0.5) { f + F::one() } else { f } }

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
    // A loop of comparisons vectorizes only when the test has no branch. The test therefore uses `&` and `|`, not `&&` and `||`. With a strict `<`, an infinity is unequal to every finite number without a finiteness test, because `inf < inf` is false. NaN fails every comparison.
    #[inline]
    fn equal<F: Float>(self, x: F, y: F) -> bool { let t = F::narrow(self.0); (x == y) | ((x - y).abs() < t * larger(x.abs(), y.abs())) }
    #[inline]
    fn floor<F: Float>(self, y: F) -> F {
        // `y` equals the whole number above it when raising `y` by `t·|y|` reaches that number. Capping the raise just below a half keeps a whole number from reaching the next one where `t·|y|` exceeds a half. Where floats are 1 or more apart, the sum rounds back to `y`.
        let (raise, most) = (F::narrow(self.0) * y.abs(), F::narrow(0.5).next_down());
        (y + if raise < most { raise } else { most }).floor()
    }
}

/// The complex versions of `Tolerant`'s comparisons, for every tolerance.
pub trait TolerantComplex: Tolerant {
    /// Whether `x` equals `y`: the magnitude of `x-y` is at most the tolerance times the larger magnitude, as Dyalog APL defines it. A number with an infinite or NaN part equals only itself, as IEEE compares it. A tolerance of 0 compares the parts exactly.
    #[inline]
    fn complex_equal<R: Native>(self, x: Complex<R>, y: Complex<R>) -> bool {
        let (equal, decided) = self.squared_equal(x, y);
        if decided { equal } else { self.scaled_equal(x, y) }
    }
    /// `complex_equal` through squared magnitudes, |x-y|² ≤ t² max(|x|², |y|²), with no branch or square root, and whether that test decides it. A tolerance of 0 compares the parts exactly, which always decides. Otherwise the test decides wherever the larger squared magnitude lies between 2^-800 and 2^1000 at f64, or 2^-80 and 2^120 at f32. In that range no square overflows, and t² times one doesn't underflow. A NaN part gives `false`, as IEEE compares it.
    #[inline(always)]
    fn squared_equal<R: Native>(self, x: Complex<R>, y: Complex<R>) -> (bool, bool) {
        let t = R::narrow(self.value());
        let (low, high) = if R::WIDE { (2f64.powi(-800), 2f64.powi(1000)) } else { (2f64.powi(-80), 2f64.powi(120)) };
        let size = x.norm_sqr().max(y.norm_sqr());
        let near = (x - y).norm_sqr() <= t * t * size;
        let exact = t == R::zero();
        ((exact & (x == y)) | (!exact & near), exact | ((size >= R::narrow(low)) & (size <= R::narrow(high))))
    }
    /// `complex_equal` from the magnitudes themselves, which `norm` computes without overflow or underflow, for the pairs that `squared_equal` leaves undecided.
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
        if x + z < R::one() - R::narrow(self.value()) { Complex::new(a, b) }
        else if x <= z { Complex::new(a, b + R::one()) }
        else { Complex::new(a + R::one(), b) }
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
    /// The greatest common divisor of `x` and `y` as complex numbers, by Euclid's algorithm with the quotient rounded to the nearest Gaussian integer. It stops once the remainder is within the tolerance of the larger magnitude, and turns the result into the first quadrant, real part above 0. Two real numbers give `gcd`. A non-finite argument, or a remainder that stops shrinking, gives NaN.
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

/// Runs `$body` with `$t` bound to `Exactly` when the `Tolerance` `$tolerance` is 0, and to the tolerance otherwise. Each binding has its own compiled copy of `$body`. The choice happens once, outside any loop in `$body`.
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

/// Writes a tolerant function object's forms from one item body over a tolerance `$t`, at `real` or at a list of concrete types, as `forms!` takes them. The object holds a `Tolerance`. Its slice form reads it once and runs the body's loop at `Exactly` where the tolerance is 0.
macro_rules! tolerant {
    ($object:ident at real $(=> $o:ty)?: $($rest:tt)+) => { tolerant!(@forms $object, (real $(=> $o)?), $($rest)+); };
    ($object:ident at $($e:ty $(=> $o:ty)?),+: $($rest:tt)+) => { tolerant!(@forms $object, ($($e $(=> $o)?),+), $($rest)+); };
    (@forms $object:ident, ($($at:tt)+), |$t:ident, [$($x:ident),+]| $body:expr) => {
        forms!($object at $($at)+: |k, [$($x),+]| with_tolerance!(k.0, |$t| $body),
            many |k, [$($x),+], n, out| with_tolerance!(k.0, |$t| each!([$($x),+], n, out, |[$($x),+]| $body)));
    };
}

/// Writes a tolerant comparison's forms at `real`: its item body over a tolerance `$t`, as `tolerant!` writes them, and a slice form that compares many items with one number through `against`. `$left` is the band an item must lie in when it is on the left of the one number, and `$right` the band when it is on the right.
macro_rules! compared {
    ($object:ident: |$t:ident, [$x:ident, $y:ident]| $body:expr, $left:ident, $right:ident) => {
        forms!($object at real => bool: |k, [$x, $y]| with_tolerance!(k.0, |$t| $body),
            many |k, [$x, $y], n, out| match against(k.0, $x, $y, n, out, Band::$left, Band::$right) {
                Some(ok) => ok,
                None => with_tolerance!(k.0, |$t| each!([$x, $y], n, out, |[$x, $y]| $body)),
            });
    };
}

/// Where an item must lie against the range of floats equal to one number (`Tolerant::range`) for a comparison with that number to hold. `NotAbove` and `NotBelow` include the range itself.
#[derive(Clone, Copy)]
enum Band { Inside, Outside, Below, NotAbove, Above, NotBelow }

impl Band {
    /// Whether `v` lies in this band against the range `lo` to `hi`.
    #[inline(always)]
    fn holds<F: PartialOrd>(self, lo: F, hi: F, v: F) -> bool {
        match self {
            Band::Inside => (lo <= v) & (v <= hi), Band::Outside => !((lo <= v) & (v <= hi)), Band::Below => v < lo, Band::NotAbove => v <= hi,
            Band::Above => v > hi, Band::NotBelow => v >= lo
        }
    }
}

/// Runs `$body` with `$b` naming the band `$band` holds, as a constant item in each arm. A closure in `$body` then compiles with the band folded in. It doesn't capture a variable, which would leave each item to test the band.
macro_rules! with_band {
    ($band:expr, |$b:ident| $body:expr) => {
        match $band {
            Band::Inside => { const $b: Band = Band::Inside; $body }
            Band::Outside => { const $b: Band = Band::Outside; $body }
            Band::Below => { const $b: Band = Band::Below; $body }
            Band::NotAbove => { const $b: Band = Band::NotAbove; $body }
            Band::Above => { const $b: Band = Band::Above; $body }
            Band::NotBelow => { const $b: Band = Band::NotBelow; $body }
        }
    };
}

/// The float types that `against` compares with one number. The default writes each result through `pairs_into`, whose loop vectorizes. Wasm's `f64` and `f32` override it in `narrowed`.
trait Banded: Copy + PartialOrd {
    /// Writes into `out` whether each item of `xs` lies in `band` against the range `lo` to `hi`.
    #[inline(always)]
    fn banded(xs: &[Self], lo: Self, hi: Self, band: Band, out: &mut [std::mem::MaybeUninit<bool>]) {
        with_band!(band, |B| pairs_into(
            Arg::Many(xs), Arg::One(()), xs.len(), out, #[inline(always)] |v, ()| (B.holds(lo, hi, v), true)
        ));
    }
}
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
impl Banded for f32 {}
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
impl Banded for f64 {}

/// Compares each item of the `Many` argument with the `One` argument `c`, through the range of floats equal to `c`, worked out once. Each item then takes one or two plain comparisons, through `Banded`. `left` is the band an item on the left of `c` must lie in, and `right` the band for an item on its right. `None`, writing nothing, unless exactly one argument is `One` and `c` isn't NaN.
#[inline(always)]
fn against<F: Float + Banded>(
    t: Tolerance, x: Arg<'_, F>, y: Arg<'_, F>, n: usize, out: &mut [std::mem::MaybeUninit<bool>], left: Band, right: Band) -> Option<bool> {
    let (items, (lo, hi), band) = match (x, y) {
        (Arg::Many(xs), Arg::One(c)) => (xs, t.range(c)?, left), (Arg::One(c), Arg::Many(ys)) => (ys, t.range(c)?, right), _ => return None
    };
    F::banded(&items[..n], lo, hi, band, &mut out[..n]);
    Some(true)
}

/// Wasm has no instruction that narrows a 64-bit lane to a byte. LLVM writes a float comparison's results two items at a time, in about ten instructions for each item. Where it can prove that lanes hold masks, it also replaces wasm's narrowing instructions with slower code. Here each block of 16 items compares with vector instructions, then shuffles the first byte of each mask into place: 7 shuffles for `f64`'s 8 vectors, and 3 for `f32`'s 4.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod narrowed {
    use super::{Band, Banded};
    use core::arch::wasm32::*;
    use std::mem::MaybeUninit;

    /// Each type's vector instructions. `$join` gathers the first byte of each lane's mask, in order.
    macro_rules! narrowed {
        ($($t:ty: $lanes:literal, $splat:ident, $lt:ident, $le:ident, $gt:ident, $ge:ident, $join:ident;)+) => {$(
            impl Banded for $t {
                fn banded(xs: &[$t], lo: $t, hi: $t, band: Band, out: &mut [MaybeUninit<bool>]) {
                    let (l, h) = ($splat(lo), $splat(hi));
                    with_band!(band, |B| blocks::<$t, $lanes, { 16 / $lanes }>(xs, out, $join, #[inline(always)] |v| match B {
                        Band::Inside => v128_and($le(l, v), $le(v, h)),
                        Band::Outside => v128_not(v128_and($le(l, v), $le(v, h))),
                        Band::Below => $lt(v, l),
                        Band::NotAbove => $le(v, h),
                        Band::Above => $gt(v, h),
                        Band::NotBelow => $ge(v, l),
                    }, #[inline(always)] |v| B.holds(lo, hi, v)))
                }
            }
        )+};
    }
    narrowed! {
        f64: 2, f64x2_splat, f64x2_lt, f64x2_le, f64x2_gt, f64x2_ge, join64;
        f32: 4, f32x4_splat, f32x4_lt, f32x4_le, f32x4_gt, f32x4_ge, join32;
    }

    /// Writes `mask` on each block of 16 items of `xs` into `out` through `join`, and `holds` on each item left over.
    #[inline(always)]
    fn blocks<T: Copy, const LANES: usize, const N: usize>(
        xs: &[T], out: &mut [MaybeUninit<bool>], join: impl Fn([v128; N]) -> v128, mask: impl Fn(v128) -> v128, holds: impl Fn(T) -> bool) {
        let (blocks, rest) = out.as_chunks_mut::<16>();
        let (whole, tail) = xs.as_chunks::<16>();
        for (block, x) in blocks.iter_mut().zip(whole) {
            // SAFETY: each load reads `LANES` of the block's 16 items.
            let masks = std::array::from_fn(|k| mask(unsafe { v128_load(x.as_ptr().add(LANES * k).cast()) }));
            // SAFETY: the store writes the block's 16 slots, each a byte of 0 or 1.
            unsafe { v128_store(block.as_mut_ptr().cast(), v128_and(join(masks), u8x16_splat(1))) }
        }
        for (slot, &x) in rest.iter_mut().zip(tail) { slot.write(holds(x)); }
    }

    #[inline(always)]
    fn join64(m: [v128; 8]) -> v128 {
        let four = |a, b| i8x16_shuffle::<0, 8, 16, 24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0>(a, b);
        let eight = |a, b| i8x16_shuffle::<0, 1, 2, 3, 16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0>(a, b);
        let (low, high) = (eight(four(m[0], m[1]), four(m[2], m[3])), eight(four(m[4], m[5]), four(m[6], m[7])));
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(low, high)
    }

    #[inline(always)]
    fn join32(m: [v128; 4]) -> v128 {
        let eight = |a, b| i8x16_shuffle::<0, 4, 8, 12, 16, 20, 24, 28, 0, 0, 0, 0, 0, 0, 0, 0>(a, b);
        i8x16_shuffle::<0, 1, 2, 3, 4, 5, 6, 7, 16, 17, 18, 19, 20, 21, 22, 23>(eight(m[0], m[1]), eight(m[2], m[3]))
    }
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

/// `Equal`'s slice form on complex items, or with `negate` `NotEqual`'s. Each `CHUNK` of items splits into slices of real and imaginary parts, as in `slice::split`, so `squared_equal` runs over plain slices with no branch and vectorizes. `scaled_equal` then takes the pairs it leaves undecided, one at a time. No result fails.
#[inline(always)]
fn complex_equals<T: TolerantComplex, R: Native>(
    t: T, [x, y]: [Arg<'_, Complex<R>>; 2], n: usize, out: &mut [std::mem::MaybeUninit<bool>], negate: bool) -> bool {
    if pairs_into(x, y, n, out, #[inline(always)] |a, b| {
        let (equal, decided) = t.squared_equal(a, b);
        (equal != negate, decided)
    }) { return true; }
    let out = crate::slice::written(out);
    let at = |a: Arg<'_, Complex<R>>, i: usize| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
    for i in 0..n { let (a, b) = (at(x, i), at(y, i)); if !t.squared_equal(a, b).1 { out[i] = t.scaled_equal(a, b) != negate } }
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

/// [`Tolerant::compare`] at the tolerance it holds, as a function object. NaN orders after every number and equal to NaN, as [`Tolerant::matches`] has NaN match NaN.
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

/// [`Tolerant::residue`], or [`TolerantComplex::complex_residue`] for complex items, at the tolerance it holds, as a function object: `y` modulo `x`, for arguments `x` and `y` in that order. Its `i64` form is [`integer::residue`](crate::integer::residue), which needs no tolerance.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Residue(pub Tolerance);
tolerant!(Residue at real: |t, [x, y]| t.residue(x, y));
tolerant!(Residue at Complex<f32>, Complex<f64>: |t, [x, y]| t.complex_residue(x, y));

/// The floor of `x ÷ y` from [`Tolerant::floor_quotient`], at the tolerance it holds, as a function object. Where the quotient equals a whole number, it gives that number and `Residue` gives 0. A float result never fails, as IEEE division doesn't. Its integer forms, from the `integer` module, divide exactly and fail on a zero divisor or an overflowing quotient.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FloorDivide(pub Tolerance);
tolerant!(FloorDivide at real: |t, [x, y]| Total(t.floor_quotient(x, y).0));

/// [`Tolerant::gcd`], or [`TolerantComplex::complex_gcd`] for complex items, at the tolerance it holds, as a function object. A non-finite argument gives NaN, which marks the item failed. Its `i64` form is [`integer::gcd`](crate::integer::gcd).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Gcd(pub Tolerance);
tolerant!(Gcd at real: |t, [x, y]| t.gcd(x, y));
tolerant!(Gcd at Complex<f32>, Complex<f64>: |t, [x, y]| t.complex_gcd(x, y));

/// The least common multiple x y / gcd(x, y), with the sign of x y, at the tolerance it holds, as a function object. Where the gcd is 0, so is the result. Its `i64` form is [`integer::lcm`](crate::integer::lcm).
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

/// For each item of `y`, the position of the first item of `x` that it matches within tolerance `t`, or the length of `x` where none does. NaN matches NaN. Up to `search::SCANNED` items of `y` each take one pass over `x`, which compares every item with the range of floats equal to it. When every item of both equals a whole number below `whole_limit(t)`, the search indexes those whole numbers (`search::index_of`), then checks that each item of `y` matches the first item of `x` with its whole number. When one doesn't, as two floats on either side of one whole number may not, the search takes the buckets below. Otherwise, with `Exactly`, the search indexes order keys. With a `Tolerance`, it files the items of `x` in buckets of neighbouring floats (`Near`), and compares each item of `y` with the items in the two buckets nearest it. The lookups in the index or the buckets split across threads.
pub fn index_of<F: Float>(t: impl Tolerant, x: &[F], y: &[F]) -> Vec<u32> { let Ok(found) = index_of_with(t, x, y, ok); found }

/// The units of `BETWEEN_CHECKS` needles that each thread looks up in one round of `index_of_with`. A round much shorter than this leaves too little for the other threads after `parallel::split` times its first unit: with 1 unit each, `v⍳w` on a million floats took as long on 4 threads as on one.
const ROUND_UNITS: usize = 16;

/// `index_of`, calling `check` on this thread before each round of `ROUND_UNITS` × `BETWEEN_CHECKS` items of `y` for each of `parallel::threads`, and returning `check`'s error at once. A round's lookups split across threads.
#[inline(never)]
pub fn index_of_with<F: Float, E>(t: impl Tolerant, x: &[F], y: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<u32>, E> {
    if y.len() <= search::SCANNED { return Ok(y.iter().map(|&v| scan(t, x, v) as u32).collect()); }
    if let Some((keys, exact)) = wholes(t, x, whole_limit(t)) && let Some((whole_y, _)) = wholes(t, y, whole_limit(t)) {
        // Two floats near one whole number each equal it, but needn't equal each other. Unless every item of `x` is a whole number, each needle must be identical to its first candidate or within tolerance of it, or the buckets below search instead. Floats near other whole numbers can't match.
        let found = search::index_of(&keys, &whole_y);
        if exact || found.iter().zip(y).all(|(&p, &v)| p as usize == x.len() || x[p as usize] == v || t.matches(x[p as usize], v)) { return Ok(found); }
    }
    if t.value() == 0.0 { return Ok(search::index_of(&order_keys(x), &order_keys(y))); }
    let near = Near::new(t, x, |j, i| x[j].order_key() == x[i].order_key());
    let mut found = Vec::with_capacity(y.len());
    for round in y.chunks(ROUND_UNITS * BETWEEN_CHECKS * crate::parallel::threads()) {
        check()?;
        search::look_up(&mut found, round, |v| {
            let mut first = x.len();
            for i in near.near(v) { if i < first && t.matches(x[i], v) { first = i; } }
            first as u32
        });
    }
    Ok(found)
}

/// The position of the earliest item of each item's class within tolerance `t`. When every item equals a whole number below `whole_limit(t)`, a class holds the items equal to one whole number. Otherwise, with `Exactly`, a class holds equal floats, and with a `Tolerance`, a class is a run that `runs` gives. Two items of one class therefore needn't match each other.
pub fn classes<F: Float>(t: impl Tolerant, x: &[F]) -> Vec<u32> { let Ok(classes) = classes_with(t, x, ok); classes }

/// `classes`, calling `check` after about every `BETWEEN_CHECKS` items, and returning `check`'s error at once.
#[inline(never)]
pub fn classes_with<F: Float, E>(t: impl Tolerant, x: &[F], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<u32>, E> {
    if let Some((keys, _)) = wholes(t, x, whole_limit(t)) { return Ok(search::classes(&keys)); }
    if t.value() == 0.0 { return Ok(search::first_classes(x, F::order_key)); }
    let (mut classes, mut since) = (vec![0; x.len()], 0);
    runs(t, x, |run| {
        since += run.len();
        if since >= BETWEEN_CHECKS {
            since = 0;
            check()?;
        }
        for &i in run { classes[i] = run[0] as u32; }
        Ok(())
    })?;
    Ok(classes)
}

/// Whether each item of `x` is the first of its class, with the classes that `classes` gives.
pub fn firsts<F: Float>(t: impl Tolerant, x: &[F]) -> Vec<bool> { let Ok(mask) = firsts_with(t, x, ok); mask }

/// `firsts`, calling `check` as `classes_with` does, and returning `check`'s error at once.
#[inline(never)]
pub fn firsts_with<F: Float, E>(t: impl Tolerant, x: &[F], check: impl FnMut() -> Result<(), E>) -> Result<Vec<bool>, E> {
    if let Some((keys, _)) = wholes(t, x, whole_limit(t)) { return Ok(search::firsts(&keys)); }
    if t.value() == 0.0 { return Ok(search::first_mask(&search::first_classes(x, F::order_key))); }
    Ok(classes_with(t, x, check)?.into_iter().enumerate().map(|(i, c)| c as usize == i).collect())
}

/// Calls `each` with the positions of each run of `x`, in ascending order of position. A run is a longest sequence of floats, taken in ascending order, in which each matches the next within tolerance `t`. Every float that a float matches lies in one interval (`Tolerant::range`), so a run holds every match of each of its floats. An error from `each` ends the walk, and `runs` returns it. The floats sort by their order keys, which read back as floats in memory order.
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

/// Floats filed in buckets of neighbouring order keys, for tolerant search. Two floats within tolerance `t` of each other lie at most `2t/((1-t)ε)` floats apart, where `ε` is the type's epsilon. A bucket holds a power of 2 of keys, at least four times that. A key in the lower half of its bucket is therefore far enough from the upper edge that every float it matches lies in its own bucket or the neighbour on the side of its half. At f64 with the default tolerance, a bucket holds 512 keys.
pub struct Near<F> { chains: Chains, shift: u32, width: PhantomData<F> }

impl<F: Float> Near<F> {
    /// Files each item of `x` in its bucket, except an item `i` for which `copies(j, i)` holds with an item `j` already in the bucket. A copy can't be a first match, because an earlier item matches whatever it matches.
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
    pub fn near(&self, y: F) -> Links<'_> { let (b, upper) = self.bucket(y); self.chains.chains(b, if upper { b.checked_add(1) } else { b.checked_sub(1) }) }
}

/// The order key of each float, in one pass that vectorises. Exact searches hash these keys, because f32 searches ran about 6% slower when each lookup computed its own key.
pub(crate) fn order_keys<F: Float>(x: &[F]) -> Vec<F::OrderKey> { x.iter().map(|v| v.order_key()).collect() }

/// The magnitude below which each whole number equals no other whole number within tolerance `t`, nor do floats equal to different whole numbers equal each other: 1/(4t), or 2^53 for a tolerance of 0. It is never above 2^53, below which an `f64` holds every whole number.
pub fn whole_limit(t: impl Tolerant) -> f64 { (0.25 / t.value()).min(9_007_199_254_740_992.0) }

/// Each item of `x` as the whole number it equals within tolerance `t`, when every item equals one below `limit` in magnitude, and whether every item is exactly its whole number. A search can then index these integers in place of the floats, with `whole_limit(t)` as the limit. A `limit` of at most 2^63 keeps every whole number within `i64`. Each block of `WHOLES` items is rounded to its nearest whole numbers in one loop that vectorizes. A block with an item that isn't exactly whole then tests each item against the tolerance in a second loop. The first block that fails ends the conversion.
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

/// The position of the first item of `x` that matches `v` within tolerance `t`, or the length of `x`. One pass compares each item with the range of floats equal to `v`. A NaN `v` matches the first NaN.
fn scan<F: Float>(t: impl Tolerant, x: &[F], v: F) -> usize {
    let found = match t.range(v) { Some((lo, hi)) => x.iter().position(|&h| (lo <= h) & (h <= hi)), None => x.iter().position(|h| h.is_nan()) };
    found.unwrap_or(x.len())
}
