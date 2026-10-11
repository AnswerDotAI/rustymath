//! The inverse circle functions. `asin` and `acos` reduce as SLEEF's `xasin_u35` and `xacos_u35` do (see `THIRD-PARTY.md`). `atan`
//! reduces to three intervals, as Cephes' `atan` does. The polynomials are minimax fits.
use crate::float::{horner, Native};

/// The coefficients of (asin s - s)/s³ as a polynomial in s², highest degree first, for |s| up to ½.
const ASIN_64: [f64; 12] = [
    0.03191217485978631,
    -0.01622411038464979,
    0.019528172105463745,
    0.006528010080512205,
    0.012169203417632283,
    0.01388523674904418,
    0.017359704628917373,
    0.02237175805979482,
    0.030381959136550615,
    0.0446428568283393,
    0.07500000000337,
    0.16666666666665408,
];
const ASIN_32: [f64; 5] = [0.04216631501913071, 0.02417951077222824, 0.04547037556767464, 0.07495297491550446, 0.16666752099990845];
/// The coefficients of (atan u - u)/u³ as a polynomial in u², highest degree first, for |u| up to tan(π/8).
const ATAN_64: [f64; 10] = [
    0.02125986000857051,
    -0.043590968108632014,
    0.05692539639740812,
    -0.06641121338420021,
    0.07690068320282498,
    -0.09090782414081072,
    0.1111110665366684,
    -0.14285714195263532,
    0.19999999999088985,
    -0.3333333333333015,
];
const ATAN_32: [f64; 4] = [0.08053722977638245, -0.13877679407596588, 0.19977709650993347, -0.3333294987678528];

/// π/2 as the nearest float and the rest. Halving or doubling both parts gives π/4 or π exactly.
#[inline]
pub(crate) fn half_pi<R: Native>() -> (R, R) {
    let (hi, lo) = if R::WIDE { (1.5707963267948966, 6.123233995736766e-17) } else { (1.5707963705062866, -4.3711390001862426e-08) };
    (R::narrow(hi), R::narrow(lo))
}

/// asin s, for |s| up to ½, with z = s².
#[inline]
fn arcsine<R: Native, const FUSED: bool>(s: R, z: R) -> R {
    let p = if R::WIDE { horner::<R, FUSED, 12>(z, ASIN_64) } else { horner::<R, FUSED, 5>(z, ASIN_32) };
    (s * z).mla::<FUSED>(p, s)
}

/// (s, z) for `arcsine`: x itself when |x| is up to ½, and otherwise √((1 - |x|)/2), whose arcsine is (π/2 - asin |x|)/2.
#[inline]
fn reduce<R: Native>(x: R) -> (R, R, bool) {
    let a = x.abs();
    let big = a > R::narrow(0.5);
    let z = if big { (R::one() - a) * R::narrow(0.5) } else { x * x };
    (if big { z.sqrt() } else { x }, z, big)
}

#[inline]
pub(crate) fn asin<R: Native, const FUSED: bool>(x: R) -> R {
    let (s, z, big) = reduce(x);
    let p = arcsine::<R, FUSED>(s, z);
    let (hi, lo) = half_pi::<R>();
    if big { (hi - (p + p - lo)).copysign(x) } else { p }
}

/// acos x: π/2 - asin x for |x| up to ½, and otherwise 2 asin s, or π less that for a negative x.
#[inline]
pub(crate) fn acos<R: Native, const FUSED: bool>(x: R) -> R {
    let (s, z, big) = reduce(x);
    let p = arcsine::<R, FUSED>(s, z);
    let (hi, lo) = half_pi::<R>();
    if !big { hi - (p - lo) } else if x > R::zero() { p + p } else { (hi + hi) - ((p + p) - (lo + lo)) }
}

/// atan x. |x| reduces to u with |u| up to tan(π/8): u = |x| up to tan(π/8), u = (|x| - 1)/(|x| + 1) up to tan(3π/8), where
/// atan |x| = π/4 + atan u, and u = -1/|x| beyond, where atan |x| = π/2 + atan u. One division serves all three.
#[inline]
pub(crate) fn atan<R: Native, const FUSED: bool>(x: R) -> R {
    let t = x.abs();
    let c = R::narrow;
    let (mid, far) = (t > c(0.41421356237309503), t > c(2.414213562373095));
    let num = if far { -R::one() } else if mid { t - R::one() } else { t };
    let den = if far { t } else if mid { t + R::one() } else { R::one() };
    let u = num / den;
    let z = u * u;
    let p = if R::WIDE { horner::<R, FUSED, 10>(z, ATAN_64) } else { horner::<R, FUSED, 4>(z, ATAN_32) };
    let (hi, lo) = half_pi::<R>();
    let (hi, lo) = if far { (hi, lo) } else if mid { (hi * c(0.5), lo * c(0.5)) } else { (R::zero(), R::zero()) };
    (hi + ((u * z).mla::<FUSED>(p, u) + lo)).copysign(x)
}

/// The angle of the point (x, y) from the positive x axis, from -π to π, as atan(y/x) in the point's quadrant. The ratio it takes the
/// arctangent of is the smaller magnitude over the larger, which never overflows. Zeros and infinities give C99's angles: atan2(±0, -0)
/// is ±π, and atan2(±∞, ±∞) is an odd multiple of ±π/4.
#[inline]
pub(crate) fn atan2<R: Native, const FUSED: bool>(y: R, x: R) -> R {
    let (ax, ay) = (x.abs(), y.abs());
    let steep = ay > ax;
    let (num, den) = if steep { (ax, ay) } else { (ay, ax) };
    // The ratio is 0 at zeros and 1 at two infinities, where the quotient is NaN.
    let q = if den == R::zero() { R::zero() } else if ax.is_infinite() & ay.is_infinite() { R::one() } else { num / den };
    let a = atan::<R, FUSED>(q);
    let (hi, lo) = half_pi::<R>();
    let a = if steep { hi - (a - lo) } else { a };
    let a = if x.is_sign_negative() { (hi + hi) - (a - (lo + lo)) } else { a };
    a.copysign(y)
}
