//! Sine and cosine. The polynomials come from Arm Optimized Routines' AdvSIMD `sin` and `sinf`. See `THIRD-PARTY.md`.
use crate::float::{horner, sealed::Core, Float, Native, FUSED};
use std::f64::consts::FRAC_1_PI;

/// sin x, within 4 ulp at f64, 2 ulp at f32 and 1 ulp at f16. `sin(±0)` is ±0, and `sin(±∞)` and `sin(NaN)` are NaN. Beyond a
/// magnitude of 2^23 at f64 or 2^20 at f32, x takes an exact reduction, which is slower and has a branch. A loop that calls `sin`
/// therefore doesn't vectorize. A loop that calls `checked_sin` does.
#[inline]
pub fn sin<F: Float>(x: F) -> F { F::from_total(sin_at::<_, FUSED>(x.total())) }

/// cos x, within 4 ulp at f64 and f32, and 1 ulp at f16. `cos(±0)` is 1, and `cos(±∞)` and `cos(NaN)` are NaN. Beyond the magnitude
/// that `sin` gives, x takes the same exact reduction.
#[inline]
pub fn cos<F: Float>(x: F) -> F { F::from_total(cos_at::<_, FUSED>(x.total())) }

/// sin x, as `sin` gives it, with no branch: `None` for a finite x beyond the magnitude where `sin` takes its exact reduction. A loop
/// that calls it vectorizes.
///
/// ```
/// assert_eq!(rustymath::checked_sin(1e6f64), Some(rustymath::sin(1e6)));
/// assert_eq!(rustymath::checked_sin(1e6f32), Some(rustymath::sin(1e6)));
/// assert_eq!(rustymath::checked_sin(2e6f32), None);
/// assert_eq!(rustymath::checked_sin(1e8f64), None);
/// assert!(rustymath::checked_sin(f64::INFINITY).unwrap().is_nan());
/// ```
#[inline]
pub fn checked_sin<F: Float>(x: F) -> Option<F> { checked(x, false) }

/// cos x, as `cos` gives it, with no branch: `None` where `checked_sin` gives `None`. A loop that calls it vectorizes.
#[inline]
pub fn checked_cos<F: Float>(x: F) -> Option<F> { checked(x, true) }

#[inline]
fn checked<F: Float>(x: F, cos: bool) -> Option<F> {
    let x = x.total();
    within(x).then_some(F::from_total(fast::<_, FUSED>(x, cos)))
}

#[inline]
pub(crate) fn sin_at<R: Native, const FUSED: bool>(x: R) -> R { trig::<R, FUSED>(x, false) }
#[inline]
pub(crate) fn cos_at<R: Native, const FUSED: bool>(x: R) -> R { trig::<R, FUSED>(x, true) }

/// sin x, or cos x when `cos`.
#[inline]
fn trig<R: Native, const FUSED: bool>(x: R, cos: bool) -> R {
    if within(x) { fast::<R, FUSED>(x, cos) } else { R::narrow(exact::<FUSED>(x.wide(), cos)) }
}

/// The magnitude up to which `fast` holds: 2^23 at f64 and 2^20 at f32.
#[inline]
fn limit<R: Native>() -> R { R::narrow(if R::WIDE { 8388608.0 } else { 1048576.0 }) }

/// Whether `fast` holds for x: a magnitude up to `limit`, or ±∞ or NaN, for which it gives NaN.
#[inline]
fn within<R: Native>(x: R) -> bool {
    let a = x.abs();
    (a <= limit()) | !(a < R::infinity())
}

/// sin x, or cos x when `cos`, for x that `within` accepts. With h = ½ for cos and 0 for sin, n is the nearest whole number to x/π + h,
/// and r = x - (n - h)π lies within about π/2. Then sin x or cos x is (-1)^n sin r.
#[inline]
fn fast<R: Native, const FUSED: bool>(x: R, cos: bool) -> R {
    let (r, z) = if FUSED { reduce_fused(x, cos) } else { reduce_split(x, cos) };
    let y = sine::<R, FUSED>(r).flip(z);
    // At x = ±0, r is the float nearest π/2, where the polynomial gives the float below 1. cos(±0) must be exactly 1.
    if cos && x == R::zero() { R::one() } else { y }
}

/// π in three positive parts at full precision. Inside a fused multiply-add, each product with a part is exact. Positive parts keep the
/// sign of x = ±0 through the reduction.
const PI_FUSED_64: [f64; 3] = [3.141592653589793, 1.224646799147353e-16, 2.165713347843828e-32];
const PI_FUSED_32: [f64; 3] = [3.141592502593994, 1.5099578831723193e-07, 1.0780605906948477e-14];

/// r and z = SHIFT + n for `fast`, with fused multiply-adds at x's own width.
#[inline]
fn reduce_fused<R: Native>(x: R, cos: bool) -> (R, R) {
    let c = R::narrow;
    let z = if cos { x.mla::<true>(c(FRAC_1_PI), c(0.5)) + R::SHIFT } else { x.mla::<true>(c(FRAC_1_PI), R::SHIFT) };
    let n = z - R::SHIFT - if cos { c(0.5) } else { R::zero() };
    (minus_multiple::<R, true, 3>(x, n, if R::WIDE { PI_FUSED_64 } else { PI_FUSED_32 }), z)
}

/// π in parts. Each part but the last has 30 significant bits, so its product with a whole number or a half-integer below 2^22, as
/// every n up to `limit` is, is exact without FMA. f64 needs four parts and f32 two.
const PI_SPLIT_64: [f64; 4] = [3.141592651605606, 1.9841871583270443e-09, 1.03403659577805e-18, 5.807711947958721e-28];
const PI_SPLIT_32: [f64; 2] = [3.141592651605606, 1.984187159361081e-09];

/// r and z = SHIFT + n for `fast`, without FMA. The reduction runs in f64 at both widths, because f32 parts with enough trailing zeros
/// would hold too few bits of π.
#[inline]
fn reduce_split<R: Native>(x: R, cos: bool) -> (R, R) {
    let wide = x.wide();
    let z = if cos { wide.mla::<false>(FRAC_1_PI, 0.5) + f64::SHIFT } else { wide.mla::<false>(FRAC_1_PI, f64::SHIFT) };
    let n = z - f64::SHIFT - if cos { 0.5 } else { 0.0 };
    let r = if R::WIDE { minus_multiple::<f64, false, 4>(wide, n, PI_SPLIT_64) } else { minus_multiple::<f64, false, 2>(wide, n, PI_SPLIT_32) };
    // At f32, z moves to f32's shift, whose lowest bit still holds n's parity.
    (R::narrow(r), if R::WIDE { R::narrow(z) } else { R::narrow(z - f64::SHIFT) + R::SHIFT })
}

/// x - nπ, with π in `parts`.
#[inline]
fn minus_multiple<R: Native, const FUSED: bool, const N: usize>(x: R, n: R, parts: [f64; N]) -> R {
    let mut r = x;
    for &p in &parts { r = n.mla::<FUSED>(R::narrow(-p), r); }
    r
}

/// The coefficients of (sin r - r)/r³ as a polynomial in r², highest degree first, for |r| up to π/2.
const POLY_64: [f64; 7] = [
    -7.364473665132213e-13,
    1.604733546032205e-10,
    -2.505185490163253e-08,
    2.755731618671281e-06,
    -0.0001984126982169595,
    0.008333333333273485,
    -0.1666666666666606,
];
const POLY_32: [f64; 4] = [2.6105958568223286e-06, -0.00019811994570773095, 0.00833310279995203, -0.16666661202907562];

/// sin r, for |r| up to π/2. r + r³ P(r²) takes the sign of r, so that sin(-0) is -0.
#[inline]
fn sine<R: Native, const FUSED: bool>(r: R) -> R {
    let r2 = r * r;
    let p = if R::WIDE { horner::<R, FUSED, 7>(r2, POLY_64) } else { horner::<R, FUSED, 4>(r2, POLY_32) };
    (r * r2).mla::<FUSED>(p, r).copysign(r)
}

/// 1/(2π), downwards, as 1280 bits after the binary point, most significant first.
const INV_2PI: [u64; 20] = [
    0x28be60db9391054a,
    0x7f09d5f47d4d3770,
    0x36d8a5664f10e410,
    0x7f9458eaf7aef158,
    0x6dc91b8e909374b8,
    0x01924bba82746487,
    0x3f877ac72c4a69cf,
    0xba208d7d4baed121,
    0x3a671c09ad17df90,
    0x4e64758e60d4ce7d,
    0x272117e2ef7e4a0e,
    0xc7fe25fff7816603,
    0xfbcbc462d6829b47,
    0xdb4d9fb3c9f2c26d,
    0xd3d18fd9a797fa8b,
    0x5d49eeb1faf97c5e,
    0xcf41ce7de294a4ba,
    0x9afed7ec47e35742,
    0x1580cc11bf1edaea,
    0xfc33ef0826bd0d87,
];
/// π × 2^126, rounded.
const PI_FIXED: u128 = 0xc90fdaa22168c234c4c6628b80dc1cd1;

/// sin x, or cos x when `cos`, for a finite x beyond `limit` in magnitude, by Payne and Hanek's reduction. Integer arithmetic on the
/// bits of 1/(2π) gives the fraction of |x|/(2π) to 128 bits, so even an x close to a multiple of π reduces accurately.
#[inline(never)]
fn exact<const FUSED: bool>(x: f64, cos: bool) -> f64 {
    let bits = x.abs().to_bits();
    let (m, k) = ((bits & ((1 << 52) - 1)) | 1 << 52, (bits >> 52) as i64 - 1075);
    // |x| = m × 2^k. The bits of 1/(2π) whose products with |x| are whole numbers drop out: the first k bits. The next 192 bits give
    // the fraction to well beyond 128 bits.
    let skip = k.max(0) as usize;
    let word = |i: usize| ((((INV_2PI[i] as u128) << 64 | INV_2PI[i + 1] as u128) << (skip % 64)) >> 64) as u64;
    let w = skip / 64;
    let low = m as u128 * word(w + 2) as u128;
    let mid = m as u128 * word(w + 1) as u128 + (low >> 64);
    let high = m as u128 * word(w) as u128 + (mid >> 64);
    // The product's bits from 2^64 up hold the fraction when k is at least 0, shifted up by -k bits when k is negative.
    let below = (-k).max(0) as u32;
    let fraction = (((high as u64 as u128) << 64 | mid as u64 as u128) >> below) | (high >> 64).checked_shl(128 - below).unwrap_or(0);
    // As a fixed-point number with 127 bits after the point, the fraction is z = |x|/π, modulo 2. Adding h + ½ and keeping the bit
    // above the point gives the parity of n, the nearest whole number to z + h. d = z + h - n lies from -½ to ½.
    let t = fraction.wrapping_add(if cos { 1 << 126 } else { 0 }).wrapping_add(1 << 126);
    let d = (t & ((1 << 127) - 1)) as i128 - (1 << 126);
    // r = dπ, from the high half of |d| × π·2^126, which has 125 bits after the point.
    let (a, b) = (d.unsigned_abs(), PI_FIXED);
    let product = (a >> 64) * (b >> 64) + ((a >> 64) * (b as u64 as u128) >> 64) + ((a as u64 as u128) * (b >> 64) >> 64);
    let r = (product as f64 * 2f64.powi(-125)).copysign(d as f64);
    let y = sine::<f64, FUSED>(r);
    if (t >> 127 == 1) ^ (!cos && x < 0.0) { -y } else { y }
}
