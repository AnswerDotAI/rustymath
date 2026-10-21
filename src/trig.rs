//! Sine, cosine and tangent. The sine polynomials come from Arm Optimized Routines' AdvSIMD `sin` and `sinf`, and the tangent's half-angle form from its AdvSIMD `tan`. The tangent polynomials are minimax fits. See `THIRD-PARTY.md`.
use crate::float::{Native, horner, sealed::Core};
use std::f64::consts::FRAC_1_PI;

/// Which function a reduction serves. Sine and cosine reduce by multiples of π, the cosine's offset by ½. Tangent reduces by multiples of π/2.
#[derive(Clone, Copy, PartialEq)]
enum Kind { Sin, Cos, Tan }

impl Kind {
    /// The factor that turns x into multiples of the period: 1/π, or 2/π for tangent.
    #[inline]
    fn scale(self) -> f64 { if self == Kind::Tan { 2.0 * FRAC_1_PI } else { FRAC_1_PI } }
    /// h, the offset that the nearest whole number n is taken after.
    #[inline]
    fn offset(self) -> f64 { if self == Kind::Cos { 0.5 } else { 0.0 } }
}

#[inline]
pub(crate) fn sin<R: Native, const FUSED: bool>(x: R) -> R { if within(x) { fast::<R, FUSED>(x, Kind::Sin) } else { slow::<R, FUSED>(x, Kind::Sin) } }

#[inline]
pub(crate) fn cos<R: Native, const FUSED: bool>(x: R) -> R { if within(x) { fast::<R, FUSED>(x, Kind::Cos) } else { slow::<R, FUSED>(x, Kind::Cos) } }

#[inline]
pub(crate) fn tan<R: Native, const FUSED: bool>(x: R) -> R { if within(x) { fast::<R, FUSED>(x, Kind::Tan) } else { slow::<R, FUSED>(x, Kind::Tan) } }

/// Writes the function objects of sine, cosine and tangent. Each slice form gives every item the fast reduction, and the rare items beyond `limit` the scalar function, with its exact reduction, in a second pass. A closure's type comes from where it's written. Closures written once in a shared function would give all three kinds one loop that tests the kind on every item. Each object writes its own closures, so that its loop has its kind fixed. The objects never fail: ±∞ give NaN, which is each function's value there, as NaN is the value of ∞ - ∞.
macro_rules! objects {
    ($($object:ident = $name:ident: $kind:expr;)+) => {$(
        #[doc = concat!("[`", stringify!($name), "`](crate::", stringify!($name), ") as a function object. Its slice form `many` gives every")]
        /// item the fast reduction, and only the rare large ones the exact reduction.
        #[derive(Clone, Copy, Debug)]
        pub struct $object;
        forms!($object at real: |_, [x]| crate::slice::Total($name::<_, { crate::FUSED }>(x)),
            many |_, [x], n, out| { two_pass!([x], n, out, |[a]| within(a).then_some(fast::<_, { crate::FUSED }>(a, $kind)), |[a]| $name::<_, { crate::FUSED }>(a)); true });
    )+};
}
objects!(Sin = sin: Kind::Sin; Cos = cos: Kind::Cos; Tan = tan: Kind::Tan;);

/// The magnitude up to which `fast` holds: 2^23 at f64 and 2^20 at f32.
#[inline]
fn limit<R: Native>() -> R { R::narrow(if R::WIDE { 8388608.0 } else { 1048576.0 }) }

/// Whether `fast` holds for x: a magnitude up to `limit`, or ±∞ or NaN, for which it gives NaN.
#[inline]
fn within<R: Native>(x: R) -> bool { let a = x.abs(); (a <= limit()) | !(a < R::infinity()) }

/// The function `kind` names at x, for x that `within` accepts. n is the nearest whole number to x × `scale` + h, and r is x less (n - h) periods. sin x and cos x are then (-1)^n sin r, with |r| up to about π/2, and tan x is tan r or -1/tan r as n is even or odd, with |r| up to about π/4.
#[inline]
fn fast<R: Native, const FUSED: bool>(x: R, kind: Kind) -> R {
    let (r, z) = if FUSED { reduce_fused(x, kind) } else { reduce_split(x, kind) };
    if kind == Kind::Tan { return tangent::<R, FUSED>(r, z.odd()); }
    let y = sine::<R, FUSED>(r).flip(z);
    // At x = ±0, r for cosine is the float nearest π/2, where the polynomial gives the float below 1, and cos(±0) must be exactly 1. Sine gives x itself there. Without this select on x, wasm runs the sine loop about 1.8 times slower.
    if x == R::zero() { if kind == Kind::Cos { R::one() } else { x } } else { y }
}

/// The function `kind` names at a finite x beyond `limit`, through the exact reduction.
#[inline(never)]
fn slow<R: Native, const FUSED: bool>(x: R, kind: Kind) -> R {
    let wide = x.wide();
    let (r, odd) = exact(wide, kind);
    let y = if kind == Kind::Tan { tangent::<f64, FUSED>(r, odd) } else { sine::<f64, FUSED>(r) };
    // The reduction works on |x|. Sine and tangent are odd, and the parity of n flips the sign of sine and cosine.
    let negate = if kind == Kind::Tan { wide < 0.0 } else { odd ^ (kind == Kind::Sin && wide < 0.0) };
    R::narrow(if negate { -y } else { y })
}

/// π in three positive parts at full precision. Inside a fused multiply-add, each product with a part is exact. Positive parts keep the sign of x = ±0 through the reduction.
const PI_FUSED_64: [f64; 3] = [3.141592653589793, 1.224646799147353e-16, 2.165713347843828e-32];
const PI_FUSED_32: [f64; 3] = [3.141592502593994, 1.5099578831723193e-07, 1.0780605906948477e-14];

/// z = SHIFT + n, from x × `scale` + h.
#[inline]
fn whole<R: Native, const FUSED: bool>(x: R, kind: Kind) -> R {
    let c = R::narrow;
    if kind.offset() == 0.0 { x.mla::<FUSED>(c(kind.scale()), R::SHIFT) } else { x.mla::<FUSED>(c(kind.scale()), c(kind.offset())) + R::SHIFT }
}

/// The multiple of π that the reduction subtracts: n - h, or n/2 for tangent. Both are exact.
#[inline]
fn multiple<R: Native>(z: R, kind: Kind) -> R { let n = z - R::SHIFT; if kind == Kind::Tan { n * R::narrow(0.5) } else { n - R::narrow(kind.offset()) } }

/// r and z = SHIFT + n for `fast`, with fused multiply-adds at x's own width.
#[inline]
fn reduce_fused<R: Native>(x: R, kind: Kind) -> (R, R) {
    let z = whole::<R, true>(x, kind);
    (minus_multiple::<R, true, 3>(x, multiple(z, kind), if R::WIDE { PI_FUSED_64 } else { PI_FUSED_32 }), z)
}

/// π in parts. Each part but the last has 30 significant bits, so its product with a whole number or a half-integer below 2^22, as every multiple up to `limit` is, is exact without FMA. f64 needs four parts and f32 two.
const PI_SPLIT_64: [f64; 4] = [3.141592651605606, 1.9841871583270443e-09, 1.03403659577805e-18, 5.807711947958721e-28];
const PI_SPLIT_32: [f64; 2] = [3.141592651605606, 1.984187159361081e-09];

/// r and z = SHIFT + n for `fast`, without FMA. The reduction runs in f64 at both widths, because f32 parts with enough trailing zeros would hold too few bits of π.
#[inline]
fn reduce_split<R: Native>(x: R, kind: Kind) -> (R, R) {
    let wide = x.wide();
    let z = whole::<f64, false>(wide, kind);
    let m = multiple(z, kind);
    let r = if R::WIDE { minus_multiple::<f64, false, 4>(wide, m, PI_SPLIT_64) } else { minus_multiple::<f64, false, 2>(wide, m, PI_SPLIT_32) };
    // At f32, z moves to f32's shift, whose lowest bit still holds n's parity.
    (R::narrow(r), if R::WIDE { R::narrow(z) } else { R::narrow(z - f64::SHIFT) + R::SHIFT })
}

/// x - mπ, with π in `parts`.
#[inline]
fn minus_multiple<R: Native, const FUSED: bool, const N: usize>(x: R, m: R, parts: [f64; N]) -> R {
    let mut r = x;
    for &p in &parts { r = m.mla::<FUSED>(R::narrow(-p), r); }
    r
}

/// The coefficients of (sin r - r)/r³ as a polynomial in r², highest degree first, for |r| up to π/2.
const SIN_64: [f64; 7] = [
    -7.364473665132213e-13, 1.604733546032205e-10, -2.505185490163253e-08, 2.755731618671281e-06, -0.0001984126982169595, 0.008333333333273485,
    -0.1666666666666606,
];
const SIN_32: [f64; 4] = [2.6105958568223286e-06, -0.00019811994570773095, 0.00833310279995203, -0.16666661202907562];

/// sin r, for |r| up to π/2. r + r³ P(r²) takes the sign of r, so that sin(-0) is -0.
#[inline]
fn sine<R: Native, const FUSED: bool>(r: R) -> R {
    let r2 = r * r;
    let p = if R::WIDE { horner::<R, FUSED, 7>(r2, SIN_64) } else { horner::<R, FUSED, 4>(r2, SIN_32) };
    (r * r2).mla::<FUSED>(p, r).copysign(r)
}

/// The coefficients of (tan u - u)/u³ as a polynomial in u², highest degree first, for |u| up to π/8.
const TAN_64: [f64; 9] = [
    0.0003287231044337369, 0.000559125460707824, 0.0014615495400569908, 0.0035914987665774814, 0.008863277952340431, 0.021869486815122453, 0.05396825400765754,
    0.13333333333289177, 0.33333333333333504,
];
const TAN_32: [f64; 4] = [0.025565408170223236, 0.053473446518182755, 0.13335898518562317, 0.33333292603492737];

/// tan r, or -1/tan r when `odd`, for |r| up to about π/4. With t = tan(r/2), tan r = 2t/(1 - t²) and -1/tan r = (t² - 1)/(2t), so one division gives either.
#[inline]
fn tangent<R: Native, const FUSED: bool>(r: R, odd: bool) -> R {
    let u = r * R::narrow(0.5);
    let u2 = u * u;
    let p = if R::WIDE { horner::<R, FUSED, 9>(u2, TAN_64) } else { horner::<R, FUSED, 4>(u2, TAN_32) };
    let t = (u * u2).mla::<FUSED>(p, u);
    let (below, twice) = (t.mla::<FUSED>(t, -R::one()), t + t);
    if odd { below / twice } else { -twice / below }
}

/// 1/(2π), downwards, as 1280 bits after the binary point, most significant first.
const INV_2PI: [u64; 20] = [
    0x28be60db9391054a, 0x7f09d5f47d4d3770, 0x36d8a5664f10e410, 0x7f9458eaf7aef158, 0x6dc91b8e909374b8, 0x01924bba82746487, 0x3f877ac72c4a69cf,
    0xba208d7d4baed121, 0x3a671c09ad17df90, 0x4e64758e60d4ce7d, 0x272117e2ef7e4a0e, 0xc7fe25fff7816603, 0xfbcbc462d6829b47, 0xdb4d9fb3c9f2c26d,
    0xd3d18fd9a797fa8b, 0x5d49eeb1faf97c5e, 0xcf41ce7de294a4ba, 0x9afed7ec47e35742, 0x1580cc11bf1edaea, 0xfc33ef0826bd0d87,
];
/// π × 2^126, rounded.
const PI_FIXED: u128 = 0xc90fdaa22168c234c4c6628b80dc1cd1;

/// r and whether n is odd, for the function `kind` names at a finite x beyond `limit`, by Payne and Hanek's reduction of |x|. Integer arithmetic on the bits of 1/(2π) gives the fraction of |x|/(2π) to 128 bits, so even an x close to a multiple of π reduces accurately.
fn exact(x: f64, kind: Kind) -> (f64, bool) {
    let bits = x.abs().to_bits();
    let (m, k) = ((bits & ((1 << 52) - 1)) | 1 << 52, (bits >> 52) as i64 - 1075);
    // |x| = m × 2^k. The bits of 1/(2π) whose products with |x| are whole numbers drop out: the first k bits. The next 192 bits give the fraction to well beyond 128 bits.
    let skip = k.max(0) as usize;
    let word = |i: usize| ((((INV_2PI[i] as u128) << 64 | INV_2PI[i + 1] as u128) << (skip % 64)) >> 64) as u64;
    let w = skip / 64;
    let low = m as u128 * word(w + 2) as u128;
    let mid = m as u128 * word(w + 1) as u128 + (low >> 64);
    let high = m as u128 * word(w) as u128 + (mid >> 64);
    // The product's bits from 2^64 up hold the fraction when k is at least 0, shifted up by -k bits when k is negative.
    let below = (-k).max(0) as u32;
    let fraction = (((high as u64 as u128) << 64 | mid as u64 as u128) >> below) | (high >> 64).checked_shl(128 - below).unwrap_or(0);
    // As a fixed-point number with 127 bits after the point, the fraction is z = |x|/π, modulo 2. Tangent's period is π/2, so its z doubles. Adding h + ½ and keeping the bit above the point gives the parity of n, the nearest whole number to z + h. d = z + h - n lies from -½ to ½.
    let z = if kind == Kind::Tan { fraction << 1 } else { fraction };
    let t = z.wrapping_add(if kind == Kind::Cos { 1 << 126 } else { 0 }).wrapping_add(1 << 126);
    let d = (t & ((1 << 127) - 1)) as i128 - (1 << 126);
    // r is d periods, from the high half of |d| × π·2^126, which has 125 bits after the point. Tangent's period is half of π.
    let (a, b) = (d.unsigned_abs(), PI_FIXED);
    let product = (a >> 64) * (b >> 64) + ((a >> 64) * (b as u64 as u128) >> 64) + ((a as u64 as u128) * (b >> 64) >> 64);
    let scale = if kind == Kind::Tan { 2f64.powi(-126) } else { 2f64.powi(-125) };
    ((product as f64 * scale).copysign(d as f64), t >> 127 == 1)
}
