//! Shared short trigonometric reductions and polynomials, with scalar repairs for rare large arguments. Polynomial sources: Arm Optimized Routines; see THIRD-PARTY.md.
use crate::{Float, buffer::FloatWidth, facts::Repair, float::Native, recipe::{NativeOps, Ops}};
use std::f64::consts::FRAC_1_PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind { Sin, Cos, Tan }

impl Kind {
    #[inline]
    fn scale(self) -> f64 { if self == Self::Tan { 2.0 * FRAC_1_PI } else { FRAC_1_PI } }
    #[inline]
    fn offset(self) -> f64 { if self == Self::Cos { 0.5 } else { 0.0 } }
    #[inline]
    pub(crate) fn predicate<F: Float, O: Ops<F> + ?Sized>(self, o: &O, x: O::V) -> O::B {
        let a = o.magnitude(x);
        let limit = o.number(F::narrow(if o.width() == FloatWidth::F64 { 8388608.0 } else { 1048576.0 }));
        o.and(o.less(limit, a), o.less(a, o.number(F::infinity())))
    }
    pub(crate) fn routine(self) -> Repair {
        match self {
            Self::Sin => Repair { single: sin_single, double: sin_double }, Self::Cos => Repair { single: cos_single, double: cos_double },
            Self::Tan => Repair { single: tan_single, double: tan_double }
        }
    }
}

extern "C" fn sin_single(x: f32) -> f32 { repair::<{ crate::FUSED }>(x as f64, Kind::Sin) as f32 }
extern "C" fn sin_double(x: f64) -> f64 { repair::<{ crate::FUSED }>(x, Kind::Sin) }
extern "C" fn cos_single(x: f32) -> f32 { repair::<{ crate::FUSED }>(x as f64, Kind::Cos) as f32 }
extern "C" fn cos_double(x: f64) -> f64 { repair::<{ crate::FUSED }>(x, Kind::Cos) }
extern "C" fn tan_single(x: f32) -> f32 { repair::<{ crate::FUSED }>(x as f64, Kind::Tan) as f32 }
extern "C" fn tan_double(x: f64) -> f64 { repair::<{ crate::FUSED }>(x, Kind::Tan) }

#[derive(Clone, Copy, Debug)]
pub struct Sin;
#[derive(Clone, Copy, Debug)]
pub struct Cos;
#[derive(Clone, Copy, Debug)]
pub struct Tan;

#[inline]
fn scalar<R: Native, const FUSED: bool>(x: R, kind: Kind) -> R {
    let o = NativeOps::<FUSED>::DEFAULT;
    if kind.predicate(&o, x) { R::narrow(repair::<FUSED>(x.wide(), kind)) } else { formula(&o, x, kind) }
}
#[inline]
pub(crate) fn sin<R: Native, const FUSED: bool>(x: R) -> R { scalar::<R, FUSED>(x, Kind::Sin) }
#[inline]
pub(crate) fn cos<R: Native, const FUSED: bool>(x: R) -> R { scalar::<R, FUSED>(x, Kind::Cos) }
#[inline]
pub(crate) fn tan<R: Native, const FUSED: bool>(x: R) -> R { scalar::<R, FUSED>(x, Kind::Tan) }

#[inline]
pub(crate) fn formula<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, kind: Kind) -> O::V {
    let zero = o.number(F::zero());
    let finite = o.less(o.magnitude(x), o.number(F::infinity()));
    let safe = o.select(o.and(finite, o.not(kind.predicate(o, x))), x, zero);
    let (r, n) = if o.fused() { reduce_fused(o, safe, kind) } else { reduce_split(o, safe, kind) };
    // n is a whole number well inside the shift's exact range, so rounding n/2 with the shift gives its parity exactly. An integer conversion would run scalar on AVX2.
    let (half, shift) = (o.multiply(n, o.number(F::narrow(0.5))), crate::recipe::rounding_shift(o));
    let even = o.subtract(o.add(half, shift), shift);
    let odd = o.not(o.equal(n, o.add(even, even)));
    let y = if kind == Kind::Tan { tangent(o, r, odd) } else {
        let y = sine(o, r);
        let y = o.select(odd, o.negate(y), y);
        o.select(o.equal(x, zero), if kind == Kind::Cos { o.number(F::one()) } else { x }, y)
    };
    o.select(finite, y, o.number(F::nan()))
}

const PI_FUSED_64: [f64; 3] = [3.141592653589793, 1.224646799147353e-16, 2.165713347843828e-32];
const PI_FUSED_32: [f64; 3] = [3.141592502593994, 1.5099578831723193e-7, 1.0780605906948477e-14];
const PI_SPLIT_64: [f64; 4] = [3.141592651605606, 1.9841871583270443e-9, 1.03403659577805e-18, 5.807711947958721e-28];
const PI_SPLIT_32: [f64; 2] = [3.141592651605606, 1.984187159361081e-9];

#[inline]
fn reduce_fused<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, kind: Kind) -> (O::V, O::V) {
    let shift = crate::recipe::rounding_shift(o);
    let z = if kind.offset() == 0.0 { o.multiply_add(x, o.number(F::narrow(kind.scale())), shift) }
    else { o.add(o.multiply_add(x, o.number(F::narrow(kind.scale())), o.number(F::narrow(kind.offset()))), shift) };
    let n = o.subtract(z, shift);
    let m = if kind == Kind::Tan { o.multiply(n, o.number(F::narrow(0.5))) } else { o.subtract(n, o.number(F::narrow(kind.offset()))) };
    let mut r = x;
    for p in if o.width() == FloatWidth::F64 { PI_FUSED_64 } else { PI_FUSED_32 } { r = o.multiply_add(m, o.number(F::narrow(-p)), r); }
    (r, n)
}

#[inline]
fn reduce_split<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, kind: Kind) -> (O::V, O::V) {
    let wide = o.widen(x);
    let z = o.wide_multiply_add(wide, o.wide_number(kind.scale()), o.wide_number(kind.offset()));
    let z = o.wide_multiply_add(z, o.wide_number(1.0), o.wide_number(6755399441055744.0));
    let n = o.wide_multiply_add(z, o.wide_number(1.0), o.wide_number(-6755399441055744.0));
    let m = if kind == Kind::Tan { o.wide_multiply_add(n, o.wide_number(0.5), o.wide_number(0.0)) }
    else { o.wide_multiply_add(n, o.wide_number(1.0), o.wide_number(-kind.offset())) };
    let mut r = wide;
    let parts: &[f64] = if o.width() == FloatWidth::F64 { &PI_SPLIT_64 } else { &PI_SPLIT_32 };
    for &p in parts { r = o.wide_multiply_add(m, o.wide_number(-p), r); }
    (o.narrow(r), o.narrow(n))
}

const SIN_64: [f64; 7] = [
    -7.364473665132213e-13, 1.604733546032205e-10, -2.505185490163253e-8, 2.755731618671281e-6, -0.0001984126982169595, 0.008333333333273485,
    -0.1666666666666606,
];
const SIN_32: [f64; 4] = [2.6105958568223286e-6, -0.00019811994570773095, 0.00833310279995203, -0.16666661202907562];
#[inline]
fn sine<F: Float, O: Ops<F> + ?Sized>(o: &O, r: O::V) -> O::V {
    let r2 = o.multiply(r, r);
    let p = crate::recipe::horner(o, r2, if o.width() == FloatWidth::F64 { &SIN_64 } else { &SIN_32 });
    o.multiply_add(o.multiply(r, r2), p, r)
}

const TAN_64: [f64; 9] = [
    0.0003287231044337369, 0.000559125460707824, 0.0014615495400569908, 0.0035914987665774814, 0.008863277952340431, 0.021869486815122453, 0.05396825400765754,
    0.13333333333289177, 0.33333333333333504,
];
const TAN_32: [f64; 4] = [0.025565408170223236, 0.053473446518182755, 0.13335898518562317, 0.33333292603492737];
#[inline]
fn tangent<F: Float, O: Ops<F> + ?Sized>(o: &O, r: O::V, odd: O::B) -> O::V {
    let u = o.multiply(r, o.number(F::narrow(0.5)));
    let u2 = o.multiply(u, u);
    let p = crate::recipe::horner(o, u2, if o.width() == FloatWidth::F64 { &TAN_64 } else { &TAN_32 });
    let t = o.multiply_add(o.multiply(u, u2), p, u);
    let below = o.multiply_add(t, t, o.number(-F::one()));
    let twice = o.add(t, t);
    o.divide(o.select(odd, below, o.negate(twice)), o.select(odd, twice, below))
}

#[inline(never)]
fn repair<const FUSED: bool>(x: f64, kind: Kind) -> f64 {
    let (r, odd) = exact(x, kind);
    let o = NativeOps::<FUSED>::DEFAULT;
    let y = if kind == Kind::Tan { tangent(&o, r, odd) } else { sine(&o, r) };
    let negate = if kind == Kind::Tan { x < 0.0 } else { odd ^ (kind == Kind::Sin && x < 0.0) };
    if negate { -y } else { y }
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
