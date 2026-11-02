//! Shared short trigonometric reductions and polynomials, with scalar repairs for rare large arguments. Polynomial sources: Arm Optimized Routines; see THIRD-PARTY.md.
use crate::{Float, float::{Native, sealed::Core}, recipe::{Ops, Integer, Cmp}, buffer::FloatWidth, facts::Repair};
use std::f64::consts::FRAC_1_PI;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) enum Kind { Sin, Cos, Tan }

impl Kind {
    fn scale(self) -> f64 { if self == Self::Tan { 2.0 * FRAC_1_PI } else { FRAC_1_PI } }
    fn offset(self) -> f64 { if self == Self::Cos { 0.5 } else { 0.0 } }
    pub(crate) fn predicate<F: Float, O: Ops<F> + ?Sized>(self, o: &O, x: O::V) -> O::B {
        let a = o.magnitude(x);
        let limit = o.number(F::narrow(if o.width() == FloatWidth::F64 { 8388608.0 } else { 1048576.0 }));
        o.and(o.less(limit, a), o.less(a, o.number(F::infinity())))
    }
    pub(crate) fn routine(self) -> Repair {
        match self {
            Self::Sin => Repair { single: sin_single, double: sin_double },
            Self::Cos => Repair { single: cos_single, double: cos_double },
            Self::Tan => Repair { single: tan_single, double: tan_double },
        }
    }
}

extern "C" fn sin_single(x: f32) -> f32 { x.sin() }
extern "C" fn sin_double(x: f64) -> f64 { x.sin() }
extern "C" fn cos_single(x: f32) -> f32 { x.cos() }
extern "C" fn cos_double(x: f64) -> f64 { x.cos() }
extern "C" fn tan_single(x: f32) -> f32 { x.tan() }
extern "C" fn tan_double(x: f64) -> f64 { x.tan() }

#[derive(Clone, Copy, Debug)]
pub struct Sin;
#[derive(Clone, Copy, Debug)]
pub struct Cos;
#[derive(Clone, Copy, Debug)]
pub struct Tan;

#[inline]
fn scalar<R: Native, const FUSED: bool>(x: R, kind: Kind) -> R {
    let o = crate::recipe::NativeOps::<FUSED>::DEFAULT;
    if kind.predicate(&o, x) { kind.routine().apply(x) } else { formula(&o, x, kind) }
}
#[inline]
pub(crate) fn sin<R: Native, const FUSED: bool>(x: R) -> R { scalar::<R, FUSED>(x, Kind::Sin) }
#[inline]
pub(crate) fn cos<R: Native, const FUSED: bool>(x: R) -> R { scalar::<R, FUSED>(x, Kind::Cos) }
#[inline]
pub(crate) fn tan<R: Native, const FUSED: bool>(x: R) -> R { scalar::<R, FUSED>(x, Kind::Tan) }

pub(crate) fn formula<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, kind: Kind) -> O::V {
    let zero = o.number(F::zero());
    let finite = o.less(o.magnitude(x), o.number(F::infinity()));
    let safe = o.select(o.and(finite, o.not(kind.predicate(o, x))), x, zero);
    let (r, n) = if o.fused() { reduce_fused(o, safe, kind) } else { reduce_split(o, safe, kind) };
    let odd = o.integer_compare(Cmp::Equal, o.integer_op(Integer::And, o.to_integer(n), o.integer(1)), o.integer(1));
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

const SIN_64: [f64; 7] = [-7.364473665132213e-13, 1.604733546032205e-10, -2.505185490163253e-8, 2.755731618671281e-6, -0.0001984126982169595, 0.008333333333273485, -0.1666666666666606];
const SIN_32: [f64; 4] = [2.6105958568223286e-6, -0.00019811994570773095, 0.00833310279995203, -0.16666661202907562];
fn sine<F: Float, O: Ops<F> + ?Sized>(o: &O, r: O::V) -> O::V {
    let r2 = o.multiply(r, r);
    let p = crate::recipe::horner(o, r2, if o.width() == FloatWidth::F64 { &SIN_64 } else { &SIN_32 });
    o.multiply_add(o.multiply(r, r2), p, r)
}

const TAN_64: [f64; 9] = [0.0003287231044337369, 0.000559125460707824, 0.0014615495400569908, 0.0035914987665774814, 0.008863277952340431, 0.021869486815122453, 0.05396825400765754, 0.13333333333289177, 0.33333333333333504];
const TAN_32: [f64; 4] = [0.025565408170223236, 0.053473446518182755, 0.13335898518562317, 0.33333292603492737];
fn tangent<F: Float, O: Ops<F> + ?Sized>(o: &O, r: O::V, odd: O::B) -> O::V {
    let u = o.multiply(r, o.number(F::narrow(0.5)));
    let u2 = o.multiply(u, u);
    let p = crate::recipe::horner(o, u2, if o.width() == FloatWidth::F64 { &TAN_64 } else { &TAN_32 });
    let t = o.multiply_add(o.multiply(u, u2), p, u);
    let below = o.multiply_add(t, t, o.number(-F::one()));
    let twice = o.add(t, t);
    o.divide(o.select(odd, below, o.negate(twice)), o.select(odd, twice, below))
}
