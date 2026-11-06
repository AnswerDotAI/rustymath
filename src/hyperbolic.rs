//! The hyperbolic functions and their inverses. The functions build on `exp` and `expm1`, the inverses on `ln`'s reduction. Each computes at |x| and restores the sign of x where the function is odd.
use crate::{float::Native, ln::ln_parts};

/// sinh |x| = h(1 + 1/(2h + 1)), with h = (e^|x| - 1)/2. The halving joins `expm1`'s scaling, so sinh stays finite where e^|x| would overflow. Where h is ∞, 1/(2h + 1) is 0, and the result is ∞.
#[inline]
pub(crate) fn sinh<R: Native, const FUSED: bool>(x: R) -> R { sinh_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x) }
#[inline]
pub(crate) fn cosh<R: Native, const FUSED: bool>(x: R) -> R { cosh_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x) }
#[inline]
pub(crate) fn tanh<R: Native, const FUSED: bool>(x: R) -> R { tanh_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x) }

pub(crate) fn sinh_formula<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V {
    let one = o.number(F::one());
    let h = crate::exp::minus_one_formula(o, o.magnitude(x), o.negate(one));
    let result = o.multiply(h, o.add(one, o.divide(one, o.add(o.add(h, h), one))));
    o.select(o.less(x, o.number(F::zero())), o.negate(result), result)
}
pub(crate) fn cosh_formula<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V {
    let e = crate::exp::scaled_formula(o, o.magnitude(x), o.number(-F::one()));
    o.add(e, o.divide(o.number(F::narrow(0.25)), e))
}
pub(crate) fn tanh_formula<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V {
    let top = o.number(F::narrow(if o.width() == crate::buffer::FloatWidth::F64 { 20.0 } else { 9.0 }));
    let a = o.magnitude(x);
    let a = o.select(o.less(top, a), top, a);
    let e = crate::exp::minus_one_formula(o, o.add(a, a), o.number(F::zero()));
    let result = o.divide(e, o.add(e, o.number(F::narrow(2.0))));
    o.select(o.less(x, o.number(F::zero())), o.negate(result), result)
}

/// Beyond this magnitude, `asinh` and `acosh` take v = x in place of x + √(x² ± 1), which rounds to 2x and may overflow, and add 1 to e. x² hasn't yet overflowed below it at either width.
#[inline]
fn large<R: Native>() -> R { R::narrow(4503599627370496.0) }

/// ln v for v = m × 2^e, with e raised by 1 when `far`. While e is 0, `near` gives the quotient s = (m-1)/(m+1) as (num, den) in a form that keeps its accuracy. An infinite or NaN v gives itself. Every caller's v is 1 or more, ∞, NaN or negative, so none needs the subnormal step of `split_positive`.
#[inline]
fn ln_of<R: Native, const FUSED: bool>(v: R, far: bool, near: (R, R)) -> R {
    let (m, e) = v.split();
    let e = if far { e + R::one() } else { e };
    let (num, den) = if e == R::zero() { near } else { (m - R::one(), m + R::one()) };
    let y = ln_parts::<R, FUSED>(num / den, e);
    if v.is_nan() | (v == R::infinity()) { v } else { y }
}

/// asinh |x| = ln v with v = |x| + r and r = √(x² + 1). Near 0, s = (v-1)/(v+1) = (|x|(1+r) + x²)/((2+|x|)(1+r) + x²).
#[inline]
pub(crate) fn asinh<R: Native, const FUSED: bool>(x: R) -> R {
    let a = x.abs();
    let far = a > large();
    let r = a.mla::<FUSED>(a, R::one()).sqrt();
    let (q, a2) = (R::one() + r, a * a);
    let near = (a.mla::<FUSED>(q, a2), (a + R::narrow(2.0)).mla::<FUSED>(q, a2));
    ln_of::<R, FUSED>(if far { a } else { a + r }, far, near).copysign(x)
}

/// acosh x = ln v with v = x + √(x² - 1). Near 1, s = t/(2 + t) with t = v - 1 = d + √(d² + 2d), where d = x - 1 is exact. x below 1 gives NaN.
#[inline]
pub(crate) fn acosh<R: Native, const FUSED: bool>(x: R) -> R {
    let far = x > large();
    let d = x - R::one();
    let t = d + d.mla::<FUSED>(d, d + d).sqrt();
    let y = ln_of::<R, FUSED>(if far { x } else { R::one() + t }, far, (t, t + R::narrow(2.0)));
    if x < R::one() { R::nan() } else { y }
}

/// atanh |x| = ln v / 2 with v = (1 + |x|)/(1 - |x|). Near 0, s = (v-1)/(v+1) is |x| itself. ±1 gives ±∞, and a magnitude beyond 1 gives NaN.
#[inline]
pub(crate) fn atanh<R: Native, const FUSED: bool>(x: R) -> R {
    let a = x.abs();
    let y = ln_of::<R, FUSED>((R::one() + a) / (R::one() - a), false, (a, R::one())) * R::narrow(0.5);
    (if a <= R::one() { y } else { R::nan() }).copysign(x)
}

/// The real part of acosh x, or of its principal value at a complex z, as a function object: acosh |x| for |x| at least 1, and 0 between -1 and 1. The real part of `6○⁻¹`.
#[derive(Clone, Copy, Debug)]
pub struct RealAcosh;

/// The real part of atanh x, or of its principal value at a complex z, as a function object: atanh x for x between -1 and 1, and atanh (1 ÷ x) beyond, which is ½ ln |(1 + x) ÷ (1 - x)|. The real part of `7○⁻¹`.
#[derive(Clone, Copy, Debug)]
pub struct RealAtanh;
