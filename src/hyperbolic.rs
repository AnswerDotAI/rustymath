//! The hyperbolic functions and their inverses. The functions build on `exp` and `expm1`, the inverses on `ln`'s reduction. Each computes at |x| and restores the sign of x where the function is odd.
use crate::{exp::{exp_times, expm1, expm1_times}, float::Native, ln::ln_parts};

/// sinh |x| = h(1 + 1/(2h + 1)), with h = (e^|x| - 1)/2. The halving joins `expm1`'s scaling, so sinh stays finite where e^|x| would overflow. Where h is ∞, 1/(2h + 1) is 0, and the result is ∞.
#[inline]
pub(crate) fn sinh<R: Native, const FUSED: bool>(x: R) -> R {
    let h = expm1_times::<R, FUSED>(x.abs(), -R::one());
    (h * (R::one() + R::one() / (h + h + R::one()))).copysign(x)
}

/// cosh x = E + 1/(4E), with E = e^|x|/2, which stays finite where e^|x| would overflow.
#[inline]
pub(crate) fn cosh<R: Native, const FUSED: bool>(x: R) -> R { let e = exp_times::<R, FUSED>(x.abs(), -R::one()); e + R::narrow(0.25) / e }

/// tanh |x| = e/(e + 2), with e = e^(2|x|) - 1. Beyond 20 at f64 and 9 at f32, tanh rounds to 1, so |x| stops there and e stays finite.
#[inline]
pub(crate) fn tanh<R: Native, const FUSED: bool>(x: R) -> R {
    let top = R::narrow(if R::WIDE { 20.0 } else { 9.0 });
    let a = x.abs();
    let a = if a > top { top } else { a };
    let e = expm1::<R, FUSED>(a + a);
    (e / (e + R::narrow(2.0))).copysign(x)
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
