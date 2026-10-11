//! The natural logarithm. The algorithm and both polynomials come from SLEEF's `xlog` and `xlogf`. See `THIRD-PARTY.md`.
use crate::float::{horner, Float, Native, FUSED};

/// The natural logarithm of x, within 2 ulp at f64 and 3 ulp at f32. `ln(0)` is -∞, `ln(∞)` is ∞, and a negative x or NaN gives NaN.
#[inline]
pub fn ln<F: Float>(x: F) -> F { F::from_total(ln_at::<_, FUSED>(x.total())) }

/// The coefficients of (ln((1+s)/(1-s)) - 2s)/s³ as a polynomial in s², highest degree first, for |s| up to 0.2.
const POLY_64: [f64; 7] = [
    0.153487338491425068243146,
    0.152519917006351951593857,
    0.181863266251982985677316,
    0.222221366518767365905163,
    0.285714294746548025383248,
    0.399999999950799600689777,
    0.6666666666667778740063,
];
/// The coefficients of ln((1+s)/(1-s))/s as a polynomial in s², highest degree first, for |s| up to 0.2.
const POLY_32: [f64; 5] = [0.2392828464508056640625, 0.28518211841583251953125, 0.400005877017974853515625, 0.666666686534881591796875, 2.0];

/// The natural logarithm of x in `R`, multiplying and adding with `mla::<FUSED>`. x = m × 2^e with m from 0.75 to 1.5, so
/// ln x = ln m + e ln 2, and ln m = ln((1+s)/(1-s)) with s = (m-1)/(m+1).
#[inline]
pub(crate) fn ln_at<R: Native, const FUSED: bool>(x: R) -> R {
    let c = R::narrow;
    // A subnormal x scales up by 2^64, so that its bits split as a normal float's do.
    let subnormal = x < R::min_positive_value();
    let (m, e) = (if subnormal { x * c(18446744073709551616.0) } else { x }).split();
    let e = if subnormal { e - c(64.0) } else { e };
    let s = (m - c(1.0)) / (m + c(1.0));
    let s2 = s * s;
    let y = if R::WIDE {
        (s * s2).mla::<FUSED>(horner::<R, FUSED, 7>(s2, POLY_64), s.mla::<FUSED>(c(2.0), c(std::f64::consts::LN_2) * e))
    } else {
        s.mla::<FUSED>(horner::<R, FUSED, 5>(s2, POLY_32), c(std::f64::consts::LN_2) * e)
    };
    let y = if x == R::infinity() { x } else { y };
    let y = if x == R::zero() { R::neg_infinity() } else { y };
    if x >= R::zero() { y } else { R::nan() }
}
