//! e^x. The f64 polynomial comes from SLEEF's `xexp`, and the f32 polynomial and the f32 split of ln 2 from Arm Optimized Routines'
//! AdvSIMD `expf`. See `THIRD-PARTY.md`.
use crate::float::{horner, Float, Native, FUSED};

/// e^x, within 1 ulp at f64 and 2 ulp at f32. `exp(∞)` is ∞, `exp(-∞)` is 0, and `exp(NaN)` is NaN.
#[inline]
pub fn exp<F: Float>(x: F) -> F { F::from_total(exp_at::<_, FUSED>(x.total())) }

/// ln 2, split into a part whose product with any whole number up to the limit below is exact, and the rest.
const LN2_64: (f64, f64) = (0.69314718055966295651160180568695068359375, 0.28235290563031577122588448175013436025525412068e-12);
const LN2_32: (f64, f64) = (0.693145751953125, 1.428606765330187e-06);
/// The coefficients of (e^r - 1)/r, highest degree first, for |r| up to about ln 2 / 2.
const POLY_64: [f64; 12] = [
    0.2081276378237164457e-8,
    0.2511210703042288022e-7,
    0.2755762628169491192e-6,
    0.2755723402025388239e-5,
    0.2480158687479686264e-4,
    0.1984126989855865850e-3,
    0.1388888888914497797e-2,
    0.8333333333314938210e-2,
    0.4166666666666602598e-1,
    0.1666666666666669072e+0,
    0.5,
    1.0,
];
const POLY_32: [f64; 5] = [0.008247390389442444, 0.04189976677298546, 0.16668395698070526, 0.4999912679195404, 0.9999994039535522];

/// e^x in `R`, multiplying and adding with `mla::<FUSED>`. x = n ln 2 + r, with n whole and |r| at most about ln 2 / 2, so
/// e^x = 2^n e^r.
#[inline]
pub(crate) fn exp_at<R: Native, const FUSED: bool>(x: R) -> R {
    let c = R::narrow;
    // Beyond this limit every result is 0 or ∞, and each half of n below stays within the exponents of normal floats. Comparisons
    // with NaN fail, so NaN passes through.
    let limit = c(if R::WIDE { 1100.0 } else { 150.0 });
    let x = if x < -limit { -limit } else if x > limit { limit } else { x };
    let n = x.mla::<FUSED>(c(std::f64::consts::LOG2_E), R::SHIFT) - R::SHIFT;
    let (hi, lo) = if R::WIDE { LN2_64 } else { LN2_32 };
    let r = n.mla::<FUSED>(c(-lo), n.mla::<FUSED>(c(-hi), x));
    let h = if R::WIDE { horner::<R, FUSED, 12>(r, POLY_64) } else { horner::<R, FUSED, 5>(r, POLY_32) };
    // 2^n in two factors. The first multiplication is exact, so a result below the normal floats rounds once, in the second.
    let half = n.mla::<FUSED>(c(0.5), R::SHIFT);
    let (a, b) = (R::pow2(half), R::pow2(n - (half - R::SHIFT) + R::SHIFT));
    r.mla::<FUSED>(h, c(1.0)) * a * b
}
