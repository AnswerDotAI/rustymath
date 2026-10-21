//! e^x and e^x - 1. The f64 reduction and the form of the final step come from SLEEF's `xexp`, and the f32 polynomial of `exp` and the f32 split of ln 2 from Arm Optimized Routines' AdvSIMD `expf`. The other polynomials are minimax fits. See `THIRD-PARTY.md`.
use crate::{dd::Double, float::{Native, horner, ln2, pow2_times, sealed::Core}};

/// The coefficients of h(r) = (e^r - 1)/r, highest degree first, for |r| up to about ln 2 / 2. `exp` needs h only to within its error in e^r = 1 + r h, which shrinks with r.
const EXP_64: [f64; 10] = [
    2.749197269771299e-07, 2.7639186521124947e-06, 2.4801900316905035e-05, 0.00019841173089486715, 0.0013888888529025865, 0.008333333382940285,
    0.04166666666820299, 0.16666666666564642, 0.4999999999999798, 1.0000000000000056,
];
const EXP_32: [f64; 5] = [0.008247390389442444, 0.04189976677298546, 0.16668395698070526, 0.4999912679195404, 0.9999994039535522];
/// The same, to about 2^-28 relative in e^r, for f32 and f16 results computed in f64. At that accuracy, a power whose exact result is a narrower float comes out exactly.
const EXP_NARROW: [f64; 6] = [0.0013941108433972674, 0.008375126398153335, 0.04166635289677515, 0.16666415514653277, 0.5000000047117757, 1.000000037716214];
/// h to within a fraction of an ulp relative to h itself, for `expm1`, whose result near 0 is r h.
const H_64: [f64; 11] = [
    2.5052099500253996e-08, 2.762626280199822e-07, 2.7557414780064832e-06, 2.4801504602137864e-05, 0.00019841269707477827, 0.001388888893225886,
    0.008333333333397829, 0.041666666666573905, 0.16666666666666563, 0.5000000000000006, 1.0,
];
const H_32: [f64; 6] = [0.0013888871762901545, 0.00836906861513853, 0.041667137295007706, 0.16666506230831146, 0.4999999701976776, 1.0];

/// The magnitude beyond which e^x is 0 or ∞ at every width this crate computes in. Each half of n stays within the exponents of normal floats.
#[inline]
pub(crate) fn limit<R: Native>() -> R { R::narrow(if R::WIDE { 1100.0 } else { 150.0 }) }

/// x limited to ±l. Comparisons with NaN fail, so NaN passes through.
#[inline]
pub(crate) fn clamp<R: Native>(x: R, l: R) -> R { if x < -l { -l } else if x > l { l } else { x } }

/// n and r with x = n ln 2 + r, n whole and |r| at most about ln 2 / 2, for |x| up to `limit`.
#[inline]
fn reduce<R: Native, const FUSED: bool>(x: R) -> (R, R) {
    let c = R::narrow;
    let n = x.mla::<FUSED>(c(std::f64::consts::LOG2_E), R::SHIFT) - R::SHIFT;
    let (hi, lo) = ln2::<R>();
    (n, n.mla::<FUSED>(-lo, n.mla::<FUSED>(-hi, x)))
}

/// 2^n e^r, with e^r = 1 + r h(r) and h from `poly`.
#[inline]
fn scaled<R: Native, const FUSED: bool, const N: usize>(n: R, r: R, poly: [f64; N]) -> R {
    pow2_times::<R, FUSED>(r.mla::<FUSED>(horner::<R, FUSED, N>(r, poly), R::one()), n)
}

/// e^(x + lo) × 2^k, for a lo far below x and a whole k. lo joins the reduced argument, except where `clamp` changes x and the result is 0 or ∞ whatever lo is. Scaling by 2^k inside keeps e^x/2 finite where e^x would overflow. A caller passes -0 for a part it lacks. The compiler drops such a part, because adding -0 changes no value.
#[inline]
fn exp_parts<R: Native, const FUSED: bool>(x: R, lo: R, k: R) -> R {
    let c = clamp(x, limit());
    let (n, r) = reduce::<R, FUSED>(c);
    let r = if c == x { r + lo } else { r };
    if R::WIDE { scaled::<R, FUSED, 10>(n + k, r, EXP_64) } else { scaled::<R, FUSED, 5>(n + k, r, EXP_32) }
}

#[inline]
pub(crate) fn exp<R: Native, const FUSED: bool>(x: R) -> R { exp_parts::<R, FUSED>(x, R::neg_zero(), R::neg_zero()) }

/// e^x × 2^k, for a whole k.
#[inline]
pub(crate) fn exp_times<R: Native, const FUSED: bool>(x: R, k: R) -> R { exp_parts::<R, FUSED>(x, R::neg_zero(), k) }

#[inline]
pub(crate) fn expm1<R: Native, const FUSED: bool>(x: R) -> R { expm1_times::<R, FUSED>(x, R::zero()) }

/// (e^x - 1) × 2^k, for a whole k. e^x - 1 = 2^n (r h) + (2^n - 1). For |x| below about ln 2 / 2, n is 0 and the result is r h, which keeps its relative accuracy down to the smallest x. Below -40 at f64 and -18 at f32, e^x - 1 rounds to -1. n can exceed the greatest exponent by one while the result is still finite, so 2^n splits into 2^m and 2^(n-m), with m no greater than that exponent.
#[inline]
pub(crate) fn expm1_times<R: Native, const FUSED: bool>(x: R, k: R) -> R {
    let c = R::narrow;
    let (low, top) = if R::WIDE { (c(-40.0), c(1023.0)) } else { (c(-18.0), c(127.0)) };
    let (n, r) = reduce::<R, FUSED>(clamp(if x < low { low } else { x }, limit()));
    let h = if R::WIDE { horner::<R, FUSED, 11>(r, H_64) } else { horner::<R, FUSED, 6>(r, H_32) };
    let m = if n > top { top } else { n };
    let scale = R::pow2(m + R::SHIFT);
    (r * h).mla::<FUSED>(scale, scale - R::one()) * R::pow2(n - m + k + R::SHIFT)
}

/// e^(hi + lo) for any double word. It renormalizes the word first, because `exp_parts` needs `lo` far below `hi`.
#[inline]
pub(crate) fn exp_plus<R: Native, const FUSED: bool>(x: Double<R>) -> R { let x = Double::sum(x.hi, x.lo); exp_parts::<R, FUSED>(x.hi, x.lo, R::neg_zero()) }

/// e^x in f64 for an f32 or f16 result. Beyond 110 in magnitude, e^x is 0 or ∞ at those widths. x stops there. One power of two then scales the result.
#[inline]
pub(crate) fn exp_narrow<const FUSED: bool>(x: f64) -> f64 {
    let (n, r) = reduce::<f64, FUSED>(clamp(x, 110.0));
    r.mla::<FUSED>(horner::<f64, FUSED, 6>(r, EXP_NARROW), 1.0) * f64::pow2(n + f64::SHIFT)
}
