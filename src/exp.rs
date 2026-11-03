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
fn reduce<R: Native, const FUSED: bool>(x: R) -> (R, R) { reduce_with(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x) }

/// e^r = 1 + r h(r), for |r| up to about ln 2 / 2.
#[inline(always)]
fn exp_reduced<R: Native, const FUSED: bool>(r: R) -> R { reduced_with(&crate::recipe::NativeOps::<FUSED>::DEFAULT, r) }

/// e^(x + lo) × 2^k, for a lo far below x and a whole k. lo joins the reduced argument, except where `clamp` changes x and the result is 0 or ∞ whatever lo is. Scaling by 2^k inside keeps e^x/2 finite where e^x would overflow. A caller passes -0 for a part it lacks. The compiler drops such a part, because adding -0 changes no value.
#[inline]
fn exp_parts<R: Native, const FUSED: bool>(x: R, lo: R, k: R) -> R { parts_with(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x, lo, k) }

#[inline]
pub(crate) fn exp<R: Native, const FUSED: bool>(x: R) -> R { exp_parts::<R, FUSED>(x, R::neg_zero(), R::neg_zero()) }

/// Whether `exp_fast` and `expm1_fast` give their functions at x: |x| is below 708 at f64 and 87 at f32, where 2^n is a normal float. NaN and the infinities fail the test. Accepting -∞ here, with a select in `exp_fast`, made other items about 18% slower on macOS.
#[inline(always)]
fn usual<R: Native>(x: R) -> bool { x.abs() < R::narrow(if R::WIDE { 708.0 } else { 87.0 }) }

/// e^x for an x that `usual` accepts, with no clamp and one power of two.
#[inline(always)]
fn exp_fast<R: Native, const FUSED: bool>(x: R) -> R { let (n, r) = reduce::<R, FUSED>(x); exp_reduced::<R, FUSED>(r) * R::pow2(n + R::SHIFT) }

/// Whether e^x rounds to 0: x is below -746 at f64 and -105 at f32, or -∞, where a causal mask puts it.
#[inline(always)]
fn under<R: Native>(x: R) -> bool { x < R::narrow(if R::WIDE { -746.0 } else { -105.0 }) }

usual_object!(/// [`exp`](crate::exp) as a function object. Its slice form clamps x and scales by two powers of two only in blocks that hold an item that neither `usual` nor `under` accepts.
    Exp: exp, usual => exp_fast::<_, { crate::FUSED }>, under => |_| 0.0; total);

/// e^x × 2^k, for a whole k.
#[inline]
pub(crate) fn exp_times<R: Native, const FUSED: bool>(x: R, k: R) -> R { exp_parts::<R, FUSED>(x, R::neg_zero(), k) }

#[inline]
pub(crate) fn expm1<R: Native, const FUSED: bool>(x: R) -> R { expm1_times::<R, FUSED>(x, R::zero()) }

/// e^x - 1 from x's reduction, 2^n (r h) + (2^n - 1), with s = 2^n. For |x| below about ln 2 / 2, n is 0 and the result is r h, which keeps its relative accuracy down to the smallest x.
#[inline(always)]
fn expm1_reduced<R: Native, const FUSED: bool>(r: R, s: R) -> R { minus_one_with(&crate::recipe::NativeOps::<FUSED>::DEFAULT, r, s) }

/// e^x - 1 for an x that `usual` accepts, with no clamp and one power of two. `tanh` calls it for every x, because it keeps its argument within that range.
#[inline(always)]
pub(crate) fn expm1_fast<R: Native, const FUSED: bool>(x: R) -> R { let (n, r) = reduce::<R, FUSED>(x); expm1_reduced::<R, FUSED>(r, R::pow2(n + R::SHIFT)) }

usual_object!(/// [`expm1`](crate::expm1) as a function object, with `Exp`'s blocks.
    Expm1: expm1, usual => expm1_fast::<_, { crate::FUSED }>; total);

/// (e^x - 1) × 2^k, for a whole k. Below -40 at f64 and -18 at f32, e^x - 1 rounds to -1. n can exceed the greatest exponent by one while the result is still finite, so 2^n splits into 2^m and 2^(n-m), with m no greater than that exponent.
#[inline]
pub(crate) fn expm1_times<R: Native, const FUSED: bool>(x: R, k: R) -> R { minus_one_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x, k) }

/// e^(hi + lo) for any double word. It renormalizes the word first, because `exp_parts` needs `lo` far below `hi`.
#[inline]
pub(crate) fn exp_plus<R: Native, const FUSED: bool>(x: Double<R>) -> R { let x = Double::sum(x.hi, x.lo); exp_parts::<R, FUSED>(x.hi, x.lo, R::neg_zero()) }

/// e^x in f64 for an f32 or f16 result. Beyond 110 in magnitude, e^x is 0 or ∞ at those widths. x stops there. One power of two then scales the result.
#[inline]
pub(crate) fn exp_narrow<const FUSED: bool>(x: f64) -> f64 {
    let (n, r) = reduce::<f64, FUSED>(clamp(x, 110.0));
    r.mla::<FUSED>(horner::<f64, FUSED, 6>(r, EXP_NARROW), 1.0) * f64::pow2(n + f64::SHIFT)
}

#[inline(always)]
fn reduce_with<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V) -> (O::V, O::V) {
    let shift = crate::recipe::rounding_shift(o);
    let n = o.subtract(o.multiply_add(x, o.number(F::narrow(std::f64::consts::LOG2_E)), shift), shift);
    let (hi, lo) = crate::recipe::ln2(o);
    (n, o.multiply_add(n, o.negate(lo), o.multiply_add(n, o.negate(hi), x)))
}

#[inline(always)]
fn reduced_with<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, r: O::V) -> O::V {
    let h = crate::recipe::horner(o, r, if o.width() == crate::buffer::FloatWidth::F64 { &EXP_64 } else { &EXP_32 });
    o.multiply_add(r, h, o.number(F::one()))
}

#[inline(always)]
fn clamp_with<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V {
    let limit = o.number(F::narrow(if o.width() == crate::buffer::FloatWidth::F64 { 1100.0 } else { 150.0 }));
    let low = o.negate(limit);
    o.select(o.less(x, low), low, o.select(o.less(limit, x), limit, x))
}

#[inline(always)]
fn parts_with<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V, lo: O::V, k: O::V) -> O::V {
    let c = clamp_with(o, x);
    let (n, r) = reduce_with(o, c);
    let r = o.select(o.equal(c, x), o.add(r, lo), r);
    crate::recipe::ldexp(o, reduced_with(o, r), o.add(n, k))
}

#[inline(always)]
pub(crate) fn formula<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V {
    let zero = o.number(-F::zero());
    parts_with(o, x, zero, zero)
}

#[inline(always)]
fn minus_one_with<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, r: O::V, scale: O::V) -> O::V {
    let h = crate::recipe::horner(o, r, if o.width() == crate::buffer::FloatWidth::F64 { &H_64 } else { &H_32 });
    o.multiply_add(o.multiply(r, h), scale, o.subtract(scale, o.number(F::one())))
}

pub(crate) fn scaled_formula<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V, scale: O::V) -> O::V {
    parts_with(o, x, o.number(-F::zero()), scale)
}

pub(crate) fn minus_one_formula<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V, scale: O::V) -> O::V {
    let (low, top) = if o.width() == crate::buffer::FloatWidth::F64 { (-40.0, 1023.0) } else { (-18.0, 127.0) };
    let low = o.number(F::narrow(low));
    let (n, r) = reduce_with(o, clamp_with(o, o.select(o.less(x, low), low, x)));
    let top = o.number(F::narrow(top));
    let m = o.select(o.less(top, n), top, n);
    let shift = crate::recipe::rounding_shift(o);
    let first = minus_one_with(o, r, crate::recipe::pow2(o, o.add(m, shift)));
    o.multiply(first, crate::recipe::pow2(o, o.add(o.add(o.subtract(n, m), scale), shift)))
}
