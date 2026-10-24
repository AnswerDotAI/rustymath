//! Logarithms. They share SLEEF's reduction from `xlog` and `xlogf`: x = m × 2^e with m from 0.75 to 1.5, and ln m = ln((1+s)/(1-s)) with s = (m-1)/(m+1), from SLEEF's f64 polynomial and minimax ones for narrower results. See `THIRD-PARTY.md`.
use crate::{Arg, dd::Double, float::{Native, horner, ln2, sealed::Core, split_positive}};
use std::f64::consts::{LN_2, LOG2_E, LOG10_E};

/// The coefficients of (ln((1+s)/(1-s)) - 2s)/s³ as a polynomial in s², highest degree first, for |s| up to 0.2.
const POLY_64: [f64; 7] = [
    0.153487338491425068243146, 0.152519917006351951593857, 0.181863266251982985677316, 0.222221366518767365905163, 0.285714294746548025383248,
    0.399999999950799600689777, 0.6666666666667778740063,
];
/// The coefficients of ln((1+s)/(1-s))/s as a polynomial in s², highest degree first, for |s| up to 0.2.
const POLY_32: [f64; 4] = [0.30434924364089966, 0.3995267450809479, 0.6666705012321472, 2.0];
/// The same, to about 2^-35 relative, for f32 and f16 results computed in f64. They need that even where a power multiplies the logarithm.
const POLY_NARROW: [f64; 5] = [0.24145279203110395, 0.28502925464850143, 0.40000986365881785, 0.6666666171721365, 2.0000000000396225];

/// ln((1+s)/(1-s)), for |s| up to 0.2, with f64's polynomial if `fine` and f32's otherwise.
#[inline]
fn series<R: Native, const FUSED: bool>(s: R, fine: bool) -> R {
    let s2 = s * s;
    if fine { (s * s2).mla::<FUSED>(horner::<R, FUSED, 7>(s2, POLY_64), s + s) } else { s * horner::<R, FUSED, 4>(s2, POLY_32) }
}

/// The logarithm e·hi + (e·lo + k·ln((1+s)/(1-s))), to the base whose logarithm of 2 is (hi, lo) and of e is k, with f64's polynomial if `fine` and f32's otherwise.
#[inline]
fn based<R: Native, const FUSED: bool>(s: R, e: R, (hi, lo): (R, R), k: R, fine: bool) -> R {
    e.mla::<FUSED>(hi, series::<R, FUSED>(s, fine).mla::<FUSED>(k, e * lo))
}

/// e ln 2 + ln((1+s)/(1-s)): the natural logarithm of m × 2^e, for s = (m-1)/(m+1) or a more accurate form of it.
#[inline]
pub(crate) fn ln_parts<R: Native, const FUSED: bool>(s: R, e: R) -> R { let (base, k) = natural(); based::<R, FUSED>(s, e, base, k, R::WIDE) }

/// The logarithm of 2 to base e, as `based` takes it, and the logarithm of e to base e.
#[inline(always)]
fn natural<R: Native>() -> ((R, R), R) { (ln2(), R::one()) }

/// Whether `log` needs none of its special values at x: x is positive, normal and finite.
#[inline(always)]
fn usual<R: Native>(x: R) -> bool { x.positive_normal() }

/// The logarithm of m × 2^e, split as `(m, e)`, to the base that `(base, k)` gives, as `based` takes them, with none of `log`'s special values.
#[inline(always)]
fn log_of<R: Native, const FUSED: bool>((m, e): (R, R), (base, k): ((R, R), R)) -> R { based::<R, FUSED>((m - R::one()) / (m + R::one()), e, base, k, R::WIDE) }

/// The logarithm of x to the base that `to` gives, as `log_of` takes it, with its special values: -∞ at 0, ∞ at ∞, and NaN for a negative x or NaN.
#[inline]
fn log<R: Native, const FUSED: bool>(x: R, to: ((R, R), R)) -> R {
    let y = log_of::<R, FUSED>(split_positive(x), to);
    let y = if x == R::infinity() { x } else { y };
    let y = if x == R::zero() { R::neg_infinity() } else { y };
    if x >= R::zero() { y } else { R::nan() }
}

#[inline]
pub(crate) fn ln<R: Native, const FUSED: bool>(x: R) -> R { log::<R, FUSED>(x, natural()) }

/// ln x for a positive, finite, nonzero x, computed in f64 for an f32 or f16 result. Such an x is never a subnormal f64. At this accuracy, ln 2 needs no lower part.
#[inline(always)]
pub(crate) fn ln_narrow<const FUSED: bool>(x: f64) -> f64 {
    let (m, e) = x.split();
    let s = (m - 1.0) / (m + 1.0);
    // Estrin's form, which shortens the chain of dependent operations.
    let [a4, a3, a2, a1, a0] = POLY_NARROW;
    let u = s * s;
    let u2 = u * u;
    let p = u2.mla::<FUSED>(a4, u.mla::<FUSED>(a3, a2)).mla::<FUSED>(u2, u.mla::<FUSED>(a1, a0));
    e.mla::<FUSED>(LN_2, s * p)
}

/// ln x for a positive x as the unrounded double word e ln 2 + ln m. Its absolute error is that of ln m, about 1e-16, whatever x's size, which suits a logarithm that a sum then cancels. `ln_double` costs more and also holds 2^-62 relative.
#[inline(always)]
pub(crate) fn ln_split<const FUSED: bool>(x: f64) -> Double<f64> {
    let (m, e) = split_positive(x);
    let (hi, lo): (f64, f64) = ln2();
    Double::sum(e * hi, e.mla::<FUSED>(lo, series::<f64, FUSED>((m - 1.0) / (m + 1.0), true)))
}

/// The logarithm of 2 to base 2, as `based` takes it, and the logarithm of e to base 2.
#[inline(always)]
fn base2<R: Native>() -> ((R, R), R) { ((R::one(), R::zero()), R::narrow(LOG2_E)) }

/// The logarithm of 2 to base 10, as `based` takes it, split as `ln2` splits ln 2, and the logarithm of e to base 10. At f32 the split has its own high part. The f64 high part, rounded to f32, would make e times it inexact.
#[inline(always)]
fn base10<R: Native>() -> ((R, R), R) {
    let (hi, lo) = if R::WIDE { (0.30102999566361177, 3.694239077158931e-13) } else { (0.30103302001953125, -3.024355464731343e-6) };
    ((R::narrow(hi), R::narrow(lo)), R::narrow(LOG10_E))
}

#[inline]
pub(crate) fn log2<R: Native, const FUSED: bool>(x: R) -> R { log::<R, FUSED>(x, base2()) }

#[inline]
pub(crate) fn log10<R: Native, const FUSED: bool>(x: R) -> R { log::<R, FUSED>(x, base10()) }

/// Base 2's value, base 10's or every other base's, chosen for base b with selects, not branches.
#[inline(always)]
fn by_base<R: Native, T>(b: R, [two, ten, other]: [T; 3]) -> T { if b == R::narrow(2.0) { two } else if b == R::narrow(10.0) { ten } else { other } }

/// The constants that `log` takes for base b, and the divisor of its result. Base 2 and base 10 take the constants of `log2` and `log10`, which are exact at powers of their base, and divide by 1. Any other base takes the natural logarithm's constants and divides by ln b. It computes ln b even at base 2 and base 10. Computing it only for other bases would take a branch.
#[inline(always)]
fn for_base<R: Native, const FUSED: bool>(b: R) -> (((R, R), R), R) {
    let lb = ln::<R, FUSED>(b);
    (by_base(b, [base2(), base10(), natural()]), by_base(b, [R::one(), R::one(), lb]))
}

/// The logarithm of x to base b. Base 2 gives `log2` and base 10 gives `log10`, which are exact at powers of their base. Any other base gives ln x / ln b. It is always inlined. Out of line, its call would stop a caller's loop vectorizing.
#[inline(always)]
pub(crate) fn logb<R: Native, const FUSED: bool>(b: R, x: R) -> R { let (to, d) = for_base::<R, FUSED>(b); log::<R, FUSED>(x, to) / d }

#[inline(always)]
fn logb_mode<R: Native, const MATHISH: bool>(b: R, x: R) -> R {
    let y = logb::<_, { crate::FUSED }>(b, x);
    if MATHISH && b == R::one() && x == R::one() { R::zero() } else { y }
}

/// [`logb`](crate::logb) as a function object. Its slice form computes the logarithm of a constant argument once, before its loop.
#[derive(Clone, Copy, Debug)]
pub struct Logb;
forms!(Logb at real: |_, [b, x]| logb::<_, { crate::FUSED }>(b, x), many |_, [b, x], n, out| logb_many::<_, false>([b, x], n, out));

/// [`Logb`] with log base one of one defined as zero.
#[derive(Clone, Copy, Debug)]
pub struct MathishLogb;
forms!(MathishLogb at real: |_, [b, x]| logb_mode::<_, true>(b, x), many |_, [b, x], n, out| logb_many::<_, true>([b, x], n, out));

#[inline]
fn logb_many<R: Native, const MATHISH: bool>([b, x]: [Arg<'_, R>; 2], n: usize, out: &mut [std::mem::MaybeUninit<R>]) -> bool {
    match (b, x) {
        (Arg::One(c), _) => {
            // Bases 2 and 10 divide by 1, which their loop skips, and take the logarithms' fast form for usual items. Any other base divides by ln b, which can be 0 or NaN, so its loop judges every result.
            let (to, d) = for_base::<_, { crate::FUSED }>(c);
            if MATHISH && c == R::one() {
                each!([b, x], n, out, |[_, x]| if x == R::one() { R::zero() } else { log::<_, { crate::FUSED }>(x, to) / d })
            } else if d == R::one() {
                crate::slice::each_usual(x, n, out, (usual, #[inline(always)] |x| log_of::<_, { crate::FUSED }>(x.split(), to)), crate::slice::none(),
                    #[inline(always)] |x| log::<_, { crate::FUSED }>(x, to))
            } else {
                each!([b, x], n, out, |[_, x]| log::<_, { crate::FUSED }>(x, to) / d)
            }
        }
        (Arg::Many(bs), Arg::One(c)) => {
            // A base of exactly 2 or 10 takes `log2` or `log10` of c. A second pass writes those items, because a select in the main loop stops it vectorizing. At those bases either result is NaN only where c is negative or NaN. The main loop's outcome holds.
            let l = ln::<_, { crate::FUSED }>(c);
            let ok = if MATHISH && c == R::one() {
                each!([b, x], n, out, |[b, _]| if b == R::one() { R::zero() } else { l / ln::<_, { crate::FUSED }>(b) })
            } else { each!([b, x], n, out, |[b, _]| l / ln::<_, { crate::FUSED }>(b)) };
            let exact = [log2::<_, { crate::FUSED }>(c), log10::<_, { crate::FUSED }>(c)];
            for (r, &b) in crate::slice::written(out).iter_mut().zip(bs) { *r = by_base(b, [exact[0], exact[1], *r]) }
            ok
        }
        _ => each!([b, x], n, out, |[b, x]| logb_mode::<_, MATHISH>(b, x)),
    }
}

/// The part of `log1p` that computes ln(1 + x) from u = 1 + x, split as `(m, e)`, with none of its special values.
#[inline(always)]
fn log1p_of<R: Native, const FUSED: bool>(x: R, u: R, (m, e): (R, R)) -> R {
    let top = R::narrow(if R::WIDE { 1000.0 } else { 100.0 });
    let c = (x - (u - R::one())) * R::pow2(R::SHIFT - if e > top { top } else { e });
    ln_parts::<R, FUSED>((m - R::one() + c) / (m + R::one() + c), e)
}

/// ln(1 + x). Rounding u = 1 + x loses c = x - (u - 1). With u = m × 2^e and c' = c × 2^-e, the quotient s = (m - 1 + c')/(m + 1 + c') restores it, and a small x keeps its relative accuracy with one division. Beyond 2^53, c is 0. e stops at `top` there, to keep 2^-e a normal float.
#[inline]
pub(crate) fn log1p<R: Native, const FUSED: bool>(x: R) -> R {
    let u = x + R::one();
    let y = log1p_of::<R, FUSED>(x, u, split_positive(u));
    // -0 must give -0, which the quotient turns into 0.
    let y = if (x == R::infinity()) | (x == R::zero()) { x } else { y };
    let y = if u == R::zero() { R::neg_infinity() } else { y };
    if u >= R::zero() { y } else { R::nan() }
}

/// `log1p` for an x that `usual1p` accepts.
#[inline(always)]
fn log1p_usual<R: Native, const FUSED: bool>(x: R) -> R { let u = x + R::one(); log1p_of::<R, FUSED>(x, u, u.split()) }

/// Whether `log1p` needs none of its special values at x: 1 + x is positive, normal and finite, and x isn't 0.
#[inline(always)]
fn usual1p<R: Native>(x: R) -> bool { usual(x + R::one()) & (x != R::zero()) }

usual_object!(/// [`ln`](crate::ln) as a function object.
    Ln: ln, usual => #[inline(always)] |x| log_of::<_, { crate::FUSED }>(x.split(), natural()));
usual_object!(/// [`log2`](crate::log2) as a function object.
    Log2: log2, usual => #[inline(always)] |x| log_of::<_, { crate::FUSED }>(x.split(), base2()));
usual_object!(/// [`log10`](crate::log10) as a function object.
    Log10: log10, usual => #[inline(always)] |x| log_of::<_, { crate::FUSED }>(x.split(), base10()));
usual_object!(/// [`log1p`](crate::log1p) as a function object.
    Log1p: log1p, usual1p => log1p_usual::<_, { crate::FUSED }>);

/// ln 2 as a double word.
const LN2: (f64, f64) = (0.693147180559945286226764, 2.319046813846299558417771e-17);
/// The centres of the four buckets that `ln_double` reduces m to, 2^(k/4) for k from -1 to 2, with ln of each as a double word, and the boundaries between buckets, 2^(j/8) for j = -1, 1, 3. A centre of exactly 1 keeps ln x accurate near x = 1.
const CENTRES: [(f64, f64, f64); 4] = [
    (0.8408964152537145, -0.17328679513998638, 9.61928934371463e-19), (1.0, 0.0, 0.0), (1.189207115002721, 0.1732867951399863, 6.856931399313918e-20),
    (1.4142135623730951, 0.3465735902799727, 2.4442169414592898e-17),
];
const BOUNDS: [f64; 3] = [0.9170040432046712, 1.0905077326652577, 1.2968395546510096];
/// 2/(2k+1) for k from 7 down to 1: ln((1+s)/(1-s)) = 2s + s³ × this polynomial in s².
const TAIL: [f64; 7] = [2.0 / 15.0, 2.0 / 13.0, 2.0 / 11.0, 2.0 / 9.0, 2.0 / 7.0, 2.0 / 5.0, 2.0 / 3.0];

/// ln x for an f64 x, as a double word accurate to about 2^-62 relative, for `pow`, which multiplies it by the exponent. It is -∞ at 0, and x itself at ∞ and NaN. ln x = e ln 2 + ln c + ln((1+s)/(1-s)), with c the nearest of `CENTRES` to m and s = (m-c)/(m+c). m - c is exact. With |s| below 0.06, only 2s needs a double word. It's inlined always: LLVM would otherwise leave it out of line, and the loops of `pow` would stop vectorizing.
#[inline(always)]
pub(crate) fn ln_double<const FUSED: bool>(x: f64) -> Double<f64> {
    let (m, e) = split_positive(x);
    let [b0, b1, b2] = BOUNDS;
    let (c, lc, lc_lo) = if m < b0 { CENTRES[0] } else if m < b1 { CENTRES[1] } else if m < b2 { CENTRES[2] } else { CENTRES[3] };
    let s = Double::new(m - c, 0.0).div::<FUSED>(Double::sum(m, c));
    let s2 = s.hi * s.hi;
    let ln_m = Double::sum(lc, s.hi + s.hi);
    let ln_m = Double::new(ln_m.hi, ln_m.lo + lc_lo + (s.lo + s.lo) + s.hi * s2 * horner::<f64, FUSED, 7>(s2, TAIL)).normalize();
    let l = Double::new(LN2.0, LN2.1).mul_float::<FUSED>(e).add(ln_m);
    let l = if x.is_nan() | (x == f64::INFINITY) { Double::new(x, 0.0) } else { l };
    if x == 0.0 { Double::new(f64::NEG_INFINITY, 0.0) } else { l }
}
