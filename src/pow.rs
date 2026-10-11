//! Powers and roots. `pow` and `root` compute e^(y ln |x|): at f64 through `table`'s logarithm and exponential, and for narrower
//! types through `ln_narrow` and `exp_narrow` in f64. `cbrt` refines an estimate taken from x's bits with Newton steps. `hypot` scales
//! both arguments by a power of two before squaring them. See `THIRD-PARTY.md`.
use crate::{
    dd::Double,
    exp::exp_narrow,
    float::{sealed::Core, Native},
    ln::ln_narrow,
    table,
};

/// 2^52, the magnitude at which consecutive floats are consecutive whole numbers.
const TWO_52: f64 = 4503599627370496.0;

/// Whether a is finite and nonzero, for an a whose sign bit is clear. 0, ∞ and NaN give false.
#[inline]
fn positive_finite(a: f64) -> bool { a.to_bits().wrapping_sub(1) < f64::INFINITY.to_bits() - 1 }

/// Whether y is a whole number, and whether it is an odd one. Infinities count as even whole numbers, and NaN as neither.
#[inline]
fn parity<R: Native>(y: R) -> (bool, bool) {
    let half = R::narrow(0.5) * y;
    let whole = y.trunc() == y;
    (whole, whole & (half.trunc() != half))
}

/// e^(y ln |x|), or e^(ln |x| / y) when `divide`. At f64, ln |x| is a double word from `table`, which keeps the error of the exponent
/// below an ulp of the result after the product or quotient with y. Narrower types compute in f64 with no table, so that loops
/// vectorize where the target has no gathers. There (|x| - 1) × ∞ stands for the logarithm that `ln_narrow` doesn't take: -∞ at 0,
/// ∞ at ∞, and NaN at NaN. It's large, and LLVM would otherwise leave it out of line, which would stop the loops that call it
/// vectorizing.
#[inline(always)]
fn power_of<R: Native, const FUSED: bool>(x: f64, y: f64, divide: bool) -> f64 {
    let a = x.abs();
    if R::WIDE {
        let l = table::ln::<FUSED>(a);
        let d = if divide { l.div::<FUSED>(Double::new(y, 0.0)) } else { l.mul_float::<FUSED>(y) };
        table::exp::<FUSED>(d.hi, d.lo)
    } else {
        let l = if positive_finite(a) { ln_narrow::<FUSED>(a) } else { (a - 1.0) * f64::INFINITY };
        exp_narrow::<FUSED>(if divide { l / y } else { l * y })
    }
}

/// r with the sign that a base x gives it: negated for a negative x and an `odd` power, and NaN for a negative, finite, nonzero x
/// where the power doesn't `exist`.
#[inline]
fn signed<R: Native>(r: R, x: R, odd: bool, exists: bool) -> R {
    let r = if x.is_sign_negative() & odd { -r } else { r };
    if (x < R::zero()) & (x > R::neg_infinity()) & !exists { R::nan() } else { r }
}

/// x^y. Special values follow IEEE 754's `pow`: x^0 and 1^y are 1, even for NaN, and a negative x has a power only for a whole y.
/// ln |x| is -∞ at 0 and ∞ at ∞, so e^(y ln |x|) is already right at zeros and infinities, and for an infinite y beside every |x|
/// but 1. Where y is 0 or |x| is 1, the magnitude is 1. e^(y ln |x|) would give NaN there wherever the product is 0 × ∞ or holds a
/// NaN.
#[inline(always)]
pub(crate) fn pow<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    let (whole, odd) = parity(y);
    let r = pow_plain::<R, FUSED>(x, y);
    signed(if (y == R::zero()) | (x.abs() == R::one()) { R::one() } else { r }, x, odd, whole)
}

/// x^y for a positive x other than 1 and a nonzero y, where `pow` needs none of its special cases. The slice form computes this for
/// every item, then recomputes with `pow` the few items where `plain` is false.
#[inline(always)]
fn pow_plain<R: Native, const FUSED: bool>(x: R, y: R) -> R { R::narrow(power_of::<R, FUSED>(x.wide(), y.wide(), false)) }

/// Whether `pow_plain` gives x^y: x is not negative, y is not 0 and |x| is not 1.
#[inline(always)]
fn plain<R: Native>(x: R, y: R) -> bool { !x.is_sign_negative() & (y != R::zero()) & (x != R::one()) }

/// The slice forms of this module's functions, which `functions!` makes the function objects' `many`.
pub(crate) mod slices {
    use crate::{float::Native, slice::{two_pass, Arg}};

    /// x^y for each pair. Every item takes `pow_plain`, and the rare items where `plain` fails take `pow` in a second pass.
    pub(crate) fn pow<R: Native, const FUSED: bool>(x: Arg<'_, R>, y: Arg<'_, R>, n: usize, out: &mut Vec<R>) -> bool {
        // Closures, because the `Fn` call of a function item such as `super::pow::<f64, true>` can stay out of line.
        two_pass(
            x, y, n, out,
            #[inline(always)] |a, b| super::pow::<R, FUSED>(a, b),
            #[inline(always)] |a, b| super::pow_plain::<R, FUSED>(a, b),
            #[inline(always)] |a, b| super::plain(a, b),
        )
    }
}

/// The real nth root of y, y^(1/n), computed as e^(ln |y| / n) without rounding 1/n first. A negative y has one only for an odd
/// whole n. n = 0, an infinite n and NaN give NaN.
#[inline(always)]
pub(crate) fn root<R: Native, const FUSED: bool>(n: R, y: R) -> R {
    let (_, odd) = parity(n);
    let r = signed(R::narrow(power_of::<R, FUSED>(y.wide(), n.wide(), true)), y, odd, odd);
    if (n == R::zero()) | !n.is_finite() { R::nan() } else { r }
}

/// The bits of |x|^(-1/3) plus a third of the bits of |x|, both read as whole numbers without their low 12 bits. The sum is nearly the
/// same for every positive normal x, and this choice keeps `cbrt`'s first estimate within 3.5%.
const CBRT_MAGIC: f64 = 1499661097332631.0;

/// The cube root of x, computed in f64. The bits of a positive float, read as a whole number, grow nearly in proportion to its
/// logarithm, so `CBRT_MAGIC` less a third of |x|'s bits gives the bits of a z near |x|^(-1/3). Newton steps with no division square
/// z's error: three at f64 and two at narrower widths. y = |x| z² is then ∛|x|, which a last step corrects: with ε = y z - 1, ∛|x| is
/// close to y(1 - 2ε/3). A subnormal f64 scales up by 2^54 first, and its root down by 2^18 last.
#[inline(always)]
pub(crate) fn cbrt<R: Native, const FUSED: bool>(x: R) -> R {
    let a = x.wide().abs();
    let sub = R::WIDE & (a < f64::MIN_POSITIVE);
    let b = if sub { a * 18014398509481984.0 } else { a };
    // b's bits, less the low 12, as a float through the bits of 2^52 plus them, and z's bits back through the same sum.
    let t = (f64::from_bits(TWO_52.to_bits() | (b.to_bits() >> 12)) - TWO_52).mla::<FUSED>(-1.0 / 3.0, CBRT_MAGIC);
    let mut z = f64::from_bits((t + TWO_52).to_bits() << 12);
    for _ in 0..if R::WIDE { 3 } else { 2 } {
        z = z * ((b * z) * (z * z)).mla::<FUSED>(-1.0 / 3.0, 4.0 / 3.0);
    }
    let y = b * z * z;
    // Without FMA, y z rounds before the subtraction, which loses ε's low bits. At f64 a double-word product keeps them, so that an
    // exact cube gives its root exactly. Narrower results round far coarser.
    let e = if FUSED | !R::WIDE { y.mla::<FUSED>(z, -1.0) } else { let p = Double::new(y, 0.0).mul_float::<FUSED>(z); (p.hi - 1.0) + p.lo };
    let y = (y * (-2.0 / 3.0)).mla::<FUSED>(e, y) * if sub { 3.814697265625e-6 } else { 1.0 };
    // 0, ∞ and NaN give themselves.
    let y = if positive_finite(a) { y } else { a };
    R::narrow(y.copysign(x.wide()))
}

/// √(x² + y²), without overflow or underflow in between. At f64, both arguments first scale by the power of two that brings the larger
/// into [2, 4), which is exact. The power and its inverse come from the larger's exponent bits with no branch. Raising a larger below
/// 2^-1021 to that bound keeps both normal. Narrower types compute in f64, whose range holds their squares. An infinite x or y gives ∞,
/// even beside NaN.
#[inline]
pub(crate) fn hypot<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    const BOTTOM: f64 = f64::from_bits(2 << 52);
    let (a, b) = (x.wide().abs(), y.wide().abs());
    let (s, back) = if R::WIDE {
        let big = if a > b { a } else { b };
        let e = (if big < BOTTOM { BOTTOM } else { big }).to_bits() >> 52;
        (f64::from_bits((2047 - e) << 52), f64::from_bits((e - 1) << 52))
    } else {
        (1.0, 1.0)
    };
    let (sa, sb) = (a * s, b * s);
    let r = sa.mla::<FUSED>(sa, sb * sb).sqrt() * back;
    R::narrow(if a.is_infinite() | b.is_infinite() { f64::INFINITY } else { r })
}
