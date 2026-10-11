//! Powers and roots. `pow` and `root` compute e^(y ln |x|) through `table`'s logarithm and exponential. `cbrt` refines an estimate
//! taken from x's bits with Newton steps. `hypot` scales both arguments by a power of two before squaring them. See `THIRD-PARTY.md`.
use crate::{
    dd::Double,
    float::{sealed::Core, Native},
    table,
};

/// 2^52. Every float of this magnitude is a whole number, and adding it to a smaller magnitude rounds that to a whole number.
const TWO_52: f64 = 4503599627370496.0;

/// Whether y is a whole number, and whether it is an odd one. Infinities are neither. Below 2^52, adding 2^52 rounds |y| to a whole
/// number whose lowest bit is its parity. From 2^52 to 2^53 every float is whole, with its parity in its lowest bit, and beyond that
/// every float is even.
#[inline]
fn parity(y: f64) -> (bool, bool) {
    let a = y.abs();
    let small = a < TWO_52;
    let t = if small { a + TWO_52 } else { a };
    let w = (!small | (t - TWO_52 == a)) & y.is_finite();
    (w, w & (a < 2.0 * TWO_52) & (t.to_bits() & 1 == 1))
}

/// The coefficients of log2((1+s)/(1-s))/s as a polynomial in s², highest degree first, for |s| up to 1/5. They keep about 2^-35
/// relative, because the exponent multiplies the logarithm's error.
const LOG2_NARROW: [f64; 5] = [0.3483427456720679, 0.41121029218967053, 0.5770922466072469, 0.9617966225204624, 2.88539008183509];
/// The coefficients of 2^r, highest degree first, for |r| up to 1/2.
const EXP2_NARROW: [f64; 7] =
    [0.0001546144469856913, 0.0013400428177615838, 0.009618056678524637, 0.05550327226670302, 0.24022650922288757, 0.6931472067028326, 1.0];

/// log2 x, for an x that a narrower float widened to: 0, ∞, NaN or a normal f64. x = m × 2^e with m from 0.75 to 1.5, and
/// log2 m = s P(s²) with s = (m-1)/(m+1).
#[inline]
fn log2_narrow<const FUSED: bool>(x: f64) -> f64 {
    let (m, e) = x.split();
    let s = (m - 1.0) / (m + 1.0);
    // Estrin's form, which shortens the chain of dependent operations.
    let [a4, a3, a2, a1, a0] = LOG2_NARROW;
    let u = s * s;
    let u2 = u * u;
    let p = u2.mla::<FUSED>(a4, u.mla::<FUSED>(a3, a2)).mla::<FUSED>(u2, u.mla::<FUSED>(a1, a0));
    let y = s.mla::<FUSED>(p, e);
    let y = if x.is_nan() | (x == f64::INFINITY) { x } else { y };
    if x == 0.0 { f64::NEG_INFINITY } else { y }
}

/// 2^v within about 2^-28 relative, far inside half an ulp of a narrower float, so a power whose exact result is a narrower float
/// comes out exactly. Beyond 160 in magnitude, the result is 0 or ∞ at those widths.
#[inline]
fn exp2_narrow<const FUSED: bool>(v: f64) -> f64 {
    let v = if v < -160.0 { -160.0 } else if v > 160.0 { 160.0 } else { v };
    let s = v + f64::SHIFT;
    let r = v - (s - f64::SHIFT);
    // Estrin's form, which shortens the chain of dependent operations.
    let [e6, e5, e4, e3, e2, e1, e0] = EXP2_NARROW;
    let r2 = r * r;
    let high = r2.mla::<FUSED>(e6, r.mla::<FUSED>(e5, e4));
    r2.mla::<FUSED>(r2.mla::<FUSED>(high, r.mla::<FUSED>(e3, e2)), r.mla::<FUSED>(e1, e0)) * f64::pow2(s)
}

/// e^(y ln |x|), or e^(ln |x| / y) when `divide`. At f64, ln |x| is a double word from `table`, which keeps the error of the exponent
/// below an ulp of the result after the product or quotient with y. Narrower types compute 2^(y log2 |x|) in plain f64, with no table,
/// so that loops vectorize where the target has no gathers. It's large, and LLVM would otherwise leave it out of line, which would
/// stop the loops that call it vectorizing.
#[inline(always)]
fn power_of<R: Native, const FUSED: bool>(x: f64, y: f64, divide: bool) -> f64 {
    if R::WIDE {
        let l = table::ln::<FUSED>(x.abs());
        let d = if divide { l.div::<FUSED>(Double::new(y, 0.0)) } else { l.mul_float::<FUSED>(y) };
        table::exp::<FUSED>(d.hi, d.lo)
    } else {
        let l = log2_narrow::<FUSED>(x.abs());
        exp2_narrow::<FUSED>(if divide { l / y } else { l * y })
    }
}

/// The factor that the sign of a base x gives a power: 1 for a positive x, -1 for a negative x and an odd power, 1 where the power
/// `exists` whatever the sign of x, and NaN otherwise.
#[inline]
fn sign(x: f64, odd: bool, exists: bool) -> f64 {
    if !x.is_sign_negative() { 1.0 } else if odd { -1.0 } else if exists { 1.0 } else { f64::NAN }
}

/// x^y. Special values follow IEEE 754's `pow`: x^0 and 1^y are 1, even for NaN, and a negative x has a power
/// only for a whole y. ln |x| is -∞ at 0 and ∞ at ∞, so e^(y ln |x|) is already right at zeros and infinities, and for an
/// infinite y beside every |x| but 1.
#[inline(always)]
pub(crate) fn pow<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    let (x, y) = (x.wide(), y.wide());
    let (whole, odd) = parity(y);
    let r = power_of::<R, FUSED>(x, y, false) * sign(x, odd, whole | (x == 0.0) | x.is_infinite() | y.is_infinite());
    R::narrow(if (y == 0.0) | (x == 1.0) | ((x == -1.0) & y.is_infinite()) { 1.0 } else { r })
}

/// The real nth root of y, y^(1/n), computed as e^(ln |y| / n) without rounding 1/n first. A negative y has one only for an odd
/// whole n. n = 0, an infinite n and NaN give NaN.
#[inline(always)]
pub(crate) fn root<R: Native, const FUSED: bool>(n: R, y: R) -> R {
    let (n, y) = (n.wide(), y.wide());
    let (_, odd) = parity(n);
    let r = power_of::<R, FUSED>(y, n, true) * sign(y, odd, (y == 0.0) | y.is_infinite());
    R::narrow(if (n == 0.0) | n.is_infinite() | n.is_nan() { f64::NAN } else { r })
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
    let y = if a.to_bits().wrapping_sub(1) >= f64::INFINITY.to_bits() - 1 { a } else { y };
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
