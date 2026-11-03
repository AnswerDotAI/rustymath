//! Powers and roots use the shared Exp/Ln bodies, except cheap constant powers and square/cube roots. Constant integer powers retain their exact cases. General power error grows with the rounding error in y·ln|x|; it is not uniformly bounded in ulps as |y·ln|x|| grows.
use crate::{buffer::FloatWidth, float::Native, recipe::{Cmp, Integer, Ops, root2}};

#[inline]
pub(crate) fn parity<R: Native>(y: R) -> (bool, bool) {
    let half = R::narrow(0.5) * y;
    let whole = y.trunc() == y;
    (whole, whole & (half.trunc() != half))
}

pub(crate) fn formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V {
    if let Some(k) = o.constant(y) { if let Some(result) = crate::recipe::power(o, Exponent::of(k.wide()), x) { return result; } }
    general(o, x, y, false)
}

pub(crate) fn root_formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, n: O::V, y: O::V) -> O::V {
    if let Some(n) = o.constant(n) {
        let result = match n.wide().abs() {
            1.0 => Some(y),
            2.0 => Some(root2(o, y)),
            3.0 => Some(cbrt_formula(o, y)),
            4.0 => Some(root2(o, root2(o, y))),
            8.0 => Some(root2(o, root2(o, root2(o, y)))),
            0.5 => {
                let x = if n < F::zero() { o.reciprocal(y) } else { y };
                return o.select(o.less(y, o.number(F::zero())), o.number(F::nan()), o.multiply(x, x));
            }
            _ => None,
        };
        if let Some(result) = result { return if n < F::zero() { o.reciprocal(result) } else { result }; }
    }
    general(o, y, n, true)
}

// A more accurate compensated implementation is in a1b7044cd3416194bf5dc423f197280c6933c9b9:src/pow.rs and src/table.rs; it can replace this formula if greater accuracy is needed.
fn general<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V, root: bool) -> O::V {
    let (zero, one, half) = (o.number(F::zero()), o.number(F::one()), o.number(F::narrow(0.5)));
    let a = o.magnitude(x);
    let l = crate::ln::formula(o, a);
    let power = if root { o.divide(l, y) } else { o.multiply(y, l) };
    let r = crate::exp::formula(o, power);
    let whole = o.equal(o.floor(y), y);
    let h = o.multiply(y, half);
    let odd = o.and(whole, o.not(o.equal(o.floor(h), h)));
    let sign = o.integer(if o.width() == FloatWidth::F64 { i64::MIN } else { 1 << 31 });
    let negative = o.not(o.integer_compare(Cmp::Equal, o.integer_op(Integer::And, o.to_bits(x), sign), o.integer(0)));
    let r = if root { r } else { o.select(o.or(o.equal(y, zero), o.equal(a, one)), one, r) };
    let r = o.select(o.and(negative, odd), o.negate(r), r);
    let exists = if root { odd } else { whole };
    let bad_base = o.and(o.and(o.less(x, zero), o.less(a, o.number(F::infinity()))), o.not(exists));
    let bad = if root { o.or(bad_base, o.or(o.equal(y, zero), o.not(o.less(o.magnitude(y), o.number(F::infinity()))))) } else { bad_base };
    o.select(bad, o.number(F::nan()), r)
}

#[inline]
pub(crate) fn pow<R: Native, const FUSED: bool>(x: R, y: R) -> R { formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x, y) }
#[inline]
pub(crate) fn root<R: Native, const FUSED: bool>(n: R, y: R) -> R { root_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, n, y) }

/// The greatest whole exponent that `pow` raises to by squaring and multiplying in single words (`recipe::multiplied`). Without fused multiply-add, larger exponents take the general route.
pub(crate) const MULTIPLIED: u32 = 8;

/// The greatest whole exponent that `pow` raises to by squaring and multiplying where the target fuses multiply-add (`FUSED`). Above `MULTIPLIED`, every step but those of bit position 0 runs in double words (`recipe::whole`). Larger exponents take the general route. From 64 on, squaring and multiplying is slower than the general route for some exponents.
pub(crate) const COMPENSATED: u32 = 63;

/// How [`Pow`] raises to an exponent y. Its slice form reads it once for a constant y, and its scalar form for each item. The JIT's lowering of a constant power reads it too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exponent {
    /// x^m, or x^-m where `negative`, by squaring and multiplying (`recipe::power`): a whole y of magnitude m from 1 to `MULTIPLIED`, or to `COMPENSATED` where the target fuses multiply-add. No result fails.
    Whole { m: u32, negative: bool },
    /// √x^m, or 1/√x^m where `negative`: a y of m halves, for m of 1 or 3. √x^3 carries √x's rounding three times and two roundings of its own, within 4 ulps. 1/√x^3 adds the reciprocal's rounding and passes 4 ulps, as rustymath's accuracy page lists. √x gives `pow`'s values at -0 and -∞, and a negative finite x has no real power, as on the general route.
    Halves { m: u32, negative: bool },
    /// Any other y: e^(y ln |x|).
    General,
}

impl Exponent {
    /// The route for y.
    #[inline(always)]
    pub fn of(y: f64) -> Self {
        let (a, negative) = (y.abs(), y < 0.0);
        // Clamped to the range and truncated, a is unchanged only if it's a whole number in that range. That one comparison is one branch. Separate tests compile to separate branches on x86, which mispredict for exponents in a random order.
        let most = if crate::FUSED { COMPENSATED } else { MULTIPLIED };
        let m = a.max(1.0).min(most.into()).trunc();
        if m == a { return Self::Whole { m: m as u32, negative }; }
        let halves = a + a;
        if (halves == 1.0) | (halves == 3.0) { Self::Halves { m: halves as u32, negative } } else { Self::General }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Pow;
#[derive(Clone, Copy, Debug)]
pub struct WrappingPow;
#[derive(Clone, Copy, Debug)]
pub struct WholePow;
#[derive(Clone, Copy, Debug)]
pub struct RealPow;

/// The real part of the principal xʸ: xʸ at and above 0, and |x|ʸ cos πy below 0. cos πy is exact where y is whole or a half (`recipe::sincospi`).
#[inline(always)]
pub(crate) fn real_pow<R: Native, const FUSED: bool>(x: R, y: R) -> R { real_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x, y) }

pub(crate) fn real_formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V {
    let r = formula(o, o.magnitude(x), y);
    let real = o.multiply(r, crate::recipe::sincospi(o, y).1);
    o.select(o.less(x, o.number(F::zero())), real, r)
}

/// [`root`](crate::root) as a function object, with a scalar form and a slice form that work as [`Pow`]'s do. A constant degree of 2 or 3 takes `sqrt` or `cbrt`.
#[derive(Clone, Copy, Debug)]
pub struct Root;

/// The top bits of |x|^(-1/3) plus a third of those of |x|, at f64 and at f32, as `top` reads them. The sum is nearly the same for every positive normal x. These values keep `cbrt`'s first estimate within 3.5%.
const CBRT_MAGIC: [f64; 2] = [1499661097332631.0, 2773273.0];

/// Two or three Newton steps from an inverse-cube-root bit estimate, followed by the product correction that preserves exact cubes.
#[inline(always)]
pub(crate) fn cbrt<R: Native, const FUSED: bool>(x: R) -> R { cbrt_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, x) }

pub(crate) fn cbrt_formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V {
    use crate::{buffer::FloatWidth, recipe::Integer};
    let c = |x| o.number(F::narrow(x));
    let wide = o.width() == FloatWidth::F64;
    let (mantissa, bits, least, up, down) =
        if wide { (52, 64, f64::MIN_POSITIVE, 18014398509481984.0, 3.814697265625e-6) } else { (23, 32, f32::MIN_POSITIVE as f64, 16777216.0, 0.00390625) };
    let a = o.magnitude(x);
    let sub = o.less(a, c(least));
    let b = o.select(sub, o.multiply(a, c(up)), a);
    let top = o.from_integer(o.integer_op(Integer::ShiftRight, o.to_bits(b), o.integer(bits - mantissa)));
    let t = o.multiply_add(top, c(-1.0 / 3.0), c(CBRT_MAGIC[usize::from(!wide)]));
    let rounded = o.to_bits(o.add(t, c((1u64 << mantissa) as f64)));
    let rounded = o.integer_op(Integer::And, rounded, o.integer((1i64 << mantissa) - 1));
    let mut z = o.from_bits(o.integer_op(Integer::ShiftLeft, rounded, o.integer(bits - mantissa)));
    for _ in 0..if wide { 3 } else { 2 } { z = o.multiply(z, o.multiply_add(o.multiply(o.multiply(b, z), o.multiply(z, z)), c(-1.0 / 3.0), c(4.0 / 3.0))); }
    let y = o.multiply(o.multiply(b, z), z);
    let e = if o.fused() { o.multiply_add(y, z, c(-1.0)) } else {
        let mask = o.integer(!((1i64 << ((mantissa + 2) / 2)) - 1));
        let yh = o.from_bits(o.integer_op(Integer::And, o.to_bits(y), mask));
        let zh = o.from_bits(o.integer_op(Integer::And, o.to_bits(z), mask));
        let (yl, zl) = (o.subtract(y, yh), o.subtract(z, zh));
        let p = o.multiply(y, z);
        let error = o.add(o.add(o.add(o.subtract(o.multiply(yh, zh), p), o.multiply(yl, zh)), o.multiply(yh, zl)), o.multiply(yl, zl));
        o.add(o.subtract(p, c(1.0)), error)
    };
    let y = o.multiply(o.multiply_add(o.multiply(y, c(-2.0 / 3.0)), e, y), o.select(sub, c(down), c(1.0)));
    let finite = o.and(o.less(c(0.0), a), o.less(a, c(f64::INFINITY)));
    let y = o.select(finite, y, a);
    let sign = o.integer_op(Integer::And, o.to_bits(x), o.integer(if wide { i64::MIN } else { i32::MIN as u32 as i64 }));
    o.from_bits(o.integer_op(Integer::Or, o.to_bits(y), sign))
}

/// √(x² + y²), without overflow or underflow in between. Both arguments first scale by the power of two that `scales` gives for the larger, which brings it into [2, 4) exactly. Each type computes at its own width. An infinite x or y gives ∞, even beside NaN.
#[inline]
pub(crate) fn hypot<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    let (a, b) = (x.abs(), y.abs());
    let (s, back) = (if a > b { a } else { b }).scales();
    let (sa, sb) = (a * s, b * s);
    let r = sa.mla::<FUSED>(sa, sb * sb).sqrt() * back;
    if a.is_infinite() | b.is_infinite() { R::infinity() } else { r }
}
