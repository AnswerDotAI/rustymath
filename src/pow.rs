//! Powers and roots use the shared Exp/Ln bodies, except cheap constant powers and square/cube roots. Runtime powers keep that same dispatch through the native kernel when a backend cannot emit it. General power error grows with the rounding error in y·ln|x|; it is not uniformly bounded in ulps as |y·ln|x|| grows.
use crate::{buffer::FloatWidth, float::Native, recipe::{Cmp, Integer, Ops, root2}};

#[inline]
pub(crate) fn parity<R: Native>(y: R) -> (bool, bool) {
    let half = R::narrow(0.5) * y;
    let whole = y.trunc() == y;
    (whole, whole & (half.trunc() != half))
}

pub(crate) fn formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V {
    let Some(k) = o.constant(y) else { return o.binary(crate::recipe::Binary::Power, x, y); };
    if let Some(result) = crate::recipe::power(o, Exponent::of(k.wide()), x) { return result; }
    general(o, x, y, false)
}

pub(crate) fn root_formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, n: O::V, y: O::V) -> O::V {
    match o.constant(n) { Some(k) => rooted(o, Degree::of(k.wide()), k < F::zero(), n, y), None => general(o, y, n, true) }
}

/// The route [`Root`] takes for a degree n, chosen by n's magnitude. `General` takes e^(ln|y|/n) for a degree with no route of its own. The slice form reads the route once for a constant n.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Degree { One, Two, Three, Four, Eight, Half, General }

impl Degree {
    /// The route for degree n.
    #[inline(always)]
    fn of(n: f64) -> Self {
        match n.abs() { 1.0 => Self::One, 2.0 => Self::Two, 3.0 => Self::Three, 4.0 => Self::Four, 8.0 => Self::Eight, 0.5 => Self::Half, _ => Self::General }
    }
}

/// The root of degree n of y by n's route `d`. A negative n (`negative`) takes the reciprocal of the route's root. For `Half` it squares y's reciprocal instead.
#[inline(always)]
fn rooted<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, d: Degree, negative: bool, n: O::V, y: O::V) -> O::V {
    let r = match d {
        Degree::One => y,
        Degree::Two => root2(o, y),
        Degree::Three => cbrt_formula(o, y),
        Degree::Four => root2(o, root2(o, y)),
        Degree::Eight => root2(o, root2(o, root2(o, y))),
        Degree::Half => {
            let x = if negative { o.reciprocal(y) } else { y };
            return o.select(o.less(y, o.number(F::zero())), o.number(F::nan()), o.multiply(x, x));
        }
        Degree::General => return general(o, y, n, true),
    };
    if negative { o.reciprocal(r) } else { r }
}

// A more accurate compensated implementation is in a1b7044cd3416194bf5dc423f197280c6933c9b9:src/pow.rs and src/table.rs; it can replace this formula if greater accuracy is needed.
#[inline]
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

/// `f` of each of `xs` into `out`, at the working precision. Each route's loop inlines it and vectorizes.
#[inline(always)]
fn each<F: crate::Float>(xs: &[F], out: &mut [std::mem::MaybeUninit<F>], f: impl Fn(F::Working) -> F::Working) {
    for (r, &x) in out.iter_mut().zip(xs) { r.write(F::from_working(f(F::working(x)))); }
}

/// `x^k` into `out` for each of `xs`, with `k`'s route (`Exponent::of`) chosen once, at the working precision. Each whole power up to `MULTIPLIED`, each power of halves and the general route have a loop of their own, so squarings unroll and each loop vectorizes. Larger whole powers share one loop.
pub(crate) fn powers<F: crate::Float>(e: Exponent, k: F, xs: &[F], out: &mut [std::mem::MaybeUninit<F>]) {
    #[inline(always)]
    fn route<F: crate::Float>(e: Exponent, xs: &[F], out: &mut [std::mem::MaybeUninit<F>]) {
        let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
        each(xs, out, #[inline(always)] |x| crate::recipe::power(&o, e, x).expect("a route off the general route"));
    }
    macro_rules! routes { ($($kind:ident $m:literal),+) => { match e {
        $(Exponent::$kind { m: $m, negative: false } => route(Exponent::$kind { m: $m, negative: false }, xs, out),
          Exponent::$kind { m: $m, negative: true } => route(Exponent::$kind { m: $m, negative: true }, xs, out),)+
        Exponent::General => {
            let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
            let k = o.number(F::working(k));
            each(xs, out, #[inline(always)] |x| general(&o, x, k, false))
        }
        e => route(e, xs, out),
    } } }
    routes!(Whole 1, Whole 2, Whole 3, Whole 4, Whole 5, Whole 6, Whole 7, Whole 8, Halves 1, Halves 3);
}

/// [`Pow`]'s slice form: [`powers`] for an exponent that every item shares, else `general` for every item, then the exact route for each item whose exponent has one. It gives whether every result is valid, as `power_valid` judges results: not NaN, or from a NaN argument.
pub(crate) fn power_slice<F: crate::float::Native>(_: &Pow, [x, y]: [crate::Arg<'_, F>; 2], n: usize, out: &mut [std::mem::MaybeUninit<F>]) -> bool {
    if let (crate::Arg::Many(xs), crate::Arg::One(k)) = (x, y) {
        let e = Exponent::of(k.wide());
        powers(e, k, &xs[..n], &mut out[..n]);
        return !matches!(e, Exponent::Halves { .. } | Exponent::General) || k.is_nan() || xs[..n].iter().zip(crate::slice::written(&mut out[..n]).iter()).all(|(x, r)| !r.is_nan() || x.is_nan());
    }
    // Exponents that vary: `general` for every item in one loop that vectorizes, then each item whose exponent has a route of its own (`Exponent::of`) takes that route, as `formula` would.
    let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
    crate::slice::mapped(&[x, y], n, out, #[inline(always)] |[a, b]| (general(&o, a, b, false), true));
    let results = crate::slice::written(&mut out[..n]);
    for (i, r) in results.iter_mut().enumerate() {
        if let Some(exact) = crate::recipe::power(&o, Exponent::of(y.nth(i).wide()), x.nth(i)) { *r = exact; }
    }
    results.iter().enumerate().all(|(i, r)| !r.is_nan() || x.nth(i).is_nan() || y.nth(i).is_nan())
}

/// The root of degree `n` of each of `ys` into `out`, at the working precision, with `n`'s route (`Degree::of`) chosen once. Each route and sign has its own loop, which vectorizes.
pub(crate) fn roots<F: crate::Float>(n: F, ys: &[F], out: &mut [std::mem::MaybeUninit<F>]) {
    let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
    let k = o.number(F::working(n));
    macro_rules! routes { ($($d:ident),+) => { match (Degree::of(n.wide()), n < F::zero()) {
        $((Degree::$d, false) => each(ys, out, #[inline(always)] |y| rooted(&o, Degree::$d, false, k, y)),
          (Degree::$d, true) => each(ys, out, #[inline(always)] |y| rooted(&o, Degree::$d, true, k, y)),)+
    } } }
    routes!(One, Two, Three, Four, Eight, Half, General);
}

/// [`Root`]'s slice form: [`roots`] for a degree that every item shares, else [`root_formula`] for each item. It gives whether every result is valid: not NaN, or from a NaN argument.
pub(crate) fn root_slice<F: crate::float::Native>(_: &Root, [n, y]: [crate::Arg<'_, F>; 2], len: usize, out: &mut [std::mem::MaybeUninit<F>]) -> bool {
    if let (crate::Arg::One(k), crate::Arg::Many(ys)) = (n, y) {
        roots(k, &ys[..len], &mut out[..len]);
        return matches!(Degree::of(k.wide()), Degree::One | Degree::Three) || k.is_nan() || ys[..len].iter().zip(crate::slice::written(&mut out[..len]).iter()).all(|(y, r)| !r.is_nan() || y.is_nan());
    }
    let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
    crate::slice::mapped(&[n, y], len, out, #[inline(always)] |[a, b]| { let r = root_formula(&o, a, b); (r, !r.is_nan() || a.is_nan() || b.is_nan()) })
}

#[derive(Clone, Copy, Debug)]
pub struct Pow;
#[derive(Clone, Copy, Debug)]
pub struct RealPow;

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
