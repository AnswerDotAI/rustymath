//! Powers and roots use compensated native block forms for their general cases and shared recipes for cheap constant exponents and degrees. Native block loops compute the usual case first, then repair exceptional items. See `THIRD-PARTY.md`.
use crate::{dd::Double, exp::exp_narrow, float::Native, ln::ln_narrow, recipe::{Ops, root2}, table};
/// Whether a is finite and nonzero, for an a whose sign bit is clear. 0, ∞ and NaN give false.
#[inline]
fn positive_finite(a: f64) -> bool { a.to_bits().wrapping_sub(1) < f64::INFINITY.to_bits() - 1 }

#[inline]
pub(crate) fn parity<R: Native>(y: R) -> (bool, bool) {
    let half = R::narrow(0.5) * y;
    let whole = y.trunc() == y;
    (whole, whole & (half.trunc() != half))
}

/// Cheap constant powers lower to the shared recipe; general powers stay native.
pub(crate) fn formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V {
    if let Some(k) = o.constant(y) { if let Some(result) = crate::recipe::power(o, Exponent::of(k.wide(), o.fused()), x) { return result; } }
    o.binary(crate::recipe::Binary::Power, x, y)
}

/// Only degrees with a short recipe enter a fused stage. Other roots use the native bulk form.
pub(crate) fn root_formula<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, n: O::V, y: O::V) -> Option<O::V> {
    let k = o.constant(n)?;
    rooted(o, Degree::of(k.wide()), k < F::zero(), y)
}

/// The route [`Root`] takes for a degree n, chosen by n's magnitude. The slice form reads the route once for a constant n.
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
fn rooted<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, d: Degree, negative: bool, y: O::V) -> Option<O::V> {
    let r = match d {
        Degree::One => y,
        Degree::Two => root2(o, y),
        Degree::Three => cbrt_formula(o, y),
        Degree::Four => root2(o, root2(o, y)),
        Degree::Eight => root2(o, root2(o, root2(o, y))),
        Degree::Half => {
            let x = if negative { o.reciprocal(y) } else { y };
            return Some(o.select(o.less(y, o.number(F::zero())), o.number(F::nan()), o.multiply(x, x)));
        }
        Degree::General => return None,
    };
    Some(if negative { o.reciprocal(r) } else { r })
}

/// e^(y ln |x|), or e^(ln |x| / y) when `divide`. At f64, ln |x| is a double word from `table`, which keeps the error of the exponent below an ulp of the result after the product or quotient with y. Narrower types compute in f64 with no table, so that loops vectorize where the target has no gathers. There (|x| - 1) × ∞ stands for the logarithm that `ln_narrow` doesn't take: -∞ at 0, ∞ at ∞, and NaN at NaN. It's large, and LLVM would otherwise leave it out of line, which would stop the loops that call it vectorizing.
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

/// r with the sign that a base x gives it: negated for a negative x and an `odd` power, and NaN for a negative, finite, nonzero x where the power doesn't `exist`.
#[inline]
fn signed<R: Native>(r: R, x: R, odd: bool, exists: bool) -> R {
    let r = if x.is_sign_negative() & odd { -r } else { r };
    if (x < R::zero()) & (x > R::neg_infinity()) & !exists { R::nan() } else { r }
}
/// x^y, or the yth root of x when `divide`, where neither function needs a special case, or `None`. x must be positive, finite and nonzero, and y `usable`. At f64 the exponent must also be within `table::NORMAL`, where the result is a normal float. That path skips the selects for special values in `table`'s logarithm and exponential. No branch.
#[inline(always)]
fn fast<R: Native, const FUSED: bool>(x: R, y: R, divide: bool) -> Option<R> {
    let (a, y) = (x.wide(), y.wide());
    let (r, normal) = if R::WIDE {
        let l = table::ln_normal::<FUSED>(a);
        let d = if divide { l.div::<FUSED>(Double::new(y, 0.0)) } else { l.mul_float::<FUSED>(y) };
        (table::exp_normal::<FUSED>(d.hi, d.lo), d.hi.abs() < table::NORMAL)
    } else { let l = ln_narrow::<FUSED>(a); (exp_narrow::<FUSED>(if divide { l / y } else { l * y }), true) };
    // A narrower float widens to a normal f64, so this tests positive, finite and nonzero at every width.
    let base = a.to_bits().wrapping_sub(f64::MIN_POSITIVE.to_bits()) < f64::INFINITY.to_bits() - f64::MIN_POSITIVE.to_bits();
    (base & normal & usable(y, divide)).then_some(R::narrow(r))
}

/// `fast` for one f32 or f16 value, through `table`'s single-precision logarithm and exponential, as Arm's scalar `powf` does. x must be a positive normal f32, y `usable`, and the exponent below 124 in magnitude, where the result is a normal f32.
#[inline(always)]
fn fast_single<R: Native, const FUSED: bool>(x: R, y: R, divide: bool) -> Option<R> {
    let (ix, y) = ((x.wide() as f32).to_bits(), y.wide());
    let l = table::log2_single::<FUSED>(ix);
    let v = if divide { l / y } else { l * y };
    let normal = ix.wrapping_sub(0x0080_0000) < 0x7f80_0000 - 0x0080_0000;
    // Below 124, |v| has top bits below 126's, which is how Arm's `powf` tests that the result is a normal f32.
    let small = (v.to_bits() >> 47 & 0xffff) < 126f64.to_bits() >> 47;
    (normal && usable(y, divide) && small).then(|| R::narrow(table::exp2_single::<FUSED>(v)))
}

/// Whether y can be the exponent of a fast path: finite, and nonzero as a root's degree.
#[inline(always)]
fn usable(y: f64, divide: bool) -> bool { y.is_finite() & !(divide & (y == 0.0)) }

/// `pow` for any arguments, with no branch. ln |x| is -∞ at 0 and ∞ at ∞, so e^(y ln |x|) is already right at zeros and infinities, and for an infinite y beside every |x| but 1. Where y is 0 or |x| is 1, the magnitude is 1. e^(y ln |x|) would give NaN there wherever the product is 0 × ∞ or holds a NaN.
#[inline(always)]
fn pow_any<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    let (whole, odd) = parity(y);
    let r = R::narrow(power_of::<R, FUSED>(x.wide(), y.wide(), false));
    signed(if (y == R::zero()) | (x.abs() == R::one()) { R::one() } else { r }, x, odd, whole)
}

/// `root` for any arguments, with no branch.
#[inline(always)]
fn root_any<R: Native, const FUSED: bool>(n: R, y: R) -> R {
    let (_, odd) = parity(n);
    let r = signed(R::narrow(power_of::<R, FUSED>(y.wide(), n.wide(), true)), y, odd, odd);
    if (n == R::zero()) | !n.is_finite() { R::nan() } else { r }
}

/// Scalar powers retain the same exact exponent shortcuts as their block form.
#[inline]
pub(crate) fn pow<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    let o = crate::recipe::NativeOps::<FUSED>::DEFAULT;
    if let Some(r) = crate::recipe::power(&o, Exponent::of(y.wide(), FUSED), x) { return r; }
    let r = if R::WIDE { fast::<R, FUSED>(x, y, false) } else { fast_single::<R, FUSED>(x, y, false) };
    r.unwrap_or_else(|| pow_any::<R, FUSED>(x, y))
}

/// Scalar roots retain the same short degree recipes as their block form.
#[inline]
pub(crate) fn root<R: Native, const FUSED: bool>(n: R, y: R) -> R {
    if let Some(r) = root_formula(&crate::recipe::NativeOps::<FUSED>::DEFAULT, n, y) { return r; }
    let r = if R::WIDE { fast::<R, FUSED>(y, n, true) } else { fast_single::<R, FUSED>(y, n, true) };
    r.unwrap_or_else(|| root_any::<R, FUSED>(n, y))
}

/// The greatest positive whole exponent that `pow` raises to by squaring and multiplying in single words (`recipe::multiplied`). Larger powers and negative powers below -1 use compensation where multiply-add fuses, otherwise the general route.
pub(crate) const MULTIPLIED: u32 = 8;

/// The greatest whole exponent that `pow` raises to by squaring and multiplying where the target fuses multiply-add (`FUSED`). Above `MULTIPLIED`, every step but those of bit position 0 runs in double words (`recipe::whole`). Larger exponents take the general route. From 64 on, squaring and multiplying is slower than the general route for some exponents.
pub(crate) const COMPENSATED: u32 = 63;

/// How [`Pow`] raises to an exponent y. Its slice form reads it once for a constant y, and its scalar form for each item. The JIT's lowering of a constant power reads it too.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Exponent {
    /// x^m, or x^-m where `negative`, through `recipe::power`. Positive m goes to `MULTIPLIED`, or `COMPENSATED` with fused multiply-add. Negative powers below -1 need fused multiply-add. No result fails.
    Whole { m: u32, negative: bool },
    /// √x^m, or 1/√x^m where `negative`, for m of 1 or 3. Negative three halves needs fused multiply-add for compensated products. √x gives `pow`'s values at -0 and -∞, and a negative finite x has no real power.
    Halves { m: u32, negative: bool },
    /// Any other y: e^(y ln |x|).
    General,
}

impl Exponent {
    /// The route for y, using compensated products only when the operations fuse multiply-add.
    #[inline(always)]
    pub fn of(y: f64, fused: bool) -> Self {
        let (a, negative) = (y.abs(), y < 0.0);
        // Clamped to the range and truncated, a is unchanged only if it's a whole number in that range. That one comparison is one branch. Separate tests compile to separate branches on x86, which mispredict for exponents in a random order.
        let most = if fused { COMPENSATED } else if negative { 1 } else { MULTIPLIED };
        let m = a.max(1.0).min(most.into()).trunc();
        if m == a { return Self::Whole { m: m as u32, negative }; }
        let halves = a + a;
        if (halves == 1.0) | ((halves == 3.0) & (!negative | fused)) { Self::Halves { m: halves as u32, negative } } else { Self::General }
    }
}

/// `f` of each of `xs` into `out`, at the working precision. Each route's loop inlines it and vectorizes.
#[inline(always)]
fn each<F: crate::Float>(xs: &[F], out: &mut [std::mem::MaybeUninit<F>], f: impl Fn(F::Working) -> F::Working) {
    for (r, &x) in out.iter_mut().zip(xs) { r.write(F::from_working(f(F::working(x)))); }
}

/// Shared block pass and exceptional-item repair for general powers and roots, including f16 storage.
#[inline]
fn general_slice<F: crate::Float>(x: crate::Arg<'_, F>, y: crate::Arg<'_, F>, n: usize, out: &mut [std::mem::MaybeUninit<F>], divide: bool) -> bool {
    crate::slice::two_pass(
        [x, y], n, out, #[inline(always)] |[a, b]| fast::<F::Working, { crate::FUSED }>(a.working(), b.working(), divide).map(F::from_working),
        #[inline(always)] |[a, b]| {
            F::from_working(if divide { root_any::<F::Working, { crate::FUSED }>(b.working(), a.working()) }
            else { pow_any::<F::Working, { crate::FUSED }>(a.working(), b.working()) })
        })
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
        Exponent::General => { general_slice(crate::Arg::Many(xs), crate::Arg::One(k), xs.len(), out, false); }
        e => route(e, xs, out),
    } } }
    routes!(Whole 1, Whole 2, Whole 3, Whole 4, Whole 5, Whole 6, Whole 7, Whole 8, Halves 1, Halves 3);
}

/// [`Pow`]'s slice form: [`powers`] for a shared exponent, otherwise the general block pass with scalar repair. Returns whether every result is valid: not NaN, or from a NaN argument.
pub(crate) fn power_slice<F: crate::float::Native>(_: &Pow, [x, y]: [crate::Arg<'_, F>; 2], n: usize, out: &mut [std::mem::MaybeUninit<F>]) -> bool {
    if let (crate::Arg::Many(xs), crate::Arg::One(k)) = (x, y) {
        let e = Exponent::of(k.wide(), crate::FUSED);
        powers(e, k, &xs[..n], &mut out[..n]);
        return !matches!(e, Exponent::Halves { .. } | Exponent::General) || k.is_nan()
            || xs[..n].iter().zip(unsafe { crate::slice::written(&mut out[..n]) }.iter()).all(|(x, r)| !r.is_nan() || x.is_nan());
    }
    general_slice(x, y, n, out, false)
}

/// The root of degree `n` of each of `ys` into `out`, at the working precision, with `n`'s route (`Degree::of`) chosen once. Each route and sign has its own loop, which vectorizes.
pub(crate) fn roots<F: crate::Float>(n: F, ys: &[F], out: &mut [std::mem::MaybeUninit<F>]) {
    let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
    macro_rules! routes { ($($d:ident),+) => { match (Degree::of(n.wide()), n < F::zero()) {
        $((Degree::$d, false) => each(ys, out, #[inline(always)] |y| rooted(&o, Degree::$d, false, y).expect("a degree off the general route")),
          (Degree::$d, true) => each(ys, out, #[inline(always)] |y| rooted(&o, Degree::$d, true, y).expect("a degree off the general route")),)+
        (Degree::General, _) => { general_slice(crate::Arg::Many(ys), crate::Arg::One(n), ys.len(), out, true); }
    } } }
    routes!(One, Two, Three, Four, Eight, Half);
}

/// [`Root`]'s slice form: [`roots`] for a shared degree, otherwise the general block pass followed by per-item short-degree recipes. It gives final validity after repair.
pub(crate) fn root_slice<F: crate::float::Native>(_: &Root, [n, y]: [crate::Arg<'_, F>; 2], len: usize, out: &mut [std::mem::MaybeUninit<F>]) -> bool {
    if let (crate::Arg::One(k), crate::Arg::Many(ys)) = (n, y) {
        roots(k, &ys[..len], &mut out[..len]);
        return matches!(Degree::of(k.wide()), Degree::One | Degree::Three) || k.is_nan()
            || ys[..len].iter().zip(unsafe { crate::slice::written(&mut out[..len]) }.iter()).all(|(y, r)| !r.is_nan() || y.is_nan());
    }
    let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
    general_slice(y, n, len, out, true);
    let results = unsafe { crate::slice::written(&mut out[..len]) };
    for (i, r) in results.iter_mut().enumerate() { if let Some(exact) = root_formula(&o, n.nth(i), y.nth(i)) { *r = exact; } }
    results.iter().enumerate().all(|(i, r)| !r.is_nan() || n.nth(i).is_nan() || y.nth(i).is_nan())
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

#[inline]
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
