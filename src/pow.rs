//! Powers and roots. `pow` and `root` compute e^(y ln |x|): at f64 through `table`'s logarithm and exponential, and for narrower types through `ln_narrow` and `exp_narrow` in f64. `Exponent` sends a whole exponent from 1 to `MULTIPLIED`, or to `COMPENSATED` where the target fuses multiply-add, to squaring and multiplying instead (`recipe::power`). One of 1 or 3 halves takes √x first. A degree of 2 or 3 takes `sqrt` or `cbrt`. `cbrt` refines an estimate taken from x's bits with Newton steps. `hypot` scales both arguments by a power of two before squaring them. See `THIRD-PARTY.md`.
use crate::{Arg, dd::Double, exp::exp_narrow, float::Native, ln::ln_narrow, slice::Total, table};
use num_complex::Complex;

/// Whether a is finite and nonzero, for an a whose sign bit is clear. 0, ∞ and NaN give false.
#[inline]
fn positive_finite(a: f64) -> bool { a.to_bits().wrapping_sub(1) < f64::INFINITY.to_bits() - 1 }

/// Whether y is a whole number, and whether it is an odd one. Infinities count as even whole numbers, and NaN as neither.
#[inline]
pub(crate) fn parity<R: Native>(y: R) -> (bool, bool) {
    let half = R::narrow(0.5) * y;
    let whole = y.trunc() == y;
    (whole, whole & (half.trunc() != half))
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

/// x^y. Special values follow IEEE 754's `pow`: x^0 and 1^y are 1, even for NaN, and a negative x has a power only for a whole y. At f64 it is `pow_one`: the f64 loop reads tables, which targets without gathers run one item at a time anyway. Narrower types have no branch, so the loops that call them vectorize.
#[inline(always)]
pub(crate) fn pow<R: Native, const FUSED: bool>(x: R, y: R) -> R { if R::WIDE { pow_one::<R, FUSED>(x, y) } else { pow_any::<R, FUSED>(x, y) } }

/// The real nth root of y, y^(1/n), computed as e^(ln |y| / n) without rounding 1/n first. A negative y has one only for an odd whole n. n = 0, an infinite n and NaN give NaN. It branches as `pow` does.
#[inline(always)]
pub(crate) fn root<R: Native, const FUSED: bool>(n: R, y: R) -> R { if R::WIDE { root_one::<R, FUSED>(n, y) } else { root_any::<R, FUSED>(n, y) } }

/// `pow` with one branch to its fast path wherever that applies, at every width, as Arm's scalar `pow` and `powf` do. An exponent that `Exponent::of` doesn't send to the general route takes another branch, to squaring and multiplying. `Pow`'s scalar form takes it, because each call computes one value, which no loop around it can vectorize. At f64 the fast path is `fast`, and at narrower widths `fast_single`.
#[inline(always)]
pub(crate) fn pow_one<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    match Exponent::of(y.wide()) {
        Exponent::Whole { m, negative } if m > MULTIPLIED => times(Exponent::Whole { m, negative }, x),
        Exponent::Whole { m, negative: true } => counted!(m, K => times(Exponent::Whole { m: K, negative: true }, x)),
        Exponent::Whole { m, negative: false } => counted!(m, K => times(Exponent::Whole { m: K, negative: false }, x)),
        Exponent::Halves { m, negative: true } => halved!(m, K => times(Exponent::Halves { m: K, negative: true }, x)),
        Exponent::Halves { m, negative: false } => halved!(m, K => times(Exponent::Halves { m: K, negative: false }, x)),
        Exponent::General => match if R::WIDE { fast::<R, FUSED>(x, y, false) } else { fast_single::<R, FUSED>(x, y, false) } { Some(r) => r, None => pow_any::<R, FUSED>(x, y) },
    }
}

/// The root of degree 2, as `recipe::root2` gives it.
#[inline(always)]
fn root2<R: Native>(y: R) -> R { crate::recipe::root2::<R, _>(&crate::recipe::Floats, y) }

/// `root` with one branch, as `pow_one` has, and one each to `sqrt` and `cbrt` for degrees 2 and 3. `Root`'s scalar form takes it.
#[inline(always)]
pub(crate) fn root_one<R: Native, const FUSED: bool>(n: R, y: R) -> R {
    if n == R::narrow(2.0) { return root2(y); }
    if n == R::narrow(3.0) { return cbrt::<R, FUSED>(y); }
    match if R::WIDE { fast::<R, FUSED>(y, n, true) } else { fast_single::<R, FUSED>(y, n, true) } { Some(r) => r, None => root_any::<R, FUSED>(n, y) }
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

/// The greatest whole exponent that `pow` raises to by squaring and multiplying in single words (`recipe::multiplied`). Without fused multiply-add, larger exponents take the general route. `Pow`'s slice form compiles a loop for each such exponent (`counted!`).
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

/// `$body` with `$k` a const for the magnitude m of `Exponent::Whole`, from 1 to `MULTIPLIED`.
macro_rules! counted {
    ($m:expr, $k:ident => $body:expr) => {
        match $m {
            1 => { const $k: u32 = 1; $body }
            2 => { const $k: u32 = 2; $body }
            3 => { const $k: u32 = 3; $body }
            4 => { const $k: u32 = 4; $body }
            5 => { const $k: u32 = 5; $body }
            6 => { const $k: u32 = 6; $body }
            7 => { const $k: u32 = 7; $body }
            _ => { const $k: u32 = 8; $body }
        }
    };
}
use counted;

/// `$body` with `$k` a const for the count m of `Exponent::Halves`, 1 or 3.
macro_rules! halved {
    ($m:expr, $k:ident => $body:expr) => { if $m == 1 { const $k: u32 = 1; $body } else { const $k: u32 = 3; $body } };
}
use halved;

/// One `each` loop for each bit length of a whole exponent m from `MULTIPLIED` + 1 to `COMPENSATED`, 4 to 6, and each sign, with a `recipe::step` written out for each bit position below m's highest (`steps!`). Rolled, as LLVM leaves such a loop at opt-level 2, those steps would stop the loop around them vectorizing.
macro_rules! unrolled {
    ($m:expr, $negative:expr, $x:expr, $n:expr, $out:expr; $($len:literal: [$($i:literal)+]),+) => {
        match (u32::BITS - $m.leading_zeros(), $negative) {
            $(
                ($len, false) => each!([$x], $n, $out, |[x]| Total(steps!($m, false, x; $($i)+))),
                ($len, true) => each!([$x], $n, $out, |[x]| Total(steps!($m, true, x; $($i)+))),
            )+
            _ => unreachable!("a whole exponent from `MULTIPLIED` + 1 to `COMPENSATED`"),
        }
    };
}

/// `recipe::whole` of `$x`, with a `recipe::step` for each bit position `$i` of `$m`.
macro_rules! steps {
    ($m:expr, $negative:literal, $x:expr; $($i:literal)+) => {{
        let o = &crate::recipe::Floats;
        let m = $m;
        let (b, mut error) = crate::recipe::base(o, $negative, $x);
        let mut w = (b, None);
        $( w = crate::recipe::step(o, w, b, m, $i, &mut error); )+
        w.0
    }};
}

/// `recipe::power` on one float, for an exponent off the general route. `counted!` and `halved!` make the exponent's magnitude a const in each arm, so each loop unrolls.
#[inline(always)]
fn times<R: Native>(e: Exponent, x: R) -> R { crate::recipe::power::<R, _>(&crate::recipe::Floats, e, x).expect("an exponent off the general route") }

/// `Pow`'s slice form, which `WholePow`'s runs too. A constant exponent takes its `Exponent` route. Whole exponents never fail, and their loop vectorizes with no test of each result. Any other call computes every item the fast way, then redoes only the rare items with special values.
macro_rules! pow_many {
    ($x:expr, $y:expr, $n:expr, $out:expr) => {
        match if let Arg::One(k) = $y { Exponent::of(<_ as crate::float::Float>::wide(k)) } else { Exponent::General } {
            Exponent::Whole { m, negative } if m > MULTIPLIED => unrolled!(m, negative, $x, $n, $out; 4: [2 1 0], 5: [3 2 1 0], 6: [4 3 2 1 0]),
            Exponent::Whole { m, negative: true } => counted!(m, K => each!([$x], $n, $out, |[x]| Total(times(Exponent::Whole { m: K, negative: true }, x)))),
            Exponent::Whole { m, negative: false } => counted!(m, K => each!([$x], $n, $out, |[x]| Total(times(Exponent::Whole { m: K, negative: false }, x)))),
            Exponent::Halves { m, negative: true } => halved!(m, K => each!([$x], $n, $out, |[x]| times(Exponent::Halves { m: K, negative: true }, x))),
            Exponent::Halves { m, negative: false } => halved!(m, K => each!([$x], $n, $out, |[x]| times(Exponent::Halves { m: K, negative: false }, x))),
            Exponent::General => two_pass!([$x, $y], $n, $out, |[a, b]| fast::<_, { crate::FUSED }>(a, b, false), |[a, b]| pow_any::<_, { crate::FUSED }>(a, b)),
        }
    };
}

/// [`pow`](crate::pow) as a function object. Its scalar form takes one branch to a fast path wherever one applies, and its slice form is `pow_many`.
#[derive(Clone, Copy, Debug)]
pub struct Pow;
forms!(Pow at real: |_, [x, y]| pow_one::<_, { crate::FUSED }>(x, y), many |_, [x, y], n, out| pow_many!(x, y, n, out));

/// [`Pow`] with wrapping integer results. Fractional negative integer powers fail.
#[derive(Clone, Copy, Debug)]
pub struct WrappingPow;
forms!(WrappingPow at real: |_, [x, y]| crate::Binary::one(&Pow, x, y), many |_, [x, y], n, out| crate::slice::Forms::slice(&Pow, [x, y], n, out));
forms!(WrappingPow at Complex<f32>, Complex<f64>: |_, [x, y]| crate::Binary::one(&Pow, x, y), many |_, [x, y], n, out| crate::slice::Forms::slice(&Pow, [x, y], n, out));

/// [`Pow`] for a whole exponent, where every real base has a real power and no result fails. Its forms are `Pow`'s. A caller passes it whole exponents only.
#[derive(Clone, Copy, Debug)]
pub struct WholePow;
forms!(WholePow at real: |_, [x, y]| Total(pow_one::<_, { crate::FUSED }>(x, y)), many |_, [x, y], n, out| crate::slice::Forms::slice(&Pow, [x, y], n, out));

/// [`root`](crate::root) as a function object, with a scalar form and a slice form that work as [`Pow`]'s do. A constant degree of 2 or 3 takes `sqrt` or `cbrt`.
#[derive(Clone, Copy, Debug)]
pub struct Root;
forms!(Root at real: |_, [n, y]| root_one::<_, { crate::FUSED }>(n, y),
many |_, [n, y], count, out| match n {
    Arg::One(d) if d == 2.0 => each!([y], count, out, |[y]| root2(y)),
    Arg::One(d) if d == 3.0 => each!([y], count, out, |[y]| cbrt::<_, { crate::FUSED }>(y)),
    _ => two_pass!([n, y], count, out, |[a, b]| fast::<_, { crate::FUSED }>(b, a, true), |[a, b]| root_any::<_, { crate::FUSED }>(a, b)),
});
// A complex degree of 2 takes `Sqrt`, and any other finite degree but 0 takes `Pow` of its reciprocal. A degree of 0 or an infinite degree gives NaN, as the real form does, which fails.
forms!(Root at Complex<f32>, Complex<f64>: |_, [n, y]| if n == Complex::new(0.0, 0.0) || !n.is_finite() {
    Complex::new(num_traits::Float::nan(), num_traits::Float::nan())
} else if n == Complex::new(2.0, 0.0) {
    crate::Unary::one(&crate::Sqrt, y)
} else {
    crate::Binary::one(&Pow, y, crate::divide(Complex::new(1.0, 0.0), n))
});

/// The top bits of |x|^(-1/3) plus a third of those of |x|, at f64 and at f32, as `top` reads them. The sum is nearly the same for every positive normal x. These values keep `cbrt`'s first estimate within 3.5%.
const CBRT_MAGIC: [f64; 2] = [1499661097332631.0, 2773273.0];

/// The cube root of x at its own width. The top bits of a positive float grow nearly in proportion to its logarithm. `CBRT_MAGIC` less a third of |x|'s top bits is therefore the top bits of a z near |x|^(-1/3). Newton steps with no division square z's error: three at f64 and two at f32. y = |x| z² is then ∛|x|, which a last step corrects: with ε = y z - 1, ∛|x| is close to y(1 - 2ε/3). A subnormal x scales up first, by 2^54 at f64 and 2^24 at f32, and its root scales down last, by 2^18 or 2^8.
#[inline(always)]
pub(crate) fn cbrt<R: Native, const FUSED: bool>(x: R) -> R {
    let c = R::narrow;
    let a = x.abs();
    let (least, up, down) =
        if R::WIDE { (f64::MIN_POSITIVE, 18014398509481984.0, 3.814697265625e-6) } else { (f32::MIN_POSITIVE as f64, 16777216.0, 0.00390625) };
    let sub = a < c(least);
    let b = if sub { a * c(up) } else { a };
    let mut z = R::from_top(b.top().mla::<FUSED>(c(-1.0 / 3.0), c(CBRT_MAGIC[usize::from(!R::WIDE)])));
    for _ in 0..if R::WIDE { 3 } else { 2 } { z = z * ((b * z) * (z * z)).mla::<FUSED>(c(-1.0 / 3.0), c(4.0 / 3.0)); }
    let y = b * z * z;
    // Without FMA, y z rounds before the subtraction, which loses ε's low bits. A double-word product keeps them, so that an exact cube gives its root exactly.
    let e = if FUSED { y.mla::<FUSED>(z, -R::one()) } else { let p = Double::new(y, R::zero()).mul_float::<FUSED>(z); (p.hi - R::one()) + p.lo };
    let y = (y * c(-2.0 / 3.0)).mla::<FUSED>(e, y) * if sub { c(down) } else { R::one() };
    // 0, ∞ and NaN give themselves.
    let y = if positive_finite(a.wide()) { y } else { a };
    y.copysign(x)
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
