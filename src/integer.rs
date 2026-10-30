//! Functions of `i64`s: greatest common divisor, least common multiple, binomial coefficients, powers, square roots and residues. Each gives `None` where its result isn't an `i64`. Their function objects are the ones that floats and complex numbers use: [`Gcd`], [`Lcm`], [`Residue`] and [`FloorDivide`] from [`tolerant`](crate::tolerant), whose integer forms ignore the tolerance they hold, and [`Binom`] and [`Pow`]. A slice form appends results to a vector and reports whether every item fitted. [`Divisor`] divides many integers by one, with multiplications in place of division instructions. Where every item has the same divisor, `Residue`'s slice form divides through `FloatDivisor`, which multiplies by the divisor's reciprocal in `f64` and vectorizes, for divisors and dividends below 2⁵¹ in magnitude. At types of 16 bits or fewer it multiplies 32-bit integers instead, and it takes `Divisor` for 64-bit dividends of 2⁵¹ or more.
//!
//! `Gcd`, `Lcm`, `Binom`, `Pow` and `Residue` also take the narrower integer types of compact buffers, `u8`, `i16` and `i32`, which they compute through these functions. A result that doesn't fit its type fails. `FloorDivide` divides each integer type exactly, and fails on a zero divisor or an overflowing quotient. The tolerant comparisons, [`Floor`] and [`Ceil`] take each of these integer types exactly, ignoring the tolerance. [`Abs`] and [`Sign`] take them through the integer arithmetic of the `arith` module, and `Abs` of a signed type's least value fails. `Floor`, `Ceil` and `Abs` keep Booleans as they are.
use crate::{
    Abs, Arg, Binom, Pow, WrappingPow, convert::{self, WIDE_LIMIT}, float::sealed::Core,
    tolerant::{Ceil, Equal, Floor, FloorDivide, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, NotEqual, Residue}
};
use strength_reduce::StrengthReducedU64;

/// Bounds for every prefix sum of up to `len` items in `[lo, hi]`, from `start` or 0. With `subtract`, bounds every running difference instead: `start` less up to `len` items, or without `start`, the first item less up to `len - 1` more. `None` where the bounds leave `i64`.
pub fn sum_bounds([lo, hi]: [i64; 2], len: usize, start: Option<i64>, subtract: bool) -> Option<[i64; 2]> {
    let (lo, hi, len) = (i128::from(lo), i128::from(hi), len as i128);
    let (least, most) = match (start, subtract) {
        (None, true) => (lo + ((len - 1) * -hi).min(0), hi + ((len - 1) * -lo).max(0)),
        _ => {
            let ([lo, hi], s) = (if subtract { [-hi, -lo] } else { [lo, hi] }, i128::from(start.unwrap_or(0)));
            (s + lo.min(len * lo), s + hi.max(len * hi))
        }
    };
    Some([i64::try_from(least).ok()?, i64::try_from(most).ok()?])
}

/// Bounds for products of one item in each inclusive range, or `None` where an endpoint product leaves `i64`.
pub fn product_bounds([a, b]: [i64; 2], [c, d]: [i64; 2]) -> Option<[i64; 2]> {
    let corners = [i128::from(a) * i128::from(c), i128::from(a) * i128::from(d), i128::from(b) * i128::from(c), i128::from(b) * i128::from(d)];
    Some([i64::try_from(*corners.iter().min()?).ok()?, i64::try_from(*corners.iter().max()?).ok()?])
}

/// The greatest common divisor of x and y, which is never negative. `gcd(0, 0)` is 0. `None` only where it is 2^63, as for `gcd(i64::MIN, 0)`.
pub fn gcd(x: i64, y: i64) -> Option<i64> { i64::try_from(unsigned_gcd(x.unsigned_abs(), y.unsigned_abs())).ok() }

/// The greatest common divisor of a and b by Stein's algorithm, with shifts and subtractions and no division. It is 0 for 0 and 0.
fn unsigned_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 || b == 0 { return a | b; }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b { std::mem::swap(&mut a, &mut b) }
        b -= a;
        if b == 0 { return a << shift; }
    }
}

/// The least common multiple of x and y, with the sign of x × y. It is 0 where either is 0.
pub fn lcm(x: i64, y: i64) -> Option<i64> { if x == 0 || y == 0 { return Some(0); } (x / gcd(x, y)?).checked_mul(y) }

/// The binomial coefficient C(n, k), the number of ways to choose k things from n, for k from 0. A negative n gives (-1)^k C(k - n - 1, k), the coefficient of t^k in (1 + t)^n. `None` for a negative k.
pub fn binom(n: i64, k: i64) -> Option<i64> {
    if k < 0 { return None; }
    let k = k as u64;
    // For a negative n, C(k - n - 1, k) is C(k + m, k) with m = -n - 1, which is never negative. k + m fits a u64.
    let (top, negative) = if n >= 0 { (n as u64, false) } else { (k + (n.unsigned_abs() - 1), k % 2 == 1) };
    if top < k { return Some(0); }
    let c = choose(top, k.min(top - k), 1 << 63)? as i128;
    i64::try_from(if negative { -c } else { c }).ok()
}

/// C(n, k) for 0 ≤ k ≤ n, through C(n, i + 1) = C(n, i) (n - i) / (i + 1), whose division is exact. Each step's product fits a `u128`. A coefficient beyond the caller's largest representable magnitude stops the steps.
fn choose(n: u64, k: u64, limit: u64) -> Option<u64> {
    (0..k).try_fold(1u64, |c, i| {
        let c = c as u128 * (n - i) as u128 / (i + 1) as u128;
        (c <= limit as u128).then_some(c as u64)
    })
}

/// x^y. 0^0 is 1. `None` where the power doesn't fit an `i64`, and for a negative y beside an x other than 1 or -1, where the power isn't a whole number.
pub fn power(x: i64, y: i64) -> Option<i64> { let (p, exact) = wrapped_power(x, y); exact.then_some(p) }

/// x^y wrapped to `i64`, and whether that is the exact power, by repeated squaring. A negative y beside an x other than 1 or -1 gives 0, which isn't the power.
pub(crate) fn wrapped_power(x: i64, y: i64) -> (i64, bool) {
    match x {
        1 => (1, true),
        -1 => (if y % 2 == 0 { 1 } else { -1 }, true),
        _ if y < 0 => (0, false),
        0 => (i64::from(y == 0), true),
        _ => {
            // A square that overflows makes the power inexact only where a later bit of y multiplies it in.
            let (mut base, mut e, mut p, mut exact, mut base_exact) = (x, y as u64, 1i64, true, true);
            loop {
                if e & 1 == 1 {
                    let (q, over) = p.overflowing_mul(base);
                    (p, exact) = (q, exact & base_exact & !over);
                }
                e >>= 1;
                if e == 0 { break (p, exact); }
                let (b, over) = base.overflowing_mul(base);
                (base, base_exact) = (b, base_exact & !over);
            }
        }
    }
}

/// The residue of y modulo x, y - x⌊y/x⌋, which takes the sign of x. A zero x gives y. Every residue fits an `i64`.
#[inline]
pub fn residue(x: i64, y: i64) -> i64 { if x == 0 { y } else { floored(y.wrapping_rem(x), x) } }

/// The mask whose bitwise and with an integer y gives y's residue modulo x, where x is a positive power of two: x - 1. In two's complement, y ∧ (x - 1) is y - x⌊y/x⌋ for every y of every width. `None` for any other x.
#[inline]
pub fn mask(x: i64) -> Option<i64> { (x > 0 && x.count_ones() == 1).then(|| x - 1) }

/// The floor of x/y, or `None` for a zero y or a quotient that overflows. A dividend below 2^53 in magnitude divides in `f64`, where integer division is slow: the quotient then rounds by less than its distance from the nearest whole number, so its floor is exact. A divisor that `f64` rounds then exceeds the dividend in magnitude, and the quotient's floor is 0 or ¯1 either way.
#[inline]
pub fn floor_quotient(x: i64, y: i64) -> Option<i64> {
    if y == 0 { return None; }
    if x.unsigned_abs() < 1 << 53 { return Some((x as f64 / y as f64).floor() as i64); }
    x.checked_div(y).map(|q| if x % y != 0 && (x < 0) != (y < 0) { q - 1 } else { q })
}

/// `r`, a remainder of a division by x with the dividend's sign, moved to x's sign.
#[inline]
fn floored<T: num_traits::PrimInt + num_traits::Signed>(r: T, x: T) -> T { if !r.is_zero() && r.is_negative() != x.is_negative() { r + x } else { r } }

/// A fixed nonzero divisor for dividing many integers. Each division multiplies and shifts, where a division instruction would take several times as long. Setting one up costs more than one division.
#[derive(Clone, Copy, Debug)]
pub struct Divisor { x: i64, magnitude: StrengthReducedU64 }

impl Divisor {
    /// The divisor x, or `None` for 0.
    pub fn new(x: i64) -> Option<Self> { (x != 0).then(|| Self { x, magnitude: StrengthReducedU64::new(x.unsigned_abs()) }) }

    /// The residue of y modulo x, as [`residue`] gives it.
    #[inline]
    pub fn residue(self, y: i64) -> i64 { let r = (y.unsigned_abs() % self.magnitude) as i64; floored(if y < 0 { -r } else { r }, self.x) }

    /// The floored quotient ⌊y/x⌋ and the residue of y modulo x. `None` where the quotient isn't an `i64`: `i64::MIN` divided by -1.
    #[inline]
    pub fn div_mod(self, y: i64) -> Option<(i64, i64)> {
        let (q, r) = StrengthReducedU64::div_rem(y.unsigned_abs(), self.magnitude);
        let q = if (y < 0) != (self.x < 0) { -(q as i128) } else { q as i128 };
        let r0 = if y < 0 { -(r as i64) } else { r as i64 };
        let r = floored(r0, self.x);
        Some((i64::try_from(q - i128::from(r != r0)).ok()?, r))
    }
}

/// A fixed divisor whose magnitude d is from 2 to 2¹⁵, for dividends whose magnitude a is at most 2¹⁵, as every `u8` and `i16` has. Each division takes the high 16 bits of a product of 16-bit integers and shifts them, which vectorizes in 16-bit lanes. [`Divisor`]'s 64-bit high product doesn't vectorize. With l = ⌈log₂ d⌉ and m = ⌈2^(15+l)/d⌉, ⌊a·m/2^(15+l)⌋ is ⌊a/d⌋. The fraction of a/d is at most 1 - 1/d. Because m exceeds 2^(15+l)/d by less than 1 and a·d is at most 2^(15+l), a·m/2^(15+l) exceeds a/d by less than 1/d, which leaves its floor unchanged. m is 2¹⁵ where d is 2^l. Any other d is at least 2^(l-1) + 1, which keeps 2^(15+l)/d below 2¹⁶ - 1. m therefore fits 16 bits and a·m fits 32. Shifting a·m's high 16 bits right by l - 1 gives ⌊a·m/2^(15+l)⌋.
#[derive(Clone, Copy)]
struct Short { x: i16, magnitude: u16, multiplier: u16, shift: u32 }

impl Short {
    /// The divisor `x`, or `None` where its magnitude is below 2.
    fn new(x: i16) -> Option<Self> {
        let d = x.unsigned_abs();
        (d >= 2).then(|| {
            let l = u16::BITS - (d - 1).leading_zeros();
            Self { x, magnitude: d, multiplier: (((1 << (15 + l)) - 1) / u32::from(d) + 1) as u16, shift: l - 1 }
        })
    }

    /// The residue of `y` modulo the divisor, as [`residue`] gives it.
    #[inline]
    fn residue(self, y: i16) -> i16 {
        let a = y.unsigned_abs();
        let q = ((u32::from(a) * u32::from(self.multiplier)) >> 16) as u16 >> self.shift;
        let r = (a - q * self.magnitude) as i16;
        floored(if y < 0 { -r } else { r }, self.x)
    }
}

/// A fixed nonzero divisor whose magnitude is below 2⁵¹, for dividends whose magnitude is below 2⁵¹. Each division multiplies by the divisor's reciprocal in `f64` and corrects the quotient once, which vectorizes at every integer width, where [`Divisor`]'s 64-bit high product doesn't. For such y and x, y·(1/x) rounds to within less than ½ of y/x, so its nearest whole number q is ⌊y/x⌋ or one more. The remainder y - q·x, which `f64` holds exactly, has the divisor's sign or is 0 only for ⌊y/x⌋. [`convert::unshifted`] and [`convert::shifted`] convert integers to and from `f64` in a form that vectorizes too.
#[derive(Clone, Copy)]
struct FloatDivisor { x: f64, reciprocal: f64 }

impl FloatDivisor {
    /// The divisor `x`, or `None` for 0 and for a magnitude of 2⁵¹ or more.
    #[inline]
    fn new(x: i64) -> Option<Self> { (x != 0 && x.unsigned_abs() <= WIDE_LIMIT as u64).then(|| Self { x: x as f64, reciprocal: 1.0 / x as f64 }) }

    /// The residue of y modulo x, as [`residue`] gives it, for y below 2⁵¹ in magnitude.
    #[inline(always)]
    fn residue(self, y: i64) -> i64 {
        let y = convert::unshifted(y);
        let q = (y * self.reciprocal + f64::SHIFT) - f64::SHIFT;
        let r = y - q * self.x;
        convert::shifted(if r * self.x < 0.0 { r + self.x } else { r })
    }
}

/// Whether each of the first `n` dividends in `y` is below 2⁵¹ in magnitude, as [`FloatDivisor`] needs. One pass finds their least and greatest.
fn fit(y: Arg<'_, i64>, n: usize) -> bool {
    let fits = |v: i64| v.unsigned_abs() <= WIDE_LIMIT as u64;
    match y { Arg::One(v) => fits(v), Arg::Many(v) => { v[..n].is_empty() || { let (lo, hi) = crate::reduce::extremes(&v[..n]); fits(lo) && fits(hi) } } }
}

/// The digits of each of `values` in the mixed radix of `bases`, as APL's `⊤` gives them: each value's digits together, the most significant first. A base of 0 takes all of what remains as its digit. `None` where a quotient isn't an `i64`.
pub fn encode(bases: &[i64], values: &[i64]) -> Option<Vec<i64>> {
    let divisors: Vec<_> = bases.iter().map(|&b| Divisor::new(b)).collect();
    crate::polynomial::encoded(&divisors, values, |d, v| match d { Some(d) => d.div_mod(v), None => Some((0, v)) })
}

/// The value of each cell of `bases.len()` digits in `digits`, in the mixed radix of `bases`, as APL's `⊥` gives it. The first base is never used. `None` where a step isn't an `i64`.
pub fn decode(bases: &[i64], digits: &[i64]) -> Option<Vec<i64>> { decoded(bases, digits, |v, b, d| v.checked_mul(b)?.checked_add(d)) }

/// [`decode`] with each step wrapped to 64 bits, as integers that wrap compute it. Wrapping a value to a narrower width then gives what wrapping each step at that width gives.
pub fn decode_wrapped(bases: &[i64], digits: &[i64]) -> Vec<i64> {
    decoded(bases, digits, |v, b, d| Some(v.wrapping_mul(b).wrapping_add(d))).expect("a wrapped step always fits")
}

/// Each cell's value by Horner's rule, through `step(value, base, digit)`, or `None` where a step gives none.
fn decoded(bases: &[i64], digits: &[i64], step: impl Fn(i64, i64, i64) -> Option<i64>) -> Option<Vec<i64>> {
    digits.chunks(bases.len()).map(|d| (1..d.len()).try_fold(d[0], |v, k| step(v, bases[k], d[k]))).collect()
}

forms!(#[inline] Gcd at i64: |_, [x, y]| gcd(x, y));
forms!(#[inline] Lcm at i64: |_, [x, y]| lcm(x, y));
// On Booleans, the greatest common divisor is `or` and the least common multiple is `and`.
forms!(#[inline] Gcd at bool: |_, [x, y]| x | y);
forms!(#[inline] Lcm at bool: |_, [x, y]| x & y);
forms!(#[inline] Binom at i64: |_, [n, k]| binom(n, k));
forms!(#[inline] Pow at i64: |_, [x, y]| wrapped_power(x, y));
fn wrapping_power(x: i64, y: i64) -> Option<i64> { let (p, exact) = wrapped_power(x, y); (exact | (y >= 0)).then_some(p) }
forms!(#[inline] WrappingPow at i64: |_, [x, y]| wrapping_power(x, y));
forms!(
    #[inline] Residue at i64: |_, [x, y]| residue(x, y), many |_, [x, y], n, out| if let Arg::One(d) = x && let Some(m) = mask(d) {
        each!([y], n, out, |[y]| y & m)
    } else if let Arg::One(d) = x && let Some(r) = FloatDivisor::new(d) && fit(y, n) {
        each!([y], n, out, |[y]| r.residue(y))
    } else if let Arg::One(d) = x && let Some(d) = Divisor::new(d) {
        each!([y], n, out, |[y]| d.residue(y))
    } else {
        each!([x, y], n, out, |[x, y]| residue(x, y))
    }
);

/// The narrower types' forms of `Gcd`, `Lcm`, `Binom`, `Pow` and `Residue`, which compute at `i64`. A result that doesn't fit the type fails, as one beyond `i64` does. A power that fails holds the power wrapped to the type. Every residue fits.
macro_rules! narrower {
    ($($t:ty),+) => {$(
        forms!(#[inline] Gcd at $t: |_, [x, y]| gcd(x.into(), y.into()).and_then(|r| <$t>::try_from(r).ok()));
        forms!(#[inline] Lcm at $t: |_, [x, y]| lcm(x.into(), y.into()).and_then(|r| <$t>::try_from(r).ok()));
        forms!(#[inline] Binom at $t: |_, [n, k]| binom(n.into(), k.into()).and_then(|r| <$t>::try_from(r).ok()));
        forms!(#[inline] Pow at $t: |_, [x, y]| { let (p, exact) = wrapped_power(x.into(), y.into()); (p as $t, exact && <$t>::try_from(p).is_ok()) });
        forms!(#[inline] WrappingPow at $t: |_, [x, y]| wrapping_power(x.into(), y.into()).map(|p| p as $t));
        forms!(
            #[inline] Residue at $t: |_, [x, y]| residue(x.into(), y.into()) as $t,
            many |_, [x, y], n, out| if let Arg::One(d) = x && let Some(m) = mask(d.into()) {
                each!([y], n, out, |[y]| y & m as $t)
            } else if let Arg::One(d) = x && <$t>::BITS <= 16 && let Some(s) = Short::new(d as i16) {
                each!([y], n, out, |[y]| s.residue(y as i16) as $t)
            } else if let Arg::One(d) = x && let Some(r) = FloatDivisor::new(d.into()) {
                each!([y], n, out, |[y]| r.residue(y.into()) as $t)
            } else {
                each!([x, y], n, out, |[x, y]| residue(x.into(), y.into()) as $t)
            }
        );
    )+};
}
narrower!(u8, i16, i32);
/// Floor division at the narrower types, through a float type in which every quotient's floor is exact, as `floor_quotient`'s doc explains. A zero divisor and a quotient that doesn't fit the type fail without a branch. The loop then vectorizes.
macro_rules! floor_divides {
    ($($t:ty => $f:ty),+) => {$(
        forms!(FloorDivide at $t: |_, [x, y]| { let q = (<$f>::from(x) / <$f>::from(y)).floor(); (y != 0 && q <= <$t>::MAX as $f).then_some(q as $t) });
    )+};
}
floor_divides!(u8 => f32, i16 => f32, i32 => f64);
forms!(FloorDivide at i64: |_, [x, y]| floor_quotient(x, y));

/// Each comparison's exact forms, which give Booleans: at each integer type of compact buffers, and at Booleans, where `false` orders before `true`. Characters compare as their code points, through these forms at `i32` ([`characters`](crate::characters)).
macro_rules! exactly {
    ($($object:ident: |$x:ident, $y:ident| $body:expr;)+) => {$(
        forms!(#[inline] $object at u8 => bool, i16 => bool, i32 => bool, i64 => bool, bool => bool: |_, [$x, $y]| $body);
    )+};
}
exactly! {
    Equal: |x, y| x == y;
    NotEqual: |x, y| x != y;
    Less: |x, y| x < y;
    LessEqual: |x, y| x <= y;
    Greater: |x, y| x > y;
    GreaterEqual: |x, y| x >= y;
}
forms!(Floor at u8, i16, i32, i64, bool: |_, [x]| x);
forms!(Ceil at u8, i16, i32, i64, bool: |_, [x]| x);
forms!(Abs at bool: |_, [x]| x);
