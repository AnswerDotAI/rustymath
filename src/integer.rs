//! Functions of `i64`s: greatest common divisor, least common multiple, binomial coefficients, powers, square roots and residues.
//! Each gives `None` where its result isn't an `i64`. Their function objects are the ones that floats and complex numbers use:
//! [`Gcd`], [`Lcm`] and [`Residue`] from [`tolerant`](crate::tolerant), whose integer forms ignore the tolerance they hold, and
//! [`Binom`] and [`Pow`]. A slice form appends results to a vector and reports whether every item fitted. [`Divisor`] divides many
//! integers by one, with multiplications in place of division instructions. `Residue`'s slice form uses one where every item has
//! the same divisor.
//!
//! `Gcd`, `Lcm`, `Pow` and `Residue` also take every narrower integer type, which they compute through these functions. Their `u64`
//! forms have functions of their own, because a `u64` can exceed `i64::MAX`. A result that doesn't fit its type fails. The tolerant
//! comparisons, [`Floor`] and [`Ceil`] take every integer type exactly, ignoring the tolerance. [`Abs`] and [`Sign`] take every
//! integer type through [`Overflowing`](crate::Overflowing): `Abs` of a signed type's least value wraps to itself, as NumPy's does.
//! `Floor`, `Ceil` and `Abs` keep Booleans as they are.
use crate::{
    tolerant::{Ceil, Equal, Floor, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, NotEqual, Residue},
    Abs, Arg, Binom, Pow,
};
use strength_reduce::StrengthReducedU64;

/// The greatest common divisor of x and y, which is never negative. `gcd(0, 0)` is 0. `None` only where it is 2^63, as for
/// `gcd(i64::MIN, 0)`.
pub fn gcd(x: i64, y: i64) -> Option<i64> { i64::try_from(unsigned_gcd(x.unsigned_abs(), y.unsigned_abs())).ok() }

/// The greatest common divisor of a and b by Stein's algorithm, with shifts and subtractions and no division. It is 0 for 0 and 0.
fn unsigned_gcd(mut a: u64, mut b: u64) -> u64 {
    if a == 0 || b == 0 { return a | b }
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b { std::mem::swap(&mut a, &mut b) }
        b -= a;
        if b == 0 { return a << shift }
    }
}

/// The least common multiple of x and y, which is 0 where either is 0. `None` where it is beyond `u64`.
fn unsigned_lcm(x: u64, y: u64) -> Option<u64> { if x == 0 || y == 0 { Some(0) } else { (x / unsigned_gcd(x, y)).checked_mul(y) } }

/// The least common multiple of x and y, with the sign of x × y. It is 0 where either is 0.
pub fn lcm(x: i64, y: i64) -> Option<i64> {
    if x == 0 || y == 0 { return Some(0) }
    (x / gcd(x, y)?).checked_mul(y)
}

/// The binomial coefficient C(n, k), the number of ways to choose k things from n, for k from 0. A negative n gives
/// (-1)^k C(k - n - 1, k), the coefficient of t^k in (1 + t)^n. `None` for a negative k.
pub fn binom(n: i64, k: i64) -> Option<i64> {
    if k < 0 { return None }
    let k = k as u64;
    // For a negative n, C(k - n - 1, k) is C(k + m, k) with m = -n - 1, which is never negative. k + m fits a u64.
    let (top, negative) = if n >= 0 { (n as u64, false) } else { (k + (n.unsigned_abs() - 1), k % 2 == 1) };
    if top < k { return Some(0) }
    let c = choose(top, k.min(top - k))? as i128;
    i64::try_from(if negative { -c } else { c }).ok()
}

/// C(n, k) for 0 ≤ k ≤ n, through C(n, i + 1) = C(n, i) (n - i) / (i + 1), whose division is exact. Each step's product fits a
/// `u128`. A coefficient beyond 2^63, the largest magnitude an `i64` holds, stops the steps.
fn choose(n: u64, k: u64) -> Option<u64> {
    (0..k).try_fold(1u64, |c, i| {
        let c = c as u128 * (n - i) as u128 / (i + 1) as u128;
        (c <= 1 << 63).then_some(c as u64)
    })
}

/// x^y. 0^0 is 1. `None` for a negative y beside an x other than 1 or -1, where the power isn't a whole number.
pub fn power(x: i64, y: i64) -> Option<i64> {
    match x {
        1 => Some(1),
        -1 => Some(if y % 2 == 0 { 1 } else { -1 }),
        _ if y < 0 => None,
        0 => Some(i64::from(y == 0)),
        _ => x.checked_pow(u32::try_from(y).ok()?),
    }
}

/// x^y, where 0^0 is 1. `None` where it is beyond `u64`.
fn unsigned_power(x: u64, y: u64) -> Option<u64> {
    match x {
        0 => Some(u64::from(y == 0)),
        1 => Some(1),
        _ => x.checked_pow(u32::try_from(y).ok()?),
    }
}

/// The residue of y modulo x, y - x⌊y/x⌋, which takes the sign of x. A zero x gives y. Every residue fits an `i64`.
pub fn residue(x: i64, y: i64) -> i64 { if x == 0 { y } else { floored(y.wrapping_rem(x), x) } }

/// `r`, a remainder of a division by x with the dividend's sign, moved to x's sign.
#[inline]
fn floored(r: i64, x: i64) -> i64 { if r != 0 && (r < 0) != (x < 0) { r + x } else { r } }

/// A fixed nonzero divisor for dividing many integers. Each division multiplies and shifts, where a division instruction would take
/// several times as long. Setting one up costs more than one division.
#[derive(Clone, Copy, Debug)]
pub struct Divisor {
    x: i64,
    magnitude: StrengthReducedU64,
}

impl Divisor {
    /// The divisor x, or `None` for 0.
    pub fn new(x: i64) -> Option<Self> { (x != 0).then(|| Self { x, magnitude: StrengthReducedU64::new(x.unsigned_abs()) }) }

    /// The residue of y modulo x, as [`residue`] gives it.
    #[inline]
    pub fn residue(self, y: i64) -> i64 {
        let r = (y.unsigned_abs() % self.magnitude) as i64;
        floored(if y < 0 { -r } else { r }, self.x)
    }

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

forms!(Gcd at i64: |_, [x, y]| gcd(x, y));
forms!(Lcm at i64: |_, [x, y]| lcm(x, y));
forms!(Binom at i64: |_, [n, k]| binom(n, k));
forms!(Pow at i64: |_, [x, y]| power(x, y));
forms!(
    Residue at i64: |_, [x, y]| residue(x, y),
    many |_, [x, y], n, out| if let Arg::One(d) = x && let Some(d) = Divisor::new(d) {
        each!([y], n, out, |[y]| d.residue(y))
    } else {
        each!([x, y], n, out, |[x, y]| residue(x, y))
    }
);
forms!(Gcd at u64: |_, [x, y]| unsigned_gcd(x, y));
forms!(Lcm at u64: |_, [x, y]| unsigned_lcm(x, y));
forms!(Pow at u64: |_, [x, y]| unsigned_power(x, y));
forms!(Residue at u64: |_, [x, y]| if x == 0 { y } else { y % x });

/// The narrower types' forms of `Gcd`, `Lcm`, `Pow` and `Residue`, which compute at `i64`. A result that doesn't fit the type fails,
/// as one beyond `i64` does. Every residue fits.
macro_rules! narrower {
    ($($t:ty),+) => {$(
        forms!(Gcd at $t: |_, [x, y]| gcd(x.into(), y.into()).and_then(|r| <$t>::try_from(r).ok()));
        forms!(Lcm at $t: |_, [x, y]| lcm(x.into(), y.into()).and_then(|r| <$t>::try_from(r).ok()));
        forms!(Pow at $t: |_, [x, y]| power(x.into(), y.into()).and_then(|r| <$t>::try_from(r).ok()));
        forms!(Residue at $t: |_, [x, y]| residue(x.into(), y.into()) as $t);
    )+};
}
narrower!(i8, i16, i32, u8, u16, u32);

/// `$object`'s forms at every integer type, with Boolean outputs where `=> bool` says so.
macro_rules! every_width {
    ($object:ident $(=> $o:ty)?: $($body:tt)+) => {
        forms!($object at i8 $(=> $o)?, i16 $(=> $o)?, i32 $(=> $o)?, i64 $(=> $o)?, u8 $(=> $o)?, u16 $(=> $o)?, u32 $(=> $o)?, u64 $(=> $o)?: $($body)+);
    };
}
every_width!(Equal => bool: |_, [x, y]| x == y);
every_width!(NotEqual => bool: |_, [x, y]| x != y);
every_width!(Less => bool: |_, [x, y]| x < y);
every_width!(LessEqual => bool: |_, [x, y]| x <= y);
every_width!(Greater => bool: |_, [x, y]| x > y);
every_width!(GreaterEqual => bool: |_, [x, y]| x >= y);
forms!(Floor at i8, i16, i32, i64, u8, u16, u32, u64, bool: |_, [x]| x);
forms!(Ceil at i8, i16, i32, i64, u8, u16, u32, u64, bool: |_, [x]| x);
forms!(Abs at bool: |_, [x]| x);

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{accuracy::random, Binary, Sign, Unary};
    use num_bigint::BigInt;

    /// Integers that exercise overflow and sign: the extremes, small numbers, powers of two and random bits.
    fn samples() -> Vec<i64> {
        let mut v = vec![0, 1, -1, 2, -2, 3, -3, 7, -12, 18, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1, 1 << 32, -(1 << 62)];
        v.extend((0..200).map(|i| random(i) as i64 >> (random(i + 1000) % 64)));
        v
    }

    #[test]
    fn gcd_lcm_power_residue() {
        let fit = |v: i128| i64::try_from(v).ok();
        for &x in &samples() {
            for &y in &samples() {
                let (a, b) = (x as i128, y as i128);
                let g = num_integer::Integer::gcd(&a, &b);
                assert_eq!(gcd(x, y), fit(g), "gcd({x}, {y})");
                assert_eq!(lcm(x, y), if g == 0 { Some(0) } else { fit(a / g * b) }, "lcm({x}, {y})");
                if x == 0 {
                    assert!(residue(0, y) == y && Divisor::new(0).is_none());
                    continue;
                }
                // The floored residue, with x's sign, and the quotient it leaves.
                let r = (b % a + a) % a;
                let d = Divisor::new(x).unwrap();
                assert!(residue(x, y) as i128 == r && d.residue(y) as i128 == r, "residue({x}, {y})");
                assert_eq!(d.div_mod(y), fit((b - r) / a).map(|q| (q, r as i64)), "div_mod({x}, {y})");
            }
            for y in -3..70 {
                let want = if y < 0 { (x.abs() == 1).then(|| if y % 2 == 0 { 1 } else { x }) } else { BigInt::from(x).pow(y as u32).try_into().ok() };
                assert_eq!(power(x, y), want, "power({x}, {y})");
            }
        }
        assert_eq!([power(0, 0), power(0, -1), power(-1, i64::MIN), power(-2, 63), power(2, 63)], [Some(1), None, Some(1), Some(i64::MIN), None]);
    }

    #[test]
    fn binomials() {
        // C(n, k) as the product of n - i for i below k, divided by k!, which holds for every integer n.
        let exact = |n: i64, k: i64| -> Option<i64> {
            let p = (0..k).fold(BigInt::from(1), |p, i| p * (n - i)) / (1..=k).fold(BigInt::from(1), |p, i| p * i);
            p.try_into().ok()
        };
        for n in -40..=70 {
            for k in 0..=75 {
                let c = exact(n, k);
                assert_eq!(binom(n, k), c, "binom({n}, {k})");
                // The float binomial coefficient of whole arguments is exact while the result times |n| + k fits the type's whole
                // numbers.
                let fits = |bits: u32| c.is_some_and(|c| (c.unsigned_abs() as u128) * (n.unsigned_abs() + k as u64) as u128 >> bits == 0);
                if fits(53) { assert_eq!(crate::binom(n as f64, k as f64), c.unwrap() as f64, "f64 binom({n}, {k})") }
                if fits(24) { assert_eq!(crate::binom(n as f32, k as f32), c.unwrap() as f32, "f32 binom({n}, {k})") }
            }
            assert_eq!(binom(n, -1), None);
        }
        assert_eq!(binom(i64::MIN, 1), Some(i64::MIN));
        assert_eq!(binom(i64::MAX, 1), Some(i64::MAX));
        assert_eq!(binom(i64::MAX, 2), None);
        assert_eq!(binom(-2, 1_000_000_000_000), Some(1_000_000_000_001));
        assert_eq!(binom(-1, i64::MAX), Some(-1));
    }

    #[test]
    fn slice_forms() {
        let (xs, ys) = (samples(), samples().into_iter().rev().collect::<Vec<_>>());
        // The integer forms ignore the tolerance.
        let t = crate::tolerant::Tolerance::DEFAULT;
        for d in [7, -12, 1, -1, 0, i64::MIN] {
            let mut out = Vec::new();
            assert!(Residue(t).many(Arg::One(d), Arg::Many(&ys), ys.len(), &mut out));
            assert_eq!(out, ys.iter().map(|&y| residue(d, y)).collect::<Vec<_>>());
        }
        let mut out = vec![5];
        let ok = Gcd(t).many(Arg::Many(&xs), Arg::Many(&ys), xs.len(), &mut out);
        assert_eq!(ok, xs.iter().zip(&ys).all(|(&x, &y)| gcd(x, y).is_some()));
        assert_eq!(out[0], 5);
        assert_eq!(Binary::<i64>::checked(&Pow, 3, 4), Some(81));
        assert_eq!(Binary::<i64>::checked(&Pow, 3, 40), None);
        assert_eq!(Binom.one(5i64, 2), 10);
        assert_eq!(Lcm(t).one(-4i64, 6), -12);
    }

    #[test]
    fn other_widths() {
        let t = crate::tolerant::Tolerance::DEFAULT;
        // Every pair of `i8`s agrees with the `i64` functions where the result fits, and fails where it doesn't.
        for x in i8::MIN..=i8::MAX {
            for y in i8::MIN..=i8::MAX {
                let (a, b) = (i64::from(x), i64::from(y));
                let fit = |r: Option<i64>| r.and_then(|r| i8::try_from(r).ok());
                assert_eq!(Binary::<i8>::checked(&Gcd(t), x, y), fit(gcd(a, b)), "gcd({x}, {y})");
                assert_eq!(Binary::<i8>::checked(&Lcm(t), x, y), fit(lcm(a, b)), "lcm({x}, {y})");
                assert_eq!(Binary::<i8>::checked(&Pow, x, y), fit(power(a, b)), "power({x}, {y})");
                assert_eq!(i64::from(Residue(t).one(x, y)), residue(a, b), "residue({x}, {y})");
            }
        }
        // `u64`s, some beyond `i64::MAX`, against exact arithmetic.
        let big = [0, 1, 2, 3, 12, 18, 1 << 32, 1 << 63, (1 << 63) + 1, u64::MAX - 1, u64::MAX];
        for &x in &big {
            for &y in &big {
                let (a, b) = (u128::from(x), u128::from(y));
                let g = num_integer::Integer::gcd(&a, &b);
                assert_eq!(u128::from(Gcd(t).one(x, y)), g, "gcd({x}, {y})");
                assert_eq!(Binary::<u64>::checked(&Lcm(t), x, y), if g == 0 { Some(0) } else { u64::try_from(a / g * b).ok() }, "lcm({x}, {y})");
                assert_eq!(Residue(t).one(x, y), if x == 0 { y } else { y % x }, "residue({x}, {y})");
                assert_eq!((Equal(t).one(x, y), Less(t).one(x, y)), (x == y, x < y), "comparisons of {x} and {y}");
            }
            for y in 0..70u64 {
                assert_eq!(Binary::<u64>::checked(&Pow, x, y), BigInt::from(x).pow(y as u32).try_into().ok(), "power({x}, {y})");
            }
        }
        // Integers compare exactly, however close they are within the tolerance.
        let (n, m) = (1_000_000_000_000_000i64, 1_000_000_000_000_001i64);
        assert_eq!([Equal(t).one(n, m), NotEqual(t).one(n, m), LessEqual(t).one(m, n), GreaterEqual(t).one(m, n), Greater(t).one(m, n)], [false, true, false, true, true]);
        assert_eq!((Floor(t).one(m), Ceil(t).one(-m), Floor(t).one(true)), (m, -m, true));
        assert_eq!((Abs.one(i8::MIN), Abs.one(-5i16), Abs.one(7u8), Abs.one(false)), (i8::MIN, 5, 7, false));
        assert_eq!((Sign.one(-9i32), Sign.one(0i64), Sign.one(200u8)), (-1, 0, 1));
    }
}
