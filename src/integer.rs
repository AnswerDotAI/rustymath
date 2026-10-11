//! Functions of `i64`s: greatest common divisor, least common multiple, binomial coefficients, powers, square roots and residues.
//! Each gives `None` where its result isn't an `i64`. Their function objects are the ones that floats and complex numbers use:
//! [`Gcd`], [`Lcm`] and [`Residue`] from [`tolerant`](crate::tolerant), whose integer forms ignore the tolerance they hold, and
//! [`Binom`] and [`Pow`]. A slice form appends results to a vector and reports whether every item fitted. [`Divisor`] divides many
//! integers by one, with multiplications in place of division instructions. `Residue`'s slice form uses one where every item has
//! the same divisor.
use crate::{
    tolerant::{Gcd, Lcm, Residue},
    Arg, Binom, Pow,
};
use strength_reduce::StrengthReducedU64;

/// The greatest common divisor of x and y, which is never negative. `gcd(0, 0)` is 0. `None` only where it is 2^63, as for
/// `gcd(i64::MIN, 0)`.
pub fn gcd(x: i64, y: i64) -> Option<i64> {
    let (mut a, mut b) = (x.unsigned_abs(), y.unsigned_abs());
    if a == 0 || b == 0 { return i64::try_from(a | b).ok() }
    // Stein's algorithm: shifts and subtractions, with no division.
    let shift = (a | b).trailing_zeros();
    a >>= a.trailing_zeros();
    loop {
        b >>= b.trailing_zeros();
        if a > b { std::mem::swap(&mut a, &mut b) }
        b -= a;
        if b == 0 { return i64::try_from(a << shift).ok() }
    }
}

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{accuracy::random, Binary};
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
}
