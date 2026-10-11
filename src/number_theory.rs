//! Number theory on unsigned integers: primality, factorisation, next and previous primes, Euler's totient, and the
//! enumeration and counting of primes.
//!
//! Each function on a single integer is generic over [`Natural`], which `u64`, `u128` and `BigUint` implement. A `u128` or
//! `BigUint` small enough for a narrower type computes in that type. Primality uses the Baillie–PSW test, which no known
//! composite passes and which has no exceptions below 2^64. Factorisation uses trial division, then Brent's form of
//! Pollard's rho. `u64` and `u128` compute in Montgomery form, with no allocation.
//!
//! A function that can run long has a `_with` version, which takes a check as its last argument. It calls the check at
//! least about once a millisecond, and returns the check's error as soon as it gets one.
//!
//! ```
//! use rustymath::number_theory::{factor, is_prime, next_prime, Primes};
//! assert!(is_prime(&((1u128 << 127) - 1)));
//! assert_eq!(factor(&600851475143u64), [(71, 1), (839, 1), (1471, 1), (6857, 1)]);
//! assert_eq!(next_prime(&u64::MAX), None);
//! assert_eq!(Primes::new(0).nth(10_000), Some(104_743));
//! ```
use crate::{ok, BETWEEN_CHECKS};
use num_bigint::BigUint;
use num_integer::{Integer, Roots};
use num_traits::{CheckedAdd, ToPrimitive};
use std::fmt::Debug;

/// An unsigned integer type that the number theory functions take: `u64`, `u128` or `BigUint`.
pub trait Natural: Integer + Roots + CheckedAdd + Clone + Debug + From<u64> + ToPrimitive + sealed::Tier {}
impl Natural for u64 {}
impl Natural for u128 {}
impl Natural for BigUint {}

mod sealed {
    use super::Natural;
    pub trait Tier: Sized {
        /// The next narrower type, or `u64` itself.
        type Narrow: Natural;
        /// `self` in `Narrow`, if `Narrow` is narrower and `self` fits it.
        fn narrow(&self) -> Option<Self::Narrow>;
        fn widen(x: Self::Narrow) -> Self;
        fn bits(&self) -> u64;
        fn bit(&self, i: u64) -> bool;
        fn rem_small(&self, m: u64) -> u64;
        /// Whether `n`, which is odd and has no factor below 41, passes the Baillie–PSW test, computed in this type.
        fn probable_prime<E>(n: &Self, check: &mut impl FnMut() -> Result<(), E>) -> Result<bool, E>;
        /// A factor of `n` other than 1 and `n`, which is odd and composite, computed in this type.
        fn divisor<E>(n: &Self, check: &mut impl FnMut() -> Result<(), E>) -> Result<Self, E>;
    }
}
use sealed::Tier;

/// Residues modulo an odd `n`, with the operations the tests and the factor search need.
trait Ring {
    type N: Natural;
    type E: Clone + PartialEq;
    fn n(&self) -> &Self::N;
    /// `x` modulo `n`.
    fn elem(&self, x: u64) -> Self::E;
    fn mul(&self, a: &Self::E, b: &Self::E) -> Self::E;
    fn add(&self, a: &Self::E, b: &Self::E) -> Self::E;
    fn sub(&self, a: &Self::E, b: &Self::E) -> Self::E;
    /// `a/2`, which exists because `n` is odd.
    fn half(&self, a: &Self::E) -> Self::E;
    /// The gcd of `n` and the integer that `a` stands for, up to a factor that is coprime to `n`.
    fn gcd(&self, a: &Self::E) -> Self::N;
    fn pow(&self, a: &Self::E, e: &Self::N) -> Self::E {
        let mut r = self.elem(1);
        for i in (0..e.bits()).rev() {
            r = self.mul(&r, &r);
            if e.bit(i) { r = self.mul(&r, a); }
        }
        r
    }
}

/// Montgomery form for a primitive type, given its double-width product as `(high, low)` words.
macro_rules! montgomery {
    ($name:ident, $t:ty, $wide:expr) => {
        /// Residues modulo `n` in Montgomery form: `x` is stored as `xR mod n`, where `R` is 2 to the type's width.
        struct $name {
            n: $t,
            /// -1/n modulo `R`.
            inv: $t,
            /// `R² mod n`.
            square: $t,
        }
        impl $name {
            fn new(n: $t) -> Self {
                // An odd n is its own inverse modulo 8, and each step doubles the number of correct bits.
                let mut inv = n;
                for _ in 0..7 { inv = inv.wrapping_mul((2 as $t).wrapping_sub(n.wrapping_mul(inv))); }
                let one = n.wrapping_neg() % n;
                let mut ring = Self { n, inv: inv.wrapping_neg(), square: one };
                for _ in 0..<$t>::BITS { ring.square = ring.add(&ring.square, &ring.square); }
                ring
            }
            #[inline]
            fn redc(&self, (hi, lo): ($t, $t)) -> $t {
                let (mh, ml) = $wide(lo.wrapping_mul(self.inv), self.n);
                let (t, over) = hi.overflowing_add(mh);
                let (t, carry) = t.overflowing_add(lo.overflowing_add(ml).1 as $t);
                if over || carry || t >= self.n { t.wrapping_sub(self.n) } else { t }
            }
        }
        impl Ring for $name {
            type N = $t;
            type E = $t;
            fn n(&self) -> &$t { &self.n }
            fn elem(&self, x: u64) -> $t { self.mul(&(<$t>::from(x) % self.n), &self.square) }
            #[inline]
            fn mul(&self, a: &$t, b: &$t) -> $t { self.redc($wide(*a, *b)) }
            #[inline]
            fn add(&self, a: &$t, b: &$t) -> $t {
                let (s, over) = a.overflowing_add(*b);
                if over || s >= self.n { s.wrapping_sub(self.n) } else { s }
            }
            #[inline]
            fn sub(&self, a: &$t, b: &$t) -> $t { if a >= b { a - b } else { a.wrapping_sub(*b).wrapping_add(self.n) } }
            fn half(&self, a: &$t) -> $t { if a % 2 == 0 { a >> 1 } else { (a >> 1) + (self.n >> 1) + 1 } }
            fn gcd(&self, a: &$t) -> $t { a.gcd(&self.n) }
        }
    };
}
montgomery!(Montgomery64, u64, |a: u64, b: u64| {
    let p = a as u128 * b as u128;
    ((p >> 64) as u64, p as u64)
});
montgomery!(Montgomery128, u128, |a: u128, b: u128| {
    let (low, (a1, a0), (b1, b0)) = (u64::MAX as u128, (a >> 64, a & u64::MAX as u128), (b >> 64, b & u64::MAX as u128));
    let (p00, p01, p10) = (a0 * b0, a0 * b1, a1 * b0);
    let mid = (p00 >> 64) + (p01 & low) + (p10 & low);
    (a1 * b1 + (p01 >> 64) + (p10 >> 64) + (mid >> 64), (p00 & low) | (mid << 64))
});

/// Plain residues modulo a `BigUint`.
struct Residues(BigUint);
impl Ring for Residues {
    type N = BigUint;
    type E = BigUint;
    fn n(&self) -> &BigUint { &self.0 }
    fn elem(&self, x: u64) -> BigUint { BigUint::from(x) % &self.0 }
    fn mul(&self, a: &BigUint, b: &BigUint) -> BigUint { a * b % &self.0 }
    fn add(&self, a: &BigUint, b: &BigUint) -> BigUint {
        let s = a + b;
        if s >= self.0 { s - &self.0 } else { s }
    }
    fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint { if a >= b { a - b } else { a + &self.0 - b } }
    fn half(&self, a: &BigUint) -> BigUint { if a.is_even() { a >> 1 } else { (a + &self.0) >> 1 } }
    fn gcd(&self, a: &BigUint) -> BigUint { a.gcd(&self.0) }
    fn pow(&self, a: &BigUint, e: &BigUint) -> BigUint { a.modpow(e, &self.0) }
}

macro_rules! primitive {
    ($t:ty, $ring:ident, $narrow:ty, $narrowed:expr) => {
        impl Tier for $t {
            type Narrow = $narrow;
            fn narrow(&self) -> Option<$narrow> { ($narrowed)(self) }
            fn widen(x: $narrow) -> $t { x.into() }
            fn bits(&self) -> u64 { (<$t>::BITS - self.leading_zeros()) as u64 }
            fn bit(&self, i: u64) -> bool { self >> i & 1 == 1 }
            fn rem_small(&self, m: u64) -> u64 { (self % m as $t) as u64 }
            fn probable_prime<E>(n: &$t, check: &mut impl FnMut() -> Result<(), E>) -> Result<bool, E> {
                baillie_psw(&$ring::new(*n), check)
            }
            fn divisor<E>(n: &$t, check: &mut impl FnMut() -> Result<(), E>) -> Result<$t, E> { rho(&$ring::new(*n), check) }
        }
    };
}
primitive!(u64, Montgomery64, u64, |_| None);
primitive!(u128, Montgomery128, u64, |n: &u128| u64::try_from(*n).ok());
impl Tier for BigUint {
    type Narrow = u128;
    fn narrow(&self) -> Option<u128> { self.to_u128() }
    fn widen(x: u128) -> BigUint { x.into() }
    fn bits(&self) -> u64 { BigUint::bits(self) }
    fn bit(&self, i: u64) -> bool { BigUint::bit(self, i) }
    fn rem_small(&self, m: u64) -> u64 { (self % m).to_u64().unwrap() }
    fn probable_prime<E>(n: &BigUint, check: &mut impl FnMut() -> Result<(), E>) -> Result<bool, E> {
        baillie_psw(&Residues(n.clone()), check)
    }
    fn divisor<E>(n: &BigUint, check: &mut impl FnMut() -> Result<(), E>) -> Result<BigUint, E> { rho(&Residues(n.clone()), check) }
}

/// `n` as `d·2^s` with `d` odd.
fn odd_part<N: Natural>(mut d: N) -> (N, u64) {
    let mut s = 0;
    while d.is_even() {
        d = d / N::from(2);
        s += 1;
    }
    (d, s)
}


/// The Jacobi symbol (a/n) for odd `n`.
fn jacobi_small(mut a: u64, mut n: u64) -> i32 {
    let mut t = 1;
    a %= n;
    while a != 0 {
        while a.is_multiple_of(2) {
            a /= 2;
            if n % 8 == 3 || n % 8 == 5 { t = -t; }
        }
        std::mem::swap(&mut a, &mut n);
        if a % 4 == 3 && n % 4 == 3 { t = -t; }
        a %= n;
    }
    if n == 1 { t } else { 0 }
}

/// The Jacobi symbol (d/n) for odd `d` and odd `n`, by reciprocity, so that only `n mod |d|` and `n mod 4` are needed.
fn jacobi<N: Natural>(d: i64, n: &N) -> i32 {
    let (a, n4) = (d.unsigned_abs(), n.rem_small(4));
    let flips = (a % 4 == 3 && n4 == 3) as u32 + (d < 0 && n4 == 3) as u32;
    jacobi_small(n.rem_small(a), a) * if flips % 2 == 1 { -1 } else { 1 }
}

/// Whether `n` is a strong probable prime to `base`, where `n - 1 = d·2^s`.
fn strong_probable_prime<R: Ring>(ring: &R, d: &R::N, s: u64, base: &R::E) -> bool {
    let (one, minus) = (ring.elem(1), ring.sub(&ring.elem(0), &ring.elem(1)));
    let mut x = ring.pow(base, d);
    if x == one || x == minus { return true; }
    for _ in 1..s {
        x = ring.mul(&x, &x);
        if x == minus { return true; }
    }
    false
}

/// Whether `n` is a strong Lucas probable prime, with Selfridge's parameters: the first `D` in 5, -7, 9, -11, … whose Jacobi
/// symbol is -1, `P = 1` and `Q = (1 - D)/4`. `n` is odd, not a square, and larger than any `D` the search reaches.
fn strong_lucas<R: Ring>(ring: &R) -> bool {
    let n = ring.n();
    let root = n.sqrt();
    if root.clone() * root == *n { return false; }
    let mut d = 5i64;
    loop {
        match jacobi(d, n) {
            -1 => break,
            0 => return false,
            _ => d = if d > 0 { -d - 2 } else { 2 - d },
        }
    }
    let signed = |x: i64| if x < 0 { ring.sub(&ring.elem(0), &ring.elem(x.unsigned_abs())) } else { ring.elem(x as u64) };
    let (d, q) = (signed(d), signed((1 - d) / 4));
    // U and V for successive prefixes of k's bits, from the top: doubling k, then adding 1 where the next bit is set.
    let (k, s) = odd_part(n.clone() + R::N::from(1));
    let (mut u, mut v, mut qk) = (ring.elem(1), ring.elem(1), q.clone());
    for i in (0..k.bits() - 1).rev() {
        u = ring.mul(&u, &v);
        v = ring.sub(&ring.mul(&v, &v), &ring.add(&qk, &qk));
        qk = ring.mul(&qk, &qk);
        if k.bit(i) {
            (u, v) = (ring.half(&ring.add(&u, &v)), ring.half(&ring.add(&ring.mul(&d, &u), &v)));
            qk = ring.mul(&qk, &q);
        }
    }
    let zero = ring.elem(0);
    if u == zero || v == zero { return true; }
    for _ in 1..s {
        v = ring.sub(&ring.mul(&v, &v), &ring.add(&qk, &qk));
        qk = ring.mul(&qk, &qk);
        if v == zero { return true; }
    }
    false
}

fn baillie_psw<R: Ring, E>(ring: &R, check: &mut impl FnMut() -> Result<(), E>) -> Result<bool, E> {
    let (d, s) = odd_part(ring.n().clone() - R::N::from(1));
    check()?;
    if !strong_probable_prime(ring, &d, s, &ring.elem(2)) { return Ok(false); }
    check()?;
    Ok(strong_lucas(ring))
}

/// A factor of the odd composite `n` other than 1 and `n`, by Brent's form of Pollard's rho, which takes one gcd for each
/// batch of 128 steps.
fn rho<R: Ring, E>(ring: &R, check: &mut impl FnMut() -> Result<(), E>) -> Result<R::N, E> {
    let (n, one) = (ring.n(), R::N::from(1));
    for c in 1.. {
        let c = ring.elem(c);
        let step = |x: &R::E| ring.add(&ring.mul(x, x), &c);
        let (mut y, mut r) = (ring.elem(2), 1u64);
        loop {
            let x = y.clone();
            for i in 0..r {
                if i % BETWEEN_CHECKS as u64 == 0 { check()?; }
                y = step(&y);
            }
            let (mut k, mut g, mut saved) = (0, one.clone(), y.clone());
            while k < r && g == one {
                check()?;
                saved = y.clone();
                let mut product = ring.elem(1);
                for _ in 0..128.min(r - k) {
                    y = step(&y);
                    product = ring.mul(&product, &ring.sub(&x, &y));
                }
                g = ring.gcd(&product);
                k += 128;
            }
            if g == *n {
                // The batch passed the factor: retrace it one step at a time.
                loop {
                    check()?;
                    saved = step(&saved);
                    g = ring.gcd(&ring.sub(&x, &saved));
                    if g != one { break; }
                }
            }
            if g == *n { break; }
            if g != one { return Ok(g); }
            r *= 2;
        }
    }
    unreachable!()
}

/// The primes that trial division tries before the other methods. A number below 41² with none of them as a factor is prime.
const SMALL_PRIMES: [u64; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
/// The product of `SMALL_PRIMES`. One remainder by it gives the remainders by all of them.
const PRIMORIAL: u64 = product(&SMALL_PRIMES);

const fn product(xs: &[u64]) -> u64 {
    let (mut product, mut i) = (1, 0);
    while i < xs.len() {
        product *= xs[i];
        i += 1;
    }
    product
}

/// A factor of the odd composite `n` other than 1 and `n`, found in the narrowest type `n` fits.
fn narrow_divisor<N: Natural, E>(n: &N, check: &mut impl FnMut() -> Result<(), E>) -> Result<N, E> {
    match n.narrow() {
        Some(x) => Ok(N::widen(narrow_divisor(&x, check)?)),
        None => N::divisor(n, check),
    }
}

/// Whether `n` is prime.
pub fn is_prime_with<N: Natural, E>(n: &N, mut check: impl FnMut() -> Result<(), E>) -> Result<bool, E> {
    if let Some(x) = n.narrow() { return is_prime_with(&x, check); }
    if let Some(x) = n.to_u64().filter(|&x| x < 41 * 41) { return Ok(x >= 2 && SMALL_PRIMES.iter().all(|&p| x == p || x % p != 0)); }
    let r = n.rem_small(PRIMORIAL);
    if SMALL_PRIMES.iter().any(|&p| r % p == 0) { return Ok(false); }
    N::probable_prime(n, &mut check)
}
/// [`is_prime_with`] with no check.
pub fn is_prime<N: Natural>(n: &N) -> bool {
    let Ok(p) = is_prime_with(n, ok);
    p
}

/// The prime factors of `n` in increasing order, each with its multiplicity. `factor(1)` is empty.
///
/// # Panics
/// If `n` is 0.
pub fn factor_with<N: Natural, E>(n: &N, mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<(N, u32)>, E> {
    assert!(!n.is_zero(), "factor needs a positive integer");
    if let Some(x) = n.narrow() { return Ok(factor_with(&x, check)?.into_iter().map(|(p, k)| (N::widen(p), k)).collect()); }
    let (mut n, mut found) = (n.clone(), Vec::new());
    let r = n.rem_small(PRIMORIAL);
    for p in SMALL_PRIMES.into_iter().filter(|p| r % p == 0) {
        while n.rem_small(p) == 0 {
            found.push(N::from(p));
            n = n / N::from(p);
        }
    }
    let mut pending = vec![n];
    while let Some(m) = pending.pop() {
        check()?;
        if m.is_one() { continue; }
        if is_prime_with(&m, &mut check)? {
            found.push(m);
        } else {
            let d = narrow_divisor(&m, &mut check)?;
            pending.push(m / d.clone());
            pending.push(d);
        }
    }
    found.sort_unstable();
    let mut result: Vec<(N, u32)> = Vec::new();
    for p in found {
        match result.last_mut() {
            Some((q, count)) if *q == p => *count += 1,
            _ => result.push((p, 1)),
        }
    }
    Ok(result)
}
/// [`factor_with`] with no check.
pub fn factor<N: Natural>(n: &N) -> Vec<(N, u32)> {
    let Ok(f) = factor_with(n, ok);
    f
}

/// The smallest prime above `n`, or `None` if it doesn't fit `N`.
pub fn next_prime_with<N: Natural, E>(n: &N, mut check: impl FnMut() -> Result<(), E>) -> Result<Option<N>, E> {
    let two = N::from(2);
    if *n < two { return Ok(Some(two)); }
    let Some(mut c) = n.checked_add(&N::from(if n.is_even() { 1 } else { 2 })) else { return Ok(None) };
    loop {
        check()?;
        if is_prime_with(&c, &mut check)? { return Ok(Some(c)); }
        let Some(next) = c.checked_add(&two) else { return Ok(None) };
        c = next;
    }
}
/// [`next_prime_with`] with no check.
pub fn next_prime<N: Natural>(n: &N) -> Option<N> {
    let Ok(p) = next_prime_with(n, ok);
    p
}

/// The largest prime below `n`, or `None` if `n` is 2 or less.
pub fn prev_prime_with<N: Natural, E>(n: &N, mut check: impl FnMut() -> Result<(), E>) -> Result<Option<N>, E> {
    if let Some(x) = n.narrow() { return Ok(prev_prime_with(&x, check)?.map(N::widen)); }
    let two = N::from(2);
    if *n <= two { return Ok(None); }
    if *n == N::from(3) { return Ok(Some(two)); }
    let mut c = n.clone() - N::from(if n.is_even() { 1 } else { 2 });
    loop {
        check()?;
        if is_prime_with(&c, &mut check)? { return Ok(Some(c)); }
        c = c - two.clone();
    }
}
/// [`prev_prime_with`] with no check.
pub fn prev_prime<N: Natural>(n: &N) -> Option<N> {
    let Ok(p) = prev_prime_with(n, ok);
    p
}

/// Euler's totient: how many integers from 1 to `n` are coprime to `n`.
///
/// # Panics
/// If `n` is 0.
pub fn totient_with<N: Natural, E>(n: &N, check: impl FnMut() -> Result<(), E>) -> Result<N, E> {
    if let Some(x) = n.narrow() { return Ok(N::widen(totient_with(&x, check)?)); }
    Ok(factor_with(n, check)?.into_iter().fold(n.clone(), |t, (p, _)| t / p.clone() * (p - N::from(1))))
}
/// [`totient_with`] with no check.
pub fn totient<N: Natural>(n: &N) -> N {
    let Ok(t) = totient_with(n, ok);
    t
}

/// The primes below 7. The sieve stores only numbers that none of them divide, so it gives these separately.
const WHEEL: [u64; 3] = [2, 3, 5];
/// The residues modulo 30 of the numbers that `WHEEL`'s primes don't divide. A segment has one byte for each 30 numbers.
/// In byte `b`, bit `j` stands for the number `30b + RESIDUES[j]` past the segment's start.
const RESIDUES: [u64; 8] = [1, 7, 11, 13, 17, 19, 23, 29];
/// The bit that stands for `x mod 30` in its byte, for `x` coprime to 30.
const BIT: [u8; 30] = {
    let (mut bits, mut j) = ([0; 30], 0);
    while j < 8 {
        bits[RESIDUES[j] as usize] = 1 << j;
        j += 1;
    }
    bits
};
/// The primes whose multiples each segment starts without, copied from a pattern instead of sieved.
const PRESIEVED: [u64; 1] = [7];
/// The bytes of a full segment. 32 KiB fits the L1 data cache of most CPUs.
const BYTES: usize = 32768;
/// The bytes of the first segment. Segments double from it up to `BYTES`, so finding a few primes takes little sieving.
const FIRST: usize = 128;
/// The bytes after which the pattern of `PRESIEVED`'s multiples repeats.
const PERIOD: u64 = product(&PRESIEVED);

/// The bytes that multiples of `PRESIEVED` leave, long enough to fill a segment from any phase.
fn pattern() -> &'static [u8] {
    static PATTERN: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| {
        (0..BYTES as u64 + PERIOD)
            .map(|b| (0..8).filter(|&j| PRESIEVED.iter().all(|p| (30 * b + RESIDUES[j]) % p != 0)).fold(0, |m, j| m | 1 << j))
            .collect()
    })
}

/// The primes from a given number upward, found a segment at a time by a sieve of Eratosthenes. A segment stores only the
/// numbers that 2, 3 and 5 don't divide, and starts from a copy of the pattern that multiples of `PRESIEVED` leave.
///
/// Sieving numbers near n needs the primes below √n, which `Primes` finds first. A start far above 2^40 therefore costs
/// memory in proportion to its square root. `next_prime` suits that range better.
pub struct Primes {
    /// Where the current segment starts, a multiple of 30.
    lo: u64,
    /// Where the next segment starts, or `None` past `u64::MAX`.
    next: Option<u64>,
    /// The current segment, with a bit set for each prime.
    bytes: Vec<u8>,
    /// The next bit to read.
    at: usize,
    /// The bits to skip in the first segment, which stand for numbers below the start.
    skip: usize,
    /// The primes from 11 to below `limit`, which sieve every segment below `limit²`.
    sieving: Vec<u64>,
    limit: u64,
    /// The bytes of the next segment.
    size: usize,
    /// How many of `WHEEL`'s primes have been given or skipped.
    wheel: usize,
}

impl Primes {
    /// The primes from `start` upward.
    pub fn new(start: u64) -> Self {
        let lo = start / 30 * 30;
        Self {
            lo,
            next: Some(lo),
            bytes: Vec::new(),
            at: 0,
            skip: RESIDUES.iter().filter(|&&r| lo + r < start).count(),
            sieving: Vec::new(),
            limit: 1,
            size: FIRST,
            wheel: WHEEL.iter().filter(|&&p| p < start).count(),
        }
    }

    fn sieve(&mut self) {
        let lo = self.next.expect("sieving past u64::MAX");
        let len = self.size;
        let hi = lo.saturating_add(30 * len as u64);
        let root = hi.sqrt() + 1;
        if root > self.limit {
            self.limit = root.max(2 * self.limit);
            self.sieving = odd_primes_below(self.limit).into_iter().filter(|&p| p > PRESIEVED[PRESIEVED.len() - 1]).collect();
        }
        let phase = ((lo / 30) % PERIOD) as usize;
        self.bytes.clear();
        self.bytes.extend_from_slice(&pattern()[phase..phase + len]);
        let bytes = &mut self.bytes[..len];
        // The multiples p·k of each prime p with k ≡ r (mod 30) are p bytes apart, all on the same bit.
        for &p in &self.sieving {
            if p.saturating_mul(p) >= hi { break; }
            let k0 = p.max(lo.div_ceil(p));
            for r in RESIDUES {
                let Some(m) = p.checked_mul(k0 + (r + 30 - k0 % 30) % 30).filter(|&m| m < hi) else { continue };
                let (mut i, mask) = (((m - lo) / 30) as usize, !BIT[(m % 30) as usize]);
                while i < len {
                    bytes[i] &= mask;
                    i += p as usize;
                }
            }
        }
        if lo == 0 {
            bytes[0] &= !BIT[1];
            for p in PRESIEVED { bytes[0] |= BIT[p as usize]; }
        }
        let top = u64::MAX - lo;
        if top < 30 * len as u64 {
            for b in (top / 30) as usize..len {
                for (j, r) in RESIDUES.into_iter().enumerate() {
                    if 30 * b as u64 + r > top { bytes[b] &= !(1 << j); }
                }
            }
        }
        let next = lo.checked_add(30 * len as u64);
        (self.lo, self.next, self.at, self.skip, self.size) = (lo, next, self.skip, 0, (2 * len).min(BYTES));
    }

    /// Skips up to `k` primes in the current segment, and gives how many more are still to skip.
    fn skip(&mut self, mut k: u64) -> u64 {
        while self.at < self.bytes.len() * 8 {
            let at = self.at / 64 * 8;
            let mut w = u64::from_le_bytes(self.bytes[at..at + 8].try_into().unwrap()) >> (self.at % 64);
            let count = w.count_ones() as u64;
            if k < count {
                for _ in 0..k { w &= w - 1; }
                self.at += w.trailing_zeros() as usize;
                return 0;
            }
            k -= count;
            self.at = (self.at / 64 + 1) * 64;
        }
        k
    }

    /// The next prime, or `None` past `u64::MAX`.
    pub fn next_with<E>(&mut self, check: impl FnMut() -> Result<(), E>) -> Result<Option<u64>, E> { self.nth_with(0, check) }

    /// The prime `k` places on: `nth_with(0, …)` is the next prime. Gives `None` past `u64::MAX`.
    pub fn nth_with<E>(&mut self, mut k: u64, mut check: impl FnMut() -> Result<(), E>) -> Result<Option<u64>, E> {
        while self.wheel < WHEEL.len() {
            self.wheel += 1;
            if k == 0 { return Ok(Some(WHEEL[self.wheel - 1])); }
            k -= 1;
        }
        loop {
            k = self.skip(k);
            if k == 0 && self.at < self.bytes.len() * 8 {
                let p = self.lo + 30 * (self.at / 8) as u64 + RESIDUES[self.at % 8];
                self.at += 1;
                return Ok(Some(p));
            }
            if self.next.is_none() { return Ok(None); }
            check()?;
            self.sieve();
        }
    }
}

impl Iterator for Primes {
    type Item = u64;
    fn next(&mut self) -> Option<u64> {
        let Ok(p) = self.next_with(ok);
        p
    }
    fn nth(&mut self, k: usize) -> Option<u64> {
        let Ok(p) = self.nth_with(k as u64, ok);
        p
    }
}

/// How many primes are below `n`.
pub fn prime_count_with<E>(n: u64, mut check: impl FnMut() -> Result<(), E>) -> Result<u64, E> {
    let mut count = WHEEL.iter().filter(|&&p| p < n).count() as u64;
    let mut primes = Primes::new(0);
    while let Some(lo) = primes.next.filter(|&lo| lo + 1 < n) {
        check()?;
        primes.sieve();
        let (full, rest) = ((((n - lo) / 30) as usize).min(primes.bytes.len()), (n - lo) % 30);
        count += primes.bytes[..full].iter().map(|b| b.count_ones() as u64).sum::<u64>();
        if full < primes.bytes.len() {
            let below = RESIDUES.iter().enumerate().filter(|&(_, &r)| r < rest).fold(0u8, |m, (j, _)| m | 1 << j);
            count += (primes.bytes[full] & below).count_ones() as u64;
        }
    }
    Ok(count)
}
/// [`prime_count_with`] with no check.
pub fn prime_count(n: u64) -> u64 {
    let Ok(c) = prime_count_with(n, ok);
    c
}

/// The odd primes below `limit`, by a plain sieve.
fn odd_primes_below(limit: u64) -> Vec<u64> {
    let mut composite = vec![false; limit as usize];
    let mut primes = Vec::new();
    for i in (3..limit as usize).step_by(2) {
        if composite[i] { continue; }
        primes.push(i as u64);
        for j in (i * i..limit as usize).step_by(2 * i) { composite[j] = true; }
    }
    primes
}

#[cfg(test)]
mod tests {
    use super::*;

    const M61: u64 = (1 << 61) - 1;
    const M89: u128 = (1 << 89) - 1;
    const M127: u128 = (1 << 127) - 1;
    /// The largest prime below 2^64.
    const TOP: u64 = u64::MAX - 58;

    #[test]
    fn primality() {
        let mut primes = Primes::new(0).peekable();
        for n in 0..1u64 << 17 {
            let prime = primes.next_if_eq(&n).is_some();
            assert!(is_prime(&n) == prime && is_prime(&(n as u128)) == prime && is_prime(&BigUint::from(n)) == prime, "{n}");
        }
        assert!(is_prime(&M61) && is_prime(&TOP) && is_prime(&M89) && is_prime(&M127));
        assert!(is_prime(&((BigUint::from(1u8) << 521) - 1u8)));
        // Strong pseudoprimes to every prime base up to 23, 37 and 41.
        assert!(!is_prime(&3825123056546413051u64));
        assert!(!is_prime(&318665857834031151167461u128));
        assert!(!is_prime(&3317044064679887385961981u128));
        assert!(!is_prime(&(BigUint::from(M89) * BigUint::from(M127))));
    }

    #[test]
    fn halves() {
        // Each half of Baillie–PSW rejects composites that pass the other half.
        let ring = Montgomery64::new(3825123056546413051);
        let (d, s) = odd_part(3825123056546413050u64);
        assert!(strong_probable_prime(&ring, &d, s, &ring.elem(2)) && !strong_lucas(&ring));
        // Strong Lucas pseudoprimes with Selfridge's parameters, from OEIS A217255.
        for n in [5459u64, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519] {
            let ring = Montgomery64::new(n);
            let (d, s) = odd_part(n - 1);
            assert!(strong_lucas(&ring) && !strong_probable_prime(&ring, &d, s, &ring.elem(2)), "{n}");
        }
    }

    #[test]
    fn factors() {
        assert!(factor(&1u64).is_empty());
        assert_eq!(factor(&u64::MAX), [(3, 1), (5, 1), (17, 1), (257, 1), (641, 1), (65537, 1), (6700417, 1)]);
        assert_eq!(factor(&((1u128 << 64) + 1)), [(274177, 1), (67280421310721, 1)]);
        assert_eq!(factor(&(((1u128 << 31) - 1) * M61 as u128)), [((1 << 31) - 1, 1), (M61 as u128, 1)]);
        let big = BigUint::from(1000003u64).pow(2) * 1000033u64 * BigUint::from(M127);
        assert_eq!(factor(&big), [(1000003u64.into(), 2), (1000033u64.into(), 1), (M127.into(), 1)]);
        assert_eq!(totient(&36u64), 12);
        assert_eq!(totient(&u64::MAX), 2 * 4 * 16 * 256 * 640 * 65536 * 6700416);
    }

    #[test]
    fn neighbours() {
        assert_eq!([next_prime(&0u64), next_prime(&2), next_prime(&13)], [Some(2), Some(3), Some(17)]);
        assert_eq!([prev_prime(&2u64), prev_prime(&3), prev_prime(&100)], [None, Some(2), Some(97)]);
        assert_eq!(next_prime(&(u64::MAX - 70)), Some(TOP));
        assert_eq!(next_prime(&TOP), None);
        assert_eq!(next_prime(&(TOP as u128)), Some((1 << 64) + 13));
        assert_eq!(prev_prime(&(1u128 << 64)), Some(TOP as u128));
    }

    #[test]
    fn counting() {
        let mut n = 1;
        for expected in [4, 25, 168, 1229, 9592, 78498, 664579] {
            n *= 10;
            assert_eq!(prime_count(n), expected);
        }
        // Counting and enumeration agree on each side of a segment boundary.
        // Counting and enumeration agree on each side of the first segments' ends.
        let (mut end, mut size) = (0, FIRST as u64);
        for _ in 0..4 {
            end += 30 * size;
            size = (2 * size).min(BYTES as u64);
            for n in [end - 1, end, end + 1] { assert_eq!(prime_count(n), Primes::new(0).take_while(|&p| p < n).count() as u64); }
        }
        let starts = [0, 3, 4, 6, 7, 29, 30, 98].map(|n| Primes::new(n).next().unwrap());
        assert_eq!(starts, [2, 3, 5, 7, 7, 29, 31, 101]);
        assert_eq!(Primes::new(0).nth(999_999), Some(15_485_863));
        let mut primes = Primes::new(0);
        assert_eq!([primes.nth(3), primes.nth(0), primes.nth(2)], [Some(7), Some(11), Some(19)]);
    }

    #[test]
    fn cancellation() {
        let stop = || Err("stopped");
        assert_eq!(factor_with(&((1u128 << 64) + 1), stop), Err("stopped"));
        assert_eq!(prime_count_with(1 << 20, stop), Err("stopped"));
        assert_eq!(Primes::new(0).nth_with(0, stop), Ok(Some(2)));
    }
}
