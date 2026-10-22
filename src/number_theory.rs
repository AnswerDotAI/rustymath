//! Number theory on unsigned integers: primality, factorisation, next and previous primes, Euler's totient, and the enumeration and counting of primes.
//!
//! Each function on a single integer is generic over [`Natural`], which `u64`, `u128` and `BigUint` implement. A `u128` or `BigUint` small enough for a narrower type computes in that type. Primality uses the Baillie–PSW test, which no known composite passes and which has no exceptions below 2^64. Factorisation uses trial division, then Brent's form of Pollard's rho. `u64` and `u128` compute in Montgomery form, with no allocation.
//!
//! A function that can run long calls [`interrupt::check`](crate::interrupt::check) at least about once a millisecond, and returns its error as soon as it gets one.
use crate::{BETWEEN_CHECKS, error::Error, interrupt::check};
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
        fn probable_prime(n: &Self) -> Result<bool, crate::error::Error>;
        /// A factor of `n` other than 1 and `n`, which is odd and composite, computed in this type.
        fn divisor(n: &Self) -> Result<Self, crate::error::Error>;
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
        for i in (0..e.bits()).rev() { r = self.mul(&r, &r); if e.bit(i) { r = self.mul(&r, a); } }
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
                for _ in 0..7 {
                    inv = inv.wrapping_mul((2 as $t).wrapping_sub(n.wrapping_mul(inv)));
                }
                let one = n.wrapping_neg() % n;
                let mut ring = Self { n, inv: inv.wrapping_neg(), square: one };
                for _ in 0..<$t>::BITS {
                    ring.square = ring.add(&ring.square, &ring.square);
                }
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
            fn n(&self) -> &$t {
                &self.n
            }
            fn elem(&self, x: u64) -> $t {
                self.mul(&(<$t>::from(x) % self.n), &self.square)
            }
            #[inline]
            fn mul(&self, a: &$t, b: &$t) -> $t {
                self.redc($wide(*a, *b))
            }
            #[inline]
            fn add(&self, a: &$t, b: &$t) -> $t {
                let (s, over) = a.overflowing_add(*b);
                if over || s >= self.n { s.wrapping_sub(self.n) } else { s }
            }
            #[inline]
            fn sub(&self, a: &$t, b: &$t) -> $t {
                if a >= b { a - b } else { a.wrapping_sub(*b).wrapping_add(self.n) }
            }
            fn half(&self, a: &$t) -> $t {
                if a % 2 == 0 { a >> 1 } else { (a >> 1) + (self.n >> 1) + 1 }
            }
            fn gcd(&self, a: &$t) -> $t {
                a.gcd(&self.n)
            }
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
    fn add(&self, a: &BigUint, b: &BigUint) -> BigUint { let s = a + b; if s >= self.0 { s - &self.0 } else { s } }
    fn sub(&self, a: &BigUint, b: &BigUint) -> BigUint { if a >= b { a - b } else { a + &self.0 - b } }
    fn half(&self, a: &BigUint) -> BigUint { if a.is_even() { a >> 1 } else { (a + &self.0) >> 1 } }
    fn gcd(&self, a: &BigUint) -> BigUint { a.gcd(&self.0) }
    fn pow(&self, a: &BigUint, e: &BigUint) -> BigUint { a.modpow(e, &self.0) }
}

macro_rules! primitive {
    ($t:ty, $ring:ident, $narrow:ty, $narrowed:expr) => {
        impl Tier for $t {
            type Narrow = $narrow;
            fn narrow(&self) -> Option<$narrow> {
                ($narrowed)(self)
            }
            fn widen(x: $narrow) -> $t {
                x.into()
            }
            fn bits(&self) -> u64 {
                (<$t>::BITS - self.leading_zeros()) as u64
            }
            fn bit(&self, i: u64) -> bool {
                self >> i & 1 == 1
            }
            fn rem_small(&self, m: u64) -> u64 {
                (self % m as $t) as u64
            }
            fn probable_prime(n: &$t) -> Result<bool, Error> {
                baillie_psw(&$ring::new(*n))
            }
            fn divisor(n: &$t) -> Result<$t, Error> {
                rho(&$ring::new(*n))
            }
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
    fn probable_prime(n: &BigUint) -> Result<bool, Error> { baillie_psw(&Residues(n.clone())) }
    fn divisor(n: &BigUint) -> Result<BigUint, Error> { rho(&Residues(n.clone())) }
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
        while a.is_multiple_of(2) { a /= 2; if n % 8 == 3 || n % 8 == 5 { t = -t; } }
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
    for _ in 1..s { x = ring.mul(&x, &x); if x == minus { return true; } }
    false
}

/// Whether `n` is a strong Lucas probable prime, with Selfridge's parameters: the first `D` in 5, -7, 9, -11, … whose Jacobi symbol is -1, `P = 1` and `Q = (1 - D)/4`. `n` is odd, not a square, and larger than any `D` the search reaches.
fn strong_lucas<R: Ring>(ring: &R) -> bool {
    let n = ring.n();
    let root = n.sqrt();
    if root.clone() * root == *n { return false; }
    let mut d = 5i64;
    loop { match jacobi(d, n) { -1 => break, 0 => return false, _ => d = if d > 0 { -d - 2 } else { 2 - d } } }
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

fn baillie_psw<R: Ring>(ring: &R) -> Result<bool, Error> {
    let (d, s) = odd_part(ring.n().clone() - R::N::from(1));
    check()?;
    if !strong_probable_prime(ring, &d, s, &ring.elem(2)) { return Ok(false); }
    check()?;
    Ok(strong_lucas(ring))
}

/// A factor of the odd composite `n` other than 1 and `n`, by Brent's form of Pollard's rho, which takes one gcd for each batch of 128 steps.
fn rho<R: Ring>(ring: &R) -> Result<R::N, Error> {
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
fn narrow_divisor<N: Natural>(n: &N) -> Result<N, Error> {
    match n.narrow() { Some(x) => Ok(N::widen(narrow_divisor(&x)?)), None => N::divisor(n) }
}

/// Whether `n` is prime.
pub fn is_prime<N: Natural>(n: &N) -> Result<bool, Error> {
    if let Some(x) = n.narrow() { return is_prime(&x); }
    if let Some(x) = n.to_u64().filter(|&x| x < 41 * 41) { return Ok(x >= 2 && SMALL_PRIMES.iter().all(|&p| x == p || x % p != 0)); }
    let r = n.rem_small(PRIMORIAL);
    if SMALL_PRIMES.iter().any(|&p| r % p == 0) { return Ok(false); }
    N::probable_prime(n)
}

/// The prime factors of `n` in increasing order, each with its multiplicity. `factor(1)` is empty.
///
/// # Panics
/// If `n` is 0.
pub fn factor<N: Natural>(n: &N) -> Result<Vec<(N, u32)>, Error> {
    assert!(!n.is_zero(), "factor needs a positive integer");
    if let Some(x) = n.narrow() { return Ok(factor(&x)?.into_iter().map(|(p, k)| (N::widen(p), k)).collect()); }
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
        if is_prime(&m)? { found.push(m); } else {
            let d = narrow_divisor(&m)?;
            pending.push(m / d.clone());
            pending.push(d);
        }
    }
    found.sort_unstable();
    let mut result: Vec<(N, u32)> = Vec::new();
    for p in found { match result.last_mut() { Some((q, count)) if *q == p => *count += 1, _ => result.push((p, 1)) } }
    Ok(result)
}

/// The smallest prime above `n`, or `None` if it doesn't fit `N`.
pub fn next_prime<N: Natural>(n: &N) -> Result<Option<N>, Error> {
    let two = N::from(2);
    if *n < two { return Ok(Some(two)); }
    let Some(mut c) = n.checked_add(&N::from(if n.is_even() { 1 } else { 2 })) else { return Ok(None) };
    loop {
        check()?;
        if is_prime(&c)? { return Ok(Some(c)); }
        let Some(next) = c.checked_add(&two) else { return Ok(None) };
        c = next;
    }
}

/// The largest prime below `n`, or `None` if `n` is 2 or less.
pub fn prev_prime<N: Natural>(n: &N) -> Result<Option<N>, Error> {
    if let Some(x) = n.narrow() { return Ok(prev_prime(&x)?.map(N::widen)); }
    let two = N::from(2);
    if *n <= two { return Ok(None); }
    if *n == N::from(3) { return Ok(Some(two)); }
    let mut c = n.clone() - N::from(if n.is_even() { 1 } else { 2 });
    loop {
        check()?;
        if is_prime(&c)? { return Ok(Some(c)); }
        c = c - two.clone();
    }
}

/// Euler's totient: how many integers from 1 to `n` are coprime to `n`.
///
/// # Panics
/// If `n` is 0.
pub fn totient<N: Natural>(n: &N) -> Result<N, Error> {
    if let Some(x) = n.narrow() { return Ok(N::widen(totient(&x)?)); }
    Ok(factor(n)?.into_iter().fold(n.clone(), |t, (p, _)| t / p.clone() * (p - N::from(1))))
}

/// n!, the product of the whole numbers from 1 to `n`.
pub fn factorial(n: u64) -> Result<BigUint, Error> {
    let mut product = BigUint::from(1u8);
    for i in 2..=n {
        if i % BETWEEN_CHECKS as u64 == 0 { check()?; }
        product *= i;
    }
    Ok(product)
}


/// The primes below 7. The sieve stores only numbers that none of them divide, so it gives these separately.
const WHEEL: [u64; 3] = [2, 3, 5];
/// The residues modulo 30 of the numbers that `WHEEL`'s primes don't divide. A segment has one byte for each 30 numbers. In byte `b`, bit `j` stands for the number `30b + RESIDUES[j]` past the segment's start.
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
#[doc(hidden)]
pub const BYTES: usize = 32768;
/// The bytes of the first segment. Segments double from it up to `BYTES`, so finding a few primes takes little sieving.
#[doc(hidden)]
pub const FIRST: usize = 128;
/// The bytes after which the pattern of `PRESIEVED`'s multiples repeats.
const PERIOD: u64 = product(&PRESIEVED);

/// The bytes that multiples of `PRESIEVED` leave, long enough to fill a segment from any phase.
fn pattern() -> &'static [u8] {
    static PATTERN: std::sync::OnceLock<Vec<u8>> = std::sync::OnceLock::new();
    PATTERN.get_or_init(|| {
        (0..BYTES as u64 + PERIOD).map(|b| (0..8).filter(|&j| PRESIEVED.iter().all(|p| (30 * b + RESIDUES[j]) % p != 0)).fold(0, |m, j| m | 1 << j)).collect()
    })
}

/// The primes from a given number upward, found a segment at a time by a sieve of Eratosthenes. A segment stores only the numbers that 2, 3 and 5 don't divide, and starts from a copy of the pattern that multiples of `PRESIEVED` leave.
///
/// Sieving numbers near n needs the primes below √n, which `Primes` finds first. A start far above 2^40 therefore costs memory in proportion to its square root. `next_prime` suits that range better.
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
        Self { lo, next: Some(lo), bytes: Vec::new(), at: 0, skip: RESIDUES.iter().filter(|&&r| lo + r < start).count(), sieving: Vec::new(), limit: 1,
            size: FIRST, wheel: WHEEL.iter().filter(|&&p| p < start).count() }
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
        if lo == 0 { bytes[0] &= !BIT[1]; for p in PRESIEVED { bytes[0] |= BIT[p as usize]; } }
        let top = u64::MAX - lo;
        if top < 30 * len as u64 {
            for b in (top / 30) as usize..len { for (j, r) in RESIDUES.into_iter().enumerate() { if 30 * b as u64 + r > top { bytes[b] &= !(1 << j); } } }
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

}

impl Iterator for Primes {
    type Item = Result<u64, Error>;
    /// The next prime, or `None` past `u64::MAX`.
    fn next(&mut self) -> Option<Result<u64, Error>> { self.nth(0) }
    /// The prime `k` places on: `nth(0)` is the next prime. Gives `None` past `u64::MAX`.
    fn nth(&mut self, k: usize) -> Option<Result<u64, Error>> {
        let mut k = k as u64;
        while self.wheel < WHEEL.len() {
            self.wheel += 1;
            if k == 0 { return Some(Ok(WHEEL[self.wheel - 1])); }
            k -= 1;
        }
        loop {
            k = self.skip(k);
            if k == 0 && self.at < self.bytes.len() * 8 {
                let p = self.lo + 30 * (self.at / 8) as u64 + RESIDUES[self.at % 8];
                self.at += 1;
                return Some(Ok(p));
            }
            self.next?;
            if let Err(e) = check() { return Some(Err(e)); }
            self.sieve();
        }
    }
}

/// How many primes are below `n`.
pub fn prime_count(n: u64) -> Result<u64, Error> {
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
