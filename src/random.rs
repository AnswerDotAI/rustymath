//! Random numbers from a caller's generator: floats between 0 and 1, rolls, each uniform below its own bound, and deals, distinct integers below a total.
use crate::Arg;
use rand::Rng;

/// Appends `n` integers to `out`, each uniform from 0 up to but not including its bound: the one bound, or each of the bounds in turn. Each bound must be at least 1. Each draw is the number that `rand`'s `Uniform` gives for its bound, by Lemire's method. A draw maps a word `w` to the high word of `w × b`, unless the low word is below 2⁶⁴ mod b. Then it discards `w` and maps the next word instead. Drawing every word before mapping any lets the mapping run as a separate pass. A discarded word, about once in 2⁶⁴ ÷ bound draws, ends the pass, and the draws from it on replay in turn, from the words already drawn and then from `rng`.
pub fn roll<R: Rng + ?Sized>(bounds: Arg<'_, u64>, n: usize, rng: &mut R, out: &mut Vec<u64>) {
    match bounds {
        Arg::One(b) => rolled(std::iter::repeat_n(b, n), rng, out),
        Arg::Many(bs) => rolled(bs[..n].iter().copied(), rng, out),
    }
}

/// One integer from 0 up to but not including `b`, as [`roll`] draws it.
pub fn below<R: Rng + ?Sized>(b: u64, rng: &mut R) -> u64 { kept(b, || rng.next_u64()) }

/// [`roll`], with one bound for each draw.
fn rolled<R: Rng + ?Sized>(bounds: impl ExactSizeIterator<Item = u64> + Clone, rng: &mut R, out: &mut Vec<u64>) {
    let (start, n) = (out.len(), bounds.len());
    // `fill` would draw the same words, but `rand`'s block generators copy each 4-byte word through a call to `memmove`, which costs several times as much.
    out.extend((0..n).map(|_| rng.next_u64()));
    // A low word of at least b is always kept, because 2⁶⁴ mod b is below b. The pass ends at the first other word, and only `kept` computes the remainder.
    let k = out[start..].iter().zip(bounds.clone()).position(|(&w, b)| product(w, b).1 < b).unwrap_or(n);
    for (w, b) in out[start..start + k].iter_mut().zip(bounds.clone()) { *w = product(*w, b).0 }
    let mut next = undrawn(out, start + k, rng);
    out.extend(bounds.skip(k).map(|b| kept(b, || next.next().unwrap())));
}

/// The draw below `b` from the first word that `next` gives and Lemire's method keeps.
fn kept(b: u64, mut next: impl FnMut() -> u64) -> u64 {
    loop { let (hi, lo) = product(next(), b); if lo >= b || lo >= b.wrapping_neg() % b { break hi } }
}

/// The high and low words of `w × b`. The high word is the draw.
#[inline]
fn product(w: u64, b: u64) -> (u64, u64) { let p = u128::from(w) * u128::from(b); ((p >> 64) as u64, p as u64) }

/// The words of `out` from `at` on, taken out of it, then the words that `rng` draws next.
fn undrawn<'a, R: Rng + ?Sized>(out: &mut Vec<u64>, at: usize, rng: &'a mut R) -> impl Iterator<Item = u64> + 'a {
    out.split_off(at).into_iter().chain(std::iter::from_fn(move || Some(rng.next_u64())))
}

/// The float in the open interval (0, 1) that `rand`'s `Open01` makes from the word `w`: its top 52 bits as the fraction of a number in [1, 2), less 1 − ε/2. Seeded draws depend on this staying bit for bit what `Open01` gives.
#[inline]
pub fn unit(w: u64) -> f64 { f64::from_bits((w >> 12) | 1.0f64.to_bits()) - (1.0 - f64::EPSILON / 2.0) }

/// Appends `n` floats to `out`, each `unit` of the generator's next `u64`, the numbers that drawing `Open01` `n` times gives. Drawing every word before converting any lets the conversion run as a separate, vectorized pass.
pub fn units<R: Rng + ?Sized>(n: usize, rng: &mut R, out: &mut Vec<f64>) {
    let start = out.len();
    out.extend((0..n).map(|_| f64::from_bits(rng.next_u64())));
    for x in &mut out[start..] { *x = unit(x.to_bits()); }
}

/// Appends `n` distinct integers below `total` to `out`, in random order: every choice of them, in every order, is equally likely. `n` must be at most `total`.
pub fn deal<R: Rng + ?Sized>(n: u64, total: u64, rng: &mut R, out: &mut Vec<u64>) {
    assert!(n <= total, "a deal can't take more items than its total");
    out.extend(rand::seq::index::sample(rng, total as usize, n as usize).into_iter().map(|i| i as u64));
}

/// SplitMix64 from its state, for rustymath's own seeded draws: differential evolution's, and the truncated SVD's test matrix.
pub(crate) struct SplitMix64(pub(crate) u64);

impl rand::TryRng for SplitMix64 {
    type Error = std::convert::Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> { self.try_next_u64().map(|z| (z >> 32) as u32) }
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Ok(z ^ (z >> 31))
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> { rand::rand_core::utils::fill_bytes_via_next_word(dst, || self.try_next_u64()) }
}
