//! Random numbers from a caller's generator: uniform floats, bounded integers and sampling without replacement.
use crate::Arg;
use rand::Rng;

/// Appends `n` uniform integers below their bounds. Every bound must be positive.
pub fn roll<R: Rng + ?Sized>(bounds: Arg<'_, u64>, n: usize, rng: &mut R, out: &mut Vec<u64>) {
    match bounds { Arg::One(b) => rolled(std::iter::repeat_n(b, n), rng, out), Arg::Many(bs) => rolled(bs[..n].iter().copied(), rng, out) }
}

/// One uniform integer below the positive bound `b`.
pub fn below<R: Rng + ?Sized>(b: u64, rng: &mut R) -> u64 { kept(b, || rng.next_u64()) }

fn rolled<R: Rng + ?Sized>(bounds: impl ExactSizeIterator<Item = u64> + Clone, rng: &mut R, out: &mut Vec<u64>) {
    let (start, n) = (out.len(), bounds.len());
    // rand block fills copy each 4-byte word through memmove. Map the raw words in a separate pass.
    out.extend((0..n).map(|_| rng.next_u64()));
    // A low word of at least b is always kept. Only the remainder needs a rejection check.
    let k = out[start..].iter().zip(bounds.clone()).position(|(&w, b)| product(w, b).1 < b).unwrap_or(n);
    for (w, b) in out[start..start + k].iter_mut().zip(bounds.clone()) { *w = product(*w, b).0 }
    for (w, b) in out[start + k..].iter_mut().zip(bounds.skip(k)) {
        let mut first = Some(*w);
        *w = kept(b, || first.take().unwrap_or_else(|| rng.next_u64()));
    }
}

/// Lemire's unbiased bounded draw. Rejected words are replaced by fresh words.
fn kept(b: u64, mut next: impl FnMut() -> u64) -> u64 { loop { let (hi, lo) = product(next(), b); if lo >= b || lo >= b.wrapping_neg() % b { break hi; } } }

#[inline]
fn product(w: u64, b: u64) -> (u64, u64) { let p = u128::from(w) * u128::from(b); ((p >> 64) as u64, p as u64) }

/// A uniform float in the open interval (0, 1) from the word's top 52 bits.
#[inline]
pub fn unit(w: u64) -> f64 { f64::from_bits((w >> 12) | 1.0f64.to_bits()) - (1.0 - f64::EPSILON / 2.0) }

/// Appends `n` uniform floats in (0, 1). Conversion of the raw words vectorizes.
pub fn units<R: Rng + ?Sized>(n: usize, rng: &mut R, out: &mut Vec<f64>) {
    let start = out.len();
    out.extend((0..n).map(|_| f64::from_bits(rng.next_u64())));
    for x in &mut out[start..] { *x = unit(x.to_bits()); }
}

/// Appends `n` distinct integers below `total`, in random order. Every choice and order is equally likely. `n` must be at most `total`.
pub fn deal<R: Rng + ?Sized>(n: u64, total: u64, rng: &mut R, out: &mut Vec<u64>) {
    assert!(n <= total, "a deal can't take more items than its total");
    out.extend(rand::seq::index::sample(rng, total as usize, n as usize).into_iter().map(|i| i as u64));
}

