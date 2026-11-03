//! Random sampling from counter streams. Each call reads one ChaCha8 stream selected by a key and stream number. Item `k` reads its first `w` words at position `k*w`, and rejection sampling uses that item's separate region. Floats, rolls and distribution samplers share this word addressing.
use crate::Arg;
use rand::{Rng, SeedableRng, rngs::ChaCha8Rng};
use std::convert::Infallible;

/// A ChaCha8 key. Every stream that one generator's draw calls read comes from it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Key(pub [u8; 32]);

impl Key {
    /// The key that `rand`'s `seed_from_u64` gives ChaCha8 for `seed`.
    pub fn seeded(seed: u64) -> Self { Self(ChaCha8Rng::seed_from_u64(seed).get_seed()) }
    /// The stream that draw call `number` reads.
    pub fn stream(self, number: u64) -> Stream { Stream { key: self, number } }
}

/// The stream that one draw call reads: a key and a stream number under it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Stream { pub key: Key, pub number: u64 }

impl Stream {
    /// The stream's generator from block `block` on. A block holds 8 words.
    pub fn at(self, block: u64) -> ChaCha8Rng {
        let mut rng = ChaCha8Rng::from_seed(self.key.0);
        rng.set_stream(self.number);
        if block != 0 { rng.set_block_pos(block); }
        rng
    }
}

/// A generator's state: its key and the count of its draw calls. Draw call c reads stream c under the key, from its start, so its numbers don't depend on how many numbers the calls before it took.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Counter { pub key: Key, pub count: u64 }

impl Counter {
    /// A generator keyed by `seed`, before its first draw call.
    pub fn seeded(seed: u64) -> Self { Self { key: Key::seeded(seed), count: 0 } }
    /// The stream that the next draw call reads, counting the call.
    pub fn next(&mut self) -> Stream {
        let stream = self.key.stream(self.count);
        self.count = self.count.wrapping_add(1);
        stream
    }
}

/// The block where items' regions start, and each region's length in blocks. Packed words fill blocks from 0. Above 2⁶³, 2²³ blocks of 8 words give each of up to 2⁴⁰ items 2²⁶ further words.
const REGIONS: u64 = 1 << 63;
const REGION_BLOCKS: u64 = 1 << 23;

/// The words of one draw call's items, item by item. Item k reads its first `w` words, the words of one attempt of the call's law, at word k × w of the call's stream. It reads any further words from its own region, from block 2⁶³ + k × 2²³ on.
pub struct Items { stream: Stream, packed: ChaCha8Rng, w: usize, next: u64 }

impl Items {
    /// The items of the call that reads `stream`, with `w` packed words each.
    pub fn new(stream: Stream, w: usize) -> Self { Self { stream, packed: stream.at(0), w, next: 0 } }
    /// The next item's words.
    pub fn item(&mut self) -> Item<'_> {
        let k = self.next;
        self.next += 1;
        Item { left: self.w, k, region: None, items: self }
    }
    /// The packed words of the next `n` items, in order, for a kernel whose items read only their packed words unless they reject. A rejection reads its item's region (`region`).
    fn words(&mut self, n: usize) -> impl Iterator<Item = u64> + '_ {
        let w = self.w;
        self.next += n as u64;
        // rand block fills copy each 4-byte word through memmove.
        (0..n * w).map(move |_| self.packed.next_u64())
    }
    /// Item `k`'s rejection region.
    fn region(&self, k: u64) -> ChaCha8Rng { self.stream.at(REGIONS + k * REGION_BLOCKS) }
}

/// One item's words: its packed ones, then its region's ([`Items`]). Dropping it skips the packed words it didn't read, so that the next item reads its own.
pub struct Item<'a> { items: &'a mut Items, left: usize, k: u64, region: Option<Box<ChaCha8Rng>> }

impl Drop for Item<'_> { fn drop(&mut self) { for _ in 0..self.left { self.items.packed.next_u64(); } } }

impl rand::TryRng for Item<'_> {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> { self.try_next_u64().map(|w| (w >> 32) as u32) }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        if self.left > 0 { self.left -= 1; return Ok(self.items.packed.next_u64()); }
        let (items, k) = (&*self.items, self.k);
        Ok(self.region.get_or_insert_with(|| Box::new(items.region(k))).next_u64())
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> { rand::rand_core::utils::fill_bytes_via_next_word(dst, || self.try_next_u64()) }
}

/// Appends `n` integers to `out`, each uniform from 0 up to but not including its bound: the one bound, or each of the bounds in turn. Each bound must be at least 1. Each draw reads the next item of `items`, which packs one word for each item, and is the number that `rand`'s `Uniform` gives for its bound from the item's words, by Lemire's method. A draw maps a word `w` to the high word of `w × b`, unless the low word is below 2⁶⁴ mod b. Then it discards `w` and maps the item's next word, from its region, instead. Reading every packed word before mapping any lets the mapping run as a separate pass.
pub fn roll(bounds: Arg<'_, u64>, n: usize, items: &mut Items, out: &mut Vec<u64>) {
    match bounds { Arg::One(b) => rolled(std::iter::repeat_n(b, n), items, out), Arg::Many(bs) => rolled(bs[..n].iter().copied(), items, out) }
}

/// One integer from 0 up to but not including `b`, as [`roll`] draws it, from the words of `rng`, which is an item's ([`Item`]) where the draw is one of a call's.
pub fn below<R: Rng + ?Sized>(b: u64, rng: &mut R) -> u64 { kept(b, || rng.next_u64()) }

/// [`roll`], with one bound for each draw.
fn rolled(bounds: impl ExactSizeIterator<Item = u64> + Clone, items: &mut Items, out: &mut Vec<u64>) {
    debug_assert_eq!(items.w, 1, "a roll packs one word for each item");
    let (start, n, first) = (out.len(), bounds.len(), items.next);
    out.extend(items.words(n));
    let mut i = 0;
    while i < n {
        // A low word of at least b is always kept, because 2⁶⁴ mod b is below b. Each pass ends at the next other word, and only `kept` computes the remainder.
        let k = i + out[start + i..].iter().zip(bounds.clone().skip(i)).position(|(&w, b)| product(w, b).1 < b).unwrap_or(n - i);
        for (w, b) in out[start + i..start + k].iter_mut().zip(bounds.clone().skip(i)) { *w = product(*w, b).0 }
        let Some(b) = bounds.clone().nth(k) else { break };
        let (mut word, mut region) = (Some(out[start + k]), items.region(first + k as u64));
        out[start + k] = kept(b, || word.take().unwrap_or_else(|| region.next_u64()));
        i = k + 1;
    }
}

/// The draw below `b` from the first word that `next` gives and Lemire's method keeps.
fn kept(b: u64, mut next: impl FnMut() -> u64) -> u64 { loop { let (hi, lo) = product(next(), b); if lo >= b || lo >= b.wrapping_neg() % b { break hi; } } }

/// The high and low words of `w × b`. The high word is the draw.
#[inline]
fn product(w: u64, b: u64) -> (u64, u64) { let p = u128::from(w) * u128::from(b); ((p >> 64) as u64, p as u64) }

/// The float in the open interval (0, 1) that `rand`'s `Open01` makes from the word `w`: its top 52 bits as the fraction of a number in [1, 2), less 1 − ε/2. Seeded draws depend on this staying bit for bit what `Open01` gives.
#[inline]
pub fn unit(w: u64) -> f64 { f64::from_bits((w >> 12) | 1.0f64.to_bits()) - (1.0 - f64::EPSILON / 2.0) }

/// Appends `n` floats to `out`, each `unit` of its item's word: the numbers that drawing `Open01` once from each of the next `n` items of `items` gives. `items` packs one word for each item. Reading every word before converting any lets the conversion run as a separate, vectorized pass.
pub fn units(n: usize, items: &mut Items, out: &mut Vec<f64>) {
    debug_assert_eq!(items.w, 1, "units pack one word for each item");
    let start = out.len();
    out.extend(items.words(n).map(f64::from_bits));
    for x in &mut out[start..] { *x = unit(x.to_bits()); }
}

/// Appends `n` distinct integers below `total` to `out`, in random order. Every choice and order is equally likely. `n` must be at most `total`.
pub fn deal<R: Rng + ?Sized>(n: u64, total: u64, rng: &mut R, out: &mut Vec<u64>) {
    assert!(n <= total, "a deal can't take more items than its total");
    out.extend(rand::seq::index::sample(rng, total as usize, n as usize).into_iter().map(|i| i as u64));
}
