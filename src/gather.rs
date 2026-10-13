//! Copying the items of a slice into a buffer by position: cycling them, repeating each, keeping those a mask selects, taking them at
//! indices, taking windows of them, and writing them at offsets. `positions` writes the positions that a mask selects. Each applies a
//! conversion to each item it writes, so a caller can widen items as it gathers them. Each is generic and `#[inline]`, so it compiles in
//! the caller, with the caller's types and conversion.
use crate::search::Key;
use std::mem::MaybeUninit;

/// Appends `len` items, cycling through the items of the nonempty `items`, each converted by `f`.
#[inline]
pub fn cycle<S: Copy, D: Clone>(out: &mut Vec<D>, items: &[S], len: usize, f: impl Fn(S) -> D) {
    if let [x] = *items { return out.resize(out.len() + len, f(x)) }
    for start in (0..len).step_by(items.len()) { out.extend(items[..items.len().min(len - start)].iter().map(|&x| f(x))) }
}

/// Appends each of `items`, converted by `f`, as many times as its count. `counts` holds one count for all the items, or one for each
/// item. The counts are nonnegative and sum to `total`. An item with a count of at most 4 writes four copies where its run starts.
/// The items after it overwrite the copies beyond its count. A larger count fills its run.
#[inline]
pub fn repeat<S: Copy, D: Copy>(out: &mut Vec<D>, items: &[S], counts: &[i64], total: usize, f: impl Fn(S) -> D) {
    let count = |j: usize| counts[if counts.len() == 1 { 0 } else { j }] as usize;
    crate::slice::appended(out, total, |d| {
        let mut k = 0;
        for (j, &x) in items.iter().enumerate() {
            let (x, n) = (MaybeUninit::new(f(x)), count(j));
            match d[k..].first_chunk_mut::<4>() {
                Some(w) if n <= 4 => *w = [x; 4],
                _ => d[k..k + n].fill(x),
            }
            k += n;
        }
        assert_eq!(k, total, "the counts must sum to `total`");
    })
}

/// Appends `item(j)` for each position `j` whose count in `mask` is 1, in order. Every count is 0 or 1, and `total` of them are 1. It
/// suits items read from a slice: where a block of counts holds both 0s and 1s, `written` reads every item in order.
#[inline]
pub fn compress<M: Key, D: Copy>(out: &mut Vec<D>, mask: &[M], total: usize, item: impl Fn(usize) -> D) { kept::<false, M, D>(out, mask, total, item) }

/// Appends `position(j)` for each position `j` whose count in `mask` is 1, in order. Every count is 0 or 1, and `total` of them are 1. It
/// suits items computed from `j`: where a block of counts holds both 0s and 1s, `selected` computes only the selected items.
#[inline]
pub fn positions<M: Key, D: Copy>(out: &mut Vec<D>, mask: &[M], total: usize, position: impl Fn(usize) -> D) {
    kept::<true, M, D>(out, mask, total, position)
}

/// `compress`, or `positions` with `TABLE`. Long runs cost little: a block of 64 counts that are all 0 is skipped, and one whose counts
/// are all 1 is copied. Each count is read by its low bit, which writes every slot whatever the counts hold.
#[inline(always)]
fn kept<const TABLE: bool, M: Key, D: Copy>(out: &mut Vec<D>, mask: &[M], total: usize, item: impl Fn(usize) -> D) {
    crate::slice::appended(out, total, |d| {
        let mut k = 0;
        for (start, m) in (0..).step_by(64).zip(mask.chunks(64)) {
            // A block's count is at most 64, which fits in a byte. Byte additions vectorize in the widest lanes.
            let mut n = 0u8;
            for x in m { n += x.key() as u8 & 1 }
            let n = usize::from(n);
            if n == m.len() {
                let run = &mut d[k..k + n];
                for j in 0..n { run[j].write(item(start + j)); }
            } else if TABLE {
                let (blocks, rest) = m.as_chunks::<8>();
                let mut at = k;
                for (b, block) in blocks.iter().enumerate() {
                    let start = start + 8 * b;
                    at = selected(d, at, mask_bits(block), |i| item(start + usize::from(i)));
                }
                let (start, last) = (start + 8 * blocks.len(), mask_bits(rest));
                selected(d, at, last, |i| item(start + usize::from(i)));
            } else if n > 0 {
                written(&mut d[k..], m, |j| item(start + j));
            }
            k += n;
        }
        assert_eq!(k, total, "`total` counts must be 1");
    })
}

/// Writes `item(j)` into `d`, in order, for each position `j` whose count in `mask` is 1. `d` holds at least as many slots as the counts
/// add up to. Each item is written where it goes if its count is 1, and the next item writes over it if its count is 0, so the loop has
/// no branch on the counts. It takes four counts at a time, so that one addition moves on past all four.
#[inline(always)]
fn written<M: Key, D: Copy>(d: &mut [MaybeUninit<D>], mask: &[M], item: impl Fn(usize) -> D) {
    let count = |m: M| (m.key() & 1) as usize;
    let (mut j, mut k) = (0, 0);
    for m in mask.as_chunks::<4>().0 {
        let Some(w) = d.get_mut(k..k + 4) else { break };
        let (a, b, c) = (count(m[0]), count(m[1]), count(m[2]));
        w[0].write(item(j));
        w[a & 3].write(item(j + 1));
        w[(a + b) & 3].write(item(j + 2));
        w[(a + b + c) & 3].write(item(j + 3));
        k += a + b + c + count(m[3]);
        j += 4;
    }
    for (j, &n) in mask.iter().enumerate().skip(j) {
        if k < d.len() { d[k].write(item(j)); }
        k += count(n);
    }
}

/// For each mask of 8 bits, the positions of its set bits in ascending order, then zeros.
const SET: [[u8; 8]; 256] = {
    let mut set = [[0; 8]; 256];
    let mut mask = 0;
    while mask < 256 {
        let (mut bit, mut k) = (0, 0);
        while bit < 8 {
            if mask >> bit & 1 == 1 {
                set[mask][k] = bit as u8;
                k += 1;
            }
            bit += 1;
        }
        mask += 1;
    }
    set
};

/// The low bits of the keys of up to 8 mask items as a byte, with the first item's in the lowest bit.
#[inline(always)]
pub(crate) fn mask_bits<M: Key>(m: &[M]) -> u8 {
    // An index loop, which compiles inline in the incremental build. A whole block's bytes take one load, which a mask on each byte
    // would split. The multiplication gathers the low bit of each byte into the top byte.
    let mut bytes = 0u64;
    for i in 0..m.len() { bytes |= u64::from(m[i].key() as u8) << (8 * i) }
    ((bytes & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8
}

/// Writes `item(i)` into `d` from place `k`, in order, for each set bit `i` of `mask`, and gives the place after the last one. Where 8
/// places remain, it writes all 8 items in a loop with no check or branch. Later calls write over the places after the selected items.
/// `D` is `Copy`, which leaves nothing to drop in the items written over.
#[inline(always)]
pub(crate) fn selected<D: Copy>(d: &mut [MaybeUninit<D>], k: usize, mask: u8, item: impl Fn(u8) -> D) -> usize {
    let set = &SET[usize::from(mask)];
    let found = mask.count_ones() as usize;
    if let Some(w) = d[k..].first_chunk_mut::<8>() {
        for i in 0..8 { w[i].write(item(set[i])); }
    } else {
        for i in 0..found { d[k + i].write(item(set[i])); }
    }
    k + found
}

/// Appends the items of `items` at `indices`, each converted by `f`. Each index is in `-n..n` for `n` items, and a negative index
/// counts back from the end.
#[inline]
pub fn take<S: Copy, D, I: Copy + Into<i64>>(out: &mut Vec<D>, items: &[S], indices: &[I], f: impl Fn(S) -> D) {
    let n = items.len() as i64;
    out.extend(indices.iter().map(move |&i| {
        let i: i64 = i.into();
        f(items[(i + (n & (i >> 63))) as usize])
    }));
}

/// Appends `m` windows of `W` items. Window `k` holds the items of `s` at `o + k × step` plus each offset in `cell`, each converted by
/// `f`. Each item is written once, into `out`'s spare capacity.
#[inline]
pub fn windows<const W: usize, S: Copy, D>(out: &mut Vec<D>, s: &[S], o: usize, step: isize, m: usize, cell: &[usize; W], f: impl Fn(S) -> D) {
    crate::slice::appended(out, m * W, |slots| {
        for (k, window) in slots.as_chunks_mut::<W>().0.iter_mut().enumerate() {
            let s = &s[o.wrapping_add_signed(k as isize * step)..];
            for i in 0..W { window[i].write(f(s[cell[i]])); }
        }
    })
}

/// Replaces the item of `out` at each offset with the matching one of `items`, converted by `f`, or with the only one when `items`
/// holds one. A later offset wins.
#[inline]
pub fn scatter<S: Copy, D>(out: &mut [D], offsets: &[usize], items: &[S], f: impl Fn(S) -> D) {
    if let [x] = *items { return offsets.iter().for_each(|&o| out[o] = f(x)) }
    for (&o, &x) in offsets.iter().zip(items) { out[o] = f(x) }
}
