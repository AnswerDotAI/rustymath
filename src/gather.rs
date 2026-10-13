//! Copying the items of a slice into a buffer by position: cycling them, repeating each, keeping those a mask selects, taking them at
//! indices, taking windows of them, and writing them at offsets. Each applies a conversion to each item it copies, so a caller can widen
//! items as it gathers them. Each is generic and `#[inline]`, so it compiles in the caller, with the caller's types and conversion.
use crate::search::Key;

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
            let (x, n) = (std::mem::MaybeUninit::new(f(x)), count(j));
            match d[k..].first_chunk_mut::<4>() {
                Some(w) if n <= 4 => *w = [x; 4],
                _ => d[k..k + n].fill(x),
            }
            k += n;
        }
        assert_eq!(k, total, "the counts must sum to `total`");
    })
}

/// Appends `item(j)` for each position `j` whose count in `mask` is 1, in order. Every count is 0 or 1, and `total` of them are 1. A
/// block of 64 counts that are all 0 is skipped, and one whose counts are all 1 is copied, so long runs cost little. Consecutive blocks
/// that mix both go through one branch-free pass that writes four places at a time, which on a random mask covers the whole array.
#[inline]
pub fn compress<M: Key, D: Clone>(out: &mut Vec<D>, mask: &[M], total: usize, item: impl Fn(usize) -> D) {
    out.reserve(total);
    let mixed = |out: &mut Vec<D>, from: usize, to: usize, count: usize| {
        let at = out.len();
        out.resize(at + count, item(from));
        written(&mut out[at..], &mask[from..to], |j| item(from + j));
    };
    let (mut from, mut count) = (0, 0);
    for (start, m) in (0..).step_by(64).zip(mask.chunks(64)) {
        // A block's count is at most 64, which fits in a byte. Byte additions vectorize in the widest lanes.
        let mut n = 0u8;
        for x in m { n += x.key() as u8 }
        let n = n as usize;
        if n != 0 && n != m.len() {
            count += n;
            continue;
        }
        if count > 0 { mixed(out, from, start, count) }
        if n > 0 { out.extend((start..start + n).map(&item)) }
        (from, count) = (start + m.len(), 0);
    }
    if count > 0 { mixed(out, from, mask.len(), count) }
}

/// Writes `item(j)` into `d`, in order, for each position `j` whose count in `mask` is 1, and gives how many it wrote. `d` holds at least
/// as many items as the counts add up to. Each item is written where it goes if its count is 1, and the next item overwrites it if its
/// count is 0, so the loop has no branch on the counts. It takes four counts at a time, so that one addition moves on past all four.
#[inline]
fn written<M: Key, D>(d: &mut [D], mask: &[M], item: impl Fn(usize) -> D) -> usize {
    let count = |m: M| m.key() as usize;
    let (mut j, mut k) = (0, 0);
    for m in mask.as_chunks::<4>().0 {
        let Some(w) = d.get_mut(k..k + 4) else { break };
        let (a, b, c) = (count(m[0]), count(m[1]), count(m[2]));
        w[0] = item(j);
        w[a & 3] = item(j + 1);
        w[(a + b) & 3] = item(j + 2);
        w[(a + b + c) & 3] = item(j + 3);
        k += a + b + c + count(m[3]);
        j += 4;
    }
    for (j, &n) in mask.iter().enumerate().skip(j) {
        if k < d.len() { d[k] = item(j) }
        k += count(n);
    }
    k
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
