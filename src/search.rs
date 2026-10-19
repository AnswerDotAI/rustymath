//! Search and sort over slices of keys. A key is an item read as a whole number (`Key`), or an unsigned word whose order is the order you want (`RadixKey`). `Float::order_key` gives each float such a word, as wide as the float. Positions take 32 bits inside these functions, so a slice holds at most `u32::MAX` items. `tolerant` builds its searches over floats on these functions.
//!
//! `index_of`, `classes` and `firsts` index the first position of each key. `index_of` passes over `x` once for each item of `y` when `y` has at most `SCANNED` items. The index is a table when the keys span at most twice the slice's length, or 1024, and a hash map otherwise. `numbered` numbers the classes in order of first position, with each one's size. `grouped_keys` numbers them the same way from a table of keys, with each one's positions, and `class_counts` gives only their sizes. `counts` counts the items with each key in such a table. `sort_rows` sorts keys that span such a table by counting them. It sorts other keys with an LSD radix sort that skips any byte that is the same in every key. With several threads, the sort splits each pass of more than `RADIX_PIECE` keys across them, and `index_of` splits its lookups.
use crate::{Float, buffer::{Buffer, DType, Element, Floats, View}, error::{Error, Kind}, tolerant::{self, Tolerance}};
use hashbrown::{HashMap, hash_map::Entry};
use std::{convert::Infallible, hash::Hash, mem::MaybeUninit, ops::Not};
use std::cmp::Ordering;

/// An item as a whole number: a Boolean's 0 or 1, an integer, a character's code point, or an order key. Two items match exactly when their keys are equal. A 64-bit unsigned key wraps to a negative number, which keeps keys distinct.
pub trait Key: Copy + Send + Sync + Eq + Ord + Hash {
    fn key(self) -> i64;
    /// The item whose key is `key`, if there is one.
    fn from_key(key: i64) -> Option<Self>;
    /// The least and the greatest key of `x`, which isn't empty, in one pass that vectorizes for narrow items.
    fn range(x: &[Self]) -> (i64, i64);
}
/// The least and the greatest key of `x`, through `reduce::extremes`, for item types that `Min` and `Max` reduce.
#[inline]
fn extreme_keys<K: Key>(x: &[K]) -> (i64, i64) where crate::reduce::Min: crate::reduce::Reduction<K, Total = K>,
    crate::reduce::Max: crate::reduce::Reduction<K, Total = K> { let (lo, hi) = crate::reduce::extremes(x); (lo.key(), hi.key()) }
macro_rules! keys {
    ($($t:ty),+) => {$(
        impl Key for $t {
            #[inline]
            fn key(self) -> i64 { self as i64 }
            #[inline]
            fn from_key(key: i64) -> Option<Self> { ((key as $t).key() == key).then_some(key as $t) }
            #[inline]
            fn range(x: &[Self]) -> (i64, i64) { extreme_keys(x) }
        }
    )+};
}
keys!(u8, u16, u32, u64, i16, i32, i64);
impl Key for bool {
    #[inline]
    fn key(self) -> i64 { self.into() }
    #[inline]
    fn from_key(key: i64) -> Option<Self> { matches!(key, 0 | 1).then_some(key == 1) }
    #[inline]
    fn range(x: &[Self]) -> (i64, i64) { extreme_keys(x) }
}
impl Key for char {
    #[inline]
    fn key(self) -> i64 { i64::from(u32::from(self)) }
    #[inline]
    fn from_key(key: i64) -> Option<Self> { char::from_u32(u32::try_from(key).ok()?) }
    #[inline]
    fn range(x: &[Self]) -> (i64, i64) { let (lo, hi) = crate::reduce::extremes_of(x, |c: char| c as u32); (lo.into(), hi.into()) }
}

/// An unsigned key that `sort_rows` sorts one byte at a time.
pub trait RadixKey: Copy + Send + Sync + Ord + Default + Not<Output = Self> {
    const BYTES: usize;
    fn byte(self, b: usize) -> usize;
    /// The key as a number.
    fn word(self) -> u64;
    /// The least and the greatest key of `x`, through [`crate::reduce::extremes`].
    fn extremes(x: &[Self]) -> (Self, Self);
}
macro_rules! radix_keys {
    ($($t:ty),+) => {$(
        impl RadixKey for $t {
            const BYTES: usize = size_of::<$t>();
            #[inline]
            fn byte(self, b: usize) -> usize { (self >> (8 * b)) as usize & 255 }
            #[inline]
            fn word(self) -> u64 { self as u64 }
            #[inline]
            fn extremes(x: &[Self]) -> (Self, Self) { crate::reduce::extremes(x) }
        }
    )+};
}
radix_keys!(u8, u16, u32, u64);

/// An unsigned key in an integer's numeric order. An unsigned integer is its own key. A signed integer's key is its bits with the sign bit flipped.
pub trait OrderKey: Copy {
    type Key: Copy + Into<u64>;
    fn order_key(self) -> Self::Key;
}
macro_rules! order_keys {
    ($($s:ty => $u:ty),+; $($own:ty),+) => {
        $(impl OrderKey for $s {
            type Key = $u;
            #[inline]
            fn order_key(self) -> $u { self as $u ^ 1 << (<$u>::BITS - 1) }
        })+
        $(impl OrderKey for $own {
            type Key = $own;
            #[inline]
            fn order_key(self) -> $own { self }
        })+
    };
}
order_keys!(i16 => u16, i32 => u32, i64 => u64; u8);
impl OrderKey for bool {
    type Key = u8;
    #[inline]
    fn order_key(self) -> u8 { self as u8 }
}
impl OrderKey for char {
    type Key = u32;
    #[inline]
    fn order_key(self) -> u32 { self.into() }
}

/// The number of items of `y` up to which `index_of` and `tolerant::index_of` pass over `x` once for each, in place of building an index.
pub const SCANNED: usize = 16;

/// For each item of `y`, the first position of its key in `x`, or the length of `x` when it has none. A slice here holds at most `u32::MAX` items, so positions are `u32`s.
#[inline(never)]
pub fn index_of<K: Key>(x: &[K], y: &[K]) -> Vec<u32> {
    let miss = x.len() as u32;
    if y.len() <= SCANNED { return y.iter().map(|n| x.iter().position(|h| h == n).map_or(miss, |p| p as u32)).collect(); }
    let Some((min, mut table)) = table(x, usize::MAX) else { return first_matches(x, y) };
    let first = write_first(x, min, &mut table, None);
    // `move` keeps `min`, `first` and `miss` in registers. Captured by reference, they would load again for every item.
    let at = move |n: K| usize::try_from(n.key().wrapping_sub(min)).ok().and_then(|i| first.get(i));
    let mut found = Vec::with_capacity(y.len());
    look_up(&mut found, y, move |n| at(n).map_or(miss, |&f| f.min(miss as usize) as u32));
    found
}

/// Appends `look(n)` to `found` for each needle `n` of `y`, in order. The lookups split across threads in units of `BETWEEN_CHECKS` needles, as `parallel::split` splits a call, so `look` must read only what no lookup changes.
pub(crate) fn look_up<K: Copy + Sync>(found: &mut Vec<u32>, y: &[K], look: impl Fn(K) -> u32 + Sync) {
    crate::parallel::extended(found, y.len(), crate::BETWEEN_CHECKS, &|r, slots| {
        slots.fill(r.len(), |slots| {
            let needles = &y[r];
            let slots = &mut slots[..needles.len()];
            for i in 0..needles.len() { slots[i].write(look(needles[i])); }
            true
        })
    });
}

/// The first position of each item's key in `x`, which represents the item's class.
#[inline(never)]
pub fn classes<K: Key>(x: &[K]) -> Vec<u32> {
    let Some((min, mut table)) = table(x, usize::MAX) else { return first_classes(x, |n| n) };
    let first = write_first(x, min, &mut table, None);
    x.iter().map(move |&n| first[n.key().wrapping_sub(min) as usize] as u32).collect()
}

/// Whether each item of `x` is the first with its key. Once a table holds the first position of each key, marking those positions gives the result, with no second pass over `x`. Wider keys mark the items that are their own class.
#[inline(never)]
pub fn firsts<K: Key>(x: &[K]) -> Vec<bool> {
    let Some((min, mut table)) = table(x, usize::MAX) else { return first_mask(&first_classes(x, |n| n)) };
    let mut mask = vec![false; x.len()];
    for &f in write_first(x, min, &mut table, None) { if let Some(m) = mask.get_mut(f) { *m = true } }
    mask
}

/// The number of items of `x` equal to `v`. Each of 32 lanes counts its matches in a byte, emptied into the total every 255 blocks, so the comparisons and the counts stay in narrow lanes. A count in `usize` widens each item to 64 bits.
#[inline]
fn count_eq<K: Key>(x: &[K], v: K) -> usize {
    const LANES: usize = 32;
    let (blocks, rest) = x.as_chunks::<LANES>();
    let mut total = rest.iter().filter(|&&k| k == v).count();
    for group in blocks.chunks(255) {
        let mut lanes = [0u8; LANES];
        for block in group { for i in 0..LANES { lanes[i] += u8::from(block[i] == v) } }
        total += lanes.iter().map(|&n| usize::from(n)).sum::<usize>();
    }
    total
}

/// The least key of `x`, and the number of items with each key from it to the greatest, when `tabled` allows that range. Up to 4 keys, each count but the first takes a pass that vectorizes. With fewer than 64 keys, runs of one key are common, and each count would wait for the previous one to be stored. So the items then go to four tables in turn, which are summed at the end. On the Mac, four tables made 2 keys 3 times as fast as one table, and 64 keys or more no faster.
#[inline(never)]
pub fn counts<K: Key>(x: &[K]) -> Option<(i64, Vec<usize>)> {
    let (min, mut counts) = table(x, 0)?;
    let n = counts.len();
    let at = move |k: &K| k.key().wrapping_sub(min) as usize;
    if n <= 4 {
        for (g, count) in counts.iter_mut().enumerate().skip(1) { *count = K::from_key(min + g as i64).map_or(0, |v| count_eq(x, v)); }
        counts[0] = x.len() - counts[1..].iter().sum::<usize>();
        return Some((min, counts));
    }
    if n >= 64 { for k in x { counts[at(k)] += 1 } return Some((min, counts)); }
    let mut parts = vec![0; 4 * n];
    let (chunks, rest) = x.as_chunks::<4>();
    for chunk in chunks { for (p, k) in chunk.iter().enumerate() { parts[p * n + at(k)] += 1 } }
    for k in rest { parts[at(k)] += 1 }
    for (g, count) in counts.iter_mut().enumerate() { *count = (0..4).map(|p| parts[p * n + g]).sum() }
    Some((min, counts))
}

/// The classes that `classes` gives, numbered from 0 in order of their first positions.
#[derive(Clone, Debug, PartialEq)]
pub struct Numbered {
    /// The first position of each class, in ascending order.
    pub firsts: Vec<usize>,
    /// Each item's class number.
    pub numbers: Vec<usize>,
    /// The number of items in each class.
    pub counts: Vec<usize>,
}

/// Numbers the classes in `classes`, which gives each item the first position of its class, as `classes` and `tolerant::classes` do.
pub fn numbered(classes: &[u32]) -> Numbered {
    let mut number = vec![0; classes.len()];
    let (mut firsts, mut numbers, mut counts) = (Vec::new(), Vec::with_capacity(classes.len()), Vec::new());
    for (i, c) in classes.iter().map(|&c| c as usize).enumerate() {
        if c == i {
            number[i] = firsts.len();
            firsts.push(i);
            counts.push(0);
        }
        let k = number[c];
        numbers.push(k);
        counts[k] += 1;
    }
    Numbered { firsts, numbers, counts }
}

/// Whether `offsets` are dense enough to use a table over the `size` positions they fall below.
pub fn dense_offsets(offsets: &[usize], size: usize) -> bool { size <= 8 * offsets.len() }

/// For each offset, how many preceding offsets equal it, and the largest occurrence count. `None` when none repeats. Every offset must be below `size`.
pub fn occurrences(offsets: &[usize], size: usize) -> Option<(Vec<u32>, usize)> {
    let next = |seen: &mut u32| { *seen += 1; *seen - 1 };
    let counts: Vec<u32> = if dense_offsets(offsets, size) {
        if offsets.len() <= size {
            let (mut taken, mut repeated) = (vec![0u64; size.div_ceil(64)], 0);
            for &o in offsets {
                let (word, bit) = (&mut taken[o / 64], 1 << (o % 64));
                repeated |= *word & bit;
                *word |= bit;
            }
            if repeated == 0 { return None; }
        }
        let mut seen = vec![0u32; size];
        offsets.iter().map(|&o| next(&mut seen[o])).collect()
    } else { let mut seen = HashMap::new(); offsets.iter().map(|&o| next(seen.entry(o).or_insert(0))).collect() };
    let rounds = counts.iter().fold(0, |most, &c| most.max(c + 1));
    (rounds > 1).then_some((counts, rounds as usize))
}

/// Inverts a permutation: each input position's rank in `order`. Every position must occur exactly once.
pub fn inverse_permutation(order: impl ExactSizeIterator<Item = usize>) -> Vec<u32> {
    let mut ranks = vec![0; order.len()];
    for (rank, position) in order.enumerate() { ranks[position] = rank as u32; }
    ranks
}

/// Marks where `pattern_shape` fits within `shape` and every pattern item matches. Shapes describe compact row-major buffers; `same` compares their flat positions. A lower-rank pattern has leading axes of length one.
#[inline]
pub fn find_with<E>(pattern_shape: &[usize], shape: &[usize], mut same: impl FnMut(usize, usize) -> Result<bool, E>) -> Result<Vec<bool>, E> {
    let mut out = vec![false; shape.iter().product()];
    if pattern_shape.len() > shape.len() { return Ok(out); }
    let mut pattern = vec![1; shape.len() - pattern_shape.len()];
    pattern.extend_from_slice(pattern_shape);
    let strides = crate::layout::strides(shape);
    let offsets: Vec<usize> = (0..pattern.iter().product()).map(|i| crate::layout::coordinates(i, &pattern).map(|(a, c)| c * strides[a]).sum()).collect();
    for (flat, matched) in out.iter_mut().enumerate() {
        if crate::layout::coordinates(flat, shape).any(|(a, c)| pattern[a] > shape[a] - c) { continue; }
        *matched = true;
        for (i, &offset) in offsets.iter().enumerate() { if !same(i, flat + offset)? { *matched = false; break; } }
    }
    Ok(out)
}

/// Marks occurrences of a compact pattern in a compact target, with the shapes and geometry of [`find_with`]. Booleans, integers and characters of one kind match exactly, and floats of any widths match within `t`, including NaN with NaN. `None` for other dtype pairs.
pub fn view_find(pattern: View<'_>, pattern_shape: &[usize], target: View<'_>, shape: &[usize], t: Tolerance) -> Option<Vec<bool>> {
    #[inline]
    fn find(pattern_shape: &[usize], shape: &[usize], mut same: impl FnMut(usize, usize) -> bool) -> Vec<bool> {
        let Ok(found) = find_with(pattern_shape, shape, |i, j| Ok::<_, Infallible>(same(i, j)));
        found
    }
    use tolerant::Tolerant;
    Some(match (pattern, target) {
        (View::Booleans(x), View::Booleans(y)) => find(pattern_shape, shape, |i, j| x[i] == y[j]),
        (View::Integers(x), View::Integers(y)) => find(pattern_shape, shape, |i, j| x.get(i) == y.get(j)),
        (View::Characters(x), View::Characters(y)) => find(pattern_shape, shape, |i, j| x[i] == y[j]),
        (View::Floats(x), View::Floats(y)) => find(pattern_shape, shape, |i, j| t.matches(x.get(i), y.get(j))),
        _ => return None,
    })
}

/// The number of items in each class of `x`, in order of the classes' first positions, as `numbered` counts them. Keys that span a table take `counts`, then a pass for first positions that stops once it has found every key `x` holds, with nothing written for each item.
#[inline(never)]
pub fn class_counts<K: Key>(x: &[K]) -> Vec<usize> {
    let Some((min, counts)) = counts(x) else { return numbered(&classes(x)).counts };
    let mut first = vec![usize::MAX; counts.len()];
    write_first(x, min, &mut first, Some(counts.iter().filter(|&&c| c > 0).count()));
    let mut keys: Vec<usize> = (0..counts.len()).filter(|&g| counts[g] > 0).collect();
    keys.sort_unstable_by_key(|&g| first[g]);
    keys.into_iter().map(|g| counts[g]).collect()
}

/// For each of `values`, the number of `boundaries` at or below it, as BQN's dyadic `⍋` and NumPy's `searchsorted` with `side='right'` give it. `boundaries` must be in ascending order. When they are, and they span few enough keys for a table, a table of the count of boundaries at or below each key answers each value with one read. Otherwise each value takes a binary search, which halves its range with a select in place of a branch, so mispredicted comparisons don't stall it. Sixteen values search together, level by level, because every search of `boundaries` halves the same lengths. Their loads at each level are then independent.
#[inline(never)]
pub fn bins<K: Key>(boundaries: &[K], values: &[K]) -> Vec<u32> {
    const LANES: usize = 16;
    let n = boundaries.len();
    let (Some(&first), Some(&last)) = (boundaries.first(), boundaries.last()) else { return vec![0; values.len()] };
    let at = move |k: K| k.key().wrapping_sub(first.key()) as u64 as usize;
    if let Some(range) = table_size(last.key().wrapping_sub(first.key()) as u64, n).filter(|_| boundaries.is_sorted()) {
        let mut below = vec![0u32; range];
        // In ascending order, the count at or below a key is one more than the position of its last boundary. Storing positions, with no read, keeps a run of one key from making each store wait for the one before. Keys between boundaries take the count below them.
        for (i, &b) in boundaries.iter().enumerate() { below[at(b)] = i as u32 + 1 }
        let mut total = 0;
        for b in &mut below { total = total.max(*b); *b = total }
        return values.iter().map(move |&v| {
            if v < first { 0 } else if v > last { n as u32 } else { below[at(v)] }
        }).collect();
    }
    let search = move |v: &[K], out: &mut [u32]| {
        let mut base = [0; LANES];
        let mut len = n;
        while len > 1 {
            let half = len / 2;
            // On x86-64, LLVM can turn a select that feeds a load in a loop into a branch. Random values mispredict that branch at half the levels. `select_unpredictable` keeps a conditional move.
            for (b, &x) in base.iter_mut().zip(v) { *b = core::hint::select_unpredictable(boundaries[*b + half] <= x, *b + half, *b) }
            len -= half;
        }
        for ((o, &b), &x) in out.iter_mut().zip(&base).zip(v) { *o = (b + usize::from(boundaries[b] <= x)) as u32 }
    };
    let mut out = vec![0; values.len()];
    let mut start = 0;
    while start < values.len() {
        let end = values.len().min(start + LANES);
        search(&values[start..end], &mut out[start..end]);
        start = end;
    }
    out
}

/// The least key of `x`, and a table with an entry for each key from it to the greatest, each `empty`, when `table_size` allows that range. `Key::range` finds the least and the greatest key in one pass.
fn table<K: Key>(x: &[K], empty: usize) -> Option<(i64, Vec<usize>)> {
    if x.is_empty() { return None; }
    let (min, max) = K::range(x);
    table_size(max.abs_diff(min), x.len()).map(|range| (min, vec![empty; range]))
}

/// The number of entries in a table indexed by `len` keys that span `span + 1` values, when that's at most twice `len`, or 1024.
fn table_size(span: u64, len: usize) -> Option<usize> { usize::try_from(span).ok()?.checked_add(1).filter(|&r| r <= len.saturating_mul(2).max(1024)) }

/// Writes the first position of each key in `x` to its entry in `first`, whose entries start at key `min` and hold `usize::MAX` until written. The pass runs forward and stops after the first block of 64 positions that fills the entry of the last key `x` holds. Keys that fill their range, such as Booleans, soon do. Keys that leave gaps in their range, as the characters of text do, never fill it. A caller that knows how many keys `x` holds passes that number as `present`. Otherwise, where the keys span at most 128 values, the pass counts them once after `STALL` blocks in a row with no new key. It stops when it has found them all. Each entry takes its new position by a select, not a branch. The table is a slice, not a `Vec`, so that the compiler keeps its bounds in registers.
fn write_first<'t, K: Key>(x: &[K], min: i64, first: &'t mut [usize], present: Option<usize>) -> &'t [usize] {
    let (mut empty, mut stall, mut counted) = match present { Some(h) => (h, 0, true), None => (first.len(), 0, first.len() > 128) };
    for (b, block) in x.chunks(64).enumerate() {
        let before = empty;
        for (j, &n) in block.iter().enumerate() {
            let entry = &mut first[n.key().wrapping_sub(min) as usize];
            let new = *entry == usize::MAX;
            empty -= usize::from(new);
            *entry = if new { b * 64 + j } else { *entry };
        }
        stall = if empty == before { stall + 1 } else { 0 };
        // Entries of keys that `x` doesn't hold stay empty.
        if !counted && stall == STALL {
            counted = true;
            empty -= first.len() - held(x, min);
        }
        if empty == 0 { break; }
    }
    first
}

/// The number of blocks of 64 positions in a row with no new key after which `write_first` counts the keys of its items. A run that long is rare while keys that fill their range are still arriving.
const STALL: usize = 4;

/// The number of distinct keys in `x`, which span at most 128 values from `min`, from four 32-bit masks of the keys seen.
fn held<K: Key>(x: &[K], min: i64) -> usize {
    let mut seen = [0u32; 4];
    for k in x {
        let d = k.key().wrapping_sub(min) as u32;
        let bit = 1 << (d & 31);
        for (q, mask) in seen.iter_mut().enumerate() { *mask |= if d >> 5 == q as u32 { bit } else { 0 } }
    }
    seen.iter().map(|mask| mask.count_ones() as usize).sum()
}

/// For each item of `y`, the position of the first item of `x` with its key, or the length of `x` where none has it, from a hash map.
#[inline(never)]
fn first_matches<K: Key>(x: &[K], y: &[K]) -> Vec<u32> {
    let (miss, mut first) = (x.len() as u32, HashMap::with_capacity(x.len()));
    for (i, &n) in x.iter().enumerate().rev() { first.insert(n, i as u32); }
    let (first, mut found) = (&first, Vec::with_capacity(y.len()));
    look_up(&mut found, y, move |n| first.get(&n).map_or(miss, |&f| f));
    found
}

/// The first position of each item's `key` in `x`, from one pass that records and reads each key together. Two items' keys are equal exactly when the items match.
pub(crate) fn first_classes<T: Copy, K: Eq + Hash>(x: &[T], key: impl Fn(T) -> K) -> Vec<u32> {
    let mut first = HashMap::with_capacity(x.len());
    x.iter().enumerate().map(|(i, &v)| *first.entry(key(v)).or_insert(i as u32)).collect()
}

/// Whether each position of `classes` is its own class, which makes its item the first with its key.
pub(crate) fn first_mask(classes: &[u32]) -> Vec<bool> { classes.iter().enumerate().map(|(i, &c)| c as usize == i).collect() }

/// The positions of `groups` in order of group, keeping their order within each group, with the number in each of the `n` groups.
#[inline(never)]
pub fn grouped(groups: &[usize], n: usize) -> Groups { grouped_by(groups, n, |g| g) }

/// Positions group by group, ascending within each group, with the number in each group. Each buffer holds whole groups. The two classes of a slice of two keys take a buffer each, which `into_buffers` hands over without a copy. Other groups share one buffer. Positions are `u32`s.
#[derive(Clone, Debug, PartialEq)]
pub struct Groups {
    /// The number of positions in each group.
    pub sizes: Vec<usize>,
    buffers: Vec<Vec<u32>>,
}

impl Groups {
    /// The positions of each group, one slice for each group in turn. An empty group gives an empty slice.
    pub fn each(&self) -> impl Iterator<Item = &[u32]> {
        let (mut buffers, mut rest) = (self.buffers.iter(), &[][..]);
        self.sizes.iter().map(move |&n| {
            while rest.len() < n { rest = buffers.next().expect("a buffer holds each group") }
            let (these, more) = rest.split_at(n);
            rest = more;
            these
        })
    }

    /// The positions of group `k`, which are none for a group past the last.
    pub fn group(&self, k: usize) -> &[u32] { self.each().nth(k).unwrap_or_default() }

    /// The positions of each group in a buffer of its own. Groups that already have one hand it over. Groups that share a buffer copy their positions from it.
    pub fn into_buffers(self) -> Vec<Vec<u32>> { if self.buffers.len() == self.sizes.len() { return self.buffers; } self.each().map(<[u32]>::to_vec).collect() }

    /// Every position, group by group, in one buffer. Groups that share a buffer hand it over.
    pub fn into_positions(mut self) -> Vec<u32> { if self.buffers.len() == 1 { return self.buffers.pop().expect("one buffer"); } self.buffers.concat() }
}

/// The classes of a slice, numbered from 0 in order of their first positions, with the positions of each.
#[derive(Clone, Debug, PartialEq)]
pub struct Grouped {
    /// The first position of each class, in ascending order.
    pub firsts: Vec<usize>,
    /// The positions of each class.
    pub groups: Groups,
}

impl Grouped {
    /// The classes that `groups` numbers, giving each item's class, whose first positions are `firsts`.
    pub fn new(firsts: Vec<usize>, groups: &[usize]) -> Grouped { grouped_classes(groups, firsts, |g| g) }

    /// Folds each class of `data` in position order by `R`, and whether every fold fits.
    #[inline]
    pub fn fold<T: Copy, R: crate::reduce::Reduction<T>>(&self, data: &[T]) -> (Vec<R::Total>, bool) {
        let mut fits = true;
        let totals = self.groups.each().map(|rows| {
            let (mut total, mut ok) = R::one(data[rows[0] as usize]);
            for &r in &rows[1..] {
                let (t, k) = R::checked(total, data[r as usize]);
                (total, ok) = (t, ok && k);
            }
            fits &= ok;
            total
        }).collect();
        (totals, fits)
    }
}

/// The classes of `x` as [`Grouped`] gives them, or `None` when its keys span too many values for a table. The table entry of each key takes its first position, then its class number, so no pass gives each item its first position.
#[inline(never)]
pub fn grouped_keys<K: Key>(x: &[K]) -> Option<Grouped> {
    let (min, mut table) = table(x, usize::MAX)?;
    write_first(x, min, &mut table, None);
    let mut firsts: Vec<usize> = table.iter().copied().filter(|&f| f != usize::MAX).collect();
    firsts.sort_unstable();
    for (g, &f) in firsts.iter().enumerate() { table[x[f].key().wrapping_sub(min) as usize] = g; }
    let table = &table[..];
    // With two keys, each item's class is its offset from the lower key, flipped when the higher key comes first.
    Some(if let [lower, _] = *table { grouped_classes(x, firsts, move |k| k.key().wrapping_sub(min) as usize ^ lower) }
    else { grouped_classes(x, firsts, move |k| table[k.key().wrapping_sub(min) as usize]) })
}

/// The positions of `items` grouped by the group that `group` gives each item, keeping their order within each group, with the number in each of the `n` groups. `grouped` is the version for groups given as numbers.
#[inline(always)]
pub fn grouped_by<T: Copy + Sync>(items: &[T], n: usize, group: impl Fn(T) -> usize + Sync) -> Groups {
    let len = checked_len(items);
    if n <= 1 { return Groups { sizes: vec![len; n], buffers: vec![(0..len as u32).collect()] }; }
    if n == 2 { return two_groups(items, group, false); }
    // Up to 256 groups, a run of one group would make each count wait for the previous one to be stored. So the positions then split into four parts, each with its own counts and next place in each group. Above 256 groups such runs are rare, and one part measured faster.
    let (positions, sizes) = if n <= 256 && n * 16 <= len { parted::<4, T>(items, n, group) } else { parted::<1, T>(items, n, group) };
    Groups { sizes, buffers: vec![positions] }
}

/// The length of `items`, which grouped positions hold as `u32`s. Panics for a longer slice, rather than truncate its positions.
fn checked_len<T>(items: &[T]) -> usize {
    assert!(u32::try_from(items.len()).is_ok(), "grouped positions hold at most u32::MAX items");
    items.len()
}

/// [`Grouped`] for the classes of `items`, each in the class that `class` gives its item, whose first positions are `firsts`. Two classes take a buffer each. More share one, as `grouped_by` writes it.
#[inline(always)]
fn grouped_classes<T: Copy + Sync>(items: &[T], firsts: Vec<usize>, class: impl Fn(T) -> usize + Sync) -> Grouped {
    checked_len(items);
    let groups = if firsts.len() == 2 { two_groups(items, class, true) } else { grouped_by(items, firsts.len(), class) };
    Grouped { firsts, groups }
}

/// Each item's group, 0 or 1, as a byte, written in a pass that vectorizes, and the number of items in group 1.
fn group_bytes<T: Copy + Sync>(items: &[T], group: impl Fn(T) -> usize + Sync) -> (Vec<u8>, usize) {
    let mut groups = Vec::with_capacity(items.len());
    crate::pairs(crate::Arg::Many(items), crate::Arg::One(()), items.len(), &mut groups, |k, ()| (group(k) as u8, true));
    let ones = count_eq(&groups, 1);
    (groups, ones)
}

/// `grouped_by` for two groups: in one buffer with group 0's positions first, or in a buffer each where `apart`. `split2` writes both groups' positions in one pass.
fn two_groups<T: Copy + Sync>(items: &[T], group: impl Fn(T) -> usize + Sync, apart: bool) -> Groups {
    let (groups, ones) = group_bytes(items, group);
    let zeros = groups.len() - ones;
    let buffers = if apart {
        let (mut d0, mut d1) = (Vec::new(), Vec::new());
        crate::slice::appended(&mut d0, zeros, |a| crate::slice::appended(&mut d1, ones, |b| split2(&groups, a, b)));
        vec![d0, d1]
    } else {
        let mut d = Vec::new();
        crate::slice::appended(&mut d, groups.len(), |d| { let (a, b) = d.split_at_mut(zeros); split2(&groups, a, b) });
        vec![d]
    };
    Groups { sizes: vec![zeros, ones], buffers }
}

/// Writes the positions of the items whose byte in `groups` is 0 into `d0`, and of those whose byte is 1 into `d1`, each in order. Each block of 8 bytes becomes a mask of the items in group 1. `gather::selected` writes group 1's positions from the mask and group 0's from its complement. No write depends on a group. Each buffer holds exactly its group's positions.
fn split2(groups: &[u8], d0: &mut [MaybeUninit<u32>], d1: &mut [MaybeUninit<u32>]) {
    let (blocks, rest) = groups.as_chunks::<8>();
    let (mut k0, mut k1) = (0, 0);
    for (b, block) in blocks.iter().enumerate() {
        let (ones, start) = (crate::gather::mask_bits(block), (8 * b) as u32);
        k1 = crate::gather::selected(d1, k1, ones, |i| start + u32::from(i));
        k0 = crate::gather::selected(d0, k0, !ones, |i| start + u32::from(i));
    }
    let (start, ones) = ((8 * blocks.len()) as u32, crate::gather::mask_bits(rest));
    k1 = crate::gather::selected(d1, k1, ones, |i| start + u32::from(i));
    k0 = crate::gather::selected(d0, k0, !ones & ((1u16 << rest.len()) - 1) as u8, |i| start + u32::from(i));
    assert_eq!((k0, k1), (d0.len(), d1.len()), "each group must be 0 or 1");
}

/// `grouped_by` for more than two groups, with the positions split into `P` parts. The number of parts is a constant, so each loop over the parts unrolls, and indexing a group's places by part needs no check.
fn parted<const P: usize, T: Copy>(items: &[T], n: usize, group: impl Fn(T) -> usize) -> (Vec<u32>, Vec<usize>) {
    let (len, part) = (items.len(), items.len() / P);
    let parts: [&[T]; P] = std::array::from_fn(|q| &items[q * part..][..part]);
    let rest = &items[P * part..];
    let (mut positions, mut sizes) = (vec![0u32; len], vec![0; n]);
    let mut next = vec![[0u32; P]; n];
    // Slices in locals, with each place read once, keep the loops from reloading them from memory after every write.
    let (slots, next) = (&mut positions[..], &mut next[..]);
    for j in 0..part { for q in 0..P { next[group(parts[q][j])][q] += 1; } }
    for &item in rest { next[group(item)][P - 1] += 1; }
    // Each group's places start after the earlier groups', with part 0 first, as its positions come first.
    let mut start = 0;
    for (places, size) in next.iter_mut().zip(&mut sizes) {
        for place in places.iter_mut() {
            *size += *place as usize;
            (*place, start) = (start, start + *place);
        }
    }
    let mut place = move |i: usize, item: T, q: usize| {
        let p = &mut next[group(item)][q];
        let at = *p;
        slots[at as usize] = i as u32;
        *p = at + 1;
    };
    for j in 0..part { for q in 0..P { place(q * part + j, parts[q][j], q); } }
    for (i, &item) in rest.iter().enumerate() { place(P * part + i, item, P - 1); }
    (positions, sizes)
}

/// The first row count at which sorting uses radix passes rather than comparisons.
const RADIX_FROM: usize = 256;

/// The rows of an `n`-row table in ascending order, comparing entries column by column from the first. Rows that compare equal keep their order of position. `columns` holds each column's `n` keys, whose unsigned order is the order you want. A caller fills them from its items in a pass that vectorizes. Fewer than 256 rows take a comparison sort. One column whose keys `tabled` allows takes a counting sort. Other tables take the radix sort once for each column, from the last. Positions are `u32`s, as a slice holds at most `u32::MAX` items.
#[inline(never)]
pub fn sort_rows<K: RadixKey>(n: usize, columns: Vec<Vec<K>>) -> Vec<u32> {
    if n < RADIX_FROM {
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| columns.iter().map(|c| c[a as usize].cmp(&c[b as usize])).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal));
        return order;
    }
    if let [keys] = &columns[..] && let Some(order) = counted(keys) { return order; }
    let mut order: Vec<u32> = (0..n as u32).collect();
    if columns.is_empty() { return order; }
    let mut spare = (vec![K::default(); n], vec![0; n]);
    for (c, mut keys) in columns.into_iter().rev().enumerate() { radix_column(&mut keys, &mut order, &mut spare, c == 0); }
    order
}

/// Adds one more significant key column to a stable order, using the existing radix pass and its scratch.
fn radix_column<K: RadixKey>(keys: &mut Vec<K>, order: &mut Vec<u32>, spare: &mut (Vec<K>, Vec<u32>), first: bool) {
    if !first {
        for (key, &i) in spare.0.iter_mut().zip(order.iter()) { *key = keys[i as usize]; }
        std::mem::swap(keys, &mut spare.0);
    }
    radix(keys, order, spare);
}

/// The order of one column's `keys` by counting, when `table_size` allows their range. Equal keys keep their order. Each key's place follows the places of all smaller keys. The range of the first 256 keys rules out most keys that span too wide a range, such as floats', before a pass over all of them.
fn counted<K: RadixKey>(keys: &[K]) -> Option<Vec<u32>> {
    let n = keys.len();
    let size = |(lo, hi): (K, K)| table_size(hi.word() - lo.word(), n);
    size(K::extremes(&keys[..n.min(256)]))?;
    let (lo, hi) = K::extremes(keys);
    let range = size((lo, hi))?;
    let lo = lo.word();
    Some(grouped_by(keys, range, move |k| (k.word() - lo) as usize).into_positions())
}

/// `keys` sorted, with each key's position, keeping the order of positions among equal keys.
pub(crate) fn sorted<K: RadixKey>(mut keys: Vec<K>) -> (Vec<K>, Vec<u32>) {
    let n = keys.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    if n < RADIX_FROM { order.sort_by_key(|&i| keys[i as usize]); return (order.iter().map(|&i| keys[i as usize]).collect(), order); }
    radix(&mut keys, &mut order, &mut (vec![K::default(); n], vec![0; n]));
    (keys, order)
}

/// Sorts `keys`, and their positions in `order`, keeping the order of equal keys: an LSD radix sort, one byte at a time from the lowest, which skips any byte that is the same in every key. Each pass moves the keys and positions into `spare`, which holds as many, then swaps the two. They move as two arrays, so a pass over 64-bit keys moves 12 bytes for each, where pairs would move 16. A key no wider than its float keeps each pass's bytes in proportion to the float's width.
fn radix<K: RadixKey>(keys: &mut Vec<K>, order: &mut Vec<u32>, spare: &mut (Vec<K>, Vec<u32>)) {
    let n = keys.len();
    if n > RADIX_PIECE && crate::parallel::threads() > 1 { return radix_pieces(keys, order, spare); }
    let mut counts = [[0usize; 256]; 8];
    let counts = &mut counts[..K::BYTES];
    for &k in keys.iter() { for (b, count) in counts.iter_mut().enumerate() { count[k.byte(b)] += 1; } }
    for (b, count) in counts.iter().enumerate() {
        if count.contains(&n) { continue; }
        let (mut next, mut total) = ([0; 256], 0);
        for (place, &c) in next.iter_mut().zip(count) { (*place, total) = (total, total + c) }
        let (from, positions, to, places) = (&keys[..n], &order[..n], &mut spare.0[..n], &mut spare.1[..n]);
        for j in 0..n {
            let place = &mut next[from[j].byte(b)];
            (to[*place], places[*place]) = (from[j], positions[j]);
            *place += 1;
        }
        std::mem::swap(keys, &mut spare.0);
        std::mem::swap(order, &mut spare.1);
    }
}

/// The fewest keys in a `radix_pieces` piece. A pass cuts one piece for each thread, or fewer where the keys run short.
const RADIX_PIECE: usize = 1 << 16;

/// `radix` across threads, for more than one `RADIX_PIECE` of keys. Each pass counts the byte's values in each piece, gives each piece its own run of every bucket, after the runs of the pieces before it, and moves each piece's keys into its runs in order. The sort stays stable, so it gives `radix`'s result. A pass splits the unwritten part of `spare` into its runs, so each piece writes only its own.
fn radix_pieces<K: RadixKey>(keys: &mut Vec<K>, order: &mut Vec<u32>, spare: &mut (Vec<K>, Vec<u32>)) {
    use crate::parallel::pieces_of;
    /// One piece's run in each bucket, for keys and for positions.
    struct Runs<'a, K> { keys: Vec<std::slice::IterMut<'a, K>>, places: Vec<std::slice::IterMut<'a, u32>> }
    let n = keys.len();
    let count = (n / RADIX_PIECE).min(crate::parallel::threads());
    let size = n.div_ceil(count);
    let piece = move |p: usize| p * size..n.min((p + 1) * size);
    let mut every = vec![[[0u32; 256]; 8]; count];
    pieces_of(&mut every, 1, usize::MAX, &|p, c| {
        for &k in &keys[piece(p)] { for b in 0..K::BYTES { c[0][b][k.byte(b)] += 1; } }
        true
    });
    let mut moved = false;
    for b in 0..K::BYTES {
        let total = (0..256).map(|d| every.iter().map(|c| c[b][d] as usize).sum::<usize>());
        if total.into_iter().any(|t| t == n) { continue; }
        let mut counts: Vec<[u32; 256]> = every.iter().map(|c| c[b]).collect();
        if moved {
            pieces_of(&mut counts, 1, usize::MAX, &|p, c| {
                c[0] = [0; 256];
                for &k in &keys[piece(p)] { c[0][k.byte(b)] += 1; }
                true
            });
        }
        {
            let (mut to, mut places) = (&mut spare.0[..n], &mut spare.1[..n]);
            let mut runs: Vec<Runs<'_, K>> = (0..count).map(|_| Runs { keys: Vec::with_capacity(256), places: Vec::with_capacity(256) }).collect();
            for d in 0..256 {
                for (run, c) in runs.iter_mut().zip(&counts) {
                    let (k, rest) = std::mem::take(&mut to).split_at_mut(c[d] as usize);
                    let (o, others) = std::mem::take(&mut places).split_at_mut(c[d] as usize);
                    (to, places) = (rest, others);
                    run.keys.push(k.iter_mut());
                    run.places.push(o.iter_mut());
                }
            }
            let (from, positions) = (&keys[..n], &order[..n]);
            pieces_of(&mut runs, 1, usize::MAX, &|p, run| {
                let run = &mut run[0];
                let range = piece(p);
                let (from, positions) = (&from[range.clone()], &positions[range]);
                for j in 0..from.len() {
                    let d = from[j].byte(b);
                    *run.keys[d].next().expect("a run holds each key that the counts gave its piece") = from[j];
                    *run.places[d].next().expect("a run holds each key that the counts gave its piece") = positions[j];
                }
                true
            });
        }
        std::mem::swap(keys, &mut spare.0);
        std::mem::swap(order, &mut spare.1);
        moved = true;
    }
}

/// Positions grouped by a 64-bit key, such as a hash. A map gives the first and last position pushed with each key, and each position links to the next one pushed with the same key. Positions pushed in order give chains in order.
pub struct Chains { ends: HashMap<u64, (u32, u32)>, next: Vec<u32> }

impl Chains {
    /// Chains for positions below `len`.
    pub fn new(len: usize) -> Self { Self { ends: HashMap::with_capacity(len), next: vec![u32::MAX; len] } }
    /// Adds position `i` to the end of `key`'s chain.
    pub fn push(&mut self, key: u64, i: usize) {
        let i = i as u32;
        match self.ends.entry(key) {
            Entry::Occupied(mut e) => {
                let (_, last) = e.get_mut();
                self.next[*last as usize] = i;
                *last = i;
            }
            Entry::Vacant(e) => { e.insert((i, i)); }
        }
    }
    /// Adds `i` to the end of `key`'s chain, unless `copy` holds for a position already in it. One map lookup does both.
    pub fn push_new(&mut self, key: u64, i: usize, copy: impl Fn(usize) -> bool) {
        match self.ends.entry(key) {
            Entry::Occupied(mut e) => {
                let (first, last) = e.get_mut();
                if links(&self.next, *first, u32::MAX).any(copy) { return; }
                self.next[*last as usize] = i as u32;
                *last = i as u32;
            }
            Entry::Vacant(e) => { e.insert((i as u32, i as u32)); }
        }
    }
    /// The positions in `key`'s chain, in the order they were pushed.
    #[inline]
    pub fn chain(&self, key: u64) -> Links<'_> { links(&self.next, self.first(key), u32::MAX) }
    /// The positions in `key`'s chain, then those in `then`'s chain if `then` is given, each in the order they were pushed.
    #[inline]
    pub fn chains(&self, key: u64, then: Option<u64>) -> Links<'_> { links(&self.next, self.first(key), then.map_or(u32::MAX, |k| self.first(k))) }
    /// The first position pushed with `key`, or `u32::MAX` where there's none.
    #[inline]
    fn first(&self, key: u64) -> u32 { self.ends.get(&key).map_or(u32::MAX, |&(first, _)| first) }
}

/// The positions linked from `at` through `next`, then those linked from `then`.
#[inline]
fn links(next: &[u32], at: u32, then: u32) -> Links<'_> { Links { next, at, then } }

/// The positions in one or two of [`Chains`]' chains, each in the order they were pushed. Keep `next` this small: the search loops that call it rely on it inlining.
pub struct Links<'a> { next: &'a [u32], at: u32, then: u32 }

impl Iterator for Links<'_> {
    type Item = usize;
    #[inline]
    fn next(&mut self) -> Option<usize> {
        if self.at == u32::MAX {
            if self.then == u32::MAX { return None; }
            (self.at, self.then) = (self.then, u32::MAX);
        }
        let i = self.at;
        self.at = self.next[i as usize];
        Some(i as usize)
    }
}

/// `Some($body)` with `$x` bound to the items of the view `$view` when they're Booleans, characters or integers, the types that implement `Key`, and `None` for other items.
macro_rules! keys_of {
    ($view:expr, |$x:ident| $body:expr) => {
        match $view {
            View::Booleans($x) => Some($body),
            View::Characters($x) => Some($body),
            View::Integers(ints) => Some(with_ints!(ints, |$x| $body)),
            _ => None,
        }
    };
}

/// `Some($body)` with `$x` and `$y` bound to the items of the views `$a` and `$b` when both hold Booleans, both characters or both integers, with integers of different widths read as `i64`, and `None` for other pairs.
macro_rules! key_pairs {
    ($a:expr, $b:expr, |$x:ident, $y:ident| $body:expr) => {
        match ($a, $b) {
            (View::Booleans($x), View::Booleans($y)) => Some($body),
            (View::Characters($x), View::Characters($y)) => Some($body),
            (View::Integers($x), View::Integers($y)) => Some(with_int_pair!($x, $y, |$x, $y| $body)),
            _ => None,
        }
    };
}

/// The body of a search of one view: `Some(Ok($keys(x)))` for Booleans, integers and characters, `Some($floats(t, x, $check))` within the tolerance `$t` for floats, and `None` for complex numbers.
macro_rules! keys_or_floats {
    ($x:expr, $t:expr, $check:expr, $keys:path, $floats:path) => {{
        if let Some(found) = keys_of!($x, |x| $keys(x)) { return Some(Ok(found)); }
        let View::Floats(x) = $x else { return None };
        Some(with_floats!(x, |x| crate::with_tolerance!($t, |t| $floats(t, x, $check))))
    }};
}

/// The class of each item of `x`, as the first position of an item that matches it: exactly for Booleans, integers and characters, through [`classes`], and within `t` for floats, through [`tolerant::classes_with`], which calls `check` as it goes. `None` for complex numbers.
pub fn view_classes<E>(x: View<'_>, t: Tolerance, check: impl FnMut() -> Result<(), E>) -> Option<Result<Vec<u32>, E>> {
    keys_or_floats!(x, t, check, classes, tolerant::classes_with)
}

/// Whether each item of `x` is the first of its class, as [`view_classes`] matches them. `None` for complex numbers.
pub fn view_firsts<E>(x: View<'_>, t: Tolerance, check: impl FnMut() -> Result<(), E>) -> Option<Result<Vec<bool>, E>> {
    keys_or_floats!(x, t, check, firsts, tolerant::firsts_with)
}

/// The size of each class of `x`, in order of first position, as [`view_classes`] matches them. Booleans, integers and characters that span a table count from it, as [`class_counts`] does. `None` for complex numbers.
pub fn view_class_counts<E>(x: View<'_>, t: Tolerance, check: impl FnMut() -> Result<(), E>) -> Option<Result<Vec<usize>, E>> {
    if let Some(counts) = keys_of!(x, |x| class_counts(x)) { return Some(Ok(counts)); }
    Some(view_classes(x, t, check)?.map(|classes| numbered(&classes).counts))
}

/// The classes of `x`, as [`view_classes`] matches them, numbered in order of first position, with the positions of each. Booleans, integers and characters that span a table group straight from it, as [`grouped_keys`] does. `None` for complex numbers.
pub fn view_grouped<E>(x: View<'_>, t: Tolerance, check: impl FnMut() -> Result<(), E>) -> Option<Result<Grouped, E>> {
    if let Some(Some(grouped)) = keys_of!(x, |x| grouped_keys(x)) { return Some(Ok(grouped)); }
    Some(view_classes(x, t, check)?.map(|classes| { let Numbered { firsts, numbers, .. } = numbered(&classes); Grouped::new(firsts, &numbers) }))
}

/// For each item of `y`, the first position in `x` of an item that matches it, or the length of `x` where none does: exactly for Booleans, integers or characters of one kind, with integers of different widths read as `i64`, and within `t` for floats of one width, calling `check` as it goes. An integer haystack also takes `f64` needles when every needle is exactly whole and below [`tolerant::whole_limit`]. `None` for other pairs.
pub fn view_index_of<E>(x: View<'_>, y: View<'_>, t: Tolerance, check: impl FnMut() -> Result<(), E>) -> Option<Result<Vec<u32>, E>> {
    if let (DType::Float(w), true) = (x.dtype(), x.dtype() == y.dtype()) {
        return with_float_width!(w, F => Some(crate::with_tolerance!(t, |t| tolerant::index_of_with(t, F::of(x)?, F::of(y)?, check))));
    }
    if let (View::Integers(x), View::Floats(Floats::F64(y))) = (x, y) {
        return with_ints!(x, |x| whole_needles(x, y, t).map(Ok));
    }
    key_pairs!(x, y, |x, y| Ok(index_of(x, y)))
}

/// Exact whole float needles small enough that tolerance cannot equate them with another integer. Other needles decline this integer fast path.
fn whole_needles<K: Key>(x: &[K], y: &[f64], t: Tolerance) -> Option<Vec<u32>> {
    let limit = tolerant::whole_limit(t);
    let whole: Vec<i64> = y.iter().map(|&v| (v.fract() == 0.0 && v.abs() < limit).then_some(v as i64)).collect::<Option<_>>()?;
    let mut found = index_of(x, &whole.iter().filter_map(|&n| K::from_key(n)).collect::<Vec<_>>()).into_iter();
    Some(whole.iter().map(|&n| K::from_key(n).and_then(|_| found.next()).unwrap_or(x.len() as u32)).collect())
}

/// Set operations retain first-occurrence order, matching as classification and membership do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Set { Union, Intersection, Difference }

/// Unique values of a set operation on two views of one dtype. Shared searches do all matching; gathered output has no keys or names. `None` for unsupported pairs.
pub fn view_set<E>(x: View<'_>, y: View<'_>, how: Set, t: Tolerance, mut check: impl FnMut() -> Result<(), E>) -> Option<Result<Buffer, E>> {
    if x.dtype() != y.dtype() { return None; }
    let mut joined = Buffer::with_capacity(x.dtype(), if how == Set::Union { x.len() + y.len() } else { 0 });
    let values = if how == Set::Union {
        joined.extend_from(x, 0..x.len()); joined.extend_from(y, 0..y.len()); joined.view()
    } else { x };
    let firsts = match view_firsts(values, t, &mut check)? { Ok(firsts) => firsts, Err(e) => return Some(Err(e)) };
    let matches = if how == Set::Union { None } else {
        match view_index_of(y, values, t, &mut check)? { Ok(found) => Some(found), Err(e) => return Some(Err(e)) }
    };
    let positions: Vec<u32> = firsts.iter().enumerate().filter_map(|(i, &first)| {
        (first && matches.as_ref().is_none_or(|found| ((found[i] as usize) < y.len()) == (how == Set::Intersection))).then_some(i as u32)
    }).collect();
    Some(Ok(Buffer::gathered(values, &positions)))
}

/// For each item of `values`, the number of `boundaries` at or below it, as [`bins`] counts them: for Booleans, integers or characters of one kind, with integers of different widths read as `i64`, and for floats of one width by their order keys. `None` for other pairs.
pub fn view_bins(boundaries: View<'_>, values: View<'_>) -> Option<Vec<u32>> {
    if let (DType::Float(w), true) = (boundaries.dtype(), boundaries.dtype() == values.dtype()) {
        return with_float_width!(w, F => Some(bins(&tolerant::order_keys(F::of(boundaries)?), &tolerant::order_keys(F::of(values)?))));
    }
    key_pairs!(boundaries, values, |x, y| bins(x, y))
}

/// For each item of `values`, the number of `boundaries` below it, for views of one dtype other than complex: [`bins`] of the complemented order keys, with `boundaries` reversed. `None` for other views.
pub fn view_bins_below(boundaries: View<'_>, values: View<'_>) -> Option<Vec<u32>> {
    if boundaries.dtype() != values.dtype() { return None; }
    let (mut a, mut v) = (view_order_keys(boundaries)?, view_order_keys(values)?);
    let n = a.len() as u32;
    a.reverse();
    for k in a.iter_mut().chain(&mut v) { *k = !*k; }
    Some(bins(&a, &v).into_iter().map(|p| n - p).collect())
}

/// Runs an ordered operation at the view's item width. Complex numbers have no ordering.
macro_rules! ordered_view {
    ($view:expr, |$x:ident, $key:ident| $body:expr) => {
        Some(match $view {
            View::Booleans($x) => { let $key = OrderKey::order_key; $body },
            View::Characters($x) => { let $key = OrderKey::order_key; $body },
            View::Integers(ints) => with_ints!(ints, |$x| { let $key = OrderKey::order_key; $body }),
            View::Floats(floats) => with_floats!(floats, |$x| { let $key = Float::order_key; $body }),
            View::Complexes(_) => return None,
        })
    };
}

/// Each item's order key, widened to 64 bits. Within one dtype, their unsigned order is the items' order, with every NaN last. `None` for complex numbers.
pub fn view_order_keys(x: View<'_>) -> Option<Vec<u64>> {
    ordered_view!(x, |x, key| x.iter().map(|&v| key(v).word()).collect())
}

/// Stable positions that order a vector. Equal items keep their original positions, including signed zeros and NaNs.
pub fn view_grade(x: View<'_>, down: bool) -> Option<Vec<u32>> {
    ordered_view!(x, |x, key| sort_rows(x.len(), vec![order_column(x, x.len(), 0, 1, down, key, |k| k)]))
}

/// Stable lexicographic order of `n` rows, with columns in primary-first order. Each column is a compact view, its first key's offset and the step between keys. Columns can have different dtypes; radix keys keep each column's item width. `None` for complex keys.
pub fn view_lexgrade<'a>(n: usize, columns: impl Iterator<Item = (View<'a>, usize, usize)>, down: bool) -> Option<Vec<u32>> {
    let columns: Vec<_> = columns.collect();
    if let [(view, start, step)] = columns[..] {
        return ordered_view!(view, |x, key| sort_rows(n, vec![order_column(x, n, start, step, down, key, |k| k)]));
    }
    if n < RADIX_FROM {
        // One comparison sort is faster than sorting once per column. This small table needs no width-dispatch representation.
        let columns = columns.into_iter().map(|(view, start, step)| ordered_view!(view, |x, key| order_column(x, n, start, step, down, key, |k| k.word()))).collect::<Option<Vec<_>>>()?;
        return Some(sort_rows(n, columns));
    }
    let (mut order, mut positions) = ((0..n as u32).collect::<Vec<_>>(), vec![0; n]);
    for (c, (view, start, step)) in columns.into_iter().rev().enumerate() {
        ordered_view!(view, |x, key| {
            let mut keys = order_column(x, n, start, step, down, key, |k| k);
            let mut spare = (vec![Default::default(); n], std::mem::take(&mut positions));
            radix_column(&mut keys, &mut order, &mut spare, c == 0);
            positions = spare.1;
        })?;
    }
    Some(order)
}

/// One column's order keys, contiguous or strided. A contiguous pass uses the same vectorized/threaded loop as vector grading.
fn order_column<T: Copy + Sync, K: RadixKey, O: Send>(x: &[T], n: usize, start: usize, step: usize, down: bool, key: impl Fn(T) -> K + Sync, convert: impl Fn(K) -> O + Sync) -> Vec<O> {
    let key = move |v| convert(if down { !key(v) } else { key(v) });
    if step != 1 { return (0..n).map(|i| key(x[start + i * step])).collect(); }
    let mut keys = Vec::with_capacity(n);
    crate::pairs(crate::Arg::Many(&x[start..start + n]), crate::Arg::One(()), n, &mut keys, move |v, ()| (key(v), true));
    keys
}

/// A counting sort with no position buffer, when the items span a small table. Extra writes avoid one unpredictable branch per sparse count.
fn spread<T: Element + Key>(x: &[T], down: bool) -> Option<Buffer> {
    if x.len() < 256 { return None; }
    let (min, counts) = counts(x)?;
    let mut sorted = vec![T::FILL; x.len() + 4];
    let mut at = 0;
    let mut write = |(k, &c): (usize, &usize)| {
        let v = T::from_key(min + k as i64).unwrap_or(T::FILL);
        sorted[at..at + 4].fill(v);
        if c > 4 { sorted[at + 4..at + c].fill(v); }
        at += c;
    };
    if down { counts.iter().enumerate().rev().for_each(&mut write); }
    else { counts.iter().enumerate().for_each(&mut write); }
    sorted.truncate(x.len());
    Some(T::buffer(sorted))
}

/// A sorted vector, counting Boolean/integer/character values directly where possible, otherwise gathering its stable grade.
pub fn view_sort(x: View<'_>, down: bool) -> Option<Buffer> {
    let counted = match x {
        View::Booleans(x) => spread(x, down), View::Characters(x) => spread(x, down),
        View::Integers(ints) => with_ints!(ints, |x| spread(x, down)), _ => None,
    };
    counted.or_else(|| Some(Buffer::gathered(x, &view_grade(x, down)?)))
}

/// Sorts each lane through the shared stack scheduler, retaining the direct counting path for values.
pub fn view_sort_along(x: View<'_>, dims @ [outer, _, inner]: [usize; 3], down: bool) -> Option<Buffer> {
    if outer * inner == 1 { return view_sort(x, down); }
    if matches!(x, View::Complexes(_)) { return None; }
    Some(with_dtype!(x.dtype(), T => T::buffer(map_lanes::<_, _, ()>(T::of(x).unwrap(), dims, dims[1], |lane, out, _| {
        let sorted = view_sort(T::view(lane), down).unwrap();
        out.copy_from_slice(T::of(sorted.view()).unwrap());
    }))))
}

/// Places each of the sorted, distinct `ranks` at its sorted position, leaving the runs between them unordered. Ranks must be below `x.len()`.
/// Selecting the middle requested rank first partitions disjoint runs for the remaining ranks; one rank takes linear time.
pub fn select<T>(x: &mut [T], ranks: &[usize], compare: impl Fn(&T, &T) -> Ordering + Copy) {
    fn within<T>(x: &mut [T], ranks: &[usize], base: usize, compare: impl Fn(&T, &T) -> Ordering + Copy) {
        if ranks.is_empty() { return; }
        let middle = ranks.len() / 2;
        let rank = ranks[middle] - base;
        let (before, _, after) = x.select_nth_unstable_by(rank, compare);
        within(before, &ranks[..middle], base, compare);
        within(after, &ranks[middle + 1..], base + rank + 1, compare);
    }
    within(x, ranks, 0, compare);
}

/// A copy with `rank` at its sorted position. Other items are partitioned around it, not sorted. `None` for complex numbers.
pub fn view_partition(x: View<'_>, rank: usize) -> Option<Buffer> {
    fn partition<T: Element, K: Ord>(x: &[T], rank: usize, key: impl Fn(T) -> K + Copy) -> Buffer {
        let mut out = x.to_vec();
        select(&mut out, &[rank], |a, b| key(*a).cmp(&key(*b)));
        T::buffer(out)
    }
    ordered_view!(x, |x, key| partition(x, rank, key))
}

/// Positions partitioned around `rank`. Equal values need not keep their original order.
pub fn view_argpartition(x: View<'_>, rank: usize) -> Option<Vec<u32>> {
    let keys = view_order_keys(x)?;
    let mut out: Vec<u32> = (0..x.len() as u32).collect();
    select(&mut out, &[rank], |&a, &b| keys[a as usize].cmp(&keys[b as usize]));
    Some(out)
}

/// Partitions each lane directly in its output buffer. Index partitioning compares the original native-width values and writes i64 positions.
pub fn view_partition_along(x: View<'_>, dims: [usize; 3], rank: usize, indices: bool) -> Option<Buffer> {
    fn run<T: Element + Default, K: Ord>(x: &[T], dims: [usize; 3], rank: usize, indices: bool, key: impl Fn(T) -> K + Copy + Sync) -> Buffer {
        if indices { i64::buffer(map_lanes::<_, _, ()>(x, dims, dims[1], |lane, out, _| {
            for (i, p) in out.iter_mut().enumerate() { *p = i as i64; }
            select(out, &[rank], |a, b| key(lane[*a as usize]).cmp(&key(lane[*b as usize])));
        })) } else { T::buffer(map_lanes::<_, _, ()>(x, dims, dims[1], |lane, out, _| {
            out.copy_from_slice(lane);
            select(out, &[rank], |a, b| key(*a).cmp(&key(*b)));
        })) }
    }
    ordered_view!(x, |x, key| run(x, dims, rank, indices, key))
}

/// Only `range` of a stable grade, by selection followed by sorting the requested run. Original positions break ties because taking from a stable grade observes them.
pub fn select_positions(n: usize, range: std::ops::Range<usize>, compare: impl Fn(usize, usize) -> Ordering + Copy) -> Vec<u32> {
    if range.is_empty() { return Vec::new(); }
    let compare = |&a: &u32, &b: &u32| compare(a as usize, b as usize).then_with(|| a.cmp(&b));
    let mut order: Vec<u32> = (0..n as u32).collect();
    let ranks = if range.start == range.end - 1 { vec![range.start] } else { vec![range.start, range.end - 1] };
    select(&mut order, &ranks, compare);
    let mut selected = order[range].to_vec();
    selected.sort_unstable_by(compare);
    selected
}

/// The first minimum or maximum position of each lane, without gathering lanes or allocating a result per lane. Comparisons use sorting's NaN-last order.
pub fn view_argextremes(x: View<'_>, [outer, len, inner]: [usize; 3], down: bool) -> Option<Vec<i64>> {
    fn extremes<T: Copy + Sync, K: Ord>(x: &[T], [outer, len, inner]: [usize; 3], down: bool, key: impl Fn(T) -> K + Sync) -> Vec<i64> {
        let mut out = Vec::with_capacity(outer * inner);
        crate::parallel::extended(&mut out, outer * inner, (crate::CHUNK / len.max(1)).max(1), &|range, slots| {
            slots.fill(range.len(), |slots| {
                for (lane, slot) in range.zip(slots) {
                    let start = lane / inner * len * inner + lane % inner;
                    let (mut best, mut at) = (key(x[start]), 0);
                    for j in 1..len {
                        let k = key(x[start + j * inner]);
                        if if down { k > best } else { k < best } { best = k; at = j; }
                    }
                    slot.write(at as i64);
                }
                true
            });
            true
        });
        out
    }
    if len == 0 { return None; }
    ordered_view!(x, |x, key| extremes(x, [outer, len, inner], down, key))
}

/// Requested order statistics and interpolation weights, shared by every lane of a quantile call.
struct Quantiles { points: Vec<(usize, usize, f64)>, ranks: Vec<usize> }

impl Quantiles {
    /// The plan for `probabilities` of `len` items. A probability outside `[0, 1]`, or NaN, gives a Domain error.
    fn new(len: usize, probabilities: &[f64]) -> Result<Self, Error> {
        if !probabilities.iter().all(|q| (0.0..=1.0).contains(q)) { return Err(Error::new(Kind::Domain, "quantile probabilities must be between zero and one")); }
        let points: Vec<_> = probabilities.iter().map(|&q| {
            let p = q * len.saturating_sub(1) as f64;
            (p.floor() as usize, p.ceil() as usize, p.fract())
        }).collect();
        let mut ranks: Vec<_> = points.iter().flat_map(|&(lo, hi, _)| [lo, hi]).collect();
        ranks.sort_unstable();
        ranks.dedup();
        Ok(Self { points, ranks })
    }

    fn apply<T: Copy, F: Float>(&self, x: &mut [T], out: &mut [F], compare: impl Fn(&T, &T) -> Ordering + Copy, read: impl Fn(T) -> F::Working) {
        if x.is_empty() { out.fill(F::nan()); return; }
        select(x, &self.ranks, compare);
        for (out, &(lo, hi, weight)) in out.iter_mut().zip(&self.points) {
            let a = read(x[lo]);
            *out = F::from_working(if lo == hi { a } else {
                let weight = F::Working::narrow(weight);
                a * (F::Working::narrow(1.0) - weight) + read(x[hi]) * weight
            });
        }
    }
}

/// Linear-interpolated quantiles of `x`. Selection keeps the input width; f16 interpolation computes in f32. Empty inputs give NaN. A probability outside `[0, 1]`, or NaN, gives a Domain error.
pub fn quantiles<F: Float>(x: &mut [F], probabilities: &[f64]) -> Result<Vec<F>, Error> {
    let mut out = vec![F::nan(); probabilities.len()];
    Quantiles::new(x.len(), probabilities)?.apply(x, &mut out, |a, b| a.order_key().cmp(&b.order_key()), Float::working);
    Ok(out)
}

/// Quantiles of each lane, with a leading probability axis. Integers select exactly before converting the selected values to f64; floats keep their width. `None` for complex numbers, which have no order. A probability outside `[0, 1]`, or NaN, gives a Domain error.
pub fn view_quantiles_along(x: View<'_>, dims @ [outer, _, inner]: [usize; 3], probabilities: &[f64]) -> Result<Option<Buffer>, Error> {
    use crate::buffer::Source;
    fn run<T: Element, F: Element + Float + Default, K: Ord>(x: &[T], dims: [usize; 3], plan: &Quantiles, key: impl Fn(T) -> K + Copy + Sync,
        read: impl Fn(T) -> F::Working + Sync) -> Vec<F> {
        map_lanes::<_, _, Vec<T>>(x, dims, plan.points.len(), |lane, out, work| {
            work.clear();
            work.extend_from_slice(lane);
            plan.apply(work, out, |a, b| key(*a).cmp(&key(*b)), &read);
        })
    }
    fn floats<F: Element + Float + Default>(x: &[F], dims: [usize; 3], plan: &Quantiles) -> Buffer {
        F::buffer(run::<_, F, _>(x, dims, plan, Float::order_key, Float::working))
    }
    let plan = Quantiles::new(dims[1], probabilities)?;
    let items = match x {
        View::Booleans(x) => f64::buffer(run::<_, f64, _>(x, dims, &plan, OrderKey::order_key, Source::<f64>::read)),
        View::Integers(ints) => with_ints!(ints, |x| f64::buffer(run::<_, f64, _>(x, dims, &plan, OrderKey::order_key, Source::<f64>::read))),
        View::Floats(x) => with_floats!(x, |x| floats(x, dims, &plan)), _ => return Ok(None),
    };
    let count = probabilities.len();
    Ok(Some(if count > 1 && outer > 1 {
        let plan = crate::structure::transpose(&crate::layout::Layout::from([outer, count, inner]), &[1, 0, 2]).expect("valid quantile axes");
        Buffer::walked(items.view(), 0, &plan.steps, items.len())
    } else { items }))
}

/// Maps lanes through the shared stack scheduler. Each callback fills its output slice and reuses its scratch. Only strided inputs need a gathered lane.
fn map_lanes<T: Copy + Sync, U: Element + Default, S: Default>(x: &[T], [outer, len, inner]: [usize; 3], width: usize,
    map: impl Fn(&[T], &mut [U], &mut S) + Sync) -> Vec<U> {
    if outer * inner * width == 0 { return Vec::new(); }
    let out = crate::parallel::stack::<U, (Vec<T>, S), std::convert::Infallible>(outer * inner, width, &|lane, out, (input, work)| {
        let (o, i) = (lane / inner, lane % inner);
        let input = if inner == 1 { &x[o * len..(o + 1) * len] } else {
            input.clear();
            input.extend((0..len).map(|j| x[(o * len + j) * inner + i]));
            &input[..]
        };
        map(input, out, work);
    }, || Ok(())).unwrap();
    if inner == 1 || width == 1 { return out; }
    let mut transposed = Vec::with_capacity(out.len());
    transposed.resize(out.len(), U::FILL);
    for o in 0..outer {
        crate::linalg::transpose_into(&out[o * inner * width..][..inner * width], width, &mut transposed[o * inner * width..][..inner * width], inner, [inner, width], |x| x);
    }
    transposed
}
