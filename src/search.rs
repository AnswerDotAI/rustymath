//! Search and sort over slices of keys. A key is an item read as a whole number (`Key`), or an unsigned word whose order is the order
//! you want (`RadixKey`). `Float::order_key` gives each float such a word, as wide as the float. Positions take 32 bits inside these
//! functions, so a slice holds at most `u32::MAX` items. `tolerant` builds its searches over floats on these functions.
//!
//! `index_of`, `classes` and `firsts` index the first position of each key. `index_of` passes over `x` once for each item of `y` when
//! `y` has at most `SCANNED` items. The index is a table when the keys span at most twice the slice's length, or 1024, and a hash map
//! otherwise. `numbered` numbers the classes in order of first position, with each one's size. `grouped_keys` numbers them the same
//! way from a table of keys, with each one's positions. `counts` counts the items with each key in such a table. `sort_rows` sorts
//! keys that span such a table by counting them. It sorts other keys with an LSD radix sort that skips any byte that is the same in
//! every key.
use hashbrown::{HashMap, hash_map::Entry};
use std::{hash::Hash, ops::Not};

/// An item as a whole number: a Boolean's 0 or 1, an integer, a character's code point, or an order key. Two items match exactly when
/// their keys are equal. A 64-bit unsigned key wraps to a negative number, which keeps keys distinct.
pub trait Key: Copy + Send + Sync + Eq + Ord + Hash {
    fn key(self) -> i64;
    /// The item whose key is `key`, if there is one.
    fn from_key(key: i64) -> Option<Self>;
    /// The least and the greatest key of `x`, which isn't empty, in one pass that vectorizes for narrow items.
    fn range(x: &[Self]) -> (i64, i64);
}
/// The least and the greatest key of `x`, through `reduce::extremes`, for item types that `Min` and `Max` reduce.
#[inline]
fn extreme_keys<K: Key>(x: &[K]) -> (i64, i64)
where
    crate::reduce::Min: crate::reduce::Reduction<K, Total = K>,
    crate::reduce::Max: crate::reduce::Reduction<K, Total = K>,
{
    let (lo, hi) = crate::reduce::extremes(x);
    (lo.key(), hi.key())
}
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
keys!(u8, u16, u32, u64, i8, i16, i32, i64);
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
    fn range(x: &[Self]) -> (i64, i64) {
        let (lo, hi) = crate::reduce::extremes_of(x, |c: char| c as u32);
        (lo.into(), hi.into())
    }
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
radix_keys!(u16, u32, u64);

/// The number of items of `y` up to which `index_of` and `tolerant::index_of` pass over `x` once for each, in place of building an
/// index.
pub const SCANNED: usize = 16;

/// For each item of `y`, the first position of its key in `x`, or the length of `x` when it has none. A slice here holds at most
/// `u32::MAX` items, so positions are `u32`s.
#[inline(never)]
pub fn index_of<K: Key>(x: &[K], y: &[K]) -> Vec<u32> {
    let miss = x.len() as u32;
    if y.len() <= SCANNED { return y.iter().map(|n| x.iter().position(|h| h == n).map_or(miss, |p| p as u32)).collect(); }
    if let Some((min, mut table)) = table(x, usize::MAX) {
        let first = write_first(x, min, &mut table);
        // `move` keeps `min`, `first` and `miss` in registers. Captured by reference, they would load again for every item.
        let at = move |n: K| usize::try_from(n.key().wrapping_sub(min)).ok().and_then(|i| first.get(i));
        return y.iter().map(move |&n| at(n).map_or(miss, |&f| f.min(miss as usize) as u32)).collect();
    }
    let first = first_positions(x);
    y.iter().map(|n| first.get(n).map_or(miss, |&f| f as u32)).collect()
}

/// The first position of each item's key in `x`, which represents the item's class.
#[inline(never)]
pub fn classes<K: Key>(x: &[K]) -> Vec<u32> {
    if let Some((min, mut table)) = table(x, usize::MAX) {
        let first = write_first(x, min, &mut table);
        return x.iter().map(move |&n| first[n.key().wrapping_sub(min) as usize] as u32).collect();
    }
    let first = first_positions(x);
    x.iter().map(|n| first[n]).collect()
}

/// Whether each item of `x` is the first with its key. Once the index holds the first position of each key, marking those positions
/// gives the result, with no second pass over `x`.
#[inline(never)]
pub fn firsts<K: Key>(x: &[K]) -> Vec<bool> {
    let mut mask = vec![false; x.len()];
    if let Some((min, mut table)) = table(x, usize::MAX) {
        for &f in write_first(x, min, &mut table) { if let Some(m) = mask.get_mut(f) { *m = true } }
        return mask;
    }
    for &f in first_positions(x).values() { mask[f as usize] = true }
    mask
}

/// The number of items of `x` equal to `v`. Each of 32 lanes counts its matches in a byte, emptied into the total every 255 blocks, so
/// the comparisons and the counts stay in narrow lanes. A count in `usize` widens each item to 64 bits.
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

/// The least key of `x`, and the number of items with each key from it to the greatest, when `tabled` allows that range. Up to 4 keys,
/// each count but the first takes a pass that vectorizes. With fewer than 64 keys, runs of one key are common, and each count would
/// wait for the previous one to be stored. So the items then go to four tables in turn, which are summed at the end. On the Mac, four
/// tables made 2 keys 3 times as fast as one table, and 64 keys or more no faster.
#[inline(never)]
pub fn counts<K: Key>(x: &[K]) -> Option<(i64, Vec<usize>)> {
    let (min, mut counts) = table(x, 0)?;
    let n = counts.len();
    let at = move |k: &K| k.key().wrapping_sub(min) as usize;
    if n <= 4 {
        for (g, count) in counts.iter_mut().enumerate().skip(1) {
            *count = K::from_key(min + g as i64).map_or(0, |v| count_eq(x, v));
        }
        counts[0] = x.len() - counts[1..].iter().sum::<usize>();
        return Some((min, counts));
    }
    if n >= 64 {
        for k in x { counts[at(k)] += 1 }
        return Some((min, counts));
    }
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

/// For each of `values`, the number of `boundaries` at or below it, as BQN's dyadic `⍋` and NumPy's `searchsorted` with `side='right'`
/// give it. `boundaries` must be in ascending order. When they are, and they span few enough keys for a table, a table of the count of
/// boundaries at or below each key answers each value with one read. Otherwise each value takes a binary search, which halves its
/// range with a select in place of a branch, so mispredicted comparisons don't stall it. Sixteen values search together, level by
/// level, because every search of `boundaries` halves the same lengths. Their loads at each level are then independent.
#[inline(never)]
pub fn bins<K: Key>(boundaries: &[K], values: &[K]) -> Vec<u32> {
    const LANES: usize = 16;
    let n = boundaries.len();
    let (Some(&first), Some(&last)) = (boundaries.first(), boundaries.last()) else { return vec![0; values.len()] };
    let at = move |k: K| k.key().wrapping_sub(first.key()) as u64 as usize;
    if let Some(range) = table_size(last.key().wrapping_sub(first.key()) as u64, n).filter(|_| boundaries.is_sorted()) {
        let mut below = vec![0u32; range];
        // In ascending order, the count at or below a key is one more than the position of its last boundary. Storing positions,
        // with no read, keeps a run of one key from making each store wait for the one before. Keys between boundaries take the count
        // below them.
        for (i, &b) in boundaries.iter().enumerate() { below[at(b)] = i as u32 + 1 }
        let mut total = 0;
        for b in &mut below { total = total.max(*b); *b = total }
        return values.iter().map(move |&v| if v < first { 0 } else if v > last { n as u32 } else { below[at(v)] }).collect();
    }
    let search = move |v: &[K], out: &mut [u32]| {
        let mut base = [0; LANES];
        let mut len = n;
        while len > 1 {
            let half = len / 2;
            // On x86-64, LLVM can turn a select that feeds a load in a loop into a branch. Random values mispredict that branch at
            // half the levels. `select_unpredictable` keeps a conditional move.
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

/// The least key of `x`, and a table with an entry for each key from it to the greatest, each `empty`, when `table_size` allows that
/// range. `Key::range` finds the least and the greatest key in one pass.
fn table<K: Key>(x: &[K], empty: usize) -> Option<(i64, Vec<usize>)> {
    if x.is_empty() { return None }
    let (min, max) = K::range(x);
    table_size(max.abs_diff(min), x.len()).map(|range| (min, vec![empty; range]))
}

/// The number of entries in a table indexed by `len` keys that span `span + 1` values, when that's at most twice `len`, or 1024.
fn table_size(span: u64, len: usize) -> Option<usize> { usize::try_from(span).ok()?.checked_add(1).filter(|&r| r <= len.saturating_mul(2).max(1024)) }

/// Writes the first position of each key in `x` to its entry in `first`, whose entries start at key `min` and hold `usize::MAX` until
/// written. The pass runs forward and stops after the first block of 64 positions that fills the entry of the last key `x` holds. Keys
/// that fill their range, such as Booleans, soon do. Keys that leave gaps in their range, as the characters of text do, never fill it.
/// So where the keys span at most 128 values, after `STALL` blocks in a row with no new key, the pass counts the keys `x` holds once,
/// and stops when it has found them all. Each entry takes its new position by a select, not a branch. The table is a slice, not a
/// `Vec`, so that the compiler keeps its bounds in registers.
fn write_first<'t, K: Key>(x: &[K], min: i64, first: &'t mut [usize]) -> &'t [usize] {
    let (mut empty, mut stall, mut counted) = (first.len(), 0, first.len() > 128);
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

/// The number of blocks of 64 positions in a row with no new key after which `write_first` counts the keys of its items. A run that
/// long is rare while keys that fill their range are still arriving.
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

/// A map from each key of `x` to its first position. The map holds the items themselves, so an entry takes their width.
fn first_positions<K: Key>(x: &[K]) -> HashMap<K, u32> {
    let mut first = HashMap::with_capacity(x.len());
    for (i, &n) in x.iter().enumerate().rev() { first.insert(n, i as u32); }
    first
}

/// The positions of `groups` in order of group, keeping their order within each group, and the number of positions in each of the
/// `n` groups.
#[inline(never)]
pub fn grouped(groups: &[usize], n: usize) -> (Vec<u32>, Vec<usize>) { grouped_by(groups, n, |g| g) }

/// The classes of a slice, numbered from 0 in order of their first positions, with the positions of each.
#[derive(Clone, Debug, PartialEq)]
pub struct Grouped {
    /// The first position of each class, in ascending order.
    pub firsts: Vec<usize>,
    /// The positions of each class in turn, ascending within each class. A slice holds at most `u32::MAX` items.
    pub positions: Vec<u32>,
    /// The number of items in each class.
    pub sizes: Vec<usize>,
}

impl Grouped {
    /// The positions of each class, one slice for each class in turn.
    pub fn each(&self) -> impl Iterator<Item = &[u32]> {
        let mut rest = &self.positions[..];
        self.sizes.iter().map(move |&n| {
            let (these, more) = rest.split_at(n);
            rest = more;
            these
        })
    }
}

/// The classes of `x` as [`Grouped`] gives them, or `None` when its keys span too many values for a table. The table entry of each key
/// takes its first position, then its class number, so no pass gives each item its first position.
#[inline(never)]
pub fn grouped_keys<K: Key>(x: &[K]) -> Option<Grouped> {
    let (min, mut table) = table(x, usize::MAX)?;
    write_first(x, min, &mut table);
    let mut firsts: Vec<usize> = table.iter().copied().filter(|&f| f != usize::MAX).collect();
    firsts.sort_unstable();
    for (g, &f) in firsts.iter().enumerate() { table[x[f].key().wrapping_sub(min) as usize] = g; }
    let table = &table[..];
    // With two keys, each item's class is its offset from the lower key, flipped when the higher key comes first.
    let (positions, sizes) = if let [lower, _] = *table {
        grouped_by(x, 2, |k| k.key().wrapping_sub(min) as usize ^ lower)
    } else {
        grouped_by(x, firsts.len(), |k| table[k.key().wrapping_sub(min) as usize])
    };
    Some(Grouped { firsts, positions, sizes })
}

/// `grouped` for the positions of `items`, each in the group that `group` gives its item.
#[inline(always)]
fn grouped_by<T: Copy + Sync>(items: &[T], n: usize, group: impl Fn(T) -> usize + Sync) -> (Vec<u32>, Vec<usize>) {
    let len = items.len();
    if n <= 1 { return ((0..len as u32).collect(), vec![len; n]) }
    if n == 2 { return two_groups(items, group) }
    // A slice here holds at most `u32::MAX` items, so the counts are `u32`s. Up to 256 groups, a run of one group would make each
    // count wait for the previous one to be stored. So the positions then split into four parts, each with its own counts and next
    // place in each group. Above 256 groups such runs are rare, and one part measured faster.
    if n <= 256 && n * 16 <= len { parted::<4, T>(items, n, group) } else { parted::<1, T>(items, n, group) }
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

/// `grouped_by` for two groups. One pass writes each item's group as a byte, which vectorizes. A multiplication turns each block of 8
/// bytes into a mask, and `SET` gives the offsets of each group's items in the block. Each group writes 8 positions from its next
/// place, of which the first are its own, then moves on by its count: group 0 from the front, group 1 from where group 0 ends. No
/// write depends on a group.
#[inline(always)]
fn two_groups<T: Copy + Sync>(items: &[T], group: impl Fn(T) -> usize + Sync) -> (Vec<u32>, Vec<usize>) {
    let len = items.len();
    let mut groups = Vec::with_capacity(len);
    crate::pairs(crate::Arg::Many(items), crate::Arg::One(()), len, &mut groups, |k, ()| (group(k) as u8, true));
    let mut ones = 0;
    for &g in &groups { ones += usize::from(g) }
    let zeros = len - ones;
    // Group 0's last writes can reach group 1's first 8 places, so `head` keeps those to write again after the loop. Eight spare places
    // at the end take group 1's last writes.
    let (mut d, mut head) = (vec![0u32; len + 8], [0u32; 8]);
    let (mut k0, mut k1) = (0, zeros);
    let (blocks, rest) = groups.as_chunks::<8>();
    for (b, block) in blocks.iter().enumerate() {
        // The multiplication gathers the low bit of each byte into the top byte.
        let mask = (u64::from_le_bytes(*block).wrapping_mul(0x0102_0408_1020_4080) >> 56) as usize;
        let start = (b * 8) as u32;
        *d[k1..].first_chunk_mut().expect("8 places") = offsets(start, &SET[mask]);
        let found = mask.count_ones() as usize;
        if k1 < zeros + 8 { for i in k1..(k1 + found).min(zeros + 8) { head[i - zeros] = d[i] } }
        *d[k0..].first_chunk_mut().expect("8 places") = offsets(start, &SET[mask ^ 255]);
        (k0, k1) = (k0 + 8 - found, k1 + found);
    }
    for (j, &g) in rest.iter().enumerate() {
        let (one, i) = (usize::from(g), (blocks.len() * 8 + j) as u32);
        if one == 1 && k1 < zeros + 8 { head[k1 - zeros] = i }
        d[if one == 1 { k1 } else { k0 }] = i;
        (k0, k1) = (k0 + 1 - one, k1 + one);
    }
    let h = ones.min(8);
    d[zeros..zeros + h].copy_from_slice(&head[..h]);
    d.truncate(len);
    (d, vec![zeros, ones])
}

/// The 8 positions from `start` at the offsets `set`, built in a fixed array so that the loop needs no call or check.
#[inline(always)]
fn offsets(start: u32, set: &[u8; 8]) -> [u32; 8] {
    let mut positions = [0; 8];
    for i in 0..8 { positions[i] = start + u32::from(set[i]) }
    positions
}

/// `grouped_by` for more than two groups, with the positions split into `P` parts. The number of parts is a constant, so each loop over
/// the parts unrolls, and indexing a group's places by part needs no check.
fn parted<const P: usize, T: Copy>(items: &[T], n: usize, group: impl Fn(T) -> usize) -> (Vec<u32>, Vec<usize>) {
    let (len, part) = (items.len(), items.len() / P);
    let parts: [&[T]; P] = std::array::from_fn(|q| &items[q * part..][..part]);
    let rest = &items[P * part..];
    let (mut positions, mut sizes) = (vec![0u32; len], vec![0; n]);
    let mut next = vec![[0u32; P]; n];
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
    let mut place = |i: usize, item: T, q: usize| {
        let p = &mut next[group(item)][q];
        positions[*p as usize] = i as u32;
        *p += 1;
    };
    for j in 0..part { for q in 0..P { place(q * part + j, parts[q][j], q); } }
    for (i, &item) in rest.iter().enumerate() { place(P * part + i, item, P - 1); }
    (positions, sizes)
}

/// The rows of an `n`-row table in ascending order, comparing entries column by column from the first. Rows that compare equal keep
/// their order of position. `columns` holds each column's `n` keys, whose unsigned order is the order you want. A caller fills them
/// from its items in a pass that vectorizes. Fewer than 256 rows take a comparison sort. One column whose keys `tabled` allows takes a
/// counting sort. Other tables take the radix sort once for each column, from the last, which sorts the last column's keys in place. Positions are `u32`s, as a slice holds at most `u32::MAX` items.
#[inline(never)]
pub fn sort_rows<K: RadixKey>(n: usize, mut columns: Vec<Vec<K>>) -> Vec<u32> {
    if n < 256 {
        let mut order: Vec<u32> = (0..n as u32).collect();
        order.sort_by(|&a, &b| columns.iter().map(|c| c[a as usize].cmp(&c[b as usize])).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal));
        return order;
    }
    if let [keys] = &columns[..] && let Some(order) = counted(keys) { return order; }
    let Some(mut keys) = columns.pop() else { return (0..n as u32).collect() };
    let mut order: Vec<u32> = (0..n as u32).collect();
    let mut spare = (vec![K::default(); n], vec![0; n]);
    loop {
        radix(&mut keys, &mut order, &mut spare);
        let Some(column) = columns.pop() else { break };
        for (k, &i) in keys.iter_mut().zip(&order) { *k = column[i as usize] }
    }
    order
}

/// The order of one column's `keys` by counting, when `table_size` allows their range. Equal keys keep their order. Each key's place
/// follows the places of all smaller keys. The range of the first 256 keys rules out most keys that span too wide a range, such as
/// floats', before a pass over all of them.
fn counted<K: RadixKey>(keys: &[K]) -> Option<Vec<u32>> {
    let n = keys.len();
    let size = |(lo, hi): (K, K)| table_size(hi.word() - lo.word(), n);
    size(K::extremes(&keys[..n.min(256)]))?;
    let (lo, hi) = K::extremes(keys);
    let range = size((lo, hi))?;
    let lo = lo.word();
    Some(grouped_by(keys, range, move |k| (k.word() - lo) as usize).0)
}

/// `keys` sorted, with each key's position, keeping the order of positions among equal keys.
pub(crate) fn sorted<K: RadixKey>(mut keys: Vec<K>) -> (Vec<K>, Vec<u32>) {
    let n = keys.len();
    let mut order: Vec<u32> = (0..n as u32).collect();
    if n < 256 {
        order.sort_by_key(|&i| keys[i as usize]);
        return (order.iter().map(|&i| keys[i as usize]).collect(), order);
    }
    radix(&mut keys, &mut order, &mut (vec![K::default(); n], vec![0; n]));
    (keys, order)
}

/// Sorts `keys`, and their positions in `order`, keeping the order of equal keys: an LSD radix sort, one byte at a time from the
/// lowest, which skips any byte that is the same in every key. Each pass moves the keys and positions into `spare`, which holds as
/// many, then swaps the two. They move as two arrays, so a pass over 64-bit keys moves 12 bytes for each, where pairs would move 16.
/// A key no wider than its float keeps each pass's bytes in proportion to the float's width.
fn radix<K: RadixKey>(keys: &mut Vec<K>, order: &mut Vec<u32>, spare: &mut (Vec<K>, Vec<u32>)) {
    let n = keys.len();
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

/// Positions grouped by a 64-bit key, such as a hash. A map gives the first and last position pushed with each key, and each position
/// links to the next one pushed with the same key. Positions pushed in order give chains in order.
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
            Entry::Vacant(e) => {
                e.insert((i, i));
            }
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
            Entry::Vacant(e) => {
                e.insert((i as u32, i as u32));
            }
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

/// The positions in one or two of [`Chains`]' chains, each in the order they were pushed. Keep `next` this small: the search loops
/// that call it rely on it inlining.
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

