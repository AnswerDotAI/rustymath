//! Search and sort over slices of keys. A key is an item read as a whole number (`Key`), or an unsigned word whose order is the order
//! you want (`RadixKey`). `Float::order_key` gives each float such a word, as wide as the float. Positions take 32 bits inside these
//! functions, so a slice holds at most `u32::MAX` items. `tolerant` builds its searches over floats on these functions.
//!
//! `index_of`, `classes` and `firsts` index the first position of each key. `index_of` passes over `x` once for each item of `y` when
//! `y` has at most `SCANNED` items. The index is a table when the keys span at most twice the slice's length, or 1024, and a hash map
//! otherwise. `numbered` numbers the classes in order of first position, with each one's size. `counts` counts the items with each
//! key in such a table. `sort_rows` sorts keys that span such a table by counting them. It sorts other keys with an LSD radix sort that
//! skips any byte that is the same in every key.
use hashbrown::{HashMap, hash_map::Entry};
use std::{hash::Hash, ops::Not};

/// An item as a whole number: a Boolean's 0 or 1, an integer, a character's code point, or an order key. Two items match exactly when
/// their keys are equal. A 64-bit unsigned key wraps to a negative number, which keeps keys distinct.
pub trait Key: Copy + Eq + Hash {
    fn key(self) -> i64;
}
macro_rules! keys {
    ($($t:ty),+) => {$(
        impl Key for $t {
            #[inline]
            fn key(self) -> i64 { self as i64 }
        }
    )+};
}
keys!(u8, u16, u32, u64, i8, i16, i32, i64);
impl Key for bool {
    #[inline]
    fn key(self) -> i64 { self.into() }
}
impl Key for char {
    #[inline]
    fn key(self) -> i64 { i64::from(u32::from(self)) }
}

/// An unsigned key that `sort_rows` sorts one byte at a time.
pub trait RadixKey: Copy + Ord + Default + Not<Output = Self> {
    const BYTES: usize;
    fn byte(self, b: usize) -> usize;
}
macro_rules! radix_keys {
    ($($t:ty),+) => {$(
        impl RadixKey for $t {
            const BYTES: usize = size_of::<$t>();
            #[inline]
            fn byte(self, b: usize) -> usize { (self >> (8 * b)) as usize & 255 }
        }
    )+};
}
radix_keys!(u16, u32, u64);

/// The number of items of `y` up to which `index_of` and `tolerant::index_of` pass over `x` once for each, in place of building an
/// index.
pub const SCANNED: usize = 16;

/// For each item of `y`, the first position of its key in `x`, or the length of `x` when it has none.
#[inline(never)]
pub fn index_of<K: Key>(x: &[K], y: &[K]) -> Vec<usize> {
    if y.len() <= SCANNED { return y.iter().map(|n| x.iter().position(|h| h == n).unwrap_or(x.len())).collect(); }
    let miss = x.len();
    if let Some((min, mut table)) = table(x, miss) {
        let first = write_first(x, min, &mut table);
        return y.iter().map(|&n| usize::try_from(n.key().wrapping_sub(min)).ok().and_then(|i| first.get(i)).map_or(miss, |&f| f)).collect();
    }
    let first = first_positions(x);
    y.iter().map(|n| first.get(n).map_or(miss, |&f| f as usize)).collect()
}

/// The first position of each item's key in `x`, which represents the item's class.
#[inline(never)]
pub fn classes<K: Key>(x: &[K]) -> Vec<usize> {
    if let Some((min, mut table)) = table(x, 0) {
        let first = write_first(x, min, &mut table);
        return x.iter().map(|&n| first[n.key().wrapping_sub(min) as usize]).collect();
    }
    let first = first_positions(x);
    x.iter().map(|n| first[n] as usize).collect()
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

/// The least key of `x`, and the number of items with each key from it to the greatest, when `tabled` allows that range. With fewer
/// than 64 keys, runs of one key are common, and each count would wait for the previous one to be stored. So the items then go to
/// four tables in turn, which are summed at the end. On the Mac, that makes 2 keys 3 times as fast, and 64 keys or more no faster.
#[inline(never)]
pub fn counts<K: Key>(x: &[K]) -> Option<(i64, Vec<usize>)> {
    let (min, mut counts) = table(x, 0)?;
    let n = counts.len();
    let at = |k: &K| k.key().wrapping_sub(min) as usize;
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
pub fn numbered(classes: &[usize]) -> Numbered {
    let mut number = vec![0; classes.len()];
    let (mut firsts, mut numbers, mut counts) = (Vec::new(), Vec::with_capacity(classes.len()), Vec::new());
    for (i, &c) in classes.iter().enumerate() {
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

/// The least key of `x`, and a table with an entry for each key from it to the greatest, each `empty`, when `tabled` allows that range.
fn table<K: Key>(x: &[K], empty: usize) -> Option<(i64, Vec<usize>)> {
    let (min, max) = x.iter().fold((i64::MAX, i64::MIN), |(lo, hi), n| (lo.min(n.key()), hi.max(n.key())));
    let range = usize::try_from(max.checked_sub(min)?).ok()?.checked_add(1)?;
    tabled(range, x.len()).then(|| (min, vec![empty; range]))
}

/// Whether `len` keys that span `range` values index a table: when `range` is at most twice `len`, or 1024.
fn tabled(range: usize, len: usize) -> bool { range <= len.saturating_mul(2).max(1024) }

/// Writes the first position of each key in `x` to its entry in `first`, whose entries start at key `min`. The positions are written
/// from the last to the first, so each entry ends with its key's first position, and the loop never reads what it writes. The table
/// is a slice, not a `Vec`, so that the compiler keeps its bounds in registers.
fn write_first<'t, K: Key>(x: &[K], min: i64, first: &'t mut [usize]) -> &'t [usize] {
    for (i, &n) in x.iter().enumerate().rev() { first[n.key().wrapping_sub(min) as usize] = i; }
    first
}

/// A map from each key of `x` to its first position. The map holds the items themselves, so an entry takes their width.
fn first_positions<K: Key>(x: &[K]) -> HashMap<K, u32> {
    let mut first = HashMap::with_capacity(x.len());
    for (i, &n) in x.iter().enumerate().rev() { first.insert(n, i as u32); }
    first
}

/// The positions of `groups` in order of group, keeping their order within each group, and the number of positions in each of the
/// `n` groups. With few groups, a run of one group would make each count wait for the previous one to be stored. So the positions
/// are then split into four parts, and each part has its own counts and its own next place in each group.
#[inline(never)]
pub fn grouped(groups: &[usize], n: usize) -> (Vec<usize>, Vec<usize>) {
    let len = groups.len();
    let parts = if n.saturating_mul(16) <= len { 4 } else { 1 };
    let part = len.div_ceil(parts).max(1);
    let mut next = vec![0; parts * n];
    for j in 0..part { for p in 0..parts { if let Some(&g) = groups.get(p * part + j) { next[p * n + g] += 1; } } }
    // Each group's places start after the earlier groups, with part 0 first, as its positions come first.
    let (mut sizes, mut start) = (vec![0; n], 0);
    for (g, size) in sizes.iter_mut().enumerate() {
        for p in 0..parts {
            let count = std::mem::replace(&mut next[p * n + g], start);
            start += count;
            *size += count;
        }
    }
    let mut positions = vec![0; len];
    for j in 0..part {
        for p in 0..parts {
            let i = p * part + j;
            if let Some(&g) = groups.get(i) {
                let place = &mut next[p * n + g];
                positions[*place] = i;
                *place += 1;
            }
        }
    }
    (positions, sizes)
}

/// The rows of an `n`-row table in ascending order, comparing entries column by column from the first. Rows that compare equal keep
/// their order of position. `key(row, column)` gives each entry as a key whose unsigned order is the order you want. Fewer than 256
/// rows take a comparison sort. One column whose keys `tabled` allows takes a counting sort. Other tables take the radix sort once
/// for each column, from the last.
#[inline(never)]
pub fn sort_rows<K: RadixKey>(n: usize, width: usize, key: impl Fn(usize, usize) -> K) -> Vec<usize> {
    if n < 256 {
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| (0..width).map(|c| key(a, c).cmp(&key(b, c))).find(|o| o.is_ne()).unwrap_or(std::cmp::Ordering::Equal));
        return order;
    }
    let (mut items, mut spare): (Vec<(K, u32)>, _) = ((0..n as u32).map(|i| (K::default(), i)).collect(), Vec::new());
    for column in (0..width).rev() {
        for item in &mut items { item.0 = key(item.1 as usize, column); }
        if width == 1 && let Some(order) = counted(&items) { return order; }
        spare.resize(n, (K::default(), 0));
        radix(&mut items, &mut spare);
    }
    items.into_iter().map(|(_, i)| i as usize).collect()
}

/// The order of one column of `items`, each a key and its row, by counting, when `tabled` allows the range of their keys. Rows with
/// equal keys keep their order. Each key's place follows the places of all smaller keys.
fn counted<K: RadixKey>(items: &[(K, u32)]) -> Option<Vec<usize>> {
    let (lo, hi) = items.iter().fold((!K::default(), K::default()), |(lo, hi), &(k, _)| (lo.min(k), hi.max(k)));
    let range = usize::try_from(word(hi) - word(lo)).ok()?.checked_add(1)?;
    if !tabled(range, items.len()) { return None; }
    let groups: Vec<usize> = items.iter().map(|&(k, _)| (word(k) - word(lo)) as usize).collect();
    Some(grouped(&groups, range).0)
}

/// `k` as a number, from its bytes.
fn word<K: RadixKey>(k: K) -> u64 { (0..K::BYTES).fold(0, |w, b| w | (k.byte(b) as u64) << (8 * b)) }

/// `items` sorted by key, keeping their order among equal keys.
pub(crate) fn sorted<K: RadixKey>(mut items: Vec<(K, u32)>) -> Vec<(K, u32)> {
    if items.len() < 256 { items.sort_by_key(|&(k, _)| k) } else {
        let mut spare = vec![(K::default(), 0); items.len()];
        radix(&mut items, &mut spare);
    }
    items
}

/// Sorts `items` by key, keeping their order among equal keys: an LSD radix sort, one byte at a time from the lowest, which skips any
/// byte that is the same in every item. The sort moves the items between `items` and `spare`, which holds as many. A key no wider than
/// its float keeps each pass's bytes in proportion to the float's width.
fn radix<K: RadixKey>(items: &mut Vec<(K, u32)>, spare: &mut Vec<(K, u32)>) {
    let n = items.len();
    let mut counts = [[0usize; 256]; 8];
    let counts = &mut counts[..K::BYTES];
    for &(k, _) in items.iter() { for (b, count) in counts.iter_mut().enumerate() { count[k.byte(b)] += 1; } }
    for (b, count) in counts.iter().enumerate() {
        if count.contains(&n) { continue; }
        scatter(items, spare, count, |&(k, _)| k.byte(b));
        std::mem::swap(items, spare);
    }
}

/// Copies the items of `source` into `target` grouped by bucket, keeping their order within each bucket. `counts` holds the number of
/// items in each bucket, and `target` holds exactly as many items as `source`. `source` is a slice because, without LTO, iterating
/// through `Copied` would make a call for each item.
fn scatter<T: Copy>(source: &[T], target: &mut [T], counts: &[usize], bucket: impl Fn(&T) -> usize) {
    let mut next = Vec::with_capacity(counts.len());
    let mut total = 0;
    for &count in counts {
        next.push(total);
        total += count;
    }
    for &item in source {
        let b = bucket(&item);
        target[next[b]] = item;
        next[b] += 1;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn search_and_sort() {
        let x = [30, 10, 20, 10];
        assert_eq!(index_of(&x, &[10, 40]), [1, 4]);
        // More than `SCANNED` needles build the index in place of scanning.
        let many: Vec<i32> = (0..20).map(|i| [10, 40, 30][i % 3]).collect();
        assert_eq!(index_of(&x, &many), (0..20).map(|i| [1, 4, 0][i % 3]).collect::<Vec<_>>());
        assert_eq!(numbered(&classes(&x)), Numbered { firsts: vec![0, 1, 2], numbers: vec![0, 1, 2, 1], counts: vec![1, 2, 1] });
        assert_eq!(classes(&x), [0, 1, 2, 1]);
        assert_eq!(sort_rows(x.len(), 1, |r, _| x[r] as u32), [1, 3, 2, 0]);
        // 300 rows take the counting sort. Equal keys keep their order of position, also under complemented keys, which reverse it.
        let y: Vec<u32> = (0..300).map(|i| i * 7 % 5).collect();
        let mut stable: Vec<usize> = (0..300).collect();
        stable.sort_by_key(|&r| y[r]);
        assert_eq!(sort_rows(y.len(), 1, |r, _| y[r]), stable);
        stable.sort_by_key(|&r| !y[r]);
        assert_eq!(sort_rows(y.len(), 1, |r, _| !y[r]), stable);
        // 299 items of 5 keys go to four tables, with 3 left over. A range of 2^20 + 1 is too wide for a table.
        let z = &y[..299];
        assert_eq!(counts(z), Some((0, (0..5).map(|k| z.iter().filter(|&&v| v == k).count()).collect())));
        assert_eq!((counts(&[3, 1, 3, 2]), counts(&[0, 1 << 20])), (Some((1, vec![1, 1, 2])), None));
    }
}
