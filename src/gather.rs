//! Copying the items of a slice into a buffer by position: cycling them, repeating each, keeping those a mask selects, taking them at indices, taking windows of them, walking an n-dimensional pattern of offsets over them, and writing them at offsets. `positions` writes the positions that a mask selects. Each applies a conversion to each item it writes, so a caller can widen items as it gathers them. Each is generic and `#[inline]`, so it compiles in the caller, with the caller's types and conversion.
use crate::search::Key;
use std::borrow::Cow;
use std::mem::MaybeUninit;

/// Appends `len` items, cycling through the items of the nonempty `items`, each converted by `f`.
#[inline]
pub fn cycle<S: Copy, D: Clone>(out: &mut Vec<D>, items: &[S], len: usize, f: impl Fn(S) -> D) {
    if let [x] = *items { return out.resize(out.len() + len, f(x)); }
    for start in (0..len).step_by(items.len()) { out.extend(items[..items.len().min(len - start)].iter().map(|&x| f(x))) }
}

/// Appends each of `items`, converted by `f`, as many times as its count. `counts` holds one count for all the items, or one for each item. The counts are nonnegative and sum to `total`. An item with a count of at most 4 writes four copies where its run starts. The items after it overwrite the copies beyond its count. A larger count fills its run.
#[inline]
pub fn repeat<S: Copy, D: Copy>(out: &mut Vec<D>, items: &[S], counts: &[i64], total: usize, f: impl Fn(S) -> D) {
    let count = |j: usize| counts[if counts.len() == 1 { 0 } else { j }] as usize;
    crate::slice::appended(out, total, |d| {
        let mut k = 0;
        for (j, &x) in items.iter().enumerate() {
            let (x, n) = (MaybeUninit::new(f(x)), count(j));
            match d[k..].first_chunk_mut::<4>() { Some(w) if n <= 4 => *w = [x; 4], _ => d[k..k + n].fill(x) }
            k += n;
        }
        assert_eq!(k, total, "the counts must sum to `total`");
    })
}

/// Appends `item(j)` for each position `j` whose count in `mask` is 1, in order. Every count is 0 or 1, and `total` of them are 1. It suits items read from a slice: where a block of counts holds both 0s and 1s, `written` reads every item in order.
#[inline]
pub fn compress<M: Key, D: Copy>(out: &mut Vec<D>, mask: &[M], total: usize, item: impl Fn(usize) -> D) { kept::<false, M, D>(out, mask, total, item) }

/// Appends `position(j)` for each position `j` whose count in `mask` is 1, in order. Every count is 0 or 1, and `total` of them are 1. It suits items computed from `j`: where a block of counts holds both 0s and 1s, `selected` computes only the selected items.
#[inline]
pub fn positions<M: Key, D: Copy>(out: &mut Vec<D>, mask: &[M], total: usize, position: impl Fn(usize) -> D) { kept::<true, M, D>(out, mask, total, position) }

/// `compress`, or `positions` with `TABLE`. Long runs cost little: a block of 64 counts that are all 0 is skipped, and one whose counts are all 1 is copied. Without `TABLE`, consecutive blocks that mix both go through `written` in one pass. Each count is read by its low bit, which writes every slot whatever the counts hold.
#[inline(always)]
fn kept<const TABLE: bool, M: Key, D: Copy>(out: &mut Vec<D>, mask: &[M], total: usize, item: impl Fn(usize) -> D) {
    crate::slice::appended(out, total, |d| {
        // `k` slots are written. Without `TABLE`, the mixed blocks from position `from`, holding `mixed` counts of 1, wait for one pass.
        let (mut k, mut from, mut mixed) = (0, 0, 0);
        for (start, m) in (0..).step_by(64).zip(mask.chunks(64)) {
            // A block's count is at most 64, which fits in a byte. Byte additions vectorize in the widest lanes.
            let mut n = 0u8;
            for x in m { n += x.key() as u8 & 1 }
            let n = usize::from(n);
            if !TABLE && 0 < n && n < m.len() { mixed += n; continue; }
            if mixed > 0 { written(&mut d[k..], &mask[from..start], |j| item(from + j)) }
            (k, mixed, from) = (k + mixed, 0, start + m.len());
            if n == m.len() { let run = &mut d[k..k + n]; for j in 0..n { run[j].write(item(start + j)); } }
            else if TABLE {
                let (blocks, rest) = m.as_chunks::<8>();
                let mut at = k;
                for (b, block) in blocks.iter().enumerate() {
                    let start = start + 8 * b;
                    at = selected(d, at, mask_bits(block), |i| item(start + usize::from(i)));
                }
                let (start, last) = (start + 8 * blocks.len(), mask_bits(rest));
                selected(d, at, last, |i| item(start + usize::from(i)));
            }
            k += n;
        }
        if mixed > 0 { written(&mut d[k..], &mask[from..], |j| item(from + j)) }
        assert_eq!(k + mixed, total, "`total` counts must be 1");
    })
}

/// Writes `item(j)` into `d`, in order, for each position `j` whose count in `mask` is 1. `d` holds at least as many slots as the counts add up to. Each item is written where it goes if its count is 1, and the next item writes over it if its count is 0, so the loop has no branch on the counts. It takes four counts at a time, so that one addition moves on past all four.
// Out of line: inlined into `kept`'s loop over blocks, the loop below ran out of x86-64's 16 registers and reloaded six values from the stack for every four counts.
#[inline(never)]
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
    // An index loop, which compiles inline in the incremental build. A whole block's bytes take one load, which a mask on each byte would split. The multiplication gathers the low bit of each byte into the top byte.
    let mut bytes = 0u64;
    for i in 0..m.len() { bytes |= u64::from(m[i].key() as u8) << (8 * i) }
    ((bytes & 0x0101_0101_0101_0101).wrapping_mul(0x0102_0408_1020_4080) >> 56) as u8
}

/// Writes `item(i)` into `d` from place `k`, in order, for each set bit `i` of `mask`, and gives the place after the last one. Where 8 places remain, it writes all 8 items in a loop with no check or branch. Later calls write over the places after the selected items. `D` is `Copy`, which leaves nothing to drop in the items written over.
#[inline(always)]
pub(crate) fn selected<D: Copy>(d: &mut [MaybeUninit<D>], k: usize, mask: u8, item: impl Fn(u8) -> D) -> usize {
    let set = &SET[usize::from(mask)];
    let found = mask.count_ones() as usize;
    if let Some(w) = d[k..].first_chunk_mut::<8>() { for i in 0..8 { w[i].write(item(set[i])); } }
    else { for i in 0..found { d[k + i].write(item(set[i])); } }
    k + found
}

/// Appends the items of `items` at `indices`, each converted by `f`. Each index is in `-n..n` for `n` items, and a negative index counts back from the end.
#[inline]
pub fn take<S: Copy, D, I: Copy + Into<i64>>(out: &mut Vec<D>, items: &[S], indices: &[I], f: impl Fn(S) -> D) {
    let n = items.len() as i64;
    out.extend(indices.iter().map(move |&i| { let i: i64 = i.into(); f(items[(i + (n & (i >> 63))) as usize]) }));
}

/// Appends `m` windows of `W` items. Window `k` holds the items of `s` at `o + k × step` plus each offset in `cell`, each converted by `f`. Each item is written once, into `out`'s spare capacity.
#[inline]
pub fn windows<const W: usize, S: Copy, D>(out: &mut Vec<D>, s: &[S], o: usize, step: isize, m: usize, cell: &[usize; W], f: impl Fn(S) -> D) {
    crate::slice::appended(out, m * W, |slots| {
        for (k, window) in slots.as_chunks_mut::<W>().0.iter_mut().enumerate() {
            let s = &s[o.wrapping_add_signed(k as isize * step)..];
            for i in 0..W { window[i].write(f(s[cell[i]])); }
        }
    })
}

/// Replaces the item of `out` at each offset with the matching one of `items`, converted by `f`, or with the only one when `items` holds one. A later offset wins.
#[inline]
pub fn scatter<S: Copy, D>(out: &mut [D], offsets: &[usize], items: &[S], f: impl Fn(S) -> D) {
    if let [x] = *items { return offsets.iter().for_each(|&o| out[o] = f(x)); }
    for (&o, &x) in offsets.iter().zip(items) { out[o] = f(x) }
}
/// Adds each item to the item of `out` at its offset. Offsets can repeat, and each repeat adds again: the transpose of a gather that reads a position more than once.
pub fn scatter_add<T: Copy + std::ops::AddAssign>(out: &mut [T], pairs: impl IntoIterator<Item = (usize, T)>) {
    for (o, x) in pairs { out[o] += x }
}

/// The source offsets that one axis of `walk`'s result reads. Regular patterns are held as a formula, so a long axis needs no table. A `None` offset gives a fill.
#[derive(Clone, Debug)]
pub enum Steps {
    /// `len` offsets from `start`, each `step` further on. A negative step walks backwards.
    Stride { start: usize, len: usize, step: isize },
    /// `len` positions from `start` along an axis of `size` items with `stride`. The start may lie outside the axis, and a position outside gives a fill.
    Clipped { start: i128, len: usize, size: usize, stride: usize },
    /// All `len` positions of an axis with `stride`, from position `shift`, wrapping round to the start.
    Rotated { shift: usize, len: usize, stride: usize },
    /// Any offsets.
    Table(Vec<Option<usize>>),
}

impl Steps {
    /// Every position of an axis of `len` items with `stride`, in order.
    pub fn along(len: usize, stride: usize) -> Self { Self::Stride { start: 0, len, step: stride as isize } }
    /// The number of offsets.
    pub fn len(&self) -> usize {
        match self { Self::Stride { len, .. } | Self::Clipped { len, .. } | Self::Rotated { len, .. } => *len, Self::Table(t) => t.len() }
    }
    /// Offset `j`, or `None` for a fill.
    pub fn get(&self, j: usize) -> Option<usize> {
        match *self {
            Self::Stride { start, step, .. } => Some(start.wrapping_add_signed(step * j as isize)),
            Self::Clipped { start, size, stride, .. } => {
                let position = start + j as i128;
                (0..size as i128).contains(&position).then(|| position as usize * stride)
            }
            Self::Rotated { shift, len, stride } => Some((shift + j) % len * stride),
            Self::Table(ref t) => t[j],
        }
    }
    /// Every offset, in order.
    pub fn offsets(&self) -> impl ExactSizeIterator<Item = Option<usize>> + '_ { (0..self.len()).map(|j| self.get(j)) }
    /// Every sum of one offset from each of `tables`, in the order that `walk` visits them. No table may reach a fill.
    pub fn combined(tables: &[impl std::borrow::Borrow<Self>]) -> Vec<usize> {
        tables.iter().fold(vec![0], |sums, t| sums.iter().flat_map(|&s| t.borrow().offsets().map(move |o| s + o.expect("no fills"))).collect())
    }
    /// `steps` without axes of one position, whose offset moves into `base`, and with neighbouring runs merged into one longer run.
    fn simplify(mut base: usize, steps: &[Self]) -> (usize, Vec<Cow<'_, Self>>) {
        let mut kept: Vec<Cow<Self>> = Vec::with_capacity(steps.len());
        for s in steps {
            if let Some(o) = if s.len() == 1 { s.get(0) } else { None } { base += o; continue; }
            kept.push(Cow::Borrowed(s));
            while let [.., outer, inner] = kept.as_slice() {
                let (&Self::Stride { start: a, len: m, step }, &Self::Stride { start: b, len: n, step: 1 }) = (outer.as_ref(), inner.as_ref()) else { break };
                if step != n as isize { break; }
                kept.truncate(kept.len() - 2);
                kept.push(Cow::Owned(Self::Stride { start: a + b, len: m * n, step: 1 }));
            }
        }
        (base, kept)
    }
}

/// Appends the items of `s` that `steps` reach from `base`, each converted by `f`: one `Steps` for each axis, with the last varying fastest. A `None` offset gives `fill`, converted. The last axis copies whole slices where its pattern allows.
#[inline]
pub fn walk<S: Copy + 'static, D: Clone + 'static>(out: &mut Vec<D>, s: &[S], base: usize, steps: &[Steps], fill: S, f: impl Fn(S) -> D) {
    let (base, steps) = Steps::simplify(base, steps);
    walk_into(out, s, &f, fill, base, &steps)
}

/// The longest run of adjacent items that `walk_into` copies item by item. A longer run copies with `extend_from_slice`.
const RUN: usize = 8;
/// The most items below one axis that `walk_into` reaches through a table of their offsets.
const CELL: usize = 64;

/// `walk` after `Steps::simplify`.
fn walk_into<S: Copy + 'static, D: Clone + 'static>(d: &mut Vec<D>, s: &[S], f: &impl Fn(S) -> D, fill: S, base: usize, steps: &[Cow<Steps>]) {
    let filled = || f(fill);
    let Some((axis, rest)) = steps.split_first() else { return d.push(f(s[base])) };
    if let [inner] = rest {
        if let (&Steps::Stride { start: a, len: m, step: 1 }, &Steps::Stride { start: b, len: n, step }) = (&**axis, &**inner) {
            if step > 1 { return transposed(d, &s[base + a + b..], f, filled(), m, n, step as usize); }
        }
        // One run of `n` adjacent items for each offset along `axis`, as windows give. A run longer than `RUN` copies fastest with `extend_from_slice`, which needs the source and the buffer to hold one type.
        if let Steps::Stride { start: b, len: n, step: 1 } = **inner {
            let same = if n > RUN { (d as &mut dyn std::any::Any).downcast_mut::<Vec<S>>() } else { None };
            if let Some(d) = same {
                d.reserve(axis.len() * n);
                for o in axis.offsets() { match o { Some(o) => d.extend_from_slice(&s[base + b + o..][..n]), None => d.extend(std::iter::repeat_n(fill, n)) } }
                return;
            }
        }
    }
    let width: usize = rest.iter().map(|s| s.len()).product();
    // Each offset along `axis` starts a copy of the same few items, at the offsets that `rest` reaches from it, as windows give. One table of those offsets serves every start. Along a stride, `windows` makes each start's items one array, written straight into spare capacity, so the loop over starts makes no call. Other axes write the fills first, which a missing start keeps.
    let run = match rest.last().map(|t| &**t) { Some(&Steps::Stride { len, step: 1, .. }) => len, _ => 1 };
    if !rest.is_empty() && (1..=CELL).contains(&width) && run <= RUN && rest.iter().all(|t| matches!(**t, Steps::Stride { .. })) {
        let cell = Steps::combined(rest);
        if let Steps::Stride { start: a, len: m, step } = **axis {
            macro_rules! widths { ($($w:literal)+) => { match width { $($w => return windows::<$w, _, _>(d, s, base + a, step, m, cell.as_slice().try_into().expect("`cell` holds `width` offsets"), |x| f(x)),)+ _ => {} } } }
            widths!(2 3 4 5 6 7 8 9);
        }
        let start = d.len();
        d.resize(start + axis.len() * width, filled());
        for (out, o) in d[start..].chunks_exact_mut(width).zip(axis.offsets()) {
            let Some(o) = o else { continue };
            let s = &s[base + o..];
            for (slot, &c) in out.iter_mut().zip(&cell) { *slot = f(s[c]) }
        }
        return;
    }
    if !rest.is_empty() {
        for j in 0..axis.len() {
            match axis.get(j) { Some(o) => walk_into(d, s, f, fill, base + o, rest), None => d.extend(std::iter::repeat_n(filled(), width)) }
        }
        return;
    }
    if axis.len() == 0 { return; }
    // Each loop gets the source from `base` onwards by value, so that the compiler keeps its bounds in registers.
    let s = &s[base..];
    let slice = move |start: usize, len: usize| s[start..start + len].iter().map(|&x| f(x));
    match **axis {
        Steps::Stride { start, len, step: 1 } => d.extend(slice(start, len)),
        Steps::Stride { start, len, step: -1 } => d.extend(slice(start + 1 - len, len).rev()),
        Steps::Stride { start, len, step } if step > 0 => { let (s, step) = (&s[start..], step as usize); d.extend((0..len).map(move |j| f(s[j * step]))) }
        Steps::Rotated { shift, len, stride: 1 } => {
            d.extend(slice(shift, len - shift));
            d.extend(slice(0, shift));
        }
        Steps::Clipped { start, len, size, stride: 1 } => {
            let (lo, hi) = (start.clamp(0, size as i128), (start + len as i128).clamp(0, size as i128));
            let before = (lo - start).clamp(0, len as i128) as usize;
            let middle = (hi - lo).max(0) as usize;
            d.extend(std::iter::repeat_n(filled(), before));
            d.extend(slice(lo as usize, middle));
            d.extend(std::iter::repeat_n(filled(), len - before - middle));
        }
        _ => d.extend(axis.offsets().map(move |o| o.map_or_else(filled, |o| f(s[o])))),
    }
}

/// Appends `m` rows of `n` items, where item `j` of row `i` is `f(s[i + j * stride])`, as `walk_into` does for a transposed matrix. Each block of 8 result rows is filled first, then overwritten by the tiled `linalg::transpose_into`.
fn transposed<S: Copy, D: Clone>(d: &mut Vec<D>, s: &[S], f: &impl Fn(S) -> D, fill: D, m: usize, n: usize, stride: usize) {
    for i0 in (0..m).step_by(8) {
        let (start, rows) = (d.len(), 8.min(m - i0));
        d.resize(start + rows * n, fill.clone());
        // `|x| f(x)` calls `f` directly. Passing `f` calls it through `&F`'s `Fn` shim, which LLVM left out of line, doubling the cost.
        crate::linalg::transpose_into(&s[i0..], stride, &mut d[start..], n, [n, rows], |x| f(x));
    }
}
