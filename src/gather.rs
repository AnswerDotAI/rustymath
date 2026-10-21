//! Copying the items of a slice into a buffer by position: repeating each, keeping those a mask selects, taking them at indices, walking an n-dimensional pattern of offsets over them, and writing them at offsets. `positions` writes the positions that a mask selects. Each applies a conversion to each item it writes, so a caller can widen items as it gathers them. Most are generic and `#[inline]`, so they compile in the caller, with the caller's types and conversion. [`walk`] and [`scatter_walk`] instead compile their walk once for each item type, and run any other conversion as a separate contiguous pass. Their fixed-width loops move [`Unit`]s, which compile once for each size of item. Booleans and integers convert into `f64` through fixed-width loops of their own as they write ([`Item::cells_from`]).
use crate::buffer::View;
use crate::search::Key;
use crate::slice::Arg;
use crate::structure::Edge;
use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::sync::Arc;

/// Appends `n` items of the nonempty `s`, read cyclically from position `from`. After one period, it copies a block of the whole periods it has written, doubling the block up to `CYCLE_BLOCK` bytes.
pub(crate) fn cycled<T: Copy>(d: &mut Vec<T>, s: &[T], from: usize, n: usize) {
    if let [x] = *s { return d.resize(d.len() + n, x); }
    let (start, from) = (d.len(), from % s.len());
    let first = (s.len() - from).min(n);
    d.extend_from_slice(&s[from..from + first]);
    d.extend_from_slice(&s[..(n - first).min(from)]);
    let mut block = d.len() - start;
    while d.len() - start < n {
        let done = d.len() - start;
        d.extend_from_within(start..start + block.min(n - done));
        if block * size_of::<T>() < CYCLE_BLOCK { block = d.len() - start; }
    }
}

/// The bytes of whole periods that `cycled` copies at a time, small enough to stay in cache.
const CYCLE_BLOCK: usize = 32 << 10;

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
    out.extend(indices.iter().map(move |&i| f(items[crate::structure::counted_back(i.into(), items.len())])));
}

/// Writes `slots` as windows of `W` units. Window `k` holds the units of `s` at `o + k × step` plus each offset in `cell`.
#[inline]
fn windows<const W: usize, U: Unit>(slots: &mut [MaybeUninit<U>], s: &[U], o: usize, step: isize, cell: &[usize; W]) {
    for (k, window) in slots.as_chunks_mut::<W>().0.iter_mut().enumerate() {
        let s = &s[o.wrapping_add_signed(k as isize * step)..];
        for i in 0..W { window[i].write(s[cell[i]]); }
    }
}

/// Writes `m` cells of `W` units into `out`, where [`windows`] reads them. Cell `r` takes the places at `o + r × step` plus each offset in `cell`, from the units of `items` at `k + r × W` on, or from its only unit when it holds one.
#[inline]
fn windows_into<const W: usize, U: Unit>(out: &mut [U], items: &[U], o: usize, step: isize, m: usize, cell: &[usize; W], k: usize) {
    match items {
        [x] => for r in 0..m {
            let at = o.wrapping_add_signed(r as isize * step);
            for i in 0..W { out[at + cell[i]] = *x }
        },
        _ => for (r, cell_items) in items[k..][..m * W].as_chunks::<W>().0.iter().enumerate() {
            let at = o.wrapping_add_signed(r as isize * step);
            for i in 0..W { out[at + cell[i]] = cell_items[i] }
        },
    }
}

/// Replaces the item of `out` at each offset with the matching one of `items`, converted by `conv`, or with the only one when `items` holds one. A later offset wins. A [`Steps::FILL`] offset writes nothing.
#[inline]
pub fn scatter<S: Copy, D>(out: &mut [D], offsets: &[usize], items: &[S], conv: impl Convert<S, D>) {
    for (o, x) in targets(offsets, Arg::of(items)) { if o != Steps::FILL { out[o] = conv.one(x) } }
}
/// Replaces the items of `out` that `steps` reach from `base`, in the order that [`walk`] reads them, with the matching ones of `items`, converted by `conv`, or with the only one when `items` holds one. A fill position writes nothing. A conversion other than [`Same`] converts a block of items at a time, whole rows of the first axis up to about `CHUNK` items, then writes that block while it's still in cache. Where `D` has a fixed-width loop that converts from `S` ([`Item::cells_from`]), short cells along a stride convert as they write instead, with no separate pass. The walk's fixed-width loops then move items as [`Unit`]s, so they compile once for each size of item, whatever the types and conversion. Rows longer than `RUN` write along a slice each. Short cells write through one table of their offsets, as [`walk`] reads them. Cells that a table of starts picks, each reaching items a stride apart, as columns' do, must be disjoint or identical. They write in blocks of 8 cells, row by row, so each block's writes share the lines they touch, as [`walk`]'s reads do. A later identical cell wins. Other steps write in [`walk`]'s order.
pub fn scatter_walk<S: Item, D: Item>(out: &mut [D], items: &[S], base: usize, steps: &[Steps], conv: impl Convert<S, D>) {
    let (base, steps) = Steps::simplify(base, steps);
    if let Some(items) = conv.borrowed(items) { return scatter_into(out, items, base, &steps, &mut 0); }
    if let [x] = *items { return scatter_into(out, &[conv.one(x)], base, &steps, &mut 0); }
    let Some((axis, rest)) = steps.split_first() else { return scatter_into(out, &conv.all(items), base, &steps, &mut 0) };
    // Short cells along a stride, as joins of short lanes and writes of a few columns give, convert as they write where `D` has a fixed-width loop for `S` (`Item::cells_from`).
    if let (&Steps::Stride { start, step, .. }, Some(_)) = (&**axis, short_cells(rest)) {
        if D::cells_from(out, items, &conv, base + start, step, rest) { return; }
    }
    let width: usize = rest.iter().map(|s| s.len()).product();
    let rows = (crate::slice::CHUNK / width.max(1)).max(1);
    let mut block = Vec::with_capacity(rows * width);
    for j in (0..axis.len()).step_by(rows) {
        let count = rows.min(axis.len() - j);
        block.clear();
        conv.slice(&mut block, &items[j * width..][..count * width]);
        let mut k = 0;
        for piece in axis.pieces(j, count) {
            let (base, first, rest) = Steps::boxed(base, piece, rest);
            scatter_axis(out, &block, base, &first, rest, &mut k);
        }
    }
}

/// [`scatter_walk`]'s walk after `Steps::simplify`, writing from item `k` on, and moving `k` past each position, fills included.
fn scatter_into<T: Item>(out: &mut [T], items: &[T], base: usize, steps: &[Cow<Steps>], k: &mut usize) {
    match steps.split_first() {
        Some((axis, rest)) => scatter_axis(out, items, base, axis, rest, k),
        None => { out[base] = nth(items, *k); *k += 1; }
    }
}

/// The item of `items` at `k`, or its only item where it holds one.
#[inline]
fn nth<T: Copy>(items: &[T], k: usize) -> T { if let [x] = items { *x } else { items[k] } }

/// [`scatter_into`] for a box whose first axis, `axis`, comes apart from the others, `rest`, as [`walk_axis`] takes one.
fn scatter_axis<T: Item>(out: &mut [T], items: &[T], base: usize, axis: &Steps, rest: &[Cow<Steps>], k: &mut usize) {
    let item = |k: usize| nth(items, k);
    if rest.is_empty() {
        match *axis {
            Steps::Table(ref t) => t.chunks(|j0, run| for (j, &o) in run.iter().enumerate() { if o != Steps::FILL { out[base + o] = item(*k + j0 + j) } }),
            Steps::Stride { start, len, step: 1 } => row(&mut out[base + start..][..len], items, *k),
            Steps::Stride { start, len, step } => for j in 0..len { out[(base + start).wrapping_add_signed(step * j as isize)] = item(*k + j) },
            _ => for (j, o) in axis.offsets().enumerate() { if let Some(o) = o { out[base + o] = item(*k + j) } },
        }
        *k += axis.len();
        return;
    }
    if let [inner] = rest {
        if let Steps::Stride { start: b, len: n, step } = **inner {
            // Rows longer than `RUN` write along a slice each.
            if step == 1 && n > RUN {
                axis.for_each(|r, o| if let Some(o) = o { row(&mut out[base + o + b..][..n], items, *k + r * n) });
                *k += axis.len() * n;
                return;
            }
            // Cells picked by a table of starts, each reaching items a stride apart, as columns are, write in blocks, as `picked` reads them.
            if step > 1 && matches!(*axis, Steps::Table(_)) {
                let mut starts = Vec::with_capacity(axis.len()); axis.for_each(|_, o| starts.push(o));
                for block in starts.chunks(8) {
                    for j in 0..n {
                        for (c, o) in block.iter().enumerate() { if let Some(o) = o { out[base + o + b + j * step as usize] = item(*k + c * n + j) } }
                    }
                    *k += block.len() * n;
                }
                return;
            }
        }
    }
    // Each position along `axis` starts a cell of the same few items, at the offsets that `rest` reaches from it, which `walk_into` reads in the same way (`short_cells`). One table of those offsets serves every cell. Along a stride, `windows_into` takes each cell's items as one array, so the loop over cells makes no call.
    if let Some(width) = short_cells(rest) {
        let cell = cell_offsets(rest, width);
        let cell = &cell[..width];
        if let Steps::Stride { start: a, len: m, step } = *axis {
            // SAFETY: `windows_into` writes only units read from `items`, which hold items of `out`'s type.
            macro_rules! widths { ($($w:literal)+) => { match width { $($w => { windows_into::<$w, T::Unit>(unsafe { written_as(out) }, units(items), base + a, step, m, cell.try_into().expect("`cell` holds `width` offsets"), *k); *k += m * width; return; })+ _ => {} } } }
            widths!(2 3 4 5 6 7 8 9);
        }
        match items {
            [x] => axis.for_each(|_, o| if let Some(o) = o { for &c in cell { out[base + o + c] = *x } }),
            _ => axis.for_each(|r, o| if let Some(o) = o {
                let items = &items[*k + r * width..][..width];
                for i in 0..width { out[base + o + cell[i]] = items[i] }
            }),
        }
        *k += axis.len() * width;
        return;
    }
    let width: usize = rest.iter().map(|s| s.len()).product();
    axis.for_each(|_, o| match o { Some(o) => scatter_into(out, items, base + o, rest, k), None => *k += width });
}
/// Writes the slice `out` from the items of `items` at `k` on, or from its only item when it holds one.
#[inline]
fn row<T: Copy>(out: &mut [T], items: &[T], k: usize) {
    match items {
        [x] => out.fill(*x),
        _ => { let items = &items[k..][..out.len()]; for i in 0..items.len() { out[i] = items[i] } }
    }
}
/// Each offset with its value: the matching item of `values`, or its only item.
#[inline]
pub fn targets<'a, A: Copy>(offsets: &'a [usize], values: Arg<'a, A>) -> impl Iterator<Item = (usize, A)> + 'a {
    offsets.iter().enumerate().map(move |(k, &o)| (o, match values { Arg::One(y) => y, Arg::Many(ys) => ys[k] }))
}
/// Combines each item with the item of `out` at its offset by `f`, in order, as `out[o] = f(out[o], x)`. Offsets can repeat, and each repeat combines again. With addition, this is the transpose of a gather that reads a position more than once. It skips each offset of [`Steps::FILL`].
pub fn scatter_with<D: Copy, S>(out: &mut [D], pairs: impl IntoIterator<Item = (usize, S)>, f: impl Fn(D, S) -> D) {
    for (o, x) in pairs { if o != Steps::FILL { out[o] = f(out[o], x) } }
}
/// [`scatter_with`] for an `f` that gives `None` for a pair it can't combine. At the first such pair, it restores every item that earlier pairs changed and gives false.
pub fn try_scatter_with<D: Copy, S>(out: &mut [D], pairs: impl IntoIterator<Item = (usize, S)>, f: impl Fn(D, S) -> Option<D>) -> bool {
    let pairs = pairs.into_iter();
    let mut undo = Vec::with_capacity(pairs.size_hint().0);
    for (o, x) in pairs {
        if o == Steps::FILL { continue; }
        let Some(new) = f(out[o], x) else {
            for (o, old) in undo.into_iter().rev() { out[o] = old }
            return false;
        };
        undo.push((o, std::mem::replace(&mut out[o], new)));
    }
    true
}
/// Whether `xs` is 0, 1, 2 and so on: offsets that read a source in order, or an axis order that a transpose leaves unchanged.
pub fn counts_up(xs: &[usize]) -> bool { xs.iter().enumerate().all(|(i, &x)| x == i) }

/// The position that position `p` of a padded axis of `size` items reads under `edge`, or `None` for a fill.
fn edged(edge: Edge, p: i128, size: usize) -> Option<usize> { edge.position(i64::try_from(p).ok()?, size) }

/// The source offsets that one axis of `walk`'s result reads. Regular patterns are held as a formula, so a long axis needs no table. A `None` offset gives a fill.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Steps {
    /// `len` offsets from `start`, each `step` further on. A negative step walks backwards.
    Stride { start: usize, len: usize, step: isize },
    /// `len` positions from `start` along an axis of `size` items with `stride`, at least one of them outside the axis, which reads by `edge`: a fill, or a position inside the axis. [`Steps::padded`] builds one, and gives a `Stride` for positions that all lie inside.
    Padded { start: i128, len: usize, size: usize, stride: usize, edge: Edge },
    /// All `len` positions of an axis with `stride`, from position `shift`, wrapping round to the start. [`Steps::rotated`] builds one, and gives a `Stride` for a whole turn.
    Rotated { shift: usize, len: usize, stride: usize },
    /// Explicit positions, as a [`Table`] holds them.
    Table(Table),
}

/// Integer index values that a table reads as positions along an axis, converting each one as it reads it, so that the values aren't first copied into a list of offsets. [`Table::indices`] takes them.
pub trait IndexSource: std::fmt::Debug + Send + Sync {
    /// The index values, as Booleans or integers.
    fn view(&self) -> View<'_>;
}

/// Explicit offsets along one axis, with [`Steps::FILL`] where a fill goes: a list of offsets, or index values that each give one. `start` and `len` choose the offsets in use, so that slicing a table copies none of them.
#[derive(Clone, Debug)]
pub struct Table { all: Positions, start: usize, len: usize }

/// Where a table's offsets come from: a list, or index values along an axis of `size` positions, each counted back from the end where it's negative, and multiplied by `scale`. `even` holds the first position and the step between positions, where the values give evenly spaced positions.
#[derive(Clone, Debug)]
enum Positions { Listed(Arc<Vec<usize>>), Indices { values: Arc<dyn IndexSource>, size: usize, scale: usize, even: Option<(usize, isize)> } }

/// The most offsets that [`Table::chunks`] converts from index values at a time, into a buffer on the stack.
const TABLE_CHUNK: usize = 256;

/// `$body` on the index values `$view`, as a slice `$x` of Booleans or integers.
macro_rules! with_index_values {
    ($view:expr, |$x:ident| $body:expr) => {
        match $view {
            View::Booleans($x) => $body,
            View::Integers(ix) => with_ints!(ix, |$x| $body),
            _ => unreachable!("index values are Booleans or integers"),
        }
    };
}
pub(crate) use with_index_values;

impl Table {
    /// A table of `offsets`.
    pub fn new(offsets: Vec<usize>) -> Self { Self { start: 0, len: offsets.len(), all: Positions::Listed(Arc::new(offsets)) } }
    /// A table that reads `values` as positions along an axis of `size` positions. A negative value counts back from the end. Every value must lie within the axis, as [`structure::index_table`](crate::structure::index_table) checks. `even` gives the first position and the step between positions, where they're evenly spaced.
    pub fn indices(values: Arc<dyn IndexSource>, size: usize, even: Option<(usize, isize)>) -> Self {
        let len = values.view().len();
        Self { start: 0, len, all: Positions::Indices { values, size, scale: 1, even } }
    }
    /// The number of offsets.
    #[inline]
    pub fn len(&self) -> usize { self.len }
    /// The listed offsets in use, or `None` where the table converts index values as it reads them.
    #[inline]
    pub fn listed(&self) -> Option<&[usize]> { match &self.all { Positions::Listed(o) => Some(&o[self.start..self.start + self.len]), Positions::Indices { .. } => None } }
    /// Offset `j`, or `None` for a fill.
    #[inline]
    pub fn get(&self, j: usize) -> Option<usize> {
        match &self.all {
            Positions::Listed(o) => Some(o[self.start + j]).filter(|&o| o != Steps::FILL),
            Positions::Indices { values, size, scale, .. } => Some(index_at(values.view(), self.start + j, *size) * scale),
        }
    }
    /// Calls `each` with the position of each run of offsets and the run, in order: the listed offsets as one run, or index values converted into runs of up to `TABLE_CHUNK` offsets.
    #[inline]
    pub fn chunks(&self, mut each: impl FnMut(usize, &[usize])) { self.runs_while(&mut |j, run| { each(j, run); true }); }
    /// Whether `f` holds for every offset, `None` for a fill. It stops at the first offset that fails, and converts no index values after it.
    #[inline]
    pub fn all(&self, mut f: impl FnMut(Option<usize>) -> bool) -> bool { self.runs_while(&mut |_, run| run.iter().all(|&o| f((o != Steps::FILL).then_some(o)))) }
    /// [`Table::chunks`], stopping where `each` gives `false`. It gives whether it went through every run.
    fn runs_while(&self, each: &mut dyn FnMut(usize, &[usize]) -> bool) -> bool {
        match &self.all {
            Positions::Listed(o) => each(0, &o[self.start..self.start + self.len]),
            Positions::Indices { values, size, scale, .. } => converted(values.view(), self.start, self.len, *size, *scale, each),
        }
    }
    /// Offsets `start..start + len` of this table.
    pub fn slice(&self, start: usize, len: usize) -> Self { Self { start: self.start + start, len, ..self.clone() } }
    /// This table with its offsets `k` times as far apart. A plan scales a table of listed offsets while it owns them alone, so this multiplies them in place. Index values keep their scale beside them.
    pub fn scaled(mut self, k: usize) -> Self {
        if k == 1 { return self; }
        match &mut self.all {
            Positions::Listed(o) => for o in &mut Arc::make_mut(o)[self.start..self.start + self.len] { if *o != Steps::FILL { *o *= k } },
            Positions::Indices { scale, .. } => *scale *= k,
        }
        self
    }
    /// Whether some offset is a fill. Index values give none.
    pub fn fills(&self) -> bool { self.listed().is_some_and(|o| o.contains(&Steps::FILL)) }
    /// The bytes that the table's listed offsets take. Index values belong to the array that holds them.
    fn bytes(&self) -> usize { self.listed().map_or(0, |o| o.len() * size_of::<usize>()) }
    /// The scale and axis length of a table that reads index values, so that every offset is a position below the length, times the scale. `None` for listed offsets, which a caller reads to learn their span.
    fn span(&self) -> Option<(usize, usize)> { match self.all { Positions::Indices { size, scale, .. } => Some((scale, size)), Positions::Listed(_) => None } }
    /// Writes each offset plus `base` into `out`, in order, and a fill as [`Steps::FILL`]. Index values convert straight into `out`.
    fn write_offsets(&self, base: usize, out: &mut [MaybeUninit<usize>]) {
        fn positions<I: Copy + Into<i64>>(values: &[I], size: usize, scale: usize, base: usize, out: &mut [MaybeUninit<usize>]) {
            for (slot, &i) in out.iter_mut().zip(values) { slot.write(base + crate::structure::counted_back(i.into(), size) * scale); }
        }
        let range = self.start..self.start + self.len;
        match &self.all {
            Positions::Listed(o) => for (slot, &o) in out.iter_mut().zip(&o[range]) { slot.write(if o == Steps::FILL { o } else { base + o }); },
            Positions::Indices { values, size, scale, .. } => with_index_values!(values.view(), |x| positions(&x[range], *size, *scale, base, out)),
        }
    }
    /// Appends the items that this table reads from `s`, taking each straight through its index value as [`take`] does, where the table reads index values one item apart along the first `size` items of `s`. Gives whether it did. Reading each value inside the gather loop overlaps those reads with the gather's own.
    fn take_into<T: Copy>(&self, d: &mut Vec<T>, s: &[T]) -> bool {
        let Positions::Indices { ref values, size, scale: 1, .. } = self.all else { return false };
        let Some(items) = s.get(..size) else { return false };
        let range = self.start..self.start + self.len;
        with_index_values!(values.view(), |x| take(d, items, &x[range], |v| v));
        true
    }
    /// This table as one stride, where it reads index values that [`structure::index_table`](crate::structure::index_table) found evenly spaced. `None` for any other table.
    fn stride(&self) -> Option<Steps> {
        let Positions::Indices { scale, even: Some((first, step)), .. } = self.all else { return None };
        Some(Steps::Stride { start: first.wrapping_add_signed(step * self.start as isize) * scale, len: self.len, step: step * scale as isize })
    }
}

impl PartialEq for Table { fn eq(&self, other: &Self) -> bool { self.len == other.len && (0..self.len).all(|j| self.get(j) == other.get(j)) } }
impl Eq for Table {}

/// Position `j` of the index values `view` along an axis of `size` positions.
fn index_at(view: View<'_>, j: usize, size: usize) -> usize {
    with_index_values!(view, |x| crate::structure::counted_back(x[j].into(), size))
}

/// Calls `each` with runs of the `len` positions from `start` that the index values `view` give along an axis of `size` positions, each multiplied by `scale`, and converted into a buffer on the stack. It stops where `each` gives `false`, and gives whether it went through every run.
fn converted(view: View<'_>, start: usize, len: usize, size: usize, scale: usize, each: &mut dyn FnMut(usize, &[usize]) -> bool) -> bool {
    fn runs<I: Copy + Into<i64>>(values: &[I], size: usize, scale: usize, each: &mut dyn FnMut(usize, &[usize]) -> bool) -> bool {
        let mut run = [0; TABLE_CHUNK];
        values.chunks(TABLE_CHUNK).enumerate().all(|(c, chunk)| {
            for (o, &i) in run.iter_mut().zip(chunk) { *o = crate::structure::counted_back(i.into(), size) * scale; }
            each(c * TABLE_CHUNK, &run[..chunk.len()])
        })
    }
    with_index_values!(view, |x| runs(&x[start..start + len], size, scale, each))
}

impl Steps {
    /// The table entry that marks a fill.
    pub const FILL: usize = usize::MAX;
    /// Every position of an axis of `len` items with `stride`, in order.
    pub fn along(len: usize, stride: usize) -> Self { Self::Stride { start: 0, len, step: stride as isize } }
    /// One entry for each of `lens`, reading an array of those lengths in row-major order from position `start`, with every stride multiplied by `scale`.
    pub fn row_major(lens: &[usize], scale: usize, start: usize) -> impl Iterator<Item = Self> + '_ {
        lens.iter().zip(crate::layout::strides(lens)).enumerate().map(move |(j, (&len, stride))| Self::Stride { start: if j == 0 { start } else { 0 }, len, step: (scale * stride) as isize })
    }
    /// A table of `offsets`, with [`Steps::FILL`] where a fill goes.
    pub fn table(offsets: Vec<usize>) -> Self { Self::Table(Table::new(offsets)) }
    /// This entry with its offsets `k` times as far apart, as where each position it reads is a cell of `k` items.
    pub fn scaled(self, k: usize) -> Self {
        match self {
            Self::Stride { start, len, step } => Self::Stride { start: start * k, len, step: step * k as isize },
            Self::Padded { start, len, size, stride, edge } => Self::Padded { start, len, size, stride: stride * k, edge },
            Self::Rotated { shift, len, stride } => Self::Rotated { shift, len, stride: stride * k },
            Self::Table(table) => Self::Table(table.scaled(k)),
        }
    }
    /// `len` positions from `start` along an axis of `size` items with `stride`, read past the axis's ends by `edge`: a `Padded` entry where a position falls outside the axis, and otherwise the `Stride` that reads the same offsets.
    pub fn padded(start: i128, len: usize, size: usize, stride: usize, edge: Edge) -> Self {
        if start >= 0 && start + len as i128 <= size as i128 { Self::Stride { start: start as usize * stride, len, step: stride as isize } } else { Self::Padded { start, len, size, stride, edge } }
    }
    /// `padded` with a fill past the axis's ends.
    pub fn clipped(start: i128, len: usize, size: usize, stride: usize) -> Self { Self::padded(start, len, size, stride, Edge::Fill) }
    /// All `len` positions of an axis with `stride`, from position `shift`, wrapping round: a `Rotated` entry, or the `Stride` that reads the same offsets where the shift is a whole turn.
    pub fn rotated(shift: usize, len: usize, stride: usize) -> Self {
        if len < 2 || shift % len == 0 { Self::along(len, stride) } else { Self::Rotated { shift: shift % len, len, stride } }
    }
    /// The number of offsets.
    #[inline]
    pub fn len(&self) -> usize {
        match self { Self::Stride { len, .. } | Self::Padded { len, .. } | Self::Rotated { len, .. } => *len, Self::Table(t) => t.len() }
    }
    /// The bytes this entry takes: itself, and a table's listed offsets.
    pub fn footprint(&self) -> usize { size_of::<Self>() + if let Self::Table(t) = self { t.bytes() } else { 0 } }
    /// Offset `j`, or `None` for a fill.
    #[inline]
    pub fn get(&self, j: usize) -> Option<usize> {
        match *self {
            Self::Stride { start, step, .. } => Some(start.wrapping_add_signed(step * j as isize)),
            Self::Padded { start, size, stride, edge, .. } => edged(edge, start + j as i128, size).map(|p| p * stride),
            Self::Rotated { shift, len, stride } => Some((shift + j) % len * stride),
            Self::Table(ref t) => t.get(j),
        }
    }
    /// Every offset, in order, `None` for a fill.
    #[inline]
    pub fn offsets(&self) -> impl ExactSizeIterator<Item = Option<usize>> + '_ { (0..self.len()).map(|j| self.get(j)) }
    /// Calls `each` with each position along this axis and its offset, or `None` where it gives a fill. It matches the kind of step once, before its loop. A loop through [`Steps::offsets`] or [`Steps::get`] matches it at every position, and makes a call there wherever LLVM keeps `get` out of line, as it can despite `#[inline]`.
    #[inline(always)]
    pub fn for_each(&self, mut each: impl FnMut(usize, Option<usize>)) {
        match *self {
            Self::Stride { start, len, step } => for r in 0..len { each(r, Some(start.wrapping_add_signed(step * r as isize))) },
            Self::Table(ref t) => t.chunks(|j0, run| for (r, &o) in run.iter().enumerate() { each(j0 + r, (o != Self::FILL).then_some(o)) }),
            _ => for (r, o) in self.offsets().enumerate() { each(r, o) },
        }
    }
    /// Offsets `start..start + len`, as at most two entries. A table's piece holding every offset is the table itself, borrowed.
    fn pieces(&self, start: usize, len: usize) -> impl Iterator<Item = Cow<'_, Self>> {
        let pieces = if len == 0 { [None, None] } else {
            match *self {
                Self::Stride { start: s, step, .. } => [Some(Cow::Owned(Self::Stride { start: s.wrapping_add_signed(step * start as isize), len, step })), None],
                Self::Padded { start: s, size, stride, edge, .. } => [Some(Cow::Owned(Self::padded(s + start as i128, len, size, stride, edge))), None],
                Self::Rotated { shift, len: n, stride } => {
                    let first = (shift + start) % n;
                    let head = len.min(n - first);
                    [Some(Cow::Owned(Self::Stride { start: first * stride, len: head, step: stride as isize })), (head < len).then(|| Cow::Owned(Self::Stride { start: 0, len: len - head, step: stride as isize }))]
                }
                Self::Table(ref t) if start == 0 && len == t.len() => [Some(Cow::Borrowed(self)), None],
                Self::Table(ref t) => [Some(Cow::Owned(Self::Table(t.slice(start, len)))), None],
            }
        };
        pieces.into_iter().flatten()
    }
    /// The offset of every position that `steps` reach from `base`, in the order that `walk` visits them, with [`Steps::FILL`] where a fill goes. [`Steps::reach`] gives one of them.
    pub fn combined(base: usize, steps: &[impl std::borrow::Borrow<Self>]) -> Vec<usize> {
        fn into<B: std::borrow::Borrow<Steps>>(out: &mut Vec<usize>, base: usize, steps: &[B]) {
            let Some((axis, rest)) = steps.split_first() else { return out.push(base) };
            let axis = axis.borrow();
            if base == Steps::FILL { return out.extend(std::iter::repeat_n(Steps::FILL, axis.len() * rest.iter().map(|s| s.borrow().len()).product::<usize>())); }
            if !rest.is_empty() { return axis.for_each(|_, o| into(out, o.map_or(Steps::FILL, |o| base + o), rest)); }
            match axis {
                Steps::Table(table) => crate::slice::appended(out, table.len(), |slots| table.write_offsets(base, slots)),
                _ => axis.for_each(|_, o| out.push(o.map_or(Steps::FILL, |o| base + o))),
            }
        }
        let mut out = Vec::with_capacity(steps.iter().map(|s| s.borrow().len()).product());
        into(&mut out, base, steps);
        out
    }
    /// `steps` without axes of one position, whose offset moves into `base`, and with neighbouring runs merged into one longer run. A table whose offsets step evenly merges as the stride it is (`Steps::even`), and becomes that stride where it's the innermost axis, which `walk_into` and `scatter_into` then reach without the table. An outer table that merges with nothing stays a table, for their routes that take a table of starts.
    fn simplify<B: std::borrow::Borrow<Self>>(mut base: usize, steps: &[B]) -> (usize, Vec<Cow<'_, Self>>) {
        let mut kept: Vec<Cow<Self>> = Vec::with_capacity(steps.len());
        for s in steps {
            let s = s.borrow();
            if let Some(o) = if s.len() == 1 { s.get(0) } else { None } { base += o; continue; }
            kept.push(Cow::Borrowed(s));
            while let [.., outer, inner] = kept.as_slice() {
                let Some(merged) = Self::merged(outer, inner) else { break };
                kept.truncate(kept.len() - 2);
                kept.push(Cow::Owned(merged));
            }
        }
        let innermost = kept.last().and_then(|s| if let Self::Table(_) = **s { s.even() } else { None });
        if let (Some(stride), Some(last)) = (innermost, kept.last_mut()) { *last = Cow::Owned(stride); }
        (base, kept)
    }
    /// `outer` and `inner` as one run, where `inner` reads adjacent items and each step of `outer` moves past one run of `inner`, as `simplify` merges neighbours. `None` where they don't form one run.
    fn merged(outer: &Self, inner: &Self) -> Option<Self> {
        let (Some(Self::Stride { start: a, len: m, step }), Some(Self::Stride { start: b, len: n, step: 1 })) = (outer.even(), inner.even()) else { return None };
        (step == n as isize).then_some(Self::Stride { start: a + b, len: m * n, step: 1 })
    }
    /// The box whose first axis is `piece`, a run of positions along an axis, and whose other axes are `rest`, already simplified, as `simplify` would give it: its base, first axis and other axes. A piece of one position moves its offset into the base, a piece that steps over whole runs of `rest`'s first axis merges with it, and an evenly stepping table with no axes after it becomes a stride. It allocates nothing, so a range's walk cuts its boxes cheaply.
    fn boxed<'a>(base: usize, piece: Cow<'a, Self>, rest: &'a [Cow<'a, Self>]) -> (usize, Cow<'a, Self>, &'a [Cow<'a, Self>]) {
        let (base, piece, rest) = match (piece.len() == 1).then(|| piece.get(0)).flatten().zip(rest.split_first()) {
            Some((o, (first, after))) => (base + o, first.clone(), after),
            None => (base, piece, rest),
        };
        match rest.split_first().and_then(|(inner, after)| Some((Self::merged(&piece, inner)?, after))) {
            Some((merged, after)) => (base, Cow::Owned(merged), after),
            None if rest.is_empty() && matches!(*piece, Self::Table(_)) => (base, piece.even().map_or(piece, Cow::Owned), rest),
            None => (base, piece, rest),
        }
    }
    /// This axis as one stride: itself where it is a stride, and a table of at least two offsets where none is a fill and each lies the same distance on from the one before. `None` for any other axis.
    fn even(&self) -> Option<Self> {
        match *self {
            Self::Stride { start, len, step } => Some(Self::Stride { start, len, step }),
            Self::Table(ref t) => {
                let Some(o) = t.listed() else { return t.stride() };
                let &[first, second, ..] = o else { return None };
                let step = second.wrapping_sub(first) as isize;
                let even = first != Self::FILL && o.windows(2).all(|p| p[1] != Self::FILL && p[1].wrapping_sub(p[0]) as isize == step);
                even.then_some(Self::Stride { start: first, len: o.len(), step })
            }
            _ => None,
        }
    }
    /// The offset that `steps`, one entry for each axis, outermost first, reach from `base` for item `i` of their frame, or `None` for a fill.
    #[inline]
    pub fn reach(base: usize, steps: &[impl std::borrow::Borrow<Self>], mut i: usize) -> Option<usize> {
        let mut offset = base;
        for s in steps.iter().rev() {
            let s = s.borrow();
            offset += s.get(i % s.len())?;
            i /= s.len();
        }
        Some(offset)
    }
    /// The first source offset and stride of items `start..start + n`, where they form one regular run. Singleton axes disappear and neighbouring strides can form one run. Padded and rotated reads must stay within their valid, unwrapped part.
    #[inline]
    pub fn run(base: usize, steps: &[Self], start: usize, n: usize) -> Option<(usize, isize)> {
        if n == 0 { return None; }
        let (mut width, mut stride) = (1usize, None);
        for axis in steps.iter().rev() {
            let len = axis.len();
            if len == 0 { return None; }
            if len == 1 { axis.get(0)?; continue; }
            let (i, end) = ((start / width) % len, ((start + n - 1) / width) % len);
            let fits = start % (width * len) + n <= width * len;
            let step = match *axis {
                Self::Stride { step, .. } => step,
                Self::Padded { start: s, size, stride, .. } if fits && s + i as i128 >= 0 && s + (end + 1) as i128 <= size as i128 => stride as isize,
                Self::Rotated { shift, stride, .. } if fits && (shift + i) % len + end - i < len => stride as isize,
                _ => return None,
            };
            if let Some(s) = stride { if step != s * width as isize { return None; } } else { stride = Some(step); }
            width *= len;
            if fits { return Some((Self::reach(base, steps, start)?, stride.unwrap())); }
        }
        (n == 1).then(|| (Self::reach(base, steps, start).expect("singleton run has no fill"), 1))
    }
    /// How many consecutive items of the frame read adjacent offsets: the product of the lengths of the innermost axes, outward, while each axis steps on from the ones inside it. Clip and wrap edges don't count, because only the runs that cross them copy. A run of `n` items reads in place through `run` only where it lies inside one such stretch, so a frame whose stretches are shorter than its blocks copies every block.
    pub fn contiguous(steps: &[Self]) -> usize {
        let mut width = 1;
        for axis in steps.iter().rev() {
            if axis.len() == 1 { continue; }
            let step = match *axis {
                Self::Stride { step, .. } => step,
                Self::Padded { stride, edge: Edge::Fill, .. } | Self::Rotated { stride, .. } => stride as isize,
                Self::Padded { .. } | Self::Table(_) => return width,
            };
            if step != width as isize { return width; }
            width *= axis.len();
        }
        width
    }
    /// Whether this entry reads items that aren't adjacent: a step or stride of more than 1, as a transpose's innermost axis has.
    pub fn skips(&self) -> bool {
        match *self { Self::Stride { step, .. } => step.unsigned_abs() > 1, Self::Padded { stride, .. } | Self::Rotated { stride, .. } => stride > 1, Self::Table(_) => false }
    }
    /// Whether this entry gives a fill at some position: a `Padded` entry does where its edge mode is fill, and a table does where it holds [`Steps::FILL`].
    pub fn fills(&self) -> bool { match self { Self::Padded { edge, .. } => *edge == Edge::Fill, Self::Table(t) => t.fills(), _ => false } }
    /// This entry's first offset and step, where it reads its whole axis as one regular run with no fill: a `Stride`, or a `Rotated` entry that doesn't rotate.
    pub fn affine(&self) -> Option<(usize, isize)> {
        match *self {
            Self::Stride { start, step, .. } => Some((start, step)),
            _ => None,
        }
    }
    /// Whether a walk through `tables` reads offsets 0, 1, 2 and so on, in order. A copy through such tables of a source with as many items as the walk reads is the source unchanged. Only `Stride` entries that start at 0 qualify.
    pub fn in_order(tables: &[Self]) -> bool {
        let mut stride = 1;
        tables.iter().rev().all(|s| match *s {
            Self::Stride { start: 0, len, step } => { let ordered = len <= 1 || step == stride as isize; stride *= len; ordered }
            _ => false,
        })
    }
    /// Whether this entry repeats an offset or gives a fill. Applying an element-wise function to the items it reads, in place of reading the function's results, would then compute an item more than once, or apply the function to a fill.
    pub fn spreads(&self) -> bool {
        match *self {
            Self::Stride { len, step, .. } => step == 0 && len > 1,
            Self::Padded { .. } => true,
            Self::Rotated { .. } => false,
            _ => { let mut seen = std::collections::HashSet::new(); !self.offsets().all(|o| o.is_some_and(|o| seen.insert(o))) }
        }
    }
    /// `tables`, a walk's entries for a result of `shape`, as one entry for each axis of `shape`, with the offset of each entry of one item moved into the base. An entry over several axes splits into one `Stride` for each. `None` where an entry that isn't a `Stride` covers several axes.
    pub fn per_axis(tables: &[Self], shape: &[usize]) -> Option<(usize, Vec<Self>)> {
        let (mut base, mut steps, mut a) = (0, Vec::with_capacity(shape.len()), 0);
        let ones = |steps: &mut Vec<Self>, a: &mut usize| while *a < shape.len() && shape[*a] == 1 {
            steps.push(Self::Stride { start: 0, len: 1, step: 0 });
            *a += 1;
        };
        for t in tables {
            if t.len() == 1 { base += t.get(0)?; continue; }
            ones(&mut steps, &mut a);
            let (first, mut n) = (a, 1);
            while a < shape.len() && n < t.len() { n *= shape[a]; a += 1; }
            if n != t.len() { return None; }
            match (&shape[first..a], t) {
                ([_], _) => steps.push(t.clone()),
                (group, &Self::Stride { start, step, .. }) => {
                    let mut width = n;
                    for (i, &len) in group.iter().enumerate() {
                        width /= len;
                        steps.push(Self::Stride { start: if i == 0 { start } else { 0 }, len, step: step * width as isize });
                    }
                }
                _ => return None,
            }
        }
        ones(&mut steps, &mut a);
        (a == shape.len()).then_some((base, steps))
    }
    /// The tables of a walk from `base` whose entries are `steps`: `steps`, after an entry of one item at `base` where `base` isn't 0. `per_axis` splits such tables back into `base` and `steps`.
    pub fn with_base(base: usize, steps: Vec<Self>) -> Vec<Self> {
        if base == 0 { steps } else { std::iter::once(Self::Stride { start: base, len: 1, step: 0 }).chain(steps).collect() }
    }
    /// `tables`, a walk's entries for a result of `shape` from a source of shape `source`, as one entry for each axis of `shape`, with the source axis that each entry walks. An entry whose offsets are all 0, as for an axis of one item or a spread, walks no axis. The first entry along each walked axis holds that axis's position in its start. The base holds only positions on axes that no entry walks. `None` where an entry doesn't walk one source axis, where a walked axis's position can't move into the start of a `Stride`, as for a `Padded` or `Rotated` entry over part of an axis, and where the source is empty.
    pub fn walked(tables: &[Self], shape: &[usize], source: &[usize]) -> Option<(usize, Vec<Self>, Vec<Option<usize>>)> {
        let (base, mut steps) = Self::per_axis(tables, shape)?;
        let strides = crate::layout::strides(source);
        if base >= source.iter().product::<usize>() { return None; }
        let axes: Vec<Option<usize>> = steps.iter().map(|e| e.axis_of(source, &strides)).collect::<Option<_>>()?;
        let mut rest = 0;
        for (a, p) in crate::layout::coordinates(base, source).filter(|&(_, p)| p > 0) {
            let Some(r) = axes.iter().position(|&b| b == Some(a)) else { rest += p * strides[a]; continue };
            let Self::Stride { start, .. } = &mut steps[r] else { return None };
            *start += p * strides[a];
            if steps[r].axis_of(source, &strides) != Some(Some(a)) { return None; }
        }
        Some((rest, steps, axes))
    }
    /// `outer`'s reads of an array of `shape`, composed with `inner`, which maps the positions along each axis of `shape` to offsets from `base`: one entry for each of `outer`'s, with the new base. Walking the result reads what walking `inner` and then `outer` reads. `None` where `outer`'s entries don't each walk one axis of `shape` ([`Steps::walked`]), where two entries walk one axis that `inner` doesn't map by a `Stride`, or where an entry reads part of an axis that `inner` rotates or clips. Callers then force the copy that `inner` describes, in one walk.
    pub fn through(outer: &[Self], shape: &[usize], mut base: usize, inner: &[Self]) -> Option<(usize, Vec<Self>)> {
        let (at, outer, axes) = Self::walked(outer, &outer.iter().map(Self::len).collect::<Vec<_>>(), shape)?;
        let strides = crate::layout::strides(shape);
        let mut walks = vec![0usize; shape.len()];
        for &a in axes.iter().flatten() { walks[a] += 1; }
        // Checked before any `read_through`, so that a composition that fails builds no table.
        if walks.iter().zip(inner).any(|(&w, r)| w > 1 && !matches!(r, Self::Stride { .. })) { return None; }
        let mut first = vec![true; shape.len()];
        let mut steps = Vec::with_capacity(outer.len());
        for (e, &axis) in outer.iter().zip(&axes) {
            steps.push(match axis {
                None => Self::Stride { start: 0, len: e.len(), step: 0 },
                Some(a) => e.read_through(strides[a], &inner[a], std::mem::replace(&mut first[a], false), &mut base)?,
            });
        }
        for (a, p) in crate::layout::coordinates(at, shape) { if walks[a] == 0 { base += inner[a].get(p)?; } }
        Some((base, steps))
    }
    /// Whether two of `outer`'s entries walk one axis of `shape`, as overlapping windows do. Applying an element-wise function to the items they read, in place of reading its results, would then compute each item once for every window that holds it.
    pub fn rereads(outer: &[Self], shape: &[usize]) -> bool {
        let Some((_, _, axes)) = Self::walked(outer, &outer.iter().map(Self::len).collect::<Vec<_>>(), shape) else { return false };
        let mut walked = vec![false; shape.len()];
        axes.iter().flatten().any(|&a| std::mem::replace(&mut walked[a], true))
    }
    /// For a copy of `shape` that `tables` read from a source of shape `source`: an order to move the copy's axes into, and the reads of the moved copy whose sum sends each item back to the source position it was read from. Each read gives the source's shape, through one table for each source axis after a table of length 1 for any offset, with a fill where it reads nothing. A copy that reads each item at most once needs one read, and keeps its order. Overlapping windows that move by 1 along an axis need one read for each position in a window, each of which reads the frames back: a correlation. The order moves each window's axis first, so that each read walks the moved copy's last axis forwards. `None` for any other copy, such as a spread, a selection that repeats items or windows that move further, and where the copy or its source is empty.
    pub fn transposed(tables: &[Self], shape: &[usize], source: &[usize]) -> Option<(Vec<usize>, Vec<Vec<Self>>)> {
        if shape.contains(&0) || source.contains(&0) { return None; }
        let (base, steps, along) = Self::walked(tables, shape, source)?;
        let from = crate::layout::strides(source);
        // The axes of the copy that walk each source axis.
        let mut walks: Vec<Vec<usize>> = vec![vec![]; source.len()];
        for (r, &a) in along.iter().enumerate() {
            match a {
                Some(a) => walks[a].push(r),
                None if steps[r].len() > 1 => return None,
                None => {}
            }
        }
        // Of each pair of axes that walk one source axis, the shorter plays the window, which gives fewer reads. Moving the windows' axes first lets each read walk the copy's last axis forwards.
        let cells: Vec<usize> = walks.iter().filter_map(|w| match *w.as_slice() { [f, c] => Some(if steps[f].len() >= steps[c].len() { c } else { f }), _ => None }).collect();
        let order: Vec<usize> = cells.iter().copied().chain((0..shape.len()).filter(|r| !cells.contains(r))).collect();
        let moved = crate::layout::strides(&order.iter().map(|&r| shape[r]).collect::<Vec<_>>());
        let mut to = vec![0; shape.len()];
        for (&r, &stride) in order.iter().zip(&moved) { to[r] = stride; }
        // Each source axis gives its entries, each with an offset into the moved copy. A source axis that no axis of the copy walks gives the copy one position, which `base` holds.
        let axes = walks.iter().enumerate().map(|(a, w)| match *w.as_slice() {
            [] => Some(vec![(0, Self::clipped(-((base / from[a] % source[a]) as i128), source[a], 1, 0))]),
            [r] => Some(vec![(0, steps[r].inverted(from[a], source[a], to[r])?)]),
            [f, c] => {
                let (f, c) = if cells.contains(&c) { (f, c) } else { (c, f) };
                steps[f].correlated(&steps[c], from[a], source[a], to[f], to[c])
            }
            _ => None,
        }).collect::<Option<Vec<_>>>()?;
        let terms = axes.iter().fold(vec![(0, vec![])], |terms, entries| {
            terms.iter().flat_map(|(offset, t)| entries.iter().map(move |(o, e)| (offset + o, [t.as_slice(), std::slice::from_ref(e)].concat()))).collect()
        });
        Some((order, terms.into_iter().map(|(offset, t)| Self::with_base(offset, t)).collect()))
    }
    /// The entries that read back the windows that this entry's frames and `cell`'s positions read along an axis of `n` items whose stride is `unit`, one for each position in a window, each with that position's offset in the copy. `frames` and `cells` are the copy's strides along the two entries. Each entry reads the frames at a negated offset from the windows' first position. `None` where `windows_range` gives no positions for the two entries.
    fn correlated(&self, cell: &Self, unit: usize, n: usize, frames: usize, cells: usize) -> Option<Vec<(usize, Self)>> {
        let first = Self::windows_range([self, cell], unit, n)?.start as i128;
        Some((0..cell.len()).map(|c| (c * cells, Self::clipped(-first - c as i128, n, self.len(), frames))).collect())
    }
    /// What `entries` read along a source axis of `n` items whose stride is `unit`, where they walk windows, such as a copy's frames and the positions in each window. Each entry must be a `Stride` that steps forward by a whole number of positions. Gives the position the entries' starts add up to, where the windows start, and each entry's length and move in positions. `None` for any other entries, and where the windows don't lie inside the axis.
    pub fn windows_read<'a>(entries: impl IntoIterator<Item = &'a Self>, unit: usize, n: usize) -> Option<(usize, Vec<[usize; 2]>)> {
        if unit == 0 { return None; }
        let (mut first, mut last, mut read) = (0, 0, Vec::new());
        for e in entries {
            let &Self::Stride { start, len, step } = e else { return None };
            if step <= 0 || step as usize % unit != 0 || start % unit != 0 || len == 0 { return None; }
            let moves = step as usize / unit;
            (first, last) = (first + start / unit, last + (len - 1) * moves);
            read.push([len, moves]);
        }
        (first + last < n).then_some((first, read))
    }
    /// The positions that `entries` read where they walk windows that move by 1 (`windows_read`), such as a copy's frames and the positions in each window. `None` where any entry moves by more.
    pub fn windows_range<'a>(entries: impl IntoIterator<Item = &'a Self>, unit: usize, n: usize) -> Option<std::ops::Range<usize>> {
        let (first, read) = Self::windows_read(entries, unit, n)?;
        read.iter().all(|&[_, moves]| moves == 1).then(|| first..first + 1 + read.iter().map(|&[len, _]| len - 1).sum::<usize>())
    }
    /// The windows of a copy of `shape` that `tables` read from `source`, laid out as frame axes, then window axes, then the source's other axes whole, as `structure::Windows::plan` lays them out without padding. Along each of the source's first axes, the frames start at 0 and move by a whole number of positions, and each frame is a window of consecutive positions. Gives each such axis's number of frames, window size and move. `None` for any other copy.
    pub fn windows_of(tables: &[Self], shape: &[usize], source: &[usize]) -> Option<Vec<[usize; 3]>> {
        let (base, steps, along) = Self::walked(tables, shape, source)?;
        let s = shape.len().checked_sub(source.len())?;
        if base != 0 || s == 0 || 2 * s > shape.len() || shape[2 * s..] != source[s..] { return None; }
        let from = crate::layout::strides(source);
        if (s..source.len()).any(|a| !matches!(steps[s + a], Self::Stride { start: 0, len, step } if len == source[a] && step == from[a] as isize)) { return None; }
        (0..s).map(|a| {
            if along[a] != Some(a) || along[s + a] != Some(a) { return None; }
            match Self::windows_read([&steps[a], &steps[s + a]], from[a], source[a])? {
                (0, read) => match read[..] { [[n, moves], [w, 1]] => Some([n, w, moves]), _ => None },
                _ => None,
            }
        }).collect()
    }
    /// The entry that reads this entry's items back, for an entry along an axis of `n` items whose stride is `unit`, from a copy whose stride along this entry's own axis is `stride`. `None` where the entry reads a position twice.
    fn inverted(&self, unit: usize, n: usize, stride: usize) -> Option<Self> {
        let (m, forwards) = (self.len(), unit as isize);
        match *self {
            Self::Stride { start: 0, step, .. } if step == forwards && m == n => return Some(Self::along(n, stride)),
            Self::Stride { start, step, .. } if step == -forwards && start == (n - 1) * unit && m == n => {
                return Some(Self::Stride { start: (n - 1) * stride, len: n, step: -(stride as isize) });
            }
            Self::Stride { start, step, .. } if step == forwards => return Some(Self::clipped(-((start / unit) as i128), n, m, stride)),
            Self::Padded { start, size, edge: Edge::Fill, .. } if size == n => return Some(Self::clipped(-start, n, m, stride)),
            Self::Rotated { shift, len, .. } if len == n => return Some(Self::rotated(n - shift % n, n, stride)),
            _ => {}
        }
        let mut back = vec![Self::FILL; n];
        let mut unique = true;
        self.for_each(|j, o| if unique && let Some(o) = o { unique = std::mem::replace(&mut back[o / unit], j * stride) == Self::FILL });
        if !unique { return None; }
        Some(Self::table(back))
    }
    /// This entry's positions along an axis whose stride is `unit`, read through `r`. The first entry along an axis takes `r`'s offset at position 0, in its own start or in `base`. `None` where neither this entry nor `r` is a table and the two don't combine into one regular entry.
    fn read_through(&self, unit: usize, r: &Self, first: bool, base: &mut usize) -> Option<Self> {
        if first && matches!(*self, Self::Stride { start: 0, len, step } if step == unit as isize && len == r.len()) { return Some(r.clone()); }
        if let Self::Stride { start: r0, step: s, .. } = *r {
            let r0 = if first { r0 } else { 0 };
            match *self {
                Self::Stride { start, len, step } => {
                    let start = usize::try_from(r0 as isize + (start / unit) as isize * s).ok()?;
                    return Some(Self::Stride { start, len, step: step / unit as isize * s });
                }
                Self::Rotated { shift, len, .. } if s >= 0 => { *base += r0; return Some(Self::rotated(shift, len, s as usize)); }
                Self::Padded { start, len, size, edge, .. } if s >= 0 => { *base += r0; return Some(Self::padded(start, len, size, s as usize, edge)); }
                _ => {}
            }
        }
        (first && (matches!(self, Self::Table(_)) || matches!(r, Self::Table(_)))).then(|| {
            let mut table = Vec::with_capacity(self.len());
            self.for_each(|_, o| table.push(o.and_then(|o| r.get(o / unit)).unwrap_or(Self::FILL)));
            Self::table(table)
        })
    }
    /// The axis of `shape` whose stride divides each of this entry's offsets, each quotient within its length: `Some(None)` where every offset is 0, and `None` where no axis fits. `strides` are `shape`'s, row by row.
    fn axis_of(&self, shape: &[usize], strides: &[usize]) -> Option<Option<usize>> {
        let zero = match *self {
            Self::Stride { start, len, step } => len == 0 || start == 0 && (step == 0 || len == 1),
            Self::Rotated { stride, .. } | Self::Padded { stride, .. } => stride == 0,
            Self::Table(ref t) => t.span().map_or_else(|| t.all(|o| o == Some(0)), |(scale, size)| scale == 0 || size <= 1),
        };
        if zero { return Some(None); }
        let fits = |a: usize| {
            let (s, n) = (strides[a], shape[a]);
            let at = |o: usize| o % s == 0 && o / s < n;
            match *self {
                Self::Stride { start, len, step } => step.unsigned_abs() % s == 0 && at(start) && at(start.wrapping_add_signed(step * (len as isize - 1))),
                Self::Rotated { stride, len, .. } => stride == s && len <= n,
                Self::Padded { stride, size, .. } => stride == s && size <= n,
                Self::Table(ref t) => t.span().map_or_else(|| t.all(|o| o.is_none_or(at)), |(scale, size)| size <= 1 || scale % s == 0 && (size - 1) * scale / s < n),
            }
        };
        (0..shape.len()).find(|&a| shape[a] > 1 && fits(a)).map(Some)
    }
}

/// How [`walk`] and [`scatter_walk`] turn a source item into a result item. Every `Fn(S) -> D` converts item by item. [`Same`] keeps each item of a source and result of one type as it is.
pub trait Convert<S: Copy, D> {
    /// `x`, converted.
    fn one(&self, x: S) -> D;
    /// Appends the items of `s`, converted, in one contiguous pass.
    fn slice(&self, d: &mut Vec<D>, s: &[S]) {
        crate::slice::appended(d, s.len(), |slots| {
            let slots = &mut slots[..s.len()];
            for i in 0..s.len() { slots[i].write(self.one(s[i])); }
        })
    }
    /// The items of `s`, converted, in one contiguous pass.
    fn all(&self, s: &[S]) -> Vec<D> {
        let mut d = Vec::with_capacity(s.len());
        self.slice(&mut d, s);
        d
    }
    /// `d` as a buffer of source items, where the conversion keeps each item as it is.
    fn same<'a>(&self, _d: &'a mut Vec<D>) -> Option<&'a mut Vec<S>> { None }
    /// `s` as result items, where the conversion keeps each item as it is.
    fn borrowed<'a>(&self, _s: &'a [S]) -> Option<&'a [D]> { None }
}
impl<S: Copy, D, F: Fn(S) -> D> Convert<S, D> for F {
    #[inline]
    fn one(&self, x: S) -> D { self(x) }
}
/// A [`Convert`] for a source and result of one type, which keeps each item as it is. A walk or scatter with it moves items straight between its source and result, with no conversion pass.
pub struct Same;
impl<T: Copy> Convert<T, T> for Same {
    #[inline]
    fn one(&self, x: T) -> T { x }
    fn same<'a>(&self, d: &'a mut Vec<T>) -> Option<&'a mut Vec<T>> { Some(d) }
    fn borrowed<'a>(&self, s: &'a [T]) -> Option<&'a [T]> { Some(s) }
}

/// An item type that [`walk`] and [`scatter_walk`] move as a [`Unit`] of its size and alignment: each element type of compact buffers. Their fixed-width loops then compile once for each size of item, whatever the item's type or its conversion. A few conversions into `f64` write through fixed-width loops of their own instead ([`Item::cells_from`]).
///
/// # Safety
/// `Self` has no padding, so every byte of every value is initialized, and [`Item::Unit`] has the same size and alignment as `Self`. `units` reads items as units on these promises, and `units!` checks size and alignment at compile time.
pub unsafe trait Item: Copy {
    /// The unit that holds an item's bytes.
    type Unit: Unit;
    /// Writes `items` into `out`, each converted by `conv`: one cell at each start along a stride of `step` from `at`, with its items at the offsets that `rest` reaches. Gives whether a fixed-width loop that converts from `S` wrote them. Only `f64` has such loops, for the sources whose [`Item::cells_into_f64`] has one. Otherwise [`scatter_walk`] converts a block at a time.
    #[inline]
    fn cells_from<S: Item, C: Convert<S, Self>>(_out: &mut [Self], _items: &[S], _conv: &C, _at: usize, _step: isize, _rest: &[Cow<'_, Steps>]) -> bool { false }
    /// [`Item::cells_from`] from this type into `f64`.
    #[inline]
    fn cells_into_f64<C: Convert<Self, f64>>(_out: &mut [f64], _items: &[Self], _conv: &C, _at: usize, _step: isize, _rest: &[Cow<'_, Steps>]) -> bool { false }
}
/// A type through which walks move items: an unsigned integer, or an array of two, with the size and alignment of the item types that use it.
///
/// # Safety
/// Every bit pattern of the type's size is a value of it, so the bytes of any item read as a unit give a valid unit.
pub unsafe trait Unit: Copy + Send + Sync + 'static {}
macro_rules! units {
    ($($u:ty),+; $($t:ty: $tu:ty $(=> $body:ident)?),+) => {
        // SAFETY: each unit is an unsigned integer or an array of two, and every bit pattern is a value of each.
        $(unsafe impl Unit for $u {})+
        $(
            // SAFETY: each item type is a primitive number, `bool`, `char` or a complex number of two floats, none of which has padding. The assertion below checks size and alignment.
            unsafe impl Item for $t { type Unit = $tu; $(units!(@$body);)? }
            const _: () = assert!(size_of::<$t>() == size_of::<$tu>() && align_of::<$t>() == align_of::<$tu>());
        )+
    };
    // A source whose conversion into `f64` writes through a fixed-width loop of its own.
    (@into_f64) => {
        #[inline]
        fn cells_into_f64<C: Convert<Self, f64>>(out: &mut [f64], items: &[Self], conv: &C, at: usize, step: isize, rest: &[Cow<'_, Steps>]) -> bool { converted_cells(out, items, conv, at, step, rest) }
    };
    // `f64` asks each source whether it has such a loop.
    (@from) => {
        #[inline]
        fn cells_from<S: Item, C: Convert<S, Self>>(out: &mut [Self], items: &[S], conv: &C, at: usize, step: isize, rest: &[Cow<'_, Steps>]) -> bool { S::cells_into_f64(out, items, conv, at, step, rest) }
    };
}
// Booleans and integers of every width convert into `f64` through fixed-width loops of their own (`converted_cells`). The blocked conversion of `scatter_walk` cost them about 0.25 ns more per item. Other pairs convert a block at a time. Add more if other pairs need speeding up in practice.
units!(u8, u16, u32, u64, [u32; 2], [u64; 2];
    bool: u8 => into_f64, u8: u8 => into_f64, i16: u16 => into_f64, i32: u32 => into_f64, i64: u64 => into_f64, half::f16: u16, f32: u32, f64: u64 => from,
    num_complex::Complex32: [u32; 2], num_complex::Complex64: [u64; 2], char: u32);

/// `items`, read as the units that hold them. Sound by the safety promises of [`Item`] and [`Unit`]: an item type has no padding and has its unit's size and alignment, and every bit pattern of a unit is a value of it.
fn units<T: Item>(items: &[T]) -> &[T::Unit] {
    // SAFETY: the promises of `Item` and `Unit`, as the doc comment says.
    unsafe { std::slice::from_raw_parts(items.as_ptr().cast(), items.len()) }
}

/// `items`, to write as `B`s: units in place of items, or slots for units in place of slots for items.
///
/// # Safety
/// `A` and `B` hold the same bytes in the same way: an item type and its unit, or slots for each. Each `B` that the caller writes holds an `A`, as a unit read through [`units`] from an `A` does.
unsafe fn written_as<A, B>(items: &mut [A]) -> &mut [B] {
    const { assert!(size_of::<A>() == size_of::<B>() && align_of::<A>() == align_of::<B>()) };
    // SAFETY: the caller's contract above.
    unsafe { std::slice::from_raw_parts_mut(items.as_mut_ptr().cast(), items.len()) }
}

/// Writes `items` into `out`, each converted by `conv`, as [`windows_into`] writes units: one cell at each start along a stride of `step` from `at`, with its items at the offsets that `rest` reaches. Each cell width from 2 to 9 has a fixed-width loop. Gives `false` for any other width. Only the sources that [`Item::cells_into_f64`] names call it, so its loops compile for those pairs alone.
fn converted_cells<S: Copy, D, C: Convert<S, D>>(out: &mut [D], items: &[S], conv: &C, at: usize, step: isize, rest: &[Cow<'_, Steps>]) -> bool {
    fn cells<const W: usize, S: Copy, D, C: Convert<S, D>>(out: &mut [D], items: &[S], conv: &C, at: usize, step: isize, cell: &[usize; W]) {
        for (r, items) in items.as_chunks::<W>().0.iter().enumerate() {
            let at = at.wrapping_add_signed(r as isize * step);
            for i in 0..W { out[at + cell[i]] = conv.one(items[i]) }
        }
    }
    let width: usize = rest.iter().map(|s| s.len()).product();
    macro_rules! widths { ($($w:literal)+) => { match width { $($w => cells::<$w, S, D, C>(out, items, conv, at, step, cell_offsets(rest, $w)[..$w].try_into().expect("`cell` holds `W` offsets")),)+ _ => return false } } }
    widths!(2 3 4 5 6 7 8 9);
    true
}

/// Appends the items of `s` that `steps` reach from `base`, each converted by `conv`: one `Steps` for each axis, with the last varying fastest. A `None` offset gives `fill`, converted. `conv` is any `Fn(S) -> D`, or [`Same`] for a source and result of one type. The walk's fixed-width loops move items as [`Unit`]s, so they compile once for each size of item. With [`Same`], the walk appends straight onto `out`. Any other conversion walks `CHUNK` items at a time into a buffer of source items, which it converts onto `out` while they're still in cache.
#[inline]
pub fn walk<S: Item, D>(out: &mut Vec<D>, s: &[S], base: usize, steps: &[Steps], fill: S, conv: impl Convert<S, D>) {
    let n = steps.iter().map(Steps::len).product();
    let (base, steps) = Steps::simplify(base, steps);
    walked(out, &conv, n, |d, from, m| if m == n { walk_into(d, s, fill, base, &steps) } else { range_into(d, s, fill, base, &steps, from, m) })
}

/// Items `from..from + n` of what [`walk`] gives for the same arguments, appended to `out`. It simplifies the plan once, and cuts the range's boxes from it without allocating, so a chunked reader's ranges cost about what one whole walk costs.
#[inline]
pub fn walk_range<S: Item, D>(out: &mut Vec<D>, s: &[S], base: usize, steps: &[Steps], fill: S, from: usize, n: usize, conv: impl Convert<S, D>) {
    let (plan_base, plan) = Steps::simplify(base, steps);
    walked(out, &conv, n, |d, at, m| match Steps::run(base, steps, from + at, m) {
        Some((start, step)) => walk_axis(d, s, fill, 0, &Steps::Stride { start, len: m, step }, &[]),
        None => range_into(d, s, fill, plan_base, &plan, from + at, m),
    })
}

/// Runs `walk` over items `0..n`, giving it a range of them and a buffer of source items to append them to. Where `conv` keeps items as they are, the one range is all `n` items and the buffer is `out`. Otherwise each range holds `CHUNK` items, which `conv` converts onto `out` while they're still in cache.
fn walked<S: Item, D, C: Convert<S, D>>(out: &mut Vec<D>, conv: &C, n: usize, walk: impl Fn(&mut Vec<S>, usize, usize)) {
    use crate::slice::CHUNK;
    match conv.same(out) {
        Some(out) => walk(out, 0, n),
        None => {
            let mut block = Vec::with_capacity(n.min(CHUNK));
            for from in (0..n).step_by(CHUNK) {
                block.clear();
                walk(&mut block, from, CHUNK.min(n - from));
                conv.slice(out, &block);
            }
        }
    }
}

/// `walk_range` over `steps` after `Steps::simplify`: a partial row at each end recurses into the next axis, and the whole rows between them walk as one box, which `Steps::boxed` cuts without allocating. A range therefore costs about its share of one whole walk.
fn range_into<T: Item>(d: &mut Vec<T>, s: &[T], fill: T, base: usize, steps: &[Cow<Steps>], from: usize, n: usize) {
    if n == 0 { return; }
    let Some((axis, rest)) = steps.split_first() else { return d.push(s[base]) };
    let width: usize = rest.iter().map(|s| s.len()).product();
    let row = |d: &mut Vec<T>, j: usize, at: usize, m: usize| match axis.get(j) {
        Some(o) => range_into(d, s, fill, base + o, rest, at, m),
        None => d.extend(std::iter::repeat_n(fill, m)),
    };
    let (mut j, at, mut left) = (from / width, from % width, n);
    if at > 0 {
        let m = left.min(width - at);
        row(d, j, at, m);
        (j, left) = (j + 1, left - m);
    }
    let whole = left / width;
    for piece in axis.pieces(j, whole) {
        let (base, first, rest) = Steps::boxed(base, piece, rest);
        walk_axis(d, s, fill, base, &first, rest);
    }
    if left > whole * width { row(d, j + whole, 0, left - whole * width) }
}

/// The longest run of adjacent items that `walk_into` copies item by item. A longer run copies with `extend_from_slice`.
const RUN: usize = 8;
/// The most items below one axis that `walk_into` reaches through a table of their offsets.
const CELL: usize = 64;

/// The number of items in each cell below an axis, where `walk_into` and `scatter_into` reach every cell's items through one table of their offsets: `rest` holds at least one axis and no fill, reaches at most `CELL` items, and ends in a run of at most `RUN` adjacent items.
fn short_cells(rest: &[Cow<Steps>]) -> Option<usize> {
    let width: usize = rest.iter().map(|s| s.len()).product();
    let run = match rest.last().map(|t| &**t) { Some(&Steps::Stride { len, step: 1, .. }) => len, _ => 1 };
    (!rest.is_empty() && (1..=CELL).contains(&width) && run <= RUN && rest.iter().all(|t| !t.fills())).then_some(width)
}

/// The offsets of a short cell's items (`short_cells`), as [`Steps::combined`] gives them, in the first `width` entries of a table on the stack.
fn cell_offsets(rest: &[Cow<Steps>], width: usize) -> [usize; CELL] {
    let mut cell = [0; CELL];
    for (i, c) in cell[..width].iter_mut().enumerate() { *c = Steps::reach(0, rest, i).expect("a short cell holds no fill") }
    cell
}

/// `walk`'s walk after `Steps::simplify`. Its fixed-width loops move [`Unit`]s, through `windows`.
fn walk_into<T: Item>(d: &mut Vec<T>, s: &[T], fill: T, base: usize, steps: &[Cow<Steps>]) {
    match steps.split_first() { Some((axis, rest)) => walk_axis(d, s, fill, base, axis, rest), None => d.push(s[base]) }
}

/// [`walk_into`] for a box whose first axis, `axis`, comes apart from the others, `rest`, so that a piece of an axis walks with the axes after it without a slice of its own.
fn walk_axis<T: Item>(d: &mut Vec<T>, s: &[T], fill: T, base: usize, axis: &Steps, rest: &[Cow<Steps>]) {
    if let [inner] = rest {
        if let (&Steps::Stride { start: a, len: m, step: 1 }, &Steps::Stride { start: b, len: n, step }) = (axis, &**inner) {
            if step > 1 { return transposed(d, &s[base + a + b..], fill, m, n, step as usize); }
        }
        // Cells picked from a matrix, as columns are: a table of starts, each reaching items a stride apart.
        if let (Steps::Table(t), &Steps::Stride { start: b, len: n, step }) = (axis, &**inner) {
            if step > 1 { t.chunks(|_, run| picked(&mut *d, s, base + b, fill, run, n, step as usize)); return; }
        }
        // One run of `n` adjacent items for each offset along `axis`, as windows give. A run longer than `RUN` copies as one slice.
        if let Steps::Stride { start: b, len: n, step: 1 } = **inner {
            if n > RUN {
                d.reserve(axis.len() * n);
                axis.for_each(|_, o| match o { Some(o) => d.extend_from_slice(&s[base + b + o..][..n]), None => d.extend(std::iter::repeat_n(fill, n)) });
                return;
            }
        }
    }
    // Each offset along `axis` starts a copy of the same few items, at the offsets that `rest` reaches from it, as windows give (`short_cells`). One table of those offsets serves every start. Along a stride, `windows` makes each start's items one array, written straight into spare capacity, so the loop over starts makes no call. Other axes write the fills first, which a missing start keeps.
    if let Some(width) = short_cells(rest) {
        let cell = cell_offsets(rest, width);
        let cell = &cell[..width];
        if let Steps::Stride { start: a, len: m, step } = *axis {
            // SAFETY: `windows` writes only units read from `s`, which hold items of `d`'s type.
            macro_rules! widths { ($($w:literal)+) => { match width { $($w => return crate::slice::appended(d, m * $w, |slots| windows::<$w, T::Unit>(unsafe { written_as(slots) }, units(s), base + a, step, cell.try_into().expect("`cell` holds `width` offsets"))),)+ _ => {} } } }
            widths!(2 3 4 5 6 7 8 9);
        }
        let start = d.len();
        d.resize(start + axis.len() * width, fill);
        let out = &mut d[start..];
        axis.for_each(|r, o| if let Some(o) = o {
            let (out, s) = (&mut out[r * width..][..width], &s[base + o..]);
            for i in 0..width { out[i] = s[cell[i]] }
        });
        return;
    }
    if !rest.is_empty() {
        let width: usize = rest.iter().map(|s| s.len()).product();
        axis.for_each(|_, o| match o { Some(o) => walk_into(d, s, fill, base + o, rest), None => d.extend(std::iter::repeat_n(fill, width)) });
        return;
    }
    if axis.len() == 0 { return; }
    // Each loop gets the source from `base` onwards by value, so that the compiler keeps its bounds in registers.
    let s = &s[base..];
    match *axis {
        Steps::Stride { start, len, step: 1 } => d.extend_from_slice(&s[start..start + len]),
        Steps::Stride { start, len, step: -1 } => d.extend(s[start + 1 - len..=start].iter().rev().copied()),
        Steps::Stride { start, len, step: 0 } => d.extend(std::iter::repeat_n(s[start], len)),
        Steps::Stride { start, len, step } => {
            let end = start.checked_add_signed(step.checked_mul((len - 1) as isize).expect("stride fits")).expect("run stays in the source");
            assert!(start < s.len() && end < s.len(), "run stays in the source");
            crate::slice::appended(d, len, |slots| {
                let item = std::convert::identity(#[inline(always)] |i: usize, slot: &mut MaybeUninit<T>| {
                    // SAFETY: both endpoints were checked; every i is within the run.
                    slot.write(unsafe { *s.get_unchecked(start.wrapping_add_signed(step * i as isize)) });
                });
                // Fixed blocks expose independent strided loads to SLP.
                let (blocks, tail) = slots.as_chunks_mut::<4>();
                for (i, block) in blocks.iter_mut().enumerate() { for (j, slot) in block.iter_mut().enumerate() { item(i * 4 + j, slot); } }
                for (j, slot) in tail.iter_mut().enumerate() { item(blocks.len() * 4 + j, slot); }
            });
        }
        Steps::Rotated { shift, len, stride: 1 } => {
            d.extend_from_slice(&s[shift..len]);
            d.extend_from_slice(&s[..shift]);
        }
        Steps::Padded { start, len, size, stride: 1, edge } => {
            let (lo, hi) = (start.clamp(0, size as i128), (start + len as i128).clamp(0, size as i128));
            let before = (lo - start).clamp(0, len as i128) as usize;
            let middle = (hi - lo).max(0) as usize;
            // The middle copies as one slice. Only the positions past the ends go through the edge mode.
            let border = |d: &mut Vec<T>, from: i128, n: usize| match edge {
                Edge::Fill => d.extend(std::iter::repeat_n(fill, n)),
                Edge::Wrap if size > 0 => cycled(d, &s[..size], from.rem_euclid(size as i128) as usize, n),
                _ => d.extend((0..n).map(|j| edged(edge, from + j as i128, size).map_or(fill, |q| s[q]))),
            };
            border(d, start, before);
            d.extend_from_slice(&s[lo as usize..lo as usize + middle]);
            border(d, start + (before + middle) as i128, len - before - middle);
        }
        Steps::Table(ref t) => if !t.take_into(d, s) { crate::slice::appended(d, t.len(), |slots| t.chunks(|j0, run| for (slot, &o) in slots[j0..].iter_mut().zip(run) { slot.write(s.get(o).copied().unwrap_or(fill)); })) },
        _ => d.extend(axis.offsets().map(move |o| o.map_or(fill, |o| s[o]))),
    }
}

/// Appends `m` rows of `n` items, where item `j` of row `i` is `s[i + j * stride]`, as `walk_into` does for a transposed matrix. Each block of 8 result rows is filled first, then overwritten by the tiled `linalg::transpose_into`.
fn transposed<T: Copy>(d: &mut Vec<T>, s: &[T], fill: T, m: usize, n: usize, stride: usize) {
    for i0 in (0..m).step_by(8) {
        let (start, rows) = (d.len(), 8.min(m - i0));
        d.resize(start + rows * n, fill);
        crate::linalg::transpose_into(&s[i0..], stride, &mut d[start..], n, [n, rows], |x| x);
    }
}

/// Appends a row of `n` items for each start in `starts`, where item `j` of row `i` is `s[base + starts[i] + j * stride]`, and a `Steps::FILL` start gives a row of `fill`: cells picked from a matrix whose items lie a stride apart, as columns' do. Each block of 8 rows reads the source row by row, so its starts share the lines they read.
fn picked<T: Copy>(d: &mut Vec<T>, s: &[T], base: usize, fill: T, starts: &[usize], n: usize, stride: usize) {
    for block in starts.chunks(8) {
        let at = d.len();
        d.resize(at + block.len() * n, fill);
        let out = &mut d[at..];
        for j in 0..n {
            for (k, &start) in block.iter().enumerate() {
                if start != Steps::FILL { out[k * n + j] = s[base + start + j * stride]; }
            }
        }
    }
}
