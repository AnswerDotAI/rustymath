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
    /// Offsets `start..start + len`, as at most two entries.
    fn pieces(&self, start: usize, len: usize) -> Vec<Self> {
        if len == 0 { return vec![]; }
        match *self {
            Self::Stride { start: s, step, .. } => vec![Self::Stride { start: s.wrapping_add_signed(step * start as isize), len, step }],
            Self::Clipped { start: s, size, stride, .. } => vec![Self::Clipped { start: s + start as i128, len, size, stride }],
            Self::Rotated { shift, len: n, stride } => {
                let first = (shift + start) % n;
                let head = len.min(n - first);
                let mut out = vec![Self::Stride { start: first * stride, len: head, step: stride as isize }];
                if head < len { out.push(Self::Stride { start: 0, len: len - head, step: stride as isize }) }
                out
            }
            Self::Table(ref t) => vec![Self::Table(t[start..start + len].to_vec())],
        }
    }
    /// Every sum of one offset from each of `tables`, in the order that `walk` visits them. No table may reach a fill.
    pub fn combined(tables: &[impl std::borrow::Borrow<Self>]) -> Vec<usize> {
        tables.iter().fold(vec![0], |sums, t| sums.iter().flat_map(|&s| t.borrow().offsets().map(move |o| s + o.expect("no fills"))).collect())
    }
    /// `steps` without axes of one position, whose offset moves into `base`, and with neighbouring runs merged into one longer run.
    fn simplify<B: std::borrow::Borrow<Self>>(mut base: usize, steps: &[B]) -> (usize, Vec<Cow<'_, Self>>) {
        let mut kept: Vec<Cow<Self>> = Vec::with_capacity(steps.len());
        for s in steps {
            let s = s.borrow();
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
    /// The offset that `steps`, one entry for each axis, outermost first, reach from `base` for item `i` of their frame, or `None` for a fill.
    #[inline]
    pub fn reach(base: usize, steps: &[Self], mut i: usize) -> Option<usize> {
        let mut offset = base;
        for s in steps.iter().rev() {
            offset += s.get(i % s.len())?;
            i /= s.len();
        }
        Some(offset)
    }
    /// The first source offset and stride of items `start..start + n`, where they form one regular run. Singleton axes disappear and neighbouring strides can form one run. Clipped and rotated reads must stay within their valid, unwrapped part.
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
                Self::Clipped { start: s, size, stride, .. } if fits && s + i as i128 >= 0 && s + (end + 1) as i128 <= size as i128 => stride as isize,
                Self::Rotated { shift, stride, .. } if fits && (shift + i) % len + end - i < len => stride as isize,
                _ => return None,
            };
            if let Some(s) = stride { if step != s * width as isize { return None; } } else { stride = Some(step); }
            width *= len;
            if fits { return Some((Self::reach(base, steps, start)?, stride.unwrap())); }
        }
        (n == 1).then(|| (Self::reach(base, steps, start).expect("singleton run has no fill"), 1))
    }
    /// Whether this entry repeats an offset or gives a fill. Applying an element-wise function to the items it reads, in place of reading the function's results, would then compute an item more than once, or apply the function to a fill.
    pub fn spreads(&self) -> bool {
        match *self {
            Self::Stride { len, step, .. } => step == 0 && len > 1,
            Self::Clipped { start, len, size, .. } => start < 0 || start + len as i128 > size as i128,
            Self::Rotated { .. } => false,
            Self::Table(ref t) => t.iter().any(Option::is_none) || { let mut seen = std::collections::HashSet::new(); !t.iter().flatten().all(|&o| seen.insert(o)) },
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
    /// `outer`'s reads of an array of `shape`, composed with `inner`, which maps the positions along each axis of `shape` to offsets from `base`: one entry for each of `outer`'s, with the new base. Walking the result reads what walking `inner` and then `outer` reads. `None` where an entry of `outer` doesn't walk one axis of `shape`, or two entries walk one axis that `inner` doesn't map by a `Stride`.
    pub fn through(outer: &[Self], shape: &[usize], mut base: usize, inner: &[Self]) -> Option<(usize, Vec<Self>)> {
        let strides = crate::layout::strides(shape);
        let mut walked = vec![false; shape.len()];
        let mut steps = Vec::with_capacity(outer.len());
        for e in outer {
            steps.push(match e.axis_of(shape, &strides)? {
                None => Self::Stride { start: 0, len: e.len(), step: 0 },
                Some(a) => e.read_through(strides[a], &inner[a], !std::mem::replace(&mut walked[a], true), &mut base)?,
            });
        }
        for (r, _) in inner.iter().zip(&walked).filter(|&(_, &w)| !w) { base += r.get(0)?; }
        Some((base, steps))
    }
    /// Whether two of `outer`'s entries walk one axis of `shape`, as overlapping windows do. Applying an element-wise function to the items they read, in place of reading its results, would then compute each item once for every window that holds it.
    pub fn rereads(outer: &[Self], shape: &[usize]) -> bool {
        let strides = crate::layout::strides(shape);
        let mut walked = vec![false; shape.len()];
        outer.iter().any(|e| matches!(e.axis_of(shape, &strides), Some(Some(a)) if std::mem::replace(&mut walked[a], true)))
    }
    /// This entry's positions along an axis whose stride is `unit`, read through `r`. The first entry along an axis takes `r`'s offset at position 0, in its own start or in `base`.
    fn read_through(&self, unit: usize, r: &Self, first: bool, base: &mut usize) -> Option<Self> {
        if first && matches!(*self, Self::Stride { start: 0, len, step } if step == unit as isize && len == r.len()) { return Some(r.clone()); }
        if let Self::Stride { start: r0, step: s, .. } = *r {
            let r0 = if first { r0 } else { 0 };
            match *self {
                Self::Stride { start, len, step } => {
                    let start = usize::try_from(r0 as isize + (start / unit) as isize * s).ok()?;
                    return Some(Self::Stride { start, len, step: step / unit as isize * s });
                }
                Self::Rotated { shift, len, .. } if s >= 0 => { *base += r0; return Some(Self::Rotated { shift, len, stride: s as usize }); }
                Self::Clipped { start, len, size, .. } if s >= 0 => { *base += r0; return Some(Self::Clipped { start, len, size, stride: s as usize }); }
                _ => {}
            }
        }
        first.then(|| Self::Table(self.offsets().map(|o| o.and_then(|o| r.get(o / unit))).collect()))
    }
    /// The axis of `shape` whose stride divides each of this entry's offsets, each quotient within its length: `Some(None)` where every offset is 0, and `None` where no axis fits.
    fn axis_of(&self, shape: &[usize], strides: &[usize]) -> Option<Option<usize>> {
        let zero = match *self {
            Self::Stride { start, len, step } => len == 0 || start == 0 && (step == 0 || len == 1),
            Self::Rotated { stride, .. } | Self::Clipped { stride, .. } => stride == 0,
            Self::Table(ref t) => t.iter().all(|&o| o == Some(0)),
        };
        if zero { return Some(None); }
        let fits = |a: usize| {
            let (s, n) = (strides[a], shape[a]);
            let at = |o: usize| o % s == 0 && o / s < n;
            match *self {
                Self::Stride { start, len, step } => step.unsigned_abs() % s == 0 && at(start) && at(start.wrapping_add_signed(step * (len as isize - 1))),
                Self::Rotated { stride, len, .. } => stride == s && len <= n,
                Self::Clipped { stride, size, .. } => stride == s && size <= n,
                Self::Table(ref t) => t.iter().flatten().all(|&o| at(o)),
            }
        };
        (0..shape.len()).find(|&a| shape[a] > 1 && fits(a)).map(Some)
    }
}

/// How `walk` turns a source item into a result item. Every `Fn(S) -> D` converts item by item, and [`Same`] copies whole slices of a source and result of one type.
pub trait Convert<S: Copy, D> {
    /// `x`, converted.
    fn one(&self, x: S) -> D;
    /// Appends the items of `s`, converted.
    fn slice(&self, d: &mut Vec<D>, s: &[S]) { d.extend(s.iter().map(|&x| self.one(x))) }
    /// Appends the items of `s` from last to first, converted.
    fn reversed(&self, d: &mut Vec<D>, s: &[S]) { d.extend(s.iter().rev().map(|&x| self.one(x))) }
}
impl<S: Copy, D, F: Fn(S) -> D> Convert<S, D> for F {
    #[inline]
    fn one(&self, x: S) -> D { self(x) }
}
/// A [`Convert`] for a source and result of one type, which copies whole slices with no conversion of each item, so that the copies vectorize.
pub struct Same;
impl<T: Copy> Convert<T, T> for Same {
    #[inline]
    fn one(&self, x: T) -> T { x }
    fn slice(&self, d: &mut Vec<T>, s: &[T]) { d.extend_from_slice(s) }
    fn reversed(&self, d: &mut Vec<T>, s: &[T]) { d.extend(s.iter().rev().copied()) }
}

/// Appends the items of `s` that `steps` reach from `base`, each converted by `conv`: one `Steps` for each axis, with the last varying fastest. A `None` offset gives `fill`, converted. `conv` is any `Fn(S) -> D`, or [`Same`] for a source and result of one type, which copies whole slices of the last axis where its pattern allows.
#[inline]
pub fn walk<S: Copy, D: Clone>(out: &mut Vec<D>, s: &[S], base: usize, steps: &[Steps], fill: S, conv: impl Convert<S, D>) {
    let (base, steps) = Steps::simplify(base, steps);
    walk_into(out, s, &conv, fill, base, &steps)
}

/// Items `from..from + n` of what [`walk`] gives for the same arguments, appended to `out`. Whole runs of the innermost axis copy as `walk` copies them, so a range in the middle of a long axis costs no more per item than the whole walk.
pub fn walk_range<S: Copy, D: Clone>(out: &mut Vec<D>, s: &[S], base: usize, steps: &[Steps], fill: S, from: usize, n: usize, conv: impl Convert<S, D>) {
    if let Some((start, step)) = Steps::run(base, steps, from, n) {
        let run = Steps::Stride { start, len: n, step };
        walk_into(out, s, &conv, fill, 0, &[Cow::Borrowed(&run)]);
    } else { range_into(out, s, &conv, fill, base, steps, from, n) }
}

/// `walk_range` over `steps`: a partial row at each end recurses into the next axis, and the whole rows between them walk as one box.
fn range_into<S: Copy, D: Clone, C: Convert<S, D>>(d: &mut Vec<D>, s: &[S], conv: &C, fill: S, base: usize, steps: &[Steps], from: usize, n: usize) {
    if n == 0 { return; }
    let Some((axis, rest)) = steps.split_first() else { return d.push(conv.one(s[base])) };
    let width: usize = rest.iter().map(Steps::len).product();
    let row = |d: &mut Vec<D>, j: usize, at: usize, m: usize| match axis.get(j) {
        Some(o) => range_into(d, s, conv, fill, base + o, rest, at, m),
        None => d.extend(std::iter::repeat_n(conv.one(fill), m)),
    };
    let (mut j, at, mut left) = (from / width, from % width, n);
    if at > 0 {
        let m = left.min(width - at);
        row(d, j, at, m);
        (j, left) = (j + 1, left - m);
    }
    let whole = left / width;
    for piece in axis.pieces(j, whole) {
        let cells: Vec<Cow<Steps>> = std::iter::once(Cow::Owned(piece)).chain(rest.iter().map(Cow::Borrowed)).collect();
        let (base, cells) = Steps::simplify(base, &cells);
        walk_into(d, s, conv, fill, base, &cells);
    }
    if left > whole * width { row(d, j + whole, 0, left - whole * width) }
}

/// The longest run of adjacent items that `walk_into` copies item by item. A longer run copies with `extend_from_slice`.
const RUN: usize = 8;
/// The most items below one axis that `walk_into` reaches through a table of their offsets.
const CELL: usize = 64;

/// `walk` after `Steps::simplify`.
fn walk_into<S: Copy, D: Clone, C: Convert<S, D>>(d: &mut Vec<D>, s: &[S], conv: &C, fill: S, base: usize, steps: &[Cow<Steps>]) {
    let filled = || conv.one(fill);
    let Some((axis, rest)) = steps.split_first() else { return d.push(conv.one(s[base])) };
    if let [inner] = rest {
        if let (&Steps::Stride { start: a, len: m, step: 1 }, &Steps::Stride { start: b, len: n, step }) = (&**axis, &**inner) {
            if step > 1 { return transposed(d, &s[base + a + b..], &|x| conv.one(x), filled(), m, n, step as usize); }
        }
        // One run of `n` adjacent items for each offset along `axis`, as windows give. A run longer than `RUN` copies as one slice.
        if let Steps::Stride { start: b, len: n, step: 1 } = **inner {
            if n > RUN {
                d.reserve(axis.len() * n);
                for o in axis.offsets() { match o { Some(o) => conv.slice(d, &s[base + b + o..][..n]), None => d.extend(std::iter::repeat_n(filled(), n)) } }
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
            macro_rules! widths { ($($w:literal)+) => { match width { $($w => return windows::<$w, _, _>(d, s, base + a, step, m, cell.as_slice().try_into().expect("`cell` holds `width` offsets"), |x| conv.one(x)),)+ _ => {} } } }
            widths!(2 3 4 5 6 7 8 9);
        }
        let start = d.len();
        d.resize(start + axis.len() * width, filled());
        for (out, o) in d[start..].chunks_exact_mut(width).zip(axis.offsets()) {
            let Some(o) = o else { continue };
            let s = &s[base + o..];
            for (slot, &k) in out.iter_mut().zip(&cell) { *slot = conv.one(s[k]) }
        }
        return;
    }
    if !rest.is_empty() {
        for j in 0..axis.len() {
            match axis.get(j) { Some(o) => walk_into(d, s, conv, fill, base + o, rest), None => d.extend(std::iter::repeat_n(filled(), width)) }
        }
        return;
    }
    if axis.len() == 0 { return; }
    // Each loop gets the source from `base` onwards by value, so that the compiler keeps its bounds in registers.
    let s = &s[base..];
    match **axis {
        Steps::Stride { start, len, step: 1 } => conv.slice(d, &s[start..start + len]),
        Steps::Stride { start, len, step: -1 } => conv.reversed(d, &s[start + 1 - len..=start]),
        Steps::Stride { start, len, step: 0 } => d.extend(std::iter::repeat_n(conv.one(s[start]), len)),
        Steps::Stride { start, len, step } => {
            let end = start.checked_add_signed(step.checked_mul((len - 1) as isize).expect("stride fits")).expect("run stays in the source");
            assert!(start < s.len() && end < s.len(), "run stays in the source");
            crate::slice::appended(d, len, |slots| {
                let item = std::convert::identity(#[inline(always)] |i: usize, slot: &mut MaybeUninit<D>| {
                    // SAFETY: both endpoints were checked; every i is within the run.
                    slot.write(conv.one(unsafe { *s.get_unchecked(start.wrapping_add_signed(step * i as isize)) }));
                });
                // Fixed blocks expose independent strided loads to SLP.
                let (blocks, tail) = slots.as_chunks_mut::<4>();
                for (i, block) in blocks.iter_mut().enumerate() { for (j, slot) in block.iter_mut().enumerate() { item(i * 4 + j, slot); } }
                for (j, slot) in tail.iter_mut().enumerate() { item(blocks.len() * 4 + j, slot); }
            });
        }
        Steps::Rotated { shift, len, stride: 1 } => {
            conv.slice(d, &s[shift..len]);
            conv.slice(d, &s[..shift]);
        }
        Steps::Clipped { start, len, size, stride: 1 } => {
            let (lo, hi) = (start.clamp(0, size as i128), (start + len as i128).clamp(0, size as i128));
            let before = (lo - start).clamp(0, len as i128) as usize;
            let middle = (hi - lo).max(0) as usize;
            d.extend(std::iter::repeat_n(filled(), before));
            conv.slice(d, &s[lo as usize..lo as usize + middle]);
            d.extend(std::iter::repeat_n(filled(), len - before - middle));
        }
        _ => d.extend(axis.offsets().map(move |o| o.map_or_else(filled, |o| conv.one(s[o])))),
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
