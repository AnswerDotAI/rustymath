//! Broadcasting: pairing the axes of two layouts, and reading each argument's items in the order of the result.
//!
//! Axes with the same name pair first. The remaining axes pair from the front, so a vector pairs with the rows of a matrix. An axis of length 1 repeats along its partner. Where both arguments key an axis, the result's keys are their union, and an item that one argument lacks reads as its type's fill.
use crate::{
    buffer::{Element, Source}, error::{Error, Kind}, gather::{walk_range, Same, Steps}, layout::{generated_len, strides, Layout}, parallel::{extended, Slots}, Arg, Binary, Unary, CHUNK
};
use std::ops::Range;

/// How one argument's items lay out over a result: the argument's offset for each result position.
#[derive(Clone)]
pub enum Mapping {
    /// One item, which every position reads.
    Single,
    /// Each item repeats over this many consecutive positions.
    Linear(usize),
    /// The input fills the trailing result axes and repeats as one block.
    Tiled(usize),
    /// Item `positions[i]` for position `i`.
    Positions(Vec<usize>),
    /// The offsets that one entry for each result axis, outermost first, reach from `base`. An axis the input broadcasts along has step 0, and a `None` offset gives a fill.
    Steps { base: usize, steps: Vec<Steps> },
}

impl Mapping {
    /// The offset for position `i`. Panics where the mapping gives a fill.
    #[inline]
    pub fn index(&self, i: usize) -> usize { self.get(i).expect("positional mapping") }
    /// The offset of the first of items `start..start + n` where they read consecutive items, within one forward run of a `Steps` mapping's innermost entry, so that a reader can take them in place.
    #[inline]
    pub fn run(&self, start: usize, n: usize) -> Option<usize> {
        let Self::Steps { base, steps } = self else { return None };
        Steps::run(*base, steps, start, n)
    }
    /// The offset for position `i`, or `None` where it reads a fill.
    #[inline]
    pub fn get(&self, i: usize) -> Option<usize> {
        Some(match self {
            Self::Single => 0, Self::Linear(repeat) => i / repeat, Self::Tiled(size) => i % size, Self::Positions(positions) => positions[i],
            Self::Steps { base, steps } => Steps::reach(*base, steps, i)?,
        })
    }
    /// For a contraction such as a matrix product: the position on `right`'s axis `raxis` of each key on `left`'s axis `laxis`. Where the keys match, or either axis has none, positions pair in order.
    pub fn contract(left: &Layout, laxis: usize, right: &Layout, raxis: usize) -> Result<Self, Error> {
        match (left.keys(laxis), right.keys(raxis)) { (Some(x), Some(y)) if x != y => Ok(Self::Positions(y.positions(x, false)?)), _ => Ok(Self::Linear(1)) }
    }
    /// How `input` lays out over `result`, which has `count` items. Axis `a` of `input` is axis `axes[a]` of `result`.
    pub fn new(input: &Layout, axes: &[usize], result: &Layout, count: usize) -> Self {
        if count == 0 { return Self::Single; }
        let (shape, output) = (input.shape(), result.shape());
        let keyed = axes.iter().enumerate().any(|(a, &b)| input.keys(a).is_some() && input.keys(a) != result.keys(b));
        if !keyed {
            let size: usize = shape.iter().product();
            if size == 1 { return Self::Single; }
            let strides = strides(output);
            let (mut source, mut linear, mut tiled, repeat) = (1, true, true, count / size);
            for (&len, &axis) in shape.iter().zip(axes).rev() {
                if len != 1 {
                    linear &= strides[axis] == source * repeat;
                    tiled &= strides[axis] == source;
                }
                source *= len;
            }
            if linear { return Self::Linear(repeat); }
            if tiled { return Self::Tiled(size); }
        }
        let mut steps: Vec<Steps> = output.iter().map(|&len| Steps::Stride { start: 0, len, step: 0 }).collect();
        let mut source = 1;
        for (a, (&len, &b)) in shape.iter().zip(axes).enumerate().rev() {
            steps[b] = match (input.keys(a), result.keys(b)) {
                (Some(src), Some(dst)) if src != dst => Steps::Table(src.align(dst).into_iter().map(|p| p.map(|p| p * source)).collect()),
                _ => Steps::Stride { start: 0, len: output[b], step: if len == 1 { 0 } else { source as isize } },
            };
            source *= len;
        }
        Self::Steps { base: 0, steps }
    }
}

/// Two arguments paired by broadcasting: the result's layout, its number of items, and how each argument lays out over it. A caller that adds its own data to layouts can carry its own layout type, through `map_layout`.
pub struct Agreement<L = Layout> { pub layout: L, pub len: usize, pub left: Mapping, pub right: Mapping }

impl Agreement {
    /// Pairs the axes of `left` and `right`: axes with the same name first, then the rest from the front. A Length error where paired axes have different lengths and neither is 1.
    pub fn new(left: &Layout, right: &Layout) -> Result<Self, Error> {
        // Plain layouts of one shape, or a plain scalar beside a plain array, need no axis maps. Each side reads item i, or its one item.
        if left.plain() && right.plain() && (left.shape() == right.shape() || left.shape().is_empty() || right.shape().is_empty()) {
            let shape = if left.shape().is_empty() { right.shape() } else { left.shape() };
            let len = generated_len(shape)?;
            let map = |l: &Layout| if len > 1 && l.shape() == shape { Mapping::Linear(1) } else { Mapping::Single };
            return Ok(Self { layout: Layout::from(shape.to_vec()), len, left: map(left), right: map(right) });
        }
        let (nx, ny) = (left.shape().len(), right.shape().len());
        let xa: Vec<_> = (0..nx).collect();
        if left.names().is_empty() && right.names().is_empty() { return Self::mapped(left, right, &xa, &(0..ny).collect::<Vec<_>>()); }
        let mut ya: Vec<_> = (0..ny).map(|b| right.name(b).and_then(|n| (0..nx).find(|&a| left.name(a) == Some(n)))).collect();
        let remaining_left: Vec<_> = (0..nx).filter(|a| !ya.contains(&Some(*a))).collect();
        let remaining_right: Vec<_> = (0..ny).filter(|&b| ya[b].is_none()).collect();
        for (a, b) in remaining_left.into_iter().zip(remaining_right) { if left.name(a).is_none() || right.name(b).is_none() { ya[b] = Some(a); } }
        let mut next = nx;
        let ya = ya.into_iter().map(|a| {
            a.unwrap_or_else(|| {
                let axis = next;
                next += 1;
                axis
            })
        }).collect::<Vec<_>>();
        Self::mapped(left, right, &xa, &ya)
    }
    /// Pairs each axis of the argument with fewer axes with the axis of the other that `axes` names.
    pub fn with_axes(left: &Layout, right: &Layout, axes: &[usize]) -> Result<Self, Error> {
        if left.shape().len() < right.shape().len() { Self::mapped(left, right, axes, &(0..right.shape().len()).collect::<Vec<_>>()) }
        else { Self::mapped(left, right, &(0..left.shape().len()).collect::<Vec<_>>(), axes) }
    }
    /// Gives each argument its own result axes. The left argument's axes come first.
    pub fn outer(left: &Layout, right: &Layout) -> Result<Self, Error> {
        let nx = left.shape().len();
        Self::mapped(left, right, &(0..nx).collect::<Vec<_>>(), &(nx..nx + right.shape().len()).collect::<Vec<_>>())
    }
    fn mapped(left: &Layout, right: &Layout, xa: &[usize], ya: &[usize]) -> Result<Self, Error> {
        let mut shape = vec![1; xa.iter().chain(ya).max().map_or(0, |a| a + 1)];
        let mut keys = vec![None; shape.len()];
        let mut names = vec![None; shape.len()];
        for axis in 0..shape.len() {
            let (x, y) = (xa.iter().position(|&a| a == axis), ya.iter().position(|&a| a == axis));
            let (nx, ny) = (x.map_or(1, |a| left.shape()[a]), y.map_or(1, |a| right.shape()[a]));
            let (kx, ky) = (x.and_then(|a| left.keys(a)), y.and_then(|a| right.keys(a)));
            names[axis] = x.and_then(|a| left.name(a)).or_else(|| y.and_then(|a| right.name(a))).cloned();
            if let (Some(x), Some(y)) = (kx, ky) {
                let key = x.union(y)?;
                shape[axis] = key.len();
                keys[axis] = Some(key);
            } else {
                let n = if nx == ny || ny == 1 { nx } else if nx == 1 { ny } else { return Err(Kind::Length.into()); };
                shape[axis] = n;
                keys[axis] = kx.filter(|_| nx == n).or(ky.filter(|_| ny == n)).cloned();
            }
        }
        let len = generated_len(&shape)?;
        let layout = Layout::from(shape).with_keys(keys)?.inherit_names(names);
        let x = Mapping::new(left, xa, &layout, len);
        let y = Mapping::new(right, ya, &layout, len);
        Ok(Self { layout, len, left: x, right: y })
    }
}

impl<L> Agreement<L> {
    /// The same agreement with `f` of its layout as the layout.
    pub fn map_layout<M>(self, f: impl FnOnce(L) -> M) -> Agreement<M> { Agreement { layout: f(self.layout), len: self.len, left: self.left, right: self.right } }
}

/// The frames of several arguments to a function of their trailing axes, such as a matrix function of the last two: the result's frame, as broadcasting pairs the arguments' frames, its number of cells, and each argument's cell under each of them.
pub struct Frames { pub layout: Layout, pub count: usize, cells: Vec<Vec<usize>> }

impl Frames {
    /// Pairs the frames of `args`, each a layout with the number of trailing axes in one of its cells. A Rank error where a layout has fewer axes than its cell, and a Length error where frames don't agree, or where keyed frames would need a cell that an argument lacks.
    pub fn new(args: &[(&Layout, usize)]) -> Result<Self, Error> {
        let frames = args.iter().map(|&(l, core)| {
            let rank = l.shape().len();
            if rank < core { return Err(Error::new(Kind::Rank, format!("an argument needs at least {core} axes, not {rank}"))); }
            Ok(l.axes(0..rank - core))
        }).collect::<Result<Vec<_>, _>>()?;
        let Some((first, rest)) = frames.split_first() else { return Ok(Self { layout: Layout::EMPTY, count: 1, cells: Vec::new() }) };
        let (mut layout, mut chains) = (first.clone(), vec![Vec::<Mapping>::new()]);
        for frame in rest {
            let a = Agreement::new(&layout, frame)?;
            for chain in &mut chains { chain.push(a.left.clone()); }
            chains.push(vec![a.right]);
            layout = a.layout;
        }
        let count = generated_len(layout.shape())?;
        // Each chain maps a position of the result's frame back through every later pairing to its argument's own frame.
        let cells = chains.iter().map(|chain| (0..count).map(|i| chain.iter().rev().try_fold(i, |j, m| m.get(j))).collect::<Option<Vec<_>>>()).collect::<Option<_>>()
            .ok_or_else(|| Error::new(Kind::Length, "keyed frames need a cell that an argument lacks"))?;
        Ok(Self { layout, count, cells })
    }

    /// Argument `k`'s cell under position `i` of the result's frame.
    pub fn cell(&self, k: usize, i: usize) -> usize { self.cells[k][i] }
}

/// Item `i` of the frame, read from `data` through `mapping`.
#[inline]
fn item<A: Element>(mapping: &Mapping, data: &[A], i: usize) -> A { mapping.get(i).map_or(A::FILL, |j| data[j]) }

/// The shortest broadcast block that runs as one call, where one argument repeats each of its items along the block. A shorter block expands into a chunk buffer instead, which measured faster than a call for each block of up to 5 items.
const SHORT_REPEATED: usize = 6;
/// The shortest broadcast block that runs as one call, where an argument cycles and no argument repeats its items. A shorter block expands into a chunk buffer, which measured faster than a call for each block of up to 12 items.
const SHORT_CYCLED: usize = 16;

/// `f` on each block of `n` of a frame's `len` items, with the block's number and length, until a call fails. Gives whether every call succeeded. After a failure, the caller discards every block's results.
#[inline]
pub fn each_block(len: usize, n: usize, mut f: impl FnMut(usize, usize) -> bool) -> bool { (0..len.div_ceil(n)).all(|k| f(k, n.min(len - k * n))) }

/// Runs `leaf` over a frame of `len` items that one argument lays out through `mapping`, appending its results to `out` in order, and gives whether every call succeeded. Ranges of the frame run in parallel, and `leaf` writes each range's results into its slots. A contiguous argument goes to `leaf` a whole range at a time. Other mappings expand `CHUNK` items at a time into a buffer for each range.
#[inline]
pub fn block<Y: Element, B: Copy + Send>(
    y: &[Y], mapping: &Mapping, len: usize, out: &mut Vec<B>, leaf: impl Fn(Arg<'_, Y>, usize, &mut Slots<'_, B>) -> bool + Sync) -> bool {
    extended(out, len, CHUNK, &|range, slots| {
        if let Mapping::Linear(1) = mapping { return leaf(Arg::Many(&y[range.clone()]), range.len(), slots); }
        let mut ys = Vec::new();
        chunks(range, |at, n| leaf(expanded(mapping, y, at, n, &mut ys), n, slots))
    })
}

/// Runs `leaf` over the frame of `agreement`, appending its results to `out` in order, and gives whether every call succeeded. Ranges of the frame run in parallel, and `leaf` writes each range's results into its slots. Within a range, contiguous arguments and long enough broadcast blocks go to `leaf` whole: an item beside a slice, or two slices. Shorter blocks and other mappings expand `CHUNK` items at a time into buffers for each range.
#[inline]
pub fn blocks<X: Element, Y: Element, B: Copy + Send, L>(
    x: &[X], y: &[Y], agreement: &Agreement<L>, out: &mut Vec<B>, leaf: impl Fn(Arg<'_, X>, Arg<'_, Y>, usize, &mut Slots<'_, B>) -> bool + Sync) -> bool {
    let (left, right) = (&agreement.left, &agreement.right);
    extended(out, agreement.len, CHUNK, &|range, slots| paired(x, y, left, right, range, slots, &leaf))
}

/// `leaf` on positions `range` of a frame that `left` and `right` lay `x` and `y` over, in order, as [`blocks`] describes. A broadcast block that `range` cuts goes to `leaf` in its parts.
fn paired<X: Element, Y: Element, B>(
    x: &[X], y: &[Y], left: &Mapping, right: &Mapping, range: Range<usize>, out: &mut Slots<'_, B>,
    leaf: &impl Fn(Arg<'_, X>, Arg<'_, Y>, usize, &mut Slots<'_, B>) -> bool) -> bool {
    use {Arg::*, Mapping::*};
    if range.is_empty() { return true; }
    let n = range.len();
    match (left, right) {
        (Single, Linear(1)) => leaf(One(x[0]), Many(&y[range]), n, out),
        (Linear(1), Single) => leaf(Many(&x[range]), One(y[0]), n, out),
        (Linear(1), Linear(1)) => leaf(Many(&x[range.clone()]), Many(&y[range]), n, out),
        (Linear(r), Tiled(m)) if r == m && *m >= SHORT_REPEATED => periods(range, *m, |k, p| leaf(One(x[k]), Many(&y[p.start - k * m..p.end - k * m]), p.len(), out)),
        (Tiled(m), Linear(r)) if r == m && *m >= SHORT_REPEATED => periods(range, *m, |k, p| leaf(Many(&x[p.start - k * m..p.end - k * m]), One(y[k]), p.len(), out)),
        (Linear(1), Linear(r)) if *r >= SHORT_REPEATED => periods(range, *r, |k, p| leaf(Many(&x[p.clone()]), One(y[k]), p.len(), out)),
        (Linear(r), Linear(1)) if *r >= SHORT_REPEATED => periods(range, *r, |k, p| leaf(One(x[k]), Many(&y[p.clone()]), p.len(), out)),
        (Linear(1), Tiled(m)) if *m >= SHORT_CYCLED => periods(range, *m, |k, p| leaf(Many(&x[p.clone()]), Many(&y[p.start - k * m..p.end - k * m]), p.len(), out)),
        (Tiled(m), Linear(1)) if *m >= SHORT_CYCLED => periods(range, *m, |k, p| leaf(Many(&x[p.start - k * m..p.end - k * m]), Many(&y[p.clone()]), p.len(), out)),
        _ => {
            let (mut xs, mut ys) = (Vec::new(), Vec::new());
            chunks(range, |at, n| leaf(expanded(left, x, at, n, &mut xs), expanded(right, y, at, n, &mut ys), n, out))
        }
    }
}

/// `f` on each period `k` of `p` frame positions that `range` meets, with the positions of `range` in it. Gives whether every call succeeded. It calls `f` on every period even after a failure, so that every slot gets written.
fn periods(range: Range<usize>, p: usize, mut f: impl FnMut(usize, Range<usize>) -> bool) -> bool {
    (range.start / p..range.end.div_ceil(p)).fold(true, |ok, k| f(k, range.start.max(k * p)..range.end.min(k * p + p)) & ok)
}

/// `f` on each run of up to `CHUNK` positions of `range`, with the run's first position and length. Gives whether every call succeeded. It calls `f` on every run even after a failure, so that every slot gets written.
pub(crate) fn chunks(range: Range<usize>, mut f: impl FnMut(usize, usize) -> bool) -> bool {
    range.clone().step_by(CHUNK).fold(true, |ok, at| f(at, CHUNK.min(range.end - at)) & ok)
}

/// `f` on the frame of `len` items that one argument lays out through `mapping`, with the argument's items read as `A`, appended to `out`. Gives whether every result succeeded. Ranges of the frame run in parallel. Items that already have type `A` go to `f` as [`block`] passes them. Items of another type convert `CHUNK` at a time into buffers for each range.
#[inline]
pub fn map1<A: Element, Y: Source<A>, B: Copy + Send>(y: &[Y], mapping: &Mapping, len: usize, f: &(impl Unary<A, Out = B> + Sync), out: &mut Vec<B>) -> bool {
    if let Some(y) = Y::borrowed(y) { return block(y, mapping, len, out, |y, n, o| f.many_into(y, n, o)); }
    extended(out, len, CHUNK, &|range, slots| {
        let (mut ye, mut ys) = (Vec::new(), Vec::new());
        chunks(range, |at, n| f.many_into(read_mapped(mapping, y, at, n, &mut ye, &mut ys), n, slots))
    })
}

/// `f` on the frame of `agreement`, with both arguments' items read as `A`, appended to `out`. Gives whether every result succeeded. Ranges of the frame run in parallel. Where both arguments' items already have type `A`, `f` takes them as [`blocks`] passes them. Otherwise `f` takes `CHUNK` pairs at a time, converted into buffers for each range.
#[inline]
pub fn map2<A: Element, X: Source<A>, Y: Source<A>, B: Copy + Send, L>(x: &[X], y: &[Y], agreement: &Agreement<L>, f: &(impl Binary<A, Out = B> + Sync), out: &mut Vec<B>) -> bool {
    if let (Some(x), Some(y)) = (X::borrowed(x), Y::borrowed(y)) { return blocks(x, y, agreement, out, |x, y, n, o| f.many_into(x, y, n, o)); }
    let (left, right) = (&agreement.left, &agreement.right);
    extended(out, agreement.len, CHUNK, &|range, slots| {
        let (mut xe, mut ye, mut xs, mut ys) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        chunks(range, |at, n| f.many_into(read_mapped(left, x, at, n, &mut xe, &mut xs), read_mapped(right, y, at, n, &mut ye, &mut ys), n, slots))
    })
}

/// Items `start..start + n` of the frame that `mapping` lays `data` over, read as `A`. A mapping that doesn't read `data` in order expands into `spread`, and items of another type convert into `buf`.
#[inline]
pub fn read_mapped<'a, A: Element, X: Source<A>>(mapping: &Mapping, data: &'a [X], start: usize, n: usize, spread: &'a mut Vec<X>, buf: &'a mut Vec<A>) -> Arg<'a, A> {
    match expanded(mapping, data, start, n, spread) {
        Arg::Many(s) => match X::borrowed(s) { Some(s) => Arg::Many(s), None => Arg::Many(s).converted(0, n, buf, X::read) },
        one => one.converted(0, n, buf, X::read),
    }
}

/// Items `start..start + n` of an argument that `mapping` lays out over the frame. A single item, a contiguous run, and items within one repeat or one cycle need no copy. Others copy into `buf`. A repeated item finds its source by a strength-reduced division, which multiplies in place of dividing. A cycled argument writes one period, then copies what it has written until `buf` is full. A `Steps` mapping reads a forward run in place through `run`, and copies anything else through `walk_range`.
pub fn expanded<'a, A: Element>(mapping: &Mapping, data: &'a [A], start: usize, n: usize, buf: &'a mut Vec<A>) -> Arg<'a, A> {
    match *mapping {
        Mapping::Single => return Arg::One(data[0]), Mapping::Linear(1) => return Arg::Many(&data[start..start + n]),
        Mapping::Linear(r) if n <= r - start % r => return Arg::One(data[start / r]),
        Mapping::Steps { base, ref steps } => {
            if let Some(o) = mapping.run(start, n) { return Arg::Many(&data[o..o + n]); }
            buf.clear();
            walk_range(buf, data, base, steps, A::FILL, start, n, Same);
            return Arg::Many(&buf[..n]);
        }
        Mapping::Tiled(m) if n <= m - start % m => return Arg::Many(&data[start % m..start % m + n]), _ => {}
    }
    if buf.len() < n { buf.resize(n, A::FILL) }
    let out = &mut buf[..n];
    match *mapping {
        Mapping::Linear(r) => {
            let r = strength_reduce::StrengthReducedUsize::new(r);
            for (i, slot) in out.iter_mut().enumerate() { *slot = data[(start + i) / r] }
        }
        Mapping::Tiled(m) => {
            let period = m.min(n);
            for (i, slot) in out[..period].iter_mut().enumerate() { *slot = data[(start + i) % m] }
            let mut filled = period;
            while filled < n {
                let more = filled.min(n - filled);
                out.copy_within(..more, filled);
                filled += more;
            }
        }
        _ => { for (i, slot) in out.iter_mut().enumerate() { *slot = item(mapping, data, start + i) } }
    }
    Arg::Many(out)
}

