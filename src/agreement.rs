//! Broadcasting: pairing the axes of two layouts, and reading each argument's items in the order of the result.
//!
//! Axes with the same name pair first. The remaining axes pair from the front unless their names conflict, so a vector pairs with the rows of a matrix. Two names conflict when they differ and are both hidden or both visible. A hidden name, which starts with `_`, meets a visible one as if that axis were unnamed, and the result keeps the visible name. An axis of length 1 repeats along its partner. Where both arguments key an axis, the result's keys are their union, and an item that one argument lacks reads as its type's fill.
use crate::{
    buffer::{Element, Source}, error::{Error, Kind}, gather::{walk_range, Same, Steps}, layout::{generated_len, hidden, names_conflict, strides, Layout}, parallel::{extended, Slots}, Arg, Binary, Unary, CHUNK
};
use std::ops::Range;

/// How one argument's items lay out over a result: the argument's offset for each result position.
#[derive(Clone, Debug)]
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

/// A contiguous block of storage that a view reads, each item once, with the block's axes in another order and some of them backwards. [`Mapping::block`] finds one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Block {
    /// The items of storage that the view reads.
    pub range: Range<usize>,
    /// The block's shape, outermost axis first, leaving out the view's axes of length 1.
    pub dims: Vec<usize>,
    /// For each axis of the view, the block axis it walks and whether it walks backwards. `None` for an axis of length 1.
    pub axes: Vec<Option<(usize, bool)>>,
}

impl Block {
    /// How a reduction folds the view over its axis `axis`, or over every item for `None`, by folding the block in its own order: the `[outer, len, inner]` dimensions to fold the block's items along their middle, and the tables that read the totals in the order of the view's other axes, where they come in another order. A reduction combines a lane's items in any order, so these totals are the view's.
    pub fn fold(&self, axis: Option<usize>) -> ([usize; 3], Option<Vec<Steps>>) {
        let Some(axis) = axis else { return ([1, self.range.len(), 1], None) };
        let folded = self.axes[axis].map(|(b, _)| b);
        let dims = &self.dims;
        let around = match folded {
            Some(f) => [dims[..f].iter().product(), dims[f], dims[f + 1..].iter().product()],
            None => [self.range.len(), 1, 1],
        };
        // Each other axis of the view reads the totals along its block axis.
        let rest: Vec<usize> = dims.iter().enumerate().filter(|&(b, _)| Some(b) != folded).map(|(_, &n)| n).collect();
        let stride = strides(&rest);
        let steps: Vec<Steps> = self.axes.iter().enumerate().filter(|&(a, _)| a != axis).map(|(_, kept)| match *kept {
            None => Steps::along(1, 0),
            Some((b, back)) => {
                let b = b - usize::from(folded.is_some_and(|f| b > f));
                let (len, step) = (rest[b], stride[b] as isize);
                if back { Steps::Stride { start: (len - 1) * stride[b], len, step: -step } } else { Steps::Stride { start: 0, len, step } }
            }
        }).collect();
        (around, (!Steps::in_order(&steps)).then_some(steps))
    }
}

impl Mapping {
    /// This mapping as one entry per axis of `shape`, or `None` where offsets aren't axis-separable.
    pub fn entries(&self, shape: &[usize]) -> Option<(usize, Vec<Steps>)> {
        if shape.contains(&0) { return Some((0, shape.iter().map(|&n| Steps::along(n, 0)).collect())); }
        if let Self::Steps { base, steps } = self {
            let (offset, steps) = Steps::per_axis(steps, shape)?;
            return Some((base + offset, steps));
        }
        if matches!(self, Self::Positions(_)) { return None; }
        if let Self::Linear(period) | Self::Tiled(period) = *self {
            let n: usize = shape.iter().product();
            if period < n && period != 1 && !shape.iter().rev().scan(1, |p, &n| { *p *= n; Some(*p) }).any(|p| p == period) { return None; }
        }
        let base = self.get(0)?;
        let steps = shape.iter().zip(strides(shape)).map(|(&len, stride)| {
            Some(Steps::Stride { start: 0, len, step: if len > 1 { self.get(stride)? as isize - base as isize } else { 0 } })
        }).collect::<Option<_>>()?;
        Some((base, steps))
    }
    /// Reads of a logical array of `shape` through `outer`, then through this mapping to its storage.
    pub fn through(&self, outer: &[Steps], shape: &[usize]) -> Option<Self> {
        if outer.iter().any(|s| s.len() == 0) { return Some(Self::Linear(1)); }
        let (base, inner) = self.entries(shape)?;
        let (base, steps) = Steps::through(outer, shape, base, &inner)?;
        Some(Self::Steps { base, steps })
    }
    /// `outer` maps `frame` to a logical array of `shape`; this maps that array to storage. Unsupported compositions return `None`.
    pub fn compose(&self, outer: &Self, shape: &[usize], frame: &[usize]) -> Option<Self> {
        if frame.contains(&0) { return Some(Self::Linear(1)); }
        if matches!(self, Self::Linear(1)) { return Some(outer.clone()); }
        if matches!(outer, Self::Linear(1)) { return Some(self.clone()); }
        let (base, steps) = outer.entries(frame)?;
        if base != 0 { return None; }
        self.through(&steps, shape)
    }
    /// First item offset and signed item strides, where the mapping is affine over `shape`: each axis reads one regular run with no fill, as [`Steps::affine`] decides.
    pub fn affine(&self, shape: &[usize]) -> Option<(usize, Vec<isize>)> {
        // Reject non-affine entries before `per_axis` clones their tables. Singleton offsets and empty shapes remain affine.
        if !shape.contains(&0) && let Self::Steps { steps, .. } = self
            && steps.iter().any(|s| s.len() > 1 && s.affine().is_none()) { return None; }
        let (base, steps) = self.entries(shape)?;
        let (mut first, mut strides) = (base, Vec::with_capacity(steps.len()));
        for s in &steps {
            let (start, step) = s.affine()?;
            first = first.wrapping_add(start);
            strides.push(step);
        }
        Some((first, strides))
    }
    /// The contiguous block of storage that the mapping reads over `shape`, each item once, where the block's axes come in another order or run backwards: as a transpose, a reversal or windows that don't overlap read their source. `None` where the mapping skips, repeats or fills an item.
    pub fn block(&self, shape: &[usize]) -> Option<Block> {
        if shape.contains(&0) { return None; }
        let (first, strides) = self.affine(shape)?;
        let mut walked: Vec<usize> = (0..shape.len()).filter(|&a| shape[a] > 1).collect();
        walked.sort_by_key(|&a| std::cmp::Reverse(strides[a].unsigned_abs()));
        let mut size = 1;
        for &a in walked.iter().rev() {
            if strides[a].unsigned_abs() != size { return None; }
            size *= shape[a];
        }
        let start = walked.iter().filter(|&&a| strides[a] < 0).try_fold(first, |s, &a| s.checked_sub(strides[a].unsigned_abs() * (shape[a] - 1)))?;
        let mut axes = vec![None; shape.len()];
        for (b, &a) in walked.iter().enumerate() { axes[a] = Some((b, strides[a] < 0)); }
        Some(Block { range: start..start + size, dims: walked.iter().map(|&a| shape[a]).collect(), axes })
    }
    /// The offset for position `i`. Panics where the mapping gives a fill.
    #[inline]
    pub fn index(&self, i: usize) -> usize { self.get(i).expect("positional mapping") }
    /// The first offset and stride of items `start..start + n`, where a `Steps` mapping reads one regular run.
    #[inline]
    pub fn run(&self, start: usize, n: usize) -> Option<(usize, isize)> {
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
                (Some(src), Some(dst)) if src != dst => Steps::table(src.align(dst).into_iter().map(|p| p.map_or(Steps::FILL, |p| p * source)).collect()),
                _ => Steps::Stride { start: 0, len: output[b], step: if len == 1 { 0 } else { source as isize } },
            };
            source *= len;
        }
        Self::Steps { base: 0, steps }
    }
    /// The fewest items in a block of a read through this mapping, where its innermost axis skips items, as a transpose's does: as many whole rows as a cache line holds items of `size` bytes, up to `LINE_BLOCK` items. The rows of such a block then share each fetch of a line. 0 where the innermost axis reads adjacent items or one item.
    pub fn line_block(&self, size: usize) -> usize {
        let Self::Steps { steps, .. } = self else { return 0 };
        match steps.iter().rev().find(|s| s.len() > 1) {
            Some(inner) if inner.skips() => (inner.len() * (LINE / size).max(1)).min(LINE_BLOCK),
            _ => 0,
        }
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
        Self::paired(left, right, &Self::paired_axes(left, right, true))
    }
    /// For each axis of `right`, the axis of `left` that it pairs with, or a new axis after `left`'s. Axes with the same name pair first. With `positional`, the remaining axes then pair from the front where their names don't conflict, as [`Agreement::new`] pairs them.
    pub fn paired_axes(left: &Layout, right: &Layout, positional: bool) -> Vec<usize> {
        let (nx, ny) = (left.shape().len(), right.shape().len());
        let mut ya: Vec<_> = (0..ny).map(|b| right.name(b).and_then(|n| (0..nx).find(|&a| left.name(a) == Some(n)))).collect();
        if positional {
            let remaining_left: Vec<_> = (0..nx).filter(|a| !ya.contains(&Some(*a))).collect();
            let remaining_right: Vec<_> = (0..ny).filter(|&b| ya[b].is_none()).collect();
            for (a, b) in remaining_left.into_iter().zip(remaining_right) { if !names_conflict(left.name(a).map(|n| &**n), right.name(b).map(|n| &**n)) { ya[b] = Some(a); } }
        }
        let mut next = nx;
        ya.into_iter().map(|a| a.unwrap_or_else(|| { next += 1; next - 1 })).collect()
    }
    /// Pairs each axis of `right` with the axis of `left` that `axes` gives, where an axis past `left`'s is a new one.
    pub fn paired(left: &Layout, right: &Layout, axes: &[usize]) -> Result<Self, Error> { Self::mapped(left, right, &(0..left.shape().len()).collect::<Vec<_>>(), axes) }
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
            let (lx, ly) = (x.and_then(|a| left.name(a)), y.and_then(|a| right.name(a)));
            names[axis] = match (lx, ly) { (Some(l), Some(r)) if hidden(l) && !hidden(r) => Some(r), _ => lx.or(ly) }.cloned();
            if let (Some(x), Some(y)) = (kx, ky) {
                let key = x.union(y)?;
                shape[axis] = key.len();
                keys[axis] = Some(key);
            } else {
                let n = if nx == ny || ny == 1 { nx } else if nx == 1 { ny } else { return Err(Error::new(Kind::Length, format!("lengths {nx} and {ny} don't agree"))); };
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
/// Bytes in a cache line on the Apple processors that rustymath is tuned on. An x86-64 line holds half as many.
pub(crate) const LINE: usize = 128;
/// The most items in a block of whole rows from [`Mapping::line_block`].
const LINE_BLOCK: usize = 1 << 16;

/// `f` on each block of `n` of a frame's `len` items, with the block's number and length, until a call fails. Gives whether every call succeeded. After a failure, the caller discards every block's results.
#[inline]
pub fn each_block(len: usize, n: usize, mut f: impl FnMut(usize, usize) -> bool) -> bool { (0..len.div_ceil(n)).all(|k| f(k, n.min(len - k * n))) }

/// Runs `leaf` over a frame of `len` items that one argument lays out through `mapping`, appending its results to `out` in order, and gives whether every call succeeded. Ranges of the frame run in parallel, and `leaf` writes each range's results into its slots. Contiguous arguments and long enough broadcast blocks go to `leaf` whole, as in [`blocks`]. Other mappings expand a run of items at a time, as `chunks` sizes it, into a buffer for each range.
#[inline]
pub fn block<Y: Element, B: Copy + Send>(
    y: &[Y], mapping: &Mapping, len: usize, out: &mut Vec<B>, leaf: impl Fn(Arg<'_, Y>, usize, &mut Slots<'_, B>) -> bool + Sync) -> bool {
    extended(out, len, CHUNK, &|range, slots| {
        use {Arg::*, Mapping::*};
        if range.is_empty() { return true; }
        match *mapping {
            Single => leaf(One(y[0]), range.len(), slots),
            Linear(1) => leaf(Many(&y[range.clone()]), range.len(), slots),
            Linear(r) if r >= SHORT_REPEATED => periods(range, r, |k, p| leaf(One(y[k]), p.len(), slots)),
            Tiled(m) if m >= SHORT_CYCLED => periods(range, m, |k, p| leaf(Many(&y[p.start - k * m..p.end - k * m]), p.len(), slots)),
            _ => {
                let mut ys = Vec::new();
                chunks(range, &[(mapping, size_of::<Y>())], |at, n| leaf(expanded(mapping, y, at, n, &mut ys), n, slots))
            }
        }
    })
}

/// Runs `leaf` over the frame of `agreement`, appending its results to `out` in order, and gives whether every call succeeded. Ranges of the frame run in parallel, and `leaf` writes each range's results into its slots. Within a range, contiguous arguments and long enough broadcast blocks go to `leaf` whole: an item beside a slice, or two slices. Shorter blocks and other mappings expand a run of items at a time, as `chunks` sizes it, into buffers for each range.
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
            chunks(range, &[(left, size_of::<X>()), (right, size_of::<Y>())], |at, n| leaf(expanded(left, x, at, n, &mut xs), expanded(right, y, at, n, &mut ys), n, out))
        }
    }
}

/// `f` on each period `k` of `p` frame positions that `range` meets, with the positions of `range` in it. Gives whether every call succeeded. It calls `f` on every period even after a failure, so that every slot gets written.
fn periods(range: Range<usize>, p: usize, mut f: impl FnMut(usize, Range<usize>) -> bool) -> bool {
    (range.start / p..range.end.div_ceil(p)).fold(true, |ok, k| f(k, range.start.max(k * p)..range.end.min(k * p + p)) & ok)
}

/// `f` on each run of positions of `range`, with the run's first position and length. A run holds `CHUNK` positions, or more where one of `reads`, each a mapping with the size of the items it reads, asks for whole rows through `Mapping::line_block`. Gives whether every call succeeded. It calls `f` on every run even after a failure, so that every slot gets written.
pub(crate) fn chunks(range: Range<usize>, reads: &[(&Mapping, usize)], mut f: impl FnMut(usize, usize) -> bool) -> bool {
    let len = reads.iter().fold(CHUNK, |n, &(mapping, size)| n.max(mapping.line_block(size)));
    range.clone().step_by(len).fold(true, |ok, at| f(at, len.min(range.end - at)) & ok)
}

/// `f` on the frame of `len` items that one argument lays out through `mapping`, with the argument's items read as `A`, appended to `out`. Gives whether every result succeeded. Ranges of the frame run in parallel. Items that already have type `A` go to `f` as [`block`] passes them. Items of another type convert a run at a time, as `chunks` sizes it, into buffers for each range.
#[inline]
pub fn map1<A: Element, Y: Source<A>, B: Copy + Send>(y: &[Y], mapping: &Mapping, len: usize, f: &(impl Unary<A, Out = B> + Sync), out: &mut Vec<B>) -> bool {
    if let Some(y) = Y::borrowed(y) { return block(y, mapping, len, out, |y, n, o| f.many_into(y, n, o)); }
    extended(out, len, CHUNK, &|range, slots| {
        let (mut ye, mut ys) = (Vec::new(), Vec::new());
        chunks(range, &[(mapping, size_of::<Y>())], |at, n| f.many_into(read_mapped(mapping, y, at, n, &mut ye, &mut ys), n, slots))
    })
}

/// `f` on the frame of `agreement`, with both arguments' items read as `A`, appended to `out`. Gives whether every result succeeded. Ranges of the frame run in parallel. Where both arguments' items already have type `A`, `f` takes them as [`blocks`] passes them. Otherwise `f` takes a run of pairs at a time, as `chunks` sizes it, converted into buffers for each range.
#[inline]
pub fn map2<A: Element, X: Source<A>, Y: Source<A>, B: Copy + Send, L>(x: &[X], y: &[Y], agreement: &Agreement<L>, f: &(impl Binary<A, Out = B> + Sync), out: &mut Vec<B>) -> bool {
    if let (Some(x), Some(y)) = (X::borrowed(x), Y::borrowed(y)) { return blocks(x, y, agreement, out, |x, y, n, o| f.many_into(x, y, n, o)); }
    let (left, right) = (&agreement.left, &agreement.right);
    extended(out, agreement.len, CHUNK, &|range, slots| {
        let (mut xe, mut ye, mut xs, mut ys) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
        chunks(range, &[(left, size_of::<X>()), (right, size_of::<Y>())], |at, n| f.many_into(read_mapped(left, x, at, n, &mut xe, &mut xs), read_mapped(right, y, at, n, &mut ye, &mut ys), n, slots))
    })
}

/// Items `start..start + n` of the frame that `mapping` lays `data` over, read as `A`. A mapping that doesn't read `data` in order expands into `spread`, and items of another type convert into `buf`.
#[inline]
pub fn read_mapped<'a, A: Element, X: Source<A>>(mapping: &Mapping, data: &'a [X], start: usize, n: usize, spread: &'a mut Vec<X>, buf: &'a mut Vec<A>) -> Arg<'a, A> {
    if let Some(data) = X::borrowed(data) { return expanded(mapping, data, start, n, buf); }
    expanded(mapping, data, start, n, spread).converted(0, n, buf, X::read)
}

/// The one item that every position in `start..start + n` reads, where `mapping` repeats a single item over all of them. `None` where the positions read more than one item.
fn single<A: Copy>(mapping: &Mapping, data: &[A], start: usize, n: usize) -> Option<A> {
    match *mapping { Mapping::Single => Some(data[0]), Mapping::Linear(r) if r != 1 && n <= r - start % r => Some(data[start / r]), _ => None }
}

/// Items `start..start + n` of an argument that `mapping` lays out, read as `expanded` reads them but always as a slice of all `n` items. A repeated item fills `buf`.
pub fn expanded_items<'a, A: Element>(mapping: &Mapping, data: &'a [A], start: usize, n: usize, buf: &'a mut Vec<A>) -> &'a [A] {
    if let Some(x) = single(mapping, data, start, n) { buf.clear(); buf.resize(n, x); return buf; }
    match expanded(mapping, data, start, n, buf) { Arg::Many(items) => items, Arg::One(_) => unreachable!("`single` gives every repeated item") }
}

/// Items `start..start + n` of an argument that `mapping` lays out, where they need no copy: a single item, a contiguous run, or items within one repeat or one cycle. A `Steps` mapping reads a forward run in place through `run`.
pub fn in_place<'a, A: Copy>(mapping: &Mapping, data: &'a [A], start: usize, n: usize) -> Option<Arg<'a, A>> {
    if let Some(x) = single(mapping, data, start, n) { return Some(Arg::One(x)); }
    match *mapping {
        Mapping::Linear(1) => Some(Arg::Many(&data[start..start + n])),
        Mapping::Steps { .. } => mapping.run(start, n).and_then(|(o, step)| (step == 1).then(|| Arg::Many(&data[o..o + n]))),
        Mapping::Tiled(m) if n <= m - start % m => Some(Arg::Many(&data[start % m..start % m + n])),
        _ => None,
    }
}

/// Items `start..start + n` of an argument that `mapping` lays out over the frame, read `in_place` where they can be. Others copy into `buf`. A repeated item finds its source by a strength-reduced division, which multiplies in place of dividing. A cycled argument writes one period, then copies what it has written until `buf` is full. A `Steps` mapping copies through `walk_range`.
pub fn expanded<'a, A: Element>(mapping: &Mapping, data: &'a [A], start: usize, n: usize, buf: &'a mut Vec<A>) -> Arg<'a, A> {
    if let Some(arg) = in_place(mapping, data, start, n) { return arg; }
    if let Mapping::Steps { base, ref steps } = *mapping {
        buf.clear();
        walk_range(buf, data, base, steps, A::FILL, start, n, Same);
        return Arg::Many(&buf[..n]);
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

