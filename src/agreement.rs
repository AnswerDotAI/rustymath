//! Broadcasting: pairing the axes of two layouts, and reading each argument's items in the order of the result.
//!
//! Axes with the same name pair first. The remaining axes pair from the front unless their names conflict, so a vector pairs with the rows of a matrix. Two names conflict when they differ and are both hidden or both visible. A hidden name, which starts with `_`, meets a visible one as if that axis were unnamed, and the result keeps the visible name. An axis of length 1 repeats along its partner. Where both arguments key an axis, the result's keys are their union, and an item that one argument lacks reads as its type's fill.
use crate::{
    buffer::{Element, Fill, Source}, error::{Error, Kind}, gather::{cycled, walk_plan, Same, Steps, Writes}, layout::{count, generated_len, hidden, names_conflict, strides, Alignment, Layout}, parallel::{extended, Slots}, Arg, Binary, Unary, CHUNK
};
use std::{borrow::Cow, ops::Range, sync::{Arc, OnceLock}};

/// How one argument's items lay out over a result: the argument's offset for each result position.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mapping {
    /// One item, which every position reads.
    Single,
    /// Each item repeats over this many consecutive positions.
    Linear(usize),
    /// The input fills the trailing result axes and repeats as one block.
    Tiled(usize),
    /// The offsets that one entry for each result axis, outermost first, reach from `base`. An axis the input broadcasts along has step 0, and a `None` offset gives a fill.
    Steps { base: usize, steps: Route },
}

/// How a mapping's positions read the same items again (`Mapping::period`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Period<'a> {
    /// Each item repeats over this many consecutive positions.
    Repeats(usize),
    /// Every this many positions read the source's first items again, in order, as `Tiled` reads them.
    Cycles(usize),
    /// Every `len` positions read the same items again, through entries: those that `steps` reach from `base`. Reading a cycle in place walks its entries, so a reader gathers one cycle first (`gather_cycle`).
    Walks { len: usize, base: usize, steps: &'a [Steps] },
}

impl Period<'_> {
    /// The period, where each period's items can be read in place: `None` for a mapping that walks entries (`Walks`).
    pub fn in_place(&self) -> Option<usize> { match *self { Self::Repeats(n) | Self::Cycles(n) => Some(n), Self::Walks { .. } => None } }
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
            Some(f) => [count(&dims[..f]), dims[f], count(&dims[f + 1..])],
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

/// The mapping that walking `tables` reads, with the tables as they are.
impl From<Vec<Steps>> for Mapping { #[inline] fn from(tables: Vec<Steps>) -> Self { Self::Steps { base: 0, steps: tables.into() } } }

/// The mapping that reads the places that `writes` writes, in order: a list of offsets as one table, and a walk as its own steps.
impl From<Writes<'_>> for Mapping {
    fn from(writes: Writes<'_>) -> Self {
        match writes { Writes::Offsets(offsets) => Self::from(vec![Steps::table(offsets.to_vec())]), Writes::Walk { base, steps } => Self::Steps { base, steps: steps.to_vec().into() } }
    }
}

/// The entries of a [`Mapping::Steps`], one for each result axis, with the walk that `Steps::simplify` plans for them and the fill that a call chose for the positions that reach no item. The plan is made on the first walk and kept, so a mapping that is read a block at a time plans its walk once.
#[derive(Clone, Debug, Default)]
pub struct Route { steps: Vec<Steps>, plan: OnceLock<(usize, Vec<Cow<'static, Steps>>)>, fill: Option<Arc<Fill>> }

impl std::ops::Deref for Route { type Target = [Steps]; #[inline] fn deref(&self) -> &[Steps] { &self.steps } }
impl From<Vec<Steps>> for Route { #[inline] fn from(steps: Vec<Steps>) -> Self { Self { steps, plan: OnceLock::new(), fill: None } } }
impl From<Route> for Vec<Steps> { #[inline] fn from(route: Route) -> Self { route.steps } }
impl PartialEq for Route { fn eq(&self, other: &Self) -> bool { self.steps == other.steps && self.fill == other.fill } }
impl Eq for Route {}

impl Route {
    /// The offset that the walk adds to its mapping's base, and the entries it walks.
    #[inline]
    pub(crate) fn plan(&self) -> (usize, &[Cow<'static, Steps>]) {
        let (offset, plan) = self.plan.get_or_init(|| { let (offset, plan) = Steps::simplify(0, &self.steps); (offset, plan.into_iter().map(|s| Cow::Owned(s.into_owned())).collect()) });
        (*offset, plan)
    }
}

impl Mapping {
    /// Whether an axis of this mapping reads through a table of positions, as a selection by data does, rather than by strides.
    pub fn tabled(&self) -> bool { matches!(self, Self::Steps { steps, .. } if steps.iter().any(|s| matches!(s, Steps::Table(_)))) }
    /// The writes at the places that a [`Mapping::Steps`] reads, in order, as `From<Writes>` makes one. `None` for any other mapping.
    pub fn as_writes(&self) -> Option<Writes<'_>> { if let Self::Steps { base, steps } = self { Some(Writes::Walk { base: *base, steps }) } else { None } }
    /// This mapping as one entry per axis of `shape`, or `None` where offsets aren't axis-separable.
    pub fn entries(&self, shape: &[usize]) -> Option<(usize, Vec<Steps>)> {
        if shape.contains(&0) { return Some((0, shape.iter().map(|&n| Steps::along(n, 0)).collect())); }
        if let Self::Steps { base, steps } = self {
            let (offset, steps) = Steps::per_axis(steps, shape)?;
            return Some((base + offset, steps));
        }
        if let Self::Linear(period) | Self::Tiled(period) = *self {
            let n = count(shape);
            if period < n && period != 1 && !shape.iter().rev().scan(1, |p, &n| { *p *= n; Some(*p) }).any(|p| p == period) { return None; }
        }
        let base = self.get(0)?;
        let steps = shape.iter().zip(strides(shape)).map(|(&len, stride)| {
            Some(Steps::Stride { start: 0, len, step: if len > 1 { self.get(stride)? as isize - base as isize } else { 0 } })
        }).collect::<Option<_>>()?;
        Some((base, steps))
    }
    /// Reads of a logical array of `shape` through `outer`, then through this mapping to its storage. A position where `outer` gives a fill reads its item type's own fill. `None` where this mapping chose a fill and `outer` gives fills.
    pub fn through(&self, outer: &[Steps], shape: &[usize]) -> Option<Self> {
        if outer.iter().any(|s| s.len() == 0) { return Some(Self::Linear(1)); }
        if self.chosen().is_some() && outer.iter().any(Steps::fills) { return None; }
        let (base, inner) = self.entries(shape)?;
        let (base, steps) = Steps::through(outer, shape, base, &inner)?;
        Some(self.with_steps(base, steps))
    }
    /// This mapping over a result of `shape` as one entry for each axis of `shape`, or as it is where its offsets don't split by axis ([`Mapping::entries`]).
    pub fn split(&self, shape: &[usize]) -> Self { self.entries(shape).map_or_else(|| self.clone(), |(base, steps)| self.with_steps(base, steps)) }
    /// The base and the tables of a walk that reads what this mapping reads over `shape`. A [`Mapping::Steps`] lends its own steps.
    #[inline]
    pub fn to_walk(&self, shape: &[usize]) -> (usize, Cow<'_, [Steps]>) {
        if let Self::Steps { base, steps } = self { return (*base, Cow::Borrowed(steps)); }
        match self.entries(shape) {
            Some((base, steps)) => (base, Cow::Owned(steps)),
            None => (0, Cow::Owned(vec![Steps::table((0..count(shape)).map(|i| self.get(i).unwrap_or(Steps::FILL)).collect())])),
        }
    }
    /// The tables of [`Mapping::to_walk`]'s walk, with its base as a leading table of one item, for code that walks or classifies tables, such as [`Walk::new`](crate::structure::Walk::new).
    pub fn tables(&self, shape: &[usize]) -> Vec<Steps> { let (base, steps) = self.to_walk(shape); Steps::with_base(base, steps.into_owned()) }
    /// The bytes that this mapping's entries take: each entry, and a table's listed offsets.
    pub fn footprint(&self) -> usize { if let Self::Steps { steps, .. } = self { steps.iter().map(Steps::footprint).sum() } else { 0 } }
    /// `outer` maps `frame` to a logical array of `shape`; this maps that array to storage. Unsupported compositions return `None`, as do two mappings that both read fills where either chose one.
    pub fn compose(&self, outer: &Self, shape: &[usize], frame: &[usize]) -> Option<Self> {
        if frame.contains(&0) { return Some(Self::Linear(1)); }
        if matches!(self, Self::Linear(1)) { return Some(outer.clone()); }
        if matches!(outer, Self::Linear(1)) { return Some(self.clone()); }
        let fill = self.joined_fill(outer)?;
        let (base, steps) = outer.entries(frame)?;
        Some(self.through(&Steps::with_base(base, steps), shape)?.filled(fill))
    }
    /// [`Mapping::compose`], or else, where `table` allows it, the composed offsets as one table over `frame`'s positions, with a fill wherever either mapping reads one. `None` where both read fills and chose different ones.
    pub fn composed(&self, outer: &Self, shape: &[usize], frame: &[usize], table: bool) -> Option<Self> {
        if let Some(mapping) = self.compose(outer, shape, frame) { return Some(mapping); }
        let fill = self.joined_fill(outer)?;
        let n = count(frame);
        table.then(|| Self::Steps { base: 0, steps: vec![Steps::table((0..n).map(|k| outer.get(k).and_then(|j| self.get(j)).unwrap_or(Steps::FILL)).collect())].into() }.filled(fill))
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
    // Per-item loops in basedpl call this for every item. The `Steps` arm calls an out-of-line walk to keep `get` small enough to inline there. `Linear(1)` skips the division.
    #[inline]
    pub fn get(&self, i: usize) -> Option<usize> {
        match self {
            Self::Single => Some(0), Self::Linear(1) => Some(i), Self::Linear(repeat) => Some(i / repeat), Self::Tiled(size) => Some(i % size),
            Self::Steps { base, steps } => Self::reached(*base, steps, i),
        }
    }
    #[inline(never)]
    fn reached(base: usize, steps: &[Steps], i: usize) -> Option<usize> { Steps::reach(base, steps, i) }
    /// For a contraction such as a matrix product: a one-axis mapping from each key on `left`'s axis `laxis` to its position on `right`'s axis `raxis`, as a table. Where the keys match, or either axis has none, positions pair in order.
    pub fn contract(left: &Layout, laxis: usize, right: &Layout, raxis: usize) -> Result<Self, Error> {
        match (left.keys(laxis), right.keys(raxis)) {
            (Some(x), Some(y)) if x != y => Ok(Self::Steps { base: 0, steps: vec![Steps::table(y.positions(x, false)?)].into() }),
            _ => Ok(Self::Linear(1)),
        }
    }
    /// How `input` lays out over `result`, which has `count` items. Axis `a` of `input` is axis `axes[a]` of `result`.
    pub fn new(input: &Layout, axes: &[usize], result: &Layout, count: usize) -> Self {
        let alignments = axes.iter().enumerate().map(|(a, &b)| match (input.keys(a), result.keys(b)) { (Some(src), Some(dst)) => src.aligned(dst), _ => Alignment::Same }).collect();
        Self::aligned(input, axes, result, count, alignments)
    }
    /// [`Mapping::new`], where `alignments` gives how each axis of `input` lies along its axis of `result`: by its keys, or by position where an axis has none.
    fn aligned(input: &Layout, axes: &[usize], result: &Layout, count: usize, alignments: Vec<Alignment>) -> Self {
        if count == 0 { return Self::Single; }
        let (shape, output) = (input.shape(), result.shape());
        if alignments.iter().all(|a| matches!(a, Alignment::Same)) {
            let size = input.count();
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
        for ((&len, &b), alignment) in shape.iter().zip(axes).zip(alignments).rev() {
            steps[b] = match alignment {
                Alignment::Same => Steps::Stride { start: 0, len: output[b], step: if len == 1 { 0 } else { source as isize } },
                Alignment::Leading => Steps::clipped(0, output[b], len, source),
                // A table whose offsets step evenly, as reversed keys give, becomes the stride it is.
                Alignment::Table(t) => { let table = Steps::table(t.into_iter().map(|p| p.map_or(Steps::FILL, |p| p * source)).collect()); table.even().unwrap_or(table) }
            };
            source *= len;
        }
        Self::Steps { base: 0, steps: steps.into() }
    }
    /// The fewest items in a block of a read through this mapping, where its innermost axis skips items, as a transpose's does: as many whole rows as a cache line holds items of `size` bytes, up to `LINE_BLOCK` items. The rows of such a block then share each fetch of a line. 0 where the innermost axis reads adjacent items or one item.
    pub fn line_block(&self, size: usize) -> usize {
        match self.inner() {
            Some(inner) if inner.skips() => (inner.len() * (LINE / size).max(1)).min(LINE_BLOCK),
            _ => 0,
        }
    }
    /// What this mapping's positions read again, and over how many positions (`Period`): each item repeated (`repeats`), `Tiled`'s cycle, or the cycle of a `Steps` mapping whose outer axes step by 0, as a broadcast along outer axes reads. `None` for one item, adjacent items, a mapping that fills, and other steps.
    pub fn period(&self) -> Option<Period<'_>> {
        if let Self::Tiled(m) = *self { return Some(Period::Cycles(m)); }
        if let Some(r) = self.repeats() { return Some(Period::Repeats(r)); }
        let Self::Steps { base, steps } = self else { return None };
        let k = steps.iter().position(|s| !matches!(*s, Steps::Stride { step: 0, .. }))?;
        let (outer, inner) = steps.split_at(k);
        let p = inner.iter().map(Steps::len).product::<usize>();
        if p < 2 || outer.iter().all(|s| s.len() < 2) || self.fills() { return None; }
        let base = base + outer.iter().map(|s| if let Steps::Stride { start, .. } = *s { start } else { 0 }).sum::<usize>();
        Some(if base == 0 && Steps::in_order(inner) { Period::Cycles(p) } else { Period::Walks { len: p, base, steps: inner } })
    }
    /// Whether `expanded` copies most blocks of `block` items read through this mapping, rather than reading them where they lie: steps whose runs of adjacent offsets (`Steps::contiguous`) are shorter than a block, as a transpose's, a reversal's, windows' and a column slice's are. Repeats and cycles don't count: a run that stays inside one repeat or cycle reads in place, and the caller decides where runs end.
    pub fn copies(&self, block: usize) -> bool {
        match self {
            Self::Steps { steps, .. } => Steps::contiguous(steps) < block,
            _ => false,
        }
    }
    /// Whether a walk of blocks of `block` items through this mapping reads each item from its own cache line, where a walk of the whole copy reads them through the tiled transpose (`Steps::transposing`): after `simplify`, the last two axes form a transpose whose inner axis holds a block or more, so that each block walks one strided run. A copy made whole then costs much less than reads through it a block at a time.
    pub fn tiles_whole(&self, block: usize) -> bool {
        let Self::Steps { steps, .. } = self else { return false };
        let (_, steps) = Steps::simplify(0, steps);
        let [.., outer, inner] = &steps[..] else { return false };
        Steps::transposing(outer, inner).is_some_and(|(_, _, n, _)| n >= block)
    }
    /// The innermost stretch of positions that read at one regular stride, as its length and that stride: `(r, 0)` for `Linear(r)`, each of whose stretches reads one item, and `Steps::regular` for steps.
    pub fn regular(&self) -> Option<(usize, isize)> { match self { Self::Linear(r) => Some((*r, 0)), Self::Steps { steps, .. } => Some(Steps::regular(steps)), _ => None } }
    /// The number of positions over which this mapping repeats each item: the length of its innermost regular stretch (`regular`) where that steps by 0, as `Linear(r)` does and a broadcast along inner axes does. `None` for one item, adjacent items, cycles and other steps.
    pub fn repeats(&self) -> Option<usize> {
        self.regular().and_then(|(len, step)| (step == 0 && len > 1).then_some(len))
    }
    /// The runs of this mapping's innermost regular stretch (`regular`) over `positions` positions, where every run reads its items in place: the stretch's length and stride, and the offset of each run's first item, in the order of the runs. A `Linear(r)` mapping's runs read its items in order. `None` where the mapping isn't linear or steps, gives a fill, or reads its stretch through a padded or rotated axis.
    pub fn runs(&self, positions: usize) -> Option<(usize, isize, Vec<usize>)> {
        let (base, steps) = match self { Self::Linear(r) => return Some((*r, 0, (0..positions.div_ceil(*r)).collect())), Self::Steps { base, steps } => (base, steps), _ => return None };
        if self.fills() { return None; }
        let (len, stride) = Steps::regular(steps);
        // Each axis of the stretch keeps only its first position, so that each position left starts one run.
        let (mut width, mut outer) = (1, steps.to_vec());
        for axis in outer.iter_mut().rev() {
            if width == len { break; }
            if axis.len() == 1 { continue; }
            let Steps::Stride { start, len: n, step } = *axis else { return None };
            *axis = Steps::Stride { start, len: 1, step };
            width *= n;
        }
        Some((len, stride, Steps::combined(*base, &outer)))
    }
    /// Whether some position reads a fill, as a padded read's does.
    pub fn fills(&self) -> bool { matches!(self, Self::Steps { steps, .. } if steps.iter().any(Steps::fills)) }
    /// This mapping, reading `fill` at each position that reaches no item, or each item type's own fill for `None`. A mapping whose positions all reach items keeps no fill. `None` on a mapping that chose no fill changes nothing, and costs nothing once inlined.
    #[inline]
    pub fn filled(self, fill: Option<Arc<Fill>>) -> Self { if fill.is_none() && self.chosen().is_none() { self } else { self.refilled(fill) } }
    /// `filled` where the fill can change.
    fn refilled(mut self, fill: Option<Arc<Fill>>) -> Self {
        if let Self::Steps { steps, .. } = &mut self { steps.fill = fill.filter(|_| steps.iter().any(Steps::fills)); }
        self
    }
    /// The fill that a call chose for the positions that reach no item, if any.
    #[inline]
    pub fn chosen(&self) -> Option<&Arc<Fill>> { match self { Self::Steps { steps, .. } => steps.fill.as_ref(), _ => None } }
    /// A [`Mapping::Steps`] of `base` and `steps`, with the fill that this mapping chose.
    #[inline]
    pub fn with_steps(&self, base: usize, steps: Vec<Steps>) -> Self { Self::Steps { base, steps: steps.into() }.filled(self.chosen().cloned()) }
    /// The fill that a composition of this mapping with `outer` reads, which either of them chose, if any. `None` where both read fills and chose different ones.
    fn joined_fill(&self, outer: &Self) -> Option<Option<Arc<Fill>>> {
        // The chosen fills are compared first, so that mappings that chose none scan no tables, which `Table::fills` reads in full.
        if self.chosen() != outer.chosen() && self.fills() && outer.fills() { return None; }
        Some(outer.chosen().or(self.chosen()).cloned())
    }
    /// The innermost axis of a `Steps` mapping that reaches more than one offset.
    fn inner(&self) -> Option<&Steps> {
        let Self::Steps { steps, .. } = self else { return None };
        steps.iter().rev().find(|s| s.len() > 1)
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
    /// Pairs the axes of `left` and `right` as [`Agreement::new`] does, except that where both key an axis, the result keeps `left`'s keys. An item of `right` whose key `left` lacks is never read, and a key of `left` that `right` lacks reads as its type's fill.
    pub fn onto(left: &Layout, right: &Layout) -> Result<Self, Error> {
        if left.plain() && right.plain() { return Self::new(left, right); }
        Self::mapped(left, right, &(0..left.shape().len()).collect::<Vec<_>>(), &Self::paired_axes(left, right, true), true)
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
    pub fn paired(left: &Layout, right: &Layout, axes: &[usize]) -> Result<Self, Error> { Self::mapped(left, right, &(0..left.shape().len()).collect::<Vec<_>>(), axes, false) }
    /// Pairs each axis of the argument with fewer axes with the axis of the other that `axes` names.
    pub fn with_axes(left: &Layout, right: &Layout, axes: &[usize]) -> Result<Self, Error> {
        if left.shape().len() < right.shape().len() { Self::mapped(left, right, axes, &(0..right.shape().len()).collect::<Vec<_>>(), false) }
        else { Self::mapped(left, right, &(0..left.shape().len()).collect::<Vec<_>>(), axes, false) }
    }
    /// Gives each argument its own result axes. The left argument's axes come first.
    pub fn outer(left: &Layout, right: &Layout) -> Result<Self, Error> {
        let nx = left.shape().len();
        Self::mapped(left, right, &(0..nx).collect::<Vec<_>>(), &(nx..nx + right.shape().len()).collect::<Vec<_>>(), false)
    }
    fn mapped(left: &Layout, right: &Layout, xa: &[usize], ya: &[usize], left_keys: bool) -> Result<Self, Error> {
        let mut shape = vec![1; xa.iter().chain(ya).max().map_or(0, |a| a + 1)];
        let mut keys = vec![None; shape.len()];
        let mut names = vec![None; shape.len()];
        let (mut lt, mut rt) = (vec![Alignment::Same; left.shape().len()], vec![Alignment::Same; right.shape().len()]);
        for axis in 0..shape.len() {
            let (x, y) = (xa.iter().position(|&a| a == axis), ya.iter().position(|&a| a == axis));
            let (nx, ny) = (x.map_or(1, |a| left.shape()[a]), y.map_or(1, |a| right.shape()[a]));
            let (kx, ky) = (x.and_then(|a| left.keys(a)), y.and_then(|a| right.keys(a)));
            let (lx, ly) = (x.and_then(|a| left.name(a)), y.and_then(|a| right.name(a)));
            names[axis] = match (lx, ly) { (Some(l), Some(r)) if hidden(l) && !hidden(r) => Some(r), _ => lx.or(ly) }.cloned();
            if let (Some(kx), Some(ky), Some(a), Some(b)) = (kx, ky, x, y) {
                let key = if left_keys { rt[b] = ky.aligned(kx); kx.clone() } else { let (key, [l, r]) = kx.union(ky)?; (lt[a], rt[b]) = (l, r); key };
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
        let (x, y) = (Mapping::aligned(left, xa, &layout, len, lt), Mapping::aligned(right, ya, &layout, len, rt));
        Ok(Self { layout, len, left: x, right: y })
    }
}

impl<L> Agreement<L> {
    /// The same agreement with `f` of its layout as the layout.
    pub fn map_layout<M>(self, f: impl FnOnce(L) -> M) -> Agreement<M> { Agreement { layout: f(self.layout), len: self.len, left: self.left, right: self.right } }
}

/// The frames of several arguments to a function of their trailing axes, such as a matrix function of the last two: the result's frame, as broadcasting pairs the arguments' frames, its number of cells, and each argument's cell under each of them.
pub struct Frames { pub layout: Layout, pub count: usize, cells: Vec<Option<Vec<usize>>> }

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
        // Each chain maps a position of the result's frame back through every later pairing to its argument's own frame. A chain of identities, an argument whose frame is the result's, needs no list.
        let cells = chains.iter().map(|chain| {
            if chain.iter().all(|m| *m == Mapping::Linear(1)) { return Some(None) }
            (0..count).map(|i| chain.iter().rev().try_fold(i, |j, m| m.get(j))).collect::<Option<Vec<_>>>().map(Some)
        }).collect::<Option<_>>().ok_or_else(|| Error::new(Kind::Length, "keyed frames need a cell that an argument lacks"))?;
        Ok(Self { layout, count, cells })
    }

    /// The frames of arguments that are one cell each, with no frame axes: one cell, which each argument's only cell fills. It allocates nothing.
    #[inline]
    pub fn one() -> Self { Self { layout: Layout::EMPTY, count: 1, cells: Vec::new() } }

    /// Argument `k`'s cell under position `i` of the result's frame. An argument with no list of cells, as in [`Frames::one`] or where its frame is the result's, has the result's cell.
    #[inline]
    pub fn cell(&self, k: usize, i: usize) -> usize { match self.cells.get(k) { Some(Some(c)) => c[i], _ => i } }

    /// The positions of the `per` items of argument `k`'s cell under position `i` of the result's frame, in an argument that holds its cells in order.
    #[inline]
    pub fn items(&self, k: usize, i: usize, per: usize) -> std::ops::Range<usize> { let c = self.cell(k, i); c * per..(c + 1) * per }
}

/// The shortest broadcast block that runs as one call, where one argument repeats each of its items along the block. A shorter block expands into a chunk buffer instead, which measured faster than a call for each block of up to 5 items.
const SHORT_REPEATED: usize = 6;
/// The shortest broadcast block that runs as one call, where an argument cycles and no argument repeats its items. A shorter block expands into a chunk buffer, which measured faster than a call for each block of up to 12 items.
const SHORT_CYCLED: usize = 16;
/// The shortest repeat or cycle at which runs of positions end with its periods, so that a read takes the repeated item, or the cycle's items, where they lie. A shorter repeat or cycle fills the run's buffer. On a broadcast `clip` through `read_runs`, reading a repeat of 128 as one item cost about 20% less than filling it, and a repeat of 64 cost about the same either way.
const LONG_PERIOD: usize = 128;
/// Bytes in a cache line on the Apple processors that rustymath is tuned on. An x86-64 line holds half as many.
pub(crate) const LINE: usize = 128;
/// The most items in a block of whole rows from [`Mapping::line_block`].
const LINE_BLOCK: usize = 1 << 16;

/// Runs `leaf` over a frame of `len` items that one argument lays out through `mapping`, appending its results to `out` in order, and gives whether every call succeeded. Ranges of the frame run in parallel, and `leaf` writes each range's results into its slots. Contiguous arguments and long enough broadcast blocks go to `leaf` whole, as in [`blocks`]. Other mappings expand a run of items at a time, as `chunks` sizes it, into a buffer for each range.
#[inline]
pub fn block<Y: Element, B: Copy + Send>(
    y: &[Y], mapping: &Mapping, len: usize, out: &mut Vec<B>, leaf: impl Fn(Arg<'_, Y>, usize, &mut Slots<'_, B>) -> bool + Sync) -> bool {
    let side = Side::new(y, mapping);
    let (y, mapping) = (&*side.data, &*side.mapping);
    extended(out, len, CHUNK, &|range, slots| {
        use {Arg::*, Mapping::*};
        if range.is_empty() { return true; }
        match *mapping {
            Single => leaf(One(y[0]), range.len(), slots),
            Linear(1) => leaf(Many(&y[range.clone()]), range.len(), slots),
            _ => match by_periods([mapping]) {
                Some((p, [w])) => periods(range, p, |k, part| leaf(w.items(y, k, &part), part.len(), slots)),
                None => {
                    let mut ys = Vec::new();
                    chunks(range, &[(mapping, size_of::<Y>())], &mut |at, n| leaf(expanded(mapping, y, at, n, &mut ys), n, slots))
                }
            },
        }
    })
}

/// Runs `leaf` over the frame of `agreement`, appending its results to `out` in order, and gives whether every call succeeded. Ranges of the frame run in parallel, and `leaf` writes each range's results into its slots. Within a range, contiguous arguments and long enough broadcast blocks go to `leaf` whole: an item beside a slice, or two slices. Shorter blocks and other mappings expand a run of items at a time, as `chunks` sizes it, into buffers for each range.
#[inline]
pub fn blocks<X: Element, Y: Element, B: Copy + Send, L>(
    x: &[X], y: &[Y], agreement: &Agreement<L>, out: &mut Vec<B>, leaf: impl Fn(Arg<'_, X>, Arg<'_, Y>, usize, &mut Slots<'_, B>) -> bool + Sync) -> bool {
    let (xs, ys) = (Side::new(x, &agreement.left), Side::new(y, &agreement.right));
    let ((x, left), (y, right)) = ((&*xs.data, &*xs.mapping), (&*ys.data, &*ys.mapping));
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
        _ => match by_periods([left, right]) {
            Some((p, [l, r])) => periods(range, p, |k, part| leaf(l.items(x, k, &part), r.items(y, k, &part), part.len(), out)),
            None => {
                let (mut xs, mut ys) = (Vec::new(), Vec::new());
                chunks(range, &[(left, size_of::<X>()), (right, size_of::<Y>())], &mut |at, n| leaf(expanded(left, x, at, n, &mut xs), expanded(right, y, at, n, &mut ys), n, out))
            }
        },
    }
}

/// `f` on each period `k` of `p` frame positions that `range` meets, with the positions of `range` in it. Gives whether every call succeeded. It calls `f` on every period even after a failure, so that every slot gets written.
fn periods(range: Range<usize>, p: usize, mut f: impl FnMut(usize, Range<usize>) -> bool) -> bool {
    (range.start / p..range.end.div_ceil(p)).fold(true, |ok, k| f(k, range.start.max(k * p)..range.end.min(k * p + p)) & ok)
}

/// How an argument's items lie within each period of a frame that runs a period at a time: in the frame's own order, as one item repeated over the period, or as a cycle of `m` items that starts again each period.
#[derive(Clone, Copy)]
enum Within { Same, Repeated, Cycled(usize) }

impl Within {
    /// How `mapping` lays its items within its periods. `None` for a mapping that neither reads in order, repeats nor cycles.
    fn of(mapping: &Mapping) -> Option<Self> {
        match *mapping { Mapping::Linear(1) => Some(Self::Same), Mapping::Linear(_) => Some(Self::Repeated), _ => match mapping.period()? { Period::Cycles(m) => Some(Self::Cycled(m)), _ => None } }
    }

    /// The items of `x` over positions `part` of period `k`.
    #[inline(always)]
    fn items<'a, X: Copy>(self, x: &'a [X], k: usize, part: &Range<usize>) -> Arg<'a, X> {
        match self { Self::Same => Arg::Many(&x[part.clone()]), Self::Repeated => Arg::One(x[k]), Self::Cycled(m) => Arg::Many(&x[part.start - k * m..part.end - k * m]) }
    }
}

/// The period at which `mappings` run as one call for each period, with how each lays its items within it. Every mapping must read in order, repeat or cycle, those that repeat or cycle must share the period, and the period must be long enough that a call for each period beats expanding it: `SHORT_REPEATED` where a mapping repeats, and `SHORT_CYCLED` otherwise.
fn by_periods<const N: usize>(mappings: [&Mapping; N]) -> Option<(usize, [Within; N])> {
    let mut ps = mappings.iter().filter_map(|m| m.period()?.in_place());
    let p = ps.next()?;
    if ps.any(|q| q != p) { return None; }
    let short = if mappings.iter().any(|m| m.repeats().is_some()) { SHORT_REPEATED } else { SHORT_CYCLED };
    let within = mappings.map(Within::of);
    (p >= short && within.iter().all(Option::is_some)).then(|| (p, within.map(|w| w.unwrap_or(Within::Same))))
}

/// `f` on each run of positions of `range`, with the run's first position and length. A run holds `CHUNK` positions, or more where one of `reads`, each a mapping with the size of the items it reads, asks for whole rows through `Mapping::line_block`. Where a read repeats each item, or cycles, over at least `LONG_PERIOD` positions, runs also end where its periods do, so that it reads its item, or its cycle's items, where they lie (`in_place`). Periods under leading-axis agreement nest, so runs end where the shortest such period does. Gives whether every call succeeded. It calls `f` on every run even after a failure, so that every slot gets written. It takes `f` as `dyn` and stays out of line, so it compiles once.
#[inline(never)]
pub(crate) fn chunks(range: Range<usize>, reads: &[(&Mapping, usize)], f: &mut dyn FnMut(usize, usize) -> bool) -> bool {
    let len = reads.iter().fold(CHUNK, |n, &(mapping, size)| n.max(mapping.line_block(size)));
    let mut runs = |part: Range<usize>| part.clone().step_by(len).fold(true, |ok, at| f(at, len.min(part.end - at)) & ok);
    match reads.iter().filter_map(|&(mapping, _)| mapping.period()?.in_place()).filter(|&p| p >= LONG_PERIOD).min() {
        Some(p) => periods(range, p, |_, part| runs(part)),
        None => runs(range),
    }
}

/// `f` on the frame of `len` items that one argument lays out through `mapping`, with the argument's items read as `A`, appended to `out`. Gives whether every result succeeded. Ranges of the frame run in parallel. An [`INLINE`](crate::Results::INLINE) object runs item by item in [`block`]'s loop, through `judged`, and converts each item as it reads it, so a short broadcast block costs no call. For any other object, items that already have type `A` go to `f` as `block` passes them, and items of another type convert a run at a time, through [`read_runs`].
#[inline]
pub fn map1<A: Element, Y: Source<A>, B: Copy + Send, K: Unary<A, Out = B> + Sync>(y: &[Y], mapping: &Mapping, len: usize, f: &K, out: &mut Vec<B>) -> bool {
    if K::INLINE { return block(y, mapping, len, out, |y, n, o| o.pairs(y, Arg::One(()), n, |b, ()| f.judged(b.read()))); }
    if let Some(y) = Y::borrowed(y) { return block(y, mapping, len, out, |y, n, o| f.many_into(y, n, o)); }
    read_runs([&Side::new(y, mapping)], len, out, &|[y], n, o| f.many_into(y, n, o))
}

/// `f` on the frame of `agreement`, with both arguments' items read as `A`, appended to `out`. Gives whether every result succeeded. Ranges of the frame run in parallel. An [`INLINE`](crate::Results::INLINE) object runs item by item in [`blocks`]' loop, as [`map1`] runs one. For any other object, where both arguments' items already have type `A`, `f` takes them as `blocks` passes them. Otherwise `f` takes a run of pairs at a time, through [`read_runs`].
#[inline]
pub fn map2<A: Element, X: Source<A>, Y: Source<A>, B: Copy + Send, L, K: Binary<A, Out = B> + Sync>(x: &[X], y: &[Y], agreement: &Agreement<L>, f: &K, out: &mut Vec<B>) -> bool {
    if K::INLINE { return blocks(x, y, agreement, out, |x, y, n, o| o.pairs(x, y, n, |a, b| f.judged(a.read(), b.read()))); }
    if let (Some(x), Some(y)) = (X::borrowed(x), Y::borrowed(y)) { return blocks(x, y, agreement, out, |x, y, n, o| f.many_into(x, y, n, o)); }
    read_runs([&Side::new(x, &agreement.left), &Side::new(y, &agreement.right)], agreement.len, out, &|[x, y], n, o| f.many_into(x, y, n, o))
}

/// One argument of a call, read a run of the frame's positions at a time as items of type `A`. A reader holds no buffers, so one reader serves every range of a parallel loop. A loop that takes readers as `dyn Read<A>` compiles once for `A`, whatever types the arguments' items have.
pub trait Read<A>: Sync {
    /// Items `start..start + n` of the argument, as its mapping lays them over the frame. Items that the read copies or converts go into `buf`.
    fn read<'b>(&'b self, start: usize, n: usize, buf: &'b mut Vec<A>) -> Arg<'b, A>;
    /// The argument's one item, where every position of the frame reads it.
    fn constant(&self) -> Option<A>;
    /// The mapping that the argument reads through, and the size of its items, which together size a run (`chunks`).
    fn reads(&self) -> (&Mapping, usize);
}

/// An argument's items and the mapping that lays them over the frame. It reads them as any type they convert to through [`Source`]. Where the mapping's positions walk entries (`Period::Walks`), it holds one cycle's items, gathered once, which `Tiled` reads in place: walking the entries for each run would read every cycle's items again.
pub struct Side<'a, X: Clone> { data: Cow<'a, [X]>, mapping: Cow<'a, Mapping> }

impl<'a, X: Element> Side<'a, X> {
    pub fn new(data: &'a [X], mapping: &'a Mapping) -> Self {
        let mut items = Vec::new();
        match gather_cycle(data, mapping, &mut items) {
            Some(tiled) => Self { data: Cow::Owned(items), mapping: Cow::Owned(tiled) },
            None => Self { data: Cow::Borrowed(data), mapping: Cow::Borrowed(mapping) },
        }
    }
}

impl<A: Element, X: Source<A>> Read<A> for Side<'_, X> {
    #[inline]
    fn read<'b>(&'b self, start: usize, n: usize, buf: &'b mut Vec<A>) -> Arg<'b, A> { read_mapped(&self.mapping, &self.data, start, n, buf) }
    fn constant(&self) -> Option<A> { if matches!(*self.mapping, Mapping::Single) { self.data.first().map(|x| x.read()) } else { None } }
    fn reads(&self) -> (&Mapping, usize) { (&self.mapping, size_of::<X>()) }
}

/// `leaf` on the frame of `len` positions, a run at a time, with the items that each of `readers` reads for the run. It appends the results to `out` in order, and gives whether every call succeeded. Ranges of the frame run in parallel, each with its own buffers. Runs take the sizes that `chunks` gives them, and every run is written even after a failure. The readers and `leaf` are `dyn`, so the loop compiles once for each item type `A`, result type `B` and number of readers, whatever the arguments' types and the kernel. A call through them costs little next to a run's items.
pub fn read_runs<A: Element, B: Copy + Send, const N: usize>(readers: [&dyn Read<A>; N], len: usize, out: &mut Vec<B>, leaf: &(dyn Fn([Arg<'_, A>; N], usize, &mut Slots<'_, B>) -> bool + Sync)) -> bool {
    let reads = readers.map(|r| r.reads());
    extended(out, len, CHUNK, &|range, slots| {
        let mut bufs: [Vec<A>; N] = std::array::from_fn(|_| Vec::new());
        chunks(range, &reads, &mut |at, n| {
            let mut k = 0;
            leaf(bufs.each_mut().map(|buf| { k += 1; readers[k - 1].read(at, n, buf) }), n, slots)
        })
    })
}

/// Items `start..start + n` of the frame that `mapping` lays `data` over, read as `A`. Items that the read copies or converts go into `buf`. Items of another type that need copying expand first into a buffer of their own type, which the call allocates, so that a reader holds no buffers.
#[inline]
fn read_mapped<'a, A: Element, X: Source<A>>(mapping: &Mapping, data: &'a [X], start: usize, n: usize, buf: &'a mut Vec<A>) -> Arg<'a, A> {
    if let Some(data) = X::borrowed(data) { return expanded(mapping, data, start, n, buf); }
    expanded(mapping, data, start, n, &mut Vec::new()).converted(0, n, buf, <X as Source<A>>::read)
}

/// The one item that every position in `start..start + n` reads, where `mapping` repeats a single item over all of them. `None` where the positions read more than one item.
fn single<A: Copy>(mapping: &Mapping, data: &[A], start: usize, n: usize) -> Option<A> {
    match *mapping { Mapping::Single => Some(data[0]), Mapping::Linear(r) if r != 1 && n <= r - start % r => Some(data[start / r]), Mapping::Steps { .. } => mapping.run(start, n).and_then(|(o, step)| (step == 0).then(|| data[o])), _ => None }
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

/// Items `start..start + n` of an argument that `mapping` lays out over the frame, read `in_place` where they can be. Others copy into `buf`. A repeated item fills each of its periods. A cycled argument copies through `gather::cycled`, and a `Steps` mapping through `walk_plan`, over the plan that the mapping keeps (`Route::plan`).
pub fn expanded<'a, A: Element>(mapping: &Mapping, data: &'a [A], start: usize, n: usize, buf: &'a mut Vec<A>) -> Arg<'a, A> {
    if let Some(arg) = in_place(mapping, data, start, n) { return arg; }
    Arg::Many(copied(mapping, data, start, n, buf))
}

/// One cycle's items of `data`, copied into `buf` once, where `mapping`'s positions walk entries (`Period::Walks`), with the `Tiled` mapping that then reads them in place. `None` for any other mapping, leaving `buf` alone. Walking the entries for every cycle would read the same items again each time.
pub fn gather_cycle<A: Element>(data: &[A], mapping: &Mapping, buf: &mut Vec<A>) -> Option<Mapping> {
    let Some(Period::Walks { len, base, steps }) = mapping.period() else { return None };
    copied(&Mapping::Steps { base, steps: steps.to_vec().into() }, data, 0, len, buf);
    Some(Mapping::Tiled(len))
}

/// Items `start..start + n` that `expanded` copies into `buf`, cleared first. It's out of line, so it compiles once for each item type rather than in each caller.
#[inline(never)]
fn copied<'a, A: Element>(mapping: &Mapping, data: &[A], start: usize, n: usize, buf: &'a mut Vec<A>) -> &'a [A] {
    buf.clear();
    let reason = match *mapping {
        Mapping::Single => { buf.resize(n, data[0]); "a repeated item" }
        Mapping::Linear(r) => { periods(start..start + n, r, |k, p| { buf.resize(buf.len() + p.len(), data[k]); true }); "repeats of each item" }
        Mapping::Tiled(m) => { cycled(buf, &data[..m], start, n); "a cycle that wraps" }
        Mapping::Steps { base, ref steps } => { let (offset, plan) = steps.plan(); walk_plan(buf, data, base + offset, plan, Fill::item(mapping.chosen()), start, n, Same); "steps other than a forward run" }
    };
    crate::events::declined("in place", reason, n);
    &buf[..n]
}

