//! Broadcasting: pairing the axes of two layouts, and reading each argument's items in the order of the result.
//!
//! Axes with the same name pair first. The remaining axes pair from the front unless their names conflict, so a vector pairs with the rows of a matrix. Two names conflict when they differ and are both hidden or both visible. A hidden name, which starts with `_`, meets a visible one as if that axis were unnamed, and the result keeps the visible name. An axis of length 1 repeats along its partner. Where both arguments key an axis, the result's keys are their union, and an item that one argument lacks reads as its type's fill.
use crate::{
    Arg, buffer::{Element, kept}, error::{Error, Kind}, gather::{Same, Steps, Writes, cycled, walk_plan},
    layout::{Alignment, Layout, count, generated_len, hidden, names_conflict, strides}
};
use std::{borrow::Cow, ops::Range, sync::{Arc, OnceLock}};

/// Positions of one argument over a result. Missing positions carry no value.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Mapping {
    /// One input item repeated.
    Single,
    /// Each input item repeats this many times.
    Linear(usize),
    /// A cycle of this many input items.
    Tiled(usize),
    /// Axis-separated offsets from the base.
    Steps { base: usize, steps: Vec<Steps> },
}

/// What an unmapped position reads. Input plans use the output position; Scalar broadcasts one source item.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Missing { Default, Input(ReadPlan), Scalar(usize), Invalid }

impl Missing {
    /// The read of a missing position in a copy of `dtype` items from `sources`. A character copy reads its prototype, a space, from one more source it adds, so projecting the sources to codes keeps the space. Other dtypes read their default.
    pub fn prototype(dtype: crate::buffer::DType, sources: &mut Vec<Binding<'_>>) -> Self {
        if dtype != crate::buffer::DType::Character { return Self::Default; }
        sources.push(Binding::full(crate::buffer::View::Characters(&[<char as crate::buffer::Element>::FILL])));
        Self::Scalar(sources.len() - 1)
    }
}

/// A run of a plan's items, as [`ReadPlan::runs`] gives them: a source's items from position `first`, `stride` apart, where a stride of 0 repeats one item; index values from `first`, `step` apart; fills; or positions that reach no source.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Run { Source { source: usize, first: usize, stride: isize }, Index { first: i64, step: i64 }, Fill, Invalid }

impl Run {
    /// The source positions of the `len` items of a source run, and none for any other run.
    pub fn positions(self, len: usize) -> impl Iterator<Item = usize> {
        let (first, stride, len) = if let Self::Source { first, stride, .. } = self { (first, stride, len) } else { (0, 0, 0) };
        (0..len).map(move |j| first.wrapping_add_signed(stride * j as isize))
    }
}

/// What a reader does with the runs of a plan's items ([`ReadPlan::runs`]). A closure gives only `run`.
pub trait RunSink {
    /// `len` items from result position `at`, read as `run`: `false` stops the read.
    fn run(&mut self, at: usize, len: usize, run: Run) -> bool;
    /// Items `range` of `map` over `input`, from result position `at`, where the reader walks them whole: `None` gives them as runs instead.
    fn walk(&mut self, _at: usize, _input: &ReadPlan, _map: &Mapping, _missing: &ReadPlan, _range: Range<usize>) -> Option<bool> { None }
}

impl<F: FnMut(usize, usize, Run) -> bool> RunSink for F { fn run(&mut self, at: usize, len: usize, run: Run) -> bool { self(at, len, run) } }

/// A part of a join: its read, and the end of its stretch of each row of the join.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadPart { pub end: usize, pub plan: ReadPlan }

/// Data-free positional routes shared by readers, compilation and read-back.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ReadRoute {
    Input(usize), Index { base: i64, step: i64 }, Map { input: ReadPlan, map: Mapping, missing: ReadPlan }, Join(Arc<[ReadPart]>), Default, Invalid
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ReadPlan { route: Arc<ReadRoute>, items: usize }

impl ReadPlan {
    pub fn input(source: usize, items: usize) -> Self { Self { route: Arc::new(ReadRoute::Input(source)), items } }
    pub fn fill(items: usize) -> Self { Self { route: Arc::new(ReadRoute::Default), items } }
    pub fn index(base: i64, step: i64, items: usize) -> Self { Self { route: Arc::new(ReadRoute::Index { base, step }), items } }
    fn fallback(missing: Missing, items: usize) -> Self {
        match missing {
            Missing::Input(plan) => plan, Missing::Scalar(source) => Self::input(source, 1).remapped(Mapping::Single, items, Missing::Invalid),
            Missing::Default => Self::fill(items), Missing::Invalid => Self { route: Arc::new(ReadRoute::Invalid), items }
        }
    }
    pub fn copy(source: usize, source_items: usize, map: Mapping, items: usize, missing: Missing) -> Self {
        Self::input(source, source_items).remapped(map, items, missing)
    }
    /// The read of `parts` side by side in `rows` rows, as a catenate reads its arguments: each row holds each part's next stretch of its items in turn, an equal share of the part's items.
    pub fn joined(rows: usize, parts: impl IntoIterator<Item = Self>) -> Result<Self, Error> {
        let too_large = || Error::new(Kind::Length, "joined source is too large");
        let (mut joined, mut width) = (Vec::new(), 0usize);
        for plan in parts {
            let share = if rows == 0 { 0 } else { plan.len() / rows };
            if share * rows != plan.len() { return Err(Error::new(Kind::Length, "a joined part doesn't fill its rows")); }
            width = width.checked_add(share).ok_or_else(too_large)?;
            joined.push(ReadPart { end: width, plan });
        }
        Ok(Self { route: Arc::new(ReadRoute::Join(joined.into())), items: width.checked_mul(rows).ok_or_else(too_large)? })
    }
    pub fn len(&self) -> usize { self.items }
    pub fn is_empty(&self) -> bool { self.items == 0 }
    pub fn route(&self) -> &ReadRoute { &self.route }
    /// What item `item` reads, as a one-item run (`runs`), or `Run::Invalid` past the end.
    pub fn at(&self, item: usize) -> Run {
        let mut found = Run::Invalid;
        if item < self.items { self.runs(item..item + 1, &mut |_, _, run| { found = run; false }); }
        found
    }
    /// Gives `sink` the runs of items `range`, in order, until it stops, and whether it took every run. A source gives a run of its items, an index plan a run of values, a default fills, and a join each part's stretch of each row. A map gives the runs of its positions (`Mapping::spans`), each read through its input, and its misses through its plan of misses, unless the sink walks it whole (`RunSink::walk`). Only a strided run through a map reads one position at a time.
    pub fn runs(&self, range: Range<usize>, sink: &mut dyn RunSink) -> bool { self.runs_through(range.start, 1, range.len(), range.start, sink) }
    /// `runs` of `len` items from position `first`, `stride` apart, given from result position `at`.
    fn runs_through(&self, first: usize, stride: isize, len: usize, at: usize, sink: &mut dyn RunSink) -> bool {
        if len == 0 { return true; }
        match self.route() {
            &ReadRoute::Input(source) => sink.run(at, len, Run::Source { source, first, stride }),
            &ReadRoute::Index { base, step } => {
                sink.run(at, len, Run::Index { first: base.wrapping_add(step.wrapping_mul(first as i64)), step: step.wrapping_mul(stride as i64) })
            }
            ReadRoute::Default => sink.run(at, len, Run::Fill),
            ReadRoute::Invalid => sink.run(at, len, Run::Invalid),
            ReadRoute::Join(parts) => {
                let mut j = 0;
                while j < len {
                    let Some((plan, item, before, after)) = joined_at(parts, first.wrapping_add_signed(stride * j as isize)) else {
                        return sink.run(at + j, len - j, Run::Invalid);
                    };
                    let n = (len - j).min(match stride.cmp(&0) {
                        std::cmp::Ordering::Greater => after.div_ceil(stride as usize), std::cmp::Ordering::Less => before / stride.unsigned_abs() + 1,
                        std::cmp::Ordering::Equal => len - j
                    });
                    if !plan.runs_through(item, stride, n, at + j, sink) { return false; }
                    j += n;
                }
                true
            }
            ReadRoute::Map { input, map, missing } => {
                if input.is_empty() { return missing.runs_through(first, stride, len, at, sink); }
                match stride {
                    1 => {
                        if let Some(go) = sink.walk(at, input, map, missing, first..first + len) { return go; }
                        let mut go = true;
                        map.spans(first..first + len, input.len(), &mut |p, n, run| {
                            go = match run {
                                Some((q, s)) => input.runs_through(q, s, n, at + p - first, sink), None => missing.runs_through(p, 1, n, at + p - first, sink)
                            };
                            go
                        });
                        go
                    }
                    0 => match map.get(first).filter(|&q| q < input.len()) {
                        Some(q) => input.runs_through(q, 0, len, at, sink), None => missing.runs_through(first, 0, len, at, sink)
                    },
                    _ => (0..len).all(|j| {
                        let p = first.wrapping_add_signed(stride * j as isize);
                        match map.get(p).filter(|&q| q < input.len()) {
                            Some(q) => input.runs_through(q, 1, 1, at + j, sink), None => missing.runs_through(p, 1, 1, at + j, sink)
                        }
                    }),
                }
            }
        }
    }
    pub fn mapped(input: ReadPlan, map: Mapping, missing: ReadPlan, items: usize) -> Self {
        if map.in_order(input.items, &[items]) { return input; }
        if let Some(plan) = Self::single_map(&input, &map, items) { return plan; }
        Self { route: Arc::new(ReadRoute::Map { input, map, missing }), items }
    }
    /// `map` over `input`'s own map, composed into one map over `input`'s input. A read of the result then walks the source once, not once for each run of `map`. `None` unless `map` stays within `input`, `input`'s misses read a fill or are invalid, and the composition is structural with no longer tables.
    fn single_map(input: &ReadPlan, map: &Mapping, items: usize) -> Option<Self> {
        let ReadRoute::Map { input: inner, map: first, missing } = input.route() else { return None };
        let frame = map.shape(items);
        if map.bounds(&frame)?.end > input.items { return None; }
        let misses = match missing.route() { ReadRoute::Default => Missing::Default, ReadRoute::Invalid => Missing::Invalid, _ => return None };
        let composed = first.compose(map, &input.shape_hint(), &frame)?;
        (composed.footprint() <= first.footprint() + map.footprint()).then(|| Self::mapped(inner.clone(), composed, Self::fallback(misses, items), items))
    }
    pub fn remapped(&self, map: Mapping, items: usize, missing: Missing) -> Self { Self::mapped(self.clone(), map, Self::fallback(missing, items), items) }
    /// Calls `visit` with each source that this read reaches and its item count. A plan's routes form a tree, so each place that reads a source visits it once.
    pub fn try_for_each_source<E>(&self, visit: &mut impl FnMut(usize, usize) -> Result<(), E>) -> Result<(), E> {
        match self.route() {
            ReadRoute::Input(source) => visit(*source, self.items),
            ReadRoute::Map { input, missing, .. } => { input.try_for_each_source(visit)?; missing.try_for_each_source(visit) }
            ReadRoute::Join(parts) => { for part in &**parts { part.plan.try_for_each_source(visit)?; } Ok(()) }
            ReadRoute::Index { .. } | ReadRoute::Default | ReadRoute::Invalid => Ok(()),
        }
    }
    /// Preserve a logical character read's space default while its bindings become integer codepoints.
    pub fn as_codes(&self, character: bool) -> Cow<'_, Self> {
        if !character { return Cow::Borrowed(self); }
        Cow::Owned(self.defaulted(<char as Element>::FILL as i64))
    }
    /// This read with each default read giving `value`.
    fn defaulted(&self, value: i64) -> Self {
        match self.route() {
            ReadRoute::Map { input, map, missing } => Self::mapped(input.defaulted(value), map.clone(), missing.defaulted(value), self.items),
            ReadRoute::Join(parts) => {
                let parts: Vec<_> = parts.iter().map(|part| ReadPart { end: part.end, plan: part.plan.defaulted(value) }).collect();
                Self { route: Arc::new(ReadRoute::Join(parts.into())), items: self.items }
            }
            ReadRoute::Default => Self::index(value, 0, self.items),
            ReadRoute::Input(_) | ReadRoute::Index { .. } | ReadRoute::Invalid => self.clone(),
        }
    }
    pub fn contiguous(&self, range: Range<usize>) -> Option<(usize, Range<usize>)> {
        let (source, start, stride) = self.run(&range)?;
        (stride == 1 || range.len() == 1).then(|| (source, start..start + range.len()))
    }
    pub fn shape_hint(&self) -> Vec<usize> { match self.route() { ReadRoute::Map { map, .. } => map.shape(self.items), _ => vec![self.items] } }
    /// The one source that a read over a result of `shape` reaches by an affine map, and that map, where every position stays within the source. `None` for a read that joins sources, fills, or maps otherwise.
    pub fn positional(&self, shape: &[usize]) -> Option<(usize, Mapping)> {
        if count(shape) != self.items { return None; }
        match self.route() {
            ReadRoute::Input(source) => Some((*source, Mapping::Linear(1))),
            ReadRoute::Map { input, map, .. } => {
                if input.is_empty() || map.bounds(shape)?.end > input.len() { return None; }
                let inner_shape = input.shape_hint();
                let (source, inner) = input.positional(&inner_shape)?;
                Some((source, inner.compose(map, &inner_shape, shape)?))
            }
            ReadRoute::Join(parts) if parts.len() == 1 => parts[0].plan.positional(shape),
            _ => None,
        }
    }
    /// One source's affine view. Missing positions and non-affine compositions have no borrowed export.
    pub fn affine(&self, shape: &[usize]) -> Option<(usize, usize, Vec<isize>)> {
        let (source, map) = self.positional(shape)?;
        let (base, strides) = map.affine(shape)?;
        Some((source, base, strides))
    }
    /// The inverse read where this plan visits every item of one source once by affine axes.
    pub fn inverse(&self, source: usize, shape: &[usize], items: usize) -> Option<Mapping> {
        if self.items != items { return None; }
        let (found, map) = self.positional(shape)?;
        if found != source { return None; }
        if items == 0 { return Some(Mapping::Linear(1)); }
        let (base, forward) = map.affine(shape)?;
        let mut axes: Vec<_> = (0..shape.len()).filter(|&axis| shape[axis] > 1).collect();
        axes.sort_unstable_by_key(|&axis| forward[axis].unsigned_abs());
        let (mut covered, mut first) = (1usize, base as i128);
        for &axis in &axes {
            if forward[axis].unsigned_abs() != covered { return None; }
            covered = covered.checked_mul(shape[axis])?;
            if forward[axis] < 0 { first += forward[axis] as i128 * (shape[axis] - 1) as i128; }
        }
        if covered != items || first != 0 { return None; }
        let strides = crate::layout::strides(shape);
        let steps = axes.into_iter().rev().map(|axis| {
            let stride = isize::try_from(strides[axis]).ok()?;
            let reverse = forward[axis] < 0;
            Some(Steps::Stride { start: if reverse { (shape[axis] - 1) * strides[axis] } else { 0 }, len: shape[axis],
                step: if reverse { -stride } else { stride } })
        }).collect::<Option<Vec<_>>>()?;
        Some(Mapping::from(steps))
    }
    /// The items that `range` reads as one run of a source: its source, first position and stride, 0 for one item that every position reads and 1 for consecutive items. `runs` must give the whole range as that one run.
    fn run(&self, range: &Range<usize>) -> Option<(usize, usize, isize)> {
        if range.is_empty() || range.end > self.items { return None; }
        let mut found = None;
        self.runs(range.clone(), &mut |_, len, run| {
            if let Run::Source { source, first, stride } = run && len == range.len() && (len == 1 || matches!(stride, 0 | 1)) {
                found = Some((source, first, if len == 1 { 1 } else { stride }));
            }
            false
        });
        found
    }
}

fn affine_bounds(base: usize, strides: &[isize], shape: &[usize]) -> Option<Range<usize>> {
    if shape.len() != strides.len() { return None; }
    if shape.contains(&0) { return Some(0..0); }
    let (mut lo, mut hi) = (base as i128, base as i128);
    for (&len, &stride) in shape.iter().zip(strides) {
        let delta = (stride as i128).checked_mul(len.saturating_sub(1) as i128)?;
        if delta < 0 { lo = lo.checked_add(delta)?; } else { hi = hi.checked_add(delta)?; }
    }
    Some(usize::try_from(lo).ok()?..usize::try_from(hi).ok()?.checked_add(1)?)
}

/// `Mapping::spans` of `n` positions of one row, from item `j0` of its innermost axis `axis`, which starts at offset `start`, given from result position `at`: a stride gives one span below `items` and one past it, and any other axis a span for each stretch that steps evenly.
fn row_spans(
    start: usize, axis: &Steps, j0: usize, n: usize, items: usize, at: usize, f: &mut dyn FnMut(usize, usize, Option<(usize, isize)>) -> bool) -> bool {
    if let Steps::Stride { start: offset, step, .. } = *axis {
        let first = start + offset.wrapping_add_signed(step * j0 as isize);
        return match step.cmp(&0) {
            // Rising positions lie below `items` for a leading stretch, and falling ones for a trailing stretch.
            std::cmp::Ordering::Greater => {
                let below = if first >= items { 0 } else { n.min((items - 1 - first) / step as usize + 1) };
                (below == 0 || f(at, below, Some((first, step)))) && (below == n || f(at + below, n - below, None))
            }
            std::cmp::Ordering::Equal => f(at, n, (first < items).then_some((first, 0))),
            std::cmp::Ordering::Less => {
                let past = if first < items { 0 } else { n.min((first - items) / step.unsigned_abs() + 1) };
                (past == 0 || f(at, past, None)) && (past == n || f(at + past, n - past, Some((first.wrapping_add_signed(step * past as isize), step))))
            }
        };
    }
    let position = |j: usize| axis.get(j).map(|o| start + o).filter(|&p| p < items);
    let (mut j, end) = (j0, j0 + n);
    while j < end {
        let first = position(j);
        let step = match (first, (j + 1 < end).then(|| position(j + 1)).flatten()) { (Some(p), Some(q)) => q as isize - p as isize, _ => 0 };
        let mut k = j + 1;
        while k < end && match (first, position(k)) {
            (None, None) => true, (Some(p), Some(q)) => q as isize - p as isize == step * (k - j) as isize, _ => false
        } { k += 1; }
        if !f(at + j - j0, k - j, first.map(|p| (p, step))) { return false; }
        j = k;
    }
    true
}

/// A whole producer or its currently available logical window.
#[derive(Clone, Copy)]
pub struct Binding<'a> { pub view: crate::buffer::View<'a>, pub start: usize, pub items: usize }

impl<'a> Binding<'a> {
    pub fn full(view: crate::buffer::View<'a>) -> Self { Self { items: view.len(), view, start: 0 } }
    pub fn as_codes(self) -> Self { Self { view: crate::characters::as_codes(self.view), ..self } }
    pub fn borrow(&self, range: Range<usize>) -> Option<crate::buffer::View<'a>> {
        if range.start > range.end || range.end > self.items { return None; }
        let local = range.start.checked_sub(self.start)?..range.end.checked_sub(self.start)?;
        (local.end <= self.view.len()).then(|| self.view.slice(local))
    }
    pub fn affine(&self, base: usize, strides: &[isize], shape: &[usize]) -> Option<(crate::buffer::View<'a>, usize)> {
        self.borrow(affine_bounds(base, strides, shape)?)?;
        Some((self.view, base.checked_sub(self.start)?))
    }
    pub fn window(view: crate::buffer::View<'a>, start: usize, items: usize) -> Result<Self, Error> {
        if start.checked_add(view.len()).is_none_or(|end| end > items) { return Err(Error::new(Kind::Length, "read window exceeds its producer")); }
        Ok(Self { view, start, items })
    }
}

/// Sources are bound at execution; plans never retain their data.
#[derive(Clone, Copy)]
pub struct Read<'a> { pub sources: &'a [Binding<'a>], pub plan: &'a ReadPlan }

/// A read by rows, from [`Read::rows`]: row `r` starts at `items[first + r × step]`, and reads a run of items from there where `run` is true, and else that one item for the whole row.
pub struct Rows<'a, T> { pub items: &'a [T], pub first: usize, pub step: isize, pub run: bool }

impl<'s> Read<'s> {
    /// Checks each source that the plan reads: bound, of the length the plan reads, with its window inside that length.
    fn bindings(&self) -> Result<(), Error> {
        self.plan.try_for_each_source(&mut |source, count| {
            let binding = self.sources.get(source).ok_or_else(|| Error::new(Kind::Length, "missing read source"))?;
            if binding.items != count { return Err(Error::new(Kind::Length, "read source changed length")); }
            if binding.start.checked_add(binding.view.len()).is_none_or(|end| end > count) {
                return Err(Error::new(Kind::Length, "read window exceeds its producer"));
            }
            Ok(())
        })
    }
    pub fn positional(&self, shape: &[usize]) -> Option<(crate::buffer::View<'_>, Mapping)> {
        let (source, mapping) = self.plan.positional(shape)?;
        let binding = self.sources.get(source)?;
        binding.borrow(mapping.bounds(shape)?)?;
        let (base, steps) = mapping.entries(shape)?;
        Some((binding.view, Mapping::Steps { base: base.checked_sub(binding.start)?, steps }))
    }
    fn range(&self, range: &Range<usize>) -> Result<(), Error> {
        if range.start > range.end || range.end > self.plan.len() { return Err(Error::new(Kind::Length, "read range exceeds its result")); }
        self.bindings()
    }
    /// The items that `range` reads as one run of a source's storage, at whatever dtype it holds: the borrowed run and its stride, 1 for consecutive items and 0 for one item that every position reads (`ReadPlan::run`).
    fn run_view(&self, range: &Range<usize>) -> Option<(crate::buffer::View<'s>, usize)> {
        let (source, start, stride) = self.plan.run(range)?;
        let binding = self.sources.get(source)?;
        let end = start.checked_add(if stride == 0 { 1 } else { range.len() })?;
        Some((binding.borrow(start..end)?, stride as usize))
    }
    /// `run_view`, where the storage holds `dtype`.
    fn storage_run(&self, range: &Range<usize>, dtype: crate::buffer::DType) -> Option<(crate::buffer::View<'_>, usize)> {
        self.run_view(range).filter(|(view, _)| view.dtype() == dtype)
    }
    /// Borrows a contiguous input range; other reads use scratch at exactly the requested length.
    pub fn part_view<'a>(
        &'a self, range: Range<usize>, dtype: crate::buffer::DType, checked: bool, scratch: &'a mut crate::buffer::Buffer
    ) -> Result<crate::buffer::View<'a>, Error> {
        self.range(&range)?;
        if let Some((view, stride)) = self.storage_run(&range, dtype) { if stride == 1 || range.len() == 1 { return Ok(view); } }
        self.read_into(range, dtype, checked, scratch)?;
        Ok(scratch.view())
    }
    pub fn part<'a, T: crate::buffer::Cast>(&'a self, range: Range<usize>, checked: bool, scratch: &'a mut Vec<T>) -> Result<Arg<'a, T>, Error> {
        self.range(&range)?;
        if range.is_empty() { return Ok(Arg::Many(&[])); }
        if let Some((view, stride)) = self.run_view(&range) {
            if view.dtype() != T::DTYPE {
                if stride == 0 { return Ok(Arg::One(kept(T::cast(view, 0), checked)?)); }
                cast_run(view, stride, range.len(), checked, scratch)?;
                return Ok(Arg::Many(scratch));
            }
            let items = T::of(view).expect("a typed read selected its dtype");
            return Ok(if stride == 0 { Arg::One(items[0]) } else { Arg::Many(items) });
        }
        scratch.clear();
        self.read_onto(range, checked, scratch)?;
        Ok(Arg::Many(scratch))
    }
    /// Whether `part` reads `range` in place, writing no scratch: one run of a source that holds `T`, or one item that every position reads.
    pub fn in_place<T: crate::buffer::Cast>(&self, range: &Range<usize>) -> bool {
        self.run_view(range).is_some_and(|(view, stride)| stride == 0 || view.dtype() == T::DTYPE)
    }
    /// This read as rows over a result of `shape`, which is rows then items, where every source is bound as the plan reads it, the plan maps one source that holds `T` affinely over `shape`, and each row reads one item or a run of items. `None` for any other read.
    pub fn rows<T: crate::buffer::Cast>(&self, shape: [usize; 2]) -> Option<Rows<'s, T>> {
        self.bindings().ok()?;
        let (source, base, strides) = self.plan.affine(&shape)?;
        let (view, first) = self.sources.get(source)?.affine(base, &strides, &shape)?;
        let &[step, inner @ (0 | 1)] = &strides[..] else { return None };
        Some(Rows { items: T::of(view)?, first, step, run: inner == 1 })
    }
    /// Appends items `range` of the plan onto `output` at `T`, a run at a time (`ReadPlan::runs`): a map over one source walks whole where it can (`walk_onto`), and each other run copies a source's run, gives index values or fills.
    fn read_onto<T: crate::buffer::Cast>(&self, range: Range<usize>, checked: bool, output: &mut Vec<T>) -> Result<(), Error> {
        output.reserve(range.len());
        if self.join_onto(&range, checked, output)? { return Ok(()); }
        let mut onto = Onto { read: self, checked, output, error: None };
        self.plan.runs(range, &mut onto);
        onto.error.map_or(Ok(()), Err)
    }
    /// Appends whole rows `range` of a join whose parts each read one run of their source, lane by lane (`gather::interleave`), and gives `false` for any other read. A part held at another dtype converts once, whole. A part that reads one item repeats it in each of its places.
    fn join_onto<T: crate::buffer::Cast>(&self, range: &Range<usize>, checked: bool, output: &mut Vec<T>) -> Result<bool, Error> {
        let ReadRoute::Join(parts) = self.plan.route() else { return Ok(false) };
        let width = parts.last().map_or(0, |part| part.end);
        if width == 0 || range.start % width != 0 || range.len() % width != 0 { return Ok(false); }
        let (first, rows) = (range.start / width, range.len() / width);
        let (mut lanes, mut start) = (Vec::with_capacity(parts.len()), 0);
        for part in parts.iter() {
            let n = part.end - std::mem::replace(&mut start, part.end);
            let Some((view, stride)) = (Read { sources: self.sources, plan: &part.plan }).run_view(&(first * n..(first + rows) * n)) else { return Ok(false) };
            lanes.push((match (T::of(view), stride) {
                (Some(items), 1) => Cow::Borrowed(items),
                (_, 0) => Cow::Owned(vec![kept(T::cast(view, 0), checked)?; rows * n]),
                _ => {
                    let mut items = Vec::new();
                    T::extend_cast(view, rows * n, checked, &mut items)?;
                    Cow::Owned(items)
                }
            }, n));
        }
        crate::gather::interleave(output, &lanes.iter().map(|(items, n)| (&items[..], *n)).collect::<Vec<_>>(), rows);
        Ok(true)
    }
    /// Appends `len` items of source `source` onto `output` at `T`, from position `first` and `stride` apart, where a stride of 0 repeats one item. Every position must lie within the source's bound window.
    fn run_onto<T: crate::buffer::Cast>(
        &self, source: usize, first: usize, stride: isize, len: usize, checked: bool, output: &mut Vec<T>) -> Result<(), Error> {
        if len == 0 { return Ok(()); }
        let outside = || Error::new(Kind::Length, "read reaches outside its bound window");
        let last = first.checked_add_signed(stride * (len as isize - 1)).ok_or_else(outside)?;
        let low = first.min(last);
        let view = self.sources[source].borrow(low..first.max(last) + 1).ok_or_else(outside)?;
        if stride == 0 {
            let item = kept(T::cast(view, 0), checked)?;
            output.resize(output.len() + len, item);
            return Ok(());
        }
        if stride == 1 {
            if let Some(items) = T::of(view) { output.extend_from_slice(items); return Ok(()); }
            return T::extend_cast(view, len, checked, output);
        }
        let walked = Walked::from(Mapping::Steps { base: 0, steps: vec![Steps::Stride { start: first - low, len, step: stride }] });
        if let Some(items) = T::of(view) { copy_walked(&walked, items, 0, len, T::FILL, output); return Ok(()); }
        with_dtype!(view.dtype(), S => {
            let mut items = Vec::with_capacity(len);
            copy_walked(&walked, S::of(view).expect("the view has its dtype"), 0, len, S::FILL, &mut items);
            T::extend_cast(S::view(&items), len, checked, output)
        })
    }
    /// The binding of the one source that `input` reads, where every position that `map` gives result items up to `end` lies within the source's whole binding.
    fn walkable(&self, input: &ReadPlan, map: &Mapping, end: usize) -> Option<&'s Binding<'s>> {
        let &ReadRoute::Input(source) = input.route() else { return None };
        let binding = self.sources.get(source)?;
        let len = binding.view.len();
        (binding.start == 0 && len == input.len() && match *map {
            Mapping::Single => true, Mapping::Linear(r) => r > 0 && end.saturating_sub(1) / r < len, Mapping::Tiled(m) => m > 0 && m <= len,
            Mapping::Steps { base, ref steps } => Steps::furthest(base, &steps[..]).is_none_or(|furthest| furthest < len)
        }).then_some(binding)
    }
    /// The whole source and the mapping of a read that maps one source with no fills, for a reader that walks the mapping itself (`linalg::Mapped`).
    pub fn mapped_source(&self) -> Option<(crate::buffer::View<'s>, &'s Mapping)> {
        let ReadRoute::Map { input, map, .. } = self.plan.route() else { return None };
        if map.fills() { return None; }
        Some((self.walkable(input, map, self.plan.len())?.view, map))
    }
    /// Walks `map` over the one source that `input` reads, onto `output`, as `copied` walks a mapping, where every position that the walk reaches lies within the source's whole binding and `missing` reads one item, the walk's fill: `false` for any other read.
    fn walk_onto<T: crate::buffer::Cast>(
        &self, input: &ReadPlan, map: &Mapping, missing: &ReadPlan, range: Range<usize>, checked: bool, output: &mut Vec<T>) -> Result<bool, Error> {
        let Some(binding) = self.walkable(input, map, range.end) else { return Ok(false) };
        let Some(fill) = Filled::of(missing).filter(|&fill| fill != Filled::Never || !map.fills()) else { return Ok(false) };
        let walked = Walked::from(map);
        if let Some(items) = T::of(binding.view) {
            let Some(fill) = fill.item::<T>(self.sources) else { return Ok(false) };
            copy_walked(&walked, items, range.start, range.len(), fill, output);
            return Ok(true);
        }
        with_dtype!(binding.view.dtype(), S => {
            let Some(fill) = fill.item::<S>(self.sources) else { return Ok(false) };
            let mut items = Vec::with_capacity(range.len());
            copy_walked(&walked, S::of(binding.view).expect("the view has its dtype"), range.start, range.len(), fill, &mut items);
            T::extend_cast(S::view(&items), items.len(), checked, output)?;
            Ok(true)
        })
    }
    pub fn read_into(&self, range: Range<usize>, dtype: crate::buffer::DType, checked: bool, output: &mut crate::buffer::Buffer) -> Result<(), Error> {
        self.range(&range)?;
        if output.dtype() != dtype { *output = crate::buffer::Buffer::with_capacity(dtype, range.len()); }
        with_dtype!(dtype, T => {
            let items = T::vec_mut(output).expect("output has the selected dtype");
            if let Some((view, stride)) = self.run_view(&range) {
                if view.dtype() != dtype { return cast_run(view, stride, range.len(), checked, items); }
                let source = T::of(view).expect("a typed read selected its dtype");
                items.clear();
                if stride == 0 { items.resize(range.len(), source[0]); } else { items.extend_from_slice(source); }
                return Ok(());
            }
            items.clear();
            self.read_onto(range, checked, items)
        })
    }
}

/// The reader that `Read::read_onto` gives a plan's runs: it appends each run's items at `T` to `output`, and keeps the first error.
struct Onto<'r, 'a, T> { read: &'r Read<'a>, checked: bool, output: &'r mut Vec<T>, error: Option<Error> }

impl<T: crate::buffer::Cast> RunSink for Onto<'_, '_, T> {
    fn run(&mut self, _at: usize, len: usize, run: Run) -> bool {
        let read = match run {
            Run::Source { source, first, stride } => self.read.run_onto(source, first, stride, len, self.checked, self.output),
            Run::Index { first, step } => {
                let values: Vec<i64> = (0..len).map(|j| first.wrapping_add(step.wrapping_mul(j as i64))).collect();
                T::extend_cast(crate::buffer::View::Integers(crate::buffer::Ints::I64(&values)), len, self.checked, self.output)
            }
            Run::Fill => { self.output.resize(self.output.len() + len, T::FILL); Ok(()) }
            Run::Invalid => Err(Error::new(Kind::Index, "read position is outside its source")),
        };
        read.map_err(|e| self.error = Some(e)).is_ok()
    }
    fn walk(&mut self, _at: usize, input: &ReadPlan, map: &Mapping, missing: &ReadPlan, range: Range<usize>) -> Option<bool> {
        match self.read.walk_onto(input, map, missing, range, self.checked, self.output) {
            Ok(true) => Some(true), Ok(false) => None, Err(e) => { self.error = Some(e); Some(false) }
        }
    }
}

/// Where position `position` of a join of `parts` lies: the part's read, the position among the part's own items, and how many of the part's positions in that row lie before it and from it on.
fn joined_at(parts: &[ReadPart], position: usize) -> Option<(&ReadPlan, usize, usize, usize)> {
    let width = parts.last().map_or(0, |part| part.end);
    if width == 0 { return None; }
    let (row, column) = (position / width, position % width);
    let k = parts.partition_point(|part| part.end <= column);
    let start = if k == 0 { 0 } else { parts[k - 1].end };
    let part = parts.get(k)?;
    Some((&part.plan, row * (part.end - start) + column - start, column - start, part.end - column))
}
/// The `len` items of a borrowed run of another dtype, as `T` in `output`: one typed loop over the run ([`Cast::extend_cast`](crate::buffer::Cast::extend_cast)), in place of a routed read for each item. A stride of 0 repeats the run's one item.
fn cast_run<T: crate::buffer::Cast>(view: crate::buffer::View<'_>, stride: usize, len: usize, checked: bool, output: &mut Vec<T>) -> Result<(), Error> {
    output.clear();
    if stride == 0 { output.resize(len, kept(T::cast(view, 0), checked)?); return Ok(()); }
    T::extend_cast(view, len, checked, output)
}

/// The one item that every missing position of a walk reads, which `Filled::of` finds in the plan of a map's misses: the dtype's fill, an index plan's one value, or a source's first item. `Never` is a plan of misses that no position may reach, where the walk's fill goes unread.
#[derive(Clone, Copy, PartialEq)]
enum Filled { Fill, Value(i64), Source(usize), Never }

impl Filled {
    fn of(missing: &ReadPlan) -> Option<Self> {
        match missing.route() {
            ReadRoute::Default => Some(Self::Fill), ReadRoute::Invalid => Some(Self::Never), &ReadRoute::Index { base, step: 0 } => Some(Self::Value(base)),
            &ReadRoute::Input(source) if missing.len() == 1 => Some(Self::Source(source)),
            ReadRoute::Map { input, map: Mapping::Single, .. } => match *input.route() { ReadRoute::Input(source) => Some(Self::Source(source)), _ => None },
            _ => None
        }
    }
    /// The item as an `A`, where it is exactly one: `None` where an index value doesn't convert, or the source holds another dtype or no first item.
    fn item<A: crate::buffer::Cast>(self, sources: &[Binding<'_>]) -> Option<A> {
        match self {
            Self::Fill | Self::Never => Some(A::FILL), Self::Value(value) => A::cast(crate::buffer::View::Integers(crate::buffer::Ints::I64(&[value])), 0).ok(),
            Self::Source(source) => A::of(sources.get(source).filter(|binding| binding.start == 0)?.view)?.first().copied()
        }
    }
}

/// How a mapping's positions read the same items again (`Mapping::period`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Period<'a> {
    /// Each item repeats over this many consecutive positions.
    Repeats(usize),
    /// Every this many positions read the source's first items again, in order, as `Tiled` reads them.
    Cycles(usize),
    /// Every `len` positions read the same items again, through entries: those that `steps` reach from `base`. Reading a cycle in place walks its entries, so a reader gathers one cycle first (`tiled_cycle`).
    Walks { len: usize, base: usize, steps: &'a [Steps] },
}

impl Period<'_> {
    /// How a reader takes this cycle in runs of up to `run` positions: the period, and how many positions it writes the cycle out over (`tiled_cycle`). A cycle shorter than `LONG_CYCLE`, read in order or by walking entries, is written out over `run` positions, so that a run reads it in place at any phase. A longer walk is gathered once with no repeats. `None` for a longer cycle read in order, which a run reads in place, and for repeats.
    pub fn tiling(&self, run: usize) -> Option<(usize, usize)> {
        match *self {
            Self::Cycles(p) | Self::Walks { len: p, .. } if p < LONG_CYCLE => Some((p, run)), Self::Walks { len, .. } => Some((len, 0)),
            Self::Cycles(_) | Self::Repeats(_) => None
        }
    }
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
        let around = match folded { Some(f) => [count(&dims[..f]), dims[f], count(&dims[f + 1..])], None => [self.range.len(), 1, 1] };
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
impl From<Vec<Steps>> for Mapping {
    #[inline]
    fn from(tables: Vec<Steps>) -> Self { Self::Steps { base: 0, steps: tables.into() } }
}

/// The mapping that reads the places that `writes` writes, in order: a list of offsets as one table, and a walk as its own steps.
impl From<Writes<'_>> for Mapping {
    fn from(writes: Writes<'_>) -> Self {
        match writes {
            Writes::Offsets(offsets) => Self::from(vec![Steps::table(offsets.to_vec())]),
            Writes::Walk { base, steps } => Self::Steps { base, steps: steps.to_vec().into() },
        }
    }
}

/// A mapping prepared for reading a block at a time. It finds the walk that a copy follows (`Steps::simplify`) and the cycle that the positions read again (`Period::Cycles`) on the first read that needs each, and keeps both. A reader makes one before its loop over blocks and shares it with every thread of the loop. It derefs to its mapping.
#[derive(Clone, Debug)]
pub struct Walked<'a> { mapping: Cow<'a, Mapping>, walk: OnceLock<(usize, Vec<Cow<'static, Steps>>)>, cycle: OnceLock<Option<usize>> }

impl<'a> Walked<'a> {
    /// `mapping`, to read a block at a time.
    pub fn new(mapping: Cow<'a, Mapping>) -> Self { Self { mapping, walk: OnceLock::new(), cycle: OnceLock::new() } }
    /// The offset that the walk adds to the mapping's base, and the entries it walks. A mapping without entries walks none.
    #[inline]
    fn walk(&self) -> (usize, &[Cow<'static, Steps>]) {
        let (offset, walk) = self.walk.get_or_init(|| match &*self.mapping {
            Mapping::Steps { steps, .. } => {
                let (offset, walk) = Steps::simplify(0, &steps[..]);
                (offset, walk.into_iter().map(|s| Cow::Owned(s.into_owned())).collect())
            }
            _ => (0, Vec::new()),
        });
        (*offset, walk)
    }
    /// The length of the cycle whose items the mapping's positions read again in order (`Period::Cycles`), where it has one.
    pub fn cycle(&self) -> Option<usize> { *self.cycle.get_or_init(|| match self.mapping.period() { Some(Period::Cycles(p)) => Some(p), _ => None }) }
}

impl<'a> From<&'a Mapping> for Walked<'a> { fn from(mapping: &'a Mapping) -> Self { Self::new(Cow::Borrowed(mapping)) } }
impl From<Mapping> for Walked<'static> { fn from(mapping: Mapping) -> Self { Self::new(Cow::Owned(mapping)) } }
impl std::ops::Deref for Walked<'_> {
    type Target = Mapping;
    #[inline]
    fn deref(&self) -> &Mapping { &self.mapping }
}
impl PartialEq for Walked<'_> { fn eq(&self, other: &Self) -> bool { self.mapping == other.mapping } }

impl Mapping {
    /// Whether an axis of this mapping reads through a table of positions, as a selection by data does, rather than by strides.
    pub fn tabled(&self) -> bool { matches!(self, Self::Steps { steps, .. } if steps.iter().any(|s| matches!(s, Steps::Table(_)))) }
    /// The writes at the places that a [`Mapping::Steps`] reads, in order, as `From<Writes>` makes one. `None` for any other mapping.
    pub fn as_writes(&self) -> Option<Writes<'_>> { if let Self::Steps { base, steps } = self { Some(Writes::Walk { base: *base, steps }) } else { None } }
    /// This mapping as one entry per axis of `shape`, or `None` where offsets aren't axis-separable.
    pub fn entries(&self, shape: &[usize]) -> Option<(usize, Vec<Steps>)> {
        if shape.contains(&0) { return Some((0, shape.iter().map(|&n| Steps::along(n, 0)).collect())); }
        if let Self::Steps { base, steps } = self { let (offset, steps) = Steps::per_axis(steps, shape)?; return Some((base + offset, steps)); }
        if let Self::Linear(period) | Self::Tiled(period) = *self {
            let n = count(shape);
            if period < n && period != 1 && !shape.iter().rev().scan(1, |p, &n| {
                *p *= n;
                Some(*p)
            }).any(|p| p == period) { return None; }
        }
        let base = self.get(0)?;
        let steps = shape.iter().zip(strides(shape))
            .map(|(&len, stride)| Some(Steps::Stride { start: 0, len, step: if len > 1 { self.get(stride)? as isize - base as isize } else { 0 } }))
            .collect::<Option<_>>()?;
        Some((base, steps))
    }
    /// Composes positional reads through the logical axes of shape.
    pub fn through(&self, outer: &[Steps], shape: &[usize]) -> Option<Self> {
        if outer.iter().any(|s| s.len() == 0) { return Some(Self::Linear(1)); }
        let (base, inner) = self.entries(shape)?;
        let (base, steps) = Steps::through(outer, shape, base, &inner)?;
        Some(Self::Steps { base, steps })
    }
    /// One entry per logical axis where the offsets separate by axis.
    pub fn split(&self, shape: &[usize]) -> Self { self.entries(shape).map_or_else(|| self.clone(), |(base, steps)| Self::Steps { base, steps }) }
    pub fn to_walk(&self, shape: &[usize]) -> (usize, Cow<'_, [Steps]>) {
        if let Self::Steps { base, steps } = self { return (*base, Cow::Borrowed(steps)); }
        let (base, steps) = match self.entries(shape) {
            Some((base, steps)) => (base, steps), None => (0, vec![Steps::table((0..count(shape)).map(|i| self.get(i).unwrap_or(Steps::FILL)).collect())])
        };
        (base, Cow::Owned(steps))
    }
    /// The result shape that this mapping's steps describe for `items` items, or one axis of `items`.
    fn shape(&self, items: usize) -> Vec<usize> {
        match self { Self::Steps { steps, .. } if Steps::count(steps) == items => steps.iter().map(Steps::len).collect(), _ => vec![items] }
    }
    /// Whether a read through this mapping over a result of `shape` takes each of a source's `len` items once, in order, from the first: a copy through it is the source unchanged, so a view of the source serves. `Linear(1)` answers at once, and any other mapping by its walk ([`Mapping::to_walk`], [`Mapping::walk_in_order`]).
    pub fn in_order(&self, len: usize, shape: &[usize]) -> bool {
        len == count(shape) && (matches!(self, Self::Linear(1)) || { let (base, steps) = self.to_walk(shape); Self::walk_in_order(base, &steps, len) })
    }
    /// Whether a walk from `base` through `steps` takes each of a source's `len` items once, in order, from the first, as [`Mapping::in_order`] asks of a mapping's walk. The walk needs exactly `len` positions, because a walk read past its last position starts again ([`Steps::reach`]).
    pub fn walk_in_order(base: usize, steps: &[Steps], len: usize) -> bool { base == 0 && Steps::count(steps) == len && Steps::in_order(steps) }
    /// Positional steps with the base as a leading singleton step.
    pub fn tables(&self, shape: &[usize]) -> Vec<Steps> { let (base, steps) = self.to_walk(shape); Steps::with_base(base, steps.into_owned()) }
    /// The bytes that this mapping's entries take: each entry, and a table's listed offsets.
    pub fn footprint(&self) -> usize { if let Self::Steps { steps, .. } = self { steps.iter().map(Steps::footprint).sum() } else { 0 } }
    /// Composes a map from frame to shape with this map from shape to storage.
    pub fn compose(&self, outer: &Self, shape: &[usize], frame: &[usize]) -> Option<Self> {
        if frame.contains(&0) { return Some(Self::Linear(1)); }
        if matches!(self, Self::Linear(1)) { return Some(outer.clone()); }
        if matches!(outer, Self::Linear(1)) { return Some(self.clone()); }
        let (base, steps) = match outer { Self::Steps { base, steps } => (*base, steps.to_vec()), _ => outer.entries(frame)? };
        Some(self.through(&Steps::with_base(base, steps), shape)?)
    }
    /// Composes maps, listing offsets when structural composition fails and table is true. The listing walks each map once and looks every position of `outer`'s walk up in `self`'s, where `shape` holds no more items than the result: a `get` divides on every axis for each item, about 400 instructions an item in negative windows' composition with their padding.
    pub fn composed(&self, outer: &Self, shape: &[usize], frame: &[usize], table: bool) -> Option<Self> {
        if let Some(mapping) = self.compose(outer, shape, frame) { return Some(mapping); }
        if !table { return None; }
        let walked = |map: &Self, shape: &[usize]| {
            let (base, steps) = map.to_walk(shape);
            Steps::combined(base, &steps[..])
        };
        let inner = (count(shape) <= count(frame)).then(|| walked(self, shape));
        let at = |j: usize| match &inner {
            _ if j == Steps::FILL => j, Some(listed) => listed.get(j).copied().unwrap_or(Steps::FILL), None => self.get(j).unwrap_or(Steps::FILL)
        };
        Some(Self::Steps { base: 0, steps: vec![Steps::table(walked(outer, frame).into_iter().map(at).collect())].into() })
    }
    fn bounds(&self, shape: &[usize]) -> Option<Range<usize>> { let (base, strides) = self.affine(shape)?; affine_bounds(base, &strides, shape) }
    /// First item offset and signed item strides, where the mapping is affine over `shape`: each axis reads one regular run with no fill, as [`Steps::affine`] decides.
    pub fn affine(&self, shape: &[usize]) -> Option<(usize, Vec<isize>)> {
        // Reject non-affine entries before `per_axis` clones their tables. Singleton offsets and empty shapes remain affine.
        if !shape.contains(&0) && let Self::Steps { steps, .. } = self && steps.iter().any(|s| s.len() > 1 && s.affine().is_none()) { return None; }
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
            Self::Linear(0) | Self::Tiled(0) => None, Self::Single => Some(0), Self::Linear(1) => Some(i), Self::Linear(repeat) => Some(i / repeat),
            Self::Tiled(size) => Some(i % size), Self::Steps { base, steps } => Self::reached(*base, steps, i)
        }
    }
    #[inline(never)]
    fn reached(base: usize, steps: &[Steps], i: usize) -> Option<usize> { Steps::reach(base, steps, i) }
    /// For a contraction such as a matrix product: a one-axis mapping from each key on `left`'s axis `laxis` to its position on `right`'s axis `raxis`, as a table. Where the keys match, or either axis has none, positions pair in order.
    pub fn contract(left: &Layout, laxis: usize, right: &Layout, raxis: usize) -> Result<Self, Error> {
        match (left.keys(laxis), right.keys(raxis)) {
            (Some(x), Some(y)) if x != y => Ok(Self::Steps { base: 0, steps: vec![Steps::table(y.positions(x, false)?)].into() }), _ => Ok(Self::Linear(1))
        }
    }
    /// How `input` lays out over `result`, which has `count` items. Axis `a` of `input` is axis `axes[a]` of `result`.
    pub fn new(input: &Layout, axes: &[usize], result: &Layout, count: usize) -> Self {
        let alignments = axes.iter().enumerate().map(|(a, &b)| match (input.keys(a), result.keys(b)) {
            (Some(src), Some(dst)) => src.aligned(dst), _ => Alignment::Same
        }).collect();
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
                Alignment::Table(t) => {
                    let table = Steps::table(t.into_iter().map(|p| p.map_or(Steps::FILL, |p| p * source)).collect());
                    table.even().unwrap_or(table)
                }
            };
            source *= len;
        }
        Self::Steps { base: 0, steps: steps.into() }
    }
    /// The fewest items in a block of a read through this mapping, where its innermost axis skips items, as a transpose's does: as many whole rows as a cache line holds items of `size` bytes, up to `LINE_BLOCK` items. The rows of such a block then share each fetch of a line. 0 where the innermost axis reads adjacent items or one item.
    pub fn line_block(&self, size: usize) -> usize {
        match self.inner() { Some(inner) if inner.skips() => (inner.len() * (LINE / size).max(1)).min(LINE_BLOCK), _ => 0 }
    }
    /// What this mapping's positions read again, and over how many positions (`Period`): each item repeated (`repeats`), `Tiled`'s cycle, the cycle of a `Steps` mapping whose outer axes step by 0, as a broadcast along outer axes reads, or the cycle of one entry that wraps its whole source from the first item, as a reshape reads. `None` for one item, adjacent items, a mapping that fills, and other steps.
    pub fn period(&self) -> Option<Period<'_>> {
        if let Self::Tiled(m) = *self { return Some(Period::Cycles(m)); }
        if let Some(r) = self.repeats() { return Some(Period::Repeats(r)); }
        let Self::Steps { base, steps } = self else { return None };
        if let [Steps::Padded { start: 0, len, size, stride: 1, edge: crate::structure::Edge::Wrap }] = steps[..] && *base == 0 && size >= 2 && len > size {
            return Some(Period::Cycles(size));
        }
        let k = steps.iter().position(|s| !matches!(*s, Steps::Stride { step: 0, .. }))?;
        let (outer, inner) = steps.split_at(k);
        let p = inner.iter().map(Steps::len).product::<usize>();
        if p < 2 || outer.iter().all(|s| s.len() < 2) || self.fills() { return None; }
        let base = base + outer.iter().map(|s| if let Steps::Stride { start, .. } = *s { start } else { 0 }).sum::<usize>();
        Some(if base == 0 && Steps::in_order(inner) { Period::Cycles(p) } else { Period::Walks { len: p, base, steps: inner } })
    }
    /// Whether `expanded` copies most blocks of `block` items read through this mapping, rather than reading them where they lie: steps whose runs of adjacent offsets (`Steps::contiguous`) are shorter than a block, as a transpose's, a reversal's, windows' and a column slice's are. Repeats and cycles don't count: a run that stays inside one repeat or cycle reads in place, and the caller decides where runs end.
    pub fn copies(&self, block: usize) -> bool { match self { Self::Steps { steps, .. } => Steps::contiguous(steps) < block, _ => false } }
    /// Whether a walk of blocks of `block` items through this mapping reads each item from its own cache line, where a walk of the whole copy reads them through the tiled transpose (`Steps::transposing`): after `simplify`, the last two axes form a transpose whose inner axis holds a block or more, so that each block walks one strided run. A copy made whole then costs much less than reads through it a block at a time.
    pub fn tiles_whole(&self, block: usize) -> bool {
        let Self::Steps { steps, .. } = self else { return false };
        let (_, steps) = Steps::simplify(0, steps);
        let [.., outer, inner] = &steps[..] else { return false };
        Steps::transposing(outer, inner).is_some_and(|(_, _, n, _)| n >= block)
    }
    /// The innermost stretch of positions that read at one regular stride, as its length and that stride: `(r, 0)` for `Linear(r)`, each of whose stretches reads one item, and `Steps::regular` for steps.
    pub fn regular(&self) -> Option<(usize, isize)> {
        match self { Self::Linear(r) => Some((*r, 0)), Self::Steps { steps, .. } => Some(Steps::regular(steps)), _ => None }
    }
    /// The number of positions over which this mapping repeats each item: the length of its innermost regular stretch (`regular`) where that steps by 0, as `Linear(r)` does and a broadcast along inner axes does. `None` for one item, adjacent items, cycles and other steps.
    pub fn repeats(&self) -> Option<usize> { self.regular().and_then(|(len, step)| (step == 0 && len > 1).then_some(len)) }
    /// The runs of this mapping's innermost regular stretch (`regular`) over `positions` positions, where every run reads its items in place: the stretch's length and stride, and the offset of each run's first item, in the order of the runs. A `Linear(r)` mapping's runs read its items in order. `None` where the mapping isn't linear or steps, gives a fill, or reads its stretch through a padded or rotated axis.
    pub fn runs(&self, positions: usize) -> Option<(usize, isize, Vec<usize>)> {
        let (base, steps) = match self {
            Self::Linear(r) => return Some((*r, 0, (0..positions.div_ceil(*r)).collect())), Self::Steps { base, steps } => (base, steps), _ => return None
        };
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
    /// Calls `f(at, len, run)` for consecutive spans of result positions `range`, in order, until it gives `false`: `run` holds the first position and stride of a span whose positions all lie below `items`, and is `None` for a span whose positions all miss. A `Steps` mapping gives the spans of each row of its innermost axis (`row_spans`), a repeat a span for each item, and a cycle a span for each pass.
    pub fn spans(&self, range: Range<usize>, items: usize, f: &mut dyn FnMut(usize, usize, Option<(usize, isize)>) -> bool) {
        let mut at = range.start;
        match *self {
            Self::Single => { f(at, range.len(), (items > 0).then_some((0, 0))); }
            Self::Linear(0) | Self::Tiled(0) => { f(at, range.len(), None); }
            Self::Linear(r) => {
                while at < range.end {
                    let (k, end) = (at / r, range.end.min((at / r + 1) * r));
                    if !f(at, end - at, (k < items).then_some((k, 0))) { return; }
                    at = end;
                }
            }
            Self::Tiled(m) => {
                while at < range.end {
                    let (p, end) = (at % m, range.end.min(at - at % m + m));
                    let below = items.saturating_sub(p).min(end - at);
                    if below > 0 && !f(at, below, Some((p, 1))) { return; }
                    if end > at + below && !f(at + below, end - at - below, None) { return; }
                    at = end;
                }
            }
            Self::Steps { base, ref steps } => {
                let Some((inner, outer)) = steps.split_last() else { f(at, range.len(), (base < items).then_some((base, 0))); return; };
                // A range that steps evenly across rows is one span.
                if let Some((first, step)) = Steps::run(base, steps, at, range.len())
                    && first.max(first.wrapping_add_signed(step * (range.len() as isize - 1))) < items { f(at, range.len(), Some((first, step))); return; }
                let width = inner.len();
                while at < range.end && width > 0 {
                    let (row, j) = (at / width, at % width);
                    let n = (width - j).min(range.end - at);
                    let go = match Steps::reach(base, outer, row) { Some(start) => row_spans(start, inner, j, n, items, at, f), None => f(at, n, None) };
                    if !go { return; }
                    at += n;
                }
            }
        }
    }

    /// The innermost axis of a `Steps` mapping that reaches more than one offset.
    fn inner(&self) -> Option<&Steps> { let Self::Steps { steps, .. } = self else { return None }; steps.iter().rev().find(|s| s.len() > 1) }
}

/// The share of an array's items, 1 in `LISTED`, at or below which a read is small. A small read lists its items' offsets through a mapping that it doesn't compose with, as [`Mapping::composed`]'s table does. Listing one item's offset costs about 50 times what a copy spends on each item. A larger read gathers the whole array first.
pub const LISTED: usize = 128;
/// Whether a read of `read` items from an array of `len` items is small (`LISTED`).
pub fn small_read(read: usize, len: usize) -> bool { read.saturating_mul(LISTED) <= len }

/// The length of a result axis that pairs an axis of length `x` with one of length `y`: their length where they agree, or the other's where one of them is 1. A Length error where they differ and neither is 1.
pub fn paired_length(x: usize, y: usize) -> Result<usize, Error> {
    if x == y || y == 1 { Ok(x) } else if x == 1 { Ok(y) } else { Err(Error::new(Kind::Length, format!("lengths {x} and {y} don't agree"))) }
}

/// Two arguments paired by broadcasting: the result's layout, its number of items, and how each argument lays out over it. A caller that adds its own data to layouts can carry its own layout type, through `map_layout`.
#[derive(Clone)]
pub struct Agreement<L = Layout> { pub layout: L, pub len: usize, pub left: Mapping, pub right: Mapping }

impl Agreement {
    /// Pairs the axes of `left` and `right`: axes with the same name first, then the rest from the front. Where every axis of `left` pairs and `right` has more axes, the result takes `right`'s axis order; otherwise `left`'s axes come first, then those only `right` has. A Length error where paired axes have different lengths and neither is 1.
    pub fn new(left: &Layout, right: &Layout) -> Result<Self, Error> {
        // Plain layouts of one shape, or a plain scalar beside a plain array, need no axis maps. Each side reads item i, or its one item.
        if left.plain() && right.plain() && (left.shape() == right.shape() || left.shape().is_empty() || right.shape().is_empty()) {
            let shape = if left.shape().is_empty() { right.shape() } else { left.shape() };
            let len = generated_len(shape)?;
            let map = |l: &Layout| if len > 1 && l.shape() == shape { Mapping::Linear(1) } else { Mapping::Single };
            return Ok(Self { layout: Layout::from(shape.to_vec()), len, left: map(left), right: map(right) });
        }
        let [xa, ya] = Self::axis_pairs(left, right);
        Self::mapped(left, right, &xa, &ya, false)
    }
    pub fn axis_pairs(left: &Layout, right: &Layout) -> [Vec<usize>; 2] {
        let (mut ya, nx) = (Self::paired_axes(left, right, true), left.shape().len());
        if ya.len() > nx && !ya[..nx].iter().copied().eq(0..nx) && let Some(xa) = (0..nx).map(|a| ya.iter().position(|&b| b == a)).collect::<Option<Vec<_>>>() {
            ya.iter_mut().enumerate().for_each(|(b, a)| *a = b);
            return [xa, ya];
        }
        [(0..nx).collect(), ya]
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
            for (a, b) in remaining_left.into_iter().zip(remaining_right) {
                if !names_conflict(left.name(a).map(|n| &**n), right.name(b).map(|n| &**n)) { ya[b] = Some(a); }
            }
        }
        let mut next = nx;
        ya.into_iter().map(|a| { a.unwrap_or_else(|| { next += 1; next - 1 }) }).collect()
    }
    /// Pairs each axis of `right` with the axis of `left` that `axes` gives, where an axis past `left`'s is a new one.
    pub fn paired(left: &Layout, right: &Layout, axes: &[usize]) -> Result<Self, Error> {
        Self::mapped(left, right, &(0..left.shape().len()).collect::<Vec<_>>(), axes, false)
    }
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
    pub fn mapped(left: &Layout, right: &Layout, xa: &[usize], ya: &[usize], left_keys: bool) -> Result<Self, Error> {
        if xa.len() != left.shape().len() || ya.len() != right.shape().len() { return Err(Error::new(Kind::Length, "axis maps must match input ranks")); }
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
                let key = if left_keys { rt[b] = ky.aligned(kx); kx.clone() } else {
                    let (key, [l, r]) = kx.union(ky)?;
                    (lt[a], rt[b]) = (l, r);
                    key
                };
                shape[axis] = key.len();
                keys[axis] = Some(key);
            } else {
                let n = paired_length(nx, ny)?;
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
    pub fn map_layout<M>(self, f: impl FnOnce(L) -> M) -> Agreement<M> {
        Agreement { layout: f(self.layout), len: self.len, left: self.left, right: self.right }
    }
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
            if chain.iter().all(|m| *m == Mapping::Linear(1)) { return Some(None); }
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

/// The shortest cycle that reads in place a period at a time. A shorter cycle is written out once (`Period::tiling`, `tiled_cycle`), so that runs a chunk long read it in place at any phase. On comparisons of two cycles of 65,536 characters, writing the cycle out cost 61-65% less than the earlier reads at 16 items and below, 43% less at 43, 5% less at 128 and 2% less at 256, then 11%, 24% and 32% more than reading a period at a time at 512, 1,000 and 2,048.
const LONG_CYCLE: usize = 256;
/// Bytes in a cache line on the Apple processors that rustymath is tuned on. An x86-64 line holds half as many.
pub(crate) const LINE: usize = 128;
/// The most items in a block of whole rows from [`Mapping::line_block`].
const LINE_BLOCK: usize = 1 << 16;

/// The one item that every position in `start..start + n` reads, where `mapping` repeats a single item over all of them. `None` where the positions read more than one item.
fn single<A: Copy>(mapping: &Mapping, data: &[A], start: usize, n: usize) -> Option<A> {
    match *mapping {
        Mapping::Single => Some(data[0]), Mapping::Linear(r) if r != 1 && n <= r - start % r => Some(data[start / r]),
        Mapping::Steps { .. } => mapping.run(start, n).and_then(|(o, step)| (step == 0).then(|| data[o])), _ => None
    }
}

/// Items `start..start + n` of an argument that `mapping` lays out, read as `expanded` reads them but always as a slice of all `n` items. A repeated item fills `buf`.
pub fn expanded_items<'a, A: Element>(mapping: &Walked, data: &'a [A], start: usize, n: usize, buf: &'a mut Vec<A>) -> &'a [A] {
    if let Some(x) = single(mapping, data, start, n) {
        buf.clear();
        buf.resize(n, x);
        return buf;
    }
    match expanded(mapping, data, start, n, buf) { Arg::Many(items) => items, Arg::One(_) => unreachable!("`single` gives every repeated item") }
}

/// Items `start..start + n` of an argument that `mapping` lays out, where they need no copy: a single item, a contiguous run, items within one repeat, or items of a cycle: within one period of a mapping that cycles (`Period::Cycles`), and as far as `data` holds a `Tiled` cycle, once or written out past its end (`tiled_cycle`). A `Steps` mapping reads a forward run in place through `run`.
pub fn in_place<'a, A: Copy>(mapping: &Walked, data: &'a [A], start: usize, n: usize) -> Option<Arg<'a, A>> {
    if let Mapping::Steps { .. } = **mapping {
        // One check of the walk finds both a repeated item and a forward run.
        return match mapping.run(start, n) {
            Some((o, 0)) => Some(Arg::One(data[o])), Some((o, 1)) => Some(Arg::Many(&data[o..o + n])),
            _ => mapping.cycle().filter(|&p| start % p + n <= p).map(|p| Arg::Many(&data[start % p..start % p + n]))
        };
    }
    if let Some(x) = single(mapping, data, start, n) { return Some(Arg::One(x)); }
    match **mapping {
        Mapping::Linear(1) => Some(Arg::Many(&data[start..start + n])),
        Mapping::Tiled(m) if start % m + n <= data.len() => Some(Arg::Many(&data[start % m..start % m + n])), _ => None
    }
}

/// Items `start..start + n` of an argument that `mapping` lays out over the frame, read `in_place` where they can be. Others copy into `buf`. A repeated item fills each of its periods. A cycled argument copies through `gather::cycled`, and a `Steps` mapping through `walk_plan`, over the walk that its reader keeps (`Walked`).
pub fn expanded<'a, A: Element>(mapping: &Walked, data: &'a [A], start: usize, n: usize, buf: &'a mut Vec<A>) -> Arg<'a, A> {
    if let Some(arg) = in_place(mapping, data, start, n) { return arg; }
    Arg::Many(copied(mapping, data, start, n, buf))
}

/// The cycle that `mapping`'s positions read from `data`, written into `buf` as `Period::tiling` says for runs of up to `run` positions, with the `Tiled` mapping that then reads it in place: a cycle read in order copies its first period, and one that walks entries (`Period::Walks`) gathers its period through them. A cycle shorter than its cover is written out past its end, so that a run reads it in place at any phase. `None` where `tiling` gives none, leaving `buf` alone.
pub fn tiled_cycle<A: Element>(data: &[A], mapping: &Mapping, run: usize, buf: &mut Vec<A>) -> Option<Mapping> {
    let period = mapping.period()?;
    let (p, cover) = period.tiling(run)?;
    match period {
        Period::Walks { base, steps, .. } => { copied(&Walked::from(Mapping::Steps { base, steps: steps.to_vec().into() }), data, 0, p, buf); }
        _ => {
            buf.clear();
            buf.extend_from_slice(&data[..p]);
        }
    }
    buf.truncate(p);
    if p < cover { *buf = buf.repeat(cover.div_ceil(p) + 1); }
    Some(Mapping::Tiled(p))
}

/// Items `start..start + n` that `mapping` lays out, copied into `buf`, cleared first: what `expanded` gives where `in_place` refuses them. A caller that has just had `in_place` refuse them copies here directly, and the copy records that refusal.
pub fn copied<'a, A: Element>(mapping: &Walked, data: &[A], start: usize, n: usize, buf: &'a mut Vec<A>) -> &'a [A] {
    buf.clear();
    let reason = copy_walked(mapping, data, start, n, A::FILL, buf);
    crate::events::declined(crate::events::Route::InPlace, reason, n);
    &buf[..n]
}

/// The one walk of a mapping's items: appends items `start..start + n` that `mapping` lays out over `data` to `buf`, with `fill` where a position misses, and gives the reason that a view couldn't serve them. It's out of line, so it compiles once for each item type rather than in each caller.
#[inline(never)]
fn copy_walked<A: Element>(mapping: &Walked, data: &[A], start: usize, n: usize, fill: A, buf: &mut Vec<A>) -> &'static str {
    let at = buf.len();
    match **mapping {
        _ if n == 0 => "no items",
        Mapping::Single => { buf.resize(at + n, data[0]); "a repeated item" }
        Mapping::Linear(r) => {
            // The range can start and end inside an item's run. The whole runs between go through `repeat`.
            let (k0, end) = (start / r, start + n);
            buf.resize(at + ((k0 + 1) * r).min(end) - start, data[k0]);
            if buf.len() - at < n {
                let (k1, k2) = (k0 + 1, end / r);
                crate::gather::repeat(buf, k2 - k1, &[r as i64], (k2 - k1) * r, |j| data[k1 + j]);
                if end % r != 0 { buf.resize(at + n, data[k2]) }
            }
            "repeats of each item"
        }
        Mapping::Tiled(m) => { cycled(buf, &data[..m], start, n); "a cycle that wraps" }
        Mapping::Steps { base, .. } => {
            let (offset, plan) = mapping.walk();
            walk_plan(buf, data, base + offset, plan, fill, start, n, Same);
            "steps other than a forward run"
        }
    }
}
