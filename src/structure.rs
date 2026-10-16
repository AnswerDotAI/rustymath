//! Plans for structural operations: take and drop, rotate and reverse, windows, transpose and catenate. A plan gives the result's layout, and `Steps` that read the result's items from the argument's items in order. `gather::walk` copies a compact array by a plan, and basedpl copies its other arrays by the same plan. Each plan checks its arguments.
use crate::error::{Error, Kind};
use crate::gather::Steps;
use crate::layout::{agreed, generated_len, strides, Keys, Lanes, Layout};
use std::sync::Arc;

/// A structural result's layout, and the steps that read its items from the argument. The steps' lengths multiply to the result's item count, outermost first. One entry can cover several result axes.
#[derive(Clone, Debug)]
pub struct Plan { pub layout: Layout, pub steps: Vec<Steps> }

/// Takes `count` items along `axis` for each `(axis, count)` in `counts`, or drops them when `take` is false. A negative count works from the end. A take past an end reads fill there. A later count for the same axis replaces an earlier one.
pub fn take(layout: &Layout, counts: &[(usize, i64)], take: bool) -> Result<Plan, Error> {
    let old = layout.shape();
    if counts.len() > old.len() { return Err(Error::new(Kind::Rank, "counts exceed argument rank")); }
    let (mut shape, mut starts, mut taken) = (old.to_vec(), vec![0i128; old.len()], vec![false; old.len()]);
    for &(axis, count) in counts {
        let Some(&size) = old.get(axis) else { return Err(Error::new(Kind::Domain, "axis is outside array rank")) };
        let n = usize::try_from(count.unsigned_abs()).unwrap_or(usize::MAX);
        shape[axis] = if take { n } else { size.saturating_sub(n) };
        starts[axis] = if take && count < 0 { size as i128 - n as i128 } else if !take && count > 0 { n.min(size) as i128 } else { 0 };
        taken[axis] = true;
    }
    generated_len(&shape)?;
    let positions = |a: usize, stride: usize| Steps::Clipped { start: starts[a], len: shape[a], size: old[a], stride };
    let stride = strides(old);
    let mut result = layout.clone();
    for a in (0..old.len()).filter(|&a| taken[a]) { result = result.select(a, positions(a, 1).offsets())?; }
    Ok(Plan { layout: result, steps: (0..old.len()).map(|a| positions(a, stride[a])).collect() })
}

/// Rotates the items along `axis`, or reverses them when `counts` is `None`. `counts` gives the counts' shape and items: one count for every lane, or one for each lane in the shape of `layout` without `axis`. A lane starts at position `count` of the axis, counted modulo its length. Keys on the axis move with their items when every lane rotates alike, and are dropped otherwise.
pub fn rotate(layout: &Layout, axis: usize, counts: Option<(&[usize], &[i64])>) -> Result<Plan, Error> {
    let shape = layout.shape();
    let lanes = Lanes::new(shape, axis)?;
    let Lanes { outer, len, inner } = lanes;
    let shift = |n: i64| if len == 0 { 0 } else { n.rem_euclid(len as i64) as usize };
    let count = match counts {
        None => None,
        Some((frame, ns)) => {
            if ns.len() != 1 && !frame.iter().eq(shape[..axis].iter().chain(&shape[axis + 1..])) {
                return Err(Error::new(Kind::Length, "rotation counts must match the axis frame"));
            }
            if len != 0 && (ns.is_empty() || ns.iter().any(|&n| shift(n) != shift(ns[0]))) {
                // Each lane has its own count, so the offsets don't separate by axis, and the axis loses its keys.
                let offsets = (0..outer).flat_map(|i| (0..len).flat_map(move |j| (0..inner).map(move |k| Some(lanes.offset(i, (j + shift(ns[i * inner + k])) % len, k)))));
                let mut keys = layout.all_keys();
                keys[axis] = None;
                return Ok(Plan { layout: layout.clone().with_keys(keys)?, steps: vec![Steps::Table(offsets.collect())] });
            }
            Some(shift(ns.first().copied().unwrap_or(0)))
        }
    };
    let source = |j: usize| match count { None => len - 1 - j, Some(s) => (j + s) % len };
    let along = match count {
        None => Steps::Stride { start: len.saturating_sub(1) * inner, len, step: -(inner as isize) },
        Some(shift) => Steps::Rotated { shift, len, stride: inner },
    };
    Ok(Plan { layout: layout.select(axis, (0..len).map(|j| Some(source(j))))?, steps: vec![Steps::along(outer, len * inner), along, Steps::along(inner, 1)] })
}

/// One axis of windows: windows of `size` items, each `step` items after the last. A padded axis has a window centred on every `step`th position, with fill past the ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowAxis { pub size: usize, pub step: usize, pub padded: bool }

impl WindowAxis {
    /// Whether an axis of length `len` can hold this window. A padded size must be less than twice the length.
    pub fn fits(&self, len: usize) -> bool { !self.padded || self.size / 2 < len }
    /// The number of windows along an axis of length `len` that `fits`.
    pub fn count(&self, len: usize) -> usize {
        if self.padded { (len - usize::from(self.size.is_multiple_of(2))).div_ceil(self.step) }
        else if len < self.size { 0 }
        else { (len - self.size) / self.step + 1 }
    }
    /// Where window `i` starts along the axis. It's negative when padding comes before the first item.
    pub fn start(&self, i: usize) -> isize { (i * self.step) as isize - if self.padded { ((self.size - 1) / 2) as isize } else { 0 } }
    /// The positions that every window covers alike, which keep their keys: all of an empty window, or the only window when it needs no padding.
    fn shared(&self, len: usize) -> Option<std::ops::Range<usize>> {
        if self.size == 0 { return Some(0..0); }
        let start = usize::try_from(self.start(0)).ok()?;
        (self.count(len) == 1 && start + self.size <= len).then_some(start..start + self.size)
    }
}

/// Windows over the leading axes of an array of one shape: a `WindowAxis` for each, the frame that counts the windows along each, and the shape of one window.
#[derive(Clone, Debug)]
pub struct Windows { shape: Vec<usize>, stride: Vec<usize>, axes: Vec<WindowAxis>, frame: Vec<usize>, cell: Vec<usize> }

impl Windows {
    /// The windows of `sizes` and `moves` over the leading axes of `shape`. A size's magnitude is the window's size, and `padded` says which sizes pad their axis. An axis with no move moves by 1.
    pub fn new(sizes: &[i64], moves: &[i64], padded: impl Fn(i64) -> bool, shape: &[usize]) -> Result<Self, Error> {
        if sizes.len() > shape.len() { return Err(Error::new(Kind::Rank, "window specification has more axes than its argument")); }
        if moves.len() > sizes.len() { return Err(Error::new(Kind::Length, "window specification has more movements than sizes")); }
        let axes = sizes.iter().zip(shape).enumerate().map(|(a, (&size, &len))| {
            let step = match moves.get(a) {
                None => 1,
                Some(&m) if m > 0 => usize::try_from(m).unwrap_or(usize::MAX),
                Some(_) => return Err(Error::new(Kind::Domain, "window movements must be positive")),
            };
            let axis = WindowAxis { size: usize::try_from(size.unsigned_abs()).unwrap_or(usize::MAX), step, padded: padded(size) };
            if axis.fits(len) { Ok(axis) } else { Err(Error::new(Kind::Domain, "padded window is too large for the argument")) }
        }).collect::<Result<Vec<_>, _>>()?;
        let frame = axes.iter().zip(shape).map(|(w, &n)| w.count(n)).collect();
        let cell = axes.iter().map(|w| w.size).chain(shape[axes.len()..].iter().copied()).collect();
        Ok(Self { shape: shape.to_vec(), stride: strides(shape), axes, frame, cell })
    }
    pub fn axes(&self) -> &[WindowAxis] { &self.axes }
    /// The number of windows along each windowed axis.
    pub fn frame(&self) -> &[usize] { &self.frame }
    /// The shape of one window: the windows' sizes, then the argument's remaining axes.
    pub fn cell(&self) -> &[usize] { &self.cell }
    /// Where the window at `position` of the frame, counted in ravel order, starts along each windowed axis.
    pub fn starts(&self, position: usize) -> Vec<isize> {
        let mut starts = vec![0; self.axes.len()];
        let mut rest = position;
        for a in (0..self.axes.len()).rev() {
            let n = self.frame[a].max(1);
            starts[a] = self.axes[a].start(rest % n);
            rest /= n;
        }
        starts
    }
    /// The fill that the window at `starts` reads along each windowed axis: the count before the first item, positive, or past the last, negative, or 0 where the window fits.
    pub fn border(&self, starts: &[isize]) -> Vec<isize> {
        self.axes.iter().enumerate().map(|(a, w)| if starts[a] < 0 { -starts[a] } else { (self.shape[a] as isize - starts[a] - w.size as isize).min(0) }).collect()
    }
    /// The keys of the frame's axes. A padded axis keeps the key of the position each window centres on, and an unpadded axis has none.
    fn frame_keys(&self, layout: &Layout) -> Result<Vec<Option<Arc<Keys>>>, Error> {
        self.axes.iter().enumerate().map(|(a, w)| match layout.keys(a) {
            Some(k) if w.padded => k.select((0..self.frame[a]).map(|i| Some(i * w.step))).map(Some),
            _ => Ok(None),
        }).collect()
    }
    /// The plan that adds fill around each padded axis, or `None` when no axis pads or there are no windows. Each padded axis of the result holds only the positions its windows read, and the result's unpadded windows are the padded windows of the argument.
    pub fn padding(&self) -> Option<Plan> {
        if !self.axes.iter().any(|w| w.padded) || self.frame.contains(&0) || self.cell.contains(&0) { return None; }
        let steps: Vec<_> = (0..self.shape.len()).map(|a| match self.axes.get(a) {
            Some(w) => Steps::Clipped { start: w.start(0) as i128, len: (self.frame[a] - 1) * w.step + w.size, size: self.shape[a], stride: self.stride[a] },
            None => Steps::along(self.shape[a], self.stride[a]),
        }).collect();
        let padded: Vec<_> = steps.iter().map(Steps::len).collect();
        Some(Plan { layout: padded.into(), steps })
    }
    /// The plan of every window over `layout`, which has the windows' shape: the frame's axes, then each window's axes. The windows read an array of shape `source`: `layout`'s shape, or the shape of `padding`'s result. A window's axis keeps its keys where every window covers the same positions.
    pub fn plan(&self, layout: &Layout, source: &[usize]) -> Result<Plan, Error> {
        let n = self.axes.len();
        let shape = [self.frame.as_slice(), &self.cell].concat();
        generated_len(&shape)?;
        generated_len(&self.cell)?;
        let stride = strides(source);
        let frame = (0..n).map(|a| Steps::Stride { start: 0, len: self.frame[a], step: (self.axes[a].step * stride[a]) as isize });
        let steps = frame.chain(self.cell.iter().zip(&stride).map(|(&len, &stride)| Steps::along(len, stride))).collect();
        let cell_keys = (0..self.cell.len()).map(|a| match self.axes.get(a) {
            None => Ok(layout.keys(a).cloned()),
            Some(w) => layout.keys(a).zip(w.shared(self.shape[a])).map(|(k, r)| k.select(r.map(Some))).transpose(),
        });
        let keys = self.frame_keys(layout)?.into_iter().map(Ok).chain(cell_keys).collect::<Result<_, Error>>()?;
        let names = (0..n).chain(0..self.cell.len()).map(|a| layout.name(a).cloned()).collect();
        Ok(Plan { layout: Layout::from(shape).with_keys(keys)?.inherit_names(names), steps })
    }
    /// The plan of the window that begins at `starts` along the windowed axes of `layout`, as `starts` gives them for a frame position. A windowed axis keeps its keys where the window needs no padding.
    pub fn window(&self, layout: &Layout, starts: &[isize]) -> Result<Plan, Error> {
        let keys = (0..self.cell.len()).map(|a| match (layout.keys(a), starts.get(a)) {
            (Some(k), Some(&s)) => match usize::try_from(s) {
                Ok(s) if s + self.cell[a] <= self.shape[a] => k.select((s..s + self.cell[a]).map(Some)).map(Some),
                _ => Ok(None),
            },
            (k, _) => Ok(k.cloned()),
        }).collect::<Result<_, Error>>()?;
        let steps = (0..self.cell.len())
            .map(|a| Steps::Clipped { start: starts.get(a).map_or(0, |&s| s as i128), len: self.cell[a], size: self.shape[a], stride: self.stride[a] }).collect();
        Ok(Plan { layout: Layout::from(self.cell.clone()).with_keys(keys)?.inherit_names(layout.names().to_vec()), steps })
    }
    /// The layout of the frame over `layout`, with the frame's keys.
    pub fn frame_layout(&self, layout: &Layout) -> Result<Layout, Error> {
        let names = (0..self.frame.len()).map(|a| layout.name(a).cloned()).collect();
        Ok(Layout::from(self.frame.clone()).with_keys(self.frame_keys(layout)?)?.inherit_names(names))
    }
}

/// Sends axis `i` of `layout` to result axis `axes[i]`. Axes sent to the same result axis select their diagonal, at the shortest of their lengths. Every result axis from 0 to the largest destination needs a source. A result axis keeps the keys and name its source axes agree on.
pub fn transpose(layout: &Layout, axes: &[usize]) -> Result<Plan, Error> {
    let source = layout.shape();
    let rank = axes.iter().max().map_or(0, |n| n + 1);
    if (0..rank).any(|r| !axes.contains(&r)) { return Err(Error::new(Kind::Rank, "transpose axes must be consecutive from 0")); }
    if axes.len() != source.len() { return Err(Error::new(Kind::Length, "transpose needs one destination for each axis")); }
    let mut shape = vec![usize::MAX; rank];
    for (&axis, &len) in axes.iter().zip(source) { shape[axis] = shape[axis].min(len); }
    let sources: Vec<Vec<_>> = (0..rank).map(|a| (0..axes.len()).filter(|&i| axes[i] == a).collect()).collect();
    let mut stride = vec![0; rank];
    for (&axis, step) in axes.iter().zip(strides(source)) { stride[axis] += step; }
    let steps = shape.iter().zip(&stride).map(|(&len, &stride)| Steps::along(len, stride)).collect();
    Ok(Plan { layout: regrouped(layout, shape, &sources, &[])?, steps })
}

/// A layout of `shape` whose axis `i` comes from `layout`'s axes `sources[i]`, as a transpose sends them. An axis with no sources is new, as a split or a merge makes it. A result axis keeps the keys its sources agree on when each source has its length. It keeps the name they agree on, unless `names[i]` gives one.
pub fn regrouped(layout: &Layout, shape: Vec<usize>, sources: &[Vec<usize>], names: &[Option<Arc<str>>]) -> Result<Layout, Error> {
    if sources.len() != shape.len() || names.len() > shape.len() { return Err(Error::new(Kind::Length, "each result axis needs one list of sources")); }
    let lengths = layout.shape();
    if sources.iter().flatten().any(|&a| a >= lengths.len()) { return Err(Error::new(Kind::Rank, "a source axis is outside array rank")); }
    let keys = sources.iter().zip(&shape)
        .map(|(s, &len)| if s.iter().all(|&a| lengths[a] == len) { agreed(s.iter().map(|&a| layout.keys(a))) } else { None }).collect();
    let names = sources.iter().enumerate().map(|(i, s)| names.get(i).and_then(Clone::clone).or_else(|| agreed(s.iter().map(|&a| layout.name(a))))).collect();
    Layout::from(shape).with_keys(keys)?.with_names(names)
}

/// The layout of `left` and `right` joined along `axis`. They must have the same rank, and the same lengths on their other axes. The joined axis has both sides' keys, with no key where a side has none, and each other axis has either side's keys. A name survives where both sides agree.
pub fn catenate(left: &Layout, right: &Layout, axis: usize) -> Result<Layout, Error> {
    let rank = left.shape().len();
    if right.shape().len() != rank { return Err(Error::new(Kind::Rank, "catenate ranks differ")); }
    if axis >= rank { return Err(Error::new(Kind::Domain, "catenate axis is outside result rank")); }
    if (0..rank).any(|a| a != axis && left.shape()[a] != right.shape()[a]) { return Err(Error::new(Kind::Length, "catenate frames differ")); }
    let names = |k: Option<&Arc<Keys>>, n: usize| k.map_or_else(|| vec![None; n], |k| k.names().to_vec());
    let keys = (0..rank).map(|a| match (left.keys(a), right.keys(a)) {
        (x, y) if a != axis => Ok(x.or(y).cloned()),
        (None, None) => Ok(None),
        (x, y) => Keys::partial([names(x, left.shape()[a]), names(y, right.shape()[a])].concat())
            .map(Some).map_err(|_| Error::new(Kind::Domain, "catenate has duplicate axis keys")),
    }).collect::<Result<_, Error>>()?;
    let mut shape = left.shape().to_vec();
    shape[axis] = shape[axis].checked_add(right.shape()[axis]).ok_or_else(|| Error::new(Kind::Limit, "catenate axis overflow"))?;
    generated_len(&shape)?;
    let names = (0..rank).map(|a| agreed([left.name(a), right.name(a)])).collect();
    Ok(Layout::from(shape).with_keys(keys)?.inherit_names(names))
}
