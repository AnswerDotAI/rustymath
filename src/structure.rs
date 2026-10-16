//! Plans for structural operations: take and drop, rotate and reverse, windows, transpose and catenate. A plan gives the result's layout, and `Steps` that read the result's items from the argument's items in order. `gather::walk` copies a compact array by a plan, and basedpl copies its other arrays by the same plan. Each plan checks its arguments.
use crate::error::{Error, Kind};
use crate::gather::Steps;
use crate::layout::{agreed, generated_len, position, strides, Keys, Lanes, Layout};
use std::sync::Arc;

/// A structural result's layout, and the steps that read its items from the argument. The steps' lengths multiply to the result's item count, outermost first. One entry can cover several result axes.
#[derive(Clone, Debug)]
pub struct Plan { pub layout: Layout, pub steps: Vec<Steps> }

/// One independent axis selector. Positions are zero-based and already normalized; a range can walk backwards. An index array replaces its axis by the axes of its layout, and a position of `None` reads the caller's fill. `NewAxis` inserts a length-one axis without consuming an input axis.
#[derive(Clone, Debug)]
pub enum Selector { At(usize), Range { start: usize, len: usize, step: isize }, Indices { layout: Layout, positions: Vec<Option<usize>> }, NewAxis }

impl Selector {
    /// The whole of an axis of length `len`.
    pub fn all(len: usize) -> Self { Selector::Range { start: 0, len, step: 1 } }
}

/// Selects each axis independently, keeping unmentioned trailing axes. Scalar selectors remove their axis. An index array with axis names gives the result its own axes, with their names and any keys it has. A named vector without keys keeps the selected positions' keys. An axis whose name an earlier index array's axis has pairs with that axis, as named axes pair in arithmetic, and both read one position along it. An unnamed vector keeps the selected positions' keys and the axis name, and an unnamed index array of another rank has plain axes.
pub fn select(layout: &Layout, selectors: &[Selector]) -> Result<Plan, Error> {
    let (shape, stride) = (layout.shape(), strides(layout.shape()));
    // Each selector's steps, scaled to the argument's items, with the result axes they run along.
    let (mut axis, mut result, mut parts, mut paired) = (0, Layout::EMPTY, Vec::<(Steps, Vec<usize>)>::new(), false);
    let mut named: Vec<(Arc<str>, usize)> = Vec::new();
    for selector in selectors {
        let rank = result.shape().len();
        if matches!(selector, Selector::NewAxis) {
            result = result.concat(&Layout::from([1]));
            parts.push((Steps::along(1, 0), vec![rank]));
            continue;
        }
        let Some(&size) = shape.get(axis) else { return Err(Error::new(Kind::Index, "too many axis selectors")) };
        let positions = match selector {
            &Selector::At(i) => Steps::Stride { start: i, len: 1, step: 0 },
            &Selector::Range { start, len, step } => Steps::Stride { start, len, step },
            Selector::Indices { layout: within, positions } => {
                if generated_len(within.shape())? != positions.len() { return Err(Error::new(Kind::Length, "index array shape doesn't fit its positions")); }
                Steps::Table(positions.clone())
            }
            Selector::NewAxis => unreachable!(),
        };
        let valid = match &positions {
            Steps::Stride { start, len, step } => *len == 0 || *start < size && start.checked_add_signed(step.saturating_mul((*len - 1) as isize)).is_some_and(|i| i < size),
            _ => positions.offsets().all(|i| i.is_none_or(|i| i < size)),
        };
        if !valid { return Err(Error::new(Kind::Index, "axis selector is outside its axis")); }
        let mut along = Vec::new();
        if !matches!(selector, Selector::At(_)) {
            // Check selected keys even when the result drops them, as BPL selection does.
            let selected = layout.select(axis, positions.offsets())?;
            match selector {
                Selector::Indices { layout: within, .. } if within.names().iter().any(Option::is_some) => {
                    for a in 0..within.shape().len() {
                        // A vector without keys of its own keeps the keys at the positions it selects, as an unnamed vector does.
                        let mine = if within.shape().len() == 1 && within.keys(0).is_none() { within.clone().with_keys(vec![selected.keys(axis).cloned()])? } else { within.axes(a..a + 1) };
                        let Some(&(_, r)) = within.name(a).and_then(|n| named.iter().find(|(m, _)| m == n)) else {
                            let r = result.shape().len();
                            if let Some(n) = within.name(a) { named.push((n.clone(), r)); }
                            result = result.concat(&mine);
                            along.push(r);
                            continue;
                        };
                        let (keys, theirs) = (result.keys(r), mine.keys(0));
                        if result.shape()[r] != within.shape()[a] || keys.zip(theirs).is_some_and(|(k, t)| k != t) {
                            return Err(Error::new(Kind::Length, "index arrays disagree on an axis they both name"));
                        }
                        if keys.is_none() && theirs.is_some() {
                            let mut all = result.all_keys();
                            all[r] = theirs.cloned();
                            result = result.with_keys(all)?;
                        }
                        along.push(r);
                        paired = true;
                    }
                }
                Selector::Indices { layout: within, .. } if within.shape().len() != 1 => {
                    along.extend(rank..rank + within.shape().len());
                    result = result.concat(&Layout::from(within.shape().to_vec()));
                }
                _ => {
                    along.push(rank);
                    result = result.concat(&selected.axes(axis..axis + 1));
                }
            }
        }
        let s = stride[axis];
        parts.push((match positions {
            Steps::Stride { start, len, step } => Steps::Stride { start: start * s, len, step: step * s as isize },
            Steps::Table(mut positions) => { for i in positions.iter_mut().flatten() { *i *= s; } Steps::Table(positions) }
            _ => unreachable!("selectors use ranges or explicit positions"),
        }, along));
        axis += 1;
    }
    let mut steps = if paired { vec![Steps::Table(paired_offsets(result.shape(), &parts)?)] } else { parts.into_iter().map(|(s, _)| s).collect() };
    result = result.concat(&layout.axes(axis..shape.len()));
    steps.extend((axis..shape.len()).map(|a| Steps::along(shape[a], stride[a])));
    generated_len(result.shape())?;
    Ok(Plan { layout: result, steps })
}

/// The argument's offset for each item of the selected axes, of lengths `dims`, as one table. Each item sums one offset from each of `parts`, read at the coordinates of the result axes that part runs along. A part's fill makes the item a fill.
fn paired_offsets(dims: &[usize], parts: &[(Steps, Vec<usize>)]) -> Result<Vec<Option<usize>>, Error> {
    let len = generated_len(dims)?;
    let (mut coords, mut table) = (vec![0; dims.len()], Vec::with_capacity(len));
    for _ in 0..len {
        table.push(parts.iter().try_fold(0, |sum, (steps, along)| steps.get(along.iter().fold(0, |j, &r| j * dims[r] + coords[r])).map(|o| sum + o)));
        for r in (0..dims.len()).rev() {
            coords[r] += 1;
            if coords[r] < dims[r] { break; }
            coords[r] = 0;
        }
    }
    Ok(table)
}

/// [`select`] with `selector` on `axis` and every other axis whole. A `NewAxis` goes in before `axis`.
pub fn select_axis(layout: &Layout, axis: usize, selector: Selector) -> Result<Plan, Error> {
    let selectors: Vec<_> = layout.shape().iter().take(axis).map(|&len| Selector::all(len)).chain([selector]).collect();
    select(layout, &selectors)
}

/// The positions that index values give along an axis of length `size`. Without axis names, a negative value counts back from the end, and a value outside the axis is an Index error. A named index array reads the caller's fill outside its axis instead, as `None`, and never counts back from the end.
pub fn index_positions(values: impl IntoIterator<Item = i64>, size: usize, named: bool) -> Result<Vec<Option<usize>>, Error> {
    values.into_iter().map(|n| if named { Ok(usize::try_from(n).ok().filter(|&i| i < size)) } else {
        position(n, size).map(Some).ok_or_else(|| Error::new(Kind::Index, format!("position {n} is outside an axis of length {size}")))
    }).collect()
}

/// Repeats each cell along `axis` by its count: one nonnegative count, or one per position. Repeating a keyed position is an error, as it is for selection.
pub fn repeat(layout: &Layout, axis: usize, counts: &[i64]) -> Result<Plan, Error> {
    let lanes = Lanes::new(layout.shape(), axis)?;
    if counts.len() != 1 && counts.len() != lanes.len { return Err(Error::new(Kind::Length, "repeat needs one count or one per axis position")); }
    let count = |i: usize| usize::try_from(counts[if counts.len() == 1 { 0 } else { i }]).map_err(|_| Error::new(Kind::Domain, "repeat counts must be nonnegative"));
    if counts.len() == 1 { count(0)?; }
    let len = (0..lanes.len).try_fold(0usize, |n, i| n.checked_add(count(i)?).ok_or_else(|| Error::new(Kind::Limit, "repeat length overflow")))?;
    let mut shape = layout.shape().to_vec();
    shape[axis] = len;
    generated_len(&shape)?;
    if counts.len() == 1 && layout.keys(axis).is_none() {
        let result = Layout::from(shape).with_keys(layout.all_keys())?.inherit_names(layout.names().to_vec());
        return Ok(Plan { layout: result, steps: vec![Steps::along(lanes.outer, lanes.len * lanes.inner), Steps::along(lanes.len, lanes.inner), Steps::along(count(0)?, 0), Steps::along(lanes.inner, 1)] });
    }
    let positions = (0..lanes.len).flat_map(|i| std::iter::repeat_n(Some(i), count(i).expect("validated counts"))).collect::<Vec<_>>();
    let result = layout.select(axis, positions.iter().copied())?;
    Ok(Plan { layout: result, steps: vec![Steps::along(lanes.outer, lanes.len * lanes.inner), Steps::Table(positions.into_iter().map(|i| i.map(|i| i * lanes.inner)).collect()), Steps::along(lanes.inner, 1)] })
}

/// Pads every axis by `(before, after)` positions, filled with the item type's fill. Existing axis keys remain on their positions.
pub fn pad(layout: &Layout, widths: &[(usize, usize)]) -> Result<Plan, Error> {
    if widths.len() != layout.shape().len() { return Err(Error::new(Kind::Rank, "padding needs one pair for each axis")); }
    let stride = strides(layout.shape());
    let mut result = layout.clone();
    let mut steps = Vec::new();
    for (axis, (&size, &(before, after))) in layout.shape().iter().zip(widths).enumerate() {
        let len = size.checked_add(before).and_then(|n| n.checked_add(after)).ok_or_else(|| Error::new(Kind::Limit, "padding length overflow"))?;
        let positions = Steps::Clipped { start: -(before as i128), len, size, stride: 1 };
        result = result.select(axis, positions.offsets())?;
        steps.push(Steps::Clipped { start: -(before as i128), len, size, stride: stride[axis] });
    }
    generated_len(result.shape())?;
    Ok(Plan { layout: result, steps })
}

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

/// Where diagonal `offset` of a matrix of `rows` by `cols` starts, as a row and a column, and its length. A positive offset starts further along the columns. A negative one starts further down the rows.
pub fn diagonal_span(offset: i64, [rows, cols]: [usize; 2]) -> (usize, usize, usize) {
    let (row, col) = if offset < 0 { (offset.unsigned_abs().min(rows as u64) as usize, 0) } else { (0, (offset as u64).min(cols as u64) as usize) };
    (row, col, (rows - row).min(cols - col))
}

/// A diagonal of two distinct axes, moved to the last result axis. Positive offsets start further along `axes[1]`; negative ones along `axes[0]`.
pub fn diagonal(layout: &Layout, axes: [usize; 2], offset: i64) -> Result<Plan, Error> {
    let [a, b] = axes;
    let shape = layout.shape();
    if a == b || a >= shape.len() || b >= shape.len() { return Err(Error::new(Kind::Index, "a diagonal needs two distinct axes")); }
    let (row, col, n) = diagonal_span(offset, [shape[a], shape[b]]);
    let selected = layout.select(a, (row..row + n).map(Some))?.select(b, (col..col + n).map(Some))?;
    let remaining: Vec<_> = (0..shape.len()).filter(|&i| i != a && i != b).collect();
    let result: Vec<_> = remaining.iter().map(|&i| shape[i]).chain([n]).collect();
    let sources: Vec<_> = remaining.iter().map(|&i| vec![i]).chain([vec![a, b]]).collect();
    let stride = strides(shape);
    let mut steps: Vec<_> = remaining.iter().map(|&i| Steps::along(shape[i], stride[i])).collect();
    steps.push(Steps::Stride { start: row * stride[a] + col * stride[b], len: n, step: (stride[a] + stride[b]) as isize });
    Ok(Plan { layout: regrouped(&selected, result, &sources, &[])?, steps })
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
