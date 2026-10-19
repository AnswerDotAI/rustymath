//! Plans for structural operations: take and drop, rotate and reverse, windows, transpose and catenate. A plan gives the result's layout, and `Steps` that read the result's items from the argument's items in order. `gather::walk` copies a compact array by a plan, and basedpl copies its other arrays by the same plan. Each plan checks its arguments.
use crate::error::{Error, Kind};
use crate::gather::Steps;
use crate::{buffer::View, float::Float, tolerant::{Tolerance, Tolerant}};
use crate::agreement::{Agreement, Mapping};
use crate::layout::{agreed, generated_len, names_conflict, strides, Keys, Lanes, Layout};
use std::sync::Arc;

/// A structural result's layout, and the steps that read its items from the argument. The steps' lengths multiply to the result's item count, outermost first. One entry can cover several result axes.
#[derive(Clone, Debug)]
pub struct Plan { pub layout: Layout, pub steps: Vec<Steps> }

/// One independent axis selector. Positions are zero-based and already normalized; a range can walk backwards. An index array replaces its axis by the axes of its layout. Its positions lie within the axis or are `Steps::FILL`, which reads the caller's fill. The code that builds an index array checks its positions, as `index_positions` does. `NewAxis` inserts a length-one axis without consuming an input axis.
#[derive(Clone, Debug)]
pub enum Selector { At(usize), Range { start: usize, len: usize, step: isize }, Indices { layout: Layout, positions: Vec<usize> }, NewAxis }

impl Selector {
    /// The whole of an axis of length `len`.
    pub fn all(len: usize) -> Self { Selector::Range { start: 0, len, step: 1 } }
}

/// Selects with one selector for each leading axis of `layout`, in order, keeping the axes that no selector mentions. A single position removes its axis, and a range keeps it in place. An index array's axes replace its axis. An unnamed vector keeps the axis's name and the keys at the positions it selects, as a named vector without keys of its own keeps those keys, and an unnamed index array of another rank has plain axes. Index arrays pair, as arguments pair in arithmetic: all of them with `broadcast`, as `.[]` reads them, and otherwise those that share an axis name, as `⌷` reads them. Paired index arrays put their axes where the first of them was. A range pairs with an index array that has its axis's name.
pub fn select(layout: &Layout, selectors: Vec<Selector>, broadcast: bool) -> Result<Plan, Error> {
    let (shape, stride) = (layout.shape(), strides(layout.shape()));
    let mut parts = Vec::with_capacity(shape.len());
    let mut axis = 0;
    for selector in selectors {
        if matches!(selector, Selector::NewAxis) { parts.push(Part::new(Steps::along(1, 0), Layout::from([1]), 0)); continue; }
        if axis == shape.len() { return Err(Error::new(Kind::Index, "too many axis selectors")); }
        parts.push(Part::of(selector, layout, axis, stride[axis])?);
        axis += 1;
    }
    parts.extend((axis..shape.len()).map(|a| Part::new(Steps::along(shape[a], 1), layout.axes(a..a + 1), stride[a])));
    let names: Vec<Arc<str>> = parts.iter().filter(|p| p.array).flat_map(|p| p.axes.names().iter().flatten().cloned()).collect();
    for part in parts.iter_mut().filter(|p| !p.array && p.axes.name(0).is_some_and(|n| names.contains(n))) {
        part.steps = Steps::table(part.steps.offsets().map(|o| o.expect("a range has no fills")).collect());
        part.array = true;
    }
    // The index arrays that pair, each group in order.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let shares = |p: usize, q: usize| parts[p].axes.names().iter().flatten().any(|n| parts[q].axes.names().contains(&Some(n.clone())));
    for p in (0..parts.len()).filter(|&p| parts[p].array) {
        let hits: Vec<usize> = (0..groups.len()).filter(|&g| broadcast || groups[g].iter().any(|&q| shares(p, q))).collect();
        let Some((&first, rest)) = hits.split_first() else { groups.push(vec![p]); continue };
        for &g in rest.iter().rev() { let moved = groups.remove(g); groups[first].extend(moved); }
        groups[first].push(p);
        groups[first].sort_unstable();
    }
    let mut group_of = vec![usize::MAX; parts.len()];
    for (g, members) in groups.iter().enumerate() { for &p in members { group_of[p] = g; } }
    let mut joined = groups.iter().map(|g| if g.len() > 1 { joined(&parts, g, broadcast).map(Some) } else { Ok(None) }).collect::<Result<Vec<_>, _>>()?;
    let (mut result, mut steps) = (Layout::EMPTY, Vec::with_capacity(parts.len()));
    for (p, part) in parts.into_iter().enumerate() {
        if group_of[p] != usize::MAX && groups[group_of[p]].len() > 1 {
            if let Some((axes, table)) = joined[group_of[p]].take() { result = result.concat(&axes); steps.push(table); }
            continue;
        }
        result = result.concat(part.alone.as_ref().unwrap_or(&part.axes));
        steps.push(scaled(part.steps, part.stride));
    }
    generated_len(result.shape())?;
    Ok(Plan { layout: result, steps })
}

/// One selector's share of a plan: positions along its axis, before scaling by the axis's `stride`, and the result axes they run along. An index array that pairs with nothing takes `alone` as its axes, where it has them.
struct Part { steps: Steps, axes: Layout, stride: usize, array: bool, alone: Option<Layout> }

impl Part {
    fn new(steps: Steps, axes: Layout, stride: usize) -> Self { Part { steps, axes, stride, array: false, alone: None } }

    /// The part that `selector` reads along `axis` of `layout`, whose items lie `stride` apart.
    fn of(selector: Selector, layout: &Layout, axis: usize, stride: usize) -> Result<Self, Error> {
        let size = layout.shape()[axis];
        let outside = || Error::new(Kind::Index, "axis selector is outside its axis");
        match selector {
            Selector::At(i) if i < size => Ok(Part::new(Steps::Stride { start: i, len: 1, step: 0 }, Layout::EMPTY, stride)),
            Selector::At(_) => Err(outside()),
            Selector::Range { start, len, step } => {
                if len > 0 && !(start < size && start.checked_add_signed(step.saturating_mul((len - 1) as isize)).is_some_and(|i| i < size)) { return Err(outside()); }
                let steps = Steps::Stride { start, len, step };
                let axes = layout.select(axis, steps.offsets())?.axes(axis..axis + 1);
                Ok(Part::new(steps, axes, stride))
            }
            Selector::Indices { layout: within, positions } => {
                if generated_len(within.shape())? != positions.len() { return Err(Error::new(Kind::Length, "index array shape doesn't fit its positions")); }
                let steps = Steps::table(positions);
                // Check selected keys even when the result drops them, as BPL selection does.
                let selected = layout.select(axis, steps.offsets())?;
                let alone = match within.shape().len() {
                    1 if within.names().is_empty() => Some(selected.axes(axis..axis + 1)),
                    1 if within.keys(0).is_none() => Some(within.clone().with_keys(vec![selected.keys(axis).cloned()])?),
                    _ if within.names().is_empty() => Some(Layout::from(within.shape().to_vec())),
                    _ => None,
                };
                Ok(Part { steps, axes: within, stride, array: true, alone })
            }
            Selector::NewAxis => unreachable!("a new axis reads no positions"),
        }
    }
}

/// The axes that paired index arrays agree on, and the offset at each of their items: the sum of each array's scaled position there. A fill in any array makes a fill.
fn joined(parts: &[Part], group: &[usize], broadcast: bool) -> Result<(Layout, Steps), Error> {
    let mut frame = parts[group[0]].axes.clone();
    let mut pairs = vec![(0..frame.shape().len()).collect::<Vec<_>>()];
    for &p in &group[1..] {
        let axes = Agreement::paired_axes(&frame, &parts[p].axes, broadcast);
        frame = Agreement::paired(&frame, &parts[p].axes, &axes)?.layout;
        pairs.push(axes);
    }
    let len = generated_len(frame.shape())?;
    let mut table = vec![0; len];
    for (&p, axes) in group.iter().zip(&pairs) {
        let Part { steps, axes: own, stride, .. } = &parts[p];
        let add = |t: &mut usize, o: Option<usize>| *t = match o { Some(o) if *t != Steps::FILL => *t + o * stride, _ => Steps::FILL };
        match (Mapping::new(own, axes, &frame, len), steps) {
            (Mapping::Linear(1), Steps::Table(positions)) => for (t, &o) in table.iter_mut().zip(positions.offsets()) { add(t, Some(o).filter(|&o| o != Steps::FILL)) },
            (map, _) => for (b, t) in table.iter_mut().enumerate() { add(t, map.get(b).and_then(|j| steps.get(j))) },
        }
    }
    Ok((frame, Steps::table(table)))
}

/// `positions` along an axis whose items lie `stride` apart, as offsets into the argument.
fn scaled(positions: Steps, stride: usize) -> Steps {
    match positions {
        Steps::Stride { start, len, step } => Steps::Stride { start: start * stride, len, step: step * stride as isize },
        Steps::Table(positions) => Steps::Table(positions.scaled(stride)),
        _ => unreachable!("selectors use ranges or explicit positions"),
    }
}

/// The name of the one named axis of `layout`, where it has exactly one.
pub fn sole_name(layout: &Layout) -> Option<&str> {
    let mut names = layout.names().iter().flatten();
    names.next().filter(|_| names.next().is_none()).map(|n| &**n)
}

/// The first axis of `layout` that each selector addresses, given the name of each selector's one named axis, where it has exactly one, and the number of axes it covers, where more axes than remain cover the rest, as a coordinate past the last axis does. A selector whose name `layout` has addresses the axis with that name, unless another selector has the same name. Selectors with the same name take free axes in order, as unnamed selectors and NumPy's index arrays do. Each other selector takes the first free axes whose names don't conflict with its name. Either way a selector claims every axis it covers. Two selectors that claim one axis, or a selector with no free axis, give an Index error.
pub fn addressed(layout: &Layout, selectors: &[(Option<&str>, usize)]) -> Result<Vec<usize>, Error> {
    let rank = layout.shape().len();
    let (mut free, mut axes) = (vec![true; rank], vec![rank; selectors.len()]);
    let span = |a: usize, width: usize| a..(a + width).min(rank);
    let names: Vec<_> = selectors.iter().map(|&(n, _)| n.filter(|_| selectors.iter().filter(|s| s.0 == n).count() == 1)).collect();
    for (k, &(_, width)) in selectors.iter().enumerate() {
        let Some((n, a)) = names[k].and_then(|n| Some((n, layout.axis_named(n)?))) else { continue };
        if span(a, width).any(|b| !free[b]) { return Err(Error::new(Kind::Index, format!("two selectors address axis {n}"))); }
        free[span(a, width)].fill(false);
        axes[k] = a;
    }
    for (k, &(_, width)) in selectors.iter().enumerate() {
        if axes[k] != rank { continue; }
        let name = names[k];
        let fits = |a: usize| a < rank && span(a, width).all(|b| free[b] && !names_conflict(name, layout.name(b).map(|n| &**n)));
        let missing = || Error::new(Kind::Index, name.map_or_else(|| "too many axis selectors".into(), |n| format!("no axis for a selector named {n}")));
        let a = (0..rank).find(|&a| fits(a)).ok_or_else(missing)?;
        free[span(a, width)].fill(false);
        axes[k] = a;
    }
    Ok(axes)
}

/// [`select`] with `selector` on `axis` and every other axis whole. A `NewAxis` goes in before `axis`.
pub fn select_axis(layout: &Layout, axis: usize, selector: Selector) -> Result<Plan, Error> {
    let selectors: Vec<_> = layout.shape().iter().take(axis).map(|&len| Selector::all(len)).chain([selector]).collect();
    select(layout, selectors, false)
}

/// The positions that index values give along an axis of length `size`. A negative value counts back from the end, and a value outside the axis is an Index error.
pub fn index_positions<I: Copy + Into<i64>>(values: &[I], size: usize) -> Result<Vec<usize>, Error> {
    let mut positions = vec![0; values.len()];
    let mut inside = true;
    for (p, &i) in positions.iter_mut().zip(values) {
        *p = counted_back(i.into(), size);
        inside &= *p < size;
    }
    if inside { return Ok(positions); }
    let k = positions.iter().position(|&p| p >= size).expect("a position is outside the axis");
    Err(outside(Into::<i64>::into(values[k]), size))
}

/// The Index error for `position` on an axis of length `size`.
fn outside(position: impl std::fmt::Display, size: usize) -> Error { Error::new(Kind::Index, format!("position {position} is outside an axis of length {size}")) }

/// Position `i` on an axis of length `size`, where a negative `i` counts back from the end. Adding `size` to a negative position without a branch keeps a loop over positions vectorised. A position outside the axis is then at least `size`.
#[inline(always)]
fn counted_back(i: i64, size: usize) -> usize { (i + (size as i64 & (i >> 63))) as usize }

/// The positions that index values give along an axis of length `size`: integers as `index_positions` reads them, and floats as `filled_positions` does. Other items are a Domain error.
pub fn positions(view: View<'_>, size: usize) -> Result<Vec<usize>, Error> {
    index_view!(view, |ix| index_positions(ix, size), |x| filled_positions(x, size))
}

/// The message of the Domain error for an index value that isn't a whole number.
pub const NOT_WHOLE: &str = "index must be an integer";

/// The Domain error for an index value that isn't a whole number.
fn not_whole() -> Error { Error::new(Kind::Domain, NOT_WHOLE) }

/// `$int` on the items of a Boolean or integer view, `$float` on a float view's items, or the Domain error for any other view.
macro_rules! index_view {
    ($view:expr, |$i:ident| $int:expr, |$f:ident| $float:expr) => {
        match $view {
            View::Booleans($i) => $int,
            View::Integers(ix) => with_ints!(ix, |$i| $int),
            View::Floats(x) => with_floats!(x, |$f| $float),
            _ => Err(not_whole()),
        }
    };
}
use index_view;

/// The whole number that a float index value equals within the default tolerance, or `None` for a NaN, which reads the fill. Any other value is a Domain error.
pub fn float_position<F: Float>(y: F) -> Result<Option<i64>, Error> {
    if y.is_nan() { return Ok(None); }
    match Tolerance::DEFAULT.nearest_whole(y) { (n, true) => Ok(Some(n.wide() as i64)), _ => Err(not_whole()) }
}

/// The positions that float index values give along an axis of length `size`, as `index_positions` gives them from integers, with each value read by `float_position`. A NaN gives `Steps::FILL`, which reads the fill and drops a write.
pub fn filled_positions<F: Float>(values: &[F], size: usize) -> Result<Vec<usize>, Error> {
    // The first loop has no branch, which keeps it vectorised. A value that isn't exactly whole is read again by `float_position`.
    let mut positions = vec![0; values.len()];
    let (mut whole, mut inside) = (true, true);
    for (p, &y) in positions.iter_mut().zip(values) {
        let (x, nan) = (y.wide(), y.is_nan());
        let q = counted_back(x as i64, size);
        whole &= nan | (x as i64 as f64 == x);
        inside &= nan | (q < size);
        *p = if nan { Steps::FILL } else { q };
    }
    if !whole {
        inside = true;
        for (p, &y) in positions.iter_mut().zip(values) {
            if let Some(i) = float_position(y)? { *p = counted_back(i, size); inside &= *p < size; }
        }
    }
    if inside { return Ok(positions); }
    let k = positions.iter().zip(values).position(|(&p, y)| !y.is_nan() && p >= size).expect("a position is outside the axis");
    let x = values[k].wide();
    Err(outside(if x.abs() < 1e16 { x.to_string() } else { format!("{x:e}") }, size))
}

/// The signed position that each index value gives, before any axis applies: integers as they are, and floats as `float_position` reads them, with `None` for a NaN. Other items are a Domain error. [`positions`] reads the values and checks them against an axis in one pass.
pub fn signed_positions(view: View<'_>) -> Result<Vec<Option<i64>>, Error> {
    index_view!(view, |ix| Ok(ix.iter().map(|&i| Some(i.into())).collect()), |x| x.iter().map(|&y| float_position(y)).collect())
}

/// The positions along an axis of length `size` that signed positions give, as `index_positions` gives them, with `Steps::FILL` for `None`.
pub fn counted_positions(values: &[Option<i64>], size: usize) -> Result<Vec<usize>, Error> {
    values.iter().map(|&q| match q.map(|i| (i, counted_back(i, size))) {
        None => Ok(Steps::FILL),
        Some((_, p)) if p < size => Ok(p),
        Some((i, _)) => Err(outside(i, size)),
    }).collect()
}

/// How a position outside an axis maps to one inside it, as edge modes of padding do. `Fill` gives no position, which reads the fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge { Fill, Clamp, Wrap, Reflect, Symmetric }

impl Edge {
    /// Each mode with its name.
    pub const MODES: [(&'static str, Edge); 5] =
        [("fill", Edge::Fill), ("clamp", Edge::Clamp), ("wrap", Edge::Wrap), ("reflect", Edge::Reflect), ("symmetric", Edge::Symmetric)];
    /// The mode named `name`.
    pub fn named(name: &str) -> Option<Edge> { Self::MODES.iter().find(|(n, _)| *n == name).map(|&(_, e)| e) }
    /// This mode's name.
    pub fn name(self) -> &'static str { Self::MODES.iter().find(|&&(_, e)| e == self).expect("every mode has a name").0 }
    /// The position on an axis of `n` items that position `p` reads, or `None` for the fill. A negative `p` is outside the axis here. `Reflect` leaves out the edge item, as NumPy's `reflect` does, so `-1` reads 1. `Symmetric` repeats it, as NumPy's `symmetric` does, so `-1` reads 0. An empty axis gives `None` in every mode.
    pub fn position(self, p: i64, n: usize) -> Option<usize> {
        let n = i64::try_from(n).ok().filter(|&n| n > 0)?;
        let q = match self {
            Edge::Fill => p,
            Edge::Clamp => p.clamp(0, n - 1),
            Edge::Wrap => p.rem_euclid(n),
            Edge::Reflect if n == 1 => 0,
            Edge::Reflect => (n - 1) - ((n - 1) - p.rem_euclid(2 * (n - 1))).abs(),
            Edge::Symmetric => { let r = p.rem_euclid(2 * n); r.min(2 * n - 1 - r) }
        };
        usize::try_from(q).ok().filter(|&q| q < n as usize)
    }
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
    let positions = (0..lanes.len).flat_map(|i| std::iter::repeat_n(i, count(i).expect("validated counts"))).collect::<Vec<_>>();
    let result = layout.select(axis, positions.iter().map(|&i| Some(i)))?;
    Ok(Plan { layout: result, steps: vec![Steps::along(lanes.outer, lanes.len * lanes.inner), Steps::table(positions.into_iter().map(|i| i * lanes.inner).collect()), Steps::along(lanes.inner, 1)] })
}

/// Pads every axis by `(before, after)` positions, filled with the item type's fill. Existing axis keys remain on their positions. An axis that pads by nothing keeps a plain walk.
pub fn pad(layout: &Layout, widths: &[(usize, usize)]) -> Result<Plan, Error> {
    if widths.len() != layout.shape().len() { return Err(Error::new(Kind::Rank, "padding needs one pair for each axis")); }
    let stride = strides(layout.shape());
    let mut result = layout.clone();
    let mut steps = Vec::new();
    for (axis, (&size, &(before, after))) in layout.shape().iter().zip(widths).enumerate() {
        if before == 0 && after == 0 { steps.push(Steps::along(size, stride[axis])); continue; }
        let len = size.checked_add(before).and_then(|n| n.checked_add(after)).ok_or_else(|| Error::new(Kind::Limit, "padding length overflow"))?;
        let positions = Steps::clipped(-(before as i128), len, size, 1);
        result = result.select(axis, positions.offsets())?;
        steps.push(Steps::clipped(-(before as i128), len, size, stride[axis]));
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
    let positions = |a: usize, stride: usize| Steps::clipped(starts[a], shape[a], old[a], stride);
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
                let offsets = (0..outer).flat_map(|i| (0..len).flat_map(move |j| (0..inner).map(move |k| lanes.offset(i, (j + shift(ns[i * inner + k])) % len, k))));
                return Ok(Plan { layout: layout.clone().without_keys(axis), steps: vec![Steps::table(offsets.collect())] });
            }
            Some(shift(ns.first().copied().unwrap_or(0)))
        }
    };
    let source = |j: usize| match count { None => len - 1 - j, Some(s) => (j + s) % len };
    let along = match count {
        None => Steps::Stride { start: len.saturating_sub(1) * inner, len, step: -(inner as isize) },
        Some(shift) => Steps::rotated(shift, len, inner),
    };
    Ok(Plan { layout: layout.select(axis, (0..len).map(|j| Some(source(j))))?, steps: vec![Steps::along(outer, len * inner), along, Steps::along(inner, 1)] })
}

/// One axis of windows: windows of `size` items, each `step` items after the last. A padded axis has a window centred on every `step`th position, with fill past the ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowAxis { pub size: usize, pub step: usize, pub padded: bool }

impl WindowAxis {
    /// Whether an axis of length `len` can hold this window. A padded size must be less than twice the length of an axis that isn't empty.
    pub fn fits(&self, len: usize) -> bool { !self.padded || len == 0 || self.size / 2 < len }
    /// The number of windows along an axis of length `len` that `fits`.
    pub fn count(&self, len: usize) -> usize {
        if self.padded { len.saturating_sub(usize::from(self.size.is_multiple_of(2))).div_ceil(self.step) }
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
            Some(w) => Steps::clipped(w.start(0) as i128, (self.frame[a] - 1) * w.step + w.size, self.shape[a], self.stride[a]),
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
            .map(|a| Steps::clipped(starts.get(a).map_or(0, |&s| s as i128), self.cell[a], self.shape[a], self.stride[a])).collect();
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
        (x, y) => Keys::partial([names(x, left.shape()[a]), names(y, right.shape()[a])].concat()).map(Some),
    }).collect::<Result<_, Error>>()?;
    let mut shape = left.shape().to_vec();
    shape[axis] = shape[axis].checked_add(right.shape()[axis]).ok_or_else(|| Error::new(Kind::Limit, "catenate axis overflow"))?;
    generated_len(&shape)?;
    let names = (0..rank).map(|a| agreed([left.name(a), right.name(a)])).collect();
    Ok(Layout::from(shape).with_keys(keys)?.inherit_names(names))
}
