//! Plans for structural operations: take and drop, rotate and reverse, windows, transpose and catenate. A plan gives the result's layout, and `Steps` that read the result's items from the argument's items in order. `gather::walk` copies a compact array by a plan, and basedpl copies its other arrays by the same plan. Each plan checks its arguments.
use crate::agreement::{Agreement, Mapping};
use crate::error::{Error, Kind};
use crate::gather::{IndexSource, IndexValue, Steps, Table, Writes};
use crate::layout::{Keys, Lanes, Layout, agreed, generated_len, names_conflict, shown, strides};
use crate::{buffer::View, float::Float};
use std::{borrow::Cow, sync::Arc};

/// A structural result's layout, and the steps that read its items from the argument. The steps' lengths multiply to the result's item count, outermost first. One entry can cover several result axes.
#[derive(Clone, Debug)]
pub struct Plan { pub layout: Layout, pub steps: Vec<Steps> }

/// A count that an argument gives, such as a take's or a diagonal's offset: its sign and magnitude.
#[derive(Clone, Copy, Debug)]
pub struct Count { pub negative: bool, pub magnitude: usize }

impl Count {
    /// The count `n`. A magnitude past `usize` saturates.
    pub fn of(n: i64) -> Self { Count { negative: n < 0, magnitude: usize::try_from(n.unsigned_abs()).unwrap_or(usize::MAX) } }
}

/// Written index parts, before the selected axes and their lengths are known.
#[derive(Clone, Debug, PartialEq)]
pub enum IndexPart<T> { Axis(T), All, NewAxis, Rest }
impl<T> IndexPart<T> {
    /// The part with its axis payload borrowed.
    pub fn as_ref(&self) -> IndexPart<&T> {
        match self {
            IndexPart::Axis(t) => IndexPart::Axis(t), IndexPart::All => IndexPart::All, IndexPart::NewAxis => IndexPart::NewAxis,
            IndexPart::Rest => IndexPart::Rest
        }
    }
    /// The part with `f` applied to its axis payload. The structural markers stay as they are.
    pub fn map<U>(self, f: impl FnOnce(T) -> U) -> IndexPart<U> {
        match self {
            IndexPart::Axis(t) => IndexPart::Axis(f(t)), IndexPart::All => IndexPart::All, IndexPart::NewAxis => IndexPart::NewAxis,
            IndexPart::Rest => IndexPart::Rest
        }
    }
    /// The part with fallible `f` applied to its axis payload, or `f`'s error.
    pub fn try_map<U, E>(self, f: impl FnOnce(T) -> Result<U, E>) -> Result<IndexPart<U>, E> {
        Ok(match self {
            IndexPart::Axis(t) => IndexPart::Axis(f(t)?), IndexPart::All => IndexPart::All, IndexPart::NewAxis => IndexPart::NewAxis,
            IndexPart::Rest => IndexPart::Rest
        })
    }
    /// The axis payload, or `None` for a structural marker.
    pub fn axis(&self) -> Option<&T> { match self { IndexPart::Axis(t) => Some(t), _ => None } }
}

/// Expands Rest once geometry is known, with one source claim per consuming part. Axis payloads identify the original written part.
pub fn index_claims<'a>(rank: usize, parts: &[IndexPart<(Option<&'a str>, usize)>]) -> Result<(Vec<IndexPart<usize>>, Vec<(Option<&'a str>, usize)>), Error> {
    let used: usize = parts.iter().map(|p| match p { IndexPart::Axis((_, n)) => *n, IndexPart::All => 1, _ => 0 }).sum();
    if used > rank { return Err(Error::new(Kind::Index, "too many axis selectors")); }
    if parts.iter().filter(|p| matches!(p, IndexPart::Rest)).count() > 1 { return Err(Error::new(Kind::Index, "only one rest selector is allowed")); }
    let mut written = Vec::new();
    let mut claims = Vec::new();
    for (i, part) in parts.iter().enumerate() {
        match part {
            IndexPart::Axis(claim) => {
                written.push(IndexPart::Axis(i));
                claims.push(*claim);
            }
            IndexPart::All => {
                written.push(IndexPart::All);
                claims.push((None, 1));
            }
            IndexPart::NewAxis => written.push(IndexPart::NewAxis),
            IndexPart::Rest => {
                for _ in used..rank {
                    written.push(IndexPart::All);
                    claims.push((None, 1));
                }
            }
        }
    }
    Ok((written, claims))
}

/// Routes written parts to source axes without padding them with implicit selectors. Axis payloads give their name and consumed width; returned payloads identify the original part.
pub fn addressed_parts(
    layout: &Layout, parts: &[IndexPart<(Option<&str>, usize)>], explicit: Option<&[usize]>) -> Result<Vec<(usize, IndexPart<usize>)>, Error> {
    let rank = layout.shape().len();
    let (written, claims) = index_claims(rank, parts)?;
    let axes = match explicit {
        Some(axes) if axes.len() == claims.len() => axes.to_vec(),
        Some(_) => return Err(Error::new(Kind::Length, "one axis is needed for each consuming selector")), None => addressed(layout, &claims)?
    };
    if axes.iter().zip(&claims).any(|(&axis, &(_, width))| axis.checked_add(width).is_none_or(|end| end > rank)) {
        return Err(Error::new(Kind::Index, "a selector spans beyond the array's axes"));
    }
    let last = axes.iter().zip(&claims).map(|(&a, &(_, n))| a + n).max().unwrap_or(0);
    let mut next = 0;
    let mut routed = written.into_iter().map(|part| {
        let axis = axes.get(next).copied().unwrap_or(last);
        if !matches!(part, IndexPart::NewAxis) { next += 1; }
        (axis, part)
    }).collect::<Vec<_>>();
    routed.sort_by_key(|(axis, _)| *axis);
    Ok(routed)
}

/// Plans a selection by written parts on `layout`, as `⌷` and the Python frontends read brackets. `claims` give each part's axis name and width, as for `addressed_parts`, and `explicit` its axes. `decode(i, axis)` gives the selector of written part `i` at source `axis`, with whether it reads one position. Axes that no part reaches stay whole, and a part wider than one axis reads its axes as one. Also gives whether the selection reads one item: one position on every source axis, in order. `broadcast` pairs index arrays as `select` does. `err` converts this module's errors to the caller's.
pub fn plan_parts<E>(
    layout: &Layout, claims: &[IndexPart<(Option<&str>, usize)>], explicit: Option<&[usize]>, broadcast: bool,
    decode: impl FnMut(usize, usize) -> Result<(Selector, bool), E>, err: impl Fn(Error) -> E) -> Result<(Plan, bool), E> {
    let r = routes(layout, claims, explicit, decode, &err)?;
    Ok((plan_routes(layout, &r, broadcast).map_err(&err)?, r.one))
}

/// The selectors of a selection by written parts (`routes`): one for each axis of the layout that `sources` regroups the source into, with the source axes that each reads, and whether the selection reads one item, one position on every source axis in order.
#[derive(Clone, Debug)]
pub struct Routes { pub selectors: Vec<Selector>, pub sources: Vec<Vec<usize>>, pub one: bool }

/// The routes of a selection by written parts on `layout`, as `plan_parts` takes them. `claims` give each part's axis name and width, as for `addressed_parts`, and `explicit` its axes. `decode(i, axis)` gives the selector of written part `i` at source `axis`, with whether it reads one position. Axes that no part reaches stay whole (`Selector::All`), and a part wider than one axis reads its axes as one.
pub fn routes<E>(
    layout: &Layout, claims: &[IndexPart<(Option<&str>, usize)>], explicit: Option<&[usize]>,
    mut decode: impl FnMut(usize, usize) -> Result<(Selector, bool), E>, err: impl Fn(Error) -> E) -> Result<Routes, E> {
    let shape = layout.shape();
    let (mut one, mut next, mut sources, mut selectors) = (claims.iter().any(|p| matches!(p, IndexPart::Axis(_))), 0, Vec::<Vec<usize>>::new(), Vec::new());
    for (axis, part) in addressed_parts(layout, claims, explicit).map_err(&err)? {
        one &= next == axis;
        for a in next..axis {
            sources.push(vec![a]);
            selectors.push(Selector::All);
        }
        next = axis;
        let (selector, width, single) = match part {
            IndexPart::NewAxis => {
                selectors.push(Selector::NewAxis);
                one = false;
                continue;
            }
            IndexPart::Axis(i) => {
                let IndexPart::Axis((_, width)) = claims[i] else { unreachable!("an axis refers to its written claim") };
                let (selector, single) = decode(i, axis)?;
                (selector, width, single)
            }
            _ => (Selector::All, 1, false),
        };
        one &= single;
        sources.push((axis..axis + width).collect());
        selectors.push(selector);
        next += width;
    }
    one &= next == shape.len();
    sources.extend((next..shape.len()).map(|a| vec![a]));
    Ok(Routes { selectors, sources, one })
}

/// The plan of the selection that `routes` gives on `layout`, as `plan_parts` plans it.
pub fn plan_routes(layout: &Layout, routes: &Routes, broadcast: bool) -> Result<Plan, Error> {
    planned(route_layout(layout, routes)?.as_ref(), routes.selectors.clone(), broadcast)
}

fn route_layout<'a>(layout: &'a Layout, routes: &Routes) -> Result<Cow<'a, Layout>, Error> {
    let shape = layout.shape();
    if routes.sources.len() == shape.len() { return Ok(Cow::Borrowed(layout)); }
    regrouped(layout, routes.sources.iter().map(|s| s.iter().map(|&a| shape[a]).product()).collect(), &routes.sources, &[]).map(Cow::Owned)
}

/// One independent axis selector. Positions are zero-based and already normalized; a range can walk backwards. `All` keeps the whole axis, whatever its length. An index array replaces its axis by the axes of its layout. Its positions, a [`Table`], lie within the axis or are `Steps::FILL`, which reads the caller's fill. The code that builds an index array checks its positions, as `index_positions` and `index_table` do. `NewAxis` inserts a length-one axis without consuming an input axis.
#[derive(Clone, Debug)]
pub enum Selector { At(usize), Range { start: usize, len: usize, step: isize }, All, Indices { layout: Layout, positions: Table }, NewAxis }

/// Selects with one selector for each leading axis of `layout`, in order, keeping the axes that no selector mentions. A single position removes its axis, and a range keeps it in place. An index array's axes replace its axis. An unnamed vector keeps the axis's name and the keys at the positions it selects if none repeats, as a named vector without keys of its own keeps those keys, and an unnamed index array of another rank has plain axes. Index arrays pair, as arguments pair in arithmetic: all of them with `broadcast`, as `.[]` reads them, and otherwise those that share an axis name, as `⌷` reads them. Paired index arrays put their axes where the first of them was. A range pairs with an index array that has its axis's name. Positions that step evenly along each axis read as strides (`strided`), whether one index array gives them or paired arrays sum to them. One index array whose positions read a stretch of its axis past the ends, as edge modes and positions that count back from the end give them, reads as one padded or rotated entry.
pub fn select(layout: &Layout, selectors: Vec<Selector>, broadcast: bool) -> Result<Plan, Error> { planned(layout, selectors, broadcast) }

fn planned(layout: &Layout, selectors: Vec<Selector>, broadcast: bool) -> Result<Plan, Error> {
    let (shape, stride) = (layout.shape(), strides(layout.shape()));
    let mut parts = Vec::with_capacity(shape.len());
    let mut axis = 0;
    for selector in selectors {
        if matches!(selector, Selector::NewAxis) { parts.push(Part::new(Steps::along(1, 0), Layout::from([1]), 0, 1)); continue; }
        if axis == shape.len() { return Err(Error::new(Kind::Index, "too many axis selectors")); }
        parts.push(Part::of(selector, layout, axis, stride[axis])?);
        axis += 1;
    }
    parts.extend((axis..shape.len()).map(|a| Part::new(Steps::along(shape[a], 1), layout.axes(a..a + 1), stride[a], shape[a])));
    let names: Vec<Arc<str>> = parts.iter().filter(|p| p.array).flat_map(|p| p.axes.names().iter().flatten().cloned()).collect();
    for part in parts.iter_mut().filter(|p| !p.array && p.axes.name(0).is_some_and(|n| names.contains(n))) { part.array = true; }
    // The index arrays that pair, each group in order.
    let mut groups: Vec<Vec<usize>> = Vec::new();
    let shares = |p: usize, q: usize| parts[p].axes.names().iter().flatten().any(|n| parts[q].axes.names().contains(&Some(n.clone())));
    for p in (0..parts.len()).filter(|&p| parts[p].array) {
        let hits: Vec<usize> = (0..groups.len()).filter(|&g| broadcast || groups[g].iter().any(|&q| shares(p, q))).collect();
        let Some((&first, rest)) = hits.split_first() else { groups.push(vec![p]); continue; };
        for &g in rest.iter().rev() {
            let moved = groups.remove(g);
            groups[first].extend(moved);
        }
        groups[first].push(p);
        groups[first].sort_unstable();
    }
    let mut group_of = vec![usize::MAX; parts.len()];
    for (g, members) in groups.iter().enumerate() { for &p in members { group_of[p] = g; } }
    // At most one entry for each axis that a part reads along, or one for a part with none.
    let (mut result, mut steps) = (Layout::EMPTY, Vec::with_capacity(parts.iter().map(|p| p.axes.shape().len().max(1)).sum()));
    for p in 0..parts.len() {
        // Paired index arrays read where the first of them was.
        if let Some(group) = groups.get(group_of[p]).filter(|g| g.len() > 1) {
            if group[0] == p { result = result.concat(&joined(&parts, group, broadcast, &mut steps)?); }
            continue;
        }
        let part = &mut parts[p];
        result = result.concat(part.alone.as_ref().unwrap_or(&part.axes));
        let entry = std::mem::replace(&mut part.steps, Steps::along(0, 0)).scaled(part.stride);
        if part.array { strided(entry, part.axes.shape(), Some((part.size, part.stride)), &mut steps); } else { steps.push(entry); }
    }
    generated_len(result.shape())?;
    Ok(Plan { layout: result, steps })
}

/// One selector's share of a plan: positions along its axis, before scaling by the axis's `stride`, the axis's `size`, and the result axes they run along. An index array that pairs with nothing takes `alone` as its axes, where it has them.
struct Part { steps: Steps, axes: Layout, stride: usize, size: usize, array: bool, alone: Option<Layout> }

impl Part {
    fn new(steps: Steps, axes: Layout, stride: usize, size: usize) -> Self { Part { steps, axes, stride, size, array: false, alone: None } }

    fn of(selector: Selector, layout: &Layout, axis: usize, stride: usize) -> Result<Self, Error> {
        let size = layout.shape()[axis];
        let outside = || Error::new(Kind::Index, "axis selector is outside its axis");
        match selector {
            Selector::At(i) if i < size => Ok(Part::new(Steps::Stride { start: i, len: 1, step: 0 }, Layout::EMPTY, stride, size)),
            Selector::At(_) => Err(outside()),
            Selector::All => Self::of(Selector::Range { start: 0, len: size, step: 1 }, layout, axis, stride),
            Selector::Range { start, len, step } => {
                if len > 0 && !(start < size && start.checked_add_signed(step.saturating_mul((len - 1) as isize)).is_some_and(|i| i < size)) {
                    return Err(outside());
                }
                let steps = Steps::Stride { start, len, step };
                let axes = layout.select(axis, steps.offsets())?.axes(axis..axis + 1);
                Ok(Part::new(steps, axes, stride, size))
            }
            Selector::Indices { layout: within, positions } => {
                if generated_len(within.shape())? != positions.len() {
                    return Err(Error::new(
                        Kind::Length, format!("index array shape {:?} needs {} positions, got {}", within.shape(), within.count(), positions.len())));
                }
                let steps = Steps::Table(positions);
                let selected = layout.select(axis, steps.offsets())?;
                Ok(Part { steps, alone: alone(&within, &selected, axis)?, axes: within, stride, size, array: true })
            }
            Selector::NewAxis => unreachable!("a new axis reads no positions"),
        }
    }
}

/// The axes that an index array laid out as `within` takes where it pairs with nothing, from `selected`, the layout whose `axis` holds the positions that it reads: a vector without names takes that axis, a named vector without keys takes its keys, and an index array of another rank without names has plain axes. `None` keeps `within`.
fn alone(within: &Layout, selected: &Layout, axis: usize) -> Result<Option<Layout>, Error> {
    Ok(match within.shape().len() {
        1 if !within.has_names() => Some(selected.axes(axis..axis + 1)),
        1 if within.keys(0).is_none() => Some(within.clone().with_keys(vec![selected.keys(axis).cloned()])?),
        _ if !within.has_names() => Some(Layout::from(within.shape().to_vec())), _ => None
    })
}

/// The axes that paired index arrays agree on, with the entries that read them pushed onto `steps`, one for each run of those axes: the fewest consecutive axes that hold all the axes of each array in the run. An array alone in its run reads through its own positions, as an array that pairs with nothing does. Where every array in a run steps evenly through its positions (`Steps::even`), and the run's axes have no keys, the run reads as one `Stride` for each of its axes, whose step is the sum of each array's step along that axis times the array's stride, and no table is built. Otherwise the arrays add their scaled positions into one table, where a fill in any of them makes a fill. An array that lies along the whole run adds its positions in chunks, and one that broadcasts along it adds them in the runs that a walk through its mapping reaches. The table reads as strides where its offsets are affine (`strided`).
fn joined(parts: &[Part], group: &[usize], broadcast: bool, steps: &mut Vec<Steps>) -> Result<Layout, Error> {
    let mut frame = parts[group[0]].axes.clone();
    let mut pairs = vec![(0..frame.shape().len()).collect::<Vec<_>>()];
    for &p in &group[1..] {
        let axes = Agreement::paired_axes(&frame, &parts[p].axes, broadcast);
        frame = Agreement::paired(&frame, &parts[p].axes, &axes)?.layout;
        pairs.push(axes);
    }
    let rank = frame.shape().len();
    // A run ends after axis `a` unless some array has axes on both sides of it. An array with no axes reads in the first run.
    let crosses = |a: usize| pairs.iter().any(|axes| axes.iter().any(|&b| b <= a) && axes.iter().any(|&b| b > a));
    let mut start = 0;
    for end in (1..=rank).filter(|&end| end == rank || !crosses(end - 1)).chain((rank == 0).then_some(0)) {
        let run = start..end;
        start = end;
        let lens = &frame.shape()[run.clone()];
        let members = || (0..group.len()).filter(|&m| pairs[m].first().map_or(run.start == 0, |a| run.contains(a)));
        let mut lone = members();
        if let (Some(m), None) = (lone.next(), lone.next()) && pairs[m].iter().copied().eq(run.clone()) {
            let Part { steps: own, stride, size, .. } = &parts[group[m]];
            strided(own.clone().scaled(*stride), lens, Some((*size, *stride)), steps);
            continue;
        }
        let at = steps.len();
        steps.extend(lens.iter().map(|&len| Steps::Stride { start: 0, len, step: 0 }));
        let mut first = 0;
        let summed = !lens.is_empty() && run.clone().all(|a| frame.keys(a).is_none())
            && members().all(|m| {
                let Part { steps: own, axes, stride, .. } = &parts[group[m]];
                let Some(Steps::Stride { start, step, .. }) = own.even() else { return false };
                first += start * stride;
                // The positions step through the array's own axes in row-major order, so each axis steps by the array's step times the axis's width.
                let mut width = step * *stride as isize;
                for (&a, &len) in pairs[m].iter().zip(axes.shape()).rev() {
                    if len > 1 && let Steps::Stride { step, .. } = &mut steps[at + a - run.start] { *step += width; }
                    width *= len as isize;
                }
                true
            });
        if summed { Steps::settle(&mut steps[at..], first); continue; }
        steps.truncate(at);
        let within = if run.len() == rank { Cow::Borrowed(&frame) } else { Cow::Owned(frame.axes(run.clone())) };
        let len = generated_len(lens)?;
        let mut table = vec![0; len];
        for m in members() {
            let Part { steps: own, axes, stride, .. } = &parts[group[m]];
            let add = |t: &mut usize, o: Option<usize>| {
                *t = match o { Some(o) if *t != Steps::FILL => *t + o * stride, _ => Steps::FILL }
            };
            let shifted: Vec<usize>;
            let along = if run.start == 0 { &pairs[m] } else { shifted = pairs[m].iter().map(|&a| a - run.start).collect(); &shifted };
            match (Mapping::new(axes, along, &within, len), own) {
                (Mapping::Linear(1), Steps::Table(positions)) => positions.chunks(|j0, offsets| {
                    for (t, &o) in table[j0..].iter_mut().zip(offsets) { add(t, Some(o).filter(|&o| o != Steps::FILL)) }
                }),
                (map, _) => {
                    let (base, tables) = map.to_walk(lens);
                    Writes::Walk { base, steps: &tables }.runs_while(&mut |k, reads| {
                        for (t, &j) in table[k..].iter_mut().zip(reads) { add(t, (j != Steps::FILL).then(|| own.get(j)).flatten()) }
                        true
                    });
                }
            }
        }
        strided(Steps::table(table), lens, None, steps);
    }
    Ok(frame)
}

/// Pushes `entry`, which reads the axes of `lens`, onto `into`: as one `Stride` for each of those axes where its offsets are affine (`Table::affine`), as one `Padded` or `Rotated` entry where `lens` is one axis read past the ends of the source axis whose size and stride `axis` gives (`Table::edged`), or else as it is. A copy reads strides in place, and the transpose of a copy through strides reads back where a table would scatter.
fn strided(entry: Steps, lens: &[usize], axis: Option<(usize, usize)>, into: &mut Vec<Steps>) {
    if let Steps::Table(table) = &entry {
        if table.affine(lens, into) { return; }
        if let ([_], Some((size, stride))) = (lens, axis) && let Some(edged) = table.edged(size, stride) { return into.push(edged); }
    }
    into.push(entry);
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
        if span(a, width).any(|b| !free[b]) { return Err(Error::new(Kind::Index, format!("two selectors address axis {}", shown(n)))); }
        free[span(a, width)].fill(false);
        axes[k] = a;
    }
    for (k, &(_, width)) in selectors.iter().enumerate() {
        if axes[k] != rank { continue; }
        let name = names[k];
        let fits = |a: usize| a < rank && span(a, width).all(|b| free[b] && !names_conflict(name, layout.name(b).map(|n| &**n)));
        let missing =
            || Error::new(Kind::Index, name.map_or_else(|| "too many axis selectors".into(), |n| format!("no axis for a selector named {}", shown(n))));
        let a = (0..rank).find(|&a| fits(a)).ok_or_else(missing)?;
        free[span(a, width)].fill(false);
        axes[k] = a;
    }
    Ok(axes)
}

/// [`select`] with `selector` on `axis` and every other axis whole. A `NewAxis` goes in before `axis`.
pub fn select_axis(layout: &Layout, axis: usize, selector: Selector) -> Result<Plan, Error> {
    let selectors: Vec<_> = layout.shape().iter().take(axis).map(|_| Selector::All).chain([selector]).collect();
    select(layout, selectors, false)
}

/// The position of key `name` on `axis` of `layout`, or `Steps::FILL` where the axis lacks it: a missing key reads the fill and writes nothing, as a position outside the axis does. BPL and both Python bindings read keys by this rule.
pub fn key_position(layout: &Layout, axis: usize, name: &str) -> Result<usize, Error> {
    let keys = layout.keys(axis).ok_or_else(|| Error::new(Kind::Index, "axis has no keys"))?;
    Ok(keys.position(name).unwrap_or(Steps::FILL))
}

/// The selector that keys give on `axis` of `layout`, by [`key_position`]: one key's position, which removes the axis, where `within` is `None`, and otherwise an index array of the keys' positions, laid out as `within`.
pub fn key_selector<'a>(layout: &Layout, axis: usize, names: impl IntoIterator<Item = &'a str>, within: Option<Layout>) -> Result<Selector, Error> {
    let positions = names.into_iter().map(|n| key_position(layout, axis, n)).collect::<Result<Vec<_>, _>>()?;
    if within.is_none() && let [p] = positions[..] && p != Steps::FILL { return Ok(Selector::At(p)); }
    Ok(Selector::Indices { layout: within.unwrap_or(Layout::EMPTY), positions: Table::new(positions) })
}

/// For each axis of `layout`, the keys that `wanted` names on it and it lacks, each once, in order. `wanted` pairs an axis with the keys that one selector names on it. A write adds these keys before it plans ([`extended`]).
pub fn missing_keys<'a, K: IntoIterator<Item = &'a str>>(layout: &Layout, wanted: impl IntoIterator<Item = (usize, K)>) -> Result<Vec<Vec<Arc<str>>>, Error> {
    let mut added: Vec<Vec<Arc<str>>> = vec![vec![]; layout.shape().len()];
    for (axis, names) in wanted {
        for name in names {
            let lacked = added.get_mut(axis).ok_or_else(|| Error::new(Kind::Rank, "too many axis selectors"))?;
            if layout.keys(axis).is_none_or(|k| k.position(name).is_none()) && !lacked.iter().any(|n| &**n == name) { lacked.push(name.into()); }
        }
    }
    Ok(added)
}

/// The plan that appends the keys in `added`, a list for each axis, to the axes of `layout`, as [`missing_keys`] gives them. An axis without keys takes them after a position without a key for each of its items. The new positions read `Steps::FILL`, and the others read the items in place.
pub fn extended(layout: &Layout, added: &[Vec<Arc<str>>]) -> Result<Plan, Error> {
    let (mut shape, mut keys) = (layout.shape().to_vec(), layout.all_keys());
    for (axis, names) in added.iter().enumerate().filter(|(_, names)| !names.is_empty()) {
        let mut all = keys[axis].as_ref().map_or_else(|| vec![None; shape[axis]], |k| k.names().to_vec());
        all.extend(names.iter().cloned().map(Some));
        shape[axis] = all.len();
        keys[axis] = Some(Keys::partial(all)?);
    }
    generated_len(&shape)?;
    let steps = shape.iter().zip(layout.shape()).zip(strides(layout.shape())).map(|((&len, &size), stride)| Steps::clipped(0, len, size, stride)).collect();
    Ok(Plan { layout: Layout::from(shape).with_keys(keys)?.inherit_names(layout.names().to_vec()), steps })
}

/// A table that reads the index values in `values` as positions along an axis of length `size`, as [`positions`] reads them. Booleans and integers that all lie inside the axis read in place, each converted as it's read: the one pass that checks them writes nothing, and also finds whether they give evenly spaced positions, so that a walk can read them as one stride. Floats, read within tolerance `t`, and values with any outside the axis become the list of positions that [`positions`] gives, with the fill position for each value that names none.
pub fn index_table(values: Arc<dyn IndexSource>, size: usize, t: crate::tolerant::Tolerance) -> Result<Table, Error> {
    /// `None` where some value is outside the axis. Otherwise the first position and the step where the positions are evenly spaced.
    fn check<I: IndexValue>(values: &[I], size: usize) -> Option<Option<(usize, isize)>> {
        let (lo, hi) = I::bounds(values);
        if !values.is_empty() && !(lo >= -(size as i64) && hi < size as i64) { return None; }
        let at = |k: usize| values[k].position();
        if values.len() < 2 { return Some(None); }
        // Values of one sign give evenly spaced positions where they're evenly spaced themselves. Checking the second step first skips the full pass for values in no order.
        let step = at(1) - at(0);
        let even = (lo >= 0 || hi < 0) && (values.len() < 3 || at(2) - at(1) == step)
            && values.windows(2).fold(true, |even, w| even & (w[1].position() - w[0].position() == step));
        Some(even.then(|| (counted_back(at(0), size), step as isize)))
    }
    let checked = match values.view() { View::Floats(_) => None, view => crate::gather::with_index_values!(view, |x| check(x, size)) };
    match checked { Some(even) => Ok(Table::indices(values, size, even)), None => positions(values.view(), size, t).map(Table::new) }
}

/// Position `i` on an axis of length `size`, where a negative `i` counts back from the end. Adding `size` to a negative position without a branch keeps a loop over positions vectorised. A position outside the axis is then at least `size`.
#[inline(always)]
pub fn position_recipe<F: Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, value: O::I, size: i64) -> (O::I, O::B) {
    use crate::recipe::{Cmp, Integer};
    let (zero, limit) = (o.integer(0), o.integer(size));
    let negative = o.integer_compare(Cmp::Less, value, zero);
    let position = o.integer_op(Integer::Add, value, o.integer_select(negative, limit, zero));
    let valid = o.and(o.not(o.integer_compare(Cmp::Less, position, zero)), o.integer_compare(Cmp::Less, position, limit));
    (position, valid)
}
pub(crate) fn counted_back(i: i64, size: usize) -> usize {
    let ops = crate::recipe::IntegerOps::<i64>::new(crate::domain::Variant::checked(crate::buffer::DType::Integer(crate::buffer::Width::I64)));
    position_recipe::<f64, _>(&ops, i as u64, size as i64).0 as usize
}

/// The positions that index values give along an axis of length `size`. A negative value counts back from the end. A value outside the axis, a float that isn't a whole number within tolerance `t` (`float_position`), or a NaN gives the fill position (`Steps::FILL`), which reads the fill and writes nothing. Items that aren't numbers are a Domain error.
pub fn positions(view: View<'_>, size: usize, t: crate::tolerant::Tolerance) -> Result<Vec<usize>, Error> {
    let at = |p: Option<i64>| p.map(|p| counted_back(p, size)).filter(|&p| p < size).unwrap_or(Steps::FILL);
    Ok(match view {
        View::Booleans(b) => b.iter().map(|&i| at(Some(i.into()))).collect(),
        View::Integers(ix) => with_ints!(ix, |i| i.iter().map(|&i| at(Some(i.into()))).collect()),
        View::Floats(x) => with_floats!(x, |x| x.iter().map(|&y| at(float_position(y, t))).collect()),
        _ => return Err(Error::new(Kind::Domain, "indices must be real numbers")),
    })
}

/// The message of the Domain error for an index value that isn't a whole number.
pub const NOT_WHOLE: &str = "index must be a whole number";

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

/// The whole number that a float index value names: the one it equals within tolerance `t`, the comparison tolerance, which only Mathish makes nonzero. `None` for a NaN or a value that isn't whole, which reads the fill. A whole number past `i64` saturates, and so lies outside every axis. The one rule for float index values: [`positions`], [`signed_positions`], [`index_table`] and [`Edge::positions`] read floats through it.
pub fn float_position<F: Float>(y: F, t: crate::tolerant::Tolerance) -> Option<i64> { crate::tolerant::Tolerant::whole(t, y.wide()).map(|n| n as i64) }

/// The signed position that each index value gives, before any axis applies: integers as they are, and floats as `float_position` reads them within tolerance `t`, with `None` for a NaN or a value that isn't whole. Items that aren't real numbers have no position. [`positions`] reads the values and checks them against an axis in one pass.
pub fn signed_positions(view: View<'_>, t: crate::tolerant::Tolerance) -> Vec<Option<i64>> {
    match view {
        View::Booleans(b) => b.iter().map(|&i| Some(i.into())).collect(), View::Integers(ix) => with_ints!(ix, |i| i.iter().map(|&i| Some(i.into())).collect()),
        View::Floats(x) => with_floats!(x, |x| x.iter().map(|&y| float_position(y, t)).collect()), _ => vec![None; view.len()]
    }
}

/// The positions along an axis of length `size` that signed positions give, as [`positions`] gives them, with `Steps::FILL` for `None` and for a position outside the axis.
pub fn counted_positions(values: &[Option<i64>], size: usize) -> Vec<usize> {
    values.iter().map(|&q| q.map(|i| counted_back(i, size)).filter(|&p| p < size).unwrap_or(Steps::FILL)).collect()
}

/// How a position outside an axis maps to one inside it, as edge modes of padding do. `Fill` gives no position, which reads the fill.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Edge { Fill, Clamp, Wrap, Reflect, Symmetric }

/// The positions that [`Edge::positions`] gives: each item's position on the axis, or floats where some item reads the fill, with NaN there.
#[derive(Debug, PartialEq)]
pub enum Edged { Positions(Vec<usize>), Floats(Vec<f64>) }

impl Edge {
    /// Each mode with its name.
    pub const MODES: [(&'static str, Edge); 5] =
        [("fill", Edge::Fill), ("clamp", Edge::Clamp), ("wrap", Edge::Wrap), ("reflect", Edge::Reflect), ("symmetric", Edge::Symmetric)];
    /// The mode named `name`.
    pub fn named(name: &str) -> Option<Edge> { Self::MODES.iter().find(|(n, _)| *n == name).map(|&(_, e)| e) }
    /// This mode's name.
    pub fn name(self) -> &'static str { Self::MODES.iter().find(|&&(_, e)| e == self).expect("every mode has a name").0 }
    /// The position on an axis of `n` items that position `p` reads, or `None` for the fill. A position inside the axis stays as it is in every mode. A negative `p` is outside the axis here. `Reflect` leaves out the edge item, as NumPy's `reflect` does, so `-1` reads 1. `Symmetric` repeats it, as NumPy's `symmetric` does, so `-1` reads 0. An empty axis gives `None` in every mode.
    #[inline]
    pub fn position(self, p: i64, n: usize) -> Option<usize> {
        if (p as u64) < n as u64 { return Some(p as usize); }
        let n = i64::try_from(n).ok().filter(|&n| n > 0)?;
        let q = match self {
            Edge::Fill => p, Edge::Clamp => p.clamp(0, n - 1), Edge::Wrap => p.rem_euclid(n), Edge::Reflect if n == 1 => 0,
            Edge::Reflect => (n - 1) - ((n - 1) - p.rem_euclid(2 * (n - 1))).abs(), Edge::Symmetric => { let r = p.rem_euclid(2 * n); r.min(2 * n - 1 - r) }
        };
        usize::try_from(q).ok().filter(|&q| q < n as usize)
    }
    /// The position on an axis of `n` items that each index value in `view` reads (`position`): integers as they are, and floats as `float_position` reads them within tolerance `t`. A NaN, or a float that isn't a whole number, reads the fill.
    pub fn positions(self, view: View<'_>, n: usize, t: crate::tolerant::Tolerance) -> Result<Edged, Error> {
        index_view!(view, |ix| Ok(self.mapped(n, false, ix.iter().map(|&i| Some(i.into())))), |x| {
            let signed: Vec<_> = x.iter().map(|&y| float_position(y, t)).collect();
            let unread = signed.iter().any(Option::is_none);
            Ok(self.mapped(n, unread, signed.into_iter()))
        })
    }
    /// What [`Edge::positions`] gives for `signed` positions, each `None` for a NaN, where `nan` says whether any is. Where `Fill` or an empty axis reads the fill for some position, or a position is NaN, the result is floats.
    pub fn mapped(self, n: usize, nan: bool, signed: impl ExactSizeIterator<Item = Option<i64>>) -> Edged {
        if self == Edge::Fill || nan || (n == 0 && signed.len() > 0) {
            return Edged::Floats(signed.map(|q| q.and_then(|p| self.position(p, n)).map_or(f64::NAN, |p| p as f64)).collect());
        }
        Edged::Positions(signed.map(|q| q.and_then(|p| self.position(p, n)).expect("a mode other than fill reads a position on an axis with items")).collect())
    }
}

/// Repeats each cell along `axis` by its count: one count for every cell, or one for each cell. A negative count gives that many fills in place of its cell. Where the axis has one cell, every count repeats it. A fill has no key, and repeating a keyed position is an error, as it is for selection.
pub fn repeat(layout: &Layout, axis: usize, counts: &[i64]) -> Result<Plan, Error> { replicated(layout, axis, counts, false) }

/// Undoes `repeat` with the same counts: one cell for each count. A positive count gives the first cell it made, and any other count a fill. One count must divide the number of cells. More counts must use every cell, unless the axis has one cell.
pub fn expand(layout: &Layout, axis: usize, counts: &[i64]) -> Result<Plan, Error> { replicated(layout, axis, counts, true) }

/// `repeat`'s items with no plan to walk: `view`, the items of `layout`, with each cell along `axis` written as many times as its count, a run at a time (`gather::repeat`, or `gather::positions` for counts of 0 and 1). `None` where a count is negative or `axis` has keys, which `repeat`'s plan serves.
pub fn repeated<C: crate::search::Key>(view: View<'_>, layout: &Layout, axis: usize, counts: &[C]) -> Result<Option<(Layout, crate::buffer::Buffer)>, Error> {
    use crate::buffer::Element;
    let (lo, hi) = if counts.is_empty() { (0, 0) } else { C::range(counts) };
    if layout.keys(axis).is_some() || lo < 0 { return Ok(None); }
    let (lanes, len, shape) = replication(layout, axis, counts, false)?;
    let (total, inner, cell) = (shape[axis], lanes.inner, |j: usize| if lanes.len == 1 { 0 } else { j });
    let mask = counts.len() > 1 && hi <= 1;
    let buffer = with_dtype!(view.dtype(), T => {
        let items = T::of(view).expect("a view's items have its dtype");
        let mut out = Vec::with_capacity(lanes.outer * total * inner);
        for block in items.chunks((lanes.len * inner).max(1)) {
            if inner > 1 { for j in 0..len { for _ in 0..counts[if counts.len() == 1 { 0 } else { j }].key() { out.extend_from_slice(&block[cell(j) * inner..][..inner]); } } }
            else if mask { crate::gather::positions(&mut out, counts, total, |j| block[cell(j)]) }
            else { crate::gather::repeat(&mut out, len, counts, total, |j| block[cell(j)]) }
        }
        T::buffer(out)
    });
    Ok(Some((unkeyed(layout, shape)?, buffer)))
}

/// The lanes along `axis` of `layout`, the number of cells that `counts` repeat or undo, and the result's shape, for `repeat`, or `expand` where `inverse`.
fn replication<C: crate::search::Key>(layout: &Layout, axis: usize, counts: &[C], inverse: bool) -> Result<(Lanes, usize, Vec<usize>), Error> {
    let lanes = Lanes::new(layout.shape(), axis)?;
    let single = counts.len() == 1;
    let size = |n: i64| Count::of(n).magnitude;
    let len = match (inverse, single) {
        (false, true) => lanes.len,
        (false, false) if lanes.len == 1 || counts.len() == lanes.len => counts.len(),
        (false, false) => return Err(Error::new(Kind::Length, "repeat needs one count, or one count for each cell")),
        (true, true) => match size(counts[0].key()) {
            0 if lanes.len == 0 => 1, n if n > 0 && lanes.len % n == 0 => lanes.len / n,
            _ => return Err(Error::new(Kind::Domain, "one count must divide the number of cells to invert"))
        },
        (true, false) => counts.len(),
    };
    // The greatest count times their number bounds the sum, which then runs unchecked.
    let most = if counts.is_empty() { 0 } else { let (lo, hi) = C::range(counts); lo.unsigned_abs().max(hi.unsigned_abs()) };
    let total = if inverse { Some(len) } else {
        most.checked_mul(len as u64)
            .and_then(|_| usize::try_from(if single { most * len as u64 } else { counts.iter().map(|n| n.key().unsigned_abs()).sum() }).ok())
    };
    let mut shape = layout.shape().to_vec();
    shape[axis] = total.ok_or_else(|| Error::new(Kind::Limit, "repeat length overflow"))?;
    generated_len(&shape)?;
    Ok((lanes, len, shape))
}

/// `shape` with `layout`'s keys on its other axes and its names, as a repeat with no keys along its axis lays out.
fn unkeyed(layout: &Layout, shape: Vec<usize>) -> Result<Layout, Error> {
    Ok(Layout::from(shape).with_keys(layout.all_keys())?.inherit_names(layout.names().to_vec()))
}

/// The plan of `repeat`, or of `expand` where `inverse`. Repeating, each count uses up one cell. Undoing, each count uses up the cells it made and gives back one.
fn replicated(layout: &Layout, axis: usize, counts: &[i64], inverse: bool) -> Result<Plan, Error> {
    let (lanes, len, shape) = replication(layout, axis, counts, inverse)?;
    let single = counts.len() == 1;
    let count = |j: usize| counts[if single { 0 } else { j }];
    let size = |n: i64| Count::of(n).magnitude;
    let keyed = layout.keys(axis).is_some();
    let rows = Steps::along(lanes.outer, lanes.len * lanes.inner);
    if !inverse && single && counts[0] >= 0 && !keyed {
        return Ok(Plan { layout: unkeyed(layout, shape)?,
            steps: vec![rows, Steps::along(lanes.len, lanes.inner), Steps::along(size(counts[0]), 0), Steps::along(lanes.inner, 1)] });
    }
    let (table, positions) = if inverse {
        let (mut table, mut positions, mut used) = (Vec::with_capacity(len), Vec::new(), 0usize);
        for j in 0..len {
            let n = count(j);
            let cell = (n > 0).then_some(if lanes.len == 1 { 0 } else { used });
            used = used.saturating_add(size(n));
            table.push(cell.map_or(Steps::FILL, |c| c * lanes.inner));
            if keyed { positions.push(cell); }
        }
        if lanes.len != 1 && used != lanes.len { return Err(Error::new(Kind::Length, "the counts must use every cell to invert")); }
        (table, positions)
    } else {
        // The table repeats each count's cell by the count's magnitude, with a fill in place of the cell where the count is negative. A mask of counts 0 and 1 takes `gather::positions`, which skips blocks of zeros.
        let cell = |j: usize| if lanes.len == 1 { 0 } else { j };
        let (mut table, mut positions) = (Vec::with_capacity(shape[axis]), Vec::new());
        if !single && !keyed && counts.iter().all(|&n| n == 0 || n == 1) {
            crate::gather::positions(&mut table, counts, shape[axis], |j| cell(j) * lanes.inner)
        } else {
            let sizes: std::borrow::Cow<'_, [i64]> =
                if counts.iter().all(|&n| n >= 0) { counts.into() } else { counts.iter().map(|&n| size(n) as i64).collect() };
            let at = |j: usize| (count(j) > 0).then_some(cell(j));
            crate::gather::repeat(&mut table, len, &sizes, shape[axis], |j| at(j).map_or(Steps::FILL, |c| c * lanes.inner));
            if keyed { crate::gather::repeat(&mut positions, len, &sizes, shape[axis], at); }
        }
        (table, positions)
    };
    let result = if keyed { layout.select(axis, positions.into_iter())? } else { unkeyed(layout, shape)? };
    Ok(Plan { layout: result, steps: vec![rows, Steps::table(table), Steps::along(lanes.inner, 1)] })
}

/// The lengths of `shape` with `widths` added before and after each axis.
pub fn padded(shape: &[usize], widths: &[(usize, usize)]) -> Result<Vec<usize>, Error> {
    if widths.len() != shape.len() { return Err(Error::new(Kind::Rank, "padding needs one pair for each axis")); }
    Ok(shape.iter().zip(widths).map(|(&len, &(before, after))| len.saturating_add(before).saturating_add(after)).collect())
}

/// The steps that pad each source axis under its edge mode.
pub fn pad_steps(shape: &[usize], widths: &[(usize, usize)], edge: Edge) -> Result<Vec<Steps>, Error> {
    let lengths = padded(shape, widths)?;
    let stride = strides(shape);
    Ok((0..shape.len()).map(|a| Steps::padded(-(widths[a].0 as i128), lengths[a], shape[a], stride[a], edge)).collect())
}

/// Pads every axis by `(before, after)` positions, to the lengths that `padded` gives. Each position past an end reads by `edge`: the item type's fill, or a position on the axis. Existing axis keys remain on their positions. Padding a keyed axis by another mode than `Edge::Fill` repeats keyed positions, which is an error. An axis that pads by nothing keeps a plain walk.
pub fn pad(layout: &Layout, widths: &[(usize, usize)], edge: Edge) -> Result<Plan, Error> {
    let shape = padded(layout.shape(), widths)?;
    generated_len(&shape)?;
    let mut result = layout.clone();
    for (axis, ((&size, &len), &(before, _))) in layout.shape().iter().zip(&shape).zip(widths).enumerate() {
        if len != size { result = result.select(axis, Steps::padded(-(before as i128), len, size, 1, edge).offsets())?; }
    }
    Ok(Plan { layout: result, steps: pad_steps(layout.shape(), widths, edge)? })
}

/// The lengths of `shape` after taking or dropping `counts`, each along its axis: a take keeps `|k|` positions, and a drop keeps `max(0, n-|k|)`. A later count for the same axis replaces an earlier one.
pub fn taken(shape: &[usize], counts: &[(usize, Count)], take: bool) -> Result<Vec<usize>, Error> {
    if counts.len() > shape.len() { return Err(Error::new(Kind::Rank, "counts exceed argument rank")); }
    let mut lengths = shape.to_vec();
    for &(axis, count) in counts {
        let Some(&len) = shape.get(axis) else { return Err(Error::new(Kind::Domain, "axis is outside array rank")) };
        lengths[axis] = if take { count.magnitude } else { len.saturating_sub(count.magnitude) };
    }
    Ok(lengths)
}

/// Where each axis of `shape` starts after taking or dropping `counts`: negative where a take pads before the axis.
fn take_starts(shape: &[usize], counts: &[(usize, Count)], take: bool) -> Vec<i128> {
    let mut starts = vec![0; shape.len()];
    for &(axis, Count { negative, magnitude: n }) in counts {
        let size = shape[axis];
        starts[axis] = match (take, negative) { (true, true) => size as i128 - n as i128, (false, false) => n.min(size) as i128, _ => 0 };
    }
    starts
}

/// The steps that read `counts` taken from `shape`, or dropped where `take` is false, one entry for each axis (`take`).
pub fn take_steps(shape: &[usize], counts: &[(usize, Count)], take: bool) -> Result<Vec<Steps>, Error> {
    let lengths = taken(shape, counts, take)?;
    let (starts, stride) = (take_starts(shape, counts, take), strides(shape));
    Ok((0..shape.len()).map(|a| Steps::padded(starts[a], lengths[a], shape[a], stride[a], Edge::Fill)).collect())
}

/// Takes `count` items along `axis` for each `(axis, count)` in `counts`, at the lengths that `taken` gives, or drops them when `take` is false. A negative count works from the end. A take past an end reads fill there. A later count for the same axis replaces an earlier one.
pub fn take(layout: &Layout, counts: &[(usize, Count)], take: bool) -> Result<Plan, Error> {
    let old = layout.shape();
    let shape = taken(old, counts, take)?;
    generated_len(&shape)?;
    let starts = take_starts(old, counts, take);
    let mut result = layout.clone();
    for a in (0..old.len()).filter(|&a| counts.iter().any(|&(axis, _)| axis == a)) {
        result = result.select(a, Steps::padded(starts[a], shape[a], old[a], 1, Edge::Fill).offsets())?;
    }
    Ok(Plan { layout: result, steps: take_steps(old, counts, take)? })
}

/// The lengths of `shape` rotated along `axis`, which are its own. `frame`, the shape of counts that give one count for each lane, must match `shape` without `axis`, which is one of its axes.
pub fn rotated<'a>(shape: &'a [usize], axis: usize, frame: Option<&[usize]>) -> Result<&'a [usize], Error> {
    if axis >= shape.len() { return Err(Error::new(Kind::Rank, "rotation axis is outside array rank")); }
    let others = || shape[..axis].iter().chain(&shape[axis + 1..]);
    if frame.is_some_and(|f| !f.iter().eq(others())) { return Err(Error::new(Kind::Length, "rotation counts must match the axis frame")); }
    Ok(shape)
}

/// The steps that rotate an axis by its count, or reverse it when no count is given.
pub fn rotate_steps(shape: &[usize], axis: usize, count: Option<Count>) -> Vec<Steps> {
    let stride = strides(shape);
    (0..shape.len()).map(|a| match count {
        _ if a != axis => Steps::along(shape[a], stride[a]),
        None => Steps::Stride { start: shape[a].saturating_sub(1) * stride[a], len: shape[a], step: -(stride[a] as isize) },
        Some(c) => {
            let n = shape[a];
            let r = if n == 0 { 0 } else { c.magnitude % n };
            Steps::rotated(if c.negative && r != 0 { n - r } else { r }, n, stride[a])
        }
    }).collect()
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
            rotated(shape, axis, (ns.len() != 1).then_some(frame))?;
            if len != 0 && (ns.is_empty() || ns.iter().any(|&n| shift(n) != shift(ns[0]))) {
                // Each lane has its own count, so the offsets don't separate by axis, and the axis loses its keys.
                let offsets =
                    (0..outer).flat_map(|i| (0..len).flat_map(move |j| (0..inner).map(move |k| lanes.offset(i, (j + shift(ns[i * inner + k])) % len, k))));
                return Ok(Plan { layout: layout.clone().without_keys(axis), steps: vec![Steps::table(offsets.collect())] });
            }
            Some(shift(ns.first().copied().unwrap_or(0)))
        }
    };
    let source = |j: usize| match count { None => len - 1 - j, Some(s) => (j + s) % len };
    Ok(Plan { layout: layout.select(axis, (0..len).map(|j| Some(source(j))))?,
        steps: rotate_steps(shape, axis, count.map(|s| Count { negative: false, magnitude: s })) })
}

/// One axis of windows: windows of `size` items, each `step` items after the last. A padded axis has a window centred on every `step`th position, and reads past its ends by its edge mode.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WindowAxis { pub size: usize, pub step: usize, pub edge: Option<Edge> }

impl WindowAxis {
    /// Whether the axis pads, with a window centred on its positions.
    pub fn padded(&self) -> bool { self.edge.is_some() }
}

impl WindowAxis {
    /// Where window `i` starts along the axis. It's negative when padding comes before the first item.
    pub fn start(&self, i: usize) -> isize { (i * self.step) as isize - if self.padded() { ((self.size - 1) / 2) as isize } else { 0 } }
}

/// The lengths of the windows of `shape` that `axes` give along its leading axes: the number of windows along each of those axes, then a window's lengths. Along an axis of length `n`, `⌊(n-size)÷step⌋+1` windows fit, or none where `size` is longer. A padded window is centred on each position, which gives `⌈(n-[size even])÷step⌉` windows. Its size must be less than twice the length of an axis that isn't empty.
pub fn windowed(shape: &[usize], axes: &[WindowAxis]) -> Result<Vec<usize>, Error> {
    if axes.len() > shape.len() { return Err(Error::new(Kind::Rank, "window specification has more axes than its argument")); }
    let mut lengths = Vec::with_capacity(axes.len() + shape.len());
    for (w, &n) in axes.iter().zip(shape) {
        lengths.push(match w.padded() {
            false if n < w.size => 0, false => (n - w.size) / w.step + 1,
            true if n == 0 || w.size / 2 < n => n.saturating_sub(usize::from(w.size.is_multiple_of(2))).div_ceil(w.step),
            true => return Err(Error::new(Kind::Domain, "padded window is too large for the argument"))
        });
    }
    lengths.extend(axes.iter().map(|w| w.size));
    lengths.extend_from_slice(&shape[axes.len()..]);
    Ok(lengths)
}

/// The windows' frame steps followed by the steps within each window.
pub fn window_steps(lengths: &[usize], axes: &[WindowAxis], source: &[usize]) -> Vec<Steps> {
    let stride = strides(source);
    let frame = axes.iter().enumerate().map(|(a, w)| Steps::along(lengths[a], w.step * stride[a]));
    frame.chain(lengths[axes.len()..].iter().zip(&stride).map(|(&len, &stride)| Steps::along(len, stride))).collect()
}

/// Pads the positions that windows read, or returns None when no axis pads or there are no windows.
pub fn window_padding_steps(shape: &[usize], axes: &[WindowAxis], lengths: &[usize]) -> Option<Vec<Steps>> {
    if !axes.iter().any(WindowAxis::padded) || lengths.contains(&0) { return None; }
    let stride = strides(shape);
    Some((0..shape.len()).map(|a| {
        let Some(w) = axes.get(a) else { return Steps::along(shape[a], stride[a]) };
        let before = if w.padded() { (w.size - 1) / 2 } else { 0 };
        let len = (lengths[a] - 1) * w.step + w.size;
        Steps::padded(-(before as i128), len, shape[a], stride[a], w.edge.unwrap_or(Edge::Fill))
    }).collect())
}

/// Windows over the leading axes of an array of one shape: a `WindowAxis` for each, and the lengths that `windowed` gives, the frame that counts the windows along each axis followed by the shape of one window.
#[derive(Clone, Debug)]
pub struct Windows { shape: Vec<usize>, axes: Vec<WindowAxis>, lengths: Vec<usize> }

impl Windows {
    /// The windows of `sizes` and `moves` over the leading axes of `shape`. Each size is a `Count`: its magnitude is the window's size, and `edge` gives the edge mode of its axis from it, or `None` for an axis that doesn't pad. An axis with no move moves by 1.
    pub fn new(sizes: &[i64], moves: &[i64], edge: impl Fn(Count) -> Option<Edge>, shape: &[usize]) -> Result<Self, Error> {
        if moves.len() > sizes.len() { return Err(Error::new(Kind::Length, "window specification has more movements than sizes")); }
        let axes = sizes.iter().enumerate().map(|(a, &size)| {
            let step = match moves.get(a) {
                None => 1, Some(&m) if m > 0 => Count::of(m).magnitude, Some(_) => return Err(Error::new(Kind::Domain, "window movements must be positive"))
            };
            let size = Count::of(size);
            let axis = WindowAxis { size: size.magnitude, step, edge: edge(size) };
            if axis.padded() && axis.size == 0 { return Err(Error::new(Kind::Domain, "padded window sizes must be positive")); }
            Ok(axis)
        }).collect::<Result<Vec<_>, _>>()?;
        let lengths = windowed(shape, &axes)?;
        Ok(Self { shape: shape.to_vec(), axes, lengths })
    }
    /// The number of windows along each windowed axis.
    pub fn frame(&self) -> &[usize] { &self.lengths[..self.axes.len()] }
    /// The shape of one window: the windows' sizes, then the argument's remaining axes.
    pub fn cell(&self) -> &[usize] { &self.lengths[self.axes.len()..] }
    /// The keys of the frame's axes. A padded axis keeps the key of the position each window centres on, and an unpadded axis has none.
    fn frame_keys(&self, layout: &Layout) -> Result<Vec<Option<Arc<Keys>>>, Error> {
        self.axes.iter().enumerate().map(|(a, w)| match layout.keys(a) {
            Some(k) if w.padded() => k.select((0..self.lengths[a]).map(|i| Some(i * w.step))).map(Some), _ => Ok(None)
        }).collect()
    }
    /// The keys of one window's axes: none on a windowed axis, so that every window has one layout, and `layout`'s own on the others.
    fn cell_keys(&self, layout: &Layout) -> Vec<Option<Arc<Keys>>> {
        (0..self.cell().len()).map(|a| if a < self.axes.len() { None } else { layout.keys(a).cloned() }).collect()
    }
    /// The plan that pads each padded axis by its edge mode, or `None` when no axis pads or there are no windows (`window_padding_steps`). Each padded axis of the result holds only the positions its windows read, and the result's unpadded windows are the padded windows of the argument.
    pub fn padding(&self) -> Option<Plan> {
        let steps = window_padding_steps(&self.shape, &self.axes, &self.lengths)?;
        let padded: Vec<_> = steps.iter().map(Steps::len).collect();
        Some(Plan { layout: padded.into(), steps })
    }
    /// The plan of every window over `layout`, which has the windows' shape: the frame's axes, then each window's axes. The windows read an array of shape `source`: `layout`'s shape, or the shape of `padding`'s result. The frame's axes have the keys that `frame_keys` gives, and each window's the keys that `cell_keys` gives.
    pub fn plan(&self, layout: &Layout, source: &[usize]) -> Result<Plan, Error> {
        let n = self.axes.len();
        generated_len(&self.lengths)?;
        generated_len(self.cell())?;
        let steps = window_steps(&self.lengths, &self.axes, source);
        let keys = self.frame_keys(layout)?.into_iter().chain(self.cell_keys(layout)).collect();
        let names = (0..n).chain(0..self.cell().len()).map(|a| layout.name(a).cloned()).collect();
        Ok(Plan { layout: Layout::from(self.lengths.clone()).with_keys(keys)?.inherit_names(names).unique_names(), steps })
    }
}

/// How a copy walks its source: one entry for each axis of the copy, the source axis that each entry walks, and the offset of the positions on source axes that no entry walks. Axes that walk one source axis together read windows, as [`Windows::plan`] lays them out. Every reader of windows from `Steps` classifies them here: [`Walk::read`] gives what the axes along one source axis read.
#[derive(Clone, Debug)]
pub struct Walk {
    /// The offset of the positions on source axes that no entry walks.
    pub base: usize,
    /// One entry for each axis of the copy. The first entry along each walked axis holds that axis's position in its start.
    pub steps: Vec<Steps>,
    /// The source axis that each entry walks, or `None` for an entry whose offsets are all 0, as for an axis of one item or a spread.
    pub along: Vec<Option<usize>>,
    /// The source's strides.
    pub strides: Vec<usize>,
}

impl Walk {
    /// The walk that `tables` give for a copy of `shape` from a source of shape `source`. `None` where an entry doesn't walk one source axis, where a walked axis's position can't move into the start of a `Stride`, as for a `Padded` or `Rotated` entry over part of an axis, and where the source is empty.
    pub fn new(tables: &[Steps], shape: &[usize], source: &[usize]) -> Option<Self> {
        let (base, mut steps) = Steps::per_axis(tables, shape)?;
        let strides = strides(source);
        if base >= crate::layout::count(source) { return None; }
        let along: Vec<Option<usize>> = steps.iter().map(|e| e.axis_of(source, &strides)).collect::<Option<_>>()?;
        let mut rest = 0;
        for (a, p) in crate::layout::coordinates(base, source).filter(|&(_, p)| p > 0) {
            let Some(r) = along.iter().position(|&b| b == Some(a)) else { rest += p * strides[a]; continue; };
            let Steps::Stride { start, .. } = &mut steps[r] else { return None };
            *start += p * strides[a];
            if steps[r].axis_of(source, &strides) != Some(Some(a)) { return None; }
        }
        Some(Self { base: rest, steps, along, strides })
    }
    /// The axes of the copy that walk source axis `a`, in order.
    pub fn walking(&self, a: usize) -> impl Iterator<Item = usize> + '_ { (0..self.along.len()).filter(move |&r| self.along[r] == Some(a)) }
    /// What the entries walking source axis `a`, of `n` items, read as windows, such as a copy's frames and the positions in each window: where the windows start, and each entry's length and move in positions. Each entry must be a `Stride` that steps forward by a whole number of positions. An axis that no entry walks reads position 0. `None` for any other entries, and where the windows don't lie inside the axis.
    pub fn read(&self, a: usize, n: usize) -> Option<(usize, Vec<[usize; 2]>)> {
        let unit = self.strides[a];
        if unit == 0 { return None; }
        let (mut first, mut last, mut read) = (0, 0, Vec::new());
        for r in self.walking(a) {
            let Steps::Stride { start, len, step } = self.steps[r] else { return None };
            if step <= 0 || step as usize % unit != 0 || start % unit != 0 || len == 0 { return None; }
            let moves = step as usize / unit;
            (first, last) = (first + start / unit, last + (len - 1) * moves);
            read.push([len, moves]);
        }
        (first + last < n).then_some((first, read))
    }
    /// The leading windows read by this copy, as `[number, width, step]` per axis. The copy's axes must be frames, windows, then unchanged source axes. Each frame starts at 0.
    pub fn windows(&self, source: &[usize]) -> Option<Vec<[usize; 3]>> {
        let s = self.steps.len().checked_sub(source.len())?;
        if self.base != 0 || s == 0 || 2 * s > self.steps.len() { return None; }
        if (s..source.len())
            .any(|a| !matches!(self.steps[s + a], Steps::Stride { start: 0, len, step } if len == source[a] && step == self.strides[a] as isize)) {
                return None;
            }
        (0..s).map(|a| {
            if self.along[a] != Some(a) || self.along[s + a] != Some(a) { return None; }
            match self.read(a, source[a])? { (0, read) => match read[..] { [[n, moves], [w, 1]] => Some([n, w, moves]), _ => None }, _ => None }
        }).collect()
    }
}

/// A fold along one or two axes of a copy whose axes walk windows of its source, run as sliding folds over the source's rows without building the windows ([`crate::reduce::sliding`] and [`crate::reduce::sliding_2d`]). [`Walk`] classifies the copy. Each folded axis is a window's axis that moves one position at a time along its source axis. The copy's other axes along that source axis move by 1, where the windows slide, or by whole windows, where they tile it, as pooling's do. basedpl folds its pending copies of windows this way, and the Python API its views of windows.
#[derive(Clone, Debug)]
pub struct Slide {
    /// How the copy walks its source.
    pub walk: Walk,
    /// The offsets of the source items that the folds read: for one fold, the rows that it reads along its source axis, or the whole source where a source axis before the fold's holds more than one item, and for two folds the whole source.
    pub region: std::ops::Range<usize>,
    /// The region as rows, with each fold along a middle axis: `[outer, len, inner]` for one folded axis, where windows that tile fold one tile to a row, and `[outer, len1, mid, len2, inner]` for two, in the order of their source axes.
    pub dims: Vec<usize>,
    /// Each fold's window width, in the order of `dims`.
    pub widths: Vec<usize>,
    /// The positions between the starts of each fold's windows, in the order of `dims`: 1 where they slide, and the width where they tile.
    pub every: Vec<usize>,
    /// The tables through which the copy without its folded axes reads the folds' results.
    pub tables: Vec<Steps>,
}

impl Slide {
    /// The sliding folds along `axes`, one axis or two, of the copy of `shape` that `tables` read from a source of shape `source`. Two folds need windows along two source axes, each covering its whole axis, and a last source axis after both of at least two items, as [`crate::reduce::sliding_2d`] takes them. `None` for any other copy: one that reads fills, one with an axis of more than one item that walks no source axis, one that reads a source axis that none of its axes walks at a position other than 0, one whose folded axis doesn't move by 1 along its source axis or is the only axis along it, and one whose other axes along a folded axis's source axis neither move by 1 nor tile it. `None` also for a window narrower than two items.
    pub fn new(tables: &[Steps], shape: &[usize], source: &[usize], axes: &[usize]) -> Option<Self> {
        let walk = Walk::new(tables, shape, source)?;
        // An axis of the copy that walks no source axis holds one item.
        if walk.base != 0 || walk.steps.iter().zip(&walk.along).any(|(s, a)| a.is_none() && s.len() != 1) { return None; }
        let reads = (0..source.len()).map(|a| walk.read(a, source[a])).collect::<Option<Vec<_>>>()?;
        let moves: Vec<usize> = walk.steps.iter().zip(&walk.along).map(|(s, a)| match (s, a) {
            (Steps::Stride { step, .. }, Some(a)) => *step as usize / walk.strides[*a], _ => 0
        }).collect();
        // Each fold's source axis, the first row and number of rows that it reads along that axis, the positions between its windows' starts, and its width. A source axis before the fold's means the fold reads every row of its axis.
        let mut lanes = axes.iter().map(|&k| {
            let (q, width) = (walk.along[k]?, shape[k]);
            if width < 2 || moves[k] != 1 || walk.walking(q).count() < 2 { return None; }
            let others: Vec<usize> = walk.walking(q).filter(|&r| r != k).map(|r| moves[r]).collect();
            let every = if others.iter().all(|&m| m == 1) { 1 } else if others.iter().all(|&m| m % width == 0) { width } else { return None; };
            let (first, read) = &reads[q];
            let len = 1 + read.iter().map(|&[n, m]| (n - 1) * m).sum::<usize>();
            let (lo, len) = if crate::layout::count(&source[..q]) == 1 { (*first, len) } else { (0, source[q]) };
            ((first - lo) % every == 0 && len % every == 0).then_some([q, lo, len, every, width])
        }).collect::<Option<Vec<_>>>()?;
        lanes.sort_unstable();
        let count = |range: std::ops::Range<usize>| crate::layout::count(&source[range]);
        let (region, dims) = match lanes[..] {
            [[q, lo, len, every, width]] => {
                let (outer, inner) = (count(0..q), walk.strides[q]);
                let region = if outer == 1 { lo * inner..(lo + len) * inner } else { 0..count(0..source.len()) };
                (region, if every == 1 { vec![outer, len, inner] } else { vec![outer * len / width, width, inner] })
            }
            [[q1, lo1, len1, ..], [q2, lo2, len2, ..]] => {
                let whole = lo1 == 0 && lo2 == 0 && len1 == source[q1] && len2 == source[q2];
                if q1 == q2 || !whole { return None; }
                let dims = vec![count(0..q1), source[q1], count(q1 + 1..q2), source[q2], count(q2 + 1..source.len())];
                if dims[4] < 2 { return None; }
                (0..count(0..source.len()), dims)
            }
            _ => return None,
        };
        let mut folded = source.to_vec();
        for &[q, _, len, every, width] in &lanes { folded[q] = (len - width) / every + 1; }
        let folded_strides = strides(&folded);
        let mut started = vec![false; folded.len()];
        // Each table moves by its axis's moves, in starts along a folded axis, and the first of them along each source axis starts where the copy does.
        let tables = (0..shape.len()).filter(|r| !axes.contains(r)).map(|r| match walk.along[r] {
            Some(a) => {
                let (every, lo) = lanes.iter().find(|l| l[0] == a).map_or((1, 0), |l| (l[3], l[1]));
                let start = if std::mem::replace(&mut started[a], true) { 0 } else { (reads[a].0 - lo) / every };
                Steps::Stride { start: start * folded_strides[a], len: shape[r], step: (moves[r] / every * folded_strides[a]) as isize }
            }
            None => Steps::Stride { start: 0, len: shape[r], step: 0 },
        }).collect();
        Some(Self { region, dims, widths: lanes.iter().map(|l| l[4]).collect(), every: lanes.iter().map(|l| l[3]).collect(), tables, walk })
    }
}

/// The row-major shape of a source of `len` items whose axes the entries of `tables` step along, for a copy of `shape`, for a caller that holds a source's storage without its shape, as a Python view does. It has one axis for each distinct stride, and each axis holds as many items as the next larger stride spans. `None` where a stride doesn't divide the next, or `len` isn't a whole number of the largest.
pub fn walked_source(tables: &[Steps], shape: &[usize], len: usize) -> Option<Vec<usize>> {
    let (_, steps) = Steps::per_axis(tables, shape)?;
    let mut units: Vec<usize> = steps.iter().filter_map(|s| match *s {
        Steps::Stride { len, step, .. } if len > 1 && step != 0 => Some(step.unsigned_abs()), _ => None
    }).collect();
    units.push(1);
    units.sort_unstable();
    units.dedup();
    let largest = *units.last()?;
    if len % largest != 0 || units.windows(2).any(|w| w[1] % w[0] != 0) { return None; }
    Some(std::iter::once(len / largest).chain(units.windows(2).rev().map(|w| w[1] / w[0])).collect())
}

/// The lengths of `shape` transposed by `axes`, which sends axis `i` to result axis `axes[i]`. Axes sent to the same result axis select their diagonal, at the shortest of their lengths. Every result axis from 0 to the largest destination needs a source.
pub fn transposed(shape: &[usize], axes: &[usize]) -> Result<Vec<usize>, Error> {
    let rank = axes.iter().max().map_or(0, |n| n + 1);
    if (0..rank).any(|r| !axes.contains(&r)) { return Err(Error::new(Kind::Rank, "transpose axes must be consecutive from 0")); }
    if axes.len() != shape.len() { return Err(Error::new(Kind::Length, "transpose needs one destination for each axis")); }
    let (mut lengths, mut seen) = (vec![0usize; rank], 0u128);
    for (&axis, &len) in axes.iter().zip(shape) {
        lengths[axis] = if seen >> axis & 1 == 1 { lengths[axis].min(len) } else { len };
        seen |= 1 << axis;
    }
    Ok(lengths)
}

/// The steps that read `shape` transposed by `axes`, one entry for each result axis (`transpose`).
pub fn transpose_steps(shape: &[usize], axes: &[usize]) -> Result<Vec<Steps>, Error> {
    let lengths = transposed(shape, axes)?;
    let mut stride = vec![0usize; lengths.len()];
    for (&axis, step) in axes.iter().zip(strides(shape)) { stride[axis] = stride[axis].saturating_add(step); }
    Ok(lengths.iter().zip(&stride).map(|(&len, &stride)| Steps::along(len, stride)).collect())
}

/// Sends axis `i` of `layout` to result axis `axes[i]`, at the lengths that `transposed` gives. A result axis keeps the keys and name its source axes agree on.
pub fn transpose(layout: &Layout, axes: &[usize]) -> Result<Plan, Error> {
    let source = layout.shape();
    let shape = transposed(source, axes)?;
    let sources: Vec<Vec<_>> = (0..shape.len()).map(|a| (0..axes.len()).filter(|&i| axes[i] == a).collect()).collect();
    let steps = transpose_steps(source, axes)?;
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
    let keys = sources.iter().zip(&shape).map(|(s, &len)| if s.iter().all(|&a| lengths[a] == len) { agreed(s.iter().map(|&a| layout.keys(a))) } else { None })
        .collect();
    let names = sources.iter().enumerate().map(|(i, s)| names.get(i).and_then(Clone::clone).or_else(|| agreed(s.iter().map(|&a| layout.name(a))))).collect();
    Layout::from(shape).with_keys(keys)?.with_names(names)
}

/// The lengths of `left` and `right` joined along `axis`: their lengths along it added. They must have the same rank, and the same lengths on their other axes.
pub fn catenated(left: &[usize], right: &[usize], axis: usize) -> Result<Vec<usize>, Error> {
    let rank = left.len();
    if right.len() != rank { return Err(Error::new(Kind::Rank, "catenate ranks differ")); }
    if axis >= rank { return Err(Error::new(Kind::Domain, "catenate axis is outside result rank")); }
    if (0..rank).any(|a| a != axis && left[a] != right[a]) { return Err(Error::new(Kind::Length, "catenate frames differ")); }
    let mut shape = left.to_vec();
    shape[axis] = left[axis].saturating_add(right[axis]);
    Ok(shape)
}

/// The layout of `left` and `right` joined along `axis`, with the lengths that `catenated` gives. The joined axis has both sides' keys, with no key where a side has none, and each other axis has either side's keys. A name survives where both sides agree.
pub fn catenate(left: &Layout, right: &Layout, axis: usize) -> Result<Layout, Error> {
    let shape = catenated(left.shape(), right.shape(), axis)?;
    generated_len(&shape)?;
    let names = |k: Option<&Arc<Keys>>, n: usize| k.map_or_else(|| vec![None; n], |k| k.names().to_vec());
    let keys = (0..shape.len()).map(|a| match (left.keys(a), right.keys(a)) {
        (x, y) if a != axis => Ok(x.or(y).cloned()), (None, None) => Ok(None),
        (x, y) => Keys::partial([names(x, left.shape()[a]), names(y, right.shape()[a])].concat()).map(Some)
    }).collect::<Result<_, Error>>()?;
    let names = (0..shape.len()).map(|a| agreed([left.name(a), right.name(a)])).collect();
    Ok(Layout::from(shape).with_keys(keys)?.inherit_names(names))
}
