use super::*;

/// The folds by `R` of the lanes along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row: one fold for each of the `outer × inner` lanes, and whether every one fits. Where a lane along the last axis, or a group of `inner` lanes along another axis, fits in one piece of [`fold_pieces`], each lane folds whole. Runs of whole lanes or groups then split across threads as [`parallel::split`] splits a call. An array with too few rows to give each of this thread's [`parallel::threads`] a piece of [`ROWS`] rows splits by columns instead, where a row holds more than one unit: a run of lanes whose part of the array holds at least [`PIECE`] items. Each thread then reads part of every row. At 4 threads on an M5 Max, `+⌿` took 41% less time on 8×524288, which had folded on one thread, and 17% less on 64×65536 than split by rows. With more rows, a split by columns took more time than a split by rows: 33% more on 256×16384 and 66% more on 4000×9000. Otherwise the array folds as `fold_pieces` folds it.
pub fn along<T: Copy + Sync, R: Reduction<T>>(data: &[T], shape: [usize; 3]) -> (Vec<R::Total>, bool) {
    let [outer, len, inner] = shape;
    let mut totals = vec![R::unit(); outer * inner];
    if len == 0 || totals.is_empty() { return (totals, true); }
    let (size, columns) = (piece_size(inner), PIECE.div_ceil(len));
    let unit = if inner > columns && outer * len <= ROWS * parallel::threads() { Some(columns) }
    else { (len * inner <= size).then(|| inner * (size / (len * inner))) };
    let fits = match unit {
        Some(unit) => parallel::split(&mut totals, unit, &|first, slots| lanes::<T, R>(slots, first, data, shape)),
        None => fold_pieces::<T, R>(&mut totals, shape, &|range, fold| fold(range.start, Part::Items(&data[range]))),
    };
    (totals, fits)
}

/// Items of an array that [`along_mapped`] copies through a mapping at a time. A fold along an earlier axis steps strips of columns down each block's rows. A block of 4096 items holds too few rows of a 1024-column matrix for that to pay. Larger blocks also cost a strided read's walk less for each item.
const MAPPED: usize = 1 << 16;

/// The shortest run of consecutive items that [`along_mapped`] folds where it lies. Each run costs its own call into the fold, so shorter runs fold faster copied a piece at a time. On one thread, folding runs of 128 floats in place took 9% less time than copying them, and runs of 64 took 15% more.
pub const RUN: usize = 128;

/// The runs of at least [`RUN`] consecutive items that `mapping` reads over `positions` positions: their length, and the offset of each run's first item. `None` where the mapping reads shorter runs, gives a fill, or reads its runs through a padded or rotated axis, as [`Mapping::runs`] says.
fn long_runs(mapping: &Mapping, positions: usize) -> Option<(usize, Vec<usize>)> {
    if !mapping.regular().is_some_and(|(len, stride)| stride == 1 && len >= RUN) { return None; }
    mapping.runs(positions).map(|(len, _, starts)| (len, starts))
}

/// Whether [`along_mapped`] reads the array of `shape` that `mapping` lays over storage where it lies, with no copy: a block of storage, as [`Mapping::block`] finds, or runs of at least [`RUN`] items. Elsewhere each read copies the array's items a piece at a time, at about a gather's cost, so a caller that will read the array again keeps a compact copy instead, as basedpl does for a pending copy that a name holds and the Python API does for a view that keeps what it gathers.
pub fn rereads_cheaply(mapping: &Mapping, shape: &[usize]) -> bool {
    mapping.block(shape).is_some() || long_runs(mapping, crate::layout::count(shape)).is_some()
}

/// How a fold reads an array that a mapping lays over storage, as [`fold_route`] chooses.
pub enum FoldRoute {
    /// Sliding folds over the source, as [`Slide`](crate::structure::Slide) classifies the array as windows along the folded axes.
    Slide(crate::structure::Slide),
    /// Through the mapping, as [`along_mapped`] reads it.
    Mapped,
    /// From the array's items, which it gathers once and keeps, or reads in place where a fold of every item reads a block of storage.
    Lanes,
}

/// The route of a fold along `axes`, one axis or two, of the array of `shape` that `mapping` lays over storage. It slides where `source` gives the shape of what the mapping reads and the array is windows along `axes`. Otherwise it reads through the mapping, unless the array keeps the items it gathers (`keeps`) and [`rereads_cheaply`] is false, where it folds the kept items. A fold of every item takes no axes, never slides, and folds a block of storage from the array's items ([`ArrayView::fold_items`](crate::array::ArrayView::fold_items) reads it in place). Python's folds of views and basedpl's folds of pending copies both choose here.
pub fn fold_route(mapping: &Mapping, shape: &[usize], source: Option<&[usize]>, keeps: bool, axes: &[usize]) -> FoldRoute {
    if !axes.is_empty() && let Some(source) = source && let Some(s) = crate::structure::Slide::new(&mapping.tables(shape), shape, source, axes) {
        return FoldRoute::Slide(s);
    }
    // A fold of every item reads a block of storage where it lies, in any order, through the array's items. Through the mapping it would read the block as a vector of every item, where the block isn't one.
    if axes.is_empty() && mapping.block(shape).is_some() { return FoldRoute::Lanes; }
    if !keeps || rereads_cheaply(mapping, shape) { FoldRoute::Mapped } else { FoldRoute::Lanes }
}

/// Folds each lane of the array described by `mapping` without gathering it. Returns totals, whether every integer total fits, and an optional mapping from those totals to result order.
pub fn along_mapped<T: Element, R: Reduction<T>>(data: &[T], mapping: &Mapping, shape: &[usize], axis: usize) -> (Vec<R::Total>, bool, Option<Vec<Steps>>) {
    let positions = crate::layout::count(shape);
    crate::events::taken(crate::events::Route::MappedFold, positions);
    if let Some(block) = mapping.block(shape).filter(|block| R::COMMUTATIVE || block.axes[axis].is_none_or(|(_, reversed)| !reversed)) {
        crate::events::taken(crate::events::Route::InPlace, positions);
        let (dims, order) = block.fold(Some(axis));
        let (totals, fits) = along::<T, R>(&data[block.range.clone()], dims);
        return (totals, fits, order);
    }
    let dims = Lanes::new(shape, axis).expect("the folded axis is within the shape").dims();
    let mut totals = vec![R::unit(); dims[0] * dims[2]];
    if dims[1] == 0 || totals.is_empty() { return (totals, true, None); }
    let runs = long_runs(mapping, positions);
    if runs.is_some() { crate::events::taken(crate::events::Route::InPlace, positions); }
    let walked = Walked::from(mapping);
    let fits = fold_pieces::<T, R>(&mut totals, dims, &|range, fold| match &runs {
        Some((len, starts)) => {
            let (len, mut at) = (*len, range.start);
            while at < range.end {
                let (start, n) = (starts[at / len] + at % len, (len - at % len).min(range.end - at));
                fold(at, Part::Items(&data[start..start + n]));
                at += n;
            }
        }
        None => {
            let mut buf = Vec::new();
            for at in range.clone().step_by(MAPPED) { fold(at, Part::Items(expanded_items(&walked, data, at, MAPPED.min(range.end - at), &mut buf))); }
        }
    });
    (totals, fits, None)
}

