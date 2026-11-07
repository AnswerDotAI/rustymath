use super::*;

/// The widest window that [`sliding`] folds directly, with one pass over each block of a lane for each of the window's rows. Wider windows take van Herk-Gil-Werman.
pub const DIRECT: usize = 8;
/// The items in each block of a lane that [`sliding`]'s direct folds finish before moving on, so that a block and the rows its windows read stay in cache.
const SLIDE_BLOCK: usize = 1024;

/// Van Herk-Gil-Werman's folds of the windows of `width` rows in `lane`, rows of `inner` items, into `folds`. `suffix` holds the lane's block suffixes, and `row` carries one row's running fold. `ONE` fixes `inner` at 1, which compiles each loop over a row's items to scalar code.
#[inline(always)]
fn hgw<T: Copy, R: Reduction<T>, const ONE: bool>(
    lane: &[T], folds: &mut [R::Total], suffix: &mut [R::Total], row: &mut [R::Total], width: usize, inner: usize, ok: &mut bool) {
    let inner = if ONE { 1 } else { inner };
    // A local row of one item stays in a register, where a row in memory would put a store and a load in each step's chain. The loops index the row item by item: `steps_of` put it in memory and made `⌈/50↕v` 1.4 times slower.
    let mut one = [R::unit()];
    let row = if ONE { &mut one[..] } else { &mut row[..inner] };
    for (b, block) in lane.chunks(width * inner).enumerate() {
        let (start, last) = (b * width * inner, block.len() - inner);
        // Each row's suffix folds it with the rest of its block, from the block's last row back to its first. Each step of a suffix or a prefix waits for the one before, so both combine through `chain`, the earlier items on the left.
        let s = &mut suffix[start..start + block.len()];
        for j in 0..inner {
            row[j] = one_of::<T, R>(block[last + j], ok);
            s[last + j] = row[j];
        }
        for k in (0..last).step_by(inner).rev() {
            for j in 0..inner {
                row[j] = chain_of::<T, R>(one_of::<T, R>(block[k + j], ok), row[j], ok);
                s[k + j] = row[j];
            }
        }
        // Each row's prefix folds the block up to it. The first block's whole prefix is the first window.
        for j in 0..inner { row[j] = one_of::<T, R>(block[j], ok); }
        if b == 0 {
            for k in (inner..block.len()).step_by(inner) { for j in 0..inner { row[j] = chain_of::<T, R>(row[j], one_of::<T, R>(block[k + j], ok), ok); } }
            folds[..inner].copy_from_slice(row);
            continue;
        }
        // In a later block, row `i` ends the window that starts `i + 1` rows into the block before: the fold of that block's suffix there and this prefix.
        let from = start + inner - width * inner;
        let (windows, earlier) = (&mut folds[from..from + block.len()], &suffix[from..from + block.len()]);
        for j in 0..inner { windows[j] = merge_of::<T, R>(earlier[j], row[j], ok); }
        for k in (inner..block.len()).step_by(inner) {
            for j in 0..inner {
                row[j] = chain_of::<T, R>(row[j], one_of::<T, R>(block[k + j], ok), ok);
                windows[k + j] = merge_of::<T, R>(earlier[k + j], row[j], ok);
            }
        }
        // A window that starts this block is the whole block, which the prefix holds once its last row is in.
        if block.len() == width * inner { folds[start..start + inner].copy_from_slice(row); }
    }
}

/// The folds by `R` of each window of `width` neighbouring rows along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row. The result has shape `[outer, len - width + 1, inner]`, with whether every fold fits. Each fold combines its window's items in order, except that a window as long as a lane of single items folds as [`along`] folds the lane. A window of at most [`DIRECT`] rows folds directly. A wider one takes van Herk-Gil-Werman's method: each lane splits into blocks of `width` rows, and each window is the fold of one block's suffix and the next block's prefix, so an item costs the same whatever the width.
pub fn sliding<T: Copy + Sync, R: Reduction<T>>(data: &[T], [outer, len, inner]: [usize; 3], width: usize) -> (Vec<R::Total>, bool) {
    assert_eq!(data.len(), outer * len * inner, "the data must fill its dimensions");
    assert!((1..=len.max(1)).contains(&width), "a window lies within its lane");
    let finished = |(totals, fits)| {
        if fits { crate::events::taken(crate::events::Route::SlidingFold, data.len()); }
        (totals, fits)
    };
    let count = (len + 1).saturating_sub(width);
    if outer * count * inner == 0 { return finished((Vec::new(), true)); }
    // A window as long as a lane of single items is that lane's fold, which `along` runs side by side with its neighbours'. Wider lanes fold directly, starting from their first row.
    if width == len && inner == 1 { return finished(along::<T, R>(data, [outer, len, inner])); }
    let mut ok = true;
    if width <= DIRECT {
        // Each block of a lane's windows folds while its rows are in cache (`rows_folded`), which reads the lane from memory once.
        let block = (SLIDE_BLOCK / inner).max(1);
        let mut out = Vec::with_capacity(outer * count * inner);
        for lane in data.chunks_exact(len * inner) {
            for r in (0..count).step_by(block) {
                let rows = block.min(count - r);
                ok &= rows_folded::<T, R>(&mut out, width, |i| &lane[(r + i) * inner..(r + i + rows) * inner]);
            }
        }
        return finished((out, ok));
    }
    let mut out = vec![R::unit(); outer * count * inner];
    let lanes = data.chunks_exact(len * inner).zip(out.chunks_exact_mut(count * inner));
    let (mut suffix, mut row) = (vec![R::unit(); len * inner], vec![R::unit(); inner]);
    for (lane, folds) in lanes {
        if inner == 1 { hgw::<T, R, true>(lane, folds, &mut suffix, &mut row, width, inner, &mut ok) }
        else { hgw::<T, R, false>(lane, folds, &mut suffix, &mut row, width, inner, &mut ok) }
    }
    finished((out, ok))
}

/// The folds by `R` of each window of `w1` rows along axis 1 and `w2` rows along axis 3 of `data`, an array of shape `[outer, len1, mid, len2, inner]` held row by row. Windows start every `e1` positions along axis 1 and every `e2` along axis 3: every position for windows that slide, or every window's width for windows that tile the axis, as pooling's do. The result has shape `[outer, c1, mid, c2, inner]`, where `c1` and `c2` count the windows along each axis, with whether every fold fits. Each band of `w1` rows folds into a buffer that stays in cache, and the windows along axis 3 fold from that buffer, so `data` is read once. A window's items fold along axis 1 and then along axis 3, which suits reductions that fold in any order. `None` for a window wider than [`DIRECT`] rows.
pub fn sliding_2d<T: Copy + Sync, R: Reduction<T>>(
    data: &[T], [outer, len1, mid, len2, inner]: [usize; 5], [w1, w2]: [usize; 2], [e1, e2]: [usize; 2]) -> Option<(Vec<R::Total>, bool)> {
    assert_eq!(data.len(), outer * len1 * mid * len2 * inner, "the data must fill its dimensions");
    assert!(w1 > 0 && w2 > 0 && e1 > 0 && e2 > 0, "windows hold items and start in order");
    if w1 > DIRECT || w2 > DIRECT { return None; }
    let (c1, c2) = ((len1 + e1).saturating_sub(w1) / e1, (len2 + e2).saturating_sub(w2) / e2);
    if outer * c1 * mid * c2 * inner == 0 { return Some((Vec::new(), true)); }
    let (row, plane) = (len2 * inner, mid * len2 * inner);
    let mut out = Vec::with_capacity(outer * c1 * mid * c2 * inner);
    let (mut band, mut ok) = (Vec::with_capacity(plane), true);
    for lane in data.chunks_exact(len1 * plane) {
        for i in 0..c1 {
            band.clear();
            ok &= rows_folded::<T, R>(&mut band, w1, |a| &lane[(e1 * i + a) * plane..(e1 * i + a + 1) * plane]);
            for m in band.chunks_exact(row) {
                let at = out.len();
                // Windows that slide fold as `sliding`'s direct folds do, one pass over the band's row for each of the window's rows. Windows that tile fold one window at a time.
                if e2 == 1 {
                    out.extend_from_slice(&m[..c2 * inner]);
                    for b in 1..w2 { ok &= merges_of::<T, R>(&mut out[at..], &m[b * inner..]); }
                } else {
                    for j in 0..c2 { out.extend_from_slice(&m[e2 * j * inner..(e2 * j + 1) * inner]); }
                    for b in 1..w2 {
                        for j in 0..c2 { ok &= merges_of::<T, R>(&mut out[at + j * inner..at + (j + 1) * inner], &m[(e2 * j + b) * inner..]); }
                    }
                }
            }
        }
    }
    if ok { crate::events::taken(crate::events::Route::SlidingFold, data.len()); }
    Some((out, ok))
}

/// A reduction that picks one of its items, as [`Max`] and [`Min`] do, over pairs of an item and its row. Each fold keeps the row of the first item that gives it.
struct Pick<R>(std::marker::PhantomData<R>);

impl<T: Copy + PartialEq + Send + Sync, R: Reduction<T, Total = T>> Reduction<(T, usize)> for Pick<R> {
    type Total = (T, usize);
    #[inline(always)]
    fn unit() -> (T, usize) { (R::unit(), 0) }
    #[inline(always)]
    fn one((x, row): (T, usize)) -> ((T, usize), bool) { ((R::one(x).0, row), true) }
    #[inline(always)]
    fn step(acc: (T, usize), x: (T, usize)) -> (T, usize) { Self::merge(acc, Self::one(x).0).0 }
    /// The earlier fold `a` where it gives the fold of both, and `b` where `b`'s item is beyond it.
    #[inline(always)]
    fn merge(a: (T, usize), b: (T, usize)) -> ((T, usize), bool) { (if R::merge(a.0, b.0).0 == a.0 { a } else { b }, true) }
    fn slice(x: &[(T, usize)]) -> ((T, usize), bool) { (x.iter().fold(Self::unit(), |t, &v| Self::step(t, v)), true) }
    fn scan(x: &[(T, usize)], start: Option<(T, usize)>, out: &mut Vec<(T, usize)>) -> bool {
        let mut total = start;
        for &v in x {
            let t = total.map_or_else(|| Self::one(v).0, |t| Self::step(t, v));
            out.push(t);
            total = Some(t);
        }
        true
    }
}

/// The folds that [`picks`] finds the extremes of: windows of a width that slide along each lane, as [`sliding`] folds them, each whole lane, as [`along`] folds it, or each lane's running folds, as [`scan_along`] gives them.
#[derive(Clone, Copy, Debug)]
pub enum Picks { Windows(usize), Lanes, Running }

/// The row of the first item that gives each fold by `R` along the middle axis of `data`, an array of shape `[outer, len, inner]`, for a reduction that picks one of its items, as [`Max`] and [`Min`] do. `of` chooses the folds, and the result lays out as their results do. Each row counts from the start of its lane. A fold whose items all leave it at `R::unit()`, as NaN items do for a float maximum, gives its first row.
pub fn picks<T: Copy + PartialEq + Send + Sync, R: Reduction<T, Total = T>>(data: &[T], shape: [usize; 3], of: Picks) -> Vec<usize> {
    let [outer, len, inner] = shape;
    match of {
        Picks::Windows(width) => {
            let pairs: Vec<(T, usize)> = data.iter().enumerate().map(|(k, &x)| (x, k / inner.max(1) % len.max(1))).collect();
            sliding::<(T, usize), Pick<R>>(&pairs, shape, width).0.into_iter().map(|(_, row)| row).collect()
        }
        // A lane's pick is its first item that, folded alone, gives the lane's fold. Along a lane of consecutive items, a vectorized search finds it after the fold.
        Picks::Lanes if inner == 1 => along::<T, R>(data, shape).0.into_iter().enumerate().map(|(o, total)| {
            let row = first_where(&data[o * len..][..len], |x| R::one(x).0 == total);
            if row == len { 0 } else { row }
        }).collect(),
        // Lanes across an axis fold in one pass in memory order, and each lane's pick steps through `moved`.
        Picks::Lanes => {
            let mut rows = vec![0; outer * inner];
            if len > 0 && inner > 0 {
                for (block, lanes) in data.chunks_exact(len * inner).zip(rows.chunks_exact_mut(inner)) {
                    let mut totals: Vec<T> = block[..inner].iter().map(|&x| R::one(x).0).collect();
                    for (k, items) in block.chunks_exact(inner).enumerate().skip(1) {
                        for ((row, total), &x) in lanes.iter_mut().zip(&mut totals).zip(items) {
                            let next = R::merge(*total, R::one(x).0).0;
                            *row = moved(*row, *total, next, k);
                            *total = next;
                        }
                    }
                }
            }
            rows
        }
        // A running fold's pick at each item steps through `moved` from the item before it in the lane.
        Picks::Running => {
            let mut totals = Vec::with_capacity(data.len());
            scan_along::<T, R>(data, shape, None, &mut totals);
            let mut rows = vec![0; data.len()];
            for at in 0..data.len() { let k = at / inner % len; if k > 0 { rows[at] = moved(rows[at - inner], totals[at - inner], totals[at], k) } }
            rows
        }
    }
}

/// [`picks`] of the items of `x` at their own type, by [`Max`], or by [`Min`] where `max` is false. `None` for complex numbers and characters, which have no order.
pub fn view_picks(x: View<'_>, dims: [usize; 3], of: Picks, max: bool) -> Option<Vec<usize>> {
    fn picked<T: Copy + PartialEq + Send + Sync>(x: &[T], dims: [usize; 3], of: Picks, max: bool) -> Vec<usize> where Max: Reduction<T, Total = T>,
        Min: Reduction<T, Total = T> { if max { picks::<T, Max>(x, dims, of) } else { picks::<T, Min>(x, dims, of) } }
    Some(match x {
        View::Booleans(x) => picked(x, dims, of, max), View::Integers(x) => with_ints!(x, |x| picked(x, dims, of, max)),
        View::Floats(x) => with_floats!(x, |x| picked(x, dims, of, max)), _ => return None
    })
}
