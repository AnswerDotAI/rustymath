use super::*;

/// A block of an array that [`fold_pieces`] folds, as its producer hands it over: the block's items, or the reduction's total of its `n` items, where the items all lie in one lane along the last axis.
pub enum Part<'a, T, S> { Items(&'a [T]), Total(usize, S) }

/// Folds an array of shape `[outer, len, inner]` by `R` into `totals`, the folds of its lanes along the middle axis, and gives whether every fold fits. `run(range, fold)` produces the array's items in `range` in order, and hands each block of them to `fold` with the position of its first item, as a [`Part`]: the items, or their total. The array folds in one piece for each of this thread's [`parallel::threads`], each of at least [`piece_size`] items. On one thread it folds as one piece. A lane that one piece holds folds as [`fold_into`] folds it, and a block's total joins it as [`fold_total`] joins it. A lane that crosses pieces folds its part in each piece, and those folds merge.
pub fn fold_pieces<T: Copy, R: Reduction<T>>(
    totals: &mut [R::Total], dims: [usize; 3], run: &(dyn Fn(Range<usize>, &mut dyn FnMut(usize, Part<'_, T, R::Total>)) + Sync)) -> bool {
    let [outer, len, inner] = dims;
    // A unit is a lane along the last axis, or a group of `inner` lanes along another axis.
    let (n, unit, least) = (outer * len * inner, len * inner, piece_size(inner));
    let size = n.div_ceil((n / least).clamp(1, parallel::threads())).max(least).next_multiple_of(CHUNK);
    let mut pieces = Vec::with_capacity(n.div_ceil(size));
    let (mut rest, mut taken) = (&mut *totals, 0);
    for start in (0..n).step_by(size) {
        let end = n.min(start + size);
        let (lo, hi) = (start.div_ceil(unit), (end / unit).max(start.div_ceil(unit)));
        let (_, after) = std::mem::take(&mut rest).split_at_mut((lo - taken) * inner);
        let (own, after) = after.split_at_mut((hi - lo) * inner);
        (rest, taken) = (after, hi);
        let (front, back) = (end.min(lo * unit), end.min(lo * unit).max(hi * unit));
        let partial = |at: usize| (at / unit, vec![R::identity(); inner]);
        pieces.push(Piece { range: start..end, lo, hi, own, front: (start < front).then(|| partial(start)), back: (back < end).then(|| partial(back)) });
    }
    let mut fits = parallel::pieces_of(&mut pieces, 1, usize::MAX, &|_, pieces| pieces.iter_mut().fold(true, |ok, piece| ok & piece.fold::<T, R>(dims, run)));
    let partials: Vec<_> = pieces.into_iter().flat_map(|piece| [piece.front, piece.back]).flatten().collect();
    let mut last = usize::MAX;
    for (u, part) in partials {
        let folds = &mut totals[u * inner..(u + 1) * inner];
        if u == last { fits &= merges_of::<T, R>(folds, &part); } else {
            folds.copy_from_slice(&part);
            last = u;
        }
    }
    fits
}

/// One piece of [`fold_pieces`]: its items, the units `lo..hi` that it holds whole with their folds, and the partial folds of the units that it shares with the pieces before and after it, each with its unit. A partial fold starts at `R::identity()`.
struct Piece<'a, S> { range: Range<usize>, lo: usize, hi: usize, own: &'a mut [S], front: Option<(usize, Vec<S>)>, back: Option<(usize, Vec<S>)> }

impl<S: Copy> Piece<'_, S> {
    /// Produces the piece's items through `run`, folds each block into the piece's own folds or its partial folds, and gives whether every fold fits.
    fn fold<T: Copy, R: Reduction<T, Total = S>>(
        &mut self, dims: [usize; 3], run: &(dyn Fn(Range<usize>, &mut dyn FnMut(usize, Part<'_, T, S>)) + Sync)) -> bool {
        let (unit, mut fits) = (dims[1] * dims[2], true);
        run(self.range.clone(), &mut |mut start, part| match part {
            Part::Items(mut items) => {
                while !items.is_empty() {
                    let (folds, at, end) = self.place(start, unit);
                    let (block, after) = items.split_at((end - start).min(items.len()));
                    fits &= fold_into::<T, R>(folds, dims, at, block);
                    (start, items) = (start + block.len(), after);
                }
            }
            Part::Total(n, total) => {
                let (folds, at, end) = self.place(start, unit);
                assert!(start + n <= end, "a block's total joins the folds of one part of a piece");
                fits &= fold_total::<T, R>(folds, dims, at, n, total);
            }
        });
        fits
    }

    /// The folds that the items from position `start` join: the partial folds of the unit that the piece shares with the piece before it, its own folds, or the partial folds of the unit that it shares with the piece after it. With them come the position of `start` among those folds' items, and the position where the items that they take end.
    fn place(&mut self, start: usize, unit: usize) -> (&mut [S], usize, usize) {
        let (a, b) = (self.lo * unit, self.hi * unit);
        match (start < a, start < b) {
            (true, _) => self.front.as_mut().map(|(u, f)| (&mut f[..], start - *u * unit, a)).expect("a piece that starts inside a unit has a front fold"),
            (_, true) => (&mut *self.own, start - a, b),
            _ => self.back.as_mut().map(|(u, f)| (&mut f[..], start - *u * unit, usize::MAX)).expect("a piece that ends inside a unit has a back fold"),
        }
    }
}

/// Joins `total`, the fold of the `n` items from position `start` of an array of shape `[outer, len, inner]` held row by row, into `totals`, the folds by `R` of its lanes along the middle axis so far, as [`fold_into`] joins those items. The items must all lie in one lane along the last axis. Gives whether the fold still fits.
fn fold_total<T, R: Reduction<T>>(totals: &mut [R::Total], [_, len, inner]: [usize; 3], start: usize, n: usize, total: R::Total) -> bool {
    assert!(inner == 1 && start % len + n <= len, "a block's total covers items of one lane along the last axis");
    let (i, at) = (start / len, start % len);
    if at == 0 { totals[i] = total; return true; }
    let (t, fits) = R::merge(totals[i], total);
    totals[i] = t;
    fits
}

/// Folds lanes `first..first + totals.len()` along the middle axis of `data`, an array of shape `[_, len, inner]` held row by row, into `totals`, and gives whether every fold fits. Each lane folds whole. A lane along the last axis folds as [`fold_rows`] folds it. Along another axis, a whole group of `inner` lanes folds its rows as [`step_rows`] folds them, and part of a group folds its columns as [`columns_by`] folds them.
pub(super) fn lanes<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], first: usize, data: &[T], [_, len, inner]: [usize; 3]) -> bool {
    if inner == 1 { return R::rows(&data[first * len..(first + totals.len()) * len], len, totals); }
    let (mut rest, mut k, mut fits) = (totals, first, true);
    let streamed = size_of_val(data) > STREAMED_BYTES;
    while !rest.is_empty() {
        let (i, c) = (k / inner, k % inner);
        let n = (inner - c).min(rest.len());
        let (folds, after) = std::mem::take(&mut rest).split_at_mut(n);
        let group = &data[i * len * inner + c..(i + 1) * len * inner];
        fits &= if folds.len() == inner { step_rows::<T, R>(folds, group, true, streamed) } else { columns_by::<T, R>(folds, group, inner, true, streamed) };
        (rest, k) = (after, k + folds.len());
    }
    fits
}

/// Steps each of `totals` by the matching item of `row`, and gives whether every fold fits. In a lane's first row, each fold is the fold of its item alone.
#[inline(always)]
fn step_row<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], row: &[T], first: bool) -> bool {
    if !first { return steps_of::<T, R>(totals, row); }
    let mut ok = true;
    for (t, &x) in totals.iter_mut().zip(row) { *t = one_of::<T, R>(x, &mut ok) }
    ok
}

/// Steps `totals` by every row of `rows`, which hold `totals.len()` items each, and gives whether every fold fits. Where `first` is true, the first row starts each fold, as in [`step_row`]. Narrow rows fold a block of whole rows at a time, into one running total for each item of the block. Each column's running totals then merge.
#[inline(always)]
fn step_rows<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], rows: &[T], first: bool, streamed: bool) -> bool {
    if !R::COMMUTATIVE { return rows_by::<T, R, 1>(totals, rows, first, streamed); }
    match crate::slice::running(size_of::<R::Total>()) {
        8 => rows_by::<T, R, 8>(totals, rows, first, streamed), 16 => rows_by::<T, R, 16>(totals, rows, first, streamed),
        _ => rows_by::<T, R, 32>(totals, rows, first, streamed)
    }
}

/// `step_rows` with up to `N` running totals. Rows wider than `N` fold as [`columns_by`] folds them. Too few rows to fill two blocks, and lanes of fewer than 4 rows, step one row at a time. Merging the columns of so few blocks cost more than the steps it saved, 13 instructions an item for the lanes of 2 rows that 2×2 pooling folds. A block is `N` items where `inner` divides `N`, and three quarters of `N` for rows of 3.
#[inline(always)]
fn rows_by<T: Copy, R: Reduction<T>, const N: usize>(totals: &mut [R::Total], rows: &[T], first: bool, streamed: bool) -> bool {
    let inner = totals.len();
    if inner > N { return columns_by::<T, R>(totals, rows, inner, first, streamed); }
    if rows.len() < 2 * N || rows.len() < 4 * inner {
        return rows.chunks_exact(inner).enumerate().fold(true, |fits, (j, row)| fits & step_row::<T, R>(totals, row, first && j == 0));
    }
    match inner {
        _ if N % inner == 0 => blocks_by::<T, R, N>(totals, rows, first, N), 3 => blocks_by::<T, R, N>(totals, rows, first, 3 * (N / 4)),
        _ => blocks_by::<T, R, N>(totals, rows, first, N - N % inner)
    }
}

/// The bytes of totals that [`columns_by`] steps in a block. They stay in the first-level cache while each row's items stream past. Blocks of 64 KB folded arrays of 64 to 256 very wide rows 20 to 25% faster than blocks of 16 KB on an M5 Max.
const COLUMN_BYTES: usize = 64 << 10;
/// The bytes of an array past which [`columns_by`] steps one row at a time: such an array streams from memory, where one stream of rows reads faster than four interleaved ones. On an M5 Max, one thread's `+⌿` of 1,000 f64 columns took 17% less time one row at a time at 64 MB and 6% less at 48 MB, and 5 to 41% more at 32 MB and below. On 4 threads, which share memory's bandwidth, both orders took the same time from 48 to 128 MB (31 October 2026).
const STREAMED_BYTES: usize = 40 << 20;

/// Steps `totals` by every row of `rows`, as [`step_rows`] does, where `totals` are the folds of the first `totals.len()` columns of rows of `inner` items. `rows` starts at the first of those columns, and its last row may stop after them, as where [`lanes`] folds part of a group's columns. `step_rows` folds rows wider than `slice::running`'s count of running totals this way. Each block of [`COLUMN_BYTES`] of `totals` steps by its columns of four rows at a time, so each row's items are read in the order they lie in memory. Where `streamed` is true, for an array past [`STREAMED_BYTES`], each block steps one row at a time. Strips of 32 columns, each folded down every row in registers, read their items a row apart, and on arrays larger than the cache took about 4 to 6 times as long as a fold along the last axis (27 October 2026).
fn columns_by<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], rows: &[T], inner: usize, first: bool, streamed: bool) -> bool {
    let (len, width) = (rows.len().div_ceil(inner), (COLUMN_BYTES / size_of::<R::Total>()).max(1));
    totals.chunks_mut(width).enumerate().fold(true, |mut fits, (b, block)| {
        let (at, w) = (b * width, block.len());
        let row = |j: usize| &rows[j * inner + at..j * inner + at + w];
        let mut j = usize::from(first && len > 0);
        if j == 1 { fits &= step_row::<T, R>(block, row(0), true); }
        while !streamed && j + 4 <= len {
            fits &= steps_of4::<T, R>(block, [row(j), row(j + 1), row(j + 2), row(j + 3)]);
            j += 4;
        }
        (j..len).fold(fits, |fits, j| fits & steps_of::<T, R>(block, row(j)))
    })
}

/// Steps each of `totals` by the matching item of each of `rows` in turn, as four calls of [`steps_of`] would, in one pass.
#[inline(always)]
fn steps_of4<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], rows: [&[T]; 4]) -> bool {
    let n = totals.len();
    let ([a, b, c, d], mut fits) = (rows.map(|r| &r[..n]), true);
    for k in 0..n {
        let (t, f) = R::checked(totals[k], a[k]);
        let (t, g) = R::checked(t, b[k]);
        let (t, h) = R::checked(t, c[k]);
        let (t, i) = R::checked(t, d[k]);
        (totals[k], fits) = (t, fits & f & g & h & i);
    }
    fits
}

/// `rows_by` in blocks of `width` items, a multiple of `inner` no greater than `N`. A width the compiler knows keeps the running totals in registers.
#[inline(always)]
fn blocks_by<T: Copy, R: Reduction<T>, const N: usize>(totals: &mut [R::Total], rows: &[T], first: bool, width: usize) -> bool {
    let inner = totals.len();
    let (blocks, rest) = rows.split_at(rows.len() - rows.len() % width);
    // `run` can stay in registers through the loop once `steps_of` inlines, unlike `all`.
    let (mut run, mut fits) = ([R::identity(); N], true);
    for block in blocks.chunks_exact(width) { fits &= steps_of::<T, R>(&mut run[..width], block) }
    let mut all = run;
    let parts = &mut all[..width];
    fits &= steps_of::<T, R>(parts, rest);
    for (c, total) in totals.iter_mut().enumerate() {
        let (column, ok) = parts[c..].iter().step_by(inner).fold((R::identity(), true), |(s, ok), &p| {
            let (t, f) = R::merge(s, p);
            (t, ok & f)
        });
        let (t, f) = if first { (column, ok) } else { let (t, f) = R::merge(*total, column); (t, f & ok) };
        (*total, fits) = (t, fits & f);
    }
    fits
}

/// Folds `block`, the items from position `start` on of an array of shape `[outer, len, inner]` held row by row, into `totals`, the folds by `R` of the array's lanes along its middle axis so far, which start at `R::unit()`. Gives whether every fold still fits. Along the last axis, a lane's items are consecutive. Whole lanes fold as `fold_rows` folds them, and the pieces of a lane that a block boundary splits fold by `slice` and join by `merge`. Along any other axis, runs of whole rows fold as [`step_rows`] folds them, and a row that a block boundary splits steps item by item.
pub fn fold_into<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], [outer, len, inner]: [usize; 3], start: usize, block: &[T]) -> bool {
    let (mut rest, mut fits) = (block, true);
    if rest.is_empty() { return true; }
    if inner == 1 {
        let (mut i, at) = (start / len, start % len);
        // The rest of a lane that an earlier block started.
        if at > 0 {
            let (piece, after) = rest.split_at((len - at).min(rest.len()));
            let (s, f) = R::slice(piece);
            let (t, g) = R::merge(totals[i], s);
            (totals[i], fits) = (t, f & g);
            if at + piece.len() < len { return fits; }
            (rest, i) = (after, i + 1);
        }
        let count = rest.len() / len;
        let (whole, started) = rest.split_at(count * len);
        fits &= R::rows(whole, len, &mut totals[i..i + count]);
        // The start of a lane that a later block finishes.
        if !started.is_empty() {
            let (s, f) = R::slice(started);
            (totals[i + count], fits) = (s, fits & f);
        }
    } else {
        let (row, streamed) = (start / inner, outer * len * inner * size_of::<T>() > STREAMED_BYTES);
        let (mut i, mut j, mut k) = (row / len, row % len, start % inner);
        while !rest.is_empty() {
            let rows = if k == 0 { (len - j).min(rest.len() / inner) } else { 0 };
            if rows > 0 {
                let (whole, after) = rest.split_at(rows * inner);
                fits &= step_rows::<T, R>(&mut totals[i * inner..(i + 1) * inner], whole, j == 0, streamed);
                (rest, j) = (after, j + rows);
            } else {
                let (piece, after) = rest.split_at((inner - k).min(rest.len()));
                fits &= step_row::<T, R>(&mut totals[i * inner + k..], piece, j == 0);
                (rest, k) = (after, k + piece.len());
                if k == inner { (j, k) = (j + 1, 0); }
            }
            if j == len { (i, j) = (i + 1, 0); }
        }
    }
    fits
}

/// Folds each row of `data`, a matrix with `columns` items in each row, by `R` into the matching item of `out`, and gives whether every fold fits. A row of fewer than 16 items folds by `checked`, with a loop of its own for each such length, such as an image's 3 channels. A loop of unknown length cost about 2 ns for each row. A longer row folds by `slice`, at the cost of a call and some setup for each row, which `running_results` keeps small for a short row.
#[inline]
pub(super) fn fold_rows<T: Copy, R: Reduction<T>>(data: &[T], columns: usize, out: &mut [R::Total]) -> bool {
    #[inline(always)]
    fn each<const K: usize, T: Copy, R: Reduction<T>>(data: &[T], out: &mut [R::Total]) -> bool {
        let mut fits = true;
        for (o, row) in out.iter_mut().zip(data.as_chunks::<K>().0) {
            let (t, f) = row[1..].iter().fold(R::one(row[0]), |(s, ok), &x| {
                let (t, f) = R::checked(s, x);
                (t, ok & f)
            });
            (*o, fits) = (t, fits & f);
        }
        fits
    }
    macro_rules! short {
        ($($k:literal)+) => {
            match columns {
                $($k => each::<$k, T, R>(data, out),)+
                _ => out.iter_mut().zip(data.chunks_exact(columns)).fold(true, |fits, (o, row)| { let (t, f) = R::slice(row); *o = t; fits & f }),
            }
        };
    }
    short!(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
}

/// [`Reduction::rows`] for a reduction whose step is one instruction: a row of 16 to fewer than [`SHORT_LANE`] items folds by `checked`, [`LANES`] rows side by side, and any other row as [`fold_rows`] folds it.
#[inline]
pub(super) fn rows_side_by_side<T: Copy, R: Reduction<T>>(data: &[T], columns: usize, out: &mut [R::Total]) -> bool {
    if !(16..SHORT_LANE).contains(&columns) { return fold_rows::<T, R>(data, columns, out); }
    let whole = out.len() / LANES * LANES;
    let (head, tail) = out.split_at_mut(whole);
    side_by_side::<LANES, _, _>(&data[..whole * columns], columns, R::one, R::checked, |_, _| {}, |row, t| head[row] = t)
        & side_by_side::<1, _, _>(&data[whole * columns..], columns, R::one, R::checked, |_, _| {}, |row, t| tail[row] = t)
}
