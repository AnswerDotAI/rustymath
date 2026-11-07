use super::*;

fn piece_values<T: crate::buffer::Cast>(range: std::ops::Range<usize>, checked: bool, pieces: &Pieces<'_>, out: &mut Vec<T>) -> Result<(), Error> {
    out.clear();
    let (mut next, mut failure) = (range.start, None);
    pieces(range.clone(), &mut |at, piece| {
        if failure.is_some() { return; }
        let Piece::Items(view) = piece else { failure = Some(Error::new(Kind::Length, "ordered fold received a precomputed total")); return; };
        if at != next || at + view.len() > range.end { failure = Some(Error::new(Kind::Length, "fold pieces do not cover the requested range")); return; }
        if let Err(error) = T::extend_cast(view, view.len(), checked, out) { failure = Some(error); return; }
        next += view.len();
    })?;
    if let Some(error) = failure { return Err(error); }
    if next != range.end { return Err(Error::new(Kind::Length, "fold pieces left an unread range")); }
    Ok(())
}

/// The ordered folds of each lane of `[outer, len, inner]`, read from `pieces` a chunk of rows at a time. Each chunk folds from the right onto the fold so far: by `lane`, one call for each lane, along the last axis, and by `block`, one call for each row, along any other axis.
pub(super) fn ordered_fold<T: crate::buffer::Cast>(
    [outer, len, inner]: [usize; 3], seed: Seed<'_>, direction: Direction, checked: bool, identity: Option<f64>, pieces: &Pieces<'_>,
    lane: &dyn Fn(&[T], T) -> (T, Option<Kind>), block: &dyn Fn(&[T], &[T], &mut [std::mem::MaybeUninit<T>]) -> Result<(), Kind>, output: &mut Buffer
) -> Result<(), Error> {
    let (mut result, mut items, mut seed_scratch) = (Vec::with_capacity(outer * inner), Vec::new(), Vec::new());
    if inner == 0 { *output = T::buffer(result); return Ok(()); }
    let right = direction == Direction::Right;
    let fold = |items: &[T], carry: Option<&[T]>| {
        let (rows, tail) = match carry { Some(tail) => (items, tail), None => items.split_at(items.len() - inner) };
        let (mut total, mut next, mut failure) = (tail.to_vec(), Vec::with_capacity(inner), None);
        if inner == 1 { (total[0], failure) = lane(rows, tail[0]); } else {
            for row in rows.chunks_exact(inner).rev() {
                next.clear();
                failure = failure.or(unsafe { crate::slice::appended(&mut next, inner, |slots| block(&total, row, slots)) }.err());
                std::mem::swap(&mut total, &mut next);
            }
        }
        failure.map_or(Ok(total), |kind| Err(Error::new(kind, "ordered fold is outside the function's domain")))
    };
    for cell in 0..outer {
        let mut given = seed.read().map(|read| crate::reduce::seed_row(&read, cell, inner, checked, &mut seed_scratch)).transpose()?;
        let upfront = len == 0 || match seed { Seed::Tail(_) => right, Seed::Initial(_) => !right, Seed::Identity => false };
        let mut carry = if upfront { given.take() } else { None };
        let rows = (crate::CHUNK / inner).max(1);
        let mut remaining = len;
        while remaining != 0 {
            let n = rows.min(remaining);
            let start = if right { remaining - n } else { len - remaining };
            let at = (cell * len + start) * inner;
            piece_values(at..at + n * inner, checked, pieces, &mut items)?;
            if !right { crate::reduce::reverse_rows(&mut items, [1, n, inner]); }
            carry = Some(if carry.is_none() && n == 1 { items.clone() } else { fold(&items, carry.as_deref())? });
            remaining -= n;
        }
        let mut total = match carry {
            Some(total) => total,
            None => {
                let value = identity.ok_or_else(|| Error::new(Kind::Length, "an empty fold needs an identity or a seed"))?;
                vec![T::cast(View::Floats(crate::buffer::Floats::F64(&[value])), 0)?; inner]
            }
        };
        if let Some(seed) = given { total = fold(&seed, Some(&total))?; }
        result.extend(total);
    }
    *output = T::buffer(result);
    Ok(())
}
