use super::*;

/// A walk that a binary formula runs item by item (`Application::binary`): a fold of `pieces`, or a scan or a walk of adjacent items of `input`, each along the middle axis of `dims`, or a modified write at `ids`.
pub(super) enum Route<'a> {
    Fold { dims: [usize; 3], seed: Seed<'a>, direction: Direction, pieces: &'a Pieces<'a> },
    Scan { dims: [usize; 3], seed: Seed<'a>, direction: Direction, input: crate::agreement::Read<'a> },
    Adjacent { dims: [usize; 3], seed: Option<crate::agreement::Read<'a>>, flip: bool, input: crate::agreement::Read<'a> },
    At { root: WriteRoot<'a>, values: crate::agreement::Read<'a>, ids: Positions<'a>, checked: bool, projection: Projection },
}

/// `route` by `object`'s per-item kernels at `variant`'s computing type: `paired` for each row or block, `folded` for each lane of a fold, `chained` for each lane of a scan, and `Writes::combine` for each chunk of writes. Everything around them is generic over the item type only.
pub(super) fn binary_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::domain::Variant, settings: Settings, route: Route<'_>, output: &mut Buffer) -> Result<(), Error> {
    if const { N != 2 || K::MAP_ONLY } { return Err(Error::new(Kind::Length, "an ordered walk needs a binary function")); }
    with_binary_body!(object, variant, settings, |T, apply| {
        // `step(total, item)` for every route, swapped for a right walk, so that the routes share one `paired`.
        let step = |right: bool| {
            #[inline(always)]
            move |x, y| if right { apply(y, x) } else { apply(x, y) }
        };
        match route {
            Route::Fold { dims, seed, direction, pieces } => {
                let step = step(direction == Direction::Right);
                ordered_fold::<T>(dims, seed, direction, variant.checked, object.identity()[1], pieces, &folded(step), &paired(step), output)
            }
            Route::Scan { dims, seed, direction, input } => {
                let step = step(direction == Direction::Right);
                let (lane, block) = (chained(step), paired(step));
                crate::reduce::scan_read::<T, T>(dims, seed, direction, variant.checked, input, output, &|items, dims, seed, out| {
                    crate::reduce::ordered_scan(items, dims, seed, out, &lane, &block)
                        .map_or(Ok(()), |kind| Err(Error::new(kind, "ordered scan is outside the function's domain")))
                })
            }
            Route::Adjacent { dims, seed, flip, input } => {
                crate::reduce::adjacent_read::<T>(dims, seed, variant.checked, flip, input, output, &paired(step(false)))
            }
            Route::At { root, values, ids, checked, projection } => {
                root.initialize(T::DTYPE, checked, projection, output)?;
                at_typed::<T>(values, ids, checked, Some(&|out, writes, items| writes.combine(out, items, &apply)), output)
            }
        }
    })
}
