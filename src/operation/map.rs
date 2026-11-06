use super::*;

mod complex;
mod integer;
mod real;
use {complex::*, integer::*, real::*};
pub(super) fn map_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::domain::Variant, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, settings: Settings, repair: bool,
    judged: bool, output: &mut Buffer) -> Result<(), Error> {
    if inputs.len() != N { return Err(Error::new(Kind::Length, "kernel operand count changed")); }
    crate::events::taken(crate::events::Route::VectorKernel, range.len());
    report_maps(inputs.iter().map(|read| read.plan), range.clone());
    // A kernel whose arguments' types can't make an error judges no item: `judge` is `variant` unchecked. Reads and `IntegerOps` keep `variant`, so an argument that doesn't convert, or an overflow that `IntegerOps::result` reports, still fails.
    let judge = crate::domain::Variant { checked: variant.checked && judged, ..variant };
    if let Some(slices) = object.native_slices() {
        let valid = |ok: bool| if judge.checked && !ok { Err(Kind::Domain) } else { Ok(()) };
        let blocks = match variant.dtype {
            DType::Float(FloatWidth::F64) => sliced(object, slices.double, inputs, range.clone(), variant.checked, &valid, output),
            DType::Float(FloatWidth::F32) => sliced(object, slices.single, inputs, range.clone(), variant.checked, &valid, output),
            DType::Float(FloatWidth::F16) => slices.single.map(|single| map_blocks::<f32, half::f16, N>(inputs, range.clone(), variant.checked, output, &|args, n, out| {
                valid(crate::slice::buffered(n, out, |out| single(object, args, n, out), half::f16::from_f32))
            })),
            DType::Integer(Width::U8) => sliced(object, slices.byte, inputs, range.clone(), variant.checked, &valid, output),
            DType::Integer(Width::I16) => sliced(object, slices.short, inputs, range.clone(), variant.checked, &valid, output),
            DType::Integer(Width::I32) => sliced(object, slices.int, inputs, range.clone(), variant.checked, &valid, output),
            DType::Integer(Width::I64) => sliced(object, slices.long, inputs, range.clone(), variant.checked, &valid, output),
            _ => None,
        };
        if let Some(blocks) = blocks { return blocks; }
    }
    match variant.dtype {
        DType::Float(FloatWidth::F16) => map_real::<K, f32, half::f16, N>(object, variant, inputs, range, settings, repair, judge, output),
        DType::Float(FloatWidth::F32) => map_real::<K, f32, f32, N>(object, variant, inputs, range, settings, repair, judge, output),
        DType::Float(FloatWidth::F64) => map_real::<K, f64, f64, N>(object, variant, inputs, range, settings, repair, judge, output),
        DType::Boolean => match object.truth_table() {
            Some(table) => map_truth::<N>(table, inputs, range, output),
            None => map_integer::<K, u8, N>(object, variant, inputs, range, judge, output),
        },
        DType::Integer(width) => with_width!(width, I => map_integer::<K, I, N>(object, variant, inputs, range, judge, output)),
        DType::Complex(crate::buffer::ComplexWidth::F32) if K::REAL_COMPLEX && real_inputs(inputs) => map_real_complex::<K, f32, N>(object, variant, inputs, range, settings, output),
        DType::Complex(crate::buffer::ComplexWidth::F64) if K::REAL_COMPLEX && real_inputs(inputs) => map_real_complex::<K, f64, N>(object, variant, inputs, range, settings, output),
        DType::Complex(crate::buffer::ComplexWidth::F32) => map_complex::<K, f32, N>(object, variant, inputs, range, settings, judge, output),
        DType::Complex(crate::buffer::ComplexWidth::F64) => map_complex::<K, f64, N>(object, variant, inputs, range, settings, judge, output),
        DType::Character => Err(Error::new(Kind::Domain, "characters are not numerical kernel operands")),
    }
}

/// `slice`, a formula's slice form at `T` (`NativeSlices`), on each block of `inputs` over `range`, with `valid` turning each block's validity into its result. `None` where the formula has no form at `T`.
fn sliced<K: Sync, T: crate::buffer::Cast + Element, const N: usize>(
    object: &K, slice: Option<crate::facts::Slice<K, T, N>>, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool,
    valid: &(dyn Fn(bool) -> Result<(), Kind> + Sync), output: &mut Buffer) -> Option<Result<(), Error>> {
    let slice = slice?;
    Some(map_blocks::<T, T, N>(inputs, range, checked, output, &|args, n, out| valid(slice(object, args, n, out))))
}

/// A function of Booleans by its truth table (`Formula::truth_table`): bit `2x+y` of the table, or bit `x` for one argument, gives its result. A function with a table gives it for every Boolean argument, so no item checks its validity.
fn map_truth<const N: usize>(table: u8, inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, output: &mut Buffer) -> Result<(), Error> {
    map_values::<bool, bool, N>(inputs, range, false, false, output, |x| crate::facts::truth_bit(table, x), |_, _| None)
}

/// Whether no source of `inputs` holds complex items, so that a formula's complex results from real arguments (`Formula::real_complex`) can read them as reals.
fn real_inputs(inputs: &[crate::agreement::Read<'_>]) -> bool { inputs.iter().all(|read| read.sources.iter().all(|b| !matches!(b.view.dtype(), DType::Complex(_)))) }

/// `apply` on each item of `inputs` over `range`, written to `output` through `map_written`.
#[inline]
fn map_values<F: crate::buffer::Cast, O: Element, const N: usize>(
    inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool, judged: bool, output: &mut Buffer,
    apply: impl Fn([F; N]) -> O + Sync, kind: impl Fn([F; N], O) -> Option<Kind> + Sync) -> Result<(), Error> {
    map_written(inputs, range, checked, judged, output, |args, n, out| { crate::slice::mapped(args, n, out, #[inline(always)] |x| (apply(x), true)); }, kind)
}

/// Each block of `inputs` over `range` as `write` writes all `n` of its slots, to `output`. Where `judged`, `kind` judges each stored result against its arguments: `None` for a valid result, otherwise its error kind. No body runs twice, because a resolved object's body gives a result for every item.
#[inline]
fn map_written<F: crate::buffer::Cast, O: Element, const N: usize>(
    inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool, judged: bool, output: &mut Buffer,
    write: impl Fn(&[Arg<'_, F>; N], usize, &mut [std::mem::MaybeUninit<O>]) + Sync, kind: impl Fn([F; N], O) -> Option<Kind> + Sync) -> Result<(), Error> {
    map_blocks(inputs, range, checked, output, &|args, n, out| {
        write(&args, n, out);
        let results = crate::slice::written(&mut out[..n]);
        if judged && !crate::slice::judged(&args, results, |x, r| kind(x, r).is_none()) {
            for (i, &r) in results.iter().enumerate() { if let Some(e) = kind(std::array::from_fn(|j| args[j].nth(i)), r) { return Err(e); } }
        }
        Ok(())
    })
}

/// `apply` on `inputs` over `range`, written to `output`: by rows where `by_rows` reads them so, and otherwise a part at a time. Where an input reads through scratch, not in place (`Read::in_place`), the part runs `CHUNK` items at a time to keep that scratch in cache.
fn map_blocks<F: crate::buffer::Cast, O: Element, const N: usize>(
    inputs: &[crate::agreement::Read<'_>], range: std::ops::Range<usize>, checked: bool, output: &mut Buffer,
    apply: &(dyn Fn([Arg<'_, F>; N], usize, &mut [std::mem::MaybeUninit<O>]) -> Result<(), Kind> + Sync)) -> Result<(), Error> {
    if output.dtype() != O::DTYPE { *output = Buffer::with_capacity(O::DTYPE, range.len()); }
    let values = O::vec_mut(output).expect("kernel output has its selected dtype");
    values.clear();
    let recorded = |result: Result<(), Kind>| result.map_err(|kind| crate::error::record(Error::new(kind, "numerical kernel argument is outside its domain"))).is_ok();
    if let Some((len, rows)) = by_rows::<F>(inputs) {
        crate::parallel::extended(values, range.len(), crate::CHUNK, &|part, slots| slots.fill(part.len(), |out| recorded(fill_rows(&rows, len, part.start + range.start, out, apply))));
        return Ok(());
    }
    crate::parallel::extended(values, range.len(), crate::CHUNK, &|part, slots| {
        let part = part.start + range.start..part.end + range.start;
        let len = if inputs.iter().all(|input| input.in_place::<F>(&part)) { part.len().max(1) } else { crate::CHUNK };
        let mut scratch: [Vec<F>; N] = std::array::from_fn(|_| Vec::new());
        part.clone().step_by(len).fold(true, |ok, at| {
            let run = at..(at + len).min(part.end);
            let n = run.len();
            let mut args = [Arg::Many(&[]); N];
            for ((arg, input), scratch) in args.iter_mut().zip(inputs).zip(scratch.iter_mut()) {
                match input.part::<F>(run.clone(), checked, scratch) {
                    Ok(read) => *arg = read,
                    Err(error) => { crate::error::record(error); return slots.fill(n, |out| { for slot in out { slot.write(O::FILL); } false }) & ok; }
                }
            }
            slots.fill(n, |out| recorded(apply(args, n, out))) & ok
        })
    });
    Ok(())
}

/// Rows shorter than this can read slower than blocks, which share each call's cost among more items.
const ROWS: usize = 16;

/// The row length `len` and each input as rows of `len` items (`Read::rows`), for a map that reads faster by rows than by blocks: at least two rows of at least `ROWS` items, and an input that blocks would copy, because each of its rows reads one item that changes from row to row, or its rows don't follow on from each other, as an outer product's operands do. `len` is the `k` of the first mapped input's repeat or cycle of `k` items, and else the last axis that input steps along. `None` otherwise.
fn by_rows<'a, F: crate::buffer::Cast>(inputs: &[crate::agreement::Read<'a>]) -> Option<(usize, Vec<crate::agreement::Rows<'a, F>>)> {
    use crate::agreement::{Mapping, ReadRoute};
    let len = inputs.iter().find_map(|read| match read.plan.route() {
        ReadRoute::Map { map: Mapping::Linear(k) | Mapping::Tiled(k), .. } => Some(*k),
        ReadRoute::Map { .. } => read.plan.shape_hint().last().copied(),
        _ => None,
    })?;
    let items = inputs.first()?.plan.len();
    if len < ROWS || items <= len || items % len != 0 { return None; }
    let rows = inputs.iter().map(|read| read.rows([items / len, len])).collect::<Option<Vec<_>>>()?;
    rows.iter().any(|row| if row.run { row.step != len as isize } else { row.step != 0 }).then_some((len, rows))
}

/// Writes `out`, the result's items from position `start`, by `apply` on each row of `len` items from `rows` in turn (`by_rows`). Every row is written, and the first error is the result.
fn fill_rows<F: Copy, O, const N: usize>(
    rows: &[crate::agreement::Rows<'_, F>], len: usize, start: usize, out: &mut [std::mem::MaybeUninit<O>],
    apply: &(dyn Fn([Arg<'_, F>; N], usize, &mut [std::mem::MaybeUninit<O>]) -> Result<(), Kind> + Sync)) -> Result<(), Kind> {
    let (mut done, mut r, mut at, mut result) = (0, start / len, start % len, Ok(()));
    while done < out.len() {
        let n = (len - at).min(out.len() - done);
        let args = std::array::from_fn(|j| {
            let (row, k) = (&rows[j], rows[j].first.wrapping_add_signed(rows[j].step * r as isize));
            if row.run { Arg::Many(&row.items[k + at..k + at + n]) } else { Arg::One(row.items[k]) }
        });
        result = result.and(apply(args, n, &mut out[done..done + n]));
        (done, r, at) = (done + n, r + 1, 0);
    }
    result
}
