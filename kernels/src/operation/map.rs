use super::*;
use crate::{FUSED, recipe::NativeOps};

mod complex;
mod integer;
mod real;
use {complex::*, integer::*, real::map_real};
mod io;
use crate::buffer::{Output, OutputItem};
pub(super) use io::Inputs;
pub(super) fn map_formula<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, variant: crate::domain::Variant, inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, settings: Settings, repair: bool, judged: bool,
    output: &mut Output<'_, '_>) -> Result<(), Error> {
    if inputs.len() != N { return Err(Error::new(Kind::Length, "kernel operand count changed")); }
    let ops = settings.ops();
    crate::events::taken(crate::events::Route::VectorKernel, range.len());
    report_maps(inputs.0.iter().map(|read| read.plan), range.clone());
    // A kernel whose arguments' types can't make an error judges no item: `judge` is `variant` unchecked. Reads and `IntegerOps` keep `variant`, so an argument that doesn't convert, or an overflow that `IntegerOps::result` reports, still fails.
    let judge = crate::domain::Variant { checked: variant.checked && judged, ..variant };
    let slices = object.native_slices();
    if let Some(slices) = &slices {
        let valid = |ok: bool| if judge.checked && !ok { Err(Kind::Domain) } else { Ok(()) };
        let blocks = match variant.dtype {
            DType::Float(FloatWidth::F64) => sliced(object, &ops, slices.double(), inputs, range.clone(), variant.checked, &valid, output),
            DType::Float(FloatWidth::F32) => sliced(object, &ops, slices.single(), inputs, range.clone(), variant.checked, &valid, output),
            DType::Float(FloatWidth::F16) => slices.single().map(|single| {
                map_blocks::<f32, half::f16, N>(inputs, range.clone(), variant.checked, output, &|args, n, out| {
                    valid(unsafe { crate::slice::buffered(n, out, |out| single(object, &ops, args, n, out), half::f16::from_f32) })
                })
            }),
            DType::Integer(Width::U8) => sliced(object, &ops, slices.byte(), inputs, range.clone(), variant.checked, &valid, output),
            DType::Integer(Width::I16) => sliced(object, &ops, slices.short(), inputs, range.clone(), variant.checked, &valid, output),
            DType::Integer(Width::I32) => sliced(object, &ops, slices.int(), inputs, range.clone(), variant.checked, &valid, output),
            DType::Integer(Width::I64) => sliced(object, &ops, slices.long(), inputs, range.clone(), variant.checked, &valid, output),
            _ => None,
        };
        if let Some(blocks) = blocks { return blocks; }
    }
    match variant.dtype {
        DType::Float(FloatWidth::F16) => map_real::<K, f32, half::f16, N>(object, variant, inputs, range, &ops, repair, judge, output),
        DType::Float(FloatWidth::F32) => map_real::<K, f32, f32, N>(object, variant, inputs, range, &ops, repair, judge, output),
        DType::Float(FloatWidth::F64) => map_real::<K, f64, f64, N>(object, variant, inputs, range, &ops, repair, judge, output),
        DType::Boolean => match object.truth_table() {
            Some(table) => map_truth::<N>(table, inputs, range, output), None => map_integer::<K, u8, N>(object, variant, inputs, range, judge, output)
        },
        DType::Integer(width) => with_width!(width, I => map_integer::<K, I, N>(object, variant, inputs, range, judge, output)),
        DType::Complex(crate::buffer::ComplexWidth::F32) if K::REAL_COMPLEX && inputs.real() => {
            map_real_complex::<K, f32, N>(object, variant, inputs, range, &ops, slices.and_then(|s| s.real_complex_single()), output)
        }
        DType::Complex(crate::buffer::ComplexWidth::F64) if K::REAL_COMPLEX && inputs.real() => {
            map_real_complex::<K, f64, N>(object, variant, inputs, range, &ops, slices.and_then(|s| s.real_complex_double()), output)
        }
        DType::Complex(crate::buffer::ComplexWidth::F32) => {
            map_complex::<K, f32, N>(object, variant, inputs, range, &ops, slices.and_then(|s| s.complex_single()), judge, output)
        }
        DType::Complex(crate::buffer::ComplexWidth::F64) => {
            map_complex::<K, f64, N>(object, variant, inputs, range, &ops, slices.and_then(|s| s.complex_double()), judge, output)
        }
        DType::Character => Err(Error::new(Kind::Domain, "characters are not numerical kernel operands")),
    }
}

/// `slice`, a formula's slice form at `T` (`NativeSlices`), on each block of `inputs` over `range`, with `valid` turning each block's validity into its result. `None` where the formula has no form at `T`.
fn sliced<K: Sync, T: crate::buffer::Cast + OutputItem, const N: usize>(
    object: &K, ops: &NativeOps<{ FUSED }>, slice: Option<crate::facts::Slice<K, T, N>>, inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, checked: bool,
    valid: &(dyn Fn(bool) -> Result<(), Kind> + Sync), output: &mut Output<'_, '_>) -> Option<Result<(), Error>> {
    let slice = slice?;
    Some(map_blocks::<T, T, N>(inputs, range, checked, output, &|args, n, out| valid(slice(object, ops, args, n, out))))
}

/// A function of Booleans by its truth table (`Formula::truth_table`): bit `2x+y` of the table, or bit `x` for one argument, gives its result. A function with a table gives it for every Boolean argument, so no item checks its validity.
fn map_truth<const N: usize>(table: u8, inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, output: &mut Output<'_, '_>) -> Result<(), Error> {
    let value = |x: [bool; N]| match N {
        1 => crate::bitwise(table, false, x[0]), 2 => crate::bitwise(table, x[0], x[1]), _ => crate::facts::truth_bit(table, x)
    };
    map_values::<bool, bool, N>(inputs, range, false, false, output, value, |_, _| None)
}

/// `apply` on each item of `inputs` over `range`, written to `output` through `map_written`.
#[inline]
fn map_values<F: crate::buffer::Cast, O: OutputItem, const N: usize>(
    inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, checked: bool, judged: bool, output: &mut Output<'_, '_>, apply: impl Fn([F; N]) -> O + Sync,
    kind: impl Fn([F; N], O) -> Option<Kind> + Sync) -> Result<(), Error> {
    map_written(inputs, range, checked, judged, output, |args, n, out| {
        crate::slice::mapped(args, n, out, #[inline(always)] |x| (apply(x), true));
    }, kind)
}

/// Each block of `inputs` over `range` as `write` writes all `n` of its slots, to `output`. Where `judged`, `kind` judges each stored result against its arguments: `None` for a valid result, otherwise its error kind. No body runs twice, because a resolved object's body gives a result for every item.
#[inline]
fn map_written<F: crate::buffer::Cast, O: OutputItem, const N: usize>(
    inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, checked: bool, judged: bool, output: &mut Output<'_, '_>,
    write: impl Fn(&[Arg<'_, F>; N], usize, &mut [std::mem::MaybeUninit<O>]) + Sync, kind: impl Fn([F; N], O) -> Option<Kind> + Sync) -> Result<(), Error> {
    map_blocks(inputs, range, checked, output, &|args, n, out| {
        write(&args, n, out);
        let results = unsafe { crate::slice::written(&mut out[..n]) };
        if judged && !crate::slice::judged(&args, results, |x, r| kind(x, r).is_none()) {
            for (i, &r) in results.iter().enumerate() { if let Some(e) = kind(std::array::from_fn(|j| args[j].nth(i)), r) { return Err(e); } }
        }
        Ok(())
    })
}

/// `apply` on `inputs` over `range`, written to `output`: by rows where `by_rows` reads them so, and otherwise a part at a time. Where an input reads through scratch, not in place (`Read::in_place`), the part runs `CHUNK` items at a time to keep that scratch in cache.
fn map_blocks<F: crate::buffer::Cast, O: OutputItem, const N: usize>(
    inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, checked: bool, output: &mut Output<'_, '_>,
    apply: &(dyn Fn([Arg<'_, F>; N], usize, &mut [std::mem::MaybeUninit<O>]) -> Result<(), Kind> + Sync)) -> Result<(), Error> {
    let recorded =
        |result: Result<(), Kind>| result.map_err(|kind| crate::error::record(Error::new(kind, "numerical kernel argument is outside its domain"))).is_ok();
    match inputs.borrowed::<F, N>(range.clone(), checked) {
        Ok(Some(args)) => return output.write(range.len(), &|part, slots, _| unsafe {
            slots.fill(part.len(), |out| recorded(apply(args.map(|arg| arg.part(part.start, part.len())), part.len(), out)))
        }),
        Ok(None) => {}, Err(error) => { output.target.fill(range.len()); return Err(error); }
    }
    if let Some((len, rows)) = by_rows::<F, N>(inputs.0) {
        return output
            .write(range.len(), &|part, slots, _| unsafe { slots.fill(part.len(), |out| recorded(fill_rows(&rows, len, part.start + range.start, out, apply))) });
    }
    output.write(range.len(), &|part, slots, work| {
        let part = part.start + range.start..part.end + range.start;
        let in_place = inputs.in_place::<F>(&part);
        let len = if in_place { part.len().max(1) } else { crate::CHUNK };
        let mut local: [Vec<F>; N] = std::array::from_fn(|_| Vec::new());
        let (mut work, mut copied) = (work, 0);
        let mut scratch = match work.as_deref_mut() { Some(work) if !in_place => work.inputs::<F, N>(), _ => local.each_mut() };
        let ok = part.clone().step_by(len).fold(true, |ok, at| {
            let run = at..(at + len).min(part.end);
            let n = run.len();
            let args = match inputs.part::<F, N>(run, checked, &mut scratch) {
                Ok((args, n)) => { copied += n; args }
                Err(error) => {
                    crate::error::record(error);
                    return unsafe { slots.fill(n, |out| { for slot in out { slot.write(O::FILL); } false }) } & ok;
                }
            };
            (unsafe { slots.fill(n, |out| recorded(apply(args, n, out))) }) & ok
        });
        if let Some(work) = work { work.record_copies(copied); }
        ok
    })
}

/// Rows shorter than this can read slower than blocks, which share each call's cost among more items.
const ROWS: usize = 16;

/// The row length `len` and each input as rows of `len` items (`Read::rows`), for a map that reads faster by rows than by blocks: at least two rows of at least `ROWS` items, and an input that blocks would copy, because each of its rows reads one item that changes from row to row, or its rows don't follow on from each other, as an outer product's operands do. `len` is the `k` of the first mapped input's repeat or cycle of `k` items, and else the last axis that input steps along. `None` otherwise.
fn by_rows<'a, F: crate::buffer::Cast, const N: usize>(inputs: &[crate::agreement::Read<'a>]) -> Option<(usize, [crate::agreement::Rows<'a, F>; N])> {
    use crate::{agreement::{Mapping, ReadRoute}, gather::Steps};
    let len = inputs.iter().find_map(|read| match read.plan.route() {
        ReadRoute::Map { map: Mapping::Linear(k) | Mapping::Tiled(k), .. } => Some(*k),
        ReadRoute::Map { map: Mapping::Steps { steps, .. }, .. } if Steps::count(steps) == read.plan.len() => steps.last().map(Steps::len),
        ReadRoute::Map { .. } => Some(read.plan.len()),
        _ => None
    })?;
    let items = inputs.first()?.plan.len();
    if len < ROWS || items <= len || items % len != 0 { return None; }
    let mut rows = std::array::from_fn(|_| None);
    for k in 0..N { rows[k] = Some(inputs.get(k)?.rows([items / len, len])?); }
    let rows: [crate::agreement::Rows<'a, F>; N] = rows.map(Option::unwrap);
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

pub(super) fn copied(inputs: Inputs<'_, '_>, range: std::ops::Range<usize>, checked: bool, output: &mut Output<'_, '_>) -> Result<(), Error> {
    with_dtype!(output.result_dtype, T => map_values::<T, T, 1>(inputs, range, checked, false, output, |x| x[0], |_, _| None))
}

pub(super) fn constant(value: bool, n: usize, output: &mut Output<'_, '_>) -> Result<(), Error> {
    output.write::<bool>(n, &|part, slots, _| slots.pairs(Arg::One(value), Arg::One(()), part.len(), |x, ()| (x, true)))
}
