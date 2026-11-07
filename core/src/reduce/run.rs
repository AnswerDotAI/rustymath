use super::*;

/// A reduction by name: the tag by which a function object's row names the reduction it folds by (`facts::Facts::fold`, `facts::Facts::truth`), and by which a frontend's block stages and compiled code fold. `with_reduce!` gives its type.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reduce { Sum, Product, Max, Min, And, Or, Xor, Xnor }

impl Reduce {
    /// Whether a block stage folds items of `item` by this reduction a block at a time: floats by a sum, a product, a maximum or a minimum; integers by a maximum or a minimum, and by a sum where its sums are exact (`exact`), as a sum whose range lies within the items' type is; Booleans by a sum (a count), a minimum, a maximum, an exclusive or or its negation.
    pub fn stage(self, item: crate::buffer::DType, exact: bool) -> bool {
        use crate::buffer::{DType, FloatWidth};
        match item {
            DType::Float(FloatWidth::F32 | FloatWidth::F64) => matches!(self, Self::Sum | Self::Product | Self::Max | Self::Min),
            DType::Integer(_) => matches!(self, Self::Max | Self::Min) || exact && self == Self::Sum,
            DType::Boolean => matches!(self, Self::Sum | Self::Min | Self::Max | Self::Xor | Self::Xnor), _ => false
        }
    }

    /// Whether a stage's fold by this reduction of items of `items` takes a seed of type `seed` into a result of type `result`, merging the seed into each total at the end: floats with a float seed, where the result keeps the items' width; integers with an integer or Boolean seed by a maximum or a minimum, or by a sum into a result of the items' type; and a count of Booleans with an integer or Boolean seed. A sum of integers into a wider integer folds after the stage, because compiled code sums integers at the items' width.
    pub fn seeds(self, items: crate::buffer::DType, seed: crate::buffer::DType, result: crate::buffer::DType) -> bool {
        use crate::buffer::DType;
        match (items, seed, result) {
            (DType::Float(a), DType::Float(_), DType::Float(b)) => a == b,
            (DType::Integer(_), DType::Boolean | DType::Integer(_), DType::Integer(_)) => match self {
                Self::Max | Self::Min => true, Self::Sum => items == result, _ => false
            },
            (DType::Boolean, DType::Boolean | DType::Integer(_), DType::Integer(_)) => self == Self::Sum,
            _ => false,
        }
    }

    /// Whether compiled code folds by this reduction: a sum, a product, a maximum or a minimum.
    pub fn compiles(self) -> bool { matches!(self, Self::Sum | Self::Product | Self::Max | Self::Min) }
}
impl Reduce {
    pub fn total_dtype(self, variant: crate::domain::Variant) -> Option<crate::buffer::DType> {
        native_reduction!(self, variant.dtype, variant.fails_on_overflow(), |R, T| Some(<<R as Reduction<T>>::Total as Element>::DTYPE))
    }
    pub fn run_pieces(self, variant: crate::domain::Variant, dims: [usize; 3], seed: Seed<'_>, pieces: &Pieces<'_>, output: &mut Buffer) -> Result<(), Error> {
        // No count of trues exceeds its lane's length: an unseeded count asked for in `i16` or `i32` totals in that type where the length fits.
        use crate::buffer::{DType, Width};
        if self == Reduce::Sum && variant.dtype == DType::Boolean && matches!(seed, Seed::Identity)
            && let DType::Integer(w @ (Width::I16 | Width::I32)) = output.dtype() && crate::domain::Range::of_dtype(output.dtype()).hi >= dims[1] as f64 {
                return if w == Width::I16 { counted::<i16>(dims, variant.checked, pieces, output) }
                else { counted::<i32>(dims, variant.checked, pieces, output) };
            }
        native_reduction!(self, variant.dtype, variant.fails_on_overflow(), |R, T| Some(
            native_pieces::<T, R>(dims, variant.checked, pieces).and_then(|result| finish_fold::<T, R>(result, seed, variant.checked, output))
        )).unwrap_or_else(|| Err(Error::new(Kind::Domain, "reduction does not support this dtype")))
    }
    pub fn run_read(
        self, variant: crate::domain::Variant, shape: &[usize], axis: Option<usize>, seed: Seed<'_>, input: Read<'_>, output: &mut Buffer
    ) -> Result<Option<Vec<Steps>>, Error> {
        native_reduction!(self, variant.dtype, variant.fails_on_overflow(), |R, T| Some(native_read::<T, R>(shape, axis, variant.checked, input).and_then(
            |(buffer, fits, order)| {
                // A seeded fold walks its totals into the result's order, which its seed reads in.
                let (buffer, order) = if seed.read().is_some() { (buffer.ordered(order), None) } else { (buffer, order) };
                finish_fold::<T, R>((buffer, fits), seed, variant.checked, output).map(|()| order)
            }
        ))).unwrap_or_else(|| Err(Error::new(Kind::Domain, "reduction does not support this dtype")))
    }
    /// The folds by this reduction of each window along axes 1 and 3 of `input`, an array of shape `dims`, as [`sliding_2d`] folds them, with each axis's window width and the positions between windows' starts. A domain error for windows wider than `sliding_2d` folds directly.
    pub fn run_sliding_2d(
        self, variant: crate::domain::Variant, dims: [usize; 5], widths: [usize; 2], every: [usize; 2], input: View<'_>, output: &mut Buffer
    ) -> Result<(), Error> {
        native_reduction!(self, variant.dtype, variant.fails_on_overflow(), |R, T| Some(items::<T>(input).and_then(|data| {
            let folds = sliding_2d::<T, R>(data, dims, widths, every).ok_or_else(|| Error::new(Kind::Domain, "windows too wide to fold directly"))?;
            totals::<T, R>(folds, variant.checked, output)
        }))).unwrap_or_else(|| Err(Error::new(Kind::Domain, "reduction does not support this dtype")))
    }
    /// The fold by this reduction of each group of `input` that `groups` gives, in group order (`Grouping::fold`).
    pub fn run_groups(self, variant: crate::domain::Variant, groups: &impl crate::search::Grouping, input: View<'_>, output: &mut Buffer) -> Result<(), Error> {
        native_reduction!(self, variant.dtype, variant.fails_on_overflow(), |R, T| Some(items::<T>(input).and_then(|data| totals::<T, R>(
            groups.fold::<T, R>(data), variant.checked, output
        )))).unwrap_or_else(|| Err(Error::new(Kind::Domain, "reduction does not support this dtype")))
    }
    #[inline(never)]
    pub fn inverse_pieces(self, variant: crate::domain::Variant, dims: [usize; 3], pieces: &Pieces<'_>) -> Result<Option<Buffer>, Error> {
        inverse_result(
            native_reduction!(@inverse self, variant.dtype, |R, T| Some(native_pieces::<T, R>(dims, variant.checked, pieces))), dims.into_iter().product())
    }
    #[inline(never)]
    pub fn inverse_read(self, variant: crate::domain::Variant, shape: &[usize], axis: usize, input: Read<'_>) -> Result<Option<Buffer>, Error> {
        inverse_result(
            native_reduction!(@inverse self, variant.dtype, |R, T| Some(native_read::<T, R>(shape, Some(axis), variant.checked, input).map(|(buffer, fits, order)| (buffer.ordered(order), fits)))),
            crate::layout::count(shape))
    }
}

fn inverse_result(result: Option<Result<(Buffer, bool), Error>>, items: usize) -> Result<Option<Buffer>, Error> {
    let result = result.transpose()?.and_then(|(buffer, fits)| fits.then_some(buffer));
    if result.is_some() { crate::events::taken(crate::events::Route::AlternatedFold, items); }
    Ok(result)
}

fn native_read<T, R>(shape: &[usize], axis: Option<usize>, checked: bool, input: Read<'_>) -> Result<(Buffer, bool, Option<Vec<Steps>>), Error> where T: Cast,
    R: Reduction<T>, R::Total: Totals {
    let dims = match axis { Some(axis) => Lanes::new(shape, axis)?.dims(), None => [1, crate::layout::count(shape), 1] };
    if let Some((view, mapping)) = input.positional(shape) && let Some(data) = T::of(view) {
        if axis.is_none() && R::COMMUTATIVE && let Some(block) = mapping.block(shape) {
            return crate::error::checked(|| {
                let (dims, order) = block.fold(None);
                let (totals, fits) = along::<T, R>(&data[block.range], dims);
                Ok((R::Total::buffer_of(totals), fits, order))
            });
        }
        if let Some(axis) = axis {
            return crate::error::checked(|| {
                let slide = if R::COMMUTATIVE {
                    let source = crate::structure::walked_source(&mapping.tables(shape), shape, data.len());
                    match fold_route(&mapping, shape, source.as_deref(), false, &[axis]) {
                        FoldRoute::Slide(s) => {
                            let data = &data[s.region.clone()];
                            let folded = match s.dims[..] {
                                [outer, len, inner] if s.every[0] == 1 => Some(sliding::<T, R>(data, [outer, len, inner], s.widths[0])),
                                [outer, len, inner] => {
                                    crate::events::taken(crate::events::Route::SlidingFold, data.len());
                                    Some(along::<T, R>(data, [outer, len, inner]))
                                }
                                [a, b, c, d, e] => sliding_2d::<T, R>(data, [a, b, c, d, e], [s.widths[0], s.widths[1]], [s.every[0], s.every[1]]),
                                _ => unreachable!("a slide folds one axis or two"),
                            };
                            folded.map(|(totals, fits)| (totals, fits, Some(s.tables)))
                        }
                        _ => None,
                    }
                } else { None };
                let (totals, fits, order) = slide.unwrap_or_else(|| along_mapped::<T, R>(data, &mapping, shape, axis));
                let order = order.filter(|order| !Mapping::walk_in_order(0, order, totals.len()));
                Ok((R::Total::buffer_of(totals), fits, order))
            });
        }
    }
    let pieces = |range: Range<usize>, emit: &mut dyn FnMut(usize, Piece<'_>)| {
        let mut scratch = Buffer::with_capacity(T::DTYPE, 0);
        let start = range.start;
        let view = input.part_view(range, T::DTYPE, checked, &mut scratch)?;
        emit(start, Piece::Items(view));
        Ok(())
    };
    native_pieces::<T, R>(dims, checked, &pieces).map(|(buffer, fits)| (buffer, fits, None))
}

/// The counts of trues along each lane of `pieces` in `C`, which holds every count (`Reduce::run_pieces`).
fn counted<C: Tally + Cast>(dims: [usize; 3], checked: bool, pieces: &Pieces<'_>, output: &mut Buffer) -> Result<(), Error> {
    native_pieces::<bool, SumAs<C>>(dims, checked, pieces).and_then(|result| finish_fold::<bool, SumAs<C>>(result, Seed::Identity, checked, output))
}

fn native_pieces<T, R>(dims: [usize; 3], checked: bool, pieces: &Pieces<'_>) -> Result<(Buffer, bool), Error> where T: Cast, R: Reduction<T>, R::Total: Totals {
    let [outer, len, inner] = dims;
    let mut totals = vec![if len == 0 { R::unit() } else { R::identity() }; outer * inner];
    crate::error::checked(|| {
        let fits = fold_pieces::<T, R>(&mut totals, dims, &|range, emit| {
            let result = pieces(range, &mut |start, piece| match piece {
                Piece::Items(view) => match T::of(view) {
                    Some(items) => emit(start, Part::Items(items)),
                    None => match Buffer::cast_results(view, T::DTYPE, checked) {
                        Ok(values) => emit(start, Part::Items(T::of(values.view()).expect("converted fold piece has its requested dtype"))),
                        Err(error) => crate::error::record(error),
                    },
                },
                Piece::Total(total) => match R::Total::from_partial(total) {
                    Ok(value) => emit(start, Part::Total(total.items, value)), Err(error) => crate::error::record(error)
                },
            });
            if let Err(error) = result { crate::error::record(error); }
        });
        Ok((R::Total::buffer_of(totals), fits))
    })
}

fn finish_fold<T, R: Reduction<T>>((mut buffer, fits): (Buffer, bool), seed: Seed<'_>, checked: bool, output: &mut Buffer) -> Result<(), Error> where
    R::Total: Cast {
    crate::error::checked(|| {
        if checked && !fits { return Err(Error::new(Kind::Overflow, "integer fold overflowed")); }
        let totals = R::Total::of_mut(&mut buffer).expect("an associative total keeps its storage type");
        if let Some(read) = seed.read() {
            let values = seed_row::<R::Total>(&read, 0, totals.len(), checked, &mut Vec::new())?;
            for (total, value) in totals.iter_mut().zip(values) {
                let (result, fits) = if matches!(seed, Seed::Initial(_)) { R::merge(value, *total) } else { R::merge(*total, value) };
                *total = result;
                if checked && !fits { crate::error::record(Error::new(Kind::Overflow, "integer fold overflowed")); }
            }
        }
        *output = buffer;
        Ok(())
    })
}

/// `view`'s items as a slice of `T`, where they have `T`'s dtype.
fn items<'a, T: Element>(view: View<'a>) -> Result<&'a [T], Error> { T::of(view).ok_or_else(|| Error::new(Kind::Domain, "fold input has the wrong dtype")) }

/// A reduction's totals as `output`, as `finish_fold` keeps them without a seed: an overflow error where a checked total doesn't fit.
fn totals<T, R: Reduction<T>>((totals, fits): (Vec<R::Total>, bool), checked: bool, output: &mut Buffer) -> Result<(), Error> where R::Total: Cast + Totals {
    finish_fold::<T, R>((R::Total::buffer_of(totals), fits), Seed::Identity, checked, output)
}

impl Reduce {
    pub fn run_scan(
        self, variant: crate::domain::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, input: crate::agreement::Read<'_>,
        output: &mut crate::buffer::Buffer) -> Result<(), Error> {
        if self == Reduce::Sum && let Some(done) = bounded_scan(variant, dims, seed, direction, false, input, output) { return done; }
        native_reduction!(self, variant.dtype, variant.fails_on_overflow(), |R, T| Some(native_scan::<T, R>(
            dims, seed, direction, variant.checked, input, output
        ))).unwrap_or_else(|| Err(Error::new(Kind::Domain, "scan does not support this dtype")))
    }
}

fn native_scan<T, R>(
    dims: [usize; 3], seed: Seed<'_>, direction: Direction, checked: bool, input: crate::agreement::Read<'_>, output: &mut crate::buffer::Buffer
) -> Result<(), Error> where T: crate::buffer::Cast, R: Reduction<T>, R::Total: crate::buffer::Cast {
    scan_read::<T, R::Total>(dims, seed, direction, checked, input, output, &|items, dims, seed, out| {
        let fits = scan_along::<T, R>(items, dims, seed, out);
        if checked && !fits { Err(Error::new(Kind::Overflow, "integer scan overflowed")) } else { Ok(()) }
    })
}

/// A left scan by `-`, run as `+`'s scan (`scan_along`) of negated items (`negated_cells`). Floats and complex numbers, 16-bit floats in 32 bits as the ordered scan computes them, and integers and Booleans whose caller asks for totals in `i16` or `i32` (`bounded_scan`). `Ok(false)`, with `output` unchanged, for other dtypes.
pub fn negated_scan(
    variant: crate::domain::Variant, dims: [usize; 3], seed: Seed<'_>, input: crate::agreement::Read<'_>, output: &mut crate::buffer::Buffer
) -> Result<bool, Error> {
    use crate::buffer::{DType, FloatWidth};
    macro_rules! negated {
        ($t:ty) => {
            scan_read::<$t, $t>(dims, seed, Direction::Left, variant.checked, input, output, &|items, dims, seed, out| {
                super::scans::negated_cells::<$t, Sum>(items, dims, seed, out, |x| -x);
                Ok(())
            })
        };
    }
    match variant.dtype {
        DType::Float(FloatWidth::F16 | FloatWidth::F32) => negated!(f32)?,
        DType::Float(FloatWidth::F64) => negated!(f64)?,
        DType::Complex(w) => with_complex_width!(w, C => negated!(C))?,
        DType::Integer(_) | DType::Boolean => {
            return bounded_scan(variant, dims, seed, Direction::Left, true, input, output).map_or(Ok(false), |done| done.map(|()| true));
        }
        _ => return Ok(false),
    }
    Ok(true)
}

/// A sum scan of integers or Booleans, or with `negate` a left scan by `-` of them, whose caller asks for totals in `output`'s dtype, `i16` or `i32`, wide enough for every stored item that the scan reads, whatever width the kernel computes at. The caller asks only where its bound on every running total fits that type. Each lane totals in it through `SumAs`, wrapping. `None` for any other request.
pub fn bounded_scan(
    variant: crate::domain::Variant, dims: [usize; 3], seed: Seed<'_>, direction: Direction, negate: bool, input: crate::agreement::Read<'_>,
    output: &mut crate::buffer::Buffer) -> Option<Result<(), Error>> {
    use crate::buffer::{DType, Width};
    let DType::Integer(total @ (Width::I16 | Width::I32)) = output.dtype() else { return None };
    let fits = |d: DType| d == DType::Boolean || matches!(d, DType::Integer(w) if total.join(w) == total);
    let stored = input.plan.try_for_each_source(&mut |s, _| if input.sources.get(s).is_some_and(|b| fits(b.view.dtype())) { Ok(()) } else { Err(()) });
    if !matches!(variant.dtype, DType::Boolean | DType::Integer(_)) || stored.is_err() { return None; }
    let checked = variant.fails_on_overflow();
    Some(if total == Width::I16 { totalled::<i16>(dims, seed, direction, checked, negate, input, output) }
    else { totalled::<i32>(dims, seed, direction, checked, negate, input, output) })
}

/// `bounded_scan`'s scan in `C`, reading the items as `C`. Where `checked`, each total is then tested from the total before it and its item: a sum overflows where its sign differs from both its arguments', and a difference where its arguments' signs differ and its sign differs from the first's. Every item fits `C`, which makes the tests exact. An overflow fails the scan.
fn totalled<C>(
    dims: [usize; 3], seed: Seed<'_>, direction: Direction, checked: bool, negate: bool, input: crate::agreement::Read<'_>, output: &mut crate::buffer::Buffer
) -> Result<(), Error> where C: crate::buffer::Cast + Tally + Term + Into<i64> + Ord + Default + std::ops::BitXor<Output = C> + std::ops::BitAnd<Output = C>
        + std::ops::BitOr<Output = C> {
    scan_read::<C, C>(dims, seed, direction, checked, input, output, &|items, [g, n, inner], seed, out| {
        let from = out.len();
        if negate { super::scans::negated_cells::<C, SumAs<C>>(items, [g, n, inner], seed, out, |x| C::low(-x.into())) }
        else { scan_along::<C, SumAs<C>>(items, [g, n, inner], seed, out); }
        if !checked { return Ok(()); }
        let test = |p: C, x: C, c: C| if negate { (p ^ x) & (p ^ c) } else { (p ^ c) & (x ^ c) };
        let along = |p: &[C], x: &[C], c: &[C]| p.iter().zip(x).zip(c).fold(C::default(), |b, ((&p, &x), &c)| b | test(p, x, c));
        let mut bad = C::default();
        for (x, c) in items.chunks_exact(n * inner).zip(out[from..].chunks_exact(n * inner)) {
            bad = bad | along(&c[..(n - 1) * inner], &x[inner..], &c[inner..]);
            if let Some(s) = seed { bad = bad | (0..inner).fold(C::default(), |b, i| b | test(s[if s.len() == 1 { 0 } else { i }], x[i], c[i])); }
        }
        if bad < C::default() { Err(Error::new(Kind::Overflow, "integer scan overflowed")) } else { Ok(()) }
    })
}

/// Cell `cell`'s `inner` seed items from `read`: its one item in every column where it holds one, else the cell's own row.
pub fn seed_row<T: crate::buffer::Cast>(
    read: &crate::agreement::Read<'_>, cell: usize, inner: usize, checked: bool, scratch: &mut Vec<T>) -> Result<Vec<T>, Error> {
    let one = read.plan.len() == 1;
    let arg = read.part::<T>(if one { 0..1 } else { cell * inner..(cell + 1) * inner }, checked, scratch)?;
    Ok((0..inner).map(|i| arg.nth(if one { 0 } else { i })).collect())
}

pub fn scan_read<T: Cast, S: Cast>(
    dims: [usize; 3], seed: Seed<'_>, direction: Direction, checked: bool, input: Read<'_>, output: &mut Buffer,
    run: &dyn Fn(&[T], [usize; 3], Option<&[S]>, &mut Vec<S>) -> Result<(), Error>) -> Result<(), Error> {
    scan_with_carry::<T, S, S>(dims, seed, direction, checked, input, output, run)
}

/// A Boolean-result scan with its raw seed and chunk carries read at the input's numeric type.
pub fn boolean_scan_read<T: Cast>(
    dims: [usize; 3], seed: Seed<'_>, direction: Direction, checked: bool, input: Read<'_>, output: &mut Buffer,
    run: &dyn Fn(&[T], [usize; 3], Option<&[T]>, &mut Vec<bool>) -> Result<(), Error>) -> Result<(), Error> where bool: Source<T> {
    scan_with_carry::<T, T, bool>(dims, seed, direction, checked, input, output, run)
}

fn scan_with_carry<T: Cast, C: Cast, S: Cast + Source<C>>(
    dims: [usize; 3], seed: Seed<'_>, direction: Direction, checked: bool, input: Read<'_>, output: &mut Buffer,
    run: &dyn Fn(&[T], [usize; 3], Option<&[C]>, &mut Vec<S>) -> Result<(), Error>) -> Result<(), Error> {
    use crate::agreement::{Mapping, Missing, Read};
    let [outer, len, inner] = dims;
    let reversed = matches!(direction, Direction::Right).then(|| {
        input.plan.remapped(Mapping::from(vec![
            Steps::along(outer, len * inner), Steps::Stride { start: len.saturating_sub(1) * inner, len, step: -(inner as isize) }, Steps::along(inner, 1),
        ]), outer * len * inner, Missing::Invalid)
    });
    let input = Read { plan: reversed.as_ref().unwrap_or(input.plan), ..input };
    let (mut values, mut scratch, mut seed_scratch) = (Vec::with_capacity(outer * len * inner), Vec::new(), Vec::new());
    if inner == 0 { *output = S::buffer(values); return Ok(()); }
    // Whole cells that start from the same seed and fit in a chunk scan in one call, so `rows_scan` runs their lanes side by side.
    let shared = !seed.read().is_some_and(|read| read.plan.len() != 1);
    let cells = if shared && len * inner <= CHUNK { CHUNK / (len * inner).max(1) } else { 1 };
    for cell in (0..outer).step_by(cells) {
        let g = cells.min(outer - cell);
        let mut carry = seed.read().map(|read| seed_row(&read, cell, inner, checked, &mut seed_scratch)).transpose()?;
        let rows = (CHUNK / inner).max(1);
        for row in (0..len).step_by(rows) {
            let n = rows.min(len - row);
            let at = (cell * len + row) * inner;
            match input.part::<T>(at..at + g * n * inner, checked, &mut scratch)? {
                Arg::Many(items) => run(items, [g, n, inner], carry.as_deref(), &mut values)?,
                Arg::One(value) => {
                    scratch.clear();
                    scratch.resize(g * n * inner, value);
                    run(&scratch, [g, n, inner], carry.as_deref(), &mut values)?;
                }
            }
            let carry = carry.get_or_insert_with(Vec::new);
            carry.clear();
            let tail = &values[values.len() - inner..];
            if let Some(items) = S::borrowed(tail) { carry.extend_from_slice(items); } else { carry.extend(tail.iter().copied().map(Source::<C>::read)); }
        }
    }
    if reversed.is_some() { reverse_rows(&mut values, dims); }
    *output = S::buffer(values);
    Ok(())
}

/// Each item of `input`, an array of shape `dims`, combined with the item one row before it along the middle axis, and each lane's first row with `seed`'s item in its column, or kept as it is without a seed, as an inverse scan runs it. `block` combines the items of two equal slices, the earlier items first unless `flip`, and gives the kind of any combination that failed: a formula's loop over a block. A failure is an error where the read is checked.
///
/// # Safety
/// `block` must initialize every output slot before returning, including when it returns `Err`.
pub unsafe fn adjacent_read<T: crate::buffer::Cast>(
    [outer, len, inner]: [usize; 3], seed: Option<crate::agreement::Read<'_>>, checked: bool, flip: bool, input: crate::agreement::Read<'_>,
    output: &mut crate::buffer::Buffer, block: &dyn Fn(&[T], &[T], &mut [std::mem::MaybeUninit<T>]) -> Result<(), Kind>) -> Result<(), Error> {
    let (mut values, mut scratch, mut seed_scratch) = (Vec::with_capacity(outer * len * inner), Vec::new(), Vec::new());
    let mut failure = None;
    let mut pair = |earlier: &[T], items: &[T], values: &mut Vec<T>| {
        failure = failure.or(unsafe { appended(values, items.len(), |out| if flip { block(items, earlier, out) } else { block(earlier, items, out) }) }.err());
    };
    for cell in (0..outer).filter(|_| len > 0) {
        let start = seed.map(|read| seed_row(&read, cell, inner, checked, &mut seed_scratch)).transpose()?;
        let lane: std::borrow::Cow<'_, [T]> = match input.part::<T>(cell * len * inner..(cell + 1) * len * inner, checked, &mut scratch)? {
            Arg::Many(items) => items.into(), Arg::One(item) => vec![item; len * inner].into()
        };
        match &start { Some(start) => pair(start, &lane[..inner], &mut values), None => values.extend_from_slice(&lane[..inner]) }
        pair(&lane[..(len - 1) * inner], &lane[inner..], &mut values);
    }
    if checked && let Some(kind) = failure { return Err(Error::new(kind, "a combination of adjacent items failed")); }
    *output = T::buffer(values);
    Ok(())
}

pub fn reverse_rows<T>(values: &mut [T], [outer, len, inner]: [usize; 3]) {
    if len < 2 || inner == 0 { return; }
    for cell in values[..outer * len * inner].chunks_exact_mut(len * inner) {
        for row in 0..len / 2 { for column in 0..inner { cell.swap(row * inner + column, (len - row - 1) * inner + column); } }
    }
}
