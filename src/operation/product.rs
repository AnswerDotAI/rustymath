use super::*;

pub fn product(
    map: &Kernel, fold: &Kernel, inputs: [crate::agreement::Read<'_>; 2], plan: &crate::linalg::ProductPlan, settings: Settings, output: &mut Buffer
) -> Result<(), Error> {
    use crate::agreement::{Missing, Read};
    let dtype = output.dtype();
    if let Some(native) = map.native_product(map.variant, fold, map.boolean_inputs, Some(dtype), settings) {
        let ran = map.projected(&inputs, None, Seed::Identity, settings, |inputs, _| (native.run)(map.variant, [inputs[0], inputs[1]], plan, output))?;
        if ran {
            return finish(output, dtype, crate::domain::Variant { checked: map.variant.checked || fold.variant.checked, ..map.variant });
        }
        crate::events::declined(crate::events::Route::Contraction, "arguments that don't suit a matrix product", plan.layout.count());
    } else {
        crate::events::declined(crate::events::Route::Contraction, "functions with no matrix product at these types", plan.layout.count());
    }
    let plans = std::array::from_fn::<_, 2, _>(|i| inputs[i].plan.remapped(plan.mappings[i].clone(), plan.iteration.count(), Missing::Invalid));
    let inputs = std::array::from_fn::<_, 2, _>(|i| Read { sources: inputs[i].sources, plan: &plans[i] });
    let shape = plan.iteration.shape();
    let split = shape.len() - plan.contracted;
    let dims = [crate::layout::count(&shape[..split]), crate::layout::count(&shape[split..]), 1];
    let pieces = |range: std::ops::Range<usize>, f: &mut dyn FnMut(usize, Piece<'_>)| {
        let mut values = Buffer::with_capacity(map.result_dtype(), range.len());
        map.map(&inputs, range.clone(), settings, &mut values)?;
        f(range.start, Piece::Items(values.view()));
        Ok(())
    };
    fold.fold_pieces(dims, Seed::Identity, Direction::Right, settings, &pieces, output)
}

type ProductRun = fn(crate::domain::Variant, [crate::agreement::Read<'_>; 2], &crate::linalg::ProductPlan, &mut Buffer) -> Result<bool, Error>;
#[derive(Clone, Copy)]
pub(super) struct NativeProduct { pub(super) result: DType, run: ProductRun }

/// The native product a map formula `K` and fold run as, where one does (`product_pair`), at `variant`'s dtype (`native`): an integer that doesn't wrap computes at i64, and a Boolean or f16 result that `wanted` names computes at that dtype. `None` where the pair has no native form, or a tolerant comparison would run as an exact count.
pub(super) fn product_for<K: crate::facts::Formula<N>, const N: usize>(
    object: &K, mut variant: crate::domain::Variant, fold: Option<crate::reduce::Reduce>, boolean_inputs: bool, wanted: Option<DType>, settings: Settings
) -> Option<NativeProduct> {
    if wanted == Some(DType::Boolean) && !boolean_inputs { return None; }
    if matches!(wanted, Some(DType::Boolean | DType::Float(FloatWidth::F16))) { variant.dtype = wanted.unwrap(); }
    if !variant.wrapping && matches!(variant.dtype, DType::Integer(_)) { variant.dtype = DType::Integer(Width::I64); }
    let pair = product_pair::<K>(fold?, if boolean_inputs { object.truth_table() } else { None }, variant.dtype == DType::Boolean)?;
    if wanted.is_some() && pair.compares_exactly() && !variant.dtype.is_integral() && settings.tolerance.value() != 0.0 { return None; }
    native(pair, variant)
}

/// Writes `product_pair` and `native` from one table. Each row names a `linalg` pair object, the map-object/fold pairs it computes, its item types, and after `wrapping` the fixed widths it computes wrapped. A Boolean row runs at `bool`, and also takes any map whose truth table matches one of its maps', as `∧` and `∨` (`tolerant::Lcm`, `tolerant::Gcd`) have `BitAnd`'s and `BitOr`'s.
macro_rules! products {
    (products {$($pair:ident: [$(($map:ty, $reduce:ident)),*] at [$($t:ty),*] $(wrapping [$($wt:ty),*])?;)*} booleans {$($bp:ident: [$(($bm:path, $bf:ident)),*];)*}) => {
        /// The native product that map object `K` and `fold` compute; for Boolean arguments, `truth` is `K`'s truth table. A `boolean` product computes at `bool`, where only the Boolean rows apply, and otherwise only the others do.
        fn product_pair<K: 'static>(fold: crate::reduce::Reduce, truth: Option<u8>, boolean: bool) -> Option<crate::linalg::Product> {
            use std::any::TypeId;
            if !boolean {
                $(if false $(|| TypeId::of::<K>() == TypeId::of::<$map>() && fold == crate::reduce::Reduce::$reduce)* { return Some(crate::linalg::Product::$pair); })*
                return None;
            }
            $(if false $(|| (TypeId::of::<K>() == TypeId::of::<$bm>() || truth.is_some() && truth == crate::facts::Formula::<2>::truth_table(&$bm))
                && fold == crate::reduce::Reduce::$bf)* { return Some(crate::linalg::Product::$bp); })*
            None
        }
        /// `pair`'s typed run at `variant`'s dtype, wrapped where `variant` wraps an integer.
        fn native(pair: crate::linalg::Product, variant: crate::domain::Variant) -> Option<NativeProduct> {
            let wraps = variant.wrapping && matches!(variant.dtype, DType::Integer(_));
            match pair {
                $(crate::linalg::Product::$pair => {
                    $(if !wraps && variant.dtype == <$t as Element>::DTYPE {
                        return typed::<$t, crate::linalg::$pair>(variant, |v, x, p, out| product_typed::<$t, _>(crate::linalg::$pair, v, x, p, out));
                    })*
                    $($(if wraps && variant.dtype == <$wt as Element>::DTYPE {
                        return typed::<$wt, crate::Wrapped<crate::linalg::$pair>>(variant, |v, x, p, out| product_typed::<$wt, _>(crate::Wrapped(crate::linalg::$pair), v, x, p, out));
                    })*)?
                })*
                $(crate::linalg::Product::$bp => {
                    return typed::<bool, crate::linalg::$bp>(variant, |v, x, p, out| product_typed::<bool, _>(crate::linalg::$bp, v, x, p, out));
                })*
            }
            None
        }
    };
}
products! {
    products {
        PlusTimes: [(crate::Multiply, Sum)] at [i64, half::f16, f32, f64, num_complex::Complex32, num_complex::Complex64] wrapping [u8, i16, i32];
        MaxPlus: [(crate::Add, Max)] at [i64, f32, f64];
        MinPlus: [(crate::Add, Min)] at [i64, f32, f64];
        PlusEqual: [(crate::tolerant::Equal, Sum)] at [i16, i32, i64, f32, f64];
        PlusNotEqual: [(crate::tolerant::NotEqual, Sum)] at [i16, i32, i64, f32, f64];
    }
    booleans {
        OrAnd: [(crate::Multiply, Sum), (crate::BitAnd, Max), (crate::BitAnd, Or), (crate::Multiply, Max)];
        AndOr: [(crate::BitOr, Min), (crate::BitOr, And)];
    }
}

/// `P`'s native product at item type `T`. `finish` converts its result to the dtype that the caller asks for.
fn typed<T: Element, P: crate::linalg::Inner<T>>(variant: crate::domain::Variant, run: ProductRun) -> Option<NativeProduct> where P::Out: Element {
    Some(NativeProduct { result: product_result::<T, P>(variant), run })
}

fn product_result<T: Element, P: crate::linalg::Inner<T>>(variant: crate::domain::Variant) -> DType where P::Out: Element {
    if variant.wrapping && T::DTYPE.is_integral() || T::DTYPE == DType::Float(FloatWidth::F16) { T::DTYPE }
    else if matches!(P::Out::DTYPE, DType::Integer(_)) { DType::Integer(Width::I64) }
    else { P::Out::DTYPE }
}

fn product_typed<T: crate::buffer::Cast, P: crate::linalg::Inner<T>>(
    pair: P, variant: crate::domain::Variant, inputs: [crate::agreement::Read<'_>; 2], plan: &crate::linalg::ProductPlan, output: &mut Buffer
) -> Result<bool, Error> where P::Out: crate::buffer::Cast {
    use crate::agreement::{Mapping, Missing, Read, Walked};
    use crate::gather::Steps;
    use crate::linalg::{Mapped, Orientation};
    let Some(matrix) = plan.matrix_for(inputs) else { return Ok(false); };
    let [m, k, n] = matrix.size;
    if !variant.wrapping && matches!(P::Out::DTYPE, DType::Integer(w) if k > i64::MAX as usize || !w.holds(k as i64)) { return Ok(false); }
    let (r, frame_shape) = (matrix.frame_rank, &plan.layout.shape()[..matrix.frame_rank]);
    let frames = crate::layout::count(frame_shape);
    let expanded = std::array::from_fn::<_, 2, _>(|i| inputs[i].plan.remapped(plan.mappings[i].clone(), plan.iteration.count(), Missing::Invalid));
    let reads = std::array::from_fn::<_, 2, _>(|i| Read { sources: inputs[i].sources, plan: &expanded[i] });
    // Each operand's walk over the frames and the matrix axes `[m, n, k]`, composed once where it reads its source positionally: a frame moves only the walk's base. A source of another dtype converts to `T` once, whole.
    let walked: Vec<usize> = frame_shape.iter().copied().chain([m, n, k]).collect();
    let positional = reads.each_ref().map(|read| match read.positional(&walked) {
        Some((view, Mapping::Steps { base, steps })) => Some((view, base, steps)), _ => None
    });
    let mut casts: [Vec<T>; 2] = [Vec::new(), Vec::new()];
    for (cast, p) in casts.iter_mut().zip(&positional) {
        if let Some((view, ..)) = p && T::of(*view).is_none() { T::extend_cast(*view, view.len(), variant.checked, cast)?; }
    }
    let walks = std::array::from_fn::<_, 2, _>(|i| positional[i].as_ref().map(|(view, base, steps)| (T::of(*view).unwrap_or(&casts[i][..]), *base, steps)));
    let mut result = Vec::with_capacity(plan.layout.count());
    let overflow = || if variant.checked { Err(Error::new(Kind::Overflow, "product does not fit its integer result")) } else { Ok(false) };
    let mut multiply = |a: Mapped<'_, T>, b: Mapped<'_, T>| -> Result<bool, Error> {
        if T::DTYPE.is_integral() {
            let (mut ab, mut bb) = (Vec::new(), Vec::new());
            let start = result.len();
            result.resize(start + m * n, P::Out::default());
            if !pair.checked_into(a.read(0, m * k, &mut ab), b.read(0, k * n, &mut bb), matrix.size, &mut result[start..]) { return overflow(); }
        } else { result.extend(pair.inner_mapped(a, b, [Orientation::Normal; 2], matrix.size)); }
        Ok(true)
    };
    if let [Some((a, a0, a_steps)), Some((b, b0, b_steps))] = &walks {
        // A frame's matrix along axes `axes` of `[m, n, k]`, at the first item of axis `other`, which the operand doesn't step along.
        let matrix_at = |base: usize, steps: &[Steps], frame: usize, axes: [usize; 2], other: usize| {
            Some(Walked::from(Mapping::Steps { base: Steps::reach(base, &steps[..r], frame)? + steps[r + other].get(0)?,
                steps: axes.map(|a| steps[r + a].clone()).into() }))
        };
        // Where both matrix axes of each operand step evenly, a stack takes each frame's two matrices as row-major slices (`frame_matrix`), so a tiny product runs `Inner`'s direct loop with nothing allocated per frame.
        let strided = |steps: &[Steps], axes: [usize; 2], other: usize| match axes.map(|x| &steps[r + x]) {
            [&Steps::Stride { start: s0, step: t0, .. }, &Steps::Stride { start: s1, step: t1, .. }] => Some((steps[r + other].get(0)? + s0 + s1, [t0, t1])),
            _ => None,
        };
        if frames > 1 && let (Some((ao, at)), Some((bo, bt))) = (strided(a_steps, [0, 2], 1), strided(b_steps, [2, 1], 0)) {
            let (mut ab, mut bb) = (Vec::new(), Vec::new());
            result.resize(frames * m * n, P::Out::default());
            for frame in 0..frames {
                let (Some(fa), Some(fb)) = (Steps::reach(*a0, &a_steps[..r], frame), Steps::reach(*b0, &b_steps[..r], frame)) else { return Ok(false) };
                let (x, y) = (frame_matrix(a, fa + ao, at, [m, k], &mut ab), frame_matrix(b, fb + bo, bt, [k, n], &mut bb));
                if !pair.checked_into(x, y, matrix.size, &mut result[frame * m * n..][..m * n]) { return overflow(); }
            }
            *output = P::Out::buffer(result);
            return Ok(true);
        }
        for frame in 0..frames {
            let (Some(am), Some(bm)) = (matrix_at(*a0, a_steps, frame, [0, 2], 1), matrix_at(*b0, b_steps, frame, [2, 1], 0)) else { return Ok(false) };
            if !multiply(Mapped { data: *a, mapping: &am }, Mapped { data: *b, mapping: &bm })? { return Ok(false); }
        }
    } else {
        for frame in 0..frames {
            let base = frame * m * n * k;
            let left = expanded[0].remapped(Mapping::Steps { base, steps: vec![Steps::along(m, n * k), Steps::along(k, 1)].into() }, m * k, Missing::Invalid);
            let right = expanded[1].remapped(Mapping::Steps { base, steps: vec![Steps::along(k, 1), Steps::along(n, k)].into() }, k * n, Missing::Invalid);
            let reads = [Read { sources: inputs[0].sources, plan: &left }, Read { sources: inputs[1].sources, plan: &right }];
            let (mut a_items, mut b_items) = (Vec::new(), Vec::new());
            let (a, am) = matrix_items::<T>(&reads[0], [m, k], variant.checked, &mut a_items)?;
            let (b, bm) = matrix_items::<T>(&reads[1], [k, n], variant.checked, &mut b_items)?;
            if !multiply(Mapped { data: &a[..], mapping: &am }, Mapped { data: &b[..], mapping: &bm })? { return Ok(false); }
        }
    }
    *output = P::Out::buffer(result);
    Ok(true)
}

/// The `len` items that `read` reads, as one slice: `scratch` holds them where the read copies them.
fn read_all<'a, T: crate::buffer::Cast>(
    read: &'a crate::agreement::Read<'_>, len: usize, checked: bool, scratch: &'a mut Vec<T>) -> Result<std::borrow::Cow<'a, [T]>, Error> {
    Ok(match read.part::<T>(0..len, checked, scratch)? { Arg::One(x) => std::borrow::Cow::Owned(vec![x; len]), Arg::Many(x) => std::borrow::Cow::Borrowed(x) })
}

/// The items of `read`, a matrix of `shape`, with the walk that reaches each one: the read's source where the read is positional or maps one source with no fills, and otherwise its items read once (`read_all`).
fn matrix_items<'a, T: crate::buffer::Cast>(
    read: &'a crate::agreement::Read<'_>, shape: [usize; 2], checked: bool, scratch: &'a mut Vec<T>
) -> Result<(std::borrow::Cow<'a, [T]>, crate::agreement::Walked<'static>), Error> {
    use crate::agreement::{Mapping, Walked};
    if let Some((view, mapping)) = read.positional(&shape) && let Some(data) = T::of(view) {
        return Ok((std::borrow::Cow::Borrowed(data), Walked::from(mapping)));
    }
    if let Some((view, mapping)) = read.mapped_source() && let Some(data) = T::of(view) {
        return Ok((std::borrow::Cow::Borrowed(data), Walked::from(mapping.clone())));
    }
    Ok((read_all(read, shape[0] * shape[1], checked, scratch)?, Walked::from(Mapping::Linear(1))))
}

/// The `rows`×`cols` matrix of `data` from `at`, with `strides` across rows and across columns, as row-major items: `data`'s own items where they lie that way, and otherwise items gathered into `scratch`.
fn frame_matrix<'a, T: Copy>(data: &'a [T], at: usize, [across_rows, across_cols]: [isize; 2], [rows, cols]: [usize; 2], scratch: &'a mut Vec<T>) -> &'a [T] {
    if (rows <= 1 || across_rows == cols as isize) && (cols <= 1 || across_cols == 1) { return &data[at..at + rows * cols]; }
    scratch.clear();
    scratch.extend((0..rows * cols).map(|ij| data[at.wrapping_add_signed((ij / cols) as isize * across_rows + (ij % cols) as isize * across_cols)]));
    scratch
}
