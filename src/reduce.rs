//! Reductions and scans over a contiguous slice: sums, products, maximums and minimums, and their running versions. Each compiles inside rustymath once for each item type. A float sum or product may combine the items in any order, which lets its loop vectorize, so its last bits can differ from a sum taken item by item. Beside a NaN item, a float maximum or minimum gives either NaN or the greatest or least other item, depending on the target and the loop. An empty slice gives -∞ or ∞. `f16` items compute in `f32`. Their sum or product is an `f32` total. An integer sum or product computes at `i64`, or at `u64` for `u64` items. Where it doesn't fit, it wraps, and reports that it doesn't fit, as the running versions do.
use crate::{Arg, Float, Native, agreement::{Mapping, expanded_items}, arith::Overflowing, buffer::{Element, View}, float::sealed::{Core, Folds}, gather::Steps, layout::Lanes, parallel, slice::{CHUNK, appended, written}};
use half::f16;
use num_complex::Complex;
use std::ops::Range;

/// The sum of `x`, 0 for an empty slice, and whether it fits. An integer sum that doesn't fit wraps.
#[inline]
pub fn sum<T>(x: &[T]) -> (<Sum as Reduction<T>>::Total, bool) where Sum: Reduction<T> { Sum::slice(x) }
/// The product of `x`, 1 for an empty slice, and whether it fits. An integer product that doesn't fit wraps.
#[inline]
pub fn prod<T>(x: &[T]) -> (<Product as Reduction<T>>::Total, bool) where Product: Reduction<T> { Product::slice(x) }
/// The greatest item of `x`.
#[inline]
pub fn max<T>(x: &[T]) -> T where Max: Reduction<T, Total = T> { Max::slice(x).0 }
/// The least item of `x`.
#[inline]
pub fn min<T>(x: &[T]) -> T where Min: Reduction<T, Total = T> { Min::slice(x).0 }
/// The least and the greatest item of `x`, in one pass of comparisons and selects. For floats, [`min`] and [`max`] take less time together, because their folds run AArch64's `fmin` and `fmax`.
#[inline]
pub fn extremes<T: Copy>(x: &[T]) -> (T, T) where Min: Reduction<T, Total = T>, Max: Reduction<T, Total = T> { extremes_of(x, |v| v) }

/// The least and the greatest `key` of the items of `x`, in the pass [`extremes`] makes. A `key` that only reads an item as another type, as `char as u32` does, costs nothing.
#[inline]
pub(crate) fn extremes_of<A: Copy, T: Copy>(x: &[A], key: impl Fn(A) -> T + Copy) -> (T, T) where Min: Reduction<T, Total = T>, Max: Reduction<T, Total = T> {
    match crate::slice::running(2 * size_of::<T>()) {
        8 => extremes_by::<A, T, 8>(x, key), 16 => extremes_by::<A, T, 16>(x, key), _ => extremes_by::<A, T, 32>(x, key)
    }
}

/// `extremes_of` with `R` running minimums and `R` running maximums, in two arrays: the compiler leaves one array of pairs unvectorized.
#[inline(always)]
fn extremes_by<A: Copy, T: Copy, const R: usize>(x: &[A], key: impl Fn(A) -> T) -> (T, T) where Min: Reduction<T, Total = T>, Max: Reduction<T, Total = T> {
    let (min, max) = (<Min as Reduction<T>>::step, <Max as Reduction<T>>::step);
    let (chunks, rest) = x.as_chunks::<R>();
    let (mut lo, mut hi) = ([<Min as Reduction<T>>::unit(); R], [<Max as Reduction<T>>::unit(); R]);
    for chunk in chunks {
        for k in 0..R {
            let v = key(chunk[k]);
            (lo[k], hi[k]) = (min(lo[k], v), max(hi[k], v));
        }
    }
    let (lo, hi) = (lo.into_iter().fold(<Min as Reduction<T>>::unit(), min), hi.into_iter().fold(<Max as Reduction<T>>::unit(), max));
    rest.iter().fold((lo, hi), |(lo, hi), &v| { let v = key(v); (min(lo, v), max(hi, v)) })
}
/// Appends the running sums of `x` to `out`, the first being `start` plus the first item where `start` is given, and gives whether every sum fits.
#[inline]
pub fn cumsum<T>(x: &[T], start: Option<<Sum as Reduction<T>>::Total>, out: &mut Vec<<Sum as Reduction<T>>::Total>) -> bool where Sum: Reduction<T> {
    Sum::scan(x, start, out)
}
/// Appends the running products of `x` to `out`, as `cumsum` does sums.
#[inline]
pub fn cumprod<T>(x: &[T], start: Option<<Product as Reduction<T>>::Total>, out: &mut Vec<<Product as Reduction<T>>::Total>) -> bool where Product: Reduction<T>
{ Product::scan(x, start, out) }
/// Appends the running maximums of `x` to `out`, the first taking `start` into account where it is given.
#[inline]
pub fn cummax<T>(x: &[T], start: Option<T>, out: &mut Vec<T>) where Max: Reduction<T, Total = T> { Max::scan(x, start, out); }
/// Appends the running minimums of `x` to `out`, as `cummax` does maximums.
#[inline]
pub fn cummin<T>(x: &[T], start: Option<T>, out: &mut Vec<T>) where Min: Reduction<T, Total = T> { Min::scan(x, start, out); }

/// The greater of x and y, as [`max`] gives the greatest item. Where they're equal, or either is NaN, it gives y. NumPy calls it `maximum`, though NumPy's gives NaN beside NaN.
#[inline(always)]
pub fn maximum<F: Float>(x: F, y: F) -> F { F::from_working(Core::maximum(x.working(), y.working())) }

/// The lesser of x and y, as [`maximum`] gives the greater. NumPy calls it `minimum`.
#[inline(always)]
pub fn minimum<F: Float>(x: F, y: F) -> F { F::from_working(Core::minimum(x.working(), y.working())) }

/// [`maximum`] as a function object, which also takes integers and Booleans. The maximum of two Booleans is their `or`.
#[derive(Clone, Copy, Debug)]
pub struct Maximum;
forms!(Maximum at real: |_, [x, y]| crate::slice::Total(maximum(x, y)));
forms!(Maximum at u8, i16, i32, i64: |_, [x, y]| x.max(y));
forms!(Maximum at bool: |_, [x, y]| x | y);

/// [`minimum`] as a function object, which also takes integers and Booleans. The minimum of two Booleans is their `and`.
#[derive(Clone, Copy, Debug)]
pub struct Minimum;
forms!(Minimum at real: |_, [x, y]| crate::slice::Total(minimum(x, y)));
forms!(Minimum at u8, i16, i32, i64: |_, [x, y]| x.min(y));
forms!(Minimum at bool: |_, [x, y]| x & y);

/// A reduction by [`Sum`], [`Product`], [`Max`], [`Min`], [`Xor`], [`Xnor`] or [`SumAs`], implemented inside rustymath once for each item type it takes. `slice` folds a whole slice of consecutive items, in any order, `merge` combines the folds of two neighbouring runs, and `scan` gives the running folds. [`along`], [`fold_into`] and [`scan_along`] fold along an axis whose items aren't consecutive. [`Sum`] and [`Product`] take floats, integers and complex numbers. [`Max`] and [`Min`] take floats, integers and Booleans, [`Xor`] and [`Xnor`] take Booleans, and [`SumAs`] takes Booleans and integers. An `f16` reduction computes in `f32`. A sum or product of `f16`s gives an `f32` total, for its caller to round once. A total that doesn't fit its type fails, with its value wrapped to that type.
pub trait Reduction<T> {
    /// The type of a fold: the item type, `i64` for a sum or product of integers narrower than 64 bits, `f32` for a sum or product of `f16`s, or the `C` of [`SumAs`].
    type Total: Copy + Send + Sync;
    /// The fold of no items: 0, 1, -∞ or ∞, the integer type's least or greatest value, or false or true.
    fn unit() -> Self::Total;
    /// The exact identity of `step`, which leaves every total unchanged. For float sums it's -0.0, because adding `unit()`, 0.0, turns a total of -0.0 into 0.0. For every other reduction it's `unit()`.
    #[inline(always)]
    fn identity() -> Self::Total { Self::unit() }
    /// `acc` combined with one more item.
    fn step(acc: Self::Total, x: T) -> Self::Total;
    /// `step`, and whether its result fits. An integer sum or product wraps where it doesn't.
    #[inline(always)]
    fn checked(acc: Self::Total, x: T) -> (Self::Total, bool) { (Self::step(acc, x), true) }
    /// The fold of the one item `x`, `checked(identity(), x)`, and whether it fits.
    #[inline(always)]
    fn one(x: T) -> (Self::Total, bool) { Self::checked(Self::identity(), x) }
    /// The folds of two neighbouring runs of items combined into the fold of both, and whether it fits.
    fn merge(a: Self::Total, b: Self::Total) -> (Self::Total, bool);
    /// The fold of `x`, which is `unit()` where `x` is empty, and whether it fits.
    fn slice(x: &[T]) -> (Self::Total, bool);
    /// Appends the running folds of `x` to `out`, the first combining `start` where it is given, and gives whether every one fits.
    fn scan(x: &[T], start: Option<Self::Total>, out: &mut Vec<Self::Total>) -> bool;
}

/// [`sum`] as a [`Reduction`].
#[derive(Clone, Copy, Debug)]
pub struct Sum;
/// [`prod`] as a [`Reduction`].
#[derive(Clone, Copy, Debug)]
pub struct Product;
/// [`max`] as a [`Reduction`]. Its scalar step on floats is [`maximum`] of the item and the total, which a NaN item leaves unchanged. Vectorized folds on some targets, such as AArch64, keep a NaN instead. A fold beside NaN therefore gives NaN or the greatest other item, depending on the target and the lane's length.
#[derive(Clone, Copy, Debug)]
pub struct Max;
/// [`min`] as a [`Reduction`]. Its scalar step on floats is [`minimum`] of the item and the total, which a NaN item leaves unchanged. Vectorized folds on some targets, such as AArch64, keep a NaN instead. A fold beside NaN therefore gives NaN or the least other item, depending on the target and the lane's length.
#[derive(Clone, Copy, Debug)]
pub struct Min;
/// The fold of Booleans by `!=`: true where an odd number of them are true.
#[derive(Clone, Copy, Debug)]
pub struct Xor;
/// The fold of Booleans by `==`: true where an even number of them are false.
#[derive(Clone, Copy, Debug)]
pub struct Xnor;
forms!(Xor at bool: |_, [x, y]| x != y);
forms!(Xnor at bool: |_, [x, y]| x == y);

/// `x` folded by `add` from `unit`, through as many running results as `running_results` gives, which `merge` combines at the end.
#[inline(always)]
pub(crate) fn folded<A: Copy, S: Copy>(x: &[A], unit: S, add: impl Fn(S, A) -> S + Copy, merge: impl Fn(S, S) -> S) -> S {
    running_results(x, unit, add, move |parts, rest| rest.iter().fold(parts.iter().fold(unit, |s, &p| merge(s, p)), |s, &v| add(s, v)))
}

/// The running results of `add` from `unit` over `x`: as many as `slice::running` gives, or fewer for a short `x`, so that each takes at least 8 items. A short slice then doesn't set up and merge more results than it has items for. Each takes the items at its own position in every whole group of that many. `finish` gets the results, and the items after the last whole group. Their count is always even.
#[inline(always)]
fn running_results<A: Copy, S: Copy, T>(x: &[A], unit: S, add: impl Fn(S, A) -> S, finish: impl FnOnce(&[S], &[A]) -> T) -> T {
    match crate::slice::running(size_of::<S>()).min(x.len() / 8) {
        ..8 => running_by::<A, S, T, 4>(x, unit, add, finish), 8..16 => running_by::<A, S, T, 8>(x, unit, add, finish),
        16..32 => running_by::<A, S, T, 16>(x, unit, add, finish), _ => running_by::<A, S, T, 32>(x, unit, add, finish)
    }
}

/// `running_results` with `R` of them. The loop updates them in one local array, which compiles to vector instructions. It indexes each chunk, because `Zip::new` isn't always inlined without LTO.
#[inline(always)]
fn running_by<A: Copy, S: Copy, T, const R: usize>(x: &[A], unit: S, add: impl Fn(S, A) -> S, finish: impl FnOnce(&[S], &[A]) -> T) -> T {
    let (chunks, rest) = x.as_chunks::<R>();
    let mut parts = [unit; R];
    for chunk in chunks { for k in 0..R { parts[k] = add(parts[k], chunk[k]) } }
    finish(&parts, rest)
}

/// The groups of four items in each block of [`grouped`]'s first pass. Its buffer of group sums stays in L1 for every total type.
const SCAN_BLOCK: usize = 64;

/// Appends the running totals of `x` by `f` to `out`, the first combining `start` where it is given. `from` reads each item as a total, and `to` writes each total. Each group of four items scans within itself in two steps, then combines with the total before it. Only one operation in each group then waits for the group before, and `f` may combine items in any order. Set `reorders` for integer sums. The compiler may reorder integer addition, and in one loop it rewrites the sums into one chain through the total, with one addition waiting for each item. With `reorders`, a first pass over each block of `SCAN_BLOCK` groups writes each group's sum to a buffer, and the total adds the sum it loads from there, which the compiler can't rewrite. Floats keep the one loop. The compiler can't reorder float operations, so their loop already keeps one operation in each group waiting, and a first pass would only add work. Integer maximums and minimums keep it too, because the first pass slows them.
#[inline(always)]
pub(crate) fn grouped<A: Copy, S: Copy, O>(x: &[A], start: Option<S>, out: &mut Vec<O>, from: impl Fn(A) -> S, f: impl Fn(S, S) -> S, to: impl Fn(S) -> O, reorders: bool) {
    let Some((&first, rest)) = x.split_first() else { return };
    appended(out, x.len(), |slots| {
        let first = from(first);
        let mut total = start.map_or(first, |s| f(s, first));
        slots[0].write(to(total));
        let ((groups, tail), (outs, last)) = (rest.as_chunks::<4>(), slots[1..].as_chunks_mut::<4>());
        let mut sums = [std::mem::MaybeUninit::<S>::uninit(); SCAN_BLOCK];
        // Indexed loops, because `Zip` over chunks isn't always inlined without LTO.
        let mut i0 = 0;
        while i0 < groups.len() {
            let n = (groups.len() - i0).min(SCAN_BLOCK);
            let (groups, outs) = (&groups[i0..i0 + n], &mut outs[i0..i0 + n]);
            if reorders {
                for g in 0..n {
                    let [a, b, c, d] = groups[g];
                    sums[g].write(f(f(from(a), from(b)), f(from(c), from(d))));
                }
            }
            for g in 0..n {
                let [a, b, c, d] = groups[g];
                let (a, b, c, d) = (from(a), from(b), from(c), from(d));
                let (ab, cd) = (f(a, b), f(c, d));
                // Four writes, not `array::map`, which the compiler can leave out of line in a build of many codegen units.
                let totals = [f(total, a), f(total, ab), f(total, f(ab, c)), f(total, f(ab, cd))];
                for k in 0..4 { outs[g][k].write(to(totals[k])); }
                // SAFETY: with `reorders`, the first pass wrote entries 0 to `n`.
                total = if reorders { f(total, unsafe { sums[g].assume_init() }) } else { totals[3] };
            }
            i0 += n;
        }
        for (o, &v) in last.iter_mut().zip(tail) {
            total = f(total, from(v));
            o.write(to(total));
        }
    })
}

/// The position of the first item of `x` that `test` picks, or its length where it picks none. Each block of 64 items is tested with no branch, so the test vectorizes.
#[inline]
fn first_where<T: Copy>(x: &[T], test: impl Fn(T) -> bool) -> usize {
    let (blocks, _) = x.as_chunks::<64>();
    let from = blocks.iter().position(|b| b.iter().fold(false, |any, &v| any | test(v))).unwrap_or(blocks.len()) * 64;
    x[from..].iter().position(|&v| test(v)).map_or(x.len(), |i| from + i)
}

/// Each float reduction at `f32` and `f64`: its unit, its identity where that isn't its unit, its step, its fold of a slice, and its scan.
macro_rules! floats {
    ($($r:ident: $unit:ident, $(identity: $identity:expr,)? |$a:ident, $b:ident| $step:expr, $fold:expr, $scan:expr;)+) => {$(
        impl<F: Native> Reduction<F> for $r {
            type Total = F;
            #[inline(always)]
            fn unit() -> F { F::$unit() }
            #[inline(always)]
            fn step($a: F, $b: F) -> F { $step }
            #[inline(always)]
            fn merge(a: F, b: F) -> (F, bool) { (Self::step(a, b), true) }
            $(#[inline(always)]
            fn identity() -> F { $identity })?
            #[inline]
            fn slice(x: &[F]) -> (F, bool) { (($fold)(x), true) }
            #[inline]
            fn scan(x: &[F], start: Option<F>, out: &mut Vec<F>) -> bool {
                ($scan)(x, start, out);
                true
            }
        }
    )+};
}
floats! {
    Sum: zero, identity: -F::zero(), |a, b| a.algebraic_add(b), |x: &[F]| if x.is_empty() { <Self as Reduction<F>>::unit() } else { folded(x, <Self as Reduction<F>>::identity(), Self::step, Self::step) }, F::scan_sum;
    Product: one, |a, b| a.algebraic_mul(b), |x| folded(x, F::one(), Self::step, Self::step), |x, start, out| grouped(x, start, out, |v| v, |a: F, b: F| a * b, |t| t, false);
    Max: neg_infinity, |acc, x| x.maximum(acc), F::fold_max, F::scan_max;
    Min: infinity, |acc, x| x.minimum(acc), F::fold_min, F::scan_min;
}

/// Writes the reduction of `f16`s by each `$r`, which runs the same reduction of `f32`s and totals in `$total`. `$narrow` converts an `f32` total to a `$total`. `$wide` converts a `$total` back to `f32`.
macro_rules! halves {
    ($total:ty, $wide:expr, $narrow:expr; $($r:ty),+) => {$(
        impl Reduction<f16> for $r {
            type Total = $total;
            #[inline(always)]
            fn unit() -> $total { $narrow(<$r as Reduction<f32>>::unit()) }
            #[inline(always)]
            fn step(acc: $total, x: f16) -> $total { $narrow(<$r as Reduction<f32>>::step($wide(acc), x.to_f32())) }
            #[inline(always)]
            fn merge(a: $total, b: $total) -> ($total, bool) { ($narrow(<$r as Reduction<f32>>::merge($wide(a), $wide(b)).0), true) }
            #[inline(always)]
            fn identity() -> $total { $narrow(<$r as Reduction<f32>>::identity()) }
            #[inline]
            fn slice(x: &[f16]) -> ($total, bool) {
                let (unit, step) = (<$r as Reduction<f32>>::identity(), <$r as Reduction<f32>>::step);
                ($narrow(if x.is_empty() { <$r as Reduction<f32>>::unit() } else { folded(x, unit, |s, v| step(s, v.to_f32()), step) }), true)
            }
            #[inline]
            fn scan(x: &[f16], start: Option<$total>, out: &mut Vec<$total>) -> bool {
                grouped(x, start.map($wide), out, f16::to_f32, <$r as Reduction<f32>>::step, $narrow, false);
                true
            }
        }
    )+};
}
// A sum or product of `f16`s totals in `f32`, for its caller to round once. A maximum or minimum is exact in `f16`.
halves!(f32, |t: f32| t, |t: f32| t; Sum, Product);
halves!(f16, f16::to_f32, f16::from_f32; Max, Min);

/// Each sum of complex numbers adds their real parts and their imaginary parts, in any order.
impl<F: Native> Reduction<Complex<F>> for Sum {
    type Total = Complex<F>;
    #[inline(always)]
    fn unit() -> Complex<F> { Complex::new(F::zero(), F::zero()) }
    #[inline(always)]
    fn step(acc: Complex<F>, x: Complex<F>) -> Complex<F> { Complex::new(acc.re.algebraic_add(x.re), acc.im.algebraic_add(x.im)) }
    #[inline(always)]
    fn merge(a: Complex<F>, b: Complex<F>) -> (Complex<F>, bool) { (<Self as Reduction<Complex<F>>>::step(a, b), true) }
    #[inline(always)]
    fn identity() -> Complex<F> { Complex::new(-F::zero(), -F::zero()) }
    #[inline]
    fn slice(x: &[Complex<F>]) -> (Complex<F>, bool) {
        let step = <Self as Reduction<Complex<F>>>::step;
        (if x.is_empty() { <Self as Reduction<Complex<F>>>::unit() } else { folded(x, <Self as Reduction<Complex<F>>>::identity(), step, step) }, true)
    }
    #[inline]
    fn scan(x: &[Complex<F>], start: Option<Complex<F>>, out: &mut Vec<Complex<F>>) -> bool {
        grouped(x, start, out, |v| v, |a: Complex<F>, b| a + b, |t| t, false);
        true
    }
}

/// Complex products regroup multiplication through the same running totals as real products.
impl<F: Native> Reduction<Complex<F>> for Product {
    type Total = Complex<F>;
    #[inline(always)]
    fn unit() -> Complex<F> { Complex::new(F::one(), F::zero()) }
    #[inline(always)]
    fn step(acc: Complex<F>, x: Complex<F>) -> Complex<F> { acc * x }
    #[inline(always)]
    fn merge(a: Complex<F>, b: Complex<F>) -> (Complex<F>, bool) { (a * b, true) }
    #[inline]
    fn slice(x: &[Complex<F>]) -> (Complex<F>, bool) {
        let step = <Self as Reduction<Complex<F>>>::step;
        (folded(x, <Self as Reduction<Complex<F>>>::unit(), step, step), true)
    }
    #[inline]
    fn scan(x: &[Complex<F>], start: Option<Complex<F>>, out: &mut Vec<Complex<F>>) -> bool {
        grouped(x, start, out, |v| v, |a: Complex<F>, b| a * b, |t| t, false);
        true
    }
}

/// AArch64's `fmax` and `fmin`, and wasm's `max` and `min`, give NaN where either argument is NaN. On AArch64 each is one instruction. In a browser on an Arm host, such as a Mac, wasm's `max` and `min` run faster than `pmax` and `pmin`, which give their first argument where either is NaN. On x86-64 hosts, wasm's `max` and `min` take about eight instructions each. A fold runs the vector forms on `PARTS` running vectors. Each step of a scan depends on the step before. In wasm, the step takes lane 0 of the vector instruction on two splats. On AArch64 that code keeps a `dup` before each `fmax` at `f32`, which has no one-lane vector type, so the step there is inline assembly. Each type's entry names its scan. AArch64's `f64` scan is `grouped`, at the speed of memory. Skipping blocks gained it nothing. Each block with a new maximum paid for a test it didn't use. The others skip blocks through `records`. AArch64's `f32` tests a block by comparing each item, as `below` does. Wasm tests it by its fold, then one comparison. In wasm, comparing each item took five instructions for two items. Each entry also names its running sum. `f32`'s is `sum_scan`, which scans four-lane vectors and makes `+\` about 1.3 times faster. `f64`'s is `grouped`, because a two-lane vector scan needs more operations per item than `grouped`'s two additions.
#[cfg(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128")))]
mod lanes {
    use super::*;
    #[cfg(target_arch = "wasm32")]
    use core::arch::wasm32::*;
    #[cfg(target_arch = "aarch64")]
    use core::arch::{aarch64::*, asm};
    /// How many running vectors a fold keeps.
    const PARTS: usize = 8;
    macro_rules! lanes {
        ($($t:ty: $lanes:literal, $splat:ident, $load:ident, $max:ident, $min:ident, $step:ident($arg:tt), $scan:ident, $sum:ident;)+) => {$(
            impl Folds for $t {
                fn fold_max(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, NEG_INFINITY, $splat, $load, $max, |a, b| lanes!(@$step $arg, "fmax", $splat, $max, a, b)) }
                fn fold_min(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, INFINITY, $splat, $load, $min, |a, b| lanes!(@$step $arg, "fmin", $splat, $min, a, b)) }
                fn scan_max(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) {
                    #[inline(always)]
                    fn step(a: $t, b: $t) -> $t { lanes!(@$step $arg, "fmax", $splat, $max, a, b) }
                    lanes!(@scan $scan, x, start, out, step, below, |b: &[$t], t: $t| Self::fold_max(b) < t)
                }
                fn scan_min(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) {
                    #[inline(always)]
                    fn step(a: $t, b: $t) -> $t { lanes!(@$step $arg, "fmin", $splat, $min, a, b) }
                    lanes!(@scan $scan, x, start, out, step, above, |b: &[$t], t: $t| Self::fold_min(b) > t)
                }
                fn scan_sum(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) { lanes!(@sum $sum, x, start, out) }
            }
        )+};
        (@fold $x:ident, $t:ty, $lanes:literal, $unit:ident, $splat:ident, $load:ident, $op:ident, $step:expr) => {{
            let (chunks, rest) = $x.as_chunks::<{ PARTS * $lanes }>();
            // SAFETY: this module compiles only for targets with these instructions. Each load reads `$lanes` items of one chunk, and a vector holds `$lanes` items of `$t`.
            let lanes: [$t; $lanes] = unsafe {
                let mut parts = [$splat(<$t>::$unit); PARTS];
                for c in chunks { for k in 0..PARTS { parts[k] = $op(parts[k], $load(c.as_ptr().add(k * $lanes).cast())) } }
                core::mem::transmute(parts.into_iter().reduce(|a, b| $op(a, b)).expect("a fold keeps `PARTS` vectors"))
            };
            lanes.into_iter().chain(rest.iter().copied()).fold(<$t>::$unit, $step)
        }};
        (@asm $reg:literal, $ins:literal, $splat:ident, $op:ident, $x:ident, $y:ident) => {{
            let r;
            // SAFETY: the instruction reads two registers and writes a third, with no memory access.
            unsafe { asm!(concat!($ins, " {0:", $reg, "}, {1:", $reg, "}, {2:", $reg, "}"), lateout(vreg) r, in(vreg) $x, in(vreg) $y, options(pure, nomem, nostack)) }
            r
        }};
        (@lane $lane:ident, $ins:literal, $splat:ident, $op:ident, $x:ident, $y:ident) => { $lane::<0>($op($splat($x), $splat($y))) };
        (@scan grouped, $x:ident, $start:ident, $out:ident, $step:ident, $compare:expr, $fold:expr) => { grouped($x, $start, $out, |v| v, $step, |t| t, false) };
        (@scan compare, $x:ident, $start:ident, $out:ident, $step:ident, $compare:expr, $fold:expr) => { records($x, $start, $out, $step, $compare) };
        (@scan fold, $x:ident, $start:ident, $out:ident, $step:ident, $compare:expr, $fold:expr) => { records($x, $start, $out, $step, $fold) };
        (@sum grouped, $x:ident, $start:ident, $out:ident) => { grouped($x, $start, $out, |v| v, |a, b| a + b, |t| t, false) };
        (@sum vector, $x:ident, $start:ident, $out:ident) => { sum_scan($x, $start, $out) };
    }
    #[cfg(target_arch = "aarch64")]
    lanes! {
        f64: 2, vdupq_n_f64, vld1q_f64, vmaxq_f64, vminq_f64, asm("d"), grouped, grouped;
        f32: 4, vdupq_n_f32, vld1q_f32, vmaxq_f32, vminq_f32, asm("s"), compare, vector;
    }
    /// Four-lane `f32` vectors, for `sum_scan`. `up1` and `up2` move a vector up one lane or two, filling the gap from `fill`. `last` copies lane 3 to every lane.
    #[cfg(target_arch = "aarch64")]
    mod quad {
        use core::arch::aarch64::*;
        pub type V = float32x4_t;
        // SAFETY: AArch64 always has these instructions. The loads and stores leave their bounds to the caller.
        #[inline(always)]
        pub fn splat(x: f32) -> V { unsafe { vdupq_n_f32(x) } }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> V { unsafe { vld1q_f32(p) } }
        #[inline(always)]
        pub unsafe fn store(p: *mut f32, v: V) { unsafe { vst1q_f32(p, v) } }
        #[inline(always)]
        pub fn add(a: V, b: V) -> V { unsafe { vaddq_f32(a, b) } }
        #[inline(always)]
        pub fn up1(fill: V, v: V) -> V { unsafe { vextq_f32::<3>(fill, v) } }
        #[inline(always)]
        pub fn up2(fill: V, v: V) -> V { unsafe { vextq_f32::<2>(fill, v) } }
        #[inline(always)]
        pub fn last(v: V) -> V { unsafe { vdupq_laneq_f32::<3>(v) } }
        #[inline(always)]
        pub fn first(v: V) -> f32 { unsafe { vgetq_lane_f32::<0>(v) } }
    }
    #[cfg(target_arch = "wasm32")]
    mod quad {
        use core::arch::wasm32::*;
        pub type V = v128;
        #[inline(always)]
        pub fn splat(x: f32) -> V { f32x4_splat(x) }
        #[inline(always)]
        pub unsafe fn load(p: *const f32) -> V { unsafe { v128_load(p.cast()) } }
        #[inline(always)]
        pub unsafe fn store(p: *mut f32, v: V) { unsafe { v128_store(p.cast(), v) } }
        #[inline(always)]
        pub fn add(a: V, b: V) -> V { f32x4_add(a, b) }
        #[inline(always)]
        pub fn up1(fill: V, v: V) -> V { i32x4_shuffle::<3, 4, 5, 6>(fill, v) }
        #[inline(always)]
        pub fn up2(fill: V, v: V) -> V { i32x4_shuffle::<2, 3, 4, 5>(fill, v) }
        #[inline(always)]
        pub fn last(v: V) -> V { i32x4_shuffle::<3, 3, 3, 3>(v, v) }
        #[inline(always)]
        pub fn first(v: V) -> f32 { f32x4_extract_lane::<0>(v) }
    }
    /// The running sums of `x` after `start`, 16 items at a time. Each vector of four scans within itself in two shifted additions. Each vector then adds the total of everything before it: the running total, plus the sums of the vectors before it in its block. Only one addition in each block waits for the block before.
    fn sum_scan(x: &[f32], start: Option<f32>, out: &mut Vec<f32>) {
        use quad::*;
        appended(out, x.len(), |slots| {
            let ((chunks, rest), (outs, tail)) = (x.as_chunks::<16>(), slots.as_chunks_mut::<16>());
            let (fill, mut c) = (splat(-0.0), splat(start.unwrap_or(-0.0)));
            for (o, x) in outs.iter_mut().zip(chunks) {
                let scanned = |k: usize| {
                    // SAFETY: the load reads four items of one chunk.
                    let v = unsafe { load(x.as_ptr().add(4 * k)) };
                    let v = add(v, up1(fill, v));
                    add(v, up2(fill, v))
                };
                let (p0, p1, p2, p3) = (scanned(0), scanned(1), scanned(2), scanned(3));
                let s0 = last(p0);
                let s01 = add(s0, last(p1));
                let s012 = add(s01, last(p2));
                let o = o.as_mut_ptr().cast::<f32>();
                // SAFETY: each store writes four slots of one chunk.
                unsafe {
                    store(o, add(c, p0));
                    store(o.add(4), add(add(c, s0), p1));
                    store(o.add(8), add(add(c, s01), p2));
                    store(o.add(12), add(add(c, s012), p3));
                }
                c = add(c, add(s012, last(p3)));
            }
            let mut total = first(c);
            for (o, &v) in tail.iter_mut().zip(rest) {
                total += v;
                o.write(total);
            }
        })
    }
    #[cfg(target_arch = "wasm32")]
    lanes! {
        f64: 2, f64x2_splat, v128_load, f64x2_max, f64x2_min, lane(f64x2_extract_lane), fold, grouped;
        f32: 4, f32x4_splat, v128_load, f32x4_max, f32x4_min, lane(f32x4_extract_lane), fold, vector;
    }
}

/// Elsewhere, the folds and scans use `Folds`' defaults. A fold's loop of `Core::maximum` vectorizes, and each step of a scan is one `maxsd` or `minsd` on x86-64.
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
mod plain {
    impl super::Folds for f32 {}
    impl super::Folds for f64 {}
}

/// The sum of 32-bit integers, wrapped to `i64`, and whether it fits. Sums of up to 2^31 of them fit `i64`, which their running results hold.
#[inline(always)]
fn narrow_sum<T: Copy + Into<i64>>(x: &[T]) -> (i64, bool) {
    let total: i128 = x.chunks(1 << 31).map(|c| i128::from(folded(c, 0i64, |s, v| s + v.into(), |a, b| a + b))).sum();
    (total as i64, i64::try_from(total).is_ok())
}

/// The sum of integers of 8 or 16 bits, which always fits `i64`. No block of 2^15 of them overflows an `i32`. Each block sums in `i32`, whose vectors hold twice as many running sums as `i64`'s.
#[inline(always)]
fn small_sum<T: Copy + Into<i32>>(x: &[T]) -> (i64, bool) {
    (x.chunks(1 << 15).map(|c| i64::from(folded(c, 0i32, |s, v| s + v.into(), |a, b| a + b))).sum(), true)
}

/// The sum of 64-bit integers, wrapped, and whether it fits. Each item splits into its high and low 32 bits. Neither half's sum can overflow `i64` within 2^31 items, and both sums vectorize.
#[inline(always)]
fn wide_sum(x: &[i64]) -> (i64, bool) {
    let halves = |c: &[i64]| folded(c, (0i64, 0i64), |(h, l), v| (h + (v >> 32), l + (v & 0xFFFF_FFFF)), |(h, l), (g, m)| (h + g, l + m));
    let total: i128 = x.chunks(1 << 31).map(|c| {
        let (h, l) = halves(c);
        (i128::from(h) << 32) + i128::from(l)
    }).sum();
    (total as i64, i64::try_from(total).is_ok())
}

/// The running sums of `x` from `start` at `i64`, wrapped, and whether every one fits. A sum overflows where its sign differs from both its arguments' signs, as `Overflowing::plus` tests it. The loop gathers those sign bits in one word and tests it once at the end. A bool for each step, as `Overflowing::plus` gives, measured 7% slower.
#[inline(always)]
fn signed_cumsum<T: Copy + Into<i64>>(x: &[T], start: i64, out: &mut Vec<i64>) -> bool {
    appended(out, x.len(), |slots| {
        let (mut total, mut overflow) = (start, 0i64);
        for (o, &v) in slots.iter_mut().zip(x) {
            let v: i64 = v.into();
            let sum = total.wrapping_add(v);
            overflow |= (total ^ sum) & (v ^ sum);
            total = sum;
            o.write(sum);
        }
        overflow >= 0
    })
}

/// Appends the running totals of `x` from `start` by `op`, which gives each total wrapped and whether it fits, and gives whether every total fits. Each step waits only for the wrapped total before it.
#[inline(always)]
fn checked_scan<T: Copy + Into<S>, S: Copy>(x: &[T], start: S, out: &mut Vec<S>, op: impl Fn(S, S) -> (S, bool)) -> bool {
    appended(out, x.len(), |slots| {
        let (mut total, mut ok) = (start, true);
        for (o, &v) in slots.iter_mut().zip(x) {
            let (next, fits) = op(total, v.into());
            (total, ok) = (next, ok & fits);
            o.write(next);
        }
        ok
    })
}

/// [`Sum`] and [`Product`] for each integer type of compact buffers, at `i64`. `$sum` sums a slice. Each step and product wraps through [`Overflowing`], which also tests whether it fits.
macro_rules! integers {
    ($($t:ty: $sum:ident),+) => {$(
        impl Reduction<$t> for Sum {
            type Total = i64;
            #[inline(always)]
            fn unit() -> i64 { 0 }
            #[inline(always)]
            fn step(acc: i64, x: $t) -> i64 { acc.plus(x.into()).0 }
            #[inline(always)]
            fn checked(acc: i64, x: $t) -> (i64, bool) { acc.plus(x.into()) }
            #[inline(always)]
            fn merge(a: i64, b: i64) -> (i64, bool) { a.plus(b) }
            #[inline]
            fn slice(x: &[$t]) -> (i64, bool) { $sum(x) }
            #[inline]
            fn scan(x: &[$t], start: Option<i64>, out: &mut Vec<i64>) -> bool { signed_cumsum(x, start.unwrap_or(0), out) }
        }
        impl Reduction<$t> for Product {
            type Total = i64;
            #[inline(always)]
            fn unit() -> i64 { 1 }
            #[inline(always)]
            fn step(acc: i64, x: $t) -> i64 { acc.times(x.into()).0 }
            #[inline(always)]
            fn checked(acc: i64, x: $t) -> (i64, bool) { acc.times(x.into()) }
            #[inline(always)]
            fn merge(a: i64, b: i64) -> (i64, bool) { a.times(b) }
            #[inline]
            fn slice(x: &[$t]) -> (i64, bool) {
                x.iter().fold((1 as i64, true), |(p, ok), &v| { let (q, fits) = p.times(v.into()); (q, ok & fits) })
            }
            #[inline]
            fn scan(x: &[$t], start: Option<i64>, out: &mut Vec<i64>) -> bool { checked_scan(x, start.unwrap_or(1), out, <i64>::times) }
        }
    )+};
}
integers!(u8: small_sum, i16: small_sum, i32: narrow_sum, i64: wide_sum);

/// [`Max`] and [`Min`] for each integer type and Booleans: `$lowest` and `$highest` are their units, and `$max` and `$min` their steps. Their steps on Booleans are `or` and `and`, as [`Maximum`] and [`Minimum`] compute them.
macro_rules! extremes {
    (@one $r:ident, $t:ty, $unit:expr, $step:expr) => {
        impl Reduction<$t> for $r {
            type Total = $t;
            #[inline(always)]
            fn unit() -> $t { $unit }
            #[inline(always)]
            fn step(acc: $t, x: $t) -> $t { ($step)(acc, x) }
            #[inline(always)]
            fn merge(a: $t, b: $t) -> ($t, bool) { (($step)(a, b), true) }
            #[inline]
            fn slice(x: &[$t]) -> ($t, bool) { (folded(x, $unit, $step, $step), true) }
            #[inline]
            fn scan(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) -> bool {
                records(x, start, out, $step, |b: &[$t], t: $t| b.iter().fold(t, |m, &v| ($step)(m, v)) == t);
                true
            }
        }
    };
    (integers $($t:ty),+) => { $(extremes!($t: <$t>::MIN, <$t>::MAX, Ord::max, Ord::min);)+ };
    ($($t:ty: $lowest:expr, $highest:expr, $max:expr, $min:expr);+) => {$(
        extremes!(@one Max, $t, $lowest, $max);
        extremes!(@one Min, $t, $highest, $min);
    )+};
}
extremes!(integers i16, i32, i64, u8, u16, u32, u64);
/// [`Max`], [`Min`], [`Xor`] and [`Xnor`] on Booleans: `or`, `and`, `!=` and `==`. Each one's scan is the scan of its truth table through [`Logical`].
macro_rules! booleans {
    ($($r:ident: $unit:expr, |$a:ident, $b:ident| $step:expr, $table:expr;)+) => {$(
        impl Reduction<bool> for $r {
            type Total = bool;
            #[inline(always)]
            fn unit() -> bool { $unit }
            #[inline(always)]
            fn step($a: bool, $b: bool) -> bool { $step }
            #[inline(always)]
            fn merge(a: bool, b: bool) -> (bool, bool) { (<Self as Reduction<bool>>::step(a, b), true) }
            #[inline]
            fn slice(x: &[bool]) -> (bool, bool) { (folded(x, $unit, <Self as Reduction<bool>>::step, <Self as Reduction<bool>>::step), true) }
            #[inline]
            fn scan(x: &[bool], start: Option<bool>, out: &mut Vec<bool>) -> bool {
                Logical($table).scan(x, start, out);
                true
            }
        }
    )+};
}
booleans! {
    Max: false, |a, b| a | b, [[false, true], [true, true]];
    Min: true, |a, b| a & b, [[false, false], [false, true]];
    Xor: false, |a, b| a ^ b, [[false, true], [true, false]];
    Xnor: true, |a, b| a == b, [[true, false], [false, true]];
}

/// The running folds of `x` from `start` by `step`, a maximum or minimum. A block of 256 items that `keeps(block, total)` passes repeats the running result, with no step for each item. Other blocks scan as `grouped` does. On random items, almost every block repeats the running result. After four blocks in a row change the running result, blocks scan with no test until one leaves it unchanged. Their lengths start at 256 items and double up to 16,384. Items that rise steadily then pay for four tests and a few calls of `grouped`. A call for every 256 items added a third to their scan. Random items start such a run about one time in 25. After two blocks, they started one about half the time. The compiler vectorizes each block's test because the block's length isn't fixed. A loop of a fixed length unrolls into one item at a time.
pub(crate) fn records<T: Copy + PartialEq>(x: &[T], start: Option<T>, out: &mut Vec<T>, step: impl Fn(T, T) -> T + Copy, keeps: impl Fn(&[T], T) -> bool) {
    let Some(&first) = x.first() else { return };
    let (mut total, mut at, mut changes) = (start.map_or(first, |s| step(s, first)), 0, 0);
    while at < x.len() {
        let test = changes < 4;
        let block = &x[at..x.len().min(at + if test { 256 } else { 256 << (changes - 4).min(6) })];
        at += block.len();
        if test && keeps(block, total) {
            out.resize(out.len() + block.len(), total);
            changes = 0;
        } else {
            grouped(block, Some(total), out, |v| v, step, |t| t, false);
            let last = out[out.len() - 1];
            (total, changes) = (last, if last == total { 0 } else { changes + 1 });
        }
    }
}

/// Whether every item of `block` is less than `total`. Only such a block repeats a float maximum's running result. A tie, a signed zero or a NaN always goes through the step.
#[inline(always)]
pub(crate) fn below<F: PartialOrd + Copy>(block: &[F], total: F) -> bool { block.iter().fold(true, |k, &v| k & (v < total)) }

/// Whether every item of `block` is greater than `total`. Only such a block repeats a float minimum's running result.
#[inline(always)]
pub(crate) fn above<F: PartialOrd + Copy>(block: &[F], total: F) -> bool { block.iter().fold(true, |k, &v| k & (v > total)) }

/// The fewest items in each piece of [`fold_pieces`]. A lane of at most this many items always folds whole.
const PIECE: usize = 1 << 18;
/// The fewest rows in each piece of [`fold_pieces`]. The partial folds of the lanes that a piece shares with its neighbours then hold at most a 256th of its items.
const ROWS: usize = 256;

/// The fewest items in each piece of [`fold_pieces`] for an array whose rows hold `inner` items: at least [`PIECE`] items and [`ROWS`] rows, in whole `CHUNK`s.
fn piece_size(inner: usize) -> usize { PIECE.max(ROWS * inner).next_multiple_of(CHUNK) }

/// The folds by `R` of the lanes along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row: one fold for each of the `outer × inner` lanes, and whether every one fits. Where a lane along the last axis, or a group of `inner` lanes along another axis, fits in one piece of [`fold_pieces`], each lane folds whole. Runs of whole lanes or groups then split across threads as [`parallel::split`] splits a call. Otherwise the array folds as `fold_pieces` folds it.
pub fn along<T: Copy + Sync, R: Reduction<T>>(data: &[T], shape: [usize; 3]) -> (Vec<R::Total>, bool) {
    let [outer, len, inner] = shape;
    let mut totals = vec![R::unit(); outer * inner];
    if len == 0 || totals.is_empty() { return (totals, true); }
    let size = piece_size(inner);
    let fits = if len * inner <= size {
        parallel::split(&mut totals, inner * (size / (len * inner)), &|first, slots| lanes::<T, R>(slots, first, data, shape))
    } else { fold_pieces::<T, R>(&mut totals, shape, &|range, fold| fold(range.start, &data[range])) };
    (totals, fits)
}

/// [`along`], followed by each lane's already-folded right `tail`. Associative reductions keep their vectorized folds and may regroup items across this boundary.
pub fn along_with_tail<T: Copy + Sync, R: Reduction<T>>(data: &[T], dims: [usize; 3], tail: Option<&[R::Total]>) -> (Vec<R::Total>, bool) {
    let (mut totals, mut ok) = along::<T, R>(data, dims);
    if let Some(tail) = tail {
        assert_eq!(tail.len(), totals.len(), "one tail per lane");
        for (total, &tail) in totals.iter_mut().zip(tail) { let (next, fits) = R::merge(*total, tail); *total = next; ok &= fits; }
    }
    (totals, ok)
}

/// Items of an array that [`along_mapped`] reads through a mapping at a time. A fold along an earlier axis steps strips of columns down each block's rows. A block of 4096 items holds too few rows of a 1024-column matrix for that to pay. Larger blocks also cost a strided read's walk less for each item.
const MAPPED: usize = 1 << 16;

/// [`along`] of the array of `shape` that `mapping` lays `data` over, folding axis `axis`, without building that array: its totals, whether they all fit, and the tables that read the totals in the order of the array's other axes, where they come in another order. A reduction combines each lane's items in any order. Where the array reads a contiguous block of `data` with its axes in another order, as [`Mapping::block`] finds, the block folds in its own order, as [`Block::fold`] plans it, and its totals keep the block's order. A caller can then copy them through the tables, or read them through the tables as another view. Otherwise the array is read [`MAPPED`] items at a time through [`expanded_items`], and folds as [`fold_pieces`] folds it.
pub fn along_mapped<T: Element, R: Reduction<T>>(data: &[T], mapping: &Mapping, shape: &[usize], axis: usize) -> (Vec<R::Total>, bool, Option<Vec<Steps>>) {
    if let Some(block) = mapping.block(shape) {
        let (dims, order) = block.fold(Some(axis));
        let (totals, fits) = along::<T, R>(&data[block.range.clone()], dims);
        return (totals, fits, order);
    }
    let dims = Lanes::new(shape, axis).expect("the folded axis is within the shape").dims();
    let mut totals = vec![R::unit(); dims[0] * dims[2]];
    if dims[1] == 0 || totals.is_empty() { return (totals, true, None); }
    let fits = fold_pieces::<T, R>(&mut totals, dims, &|range, fold| {
        let mut buf = Vec::new();
        for at in range.clone().step_by(MAPPED) { fold(at, expanded_items(mapping, data, at, MAPPED.min(range.end - at), &mut buf)); }
    });
    (totals, fits, None)
}

/// The widest window that [`sliding`] folds directly, one pass for each of its rows. Wider windows take van Herk-Gil-Werman.
const DIRECT: usize = 8;

/// The fold of the one item `x`, clearing `ok` where it doesn't fit.
#[inline(always)]
fn one_of<T, R: Reduction<T>>(x: T, ok: &mut bool) -> R::Total { let (t, fits) = R::one(x); *ok &= fits; t }

/// `total` combined with the item `x`, clearing `ok` where it doesn't fit.
#[inline(always)]
fn step_of<T, R: Reduction<T>>(total: R::Total, x: T, ok: &mut bool) -> R::Total { let (t, fits) = R::checked(total, x); *ok &= fits; t }

/// The folds `a` and `b` of neighbouring runs combined, clearing `ok` where the result doesn't fit.
#[inline(always)]
fn merge_of<T, R: Reduction<T>>(a: R::Total, b: R::Total, ok: &mut bool) -> R::Total { let (t, fits) = R::merge(a, b); *ok &= fits; t }

/// Van Herk-Gil-Werman's folds of the windows of `width` rows in `lane`, rows of `inner` items, into `folds`. `suffix` holds the lane's block suffixes, and `row` carries one row's running fold. `ONE` fixes `inner` at 1, which compiles each loop over a row's items to scalar code.
#[inline(always)]
fn hgw<T: Copy, R: Reduction<T>, const ONE: bool>(lane: &[T], folds: &mut [R::Total], suffix: &mut [R::Total], row: &mut [R::Total], width: usize, inner: usize, ok: &mut bool) {
    let inner = if ONE { 1 } else { inner };
    // A local row of one item stays in a register, where a row in memory would put a store and a load in each step's chain.
    let mut one = [R::unit()];
    let row = if ONE { &mut one[..] } else { &mut row[..inner] };
    for (b, block) in lane.chunks(width * inner).enumerate() {
        let (start, last) = (b * width * inner, block.len() - inner);
        // Each row's suffix folds it with the rest of its block, from the block's last row back to its first.
        let s = &mut suffix[start..start + block.len()];
        for j in 0..inner { row[j] = one_of::<T, R>(block[last + j], ok); s[last + j] = row[j]; }
        for k in (0..last).step_by(inner).rev() { for j in 0..inner { row[j] = merge_of::<T, R>(one_of::<T, R>(block[k + j], ok), row[j], ok); s[k + j] = row[j]; } }
        // Each row's prefix folds the block up to it. The first block's whole prefix is the first window.
        for j in 0..inner { row[j] = one_of::<T, R>(block[j], ok); }
        if b == 0 {
            for k in (inner..block.len()).step_by(inner) { for j in 0..inner { row[j] = step_of::<T, R>(row[j], block[k + j], ok); } }
            folds[..inner].copy_from_slice(row);
            continue;
        }
        // In a later block, row `i` ends the window that starts `i + 1` rows into the block before: the fold of that block's suffix there and this prefix.
        let from = start + inner - width * inner;
        let (windows, earlier) = (&mut folds[from..from + block.len()], &suffix[from..from + block.len()]);
        for j in 0..inner { windows[j] = merge_of::<T, R>(earlier[j], row[j], ok); }
        for k in (inner..block.len()).step_by(inner) {
            for j in 0..inner {
                row[j] = step_of::<T, R>(row[j], block[k + j], ok);
                windows[k + j] = merge_of::<T, R>(earlier[k + j], row[j], ok);
            }
        }
        // A window that starts this block is the whole block, which the prefix holds once its last row is in.
        if block.len() == width * inner { folds[start..start + inner].copy_from_slice(row); }
    }
}

/// The folds by `R` of each window of `width` neighbouring rows along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row. The result has shape `[outer, len - width + 1, inner]`, with whether every fold fits. Each fold combines its window's items in order. A window of at most [`DIRECT`] rows folds directly. A wider one takes van Herk-Gil-Werman's method: each lane splits into blocks of `width` rows, and each window is the fold of one block's suffix and the next block's prefix, so an item costs the same whatever the width.
pub fn sliding<T: Copy, R: Reduction<T>>(data: &[T], [outer, len, inner]: [usize; 3], width: usize) -> (Vec<R::Total>, bool) {
    assert_eq!(data.len(), outer * len * inner, "the data must fill its dimensions");
    assert!((1..=len.max(1)).contains(&width), "a window lies within its lane");
    let count = (len + 1).saturating_sub(width);
    if outer * count * inner == 0 { return (Vec::new(), true); }
    let mut out = vec![R::unit(); outer * count * inner];
    let mut ok = true;
    let lanes = data.chunks_exact(len * inner).zip(out.chunks_exact_mut(count * inner));
    if width <= DIRECT {
        for (lane, folds) in lanes {
            for (t, &x) in folds.iter_mut().zip(lane) { *t = one_of::<T, R>(x, &mut ok); }
            for i in 1..width {
                for (t, &x) in folds.iter_mut().zip(&lane[i * inner..]) { *t = step_of::<T, R>(*t, x, &mut ok); }
            }
        }
        return (out, ok);
    }
    let (mut suffix, mut prefix) = (vec![R::unit(); len * inner], vec![R::unit(); inner]);
    for (lane, folds) in lanes {
        if inner == 1 { hgw::<T, R, true>(lane, folds, &mut suffix, &mut prefix, width, inner, &mut ok) }
        else { hgw::<T, R, false>(lane, folds, &mut suffix, &mut prefix, width, inner, &mut ok) }
    }
    (out, ok)
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
        for &v in x { let t = total.map_or_else(|| Self::one(v).0, |t| Self::step(t, v)); out.push(t); total = Some(t); }
        true
    }
}

/// The row of the first item in each window of `width` rows that gives the window's fold by `R`, for a reduction that picks one of its items, as [`Max`] and [`Min`] do. The result has [`sliding`]'s shape, and each row counts from the start of its lane. A window whose items all leave the fold at `R::unit()`, as NaN items do for a float maximum, gives its first row.
pub fn sliding_picks<T: Copy + PartialEq + Send + Sync, R: Reduction<T, Total = T>>(data: &[T], shape: [usize; 3], width: usize) -> Vec<usize> {
    let [_, len, inner] = shape;
    let pairs: Vec<(T, usize)> = data.iter().enumerate().map(|(k, &x)| (x, k / inner.max(1) % len.max(1))).collect();
    sliding::<(T, usize), Pick<R>>(&pairs, shape, width).0.into_iter().map(|(_, row)| row).collect()
}

/// [`sliding_picks`] of the items of `x` at their own type, by [`Max`], or by [`Min`] where `max` is false. `None` for complex numbers and characters, which have no order.
pub fn view_sliding_picks(x: View<'_>, dims: [usize; 3], width: usize, max: bool) -> Option<Vec<usize>> {
    fn picks<T: Copy + PartialEq + Send + Sync>(x: &[T], dims: [usize; 3], width: usize, max: bool) -> Vec<usize> where Max: Reduction<T, Total = T>, Min: Reduction<T, Total = T> {
        if max { sliding_picks::<T, Max>(x, dims, width) } else { sliding_picks::<T, Min>(x, dims, width) }
    }
    Some(match x {
        View::Booleans(x) => picks(x, dims, width, max),
        View::Integers(x) => with_ints!(x, |x| picks(x, dims, width, max)),
        View::Floats(x) => with_floats!(x, |x| picks(x, dims, width, max)),
        _ => return None,
    })
}

/// Folds each lane of `[outer, len, inner]` from the right: `x0 f (x1 f (...))`. A `tail` supplies the already-folded items to the right of each lane, in the input type. Otherwise each lane needs at least two items. `cast` reads a running result as the next step's right argument, as a Boolean comparison result reads as 0 or 1. A failed step gives `None`.
#[inline]
pub fn right_fold<T: Copy, O: Copy>(data: &[T], [outer, len, inner]: [usize; 3], tail: Option<&[T]>, f: impl Fn(T, T) -> Option<O>, cast: impl Fn(O) -> T) -> Option<Vec<O>> {
    assert_eq!(data.len(), outer * len * inner, "the data must fill its dimensions");
    assert!(inner == 0 || if tail.is_some() { len > 0 } else { len >= 2 }, "a right fold needs two items or a tail");
    if let Some(tail) = tail { assert_eq!(tail.len(), outer * inner, "one tail per lane"); }
    let mut result = Vec::with_capacity(outer * inner);
    if outer * inner == 0 { return Some(result); }
    if inner == 1 {
        for (i, lane) in data.chunks_exact(len).enumerate() {
            let (last, first) = match tail { Some(tail) => (tail[i], len - 1), None => (lane[len - 1], len - 2) };
            let mut total = f(lane[first], last)?;
            for &x in lane[..first].iter().rev() { total = f(x, cast(total))?; }
            result.push(total);
        }
        return Some(result);
    }
    for i in 0..outer {
        let row = |j: usize| &data[(i * len + j) * inner..(i * len + j + 1) * inner];
        let (last, first) = match tail {
            Some(tail) => (&tail[i * inner..(i + 1) * inner], len - 1),
            None => (row(len - 1), len - 2),
        };
        let start = result.len();
        for (&x, &y) in row(first).iter().zip(last) { result.push(f(x, y)?); }
        for j in (0..first).rev() {
            for (value, &x) in result[start..].iter_mut().zip(row(j)) { *value = f(x, cast(*value))?; }
        }
    }
    Some(result)
}

/// Folds an array of shape `[outer, len, inner]` by `R` into `totals`, the folds of its lanes along the middle axis, and gives whether every fold fits. `run(range, fold)` produces the array's items in `range` in order, and hands each block of them to `fold` with the position of its first item. The array folds in one piece for each of this thread's [`parallel::threads`], each of at least [`piece_size`] items. On one thread it folds as one piece. A lane that one piece holds folds as [`fold_into`] folds it. A lane that crosses pieces folds its part in each piece, and those folds merge.
pub fn fold_pieces<T: Copy, R: Reduction<T>>(
    totals: &mut [R::Total], dims: [usize; 3], run: &(dyn Fn(Range<usize>, &mut dyn FnMut(usize, &[T])) + Sync)) -> bool {
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
        if u == last {
            for (s, &x) in folds.iter_mut().zip(&part) {
                let (t, f) = R::merge(*s, x);
                (*s, fits) = (t, fits & f);
            }
        } else {
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
    fn fold<T: Copy, R: Reduction<T, Total = S>>(&mut self, dims: [usize; 3], run: &(dyn Fn(Range<usize>, &mut dyn FnMut(usize, &[T])) + Sync)) -> bool {
        let unit = dims[1] * dims[2];
        let (a, b, mut fits) = (self.lo * unit, self.hi * unit, true);
        run(self.range.clone(), &mut |mut start, mut items| {
            while !items.is_empty() {
                let (block, after) = items.split_at((if start < a { a } else if start < b { b } else { usize::MAX } - start).min(items.len()));
                let (folds, at) = match (start < a, start < b) {
                    (true, _) => self.front.as_mut().map(|(u, f)| (&mut f[..], start - *u * unit)).expect("a piece that starts inside a unit has a front fold"),
                    (_, true) => (&mut *self.own, start - a),
                    _ => self.back.as_mut().map(|(u, f)| (&mut f[..], start - *u * unit)).expect("a piece that ends inside a unit has a back fold"),
                };
                fits &= fold_into::<T, R>(folds, dims, at, block);
                (start, items) = (start + block.len(), after);
            }
        });
        fits
    }
}

/// Folds lanes `first..first + totals.len()` along the middle axis of `data`, an array of shape `[_, len, inner]` held row by row, into `totals`, and gives whether every fold fits. Each lane folds whole. A lane along the last axis folds as [`fold_rows`] folds it. Along another axis, the lanes come in whole groups of `inner`, and each group folds its rows as [`step_rows`] folds them.
fn lanes<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], first: usize, data: &[T], [_, len, inner]: [usize; 3]) -> bool {
    let block = &data[first * len..(first + totals.len()) * len];
    if inner == 1 { return fold_rows::<T, R>(block, len, totals); }
    totals.chunks_mut(inner).zip(block.chunks_exact(len * inner)).fold(true, |fits, (folds, group)| fits & step_rows::<T, R>(folds, group, true))
}

/// Steps each of `totals` by the matching item of `row`, and gives whether every fold fits. In a lane's first row, each fold is the fold of its item alone.
#[inline(always)]
fn step_row<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], row: &[T], first: bool) -> bool {
    let row = totals.iter_mut().zip(row);
    if first {
        row.fold(true, |ok, (s, &x)| {
            let (t, f) = R::one(x);
            *s = t;
            ok & f
        })
    } else {
        row.fold(true, |ok, (s, &x)| {
            let (t, f) = R::checked(*s, x);
            *s = t;
            ok & f
        })
    }
}

/// Steps `totals` by every row of `rows`, which hold `totals.len()` items each, and gives whether every fold fits. Where `first` is true, the first row starts each fold, as in [`step_row`]. Narrow rows fold a block of whole rows at a time, into one running total for each item of the block. Each column's running totals then merge.
#[inline(always)]
fn step_rows<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], rows: &[T], first: bool) -> bool {
    match crate::slice::running(size_of::<R::Total>()) {
        8 => rows_by::<T, R, 8>(totals, rows, first), 16 => rows_by::<T, R, 16>(totals, rows, first), _ => rows_by::<T, R, 32>(totals, rows, first)
    }
}

/// `step_rows` with up to `N` running totals. Rows wider than `N` fold in strips of `N` columns, each down every row, where there are at least 4 rows and the totals are narrower than complex numbers. Complex strips measured slower than stepping row by row. Other wide rows, and too few rows to fill two blocks, step one row at a time. A block is `N` items where `inner` divides `N`, and three quarters of `N` for rows of 3.
#[inline(always)]
fn rows_by<T: Copy, R: Reduction<T>, const N: usize>(totals: &mut [R::Total], rows: &[T], first: bool) -> bool {
    let inner = totals.len();
    if inner > N && rows.len() >= 4 * inner && size_of::<R::Total>() < 16 {
        return totals.chunks_mut(N).enumerate().fold(true, |fits, (s, out)| {
            fits & if out.len() == N { strip_by::<T, R, N>(out, rows, inner, s * N, N, first) }
            else { strip_by::<T, R, N>(out, rows, inner, s * N, out.len(), first) }
        });
    }
    if inner > N || rows.len() < 2 * N {
        return rows.chunks_exact(inner).enumerate().fold(true, |fits, (j, row)| fits & step_row::<T, R>(totals, row, first && j == 0));
    }
    match inner {
        _ if N % inner == 0 => blocks_by::<T, R, N>(totals, rows, first, N), 3 => blocks_by::<T, R, N>(totals, rows, first, 3 * (N / 4)),
        _ => blocks_by::<T, R, N>(totals, rows, first, N - N % inner)
    }
}

/// Folds columns `start..start + width` of `rows`, which hold `inner` items each, down every row into `out`, their `width` totals, and gives whether every fold fits. `width` is at most `N`. Where `first` is true, the fold of the strip replaces `out`. A width the compiler knows keeps the running totals in registers.
#[inline(always)]
fn strip_by<T: Copy, R: Reduction<T>, const N: usize>(out: &mut [R::Total], rows: &[T], inner: usize, start: usize, width: usize, first: bool) -> bool {
    let (mut run, mut fits) = ([R::identity(); N], true);
    for row in rows.chunks_exact(inner) {
        let strip = &row[start..start + width];
        for k in 0..width { let (t, f) = R::checked(run[k], strip[k]); (run[k], fits) = (t, fits & f) }
    }
    for k in 0..width {
        let (t, f) = if first { (run[k], true) } else { R::merge(out[k], run[k]) };
        (out[k], fits) = (t, fits & f);
    }
    fits
}

/// `rows_by` in blocks of `width` items, a multiple of `inner` no greater than `N`. A width the compiler knows keeps the running totals in registers.
#[inline(always)]
fn blocks_by<T: Copy, R: Reduction<T>, const N: usize>(totals: &mut [R::Total], rows: &[T], first: bool, width: usize) -> bool {
    let inner = totals.len();
    let (blocks, rest) = rows.split_at(rows.len() - rows.len() % width);
    // `run` can stay in registers through the loop because nothing takes its address, unlike `all`.
    let (mut run, mut fits) = ([R::identity(); N], true);
    for block in blocks.chunks_exact(width) { for k in 0..width { let (t, f) = R::checked(run[k], block[k]); (run[k], fits) = (t, fits & f) } }
    let mut all = run;
    let parts = &mut all[..width];
    for (k, &x) in rest.iter().enumerate() { let (t, f) = R::checked(parts[k], x); (parts[k], fits) = (t, fits & f) }
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
pub fn fold_into<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], [_, len, inner]: [usize; 3], start: usize, block: &[T]) -> bool {
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
        fits &= fold_rows::<T, R>(whole, len, &mut totals[i..i + count]);
        // The start of a lane that a later block finishes.
        if !started.is_empty() {
            let (s, f) = R::slice(started);
            (totals[i + count], fits) = (s, fits & f);
        }
    } else {
        let row = start / inner;
        let (mut i, mut j, mut k) = (row / len, row % len, start % inner);
        while !rest.is_empty() {
            let rows = if k == 0 { (len - j).min(rest.len() / inner) } else { 0 };
            if rows > 0 {
                let (whole, after) = rest.split_at(rows * inner);
                fits &= step_rows::<T, R>(&mut totals[i * inner..(i + 1) * inner], whole, j == 0);
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
fn fold_rows<T: Copy, R: Reduction<T>>(data: &[T], columns: usize, out: &mut [R::Total]) -> bool {
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

/// Appends the running folds by `R` along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`, in the same layout. Each lane's first fold combines `start` where it is given. Gives whether every fold fits. A lane of consecutive items scans by `scan`. Along any other axis, each row's folds step the row before's by `checked`.
pub fn scan_along<T: Copy, R: Reduction<T>>(data: &[T], dims: [usize; 3], start: Option<R::Total>, out: &mut Vec<R::Total>) -> bool {
    rows_scan(data, dims, out, |lane, o| R::scan(lane, start, o), |x| start.map_or_else(|| R::one(x), |s| R::checked(s, x)), R::checked)
}

/// Appends the ordered left scan of each lane of `[outer, len, inner]`. Without a seed, the first item is unchanged. A failed step reports `false`; the remaining output is still initialized.
#[inline]
pub fn left_scan<T: Copy>(data: &[T], dims: [usize; 3], seed: Option<T>, f: impl Fn(T, T) -> Option<T>, out: &mut Vec<T>) -> bool {
    let step = |a, b| f(a, b).map_or((a, false), |v| (v, true));
    let first = |x| seed.map_or((x, true), |s| step(s, x));
    rows_scan(data, dims, out, |lane, out| {
        let (total, ok) = first(lane[0]);
        out.push(total);
        checked_scan(&lane[1..], total, out, step) & ok
    }, first, step)
}

/// Applies a bulk function to adjacent rows along the middle axis of `[outer, len, inner]`. `first` writes or omits each lane's first row; `combine` receives the borrowed previous and next rows. Both append to `out` and report whether their results fit.
#[inline]
pub fn adjacent<T: Copy, O>(data: &[T], [outer, len, inner]: [usize; 3], out: &mut Vec<O>,
    first: impl Fn(&[T], &mut Vec<O>) -> bool, combine: impl Fn(&[T], &[T], &mut Vec<O>) -> bool) -> bool {
    assert_eq!(data.len(), outer * len * inner, "the data must fill its dimensions");
    if data.is_empty() { return true; }
    data.chunks(len * inner).fold(true, |ok, lane| {
        let rest = lane.len() - inner;
        first(&lane[..inner], out) & combine(&lane[..rest], &lane[inner..], out) & ok
    })
}

/// A scan whose result after the first item is Boolean. `both` appends comparisons of a running total with items; `zero` and `one` read the two possible totals in the input type. `starts` supplies each lane's first Boolean result for a seeded scan. Without it, each lane's first result is a placeholder for the caller to replace with the original item.
pub fn comparison_scan<T: Copy>(data: &[T], [_, len, inner]: [usize; 3], starts: Option<&[bool]>, zero: T, one: T,
    both: &dyn Fn(Arg<'_, T>, &[T], &mut Vec<bool>) -> bool) -> (Vec<bool>, bool) {
    let (mut bits, mut low, mut high) = (Vec::with_capacity(data.len()), Vec::new(), Vec::new());
    let (mut heads, mut seconds, mut pair) = (Vec::new(), Vec::new(), Vec::new());
    if data.is_empty() { return (bits, true); }
    let (fixed, mut ok) = (1 + usize::from(starts.is_none()), true);
    let at_both = |xs: &[T], low: &mut Vec<bool>, high: &mut Vec<bool>| {
        low.clear(); high.clear();
        both(Arg::One(zero), xs, low) & both(Arg::One(one), xs, high)
    };
    if inner == 1 && len > 0 && len <= fixed {
        if let Some(starts) = starts { return (starts.to_vec(), true); }
        bits.resize(data.len(), false);
        if len == 2 {
            let (firsts, seconds): (Vec<T>, Vec<T>) = data.chunks_exact(2).map(|p| (p[0], p[1])).unzip();
            ok &= both(Arg::Many(&firsts), &seconds, &mut pair);
            for (r, &v) in bits[1..].iter_mut().step_by(2).zip(&pair) { *r = v; }
        }
        return (bits, ok);
    }
    if len > 0 && inner == 1 {
        let mut total = false;
        for (at, xs) in (0..).step_by(CHUNK).zip(data.chunks(CHUNK)) {
            ok &= at_both(xs, &mut low, &mut high);
            let first = (fixed - 1 + len - at % len) % len;
            let count = xs.len().saturating_sub(first).div_ceil(len);
            let restarts = match starts {
                Some(starts) => &starts[(at + first) / len..][..count],
                None => {
                    heads.clear(); seconds.clear(); pair.clear();
                    if count > 0 {
                        let base = at + first - 1;
                        heads.extend((0..count).map(|i| data[base + i * len]));
                        seconds.extend((0..count).map(|i| data[base + i * len + 1]));
                        ok &= both(Arg::Many(&heads), &seconds, &mut pair);
                    }
                    &pair[..]
                }
            };
            chosen(total, &low, &high, len, first, restarts, &mut bits);
            total = bits[bits.len() - 1];
        }
    } else if len > 0 {
        let rows = (CHUNK / inner).max(1);
        for (first, span) in (0..).step_by(rows).zip(data.chunks(rows * inner)) {
            let span_rows = first..first + span.len() / inner;
            if span_rows.clone().any(|r| r % len >= fixed) { ok &= at_both(span, &mut low, &mut high); }
            pair.clear();
            if starts.is_none() {
                let row = |r: usize| &data[r * inner..(r + 1) * inner];
                let mut second = span_rows.clone().filter(|r| r % len == 1);
                if span.len() == inner { if let Some(r) = second.next() { ok &= both(Arg::Many(row(r - 1)), row(r), &mut pair); } } else {
                    heads.clear(); seconds.clear();
                    for r in second { heads.extend_from_slice(row(r - 1)); seconds.extend_from_slice(row(r)); }
                    ok &= both(Arg::Many(&heads), &seconds, &mut pair);
                }
            }
            let mut pairs = pair.chunks(inner);
            for (j, r) in span_rows.enumerate() {
                match (r % len, starts) {
                    (0, Some(starts)) => bits.extend_from_slice(&starts[r / len * inner..(r / len + 1) * inner]),
                    (0, None) => bits.extend(std::iter::repeat_n(false, inner)),
                    (1, None) => bits.extend_from_slice(pairs.next().expect("a pair for each lane's second row")),
                    _ => {
                        let before = bits.len() - inner;
                        bits.extend_from_within(before..);
                        let (lo, hi) = (&low[j * inner..(j + 1) * inner], &high[j * inner..(j + 1) * inner]);
                        for ((t, &l), &h) in bits[before + inner..].iter_mut().zip(lo).zip(hi) { *t = (*t & h) | (!*t & l); }
                    }
                }
            }
        }
    }
    (bits, ok)
}

/// Appends the running differences along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`, in the same layout: each lane's first row, or `start` less it, and then each later row subtracted from the total before it. `read` gives an item as a total, and `negated` gives it negated. A lane of consecutive items adds its negated items through [`grouped`]'s running sums, as `+\` does, in one pass, with `reorders` passed on. Each caller defines its closures in this module, because a closure from another crate can stay out of line in a build of many codegen units, at a call for each item. `grouped` takes `negated` and `add` by value, because such a build can also leave core's `<&F as Fn>::call` for a reference out of line.
#[inline(always)]
fn differences<T: Copy, S: Copy>(
    data: &[T], dims: [usize; 3], start: Option<S>, out: &mut Vec<S>, read: impl Fn(T) -> S, negated: impl Fn(T) -> S + Copy, add: impl Fn(S, S) -> S + Copy, reorders: bool) {
    let lane = |lane: &[T], o: &mut Vec<S>| {
        let (total, rest) = match start { Some(s) => (s, lane), None => { let first = read(lane[0]); o.push(first); (first, &lane[1..]) } };
        grouped(rest, Some(total), o, negated, add, |t| t, reorders);
        true
    };
    rows_scan(data, dims, out, lane, |x| (start.map_or_else(|| read(x), |s| add(s, negated(x))), true), |t, x| (add(t, negated(x)), true));
}

/// The running differences of items that negate and add as they are, such as floats and complex numbers, through [`differences`].
pub fn difference_scan_along<T: Copy + std::ops::Neg<Output = T> + std::ops::Add<Output = T>>(data: &[T], dims: [usize; 3], start: Option<T>, out: &mut Vec<T>) {
    differences(data, dims, start, out, |x| x, |x: T| -x, |a, b| a + b, false)
}

/// The running differences of [`Term`]s in `C`, which the caller picks to hold every result, through [`differences`]. They wrap at that width, which leaves every result exact.
pub fn difference_scan_as<T: Term, C: Tally>(data: &[T], dims: [usize; 3], start: Option<C>, out: &mut Vec<C>) {
    differences(data, dims, start, out, T::term, |x: T| C::default().subtract(x.term()), C::add, true)
}

/// An item that sums into any [`Tally`]: a Boolean, as 0 or 1, or an integer.
pub trait Term: Copy {
    /// `self` as a term of a sum in `C`. An integer wider than `C` keeps its low bits, which leaves exact every sum that fits `C`.
    fn term<C: Tally>(self) -> C;
    /// The sum of `items` in `C`.
    #[inline]
    fn terms<C: Tally>(items: &[Self]) -> C { items.iter().fold(C::default(), |s, &x| s.add(x.term())) }
}
impl Term for bool {
    #[inline(always)]
    fn term<C: Tally>(self) -> C { C::low(self as i64) }
    /// Each chunk of 192 items counts into byte totals, 16 to a vector. In wasm, a running 64-bit total doesn't vectorise, and 16-bit totals cost about 4 times as much as bytes.
    #[inline]
    fn terms<C: Tally>(items: &[bool]) -> C { items.chunks(192).fold(C::default(), |s, c| s.add(C::low(c.iter().fold(0u8, |n, &b| n + u8::from(b)).into()))) }
}
macro_rules! terms {
    ($($t:ty),+) => {$(
        impl Term for $t {
            #[inline(always)]
            fn term<C: Tally>(self) -> C { C::low(self as i64) }
        }
    )+};
}
terms!(i16, i32, i64, u8);

/// An integer type that sums [`Term`]s, wrapping where a sum doesn't fit: `u8`, `i16`, `i32` or `i64`. A caller picks one that holds every sum it needs. A count of fewer than 256 Booleans fits a byte, which takes an eighth of the memory of an `i64`.
pub trait Tally: Copy + Default + Send + Sync {
    /// `self + y`, wrapping.
    fn add(self, y: Self) -> Self;
    /// `self - y`, wrapping.
    fn subtract(self, y: Self) -> Self;
    /// The low bits of `n`.
    fn low(n: i64) -> Self;
}
macro_rules! tallies {
    ($($t:ty),+) => {$(
        impl Tally for $t {
            #[inline(always)]
            fn add(self, y: $t) -> $t { self.wrapping_add(y) }
            #[inline(always)]
            fn subtract(self, y: $t) -> $t { self.wrapping_sub(y) }
            #[inline(always)]
            fn low(n: i64) -> $t { n as $t }
        }
    )+};
}
tallies!(u8, i16, i32, i64);

/// Sums of [`Term`]s as a [`Reduction`], in the type `C`, which the caller picks to hold every sum, so no step checks whether it fits: counts of the trues among Booleans, or sums of integers.
#[derive(Clone, Copy, Debug)]
pub struct SumAs<C>(std::marker::PhantomData<C>);
impl<T: Term, C: Tally> Reduction<T> for SumAs<C> {
    type Total = C;
    #[inline(always)]
    fn unit() -> C { C::default() }
    #[inline(always)]
    fn step(acc: C, x: T) -> C { acc.add(x.term()) }
    #[inline(always)]
    fn merge(a: C, b: C) -> (C, bool) { (a.add(b), true) }
    #[inline]
    fn slice(x: &[T]) -> (C, bool) { (T::terms(x), true) }
    #[inline]
    fn scan(x: &[T], start: Option<C>, out: &mut Vec<C>) -> bool { grouped(x, start, out, T::term, C::add, |t| t, true); true }
}

/// The right fold of a slice by the inverse of a reduction, through pairs of items: `-` for [`Sum`], and `÷` for [`Product`]. The fold `x0 - (x1 - (x2 - …))` is the sum of the pairs' differences, and `x0 ÷ (x1 ÷ (x2 ÷ …))` is the product of the pairs' ratios, each with the last item where it has no pair. Both combine the pairs in any order, as sums and products do.
pub trait Alternating<T> {
    /// The type of the fold: the item type, or `i64` for integers.
    type Total;
    /// The right fold of `x` by the inverse, 0 or 1 for an empty slice, and whether it fits. An integer fold fits where it is within `i64`. A quotient of floats doesn't fit where a running product or the result reaches zero, a subnormal number, an infinity or NaN. The right fold taken item by item may still be finite there.
    fn alternating(x: &[T]) -> (Self::Total, bool);
}

/// Alternating folds of consecutive lanes, retaining [`Alternating`]'s slice kernels. Gives `None` where a lane's result doesn't fit.
#[inline]
pub fn alternating_lanes<T: Copy, R: Alternating<T>>(data: &[T], len: usize) -> Option<Vec<R::Total>> {
    data.chunks(len.max(1)).map(|lane| match R::alternating(lane) { (total, true) => Some(total), _ => None }).collect()
}

/// The fold by `merge` from `unit` of the pairs of `x`, each combined into a total by `pair`, and then of the item left over where `x` has an odd length, combined by `last`.
#[inline(always)]
fn pairwise<A: Copy, S: Copy>(x: &[A], unit: S, pair: impl Fn(S, A, A) -> S, merge: impl Fn(S, S) -> S, last: impl Fn(S, A) -> S) -> S {
    let (pairs, rest) = x.as_chunks::<2>();
    let total = folded(pairs, unit, |s, [a, b]| pair(s, a, b), merge);
    rest.first().map_or(total, |&z| last(total, z))
}

/// The product of the pairs' ratios of `x` by `divide` and `times`, with whether every running product and the result stay inside the normal range. A product that falls below that range stays out of it, or comes back with fewer significant bits, so each step keeps the least `size` of the running products. A product that overflows stays infinite or turns NaN, and a NaN stays NaN, so `finite` catches both in the result.
#[inline(always)]
fn quotient<T: Copy, R: Native>(
    x: &[T], one: T, divide: impl Fn(T, T) -> T, times: impl Fn(T, T) -> T + Copy, size: impl Fn(T) -> R + Copy, finite: impl Fn(T) -> bool) -> (T, bool) {
    let step = move |(p, low): (T, R), q: T| { let r = times(p, q); (r, low.min(size(r))) };
    let (total, low) = pairwise(x, (one, R::one()), |s, a, b| step(s, divide(a, b)), move |s, (q, l)| {
        let (r, m) = step(s, q);
        (r, m.min(l))
    }, step);
    (total, low >= R::min_positive_value() && finite(total))
}

/// The sum by `add` and `merge` of the items of `x` at even positions, less the sum of those at odd positions. There is an even number of running results, so each one takes the items of one parity.
#[inline(always)]
fn alternated<A: Copy, S: Copy + std::ops::Sub<Output = S>>(x: &[A], unit: S, add: impl Fn(S, A) -> S + Copy, merge: impl Fn(S, S) -> S) -> S {
    running_results(x, unit, add, move |parts, rest| {
        let ((pairs, _), (tail, last)) = (parts.as_chunks::<2>(), rest.as_chunks::<2>());
        let (even, odd) = pairs.iter().fold((unit, unit), |(e, o), &[p, q]| (merge(e, p), merge(o, q)));
        let (even, odd) = tail.iter().fold((even, odd), |(e, o), &[p, q]| (add(e, p), add(o, q)));
        last.first().map_or(even, |&z| add(even, z)) - odd
    })
}

impl<F: Native> Alternating<F> for Sum {
    type Total = F;
    #[inline]
    fn alternating(x: &[F]) -> (F, bool) { (pairwise(x, F::zero(), |s, a, b| s.algebraic_add(a - b), F::algebraic_add, |s, z| s + z), true) }
}

impl<F: Native> Alternating<Complex<F>> for Sum {
    type Total = Complex<F>;
    #[inline]
    fn alternating(x: &[Complex<F>]) -> (Complex<F>, bool) {
        let add = <Sum as Reduction<Complex<F>>>::step;
        (pairwise(x, Complex::new(F::zero(), F::zero()), |s, a, b| add(s, a - b), add, |s, z| s + z), true)
    }
}

/// [`Sum`]'s alternating folds of integers: the sum of the items at even positions less the sum of those at odd positions, which is exact. Each run of `$run` items, an even number, sums at `$acc`, where it can't overflow, and the runs total at `i128`. Bytes and 16-bit integers sum at `i32`, which fits twice as many in each vector as `i64`. Separate running sums for each parity take less time than `pairwise`, which must split each pair out of the vector lanes.
macro_rules! alternating_integers {
    ($($t:ty => $acc:ty, $run:expr);+) => {$(
        impl Alternating<$t> for Sum {
            type Total = i64;
            #[inline]
            fn alternating(x: &[$t]) -> (i64, bool) {
                let run = |c: &[$t]| i128::from(alternated(c, 0 as $acc, |s, v| s + <$acc>::from(v), |s, t| s + t));
                let total: i128 = x.chunks($run).map(run).sum();
                (total as i64, i64::try_from(total).is_ok())
            }
        }
    )+};
}
alternating_integers!(i16 => i32, 1 << 15; u8 => i32, 1 << 15; i32 => i64, 1 << 31; i64 => i128, 1 << 31);

impl<F: Native> Alternating<F> for Product {
    type Total = F;
    #[inline]
    fn alternating(x: &[F]) -> (F, bool) { quotient(x, F::one(), |a, b| a / b, F::algebraic_mul, F::abs, F::is_finite) }
}

impl<F: Native> Alternating<Complex<F>> for Product {
    type Total = Complex<F>;
    /// The pairs' ratios divide as [`crate::divide`] does.
    #[inline]
    fn alternating(x: &[Complex<F>]) -> (Complex<F>, bool) {
        let times = |p: Complex<F>, q: Complex<F>| {
            Complex::new(p.re.algebraic_mul(q.re).algebraic_add(-p.im.algebraic_mul(q.im)), p.re.algebraic_mul(q.im).algebraic_add(p.im.algebraic_mul(q.re)))
        };
        quotient(x, Complex::new(F::one(), F::zero()), crate::divide, times, |p| p.re.abs().max(p.im.abs()), |p| p.re.is_finite() & p.im.is_finite())
    }
}

/// The length below which a lane of consecutive items runs through `rows_scan`'s own loop, [`LANES`] lanes side by side, because a slice form costs more to start than such a lane's items do.
const SHORT_LANE: usize = 64;

/// How many lanes of fewer than [`SHORT_LANE`] items `rows_scan` runs side by side. Each lane's running result waits on the step before it. Run one lane at a time, a loop keeps one chain in flight and leaves any overlap to the processor running ahead into the next lane, which changes with where the code sits. Side by side, `LANES` chains are always in flight.
const LANES: usize = 8;

/// Runs `first` and `step` along the lanes of `len` items in `data`, `B` lanes side by side, writing the running results to `slots`, and gives whether every result fits. `data` holds a whole number of groups of `B` lanes.
#[inline(always)]
fn side_by_side<const B: usize, T: Copy, S: Copy>(
    data: &[T], slots: &mut [std::mem::MaybeUninit<S>], len: usize, first: impl Fn(T) -> (S, bool), step: impl Fn(S, T) -> (S, bool)) -> bool {
    let mut fits = true;
    for (group, outs) in data.chunks_exact(B * len).zip(slots.chunks_exact_mut(B * len)) {
        let mut ok = [true; B];
        let mut t: [S; B] = std::array::from_fn(|b| { let (t, f) = first(group[b * len]); ok[b] = f; outs[b * len].write(t); t });
        for i in 1..len {
            for b in 0..B {
                // SAFETY: `group` and `outs` hold `B * len` items, and `b < B` and `i < len`.
                let (s, f) = step(t[b], unsafe { *group.get_unchecked(b * len + i) });
                (t[b], ok[b]) = (s, ok[b] & f);
                unsafe { outs.get_unchecked_mut(b * len + i) }.write(s);
            }
        }
        fits &= ok.into_iter().all(|f| f);
    }
    fits
}

/// Appends the running results along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`, in the same layout, and gives whether every one fits. A lane of at least [`SHORT_LANE`] consecutive items runs through `lane`. Shorter lanes run [`LANES`] side by side, and lanes along any other axis run a row at a time: `first` gives each lane's first item or row, and `step` steps each from the one before. `first` and `step` go to `side_by_side` by value, for the reason that [`differences`] gives.
#[inline(always)]
fn rows_scan<T: Copy, S: Copy>(
    data: &[T], [_, len, inner]: [usize; 3], out: &mut Vec<S>, lane: impl Fn(&[T], &mut Vec<S>) -> bool, first: impl Fn(T) -> (S, bool) + Copy,
    step: impl Fn(S, T) -> (S, bool) + Copy) -> bool {
    if data.is_empty() { return true; }
    if inner == 1 && len >= SHORT_LANE { return data.chunks(len).fold(true, |fits, l| lane(l, out) & fits); }
    if inner == 1 {
        return appended(out, data.len(), |slots| {
            let whole = data.len() / (LANES * len) * (LANES * len);
            let (groups, rest) = slots.split_at_mut(whole);
            side_by_side::<LANES, _, _>(&data[..whole], groups, len, first, step) & side_by_side::<1, _, _>(&data[whole..], rest, len, first, step)
        });
    }
    // Each row steps from the row before it, which `out` already holds. Indexed loops, because a `Zip` or an `extend_from_slice` for each row can stay out of line in a build of many codegen units.
    appended(out, data.len(), |slots| {
        let mut fits = true;
        let mut at = 0;
        while at < data.len() {
            let (row, outs) = (&data[at..at + inner], &mut slots[at..at + inner]);
            for k in 0..inner {
                let (t, f) = first(row[k]);
                outs[k].write(t);
                fits &= f;
            }
            let mut r = at + inner;
            while r < at + len * inner {
                let (done, rest) = slots.split_at_mut(r);
                let (before, row, outs) = (written(&mut done[r - inner..]), &data[r..r + inner], &mut rest[..inner]);
                for k in 0..inner {
                    let (t, f) = step(before[k], row[k]);
                    outs[k].write(t);
                    fits &= f;
                }
                r += inner;
            }
            at += len * inner;
        }
        fits
    })
}

/// A function of two Booleans, held as its truth table: `Logical(t)` gives `t[r][x]` for the running result `r` and the next item `x`. Its scans run any such function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Logical(pub [[bool; 2]; 2]);

impl Logical {
    /// `r f x`, with no branch.
    #[inline(always)]
    fn apply(self, r: bool, x: bool) -> bool { let [[f00, f01], [f10, f11]] = self.0; (r & ((x & f11) | (!x & f10))) | (!r & ((x & f01) | (!x & f00))) }

    /// Appends the running results of `x` to `out`: the first item, or `start` combined with it where `start` is given, then each result combined with the next item.
    ///
    /// For each value of the next item, the function maps the running result to itself, to its negation, or to a constant. Where one value maps it to itself and the other to a constant, as for `and` and `or`, the results change at most once. `first_where` finds that place, and two runs fill the rest. Otherwise each map is `r ↦ (s ∧ r) ≠ t` for two bits `s` and `t`, and the maps compose as `r ↦ (s₂ ∧ s₁ ∧ r) ≠ ((s₂ ∧ t₁) ≠ t₂)`. Each block of 64 items packs into a word of bits, whose maps compose in six steps of shifts, with no branch and no step for each item.
    pub fn scan(self, x: &[bool], start: Option<bool>, out: &mut Vec<bool>) {
        let Some((&head, rest)) = x.split_first() else { return };
        let first = self.first(start, head);
        out.push(first);
        // `maps[v]` gives the results of an item `v` for a running result of false and of true.
        let [[f00, f01], [f10, f11]] = self.0;
        let maps = [[f00, f10], [f01, f11]];
        for v in [false, true] {
            let [lo, hi] = maps[usize::from(!v)];
            if maps[usize::from(v)] == [false, true] && lo == hi {
                let at = first_where(rest, |x| x != v);
                out.extend(std::iter::repeat_n(first, at));
                out.extend(std::iter::repeat_n(lo, rest.len() - at));
                return;
            }
        }
        let bit_maps = self.bit_maps();
        mapped(rest.len(), first, |start| bit_maps(packed(rest, start)), out);
    }

    /// A lane's first result: its first item `x`, combined with `start` where it's given.
    #[inline(always)]
    fn first(self, start: Option<bool>, x: bool) -> bool { start.map_or(x, |s| self.apply(s, x)) }

    /// The maps of 64 items given as the bits of a word, as [`mapped`] takes them: item `k` maps `r` to `(s ∧ r) ≠ t` for bit `k` of each word of `(s, t)`.
    #[inline(always)]
    fn bit_maps(self) -> impl Fn(u64) -> (u64, u64) + Copy {
        let [[f00, f01], [f10, f11]] = self.0;
        let word = |b: bool| if b { u64::MAX } else { 0 };
        // An item `v` maps a result of false to `f0v` and a result of true to `f1v`.
        let (s0, t0, s1, t1) = (word(f00 != f10), word(f00), word(f01 != f11), word(f01));
        move |bits| ((bits & s1) | (!bits & s0), (bits & t1) | (!bits & t0))
    }

    /// Appends the running results along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`, in the same layout, as [`scan_along`] does for a [`Reduction`]. Each lane's first result combines `start` where it is given.
    pub fn scan_along(self, data: &[bool], dims: [usize; 3], start: Option<bool>, out: &mut Vec<bool>) {
        let [_, len, inner] = dims;
        if inner == 1 && (1..=64).contains(&len) { return self.short_lanes(data, len, start, out); }
        rows_scan(data, dims, out, |l, o| {
            self.scan(l, start, o);
            true
        }, |x| (self.first(start, x), true), |r, x| (self.apply(r, x), true));
    }

    /// [`Logical::scan_along`] on lanes of at most 64 consecutive items, in one [`mapped`] call. Each word restarts the lanes that start in it with constant maps, at places that shift by `64 % len` from one word to the next. A lane's first result depends only on its first item, so it comes from the same bits.
    fn short_lanes(self, data: &[bool], len: usize, start: Option<bool>, out: &mut Vec<bool>) {
        let (bit_maps, word) = (self.bit_maps(), |b: bool| if b { u64::MAX } else { 0 });
        let (yes, no) = (word(self.first(start, true)), word(self.first(start, false)));
        let starts = (0..64).step_by(len).fold(0u64, |p, k| p | 1 << k);
        let (shift, at) = (64 % len, std::cell::Cell::new(0));
        mapped(data.len(), false, |from| {
            let (bits, k) = (packed(data, from), at.get());
            at.set(if k >= shift { k - shift } else { k + len - shift });
            let (places, (s, t)) = (starts << k, bit_maps(bits));
            (s & !places, (t & !places) | ((bits & yes) | (!bits & no)) & places)
        }, out);
    }
}

/// The 64 Booleans of `block` as the bits of a word, the first in the lowest bit. Each 8 Booleans, one byte each, gather into a byte through one multiplication.
#[inline(always)]
fn pack(block: &[bool; 64]) -> u64 {
    let mut bits = 0;
    for (k, b) in block.as_chunks::<8>().0.iter().enumerate() {
        let bytes = (0..8).fold(0u64, |w, i| w | u64::from(b[i]) << (8 * i));
        bits |= (bytes.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
    }
    bits
}

/// Writes the 64 bits of `r` to `out` as Booleans, the lowest bit first. Each byte of `r` becomes 8 bytes of 0 or 1, bit `j` in byte `j`, through one multiplication, two masks, an add and a shift that's the same for every byte. LLVM compiles a test of bit `j` in byte `j` to a different shift for each byte, which wasm's SIMD can only emulate.
#[inline(always)]
fn unpack(r: u64, out: &mut [std::mem::MaybeUninit<bool>; 64]) {
    for (k, lanes) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let bits = (r >> (8 * k) & 0xff).wrapping_mul(0x0101_0101_0101_0101) & 0x8040_2010_0804_0201;
        let bytes = ((bits + 0x7f7f_7f7f_7f7f_7f7f) >> 7 & 0x0101_0101_0101_0101).to_le_bytes();
        *lanes = std::array::from_fn(|j| std::mem::MaybeUninit::new(bytes[j] != 0));
    }
}

/// The results of 64 items, bit by bit, from the running result `carry`, where item `k`'s map is `r ↦ (s ∧ r) ≠ t` for bit `k` of `s` and `t`. Each step composes each bit's map with the map it holds from the bits before it. A bit with none before it takes the identity map, whose `s` is 1 and whose `t` is 0.
#[inline(always)]
fn composed(mut s: u64, mut t: u64, carry: bool) -> u64 {
    for k in [1, 2, 4, 8, 16, 32] {
        t ^= s & (t << k);
        s &= (s << k) | ((1u64 << k) - 1);
    }
    (s & if carry { u64::MAX } else { 0 }) ^ t
}

/// The Booleans of `x` from `start`, at most 64, as the bits of a word, the first in the lowest bit. Bits past the end of `x` are 0.
#[inline(always)]
fn packed(x: &[bool], start: usize) -> u64 {
    match x[start..].first_chunk::<64>() {
        Some(block) => pack(block),
        None => {
            let mut block = [false; 64];
            block[..x.len() - start].copy_from_slice(&x[start..]);
            pack(&block)
        }
    }
}

/// Appends the running results of `n` items to `out`, from the running result `carry` before the first. `maps(start)` gives the maps of the 64 items from `start` as two words of bits `(s, t)`: item `start + k` maps `r` to `(s ∧ r) ≠ t` for bit `k` of each. Bits past the last item are ignored.
#[inline(always)]
fn mapped(n: usize, mut carry: bool, maps: impl Fn(usize) -> (u64, u64), out: &mut Vec<bool>) {
    appended(out, n, |slots| {
        let (outs, last) = slots.as_chunks_mut::<64>();
        for (i, o) in outs.iter_mut().enumerate() {
            let (s, t) = maps(64 * i);
            let r = composed(s, t, carry);
            unpack(r, o);
            carry = r >> 63 != 0;
        }
        if !last.is_empty() {
            let (s, t) = maps(n - last.len());
            let r = composed(s, t, carry);
            for (k, slot) in last.iter_mut().enumerate() { slot.write(r >> k & 1 != 0); }
        }
    });
}

/// Appends the running results of a scan whose running result is a Boolean, from `carry` before the first item. Item `k`'s result is `high[k]` where the result before it is true, and `low[k]` where it's false. `low` and `high` have the same length. Item `first + i·len` takes `fixed[i]` whatever the result before it, which restarts a lane of `len` items. An empty `fixed` restarts none. The results compose 64 at a time, with no branch for each item, as [`Logical::scan`]'s do.
pub fn chosen(carry: bool, low: &[bool], high: &[bool], len: usize, first: usize, fixed: &[bool], out: &mut Vec<bool>) {
    let picks = |start| { let (l, h) = (packed(low, start), packed(high, start)); (l ^ h, l) };
    // Most runs of long lanes hold no restart, and skip `restarts`.
    if fixed.is_empty() { return mapped(low.len(), carry, picks, out); }
    let next = std::cell::Cell::new(0);
    mapped(low.len(), carry, |start| {
        let ((s, t), (places, values)) = (picks(start), restarts(start, len, first, fixed, &next));
        (s & !places, (t & !places) | values)
    }, out);
}

/// The items among the 64 from `start` that are items `first + i·len`, as bits, and `fixed[i]` at the same bits. A restarted item's map is the constant `fixed[i]`. `next` holds the first `i` not yet placed. [`mapped`] asks for its words in order, so each call starts where the one before stopped, with no division.
#[inline(always)]
fn restarts(start: usize, len: usize, first: usize, fixed: &[bool], next: &std::cell::Cell<usize>) -> (u64, u64) {
    let (mut places, mut values, mut i) = (0, 0, next.get());
    while let Some(&v) = fixed.get(i) {
        let k = first + i * len - start;
        if k >= 64 { break; }
        places |= 1 << k;
        values |= u64::from(v) << k;
        i += 1;
    }
    next.set(i);
    (places, values)
}
