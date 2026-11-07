//! Reductions and scans over a contiguous slice: sums, products, maximums and minimums, and their running versions. Each compiles inside rustymath once for each item type. A float sum or product may combine the items in any order, which lets its loop vectorize, so its last bits can differ from a sum taken item by item. Beside NaN, a float maximum or minimum gives NaN or the greatest or least of the items that aren't NaN, and never any other number. Which of the two it gives can depend on the target and the loop. An empty slice gives -∞ or ∞. `f16` items compute in `f32`. Their sum or product is an `f32` total. An integer sum or product computes at `i64`, or at `u64` for `u64` items. Where it doesn't fit, it wraps, and reports that it doesn't fit, as the running versions do.
use crate::error::{Error, Kind};
pub use crate::slice::running;
use crate::{
    Arg, Float, Native, agreement::{Mapping, Read, Walked, expanded_items}, arith::Overflowing, buffer::{Buffer, Cast, Element, Source, View},
    float::sealed::{Core, Folds}, gather::Steps, layout::Lanes, linalg::Scalar, operation::{Piece as InputPiece, Pieces, Seed}, parallel,
    slice::{CHUNK, appended, written}
};
use half::f16;
use num_complex::Complex;
use std::ops::Range;

/// The sum of `x`, 0 for an empty slice, and whether it fits. An integer sum that doesn't fit wraps.
#[inline]
pub fn sum<T: Copy>(x: &[T]) -> (<Sum as Reduction<T>>::Total, bool) where Sum: Reduction<T> { Sum::slice(x) }
/// The product of `x`, 1 for an empty slice, and whether it fits. An integer product that doesn't fit wraps.
#[inline]
pub fn prod<T: Copy>(x: &[T]) -> (<Product as Reduction<T>>::Total, bool) where Product: Reduction<T> { Product::slice(x) }
/// The greatest item of `x`.
#[inline]
pub fn max<T: Copy>(x: &[T]) -> T where Max: Reduction<T, Total = T> { Max::slice(x).0 }
/// The least item of `x`.
#[inline]
pub fn min<T: Copy>(x: &[T]) -> T where Min: Reduction<T, Total = T> { Min::slice(x).0 }
/// The least and the greatest item of `x`, in one pass of comparisons and selects. For floats, [`min`] and [`max`] take less time together, because their folds run AArch64's `fmin` and `fmax`.
#[inline]
pub fn extremes<T: Copy>(x: &[T]) -> (T, T) where Min: Reduction<T, Total = T>, Max: Reduction<T, Total = T> { extremes_of(x, |v| v) }

/// The least and the greatest `key` of the items of `x`, in the pass [`extremes`] makes. A `key` that only reads an item as another type, as `char as u32` does, costs nothing.
#[inline]
pub(crate) fn extremes_of<A: Copy, T: Copy>(x: &[A], key: impl Fn(A) -> T) -> (T, T) where Min: Reduction<T, Total = T>, Max: Reduction<T, Total = T> {
    let (min, max) = (<Min as Reduction<T>>::step, <Max as Reduction<T>>::step);
    x.iter().fold((<Min as Reduction<T>>::unit(), <Max as Reduction<T>>::unit()), |(lo, hi), &a| {
        let v = key(a);
        (min(lo, v), max(hi, v))
    })
}

/// The greater of x and y, as [`max`] gives the greatest item. Beside NaN, or where they're equal, it gives y. A fold that passes its running result as y keeps it beside a NaN item. NumPy calls it `maximum`.
#[inline(always)]
pub fn maximum<F: Float>(x: F, y: F) -> F { F::from_working(Core::maximum(x.working(), y.working())) }

/// The lesser of x and y, as [`maximum`] gives the greater. NumPy calls it `minimum`.
#[inline(always)]
pub fn minimum<F: Float>(x: F, y: F) -> F { F::from_working(Core::minimum(x.working(), y.working())) }

/// [`maximum`] as a function object, which also takes integers and Booleans. The maximum of two Booleans is their `or`.
#[derive(Clone, Copy, Debug)]
pub struct Maximum;
forms!(#[inline] Maximum at real: |_, [x, y]| crate::slice::Total(maximum(x, y)));
forms!(#[inline] Maximum at u8, i16, i32, i64: |_, [x, y]| x.max(y));
forms!(#[inline] Maximum at bool: |_, [x, y]| x | y);

/// [`minimum`] as a function object, which also takes integers and Booleans. The minimum of two Booleans is their `and`.
#[derive(Clone, Copy, Debug)]
pub struct Minimum;
forms!(#[inline] Minimum at real: |_, [x, y]| crate::slice::Total(minimum(x, y)));
forms!(#[inline] Minimum at u8, i16, i32, i64: |_, [x, y]| x.min(y));
forms!(#[inline] Minimum at bool: |_, [x, y]| x & y);

/// A reduction by [`Sum`], [`Product`], [`ScaledProduct`], [`Max`], [`Min`], [`And`], [`Or`], [`Xor`], [`Xnor`], [`SumAs`], [`Squares`] or [`Alternated`], implemented inside rustymath once for each item type it takes. `slice` folds a whole slice of consecutive items, in any order, `merge` combines the folds of two neighbouring runs, and `scan` gives the running folds. [`along`], [`fold_into`] and [`scan_along`] fold along an axis whose items aren't consecutive. [`Sum`] and [`Product`] take floats, integers and complex numbers. [`Max`] and [`Min`] take floats, integers and Booleans, [`And`], [`Or`] and [`Xor`] take integers and Booleans, [`Xnor`] takes signed integers, bytes and Booleans, [`SumAs`] takes Booleans and integers, [`Squares`] takes floats and complex numbers, [`ScaledProduct`] takes floats, `Alternated<Sum>` takes floats, integers and complex numbers, `Alternated<ScaledProduct>` takes floats, and `Alternated<Product>` takes complex numbers. An `f16` reduction computes in `f32`. A sum or product of `f16`s gives an `f32` total, for its caller to round once. A total that doesn't fit its type fails, with its value wrapped to that type.
pub trait Reduction<T> {
    /// The type of a fold: the item type, `i64` for a sum or product of integers narrower than 64 bits, `f32` for a sum or product of `f16`s, `i16` for an XNOR of bytes, the `C` of [`SumAs`], `f64` for [`Squares`], a [`Scaled`] product for [`ScaledProduct`], or a total with the parity of its count for [`Alternated`].
    type Total: Copy + Send + Sync;
    /// Whether disjoint subsequences may be merged in any order.
    const COMMUTATIVE: bool = false;
    /// The fold of no items: 0, 1, -∞ or ∞, the integer type's least or greatest value, all ones, or false or true.
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
    /// `merge`, for a chain of folds whose every step waits for the one before, as van Herk-Gil-Werman's suffixes and prefixes do. As in `merge`, `a` folds the earlier items. Float maximums and minimums override it with `Folds::step_max` and `step_min`, which use AArch64's one-instruction `fmaxnm` and `fminnm`. LLVM can't vectorize those, so they stay out of `merge`.
    #[inline(always)]
    fn chain(a: Self::Total, b: Self::Total) -> (Self::Total, bool) { Self::merge(a, b) }
    /// The fold of `x`, which is `unit()` where `x` is empty, and whether it fits. The default takes each item in turn by `checked`.
    fn slice(x: &[T]) -> (Self::Total, bool) where Self: Sized, T: Copy {
        if x.is_empty() { return (Self::unit(), true); }
        x.iter().fold((Self::identity(), true), |(t, fits), &v| { let (t, ok) = Self::checked(t, v); (t, fits & ok) })
    }
    /// Appends the running folds of `x` to `out`, the first combining `start` where it is given, and gives whether every one fits. The default takes each item in turn by `checked` (`checked_steps`).
    fn scan(x: &[T], start: Option<Self::Total>, out: &mut Vec<Self::Total>) -> bool where Self: Sized, T: Copy { checked_steps::<T, Self>(x, start, out) }
    /// Folds each row of `data`, a matrix with `columns` items in each row, into the matching item of `out`, and gives whether every fold fits. The default is `fold_rows`, which folds each row of 16 items or more by `slice`. Float sums and products, integer maximums and minimums, and bitwise folds fold rows of 16 to 63 items by `checked` instead, eight rows side by side (`rows_side_by_side`). Each of their steps is one instruction, so keeping several rows' steps in flight costs less than a call of `slice` for each row. Other reductions keep `fold_rows`, where `slice` vectorizes within the row: an integer sum's step checks for overflow, and a float maximum's or minimum's `slice` runs `Folds`' vector kernels. Nobody has timed float maximums side by side since their step became one select on x86.
    #[inline]
    fn rows(data: &[T], columns: usize, out: &mut [Self::Total]) -> bool where Self: Sized, T: Copy { fold_rows::<T, Self>(data, columns, out) }
    /// Appends the running folds of `data`'s lanes of `len` consecutive items to `out`, each lane from `start`, through a route that runs many short lanes at once, and gives whether every fold fits. A reduction without such a route gives `None` and appends nothing, as the default does. [`scan_along`] then runs the lanes itself. The Boolean reductions run lanes of 1 to 64 items through [`Logical`], 64 items to a word.
    #[inline]
    fn short_lanes(_data: &[T], _len: usize, _start: Option<Self::Total>, _out: &mut Vec<Self::Total>) -> Option<bool> where Self: Sized { None }
    /// The most classes for which [`crate::search::ClassNumbers`] folds each class into four totals by position. Each step of a class's total waits on the last through memory, so a reduction whose step takes longer gains from the split over more classes.
    const SPLIT: usize = 8;
}

/// A fold's total as an item of a buffer: an [`Element`] stores itself, a [`Scaled`] product stores its value, and [`Alternated`]'s total stores its fold without the parity of its count.
pub trait Totals: Copy {
    /// The buffer that holds `totals`.
    fn buffer_of(totals: Vec<Self>) -> crate::buffer::Buffer;
    fn from_partial(_: crate::operation::Total) -> Result<Self, Error> { Err(Error::new(Kind::Domain, "this reduction cannot read compiled partial totals")) }
}
impl<T: crate::buffer::Cast> Totals for T {
    #[inline]
    fn buffer_of(totals: Vec<T>) -> crate::buffer::Buffer { T::buffer(totals) }
    fn from_partial(total: crate::operation::Total) -> Result<Self, Error> { total.value() }
}
impl<S: Totals> Totals for (S, bool) {
    #[inline]
    fn buffer_of(totals: Vec<(S, bool)>) -> crate::buffer::Buffer { S::buffer_of(totals.into_iter().map(|(s, _)| s).collect()) }
}
impl<F: Native + Element> Totals for Scaled<F> {
    #[inline]
    fn buffer_of(totals: Vec<Scaled<F>>) -> crate::buffer::Buffer { F::buffer(totals.into_iter().map(Scaled::value).collect()) }
}

/// [`sum`] as a [`Reduction`].
#[derive(Clone, Copy, Debug)]
pub struct Sum;
/// [`prod`] as a [`Reduction`].
#[derive(Clone, Copy, Debug)]
pub struct Product;
/// [`max`] as a [`Reduction`]. Its step on floats is [`maximum`] of the item and the total, which a NaN item leaves unchanged. Every fold and scan starts at `-∞`, and AArch64's and wasm's vector folds keep the running result beside a NaN item, as that step does. So a NaN item changes no running result on any target, and a fold of NaN items alone gives `-∞`.
#[derive(Clone, Copy, Debug)]
pub struct Max;
/// [`min`] as a [`Reduction`], as [`Max`] is [`max`], starting at `∞`.
#[derive(Clone, Copy, Debug)]
pub struct Min;
/// The fold of integers by bitwise `&`, and of Booleans by `and`.
#[derive(Clone, Copy, Debug)]
pub struct And;
/// The fold of integers by bitwise `|`, and of Booleans by `or`.
#[derive(Clone, Copy, Debug)]
pub struct Or;
/// The fold of integers by bitwise `^`, and of Booleans by `!=`: true where an odd number of them are true.
#[derive(Clone, Copy, Debug)]
pub struct Xor;
/// The fold of integers by bitwise XNOR, `!(a ^ b)`, and of Booleans by `==`: true where an even number of them are false.
#[derive(Clone, Copy, Debug)]
pub struct Xnor;
forms!(Xor at bool: |_, [x, y]| x != y);
forms!(Xnor at bool: |_, [x, y]| x == y);

/// `x` folded by `add` from `unit`, through the running results that `running_results` gives, which `merge` combines.
#[inline(always)]
pub(crate) fn folded<A: Copy, S: Copy>(x: &[A], unit: S, add: impl Fn(S, A) -> S + Copy, merge: impl Fn(S, S) -> S) -> S {
    let merge = &merge;
    running_results(x, unit, add, merge, move |parts, rest| rest.iter().fold(parts.iter().fold(unit, |s, &p| merge(s, p)), |s, &v| add(s, v)))
}

/// The running results of `add` from `unit` over `x`: as many as `slice::running` gives, or fewer for a short `x`, so that each takes at least 8 items. A short slice then doesn't set up and merge more results than it has items for. Each takes the items at its own position in every whole group of that many. `finish` gets the results, and the items after the last whole group. Their count is always even. `merge` combines two running results, as [`tree`] does for a long `x`.
#[inline(always)]
fn running_results<A: Copy, S: Copy, T>(x: &[A], unit: S, add: impl Fn(S, A) -> S, merge: impl Fn(S, S) -> S, finish: impl FnOnce(&[S], &[A]) -> T) -> T {
    match crate::slice::running(size_of::<S>()).min(x.len() / 8) {
        ..8 => running_by::<A, S, T, 4>(x, unit, add, merge, finish), 8..16 => running_by::<A, S, T, 8>(x, unit, add, merge, finish),
        16..32 => running_by::<A, S, T, 16>(x, unit, add, merge, finish), _ => running_by::<A, S, T, 32>(x, unit, add, merge, finish)
    }
}

/// `running_results` with `R` of them, folded by [`tree`]. Float callers pass a strict `+` or `*`, not `algebraic_add`: under thin LTO the inner loop is unrolled before the loop vectorizer runs, and reassociation let it treat each running result as a reduction across chunks, which on wasm read every lane from a different chunk. A strict operation also fixes the order of a float total in every build, which changed some totals in their last bits from earlier builds, whose order varied by build and target.
#[inline(always)]
fn running_by<A: Copy, S: Copy, T, const R: usize>(
    x: &[A], unit: S, add: impl Fn(S, A) -> S, merge: impl Fn(S, S) -> S, finish: impl FnOnce(&[S], &[A]) -> T) -> T {
    let (chunks, rest) = x.as_chunks::<R>();
    let parts = if chunks.len() * R <= LEAF { leaf(chunks, unit, &add) } else { tree(chunks, unit, &add, &merge) };
    finish(&parts, rest)
}

/// The most items that one loop of [`tree`] folds. Each leaf adds a call and a merge. Leaves of 16,384 items cost a 65,536-item sum about 2% fewer instructions than 4,096, but left a million tenths' sum in `f32` with 3.4 times the error.
const LEAF: usize = 4096;

/// The `R` running results of `add` from `unit` over `chunks`, each taking the item at its own position in every chunk. Up to [`LEAF`] items fold in one loop, which updates the results in one local array and compiles to vector instructions. More items split into halves, whose results `merge` combines position by position. Each result's chain of steps then stays short, so the rounding error of a float total grows with the logarithm of its length past a leaf, not in proportion to it.
fn tree<A: Copy, S: Copy, const R: usize>(chunks: &[[A; R]], unit: S, add: &impl Fn(S, A) -> S, merge: &impl Fn(S, S) -> S) -> [S; R] {
    if chunks.len() * R > LEAF {
        let (a, b) = chunks.split_at(chunks.len() / 2);
        let (mut a, b) = (tree(a, unit, add, merge), tree(b, unit, add, merge));
        for k in 0..R { a[k] = merge(a[k], b[k]) }
        return a;
    }
    leaf(chunks, unit, add)
}

#[inline(always)]
fn leaf<A: Copy, S: Copy, const R: usize>(chunks: &[[A; R]], unit: S, add: &impl Fn(S, A) -> S) -> [S; R] {
    let mut parts = [unit; R];
    for chunk in chunks { for k in 0..R { parts[k] = add(parts[k], chunk[k]) } }
    parts
}

/// The groups of four items in each block of [`grouped`]'s first pass. Its buffer of running sums stays in L1 for every total type.
const SCAN_BLOCK: usize = 64;

/// Appends the running totals of `x` by `f` to `out`, the first combining `start` where it is given. `from` reads each item as a total, and `to` writes each total. Each group of four items scans within itself, then combines with the total before it. Only one operation in each group then waits for the group before, and `f` may combine items in any order. Set `reorders` for integer sums and bitwise steps, which the compiler may reorder: in one loop it rewrites them into one chain through the total, with one step waiting for each item. With `reorders`, a first pass over each block of `SCAN_BLOCK` groups writes each group's four running sums to a buffer, in the order the outputs take them. The second pass loads them as one vector, adds the total, and adds the group's last sum to the total, which the compiler can't rewrite. Floats keep the one loop. The compiler can't reorder float operations, so their loop already keeps one operation in each group waiting, and a first pass would only add work. Integer maximums and minimums keep the one loop too. On Apple silicon, the one loop costs them up to about twice as much in some processes as in others. With the first pass they cost the same in every process, but 64-bit items then cost about a quarter more than the one loop usually does. Sums into 64-bit totals keep the one loop for the same reason. Its chain of one-cycle additions, one for each item, took a fifth fewer instructions and about a quarter less time than the first pass, whose vectors hold only two 64-bit sums.
#[inline(always)]
pub(crate) fn grouped<A: Copy, S: Copy, O>(
    x: &[A], start: Option<S>, out: &mut Vec<O>, from: impl Fn(A) -> S, f: impl Fn(S, S) -> S, to: impl Fn(S) -> O, reorders: bool) {
    let Some((&first, rest)) = x.split_first() else { return };
    appended(out, x.len(), |slots| {
        let first = from(first);
        let mut total = start.map_or(first, |s| f(s, first));
        slots[0].write(to(total));
        let ((groups, tail), (outs, last)) = (rest.as_chunks::<4>(), slots[1..].as_chunks_mut::<4>());
        let scanned = |[a, b, c, d]: [A; 4]| {
            let (a, b, c, d) = (from(a), from(b), from(c), from(d));
            let ab = f(a, b);
            [a, ab, f(ab, c), f(ab, f(c, d))]
        };
        let (mut sums, mut ends) = ([std::mem::MaybeUninit::<[S; 4]>::uninit(); SCAN_BLOCK], [std::mem::MaybeUninit::<S>::uninit(); SCAN_BLOCK]);
        // Blocks by index: a loop over `chunks(SCAN_BLOCK)` zipped with `chunks_mut` cost `+\b` 90% more instructions.
        let mut i0 = 0;
        while i0 < groups.len() {
            let n = (groups.len() - i0).min(SCAN_BLOCK);
            let (groups, outs) = (&groups[i0..i0 + n], &mut outs[i0..i0 + n]);
            if reorders {
                for g in 0..n {
                    let s = scanned(groups[g]);
                    sums[g].write(s);
                    ends[g].write(s[3]);
                }
            }
            for g in 0..n {
                // SAFETY: with `reorders`, the first pass wrote entries 0 to `n`.
                let sums = if reorders { unsafe { sums[g].assume_init() } } else { scanned(groups[g]) };
                // Four writes, not `array::map`, which cost `+\b` 53% more instructions.
                for k in 0..4 { outs[g][k].write(to(f(total, sums[k]))); }
                // The group's sum from its own buffer, which the compiler can't see is the last of `sums`. Taken from `sums`, the total would wait on the vector the outputs use.
                total = f(total, if reorders { unsafe { ends[g].assume_init() } } else { sums[3] });
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

/// A pick after a running fold steps from `before` to `after` at row `k`: `k` where the fold changed, and `pick` where it didn't. A pick that only moves where its fold changes stays on the first item that gives the fold.
#[inline(always)]
fn moved<T: PartialEq>(pick: usize, before: T, after: T, k: usize) -> usize { if after != before { k } else { pick } }

/// Each float reduction at `f32` and `f64`: its unit, its identity where that isn't its unit, its step, its `chain` where that differs from its merge, its fold of a slice, and its scan.
macro_rules! floats {
    ($($r:ident: $unit:ident, $(identity: $identity:expr,)? $(rows: $rows:ident,)? $(chain: $chain:expr,)? |$a:ident, $b:ident| $step:expr, $fold:expr, $scan:expr;)+) => {$(
        impl<F: Native> Reduction<F> for $r {
            type Total = F;
            const COMMUTATIVE: bool = true;
            const SPLIT: usize = 64;
            #[inline(always)]
            fn unit() -> F { F::$unit() }
            #[inline(always)]
            fn step($a: F, $b: F) -> F { $step }
            #[inline(always)]
            fn merge(a: F, b: F) -> (F, bool) { (Self::step(a, b), true) }
            $(#[inline(always)]
            fn identity() -> F { $identity })?
            $(#[inline]
            fn rows(data: &[F], columns: usize, out: &mut [F]) -> bool { $rows::<F, Self>(data, columns, out) })?
            $(#[inline(always)]
            fn chain(a: F, b: F) -> (F, bool) { (($chain)(a, b), true) })?
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
    Sum: zero, identity: -F::zero(), rows: rows_side_by_side, |a, b| a.algebraic_add(b), |x: &[F]| if x.is_empty() { <Self as Reduction<F>>::unit() } else { folded(x, <Self as Reduction<F>>::identity(), |a: F, b: F| a + b, |a: F, b: F| a + b) }, F::scan_sum;
    Product: one, rows: rows_side_by_side, |a, b| a.algebraic_mul(b), |x| folded(x, F::one(), |a: F, b: F| a * b, |a: F, b: F| a * b), |x, start, out| grouped(x, start, out, |v| v, |a: F, b: F| a * b, |t| t, false);
    Max: neg_infinity, chain: F::step_max, |acc, x| x.maximum(acc), F::fold_max, F::scan_max;
    Min: infinity, chain: F::step_min, |acc, x| x.minimum(acc), F::fold_min, F::scan_min;
}

/// Writes the reduction of `f16`s by each `$r`, which runs the same reduction of `f32`s and totals in `$total`. `$narrow` converts an `f32` total to a `$total`. `$wide` converts a `$total` back to `f32`. `$add` combines `f32` totals in `slice`: a strict operation for a sum or product, as `running_by` needs.
macro_rules! halves {
    ($total:ty, $wide:expr, $narrow:expr; $($r:ty: $add:expr),+) => {$(
        impl Reduction<f16> for $r {
            type Total = $total;
            const COMMUTATIVE: bool = true;
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
                let (unit, add) = (<$r as Reduction<f32>>::identity(), $add);
                ($narrow(if x.is_empty() { <$r as Reduction<f32>>::unit() } else { folded(x, unit, |s, v| add(s, v.to_f32()), add) }), true)
            }
            #[inline]
            fn scan(x: &[f16], start: Option<$total>, out: &mut Vec<$total>) -> bool {
                grouped(x, start.map($wide), out, |v: f16| <$r as Reduction<f32>>::one(v.to_f32()).0, <$r as Reduction<f32>>::step, $narrow, false);
                true
            }
        }
    )+};
}
// A sum or product of `f16`s totals in `f32`, for its caller to round once. A maximum or minimum is exact in `f16`.
halves!(f32, |t: f32| t, |t: f32| t; Sum: |a: f32, b: f32| a + b, Product: |a: f32, b: f32| a * b);
halves!(f16, f16::to_f32, f16::from_f32; Max: <Max as Reduction<f32>>::step, Min: <Min as Reduction<f32>>::step);

/// Each sum of complex numbers adds their real parts and their imaginary parts, in any order.
impl<F: Native> Reduction<Complex<F>> for Sum {
    type Total = Complex<F>;
    const COMMUTATIVE: bool = true;
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
        let add = |a: Complex<F>, b: Complex<F>| a + b;
        (if x.is_empty() { <Self as Reduction<Complex<F>>>::unit() } else { folded(x, <Self as Reduction<Complex<F>>>::identity(), add, add) }, true)
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
    const COMMUTATIVE: bool = true;
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

/// AArch64's `fmaxnm` and `fminnm` give the argument that isn't NaN, and wasm's `pmax` and `pmin` give their first argument where either is NaN. A fold starts at `-∞` or `∞` and passes its running result first, so each gives what `Core::maximum` and `Core::minimum` give, in one instruction. A scan steps the same way. `Folds::step_max` and `step_min` hold that step. `fmaxnm` drops a NaN item wherever it stands, but `pmax` keeps a NaN running result, so a wasm scan turns each NaN item into the start as it reads it, as `records` describes. A NaN start fills a wasm scan with NaN, and `fmaxnm` drops it. Wasm's `max` and `min` give NaN beside NaN, and on x86-64 hosts take about eight instructions each. A fold runs the vector forms on `PARTS` running vectors. Each step of a scan depends on the step before. In wasm, the step takes lane 0 of the vector instruction on two splats. On AArch64 that code keeps a `dup` before each `fmaxnm` at `f32`, which has no one-lane vector type, and `f64::max`, LLVM's `maxnum`, quiets each argument with an `fminnm` of itself first. The step there is inline assembly. Each type's entry names its scan. AArch64's `f64` scan is `grouped`, at the speed of memory. Skipping blocks gained it nothing. Each block with a new maximum paid for a test it didn't use. The others skip blocks through `records`. AArch64's `f32` tests a block by comparing each item, as `below` does. Wasm tests it by its fold, then one comparison. In wasm, comparing each item took five instructions for two items. Each entry also names its running sum. `f32`'s is `sum_scan`, which scans four-lane vectors and makes `+\` about 1.3 times faster. `f64`'s is `grouped`, because a two-lane vector scan needs more operations per item than `grouped`'s two additions.
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
                #[inline(always)]
                fn step_max(t: $t, v: $t) -> $t { lanes!(@$step $arg, "fmaxnm", $splat, $max, t, v) }
                #[inline(always)]
                fn step_min(t: $t, v: $t) -> $t { lanes!(@$step $arg, "fminnm", $splat, $min, t, v) }
                fn fold_max(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, NEG_INFINITY, $splat, $load, $max, Self::step_max) }
                fn fold_min(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, INFINITY, $splat, $load, $min, Self::step_min) }
                fn scan_max(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) {
                    lanes!(@scan $scan, x, start, out, lanes!(@read $step, $t, NEG_INFINITY), Self::step_max, below, |b: &[$t], t: $t| Self::fold_max(b) < t)
                }
                fn scan_min(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) {
                    lanes!(@scan $scan, x, start, out, lanes!(@read $step, $t, INFINITY), Self::step_min, above, |b: &[$t], t: $t| Self::fold_min(b) > t)
                }
                fn scan_sum(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) { lanes!(@sum $sum, x, start, out) }
                #[inline(always)]
                fn ordered_lanes<const W: usize>(a: [$t; W], c: [$t; W]) -> ([$t; W], [$t; W]) {
                    const { assert!(W == $lanes) };
                    // SAFETY: this module compiles only for targets with these instructions, and each row of `$lanes` items is one vector. `$max` takes `c` first so that equal items keep their places beside wasm's `pmax`, which gives its first argument where they're equal.
                    unsafe {
                        let (x, y) = ($load(a.as_ptr().cast()), $load(c.as_ptr().cast()));
                        (core::mem::transmute_copy(&$min(x, y)), core::mem::transmute_copy(&$max(y, x)))
                    }
                }
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
        // AArch64's `fmaxnm` drops a NaN item wherever it stands, so its scans read items as they are. Wasm's `pmax` keeps a NaN running result, so its scans turn each NaN item into the start as they read it, with a test for NaN and a select.
        (@read asm, $t:ty, $unit:ident) => { |v: $t| v };
        (@read lane, $t:ty, $unit:ident) => { |v: $t| if v.is_nan() { <$t>::$unit } else { v } };
        (@scan grouped, $x:ident, $start:ident, $out:ident, $from:expr, $step:expr, $compare:expr, $fold:expr) => { grouped($x, $start, $out, $from, $step, |t| t, false) };
        (@scan compare, $x:ident, $start:ident, $out:ident, $from:expr, $step:expr, $compare:expr, $fold:expr) => { records($x, $start, $out, $from, $step, $compare) };
        (@scan fold, $x:ident, $start:ident, $out:ident, $from:expr, $step:expr, $compare:expr, $fold:expr) => { records($x, $start, $out, $from, $step, $fold) };
        (@sum grouped, $x:ident, $start:ident, $out:ident) => { grouped($x, $start, $out, |v| v, |a, b| a + b, |t| t, false) };
        (@sum vector, $x:ident, $start:ident, $out:ident) => { sum_scan($x, $start, $out) };
    }
    #[cfg(target_arch = "aarch64")]
    lanes! {
        f64: 2, vdupq_n_f64, vld1q_f64, vmaxnmq_f64, vminnmq_f64, asm("d"), grouped, grouped;
        f32: 4, vdupq_n_f32, vld1q_f32, vmaxnmq_f32, vminnmq_f32, asm("s"), compare, vector;
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
        f64: 2, f64x2_splat, v128_load, f64x2_pmax, f64x2_pmin, lane(f64x2_extract_lane), fold, grouped;
        f32: 4, f32x4_splat, v128_load, f32x4_pmax, f32x4_pmin, lane(f32x4_extract_lane), fold, vector;
    }
}

/// Elsewhere, the folds and scans use `Folds`' defaults. A fold's loop of `Core::maximum` vectorizes, and each step of a scan is one `maxsd` or `minsd` on x86-64. With AVX2, `ordered_lanes` compares each row of `search::network` as one vector, through `vminps` and `vmaxps` or their `f64` forms, which give what `Core::minimum` and `Core::maximum` give. The default leaves its lanes to LLVM, which vectorizes `f32`'s block of 8 rows of 8 lanes across its rows, with shuffles and spills, and stores each lesser row through a masked store.
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
mod plain {
    #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
    use core::arch::x86_64::*;
    macro_rules! plain {
        ($($t:ty: $lanes:literal, $load:ident, $min:ident, $max:ident;)+) => {$(
            impl super::Folds for $t {
                #[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
                #[inline(always)]
                fn ordered_lanes<const W: usize>(a: [$t; W], c: [$t; W]) -> ([$t; W], [$t; W]) {
                    const { assert!(W == $lanes) };
                    // SAFETY: this code compiles only with AVX2, and each row of `$lanes` items is one 256-bit vector. `$min(y, x)` and `$max(x, y)` give `c[j].minimum(a[j])` and `a[j].maximum(c[j])`, as the default does.
                    unsafe {
                        let (x, y) = ($load(a.as_ptr()), $load(c.as_ptr()));
                        (core::mem::transmute_copy(&$min(y, x)), core::mem::transmute_copy(&$max(x, y)))
                    }
                }
            }
        )+};
    }
    plain! {
        f32: 8, _mm256_loadu_ps, _mm256_min_ps, _mm256_max_ps;
        f64: 4, _mm256_loadu_pd, _mm256_min_pd, _mm256_max_pd;
    }
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
            const COMMUTATIVE: bool = true;
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
            const COMMUTATIVE: bool = true;
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

/// [`Sum`] of integers wrapped to `i64`, with no test of whether a total fits, as static kinds and fixed integers keep their sums. A sum of `i64`s is then one vector add for each item, where `Sum` splits each item into halves to test its total. Narrower items take `Sum`'s own sums, which either always fit or test once for each block.
macro_rules! wrapped_sums {
    ($($t:ty: $slice:expr),+) => {$(
        impl Reduction<$t> for crate::Wrapped<Sum> {
            type Total = i64;
            const COMMUTATIVE: bool = true;
            #[inline(always)]
            fn unit() -> i64 { 0 }
            #[inline(always)]
            fn step(acc: i64, x: $t) -> i64 { acc.wrapping_add(x.into()) }
            #[inline(always)]
            fn merge(a: i64, b: i64) -> (i64, bool) { (a.wrapping_add(b), true) }
            #[inline]
            fn slice(x: &[$t]) -> (i64, bool) { (($slice)(x), true) }
            #[inline]
            fn scan(x: &[$t], start: Option<i64>, out: &mut Vec<i64>) -> bool { signed_cumsum(x, start.unwrap_or(0), out); true }
        }
    )+};
}
wrapped_sums!(u8: |x| small_sum(x).0, i16: |x| small_sum(x).0, i32: |x| narrow_sum(x).0, i64: |x: &[i64]| folded(x, 0i64, i64::wrapping_add, i64::wrapping_add));

/// The XNOR fold from `!0` of `n` items whose XOR fold is `xor`: their XOR, complemented for an even count. The compiler vectorizes an XOR fold, and no spelling of an XNOR one.
fn xnor_of<T: std::ops::Not<Output = T>>(xor: T, n: usize) -> T { if n % 2 == 0 { !xor } else { xor } }

/// [`Max`], [`Min`], [`And`], [`Or`] and [`Xor`] for each integer type, and [`Xnor`] for each signed one, from each one's unit and step. No step overflows. A slice folds in one plain loop, which the compiler vectorizes and reorders as it does integer addition; [`Xnor`]'s folds through [`xnor_of`]. A maximum or minimum scans through [`records`]. A bitwise scan takes [`grouped`] with `reorders` set.
macro_rules! closed {
    (@one $r:ident, $t:ty, $unit:expr, $step:expr, $scan:ident) => { closed!(@one $r, $t, $unit, $step, $scan, |x: &[$t]| x.iter().copied().fold($unit, $step)); };
    (@one $r:ident, $t:ty, $unit:expr, $step:expr, $scan:ident, $slice:expr) => {
        impl Reduction<$t> for $r {
            type Total = $t;
            const COMMUTATIVE: bool = true;
            #[inline(always)]
            fn unit() -> $t { $unit }
            #[inline(always)]
            fn step(acc: $t, x: $t) -> $t { ($step)(acc, x) }
            #[inline(always)]
            fn merge(a: $t, b: $t) -> ($t, bool) { (($step)(a, b), true) }
            #[inline]
            fn slice(x: &[$t]) -> ($t, bool) { (($slice)(x), true) }
            #[inline]
            fn scan(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) -> bool {
                closed!(@$scan x, start, out, $t, $step);
                true
            }
            #[inline]
            fn rows(data: &[$t], columns: usize, out: &mut [$t]) -> bool { rows_side_by_side::<$t, Self>(data, columns, out) }
        }
    };
    (@records $x:ident, $start:ident, $out:ident, $t:ty, $step:expr) => { records($x, $start, $out, |v| v, $step, |b: &[$t], t: $t| b.iter().fold(t, |m, &v| ($step)(m, v)) == t) };
    (@grouped $x:ident, $start:ident, $out:ident, $t:ty, $step:expr) => { grouped($x, $start, $out, |v| v, $step, |t| t, true) };
    (signed $($t:ty),+) => {$( closed!(@one Xnor, $t, !0, |a: $t, b: $t| !(a ^ b), grouped, |x: &[$t]| xnor_of(x.iter().fold(0, |a: $t, &b| a ^ b), x.len())); )+};
    ($($t:ty),+) => {$(
        closed!(@one Max, $t, <$t>::MIN, Ord::max, records);
        closed!(@one Min, $t, <$t>::MAX, Ord::min, records);
        closed!(@one And, $t, !0, |a: $t, b: $t| a & b, grouped);
        closed!(@one Or, $t, 0, |a: $t, b: $t| a | b, grouped);
        closed!(@one Xor, $t, 0, |a: $t, b: $t| a ^ b, grouped);
    )+};
}
closed!(i16, i32, i64, u8, u16, u32, u64);
closed!(signed i16, i32, i64);
/// [`Xnor`] of bytes, totalled at `i16`. Each total lies between -256 and 255, and the XNOR of two bytes is negative.
impl Reduction<u8> for Xnor {
    type Total = i16;
    const COMMUTATIVE: bool = true;
    #[inline(always)]
    fn unit() -> i16 { !0 }
    #[inline(always)]
    fn step(acc: i16, x: u8) -> i16 { !(acc ^ i16::from(x)) }
    #[inline(always)]
    fn merge(a: i16, b: i16) -> (i16, bool) { (!(a ^ b), true) }
    #[inline]
    fn slice(x: &[u8]) -> (i16, bool) { (xnor_of(x.iter().fold(0i16, |a, &b| a ^ i16::from(b)), x.len()), true) }
    #[inline]
    fn scan(x: &[u8], start: Option<i16>, out: &mut Vec<i16>) -> bool { grouped(x, start, out, i16::from, |a: i16, b: i16| !(a ^ b), |t| t, true); true }
    #[inline]
    fn rows(data: &[u8], columns: usize, out: &mut [i16]) -> bool { rows_side_by_side::<u8, Self>(data, columns, out) }
}
/// The reductions on Booleans, with one row for each truth table: the reductions that fold by it, its unit, its step and the table itself. Each scan runs through [`Logical`]. [`Max`] and [`Or`] fold Booleans by `or`, and [`Min`] and [`And`] by `and`.
macro_rules! booleans {
    ($($($r:ident)|+: $unit:expr, |$a:ident, $b:ident| $step:expr, $table:expr;)+) => {$($(
        impl Reduction<bool> for $r {
            type Total = bool;
            const COMMUTATIVE: bool = true;
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
            #[inline]
            fn short_lanes(data: &[bool], len: usize, start: Option<bool>, out: &mut Vec<bool>) -> Option<bool> {
                (1..=64).contains(&len).then(|| { Logical($table).short_lanes(data, len, start, out); true })
            }
        }
    )+)+};
}
booleans! {
    Max | Or: false, |a, b| a | b, [[false, true], [true, true]];
    Min | And: true, |a, b| a & b, [[false, false], [false, true]];
    Xor: false, |a, b| a ^ b, [[false, true], [true, false]];
    Xnor: true, |a, b| a == b, [[true, false], [false, true]];
}

/// The running folds of `x` from `start` by `step`, a maximum or minimum, reading each item through `from`. Where a float maximum's step keeps a NaN running result, its `from` turns a NaN item into `-∞`: `grouped` combines the items of each group of four with each other before it steps from the total, and a NaN that comes first in one of those combinations, the group's first or third item, would hide the items after it in the group. A block of 256 items that `keeps(block, total)` passes repeats the running result, with no step for each item. Other blocks scan as `grouped` does. On random items, almost every block repeats the running result. After four blocks in a row change the running result, blocks scan with no test until one leaves it unchanged. Their lengths start at 256 items and double up to 16,384. Items that rise steadily then pay for four tests and a few calls of `grouped`. A call for every 256 items added a third to their scan. Random items start such a run about one time in 25. After two blocks, they started one about half the time. The compiler vectorizes each block's test because the block's length isn't fixed. A loop of a fixed length unrolls into one item at a time.
pub(crate) fn records<T: Copy + PartialEq>(
    x: &[T], start: Option<T>, out: &mut Vec<T>, from: impl Fn(T) -> T + Copy, step: impl Fn(T, T) -> T + Copy, keeps: impl Fn(&[T], T) -> bool) {
    let Some(&first) = x.first() else { return };
    let first = from(first);
    let (mut total, mut at, mut changes) = (start.map_or(first, |s| step(s, first)), 0, 0);
    while at < x.len() {
        let test = changes < 4;
        let block = &x[at..x.len().min(at + if test { 256 } else { 256 << (changes - 4).min(6) })];
        at += block.len();
        if test && keeps(block, total) {
            out.resize(out.len() + block.len(), total);
            changes = 0;
        } else {
            grouped(block, Some(total), out, from, step, |t| t, false);
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
/// The fewest rows in each piece of [`fold_pieces`]. The partial folds of the lanes that a piece shares with its neighbours then hold at most a 16th of its items, and merge on the calling thread. With 256 rows, an array of fewer rows than that folded on one thread: at 4 threads, 16 rows took 19% less time for `+⌿` on 64×65536 and 32% less for `⌊⌿` on 64×16384, as kmeans folds.
const ROWS: usize = 16;

/// The fewest items in each piece of [`fold_pieces`] for an array whose rows hold `inner` items: at least [`PIECE`] items and [`ROWS`] rows, in whole `CHUNK`s.
fn piece_size(inner: usize) -> usize { PIECE.max(ROWS * inner).next_multiple_of(CHUNK) }

/// The fold of the one item `x`, clearing `ok` where it doesn't fit.
#[inline(always)]
fn one_of<T, R: Reduction<T>>(x: T, ok: &mut bool) -> R::Total {
    let (t, fits) = R::one(x);
    *ok &= fits;
    t
}

/// Steps each of `totals` by the matching item of `x`, as far as the shorter goes, and gives whether every fold fits. It loops by index, because column folds ran 1.6 times slower when `strip_by`'s running totals went through zipped slices.
#[inline(always)]
fn steps_of<T: Copy, R: Reduction<T>>(totals: &mut [R::Total], x: &[T]) -> bool {
    let (n, mut fits) = (totals.len().min(x.len()), true);
    for k in 0..n { let (t, f) = R::checked(totals[k], x[k]); (totals[k], fits) = (t, fits & f) }
    fits
}

/// Appends to `out` the fold of each column of the first one, two or three of `rows`, as far as the shorter goes, and gives whether every fold fits. One pass reads those rows, where a pass for each row would also read and write `out` again.
#[inline(always)]
fn started<T: Copy, R: Reduction<T>>(out: &mut Vec<R::Total>, rows: &[&[T]]) -> bool {
    #[inline(always)]
    fn step<T, R: Reduction<T>>((t, f): (R::Total, bool), x: T) -> (R::Total, bool) { let (u, g) = R::checked(t, x); (u, f & g) }
    let mut fits = true;
    match *rows {
        [] => {}
        [a] => out.extend(a.iter().map(|&x| {
            let (t, f) = R::one(x);
            fits &= f;
            t
        })),
        [a, b] => out.extend(a.iter().zip(b).map(|(&x, &y)| {
            let (t, f) = step::<T, R>(R::one(x), y);
            fits &= f;
            t
        })),
        [a, b, c, ..] => out.extend(a.iter().zip(b).zip(c).map(|((&x, &y), &z)| {
            let (t, f) = step::<T, R>(step::<T, R>(R::one(x), y), z);
            fits &= f;
            t
        })),
    }
    fits
}

/// Appends to `out` the folds of `width` rows, `row(0)` up to `row(width - 1)`, item by item, as far as the shortest row goes, and gives whether every fold fits. One pass folds the first three rows (`started`), and each later row steps those folds while they're still in cache.
#[inline(always)]
fn rows_folded<'a, T: Copy + 'a, R: Reduction<T>>(out: &mut Vec<R::Total>, width: usize, row: impl Fn(usize) -> &'a [T]) -> bool {
    let at = out.len();
    let mut fits = match width {
        1 => started::<T, R>(out, &[row(0)]), 2 => started::<T, R>(out, &[row(0), row(1)]), _ => started::<T, R>(out, &[row(0), row(1), row(2)])
    };
    for i in 3..width { fits &= steps_of::<T, R>(&mut out[at..], row(i)); }
    fits
}

/// Merges each of `totals` with the matching total of `parts`, as far as the shorter goes, and gives whether every result fits. It loops by index, as [`steps_of`] does.
#[inline(always)]
fn merges_of<T, R: Reduction<T>>(totals: &mut [R::Total], parts: &[R::Total]) -> bool {
    let (n, mut fits) = (totals.len().min(parts.len()), true);
    for k in 0..n { let (t, f) = R::merge(totals[k], parts[k]); (totals[k], fits) = (t, fits & f) }
    fits
}

/// The folds `a` and `b` of neighbouring runs combined, clearing `ok` where the result doesn't fit.
#[inline(always)]
fn merge_of<T, R: Reduction<T>>(a: R::Total, b: R::Total, ok: &mut bool) -> R::Total {
    let (t, fits) = R::merge(a, b);
    *ok &= fits;
    t
}

/// The folds `a` and `b` of neighbouring runs combined by `chain`, clearing `ok` where the result doesn't fit.
#[inline(always)]
fn chain_of<T, R: Reduction<T>>(a: R::Total, b: R::Total, ok: &mut bool) -> R::Total {
    let (t, fits) = R::chain(a, b);
    *ok &= fits;
    t
}

/// The length below which a lane of consecutive items runs through [`side_by_side`]'s loop in `rows_scan` and `rows_side_by_side`, [`LANES`] lanes side by side, because a slice form costs more to start than such a lane's items do.
const SHORT_LANE: usize = 64;

/// How many lanes of fewer than [`SHORT_LANE`] items `rows_scan` and `rows_side_by_side` run side by side. Each lane's running result waits on the step before it. Run one lane at a time, a loop keeps one chain in flight and leaves any overlap to the processor running ahead into the next lane, which changes with where the code sits. Side by side, `LANES` chains are always in flight.
const LANES: usize = 8;

/// Runs `first` and `step` along the lanes of `len` items in `data`, `B` lanes side by side, and gives whether every result fits. `data` holds a whole number of groups of `B` lanes. `each` gets each running result with its position in `data`, and `last` each lane's last result with the lane's number.
#[inline(always)]
fn side_by_side<const B: usize, T: Copy, S: Copy>(
    data: &[T], len: usize, first: impl Fn(T) -> (S, bool), step: impl Fn(S, T) -> (S, bool), mut each: impl FnMut(usize, S), mut last: impl FnMut(usize, S)
) -> bool {
    let mut fits = true;
    for (g, group) in data.chunks_exact(B * len).enumerate() {
        let at = g * B * len;
        let mut ok = [true; B];
        let mut t: [S; B] = std::array::from_fn(|b| {
            let (t, f) = first(group[b * len]);
            ok[b] = f;
            each(at + b * len, t);
            t
        });
        for i in 1..len {
            for b in 0..B {
                // SAFETY: `group` holds `B * len` items, and `b < B` and `i < len`.
                let (s, f) = step(t[b], unsafe { *group.get_unchecked(b * len + i) });
                (t[b], ok[b]) = (s, ok[b] & f);
                each(at + b * len + i, s);
            }
        }
        for (b, &t) in t.iter().enumerate() { last(g * B + b, t) }
        fits &= ok.into_iter().all(|f| f);
    }
    fits
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
    /// The low bits of `n`.
    fn low(n: i64) -> Self;
}
macro_rules! tallies {
    ($($t:ty),+) => {$(
        impl Tally for $t {
            #[inline(always)]
            fn add(self, y: $t) -> $t { self.wrapping_add(y) }
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
    const COMMUTATIVE: bool = true;
    #[inline(always)]
    fn unit() -> C { C::default() }
    #[inline(always)]
    fn step(acc: C, x: T) -> C { acc.add(x.term()) }
    #[inline(always)]
    fn merge(a: C, b: C) -> (C, bool) { (a.add(b), true) }
    #[inline]
    fn slice(x: &[T]) -> (C, bool) { (T::terms(x), true) }
    #[inline]
    fn scan(x: &[T], start: Option<C>, out: &mut Vec<C>) -> bool { grouped(x, start, out, T::term, C::add, |t| t, size_of::<C>() < 8); true }
}

/// The sum of the squares of floats' or complex numbers' magnitudes as a [`Reduction`], with each square computed and totalled in `f64`, where an `f32`'s square neither overflows nor underflows. It folds the items themselves, with no buffer of squares.
#[derive(Clone, Copy, Debug)]
pub struct Squares;

/// `s` plus the square of `x`'s magnitude, computed in `f64`, each multiply-add rounding once where `FUSED` is true. A real item has no imaginary part to add.
#[inline(always)]
fn add_square<T: Scalar>(s: f64, x: T) -> f64 {
    let re = x.re().wide();
    let s = re.mul_acc(re, s);
    if T::COMPLEX { let im = x.im().wide(); im.mul_acc(im, s) } else { s }
}

impl<T: Scalar> Reduction<T> for Squares {
    type Total = f64;
    const COMMUTATIVE: bool = true;
    #[inline(always)]
    fn unit() -> f64 { 0.0 }
    #[inline(always)]
    fn step(acc: f64, x: T) -> f64 { add_square(acc, x) }
    #[inline(always)]
    fn merge(a: f64, b: f64) -> (f64, bool) { (a + b, true) }
    #[inline]
    fn slice(x: &[T]) -> (f64, bool) { (folded(x, 0.0, add_square, |a, b| a + b), true) }
    #[inline]
    fn scan(x: &[T], start: Option<f64>, out: &mut Vec<f64>) -> bool { grouped(x, start, out, |v| add_square(0.0, v), |a, b| a + b, |t| t, false); true }
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
    let merge = &merge;
    running_results(x, unit, add, merge, move |parts, rest| {
        let ((pairs, _), (tail, last)) = (parts.as_chunks::<2>(), rest.as_chunks::<2>());
        let (even, odd) = pairs.iter().fold((unit, unit), |(e, o), &[p, q]| (merge(e, p), merge(o, q)));
        let (even, odd) = tail.iter().fold((even, odd), |(e, o), &[p, q]| (add(e, p), add(o, q)));
        last.first().map_or(even, |&z| add(even, z)) - odd
    })
}

/// The running folds of `x` by `R`, each item taken in turn by `checked` from `start` or `R`'s identity, appended to `out`, and whether every one fits.
fn checked_steps<T: Copy, R: Reduction<T>>(x: &[T], start: Option<R::Total>, out: &mut Vec<R::Total>) -> bool {
    let (mut total, mut fits) = (start.unwrap_or_else(R::identity), true);
    out.extend(x.iter().map(|&v| {
        let ok;
        (total, ok) = R::checked(total, v);
        fits &= ok;
        total
    }));
    fits
}

/// The right fold by the inverse of a reduction: `-` for [`Sum`], and `÷` for [`ScaledProduct`] on floats and [`Product`] on complex numbers. `x0 - (x1 - (x2 - …))` is the sum of the items at even positions less the sum of those at odd positions, and `x0 ÷ (x1 ÷ (x2 ÷ …))` is the quotient of their products. A total holds that sum or quotient for the items folded so far, with whether their number is odd. `step` takes an item at an odd position away, and `merge` takes the later run's total away where the earlier run's count is odd, each through [`Inverse`]. Every fold that takes a [`Reduction`] folds it, as [`along`], [`fold_into`], [`along_mapped`] and [`fold_pieces`] do. [`Totals`] stores its folds without the parity.
#[derive(Clone, Copy, Debug, Default)]
pub struct Alternated<R>(std::marker::PhantomData<R>);

/// A reduction whose step can be undone, as [`Alternated`] folds by it: [`Sum`] subtracts, and [`ScaledProduct`] and [`Product`] divide. An integer difference fits only within `i64`, a complex quotient only inside the normal range, and a float quotient only where no item, or product of the running mantissa and an item, leaves the normal floats. An alternated fold can then fail where the right fold taken item by item, or its exact result, would fit.
pub trait Inverse<T>: Reduction<T> {
    /// `acc` with the item `x` taken away, and whether it fits.
    fn inverse(acc: Self::Total, x: T) -> (Self::Total, bool);
    /// The fold `a` with the fold `b` of the next run taken away, and whether it fits.
    fn undo(a: Self::Total, b: Self::Total) -> (Self::Total, bool);
    /// Whether a running total fits, as every one does apart from a float quotient outside the normal range.
    #[inline(always)]
    fn fits(_: Self::Total) -> bool { true }
    /// The right fold of the slice `x` by the inverse, and whether it fits: for sums through a sum at each parity (`alternated`), and for products through the pairs' quotients (`pairwise`).
    fn alternating(x: &[T]) -> (Self::Total, bool);
}

impl<T: Copy, R: Inverse<T>> Reduction<T> for Alternated<R> {
    type Total = (R::Total, bool);
    #[inline(always)]
    fn unit() -> Self::Total { (R::unit(), false) }
    #[inline(always)]
    fn identity() -> Self::Total { (R::identity(), false) }
    #[inline(always)]
    fn step((s, odd): Self::Total, x: T) -> Self::Total { (if odd { R::inverse(s, x).0 } else { R::step(s, x) }, !odd) }
    #[inline(always)]
    fn checked((s, odd): Self::Total, x: T) -> (Self::Total, bool) {
        let (t, ok) = if odd { R::inverse(s, x) } else { R::checked(s, x) };
        ((t, !odd), ok && R::fits(t))
    }
    #[inline(always)]
    fn merge((a, odd): Self::Total, (b, more): Self::Total) -> (Self::Total, bool) {
        let (t, ok) = if odd { R::undo(a, b) } else { R::merge(a, b) };
        ((t, odd ^ more), ok && R::fits(t))
    }
    #[inline]
    fn slice(x: &[T]) -> (Self::Total, bool) { let (t, ok) = R::alternating(x); ((t, x.len() % 2 == 1), ok) }
}

impl<F: Native> Inverse<F> for Sum {
    #[inline(always)]
    fn inverse(acc: F, x: F) -> (F, bool) { (acc.algebraic_add(-x), true) }
    #[inline(always)]
    fn undo(a: F, b: F) -> (F, bool) { (a.algebraic_add(-b), true) }
    #[inline]
    fn alternating(x: &[F]) -> (F, bool) { (alternated(x, F::zero(), |s, v| s + v, |a, b| a + b), true) }
}

impl<F: Native> Inverse<Complex<F>> for Sum {
    #[inline(always)]
    fn inverse(acc: Complex<F>, x: Complex<F>) -> (Complex<F>, bool) { (<Sum as Reduction<Complex<F>>>::step(acc, -x), true) }
    #[inline(always)]
    fn undo(a: Complex<F>, b: Complex<F>) -> (Complex<F>, bool) { (<Sum as Reduction<Complex<F>>>::step(a, -b), true) }
    #[inline]
    fn alternating(x: &[Complex<F>]) -> (Complex<F>, bool) { (alternated(x, Complex::new(F::zero(), F::zero()), |s, v| s + v, |a, b| a + b), true) }
}

/// [`Sum`]'s inverse on integers, which subtracts at `i64`. A slice sums the items at even positions less those at odd positions, which is exact: each run of `$run` items, an even number, sums at `$acc`, where it can't overflow, and the runs total at `i128`. Bytes and 16-bit integers sum at `i32`, which fits twice as many in each vector as `i64`. Separate running sums for each parity take less time than `pairwise`, which must split each pair out of the vector lanes.
macro_rules! alternated_integers {
    ($($t:ty => $acc:ty, $run:expr);+) => {$(
        impl Inverse<$t> for Sum {
            #[inline(always)]
            fn inverse(acc: i64, x: $t) -> (i64, bool) { let (t, over) = acc.overflowing_sub(i64::from(x)); (t, !over) }
            #[inline(always)]
            fn undo(a: i64, b: i64) -> (i64, bool) { let (t, over) = a.overflowing_sub(b); (t, !over) }
            #[inline]
            fn alternating(x: &[$t]) -> (i64, bool) {
                let run = |c: &[$t]| i128::from(alternated(c, 0 as $acc, |s, v| s + <$acc>::from(v), |s, t| s + t));
                let total: i128 = x.chunks($run).map(run).sum();
                (total as i64, i64::try_from(total).is_ok())
            }
        }
    )+};
}
alternated_integers!(i16 => i32, 1 << 15; u8 => i32, 1 << 15; i32 => i64, 1 << 31; i64 => i128, 1 << 31);

/// Whether a quotient lies inside the normal range of floats, where [`Product`]'s inverse on complex numbers fits.
#[inline(always)]
fn normal<F: Native>(q: F) -> bool { q.abs() >= F::min_positive_value() && q.is_finite() }

/// A product of floats held as `m × 2^e`, a mantissa `m` and a whole exponent `e` kept apart, so that the product never leaves the normal floats however many items it takes. A step leaves the mantissa where it is while its magnitude stays inside 2^±511 (2^±63 in `f32`), half the floats' exponents, so that the product of two mantissas stays normal, and moves the mantissa's exponent into `e` once it leaves. `normal` says whether every step so far was a nonzero normal float. [`ScaledProduct`] folds into it. The exponent sums exactly up to 2^53 in `f64` and 2^24 in `f32`, which a product that ends inside the floats' range never nears.
#[derive(Clone, Copy, Debug)]
pub struct Scaled<F> { m: F, e: F, normal: bool }

impl<F: Native> Scaled<F> {
    /// `p × 2^e`, with `p`'s exponent moved into `e` where `p`'s magnitude lies outside 2^±511 (2^±63 in `f32`), still `normal` where `p` is a nonzero normal float.
    #[inline(always)]
    fn of(p: F, e: F, normal: bool) -> Self {
        let (lo, hi) = if F::WIDE { (1.4916681462400413e-154, 6.703903964971299e153) } else { (1.0842021724855044e-19, 9.223372036854776e18) };
        if p.abs() >= F::narrow(lo) && p.abs() <= F::narrow(hi) { Self { m: p, e, normal } } else { Self::rescaled(p, e, normal) }
    }
    /// `p × 2^e` with all of `p`'s exponent moved into `e`, its mantissa from 0.75 to 1.5 in magnitude, still `normal` where `p` is a nonzero normal float.
    #[inline]
    fn rescaled(p: F, e: F, normal: bool) -> Self {
        let (m, f) = p.abs().split();
        Self { m: if p < F::zero() { -m } else { m }, e: e + f, normal: normal & p.abs().positive_normal() }
    }
    /// This product times `q`.
    #[inline(always)]
    fn times(self, q: F) -> Self { Self::of(self.m * q, self.e, self.normal) }
    /// This product divided by `q`.
    #[inline(always)]
    fn over(self, q: F) -> Self { Self::of(self.m / q, self.e, self.normal) }
    /// This product times the product `b`.
    #[inline(always)]
    fn joined(self, b: Self) -> Self { Self::of(self.m * b.m, self.e + b.e, self.normal & b.normal) }
    /// This product divided by the product `b`.
    #[inline(always)]
    fn undone(self, b: Self) -> Self { Self::of(self.m / b.m, self.e - b.e, self.normal & b.normal) }
    /// The float nearest the product: infinite or zero where it lies past the floats' range, and rounded at most once, where it lies below the normal floats.
    #[inline]
    pub fn value(self) -> F {
        let Self { m, e, .. } = Self::rescaled(self.m, self.e, true);
        let far = F::narrow(if F::WIDE { 2040.0 } else { 250.0 });
        crate::float::pow2_times::<F, { crate::FUSED }>(m, e.max(-far).min(far))
    }
}

/// The product of floats with its exponent kept apart, as [`Scaled`] holds it, so that a running product can't overflow or underflow where the whole product doesn't. A step fails only where an item, or a step's product, isn't a nonzero normal float: on zeros, infinities and NaNs, on subnormals, and on items so large or small that a mantissa times them leaves the normal floats. The exact fold takes those. [`Alternated`] folds `÷` on floats by it.
#[derive(Clone, Copy, Debug, Default)]
pub struct ScaledProduct;

impl<F: Native> Reduction<F> for ScaledProduct {
    type Total = Scaled<F>;
    const COMMUTATIVE: bool = true;
    #[inline(always)]
    fn unit() -> Scaled<F> { Scaled { m: F::one(), e: F::zero(), normal: true } }
    #[inline(always)]
    fn step(acc: Scaled<F>, x: F) -> Scaled<F> { acc.times(x) }
    #[inline(always)]
    fn checked(acc: Scaled<F>, x: F) -> (Scaled<F>, bool) { let t = acc.times(x); (t, t.normal) }
    #[inline(always)]
    fn merge(a: Scaled<F>, b: Scaled<F>) -> (Scaled<F>, bool) { let t = a.joined(b); (t, t.normal) }
}

impl<F: Native> Inverse<F> for ScaledProduct {
    #[inline(always)]
    fn inverse(acc: Scaled<F>, x: F) -> (Scaled<F>, bool) { let t = acc.over(x); (t, t.normal) }
    #[inline(always)]
    fn undo(a: Scaled<F>, b: Scaled<F>) -> (Scaled<F>, bool) { let t = a.undone(b); (t, t.normal) }
    #[inline]
    fn alternating(x: &[F]) -> (Scaled<F>, bool) {
        // The quotient of plain floats vectorizes, so it goes first. One that leaves the normal floats runs again with its exponent apart.
        let (q, fits) = quotient(x, F::one(), |a, b| a / b, F::algebraic_mul, F::abs, F::is_finite);
        let t = if fits { Scaled::of(q, F::zero(), true) }
        else { pairwise(x, <Self as Reduction<F>>::unit(), |s, a, b| s.times(a / b), Scaled::joined, Scaled::times) };
        (t, t.normal)
    }
}

/// The product of two complex numbers, as the quotients of [`Product`]'s inverse on complex numbers take it.
#[inline(always)]
fn complex_times<F: Native>(p: Complex<F>, q: Complex<F>) -> Complex<F> {
    Complex::new(p.re.algebraic_mul(q.re).algebraic_add(-p.im.algebraic_mul(q.im)), p.re.algebraic_mul(q.im).algebraic_add(p.im.algebraic_mul(q.re)))
}

/// The larger magnitude of a complex number's parts, which [`Product`]'s inverse on complex numbers keeps inside the normal range.
#[inline(always)]
fn complex_size<F: Native>(p: Complex<F>) -> F { p.re.abs().max(p.im.abs()) }

/// [`Product`]'s inverse on complex numbers, which divides as [`crate::divide`] does.
impl<F: Native> Inverse<Complex<F>> for Product {
    #[inline(always)]
    fn inverse(acc: Complex<F>, x: Complex<F>) -> (Complex<F>, bool) { (crate::divide(acc, x), true) }
    #[inline(always)]
    fn undo(a: Complex<F>, b: Complex<F>) -> (Complex<F>, bool) { (crate::divide(a, b), true) }
    #[inline(always)]
    fn fits(q: Complex<F>) -> bool { normal(complex_size(q)) }
    #[inline]
    fn alternating(x: &[Complex<F>]) -> (Complex<F>, bool) {
        quotient(x, Complex::new(F::one(), F::zero()), crate::divide, complex_times, complex_size, |p| p.re.is_finite() & p.im.is_finite())
    }
}

/// Runs `$body` with the type `$t` standing for the reduction that `$r` (a [`Reduce`]) names.
#[macro_export]
macro_rules! with_reduce {
    ($r:expr, |$t:ident| $body:expr) => {
        match $r {
            $crate::reduce::Reduce::Sum => { type $t = $crate::reduce::Sum; $body }
            $crate::reduce::Reduce::Product => { type $t = $crate::reduce::Product; $body }
            $crate::reduce::Reduce::Max => { type $t = $crate::reduce::Max; $body }
            $crate::reduce::Reduce::Min => { type $t = $crate::reduce::Min; $body }
            $crate::reduce::Reduce::And => { type $t = $crate::reduce::And; $body }
            $crate::reduce::Reduce::Or => { type $t = $crate::reduce::Or; $body }
            $crate::reduce::Reduce::Xor => { type $t = $crate::reduce::Xor; $body }
            $crate::reduce::Reduce::Xnor => { type $t = $crate::reduce::Xnor; $body }
        }
    };
}

macro_rules! native_reduction {
    (@inverse $kind:expr, $dtype:expr, |$R:ident, $T:ident| $body:expr) => {{
        use crate::buffer::{DType, FloatWidth};
        match ($dtype, $kind) {
            (DType::Integer(w), Reduce::Sum) => with_width!(w, I => native_reduction!(@at Alternated<Sum>, I, |$R, $T| $body)),
            (DType::Float(FloatWidth::F16 | FloatWidth::F32), Reduce::Sum) => native_reduction!(@at Alternated<Sum>, f32, |$R, $T| $body),
            (DType::Float(FloatWidth::F64), Reduce::Sum) => native_reduction!(@at Alternated<Sum>, f64, |$R, $T| $body),
            (DType::Float(FloatWidth::F16 | FloatWidth::F32), Reduce::Product) => native_reduction!(@at Alternated<ScaledProduct>, f32, |$R, $T| $body),
            (DType::Float(FloatWidth::F64), Reduce::Product) => native_reduction!(@at Alternated<ScaledProduct>, f64, |$R, $T| $body),
            (DType::Complex(w), Reduce::Sum) => with_complex_width!(w, C => native_reduction!(@at Alternated<Sum>, C, |$R, $T| $body)),
            (DType::Complex(w), Reduce::Product) => with_complex_width!(w, C => native_reduction!(@at Alternated<Product>, C, |$R, $T| $body)),
            _ => None,
        }
    }};
    (@at $r:ty, $t:ty, |$R:ident, $T:ident| $body:expr) => {{ type $R = $r; type $T = $t; $body }};
    ($kind:expr, $dtype:expr, $checked:expr, |$R:ident, $T:ident| $body:expr) => {{
        use crate::buffer::DType;
        use crate::reduce::Reduce;
        match ($dtype, $kind) {
            (DType::Boolean, Reduce::Sum) => native_reduction!(@at crate::reduce::SumAs<i64>, bool, |$R, $T| $body),
            (DType::Boolean, Reduce::Product | Reduce::And | Reduce::Min) => native_reduction!(@at crate::reduce::And, bool, |$R, $T| $body),
            (DType::Boolean, Reduce::Or | Reduce::Max) => native_reduction!(@at crate::reduce::Or, bool, |$R, $T| $body),
            (DType::Boolean, Reduce::Xor) => native_reduction!(@at crate::reduce::Xor, bool, |$R, $T| $body),
            (DType::Boolean, Reduce::Xnor) => native_reduction!(@at crate::reduce::Xnor, bool, |$R, $T| $body),
            (DType::Integer(w), Reduce::Sum) if !$checked => with_width!(w, I => native_reduction!(@at crate::Wrapped<crate::reduce::Sum>, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Sum) => with_width!(w, I => native_reduction!(@at crate::reduce::Sum, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Product) if !$checked => with_width!(w, I => native_reduction!(@at crate::Wrapped<crate::reduce::Product>, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Product) => with_width!(w, I => native_reduction!(@at crate::reduce::Product, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Max) => with_width!(w, I => native_reduction!(@at crate::reduce::Max, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Min) => with_width!(w, I => native_reduction!(@at crate::reduce::Min, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::And) => with_width!(w, I => native_reduction!(@at crate::reduce::And, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Or) => with_width!(w, I => native_reduction!(@at crate::reduce::Or, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Xor) => with_width!(w, I => native_reduction!(@at crate::reduce::Xor, I, |$R, $T| $body)),
            (DType::Integer(w), Reduce::Xnor) => with_width!(w, I => native_reduction!(@at crate::reduce::Xnor, I, |$R, $T| $body)),
            (DType::Float(w), Reduce::Sum) => with_float_width!(w, F => native_reduction!(@at crate::reduce::Sum, F, |$R, $T| $body)),
            (DType::Float(w), Reduce::Product) => with_float_width!(w, F => native_reduction!(@at crate::reduce::Product, F, |$R, $T| $body)),
            (DType::Float(w), Reduce::Max) => with_float_width!(w, F => native_reduction!(@at crate::reduce::Max, F, |$R, $T| $body)),
            (DType::Float(w), Reduce::Min) => with_float_width!(w, F => native_reduction!(@at crate::reduce::Min, F, |$R, $T| $body)),
            (DType::Complex(w), Reduce::Sum) => with_complex_width!(w, C => native_reduction!(@at crate::reduce::Sum, C, |$R, $T| $body)),
            (DType::Complex(w), Reduce::Product) => with_complex_width!(w, C => native_reduction!(@at crate::reduce::Product, C, |$R, $T| $body)),
            _ => None,
        }
    }};
}
pub(crate) use native_reduction;

macro_rules! wrapped_products {
    ($($t:ty),+) => {$(
        impl Reduction<$t> for crate::Wrapped<Product> {
            type Total = i64;
            const COMMUTATIVE: bool = true;
            fn unit() -> i64 { 1 }
            fn step(total: i64, x: $t) -> i64 { total.wrapping_mul(x.into()) }
            fn merge(x: i64, y: i64) -> (i64, bool) { (x.wrapping_mul(y), true) }
            fn slice(x: &[$t]) -> (i64, bool) { (folded(x, 1i64, |total, x| total.wrapping_mul(x.into()), i64::wrapping_mul), true) }
            fn scan(x: &[$t], start: Option<i64>, out: &mut Vec<i64>) -> bool {
                grouped(x, start, out, |x| x.into(), i64::wrapping_mul, |x| x, true);
                true
            }
        }
    )+};
}
wrapped_products!(u8, i16, i32, i64);

mod along;
mod pieces;
mod run;
mod scans;
mod sliding;
pub use {along::*, pieces::*, run::*, scans::*, sliding::*};
