//! Reductions and scans over a contiguous slice: sums, products, maximums and minimums, and their running versions. Each compiles
//! inside rustymath once for each item type. A float sum or product may combine the items in any order, which lets its loop
//! vectorize, so its last bits can differ from a sum taken item by item. A float maximum or minimum is NaN where any item is NaN,
//! as [`maximum`] and [`minimum`] are for two floats. An empty slice gives -∞ or ∞. `f16` items compute in `f32` and round once. An
//! integer sum or product computes at `i64`, or at `u64` for `u64` items. Where it doesn't fit, it wraps, and reports that it
//! doesn't fit, as the running versions do.
use crate::{
    float::sealed::{Core, Extremes},
    slice::appended,
    Float, Native, Overflowing,
};
use half::f16;
use num_complex::Complex;

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
/// The least and the greatest item of `x`, in one pass of comparisons and selects. For floats, [`min`] and [`max`] take less time
/// together, because their folds run AArch64's `fmin` and `fmax`.
#[inline]
pub fn extremes<T: Copy>(x: &[T]) -> (T, T)
where
    Min: Reduction<T, Total = T>,
    Max: Reduction<T, Total = T>,
{
    extremes_of(x, |v| v)
}

/// The least and the greatest `key` of the items of `x`, in the pass [`extremes`] makes. A `key` that only reads an item as another
/// type, as `char as u32` does, costs nothing.
#[inline]
pub(crate) fn extremes_of<A: Copy, T: Copy>(x: &[A], key: impl Fn(A) -> T + Copy) -> (T, T)
where
    Min: Reduction<T, Total = T>,
    Max: Reduction<T, Total = T>,
{
    match crate::slice::running(2 * size_of::<T>()) {
        8 => extremes_by::<A, T, 8>(x, key),
        16 => extremes_by::<A, T, 16>(x, key),
        _ => extremes_by::<A, T, 32>(x, key),
    }
}

/// `extremes_of` with `R` running minimums and `R` running maximums, in two arrays: the compiler leaves one array of pairs unvectorized.
#[inline(always)]
fn extremes_by<A: Copy, T: Copy, const R: usize>(x: &[A], key: impl Fn(A) -> T) -> (T, T)
where
    Min: Reduction<T, Total = T>,
    Max: Reduction<T, Total = T>,
{
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
/// Appends the running sums of `x` to `out`, the first being `start` plus the first item where `start` is given, and gives whether
/// every sum fits.
#[inline]
pub fn cumsum<T>(x: &[T], start: Option<<Sum as Reduction<T>>::Total>, out: &mut Vec<<Sum as Reduction<T>>::Total>) -> bool
where
    Sum: Reduction<T>,
{
    Sum::scan(x, start, out)
}
/// Appends the running products of `x` to `out`, as `cumsum` does sums.
#[inline]
pub fn cumprod<T>(x: &[T], start: Option<<Product as Reduction<T>>::Total>, out: &mut Vec<<Product as Reduction<T>>::Total>) -> bool
where
    Product: Reduction<T>,
{
    Product::scan(x, start, out)
}
/// Appends the running maximums of `x` to `out`, the first taking `start` into account where it is given.
#[inline]
pub fn cummax<T>(x: &[T], start: Option<T>, out: &mut Vec<T>) where Max: Reduction<T, Total = T> { Max::scan(x, start, out); }
/// Appends the running minimums of `x` to `out`, as `cummax` does maximums.
#[inline]
pub fn cummin<T>(x: &[T], start: Option<T>, out: &mut Vec<T>) where Min: Reduction<T, Total = T> { Min::scan(x, start, out); }

/// The greater of x and y, or NaN where either is NaN, as [`max`] gives the greatest item. Of +0 and -0, it gives y. NumPy calls it
/// `maximum`.
#[inline(always)]
pub fn maximum<F: Float>(x: F, y: F) -> F { F::from_working(Core::maximum(x.working(), y.working())) }

/// The lesser of x and y, or NaN where either is NaN, as [`min`] gives the least item. Of +0 and -0, it gives y. NumPy calls it
/// `minimum`.
#[inline(always)]
pub fn minimum<F: Float>(x: F, y: F) -> F { F::from_working(Core::minimum(x.working(), y.working())) }

/// [`maximum`] as a function object, which also takes integers and Booleans. The maximum of two Booleans is their `or`. Where every
/// pair shares one `x`, its slice form tests `y` for NaN in place of `x`, with the same results. A loop hoists the test of a shared
/// `x` out of its body. On AArch64 it then rebuilds that one Boolean as a mask for each vector, at a cost of five instructions per
/// vector.
#[derive(Clone, Copy, Debug)]
pub struct Maximum;
forms!(Maximum at real: |_, [x, y]| crate::slice::Total(maximum(x, y)),
    many |_, [x, y], n, out| match x {
        crate::Arg::One(_) => each!([x, y], n, out, |[x, y]| crate::slice::Total(if x <= y || y.is_nan() { y } else { x })),
        crate::Arg::Many(_) => each!([x, y], n, out, |[x, y]| crate::slice::Total(maximum(x, y))),
    });
forms!(Maximum at i8, i16, i32, i64, u8, u16, u32, u64: |_, [x, y]| x.max(y));
forms!(Maximum at bool: |_, [x, y]| x | y);

/// [`minimum`] as a function object, which also takes integers and Booleans. The minimum of two Booleans is their `and`. Its slice
/// form tests `y` for NaN where every pair shares one `x`, as [`Maximum`]'s does.
#[derive(Clone, Copy, Debug)]
pub struct Minimum;
forms!(Minimum at real: |_, [x, y]| crate::slice::Total(minimum(x, y)),
    many |_, [x, y], n, out| match x {
        crate::Arg::One(_) => each!([x, y], n, out, |[x, y]| crate::slice::Total(if x >= y || y.is_nan() { y } else { x })),
        crate::Arg::Many(_) => each!([x, y], n, out, |[x, y]| crate::slice::Total(minimum(x, y))),
    });
forms!(Minimum at i8, i16, i32, i64, u8, u16, u32, u64: |_, [x, y]| x.min(y));
forms!(Minimum at bool: |_, [x, y]| x & y);

/// A reduction by [`Sum`], [`Product`], [`Max`], [`Min`], [`Xor`], [`Xnor`] or [`SumAs`], implemented inside rustymath once for each
/// item type it takes. `slice` folds a whole slice of consecutive items, in any order, `merge` combines the folds of two neighbouring
/// runs, and `scan` gives the running folds. [`along`], [`fold_into`] and [`scan_along`] fold along an axis whose items aren't
/// consecutive. [`Sum`] and [`Product`] take floats and integers, and [`Sum`] also takes complex numbers. [`Max`] and [`Min`] take
/// floats, integers and Booleans, [`Xor`] and [`Xnor`] take Booleans, and [`SumAs`] takes Booleans and integers. An `f16` reduction
/// computes in `f32` and rounds its result, as `f16` arithmetic does.
pub trait Reduction<T> {
    /// The type of a fold: the item type, `i64` for a sum or product of integers narrower than 64 bits, or the `C` of [`SumAs`].
    type Total: Copy;
    /// The fold of no items: 0, 1, -∞ or ∞, the integer type's least or greatest value, or false or true.
    fn unit() -> Self::Total;
    /// `acc` combined with one more item.
    fn step(acc: Self::Total, x: T) -> Self::Total;
    /// `step`, and whether its result fits. An integer sum or product wraps where it doesn't.
    #[inline(always)]
    fn checked(acc: Self::Total, x: T) -> (Self::Total, bool) { (Self::step(acc, x), true) }
    /// The folds of two neighbouring runs of items combined into the fold of both, and whether it fits.
    fn merge(a: Self::Total, b: Self::Total) -> (Self::Total, bool);
    /// The fold of `x` from `unit()`, and whether it fits.
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
/// [`max`] as a [`Reduction`]. Its step on floats is [`maximum`].
#[derive(Clone, Copy, Debug)]
pub struct Max;
/// [`min`] as a [`Reduction`]. Its step on floats is [`minimum`].
#[derive(Clone, Copy, Debug)]
pub struct Min;
/// The fold of Booleans by `!=`: true where an odd number of them are true.
#[derive(Clone, Copy, Debug)]
pub struct Xor;
/// The fold of Booleans by `==`: true where an even number of them are false.
#[derive(Clone, Copy, Debug)]
pub struct Xnor;

/// `x` folded by `add` from `unit`, through as many running results as `slice::running` gives, which `merge` combines at the end.
#[inline(always)]
pub(crate) fn folded<A: Copy, S: Copy>(x: &[A], unit: S, add: impl Fn(S, A) -> S + Copy, merge: impl Fn(S, S) -> S) -> S {
    running_results(x, unit, add, move |parts, rest| rest.iter().fold(parts.iter().fold(unit, |s, &p| merge(s, p)), |s, &v| add(s, v)))
}

/// The running results of `add` from `unit` over `x`, as many as `slice::running` gives. Each takes the items at its own position in
/// every whole group of that many. `finish` gets the results, and the items after the last whole group.
#[inline(always)]
fn running_results<A: Copy, S: Copy, T>(x: &[A], unit: S, add: impl Fn(S, A) -> S, finish: impl FnOnce(&[S], &[A]) -> T) -> T {
    match crate::slice::running(size_of::<S>()) {
        8 => running_by::<A, S, T, 8>(x, unit, add, finish),
        16 => running_by::<A, S, T, 16>(x, unit, add, finish),
        _ => running_by::<A, S, T, 32>(x, unit, add, finish),
    }
}

/// `running_results` with `R` of them. The loop updates them in one local array, which compiles to vector instructions. It indexes
/// each chunk, because `Zip::new` isn't always inlined without LTO.
#[inline(always)]
fn running_by<A: Copy, S: Copy, T, const R: usize>(x: &[A], unit: S, add: impl Fn(S, A) -> S, finish: impl FnOnce(&[S], &[A]) -> T) -> T {
    let (chunks, rest) = x.as_chunks::<R>();
    let mut parts = [unit; R];
    for chunk in chunks {
        for k in 0..R { parts[k] = add(parts[k], chunk[k]) }
    }
    finish(&parts, rest)
}

/// Appends the running totals of `x` by `f` to `out`, the first combining `start` where it is given. `from` reads each item as a total,
/// and `to` writes each total. Each group of four items scans within itself in two steps, then combines with the total before it, so
/// only one operation in each group waits for the group before. `f` may therefore combine items in any order.
#[inline(always)]
fn grouped<A: Copy, S: Copy, O>(x: &[A], start: Option<S>, out: &mut Vec<O>, from: impl Fn(A) -> S, f: impl Fn(S, S) -> S, to: impl Fn(S) -> O) {
    let Some((&first, rest)) = x.split_first() else { return };
    appended(out, x.len(), |slots| {
        let first = from(first);
        let mut total = start.map_or(first, |s| f(s, first));
        slots[0].write(to(total));
        let ((groups, tail), (outs, last)) = (rest.as_chunks::<4>(), slots[1..].as_chunks_mut::<4>());
        for (o, &[a, b, c, d]) in outs.iter_mut().zip(groups) {
            let (a, b, c, d) = (from(a), from(b), from(c), from(d));
            let (ab, cd) = (f(a, b), f(c, d));
            let (abc, abcd) = (f(ab, c), f(ab, cd));
            // Four writes, not `array::map`, which the compiler can leave out of line in a build of many codegen units.
            let totals = [f(total, a), f(total, ab), f(total, abc), f(total, abcd)];
            for k in 0..4 { o[k].write(to(totals[k])); }
            total = f(total, abcd);
        }
        for (o, &v) in last.iter_mut().zip(tail) {
            total = f(total, from(v));
            o.write(to(total));
        }
    })
}

/// Appends the running totals of `x` by `f` to `out`, the first combining `start` where it is given, for an `f` that ignores NaN.
/// Every total from the first NaN on, in `start` or in `x`, is NaN. A first pass finds that NaN, 64 items at a time, so that `f`
/// needs no NaN test.
#[inline]
pub(crate) fn until_nan<F: num_traits::Float>(x: &[F], start: Option<F>, out: &mut Vec<F>, f: impl Fn(F, F) -> F) {
    let nan = if start.is_some_and(|s| s.is_nan()) { 0 } else { first_where(x, |v: F| v.is_nan()) };
    grouped(&x[..nan], start, out, |v| v, f, |t| t);
    out.resize(out.len() + x.len() - nan, F::nan());
}

/// The position of the first item of `x` that `test` picks, or its length where it picks none. Each block of 64 items is tested with
/// no branch, so the test vectorizes.
#[inline]
fn first_where<T: Copy>(x: &[T], test: impl Fn(T) -> bool) -> usize {
    let (blocks, _) = x.as_chunks::<64>();
    let from = blocks.iter().position(|b| b.iter().fold(false, |any, &v| any | test(v))).unwrap_or(blocks.len()) * 64;
    x[from..].iter().position(|&v| test(v)).map_or(x.len(), |i| from + i)
}

/// Each float reduction at `f32` and `f64`: its unit, its step, its fold of a slice, and its scan.
macro_rules! floats {
    ($($r:ident: $unit:ident, |$a:ident, $b:ident| $step:expr, $fold:expr, $scan:expr;)+) => {$(
        impl<F: Native> Reduction<F> for $r {
            type Total = F;
            #[inline(always)]
            fn unit() -> F { F::$unit() }
            #[inline(always)]
            fn step($a: F, $b: F) -> F { $step }
            #[inline(always)]
            fn merge(a: F, b: F) -> (F, bool) { (Self::step(a, b), true) }
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
    Sum: zero, |a, b| a.algebraic_add(b), |x| folded(x, F::zero(), Self::step, Self::step), |x, start, out| grouped(x, start, out, |v| v, |a: F, b: F| a + b, |t| t);
    Product: one, |a, b| a.algebraic_mul(b), |x| folded(x, F::one(), Self::step, Self::step), |x, start, out| grouped(x, start, out, |v| v, |a: F, b: F| a * b, |t| t);
    Max: neg_infinity, |a, b| a.maximum(b), F::fold_max, F::scan_max;
    Min: infinity, |a, b| a.minimum(b), F::fold_min, F::scan_min;
}

/// Each reduction of `f16`s runs the same reduction of `f32`s, and rounds each result once.
impl<R: Reduction<f32, Total = f32>> Reduction<f16> for R {
    type Total = f16;
    #[inline(always)]
    fn unit() -> f16 { f16::from_f32(R::unit()) }
    #[inline(always)]
    fn step(acc: f16, x: f16) -> f16 { f16::from_f32(R::step(acc.to_f32(), x.to_f32())) }
    #[inline(always)]
    fn merge(a: f16, b: f16) -> (f16, bool) { (f16::from_f32(R::merge(a.to_f32(), b.to_f32()).0), true) }
    #[inline]
    fn slice(x: &[f16]) -> (f16, bool) { (f16::from_f32(folded(x, R::unit(), |s, v| R::step(s, v.to_f32()), R::step)), true) }
    #[inline]
    fn scan(x: &[f16], start: Option<f16>, out: &mut Vec<f16>) -> bool {
        grouped(x, start.map(f16::to_f32), out, f16::to_f32, R::step, f16::from_f32);
        true
    }
}

/// Each sum of complex numbers adds their real parts and their imaginary parts, in any order.
impl<F: Native> Reduction<Complex<F>> for Sum {
    type Total = Complex<F>;
    #[inline(always)]
    fn unit() -> Complex<F> { Complex::new(F::zero(), F::zero()) }
    #[inline(always)]
    fn step(acc: Complex<F>, x: Complex<F>) -> Complex<F> { Complex::new(acc.re.algebraic_add(x.re), acc.im.algebraic_add(x.im)) }
    #[inline(always)]
    fn merge(a: Complex<F>, b: Complex<F>) -> (Complex<F>, bool) { (<Self as Reduction<Complex<F>>>::step(a, b), true) }
    #[inline]
    fn slice(x: &[Complex<F>]) -> (Complex<F>, bool) { (folded(x, Complex::new(F::zero(), F::zero()), <Self as Reduction<Complex<F>>>::step, <Self as Reduction<Complex<F>>>::step), true) }
    #[inline]
    fn scan(x: &[Complex<F>], start: Option<Complex<F>>, out: &mut Vec<Complex<F>>) -> bool {
        grouped(x, start, out, |v| v, |a: Complex<F>, b| a + b, |t| t);
        true
    }
}

/// AArch64's `fmax` and `fmin`, and wasm's `max` and `min`, give NaN where either argument is NaN, each in one instruction, where
/// `Core::maximum` takes four. A fold runs their vector forms on `PARTS` running vectors. Each step of a scan depends on the step
/// before. It runs the scalar instruction. In wasm, the step takes lane 0 of the vector instruction on two splats, which the
/// compiler turns into the scalar instruction. The same code on AArch64 keeps a `dup` before each `fmax` at `f32`, which has no
/// one-lane vector type. The step there is inline assembly.
#[cfg(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128")))]
mod lanes {
    use super::*;
    #[cfg(target_arch = "aarch64")]
    use core::arch::{aarch64::*, asm};
    #[cfg(target_arch = "wasm32")]
    use core::arch::wasm32::*;
    /// How many running vectors a fold keeps.
    const PARTS: usize = 8;
    macro_rules! lanes {
        ($($t:ty: $lanes:literal, $splat:ident, $load:ident, $max:ident, $min:ident, $step:ident($arg:tt);)+) => {$(
            impl Extremes for $t {
                fn fold_max(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, NEG_INFINITY, $splat, $load, $max, |a, b| lanes!(@$step $arg, "fmax", $splat, $max, a, b)) }
                fn fold_min(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, INFINITY, $splat, $load, $min, |a, b| lanes!(@$step $arg, "fmin", $splat, $min, a, b)) }
                fn scan_max(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) {
                    grouped(x, start, out, |v| v, #[inline(always)] |a, b| lanes!(@$step $arg, "fmax", $splat, $max, a, b), |t| t)
                }
                fn scan_min(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) {
                    grouped(x, start, out, |v| v, #[inline(always)] |a, b| lanes!(@$step $arg, "fmin", $splat, $min, a, b), |t| t)
                }
            }
        )+};
        (@fold $x:ident, $t:ty, $lanes:literal, $unit:ident, $splat:ident, $load:ident, $op:ident, $step:expr) => {{
            let (chunks, rest) = $x.as_chunks::<{ PARTS * $lanes }>();
            // SAFETY: this module compiles only for targets with these instructions. Each load reads `$lanes` items of one chunk, and
            // a vector holds `$lanes` items of `$t`.
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
    }
    #[cfg(target_arch = "aarch64")]
    lanes! {
        f64: 2, vdupq_n_f64, vld1q_f64, vmaxq_f64, vminq_f64, asm("d");
        f32: 4, vdupq_n_f32, vld1q_f32, vmaxq_f32, vminq_f32, asm("s");
    }
    #[cfg(target_arch = "wasm32")]
    lanes! {
        f64: 2, f64x2_splat, v128_load, f64x2_max, f64x2_min, lane(f64x2_extract_lane);
        f32: 4, f32x4_splat, v128_load, f32x4_max, f32x4_min, lane(f32x4_extract_lane);
    }
}

/// Elsewhere, the folds and scans use `Extremes`' defaults. A fold's loop of `Core::maximum` vectorizes. A scan finds the first NaN
/// before it starts, then compares with no NaN test, which on x86-64 compiles to one `maxsd` or `minsd` for each step. With a NaN
/// test in every step, x86-64's `cummax` cost 4.7 copies on CI, where the Mac's cost 1.0.
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
mod plain {
    impl super::Extremes for f32 {}
    impl super::Extremes for f64 {}
}

/// The sum of 32-bit integers, wrapped to `i64`, and whether it fits. Sums of up to 2^31 of them fit `i64`, which their running
/// results hold.
#[inline(always)]
fn narrow_sum<T: Copy + Into<i64>>(x: &[T]) -> (i64, bool) {
    let total: i128 = x.chunks(1 << 31).map(|c| i128::from(folded(c, 0i64, |s, v| s + v.into(), |a, b| a + b))).sum();
    (total as i64, i64::try_from(total).is_ok())
}

/// The sum of integers of 8 or 16 bits, which always fits `i64`. No block of 2^15 of them overflows an `i32`. Each block sums in
/// `i32`, whose vectors hold twice as many running sums as `i64`'s.
#[inline(always)]
fn small_sum<T: Copy + Into<i32>>(x: &[T]) -> (i64, bool) {
    (x.chunks(1 << 15).map(|c| i64::from(folded(c, 0i32, |s, v| s + v.into(), |a, b| a + b))).sum(), true)
}

/// The sum of 64-bit integers, wrapped, and whether it fits. Each item splits into its high and low 32 bits. Neither half's sum can
/// overflow `i64` within 2^31 items, and both sums vectorize.
#[inline(always)]
fn wide_sum(x: &[i64]) -> (i64, bool) {
    let halves = |c: &[i64]| folded(c, (0i64, 0i64), |(h, l), v| (h + (v >> 32), l + (v & 0xFFFF_FFFF)), |(h, l), (g, m)| (h + g, l + m));
    let total: i128 = x.chunks(1 << 31).map(|c| { let (h, l) = halves(c); (i128::from(h) << 32) + i128::from(l) }).sum();
    (total as i64, i64::try_from(total).is_ok())
}

/// The sum of `u64`s, wrapped, and whether it fits. Each item splits into its high and low 32 bits. Neither half's sum can overflow
/// `u64` within 2^31 items, and both sums vectorize.
#[inline(always)]
fn unsigned_sum(x: &[u64]) -> (u64, bool) {
    let halves = |c: &[u64]| folded(c, (0u64, 0u64), |(h, l), v| (h + (v >> 32), l + (v & 0xFFFF_FFFF)), |(h, l), (g, m)| (h + g, l + m));
    let total: u128 = x.chunks(1 << 31).map(|c| { let (h, l) = halves(c); (u128::from(h) << 32) + u128::from(l) }).sum();
    (total as u64, total <= u128::from(u64::MAX))
}

/// The running sums of `x` from `start` at `i64`, wrapped, and whether every one fits. A sum overflows where its sign differs from
/// both its arguments' signs, as `Overflowing::plus` tests it. The loop gathers those sign bits in one word and tests it once at the
/// end. A bool for each step, as `Overflowing::plus` gives, measured 7% slower.
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

/// The running sums of `x` from `start`, wrapped, and whether every one fits.
#[inline(always)]
fn unsigned_cumsum(x: &[u64], start: u64, out: &mut Vec<u64>) -> bool { checked_scan(x, start, out, u64::plus) }

/// Appends the running totals of `x` from `start` by `op`, which gives each total wrapped and whether it fits, and gives whether every
/// total fits. Each step waits only for the wrapped total before it.
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

/// [`Sum`] and [`Product`] for each integer type, at `$wide`: `i64`, or `u64` for `u64` items. `$sum` sums a slice, and `$cumsum`
/// gives running sums. Each step and product wraps through [`Overflowing`], which also tests whether it fits.
macro_rules! integers {
    ($($t:ty => $wide:ty: $sum:ident, $cumsum:ident),+) => {$(
        impl Reduction<$t> for Sum {
            type Total = $wide;
            #[inline(always)]
            fn unit() -> $wide { 0 }
            #[inline(always)]
            fn step(acc: $wide, x: $t) -> $wide { acc.plus(x.into()).0 }
            #[inline(always)]
            fn checked(acc: $wide, x: $t) -> ($wide, bool) { acc.plus(x.into()) }
            #[inline(always)]
            fn merge(a: $wide, b: $wide) -> ($wide, bool) { a.plus(b) }
            #[inline]
            fn slice(x: &[$t]) -> ($wide, bool) { $sum(x) }
            #[inline]
            fn scan(x: &[$t], start: Option<$wide>, out: &mut Vec<$wide>) -> bool { $cumsum(x, start.unwrap_or(0), out) }
        }
        impl Reduction<$t> for Product {
            type Total = $wide;
            #[inline(always)]
            fn unit() -> $wide { 1 }
            #[inline(always)]
            fn step(acc: $wide, x: $t) -> $wide { acc.times(x.into()).0 }
            #[inline(always)]
            fn checked(acc: $wide, x: $t) -> ($wide, bool) { acc.times(x.into()) }
            #[inline(always)]
            fn merge(a: $wide, b: $wide) -> ($wide, bool) { a.times(b) }
            #[inline]
            fn slice(x: &[$t]) -> ($wide, bool) {
                x.iter().fold((1 as $wide, true), |(p, ok), &v| { let (q, fits) = p.times(v.into()); (q, ok & fits) })
            }
            #[inline]
            fn scan(x: &[$t], start: Option<$wide>, out: &mut Vec<$wide>) -> bool { checked_scan(x, start.unwrap_or(1), out, <$wide>::times) }
        }
    )+};
}
integers!(
    i8 => i64: small_sum, signed_cumsum, i16 => i64: small_sum, signed_cumsum, i32 => i64: narrow_sum, signed_cumsum,
    i64 => i64: wide_sum, signed_cumsum, u8 => i64: small_sum, signed_cumsum, u16 => i64: small_sum, signed_cumsum,
    u32 => i64: narrow_sum, signed_cumsum, u64 => u64: unsigned_sum, unsigned_cumsum
);

/// [`Max`] and [`Min`] for each integer type and Booleans: `$lowest` and `$highest` are their units, and `$max` and `$min` their
/// steps. Their steps on Booleans are `or` and `and`, as [`Maximum`] and [`Minimum`] compute them.
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
                records(x, start, out, $step);
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
extremes!(integers i8, i16, i32, i64, u8, u16, u32, u64);
/// [`Max`], [`Min`], [`Xor`] and [`Xnor`] on Booleans: `or`, `and`, `!=` and `==`. Each one's scan is the scan of its truth table
/// through [`Logical`].
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

/// The running folds of `x` from `start` by `step`, a maximum or minimum. A block of 256 items whose fold leaves the running result
/// unchanged repeats it, with no step for each item. Other blocks scan as `grouped` does. On random items, almost every block repeats
/// the running result. The compiler vectorizes each block's fold because the block's length isn't fixed. A loop of a fixed length
/// unrolls into one item at a time.
fn records<T: Copy + PartialEq>(x: &[T], start: Option<T>, out: &mut Vec<T>, step: impl Fn(T, T) -> T + Copy) {
    let Some(&first) = x.first() else { return };
    let mut total = start.map_or(first, |s| step(s, first));
    for block in x.chunks(256) {
        if block.iter().fold(total, |m, &v| step(m, v)) == total {
            out.resize(out.len() + block.len(), total);
        } else {
            grouped(block, Some(total), out, |v| v, step, |t| t);
            total = out[out.len() - 1];
        }
    }
}

/// The folds by `R` of the lanes along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row: one fold
/// for each of the `outer × inner` lanes, and whether every one fits.
pub fn along<T: Copy, R: Reduction<T>>(data: &[T], shape: [usize; 3]) -> (Vec<R::Total>, bool) {
    let mut totals = vec![R::unit(); shape[0] * shape[2]];
    let fits = fold_into::<T, R>(&mut totals, shape, 0, data);
    (totals, fits)
}

/// Folds `block`, the items from position `start` on of an array of shape `[outer, len, inner]` held row by row, into `totals`, the
/// folds by `R` of the array's lanes along its middle axis so far, which start at `R::unit()`. Gives whether every fold still fits.
/// Along the last axis, a lane's items are consecutive. Whole lanes fold as `fold_rows` folds them, and the pieces of a lane that a
/// block boundary splits fold by `slice` and join by `merge`. Along any other axis, each item steps its lane's fold by `checked`.
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
            let (piece, after) = rest.split_at((inner - k).min(rest.len()));
            for (s, &x) in totals[i * inner + k..].iter_mut().zip(piece) {
                let (t, f) = R::checked(*s, x);
                (*s, fits) = (t, fits & f);
            }
            (rest, k) = (after, k + piece.len());
            if k == inner { (j, k) = (j + 1, 0); }
            if j == len { (i, j) = (i + 1, 0); }
        }
    }
    fits
}

/// Folds each row of `data`, a matrix with `columns` items in each row, by `R` into the matching item of `out`, and gives whether every
/// fold fits. A row of fewer than 16 items folds by `checked`, with a loop of its own for each such length, such as an image's 3
/// channels. A loop of unknown length cost about 2 ns for each row. A longer row folds by `slice`, at the cost of a call and some setup
/// for each row.
#[inline]
fn fold_rows<T: Copy, R: Reduction<T>>(data: &[T], columns: usize, out: &mut [R::Total]) -> bool {
    #[inline(always)]
    fn each<const K: usize, T: Copy, R: Reduction<T>>(data: &[T], out: &mut [R::Total]) -> bool {
        let mut fits = true;
        for (o, row) in out.iter_mut().zip(data.as_chunks::<K>().0) {
            let (t, f) = row.iter().fold((R::unit(), true), |(s, ok), &x| { let (t, f) = R::checked(s, x); (t, ok & f) });
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

/// Appends the running folds by `R` along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to
/// `out`, in the same layout. Each lane's first fold combines `start` where it is given. Gives whether every fold fits. A lane of
/// consecutive items scans by `scan`. Along any other axis, each row's folds step the row before's by `checked`.
pub fn scan_along<T: Copy, R: Reduction<T>>(data: &[T], dims: [usize; 3], start: Option<R::Total>, out: &mut Vec<R::Total>) -> bool {
    let first = start.unwrap_or_else(R::unit);
    rows_scan(data, dims, out, |lane, o| R::scan(lane, start, o), |x| R::checked(first, x), R::checked)
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
    /// Each chunk of 192 items counts into byte totals, 16 to a vector. In wasm, a running 64-bit total doesn't vectorise, and 16-bit
    /// totals cost about 4 times as much as bytes.
    #[inline]
    fn terms<C: Tally>(items: &[bool]) -> C {
        items.chunks(192).fold(C::default(), |s, c| s.add(C::low(c.iter().fold(0u8, |n, &b| n + u8::from(b)).into())))
    }
}
macro_rules! terms {
    ($($t:ty),+) => {$(
        impl Term for $t {
            #[inline(always)]
            fn term<C: Tally>(self) -> C { C::low(self as i64) }
        }
    )+};
}
terms!(i8, i16, i32, i64, u8, u16, u32, u64);

/// An integer type that sums [`Term`]s, wrapping where a sum doesn't fit: `u8`, `i16`, `i32` or `i64`. A caller picks one that holds
/// every sum it needs. A count of fewer than 256 Booleans fits a byte, which takes an eighth of the memory of an `i64`.
pub trait Tally: Copy + Default {
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

/// Sums of [`Term`]s as a [`Reduction`], in the type `C`, which the caller picks to hold every sum, so no step checks whether it fits:
/// counts of the trues among Booleans, or sums of integers.
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
    fn scan(x: &[T], start: Option<C>, out: &mut Vec<C>) -> bool {
        grouped(x, start, out, T::term, C::add, |t| t);
        true
    }
}

/// The right fold of a slice by the inverse of a reduction, through pairs of items: `-` for [`Sum`], and `÷` for [`Product`]. The
/// fold `x0 - (x1 - (x2 - …))` is the sum of the pairs' differences, and `x0 ÷ (x1 ÷ (x2 ÷ …))` is the product of the pairs' ratios,
/// each with the last item where it has no pair. Both combine the pairs in any order, as sums and products do.
pub trait Alternating<T> {
    /// The type of the fold: the item type, or `i64` for integers.
    type Total;
    /// The right fold of `x` by the inverse, 0 or 1 for an empty slice, and whether it fits. An integer fold fits where it is
    /// within `i64`. A quotient of floats doesn't fit where a running product or the result reaches zero, a subnormal number, an
    /// infinity or NaN. The right fold taken item by item may still be finite there.
    fn alternating(x: &[T]) -> (Self::Total, bool);
}

/// The fold by `merge` from `unit` of the pairs of `x`, each combined into a total by `pair`, and then of the item left over where `x`
/// has an odd length, combined by `last`.
#[inline(always)]
fn pairwise<A: Copy, S: Copy>(x: &[A], unit: S, pair: impl Fn(S, A, A) -> S, merge: impl Fn(S, S) -> S, last: impl Fn(S, A) -> S) -> S {
    let (pairs, rest) = x.as_chunks::<2>();
    let total = folded(pairs, unit, |s, [a, b]| pair(s, a, b), merge);
    rest.first().map_or(total, |&z| last(total, z))
}

/// The product of the pairs' ratios of `x` by `divide` and `times`, with whether every running product and the result stay inside the
/// normal range. A product that falls below that range stays out of it, or comes back with fewer significant bits, so each step keeps
/// the least `size` of the running products. A product that overflows stays infinite or turns NaN, and a NaN stays NaN, so `finite`
/// catches both in the result.
#[inline(always)]
fn quotient<T: Copy, R: Native>(x: &[T], one: T, divide: impl Fn(T, T) -> T, times: impl Fn(T, T) -> T + Copy, size: impl Fn(T) -> R + Copy, finite: impl Fn(T) -> bool) -> (T, bool) {
    let step = move |(p, low): (T, R), q: T| { let r = times(p, q); (r, low.min(size(r))) };
    let (total, low) = pairwise(x, (one, R::one()), |s, a, b| step(s, divide(a, b)), move |s, (q, l)| { let (r, m) = step(s, q); (r, m.min(l)) }, step);
    (total, low >= R::min_positive_value() && finite(total))
}

/// The sum by `add` and `merge` of the items of `x` at even positions, less the sum of those at odd positions. There is an even number
/// of running results, so each one takes the items of one parity.
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

/// [`Sum`]'s alternating folds of integers: the sum of the items at even positions less the sum of those at odd positions, which is
/// exact. Each run of `$run` items, an even number, sums at `$acc`, where it can't overflow, and the runs total at `i128`. Bytes and
/// 16-bit integers sum at `i32`, which fits twice as many in each vector as `i64`. Separate running sums for each parity take less time
/// than `pairwise`, which must split each pair out of the vector lanes.
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
alternating_integers!(i8 => i32, 1 << 15; i16 => i32, 1 << 15; u8 => i32, 1 << 15; u16 => i32, 1 << 15; i32 => i64, 1 << 31; u32 => i64, 1 << 31; i64 => i128, 1 << 31);

impl<F: Native> Alternating<F> for Product {
    type Total = F;
    #[inline]
    fn alternating(x: &[F]) -> (F, bool) {
        quotient(x, F::one(), |a, b| a / b, F::algebraic_mul, F::abs, F::is_finite)
    }
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

/// Appends the running results along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`, in
/// the same layout, and gives whether every one fits. A lane of consecutive items runs through `lane`. Along any other axis, `first`
/// gives each lane's first row, and `step` steps each row from the row before.
#[inline(always)]
fn rows_scan<T: Copy, S: Copy>(
    data: &[T], [_, len, inner]: [usize; 3], out: &mut Vec<S>, lane: impl Fn(&[T], &mut Vec<S>) -> bool, first: impl Fn(T) -> (S, bool),
    step: impl Fn(S, T) -> (S, bool),
) -> bool {
    if data.is_empty() { return true; }
    if inner == 1 { return data.chunks(len).fold(true, |fits, l| lane(l, out) & fits); }
    let (mut totals, mut fits) = (Vec::with_capacity(inner), true);
    out.reserve(data.len());
    for l in data.chunks(len * inner) {
        let (head, rows) = l.split_at(inner);
        totals.clear();
        totals.extend(head.iter().map(|&x| { let (t, f) = first(x); fits &= f; t }));
        out.extend_from_slice(&totals);
        for row in rows.chunks_exact(inner) {
            for (s, &x) in totals.iter_mut().zip(row) {
                let (t, f) = step(*s, x);
                (*s, fits) = (t, fits & f);
            }
            out.extend_from_slice(&totals);
        }
    }
    fits
}

/// A function of two Booleans, held as its truth table: `Logical(t)` gives `t[r][x]` for the running result `r` and the next item `x`.
/// Its scans run any such function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Logical(pub [[bool; 2]; 2]);

impl Logical {
    /// `r f x`, with no branch.
    #[inline(always)]
    fn apply(self, r: bool, x: bool) -> bool {
        let [[f00, f01], [f10, f11]] = self.0;
        (r & ((x & f11) | (!x & f10))) | (!r & ((x & f01) | (!x & f00)))
    }

    /// Appends the running results of `x` to `out`: the first item, or `start` combined with it where `start` is given, then each
    /// result combined with the next item.
    ///
    /// For each value of the next item, the function maps the running result to itself, to its negation, or to a constant. Where one
    /// value maps it to itself and the other to a constant, as for `and` and `or`, the results change at most once. `first_where`
    /// finds that place, and two runs fill the rest. Otherwise each map is `r ↦ (s ∧ r) ≠ t` for two bits `s` and `t`, and the maps
    /// compose as `r ↦ (s₂ ∧ s₁ ∧ r) ≠ ((s₂ ∧ t₁) ≠ t₂)`. Each block of 64 items packs into a word of bits, whose maps compose in six
    /// steps of shifts, with no branch and no step for each item.
    pub fn scan(self, x: &[bool], start: Option<bool>, out: &mut Vec<bool>) {
        let Some((&head, rest)) = x.split_first() else { return };
        let first = start.map_or(head, |s| self.apply(s, head));
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
        let word = |b: bool| if b { u64::MAX } else { 0 };
        let ((t0, s0), (t1, s1)) = ((word(maps[0][0]), word(maps[0][0] != maps[0][1])), (word(maps[1][0]), word(maps[1][0] != maps[1][1])));
        let affine = move |bits: u64| ((bits & s1) | (!bits & s0), (bits & t1) | (!bits & t0));
        let mut carry = first;
        appended(out, rest.len(), |slots| {
            let ((blocks, tail), (outs, last)) = (rest.as_chunks::<64>(), slots.as_chunks_mut::<64>());
            for (o, block) in outs.iter_mut().zip(blocks) {
                let (s, t) = affine(pack(block));
                let r = composed(s, t, carry);
                for (k, slot) in o.iter_mut().enumerate() { slot.write(r >> k & 1 != 0); }
                carry = r >> 63 != 0;
            }
            if !tail.is_empty() {
                let mut block = [false; 64];
                block[..tail.len()].copy_from_slice(tail);
                let (s, t) = affine(pack(&block));
                let r = composed(s, t, carry);
                for (k, slot) in last.iter_mut().enumerate() { slot.write(r >> k & 1 != 0); }
            }
        });
    }

    /// Appends the running results along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`,
    /// in the same layout, as [`scan_along`] does for a [`Reduction`]. Each lane's first result combines `start` where it is given.
    pub fn scan_along(self, data: &[bool], dims: [usize; 3], start: Option<bool>, out: &mut Vec<bool>) {
        rows_scan(data, dims, out, |l, o| { self.scan(l, start, o); true }, |x| (start.map_or(x, |s| self.apply(s, x)), true), |r, x| (self.apply(r, x), true));
    }
}

/// The 64 Booleans of `block` as the bits of a word, the first in the lowest bit. Each 8 Booleans, one byte each, gather into a byte
/// through one multiplication.
#[inline(always)]
fn pack(block: &[bool; 64]) -> u64 {
    let mut bits = 0;
    for (k, b) in block.as_chunks::<8>().0.iter().enumerate() {
        let bytes = (0..8).fold(0u64, |w, i| w | u64::from(b[i]) << (8 * i));
        bits |= (bytes.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
    }
    bits
}

/// The results of 64 items, bit by bit, from the running result `carry`, where item `k`'s map is `r ↦ (s ∧ r) ≠ t` for bit `k` of `s`
/// and `t`. Each step composes each bit's map with the map it holds from the bits before it. A bit with none before it takes the identity
/// map, whose `s` is 1 and whose `t` is 0.
#[inline(always)]
fn composed(mut s: u64, mut t: u64, carry: bool) -> u64 {
    for k in [1, 2, 4, 8, 16, 32] {
        t ^= s & (t << k);
        s &= (s << k) | ((1u64 << k) - 1);
    }
    (s & if carry { u64::MAX } else { 0 }) ^ t
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Each reduction and scan against the same taken item by item, in order: sums and running sums of `sums`, and the product of
    /// `prods`. Their items keep every sum and product exact at every width, so any grouping gives the same result.
    fn exact<T: Copy + PartialEq + std::fmt::Debug, S: Copy + PartialEq + std::fmt::Debug>(
        sums: &[T], prods: &[T], add: impl Fn(S, T) -> Option<S>, mul: impl Fn(S, T) -> Option<S>, zero: S, one: S,
    ) where
        Sum: Reduction<T, Total = S>,
        Product: Reduction<T, Total = S>,
    {
        let mut total = Some(zero);
        let running: Vec<_> = sums.iter().map(|&v| { total = total.and_then(|t| add(t, v)); total.unwrap() }).collect();
        assert_eq!(sum(sums), (total.unwrap(), true));
        let mut out = Vec::new();
        assert!(cumsum(sums, None, &mut out));
        assert_eq!(out, running);
        assert_eq!(prod(prods), (prods.iter().try_fold(one, |t, &v| mul(t, v)).unwrap(), true));
    }

    #[test]
    fn folds_match_item_by_item() {
        // 101 items cover the running results, the remainder, and the scan's groups of four and its tail. Ten 2s keep the product
        // within f16.
        let sums: Vec<i64> = (0..101).map(|i| (i * 7919) % 23 - 11).collect();
        let prods: Vec<i64> = (0..101).map(|i| if i % 10 == 3 { 2 } else if i % 3 == 0 { -1 } else { 1 }).collect();
        let at = |f: fn(i64) -> f64| (sums.iter().map(|&v| f(v)).collect::<Vec<_>>(), prods.iter().map(|&v| f(v)).collect::<Vec<_>>());
        exact(&sums, &prods, |t, v| t.checked_add(v), |t, v| t.checked_mul(v), 0, 1);
        let (s, p): (Vec<i8>, Vec<i8>) = (sums.iter().map(|&v| v as i8).collect(), prods.iter().map(|&v| v as i8).collect());
        exact(&s, &p, |t, v| t.checked_add(v.into()), |t, v| t.checked_mul(v.into()), 0, 1);
        let (s, p) = at(|v| v as f64);
        exact(&s, &p, |t, v| Some(t + v), |t, v| Some(t * v), 0.0, 1.0);
        let (s, p): (Vec<f32>, Vec<f32>) = (s.iter().map(|&v| v as f32).collect(), p.iter().map(|&v| v as f32).collect());
        exact(&s, &p, |t, v| Some(t + v), |t, v| Some(t * v), 0.0, 1.0);
        let (s, p): (Vec<f16>, Vec<f16>) = (s.iter().map(|&v| f16::from_f32(v)).collect(), p.iter().map(|&v| f16::from_f32(v)).collect());
        let narrow = |f: fn(f32, f32) -> f32| move |t: f16, v: f16| Some(f16::from_f32(f(t.to_f32(), v.to_f32())));
        exact(&s, &p, narrow(|t, v| t + v), narrow(|t, v| t * v), f16::ZERO, f16::ONE);
        assert_eq!(extremes(&sums), (*sums.iter().min().unwrap(), *sums.iter().max().unwrap()));
        let mut maxima = Vec::new();
        cummax(&sums, Some(-5), &mut maxima);
        assert_eq!(maxima, sums.iter().scan(-5, |m, &v| { *m = (*m).max(v); Some(*m) }).collect::<Vec<_>>());
        assert_eq!((max(&sums), min(&sums)), (*sums.iter().max().unwrap(), *sums.iter().min().unwrap()));
        // Complex sums of whole numbers are exact in any order.
        let z: Vec<Complex<f64>> = (0..101).map(|k| Complex::new(k as f64, -(k as f64) / 2.0)).collect();
        assert_eq!(sum(&z), (z.iter().fold(Complex::new(0.0, 0.0), |s, &v| s + v), true));
        let mut out = Vec::new();
        assert!(cumsum(&z, Some(Complex::new(1.0, 1.0)), &mut out));
        assert_eq!(out, z.iter().scan(Complex::new(1.0, 1.0), |s, &v| { *s += v; Some(*s) }).collect::<Vec<_>>());
    }

    #[test]
    fn edges() {
        assert_eq!((sum::<f64>(&[]), prod::<i32>(&[]), max::<f32>(&[]), min::<i16>(&[])), ((0.0, true), (1, true), f32::NEG_INFINITY, i16::MAX));
        // NaN at item 40 of 101 lies inside a fold's running results, which vectorize, and not in the remainder.
        let mut v = vec![1.0f64; 101];
        v[40] = f64::NAN;
        let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        assert_eq!((max(&v).is_nan(), min(&v32).is_nan(), max(&[f16::ONE, f16::NAN]).is_nan()), (true, true, true));
        assert_eq!((maximum(1.0, f64::NAN).is_nan(), minimum(f32::NAN, 1.0).is_nan(), maximum(-1.0, 2.0), minimum(f16::ONE, f16::ZERO)), (true, true, 2.0, f16::ZERO));
        // With a shared `x`, the slice forms test the other argument for NaN. They still give `maximum`'s and `minimum`'s results.
        let e = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 0.0, -0.0, 1.5, -2.0];
        let same = |a: f64, b: f64| a.to_bits() == b.to_bits() || a.is_nan() && b.is_nan();
        for &c in &e {
            let (mut most, mut least) = (Vec::new(), Vec::new());
            assert!(crate::Binary::many(&Maximum, crate::Arg::One(c), crate::Arg::Many(&e), e.len(), &mut most));
            assert!(crate::Binary::many(&Minimum, crate::Arg::One(c), crate::Arg::Many(&e), e.len(), &mut least));
            for (k, &y) in e.iter().enumerate() { assert!(same(most[k], maximum(c, y)) && same(least[k], minimum(c, y)), "{c} and {y}"); }
        }
        let mut out = Vec::new();
        cummax(&[1.0, 3.0, f64::NAN, 5.0, 2.0, 1.0], None, &mut out);
        assert_eq!(out.iter().map(|v| v.is_nan()).collect::<Vec<_>>(), [false, false, true, true, true, true]);
        assert_eq!((sum(&[i64::MAX, 1]), sum(&[i64::MAX, 1, -2]), prod(&[i64::MAX, 2])), ((i64::MIN, false), (i64::MAX - 1, true), (-2, false)));
        assert_eq!((sum(&[u64::MAX, 1]), prod(&[1u64 << 32, 1 << 32]), sum(&[u64::MAX, 0])), ((0, false), (0, false), (u64::MAX, true)));
        let mut out = Vec::new();
        assert!(!cumsum(&[u64::MAX, 1, 5], None, &mut out));
        assert_eq!(out, [u64::MAX, 0, 5]);
        assert!(!cumsum(&[i64::MAX, 1], None, &mut Vec::new()));
        assert!(!cumprod(&[i32::MAX, i32::MAX, i32::MAX], None, &mut Vec::new()));
        let mut out = vec![7.0];
        assert!(cumsum(&[1.0f64, 2.0], Some(10.0), &mut out));
        assert_eq!(out, [7.0, 11.0, 13.0]);
    }

    /// The scan that x86-64 runs, which every target can call: NaN from the first NaN on, in `start` or in the items, wherever it falls
    /// in the blocks of 64 that the first pass reads, and the running maximum before it.
    #[test]
    fn scans_until_the_first_nan() {
        let x: Vec<f64> = (0..200).map(|i| (i * 37 % 101) as f64 - 50.0).collect();
        for nan in [None, Some(0), Some(3), Some(63), Some(64), Some(130), Some(199)] {
            for start in [None, Some(10.0), Some(f64::NAN)] {
                let mut y = x.clone();
                if let Some(k) = nan { y[k] = f64::NAN }
                let mut out = Vec::new();
                until_nan(&y, start, &mut out, |t, v| if t > v { t } else { v });
                let mut total = start;
                let want = y.iter().map(|&v| { let t = total.map_or(v, |t| maximum(t, v)); total = Some(t); t });
                assert!(out.len() == y.len() && out.iter().zip(want).all(|(&a, b)| a == b || a.is_nan() && b.is_nan()), "{nan:?} {start:?}");
            }
        }
    }

    /// Every lane of an array of shape `[outer, len, inner]`, its items in order.
    fn lanes_of<T: Copy>(x: &[T], [outer, len, inner]: [usize; 3]) -> Vec<Vec<T>> {
        (0..outer).flat_map(|i| (0..inner).map(move |k| (0..len).map(|j| x[(i * len + j) * inner + k]).collect())).collect()
    }

    /// Folds and scans along the middle axis against the same taken lane by lane, item by item, for lanes of consecutive items and
    /// otherwise, short and long, and empty. Folding in blocks of 13 splits lanes at many offsets. Booleans fold by each of their
    /// reductions, and an `i64` sum that wraps reports it, along a lane and down the rows.
    #[test]
    fn axes_match_lane_by_lane() {
        for shape in [[3, 37, 5], [15, 37, 1], [4, 3, 2], [2, 0, 3], [2, 200, 1]] {
            let [outer, len, inner] = shape;
            let x: Vec<i64> = (0..outer * len * inner).map(|i| (i * 37 % 101) as i64 - 50).collect();
            let sums: Vec<i64> = lanes_of(&x, shape).iter().map(|l| l.iter().sum()).collect();
            assert_eq!(along::<i64, Sum>(&x, shape), (sums.clone(), true));
            let mut totals = vec![0; sums.len()];
            for (b, block) in x.chunks(13).enumerate() { assert!(fold_into::<i64, Sum>(&mut totals, shape, b * 13, block)) }
            assert_eq!(totals, sums, "{shape:?}");
            let mut want = vec![0; x.len()];
            for i in 0..outer {
                for k in 0..inner {
                    (0..len).fold(10, |t, j| { let p = (i * len + j) * inner + k; want[p] = t + x[p]; want[p] });
                }
            }
            let mut out = Vec::new();
            assert!(scan_along::<i64, Sum>(&x, shape, Some(10), &mut out));
            assert_eq!(out, want, "{shape:?}");
            // `SumAs`'s running sums of Booleans and of narrow integers match `Sum`'s, which run at 64 bits.
            let wide = |items: Vec<i64>, start: Option<i64>| { let mut out = Vec::new(); scan_along::<i64, Sum>(&items, shape, start, &mut out); out.into_iter().map(|v| v as i32).collect::<Vec<_>>() };
            let mut counts = Vec::new();
            scan_along::<bool, SumAs<i32>>(&x.iter().map(|v| v % 3 == 0).collect::<Vec<_>>(), shape, Some(3), &mut counts);
            assert_eq!(counts, wide(x.iter().map(|v| i64::from(v % 3 == 0)).collect(), Some(3)), "{shape:?}");
            let mut sums = Vec::new();
            scan_along::<i16, SumAs<i32>>(&x.iter().map(|&v| v as i16).collect::<Vec<_>>(), shape, None, &mut sums);
            assert_eq!(sums, wide(x.clone(), None), "{shape:?}");
            let b: Vec<bool> = x.iter().map(|v| v % 3 == 0).collect();
            let lanes = lanes_of(&b, shape);
            let ones = |l: &Vec<bool>| l.iter().filter(|&&v| v).count();
            assert_eq!(along::<bool, Max>(&b, shape).0, lanes.iter().map(|l| l.contains(&true)).collect::<Vec<_>>());
            assert_eq!(along::<bool, Min>(&b, shape).0, lanes.iter().map(|l| !l.contains(&false)).collect::<Vec<_>>());
            assert_eq!(along::<bool, Xor>(&b, shape).0, lanes.iter().map(|l| ones(l) % 2 == 1).collect::<Vec<_>>());
            assert_eq!(along::<bool, Xnor>(&b, shape).0, lanes.iter().map(|l| (l.len() - ones(l)) % 2 == 0).collect::<Vec<_>>());
            assert_eq!(along::<bool, SumAs<u8>>(&b, shape).0, lanes.iter().map(|l| ones(l) as u8).collect::<Vec<_>>());
            for start in [None, Some(false), Some(true)] {
                for code in 0..16 {
                    let table = Logical([[code & 1 != 0, code & 2 != 0], [code & 4 != 0, code & 8 != 0]]);
                    let mut want = b.clone();
                    for i in 0..outer {
                        for k in 0..inner {
                            (0..len).fold(start, |t, j| {
                                let p = (i * len + j) * inner + k;
                                want[p] = t.map_or(b[p], |t| table.0[usize::from(t)][usize::from(b[p])]);
                                Some(want[p])
                            });
                        }
                    }
                    let mut out = Vec::new();
                    table.scan_along(&b, shape, start, &mut out);
                    assert_eq!(out, want, "{shape:?} {start:?} {code}");
                    // The Boolean reductions scan by their tables: `or` is 14, `and` 8, `!=` 6 and `==` 9.
                    let mut reduced = Vec::new();
                    match code {
                        14 => scan_along::<bool, Max>(&b, shape, start, &mut reduced),
                        8 => scan_along::<bool, Min>(&b, shape, start, &mut reduced),
                        6 => scan_along::<bool, Xor>(&b, shape, start, &mut reduced),
                        9 => scan_along::<bool, Xnor>(&b, shape, start, &mut reduced),
                        _ => continue,
                    };
                    assert_eq!(reduced, want, "{shape:?} {start:?} {code}");
                }
            }
        }
        assert!(!along::<i64, Sum>(&[i64::MAX, 1], [1, 2, 1]).1);
        assert!(!along::<i64, Sum>(&[i64::MAX, 0, 1, 0], [1, 2, 2]).1);
        assert!(!scan_along::<i64, Sum>(&[i64::MAX, 0, 1, 0], [1, 2, 2], None, &mut Vec::new()));
    }

    #[test]
    fn alternating_folds_match_the_right_fold() {
        fn right<T: Copy>(x: &[T], unit: T, f: impl Fn(T, T) -> T) -> T { x.iter().rev().copied().reduce(|t, v| f(v, t)).unwrap_or(unit) }
        for n in [0, 1, 2, 3, 64, 101] {
            let x: Vec<i64> = (0..n).map(|i| (i * 7919) % 23 - 11).collect();
            let want = right(&x, 0, |a, b| a - b);
            assert_eq!(<Sum as Alternating<i64>>::alternating(&x), (want, true));
            assert_eq!(<Sum as Alternating<i16>>::alternating(&x.iter().map(|&v| v as i16).collect::<Vec<_>>()), (want, true));
            // Whole numbers, and powers of 2, keep every sum and quotient exact in any order.
            let f: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            assert_eq!(<Sum as Alternating<f64>>::alternating(&f), (want as f64, true));
            let z: Vec<Complex<f64>> = f.iter().map(|&v| Complex::new(v, -v)).collect();
            assert_eq!(<Sum as Alternating<Complex<f64>>>::alternating(&z), (Complex::new(want as f64, -want as f64), true));
            let p: Vec<f64> = x.iter().map(|&v| 2f64.powi(v as i32 % 5)).collect();
            let q = right(&p, 1.0, |a, b| a / b);
            assert_eq!(<Product as Alternating<f64>>::alternating(&p), (q, true));
            let pz: Vec<Complex<f64>> = p.iter().map(|&v| Complex::new(v, 0.0)).collect();
            assert_eq!(<Product as Alternating<Complex<f64>>>::alternating(&pz), (Complex::new(q, 0.0), true));
        }
        assert!(!<Sum as Alternating<i64>>::alternating(&[i64::MAX, -1]).1);
        // The pairs' ratios alternate between 1e200 and 1e-200, so the right fold is 1. Each running product takes every 16th ratio, or
        // every 8th in wasm, so half of them overflow and half underflow.
        let x: Vec<f64> = (0..256).map(|i| if (i % 4 == 0) | (i % 4 == 3) { 1e200 } else { 1.0 }).collect();
        assert_eq!(right(&x, 1.0, |a, b| a / b), 1.0);
        assert!(!<Product as Alternating<f64>>::alternating(&x).1);
        assert!(!<Product as Alternating<f64>>::alternating(&[1.0, 0.0]).1);
    }
}
