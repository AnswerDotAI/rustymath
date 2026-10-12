//! Reductions and scans over a contiguous slice: sums, products, maximums and minimums, and their running versions. Each compiles
//! inside rustymath once for each item type. A float sum or product may combine the items in any order, which lets its loop
//! vectorize, so its last bits can differ from a sum taken item by item. A float maximum or minimum is NaN where any item is NaN,
//! as [`maximum`] and [`minimum`] are for two floats. An empty slice gives -∞ or ∞. `f16` items compute in `f32` and round once. An
//! integer sum or product is exact at `i64`, and is `None` where it doesn't fit.
use crate::{
    float::sealed::{Core, Item, Sealed},
    slice::appended,
    Float, Native,
};
use half::f16;

/// An item type that rustymath reduces and scans: `f16`, `f32`, `f64`, a signed integer type, `u8`, `u16` or `u32`. Only rustymath
/// implements it, and the functions of this module call it.
pub trait Fold: Item + Sealed {
    /// The type of a sum or product: the item type for a float, and `i64` for an integer.
    type Sum: Item;
    fn slice_sum(x: &[Self]) -> Option<Self::Sum>;
    fn slice_prod(x: &[Self]) -> Option<Self::Sum>;
    fn slice_max(x: &[Self]) -> Self;
    fn slice_min(x: &[Self]) -> Self;
    fn slice_cumsum(x: &[Self], start: Option<Self::Sum>, out: &mut Vec<Self::Sum>) -> bool;
    fn slice_cumprod(x: &[Self], start: Option<Self::Sum>, out: &mut Vec<Self::Sum>) -> bool;
    fn slice_cummax(x: &[Self], start: Option<Self>, out: &mut Vec<Self>);
    fn slice_cummin(x: &[Self], start: Option<Self>, out: &mut Vec<Self>);
}

/// The sum of `x`, 0 for an empty slice.
#[inline]
pub fn sum<T: Fold>(x: &[T]) -> Option<T::Sum> { T::slice_sum(x) }
/// The product of `x`, 1 for an empty slice.
#[inline]
pub fn prod<T: Fold>(x: &[T]) -> Option<T::Sum> { T::slice_prod(x) }
/// The greatest item of `x`.
#[inline]
pub fn max<T: Fold>(x: &[T]) -> T { T::slice_max(x) }
/// The least item of `x`.
#[inline]
pub fn min<T: Fold>(x: &[T]) -> T { T::slice_min(x) }
/// Appends the running sums of `x` to `out`, the first being `start` plus the first item where `start` is given, and gives whether
/// every sum fits.
#[inline]
pub fn cumsum<T: Fold>(x: &[T], start: Option<T::Sum>, out: &mut Vec<T::Sum>) -> bool { T::slice_cumsum(x, start, out) }
/// Appends the running products of `x` to `out`, as `cumsum` does sums.
#[inline]
pub fn cumprod<T: Fold>(x: &[T], start: Option<T::Sum>, out: &mut Vec<T::Sum>) -> bool { T::slice_cumprod(x, start, out) }
/// Appends the running maximums of `x` to `out`, the first taking `start` into account where it is given.
#[inline]
pub fn cummax<T: Fold>(x: &[T], start: Option<T>, out: &mut Vec<T>) { T::slice_cummax(x, start, out) }
/// Appends the running minimums of `x` to `out`, as `cummax` does maximums.
#[inline]
pub fn cummin<T: Fold>(x: &[T], start: Option<T>, out: &mut Vec<T>) { T::slice_cummin(x, start, out) }

/// The greater of x and y, or NaN where either is NaN, as [`max`] gives the greatest item. Of +0 and -0, it gives y. NumPy calls it
/// `maximum`.
#[inline(always)]
pub fn maximum<F: Float>(x: F, y: F) -> F { F::from_working(Core::maximum(x.working(), y.working())) }

/// The lesser of x and y, or NaN where either is NaN, as [`min`] gives the least item. Of +0 and -0, it gives y. NumPy calls it
/// `minimum`.
#[inline(always)]
pub fn minimum<F: Float>(x: F, y: F) -> F { F::from_working(Core::minimum(x.working(), y.working())) }

/// [`maximum`] as a function object, which also takes integers.
#[derive(Clone, Copy, Debug)]
pub struct Maximum;
forms!(Maximum at real: |_, [x, y]| crate::slice::Total(maximum(x, y)));
forms!(Maximum at i8, i16, i32, i64, u8, u16, u32, u64: |_, [x, y]| x.max(y));

/// [`minimum`] as a function object, which also takes integers.
#[derive(Clone, Copy, Debug)]
pub struct Minimum;
forms!(Minimum at real: |_, [x, y]| crate::slice::Total(minimum(x, y)));
forms!(Minimum at i8, i16, i32, i64, u8, u16, u32, u64: |_, [x, y]| x.min(y));

/// How many running results a fold keeps along its slice: 32, which measured faster on the Mac than 8 or 16 for every item type.
const RUNNING: usize = 32;

/// `x` folded by `add` from `unit`, through `RUNNING` running results that `merge` combines at the end. The loop updates them in one
/// local array, which compiles to vector instructions. It indexes each chunk, because `Zip::new` isn't always inlined without LTO.
#[inline(always)]
fn folded<A: Copy, S: Copy>(x: &[A], unit: S, add: impl Fn(S, A) -> S, merge: impl Fn(S, S) -> S) -> S {
    let (chunks, rest) = x.as_chunks::<RUNNING>();
    let mut parts = [unit; RUNNING];
    for chunk in chunks {
        for k in 0..RUNNING { parts[k] = add(parts[k], chunk[k]) }
    }
    rest.iter().fold(parts.into_iter().fold(unit, merge), |s, &v| add(s, v))
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

macro_rules! native {
    ($($t:ty),+) => {$(
        impl Fold for $t {
            type Sum = $t;
            fn slice_sum(x: &[$t]) -> Option<$t> { Some(folded(x, 0.0, Native::algebraic_add, Native::algebraic_add)) }
            fn slice_prod(x: &[$t]) -> Option<$t> { Some(folded(x, 1.0, Native::algebraic_mul, Native::algebraic_mul)) }
            fn slice_max(x: &[$t]) -> $t { <$t>::fold_max(x) }
            fn slice_min(x: &[$t]) -> $t { <$t>::fold_min(x) }
            fn slice_cumsum(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) -> bool {
                grouped(x, start, out, |v| v, |a, b| a + b, |t| t);
                true
            }
            fn slice_cumprod(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) -> bool {
                grouped(x, start, out, |v| v, |a, b| a * b, |t| t);
                true
            }
            fn slice_cummax(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) { grouped(x, start, out, |v| v, <$t>::step_max, |t| t) }
            fn slice_cummin(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) { grouped(x, start, out, |v| v, <$t>::step_min, |t| t) }
        }
    )+};
}
native!(f32, f64);

/// The folds of `f32` and `f64` by maximum and minimum, and the step of their scans. Their defaults run through `Core::maximum` and
/// `minimum`, whose loops vectorize.
trait Extremes: Native {
    fn fold_max(x: &[Self]) -> Self { folded(x, Self::neg_infinity(), Core::maximum, Core::maximum) }
    fn fold_min(x: &[Self]) -> Self { folded(x, Self::infinity(), Core::minimum, Core::minimum) }
    #[inline(always)]
    fn step_max(x: Self, y: Self) -> Self { Core::maximum(x, y) }
    #[inline(always)]
    fn step_min(x: Self, y: Self) -> Self { Core::minimum(x, y) }
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
                fn fold_max(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, NEG_INFINITY, $splat, $load, $max, step_max) }
                fn fold_min(x: &[$t]) -> $t { lanes!(@fold x, $t, $lanes, INFINITY, $splat, $load, $min, step_min) }
                #[inline(always)]
                fn step_max(x: $t, y: $t) -> $t { lanes!(@$step $arg, "fmax", $splat, $max, x, y) }
                #[inline(always)]
                fn step_min(x: $t, y: $t) -> $t { lanes!(@$step $arg, "fmin", $splat, $min, x, y) }
            }
        )+};
        (@fold $x:ident, $t:ty, $lanes:literal, $unit:ident, $splat:ident, $load:ident, $op:ident, $step:ident) => {{
            let (chunks, rest) = $x.as_chunks::<{ PARTS * $lanes }>();
            // SAFETY: this module compiles only for targets with these instructions. Each load reads `$lanes` items of one chunk, and
            // a vector holds `$lanes` items of `$t`.
            let lanes: [$t; $lanes] = unsafe {
                let mut parts = [$splat(<$t>::$unit); PARTS];
                for c in chunks { for k in 0..PARTS { parts[k] = $op(parts[k], $load(c.as_ptr().add(k * $lanes).cast())) } }
                core::mem::transmute(parts.into_iter().reduce(|a, b| $op(a, b)).expect("a fold keeps `PARTS` vectors"))
            };
            lanes.into_iter().chain(rest.iter().copied()).fold(<$t>::$unit, <$t>::$step)
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

/// Elsewhere, as on x86-64, whose `maxpd` gives its second argument beside NaN, the folds and scans use `Extremes`' defaults.
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"))))]
mod plain {
    impl super::Extremes for f32 {}
    impl super::Extremes for f64 {}
}

impl Fold for f16 {
    type Sum = f16;
    fn slice_sum(x: &[f16]) -> Option<f16> { Some(f16::from_f32(folded(x, 0.0f32, |s, v| s.algebraic_add(v.to_f32()), f32::algebraic_add))) }
    fn slice_prod(x: &[f16]) -> Option<f16> { Some(f16::from_f32(folded(x, 1.0f32, |s, v| s.algebraic_mul(v.to_f32()), f32::algebraic_mul))) }
    // The greatest or least item is one of the items, or ∓∞, so it narrows exactly.
    fn slice_max(x: &[f16]) -> f16 { f16::from_f32(folded(x, f32::NEG_INFINITY, |s, v| Core::maximum(s, v.to_f32()), <f32 as Core>::maximum)) }
    fn slice_min(x: &[f16]) -> f16 { f16::from_f32(folded(x, f32::INFINITY, |s, v| Core::minimum(s, v.to_f32()), <f32 as Core>::minimum)) }
    fn slice_cumsum(x: &[f16], start: Option<f16>, out: &mut Vec<f16>) -> bool {
        grouped(x, start.map(f16::to_f32), out, f16::to_f32, |a, b| a + b, f16::from_f32);
        true
    }
    fn slice_cumprod(x: &[f16], start: Option<f16>, out: &mut Vec<f16>) -> bool {
        grouped(x, start.map(f16::to_f32), out, f16::to_f32, |a, b| a * b, f16::from_f32);
        true
    }
    fn slice_cummax(x: &[f16], start: Option<f16>, out: &mut Vec<f16>) { grouped(x, start.map(f16::to_f32), out, f16::to_f32, f32::step_max, f16::from_f32) }
    fn slice_cummin(x: &[f16], start: Option<f16>, out: &mut Vec<f16>) { grouped(x, start.map(f16::to_f32), out, f16::to_f32, f32::step_min, f16::from_f32) }
}

/// The exact sum of integers narrower than 64 bits. Sums of up to 2^31 of them fit `i64`, which their running results hold.
#[inline(always)]
fn narrow_sum<T: Copy + Into<i64>>(x: &[T]) -> Option<i64> {
    let total: i128 = x.chunks(1 << 31).map(|c| i128::from(folded(c, 0i64, |s, v| s + v.into(), |a, b| a + b))).sum();
    i64::try_from(total).ok()
}

/// The exact sum of 64-bit integers. Each item splits into its high and low 32 bits. Neither half's sum can overflow `i64` within
/// 2^31 items, and both sums vectorize.
#[inline(always)]
fn wide_sum(x: &[i64]) -> Option<i64> {
    let halves = |c: &[i64]| folded(c, (0i64, 0i64), |(h, l), v| (h + (v >> 32), l + (v & 0xFFFF_FFFF)), |(h, l), (g, m)| (h + g, l + m));
    let total: i128 = x.chunks(1 << 31).map(|c| { let (h, l) = halves(c); (i128::from(h) << 32) + i128::from(l) }).sum();
    i64::try_from(total).ok()
}

macro_rules! integers {
    ($($t:ty: $sum:expr),+) => {$(
        impl Fold for $t {
            type Sum = i64;
            fn slice_sum(x: &[$t]) -> Option<i64> { $sum(x) }
            fn slice_prod(x: &[$t]) -> Option<i64> { x.iter().try_fold(1i64, |p, &v| p.checked_mul(v.into())) }
            fn slice_max(x: &[$t]) -> $t { folded(x, <$t>::MIN, Ord::max, Ord::max) }
            fn slice_min(x: &[$t]) -> $t { folded(x, <$t>::MAX, Ord::min, Ord::min) }
            // The running sum wraps, and overflow is tested apart from it, so each step waits only for one addition.
            fn slice_cumsum(x: &[$t], start: Option<i64>, out: &mut Vec<i64>) -> bool {
                appended(out, x.len(), |slots| {
                    let (mut total, mut overflow) = (start.unwrap_or(0), 0i64);
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
            fn slice_cumprod(x: &[$t], start: Option<i64>, out: &mut Vec<i64>) -> bool {
                appended(out, x.len(), |slots| {
                    let (mut total, mut ok) = (start.unwrap_or(1), true);
                    for (o, &v) in slots.iter_mut().zip(x) {
                        let (product, over) = total.overflowing_mul(v.into());
                        (total, ok) = (product, ok & !over);
                        o.write(total);
                    }
                    ok
                })
            }
            fn slice_cummax(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) { grouped(x, start, out, |v| v, Ord::max, |t| t) }
            fn slice_cummin(x: &[$t], start: Option<$t>, out: &mut Vec<$t>) { grouped(x, start, out, |v| v, Ord::min, |t| t) }
        }
    )+};
}
integers!(i8: narrow_sum, i16: narrow_sum, i32: narrow_sum, i64: wide_sum, u8: narrow_sum, u16: narrow_sum, u32: narrow_sum);

#[cfg(test)]
mod tests {
    use super::*;

    /// Each reduction and scan against the same taken item by item, in order: sums and running sums of `sums`, and the product of
    /// `prods`. Their items keep every sum and product exact at every width, so any grouping gives the same result.
    fn exact<T: Fold + Copy + PartialEq + std::fmt::Debug>(
        sums: &[T], prods: &[T], add: impl Fn(T::Sum, T) -> Option<T::Sum>, mul: impl Fn(T::Sum, T) -> Option<T::Sum>, zero: T::Sum,
        one: T::Sum,
    ) where
        T::Sum: PartialEq + std::fmt::Debug,
    {
        let mut total = Some(zero);
        let running: Vec<_> = sums.iter().map(|&v| { total = total.and_then(|t| add(t, v)); total.unwrap() }).collect();
        assert_eq!(sum(sums), total);
        let mut out = Vec::new();
        assert!(cumsum(sums, None, &mut out));
        assert_eq!(out, running);
        assert_eq!(prod(prods), prods.iter().try_fold(one, |t, &v| mul(t, v)));
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
        let mut maxima = Vec::new();
        cummax(&sums, Some(-5), &mut maxima);
        assert_eq!(maxima, sums.iter().scan(-5, |m, &v| { *m = (*m).max(v); Some(*m) }).collect::<Vec<_>>());
        assert_eq!((max(&sums), min(&sums)), (*sums.iter().max().unwrap(), *sums.iter().min().unwrap()));
    }

    #[test]
    fn edges() {
        assert_eq!((sum::<f64>(&[]), prod::<i32>(&[]), max::<f32>(&[]), min::<i16>(&[])), (Some(0.0), Some(1), f32::NEG_INFINITY, i16::MAX));
        // NaN at item 40 of 101 lies inside a fold's running results, which vectorize, and not in the remainder.
        let mut v = vec![1.0f64; 101];
        v[40] = f64::NAN;
        let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        assert_eq!((max(&v).is_nan(), min(&v32).is_nan(), max(&[f16::ONE, f16::NAN]).is_nan()), (true, true, true));
        assert_eq!((maximum(1.0, f64::NAN).is_nan(), minimum(f32::NAN, 1.0).is_nan(), maximum(-1.0, 2.0), minimum(f16::ONE, f16::ZERO)), (true, true, 2.0, f16::ZERO));
        let mut out = Vec::new();
        cummax(&[1.0, 3.0, f64::NAN, 5.0, 2.0, 1.0], None, &mut out);
        assert_eq!(out.iter().map(|v| v.is_nan()).collect::<Vec<_>>(), [false, false, true, true, true, true]);
        assert_eq!(sum(&[i64::MAX, 1]), None);
        assert_eq!(sum(&[i64::MAX, 1, -2]), Some(i64::MAX - 1));
        assert!(!cumsum(&[i64::MAX, 1], None, &mut Vec::new()));
        assert!(!cumprod(&[i32::MAX, i32::MAX, i32::MAX], None, &mut Vec::new()));
        let mut out = vec![7.0];
        assert!(cumsum(&[1.0f64, 2.0], Some(10.0), &mut out));
        assert_eq!(out, [7.0, 11.0, 13.0]);
    }
}
