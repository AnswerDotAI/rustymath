//! Conversions between floats and integers whose loops vectorize on every target. AVX2 and wasm have no vector instruction for converting between `f64` and a 64-bit integer. AArch64 converts bytes to floats two lanes at a time. A loop that converts such types with `as` runs one item at a time on those targets. [`int`] and [`float`] convert them through a float's bits. They use the conversion instruction for types that every target converts in vector form, and for 64-bit integers where the target can. [`count`] writes runs of whole numbers as floats. Where the target can't convert 64-bit integers in vector form, it converts one item in 8.
use crate::float::{Float, sealed::{Core, Sealed}};

/// An integer type that [`int`] and [`float`] convert: `i8`, `i16`, `i32`, `i64`, `u8`, `u16` or `u32`.
pub trait Int: Copy + Sealed + sealed::Ways {}

pub(crate) mod sealed {
    /// One integer type's conversions from and to `f32` and `f64`, which [`int`](super::int), [`saturated`](super::saturated) and [`float`](super::float) pick by the float type.
    pub trait Ways: Copy {
        fn from_f32(x: f32) -> Option<Self>;
        fn from_f64(x: f64) -> Option<Self>;
        fn saturated_f32(x: f32) -> Self;
        fn saturated_f64(x: f64) -> Self;
        fn to_f32(self) -> f32;
        fn to_f64(self) -> f64;
    }
}

/// `x`, a whole float, as an integer of type `I`. `None` where `x` is NaN or infinite, or lies outside `I`. For `i64` and `u32`, a magnitude beyond 2^51 - 1 also gives `None`, and so does `i32::MIN` from an `f32`. A fraction gives an integer next to it, or `None`.
#[inline(always)]
pub fn int<I: Int, F: Float>(x: F) -> Option<I> { if <F::Working as Core>::WIDE { I::from_f64(x.wide()) } else { I::from_f32(x.wide() as f32) } }

/// `x` as an integer of type `I`, as Rust's `as` converts a float: NaN gives 0, a fraction gives the integer toward 0, and a value past `I`'s range gives its largest or smallest value. A type that fits an `i32` converts in vector form on every target, where `as` would convert one item at a time, as it does from an `f64` on AArch64 and from any float on x86-64.
#[inline(always)]
pub fn saturated<I: Int, F: Float>(x: F) -> I { if <F::Working as Core>::WIDE { I::saturated_f64(x.wide()) } else { I::saturated_f32(x.wide() as f32) } }

/// `i` as the nearest float of type `F`.
#[inline(always)]
pub fn float<F: Float, I: Int>(i: I) -> F { if <F::Working as Core>::WIDE { F::narrow(i.to_f64()) } else { F::narrow(i.to_f32().into()) } }

/// Appends the `n` whole numbers `start`, `start + 1`, `start + 2` and so on to `out` as floats of type `F`, or `start`, `start - 1` and so on where `down` is true. Each is the whole number rounded once to `F`. A float narrower than `f64` counts in `i32` where every item fits one, because every target converts `i32` to it in vector form. Otherwise a target with `NATIVE_64` converts each item from an integer, as [`float`] does. Wasm and x86-64 without AVX-512 have no vector instruction for that, so there each block of 8 items adds 0 to 7 steps to the block's first item, which comes from the block's position.
pub fn count<F: Float>(out: &mut Vec<F>, start: i64, down: bool, n: usize) {
    let end = if down { start - n as i64 } else { start + n as i64 };
    if !<F::Working as Core>::WIDE && i32::try_from(start).is_ok() && i32::try_from(end).is_ok() {
        let start = start as i32;
        return if down { out.extend((0..n as i32).map(move |i| float::<F, i32>(start - i))) } else { out.extend((0..n as i32).map(move |i| float::<F, i32>(start + i))) };
    }
    if NATIVE_64 {
        return if down { out.extend((0..n).map(move |i| F::narrow(wide_float(start - i as i64)))) } else { out.extend((0..n).map(move |i| F::narrow(wide_float(start + i as i64)))) };
    }
    let step = if down { -1 } else { 1 };
    let offsets: [f64; 8] = std::array::from_fn(|k| wide_float(step * k as i64));
    let first = move |j: usize| wide_float(start + step * (8 * j) as i64);
    crate::slice::appended(out, n, |slots| {
        let (blocks, rest) = slots.as_chunks_mut::<8>();
        let whole = blocks.len();
        for (j, block) in blocks.iter_mut().enumerate() { let first = first(j); for k in 0..8 { block[k].write(F::narrow(first + offsets[k])); } }
        for (slot, &offset) in rest.iter_mut().zip(&offsets) { slot.write(F::narrow(first(whole) + offset)); }
    });
}

/// Whether the target converts between `f64` and 64-bit integers in vector form, as AArch64 and x86-64 with AVX-512 do. Elsewhere they convert through a float's bits. Both ways give the same results for whole floats.
const NATIVE_64: bool = cfg!(any(target_arch = "aarch64", target_feature = "avx512dq"));

/// Whether the target converts floats to `i32` with saturation in vector form, as AArch64's `fcvtzs` and wasm's `trunc_sat` do. On x86-64, where Rust's saturating `as` compiles to one scalar conversion per item, floats convert without saturation after a range check.
const SATURATING: bool = !cfg!(target_arch = "x86_64");

/// The magnitude up to which the bits of `f64::SHIFT` + x hold the whole number x.
pub(crate) const WIDE_LIMIT: f64 = 2251799813685247.0;

/// `i` as the nearest `f64`, natively or through `halves`.
#[inline(always)]
fn wide_float(i: i64) -> f64 { if NATIVE_64 { i as f64 } else { halves(i) } }

/// `i` as the nearest `f64` through bits. The high half converts through the bits of 2^84 + 2^63 plus it times 2^32, and the low half through `from_low`. Both convert exactly, and their sum rounds once.
#[inline(always)]
fn halves(i: i64) -> f64 {
    let high = f64::from_bits(0x4530_0000_0000_0000 | ((i as u64 >> 32) ^ 0x8000_0000)) - f64::from_bits(0x4530_0000_8000_0000);
    high + f64::from_low(i as u64 & 0xffff_ffff, 0.0)
}

/// The whole number nearest `x`, for a magnitude up to `WIDE_LIMIT`, through the bits of `f64::SHIFT` + x.
#[inline(always)]
pub(crate) fn shifted(x: f64) -> i64 { (x + f64::SHIFT).to_bits().wrapping_sub(f64::SHIFT.to_bits()) as i64 }

/// The whole number `i`, of magnitude up to `WIDE_LIMIT`, as an `f64`, through the bits of `f64::SHIFT` + i. It undoes [`shifted`].
#[inline(always)]
pub(crate) fn unshifted(i: i64) -> f64 { f64::from_bits(f64::SHIFT.to_bits().wrapping_add(i as u64)) - f64::SHIFT }

/// Writes each integer type's conversions. `narrow` types fit an `i32`, and every target converts a float to an `i32` in vector form. `wide` types convert from a float through the bits of `f64::SHIFT` + x, unless the target has `NATIVE_64`. The two expressions convert `i` to `f32` and `f64`.
macro_rules! ints {
    ($($t:ty: $kind:ident, |$i:ident| $f32:expr, $f64:expr;)+) => {$(
        impl Int for $t {}
        impl sealed::Ways for $t {
            #[inline(always)]
            fn from_f32(x: f32) -> Option<$t> { ints!(@$kind f32, $t, x) }
            #[inline(always)]
            fn from_f64(x: f64) -> Option<$t> { ints!(@$kind f64, $t, x) }
            #[inline(always)]
            fn saturated_f32(x: f32) -> $t { ints!(@saturated $kind f32, $t, x) }
            #[inline(always)]
            fn saturated_f64(x: f64) -> $t { ints!(@saturated $kind f64, $t, x) }
            #[inline(always)]
            fn to_f32(self) -> f32 { let $i = self; $f32 }
            #[inline(always)]
            fn to_f64(self) -> f64 { let $i = self; $f64 }
        }
    )+};
    (@narrow $f:ty, $t:ty, $x:ident) => {{
        // One comparison, of the distance from the middle of `$t`'s range, keeps the loop to one mask.
        let (mid, half) = ((<$t>::MIN as f64 + <$t>::MAX as f64) / 2.0, (<$t>::MAX as f64 - <$t>::MIN as f64 + 1.0) / 2.0);
        (($x - mid as $f).abs() < half as $f).then(|| {
            if SATURATING { return $x as i32 as $t }
            // SAFETY: the comparison excludes NaN, the infinities and every float that truncates outside `$t`, which `i32` holds. The conversion then needs no saturation.
            unsafe { $x.to_int_unchecked::<i32>() as $t }
        })
    }};
    (@wide f32, $t:ty, $x:ident) => { Self::from_f64($x.into()) };
    (@wide f64, $t:ty, $x:ident) => {{
        let n = if NATIVE_64 { $x as i64 } else { shifted($x) };
        ($x.abs() <= WIDE_LIMIT).then_some(n).and_then(|n| <$t>::try_from(n).ok())
    }};
    (@saturated narrow f32, $t:ty, $x:ident) => { if SATURATING { $x as $t } else { ints!(@clamped f32, $t, $x) } };
    // A target with `NATIVE_64` converts an `f64` to an `i64` with saturation in vector form, which the clamp then narrows as `as` would.
    (@saturated narrow f64, $t:ty, $x:ident) => { if NATIVE_64 { ($x as i64).clamp(<$t>::MIN.into(), <$t>::MAX.into()) as $t } else { ints!(@clamped f64, $t, $x) } };
    (@saturated wide $f:ty, $t:ty, $x:ident) => { $x as $t };
    (@clamped $f:ty, $t:ty, $x:ident) => {{
        // Where `$f` can't hold `$t::MAX`, as an `f32` can't hold `i32::MAX`, the maximum rounds up to a float outside `$t`. The clamp then stops one float below it, and a float at or past it gives `$t::MAX`.
        let top = <$t>::MAX as $f;
        let below = if top as i128 > <$t>::MAX as i128 { top.next_down() } else { top };
        let c = if $x.is_nan() { 0.0 } else { $x.max(<$t>::MIN as $f).min(below) };
        // SAFETY: `c` is never NaN, and the clamp keeps it within `$t`.
        let i = unsafe { c.to_int_unchecked::<$t>() };
        if $x >= top { <$t>::MAX } else { i }
    }};
}

// A byte or a `u16` reads as a float through `from_low`. AArch64's own conversion handles bytes two lanes at a time.
ints! {
    i8: narrow, |i| i as f32, i as f64;
    i16: narrow, |i| i as f32, i as f64;
    i32: narrow, |i| i as f32, i as f64;
    u8: narrow, |i| f32::from_low(i.into(), 0.0), f32::from_low(i.into(), 0.0).into();
    u16: narrow, |i| f32::from_low(i.into(), 0.0), f32::from_low(i.into(), 0.0).into();
    u32: wide, |i| f64::from_low(i.into(), 0.0) as f32, f64::from_low(i.into(), 0.0);
    i64: wide, |i| wide_float(i) as f32, wide_float(i);
}
