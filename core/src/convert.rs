//! Conversions between floats and integers whose loops vectorize on every target. AVX2 and wasm have no vector instruction for converting between `f64` and a 64-bit integer. AArch64 converts bytes to floats two lanes at a time. A loop that converts such types with `as` runs one item at a time on those targets. [`int`] and [`float`] convert them through a float's bits. They use the conversion instruction for types that every target converts in vector form, and for 64-bit integers where the target can. [`count`] writes runs of whole numbers as floats. Where the target can't convert 64-bit integers in vector form, it converts one item in 8. [`saturated`] also converts big integers and `u64`s, saturating them at an integer type's range.
use crate::float::{Float, sealed::{Core, Sealed}};
use num_bigint::{BigInt, Sign};
use num_traits::Bounded;

/// An integer type that [`int`] and [`float`] convert: `i8`, `i16`, `i32`, `i64`, `u8`, `u16` or `u32`.
pub trait Int: Copy + Sealed + sealed::Ways {}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ConversionKind { Cast, Truncate, Bits, Keep, Floor }

/// Resolved kinds include the working float width even when the other kind is a word.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Conversion { pub input: crate::buffer::DType, pub output: crate::buffer::DType, pub kind: ConversionKind }

macro_rules! conversion {
    ($call:expr, $checked:expr, $tolerance:expr, $write:ident) => {{
        use crate::{buffer::{DType, Element, FloatWidth, Width}, error::{Error, Kind}};
        match ($call.kind, $call.input, $call.output) {
            (ConversionKind::Floor, DType::Float(width), DType::Integer(to)) => with_width!(to, I => with_float_width!(width, F => {
                crate::with_tolerance!($tolerance, |t| $write!(F, I, |x| Ok(floored(x, t))))
            })),
            (ConversionKind::Floor, DType::Integer(from), DType::Integer(to)) => {
                with_width!(to, I => with_width!(from, S => $write!(S, I, |x| Ok(x as I))))
            }
            (ConversionKind::Floor | ConversionKind::Truncate, DType::Boolean, DType::Integer(to)) => {
                with_width!(to, I => $write!(bool, I, |x| Ok(u8::from(x) as I)))
            }
            (ConversionKind::Truncate, DType::Float(width), DType::Integer(to)) => {
                with_width!(to, I => with_float_width!(width, F => $write!(F, I, |x| Ok(saturated(x)))))
            }
            (ConversionKind::Cast | ConversionKind::Truncate, DType::Integer(from), DType::Integer(to)) => with_width!(to, I => with_width!(from, S => {
                $write!(S, I, move |x| {
                    if $checked { I::try_from(x).map_err(|_| Error::new(Kind::Overflow, "integer doesn't fit its converted dtype")) } else { Ok(x as I) }
                })
            })),
            (ConversionKind::Truncate, input_type, DType::Character) if input_type.is_integral() => $write!(i64, char, move |x| {
                match u32::try_from(x).ok().and_then(char::from_u32) {
                    Some(x) => Ok(x), None if $checked => Err(Error::new(Kind::Domain, "result is not a Unicode character")), None => Ok(char::FILL)
                }
            }),
            (ConversionKind::Bits, DType::Float(FloatWidth::F32), DType::Integer(Width::I64)) => {
                $write!(f32, i64, |x| Ok(i64::from(x.to_bits())))
            }
            (ConversionKind::Bits, DType::Float(FloatWidth::F64), DType::Integer(Width::I64)) => {
                $write!(f64, i64, |x| Ok(x.to_bits() as i64))
            }
            (ConversionKind::Bits, DType::Integer(_), DType::Float(FloatWidth::F32)) => {
                $write!(i64, f32, |x| Ok(f32::from_bits(x as u32)))
            }
            (ConversionKind::Bits, DType::Integer(_), DType::Float(FloatWidth::F64)) => {
                $write!(i64, f64, |x| Ok(f64::from_bits(x as u64)))
            }
            (ConversionKind::Keep, _, dtype) => with_dtype!(dtype, T => $write!(@keep T)),
            (ConversionKind::Cast | ConversionKind::Floor | ConversionKind::Truncate, _, dtype) => with_dtype!(dtype, T => $write!(@cast T)),
            _ => $write!(@unsupported),
        }
    }};
}
impl Conversion {
    pub fn validity<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(self, o: &O, x: O::V) -> Option<(crate::error::Kind, O::B)> {
        use crate::buffer::{DType, Width};
        if self.kind == ConversionKind::Keep && self.input.is_integral() && self.output.is_integral() {
            let bounds = |dtype| match dtype {
                DType::Boolean => (0, 1), DType::Integer(width) => with_width!(width, T => (T::MIN as i64, T::MAX as i64)), _ => unreachable!()
            };
            let ((input_lo, input_hi), (lo, hi)) = (bounds(self.input), bounds(self.output));
            if lo <= input_lo && hi >= input_hi { return None; }
            let x = o.to_integer(x);
            let valid = o.and(
                o.integer_compare(crate::recipe::Cmp::LessEqual, o.integer(lo.max(input_lo)), x),
                o.integer_compare(crate::recipe::Cmp::LessEqual, x, o.integer(hi.min(input_hi))));
            return Some((crate::error::Kind::Domain, valid));
        }
        if self.kind != ConversionKind::Cast { return None; }
        let valid = match self.output {
            DType::Boolean => o.or(o.equal(x, o.number(F::zero())), o.equal(x, o.number(F::one()))),
            DType::Integer(width) if matches!(self.input, DType::Float(_)) => {
                let upper = if width == Width::U8 { 256.0 } else { 2f64.powi((8 * self.output.bytes() - 1) as i32) };
                let lower = if width == Width::U8 { 0.0 } else { -upper };
                let x = if width == Width::I64 { x } else { o.from_integer(o.to_integer(x)) };
                o.and(o.less_equal(o.number(F::narrow(lower)), x), o.less(x, o.number(F::narrow(upper))))
            }
            _ => return None,
        };
        Some((crate::error::Kind::Overflow, valid))
    }
    pub fn may_raise(self, checked: bool) -> bool {
        self.kind == ConversionKind::Keep || checked && (self.output == crate::buffer::DType::Character || self
                        .validity::<f64, _>(&crate::recipe::RangeOps(self.input), crate::domain::Range::of_dtype(self.input))
                        .is_some_and(|(_, valid)| valid != Some(true)))
    }
    /// Converts contiguous storage through the mapped conversion executor.
    pub(crate) fn buffer(self, view: crate::buffer::View<'_>, checked: bool, tolerance: crate::tolerant::Tolerance) -> Result<crate::buffer::Buffer, crate::error::Error> {
        let mut result = crate::buffer::Buffer::with_capacity(self.output, view.len());
        self.append(view, checked, tolerance, None, crate::buffer::MapOutput::Buffer(&mut result))?;
        Ok(result)
    }
    pub(crate) fn append<'a, 'b>(
        self, view: crate::buffer::View<'_>, checked: bool, tolerance: crate::tolerant::Tolerance,
        scratch: Option<&'a mut crate::buffer::Scratch>, output: crate::buffer::MapOutput<'a, 'b>
    ) -> Result<(), crate::error::Error> {
        let sources = [crate::agreement::Binding::full(view)];
        let plan = crate::agreement::ReadPlan::input(0, view.len());
        self.map_in(crate::agreement::Read { sources: &sources, plan: &plan }, 0..view.len(), checked, tolerance, scratch, output)
    }
    pub fn map_in<'a, 'b>(
        self, input: crate::agreement::Read<'_>, range: std::ops::Range<usize>, checked: bool, tolerance: crate::tolerant::Tolerance,
        scratch: Option<&'a mut crate::buffer::Scratch>, target: crate::buffer::MapOutput<'a, 'b>
    ) -> Result<(), crate::error::Error> {
        let mut output = crate::buffer::Output { target, result_dtype: self.output, checked, scratch };
        macro_rules! write {
            ($s:ty, $t:ty, $convert:expr) => { written::<$s, $t>(input, range, checked, &mut output, $convert) };
            (@cast $t:ty) => {{
                if let crate::buffer::MapOutput::Buffer(buffer) = &mut output.target && let Some(values) = <$t>::vec_mut(buffer) {
                    let (at, n) = (values.len(), range.len());
                    if let Err(error) = input.append(range, checked, values) {
                        crate::error::record(error); values.truncate(at); values.resize(at + n, <$t>::FILL);
                    }
                    Ok(())
                } else { write!($t, $t, Ok) }
            }};
            (@keep $t:ty) => { kept_mapped::<$t>(self.input, input, range, checked, &mut output) };
            (@unsupported) => {{ output.target.fill(range.len()); Err(Error::new(Kind::Domain, "unsupported resolved conversion")) }};
        }
        crate::error::checked(|| conversion!(self, checked, tolerance, write))
    }
}

#[inline]
fn converted_item<T: crate::buffer::Element>(value: Result<T, crate::error::Error>) -> (T, bool) {
    match value { Ok(value) => (value, true), Err(error) => { crate::error::record(error); (T::FILL, false) } }
}

#[inline]
fn written<S: crate::buffer::Cast, T: crate::buffer::OutputItem>(
    input: crate::agreement::Read<'_>, range: std::ops::Range<usize>, checked: bool, output: &mut crate::buffer::Output<'_, '_>,
    convert: impl Fn(S) -> Result<T, crate::error::Error> + Sync
) -> Result<(), crate::error::Error> {
    output.write::<T>(range.len(), &|part, slots, work| {
        let part = part.start + range.start..part.end + range.start;
        let in_place = input.in_place::<S>(&part);
        let len = if in_place { part.len().max(1) } else { crate::CHUNK };
        let mut local = Vec::new();
        let (mut work, mut copied) = (work, 0);
        let scratch = match work.as_deref_mut() { Some(work) if !in_place => { let [scratch] = work.inputs::<S, 1>(); scratch }, _ => &mut local };
        let ok = part.clone().step_by(len).fold(true, |ok, start| {
            let range = start..(start + len).min(part.end);
            let n = range.len();
            let input = match input.part_counted::<S>(range, checked, scratch) {
                Ok((input, n)) => { copied += n; input }
                Err(error) => {
                    crate::error::record(error);
                    return unsafe { slots.fill(n, |out| { for slot in out { slot.write(T::FILL); } false }) } & ok;
                }
            };
            slots.pairs(input, crate::Arg::One(()), n, |x, ()| converted_item(convert(x))) & ok
        });
        if let Some(work) = work { work.record_copies(copied); }
        ok
    })
}

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

/// `x` as an integer of type `I`, or `I`'s largest or smallest value where `x` lies past its range. A float converts as Rust's `as` converts it: NaN gives 0, and a fraction gives the integer toward 0. A type that fits an `i32` converts from a float in vector form on every target, where `as` would convert one item at a time, as it does from an `f64` on AArch64 and from any float on x86-64. [`Saturates`] lists the other sources.
#[inline(always)]
pub fn saturated<I, S: Saturates<I>>(x: S) -> I { x.saturated() }
/// A number that [`saturated`] converts to the integer type `I`: a float to any [`Int`], and a big integer or a `u64` to any integer type that `TryFrom` converts it to.
pub trait Saturates<I> { fn saturated(self) -> I; }
impl<I: Int, F: Float> Saturates<I> for F {
    #[inline(always)]
    fn saturated(self) -> I { if <F::Working as Core>::WIDE { I::saturated_f64(self.wide()) } else { I::saturated_f32(self.wide() as f32) } }
}
impl<I: Bounded + for<'a> TryFrom<&'a BigInt>> Saturates<I> for &BigInt {
    fn saturated(self) -> I { I::try_from(self).unwrap_or_else(|_| if self.sign() == Sign::Minus { I::min_value() } else { I::max_value() }) }
}
impl<I: Bounded + TryFrom<u64>> Saturates<I> for u64 { fn saturated(self) -> I { I::try_from(self).unwrap_or_else(|_| I::max_value()) } }
/// The floor of `x` within `t` as an integer of type `I`, saturated at `I`'s range as [`saturated`] saturates: NaN gives 0. The one float-to-integer rule for conversions that change a number's domain: BPL's `ₓ` marks and Python's `astype` and integer `dtype=`.
#[inline(always)]
pub fn floored<I: Int, F: Float>(x: F, t: impl crate::tolerant::Tolerant) -> I { saturated(t.floor(x)) }

/// `i` as the nearest float of type `F`.
#[inline(always)]
pub fn float<F: Float, I: Int>(i: I) -> F { if <F::Working as Core>::WIDE { F::narrow(i.to_f64()) } else { F::narrow(i.to_f32().into()) } }

/// Appends the `n` whole numbers `start`, `start + 1`, `start + 2` and so on to `out` as floats of type `F`, or `start`, `start - 1` and so on where `down` is true. Each is the whole number rounded once to `F`. A float narrower than `f64` counts in `i32` where every item fits one, because every target converts `i32` to it in vector form. Otherwise a target with `NATIVE_64` converts each item from an integer, as [`float`] does. Wasm and x86-64 without AVX-512 have no vector instruction for that, so there each block of 8 items adds 0 to 7 steps to the block's first item, which comes from the block's position.
pub fn count<F: Float>(out: &mut Vec<F>, start: i64, down: bool, n: usize) {
    let end = if down { start - n as i64 } else { start + n as i64 };
    if !<F::Working as Core>::WIDE && i32::try_from(start).is_ok() && i32::try_from(end).is_ok() {
        let start = start as i32;
        return if down { out.extend((0..n as i32).map(move |i| float::<F, i32>(start - i))) }
        else { out.extend((0..n as i32).map(move |i| float::<F, i32>(start + i))) };
    }
    if NATIVE_64 {
        return if down { out.extend((0..n).map(move |i| F::narrow(wide_float(start - i as i64)))) }
        else { out.extend((0..n).map(move |i| F::narrow(wide_float(start + i as i64)))) };
    }
    let step = if down { -1 } else { 1 };
    let offsets: [f64; 8] = std::array::from_fn(|k| wide_float(step * k as i64));
    let first = move |j: usize| wide_float(start + step * (8 * j) as i64);
    unsafe {
        crate::slice::appended(out, n, |slots| {
            let (blocks, rest) = slots.as_chunks_mut::<8>();
            let whole = blocks.len();
            for (j, block) in blocks.iter_mut().enumerate() { let first = first(j); for k in 0..8 { block[k].write(F::narrow(first + offsets[k])); } }
            for (slot, &offset) in rest.iter_mut().zip(&offsets) { slot.write(F::narrow(first(whole) + offset)); }
        })
    };
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

fn kept_mapped<T: crate::buffer::OutputItem + crate::buffer::Cast>(
    dtype: crate::buffer::DType, input: crate::agreement::Read<'_>, range: std::ops::Range<usize>, checked: bool, output: &mut crate::buffer::Output<'_, '_>
) -> Result<(), crate::error::Error> {
    output.write::<T>(range.len(), &|part, slots, work| {
        let mut local = crate::buffer::Buffer::with_capacity(dtype, 0);
        let scratch = match work { Some(work) => work.buffer(dtype), None => &mut local };
        match input.part_view(part.start + range.start..part.end + range.start, dtype, checked, scratch) {
            Ok(values) => unsafe { slots.fill(part.len(), |out| {
                let mut good = true;
                for (i, slot) in out.iter_mut().enumerate() {
                    slot.write(match crate::buffer::kept_as::<T>(values, i) {
                        Ok(value) => value, Err(error) => { crate::error::record(error); good = false; T::FILL }
                    });
                }
                good
            }) },
            Err(error) => {
                crate::error::record(error);
                unsafe { slots.fill(part.len(), |out| { for slot in out { slot.write(T::FILL); } false }) }
            }
        }
    })
}
