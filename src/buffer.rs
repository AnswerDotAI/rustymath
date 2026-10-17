//! Compact buffers: owned items of one element type, borrowed views of them, and the [`DType`] that names each. Booleans take one byte each. Integers come in four widths, floats in three and complex numbers in two, named by the width of their parts. [`Source`] reads an item of one element type as another.
use num_complex::{Complex32, Complex64};
use std::borrow::Cow;
use crate::error::{Error, Kind};

/// Declares the owned buffers `$owned`, their borrowed view `$view` and the width `$tag` that names a variant, with one variant for each element type, which fills with `$fill`. `$outer` is the family's variant in `Buffer` and `View`, and `$widest` its widest type, which `widened` reads every width as. Code that needs a variant's element type matches on the variant, as `with_ints!` does.
macro_rules! buffers {
    ($owned:ident, $view:ident, $tag:ident, $outer:ident, $kind:ident, $widest:ty: $($variant:ident($t:ty = $fill:expr)),+) => {
        #[derive(Clone, Debug)]
        pub enum $owned { $($variant(Vec<$t>)),+ }
        #[derive(Clone, Copy, Debug)]
        pub enum $view<'a> { $($variant(&'a [$t])),+ }
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
        pub enum $tag { $($variant),+ }
        impl $owned {
            pub fn with_capacity(tag: $tag, capacity: usize) -> Self { match tag { $($tag::$variant => Self::$variant(Vec::with_capacity(capacity))),+ } }
            #[inline]
            pub fn view(&self) -> $view<'_> { match self { $(Self::$variant(v) => $view::$variant(v)),+ } }
            #[inline]
            pub fn capacity(&self) -> usize { match self { $(Self::$variant(v) => v.capacity()),+ } }
            #[inline]
            pub fn as_mut_ptr(&mut self) -> *mut u8 { match self { $(Self::$variant(v) => v.as_mut_ptr().cast()),+ } }
            /// Copies `range` of `source` when it has this buffer's element type, and gives whether it could.
            pub fn extend_from(&mut self, source: $view<'_>, range: std::ops::Range<usize>) -> bool {
                match (self, source) {
                    $((Self::$variant(d), $view::$variant(s)) => d.extend_from_slice(&s[range]),)+
                    #[allow(unreachable_patterns)]
                    _ => return false,
                }
                true
            }
        }
        impl $view<'_> {
            #[inline]
            pub fn len(self) -> usize { match self { $(Self::$variant(v) => v.len()),+ } }
            #[inline]
            pub fn is_empty(self) -> bool { self.len() == 0 }
            #[inline]
            pub fn as_ptr(self) -> *const u8 { match self { $(Self::$variant(v) => v.as_ptr().cast()),+ } }
            #[inline]
            pub fn tag(self) -> $tag { match self { $(Self::$variant(_) => $tag::$variant),+ } }
        }
        impl<'a> $view<'a> {
            /// The items at the widest width, borrowed when they have it. A narrower item converts exactly.
            pub fn widened(self) -> Cow<'a, [$widest]> {
                match self { $(Self::$variant(v) => Source::<$widest>::borrowed(v).map_or_else(|| Cow::Owned(read_all(v)), Cow::Borrowed)),+ }
            }
        }
        // Views are equal when they hold the same values, whatever their widths. A float NaN equals nothing, as in IEEE.
        impl PartialEq for $view<'_> {
            fn eq(&self, other: &Self) -> bool {
                match (*self, *other) { $((Self::$variant(x), Self::$variant(y)) => x == y,)+ (x, y) => x.widened() == y.widened() }
            }
        }
        $(impl Element for $t {
            const FILL: Self = $fill;
            const DTYPE: DType = DType::$kind($tag::$variant);
            #[inline]
            fn of(view: View<'_>) -> Option<&[Self]> { match view { View::$outer($view::$variant(v)) => Some(v), _ => None } }
            #[inline]
            fn of_mut(buffer: &mut Buffer) -> Option<&mut [Self]> { match buffer { Buffer::$outer($owned::$variant(v)) => Some(v), _ => None } }
            #[inline]
            fn view(items: &[Self]) -> View<'_> { View::$outer($view::$variant(items)) }
            #[inline]
            fn buffer(data: Vec<Self>) -> Buffer { Buffer::$outer($owned::$variant(data)) }
            #[inline]
            fn vec(buffer: Buffer) -> Option<Vec<Self>> { match buffer { Buffer::$outer($owned::$variant(v)) => Some(v), _ => None } }
        })+
    };
}
buffers!(IntBuf, Ints, Width, Integers, Integer, i64: U8(u8 = 0), I16(i16 = 0), I32(i32 = 0), I64(i64 = 0));
// Floats come in three widths, which sort from narrowest to widest. A narrower width takes less memory. The CPU computes each operation on 16-bit floats in f32 and rounds the result, through `half`.
buffers!(FloatBuf, Floats, FloatWidth, Floats, Float, f64: F16(half::f16 = half::f16::ZERO), F32(f32 = 0.0), F64(f64 = 0.0));
// Complex numbers come in two widths, named by the width of their parts. A 16-bit float that becomes complex takes 32-bit parts.
buffers!(ComplexBuf, Complexes, ComplexWidth, Complexes, Complex, Complex64: F32(Complex32 = Complex32::new(0.0, 0.0)), F64(Complex64 = Complex64::new(0.0, 0.0)));
/// Booleans and characters each have one element type.
macro_rules! single {
    ($($t:ty: $variant:ident($dtype:ident) = $fill:expr),+) => {$(
        impl Element for $t {
            const FILL: Self = $fill;
            const DTYPE: DType = DType::$dtype;
            #[inline]
            fn of(view: View<'_>) -> Option<&[Self]> { match view { View::$variant(v) => Some(v), _ => None } }
            #[inline]
            fn of_mut(buffer: &mut Buffer) -> Option<&mut [Self]> { match buffer { Buffer::$variant(v) => Some(v), _ => None } }
            #[inline]
            fn view(items: &[Self]) -> View<'_> { View::$variant(items) }
            #[inline]
            fn buffer(data: Vec<Self>) -> Buffer { Buffer::$variant(data) }
            #[inline]
            fn vec(buffer: Buffer) -> Option<Vec<Self>> { match buffer { Buffer::$variant(v) => Some(v), _ => None } }
        }
    )+};
}
single!(bool: Booleans(Boolean) = false, char: Characters(Character) = ' ');

/// The narrowest complex width whose parts hold floats of `width`.
impl From<FloatWidth> for ComplexWidth {
    fn from(width: FloatWidth) -> Self { match width { FloatWidth::F64 => Self::F64, FloatWidth::F16 | FloatWidth::F32 => Self::F32 } }
}
impl From<ComplexWidth> for FloatWidth {
    fn from(width: ComplexWidth) -> Self { match width { ComplexWidth::F32 => Self::F32, ComplexWidth::F64 => Self::F64 } }
}

/// Writes the two dispatch macros of a family of widths from each width's variant name and element type. `$slices` is the family's view and `$widths` its enum of widths, whose variants share names. `$values!(v, |x| body)` runs `body` with `x` bound to the slice that `v` holds. `$types!(w, T => body)` runs `body` with the type `T` standing for the element type of the width `w`. Each expands to one `match` with an arm for each variant. With `kernel` before `v` or `w`, each variant listed before the `;` gives `None`, and `body` runs only for the others, for a caller whose own kernels take no 16-bit floats. rustymath's function objects have 16-bit forms, which compute in 32 bits. `$d` is a `$` token, which the written macros need for their own metavariables. Element types come in brackets as token trees. A type from another crate is written `$ crate::half::f16`, so a caller that doesn't depend on `half` reaches it through rustymath's re-export.
macro_rules! width_macros {
    ($d:tt $values:ident: $slices:ident, $types:ident: $widths:ident { $($skip:ident => [$($st:tt)+]),* ; $($kept:ident => [$($kt:tt)+]),+ }) => {
        #[macro_export]
        macro_rules! $values {
            (kernel $d v:expr, |$d x:ident| $d body:expr) => {
                match $d v {
                    $($crate::buffer::$slices::$skip(_) => None,)*
                    $($crate::buffer::$slices::$kept($d x) => $d body,)+
                }
            };
            ($d v:expr, |$d x:ident| $d body:expr) => {
                match $d v {
                    $($crate::buffer::$slices::$skip($d x) => $d body,)*
                    $($crate::buffer::$slices::$kept($d x) => $d body,)+
                }
            };
        }
        #[macro_export]
        macro_rules! $types {
            (kernel $d w:expr, $d t:ident => $d body:expr) => {
                match $d w {
                    $($crate::buffer::$widths::$skip => None,)*
                    $($crate::buffer::$widths::$kept => { type $d t = $($kt)+; $d body })+
                }
            };
            ($d w:expr, $d t:ident => $d body:expr) => {
                match $d w {
                    $($crate::buffer::$widths::$skip => { type $d t = $($st)+; $d body })*
                    $($crate::buffer::$widths::$kept => { type $d t = $($kt)+; $d body })+
                }
            };
        }
    };
}
width_macros!($ with_ints: Ints, with_width: Width { ; U8 => [u8], I16 => [i16], I32 => [i32], I64 => [i64] });
width_macros!($ with_floats: Floats, with_float_width: FloatWidth { F16 => [$ crate::half::f16]; F32 => [f32], F64 => [f64] });
width_macros!($ with_complexes: Complexes, with_complex_width: ComplexWidth { ; F32 => [$ crate::num_complex::Complex32], F64 => [$ crate::num_complex::Complex64] });

/// Runs `$body` with `$t` standing for the element type of `$dtype`, a computing dtype that [`DType::inexact`] gives: a 32- or 64-bit float, or a complex number. Any other dtype panics.
#[macro_export]
macro_rules! with_inexact {
    ($dtype:expr, $t:ident => $body:expr) => {
        match $dtype {
            $crate::buffer::DType::Float(w) => $crate::with_float_width!(kernel w, $t => Some($body)),
            $crate::buffer::DType::Complex(w) => Some($crate::with_complex_width!(w, $t => $body)),
            _ => None,
        }.expect("with_inexact! takes a computing dtype that DType::inexact gives")
    };
}

/// Runs `$body` with `$x` and `$y` bound to the integers of `$a` and `$b` at one width: their own when they share it, and otherwise both widened to 64 bits.
#[macro_export]
macro_rules! with_int_pair {
    ($a:expr, $b:expr, |$x:ident, $y:ident| $body:expr) => {
        match ($a, $b) {
            ($crate::buffer::Ints::U8($x), $crate::buffer::Ints::U8($y)) => $body,
            ($crate::buffer::Ints::I16($x), $crate::buffer::Ints::I16($y)) => $body,
            ($crate::buffer::Ints::I32($x), $crate::buffer::Ints::I32($y)) => $body,
            (a, b) => {
                let (a, b) = (a.widened(), b.widened());
                let ($x, $y) = (&*a, &*b);
                $body
            }
        }
    };
}

impl Width {
    /// The bytes that one integer of this width takes.
    fn bytes(self) -> usize { with_width!(self, T => std::mem::size_of::<T>()) }
    #[inline]
    fn holds(self, n: i64) -> bool { with_width!(self, T => T::try_from(n).is_ok()) }
    /// The narrowest width that holds `min`, `max` and every integer between. Every width holds the empty range, where `min` is above `max`.
    #[inline]
    pub fn narrowest<T: Into<i64>>(min: T, max: T) -> Self {
        let (min, max) = (min.into(), max.into());
        for w in [Self::U8, Self::I16, Self::I32] { if min > max || w.holds(min) && w.holds(max) { return w; } }
        Self::I64
    }
    /// The narrowest width that holds every position below `bound`.
    #[inline]
    pub fn below(bound: usize) -> Self { Self::narrowest(0, bound as i64 - 1) }
    /// The narrowest width that holds the integers of both widths.
    #[inline]
    pub fn join(self, other: Self) -> Self { if self.bytes() >= other.bytes() { self } else { other } }
    /// The next wider width, if there is one.
    #[inline]
    pub fn wider(self) -> Option<Self> {
        match self { Self::U8 => Some(Self::I16), Self::I16 => Some(Self::I32), Self::I32 => Some(Self::I64), Self::I64 => None }
    }
    /// The float width that NumPy promotes integers of this width to beside a float: the narrowest that holds each of them exactly. That's `f16` for `u8`, `f32` for `i16` and `f64` for `i32`. `i64` also gives `f64`, which holds its integers exactly only up to 2^53.
    #[inline]
    pub fn float(self) -> FloatWidth {
        match self { Self::U8 => FloatWidth::F16, Self::I16 => FloatWidth::F32, Self::I32 | Self::I64 => FloatWidth::F64 }
    }
}

impl FloatWidth {
    /// The gap between 1 and the next float of this width.
    pub fn epsilon(self) -> f64 { with_float_width!(self, F => <F as num_traits::Float>::epsilon().into()) }
    /// The bits of a float of this width.
    pub fn bits(self) -> i64 { with_float_width!(self, F => 8 * std::mem::size_of::<F>() as i64) }
    /// The width of floats of `bits` bits.
    pub fn of_bits(bits: i64) -> Option<Self> { match bits { 16 => Some(Self::F16), 32 => Some(Self::F32), 64 => Some(Self::F64), _ => None } }
}

impl<'a> Ints<'a> {
    /// Item `i` as an `i64`.
    #[inline]
    pub fn get(self, i: usize) -> i64 { with_ints!(self, |x| x[i].into()) }
}

impl<'a> Floats<'a> {
    /// Item `i` as an `f64`, which a narrower float converts to exactly.
    #[inline]
    pub fn get(self, i: usize) -> f64 { with_floats!(self, |v| v[i].into()) }
}

impl<'a> Complexes<'a> {
    /// Item `i` as a `Complex64`, which a narrower complex number converts to exactly.
    #[inline]
    pub fn get(self, i: usize) -> Complex64 { with_complexes!(self, |v| Source::<Complex64>::read(v[i])) }
}
impl FloatBuf {
    /// The floats `values` at `width`, each rounded to it.
    pub fn collect(width: FloatWidth, values: impl Iterator<Item = f64>) -> Self {
        match width {
            FloatWidth::F16 => Self::F16(values.map(Source::read).collect()), FloatWidth::F32 => Self::F32(values.map(Source::read).collect()),
            FloatWidth::F64 => Self::F64(values.collect())
        }
    }
}

impl ComplexBuf {
    /// The complex numbers `values` at `width`, each part rounded to it.
    pub fn collect(width: ComplexWidth, values: impl Iterator<Item = Complex64>) -> Self {
        match width {
            ComplexWidth::F32 => Self::F32(values.map(Source::read).collect()), ComplexWidth::F64 => Self::F64(values.collect())
        }
    }
}

/// The element type of a compact buffer. Numbers widen from Boolean through the integer widths to the float widths to the complex widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DType { Boolean, Integer(Width), Float(FloatWidth), Complex(ComplexWidth), Character }

impl DType {
    /// Whether items of this dtype are whole numbers: Booleans and integers.
    pub fn is_integral(self) -> bool { matches!(self, Self::Boolean | Self::Integer(_)) }
    /// This dtype's kind, in the order that NEP 50 promotes kinds: 0 for Booleans, 1 for integers, 2 for floats, 3 for complex numbers and 4 for characters, which don't mix with numbers.
    pub fn kind(self) -> u8 { match self { Self::Boolean => 0, Self::Integer(_) => 1, Self::Float(_) => 2, Self::Complex(_) => 3, Self::Character => 4 } }
    /// Whether this dtype widens to `to`: whether [`DType::common`] promotes this dtype and `to` to `to`.
    pub fn widens_to(self, to: DType) -> bool { self.common(to) == Some(to) }
    /// The dtypes that a function of floats or complex numbers computes in and gives, for arguments of this dtype. A Boolean or an integer takes the float that [`Width::float`] gives, but at least `f32`, which computes faster than `f16`: `f32` for Booleans, `u8` and `i16`, and `f64` for `i32` and `i64`. `f16` computes in `f32`, and its results round back. Other floats and complex numbers stay as they are. `None` for characters.
    pub fn inexact(self) -> Option<(DType, DType)> {
        let at_least_f32 = |w: FloatWidth| { let f = Self::Float(w.max(FloatWidth::F32)); (f, f) };
        Some(match self {
            Self::Boolean => at_least_f32(FloatWidth::F16),
            Self::Integer(w) => at_least_f32(w.float()),
            Self::Float(FloatWidth::F16) => (Self::Float(FloatWidth::F32), self),
            Self::Float(_) | Self::Complex(_) => (self, self),
            Self::Character => return None,
        })
    }
    /// The dtype that NumPy promotes arguments of this dtype and `other` to: the higher kind, wide enough for both. A Boolean takes the other dtype. An integer beside a float or complex dtype takes at least the float that [`Width::float`] gives, and a float beside a complex dtype takes parts at least as wide. `None` for characters beside numbers.
    pub fn common(self, other: DType) -> Option<DType> {
        use DType::*;
        let (low, high) = if self.kind() <= other.kind() { (self, other) } else { (other, self) };
        Some(match (low, high) {
            (Character, Character) => Character,
            (_, Character) => return None,
            (Boolean, d) => d,
            (Integer(a), Integer(b)) => Integer(a.join(b)),
            (Integer(w), Float(f)) => Float(f.max(w.float())),
            (Integer(w), Complex(c)) => Complex(c.max(w.float().into())),
            (Float(a), Float(b)) => Float(a.max(b)),
            (Float(f), Complex(c)) => Complex(c.max(f.into())),
            (Complex(a), Complex(b)) => Complex(a.max(b)),
            _ => unreachable!("the pair is ordered by kind"),
        })
    }
    /// The real dtype of a complex dtype's parts. Other dtypes are their own.
    pub fn real(self) -> DType {
        match self { Self::Complex(ComplexWidth::F32) => Self::Float(FloatWidth::F32), Self::Complex(ComplexWidth::F64) => Self::Float(FloatWidth::F64), d => d }
    }
    /// The complex dtype that numbers of this dtype take: `complex64` for `f16`, `f32` and `complex64`, and `complex128` for everything else.
    pub fn complex(self) -> DType {
        Self::Complex(match self { Self::Float(FloatWidth::F16 | FloatWidth::F32) | Self::Complex(ComplexWidth::F32) => ComplexWidth::F32, _ => ComplexWidth::F64 })
    }
}

/// `element_types!((callback) args)` calls `callback!` with `args`, then a bracketed list of every element type: `name: (dtype) type` for each, in the order that numbers widen, with characters last. Each name is unique. [`DType::ALL`], [`DType::index`] and [`kernels!`](crate::kernels) come from this list, so their orders agree.
#[doc(hidden)]
#[macro_export]
macro_rules! element_types {
    (($($callback:tt)+) $($args:tt)*) => {
        $($callback)+! {$($args)* [
            Boolean: ($crate::buffer::DType::Boolean) bool,
            U8: ($crate::buffer::DType::Integer($crate::buffer::Width::U8)) u8,
            I16: ($crate::buffer::DType::Integer($crate::buffer::Width::I16)) i16,
            I32: ($crate::buffer::DType::Integer($crate::buffer::Width::I32)) i32,
            I64: ($crate::buffer::DType::Integer($crate::buffer::Width::I64)) i64,
            F16: ($crate::buffer::DType::Float($crate::buffer::FloatWidth::F16)) $crate::half::f16,
            F32: ($crate::buffer::DType::Float($crate::buffer::FloatWidth::F32)) f32,
            F64: ($crate::buffer::DType::Float($crate::buffer::FloatWidth::F64)) f64,
            C32: ($crate::buffer::DType::Complex($crate::buffer::ComplexWidth::F32)) $crate::num_complex::Complex32,
            C64: ($crate::buffer::DType::Complex($crate::buffer::ComplexWidth::F64)) $crate::num_complex::Complex64,
            Character: ($crate::buffer::DType::Character) char
        ]}
    };
}

/// Writes [`DType::ALL`] and [`DType::index`] from the list that [`element_types!`] gives.
macro_rules! dtypes {
    ([$($name:ident: ($($d:tt)+) $t:ty),+]) => {
        impl DType {
            /// Every dtype, in the order that numbers widen, with characters last.
            pub const ALL: [DType; [$(stringify!($name)),+].len()] = [$($($d)+),+];
            /// This dtype's position in [`DType::ALL`].
            #[inline]
            pub const fn index(self) -> usize {
                enum Position { $($name),+ }
                match self { $($($d)+ => Position::$name as usize),+ }
            }
        }
    };
}
element_types!((dtypes));

/// Writes the flat dtype dispatch from the element table, without a second width/type list.
macro_rules! dtype_macro {
    ($d:tt [$($name:ident: ($($dtype:tt)+) $t:ty),+]) => {
        /// Runs the body with its type name standing for the element type of the dtype.
        #[macro_export]
        macro_rules! with_dtype {
            ($d dtype:expr, $d t:ident => $d body:expr) => {
                match $d dtype { $($($dtype)+ => { type $d t = $t; $d body }),+ }
            };
        }
    };
}
element_types!((dtype_macro) $);

impl DType {
    /// The bytes that one item of this dtype takes.
    pub fn bytes(self) -> usize { with_dtype!(self, T => std::mem::size_of::<T>()) }
}

/// Items of one element type.
#[derive(Clone, Debug)]
pub enum Buffer { Booleans(Vec<bool>), Integers(IntBuf), Floats(FloatBuf), Complexes(ComplexBuf), Characters(Vec<char>) }

/// A buffer's items, borrowed in their element type.
#[derive(Clone, Copy, Debug)]
pub enum View<'a> { Booleans(&'a [bool]), Integers(Ints<'a>), Floats(Floats<'a>), Complexes(Complexes<'a>), Characters(&'a [char]) }

/// Compact items at one dtype, borrowed or prepared by gathering and conversion.
pub enum Items<'a> { Borrowed(View<'a>), Owned(Buffer) }

impl<'a> Items<'a> {
    pub fn view(&self) -> View<'_> { match self { Self::Borrowed(v) => *v, Self::Owned(b) => b.view() } }
    pub fn of<T: Element>(&self) -> &[T] { T::of(self.view()).expect("items at their own element type") }
    /// Unpacks typed items, without allocating or converting again.
    pub fn items<T: Element>(self) -> Cow<'a, [T]> {
        match self {
            Self::Borrowed(v) => Cow::Borrowed(T::of(v).expect("items at their own element type")),
            Self::Owned(b) => Cow::Owned(T::vec(b).expect("items at their own element type")),
        }
    }
}

impl Buffer {
    /// An empty buffer of `dtype`, with room for `capacity` items.
    pub fn with_capacity(dtype: DType, capacity: usize) -> Self {
        match dtype {
            DType::Boolean => Self::Booleans(Vec::with_capacity(capacity)), DType::Integer(width) => Self::Integers(IntBuf::with_capacity(width, capacity)),
            DType::Float(width) => Self::Floats(FloatBuf::with_capacity(width, capacity)),
            DType::Complex(width) => Self::Complexes(ComplexBuf::with_capacity(width, capacity)), DType::Character => Self::Characters(Vec::with_capacity(capacity))
        }
    }
    #[inline]
    pub fn view(&self) -> View<'_> {
        match self {
            Self::Booleans(v) => View::Booleans(v), Self::Integers(v) => View::Integers(v.view()), Self::Floats(v) => View::Floats(v.view()),
            Self::Complexes(v) => View::Complexes(v.view()), Self::Characters(v) => View::Characters(v)
        }
    }
    #[inline]
    pub fn len(&self) -> usize { self.view().len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.len() == 0 }
    #[inline]
    pub fn dtype(&self) -> DType { self.view().dtype() }
    pub fn capacity(&self) -> usize {
        match self {
            Self::Booleans(v) => v.capacity(), Self::Integers(v) => v.capacity(), Self::Floats(v) => v.capacity(), Self::Complexes(v) => v.capacity(),
            Self::Characters(v) => v.capacity()
        }
    }
    /// The address of the first item, for writing through, as an exported buffer may.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Booleans(v) => v.as_mut_ptr().cast(), Self::Integers(v) => v.as_mut_ptr(), Self::Floats(v) => v.as_mut_ptr(), Self::Complexes(v) => v.as_mut_ptr(),
            Self::Characters(v) => v.as_mut_ptr().cast()
        }
    }
    /// Copies `range` of `source` when it has this buffer's dtype, and gives whether it could. Integers of another width don't copy.
    pub fn extend_from(&mut self, source: View<'_>, range: std::ops::Range<usize>) -> bool {
        match (self, source) {
            (Self::Booleans(d), View::Booleans(s)) => d.extend_from_slice(&s[range]),
            (Self::Integers(d), View::Integers(s)) => return d.extend_from(s, range),
            (Self::Floats(d), View::Floats(s)) => return d.extend_from(s, range), (Self::Complexes(d), View::Complexes(s)) => return d.extend_from(s, range),
            (Self::Characters(d), View::Characters(s)) => d.extend_from_slice(&s[range]), _ => return false
        }
        true
    }
}
// Buffers are equal when they hold the same items. Numbers of one kind compare by value across widths. Buffers of different kinds are never equal.
impl PartialEq for Buffer { fn eq(&self, other: &Self) -> bool { self.view() == other.view() } }

impl<'a> View<'a> {
    /// Borrows a consecutive range without copying or changing its dtype.
    pub fn slice(self, range: std::ops::Range<usize>) -> Self {
        with_dtype!(self.dtype(), T => T::view(&T::of(self).expect("the view has its dtype")[range]))
    }
    #[inline]
    pub fn len(self) -> usize {
        match self { Self::Booleans(v) => v.len(), Self::Integers(v) => v.len(), Self::Floats(v) => v.len(), Self::Complexes(v) => v.len(), Self::Characters(v) => v.len() }
    }
    #[inline]
    pub fn is_empty(self) -> bool { self.len() == 0 }
    #[inline]
    pub fn as_ptr(self) -> *const u8 {
        match self { Self::Booleans(v) => v.as_ptr().cast(), Self::Integers(v) => v.as_ptr(), Self::Floats(v) => v.as_ptr(), Self::Complexes(v) => v.as_ptr(), Self::Characters(v) => v.as_ptr().cast() }
    }
    #[inline]
    pub fn dtype(self) -> DType {
        match self {
            Self::Booleans(_) => DType::Boolean, Self::Integers(v) => DType::Integer(v.tag()), Self::Floats(v) => DType::Float(v.tag()),
            Self::Complexes(v) => DType::Complex(v.tag()), Self::Characters(_) => DType::Character
        }
    }
}
impl PartialEq for View<'_> {
    fn eq(&self, other: &Self) -> bool {
        match (*self, *other) {
            (Self::Booleans(x), Self::Booleans(y)) => x == y, (Self::Integers(x), Self::Integers(y)) => x == y, (Self::Floats(x), Self::Floats(y)) => x == y,
            (Self::Complexes(x), Self::Complexes(y)) => x == y, (Self::Characters(x), Self::Characters(y)) => x == y, _ => false
        }
    }
}

/// An element type of compact buffers: `bool`, the integer widths `u8`, `i16`, `i32` and `i64`, the float widths `half::f16`, `f32` and `f64`, `Complex32` and `Complex64`, and `char`.
pub trait Element: Copy + Send + Sync {
    /// The item that a missing keyed entry reads as: zero, false or a space.
    const FILL: Self;
    /// The dtype of buffers of this type.
    const DTYPE: DType;
    /// The items of `view`, when they have this type.
    fn of(view: View<'_>) -> Option<&[Self]>;
    /// The writable items of `buffer`, when they have this type. The allocation and length don't change.
    fn of_mut(buffer: &mut Buffer) -> Option<&mut [Self]>;
    /// `items` as a view.
    fn view(items: &[Self]) -> View<'_>;
    /// A buffer that holds `data`.
    fn buffer(data: Vec<Self>) -> Buffer;
    /// The items of `buffer`, moved out, when they have this type.
    fn vec(buffer: Buffer) -> Option<Vec<Self>>;
}

/// An item that reads as type `A`: a Boolean as a number, an integer as a wider integer, a float or a complex number, a float as a float of another width or a complex number, and a complex number as one of the other width.
pub trait Source<A>: Element {
    fn read(self) -> A;
    /// `items` as `A` with no copy, where the items already have type `A`.
    fn borrowed(_items: &[Self]) -> Option<&[A]> { None }
}
impl<A: Element> Source<A> for A {
    #[inline]
    fn read(self) -> A { self }
    #[inline]
    fn borrowed(items: &[A]) -> Option<&[A]> { Some(items) }
}
macro_rules! truth {
    ($($a:ty),+) => {$(
        impl Source<$a> for bool { #[inline] fn read(self) -> $a { self.into() } }
    )+};
}
truth!(u8, i16, i32, i64);
/// A narrower integer reads as a wider one exactly.
macro_rules! wider {
    ($($s:ty => $($a:ty),+);+) => {$($(
        impl Source<$a> for $s { #[inline] fn read(self) -> $a { self.into() } }
    )+)+};
}
wider!(u8 => i16, i32, i64; i16 => i32, i64; i32 => i64);
/// An integer reads as a float through `convert`, whose conversions vectorize on every target.
macro_rules! numbers {
    ($($s:ty),+) => {$(
        impl Source<f64> for $s { #[inline] fn read(self) -> f64 { crate::convert::float(self) } }
        impl Source<f32> for $s { #[inline] fn read(self) -> f32 { crate::convert::float(self) } }
        impl Source<half::f16> for $s { #[inline] fn read(self) -> half::f16 { crate::convert::float(self) } }
    )+};
}
numbers!(u8, i16, i32, i64);
/// A Boolean reads as a float through its byte.
macro_rules! truths {
    ($($a:ty),+) => {$(
        impl Source<$a> for bool { #[inline] fn read(self) -> $a { Source::<$a>::read(u8::from(self)) } }
    )+};
}
truths!(f64, f32, half::f16);
/// A float reads as a wider float exactly, and as a narrower one rounded to the nearest.
macro_rules! floats {
    ($($s:ty => $a:ty: |$x:ident| $read:expr),+) => {$(
        impl Source<$a> for $s { #[inline] fn read(self) -> $a { let $x = self; $read } }
    )+};
}
floats!(
    half::f16 => f64: |x| x.into(), f32 => f64: |x| x.into(), half::f16 => f32: |x| x.into(),
    f64 => f32: |x| x as f32, f32 => half::f16: |x| half::f16::from_f32(x), f64 => half::f16: |x| half::f16::from_f64(x)
);
/// A real number reads as a complex number with no imaginary part.
macro_rules! complex {
    ($($s:ty),+) => {$(
        impl Source<Complex64> for $s { #[inline] fn read(self) -> Complex64 { Complex64::new(Source::<f64>::read(self), 0.0) } }
        impl Source<Complex32> for $s { #[inline] fn read(self) -> Complex32 { Complex32::new(Source::<f32>::read(self), 0.0) } }
    )+};
}
complex!(bool, u8, i16, i32, i64, half::f16, f32, f64);
impl Source<Complex64> for Complex32 {
    #[inline]
    fn read(self) -> Complex64 { Complex64::new(self.re.into(), self.im.into()) }
}
impl Source<Complex32> for Complex64 {
    #[inline]
    fn read(self) -> Complex32 { Complex32::new(self.re as f32, self.im as f32) }
}

/// Every item of `items`, read as type `A`, through the same scheduler as native elementwise calls.
pub fn read_all<A: Element, S: Source<A>>(items: &[S]) -> Vec<A> {
    let mut out = Vec::with_capacity(items.len());
    crate::pairs(crate::Arg::Many(items), crate::Arg::One(()), items.len(), &mut out, |x, ()| (x.read(), true));
    out
}

/// The real items of `view` read as `A`, or `None` for complex numbers or characters.
fn reals<A: Element>(view: View<'_>, mapping: &crate::agreement::Mapping, n: usize) -> Option<Vec<A>>
where bool: Source<A>, u8: Source<A>, i16: Source<A>, i32: Source<A>, i64: Source<A>, half::f16: Source<A>, f32: Source<A>, f64: Source<A> {
    Some(match view { View::Booleans(v) => read_as(v, mapping, n), View::Integers(v) => with_ints!(v, |x| read_as(x, mapping, n)), View::Floats(v) => with_floats!(v, |x| read_as(x, mapping, n)), _ => return None })
}

/// Gathers source items in blocks, then converts each block contiguously.
fn read_as<A: Element, S: Source<A>>(items: &[S], mapping: &crate::agreement::Mapping, n: usize) -> Vec<A> {
    if matches!(mapping, crate::agreement::Mapping::Linear(1)) { return read_all(&items[..n]); }
    let mut out = Vec::with_capacity(n);
    crate::agreement::block(items, mapping, n, &mut out, |x, n, slots| slots.pairs(x, crate::Arg::One(()), n, |x, ()| (x.read(), true)));
    out
}

/// An Overflow error for a value `x` that doesn't fit the dtype it converts to.
fn overflow(x: impl std::fmt::Display) -> Error { Error::new(Kind::Overflow, format!("{x} doesn't fit the dtype it converts to")) }

/// The real part of `z`, or a Domain error where its imaginary part isn't zero.
fn real_part(z: Complex64) -> Result<f64, Error> {
    if z.im == 0.0 { Ok(z.re) } else { Err(Error::new(Kind::Domain, format!("{z} has an imaginary part, so it doesn't convert to a real dtype"))) }
}

/// Item `i` of a numeric `view` as an `i64`. A float truncates toward zero, and gives an Overflow error where it doesn't fit, as do NaN and infinities.
fn integer(view: View<'_>, i: usize) -> Result<i64, Error> {
    let x = match view {
        View::Booleans(v) => return Ok(v[i].into()),
        View::Integers(v) => return Ok(v.get(i)),
        View::Floats(v) => v.get(i),
        View::Complexes(v) => real_part(v.get(i))?,
        View::Characters(_) => unreachable!("cast refuses characters first"),
    };
    let t = x.trunc();
    // -2^63 and 2^63 are exact floats. NaN fails both comparisons.
    if (-9.223372036854775808e18..9.223372036854775808e18).contains(&t) { Ok(t as i64) } else { Err(overflow(x)) }
}

/// Item `i` of a numeric `view` as a Boolean. Only exactly 0 and 1 convert, and anything else gives an Overflow error.
fn boolean(view: View<'_>, i: usize) -> Result<bool, Error> {
    let x = match view {
        View::Booleans(v) => return Ok(v[i]),
        View::Integers(v) => return match v.get(i) { 0 => Ok(false), 1 => Ok(true), k => Err(overflow(k)) },
        View::Floats(v) => v.get(i),
        View::Complexes(v) => real_part(v.get(i))?,
        View::Characters(_) => unreachable!("cast refuses characters first"),
    };
    if x == 0.0 { Ok(false) } else if x == 1.0 { Ok(true) } else { Err(overflow(x)) }
}

impl Buffer {
    /// Flattened positions whose item differs from its type's fill (zero, false or a space).
    pub fn nonzero(view: View<'_>) -> Vec<usize> {
        with_dtype!(view.dtype(), T => T::of(view).expect("matching dtype").iter().enumerate().filter_map(|(i, &x)| (x != T::FILL).then_some(i)).collect())
    }
    /// Copies the first `n` items of a structural walk from `base`, using the type's fill for missing positions and whole-slice copies for contiguous runs.
    pub fn walked(view: View<'_>, base: usize, steps: &[crate::gather::Steps], n: usize) -> Self {
        with_dtype!(view.dtype(), T => {
            let mut out = Vec::with_capacity(n);
            crate::gather::walk_range(&mut out, T::of(view).expect("matching dtype"), base, steps, T::FILL, 0, n, crate::gather::Same);
            T::buffer(out)
        })
    }
    /// A new buffer holding the items of `view` in `range`.
    pub fn copied(view: View<'_>, range: std::ops::Range<usize>) -> Self {
        let mut b = Self::with_capacity(view.dtype(), range.len());
        b.extend_from(view, range);
        b
    }
    /// The items of `view` at `indices`, as [`take`](crate::gather::take) reads them: a negative index counts back from the end.
    pub fn gathered<I: Copy + Into<i64>>(view: View<'_>, indices: &[I]) -> Self {
        fn taken<T: Element, I: Copy + Into<i64>>(items: &[T], indices: &[I]) -> Buffer {
            let mut out = Vec::with_capacity(indices.len());
            crate::gather::take(&mut out, items, indices, |x| x);
            T::buffer(out)
        }
        match view {
            View::Booleans(v) => taken(v, indices),
            View::Characters(v) => taken(v, indices),
            View::Integers(v) => with_ints!(v, |v| taken(v, indices)),
            View::Floats(v) => with_floats!(v, |v| taken(v, indices)),
            View::Complexes(v) => with_complexes!(v, |v| taken(v, indices)),
        }
    }
    /// The items of `view` at `dtype`. Matching dtypes copy. Otherwise each item converts as [`Source`] reads it, and conversions to a narrower dtype follow these rules. A float converts to an integer by truncating toward zero. A value that doesn't fit its target gives an Overflow error, as do NaN, infinities and anything but exactly 0 or 1 for Booleans. A complex number converts to a real dtype only when its imaginary part is zero, and otherwise gives a Domain error. Characters convert only to characters.
    pub fn cast(view: View<'_>, dtype: DType) -> Result<Self, Error> {
        Self::cast_mapped(view, &crate::agreement::Mapping::Linear(1), view.len(), dtype)
    }
    /// Shared conversion dispatch for compact and mapped logical items. Unselected storage never participates in checked conversion.
    pub(crate) fn cast_mapped(view: View<'_>, mapping: &crate::agreement::Mapping, n: usize, dtype: DType) -> Result<Self, Error> {
        if view.dtype() == dtype {
            if matches!(mapping, crate::agreement::Mapping::Linear(1)) { return Ok(Self::copied(view, 0..n)); }
            if let crate::agreement::Mapping::Steps { base, steps } = mapping { return Ok(Self::walked(view, *base, steps, n)); }
            return Ok(with_dtype!(dtype, T => T::buffer(read_as::<T, T>(T::of(view).expect("matching dtype"), mapping, n))));
        }
        Ok(match (dtype, view) {
            (DType::Character, _) | (_, View::Characters(_)) => return Err(Error::new(Kind::Domain, "characters convert only to characters")),
            (DType::Integer(w), View::Booleans(v)) => with_width!(w, A => A::buffer(read_as::<A, bool>(v, mapping, n))),
            (DType::Integer(Width::I64), View::Integers(v)) => i64::buffer(with_ints!(v, |x| read_as(x, mapping, n))),
            (DType::Integer(Width::I32), View::Integers(Ints::I16(v))) => i32::buffer(read_as(v, mapping, n)),
            (DType::Integer(Width::I32), View::Integers(Ints::U8(v))) => i32::buffer(read_as(v, mapping, n)),
            (DType::Integer(Width::I16), View::Integers(Ints::U8(v))) => i16::buffer(read_as(v, mapping, n)),
            (DType::Integer(w), _) => with_width!(w, A => A::buffer((0..n).map(|i| mapping.get(i).map_or(Ok(0), |j| integer(view, j)).and_then(|k| A::try_from(k).map_err(|_| overflow(k)))).collect::<Result<_, _>>()?)),
            (DType::Boolean, _) => Self::Booleans((0..n).map(|i| mapping.get(i).map_or(Ok(false), |j| boolean(view, j))).collect::<Result<_, _>>()?),
            (DType::Float(w), View::Complexes(v)) => with_float_width!(w, A => A::buffer((0..n).map(|i| mapping.get(i).map_or(Ok(0.), |j| real_part(v.get(j))).map(Source::<A>::read)).collect::<Result<_, _>>()?)),
            (DType::Float(w), v) => with_float_width!(w, A => A::buffer(reals::<A>(v, mapping, n).expect("cast handles complex numbers and characters first"))),
            (DType::Complex(w), View::Complexes(v)) => with_complex_width!(w, A => A::buffer(with_complexes!(v, |x| read_as(x, mapping, n)))),
            (DType::Complex(w), v) => with_complex_width!(w, A => A::buffer(reals::<A>(v, mapping, n).expect("cast handles characters first"))),
        })
    }
}
