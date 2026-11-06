//! Compact buffers: owned items of one element type, borrowed views of them, and the [`DType`] that names each. Booleans take one byte each. Integers come in four widths, floats in three and complex numbers in two, named by the width of their parts. [`Source`] reads an item of one element type as another.
use crate::error::{Error, Kind};
use num_complex::{Complex32, Complex64};
use std::borrow::Cow;

/// Declares the owned buffers `$owned`, their borrowed view `$view` and the width `$tag` that names a variant, with one variant for each element type, which fills with `$fill`. `$outer` is the family's variant in `Buffer` and `View`, and `$widest` its widest type, which `widened` reads every width as. Code that needs a variant's element type matches on the variant, as `with_ints!` does.
macro_rules! buffers {
    ($owned:ident, $view:ident, $tag:ident, $outer:ident, $kind:ident, $widest:ty: $($variant:ident($t:ty = $fill:expr)),+) => {
        #[derive(Clone, Debug)]
        pub enum $owned { $($variant(Vec<$t>)),+ }
        #[derive(Clone, Copy, Debug)]
        pub enum $view<'a> { $($variant(&'a [$t])),+ }
        #[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
        pub enum $tag { $($variant),+ }
        impl $owned {
            pub fn with_capacity(tag: $tag, capacity: usize) -> Self { match tag { $($tag::$variant => Self::$variant(Vec::with_capacity(capacity))),+ } }
            #[inline]
            pub fn view(&self) -> $view<'_> { match self { $(Self::$variant(v) => $view::$variant(v)),+ } }
            #[inline]
            pub fn capacity(&self) -> usize { match self { $(Self::$variant(v) => v.capacity()),+ } }
            #[inline]
            pub fn as_mut_ptr(&mut self) -> *mut u8 { match self { $(Self::$variant(v) => v.as_mut_ptr().cast()),+ } }
            /// Sets the length to `n` after a writer outside Rust, such as compiled code, filled the first `n` items through `as_mut_ptr`.
            ///
            /// # Safety
            /// `n` is at most the capacity, and the first `n` items are initialized.
            #[inline]
            pub unsafe fn set_len(&mut self, n: usize) { match self { $(Self::$variant(v) => unsafe { v.set_len(n) }),+ } }
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
            #[inline]
            fn vec_mut(buffer: &mut Buffer) -> Option<&mut Vec<Self>> { match buffer { Buffer::$outer($owned::$variant(v)) => Some(v), _ => None } }
            #[inline]
            fn truth(b: bool) -> Self { Source::<Self>::read(b) }
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
    ($($t:ty: $variant:ident($dtype:ident) = $fill:expr, $truth:expr),+) => {$(
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
            #[inline]
            fn vec_mut(buffer: &mut Buffer) -> Option<&mut Vec<Self>> { match buffer { Buffer::$variant(v) => Some(v), _ => None } }
            #[inline]
            fn truth(b: bool) -> Self { ($truth)(b) }
        }
    )+};
}
single!(bool: Booleans(Boolean) = false, |b| b, char: Characters(Character) = ' ', |b| if b { '\u{1}' } else { '\0' });

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
            $crate::buffer::DType::Float($crate::buffer::FloatWidth::F32) => { type $t = f32; $body }
            $crate::buffer::DType::Float($crate::buffer::FloatWidth::F64) => { type $t = f64; $body }
            $crate::buffer::DType::Complex($crate::buffer::ComplexWidth::F32) => { type $t = $crate::num_complex::Complex32; $body }
            $crate::buffer::DType::Complex($crate::buffer::ComplexWidth::F64) => { type $t = $crate::num_complex::Complex64; $body }
            _ => panic!("with_inexact! takes a computing dtype that DType::inexact gives"),
        }
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
    #[inline]
    fn bytes(self) -> usize { with_width!(self, T => std::mem::size_of::<T>()) }
    #[inline]
    pub fn holds(self, n: i64) -> bool { with_width!(self, T => T::try_from(n).is_ok()) }
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
    /// The float width that integers of this width become beside a float, no wider than `limit`. With a `limit` of `f64`, this is NumPy's rule: the narrowest float that holds each of them exactly. That's `f16` for `u8`, `f32` for `i16` and `f64` for `i32`. `i64` also gives `f64`, which holds its integers exactly only up to 2^53. A session whose default float is narrower passes that width as `limit`.
    #[inline]
    pub fn float(self, limit: FloatWidth) -> FloatWidth {
        (match self { Self::U8 => FloatWidth::F16, Self::I16 => FloatWidth::F32, Self::I32 | Self::I64 => FloatWidth::F64 }).min(limit)
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
        match width { ComplexWidth::F32 => Self::F32(values.map(Source::read).collect()), ComplexWidth::F64 => Self::F64(values.collect()) }
    }
}

/// The element type of a compact buffer. Numbers widen from Boolean through the integer widths to the float widths to the complex widths.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType { Boolean, Integer(Width), Float(FloatWidth), Complex(ComplexWidth), Character }

impl DType {
    /// Whether items of this dtype are whole numbers: Booleans and integers.
    #[inline]
    pub fn is_integral(self) -> bool { matches!(self, Self::Boolean | Self::Integer(_)) }
    /// Whether Mathish forms (`tolerant::MathishFloorDivide` and the rest) apply to a call that computes at this dtype: 64-bit floats and complex numbers with 64-bit parts.
    #[inline]
    pub fn mathish(self) -> bool { matches!(self, Self::Float(FloatWidth::F64) | Self::Complex(ComplexWidth::F64)) }
    /// Whether items of this dtype are real numbers: Booleans, integers and floats.
    #[inline]
    pub fn is_real(self) -> bool { matches!(self, Self::Boolean | Self::Integer(_) | Self::Float(_)) }
    /// This dtype's kind, in the order that NEP 50 promotes kinds: 0 for Booleans, 1 for integers, 2 for floats, 3 for complex numbers and 4 for characters, which don't mix with numbers.
    pub fn kind(self) -> u8 {
        match self { Self::Boolean => 0, Self::Integer(_) => 1, Self::Float(_) => 2, Self::Complex(_) => 3, Self::Character => 4 }
    }
    /// The dtype's name, as NumPy names its dtypes: `"bool"`, `"uint8"`, `"int16"`, `"int32"`, `"int64"`, `"float16"`, `"float32"`, `"float64"`, `"complex64"` and `"complex128"`. NumPy has no dtype for characters, which are `"character"`. Rustymath's Python API and BPL's `•storage` both use these names.
    pub fn name(self) -> &'static str {
        match self {
            Self::Boolean => "bool", Self::Integer(Width::U8) => "uint8", Self::Integer(Width::I16) => "int16", Self::Integer(Width::I32) => "int32",
            Self::Integer(Width::I64) => "int64", Self::Float(FloatWidth::F16) => "float16", Self::Float(FloatWidth::F32) => "float32",
            Self::Float(FloatWidth::F64) => "float64", Self::Complex(ComplexWidth::F32) => "complex64", Self::Complex(ComplexWidth::F64) => "complex128",
            Self::Character => "character"
        }
    }
    /// The dtype that [`DType::name`] calls `name`, or `None` for any other name.
    pub fn named(name: &str) -> Option<Self> { Self::ALL.into_iter().find(|d| d.name() == name) }
    /// Whether this dtype widens to `to`: whether [`DType::common`] promotes this dtype and `to` to `to`. An integer widens to every float and complex dtype.
    pub fn widens_to(self, to: DType) -> bool { self.common(to) == Some(to) }
    /// The dtypes that a function of floats or complex numbers computes in and gives, for arguments of this dtype. A Boolean or an integer takes the float that [`Width::float`] gives under `limit`, but at least `f32`, which computes faster than `f16`. Under NumPy's limit, `f64`, that's `f32` for Booleans, `u8` and `i16`, and `f64` for `i32` and `i64`. `f16` computes in `f32`, and its results round back. Other floats and complex numbers stay as they are. `None` for characters.
    pub fn inexact(self, limit: FloatWidth) -> Option<(DType, DType)> {
        let at_least_f32 = |w: FloatWidth| { let f = Self::Float(w.max(FloatWidth::F32)); (f, f) };
        Some(match self {
            Self::Boolean => at_least_f32(FloatWidth::F16), Self::Integer(w) => at_least_f32(w.float(limit)),
            Self::Float(FloatWidth::F16) => (Self::Float(FloatWidth::F32), self), Self::Float(_) | Self::Complex(_) => (self, self),
            Self::Character => return None
        })
    }
    /// The dtype that an argument of this dtype computes in, for a function with a kernel at each dtype where `has` holds. That's this dtype where it has a kernel. Otherwise it's the first with a kernel of these: for a Boolean or an integer, integers of `int` or of this dtype's width, whichever is wider, then floats of `float`; for a float or a complex number, the dtype that [`DType::inexact`] gives it; then each wider float, and each complex dtype whose parts are at least as wide; and last, for a float, integers of `int`, which compute it as the whole number it holds (`operation::whole_numbers`). `int` and `float` are the caller's policy for exact numbers. NumPy's is `i64` and the float that `inexact` gives under `f64`. `None` where no dtype on the way has a kernel.
    pub fn computes_in(self, int: Width, float: FloatWidth, has: impl Fn(DType) -> bool) -> Option<DType> {
        if has(self) { return Some(self); }
        let (integer, float, whole) = match self {
            Self::Boolean => (Some(Self::Integer(int)), Self::Float(float), None), Self::Integer(w) => (Some(Self::Integer(w.join(int))), Self::Float(float), None),
            Self::Float(_) => (None, self.inexact(float)?.0, Some(Self::Integer(int))),
            _ => (None, self.inexact(float)?.0, None)
        };
        integer.into_iter().chain(Self::ALL.into_iter().filter(|&d| float.widens_to(d))).chain(whole).find(|&d| has(d))
    }
    /// The dtype that arguments of this dtype and `other` promote to: the higher kind, wide enough for both. A Boolean takes the other dtype. An integer beside a float or complex dtype takes that dtype, as in JAX, where NumPy widens the float to hold every integer exactly. A float beside a complex dtype takes parts at least as wide. `None` for characters beside numbers.
    pub fn common(self, other: DType) -> Option<DType> {
        use DType::*;
        let (low, high) = if self.kind() <= other.kind() { (self, other) } else { (other, self) };
        Some(match (low, high) {
            (Character, Character) => Character, (_, Character) => return None, (Boolean, d) | (Integer(_), d @ (Float(_) | Complex(_))) => d,
            (Integer(a), Integer(b)) => Integer(a.join(b)), (Float(a), Float(b)) => Float(a.max(b)), (Float(f), Complex(c)) => Complex(c.max(f.into())),
            (Complex(a), Complex(b)) => Complex(a.max(b)), _ => unreachable!("the pair is ordered by kind")
        })
    }
    /// The dtype that NEP 50 promotes an array of this dtype and a weak value of dtype `weak` to. A weak value is a Python scalar, which has no dtype of its own. BPL's weak numbers follow the same rule. The caller gives the weak value's dtype at the default width of its kind. Where its kind is no higher, the array's dtype wins. A weak complex value beside floats takes their width. Otherwise the weak dtype wins. `None` for characters beside numbers.
    pub fn common_weak(self, weak: DType) -> Option<DType> {
        use DType::*;
        Some(match (self, weak) {
            (Character, Character) => Character, (Character, _) | (_, Character) => return None, (Float(_) | Complex(_), Complex(_)) => self.complex(),
            _ if weak.kind() > self.kind() => weak, _ => self
        })
    }
    /// The real dtype of a complex dtype's parts. Other dtypes are their own.
    pub fn real(self) -> DType {
        match self {
            Self::Complex(ComplexWidth::F32) => Self::Float(FloatWidth::F32), Self::Complex(ComplexWidth::F64) => Self::Float(FloatWidth::F64), d => d
        }
    }
    /// The complex dtype that numbers of this dtype take: `complex64` for `f16`, `f32` and `complex64`, and `complex128` for everything else.
    pub fn complex(self) -> DType {
        Self::Complex(match self {
            Self::Float(FloatWidth::F16 | FloatWidth::F32) | Self::Complex(ComplexWidth::F32) => ComplexWidth::F32, _ => ComplexWidth::F64
        })
    }
}

/// `element_types!((callback) args)` calls `callback!` with `args`, then a bracketed list of every element type: `name: (dtype) type` for each, in the order that numbers widen, with characters last. Each name is unique. [`DType::ALL`], [`DType::index`] and [`kernels!`](crate::kernels) come from this list, so their orders agree.
#[doc(hidden)]
#[macro_export]
macro_rules! element_types {
    (($($callback:tt)+) $($args:tt)*) => {
        $($callback)+! {$($args)* [
            Boolean: ($crate::buffer::DType::Boolean) bool, U8: ($crate::buffer::DType::Integer($crate::buffer::Width::U8)) u8,
            I16: ($crate::buffer::DType::Integer($crate::buffer::Width::I16)) i16, I32: ($crate::buffer::DType::Integer($crate::buffer::Width::I32)) i32,
            I64: ($crate::buffer::DType::Integer($crate::buffer::Width::I64)) i64,
            F16: ($crate::buffer::DType::Float($crate::buffer::FloatWidth::F16)) $crate::half::f16,
            F32: ($crate::buffer::DType::Float($crate::buffer::FloatWidth::F32)) f32, F64: ($crate::buffer::DType::Float($crate::buffer::FloatWidth::F64)) f64,
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
            DType::Complex(width) => Self::Complexes(ComplexBuf::with_capacity(width, capacity)),
            DType::Character => Self::Characters(Vec::with_capacity(capacity))
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
    /// Reserve room for `additional` items after the current length.
    pub fn reserve(&mut self, additional: usize) { with_dtype!(self.dtype(), T => T::vec_mut(self).expect("the buffer has its dtype").reserve(additional)) }
    /// The address of the first item, for writing through, as an exported buffer may.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            Self::Booleans(v) => v.as_mut_ptr().cast(), Self::Integers(v) => v.as_mut_ptr(), Self::Floats(v) => v.as_mut_ptr(),
            Self::Complexes(v) => v.as_mut_ptr(), Self::Characters(v) => v.as_mut_ptr().cast()
        }
    }
    /// Copies `range` of `source` when it has this buffer's dtype, and gives whether it could. Integers of another width don't copy.
    pub fn extend_from(&mut self, source: View<'_>, range: std::ops::Range<usize>) -> bool {
        match (self, source) {
            (Self::Booleans(d), View::Booleans(s)) => d.extend_from_slice(&s[range]),
            (Self::Integers(d), View::Integers(s)) => return d.extend_from(s, range), (Self::Floats(d), View::Floats(s)) => return d.extend_from(s, range),
            (Self::Complexes(d), View::Complexes(s)) => return d.extend_from(s, range),
            (Self::Characters(d), View::Characters(s)) => d.extend_from_slice(&s[range]), _ => return false
        }
        true
    }
    /// Sets the length to `n` after a writer outside Rust, such as compiled code, filled the first `n` items through `as_mut_ptr`.
    ///
    /// # Safety
    /// `n` is at most the capacity, and the first `n` items are initialized.
    pub unsafe fn set_len(&mut self, n: usize) {
        match self {
            Self::Booleans(v) => unsafe { v.set_len(n) }, Self::Integers(v) => unsafe { v.set_len(n) }, Self::Floats(v) => unsafe { v.set_len(n) },
            Self::Complexes(v) => unsafe { v.set_len(n) }, Self::Characters(v) => unsafe { v.set_len(n) }
        }
    }
    /// Removes the items, keeping the allocation.
    pub fn clear(&mut self) {
        // SAFETY: no items remain, and items have no destructors.
        unsafe { self.set_len(0) }
    }
}

/// A fill that a call chooses: the item that a read gives at each position that reaches no item, as one native item, checked at the dtype that reads it. A read with no chosen fill gives its dtype's [`Element::FILL`]. A frontend's own storage, such as basedpl's nested arrays, keeps its fills in the frontend.
#[derive(Clone, Debug)]
pub struct Fill(Buffer);

impl Fill {
    /// The one item of `value` at `dtype`, where the conversion keeps its value, as [`Buffer::kept`] converts it. A Length error where `value` doesn't hold one item, and a Domain error where the item doesn't keep its value.
    pub fn new(value: View<'_>, dtype: DType) -> Result<Self, Error> {
        if value.len() != 1 { return Err(Error::new(Kind::Length, "a fill must contain one item")); }
        Buffer::kept(value, dtype).map(Self)
    }
    pub fn view(&self) -> View<'_> { self.0.view() }
    /// Convert with the source's items, including its rounding and truncation.
    pub fn cast(&self, dtype: DType) -> Result<Self, Error> { Buffer::cast(self.view(), dtype).map(Self) }
    /// The item that a read of items of type `A` gives at a position that reaches no item: the chosen `fill`, made for items of type `A`, or `A`'s own fill where there's none.
    #[inline]
    pub fn item<A: Element>(fill: Option<&std::sync::Arc<Self>>) -> A { fill.map_or(A::FILL, |f| A::of(f.view()).expect("a fill has its source's dtype")[0]) }
}
impl PartialEq for Fill { fn eq(&self, other: &Self) -> bool { self.view().identical(other.view()) } }
impl Eq for Fill {}
// Buffers are equal when they hold the same items. Numbers of one kind compare by value across widths. Buffers of different kinds are never equal.
impl PartialEq for Buffer { fn eq(&self, other: &Self) -> bool { self.view() == other.view() } }

impl<'a> View<'a> {
    /// Borrows a consecutive range without copying or changing its dtype.
    #[inline]
    pub fn slice(self, range: std::ops::Range<usize>) -> Self { with_dtype!(self.dtype(), T => T::view(&T::of(self).expect("the view has its dtype")[range])) }
    #[inline]
    pub fn len(self) -> usize {
        match self {
            Self::Booleans(v) => v.len(), Self::Integers(v) => v.len(), Self::Floats(v) => v.len(), Self::Complexes(v) => v.len(),
            Self::Characters(v) => v.len()
        }
    }
    #[inline]
    pub fn is_empty(self) -> bool { self.len() == 0 }
    #[inline]
    pub fn as_ptr(self) -> *const u8 {
        match self {
            Self::Booleans(v) => v.as_ptr().cast(), Self::Integers(v) => v.as_ptr(), Self::Floats(v) => v.as_ptr(), Self::Complexes(v) => v.as_ptr(),
            Self::Characters(v) => v.as_ptr().cast()
        }
    }
    #[inline]
    pub fn dtype(self) -> DType {
        match self {
            Self::Booleans(_) => DType::Boolean, Self::Integers(v) => DType::Integer(v.tag()), Self::Floats(v) => DType::Float(v.tag()),
            Self::Complexes(v) => DType::Complex(v.tag()), Self::Characters(_) => DType::Character
        }
    }
    /// Whether the views hold the same items of one dtype, bit for bit. `-0.0` differs from `0.0`, and a NaN is identical to a NaN with the same bits. `==` instead compares values across widths.
    pub fn identical(self, other: View<'_>) -> bool {
        // SAFETY: every element type is plain data with no padding, so a view's items read as initialized bytes.
        let bytes = |v: View<'_>| unsafe { std::slice::from_raw_parts(v.as_ptr(), v.len() * v.dtype().bytes()) };
        self.dtype() == other.dtype() && bytes(self) == bytes(other)
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
pub trait Element: Copy + Send + Sync + 'static + crate::gather::Item {
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
    /// The items of `buffer` as their vector, when they have this type, for a writer that appends or clears them.
    fn vec_mut(buffer: &mut Buffer) -> Option<&mut Vec<Self>>;
    /// `b` as an item of this type: 1 or 0 for a number, the Boolean itself, or the characters U+0001 and U+0000. A Boolean mask becomes lanes of the items that a kernel such as [`Choose`](crate::Choose) chooses between. A number converts through an integer, which suits stored Booleans. A comparison's lane mask turns into 1 or 0 faster by a choice between the two, so `Indicator` writes its own (`tolerant::indicated`).
    fn truth(b: bool) -> Self;
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

/// An Overflow error for a value `x` that doesn't fit the dtype it converts to.
fn overflow(x: impl std::fmt::Display) -> Error { Error::new(Kind::Overflow, format!("{x} doesn't fit the dtype it converts to")) }

/// The real part of `z`, or a Domain error where its imaginary part isn't zero.
fn real_part(z: Complex64) -> Result<f64, Error> {
    if z.im == 0.0 { Ok(z.re) } else { Err(Error::new(Kind::Domain, format!("{z} has an imaginary part, so it doesn't convert to a real dtype"))) }
}

/// Item `i` of a numeric `view` as an `i64`. A float truncates toward zero, and gives an Overflow error where it doesn't fit, as do NaN and infinities.
fn integer(view: View<'_>, i: usize) -> Result<i64, Error> {
    let x = match view {
        View::Booleans(v) => return Ok(v[i].into()), View::Integers(v) => return Ok(v.get(i)), View::Floats(v) => v.get(i),
        View::Complexes(v) => real_part(v.get(i))?, View::Characters(_) => unreachable!("cast refuses characters first")
    };
    let cast =
        crate::convert::Conversion { input: DType::Float(FloatWidth::F64), output: DType::Integer(Width::I64), kind: crate::convert::ConversionKind::Cast };
    if cast.validity::<f64, _>(&crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT, x).is_some_and(|(_, valid)| valid) { Ok(x as i64) }
    else { Err(overflow(x)) }
}

/// Item `i` of a numeric `view` as a Boolean. Only exactly 0 and 1 convert, and anything else gives an Overflow error.
fn boolean(view: View<'_>, i: usize) -> Result<bool, Error> {
    let x = match view {
        View::Booleans(v) => return Ok(v[i]), View::Integers(v) => { return match v.get(i) { 0 => Ok(false), 1 => Ok(true), k => Err(overflow(k)) }; }
        View::Floats(v) => v.get(i), View::Complexes(v) => real_part(v.get(i))?, View::Characters(_) => unreachable!("cast refuses characters first")
    };
    let cast = crate::convert::Conversion { input: DType::Float(FloatWidth::F64), output: DType::Boolean, kind: crate::convert::ConversionKind::Cast };
    if cast.validity::<f64, _>(&crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT, x).is_some_and(|(_, valid)| valid) { Ok(x != 0.0) }
    else { Err(overflow(x)) }
}

/// Convert stored items using the destination dtype's checked conversion.
pub trait Cast: Element {
    /// Item `item` of `view`, or an error where it doesn't convert.
    fn cast(view: View<'_>, item: usize) -> Result<Self, Error>;
    /// Appends the first `len` items of `view` to `out`, each as [`Cast::cast`] converts it and [`kept`] keeps it. Element types override this with one typed loop for each source dtype they read often.
    fn extend_cast(view: View<'_>, len: usize, checked: bool, out: &mut Vec<Self>) -> Result<(), Error> { itemwise(view, len, checked, out) }
}
/// `value`, or the fill where it didn't convert, unless `checked`.
pub(crate) fn kept<T: Element>(value: Result<T, Error>, checked: bool) -> Result<T, Error> {
    match value { Ok(value) => Ok(value), Err(error) if checked => Err(error), Err(_) => Ok(T::FILL) }
}
/// [`Cast::extend_cast`] one item at a time, through [`Cast::cast`].
fn itemwise<T: Cast>(view: View<'_>, len: usize, checked: bool, out: &mut Vec<T>) -> Result<(), Error> {
    out.reserve(len);
    for i in 0..len { out.push(kept(T::cast(view, i), checked)?); }
    Ok(())
}
fn numeric(view: View<'_>) -> Result<(), Error> {
    if matches!(view, View::Characters(_)) { Err(Error::new(Kind::Domain, "characters convert only to characters")) } else { Ok(()) }
}
fn real(view: View<'_>, item: usize) -> Result<f64, Error> {
    Ok(match view {
        View::Booleans(v) => u8::from(v[item]).into(), View::Integers(v) => Source::<f64>::read(v.get(item)), View::Floats(v) => v.get(item),
        View::Complexes(v) => real_part(v.get(item))?, View::Characters(_) => return Err(Error::new(Kind::Domain, "characters convert only to characters"))
    })
}
impl Cast for bool { fn cast(view: View<'_>, item: usize) -> Result<Self, Error> { numeric(view)?; boolean(view, item) } }
impl Cast for char {
    fn cast(view: View<'_>, item: usize) -> Result<Self, Error> {
        match view { View::Characters(v) => Ok(v[item]), _ => Err(Error::new(Kind::Domain, "characters convert only to characters")) }
    }
}
macro_rules! casts {
    (integers $($t:ty),+) => {$(impl Cast for $t {
        fn cast(view: View<'_>, item: usize) -> Result<Self, Error> {
            numeric(view)?;
            let value = integer(view, item)?;
            Self::try_from(value).map_err(|_| overflow(value))
        }
        // A checked cast first looks for an item that doesn't fit. The typed loop gives the fill for one.
        fn extend_cast(view: View<'_>, len: usize, checked: bool, out: &mut Vec<Self>) -> Result<(), Error> {
            match view {
                View::Booleans(v) => out.extend(v[..len].iter().map(|&b| Self::from(b))),
                View::Integers(v) => with_ints!(v, |v| {
                    let v = &v[..len];
                    if checked { if let Some(&x) = v.iter().find(|&&x| Self::try_from(x).is_err()) { return Err(overflow(x)); } }
                    out.extend(v.iter().map(|&x| Self::try_from(x).unwrap_or(Self::FILL)));
                }),
                _ => return itemwise(view, len, checked, out),
            }
            Ok(())
        }
    })+};
    (floats $($t:ty),+) => {$(impl Cast for $t {
        fn cast(view: View<'_>, item: usize) -> Result<Self, Error> { real(view, item).map(Source::<Self>::read) }
        // Converting a real item can't fail.
        fn extend_cast(view: View<'_>, len: usize, checked: bool, out: &mut Vec<Self>) -> Result<(), Error> {
            match view {
                View::Booleans(v) => out.extend(v[..len].iter().map(|&b| Source::<Self>::read(f64::from(u8::from(b))))),
                View::Integers(v) => with_ints!(v, |v| out.extend(v[..len].iter().map(|&x| Source::<Self>::read(Source::<f64>::read(i64::from(x)))))),
                View::Floats(v) => with_floats!(v, |v| out.extend(v[..len].iter().map(|&x| Source::<Self>::read(f64::from(x))))),
                _ => return itemwise(view, len, checked, out),
            }
            Ok(())
        }
    })+};
    (complex $($t:ty),+) => {$(impl Cast for $t {
        fn cast(view: View<'_>, item: usize) -> Result<Self, Error> {
            let value = match view { View::Complexes(v) => v.get(item), _ => Complex64::new(real(view, item)?, 0.0) };
            Ok(Source::<Self>::read(value))
        }
    })+};
}
casts!(integers u8, i16, i32, i64);
casts!(floats half::f16, f32, f64);
casts!(complex Complex32, Complex64);

impl Buffer {
    /// Flattened positions whose item differs from its type's fill (zero, false or a space).
    pub fn nonzero(view: View<'_>) -> Vec<usize> {
        with_dtype!(view.dtype(), T => T::of(view).expect("matching dtype").iter().enumerate().filter_map(|(i, &x)| (x != T::FILL).then_some(i)).collect())
    }
    /// Whether every item has the first item's bits, as an array of one repeated value does: `0` and `¯0` differ, and a NaN matches only a NaN with its bits. An empty view gives true.
    pub fn same(view: View<'_>) -> bool {
        let size = view.dtype().bytes();
        // SAFETY: a view holds `len` initialised items of `size` bytes each at `as_ptr`, and no element type has padding.
        let bytes = unsafe { std::slice::from_raw_parts(view.as_ptr(), view.len() * size) };
        bytes.chunks_exact(size).all(|item| item == &bytes[..size])
    }
    /// Copies the first `n` items of a structural walk from `base`, using the type's fill for missing positions and whole-slice copies for contiguous runs. `n` can be shorter than the walk, as it is for a view whose mapping reaches past its own items.
    pub fn walked(view: View<'_>, base: usize, steps: &[crate::gather::Steps], n: usize) -> Self {
        with_dtype!(view.dtype(), T => {
            let mut out = Vec::with_capacity(n);
            let (base, plan) = crate::gather::Steps::simplify(base, steps);
            crate::gather::walk_plan(&mut out, T::of(view).expect("matching dtype"), base, &plan, T::FILL, 0, n, crate::gather::Same);
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
    pub fn gathered<I: crate::gather::IndexValue>(view: View<'_>, indices: &[I]) -> Self {
        fn taken<T: Element, I: crate::gather::IndexValue>(items: &[T], indices: &[I]) -> Buffer {
            let mut out = Vec::with_capacity(indices.len());
            crate::gather::take(&mut out, items, indices, |x| x);
            T::buffer(out)
        }
        match view {
            View::Booleans(v) => taken(v, indices), View::Characters(v) => taken(v, indices), View::Integers(v) => with_ints!(v, |v| taken(v, indices)),
            View::Floats(v) => with_floats!(v, |v| taken(v, indices)), View::Complexes(v) => with_complexes!(v, |v| taken(v, indices))
        }
    }
    /// The items of `view` at `dtype`. Matching dtypes copy. Otherwise each item converts as [`Source`] reads it, and conversions to a narrower dtype follow these rules. A float converts to an integer by truncating toward zero. A value that doesn't fit its target gives an Overflow error, as do NaN, infinities and anything but exactly 0 or 1 for Booleans. A complex number converts to a real dtype only when its imaginary part is zero, and otherwise gives a Domain error. Characters convert only to characters.
    pub fn cast(view: View<'_>, dtype: DType) -> Result<Self, Error> { Self::cast_mapped(view, &crate::agreement::Mapping::Linear(1), view.len(), dtype) }
    /// The items of `view` at `dtype`, where every item keeps its value. Items convert as [`Buffer::cast`] converts them, and a float that becomes a Boolean or an integer must be whole. A float rounds to a float of another width. A Domain error where an item doesn't keep its value.
    pub fn kept(view: View<'_>, dtype: DType) -> Result<Self, Error> {
        let whole = match view {
            View::Floats(v) => (0..v.len()).all(|i| v.get(i).fract() == 0.0), View::Complexes(v) => (0..v.len()).all(|i| v.get(i).re.fract() == 0.0), _ => true
        };
        let lost = || Error::new(Kind::Domain, format!("a {} item doesn't keep its value as {}", view.dtype().name(), dtype.name()));
        if dtype.is_integral() && !whole { return Err(lost()); }
        Self::cast(view, dtype).map_err(|_| lost())
    }
    /// Shared conversion dispatch for compact and mapped logical items. Unselected storage never participates in checked conversion.
    pub fn cast_mapped(view: View<'_>, mapping: &crate::agreement::Mapping, n: usize, dtype: DType) -> Result<Self, Error> {
        use crate::agreement::{Binding, Missing, Read, ReadPlan};
        let mut sources = vec![Binding::full(view)];
        let missing = Missing::prototype(view.dtype(), &mut sources);
        let plan = ReadPlan::copy(0, view.len(), mapping.clone(), n, missing);
        let mut output = Self::with_capacity(dtype, n);
        Read { sources: &sources, plan: &plan }.read_into(0..n, dtype, true, &mut output)?;
        Ok(output)
    }
}

impl Buffer {
    /// The items of `view` as integers of `dtype`, as BPL's `ₓ` marks and Python's `astype` convert them: a float takes its floor within `t` and saturates at the range, NaN giving 0 ([`crate::convert::floored`]); an integer wraps to the width; a Boolean gives 0 or 1. Other items and targets convert as [`Buffer::cast`] converts them.
    pub fn floored(view: View<'_>, dtype: DType, t: impl crate::tolerant::Tolerant) -> Result<Self, Error> {
        match (view, dtype) {
            (View::Floats(values), DType::Integer(width)) => {
                Ok(with_width!(width, I => I::buffer(with_floats!(values, |values| values.iter().map(|&x| crate::convert::floored::<I, _>(x, t)).collect()))))
            }
            (View::Integers(_) | View::Booleans(_), DType::Integer(_)) => Self::cast_results(view, dtype, false),
            _ => Self::cast(view, dtype),
        }
    }
    pub fn cast_results(view: View<'_>, dtype: DType, checked: bool) -> Result<Self, Error> {
        match (view, dtype) {
            (View::Floats(values), DType::Integer(width)) => {
                Ok(with_width!(width, I => I::buffer(with_floats!(values, |values| values.iter().map(|&x| crate::convert::saturated::<I, _>(x)).collect()))))
            }
            (View::Integers(values), DType::Integer(width)) if !checked => {
                Ok(with_width!(width, T => with_ints!(values, |values| T::buffer(values.iter().map(|&x| x as T).collect()))))
            }
            (View::Booleans(values), DType::Integer(width)) => Ok(with_width!(width, T => T::buffer(values.iter().map(|&x| x as T).collect()))),
            (_, DType::Character) if view.dtype().is_integral() => {
                let items = (0..view.len()).map(|i| {
                    let value = integer(view, i).ok().and_then(|x| u32::try_from(x).ok()).and_then(char::from_u32);
                    match value {
                        Some(value) => Ok(value), None if checked => Err(Error::new(Kind::Domain, "result is not a Unicode character")), None => Ok(char::FILL)
                    }
                }).collect::<Result<Vec<_>, _>>()?;
                Ok(Self::Characters(items))
            }
            _ => with_dtype!(dtype, T => {
                let mut items = Vec::new();
                T::extend_cast(view, view.len(), checked, &mut items)?;
                Ok(T::buffer(items))
            }),
        }
    }
}
