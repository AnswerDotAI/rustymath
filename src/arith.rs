//! Arithmetic as function objects. Addition, subtraction, multiplication and negation take every float and complex type and the integer types of compact buffers, `u8`, `i16`, `i32` and `i64`. Division, reciprocals and multiples of π take every float and complex type, and squares take floats. Exact division and reciprocals take the integer types, and bitwise and logical functions take them and Booleans. A float or complex result never fails. An integer sum, difference, product or negation that doesn't fit its type fails, and holds the result wrapped to the type, as NumPy gives it. Any other integer result that the type can't hold fails.
use crate::{Abs, Sign, float::sealed::Sealed, slice::Total};
use num_complex::Complex;

/// x + y.
#[derive(Clone, Copy, Debug)]
pub struct Add;
/// x - y.
#[derive(Clone, Copy, Debug)]
pub struct Subtract;
/// x × y.
#[derive(Clone, Copy, Debug)]
pub struct Multiply;
/// x ÷ y, at every float and complex type. Complex numbers divide by [`divide`](crate::divide).
#[derive(Clone, Copy, Debug)]
pub struct Divide;

/// Division with zero divided by zero defined as zero.
#[derive(Clone, Copy, Debug)]
pub struct MathishDivide;

#[inline]
pub(crate) fn both_zero<T: num_traits::Zero>(x: T, y: T) -> bool { x.is_zero() & y.is_zero() }

macro_rules! bitwise {
    ($($(#[doc = $doc:literal])* $name:ident [$($x:ident),+] $body:expr;)+) => {$(
        $(#[doc = $doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $name;
        forms!($name at bool, u8, i16, i32, i64: |_, [$($x),+]| $body);
    )+};
}
bitwise! {
    /// Bitwise AND, or logical AND of Booleans.
    BitAnd [x, y] x & y;
    /// Bitwise OR, or logical OR of Booleans.
    BitOr [x, y] x | y;
    /// Bitwise XOR, or logical XOR of Booleans.
    BitXor [x, y] x ^ y;
    /// Bitwise complement, or logical negation of a Boolean.
    BitNot [x] !x;
}

/// Left shift, discarding high bits. A negative count or one at least the item width fails.
#[derive(Clone, Copy, Debug)]
pub struct LeftShift;
/// Right shift, retaining the sign bit for signed integers. A negative count or one at least the item width fails.
#[derive(Clone, Copy, Debug)]
pub struct RightShift;
forms!(LeftShift at u8, i16, i32, i64: |_, [x, y]| u32::try_from(y).ok().and_then(|y| x.checked_shl(y)));
forms!(RightShift at u8, i16, i32, i64: |_, [x, y]| u32::try_from(y).ok().and_then(|y| x.checked_shr(y)));

// The float and complex forms are `#[inline]`, as the integer forms are, so that a caller's own loop over items, such as a fold's, inlines them.
forms!(#[inline] Add at real: |_, [x, y]| Total(x + y));
forms!(#[inline] Subtract at real: |_, [x, y]| Total(x - y));
forms!(#[inline] Multiply at real: |_, [x, y]| Total(x * y));
forms!(#[inline] Add at Complex<f32>, Complex<f64>: |_, [x, y]| Total(x + y));
forms!(#[inline] Subtract at Complex<f32>, Complex<f64>: |_, [x, y]| Total(x - y));
forms!(#[inline] Multiply at Complex<f32>, Complex<f64>: |_, [x, y]| Total(x * y));
forms!(#[inline] Divide at real: |_, [x, y]| Total(x / y));
forms!(#[inline] Divide at Complex<f32>, Complex<f64>: |_, [x, y]| Total(crate::divide(x, y)));
forms!(#[inline] MathishDivide at real: |_, [x, y]| Total(if both_zero(x, y) { x } else { x / y }));
forms!(#[inline] MathishDivide at Complex<f32>, Complex<f64>: |_, [x, y]| Total(if both_zero(x, y) { x } else { crate::divide(x, y) }));

/// -x. A float or complex result never fails. An integer result fails where it doesn't fit its type, as the negation of a signed type's least value or of an unsigned byte other than 0 doesn't, and holds the result wrapped to the type.
#[derive(Clone, Copy, Debug)]
pub struct Negate;
/// 1 ÷ x. A float or complex result never fails. An integer's reciprocal is an integer only for 1 and -1, and fails for every other integer.
#[derive(Clone, Copy, Debug)]
pub struct Reciprocal;
/// π × x, or π × (x ÷ y) of two arguments, at every float and complex type.
#[derive(Clone, Copy, Debug)]
pub struct PiTimes;
/// x ÷ y, of integers whose quotient is an integer. Any other quotient, or a zero or overflowing divisor, fails.
#[derive(Clone, Copy, Debug)]
pub struct WholeDivide;
forms!(#[inline] Negate at real: |_, [x]| Total(-x));
forms!(#[inline] Negate at Complex<f32>, Complex<f64>: |_, [x]| Total(-x));
forms!(#[inline] Negate at u8, i16, i32, i64: |_, [x]| { let (r, overflowed) = x.overflowing_neg(); (r, !overflowed) });
forms!(#[inline] Reciprocal at real: |_, [x]| Total(1.0 / x));
forms!(#[inline] Reciprocal at Complex<f32>, Complex<f64>: |_, [x]| Total(crate::divide(Complex::new(1.0, 0.0), x)));
forms!(#[inline] Reciprocal at u8, i16, i32, i64: |_, [x]| (x.checked_mul(x) == Some(1)).then_some(x));
forms!(#[inline] PiTimes at real: |_, [x]| Total(pi_times(x)));
forms!(#[inline] PiTimes at real: |_, [x, y]| Total(pi_times(x / y)));
forms!(#[inline] PiTimes at Complex<f32>: |_, [x]| Total(x * std::f32::consts::PI));
forms!(#[inline] PiTimes at Complex<f64>: |_, [x]| Total(x * std::f64::consts::PI));
forms!(#[inline] PiTimes at Complex<f32>: |_, [x, y]| Total(crate::divide(x, y) * std::f32::consts::PI));
forms!(#[inline] PiTimes at Complex<f64>: |_, [x, y]| Total(crate::divide(x, y) * std::f64::consts::PI));
forms!(#[inline] WholeDivide at u8, i16, i32, i64: |_, [x, y]| if x.checked_rem(y) == Some(0) { x.checked_div(y) } else { None });

/// π × y, at the width of `y`.
#[inline]
fn pi_times<F: crate::Float>(y: F) -> F { y * F::narrow(std::f64::consts::PI) }

/// Applies the Boolean function whose truth table is `table`, indexed by `2*x+y`, to every bit of `x` and `y`, the leading sign bits included. It computes the table's algebraic normal form, `a0 ^ (a1 & x) ^ (a2 & y) ^ (a3 & x & y)`, with each coefficient spread to every bit, so every table runs the same code. A `match` on the table would put 16 paths into each loop that inlines it. It takes any integer type, exact ones included.
#[inline]
pub fn bitwise<T>(table: u8, x: T, y: T) -> T where T: Clone + std::ops::BitAnd<Output = T> + std::ops::BitOr<Output = T> + std::ops::BitXor<Output = T> + std::ops::Not<Output = T> {
    let ones = x.clone() | !x.clone();
    let zero = ones.clone() ^ ones.clone();
    let spread = |bit: u8| if bit & 1 == 1 { ones.clone() } else { zero.clone() };
    let entry = |i: u8| table >> i;
    // `x`'s coefficient is the parity of entries 0 and 2, `y`'s of entries 0 and 1, and `x & y`'s of all four.
    let (a0, a1, a2, a3) = (spread(entry(0)), spread(entry(0) ^ entry(2)), spread(entry(0) ^ entry(1)), spread(entry(0) ^ entry(1) ^ entry(2) ^ entry(3)));
    a0 ^ (x & (a1 ^ (y.clone() & a3))) ^ (y & a2)
}

/// [`bitwise`] with the truth table it holds, as a function object. One argument is the second, with 0 as the first. Booleans and the signed types hold every result, because each leading sign bit takes the same entry of the table. An unsigned byte computes at `i64`, where its leading zeros can become ones, and fails where the result doesn't fit.
#[derive(Clone, Copy, Debug)]
pub struct Bitwise(pub u8);
forms!(#[inline] Bitwise at bool, i16, i32, i64: |k, [x, y]| bitwise(k.0, x, y));
forms!(#[inline] Bitwise at bool: |k, [y]| bitwise(k.0, false, y));
forms!(#[inline] Bitwise at i16, i32, i64: |k, [y]| bitwise(k.0, 0, y));
forms!(#[inline] Bitwise at u8: |k, [x, y]| u8::try_from(bitwise(k.0, i64::from(x), i64::from(y))).ok());
forms!(#[inline] Bitwise at u8: |k, [y]| u8::try_from(bitwise(k.0, 0, i64::from(y))).ok());

/// Logical NOT, NAND and NOR, of Booleans, and of integers that are 0 or 1, which give Booleans. Any other integer fails.
#[derive(Clone, Copy, Debug)]
pub struct Not;
/// See [`Not`].
#[derive(Clone, Copy, Debug)]
pub struct Nand;
/// See [`Not`].
#[derive(Clone, Copy, Debug)]
pub struct Nor;
/// An integer that is 0 or 1 as a Boolean.
#[inline]
fn truth<T: Into<i64>>(x: T) -> Option<bool> { match x.into() { 0 => Some(false), 1 => Some(true), _ => None } }
forms!(#[inline] Not at bool: |_, [x]| !x);
forms!(#[inline] Nand at bool: |_, [x, y]| !(x & y));
forms!(#[inline] Nor at bool: |_, [x, y]| !(x | y));
forms!(#[inline] Not at u8 => bool, i16 => bool, i32 => bool, i64 => bool: |_, [x]| truth(x).map(|x| !x));
forms!(#[inline] Nand at u8 => bool, i16 => bool, i32 => bool, i64 => bool: |_, [x, y]| truth(x).zip(truth(y)).map(|(x, y)| !(x & y)));
forms!(#[inline] Nor at u8 => bool, i16 => bool, i32 => bool, i64 => bool: |_, [x, y]| truth(x).zip(truth(y)).map(|(x, y)| !(x | y)));

/// An integer type's arithmetic. Each operation that can overflow gives its result wrapped to the type, and whether that is the exact result, so that a caller with exact integers can take another path. The tests have no branch, so a loop of them vectorizes. The one exception is a 64-bit product, because no vector instruction gives its 128-bit result. [`Add`], [`Subtract`], [`Multiply`], [`Abs`](crate::Abs) and [`Sign`](crate::Sign) run these on slices.
pub(crate) trait Overflowing: Copy + Sealed {
    /// x + y.
    fn plus(self, y: Self) -> (Self, bool);
    /// x - y.
    fn minus(self, y: Self) -> (Self, bool);
    /// x × y.
    fn times(self, y: Self) -> (Self, bool);
    /// |x|. A signed type's least value has no magnitude of its type.
    fn magnitude(self) -> (Self, bool);
    /// -1, 0 or 1.
    fn sign(self) -> Self;
}

/// The product of `$x` and `$y` of type `$t`, computed exactly at `$wide`, the next width up, and wrapped to `$t`.
macro_rules! wide_product {
    ($t:ty, $wide:ty, $x:ident, $y:ident) => {{
        let p = <$wide>::from($x) * <$wide>::from($y);
        (p as $t, (p >= <$t>::MIN as $wide) & (p <= <$t>::MAX as $wide))
    }};
}

/// The signed types' arithmetic. A sum overflows where its sign differs from both arguments'. A difference overflows where the arguments' signs differ and its sign differs from the first argument's.
macro_rules! signed {
    ($($t:ty: |$x:ident, $y:ident| $times:expr),+) => {$(
        impl Overflowing for $t {
            #[inline(always)]
            fn plus(self, y: Self) -> (Self, bool) { let s = self.wrapping_add(y); (s, ((self ^ s) & (y ^ s)) >= 0) }
            #[inline(always)]
            fn minus(self, y: Self) -> (Self, bool) { let d = self.wrapping_sub(y); (d, ((self ^ y) & (self ^ d)) >= 0) }
            #[inline(always)]
            fn times(self, $y: Self) -> (Self, bool) { let $x = self; $times }
            #[inline(always)]
            fn magnitude(self) -> (Self, bool) { (self.wrapping_abs(), self != <$t>::MIN) }
            #[inline(always)]
            fn sign(self) -> Self { self.signum() }
        }
    )+};
}
signed!(
    i16: |x, y| wide_product!(i16, i32, x, y), i32: |x, y| wide_product!(i32, i64, x, y),
    i64: |x, y| { let (p, over) = x.overflowing_mul(y); (p, !over) }
);

/// The unsigned types' arithmetic. A sum overflows where it wraps below an argument, and a difference where it would be negative.
macro_rules! unsigned {
    ($($t:ty: |$x:ident, $y:ident| $times:expr),+) => {$(
        impl Overflowing for $t {
            #[inline(always)]
            fn plus(self, y: Self) -> (Self, bool) { let s = self.wrapping_add(y); (s, s >= self) }
            #[inline(always)]
            fn minus(self, y: Self) -> (Self, bool) { (self.wrapping_sub(y), self >= y) }
            #[inline(always)]
            fn times(self, $y: Self) -> (Self, bool) { let $x = self; $times }
            #[inline(always)]
            fn magnitude(self) -> (Self, bool) { (self, true) }
            #[inline(always)]
            fn sign(self) -> Self { self.min(1) }
        }
    )+};
}
unsigned!(u8: |x, y| wide_product!(u8, u16, x, y));

/// Each integer type's forms of the arithmetic objects. A result that doesn't fit the type fails, as [`Overflowing`] reports it, and its slot holds the result wrapped to the type. The forms are `#[inline]`, so that a caller's loop over items inlines them.
macro_rules! integer_forms {
    ($($t:ty),+) => {$(
        forms!(#[inline] Add at $t: |_, [x, y]| x.plus(y));
        forms!(#[inline] Subtract at $t: |_, [x, y]| x.minus(y));
        forms!(#[inline] Multiply at $t: |_, [x, y]| x.times(y));
        forms!(#[inline] Abs at $t: |_, [x]| x.magnitude());
        forms!(#[inline] Sign at $t: |_, [x]| Total(x.sign()));
    )+};
}
integer_forms!(u8, i16, i32, i64);
