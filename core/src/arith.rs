//! Arithmetic as formula objects, which `rustymath::operation::Kernel` runs through their [`Formula`](crate::facts::Formula) rows. Addition, subtraction, multiplication and negation take every float and complex type and the integer types of compact buffers, `u8`, `i16`, `i32` and `i64`. Division, reciprocals and multiples of π take every float and complex type, and squares take floats. Reciprocals also take the integer types, where they floor, as integer division of 1 does. Bitwise and logical functions take them and Booleans. A float or complex result never fails. An integer sum, difference, product or negation that doesn't fit its type fails, and holds the result wrapped to the type, as NumPy gives it. Any other integer result that the type can't hold fails.
use crate::float::sealed::Sealed;

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

/// Bitwise AND, or logical AND of Booleans.
#[derive(Clone, Copy, Debug)]
pub struct BitAnd;
/// Bitwise OR, or logical OR of Booleans.
#[derive(Clone, Copy, Debug)]
pub struct BitOr;

/// Left shift, discarding high bits. A negative count shifts right, and a count at least the item width saturates to 0.
#[derive(Clone, Copy, Debug)]
pub struct LeftShift;
/// Right shift, retaining the sign bit for signed integers. A negative count shifts left, and a count at least the item width saturates to the sign.
#[derive(Clone, Copy, Debug)]
pub struct RightShift;

/// -x. A float or complex result never fails. An integer result fails where it doesn't fit its type, as the negation of a signed type's least value or of an unsigned byte other than 0 doesn't, and holds the result wrapped to the type.
#[derive(Clone, Copy, Debug)]
pub struct Negate;
/// 1 ÷ x. A float or complex result never fails. An integer's reciprocal floors, as integer [`FloorDivide`](crate::tolerant::FloorDivide) of 1 does: 1 gives 1, 2 or more give 0, -1 or less give -1, and 0 gives the type's greatest value.
#[derive(Clone, Copy, Debug)]
pub struct Reciprocal;
/// π × x, or π × (x ÷ y) of two arguments, at every float and complex type.
#[derive(Clone, Copy, Debug)]
pub struct PiTimes;

/// Applies the Boolean function whose truth table is `table`, indexed by `2*x+y`, to every bit of `x` and `y`, the leading sign bits included. It computes the table's algebraic normal form, `a0 ^ (a1 & x) ^ (a2 & y) ^ (a3 & x & y)`, with each coefficient spread to every bit, so every table runs the same code. A `match` on the table would put 16 paths into each loop that inlines it. It takes any integer type, exact ones included.
#[inline]
pub fn bitwise<T>(table: u8, x: T, y: T) -> T where
    T: Clone + std::ops::BitAnd<Output = T> + std::ops::BitOr<Output = T> + std::ops::BitXor<Output = T> + std::ops::Not<Output = T> {
    bitwise_with(table, x.clone(), y, x.clone() | !x, |a, b| a & b, |a, b| a ^ b)
}

pub(crate) fn bitwise_with<T: Clone>(table: u8, x: T, y: T, ones: T, and: impl Fn(T, T) -> T, xor: impl Fn(T, T) -> T) -> T {
    let zero = xor(ones.clone(), ones.clone());
    let spread = |bit: u8| if bit & 1 == 1 { ones.clone() } else { zero.clone() };
    let entry = |i: u8| table >> i;
    let (a0, a1, a2, a3) = (spread(entry(0)), spread(entry(0) ^ entry(2)), spread(entry(0) ^ entry(1)), spread(entry(0) ^ entry(1) ^ entry(2) ^ entry(3)));
    xor(xor(a0, and(x, xor(a1, and(y.clone(), a3)))), and(y, a2))
}

/// [`bitwise`] with the truth table it holds, as a function object. One argument is the second, with 0 as the first. Booleans and the signed types hold every result, because each leading sign bit takes the same entry of the table. An unsigned byte computes at `i64`, where its leading zeros can become ones, and fails where the result doesn't fit.
///
/// A float argument computes at `i64`, as the whole number it equals within the default tolerance (`operation::with_projected`). Any other float is DOMAIN. BPL's `ᵇ` and the Python wrapper's `bitwise_and`, `bitwise_or`, `bitwise_xor` and `bitwise_not` run its kernels.
#[derive(Clone, Copy, Debug)]
pub struct Bitwise(pub u8);

/// Logical NOT, NAND and NOR, of Booleans, and of integers that are 0 or 1, which give Booleans. Any other integer fails.
#[derive(Clone, Copy, Debug)]
pub struct Not;
/// See [`Not`].
#[derive(Clone, Copy, Debug)]
pub struct Nand;
/// See [`Not`].
#[derive(Clone, Copy, Debug)]
pub struct Nor;

/// An integer type's arithmetic. Each operation that can overflow gives its result wrapped to the type, and whether that is the exact result, so that a caller with exact integers can take another path. The tests have no branch, so a loop of them vectorizes. The one exception is a 64-bit product, because no vector instruction gives its 128-bit result. Integer kernels (`recipe::IntegerOps`) and integer folds run these.
pub trait Overflowing: crate::buffer::Cast + Ord + Into<i64> + Sealed {
    fn low(x: i64) -> Self;
    /// The type's greatest value for a positive `sign`, its least for a negative one, and 0 for 0: the floor of a quotient by 0, and of a negative power of 0 where `sign` is 1.
    fn limit(sign: i64) -> Self;
    /// x + y.
    fn plus(self, y: Self) -> (Self, bool);
    /// x - y.
    fn minus(self, y: Self) -> (Self, bool);
    /// x × y.
    fn times(self, y: Self) -> (Self, bool);
    /// |x|. A signed type's least value has no magnitude of its type.
    fn magnitude(self) -> (Self, bool);
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
            fn low(x: i64) -> Self { x as $t }
            #[inline(always)]
            fn limit(sign: i64) -> Self { match sign.signum() { 1 => <$t>::MAX, -1 => <$t>::MIN, _ => 0 } }
            #[inline(always)]
            fn plus(self, y: Self) -> (Self, bool) { let s = self.wrapping_add(y); (s, ((self ^ s) & (y ^ s)) >= 0) }
            #[inline(always)]
            fn minus(self, y: Self) -> (Self, bool) { let d = self.wrapping_sub(y); (d, ((self ^ y) & (self ^ d)) >= 0) }
            #[inline(always)]
            fn times(self, $y: Self) -> (Self, bool) { let $x = self; $times }
            #[inline(always)]
            fn magnitude(self) -> (Self, bool) { (self.wrapping_abs(), self != <$t>::MIN) }
        }
    )+};
}
signed!(
    i16: |x, y| wide_product!(i16, i32, x, y), i32: |x, y| wide_product!(i32, i64, x, y), i64: |x, y| { let (p, over) = x.overflowing_mul(y); (p, !over) }
);

/// The unsigned types' arithmetic. A sum overflows where it wraps below an argument, and a difference where it would be negative.
macro_rules! unsigned {
    ($($t:ty: |$x:ident, $y:ident| $times:expr),+) => {$(
        impl Overflowing for $t {
            fn low(x: i64) -> Self { x as $t }
            #[inline(always)]
            fn limit(sign: i64) -> Self { match sign.signum() { 1 => <$t>::MAX, -1 => <$t>::MIN, _ => 0 } }
            #[inline(always)]
            fn plus(self, y: Self) -> (Self, bool) { let s = self.wrapping_add(y); (s, s >= self) }
            #[inline(always)]
            fn minus(self, y: Self) -> (Self, bool) { (self.wrapping_sub(y), self >= y) }
            #[inline(always)]
            fn times(self, $y: Self) -> (Self, bool) { let $x = self; $times }
            #[inline(always)]
            fn magnitude(self) -> (Self, bool) { (self, true) }
        }
    )+};
}
unsigned!(u8: |x, y| wide_product!(u8, u16, x, y));
