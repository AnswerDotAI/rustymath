//! Addition, subtraction and multiplication as function objects, at every float, complex and integer type, and the integer arithmetic they run. No result fails. An integer result wraps to its type, as Rust's wrapping arithmetic and NumPy's integer arrays do.
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

forms!(Add at real: |_, [x, y]| Total(x + y));
forms!(Subtract at real: |_, [x, y]| Total(x - y));
forms!(Multiply at real: |_, [x, y]| Total(x * y));
forms!(Add at Complex<f32>, Complex<f64>: |_, [x, y]| Total(x + y));
forms!(Subtract at Complex<f32>, Complex<f64>: |_, [x, y]| Total(x - y));
forms!(Multiply at Complex<f32>, Complex<f64>: |_, [x, y]| Total(x * y));

/// An integer type's arithmetic. Each operation that can overflow gives its result wrapped to the type, and whether that is the exact result, so that a caller with exact integers can take another path. The tests have no branch, so a loop of them vectorizes. The one exception is a 64-bit product, because no vector instruction gives its 128-bit result. [`Add`], [`Subtract`], [`Multiply`], [`Abs`](crate::Abs) and [`Sign`](crate::Sign) run these on slices.
pub trait Overflowing: Copy + Sealed {
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
    i8: |x, y| wide_product!(i8, i16, x, y), i16: |x, y| wide_product!(i16, i32, x, y), i32: |x, y| wide_product!(i32, i64, x, y),
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
unsigned!(
    u8: |x, y| wide_product!(u8, u16, x, y), u16: |x, y| wide_product!(u16, u32, x, y), u32: |x, y| wide_product!(u32, u64, x, y),
    u64: |x, y| { let (p, over) = x.overflowing_mul(y); (p, !over) }
);

/// Each integer type's forms of the arithmetic objects, which keep [`Overflowing`]'s wrapped results and never fail.
macro_rules! integer_forms {
    ($($t:ty),+) => {$(
        forms!(Add at $t: |_, [x, y]| Total(x.plus(y).0));
        forms!(Subtract at $t: |_, [x, y]| Total(x.minus(y).0));
        forms!(Multiply at $t: |_, [x, y]| Total(x.times(y).0));
        forms!(Abs at $t: |_, [x]| Total(x.magnitude().0));
        forms!(Sign at $t: |_, [x]| Total(x.sign()));
    )+};
}
integer_forms!(i8, i16, i32, i64, u8, u16, u32, u64);
