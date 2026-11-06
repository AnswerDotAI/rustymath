//! Integer instantiation of the shared arithmetic formulas.
use super::{Binary, Cmp, Integer, Ops, Overflowing, Unary};
use crate::buffer::FloatWidth;

pub struct IntegerOps<T> { pub variant: crate::domain::Variant, item: std::marker::PhantomData<T> }
impl<T> IntegerOps<T> { #[inline(always)] pub fn new(variant: crate::domain::Variant) -> Self { Self { variant, item: std::marker::PhantomData } } }
impl<T: Overflowing> IntegerOps<T> {
    #[inline(always)]
    pub fn result(&self, value: i64) -> T {
        if self.variant.fails_on_overflow() && matches!(self.variant.dtype, crate::buffer::DType::Integer(w) if !w.holds(value)) {
            (crate::error::Kind::Overflow.reporter())();
        }
        T::low(value)
    }
}
impl<T: Overflowing> Ops<f64> for IntegerOps<T> {
    type V = T;
    type B = bool;
    type I = u64;
    type W = f64;
    #[inline(always)]
    fn width(&self) -> FloatWidth { FloatWidth::F64 }
    #[inline(always)]
    fn fused(&self) -> bool { false }
    #[inline(always)]
    fn integral(&self) -> bool { true }
    #[inline(always)]
    fn widen(&self, x: T) -> f64 { Into::<i64>::into(x) as f64 }
    #[inline(always)]
    fn wide_number(&self, x: f64) -> f64 { x }
    #[inline(always)]
    fn wide_multiply_add(&self, x: f64, y: f64, z: f64) -> f64 { x * y + z }
    #[inline(always)]
    fn narrow(&self, x: f64) -> T { T::low(x as i64) }
    u64_integers!(8 * std::mem::size_of::<T>() as u32);
    #[inline(always)]
    fn to_integer(&self, x: T) -> u64 { Into::<i64>::into(x) as u64 }
    #[inline(always)]
    fn from_integer(&self, x: u64) -> T { self.from_bits(x) }
    #[inline(always)]
    fn to_bits(&self, x: T) -> u64 { Into::<i64>::into(x) as u64 }
    #[inline(always)]
    fn from_bits(&self, x: u64) -> T { T::low(if self.variant.dtype == crate::buffer::DType::Boolean { (x & 1) as i64 } else { x as i64 }) }
    #[inline(always)]
    fn number(&self, x: f64) -> T { T::low(x as i64) }
    #[inline(always)]
    fn unary(&self, op: Unary, x: T) -> T {
        match op {
            Unary::Negate => T::low(0).minus(x).0, Unary::Magnitude => x.magnitude().0, Unary::Floor => x,
            Unary::Sqrt => { let x = Into::<i64>::into(x); T::low(if x >= 0 { x.isqrt() } else { 0 }) } Unary::Not => T::low(1).minus(x).0,
            _ => unreachable!("non-integer instruction in an integer formula")
        }
    }
    #[inline(always)]
    fn binary(&self, op: Binary, x: T, y: T) -> T {
        match op {
            Binary::Add => x.plus(y).0, Binary::Subtract => x.minus(y).0, Binary::Multiply => x.times(y).0,
            // A zero divisor gives the limit of the dividend's sign, or 0 for 0 ÷ 0. Under dynamic kinds that's i64's limit, which a narrower width reports as an overflow (`result`), so the call widens to i64.
            Binary::Divide => {
                let (x, y): (i64, i64) = (x.into(), y.into());
                if y != 0 { T::low(x.wrapping_div(y)) }
                else if self.variant.fails_on_overflow() { self.result(if x == 0 { 0 } else if x > 0 { i64::MAX } else { i64::MIN }) }
                else { T::limit(x) }
            }
            Binary::Larger => x.max(y), Binary::Smaller => x.min(y), Binary::BitAnd => T::low(Into::<i64>::into(x) & Into::<i64>::into(y)),
            _ => unreachable!("non-integer instruction in an integer formula")
        }
    }
    #[inline(always)]
    fn overflow(&self, op: Binary, x: T, y: T) -> bool {
        !match op {
            Binary::Add => x.plus(y).1, Binary::Subtract => x.minus(y).1, Binary::Multiply => x.times(y).1,
            // A quotient overflows only as the least value divided by -1 does. A zero divisor's limit reports its own overflow (`binary`).
            Binary::Divide => { let (x, y): (i64, i64) = (x.into(), y.into()); x.checked_div(y).map_or(y == 0, |q| Into::<i64>::into(T::low(q)) == q) }
            _ => true
        }
    }
    plain_logic!(T);
    #[inline(always)]
    fn multiply_add(&self, x: T, y: T, z: T) -> T { self.binary(Binary::Add, self.binary(Binary::Multiply, x, y), z) }
}
