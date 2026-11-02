//! Integer instantiation of the shared arithmetic formulas.
use super::{Ops, Unary, Binary, Cmp, Integer, Overflowing};
use crate::buffer::FloatWidth;

pub struct IntegerOps<T> { pub variant: crate::facts::Variant, item: std::marker::PhantomData<T> }
impl<T> IntegerOps<T> {
    pub fn new(variant: crate::facts::Variant) -> Self { Self { variant, item: std::marker::PhantomData } }
}
impl<T: Overflowing> IntegerOps<T> {
    pub fn result(&self, value: i64) -> T {
        if self.variant.checked && !self.variant.wrapping && matches!(self.variant.dtype, crate::buffer::DType::Integer(w) if !w.holds(value)) {
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
    fn width(&self) -> FloatWidth { FloatWidth::F64 }
    fn fused(&self) -> bool { false }
    fn integral(&self) -> bool { true }
    fn widen(&self, x: T) -> f64 { Into::<i64>::into(x) as f64 }
    fn wide_number(&self, x: f64) -> f64 { x }
    fn wide_multiply_add(&self, x: f64, y: f64, z: f64) -> f64 { x * y + z }
    fn narrow(&self, x: f64) -> T { T::low(x as i64) }
    fn integer(&self, x: i64) -> u64 { x as u64 }
    fn integer_op(&self, op: Integer, x: u64, y: u64) -> u64 {
        match op {
            Integer::Add => x.wrapping_add(y), Integer::Subtract => x.wrapping_sub(y), Integer::And => x & y, Integer::Or => x | y, Integer::Xor => x ^ y,
            Integer::ShiftLeft => x.wrapping_shl(y as u32), Integer::ShiftRight => (x & (u64::MAX >> (64 - 8 * self.variant.dtype.bytes()))).wrapping_shr(y as u32), Integer::SignedShiftRight => (x as i64).wrapping_shr(y as u32) as u64,
        }
    }
    fn integer_compare(&self, op: Cmp, x: u64, y: u64) -> bool {
        match op { Cmp::Less => (x as i64) < y as i64, Cmp::LessEqual => (x as i64) <= y as i64, Cmp::Equal => x == y }
    }
    fn integer_select(&self, test: bool, x: u64, y: u64) -> u64 { if test { x } else { y } }
    fn to_integer(&self, x: T) -> u64 { Into::<i64>::into(x) as u64 }
    fn from_integer(&self, x: u64) -> T { self.from_bits(x) }
    fn to_bits(&self, x: T) -> u64 { Into::<i64>::into(x) as u64 }
    fn from_bits(&self, x: u64) -> T { T::low(if self.variant.dtype == crate::buffer::DType::Boolean { (x & 1) as i64 } else { x as i64 }) }
    fn number(&self, x: f64) -> T { T::low(x as i64) }
    fn unary(&self, op: Unary, x: T) -> T {
        match op {
            Unary::Negate => T::low(0).minus(x).0, Unary::Magnitude => x.magnitude().0, Unary::Floor => x,
            Unary::Sqrt => { let x = Into::<i64>::into(x); T::low(if x >= 0 { x.isqrt() } else { 0 }) }, Unary::Not => T::low(1).minus(x).0,
            _ => unreachable!("non-integer instruction in an integer formula"),
        }
    }
    fn binary(&self, op: Binary, x: T, y: T) -> T {
        match op {
            Binary::Add => x.plus(y).0, Binary::Subtract => x.minus(y).0, Binary::Multiply => x.times(y).0,
            Binary::Divide => T::low(if Into::<i64>::into(y) == 0 { 0 } else { Into::<i64>::into(x).wrapping_div(Into::<i64>::into(y)) }), Binary::Larger => x.max(y), Binary::Smaller => x.min(y),
            Binary::BitAnd => T::low(Into::<i64>::into(x) & Into::<i64>::into(y)), _ => unreachable!("non-integer instruction in an integer formula"),
        }
    }
    fn overflow(&self, op: Binary, x: T, y: T) -> bool {
        !match op { Binary::Add => x.plus(y).1, Binary::Subtract => x.minus(y).1, Binary::Multiply => x.times(y).1, _ => true }
    }
    fn compare(&self, c: Cmp, x: T, y: T) -> bool { match c { Cmp::Less => x < y, Cmp::LessEqual => x <= y, Cmp::Equal => x == y } }
    fn multiply_add(&self, x: T, y: T, z: T) -> T { self.binary(Binary::Add, self.binary(Binary::Multiply, x, y), z) }
    fn and(&self, x: bool, y: bool) -> bool { x & y }
    fn or(&self, x: bool, y: bool) -> bool { x | y }
    fn not(&self, x: bool) -> bool { !x }
    fn select(&self, test: bool, x: T, y: T) -> T { if test { x } else { y } }
}
