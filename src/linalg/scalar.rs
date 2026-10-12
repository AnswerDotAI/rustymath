//! The item types of the matrices that `factor` decomposes: `f32`, `f64`, and complex numbers with parts of either type. Each
//! decomposition is written once for all of them. Where a real algorithm transposes, a complex one also conjugates. Real quantities,
//! such as norms, the sizes of pivots and singular values, have the parts' type.
use crate::{float::{sealed::Sealed, FUSED}, Native};
use num_complex::Complex;
use num_traits::{One, Zero};
use std::ops::{Add, Mul, Neg, Sub};

/// A real or complex number that the decompositions take: `f32`, `f64`, or a complex number with parts of either type. rustymath
/// implements it for those four types, and no other crate can. For a real type each method is the plain operation, and the generic
/// algorithms give the same real results as algorithms written for reals alone. The trait has no `/`. Quotients go through `quotient`
/// and `reciprocal`, which keep a complex quotient from overflowing where its result fits.
pub trait Scalar:
    Sealed + Copy + PartialEq + Zero + One + Add<Output = Self> + Sub<Output = Self> + Mul<Output = Self> + Neg<Output = Self>
{
    /// The type of the real and imaginary parts.
    type Real: Native;
    fn conj(self) -> Self;
    fn re(self) -> Self::Real;
    fn im(self) -> Self::Real;
    fn from_real(x: Self::Real) -> Self;
    /// `|re| + |im|`, within a factor of √2 of the modulus and cheaper. Pivot searches and scaling compare it.
    fn abs1(self) -> Self::Real;
    /// `|self|`, computed without overflow where the result fits, as `hypot` computes it.
    fn modulus(self) -> Self::Real;
    fn mul_real(self, x: Self::Real) -> Self;
    fn div_real(self, x: Self::Real) -> Self;
    /// `self × b + c`, each multiply-add rounding once where `FUSED` is true, as `Core::mla` does.
    fn mul_acc(self, b: Self, c: Self) -> Self;
    fn reciprocal(self) -> Self;
    /// `self / d`. A complex quotient follows Smith's algorithm, which overflows or underflows only where the quotient itself does.
    fn quotient(self, d: Self) -> Self;
}

impl<T: Native> Scalar for T {
    type Real = T;
    #[inline(always)]
    fn conj(self) -> T { self }
    #[inline(always)]
    fn re(self) -> T { self }
    #[inline(always)]
    fn im(self) -> T { T::zero() }
    #[inline(always)]
    fn from_real(x: T) -> T { x }
    #[inline(always)]
    fn abs1(self) -> T { self.abs() }
    #[inline(always)]
    fn modulus(self) -> T { self.abs() }
    #[inline(always)]
    fn mul_real(self, x: T) -> T { self * x }
    #[inline(always)]
    fn div_real(self, x: T) -> T { self / x }
    #[inline(always)]
    fn mul_acc(self, b: T, c: T) -> T { self.mla::<FUSED>(b, c) }
    #[inline(always)]
    fn reciprocal(self) -> T { self.recip() }
    #[inline(always)]
    fn quotient(self, d: T) -> T { self / d }
}

impl<T: Native> Scalar for Complex<T> {
    type Real = T;
    #[inline(always)]
    fn conj(self) -> Self { Complex::new(self.re, -self.im) }
    #[inline(always)]
    fn re(self) -> T { self.re }
    #[inline(always)]
    fn im(self) -> T { self.im }
    #[inline(always)]
    fn from_real(x: T) -> Self { Complex::new(x, T::zero()) }
    #[inline(always)]
    fn abs1(self) -> T { self.re.abs() + self.im.abs() }
    #[inline(always)]
    fn modulus(self) -> T { self.norm() }
    #[inline(always)]
    fn mul_real(self, x: T) -> Self { Complex::new(self.re * x, self.im * x) }
    #[inline(always)]
    fn div_real(self, x: T) -> Self { Complex::new(self.re / x, self.im / x) }
    #[inline(always)]
    fn mul_acc(self, b: Self, c: Self) -> Self {
        let re = self.re.mla::<FUSED>(b.re, (-self.im).mla::<FUSED>(b.im, c.re));
        Complex::new(re, self.re.mla::<FUSED>(b.im, self.im.mla::<FUSED>(b.re, c.im)))
    }
    #[inline(always)]
    fn reciprocal(self) -> Self { Self::one().quotient(self) }
    #[inline(always)]
    fn quotient(self, d: Self) -> Self {
        if d.re.abs() >= d.im.abs() {
            let r = d.im / d.re;
            let s = d.re + d.im * r;
            Complex::new((self.re + self.im * r) / s, (self.im - self.re * r) / s)
        } else {
            let r = d.re / d.im;
            let s = d.re * r + d.im;
            Complex::new((self.re * r + self.im) / s, (self.im * r - self.re) / s)
        }
    }
}
