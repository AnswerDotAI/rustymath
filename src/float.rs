//! The float types that rustymath's functions take, and the operations that its algorithms build on.

/// Whether `mla` fuses its multiply and add into one operation with one rounding. It does where the target has a fused multiply-add
/// instruction, as on AArch64 and on x86-64 with FMA. Elsewhere, as in wasm and on baseline x86-64, `mul_add` calls a software `fma`,
/// which is slow and stops the loop around it vectorizing. There `mla` multiplies and adds with two roundings.
pub const FUSED: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// A float type that rustymath's functions take: `half::f16`, `f32` or `f64`.
pub trait Float: num_traits::Float + sealed::Sealed {
    /// The type this float computes in: `f32` for `f16`, and the type itself otherwise. A function of an `f16` computes in `f32` and
    /// rounds once. Sums keep their running totals in this type.
    type Total: Native;
    fn total(self) -> Self::Total;
    /// `t` rounded to this type.
    fn from_total(t: Self::Total) -> Self;
    /// `x` rounded to the nearest float of this type.
    fn narrow(x: f64) -> Self;
    /// The least float of this type above `self`, as `f64::next_up` gives it.
    fn next_up(self) -> Self;
    /// The greatest float of this type below `self`, as `f64::next_down` gives it.
    fn next_down(self) -> Self;
}

/// A float type that rustymath's algorithms run in: `f32` or `f64`.
pub trait Native: Float<Total = Self> + sealed::Core {
    /// `+` that may be reordered with other algebraic operations. A sum of them vectorizes.
    fn algebraic_add(self, other: Self) -> Self;
    /// `×` that may be reordered with other algebraic operations.
    fn algebraic_mul(self, other: Self) -> Self;
}

/// Traits that only rustymath implements. `Core` holds the bit-level operations that the algorithms build on.
pub(crate) mod sealed {
    pub trait Sealed {}
    pub trait Core: Copy {
        /// Whether this is `f64`. Algorithms pick their constants by it, and the compiler removes the branch.
        const WIDE: bool;
        /// 1.5 × 2^m, where m is the number of mantissa bits. Adding it to a float of magnitude below 2^(m-1) rounds that float to a
        /// whole number, which the low bits of the sum hold.
        const SHIFT: Self;
        /// `self×b+c`, with one rounding when `FUSED` is true and two otherwise.
        fn mla<const FUSED: bool>(self, b: Self, c: Self) -> Self;
        /// 2^n, from `z` = `SHIFT + n`, for a whole n within the exponents of normal floats.
        fn pow2(z: Self) -> Self;
        /// `(m, e)` with `self` = m × 2^e, m from 0.75 to 1.5 and e whole, for a positive normal `self`.
        fn split(self) -> (Self, Self);
        /// This float as an `f64`, which holds it exactly.
        fn wide(self) -> f64;
        /// `self` negated when the whole number n is odd, for `z` = `SHIFT + n`.
        fn flip(self, z: Self) -> Self;
    }
}

macro_rules! native {
    ($t:ident, $bits:ty, $mantissa:literal, $bias:literal) => {
        impl sealed::Sealed for $t {}
        impl Float for $t {
            type Total = $t;
            #[inline]
            fn total(self) -> $t { self }
            #[inline]
            fn from_total(t: $t) -> $t { t }
            #[inline]
            fn narrow(x: f64) -> $t { x as $t }
            #[inline]
            fn next_up(self) -> $t { $t::next_up(self) }
            #[inline]
            fn next_down(self) -> $t { $t::next_down(self) }
        }
        impl Native for $t {
            #[inline]
            fn algebraic_add(self, other: $t) -> $t { $t::algebraic_add(self, other) }
            #[inline]
            fn algebraic_mul(self, other: $t) -> $t { $t::algebraic_mul(self, other) }
        }
        impl sealed::Core for $t {
            const WIDE: bool = $mantissa == 52;
            const SHIFT: $t = 1.5 * (1u64 << $mantissa) as $t;
            #[inline]
            fn mla<const FUSED: bool>(self, b: $t, c: $t) -> $t { if FUSED { self.mul_add(b, c) } else { self * b + c } }
            // The low bits of `z` hold n. Adding the bias to them and shifting them into the exponent field gives 2^n. Integer adds
            // and shifts vectorize on every target, where a conversion from a float to a 64-bit integer doesn't on x86-64.
            #[inline]
            fn pow2(z: $t) -> $t { $t::from_bits(z.to_bits().wrapping_add($bias) << $mantissa) }
            // e comes from the exponent field of `self÷0.75`. It converts to a float through the bits of 2^m + e, which avoids a
            // conversion from an integer that wouldn't vectorize.
            #[inline]
            fn split(self) -> ($t, $t) {
                let exponent = ((self * (1.0 / 0.75)).to_bits() >> $mantissa) & (2 * $bias + 1);
                let m = $t::from_bits(self.to_bits().wrapping_sub(exponent.wrapping_sub($bias) << $mantissa));
                let two_m = (1 as $bits << $mantissa) as $t;
                (m, $t::from_bits(two_m.to_bits() | exponent) - (two_m + $bias as $t))
            }
            #[inline]
            fn wide(self) -> f64 { self as f64 }
            // The lowest bit of `z` is the parity of n. Shifted into the sign bit, it negates `self` with no branch.
            #[inline]
            fn flip(self, z: $t) -> $t { $t::from_bits(self.to_bits() ^ (z.to_bits() << (<$bits>::BITS - 1))) }
        }
    };
}
native!(f32, u32, 23, 127);
native!(f64, u64, 52, 1023);

impl sealed::Sealed for half::f16 {}
impl Float for half::f16 {
    type Total = f32;
    #[inline]
    fn total(self) -> f32 { self.to_f32() }
    #[inline]
    fn from_total(t: f32) -> Self { half::f16::from_f32(t) }
    #[inline]
    fn narrow(x: f64) -> Self { half::f16::from_f64(x) }
    // The float above a positive float has the next bit pattern. The float above a negative float has the previous one. Both zeros
    // step to the least positive subnormal.
    #[inline]
    fn next_up(self) -> Self {
        let bits = self.to_bits();
        let next = if bits & 0x7fff == 0 { 1 } else if bits >> 15 == 0 { bits + 1 } else { bits - 1 };
        if self.is_nan() || self == half::f16::INFINITY { self } else { half::f16::from_bits(next) }
    }
    #[inline]
    fn next_down(self) -> Self { -(-self).next_up() }
}

/// The polynomial with coefficients `c`, highest degree first, at `x`, by Horner's rule. The coefficients convert to `R` when the
/// compiler inlines the call.
#[inline]
pub(crate) fn horner<R: Native, const FUSED: bool, const N: usize>(x: R, c: [f64; N]) -> R {
    let mut y = R::narrow(c[0]);
    for &k in &c[1..] { y = y.mla::<FUSED>(x, R::narrow(k)); }
    y
}
