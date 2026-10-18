//! The float types that rustymath's functions take, and the operations that its algorithms build on.
use crate::search::{Key, RadixKey};

/// Whether `mla` fuses its multiply and add into one operation with one rounding. It does where the target has a fused multiply-add instruction, as on AArch64 and on x86-64 with FMA. Elsewhere, as in wasm and on baseline x86-64, `mul_add` calls a software `fma`, which is slow and stops the loop around it vectorizing. There `mla` multiplies and adds with two roundings.
pub const FUSED: bool = cfg!(any(target_arch = "aarch64", target_feature = "fma"));

/// A float type that rustymath's functions take: `half::f16`, `f32` or `f64`.
pub trait Float: num_traits::Float + sealed::Sealed + sealed::Item {
    /// The type this float computes in, its working precision: `f32` for `f16`, and the type itself otherwise. A function of an `f16` computes in `f32` and rounds once. Sums keep their running totals in this type.
    type Working: Native;
    /// This float in its working type, which holds it exactly.
    fn working(self) -> Self::Working;
    /// `t` rounded to this type.
    fn from_working(t: Self::Working) -> Self;
    /// This float as an `f64`, which holds it exactly.
    fn wide(self) -> f64;
    /// `x` rounded to the nearest float of this type.
    fn narrow(x: f64) -> Self;
    /// The least float of this type above `self`, as `f64::next_up` gives it.
    fn next_up(self) -> Self;
    /// The greatest float of this type below `self`, as `f64::next_down` gives it.
    fn next_down(self) -> Self;
    /// The unsigned type, of this float's width, that `order_key` gives.
    type OrderKey: RadixKey + Key;
    /// A key whose unsigned order is the order of `total_cmp`, after `¯0` becomes `0` and every NaN becomes one NaN. So `¯0` and `0` share a key, NaN's key follows every other, and two floats match exactly when their keys are equal.
    fn order_key(self) -> Self::OrderKey;
    /// The float whose `order_key` is `k`. Every NaN reads back as one NaN, and `¯0` as `0`.
    fn from_order_key(k: Self::OrderKey) -> Self;
}

/// A float type that rustymath's algorithms run in: `f32` or `f64`.
pub trait Native: Float<Working = Self> + sealed::Core + sealed::Folds {
    /// `+` that may be reordered with other algebraic operations. A sum of them vectorizes.
    fn algebraic_add(self, other: Self) -> Self;
    /// `×` that may be reordered with other algebraic operations.
    fn algebraic_mul(self, other: Self) -> Self;
}

/// Traits that only rustymath implements. `Core` holds the bit-level operations that the algorithms build on. `Item` and `Forms` describe function objects.
pub(crate) mod sealed {
    use crate::slice::Arg;
    use num_complex::Complex;
    pub trait Sealed {}
    /// A type that function objects take or give: a float, a complex number, an integer or a Boolean. Pieces on several threads read and write items.
    pub trait Item: Copy + Send + Sync {
        /// Whether the type has NaN, as floats and complex numbers do.
        const NAN: bool;
        /// Whether this item is NaN, or for a complex number has a NaN part. An integer or Boolean never is.
        fn has_nan(self) -> bool;
    }
    /// A function object of `N` arguments at item type `E`, compiled inside rustymath and never inlined into a caller. `forms!` writes the impls, and [`Unary`](crate::Unary), [`Binary`](crate::Binary) and [`Ternary`](crate::Ternary) call them. A call can split its slice across threads.
    pub trait Forms<E: Item, const N: usize>: Sync {
        type Out: Item;
        fn scalar(&self, x: [E; N]) -> Self::Out;
        fn checked_scalar(&self, x: [E; N]) -> Option<Self::Out>;
        /// Whether no item fails: `checked_scalar` always gives `Some`, and `slice` always gives true.
        const TOTAL: bool;
        /// Whether a failed item's slot holds its result wrapped to the output type, as integer arithmetic's does.
        const WRAPS: bool;
        fn slice(&self, x: [Arg<'_, E>; N], n: usize, out: &mut [std::mem::MaybeUninit<Self::Out>]) -> bool;
    }
    macro_rules! nan_items { ($($t:ty),+) => {$( impl Item for $t { const NAN: bool = true; #[inline(always)] fn has_nan(self) -> bool { self.is_nan() } } )+} }
    nan_items!(f32, f64, half::f16, Complex<f32>, Complex<f64>);
    macro_rules! exact_items { ($($t:ty),+) => {$( impl Item for $t { const NAN: bool = false; #[inline(always)] fn has_nan(self) -> bool { false } } )+} }
    exact_items!(u8, i16, i32, i64, bool, char, std::cmp::Ordering);
    macro_rules! sealed { ($($t:ty),+) => {$( impl Sealed for $t {} )+} }
    sealed!(i8, i16, i32, i64, u8, u16, u32);
    impl<T: super::Native> Sealed for Complex<T> {}

    pub trait Core: Copy {
        /// Whether this is `f64`. Algorithms pick their constants by it, and the compiler removes the branch.
        const WIDE: bool;
        /// 1.5 × 2^m, where m is the number of mantissa bits. Adding it to a float of magnitude below 2^(m-1) rounds that float to a whole number, which the low bits of the sum hold.
        const SHIFT: Self;
        /// The unsigned integer type with this float's bits.
        type Bits: Copy;
        /// `u - less` as a float, for a whole number u below 2^m, through the bits of 2^m + u. A loop that calls it vectorizes even where the target has no vector conversion for u's type. [`convert`](crate::convert) builds on it.
        fn from_low(u: Self::Bits, less: Self) -> Self;
        /// The whole number nearest `self`, for `self` from 0 to below 2^m, as the low bits of 2^m + `self`. `from_low` reverses it.
        fn to_low(self) -> Self::Bits;
        /// The bits of a positive `self` without their low `BITS - m`, read as a whole number, as a float. They grow nearly in proportion to `self`'s logarithm. `from_top` reverses it.
        fn top(self) -> Self;
        /// The float whose bits without their low `BITS - m` are the whole number `t`, for a `t` from 0 to below 2^m.
        fn from_top(t: Self) -> Self;
        /// `self×b+c`, with one rounding when `FUSED` is true and two otherwise.
        fn mla<const FUSED: bool>(self, b: Self, c: Self) -> Self;
        /// 2^n, from `z` = `SHIFT + n`, for a whole n within the exponents of normal floats.
        fn pow2(z: Self) -> Self;
        /// `(m, e)` with `self` = m × 2^e, m from 0.75 to 1.5 and e whole, for a positive normal `self`.
        fn split(self) -> (Self, Self);
        /// `self` negated when the whole number n is odd, for `z` = `SHIFT + n`.
        fn flip(self, z: Self) -> Self;
        /// Whether the whole number n is odd, for `self` = `SHIFT + n`.
        fn odd(self) -> bool;
        /// `self` with the low half of its mantissa bits cleared. The product of two such floats is exact, which double-word products without FMA rely on.
        fn upper(self) -> Self;
        /// `(s, 1/s)` for the power of two s that brings `self` into [2, 4), after raising a smaller `self` to 2^(2 - bias). Both are normal for every finite `self`, and they come from its exponent bits with no branch.
        fn scales(self) -> (Self, Self);
        /// The greater of `self` and `y`, from a comparison and a select. Where they're equal, or either is NaN, it gives `y`, as x86-64's `maxsd` does. A loop of them vectorizes on every target.
        fn maximum(self, y: Self) -> Self;
        /// The lesser of `self` and `y`, as `maximum` gives the greater.
        fn minimum(self, y: Self) -> Self;
    }
    /// The folds and scans of floats by maximum and minimum, and their running sums. The defaults' fold loops vectorize. Each step of a maximum or minimum passes the running result as `Core::maximum`'s or `Core::minimum`'s `y`, which a NaN item leaves unchanged. The default running sum is `grouped`'s. `reduce.rs` overrides them where a target has instructions that do better.
    pub trait Folds: Core + num_traits::Float {
        fn fold_max(x: &[Self]) -> Self { crate::reduce::folded(x, Self::neg_infinity(), |t, v| Self::maximum(v, t), Self::maximum) }
        fn fold_min(x: &[Self]) -> Self { crate::reduce::folded(x, Self::infinity(), |t, v| Self::minimum(v, t), Self::minimum) }
        fn scan_max(x: &[Self], start: Option<Self>, out: &mut Vec<Self>) {
            crate::reduce::records(x, start, out, |t, v| Self::maximum(v, t), crate::reduce::below)
        }
        fn scan_min(x: &[Self], start: Option<Self>, out: &mut Vec<Self>) {
            crate::reduce::records(x, start, out, |t, v| Self::minimum(v, t), crate::reduce::above)
        }
        fn scan_sum(x: &[Self], start: Option<Self>, out: &mut Vec<Self>) { crate::reduce::grouped(x, start, out, |v| v, |a, b| a + b, |t| t) }
    }
}

/// `Float`'s order-key items, for a float whose bits have type `$k`. A key is the bits with the sign bit flipped, and a negative float's other bits flipped too. `¯0` adds to `0` first, and every NaN takes the greatest key.
macro_rules! order_key {
    ($k:ty) => {
        type OrderKey = $k;
        #[inline(always)]
        fn order_key(self) -> $k {
            if self.is_nan() {
                return <$k>::MAX;
            }
            let bits = (self + <Self as num_traits::Zero>::zero()).to_bits();
            if bits >> (<$k>::BITS - 1) == 1 { !bits } else { bits | 1 << (<$k>::BITS - 1) }
        }
        #[inline(always)]
        fn from_order_key(k: $k) -> Self {
            Self::from_bits(if k >> (<$k>::BITS - 1) == 1 { k & !(1 << (<$k>::BITS - 1)) } else { !k })
        }
    };
}

macro_rules! native {
    ($t:ident, $bits:ty, $mantissa:literal, $bias:literal) => {
        impl sealed::Sealed for $t {}
        impl Float for $t {
            type Working = $t;
            #[inline]
            fn working(self) -> $t { self }
            #[inline]
            fn from_working(t: $t) -> $t { t }
            #[inline]
            fn wide(self) -> f64 { self as f64 }
            #[inline]
            fn narrow(x: f64) -> $t { x as $t }
            #[inline]
            fn next_up(self) -> $t { $t::next_up(self) }
            #[inline]
            fn next_down(self) -> $t { $t::next_down(self) }
            order_key!($bits);
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
            type Bits = $bits;
            #[inline]
            fn from_low(u: $bits, less: $t) -> $t {
                let two_m = (1 as $bits << $mantissa) as $t;
                $t::from_bits(two_m.to_bits() | u) - (two_m + less)
            }
            #[inline]
            fn to_low(self) -> $bits { (self + (1 as $bits << $mantissa) as $t).to_bits() & ((1 << $mantissa) - 1) }
            #[inline]
            fn top(self) -> $t { Self::from_low(self.to_bits() >> (<$bits>::BITS - $mantissa), 0.0) }
            #[inline]
            fn from_top(t: $t) -> $t { $t::from_bits(t.to_low() << (<$bits>::BITS - $mantissa)) }
            #[inline]
            fn mla<const FUSED: bool>(self, b: $t, c: $t) -> $t { if FUSED { self.mul_add(b, c) } else { self * b + c } }
            // The low bits of `z` hold n. Adding the bias to them and shifting them into the exponent field gives 2^n. Integer adds and shifts vectorize on every target, where a conversion from a float to a 64-bit integer doesn't on x86-64.
            #[inline]
            fn pow2(z: $t) -> $t { $t::from_bits(z.to_bits().wrapping_add($bias) << $mantissa) }
            // e comes from the exponent field of `self÷0.75`, converted through `from_low`.
            #[inline]
            fn split(self) -> ($t, $t) {
                let exponent = ((self * (1.0 / 0.75)).to_bits() >> $mantissa) & (2 * $bias + 1);
                let m = $t::from_bits(self.to_bits().wrapping_sub(exponent.wrapping_sub($bias) << $mantissa));
                (m, Self::from_low(exponent, $bias as $t))
            }
            // The lowest bit of `z` is the parity of n. Shifted into the sign bit, it negates `self` with no branch.
            #[inline]
            fn flip(self, z: $t) -> $t { $t::from_bits(self.to_bits() ^ (z.to_bits() << (<$bits>::BITS - 1))) }
            #[inline]
            fn odd(self) -> bool { self.to_bits() & 1 == 1 }
            #[inline]
            fn upper(self) -> $t { $t::from_bits(self.to_bits() & !((1 << (($mantissa + 2) / 2)) - 1)) }
            #[inline]
            fn scales(self) -> ($t, $t) {
                let bottom = $t::from_bits(2 << $mantissa);
                let e = (if self < bottom { bottom } else { self }).to_bits() >> $mantissa;
                ($t::from_bits((2 * $bias + 1 - e) << $mantissa), $t::from_bits((e - 1) << $mantissa))
            }
            #[inline(always)]
            fn maximum(self, y: $t) -> $t { if self > y { self } else { y } }
            #[inline(always)]
            fn minimum(self, y: $t) -> $t { if self < y { self } else { y } }
        }
    };
}
native!(f32, u32, 23, 127);
native!(f64, u64, 52, 1023);

impl sealed::Sealed for half::f16 {}
impl Float for half::f16 {
    type Working = f32;
    #[inline]
    fn working(self) -> f32 { self.to_f32() }
    #[inline]
    fn from_working(t: f32) -> Self { half::f16::from_f32(t) }
    #[inline]
    fn wide(self) -> f64 { self.to_f64() }
    #[inline]
    fn narrow(x: f64) -> Self { half::f16::from_f64(x) }
    // The float above a positive float has the next bit pattern. The float above a negative float has the previous one. Both zeros step to the least positive subnormal.
    #[inline]
    fn next_up(self) -> Self {
        let bits = self.to_bits();
        let next = if bits & 0x7fff == 0 { 1 } else if bits >> 15 == 0 { bits + 1 } else { bits - 1 };
        if self.is_nan() || self == half::f16::INFINITY { self } else { half::f16::from_bits(next) }
    }
    #[inline]
    fn next_down(self) -> Self { -(-self).next_up() }
    order_key!(u16);
}

/// The polynomial with coefficients `c`, highest degree first, at `x`, by Horner's rule. The coefficients convert to `R` when the compiler inlines the call.
#[inline]
pub(crate) fn horner<R: Native, const FUSED: bool, const N: usize>(x: R, c: [f64; N]) -> R {
    let mut y = R::narrow(c[0]);
    for &k in &c[1..] { y = y.mla::<FUSED>(x, R::narrow(k)); }
    y
}

/// ln 2, split into a part whose product with any whole number up to 2^11 at f64 or 2^8 at f32 is exact, and the rest. That covers every exponent of a float and every n that `exp` reduces by.
#[inline]
pub(crate) fn ln2<R: Native>() -> (R, R) {
    let (hi, lo) = if R::WIDE { (0.69314718055966295651160180568695068359375, 0.28235290563031577122588448175013436025525412068e-12) }
    else { (0.693145751953125, 1.428606765330187e-06) };
    (R::narrow(hi), R::narrow(lo))
}

/// y × 2^n, for a whole n, in two factors. The first multiplication is exact, so a result below the normal floats rounds once, in the second.
#[inline]
pub(crate) fn pow2_times<R: Native, const FUSED: bool>(y: R, n: R) -> R {
    let half = n.mla::<FUSED>(R::narrow(0.5), R::SHIFT);
    y * R::pow2(half) * R::pow2(n - (half - R::SHIFT) + R::SHIFT)
}

/// (m, e) with x = m × 2^e, m from 0.75 to 1.5 and e whole, for a positive x, subnormals included. A subnormal x scales up by 2^64 first, so that its bits split as a normal float's do.
#[inline]
pub(crate) fn split_positive<R: Native>(x: R) -> (R, R) {
    let subnormal = x < R::min_positive_value();
    let (m, e) = (if subnormal { x * R::narrow(18446744073709551616.0) } else { x }).split();
    (m, if subnormal { e - R::narrow(64.0) } else { e })
}
