#![doc = include_str!("../README.md")]
mod arc;
mod dd;
mod exp;
mod float;
mod hyperbolic;
mod ln;
mod pow;
mod special;
mod table;
mod trig;
pub mod distribution;
pub mod number_theory;
pub mod polynomial;
pub mod search;
pub mod tolerant;
#[cfg(test)]
mod accuracy;

pub use float::{Float, Native, FUSED};

/// The check for a `_with` function's plain version, which can't fail.
pub(crate) fn ok() -> Result<(), std::convert::Infallible> { Ok(()) }
/// How many items a `_with` function handles between calls to its check, which keeps the call out of the loop over items.
pub(crate) const BETWEEN_CHECKS: usize = 1024;

/// Declares each function at the crate root, where it multiplies and adds as the target's `FUSED` says, and in `fused` and
/// `unfused`, which choose for themselves. Each entry names the module whose function of the same name implements it, generic over
/// `R: Native` and `const FUSED: bool`. An entry marked `Option` gives an `Option`. Every wrapper is `#[inline(always)]`, because
/// LLVM leaves a large function such as `gamma` out of line in a caller's loop, and the call stops that loop vectorizing.
macro_rules! functions {
    ($($(#[doc = $doc:literal])* $name:ident($($x:ident),+) $(-> $option:ident)? in $module:ident;)+) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name<F: Float>($($x: F),+) -> functions!(@ret $($option)?) { functions!(@call FUSED, $module, $name, ($($x),+) $($option)?) }
        )+
        /// rustymath's functions with `mla` fused, whatever the target's `FUSED`. Code compiled with FMA enabled for itself, such as an
        /// AVX2 version of a loop chosen when the program runs, calls these on a target whose baseline lacks FMA.
        pub mod fused {
            use crate::Float;
            $(
                #[doc = concat!("[`", stringify!($name), "`](crate::", stringify!($name), "), with `mla` fused.")]
                #[inline(always)]
                pub fn $name<F: Float>($($x: F),+) -> functions!(@ret $($option)?) { functions!(@call true, $module, $name, ($($x),+) $($option)?) }
            )+
        }
        /// rustymath's functions with `mla` unfused, whatever the target's `FUSED`.
        pub mod unfused {
            use crate::Float;
            $(
                #[doc = concat!("[`", stringify!($name), "`](crate::", stringify!($name), "), with `mla` unfused.")]
                #[inline(always)]
                pub fn $name<F: Float>($($x: F),+) -> functions!(@ret $($option)?) { functions!(@call false, $module, $name, ($($x),+) $($option)?) }
            )+
        }
    };
    (@ret) => { F };
    (@ret Option) => { Option<F> };
    (@call $fused:expr, $module:ident, $name:ident, ($($x:ident),+)) => { F::from_total(crate::$module::$name::<_, { $fused }>($($x.total()),+)) };
    (@call $fused:expr, $module:ident, $name:ident, ($($x:ident),+) Option) => {
        crate::$module::$name::<_, { $fused }>($($x.total()),+).map(F::from_total)
    };
}

functions! {
    /// e^x. `exp(∞)` is ∞, `exp(-∞)` is 0, and `exp(NaN)` is NaN.
    exp(x) in exp;
    /// e^x - 1, without the loss of digits near 0 that `exp(x) - 1` has.
    expm1(x) in exp;
    /// The natural logarithm of x. `ln(0)` is -∞, `ln(∞)` is ∞, and a negative x or NaN gives NaN.
    ln(x) in ln;
    /// ln(1 + x), without the loss of digits near 0 that `ln(1 + x)` has. `log1p(-1)` is -∞.
    log1p(x) in ln;
    /// The logarithm of x to base 2.
    log2(x) in ln;
    /// The logarithm of x to base 10.
    log10(x) in ln;
    /// sin x. `sin(±0)` is ±0, and `sin(±∞)` and `sin(NaN)` are NaN. Beyond a magnitude of 2^23 at
    /// f64 or 2^20 at f32, x takes an exact reduction, which is slower and has a branch. A loop that calls `sin` therefore doesn't
    /// vectorize. A loop that calls `checked_sin` does.
    sin(x) in trig;
    /// cos x. `cos(±0)` is 1, and `cos(±∞)` and `cos(NaN)` are NaN. Beyond the magnitude that `sin` gives,
    /// x takes the same exact reduction.
    cos(x) in trig;
    /// tan x. `tan(±0)` is ±0, and `tan(±∞)` and `tan(NaN)` are NaN. Beyond the magnitude that `sin`
    /// gives, x takes the same exact reduction.
    tan(x) in trig;
    /// sin x, as `sin` gives it, with no branch: `None` for a finite x beyond the magnitude where `sin` takes its exact reduction. A
    /// loop that calls it vectorizes.
    ///
    /// ```
    /// assert_eq!(rustymath::checked_sin(1e6f64), Some(rustymath::sin(1e6)));
    /// assert_eq!(rustymath::checked_sin(1e6f32), Some(rustymath::sin(1e6)));
    /// assert_eq!(rustymath::checked_sin(2e6f32), None);
    /// assert_eq!(rustymath::checked_sin(1e8f64), None);
    /// assert!(rustymath::checked_sin(f64::INFINITY).unwrap().is_nan());
    /// ```
    checked_sin(x) -> Option in trig;
    /// cos x, as `cos` gives it, with no branch: `None` where `checked_sin` gives `None`.
    checked_cos(x) -> Option in trig;
    /// tan x, as `tan` gives it, with no branch: `None` where `checked_sin` gives `None`.
    checked_tan(x) -> Option in trig;
    /// asin x, from -π/2 to π/2. A magnitude beyond 1 gives NaN.
    asin(x) in arc;
    /// acos x, from 0 to π. A magnitude beyond 1 gives NaN.
    acos(x) in arc;
    /// atan x, from -π/2 to π/2.
    atan(x) in arc;
    /// sinh x. It stays finite up to where sinh itself overflows, beyond the point where e^x does.
    sinh(x) in hyperbolic;
    /// cosh x. It stays finite up to where cosh itself overflows, beyond the point where e^x does.
    cosh(x) in hyperbolic;
    /// tanh x.
    tanh(x) in hyperbolic;
    /// asinh x.
    asinh(x) in hyperbolic;
    /// acosh x. x below 1 gives NaN.
    acosh(x) in hyperbolic;
    /// atanh x. ±1 gives ±∞, and a magnitude beyond 1 gives NaN.
    atanh(x) in hyperbolic;
    /// x^y. Special values follow IEEE 754's `pow`: x^0 and 1^y are 1, even for NaN, and a negative x has a power only for a whole y.
    ///
    /// ```
    /// assert!((rustymath::pow(2.0f64, 0.5) - std::f64::consts::SQRT_2).abs() < 1e-15);
    /// assert!(rustymath::pow(-2.0f32, 3.0) < 0.0);
    /// assert!(rustymath::pow(-2.0f64, 0.5).is_nan());
    /// assert_eq!(rustymath::pow(f64::NAN, 0.0), 1.0);
    /// ```
    pow(x, y) in pow;
    /// The real nth root of y, computed without rounding 1/n first. A negative y has one only for an odd
    /// whole n. n = 0, an infinite n and NaN give NaN.
    root(n, y) in pow;
    /// The cube root of x.
    cbrt(x) in pow;
    /// √(x² + y²), with no overflow or underflow in between. An infinite argument gives ∞, even beside NaN.
    hypot(x, y) in pow;
    /// Γ(x). Γ(±0) is ±∞, a negative whole x and -∞ give NaN, and Γ(∞) is ∞.
    gamma(x) in special;
    /// ln |Γ(x)|. It is ∞ at 0, at the negative whole numbers and at ±∞. Near its zeros at 1 and 2 its error is within 4 ulps of 2,
    /// not of the result.
    lgamma(x) in special;
    /// The error function, 2/√π times the integral of e^-t² from 0 to x.
    erf(x) in special;
    /// 1 - erf(x), without the loss of digits for large x that `1 - erf(x)` has.
    erfc(x) in special;
    /// The x with erf(x) = p. ±1 gives ±∞, and a magnitude beyond 1 gives NaN.
    erfinv(p) in special;
    /// The x with erfc(x) = q. 0 gives ∞, 2 gives -∞, and q outside 0 to 2 gives NaN.
    erfcinv(q) in special;
    /// The regularized lower incomplete gamma function P(a, x): the integral of t^(a-1) e^-t from 0 to x, divided by Γ(a). a must be
    /// positive and x nonnegative, or the result is NaN.
    gammainc(a, x) in special;
    /// The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x), without the loss of digits that `1 - gammainc(a, x)`
    /// has where Q is small.
    gammaincc(a, x) in special;
    /// The x with P(a, x) = p, for p from 0 to 1.
    gammaincinv(a, p) in special;
    /// The regularized incomplete beta function I_x(a, b): the integral of t^(a-1) (1-t)^(b-1) from 0 to x, divided by B(a, b). a and
    /// b must be positive and x from 0 to 1, or the result is NaN.
    betainc(a, b, x) in special;
    /// The x with I_x(a, b) = p, for p from 0 to 1.
    betaincinv(a, b, p) in special;
}
