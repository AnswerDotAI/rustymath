//! Maths in plain Rust. Its float functions take `f16`, `f32` and `f64`, each within 4 units in the last place (ulps) of the correctly
//! rounded result. The crate also has APL's tolerant comparison, exact rationals, linear algebra, number theory and polynomials. Its
//! Python package gives the float functions as NumPy ufuncs.
#![doc = include_str!("lib.md")]
#[macro_use]
mod slice;
mod arc;
mod arith;
mod complex;
mod dd;
mod exp;
mod field;
mod float;
mod hyperbolic;
mod lambert;
mod ln;
mod parts;
mod pow;
mod special;
mod sqrts;
mod table;
mod trig;
pub mod convert;
pub mod distribution;
pub mod gather;
pub mod exact;
pub mod integer;
pub mod linalg;
pub mod number_theory;
pub mod parallel;
pub mod polynomial;
pub mod reduce;
pub mod random;
pub mod search;
pub mod tolerant;

pub use float::{Float, Native, FUSED};
pub use slice::{pairs, Arg, Binary, Flipped, Ternary, Unary, CHUNK};
pub use parts::{Abs, Angle, Sign};
pub use sqrts::Sqrt;
pub use ln::Logb;
pub use pow::{Pow, Root};
pub use trig::{Cos, Sin, Tan};
pub use special::{Erfc, Erfcinv, Erfinv};
pub use complex::{divide, Cis};
pub use arith::{Add, Multiply, Overflowing, Subtract};

/// The check for a `_with` function's plain version, which can't fail.
pub(crate) fn ok() -> Result<(), std::convert::Infallible> { Ok(()) }
/// How many items a `_with` function handles between calls to its check, which keeps the call out of the loop over items.
pub const BETWEEN_CHECKS: usize = 1024;

/// Declares each function at the crate root, where it multiplies and adds as the target's `FUSED` says, and in `fused` and
/// `unfused`, which choose for themselves. Each entry names the module whose function of the same name implements it, generic over
/// `R: Native` and `const FUSED: bool`. Every wrapper is `#[inline(always)]`, because
/// LLVM leaves a large function such as `gamma` out of line in a caller's loop, and the call stops that loop vectorizing. An entry
/// marked `as` also declares a function object of that name, whose forms rustymath compiles and never inlines into a caller. Its
/// scalar form calls the function, and its slice form runs the function on each item. An entry marked `: total` never gives NaN from
/// arguments that aren't NaN. Its object's body gives a [`Total`](slice::Total) output, and the object's slice loop tests no item. An
/// object with forms of its own, such as [`Pow`], is written with `forms!` beside its function.
macro_rules! functions {
    ($($(#[doc = $doc:literal])* $name:ident($($x:ident),+) in $module:ident $(as $object:ident $(: $outcome:ident)?)?;)+) => {
        $(
            $(#[doc = $doc])*
            #[inline(always)]
            pub fn $name<F: Float>($($x: F),+) -> F { functions!(@call FUSED, $module, $name, ($($x),+)) }
        )+
        $(functions!(@object $($object $(: $outcome)?)?; $name, $module, ($($x),+));)+
        /// rustymath's functions with `mla` fused, whatever the target's `FUSED`. Code compiled with FMA enabled for itself, such as an
        /// AVX2 version of a loop chosen when the program runs, calls these on a target whose baseline lacks FMA.
        pub mod fused {
            use crate::Float;
            $(
                #[doc = concat!("[`", stringify!($name), "`](crate::", stringify!($name), "), with `mla` fused.")]
                #[inline(always)]
                pub fn $name<F: Float>($($x: F),+) -> F { functions!(@call true, $module, $name, ($($x),+)) }
            )+
        }
        /// rustymath's functions with `mla` unfused, whatever the target's `FUSED`.
        pub mod unfused {
            use crate::Float;
            $(
                #[doc = concat!("[`", stringify!($name), "`](crate::", stringify!($name), "), with `mla` unfused.")]
                #[inline(always)]
                pub fn $name<F: Float>($($x: F),+) -> F { functions!(@call false, $module, $name, ($($x),+)) }
            )+
        }
    };
    (@object; $name:ident, $module:ident, ($($x:ident),+)) => {};
    (@object $object:ident $(: $outcome:ident)?; $name:ident, $module:ident, ($($x:ident),+)) => {
        #[doc = concat!("[`", stringify!($name), "`](crate::", stringify!($name), ") as a function object, with the slice form `many`.")]
        #[derive(Clone, Copy, Debug)]
        pub struct $object;
        forms!($object at real: |_, [$($x),+]| functions!(@outcome [$($outcome)?], crate::$module::$name::<_, { crate::FUSED }>($($x),+)));
    };
    (@outcome [], $e:expr) => { $e };
    (@outcome [total], $e:expr) => { crate::slice::Total($e) };
    (@call $fused:expr, $module:ident, $name:ident, ($($x:ident),+)) => { F::from_working(crate::$module::$name::<_, { $fused }>($($x.working()),+)) };
}

functions! {
    /// e^x. `exp(∞)` is ∞, `exp(-∞)` is 0, and `exp(NaN)` is NaN.
    exp(x) in exp as Exp: total;
    /// e^x - 1, without the loss of digits near 0 that `exp(x) - 1` has.
    expm1(x) in exp as Expm1: total;
    /// The natural logarithm of x. `ln(0)` is -∞, `ln(∞)` is ∞, and a negative x or NaN gives NaN.
    ln(x) in ln as Ln;
    /// ln(1 + x), without the loss of digits near 0 that `ln(1 + x)` has. `log1p(-1)` is -∞.
    log1p(x) in ln as Log1p;
    /// The logarithm of x to base 2. It is k exactly at 2^k.
    log2(x) in ln as Log2;
    /// The logarithm of x to base 10. It is k exactly at 10^k, and at the float nearest 10^k where 10^k isn't a float.
    log10(x) in ln as Log10;
    /// The logarithm of x to base b, ln x / ln b. Base 2 and base 10 take `log2` and `log10`, which are exact at powers of their base.
    logb(b, x) in ln;
    /// sin x. `sin(±0)` is ±0, and `sin(±∞)` and `sin(NaN)` are NaN. Beyond a magnitude of 2^23 at f64 or 2^20 at f32, x takes an
    /// exact reduction, which is slower and has a branch. A loop that calls `sin` therefore doesn't vectorize. [`Sin`]'s slice form
    /// does: it reduces every item the fast way, then gives only the rare large ones the exact reduction.
    sin(x) in trig;
    /// cos x. `cos(±0)` is 1, and `cos(±∞)` and `cos(NaN)` are NaN. Beyond the magnitude that `sin` gives,
    /// x takes the same exact reduction.
    cos(x) in trig;
    /// tan x. `tan(±0)` is ±0, and `tan(±∞)` and `tan(NaN)` are NaN. Beyond the magnitude that `sin`
    /// gives, x takes the same exact reduction.
    tan(x) in trig;
    /// asin x, from -π/2 to π/2. A magnitude beyond 1 gives NaN.
    asin(x) in arc as Asin;
    /// acos x, from 0 to π. A magnitude beyond 1 gives NaN.
    acos(x) in arc as Acos;
    /// atan x, from -π/2 to π/2.
    atan(x) in arc as Atan: total;
    /// The angle of the point (x, y) from the positive x axis, from -π to π, with y first as in `f64::atan2`. Zeros and infinities give
    /// C99's angles: atan2(±0, -0) is ±π, and atan2(±∞, ±∞) is an odd multiple of ±π/4.
    atan2(y, x) in arc as Atan2: total;
    /// sinh x. It stays finite up to where sinh itself overflows, beyond the point where e^x does.
    sinh(x) in hyperbolic as Sinh: total;
    /// cosh x. It stays finite up to where cosh itself overflows, beyond the point where e^x does.
    cosh(x) in hyperbolic as Cosh: total;
    /// tanh x.
    tanh(x) in hyperbolic as Tanh: total;
    /// asinh x.
    asinh(x) in hyperbolic as Asinh: total;
    /// acosh x. x below 1 gives NaN.
    acosh(x) in hyperbolic as Acosh;
    /// atanh x. ±1 gives ±∞, and a magnitude beyond 1 gives NaN.
    atanh(x) in hyperbolic as Atanh;
    /// x^y. Special values follow IEEE 754's `pow`: x^0 and 1^y are 1, even for NaN, and a negative x has a power only for a whole y.
    pow(x, y) in pow;
    /// The real nth root of y, computed without rounding 1/n first. A negative y has one only for an odd
    /// whole n. n = 0, an infinite n and NaN give NaN.
    root(n, y) in pow;
    /// The cube root of x.
    cbrt(x) in pow as Cbrt: total;
    /// √(x² + y²), with no overflow or underflow in between. An infinite argument gives ∞, even beside NaN.
    hypot(x, y) in pow as Hypot: total;
    /// √(1 - x²), without the digits that squaring x first loses near |x| = 1. A magnitude beyond 1 gives NaN.
    sqrt1mx2(x) in sqrts as Sqrt1mx2;
    /// √(1 + x²), with no overflow where x² would overflow.
    sqrt1px2(x) in sqrts as Sqrt1px2: total;
    /// √(x - 1) √(x + 1): √(x² - 1) with the sign of x, without the digits that squaring x first loses near |x| = 1. A magnitude
    /// below 1 gives NaN.
    sqrtx2m1(x) in sqrts as Sqrtx2m1;
    /// Γ(x). Γ(±0) is ±∞, a negative whole x and -∞ give NaN, and Γ(∞) is ∞.
    gamma(x) in special as Gamma;
    /// Γ(x + 1), which is x! for a whole x from 0.
    factorial(x) in special as Factorial;
    /// ln |Γ(x)|. It is ∞ at 0, at the negative whole numbers and at ±∞. Near its zeros at 1 and 2 its error is within 4 ulps of 2,
    /// not of the result.
    lgamma(x) in special as Lgamma: total;
    /// C(n, k), the binomial coefficient. Whole n and k from 0 multiply out exactly while the result times |n| + k is a whole number
    /// the type holds, and a negative whole n gives (-1)^k C(k - n - 1, k). Other arguments go through Γ, with the complex binomial
    /// coefficient's error.
    binom(n, k) in special as Binom;
    /// The error function, 2/√π times the integral of e^-t² from 0 to x.
    erf(x) in special as Erf: total;
    /// 1 - erf(x), without the loss of digits for large x that `1 - erf(x)` has.
    erfc(x) in special;
    /// The x with erf(x) = p. ±1 gives ±∞, and a magnitude beyond 1 gives NaN.
    erfinv(p) in special;
    /// The x with erfc(x) = q. 0 gives ∞, 2 gives -∞, and q outside 0 to 2 gives NaN.
    erfcinv(q) in special;
    /// W0(x), the principal branch of the Lambert W function: the w of at least -1 with w e^w = x. x below -1/e gives NaN, except
    /// within half an ulp of -1/e, where W0 is -1.
    lambertw(x) in lambert as LambertW;
    /// The regularized lower incomplete gamma function P(a, x): the integral of t^(a-1) e^-t from 0 to x, divided by Γ(a). a must be
    /// positive and x nonnegative, or the result is NaN.
    gammainc(a, x) in special as Gammainc;
    /// The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x), without the loss of digits that `1 - gammainc(a, x)`
    /// has where Q is small.
    gammaincc(a, x) in special as Gammaincc;
    /// The x with P(a, x) = p, for p from 0 to 1.
    gammaincinv(a, p) in special as Gammaincinv;
    /// The regularized incomplete beta function I_x(a, b): the integral of t^(a-1) (1-t)^(b-1) from 0 to x, divided by B(a, b). a and
    /// b must be positive and x from 0 to 1, or the result is NaN.
    betainc(a, b, x) in special as Betainc;
    /// The x with I_x(a, b) = p, for p from 0 to 1.
    betaincinv(a, b, p) in special as Betaincinv;
}
