//! Square roots: √x, and √(1 - x²), √(1 + x²) and √(x - 1) √(x + 1) without the digits that squaring x first loses or the overflow it risks. `complex.rs` computes all four at complex arguments.
use crate::float::Native;

/// √x, or the principal √z of a complex z, whose real part is never negative, as a function object. A negative x gives NaN.
#[derive(Clone, Copy, Debug)]
pub struct Sqrt;

/// The real part of √x, or of the principal √z of a complex z, as a function object: √x for a positive x, and 0 at or below 0. Its formula is `recipe::real_sqrt`.
#[derive(Clone, Copy, Debug)]
pub struct RealSqrt;

/// The real part of √(1 - x²), or of its principal value at a complex z, as a function object: √(1 - x²) for x from -1 to 1, and 0 beyond. The real part of `0○` and `0○⁻¹`.
#[derive(Clone, Copy, Debug)]
pub struct RealSqrt1mx2;

/// The real part of √(x - 1) √(x + 1), or of its principal value at a complex z, as a function object: x's sign times √(x² - 1) for |x| at least 1, and 0 between -1 and 1. The real part of `4○⁻¹`, on APL's branch.
#[derive(Clone, Copy, Debug)]
pub struct RealSqrtx2m1;

/// The real part of √(-1 - x²), or of its principal value at a complex z, negated where the flag is set, as a function object: 0 for every real x, whose `8○` is imaginary. The real part of `8○`, and of `8○⁻¹` with the flag.
#[derive(Clone, Copy, Debug)]
pub struct RealSqrtm1mx2(pub bool);

/// √(1 - x²) as √((1 - x)(1 + x)). For |x| from 1/2 to 1, the factor that nears 0 is exact, and elsewhere neither factor loses digits. A magnitude beyond 1 gives NaN.
#[inline]
pub(crate) fn sqrt1mx2<R: Native, const FUSED: bool>(x: R) -> R { ((R::one() - x) * (R::one() + x)).sqrt() }

/// √(1 + x²) as the hypotenuse of x and 1, which never overflows.
#[inline]
pub(crate) fn sqrt1px2<R: Native, const FUSED: bool>(x: R) -> R { crate::pow::hypot::<R, FUSED>(x, R::one()) }

/// √(x - 1) √(x + 1): √(x² - 1) with the sign of x, computed as √((|x| - 1)(|x| + 1)). Where that product overflows, the root rounds to |x|. A magnitude below 1 gives NaN.
#[inline]
pub(crate) fn sqrtx2m1<R: Native, const FUSED: bool>(x: R) -> R {
    let a = x.abs();
    let p = (a - R::one()) * (a + R::one());
    (if p == R::infinity() { a } else { p.sqrt() }).copysign(x)
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct Sqrtm1mx2(pub(crate) bool);
