//! Square roots: √x, and √(1 - x²), √(1 + x²) and √(x - 1) √(x + 1) without the digits that squaring x first loses or the overflow
//! it risks. `complex.rs` attaches the complex forms of all four.
use crate::float::Native;

/// √x, or the principal √z of a complex z, whose real part is never negative, as a function object. A negative x gives NaN.
#[derive(Clone, Copy, Debug)]
pub struct Sqrt;
forms!(Sqrt at real: |_, [x]| x.sqrt());

/// √(1 - x²) as √((1 - x)(1 + x)). For |x| from 1/2 to 1, the factor that nears 0 is exact, and elsewhere neither factor loses
/// digits. A magnitude beyond 1 gives NaN.
#[inline]
pub(crate) fn sqrt1mx2<R: Native, const FUSED: bool>(x: R) -> R { ((R::one() - x) * (R::one() + x)).sqrt() }

/// √(1 + x²) as the hypotenuse of x and 1, which never overflows.
#[inline]
pub(crate) fn sqrt1px2<R: Native, const FUSED: bool>(x: R) -> R { crate::pow::hypot::<R, FUSED>(x, R::one()) }

/// √(x - 1) √(x + 1): √(x² - 1) with the sign of x, computed as √((|x| - 1)(|x| + 1)). Where that product overflows, the root
/// rounds to |x|. A magnitude below 1 gives NaN.
#[inline]
pub(crate) fn sqrtx2m1<R: Native, const FUSED: bool>(x: R) -> R {
    let a = x.abs();
    let p = (a - R::one()) * (a + R::one());
    (if p == R::infinity() { a } else { p.sqrt() }).copysign(x)
}
