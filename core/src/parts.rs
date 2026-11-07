//! The magnitude, sign and angle of a number, as function objects.
use crate::slice::Total;
use num_complex::{Complex32, Complex64};

/// |x|, or the magnitude |z| of a complex z, which is real, as a function object.
#[derive(Clone, Copy, Debug)]
pub struct Abs;
/// The sign of x as -1 or 1, or z/|z| for a complex z, as a function object. A zero gives itself, and NaN gives NaN.
#[derive(Clone, Copy, Debug)]
pub struct Sign;
/// The angle from the positive real axis, from -π to π, as a function object. It is real for complex items too. A real x has angle 0, or π where its sign bit is set, as the angle of x + 0i is.
#[derive(Clone, Copy, Debug)]
pub struct Angle;

macro_rules! objects {
    ($($(#[doc = $doc:literal])+ $name:ident;)+) => {$(
        $(#[doc = $doc])+
        #[derive(Clone, Copy, Debug)]
        pub struct $name;
    )+};
}
objects! {
    /// Whether a number has a NaN part.
    Isnan;
    /// Whether every part of a number is finite.
    Isfinite;
    /// Whether a number has an infinite part.
    Isinf;
    /// The real part of a number, with a real number unchanged.
    Real;
    /// The imaginary part of a number, zero for a real number.
    Imag;
    /// The complex conjugate, with a real number unchanged.
    Conj;
    /// Rounding toward zero, with whole numbers unchanged.
    Trunc;
    /// Rounding to the nearest whole number, with ties to even.
    Rint;
    /// Clipping x between lower and upper bounds in one pass: `minimum(upper, maximum(lower, x))`. A lower bound above the upper one gives the upper. A NaN argument gives NaN, x or a bound, depending on the target.
    Clip;
    /// `x` where `m` is nonzero and `y` where it's zero, item by item: a select, with the mask `m` as lanes of the items' type.
    Choose;
}

/// Multiplication by a fixed real scale, computing f16 in f32 before rounding once.
#[derive(Clone, Copy, Debug)]
pub struct Scale(pub f64);

/// Decimal rounding to nearest, ties to even. Constructing it precomputes the scale; extreme negative/positive precisions give zero/leave inputs unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Round { pub(crate) scale: f64 }
impl Round {
    /// Rounds to this many digits after the decimal point; negative values round to tens, hundreds and so on.
    pub fn new(decimals: i32) -> Self { Self { scale: 10f64.powi(decimals) } }
}

#[inline(always)]
pub(crate) fn rounded<F: crate::Float>(x: F, scale: f64) -> F {
    let scale = F::narrow(scale);
    let scaled = x * scale;
    if !x.is_finite() || scale.is_infinite() || scaled.is_infinite() { x }
    else if scale == F::zero() { x * F::zero() }
    else { scaled.round_ties_even() / scale }
}

/// `x` rounded to the nearest multiple of 1 ÷ `scale`, ties to even, exactly at `i128`, where no `i64` rounds past the type.
#[inline(always)]
pub(crate) fn rounded_wide(x: i128, scale: f64) -> i128 {
    if scale >= 1.0 { return x; }
    let scale = (1.0 / scale).round() as i128;
    let (q, rem) = (x / scale, x % scale);
    let away = rem.abs() * 2;
    let q = q + if away > scale || away == scale && q & 1 != 0 { x.signum() } else { 0 };
    q * scale
}

forms!(Choose at real: |_, [x, y, m]| Total(if m != 0.0 { x } else { y }));
forms!(Choose at Complex32, Complex64: |_, [x, y, m]| Total(if m.re != 0.0 { x } else { y }));
forms!(Choose at u8, i16, i32, i64: |_, [x, y, m]| if m != 0 { x } else { y });
forms!(Choose at bool: |_, [x, y, m]| (m & x) | (!m & y));
forms!(Choose at char: |_, [x, y, m]| if m == '\u{1}' { x } else { y });
