//! The magnitude, sign and angle of a number, as function objects. Their real forms are here, and `complex.rs` attaches their complex ones.
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

forms!(Abs at real: |_, [x]| Total(x.abs()));
forms!(#[inline] Sign at real: |_, [x]| Total(if x == 0.0 { x } else { x.signum() }));
forms!(Angle at real: |_, [x]| Total(crate::arc::real_angle(x)));

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
forms!(Scale at real: |k, [x]| Total(scaled(x, k.0)));

#[inline(always)]
fn scaled<F: crate::Float>(x: F, scale: f64) -> F { x * F::narrow(scale) }

/// Decimal rounding to nearest, ties to even. Constructing it precomputes the scale; extreme negative/positive precisions give zero/leave inputs unchanged.
#[derive(Clone, Copy, Debug)]
pub struct Round { scale: f64 }
impl Round {
    /// Rounds to this many digits after the decimal point; negative values round to tens, hundreds and so on.
    pub fn new(decimals: i32) -> Self { Self { scale: 10f64.powi(decimals) } }
}
forms!(Round at real: |k, [x]| Total(rounded(x, k.scale)));
forms!(Round at Complex32, Complex64: |k, [x]| Total(num_complex::Complex::new(rounded(x.re, k.scale), rounded(x.im, k.scale))));
forms!(Round at u8, i16, i32, i64: |k, [x]| rounded_integer(x, k.scale));

#[inline(always)]
fn rounded<F: crate::Float>(x: F, scale: f64) -> F where Rint: crate::Unary<F, Out = F> {
    let scale = F::narrow(scale);
    let scaled = x * scale;
    if !x.is_finite() || scale.is_infinite() || scaled.is_infinite() { x }
    else if scale == F::zero() { x * F::zero() }
    else { crate::Unary::one(&Rint, scaled) / scale }
}

#[inline(always)]
fn rounded_integer<T: Into<i128> + TryFrom<i128>>(x: T, scale: f64) -> Option<T> {
    if scale >= 1.0 { return Some(x); }
    let scale = (1.0 / scale).round() as i128;
    let x = x.into();
    let (q, rem) = (x / scale, x % scale);
    let away = rem.abs() * 2;
    let q = q + if away > scale || away == scale && q & 1 != 0 { x.signum() } else { 0 };
    T::try_from(q * scale).ok()
}

forms!(Isnan at real => bool: |_, [x]| x.is_nan());
forms!(Isfinite at real => bool: |_, [x]| x.is_finite());
forms!(Isinf at real => bool: |_, [x]| x.is_infinite());
forms!(Isnan at bool => bool, u8 => bool, i16 => bool, i32 => bool, i64 => bool: |_, [_x]| false);
forms!(Isfinite at bool => bool, u8 => bool, i16 => bool, i32 => bool, i64 => bool: |_, [_x]| true);
forms!(Isinf at bool => bool, u8 => bool, i16 => bool, i32 => bool, i64 => bool: |_, [_x]| false);
forms!(Isnan at Complex32 => bool, Complex64 => bool: |_, [x]| x.is_nan());
forms!(Isfinite at Complex32 => bool, Complex64 => bool: |_, [x]| x.re.is_finite() & x.im.is_finite());
forms!(Isinf at Complex32 => bool, Complex64 => bool: |_, [x]| x.re.is_infinite() | x.im.is_infinite());

forms!(Real at real: |_, [x]| Total(x));
forms!(Imag at real: |_, [_x]| Total(0.0));
forms!(Conj at real: |_, [x]| Total(x));
forms!(Trunc at real: |_, [x]| Total(x.trunc()));
forms!(Rint at real: |_, [x]| Total(x.round_ties_even()));
forms!(Real at bool, u8, i16, i32, i64: |_, [x]| x);
forms!(Imag at bool: |_, [_x]| false);
forms!(Imag at u8, i16, i32, i64: |_, [_x]| 0);
forms!(Conj at bool, u8, i16, i32, i64: |_, [x]| x);
forms!(Trunc at bool, u8, i16, i32, i64: |_, [x]| x);
forms!(Rint at bool, u8, i16, i32, i64: |_, [x]| x);
forms!(Conj at Complex32, Complex64: |_, [x]| Total(x.conj()));
forms!(Rint at Complex32, Complex64: |_, [x]| Total(num_complex::Complex::new(x.re.round_ties_even(), x.im.round_ties_even())));
forms!(Real at Complex32 => f32: |_, [x]| Total(x.re));
forms!(Real at Complex64 => f64: |_, [x]| Total(x.re));
forms!(Imag at Complex32 => f32: |_, [x]| Total(x.im));
forms!(Imag at Complex64 => f64: |_, [x]| Total(x.im));

forms!(Clip at real: |_, [x, lo, hi]| Total(crate::reduce::minimum(hi, crate::reduce::maximum(lo, x))));
forms!(Clip at u8, i16, i32, i64: |_, [x, lo, hi]| x.max(lo).min(hi));
forms!(Clip at bool: |_, [x, lo, hi]| (x | lo) & hi);

forms!(Choose at real: |_, [x, y, m]| Total(if m != 0.0 { x } else { y }));
forms!(Choose at Complex32, Complex64: |_, [x, y, m]| Total(if m.re != 0.0 { x } else { y }));
forms!(Choose at u8, i16, i32, i64: |_, [x, y, m]| if m != 0 { x } else { y });
forms!(Choose at bool: |_, [x, y, m]| (m & x) | (!m & y));
forms!(Choose at char: |_, [x, y, m]| if m == '\u{1}' { x } else { y });
