//! The magnitude, sign and angle of a number, as function objects. Their real forms are here, and `complex.rs` attaches their complex
//! ones.
use crate::{slice::Total, FUSED};

/// |x|, or the magnitude |z| of a complex z, which is real, as a function object.
#[derive(Clone, Copy, Debug)]
pub struct Abs;
/// The sign of x as -1 or 1, or z/|z| for a complex z, as a function object. A zero gives itself, and NaN gives NaN.
#[derive(Clone, Copy, Debug)]
pub struct Sign;
/// The angle from the positive real axis, from -π to π, as a function object. It is real for complex items too. A real x has angle 0,
/// or π where its sign bit is set, as the angle of x + 0i is.
#[derive(Clone, Copy, Debug)]
pub struct Angle;

forms!(Abs at real: |_, [x]| Total(x.abs()));
forms!(Sign at real: |_, [x]| Total(if x == 0.0 { x } else { x.signum() }));
forms!(Angle at real: |_, [x]| Total(crate::arc::atan2::<_, FUSED>(0.0, x)));
