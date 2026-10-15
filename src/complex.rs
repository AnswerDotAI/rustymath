//! Complex functions at `Complex<f32>` and `Complex<f64>`: the exponential and logarithm, square root and powers, the circle and hyperbolic functions and their inverses, ln Γ and Γ, binomial coefficients, the magnitude and angle, and the sign z/|z|. Each computes at its own width, and attaches its forms to the function object of the same name, which serves real arguments too.
//!
//! Their error counts in ulps of the result's magnitude: |f(z) - exact| is within 4 ulps of |exact|. A part much smaller than the other can carry most of that error. ln Γ counts in ulps of the larger of its magnitude and 2. The functions that exponentiate a logarithm count in ulps of their magnitude times one more than the exponent's magnitude, or than 2 where that is smaller: |w ln z| for powers, and |ln Γ| for Γ and binomial coefficients. Branch cuts follow C99's Annex G: the sign of a zero part chooses the side of a cut. The inverse functions use the formulas from Kahan's "Branch Cuts for Complex Elementary Functions" (1987), built on the complex square root, which keep their digits near the branch points.
//!
//! `exp` and the powers that compute through it give an exact direction at the float nearest a whole number of quarter turns. e^(iπ/2) is then exactly i, where the exponential of that float has a real part of about 6e-17. These results lie outside the error bound.
use crate::{
    Abs, Acos, Acosh, Angle, Arg, Asin, Asinh, Atan, Atan2, Atanh, Binary, Binom, Cos, Cosh, Exp, Factorial, Gamma, Hypot, Lgamma, Ln, Log1p, Logb, Pow, Sign,
    Sin, Sinh, Sqrt, Sqrt1mx2, Sqrt1px2, Sqrtx2m1, Tan, Tanh, Unary,
};
use crate::{Native, arc, exp as e, float::ln2, hyperbolic, ln as l, pow as p, special, trig};
use num_complex::Complex;
use std::mem::MaybeUninit;

#[inline(always)]
fn c<R>(re: R, im: R) -> Complex<R> { Complex::new(re, im) }

/// iz, a quarter turn of z.
#[inline(always)]
fn turned<R: Native>(z: Complex<R>) -> Complex<R> { c(-z.im, z.re) }

/// -iz, a quarter turn back.
#[inline(always)]
fn unturned<R: Native>(z: Complex<R>) -> Complex<R> { c(z.im, -z.re) }

/// π.
#[inline(always)]
fn pi<R: Native>() -> R { let (hi, _) = arc::half_pi::<R>(); hi + hi }

/// The whole number k nearest y/(π/2), and whether y is the float nearest kπ/2, with |k| up to 256.
#[inline(always)]
fn turns<R: Native>(y: R) -> (R, bool) {
    let (q, _) = arc::half_pi::<R>();
    let k = (y * R::narrow(std::f64::consts::FRAC_2_PI)).round();
    (k, (k.abs() <= R::narrow(256.0)) & (k * q == y))
}

/// cos y and sin y as exactly 1, 0 or -1, where y is the float nearest kπ/2 for a whole number k of quarter turns, with |k| up to 256.
#[inline(always)]
fn quarter<R: Native>(y: R) -> Option<(R, R)> {
    let (k, exact) = turns(y);
    if !exact { return None; }
    let (zero, one, m) = (R::zero(), R::one(), k - R::narrow(4.0) * (k * R::narrow(0.25)).floor());
    Some(if m == zero { (one, zero) } else if m == one { (zero, one) } else if m == one + one { (-one, zero) } else { (zero, -one) })
}

/// A part of e^z: `scale` times `d`, the cosine or sine of y. A part that `quarter` makes exactly 0 stays 0, even where e^x overflows, unless x is NaN.
#[inline(always)]
fn part<R: Native>(x: R, d: R, scale: R) -> R { if (d == R::zero()) & !x.is_nan() { d } else { scale * d } }

/// x / y for complex numbers, inline in the caller's loop, by num_complex's formula. Two real arguments divide as reals, so a real zero divisor gives ±∞ or NaN with an imaginary part of 0. Any other zero divisor gives NaN parts. |y|² overflows beyond about 1e154 at f64 and 1e19 at f32, and underflows below their inverses, where the quotient loses its digits. The decompositions' `Scalar::quotient` uses Smith's algorithm, which avoids that at the cost of speed in a vector loop.
#[inline(always)]
pub fn divide<R: Native>(x: Complex<R>, y: Complex<R>) -> Complex<R> {
    if (x.im == R::zero()) & (y.im == R::zero()) { c(x.re / y.re, R::zero()) } else { x / y }
}

/// |z| for the scalar functions. At f64 it is `hypot`. At f32 it adds the squares of z's parts in f64, where no square overflows or loses bits, and rounds the root once. That is within half an ulp, where `hypot` at f32's own width is within about one, and it costs less than `hypot`'s scaling.
#[inline(always)]
pub(crate) fn modulus<R: Native, const FUSED: bool>(x: R, y: R) -> R {
    let (a, b) = (x.wide(), y.wide());
    if R::WIDE { p::hypot::<R, FUSED>(x, y) } else { R::narrow((a * a + b * b).sqrt()) }
}

/// e^z = e^x (cos y + i sin y). A y that is the float nearest a whole number of quarter turns, up to 256 of them, gives an exact direction: e^(iπ/2) is i and e^(iπ) is -1. e^x/2 times 2 cos y keeps the result finite where e^x overflows but the result doesn't. A real z gives a real result. At x = -∞ the result is 0, and at x = ∞ with an infinite or NaN y it is ∞ + i NaN.
#[inline]
pub(crate) fn exp<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let half = e::exp_times::<R, FUSED>(x, -R::one());
    let (cy, sy) = quarter(y).unwrap_or_else(|| (trig::cos::<R, FUSED>(y), trig::sin::<R, FUSED>(y)));
    let r = c(part(x, cy + cy, half), if y == R::zero() { y } else { part(x, sy + sy, half) });
    let lost = x.is_infinite() & !y.is_finite();
    if lost { if x > R::zero() { c(x, R::nan()) } else { c(R::zero(), R::zero().copysign(y)) } } else { r }
}

/// e^(iy) for a real y, from cos y and sin y as `trig` computes them, as `exp` gives it: exact at the quarter turns that `quarter` finds, and with a zero y, sign included, as its imaginary part.
#[inline(always)]
fn cis_parts<R: Native>(y: R, cy: R, sy: R) -> Complex<R> { let (cy, sy) = quarter(y).unwrap_or((cy, sy)); c(cy, if y == R::zero() { y } else { sy }) }

/// Sine and cosine over slices, computed together. vForce implements it on macOS, in `vforce.rs`.
pub(crate) trait SinCos: Native {
    /// Appends sin y to `sin` and cos y to `cos` for each item y of `ys`.
    fn sin_cos_many(ys: &[Self], sin: &mut Vec<Self>, cos: &mut Vec<Self>);
    /// Writes cos y + i sin y into `out` for each item y of `ys`. `parts` holds buffers that calls can reuse.
    fn cis(ys: &[Self], out: &mut [MaybeUninit<Complex<Self>>], parts: &mut (Vec<Self>, Vec<Self>));
}

/// `SinCos` through the slice forms of `Sin` and `Cos`.
#[cfg(not(accelerate))]
impl<R: Native> SinCos for R where Sin: Unary<R, Out = R>, Cos: Unary<R, Out = R> {
    fn sin_cos_many(ys: &[R], sin: &mut Vec<R>, cos: &mut Vec<R>) {
        Sin.many(Arg::Many(ys), ys.len(), sin);
        Cos.many(Arg::Many(ys), ys.len(), cos);
    }

    fn cis(ys: &[R], out: &mut [MaybeUninit<Complex<R>>], (sin, cos): &mut (Vec<R>, Vec<R>)) {
        sin.clear();
        cos.clear();
        R::sin_cos_many(ys, sin, cos);
        for ((z, &cy), &sy) in out.iter_mut().zip(&*cos).zip(&*sin) { z.write(c(cy, sy)); }
    }
}

/// z scaled up exactly by 2^(2p) where it is so small that its halves or its magnitude would lose bits, and p, or z itself and 0. p is the number of bits in `R`'s significand.
#[inline(always)]
fn scaled<R: Native>(z: Complex<R>) -> (Complex<R>, R) {
    let (low, scale, p) = if R::WIDE { (1.0020841800044864e-292, 8.112963841460668e31, 53.0) } else { (9.860761315262648e-32, 2.81474976710656e14, 24.0) };
    if z.re.abs().max(z.im.abs()) < R::narrow(low) { (z * R::narrow(scale), R::narrow(p)) } else { (z, R::zero()) }
}

/// ln z = ln |z| + i arg z. Where |z| is near 1, ln |z| is ½ log1p((x-1)(x+1) + y²), which keeps its digits. A tiny z scales up first, so that |z| stays normal.
#[inline]
pub(crate) fn ln<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let (w, p) = scaled(z);
    let (hi, lo) = ln2::<R>();
    let h = modulus::<R, FUSED>(w.re, w.im);
    let near = (h > R::narrow(0.5)) & (h < R::narrow(2.0));
    let t = (x - R::one()).mla::<FUSED>(x + R::one(), y * y);
    let k = p + p;
    c(if near { l::log1p::<R, FUSED>(t) * R::narrow(0.5) } else { l::ln::<R, FUSED>(h) - k * hi - k * lo }, arc::atan2::<R, FUSED>(y, x))
}

/// ln(1 + w) for |w| up to 1. Below ½, ln |1 + w| is ½ log1p(u(2+u) + v²), which keeps its digits.
#[inline]
fn ln1p<R: Native, const FUSED: bool>(w: Complex<R>) -> Complex<R> {
    let (u, v) = (w.re, w.im);
    let small = modulus::<R, FUSED>(u, v) < R::narrow(0.5);
    let t = u.mla::<FUSED>(u + R::narrow(2.0), v * v);
    let re = if small { l::log1p::<R, FUSED>(t) * R::narrow(0.5) } else { l::ln::<R, FUSED>(modulus::<R, FUSED>(R::one() + u, v)) };
    c(re, arc::atan2::<R, FUSED>(v, R::one() + u))
}

/// √z, with a real part that is never negative. t = √((|x| + |z|)/2) is one part, and y/2t the other, which doesn't cancel. A tiny z scales up by 2^(2p) first, so that its halves stay normal, and its root back by 2^-p. Zero gives itself, and an infinite y gives ∞ + iy, even beside NaN.
#[inline]
pub(crate) fn sqrt<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let (w, p) = scaled(z);
    let h = modulus::<R, FUSED>(w.re, w.im);
    let t = w.re.abs().mla::<FUSED>(R::narrow(0.5), h * R::narrow(0.5)).sqrt();
    let (re, im) = if w.re >= R::zero() { (t, w.im / (t + t)) } else { (w.im.abs() / (t + t), t.copysign(w.im)) };
    let k = R::pow2(R::SHIFT - p);
    if y.is_infinite() { c(R::infinity(), y) } else if (x == R::zero()) & (y == R::zero()) { c(R::zero(), y) } else { c(re * k, im * k) }
}

/// Whether z^w takes a special case of `pow`: w = 0, z = 0 with Re w > 0, or a real z and w whose power is real, which is a z from 0 up or a whole w.
#[inline(always)]
fn pow_cases<R: Native>(z: Complex<R>, w: Complex<R>) -> (bool, bool, bool) {
    let unit = (w.re == R::zero()) & (w.im == R::zero());
    let zero = (z.re == R::zero()) & (z.im == R::zero()) & (w.re > R::zero());
    (unit, zero, (z.im == R::zero()) & (w.im == R::zero()) & ((z.re >= R::zero()) | (w.re == w.re.trunc())))
}

/// z^w = e^(w ln z). w = 0 gives 1, even for NaN, and z = 0 gives 0 where Re w > 0. A real z and w whose power is real give the real power, as `pow` gives it. Its error grows with |w ln z|, as the module doc gives it.
#[inline]
pub(crate) fn pow<R: Native, const FUSED: bool>(z: Complex<R>, w: Complex<R>) -> Complex<R> {
    let (unit, zero, real) = pow_cases(z, w);
    let r = p::pow::<R, FUSED>(z.re, w.re);
    let general = exp::<R, FUSED>(w * ln::<R, FUSED>(z));
    if unit { c(R::one(), R::zero()) } else if zero { c(R::zero(), R::zero()) } else if real { c(r, R::zero()) } else { general }
}

/// sinh z = sinh x cos y + i cosh x sin y. A real z gives a real result and an imaginary z an imaginary one, even where cosh x overflows or cos y is NaN.
#[inline]
pub(crate) fn sinh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let re = hyperbolic::sinh::<R, FUSED>(x) * trig::cos::<R, FUSED>(y);
    let im = hyperbolic::cosh::<R, FUSED>(x) * trig::sin::<R, FUSED>(y);
    c(if x == R::zero() { x } else { re }, if y == R::zero() { y } else { im })
}

/// cosh z = cosh x cos y + i sinh x sin y. A real or imaginary z gives an imaginary part of ±0.
#[inline]
pub(crate) fn cosh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let re = hyperbolic::cosh::<R, FUSED>(x) * trig::cos::<R, FUSED>(y);
    let im = hyperbolic::sinh::<R, FUSED>(x) * trig::sin::<R, FUSED>(y);
    c(re, if (x == R::zero()) | (y == R::zero()) { R::zero().copysign(x) * R::zero().copysign(y) } else { im })
}

/// tanh z by Kahan's formula: with t = tan y, s = sinh x and ρ = √(1 + s²) = cosh x, tanh z = ((1+t²)ρs + it)/(1 + (1+t²)s²). Beyond a magnitude where tanh x rounds to ±1, the result is ±1 + 4 sin y cos y e^(-2|x|) i, which can't overflow.
#[inline]
pub(crate) fn tanh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let (sy, cy) = (trig::sin::<R, FUSED>(y), trig::cos::<R, FUSED>(y));
    let t = sy / cy;
    let b = t.mla::<FUSED>(t, R::one());
    let s = hyperbolic::sinh::<R, FUSED>(x);
    let rho = s.mla::<FUSED>(s, R::one()).sqrt();
    let d = (b * s).mla::<FUSED>(s, R::one());
    let near = c(b * s * rho / d, t / d);
    let far = c(R::one().copysign(x), e::exp_times::<R, FUSED>(R::narrow(-2.0) * x.abs(), R::narrow(2.0)) * (sy * cy));
    let r = if x.abs() > tanh_limit::<R>() { far } else { near };
    if x.is_infinite() & !y.is_finite() { c(r.re, R::zero().copysign(y)) } else { r }
}

/// The `|x|` beyond which `tanh` takes its formula for a large `x`.
fn tanh_limit<R: Native>() -> R { R::narrow(if R::WIDE { 22.0 } else { 10.0 }) }

/// sin z = -i sinh(iz).
#[inline]
pub(crate) fn sin<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { unturned(sinh::<R, FUSED>(turned(z))) }

/// cos z = cosh(iz).
#[inline]
pub(crate) fn cos<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { cosh::<R, FUSED>(turned(z)) }

/// tan z = -i tanh(iz).
#[inline]
pub(crate) fn tan<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { unturned(tanh::<R, FUSED>(turned(z))) }

/// √(1 - z) and √(1 + z), whose products give the inverse circle functions.
#[inline(always)]
fn roots<R: Native, const FUSED: bool>(z: Complex<R>) -> (Complex<R>, Complex<R>) {
    (sqrt::<R, FUSED>(c(R::one() - z.re, -z.im)), sqrt::<R, FUSED>(c(R::one() + z.re, z.im)))
}

/// √(z - 1) and √(z + 1), whose products give acosh and √(z² - 1).
#[inline(always)]
fn shifted_roots<R: Native, const FUSED: bool>(z: Complex<R>) -> (Complex<R>, Complex<R>) {
    (sqrt::<R, FUSED>(c(z.re - R::one(), z.im)), sqrt::<R, FUSED>(c(z.re + R::one(), z.im)))
}

/// `r`, a square root of a number whose imaginary part is 2xy, with its smaller part recomputed as xy divided by its larger part. A product of two square roots, as `sqrt1mx2` and `sqrtx2m1` form, loses the smaller part where x is too small to change 1 ± x. The product xy keeps it. The larger of x and y is divided first, so a huge x with a tiny y doesn't underflow. `r` stays as it is where x or y is 0, to keep the signs of zeros on the branch cuts, and where a part isn't finite.
#[inline(always)]
fn balanced<R: Native>(r: Complex<R>, x: R, y: R) -> Complex<R> {
    let zero = R::zero();
    if (x == zero) | (y == zero) | !(r.re.is_finite() & r.im.is_finite() & x.is_finite() & y.is_finite()) { return r; }
    let (hi, lo) = if x.abs() >= y.abs() { (x, y) } else { (y, x) };
    if r.re.abs() >= r.im.abs() { if r.re == zero { r } else { c(r.re, hi / r.re * lo) } } else { c(hi / r.im * lo, r.im) }
}

/// √(1 - z²) as √(1 - z) √(1 + z), whose real part is never negative. Neither factor loses digits near z = ±1, or overflows where z² would. [`balanced`] restores the smaller part where z's real part is too small to change 1.
#[inline]
pub(crate) fn sqrt1mx2<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { let (a, b) = roots::<R, FUSED>(z); balanced(a * b, -z.re, z.im) }

/// √(1 + z²) as `sqrt1mx2` at iz, because 1 + z² is 1 - (iz)². Its real part is never negative. Neither factor overflows where z² would.
#[inline]
pub(crate) fn sqrt1px2<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { sqrt1mx2::<R, FUSED>(turned(z)) }

/// √(z - 1) √(z + 1), the continuation of the real `sqrtx2m1` off the real line. Neither factor loses digits near z = ±1, or overflows where z² would. [`balanced`] restores the smaller part where z's real part is too small to change 1.
#[inline]
pub(crate) fn sqrtx2m1<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { let (a, b) = shifted_roots::<R, FUSED>(z); balanced(a * b, z.re, z.im) }

/// asin z, by Kahan's formula: with a = √(1 - z) and b = √(1 + z), Re asin z = atan(x / Re(ab)) and Im asin z = asinh(Im(conj(a) b)).
#[inline]
pub(crate) fn asin<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = roots::<R, FUSED>(z);
    let re = arc::atan2::<R, FUSED>(z.re, a.re.mla::<FUSED>(b.re, -(a.im * b.im)));
    c(re, hyperbolic::asinh::<R, FUSED>(a.re.mla::<FUSED>(b.im, -(a.im * b.re))))
}

/// acos z, by Kahan's formula: with a = √(1 - z) and b = √(1 + z), Re acos z = 2 atan(Re a / Re b) and Im acos z = asinh(Im(conj(b) a)).
#[inline]
pub(crate) fn acos<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = roots::<R, FUSED>(z);
    let re = arc::atan2::<R, FUSED>(a.re, b.re);
    c(re + re, hyperbolic::asinh::<R, FUSED>(b.re.mla::<FUSED>(a.im, -(b.im * a.re))))
}

/// acosh z, by Kahan's formula: with a = √(z - 1) and b = √(z + 1), Re acosh z = asinh(Re(conj(a) b)) and Im acosh z = 2 atan(Im a / Re b).
#[inline]
pub(crate) fn acosh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = shifted_roots::<R, FUSED>(z);
    let im = arc::atan2::<R, FUSED>(a.im, b.re);
    c(hyperbolic::asinh::<R, FUSED>(a.re.mla::<FUSED>(b.re, a.im * b.im)), im + im)
}

/// asinh z = -i asin(iz).
#[inline]
pub(crate) fn asinh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { unturned(asin::<R, FUSED>(turned(z))) }

/// atanh z, with Re atanh z = ¼ log1p(4x / ((1-x)² + y²)) and Im atanh z = ½ atan2(2y, (1-x)(1+x) - y²). atanh is odd, so it computes at |x|, where the argument of log1p is from 0 up and loses nothing. Where the squares would overflow, Re atanh z is x/|z|² and Im atanh z is ±π/2. An infinite part gives ±0 ± iπ/2.
#[inline]
pub(crate) fn atanh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let s = R::one().copysign(z.re);
    let (x, y) = (z.re.abs(), z.im * s);
    let (m, quarter, half) = (R::one() - x, R::narrow(0.25), R::narrow(0.5));
    let re = l::log1p::<R, FUSED>(R::narrow(4.0) * x / m.mla::<FUSED>(m, y * y)) * quarter;
    let im = arc::atan2::<R, FUSED>(y + y, m.mla::<FUSED>(R::one() + x, -(y * y))) * half;
    let h = modulus::<R, FUSED>(x, y);
    let (right, _) = arc::half_pi::<R>();
    let huge = h > atanh_huge::<R>();
    let (re, im) = if x.is_infinite() | y.is_infinite() { (R::zero(), right.copysign(y)) } else if huge { (x / h / h, right.copysign(y)) } else { (re, im) };
    c(re * s, im * s)
}

/// The modulus beyond which `atanh` takes its formula for a huge `z`.
fn atanh_huge<R: Native>() -> R { R::narrow(if R::WIDE { 1e150 } else { 1e18 }) }

/// atan z = -i atanh(iz).
#[inline]
pub(crate) fn atan<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { unturned(atanh::<R, FUSED>(turned(z))) }

/// (sin πx, cos πx), exact at whole and half x. x = n/2 + r with n whole and |r| up to ¼, and n mod 4 picks the quadrant.
#[inline]
fn sincospi<R: Native, const FUSED: bool>(x: R) -> (R, R) {
    let n = (x + x).round();
    let a = (x - n * R::narrow(0.5)) * pi::<R>();
    let (s, k) = (trig::sin::<R, FUSED>(a), trig::cos::<R, FUSED>(a));
    let q = n - R::narrow(4.0) * (n * R::narrow(0.25)).floor();
    if q == R::zero() { (s, k) } else if q == R::one() { (k, -s) } else if q == R::narrow(2.0) { (-s, -k) } else { (-k, s) }
}

/// Whether z is a pole of Γ: a whole number from 0 down.
#[inline(always)]
fn pole<R: Native>(z: Complex<R>) -> bool { (z.im == R::zero()) & (z.re <= R::zero()) & (z.re == z.re.trunc()) }

/// ln Γ(w) by Stirling's series, for |w| from 7 up, away from the negative real axis: (w - ½) ln w - w + ½ ln 2π + (1/w) S(1/w²), with the ten terms of `special::STIRLING`.
#[inline]
fn stirling<R: Native, const FUSED: bool>(w: Complex<R>) -> Complex<R> {
    let n = w.re.mla::<FUSED>(w.re, w.im * w.im);
    let v = c(w.re / n, -w.im / n);
    let v2 = v * v;
    let mut s = c(R::narrow(special::STIRLING[0]), R::zero());
    for &k in &special::STIRLING[1..] { s = s * v2 + c(R::narrow(k), R::zero()); }
    c(w.re - R::narrow(0.5), w.im) * ln::<R, FUSED>(w) - w + c(R::narrow(special::HALF_LN_2PI), R::zero()) + v * s
}

/// ln Γ(w) for Re w from 0.1 up: `stirling` itself where Re w or |Im w| is 7 or more, and otherwise ln Γ(w + n) - ln(w(w+1)…(w+n-1)) with n the steps that bring Re w to 7. Above the real axis, the product's argument grows with each factor. Each time it passes π the product's imaginary part turns negative, and the principal logarithm drops 2π, which the result adds back. Below the axis it is the conjugate of the result above.
#[inline]
fn shifted<R: Native, const FUSED: bool>(w: Complex<R>) -> Complex<R> {
    let below = w.im.is_sign_negative();
    let mut z = if below { w.conj() } else { w };
    let seven = R::narrow(7.0);
    let r = if (z.re >= seven) | (z.im >= seven) { stirling::<R, FUSED>(z) } else {
        let (mut product, mut turns, mut negative) = (z, R::zero(), false);
        z.re = z.re + R::one();
        while z.re < seven {
            product = product * z;
            let now = product.im.is_sign_negative();
            if now & !negative { turns = turns + R::one(); }
            negative = now;
            z.re = z.re + R::one();
        }
        stirling::<R, FUSED>(z) - ln::<R, FUSED>(product) - c(R::zero(), (pi::<R>() + pi::<R>()) * turns)
    };
    if below { r.conj() } else { r }
}

/// ln π - ln sin πu, for Im u from 0 up. sin πu = (i/2) e^(-iπu) (1 - e^(2πiu)), so this is ln 2π - πy + iπ(x - ½) - ln(1 - e^(2πiu)), which stays finite where sin πu overflows. The imaginary part isn't reduced, which keeps it continuous over the half-plane.
#[inline]
fn reflected<R: Native, const FUSED: bool>(u: Complex<R>) -> Complex<R> {
    let (x, y) = (u.re, u.im);
    let (s, k) = sincospi::<R, FUSED>(x + x);
    let m = e::exp::<R, FUSED>(R::narrow(-2.0) * pi::<R>() * y);
    let t = ln1p::<R, FUSED>(c(-(m * k), -(m * s)));
    c((R::narrow(1.8378770664093453) - pi::<R>() * y) - t.re, pi::<R>() * (x - R::narrow(0.5)) - t.im)
}

/// ln Γ(z), on the branch that is real on the positive real axis and continuous everywhere but across the negative real axis, as SciPy's and mpmath's `loggamma` give it. Below Re z = 0.1 it reflects: ln Γ(z) = ln π - ln sin πz - ln Γ(1 - z), and the lower half-plane gives the conjugate of the upper. Elsewhere it is `shifted`. Poles and non-finite arguments give NaN. Near the zeros at 1 and 2 the error counts in ulps of 2, not of the result.
#[inline]
pub(crate) fn lgamma<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    if !(z.re.is_finite() & z.im.is_finite()) | pole(z) { return c(R::nan(), R::nan()); }
    if z.re >= R::narrow(0.1) { return shifted::<R, FUSED>(z); }
    let below = z.im.is_sign_negative();
    let u = if below { z.conj() } else { z };
    let r = reflected::<R, FUSED>(u) - shifted::<R, FUSED>(c(R::one() - u.re, -u.im));
    if below { r.conj() } else { r }
}

/// Γ(z) = e^(ln Γ(z)). Its error grows with |ln Γ(z)|, as the module doc gives it. The poles give NaN.
#[inline]
pub(crate) fn gamma<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { exp::<R, FUSED>(lgamma::<R, FUSED>(z)) }
/// Γ(z + 1).
#[inline]
pub(crate) fn factorial<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { gamma::<R, FUSED>(c(z.re + R::one(), z.im)) }

/// The binomial coefficient Γ(n+1) / (Γ(k+1) Γ(n-k+1)), as e^(ln Γ(n+1) - ln Γ(k+1) - ln Γ(n-k+1)). A pole in the denominator gives 0, and one in the numerator NaN.
#[inline]
pub(crate) fn binom<R: Native, const FUSED: bool>(n: Complex<R>, k: Complex<R>) -> Complex<R> {
    let one = c(R::one(), R::zero());
    let (a, b, d) = (n + one, k + one, n - k + one);
    let r = exp::<R, FUSED>(lgamma::<R, FUSED>(a) - lgamma::<R, FUSED>(b) - lgamma::<R, FUSED>(d));
    if !pole(a) & (pole(b) | pole(d)) { c(R::zero(), R::zero()) } else { r }
}

/// z/|z|, of magnitude 1 in the direction of z. Zero gives itself, and an infinity the direction of its infinite parts.
#[inline]
pub(crate) fn sign<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let unit = |v: R| (if v.is_infinite() { R::one() } else { R::zero() }).copysign(v);
    let w = if z.re.is_infinite() | z.im.is_infinite() { c(unit(z.re), unit(z.im)) } else { z };
    let h = modulus::<R, FUSED>(w.re, w.im);
    if h == R::zero() { z } else { c(w.re / h, w.im / h) }
}

/// Whether `scaled` scales z up, which the fast formulas of `ln` and `pow` leave to the scalar functions.
#[inline(always)]
fn tiny<R: Native>(z: Complex<R>) -> bool { scaled(z).1 != R::zero() }

/// Whether both parts of z are finite.
#[inline(always)]
fn finite<R: Native>(z: Complex<R>) -> bool { z.re.is_finite() & z.im.is_finite() }

/// e^z's parts, computed through real slice forms from slices of z's parts, as `exp` combines them. Where e^x overflows, or x is -∞ beside an infinite or NaN y, a part is non-finite here, and the slice form gives the item to `exp`.
fn exp_parts<R: SinCos>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Exp: Unary<R, Out = R> {
    let (n, mut e, mut cy, mut sy) = (x.len(), Vec::new(), Vec::new(), Vec::new());
    Exp.many(Arg::Many(x), n, &mut e);
    R::sin_cos_many(y, &mut sy, &mut cy);
    re.extend(x.iter().zip(y).zip(e.iter().zip(&cy)).map(|((&x, &y), (&e, &c))| part(x, quarter(y).map_or(c, |q| q.0), e)));
    im.extend(x.iter().zip(y).zip(e.iter().zip(&sy)).map(|((&x, &y), (&e, &s))| if y == R::zero() { y } else { part(x, quarter(y).map_or(s, |q| q.1), e) }));
}

/// sinh z's parts, or with `cosh` cosh z's, computed through real slice forms from slices of z's parts, as `sinh` and `cosh` combine them.
fn hyperbolic_parts<R: SinCos>(x: &[R], y: &[R], cosh: bool, re: &mut Vec<R>, im: &mut Vec<R>) where Sinh: Unary<R, Out = R>, Cosh: Unary<R, Out = R> {
    let (n, mut sh, mut ch, mut s, mut c) = (x.len(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    Sinh.many(Arg::Many(x), n, &mut sh);
    Cosh.many(Arg::Many(x), n, &mut ch);
    R::sin_cos_many(y, &mut s, &mut c);
    let zero = R::zero();
    if cosh {
        re.extend(ch.iter().zip(&c).map(|(&ch, &c)| ch * c));
        im.extend(x.iter().zip(y).zip(sh.iter().zip(&s)).map(|((&x, &y), (&sh, &s))| {
            if (x == zero) | (y == zero) { zero.copysign(x) * zero.copysign(y) } else { sh * s }
        }));
    } else {
        re.extend(x.iter().zip(&sh).zip(&c).map(|((&x, &sh), &c)| if x == zero { x } else { sh * c }));
        im.extend(y.iter().zip(&ch).zip(&s).map(|((&y, &ch), &s)| if y == zero { y } else { ch * s }));
    }
}

/// ln z's parts, computed through real slice forms from slices of z's parts, as `ln` combines them for a z that isn't tiny.
fn ln_parts<R: Native>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Hypot: Binary<R, Out = R>, Ln: Unary<R, Out = R>, Log1p: Unary<R, Out = R>,
    Atan2: Binary<R, Out = R> {
    let (n, mut h, mut lh, mut lt) = (x.len(), Vec::new(), Vec::new(), Vec::new());
    Hypot.many(Arg::Many(x), Arg::Many(y), n, &mut h);
    Ln.many(Arg::Many(&h), n, &mut lh);
    let one = R::one();
    let t: Vec<R> = x.iter().zip(y).map(|(&x, &y)| (x - one).mla::<{ crate::FUSED }>(x + one, y * y)).collect();
    Log1p.many(Arg::Many(&t), n, &mut lt);
    let near = |h: R| (h > R::narrow(0.5)) & (h < R::narrow(2.0));
    re.extend(h.iter().zip(&lh).zip(&lt).map(|((&h, &lh), &lt)| if near(h) { lt * R::narrow(0.5) } else { lh }));
    Atan2.many(Arg::Many(y), Arg::Many(x), n, im);
}

/// The moduli of `x + iy`, as `modulus` computes them: at f64 through `Hypot`'s slice form, at f32 from squares summed in f64.
fn moduli<R: Native>(x: &[R], y: &[R], out: &mut Vec<R>) where Hypot: Binary<R, Out = R> {
    if R::WIDE { Hypot.many(Arg::Many(x), Arg::Many(y), x.len(), out); }
    else { out.extend(x.iter().zip(y).map(|(&x, &y)| modulus::<R, { crate::FUSED }>(x, y))) }
}

/// The square roots of `x + iy` through real slice forms, as `sqrt` computes them where `rough_root` is false.
fn sqrt_parts<R: Native>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Hypot: Binary<R, Out = R> {
    let (mut h, half, zero) = (Vec::new(), R::narrow(0.5), R::zero());
    moduli(x, y, &mut h);
    let t: Vec<R> = x.iter().zip(&h).map(|(&x, &h)| x.abs().mla::<{ crate::FUSED }>(half, h * half).sqrt()).collect();
    // `sqrt` divides |y| or y by 2t, whichever its branch needs. |y|/2t is |y/2t| exactly, so one division serves both branches.
    let q: Vec<R> = y.iter().zip(&t).map(|(&y, &t)| y / (t + t)).collect();
    re.extend(x.iter().zip(&t).zip(&q).map(|((&x, &t), &q)| if x >= zero { t } else { q.abs() }));
    im.extend(x.iter().zip(y).zip(t.iter().zip(&q)).map(|((&x, &y), (&t, &q))| if x >= zero { q } else { t.copysign(y) }));
}

/// Whether `sqrt` takes a path for `z` that `sqrt_parts` leaves out: a tiny `z`, which it scales first, an infinite imaginary part, or zero.
fn rough_root<R: Native>(z: Complex<R>) -> bool { tiny(z) | z.im.is_infinite() | ((z.re == R::zero()) & (z.im == R::zero())) }

/// The square roots of `a` and of `b`, each given and returned as its real and imaginary parts, through `sqrt_parts`.
fn root_pair<R: Native>([ax, ay]: [&[R]; 2], [bx, by]: [&[R]; 2]) -> [(Vec<R>, Vec<R>); 2] where Hypot: Binary<R, Out = R> {
    let mut roots = [(Vec::new(), Vec::new()), (Vec::new(), Vec::new())];
    let [a, b] = &mut roots;
    sqrt_parts(ax, ay, &mut a.0, &mut a.1);
    sqrt_parts(bx, by, &mut b.0, &mut b.1);
    roots
}

/// `roots`' square roots of `1 - z` and `1 + z`, for `z` = `x + iy`.
fn roots_parts<R: Native>(x: &[R], y: &[R]) -> [(Vec<R>, Vec<R>); 2] where Hypot: Binary<R, Out = R> {
    let one = R::one();
    let below: Vec<R> = x.iter().map(|&x| one - x).collect();
    let minus_y: Vec<R> = y.iter().map(|&y| -y).collect();
    let above: Vec<R> = x.iter().map(|&x| one + x).collect();
    root_pair([&below, &minus_y], [&above, y])
}

/// Whether `roots` takes a path for `z` that `roots_parts` leaves out.
fn rough_roots<R: Native>(z: Complex<R>) -> bool { rough_root(c(R::one() - z.re, -z.im)) | rough_root(c(R::one() + z.re, z.im)) }

/// `shifted_roots`' square roots of `z - 1` and `z + 1`, for `z` = `x + iy`.
fn shifted_roots_parts<R: Native>(x: &[R], y: &[R]) -> [(Vec<R>, Vec<R>); 2] where Hypot: Binary<R, Out = R> {
    let one = R::one();
    let below: Vec<R> = x.iter().map(|&x| x - one).collect();
    let above: Vec<R> = x.iter().map(|&x| x + one).collect();
    root_pair([&below, y], [&above, y])
}

/// Whether `shifted_roots` takes a path for `z` that `shifted_roots_parts` leaves out.
fn rough_shifted_roots<R: Native>(z: Complex<R>) -> bool { rough_root(c(z.re - R::one(), z.im)) | rough_root(c(z.re + R::one(), z.im)) }

/// Appends the products `a × b`, computed as `Complex`'s multiplication computes them, of two complex slices given as their parts.
fn times_parts<R: Native>((ar, ai): &(Vec<R>, Vec<R>), (br, bi): &(Vec<R>, Vec<R>), re: &mut Vec<R>, im: &mut Vec<R>) {
    re.extend(ar.iter().zip(ai).zip(br.iter().zip(bi)).map(|((&ar, &ai), (&br, &bi))| ar * br - ai * bi));
    im.extend(ar.iter().zip(ai).zip(br.iter().zip(bi)).map(|((&ar, &ai), (&br, &bi))| ar * bi + ai * br));
}

/// Appends, for each `z` = `x + iy`, the product of the two square roots that `roots` gives, with its smaller part restored by [`balanced`]. `negate` negates x for `balanced`, as √(1 - z²) needs.
fn root_products_parts<R: Native>(roots: impl Fn(&[R], &[R]) -> [(Vec<R>, Vec<R>); 2], negate: bool, x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) {
    let [a, b] = roots(x, y);
    let start = re.len();
    times_parts(&a, &b, re, im);
    for (i, (&x, &y)) in x.iter().zip(y).enumerate() {
        let r = balanced(c(re[start + i], im[start + i]), if negate { -x } else { x }, y);
        (re[start + i], im[start + i]) = (r.re, r.im);
    }
}

/// Appends √(1 - z²) for each `z` = `x + iy`, as `sqrt1mx2` computes it.
fn sqrt1mx2_parts<R: Native>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Hypot: Binary<R, Out = R> { root_products_parts(roots_parts, true, x, y, re, im) }

/// Appends `g` of `turned(z)`, which is iz, for each `z` = `x + iy`, where `g` appends its results' parts.
fn turned_parts<R: Native>(x: &[R], y: &[R], g: impl Fn(&[R], &[R], &mut Vec<R>, &mut Vec<R>), re: &mut Vec<R>, im: &mut Vec<R>) {
    let minus_y: Vec<R> = y.iter().map(|&v| -v).collect();
    g(&minus_y, x, re, im);
}

/// [`turned_parts`], with each result turned back by `unturned`.
fn unturned_parts<R: Native>(x: &[R], y: &[R], g: impl Fn(&[R], &[R], &mut Vec<R>, &mut Vec<R>), re: &mut Vec<R>, im: &mut Vec<R>) {
    let (mut a, mut b) = (Vec::new(), Vec::new());
    turned_parts(x, y, g, &mut a, &mut b);
    re.extend(b);
    im.extend(a.iter().map(|&v| -v));
}

/// `asin` of `x + iy` through real slice forms, where `rough_roots` is false.
fn asin_parts<R: Native>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Hypot: Binary<R, Out = R>, Atan2: Binary<R, Out = R>,
    Asinh: Unary<R, Out = R> {
    let n = x.len();
    let [(ar, ai), (br, bi)] = roots_parts(x, y);
    let d: Vec<R> = ar.iter().zip(&ai).zip(br.iter().zip(&bi)).map(|((&ar, &ai), (&br, &bi))| ar.mla::<{ crate::FUSED }>(br, -(ai * bi))).collect();
    let s: Vec<R> = ar.iter().zip(&ai).zip(br.iter().zip(&bi)).map(|((&ar, &ai), (&br, &bi))| ar.mla::<{ crate::FUSED }>(bi, -(ai * br))).collect();
    Atan2.many(Arg::Many(x), Arg::Many(&d), n, re);
    Asinh.many(Arg::Many(&s), n, im);
}

/// `acos` of `x + iy` through real slice forms, where `rough_roots` is false.
fn acos_parts<R: Native>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Hypot: Binary<R, Out = R>, Atan2: Binary<R, Out = R>,
    Asinh: Unary<R, Out = R> {
    let (n, mut half) = (x.len(), Vec::new());
    let [(ar, ai), (br, bi)] = roots_parts(x, y);
    let s: Vec<R> = br.iter().zip(&bi).zip(ar.iter().zip(&ai)).map(|((&br, &bi), (&ar, &ai))| br.mla::<{ crate::FUSED }>(ai, -(bi * ar))).collect();
    Atan2.many(Arg::Many(&ar), Arg::Many(&br), n, &mut half);
    re.extend(half.iter().map(|&h| h + h));
    Asinh.many(Arg::Many(&s), n, im);
}

/// `acosh` of `x + iy` through real slice forms, where `rough_shifted_roots` is false.
fn acosh_parts<R: Native>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Hypot: Binary<R, Out = R>, Atan2: Binary<R, Out = R>,
    Asinh: Unary<R, Out = R> {
    let (n, mut half) = (x.len(), Vec::new());
    let [(ar, ai), (br, bi)] = shifted_roots_parts(x, y);
    let s: Vec<R> = ar.iter().zip(&ai).zip(br.iter().zip(&bi)).map(|((&ar, &ai), (&br, &bi))| ar.mla::<{ crate::FUSED }>(br, ai * bi)).collect();
    Asinh.many(Arg::Many(&s), n, re);
    Atan2.many(Arg::Many(&ai), Arg::Many(&br), n, &mut half);
    im.extend(half.iter().map(|&h| h + h));
}

/// `tanh` of `x + iy` through real slice forms, where `rough_tanh` is false.
fn tanh_parts<R: SinCos>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Sinh: Unary<R, Out = R> {
    let (n, one, mut sy, mut cy, mut s) = (x.len(), R::one(), Vec::new(), Vec::new(), Vec::new());
    R::sin_cos_many(y, &mut sy, &mut cy);
    Sinh.many(Arg::Many(x), n, &mut s);
    let t: Vec<R> = sy.iter().zip(&cy).map(|(&sy, &cy)| sy / cy).collect();
    let b: Vec<R> = t.iter().map(|&t| t.mla::<{ crate::FUSED }>(t, one)).collect();
    let d: Vec<R> = b.iter().zip(&s).map(|(&b, &s)| (b * s).mla::<{ crate::FUSED }>(s, one)).collect();
    re.extend(b.iter().zip(&s).zip(&d).map(|((&b, &s), &d)| b * s * s.mla::<{ crate::FUSED }>(s, one).sqrt() / d));
    im.extend(t.iter().zip(&d).map(|(&t, &d)| t / d));
}

/// Whether `tanh` takes a path for `z` that `tanh_parts` leaves out: `|x|` beyond `tanh_limit`, or a part that isn't finite.
fn rough_tanh<R: Native>(z: Complex<R>) -> bool { !(z.re.abs() <= tanh_limit::<R>()) | !z.im.is_finite() }

/// `atanh` of `x + iy` through real slice forms, where `rough_atanh` is false.
fn atanh_parts<R: Native>(x: &[R], y: &[R], re: &mut Vec<R>, im: &mut Vec<R>) where Log1p: Unary<R, Out = R>, Atan2: Binary<R, Out = R> {
    let (n, one, mut lr, mut li) = (x.len(), R::one(), Vec::new(), Vec::new());
    let s: Vec<R> = x.iter().map(|&x| one.copysign(x)).collect();
    let ax: Vec<R> = x.iter().map(|&x| x.abs()).collect();
    let ay: Vec<R> = y.iter().zip(&s).map(|(&y, &s)| y * s).collect();
    let q: Vec<R> = ax.iter().zip(&ay).map(|(&x, &y)| R::narrow(4.0) * x / (one - x).mla::<{ crate::FUSED }>(one - x, y * y)).collect();
    let u: Vec<R> = ay.iter().map(|&y| y + y).collect();
    let v: Vec<R> = ax.iter().zip(&ay).map(|(&x, &y)| (one - x).mla::<{ crate::FUSED }>(one + x, -(y * y))).collect();
    Log1p.many(Arg::Many(&q), n, &mut lr);
    Atan2.many(Arg::Many(&u), Arg::Many(&v), n, &mut li);
    re.extend(lr.iter().zip(&s).map(|(&r, &s)| r * R::narrow(0.25) * s));
    im.extend(li.iter().zip(&s).map(|(&i, &s)| i * R::narrow(0.5) * s));
}

/// Whether `atanh` may take a path for `z` that `atanh_parts` leaves out. It tests for a part that isn't finite, or a part above half of `atanh_huge`. Every `z` with a modulus above `atanh_huge` has such a part.
fn rough_atanh<R: Native>(z: Complex<R>) -> bool { !z.re.is_finite() | !z.im.is_finite() | (z.re.abs().max(z.im.abs()) > atanh_huge::<R>() * R::narrow(0.5)) }

/// Attaches each function of one complex argument to its object, at both widths, with the slice form that runs it on each item.
macro_rules! unary {
    ($($object:ident: $f:ident;)+) => {$(
        forms!($object at Complex<f32>, Complex<f64>: |_, [z]| $f::<_, { crate::FUSED }>(z));
    )+};
}
unary! {
    Lgamma: lgamma; Gamma: gamma; Factorial: factorial; Sign: sign; Sqrt: sqrt;
}
// The functions below have slice forms computed through real slice forms, by `split!`. cis z is the exponential of iz, sin z and tan z are sinh and tanh of iz turned back, and cos z is cosh of iz, as the scalar forms compute them.
forms!(Exp at Complex<f32>, Complex<f64>: |_, [z]| exp::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| exp_parts(x, y, re, im), |_, r| !finite(r), |[z]| exp::<_, { crate::FUSED }>(z)));
/// e^(iz), the cis of z, as a function object, which is cos z + i sin z for a real z. It takes real and complex items. On reals, its slice form runs vForce's `vvcosisin` on macOS, and `Cos`'s and `Sin`'s elsewhere.
#[derive(Clone, Copy, Debug)]
pub struct Cis;
forms!(Cis at Complex<f32>, Complex<f64>: |_, [z]| exp::<_, { crate::FUSED }>(turned(z)),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| turned_parts(x, y, exp_parts, re, im), |_, r| !finite(r),
        |[z]| exp::<_, { crate::FUSED }>(turned(z))));
forms!(Cis at f32 => Complex<f32>, f64 => Complex<f64>: |_, [y]| exp::<_, { crate::FUSED }>(c(0.0, y)),
many |_, [y], n, out| {
    let (mut fill, mut parts, mut ok) = (Vec::new(), (Vec::new(), Vec::new()), true);
    for (from, slots) in (0..n).step_by(crate::CHUNK).zip(out.chunks_mut(crate::CHUNK)) {
        let m = slots.len();
        let ys = match y.part(from, m) {
            Arg::One(v) => {
                fill.clear();
                fill.resize(m, v);
                &fill[..]
            }
            Arg::Many(ys) => ys,
        };
        SinCos::cis(ys, slots, &mut parts);
        // An infinite y fails, as `Cos` and `Sin` judge it. Only quarter turns, zero among them, need `cis_parts`. They're rare.
        let (infinite, turned) = ys.iter().fold((false, false), |(i, t), &y| (i | y.is_infinite(), t | turns(y).1));
        ok &= !infinite;
        if turned {
            for (z, &y) in crate::slice::written(slots).iter_mut().zip(ys) { if turns(y).1 { *z = cis_parts(y, z.re, z.im) } }
        }
    }
    ok
});
forms!(Sinh at Complex<f32>, Complex<f64>: |_, [z]| sinh::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| hyperbolic_parts(x, y, false, re, im), |_, _| false, |[z]| sinh::<_, { crate::FUSED }>(z)));
forms!(Cosh at Complex<f32>, Complex<f64>: |_, [z]| cosh::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| hyperbolic_parts(x, y, true, re, im), |_, _| false, |[z]| cosh::<_, { crate::FUSED }>(z)));
forms!(Sin at Complex<f32>, Complex<f64>: |_, [z]| sin::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| unturned_parts(x, y, |x, y, re, im| hyperbolic_parts(x, y, false, re, im), re, im),
        |_, _| false, |[z]| sin::<_, { crate::FUSED }>(z)));
forms!(Cos at Complex<f32>, Complex<f64>: |_, [z]| cos::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| turned_parts(x, y, |x, y, re, im| hyperbolic_parts(x, y, true, re, im), re, im),
        |_, _| false, |[z]| cos::<_, { crate::FUSED }>(z)));
forms!(Tan at Complex<f32>, Complex<f64>: |_, [z]| tan::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| unturned_parts(x, y, tanh_parts, re, im), |[z], _| rough_tanh(turned(z)), |[z]| tan::<_, { crate::FUSED }>(z)));
forms!(Tanh at Complex<f32>, Complex<f64>: |_, [z]| tanh::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| tanh_parts(x, y, re, im), |[z], _| rough_tanh(z), |[z]| tanh::<_, { crate::FUSED }>(z)));
forms!(Asin at Complex<f32>, Complex<f64>: |_, [z]| asin::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| asin_parts(x, y, re, im), |[z], _| rough_roots(z), |[z]| asin::<_, { crate::FUSED }>(z)));
forms!(Asinh at Complex<f32>, Complex<f64>: |_, [z]| asinh::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| unturned_parts(x, y, asin_parts, re, im), |[z], _| rough_roots(turned(z)), |[z]| asinh::<_, { crate::FUSED }>(z)));
forms!(Acos at Complex<f32>, Complex<f64>: |_, [z]| acos::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| acos_parts(x, y, re, im), |[z], _| rough_roots(z), |[z]| acos::<_, { crate::FUSED }>(z)));
forms!(Acosh at Complex<f32>, Complex<f64>: |_, [z]| acosh::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| acosh_parts(x, y, re, im), |[z], _| rough_shifted_roots(z), |[z]| acosh::<_, { crate::FUSED }>(z)));
forms!(Atanh at Complex<f32>, Complex<f64>: |_, [z]| atanh::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| atanh_parts(x, y, re, im), |[z], _| rough_atanh(z), |[z]| atanh::<_, { crate::FUSED }>(z)));
forms!(Atan at Complex<f32>, Complex<f64>: |_, [z]| atan::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| unturned_parts(x, y, atanh_parts, re, im), |[z], _| rough_atanh(turned(z)), |[z]| atan::<_, { crate::FUSED }>(z)));
forms!(Sqrt1mx2 at Complex<f32>, Complex<f64>: |_, [z]| sqrt1mx2::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| sqrt1mx2_parts(x, y, re, im), |[z], _| rough_roots(z), |[z]| sqrt1mx2::<_, { crate::FUSED }>(z)));
forms!(Sqrt1px2 at Complex<f32>, Complex<f64>: |_, [z]| sqrt1px2::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| turned_parts(x, y, sqrt1mx2_parts, re, im), |[z], _| rough_roots(turned(z)),
        |[z]| sqrt1px2::<_, { crate::FUSED }>(z)));
forms!(Sqrtx2m1 at Complex<f32>, Complex<f64>: |_, [z]| sqrtx2m1::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| root_products_parts(shifted_roots_parts, false, x, y, re, im), |[z], _| rough_shifted_roots(z),
        |[z]| sqrtx2m1::<_, { crate::FUSED }>(z)));
forms!(Ln at Complex<f32>, Complex<f64>: |_, [z]| ln::<_, { crate::FUSED }>(z),
    many |_, [z], n, out| split!([z], n, out, |[(x, y)], re, im| ln_parts(x, y, re, im), |[z], r| tiny(z) | !finite(r), |[z]| ln::<_, { crate::FUSED }>(z)));
forms!(Pow at Complex<f32>, Complex<f64>: |_, [z, w]| pow::<_, { crate::FUSED }>(z, w),
    many |_, [z, w], n, out| split!([z, w], n, out, |[(zx, zy), (wx, wy)], re, im| {
        let (mut lr, mut li) = (Vec::new(), Vec::new());
        ln_parts(zx, zy, &mut lr, &mut li);
        let pr: Vec<_> = wx.iter().zip(wy).zip(lr.iter().zip(&li)).map(|((&a, &b), (&c, &d))| a * c - b * d).collect();
        let pi: Vec<_> = wx.iter().zip(wy).zip(lr.iter().zip(&li)).map(|((&a, &b), (&c, &d))| a * d + b * c).collect();
        exp_parts(&pr, &pi, re, im);
    }, |[z, w], r| {
        let (unit, zero, real) = pow_cases(z, w);
        unit | zero | real | tiny(z) | !finite(r)
    }, |[z, w]| pow::<_, { crate::FUSED }>(z, w)));
forms!(Logb at Complex<f32>, Complex<f64>: |_, [b, z]| divide(ln::<_, { crate::FUSED }>(z), ln::<_, { crate::FUSED }>(b)));
forms!(Binom at Complex<f32>, Complex<f64>: |_, [n, k]| binom::<_, { crate::FUSED }>(n, k));
forms!(Abs at Complex<f32> => f32, Complex<f64> => f64: |_, [z]| crate::slice::Total(p::hypot::<_, { crate::FUSED }>(z.re, z.im)));
forms!(Angle at Complex<f32> => f32, Complex<f64> => f64: |_, [z]| crate::slice::Total(arc::atan2::<_, { crate::FUSED }>(z.im, z.re)));
