//! Complex functions at `Complex<f32>` and `Complex<f64>`: the exponential and logarithm, square root and powers, the circle and
//! hyperbolic functions and their inverses, ln Γ and Γ, binomial coefficients, the magnitude and angle, and the sign z/|z|. Each
//! computes at its own width, and attaches its forms to the function object of the same name, which serves real arguments too.
//!
//! Their error counts in ulps of the result's magnitude: |f(z) - exact| is within 4 ulps of |exact|. A part much smaller than the
//! other can carry most of that error. ln Γ counts in ulps of the larger of its magnitude and 2. The functions that exponentiate a
//! logarithm count in ulps of their magnitude times one more than the exponent's magnitude, or than 2 where that is smaller: |w ln z|
//! for powers, and |ln Γ| for Γ and binomial coefficients. Branch cuts follow C99's Annex G: the sign of a zero part chooses the side of
//! a cut.
//! The inverse functions use the formulas from Kahan's "Branch Cuts for Complex Elementary Functions" (1987), built on the complex
//! square root, which keep their digits near the branch points.
use crate::{arc, exp as e, float::ln2, hyperbolic, ln as l, pow as p, special, trig, Native};
use crate::{
    Abs, Acos, Acosh, Angle, Asin, Asinh, Atan, Atanh, Binom, Cos, Cosh, Exp, Factorial, Gamma, Lgamma, Ln, Logb, Pow, Sign, Sin, Sinh,
    Sqrt, Sqrt1mx2, Sqrt1px2, Sqrtx2m1, Tan, Tanh,
};
use num_complex::Complex;

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

/// e^z = e^x (cos y + i sin y). e^x/2 times 2 cos y keeps the result finite where e^x overflows but the result doesn't. A real z gives
/// a real result. At x = -∞ the result is 0, and at x = ∞ with an infinite or NaN y it is ∞ + i NaN.
#[inline]
pub(crate) fn exp<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let half = e::exp_times::<R, FUSED>(x, -R::one());
    let (cy, sy) = (trig::cos::<R, FUSED>(y), trig::sin::<R, FUSED>(y));
    let r = c(half * (cy + cy), if y == R::zero() { y } else { half * (sy + sy) });
    let lost = x.is_infinite() & !y.is_finite();
    if lost { if x > R::zero() { c(x, R::nan()) } else { c(R::zero(), R::zero().copysign(y)) } } else { r }
}

/// z scaled up exactly by 2^(2p) where it is so small that its halves or its magnitude would lose bits, and p, or z itself and 0. p
/// is the number of bits in `R`'s significand.
#[inline(always)]
fn scaled<R: Native>(z: Complex<R>) -> (Complex<R>, R) {
    let (low, scale, p) = if R::WIDE { (1.0020841800044864e-292, 8.112963841460668e31, 53.0) } else { (9.860761315262648e-32, 2.81474976710656e14, 24.0) };
    if z.re.abs().max(z.im.abs()) < R::narrow(low) { (z * R::narrow(scale), R::narrow(p)) } else { (z, R::zero()) }
}

/// ln z = ln |z| + i arg z. Where |z| is near 1, ln |z| is ½ log1p((x-1)(x+1) + y²), which keeps its digits. A tiny z scales up
/// first, so that |z| stays normal.
#[inline]
pub(crate) fn ln<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let (w, p) = scaled(z);
    let (hi, lo) = ln2::<R>();
    let h = p::hypot::<R, FUSED>(w.re, w.im);
    let near = (h > R::narrow(0.5)) & (h < R::narrow(2.0));
    let t = (x - R::one()).mla::<FUSED>(x + R::one(), y * y);
    let k = p + p;
    c(if near { l::log1p::<R, FUSED>(t) * R::narrow(0.5) } else { l::ln::<R, FUSED>(h) - k * hi - k * lo }, arc::atan2::<R, FUSED>(y, x))
}

/// ln(1 + w) for |w| up to 1. Below ½, ln |1 + w| is ½ log1p(u(2+u) + v²), which keeps its digits.
#[inline]
fn ln1p<R: Native, const FUSED: bool>(w: Complex<R>) -> Complex<R> {
    let (u, v) = (w.re, w.im);
    let small = p::hypot::<R, FUSED>(u, v) < R::narrow(0.5);
    let t = u.mla::<FUSED>(u + R::narrow(2.0), v * v);
    let re = if small { l::log1p::<R, FUSED>(t) * R::narrow(0.5) } else { l::ln::<R, FUSED>(p::hypot::<R, FUSED>(R::one() + u, v)) };
    c(re, arc::atan2::<R, FUSED>(v, R::one() + u))
}

/// √z, with a real part that is never negative. t = √((|x| + |z|)/2) is one part, and y/2t the other, which doesn't cancel. A tiny z
/// scales up by 2^(2p) first, so that its halves stay normal, and its root back by 2^-p. Zero gives itself, and an infinite y gives
/// ∞ + iy, even beside NaN.
#[inline]
pub(crate) fn sqrt<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let (w, p) = scaled(z);
    let h = p::hypot::<R, FUSED>(w.re, w.im);
    let t = w.re.abs().mla::<FUSED>(R::narrow(0.5), h * R::narrow(0.5)).sqrt();
    let (re, im) = if w.re >= R::zero() { (t, w.im / (t + t)) } else { (w.im.abs() / (t + t), t.copysign(w.im)) };
    let k = R::pow2(R::SHIFT - p);
    if y.is_infinite() { c(R::infinity(), y) } else if (x == R::zero()) & (y == R::zero()) { c(R::zero(), y) } else { c(re * k, im * k) }
}

/// z^w = e^(w ln z). w = 0 gives 1, even for NaN, and z = 0 gives 0 where Re w > 0. A real z and w whose power is real give the real
/// power, as `pow` gives it: a z from 0 up, or a whole w. Its error grows with |w ln z|, as the module doc gives it.
#[inline]
pub(crate) fn pow<R: Native, const FUSED: bool>(z: Complex<R>, w: Complex<R>) -> Complex<R> {
    let real = (z.im == R::zero()) & (w.im == R::zero()) & ((z.re >= R::zero()) | (w.re == w.re.trunc()));
    let r = p::pow::<R, FUSED>(z.re, w.re);
    let zero = (z.re == R::zero()) & (z.im == R::zero()) & (w.re > R::zero());
    let unit = (w.re == R::zero()) & (w.im == R::zero());
    let general = exp::<R, FUSED>(w * ln::<R, FUSED>(z));
    if unit { c(R::one(), R::zero()) } else if zero { c(R::zero(), R::zero()) } else if real { c(r, R::zero()) } else { general }
}

/// sinh z = sinh x cos y + i cosh x sin y. A real z gives a real result and an imaginary z an imaginary one, even where cosh x
/// overflows or cos y is NaN.
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

/// tanh z by Kahan's formula: with t = tan y, s = sinh x and ρ = √(1 + s²) = cosh x, tanh z = ((1+t²)ρs + it)/(1 + (1+t²)s²).
/// Beyond a magnitude where tanh x rounds to ±1, the result is ±1 + 4 sin y cos y e^(-2|x|) i, which can't overflow.
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
    let limit = R::narrow(if R::WIDE { 22.0 } else { 10.0 });
    let r = if x.abs() > limit { far } else { near };
    if x.is_infinite() & !y.is_finite() { c(r.re, R::zero().copysign(y)) } else { r }
}

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

/// √(1 - z²) as √(1 - z) √(1 + z), whose real part is never negative. Neither factor loses digits near z = ±1, or overflows where
/// z² would.
#[inline]
pub(crate) fn sqrt1mx2<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = roots::<R, FUSED>(z);
    a * b
}

/// √(1 + z²) as √(1 + iz) √(1 - iz), the principal root, whose real part is never negative: `roots` at -iz. Neither factor
/// overflows where z² would.
#[inline]
pub(crate) fn sqrt1px2<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = roots::<R, FUSED>(c(z.im, -z.re));
    a * b
}

/// √(z - 1) √(z + 1), the continuation of the real `sqrtx2m1` off the real line. Neither factor loses digits near z = ±1, or
/// overflows where z² would.
#[inline]
pub(crate) fn sqrtx2m1<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = shifted_roots::<R, FUSED>(z);
    a * b
}

/// asin z, by Kahan's formula: with a = √(1 - z) and b = √(1 + z), Re asin z = atan(x / Re(ab)) and Im asin z = asinh(Im(conj(a) b)).
#[inline]
pub(crate) fn asin<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = roots::<R, FUSED>(z);
    let re = arc::atan2::<R, FUSED>(z.re, a.re.mla::<FUSED>(b.re, -(a.im * b.im)));
    c(re, hyperbolic::asinh::<R, FUSED>(a.re.mla::<FUSED>(b.im, -(a.im * b.re))))
}

/// acos z, by Kahan's formula: with a = √(1 - z) and b = √(1 + z), Re acos z = 2 atan(Re a / Re b) and
/// Im acos z = asinh(Im(conj(b) a)).
#[inline]
pub(crate) fn acos<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = roots::<R, FUSED>(z);
    let re = arc::atan2::<R, FUSED>(a.re, b.re);
    c(re + re, hyperbolic::asinh::<R, FUSED>(b.re.mla::<FUSED>(a.im, -(b.im * a.re))))
}

/// acosh z, by Kahan's formula: with a = √(z - 1) and b = √(z + 1), Re acosh z = asinh(Re(conj(a) b)) and
/// Im acosh z = 2 atan(Im a / Re b).
#[inline]
pub(crate) fn acosh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (a, b) = shifted_roots::<R, FUSED>(z);
    let im = arc::atan2::<R, FUSED>(a.im, b.re);
    c(hyperbolic::asinh::<R, FUSED>(a.re.mla::<FUSED>(b.re, a.im * b.im)), im + im)
}

/// asinh z = -i asin(iz).
#[inline]
pub(crate) fn asinh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> { unturned(asin::<R, FUSED>(turned(z))) }

/// atanh z, with Re atanh z = ¼ log1p(4x / ((1-x)² + y²)) and Im atanh z = ½ atan2(2y, (1-x)(1+x) - y²). atanh is odd, so it computes
/// at |x|, where the argument of log1p is from 0 up and loses nothing. Where the squares would overflow, Re atanh z is x/|z|² and
/// Im atanh z is ±π/2. An infinite part gives ±0 ± iπ/2.
#[inline]
pub(crate) fn atanh<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let s = R::one().copysign(z.re);
    let (x, y) = (z.re.abs(), z.im * s);
    let (m, quarter, half) = (R::one() - x, R::narrow(0.25), R::narrow(0.5));
    let re = l::log1p::<R, FUSED>(R::narrow(4.0) * x / m.mla::<FUSED>(m, y * y)) * quarter;
    let im = arc::atan2::<R, FUSED>(y + y, m.mla::<FUSED>(R::one() + x, -(y * y))) * half;
    let h = p::hypot::<R, FUSED>(x, y);
    let (right, _) = arc::half_pi::<R>();
    let huge = h > R::narrow(if R::WIDE { 1e150 } else { 1e18 });
    let (re, im) = if x.is_infinite() | y.is_infinite() { (R::zero(), right.copysign(y)) } else if huge { (x / h / h, right.copysign(y)) } else { (re, im) };
    c(re * s, im * s)
}

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

/// ln Γ(w) by Stirling's series, for |w| from 7 up, away from the negative real axis: (w - ½) ln w - w + ½ ln 2π + (1/w) S(1/w²),
/// with the ten terms of `special::STIRLING`.
#[inline]
fn stirling<R: Native, const FUSED: bool>(w: Complex<R>) -> Complex<R> {
    let n = w.re.mla::<FUSED>(w.re, w.im * w.im);
    let v = c(w.re / n, -w.im / n);
    let v2 = v * v;
    let mut s = c(R::narrow(special::STIRLING[0]), R::zero());
    for &k in &special::STIRLING[1..] { s = s * v2 + c(R::narrow(k), R::zero()); }
    c(w.re - R::narrow(0.5), w.im) * ln::<R, FUSED>(w) - w + c(R::narrow(special::HALF_LN_2PI), R::zero()) + v * s
}

/// ln Γ(w) for Re w from 0.1 up: `stirling` itself where Re w or |Im w| is 7 or more, and otherwise ln Γ(w + n) - ln(w(w+1)…(w+n-1))
/// with n the steps that bring Re w to 7. Above the real axis, the product's argument grows with each factor. Each time it passes π
/// the product's imaginary part turns negative, and the principal logarithm drops 2π, which the result adds back. Below the axis it
/// is the conjugate of the result above.
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

/// ln π - ln sin πu, for Im u from 0 up. sin πu = (i/2) e^(-iπu) (1 - e^(2πiu)), so this is ln 2π - πy + iπ(x - ½) - ln(1 - e^(2πiu)),
/// which stays finite where sin πu overflows. The imaginary part isn't reduced, which keeps it continuous over the half-plane.
#[inline]
fn reflected<R: Native, const FUSED: bool>(u: Complex<R>) -> Complex<R> {
    let (x, y) = (u.re, u.im);
    let (s, k) = sincospi::<R, FUSED>(x + x);
    let m = e::exp::<R, FUSED>(R::narrow(-2.0) * pi::<R>() * y);
    let t = ln1p::<R, FUSED>(c(-(m * k), -(m * s)));
    c((R::narrow(1.8378770664093453) - pi::<R>() * y) - t.re, pi::<R>() * (x - R::narrow(0.5)) - t.im)
}

/// ln Γ(z), on the branch that is real on the positive real axis and continuous everywhere but across the negative real axis, as
/// SciPy's and mpmath's `loggamma` give it. Below Re z = 0.1 it reflects: ln Γ(z) = ln π - ln sin πz - ln Γ(1 - z), and the lower
/// half-plane gives the conjugate of the upper. Elsewhere it is `shifted`. Poles and non-finite arguments give NaN. Near the zeros at 1
/// and 2 the error counts in ulps of 2, not of the result.
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


/// The binomial coefficient Γ(n+1) / (Γ(k+1) Γ(n-k+1)), as e^(ln Γ(n+1) - ln Γ(k+1) - ln Γ(n-k+1)). A pole in the denominator gives
/// 0, and one in the numerator NaN.
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
    let h = p::hypot::<R, FUSED>(w.re, w.im);
    if h == R::zero() { z } else { c(w.re / h, w.im / h) }
}

/// Attaches each function of one complex argument to its object, at both widths.
macro_rules! unary {
    ($($object:ident: $f:ident;)+) => {$(
        forms!($object at Complex<f32>, Complex<f64>: |_, [z]| $f::<_, { crate::FUSED }>(z));
    )+};
}
unary! {
    Exp: exp; Ln: ln; Sqrt: sqrt; Sin: sin; Cos: cos; Tan: tan; Sinh: sinh; Cosh: cosh; Tanh: tanh; Asin: asin; Acos: acos;
    Atan: atan; Asinh: asinh; Acosh: acosh; Atanh: atanh; Lgamma: lgamma; Gamma: gamma; Factorial: factorial; Sign: sign;
    Sqrt1mx2: sqrt1mx2; Sqrt1px2: sqrt1px2; Sqrtx2m1: sqrtx2m1;
}
forms!(Pow at Complex<f32>, Complex<f64>: |_, [z, w]| pow::<_, { crate::FUSED }>(z, w));
forms!(Logb at Complex<f32>, Complex<f64>: |_, [b, z]| ln::<_, { crate::FUSED }>(z) / ln::<_, { crate::FUSED }>(b));
forms!(Binom at Complex<f32>, Complex<f64>: |_, [n, k]| binom::<_, { crate::FUSED }>(n, k));
forms!(Abs at Complex<f32> => f32, Complex<f64> => f64: |_, [z]| crate::slice::Total(p::hypot::<_, { crate::FUSED }>(z.re, z.im)));
forms!(Angle at Complex<f32> => f32, Complex<f64> => f64: |_, [z]| crate::slice::Total(arc::atan2::<_, { crate::FUSED }>(z.im, z.re)));

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    /// The bound on each function's error, in ulps of the result's magnitude.
    const BOUND: f64 = 4.0;

    /// The function named as in `complex.txt`, at `R`, with `mla` fused or not. Other modules' functions in that file give `None`.
    fn apply<R: Native, const FUSED: bool>(name: &str, z: &[Complex<R>]) -> Option<Complex<R>> {
        Some(match name {
            "exp" => exp::<R, FUSED>(z[0]),
            "log" => ln::<R, FUSED>(z[0]),
            "sqrt" => sqrt::<R, FUSED>(z[0]),
            "sin" => sin::<R, FUSED>(z[0]),
            "cos" => cos::<R, FUSED>(z[0]),
            "tan" => tan::<R, FUSED>(z[0]),
            "sinh" => sinh::<R, FUSED>(z[0]),
            "cosh" => cosh::<R, FUSED>(z[0]),
            "tanh" => tanh::<R, FUSED>(z[0]),
            "asin" => asin::<R, FUSED>(z[0]),
            "acos" => acos::<R, FUSED>(z[0]),
            "atan" => atan::<R, FUSED>(z[0]),
            "asinh" => asinh::<R, FUSED>(z[0]),
            "acosh" => acosh::<R, FUSED>(z[0]),
            "atanh" => atanh::<R, FUSED>(z[0]),
            "lgamma" => lgamma::<R, FUSED>(z[0]),
            "gamma" => gamma::<R, FUSED>(z[0]),
            "pow" => pow::<R, FUSED>(z[0], z[1]),
            "binom" => binom::<R, FUSED>(z[0], z[1]),
            "lambertw" => crate::lambert::complex(z[0]),
            "sqrt1mx2" => sqrt1mx2::<R, FUSED>(z[0]),
            "sqrtx2m1" => sqrtx2m1::<R, FUSED>(z[0]),
            "sqrt1px2" => sqrt1px2::<R, FUSED>(z[0]),
            _ => return None,
        })
    }

    /// Each line of `complex.txt` as its function's name, its arguments and the correctly rounded result.
    fn cases() -> Vec<(String, Vec<Complex<f64>>, Complex<f64>)> {
        include_str!("../testdata/complex.txt").lines().filter(|l| !l.starts_with('#')).map(|l| {
            let mut fields = l.split_whitespace();
            let name = fields.next().unwrap().to_string();
            let v: Vec<f64> = fields.map(|f| f.parse().unwrap()).collect();
            let z = v.chunks(2).map(|p| c(p[0], p[1])).collect::<Vec<_>>();
            let (args, want) = z.split_at(z.len() - 1);
            (name, args.to_vec(), want[0])
        }).collect()
    }

    /// The magnitude that a function's error counts in ulps of, as the module doc gives it.
    fn scale(name: &str, args: &[Complex<f64>], want: Complex<f64>) -> f64 {
        let size = |z: Complex<f64>| z.re.hypot(z.im);
        let lg = |z: Complex<f64>| size(lgamma::<f64, false>(z));
        let exponentiated = |exponent: f64| size(want) * (exponent.max(2.0) + 1.0);
        let one = c(1.0, 0.0);
        match name {
            "lgamma" => size(want).max(2.0),
            "gamma" => exponentiated(lg(args[0])),
            "binom" => exponentiated(lg(args[0] + one) + lg(args[1] + one) + lg(args[0] - args[1] + one)),
            "pow" => exponentiated(size(args[1] * ln::<f64, false>(args[0]))),
            _ => size(want),
        }
    }

    /// The worst error of each function, in ulps of its scale, over the cases whose arguments and result are finite and normal at `R`.
    fn worst<R: Native, const FUSED: bool>(eps: f64, narrow: impl Fn(f64) -> R) -> Vec<(String, f64, Vec<Complex<f64>>)> {
        let mut out: Vec<(String, f64, Vec<Complex<f64>>)> = Vec::new();
        let fits = |v: f64| v.is_finite() && narrow(v).wide() == v;
        let normal = |v: f64| v == 0.0 || (v.abs() >= R::min_positive_value().wide() && v.abs() <= R::max_value().wide());
        for (name, args, want) in cases() {
            if !args.iter().all(|z| fits(z.re) && fits(z.im)) || !(normal(want.re) && normal(want.im)) || !(want.re.is_finite() && want.im.is_finite()) { continue; }
            let Some(got) = apply::<R, FUSED>(&name, &args.iter().map(|z| c(narrow(z.re), narrow(z.im))).collect::<Vec<_>>()) else { continue };
            let d = (got.re.wide() - want.re).hypot(got.im.wide() - want.im);
            let s = scale(&name, &args, want);
            let err = if s == 0.0 { if d == 0.0 { 0.0 } else { f64::INFINITY } } else { d / (s * eps) };
            let err = if err.is_nan() { f64::INFINITY } else { err };
            match out.iter_mut().find(|(n, _, _)| *n == name) {
                Some(row) => if err > row.1 { *row = (name, err, args) },
                None => out.push((name, err, args)),
            }
        }
        out
    }

    #[test]
    fn accuracy() {
        let rows = [
            ("f64 fused", worst::<f64, true>(f64::EPSILON, |v| v)),
            ("f64 unfused", worst::<f64, false>(f64::EPSILON, |v| v)),
            ("f32 fused", worst::<f32, true>(f32::EPSILON.into(), |v| v as f32)),
            ("f32 unfused", worst::<f32, false>(f32::EPSILON.into(), |v| v as f32)),
        ];
        let over: Vec<_> = rows.iter().flat_map(|(how, r)| r.iter().filter(|(_, e, _)| *e > BOUND).map(move |(n, e, a)| (*how, n, *e, a))).collect();
        if std::env::var("RUSTYMATH_COMPLEX").is_ok() { for (how, r) in &rows { for (n, e, a) in r { println!("{how} {n} {e:.2} {a:?}"); } } }
        assert!(over.is_empty(), "{over:#?}");
    }

    /// C99's values on the branch cuts, where the sign of a zero part chooses the side, and exact real or imaginary results on the axes.
    #[test]
    fn cuts_and_axes() {
        let k = |re: f64, im: f64| c(re, im);
        let (a, b) = (1.3169578969248166, 0.5493061443340549);
        let cases = [
            ("sqrt", k(-4.0, 0.0), k(0.0, 2.0)), ("sqrt", k(-4.0, -0.0), k(0.0, -2.0)),
            ("log", k(-1.0, 0.0), k(0.0, PI)), ("log", k(-1.0, -0.0), k(0.0, -PI)),
            ("asin", k(2.0, 0.0), k(FRAC_PI_2, a)), ("asin", k(2.0, -0.0), k(FRAC_PI_2, -a)),
            ("acos", k(2.0, 0.0), k(0.0, -a)), ("acos", k(2.0, -0.0), k(0.0, a)),
            ("acosh", k(-2.0, 0.0), k(a, PI)), ("acosh", k(-2.0, -0.0), k(a, -PI)),
            ("atanh", k(2.0, 0.0), k(b, FRAC_PI_2)), ("atanh", k(2.0, -0.0), k(b, -FRAC_PI_2)),
            ("asinh", k(0.0, 2.0), k(a, FRAC_PI_2)), ("asinh", k(-0.0, 2.0), k(-a, FRAC_PI_2)),
            ("atan", k(0.0, 2.0), k(FRAC_PI_2, b)), ("atan", k(-0.0, 2.0), k(-FRAC_PI_2, b)),
        ];
        for (name, z, want) in cases {
            let got = apply::<f64, false>(name, &[z]).unwrap();
            let close = |g: f64, w: f64| (g - w).abs() <= 4.0 * f64::EPSILON * w.abs().max(1.0) && g.is_sign_negative() == w.is_sign_negative();
            assert!(close(got.re, want.re) && close(got.im, want.im), "{name}({z}) gave {got}, not {want}");
        }
        for x in [0.5, 3.0, -2.0, 700.0, 1e-300] {
            for name in ["exp", "sin", "cos", "sinh", "cosh", "tanh", "tan"] { assert_eq!(apply::<f64, false>(name, &[k(x, 0.0)]).unwrap().im, 0.0, "{name}({x})"); }
            if x > 0.0 { for name in ["log", "sqrt"] { assert_eq!(apply::<f64, false>(name, &[k(x, 0.0)]).unwrap().im, 0.0, "{name}({x})"); } }
            for name in ["sin", "sinh", "tan", "tanh"] { assert_eq!(apply::<f64, false>(name, &[k(0.0, x)]).unwrap().re, 0.0, "{name}({x}i)"); }
        }
        assert_eq!(pow::<f64, false>(k(1.0, 1.0), k(0.0, 0.0)), k(1.0, 0.0));
        assert_eq!(pow::<f64, false>(k(-8.0, 0.0), k(3.0, 0.0)), k(-512.0, 0.0));
        assert_eq!(pow::<f64, false>(k(0.0, 0.0), k(2.0, 5.0)), k(0.0, 0.0));
        assert_eq!(sign::<f64, false>(k(f64::INFINITY, 3.0)), k(1.0, 0.0));
        assert_eq!(sign::<f64, false>(k(-0.0, 0.0)), k(-0.0, 0.0));
        assert!(lgamma::<f64, false>(k(-3.0, 0.0)).re.is_nan());
        assert_eq!(binom::<f64, false>(k(2.0, 0.0), k(3.0, 0.0)), k(0.0, 0.0));
    }
}
