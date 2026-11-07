//! Complex functions at `Complex<f32>` and `Complex<f64>`: the exponential and logarithm, square root and powers, the circle and hyperbolic functions and their inverses, ln Γ and Γ, binomial coefficients, the magnitude and angle, and the sign z/|z|. Each computes at its own width. The facts table calls their scalar bodies or bulk bodies that run real-array kernels over the parts.
//!
//! Their error counts in ulps of the result's magnitude: |f(z) - exact| is within 4 ulps of |exact|. A part much smaller than the other can carry most of that error. ln Γ counts in ulps of the larger of its magnitude and 2. The functions that exponentiate a logarithm count in ulps of their magnitude times one more than the exponent's magnitude, or than 2 where that is smaller: |w ln z| for powers, and |ln Γ| for Γ and binomial coefficients. Branch cuts follow C99's Annex G: the sign of a zero part chooses the side of a cut. The inverse functions use the formulas from Kahan's "Branch Cuts for Complex Elementary Functions" (1987), built on the complex square root, which keep their digits near the branch points.
//!
//! `exp` and the powers that compute through it give an exact direction at the float nearest a whole number of quarter turns. e^(iπ/2) is then exactly i, where the exponential of that float has a real part of about 6e-17. These results lie outside the error bound.
use crate::{Native, arc, exp as e, float::ln2, hyperbolic, ln as l, pow as p, special, trig};
use num_complex::Complex;

pub(crate) mod bulk;

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
    let k = (y * R::narrow(std::f64::consts::FRAC_2_PI)).round_ties_even();
    (k, (k.abs() <= R::narrow(256.0)) & (k * q == y))
}

/// cos y and sin y as exactly 1, 0 or -1, where y is the float nearest kπ/2 for a whole number k of quarter turns, with |k| up to 256. A zero y keeps its sign as sin y.
#[inline(always)]
pub(crate) fn quarter<R: Native>(y: R) -> Option<(R, R)> {
    let (k, exact) = turns(y);
    if !exact { return None; }
    let (zero, one, m) = (R::zero(), R::one(), k - R::narrow(4.0) * (k * R::narrow(0.25)).floor());
    Some(if m == zero { (one, if y == zero { y }
    else { zero }) }
    else if m == one { (zero, one) }
    else if m == one + one { (-one, zero) }
    else { (zero, -one) })
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

/// ln(1 + w). Below |w| = ½, ln |1 + w| is ½ log1p(u(2+u) + v²), preserving digits near zero.
#[inline]
pub(crate) fn ln1p<R: Native, const FUSED: bool>(w: Complex<R>) -> Complex<R> {
    let (u, v) = (w.re, w.im);
    let small = modulus::<R, FUSED>(u, v) < R::narrow(0.5);
    let t = u.mla::<FUSED>(u + R::narrow(2.0), v * v);
    let re = if small { l::log1p::<R, FUSED>(t) * R::narrow(0.5) } else { l::ln::<R, FUSED>(modulus::<R, FUSED>(R::one() + u, v)) };
    c(re, arc::atan2::<R, FUSED>(v, R::one() + u))
}

/// The parts of √(x + iy), whose real part is never negative, from x, y and h = |x + iy|. t = √((|x| + h)/2) is one part, and y/2t the other, which doesn't cancel. |y|/2t is |y/2t| exactly, so one division serves both signs of x.
#[inline(always)]
fn root<R: Native, const FUSED: bool>(x: R, y: R, h: R) -> (R, R) {
    let t = x.abs().mla::<FUSED>(R::narrow(0.5), h * R::narrow(0.5)).sqrt();
    let q = y / (t + t);
    if x >= R::zero() { (t, q) } else { (q.abs(), t.copysign(y)) }
}

/// √z, through `root` on z and |z|. A tiny z scales up by 2^(2p) first, so that its halves stay normal, and its root back by 2^-p. Zero gives itself, and an infinite y gives ∞ + iy, even beside NaN.
#[inline]
pub(crate) fn sqrt<R: Native, const FUSED: bool>(z: Complex<R>) -> Complex<R> {
    let (x, y) = (z.re, z.im);
    let (w, p) = scaled(z);
    let (re, im) = root::<R, FUSED>(w.re, w.im, modulus::<R, FUSED>(w.re, w.im));
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

pub(crate) fn nth_root<R: Native, const FUSED: bool>(n: Complex<R>, y: Complex<R>) -> Complex<R> {
    if n == c(R::zero(), R::zero()) || !finite(n) { return c(R::nan(), R::nan()); }
    if n == c(R::narrow(2.0), R::zero()) { sqrt::<R, FUSED>(y) } else { pow::<R, FUSED>(y, divide(c(R::one(), R::zero()), n)) }
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

/// `r`, a square root of a number whose imaginary part is 2xy, with its smaller part recomputed as xy divided by its larger part. A product of two square roots, as `sqrt1mx2` and `sqrtx2m1` form, loses the smaller part where x is too small to change 1 ± x. The product xy keeps it. The larger of x and y is divided first, so a huge x with a tiny y doesn't underflow. `r` stays as it is where x or y is 0, to keep the signs of zeros on the branch cuts, and where a part isn't finite. It selects between values in place of branching, so a loop over a slice vectorizes.
#[inline(always)]
fn balanced<R: Native>(r: Complex<R>, x: R, y: R) -> Complex<R> {
    let zero = R::zero();
    let swap = x.abs() < y.abs();
    let (hi, lo) = (if swap { y } else { x }, if swap { x } else { y });
    let wide = r.re.abs() >= r.im.abs();
    let big = if wide { r.re } else { r.im };
    let keep = (x == zero) | (y == zero) | (big == zero) | !(r.re.is_finite() & r.im.is_finite() & x.is_finite() & y.is_finite());
    let small = if keep { if wide { r.im } else { r.re } } else { hi / big * lo };
    if wide { c(r.re, small) } else { c(small, r.im) }
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

pub(crate) fn sqrtm1mx2<R: Native, const FUSED: bool>(z: Complex<R>, inverse: bool) -> Complex<R> {
    let r = if z.im == R::zero() { c(R::zero(), crate::sqrts::sqrt1px2::<R, FUSED>(z.re)) } else { sqrt::<R, FUSED>(-c(R::one(), R::zero()) - z * z) };
    if inverse { -r } else { r }
}

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
    let (s, k) = crate::recipe::sincospi(&crate::recipe::Floats, x + x);
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

/// Whether both parts of z are finite.
#[inline(always)]
fn finite<R: Native>(z: Complex<R>) -> bool { z.re.is_finite() & z.im.is_finite() }

/// e^(iz), the cis of z, as a function object, which is cos z + i sin z for a real z. It takes real and complex items.
#[derive(Clone, Copy, Debug)]
pub struct Cis;
