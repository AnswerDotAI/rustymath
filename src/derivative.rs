//! Partial-derivative recipes shared by [`Formula`](crate::facts::Formula) rows. They use [`Ops`] on the original arguments and result. [`Slope`] presents a selected partial as another formula. A masked case keeps discarded intermediate values finite before selecting the slope.
use crate::recipe::Ops;
use crate::{Exponent, Float};

/// The slopes of ψ⁽ⁿ⁾(x), with its whole order n first: ψ⁽ⁿ⁺¹⁾(x) in x, and none in n.
#[inline(always)]
pub(crate) fn polygamma_slopes<F: Float, O: Ops<F> + ?Sized>(o: &O, [n, x]: [O::V; 2], _: O::V, [_, nx]: [bool; 2]) -> [Option<O::V>; 2] {
    [None, nx.then(|| o.polygamma(o.add(n, o.number(F::one())), x))]
}

/// The slopes of P(a, x), the regularized lower incomplete gamma function, with a first. Its slope in x is x^(a-1) e^-x / Γ(a), as e^((a-1) ln x - x - ln Γ(a)), where (a-1) ln x is 0 at a = 1 for every x, so that x⁰ is 1 at x = 0 too: a constant 1 leaves the term out, and an a that varies by item stands x in as 1 there. Its slope in a has no closed form, so it has none here.
#[inline(always)]
pub(crate) fn gammainc_slopes<F: Float, O: Ops<F> + ?Sized>(o: &O, [a, x]: [O::V; 2], _: O::V, [_, nx]: [bool; 2]) -> [Option<O::V>; 2] {
    let one = o.number(F::one());
    let power = || match o.constant(a) {
        Some(c) if c == F::one() => o.number(F::zero()), _ => o.multiply(o.subtract(a, one), o.ln(o.add(x, o.one_where(o.equal(a, one)))))
    };
    [None, nx.then(|| o.exp(o.subtract(o.subtract(power(), x), o.lgamma(a))))]
}

/// The slopes of the logarithm of `y` to base `x`, with the base first: ln y / ln x. They share ln x.
#[inline(always)]
pub(crate) fn logb<F: Float, O: Ops<F> + ?Sized>(o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
    let base = o.ln(x);
    [nx.then(|| o.negate(o.divide(r, o.multiply(x, base)))), ny.then(|| o.reciprocal(o.multiply(y, base)))]
}

/// What the slopes of ∂C(n, k)/∂n and ∂C(n, k)/∂k share, with n first: C(n, k), and n + 1, k + 1 and n - k + 1, where they read ψ and ψ⁽¹⁾.
#[inline(always)]
fn binom_points<F: Float, O: Ops<F> + ?Sized>(o: &O, n: O::V, k: O::V) -> (O::V, [O::V; 3]) {
    let one = o.number(F::one());
    (o.binom(n, k), [o.add(n, one), o.add(k, one), o.add(o.subtract(n, k), one)])
}

/// ψ⁽¹⁾(x).
#[inline(always)]
fn trigamma<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V { o.polygamma(o.number(F::one()), x) }

/// The slopes of r = ∂C(n, k)/∂n, with n first. r is C(n, k)(ψ(n + 1) - ψ(n - k + 1)), so its slope in n is r times that difference plus C(n, k)(ψ⁽¹⁾(n + 1) - ψ⁽¹⁾(n - k + 1)), and its slope in k is ∂C/∂k times the difference plus C(n, k)ψ⁽¹⁾(n - k + 1). A pole of ψ gives NaN.
#[inline(always)]
pub(crate) fn binom_dn_slopes<F: Float, O: Ops<F> + ?Sized>(o: &O, [n, k]: [O::V; 2], r: O::V, [nn, nk]: [bool; 2]) -> [Option<O::V>; 2] {
    let (c, [n1, _, m]) = binom_points(o, n, k);
    let (a, tm) = (o.subtract(o.digamma(n1), o.digamma(m)), trigamma(o, m));
    [nn.then(|| o.add(o.multiply(r, a), o.multiply(c, o.subtract(trigamma(o, n1), tm)))), nk.then(|| o.add(o.multiply(o.binom_dk(n, k), a), o.multiply(c, tm)))]
}

/// The slopes of r = ∂C(n, k)/∂k, with n first. r is C(n, k)(ψ(n - k + 1) - ψ(k + 1)), so its slope in n is ∂C/∂n times that difference plus C(n, k)ψ⁽¹⁾(n - k + 1), and its slope in k is r times the difference less C(n, k)(ψ⁽¹⁾(n - k + 1) + ψ⁽¹⁾(k + 1)). A pole of ψ gives NaN.
#[inline(always)]
pub(crate) fn binom_dk_slopes<F: Float, O: Ops<F> + ?Sized>(o: &O, [n, k]: [O::V; 2], r: O::V, [nn, nk]: [bool; 2]) -> [Option<O::V>; 2] {
    let (c, [_, k1, m]) = binom_points(o, n, k);
    let (b, tm) = (o.subtract(o.digamma(m), o.digamma(k1)), trigamma(o, m));
    [nn.then(|| o.add(o.multiply(o.binom_dn(n, k), b), o.multiply(c, tm))), nk.then(|| o.subtract(o.multiply(r, b), o.multiply(c, o.add(tm, trigamma(o, k1)))))]
}

/// x^y's slope in x, y x^(y-1), where `Pow` or `RealPow` gave r:
/// - for a constant y of 1, 1, and of 0, 0;
/// - for a constant y that `Exponent` routes off the general route, y x^(y-1) by that route, as x²'s slope is 2x. With `halves` false, as for `RealPow`, only whole constants take it, since a half power of a negative x has no real value;
/// - for any other y, y r/x, from the result. At x = 0, 1 replaces r where 0 < y ≤ 1. It replaces x unless 0 < y < 1, where y/0 then gives ∞.
///
/// At x = 0 every route gives the slope's limit from above, as JAX does: 0 for a y above 1, 1 for a y of 1, ∞ for a y between 0 and 1, 0 for a y of 0 and -∞ for a y below 0.
#[inline(always)]
pub fn power_slope<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V, r: O::V, halves: bool) -> O::V {
    let (zero, one) = (o.number(F::zero()), o.number(F::one()));
    match o.constant(y).map(|c| (c, c - F::one())) {
        Some((c, _)) if c == F::one() => one,
        Some((c, _)) if c == F::zero() => zero,
        Some((_, k))
            if match Exponent::of(k.wide()) { Exponent::General => false, Exponent::Halves { .. } => halves, Exponent::Whole { .. } => true } =>
        { o.multiply(y, o.power(k, x)) }
        route => {
            let at_zero = o.equal(x, zero);
            let stand_in = |v, m| o.add(v, o.one_where(m));
            // A constant y, which isn't 1 here, decides where the 1 goes for every item.
            let (r, x) = match route {
                Some((c, _)) if F::zero() < c && c < F::one() => (stand_in(r, at_zero), x),
                Some(_) => (r, stand_in(x, at_zero)),
                None => {
                    let between = o.and(o.less(zero, y), o.less(y, one));
                    (stand_in(r, o.and(at_zero, o.or(between, o.equal(y, one)))), stand_in(x, o.and(at_zero, o.not(between))))
                }
            };
            o.multiply(y, o.divide(r, x))
        }
    }
}

/// `y x^(y-1)` as a smooth expression, avoiding `0 * 0^-1` where both arguments are zero.
#[inline(always)]
pub fn reduced_power_slope<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V {
    let (zero, one) = (o.number(F::zero()), o.number(F::one()));
    let base = o.select(o.and(o.equal(x, zero), o.equal(y, zero)), one, x);
    o.multiply(y, o.real_power(base, o.subtract(y, one)))
}

/// The slopes of x^y: in x as `power_slope` gives it, and in y r ln x, which is 0 where x isn't positive, where x stands in as 1.
#[inline(always)]
pub(crate) fn power<F: Float, O: Ops<F> + ?Sized>(o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
    let zero = o.number(F::zero());
    [nx.then(|| o.power_slope(x, y, r, true)), ny.then(|| o.multiply(r, o.ln(o.larger(x, o.one_where(o.less_equal(x, zero))))))]
}

/// The slopes of the real part of the principal x^y (`RealPow`): in x as `power_slope` gives it, and in y the real part of x^y ln x, with ln x = ln|x| + iπ below 0: r ln|x| - π |x|^y sin πy. ln|x| is 0 at x = 0, as `power`'s is where x isn't positive.
#[inline(always)]
pub(crate) fn real_power<F: Float, O: Ops<F> + ?Sized>(o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
    let zero = o.number(F::zero());
    [
        nx.then(|| o.power_slope(x, y, r, false)), ny.then(|| {
            let ln = o.ln(o.select(o.equal(x, zero), o.number(F::one()), o.magnitude(x)));
            // The select's cases are atoms, so its mask, built from x, needs no layout of y's.
            let below = o.select(o.less(x, zero), o.number(F::narrow(core::f64::consts::PI)), zero);
            o.subtract(o.multiply(r, ln), o.multiply(below, o.multiply(o.exp(o.multiply(y, ln)), crate::recipe::sincospi(o, y).0)))
        }),
    ]
}

/// `K`'s partial derivative in its argument `SIDE`, as a function object at `f32` and `f64`: of `K`'s argument and result where `K` takes one argument, and of its two arguments and its result where it takes two. Each item runs `K`'s recipe on floats ([`Floats`]), which see no constant. A slope that is NaN where no argument is, such as `Pow`'s in `y` at an infinite `x` and a negative `y`, is its value, as [`Subtract`](crate::Subtract) gives NaN for ∞ - ∞. A slope in an argument that has none, such as `Polygamma`'s in its order, is NaN.
#[derive(Clone, Copy, Debug)]
pub struct Slope<K, const SIDE: usize = 0>(pub K);
