//! The partial derivatives of rustymath's elementwise functions. Each function object's [`Partials`] gives them as a recipe from the function's arguments and its result, written once over [`Ops`]. [`Slope`] runs a recipe on floats ([`Floats`]) as a function object of the arguments and the result. basedpl's `∂` runs that object where it can, and otherwise records the recipe as calls of its own primitives. A recipe calls no [`Ops::select`], which a frontend's primitives may lack. It masks with [`Ops::one_where`]'s 1 or 0, times a value made finite where the mask is 0, as a power's slope does at 0.
use crate::recipe::{Floats, Ops};
use crate::reduce::{Maximum, Minimum};
use crate::tolerant::Residue;
use crate::slice::{Arg, Forms};
use crate::{Exponent, Float, Flipped, Native};
use std::mem::MaybeUninit;

/// A function whose partial derivatives a recipe gives, from its `N` arguments `x` and its result `r`.
pub trait Partials<const N: usize> {
    /// The partial derivative in each argument that `need` asks for, computed through `o`.
    fn partials<F: Float, O: Ops<F>>(&self, o: &O, x: [O::V; N], r: O::V, need: [bool; N]) -> [Option<O::V>; N];
}

/// `Partials<1>` for each function object, from its slope `|o, x, r| slope`.
macro_rules! slope {
    ($($($object:ty),+: |$o:tt, $x:tt, $r:tt| $slope:expr;)+) => {$($(
        impl Partials<1> for $object {
            #[inline(always)]
            fn partials<F: Float, O: Ops<F>>(&self, $o: &O, [$x]: [O::V; 1], $r: O::V, [need]: [bool; 1]) -> [Option<O::V>; 1] { [need.then(|| $slope)] }
        }
    )+)+};
}

/// `Partials<2>` for each function object, from its slopes in its two arguments, `|o, x, y, r| [dx, dy]`.
macro_rules! slopes {
    ($($($object:ty),+: |$o:tt, $x:tt, $y:tt, $r:tt| [$dx:expr, $dy:expr];)+) => {$($(
        impl Partials<2> for $object {
            #[inline(always)]
            fn partials<F: Float, O: Ops<F>>(&self, $o: &O, [$x, $y]: [O::V; 2], $r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
                [nx.then(|| $dx), ny.then(|| $dy)]
            }
        }
    )+)+};
}

/// `Partials<2>` for each function object, from a function that gives both slopes, where they share work.
macro_rules! shared {
    ($($($object:ty),+ => $slopes:ident;)+) => {$($(
        impl Partials<2> for $object {
            #[inline(always)]
            fn partials<F: Float, O: Ops<F>>(&self, o: &O, x: [O::V; 2], r: O::V, need: [bool; 2]) -> [Option<O::V>; 2] { $slopes(o, x, r, need) }
        }
    )+)+};
}

slope! {
    crate::Conj: |o, _, _| o.number(F::one());
    crate::Negate: |o, _, _| o.number(-F::one());
    crate::Reciprocal: |o, _, r| o.negate(o.multiply(r, r));
    crate::Exp: |_, _, r| r;
    crate::Ln: |o, x, _| o.reciprocal(x);
    crate::Log1p: |o, x, _| o.reciprocal(o.add(o.number(F::one()), x));
    crate::Expm1: |o, _, r| o.add(r, o.number(F::one()));
    crate::Sqrt: |o, _, r| o.reciprocal(o.add(r, r));
    crate::Cbrt: |o, _, r| o.reciprocal(o.multiply(o.number(F::narrow(3.0)), o.multiply(r, r)));
    crate::Abs: |o, x, _| o.sign(x);
    crate::PiTimes: |o, _, _| o.number(F::narrow(std::f64::consts::PI));
    // Γ(x + 1)'s slope is Γ(x + 1) ψ(x + 1).
    crate::Factorial: |o, x, r| o.multiply(r, o.digamma(o.add(x, o.number(F::one()))));
    crate::Sin: |o, x, _| o.cos(x);
    crate::Cos: |o, x, _| o.negate(o.sin(x));
    crate::Tan: |o, _, r| o.add(o.number(F::one()), o.multiply(r, r));
    crate::Asin: |o, x, _| o.reciprocal(o.sqrt1mx2(x));
    crate::Acos: |o, x, _| o.negate(o.reciprocal(o.sqrt1mx2(x)));
    crate::Atan: |o, x, _| o.reciprocal(o.add(o.number(F::one()), o.multiply(x, x)));
    crate::Sinh: |o, x, _| o.cosh(x);
    crate::Cosh: |o, x, _| o.sinh(x);
    crate::Tanh: |o, _, r| o.subtract(o.number(F::one()), o.multiply(r, r));
    crate::Asinh: |o, x, _| o.reciprocal(o.sqrt1px2(x));
    // sinh(acosh x) is √(x² - 1).
    crate::Acosh: |o, _, r| o.reciprocal(o.sinh(r));
    crate::Atanh: |o, x, _| o.reciprocal(o.subtract(o.number(F::one()), o.multiply(x, x)));
    crate::Sqrt1mx2: |o, x, r| o.negate(o.divide(x, r));
    crate::Sqrt1px2, crate::Sqrtx2m1: |o, x, r| o.divide(x, r);
    crate::Lgamma: |o, x, _| o.digamma(x);
    // ψ's slope is ψ⁽¹⁾.
    crate::Digamma: |o, x, _| o.polygamma(o.number(F::one()), x);
    // 2/√π e^-x², and its negative for erfc.
    crate::Erf: |o, x, _| o.multiply(o.number(F::narrow(std::f64::consts::FRAC_2_SQRT_PI)), o.exp(o.negate(o.multiply(x, x))));
    crate::Erfc: |o, x, _| o.multiply(o.number(F::narrow(-std::f64::consts::FRAC_2_SQRT_PI)), o.exp(o.negate(o.multiply(x, x))));
    // √π/2 e^(erfinv x)², from the result.
    crate::Erfinv: |o, _, r| o.multiply(o.number(F::narrow(std::f64::consts::FRAC_2_SQRT_PI.recip())), o.exp(o.multiply(r, r)));
}

slopes! {
    crate::Add: |o, _, _, _| [o.number(F::one()), o.number(F::one())];
    crate::Subtract: |o, _, _, _| [o.number(F::one()), o.number(-F::one())];
    crate::Multiply: |_, x, y, _| [y, x];
    crate::Divide, crate::MathishDivide: |o, _, y, r| [o.reciprocal(y), o.negate(o.divide(r, y))];
    // π(x ÷ y).
    crate::PiTimes: |o, _, y, r| [crate::recipe::pi_ratio(o, o.number(F::one()), y), o.negate(o.divide(r, y))];
    // The yth root of x, x^(1/y).
    crate::Root: |o, x, y, r| [o.negate(o.divide(o.multiply(r, o.ln(y)), o.multiply(x, x))), o.divide(r, o.multiply(x, y))];
    // C(n, k), with n first.
    crate::Binom: |o, n, k, _| [o.binom_dn(n, k), o.binom_dk(n, k)];
    // A tie goes to the left argument, as in Futhark.
    Maximum: |o, x, y, _| [o.one_where(o.less_equal(y, x)), o.one_where(o.less(x, y))];
    Minimum: |o, x, y, _| [o.one_where(o.less_equal(x, y)), o.one_where(o.less(y, x))];
    // `y` modulo `x`. Its slope in `x` is minus the quotient's floor, (y - r) ÷ x from the result, and 0 where `x` is 0, where the residue is `y` itself.
    Residue: |o, x, y, r| [{
        let at_zero = o.equal(x, o.number(F::zero()));
        o.negate(o.multiply(o.one_where(o.not(at_zero)), o.divide(o.subtract(y, r), o.add(x, o.one_where(at_zero)))))
    }, o.number(F::one())];
    // B(a, b): B(a, b)(ψ(a) - ψ(a + b)) in a and B(a, b)(ψ(b) - ψ(a + b)) in b, from the result.
    crate::Beta: |o, a, b, r| [o.multiply(r, o.subtract(o.digamma(a), o.digamma(o.add(a, b)))), o.multiply(r, o.subtract(o.digamma(b), o.digamma(o.add(a, b))))];
}

shared! {
    crate::Logb, crate::MathishLogb => logb;
    crate::Pow, crate::WholePow, crate::WrappingPow => power;
}

/// ψ⁽ⁿ⁾(x), with its whole order n first. Its slope in x is ψ⁽ⁿ⁺¹⁾(x). It has none in n.
impl Partials<2> for crate::Polygamma {
    #[inline(always)]
    fn partials<F: Float, O: Ops<F>>(&self, o: &O, [n, x]: [O::V; 2], _: O::V, [_, nx]: [bool; 2]) -> [Option<O::V>; 2] {
        [None, nx.then(|| o.polygamma(o.add(n, o.number(F::one())), x))]
    }
}

/// P(a, x), the regularized lower incomplete gamma function, with a first. Its slope in x is x^(a-1) e^-x / Γ(a), as e^((a-1) ln x - x - ln Γ(a)), where (a-1) ln x is 0 at a = 1 for every x, so that x⁰ is 1 at x = 0 too: a constant 1 leaves the term out, and an a that varies by item stands x in as 1 there. Its slope in a has no closed form, so it has none here.
impl Partials<2> for crate::Gammainc {
    #[inline(always)]
    fn partials<F: Float, O: Ops<F>>(&self, o: &O, [a, x]: [O::V; 2], _: O::V, [_, nx]: [bool; 2]) -> [Option<O::V>; 2] {
        let one = o.number(F::one());
        let power = || match o.constant(a) {
            Some(c) if c == F::one() => o.number(F::zero()),
            _ => o.multiply(o.subtract(a, one), o.ln(o.add(x, o.one_where(o.equal(a, one))))),
        };
        [None, nx.then(|| o.exp(o.subtract(o.subtract(power(), x), o.lgamma(a))))]
    }
}

/// The slopes of the logarithm of `y` to base `x`, with the base first: ln y / ln x. They share ln x.
#[inline(always)]
fn logb<F: Float, O: Ops<F>>(o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
    let base = o.ln(x);
    [nx.then(|| o.negate(o.divide(r, o.multiply(x, base)))), ny.then(|| o.reciprocal(o.multiply(y, base)))]
}

/// The slopes of x^y. In y it is r ln x, and 0 where x isn't positive, where x stands in as 1. In x it is y x^(y-1):
/// - for a constant y of 1, 1, and of 0, 0;
/// - for a constant y that `Exponent` routes off the general route, y x^(y-1) by that route, as x²'s slope is 2x;
/// - for any other y, y r/x, from the result. At x = 0, 1 replaces r where 0 < y ≤ 1. It replaces x unless 0 < y < 1, where y/0 then gives ∞.
///
/// At x = 0 every route gives the slope's limit from above, as JAX does: 0 for a y above 1, 1 for a y of 1, ∞ for a y between 0 and 1, 0 for a y of 0 and -∞ for a y below 0.
#[inline(always)]
fn power<F: Float, O: Ops<F>>(o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
    let (zero, one) = (o.number(F::zero()), o.number(F::one()));
    [nx.then(|| match o.constant(y).map(|c| (c, c - F::one())) {
        Some((c, _)) if c == F::one() => one,
        Some((c, _)) if c == F::zero() => zero,
        Some((_, k)) if Exponent::of(k.wide()) != Exponent::General => o.multiply(y, o.power(k, x)),
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
    }), ny.then(|| o.multiply(r, o.ln(o.larger(x, o.one_where(o.less_equal(x, zero))))))]
}

/// `K` with its arguments in the other order, whose slopes are `K`'s in the other order.
impl<K: Partials<2>> Partials<2> for Flipped<K> {
    #[inline(always)]
    fn partials<F: Float, O: Ops<F>>(&self, o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
        let [dy, dx] = self.0.partials(o, [y, x], r, [ny, nx]);
        [dx, dy]
    }
}

/// `K` with its integer results wrapped to their width, whose slopes on floats are `K`'s.
impl<K: Partials<N>, const N: usize> Partials<N> for crate::Wrapped<K> {
    #[inline(always)]
    fn partials<F: Float, O: Ops<F>>(&self, o: &O, x: [O::V; N], r: O::V, need: [bool; N]) -> [Option<O::V>; N] { self.0.partials(o, x, r, need) }
}

/// `K`'s partial derivative in its argument `SIDE`, as a function object at `f32` and `f64`: of `K`'s argument and result where `K` takes one argument, and of its two arguments and its result where it takes two. Each item runs `K`'s recipe on floats ([`Floats`]), which see no constant. A slope that is NaN where no argument is, such as `Pow`'s in `y` at an infinite `x` and a negative `y`, is its value, as [`Subtract`](crate::Subtract) gives NaN for ∞ - ∞. A slope in an argument that has none, such as `Polygamma`'s in its order, is NaN.
#[derive(Clone, Copy, Debug)]
pub struct Slope<K, const SIDE: usize = 0>(pub K);

impl<K, const SIDE: usize> Slope<K, SIDE> {
    /// The slope at the arguments `x`, where `K` gave `r`.
    #[inline(always)]
    fn at<F: Native, const N: usize>(&self, x: [F; N], r: F) -> F where K: Partials<N> {
        self.0.partials(&Floats, x, r, std::array::from_fn(|k| k == SIDE))[SIDE].unwrap_or(F::narrow(f64::NAN))
    }
}

unsafe impl<F: Native, K: Partials<1> + Sync> Forms<F, 2> for Slope<K> {
    type Out = F;
    const TOTAL: bool = true;
    const WRAPS: bool = false;
    const INLINE: bool = true;
    #[inline]
    fn scalar(&self, [x, r]: [F; 2]) -> F { self.at([x], r) }
    #[inline]
    fn judged_scalar(&self, x: [F; 2]) -> (F, bool) { (self.scalar(x), true) }
    fn slice(&self, [x, r]: [Arg<'_, F>; 2], n: usize, out: &mut [MaybeUninit<F>]) -> bool { crate::slice::pairs_into(x, r, n, out, |x, r| (self.at([x], r), true)) }
}

unsafe impl<F: Native, K: Partials<2> + Sync, const SIDE: usize> Forms<F, 3> for Slope<K, SIDE> {
    type Out = F;
    const TOTAL: bool = true;
    const WRAPS: bool = false;
    const INLINE: bool = true;
    #[inline]
    fn scalar(&self, [x, y, r]: [F; 3]) -> F { self.at([x, y], r) }
    #[inline]
    fn judged_scalar(&self, x: [F; 3]) -> (F, bool) { (self.scalar(x), true) }
    fn slice(&self, [x, y, r]: [Arg<'_, F>; 3], n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        crate::slice::triples_into(x, y, r, n, out, |x, y, r| (self.at([x, y], r), true))
    }
}
