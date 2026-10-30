//! Formulas that rustymath's kernels compute and basedpl's JIT compiles, each written once over the operations it uses (`Ops`): tolerant closeness, comparison (each item, or through the range of floats equal to one number), floor, ceiling and the residue; Mathish division, sign, the reciprocal and `π`'s two forms; and the powers that `Exponent` routes off the general route, with the square root they take for halves. `Floats` runs a recipe on floats, as the kernels do. A compiler's `Ops` records the same operations as steps, so both routes round alike, step for step.
use crate::{Exponent, FUSED, Float, float::sealed::Core, tolerant::Band};

/// An elementwise operation of one value that a recipe uses, named by id: an `Ops` implementor computes each in one match (`Ops::unary`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Unary {
    Negate, Magnitude, Sqrt, Floor, Ln, Sin, Cos, Sinh, Cosh,
    /// √(1 - x²).
    Sqrt1mx2,
    /// √(1 + x²).
    Sqrt1px2,
    /// ψ(x).
    Digamma,
    /// e^x.
    Exp,
    /// ln |Γ(x)|.
    Lgamma,
}

/// An elementwise operation of two values that a recipe uses, named by id (`Ops::binary`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Binary {
    Add, Subtract, Multiply, Divide,
    /// `x` where `x > y`, otherwise `y`: `y` where either is NaN.
    Larger,
    /// `x` where `x < y`, otherwise `y`: `y` where either is NaN.
    Smaller,
    /// ∂C(n, k)/∂n, with `n` first.
    BinomDn,
    /// ∂C(n, k)/∂k, with `n` first.
    BinomDk,
    /// ψ⁽ⁿ⁾(x), the polygamma function, with its whole order `n` first.
    Polygamma,
}

/// A comparison that a recipe uses, which gives a truth value (`Ops::compare`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cmp { Less, LessEqual, Equal }

/// The named methods of `Ops`, each the operation of its id.
macro_rules! named {
    (unary: $($u:ident $uid:ident),+; binary: $($b:ident $bid:ident),+; compare: $($c:ident $cid:ident),+) => {
        $( #[inline(always)] fn $u(&self, x: Self::V) -> Self::V { self.unary(Unary::$uid, x) } )+
        $( #[inline(always)] fn $b(&self, x: Self::V, y: Self::V) -> Self::V { self.binary(Binary::$bid, x, y) } )+
        $( #[inline(always)] fn $c(&self, x: Self::V, y: Self::V) -> Self::B { self.compare(Cmp::$cid, x, y) } )+
    };
}

/// The operations that a recipe uses, on values `V` and truth values `B`, at float type `F`. Each rounds as IEEE's operation, or rustymath's function of its name, does at `F`'s width. An implementor computes the operations by id, in `unary`, `binary` and `compare`, and recipes call them by their names.
pub trait Ops<F: Float> {
    /// A value.
    type V: Copy;
    /// A truth value.
    type B: Copy;
    /// The number `x`.
    fn number(&self, x: F) -> Self::V;
    fn unary(&self, op: Unary, x: Self::V) -> Self::V;
    fn binary(&self, op: Binary, x: Self::V, y: Self::V) -> Self::V;
    fn compare(&self, c: Cmp, x: Self::V, y: Self::V) -> Self::B;
    /// `x·y + z`, rounded once where the target fuses multiply-add (`FUSED`), as rustymath's `mla` gives it.
    fn multiply_add(&self, x: Self::V, y: Self::V, z: Self::V) -> Self::V;
    fn and(&self, x: Self::B, y: Self::B) -> Self::B;
    fn or(&self, x: Self::B, y: Self::B) -> Self::B;
    fn not(&self, x: Self::B) -> Self::B;
    /// `x` where `m` holds, otherwise `y`.
    fn select(&self, m: Self::B, x: Self::V, y: Self::V) -> Self::V;
    /// 1 where `m` holds, otherwise 0.
    #[inline(always)]
    fn one_where(&self, m: Self::B) -> Self::V { self.select(m, self.number(F::one()), self.number(F::zero())) }
    /// `v` where it's one number that every item shares, known before any item is read, such as the 3 of `x^3`. A recipe can pick a cheaper formula by it. `None` where `v` varies by item.
    fn constant(&self, _v: Self::V) -> Option<F> { None }
    /// The sign of `x`, as [`sign`] gives it.
    #[inline(always)]
    fn sign(&self, x: Self::V) -> Self::V where Self: Sized { sign(self, x) }
    /// `1 ÷ x`, as [`reciprocal`] gives it.
    #[inline(always)]
    fn reciprocal(&self, x: Self::V) -> Self::V where Self: Sized { reciprocal(self, x) }
    /// `x` raised to `k`, a constant that `Exponent::of` routes off the general route, as [`power`] gives it.
    #[inline(always)]
    fn power(&self, k: F, x: Self::V) -> Self::V where Self: Sized { power(self, Exponent::of(k.wide()), x).expect("an exponent off the general route") }
    named! {
        unary: negate Negate, magnitude Magnitude, sqrt Sqrt, floor Floor, ln Ln, sin Sin, cos Cos, sinh Sinh, cosh Cosh, sqrt1mx2 Sqrt1mx2, sqrt1px2 Sqrt1px2, digamma Digamma, exp Exp, lgamma Lgamma;
        binary: add Add, subtract Subtract, multiply Multiply, divide Divide, larger Larger, smaller Smaller, binom_dn BinomDn, binom_dk BinomDk, polygamma Polygamma;
        compare: less Less, less_equal LessEqual, equal Equal
    }
}

/// `Ops` on floats themselves, as a kernel computes. Truth values combine through `&` and `|`, so a loop of recipes has no branch and vectorizes.
#[derive(Clone, Copy, Debug, Default)]
pub struct Floats;

impl<F: Float> Ops<F> for Floats {
    type V = F;
    type B = bool;
    #[inline(always)]
    fn number(&self, x: F) -> F { x }
    #[inline(always)]
    fn unary(&self, op: Unary, x: F) -> F {
        match op {
            Unary::Negate => -x, Unary::Magnitude => x.abs(), Unary::Sqrt => x.sqrt(), Unary::Floor => x.floor(),
            Unary::Ln => crate::ln(x), Unary::Sin => crate::sin(x), Unary::Cos => crate::cos(x), Unary::Sinh => crate::sinh(x), Unary::Cosh => crate::cosh(x),
            Unary::Sqrt1mx2 => crate::sqrt1mx2(x), Unary::Sqrt1px2 => crate::sqrt1px2(x), Unary::Digamma => crate::digamma(x), Unary::Exp => crate::exp(x), Unary::Lgamma => crate::lgamma(x),
        }
    }
    #[inline(always)]
    fn binary(&self, op: Binary, x: F, y: F) -> F {
        match op {
            Binary::Add => x + y, Binary::Subtract => x - y, Binary::Multiply => x * y, Binary::Divide => x / y,
            Binary::Larger => if x > y { x } else { y }, Binary::Smaller => if x < y { x } else { y },
            Binary::BinomDn => crate::binom_dn(x, y), Binary::BinomDk => crate::binom_dk(x, y), Binary::Polygamma => crate::polygamma(x, y),
        }
    }
    #[inline(always)]
    fn compare(&self, c: Cmp, x: F, y: F) -> bool { match c { Cmp::Less => x < y, Cmp::LessEqual => x <= y, Cmp::Equal => x == y } }
    #[inline(always)]
    fn multiply_add(&self, x: F, y: F, z: F) -> F { F::from_working(x.working().mla::<FUSED>(y.working(), z.working())) }
    #[inline(always)]
    fn and(&self, x: bool, y: bool) -> bool { x & y }
    #[inline(always)]
    fn or(&self, x: bool, y: bool) -> bool { x | y }
    #[inline(always)]
    fn not(&self, x: bool) -> bool { !x }
    #[inline(always)]
    fn select(&self, m: bool, x: F, y: F) -> F { if m { x } else { y } }
}

/// Whether `x` and `y` lie within tolerance `t` of each other: `|x-y| < t·max(|x|, |y|)`. With a strict `<`, an infinity is close to no number without a finiteness test, because `inf < inf` is false. NaN is close to nothing.
#[inline(always)]
pub fn close<F: Float, O: Ops<F>>(o: &O, t: F, x: O::V, y: O::V) -> O::B {
    let gap = o.subtract(x, y);
    let gap = o.magnitude(gap);
    let (a, b) = (o.magnitude(x), o.magnitude(y));
    let size = o.larger(a, b);
    let t = o.number(t);
    let bound = o.multiply(t, size);
    o.less(gap, bound)
}

/// Whether `x` equals `y`: `x == y`, or `close`, which gives whether they're close. Each of these compositions runs `close` after its plain comparison: in the kernels' loops, that order is what lets LLVM vectorize 16 items at a time rather than 8.
#[inline(always)]
pub fn equal<F: Float, O: Ops<F>>(o: &O, x: O::V, y: O::V, close: impl FnOnce(&O) -> O::B) -> O::B {
    let same = o.equal(x, y);
    let close = close(o);
    o.or(same, close)
}

/// Whether `x` is below `y` and doesn't equal it, where `close` gives whether they're close. `x < y` already implies `x != y`, so only `close` is tested.
#[inline(always)]
pub fn less<F: Float, O: Ops<F>>(o: &O, x: O::V, y: O::V, close: impl FnOnce(&O) -> O::B) -> O::B {
    let below = o.less(x, y);
    let close = close(o);
    let apart = o.not(close);
    o.and(below, apart)
}

/// Whether `x` is below `y` or equals it, where `close` gives whether they're close. `x <= y` already holds where `x == y`, so only `close` is added.
#[inline(always)]
pub fn less_equal<F: Float, O: Ops<F>>(o: &O, x: O::V, y: O::V, close: impl FnOnce(&O) -> O::B) -> O::B {
    let below = o.less_equal(x, y);
    let close = close(o);
    o.or(below, close)
}

/// The floor of `y` within tolerance `t`. `y` equals the whole number above it when raising `y` by `t·|y|` reaches that number. Capping the raise just below a half keeps a whole number from reaching the next one where `t·|y|` exceeds a half. Where floats are 1 or more apart, the sum rounds back to `y`. A zero result is `0`, since `-0` plus a zero raise is `0`.
#[inline(always)]
pub fn floor<F: Float, O: Ops<F>>(o: &O, t: F, y: O::V) -> O::V {
    let size = o.magnitude(y);
    let t = o.number(t);
    let raise = o.multiply(t, size);
    let most = o.number(F::narrow(0.5).next_down());
    let raise = o.smaller(raise, most);
    let raised = o.add(y, raise);
    o.floor(raised)
}

/// The exact floor of `y`. Adding 0 turns IEEE's `-0` into `0`.
#[inline(always)]
pub fn exact_floor<F: Float, O: Ops<F>>(o: &O, y: O::V) -> O::V {
    let whole = o.floor(y);
    let zero = o.number(F::zero());
    o.add(whole, zero)
}

/// The ceiling of `y` through `floor`, the floor at the same tolerance: `0 - floor(-y)`.
#[inline(always)]
pub fn ceil<F: Float, O: Ops<F>>(o: &O, y: O::V, floor: impl FnOnce(&O, O::V) -> O::V) -> O::V {
    let negated = o.negate(y);
    let whole = floor(o, negated);
    let zero = o.number(F::zero());
    o.subtract(zero, whole)
}

/// `y` modulo `x`: `y - x·q`, for `q` the floor of `y÷x` that `floor` gives, rounded once where the target fuses multiply-add. Within a tolerance `t`, a product `x·q` that equals `y` leaves no remainder. `x` = 0 gives `y`. At no tolerance, where `y÷x` passes the float's precision, the rounded quotient has lost its low digits: `tolerant::Residue` recomputes those items.
#[inline(always)]
pub fn residue<F: Float, O: Ops<F>>(o: &O, t: Option<F>, x: O::V, y: O::V, floor: impl FnOnce(&O, O::V) -> O::V) -> O::V {
    let quotient = o.divide(y, x);
    let q = floor(o, quotient);
    let minus = o.negate(x);
    let r = o.multiply_add(minus, q, y);
    let r = match t {
        Some(t) => {
            let product = o.multiply(x, q);
            let whole = equal(o, product, y, |o| close(o, t, product, y));
            let zero = o.number(F::zero());
            o.select(whole, zero, r)
        }
        None => r,
    };
    let zero = o.number(F::zero());
    let none = o.equal(x, zero);
    o.select(none, y, r)
}

/// The square root of `y`, with the values that `pow` gives `y^0.5` at -0 and -∞: 0 and ∞.
#[inline(always)]
pub fn root2<F: Float, O: Ops<F>>(o: &O, y: O::V) -> O::V {
    let root = o.sqrt(y);
    let zero = o.number(F::zero());
    let root = o.add(root, zero);
    let low = o.number(F::neg_infinity());
    let infinite = o.equal(y, low);
    let flipped = o.negate(y);
    o.select(infinite, flipped, root)
}

/// Whether `v` lies in `band` against the range `lo` to `hi` of floats equal to one number (`Tolerant::range`), by one or two plain comparisons.
#[inline(always)]
pub fn in_band<F: Float, O: Ops<F>>(o: &O, band: Band, lo: O::V, hi: O::V, v: O::V) -> O::B {
    match band {
        Band::Inside | Band::Outside => {
            let (above, below) = (o.less_equal(lo, v), o.less_equal(v, hi));
            let inside = o.and(above, below);
            if matches!(band, Band::Outside) { o.not(inside) } else { inside }
        }
        Band::Below => o.less(v, lo),
        Band::NotAbove => o.less_equal(v, hi),
        Band::Above => o.less(hi, v),
        Band::NotBelow => o.less_equal(lo, v),
    }
}

/// `1 ÷ x`.
#[inline(always)]
pub fn reciprocal<F: Float, O: Ops<F>>(o: &O, x: O::V) -> O::V {
    let one = o.number(F::one());
    o.divide(one, x)
}

/// `π × x`, with π at `F`'s width.
#[inline(always)]
pub fn pi_times<F: Float, O: Ops<F>>(o: &O, x: O::V) -> O::V {
    let pi = o.number(F::narrow(std::f64::consts::PI));
    o.multiply(x, pi)
}

/// `π × (x ÷ y)`, an angle of `x` over `y` turns of half a circle.
#[inline(always)]
pub fn pi_ratio<F: Float, O: Ops<F>>(o: &O, x: O::V, y: O::V) -> O::V {
    let ratio = o.divide(x, y);
    pi_times(o, ratio)
}

/// `x ÷ y`, or `x` where both are zero, as Mathish division gives `0÷0`.
#[inline(always)]
pub fn mathish_divide<F: Float, O: Ops<F>>(o: &O, x: O::V, y: O::V) -> O::V {
    let quotient = o.divide(x, y);
    let zero = o.number(F::zero());
    let (a, b) = (o.equal(x, zero), o.equal(y, zero));
    let both = o.and(a, b);
    o.select(both, x, quotient)
}

/// The sign of `x`: `-1`, `1`, or `x` itself where `x` is a zero or NaN.
#[inline(always)]
pub fn sign<F: Float, O: Ops<F>>(o: &O, x: O::V) -> O::V {
    let (zero, one, minus) = (o.number(F::zero()), o.number(F::one()), o.number(-F::one()));
    let (above, below) = (o.less(zero, x), o.less(x, zero));
    o.select(above, one, o.select(below, minus, x))
}

/// `x` raised to a power that `e` routes off the general route: a whole exponent up to `MULTIPLIED` through [`multiplied`], a larger one through [`whole`], and halves as [`multiplied`] of `root2` of `x`. `None` for `Exponent::General`.
#[inline(always)]
pub fn power<F: Float, O: Ops<F>>(o: &O, e: Exponent, x: O::V) -> Option<O::V> {
    match e {
        Exponent::Whole { m, negative } if m > crate::pow::MULTIPLIED => Some(whole(o, m, negative, x)),
        Exponent::Whole { m, negative } => Some(multiplied(o, m, negative, x)),
        Exponent::Halves { m, negative } => Some(multiplied(o, m, negative, root2(o, x))),
        Exponent::General => None,
    }
}

/// x^m, or x^-m where `negative`, for a whole m up to `MULTIPLIED`: single words, squaring from m's lowest set bit, so x^6 is x²·x⁴, then the reciprocal for a negative power. A rounding at the step that gives x^k carries into x^m about m/k times. x^7, x^8, x^-5 to x^-8 and, through halves, 1/√x^3 pass 4 ulps (rustymath's accuracy page). A result that the float holds exactly comes out exactly. Where x^m overflows, x^-m is 0, though the true result can be a subnormal.
#[inline(always)]
fn multiplied<F: Float, O: Ops<F>>(o: &O, m: u32, negative: bool, x: O::V) -> O::V {
    let (mut base, mut k) = (x, m);
    while k & 1 == 0 {
        base = o.multiply(base, base);
        k >>= 1;
    }
    let mut r = base;
    while k > 1 {
        k >>= 1;
        base = o.multiply(base, base);
        if k & 1 == 1 { r = o.multiply(r, base); }
    }
    if negative { reciprocal(o, r) } else { r }
}

/// A power that [`whole`] has part computed: `h + l` in double words, or `h` alone where `l` is `None`.
pub(crate) type Word<V> = (V, Option<V>);

/// x^m, or x^-m where `negative`, for a whole m from `MULTIPLIED` + 1 to `COMPENSATED`, which `Exponent::of` routes here only where the target fuses multiply-add. It raises [`base`] by squaring and multiplying from m's highest set bit, with a [`step`] for each lower bit. A result that the float holds exactly comes out exactly. `Pow`'s slice form writes the steps out for each bit length of m, so that its loop vectorizes.
#[inline(always)]
pub(crate) fn whole<F: Float, O: Ops<F>>(o: &O, m: u32, negative: bool, x: O::V) -> O::V {
    let (b, mut error) = base(o, negative, x);
    let mut w = (b, None);
    for i in (0..u32::BITS - 1 - m.leading_zeros()).rev() { w = step(o, w, b, m, i, &mut error); }
    w.0
}

/// The number that [`whole`] raises to m: x, or q = 1/x with its relative error ε = 1 - x·q where `negative`, since 1/x is q·(1 + ε) to within ε².
#[inline(always)]
pub(crate) fn base<F: Float, O: Ops<F>>(o: &O, negative: bool, x: O::V) -> (O::V, Option<O::V>) {
    if !negative { return (x, None); }
    let q = reciprocal(o, x);
    let one = o.number(F::one());
    (q, Some(o.multiply_add(o.negate(x), q, one)))
}

/// The steps of bit position `i` of [`whole`]'s m: `w` squared, then times the base `b` where that bit is set.
#[inline(always)]
pub(crate) fn step<F: Float, O: Ops<F>>(o: &O, w: Word<O::V>, b: O::V, m: u32, i: u32, error: &mut Option<O::V>) -> Word<O::V> {
    let w = times(o, w, None, m, i, error);
    if (m >> i) & 1 == 1 { times(o, w, Some(b), m, i, error) } else { w }
}

/// One step of [`whole`] at bit position `i`: `w` squared, or times `b` for `by` = `Some(b)`. A rounding at the step that gives x^k carries into x^m about m/k times. Each step at a position above 0 runs in double words, taking its product's rounding error exactly from a fused multiply-add. The first step after them rounds the double word to one word. That step also adds the base's error `error`, times m and halved once for each square from that step on. This turns a power of q into a power of 1/x. The steps after it run in single words. The roundings left keep x^m within 4 ulps.
#[inline(always)]
fn times<F: Float, O: Ops<F>>(o: &O, (h, l): Word<O::V>, by: Option<O::V>, m: u32, i: u32, error: &mut Option<O::V>) -> Word<O::V> {
    let y = by.unwrap_or(h);
    // The coefficient of l in the product: 2h for a square, b for a product with b.
    let twice = |o: &O| by.unwrap_or_else(|| o.add(h, h));
    if i > 0 {
        let p = o.multiply(h, y);
        let e = o.multiply_add(h, y, o.negate(p));
        return (p, Some(match l { Some(l) => o.multiply_add(twice(o), l, e), None => e }));
    }
    let squares = i + by.is_none() as u32;
    let fold = error.take().map(|e| o.multiply(e, o.number(F::narrow(m as f64 / (1u64 << squares) as f64))));
    let l = match (l, fold) {
        (None, None) => return (o.multiply(h, y), None),
        (Some(l), None) => l,
        (None, Some(c)) => o.multiply(h, c),
        (Some(l), Some(c)) => o.multiply_add(h, c, l),
    };
    let k = o.multiply(twice(o), l);
    // k is infinite or NaN only where h is, or where h·y overflows. h·y alone is the power there.
    let k = o.select(finite(o, k), k, o.number(F::zero()));
    (o.multiply_add(h, y, k), None)
}

/// Whether `v` is finite: `v - v` is 0 there, and NaN at an infinity or NaN.
#[inline(always)]
fn finite<F: Float, O: Ops<F>>(o: &O, v: O::V) -> O::B { o.equal(o.subtract(v, v), o.number(F::zero())) }
