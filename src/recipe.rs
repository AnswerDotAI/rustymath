//! Formulas that rustymath's kernels compute and basedpl's JIT compiles, each written once over the operations it uses (`Ops`): tolerant closeness, comparison (each item, or through the range of floats equal to one number), floor, ceiling and the residue; Mathish division, sign, the reciprocal and `π`'s two forms; and the powers that `Exponent` routes off the general route, with the square root they take for halves. `Floats` runs a recipe on floats, as the kernels do. A compiler's `Ops` records the same operations as steps, so both routes round alike, step for step.
use crate::{Exponent, Float, float::sealed::Core, tolerant::Band};
mod integer;
pub use crate::arith::Overflowing;
pub use integer::IntegerOps;

/// An elementwise operation of one value that a recipe uses, named by id: an `Ops` implementor computes each in one match (`Ops::unary`).
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum Unary {
    Negate,
    Magnitude,
    Sqrt,
    Floor,
    Ln,
    Sin,
    Cos,
    Sinh,
    Cosh,
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
    /// 1 - x: the negation of a truth value held as 0 or 1, as a Boolean is.
    Not,
}

/// An elementwise operation of two values that a recipe uses, named by id (`Ops::binary`).
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum Binary {
    Add,
    Subtract,
    Multiply,
    Divide,
    /// Runtime power retains the same exponent dispatch as the native kernel.
    Power,
    /// `x` where `x > y`, otherwise `y`: `y` where either is NaN.
    Larger,
    /// `x` where `x < y`, otherwise `y`: `y` where either is NaN.
    Smaller,
    /// C(n, k), the binomial coefficient, with `n` first.
    Binom,
    /// ∂C(n, k)/∂n, with `n` first.
    BinomDn,
    /// ∂C(n, k)/∂k, with `n` first.
    BinomDk,
    /// ψ⁽ⁿ⁾(x), the polygamma function, with its whole order `n` first.
    Polygamma,
    /// The bitwise and of two integers, which only integer values take (`Ops::integral`).
    BitAnd,
}

/// A comparison that a recipe uses, which gives a truth value (`Ops::compare`).
#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum Cmp { Less, LessEqual, Equal }

#[derive(Clone, Copy, Debug, PartialEq, Hash, Eq)]
pub enum Integer { Add, Subtract, And, Or, Xor, ShiftLeft, ShiftRight, SignedShiftRight }

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
    /// Integer bits at the resolved working width.
    type I: Copy;
    fn width(&self) -> crate::buffer::FloatWidth;
    fn fused(&self) -> bool;
    /// A binary64 temporary for short mixed-width reductions.
    type W: Copy;
    fn widen(&self, x: Self::V) -> Self::W;
    fn wide_number(&self, x: f64) -> Self::W;
    fn wide_multiply_add(&self, x: Self::W, y: Self::W, z: Self::W) -> Self::W;
    fn narrow(&self, x: Self::W) -> Self::V;
    fn integer(&self, x: i64) -> Self::I;
    fn integer_op(&self, op: Integer, x: Self::I, y: Self::I) -> Self::I;
    fn integer_compare(&self, op: Cmp, x: Self::I, y: Self::I) -> Self::B;
    fn integer_select(&self, test: Self::B, x: Self::I, y: Self::I) -> Self::I;
    fn to_integer(&self, x: Self::V) -> Self::I;
    fn from_integer(&self, x: Self::I) -> Self::V;
    fn to_bits(&self, x: Self::V) -> Self::I;
    fn from_bits(&self, x: Self::I) -> Self::V;
    /// The number `x`.
    fn number(&self, x: F) -> Self::V;
    fn unary(&self, op: Unary, x: Self::V) -> Self::V;
    fn binary(&self, op: Binary, x: Self::V, y: Self::V) -> Self::V;
    fn compare(&self, c: Cmp, x: Self::V, y: Self::V) -> Self::B;
    /// `x·y + z`, rounded once where the target fuses multiply-add (`FUSED`), as rustymath's `mla` gives it.
    fn multiply_add(&self, x: Self::V, y: Self::V, z: Self::V) -> Self::V;
    fn overflow(&self, op: Binary, x: Self::V, y: Self::V) -> Self::B;
    fn and(&self, x: Self::B, y: Self::B) -> Self::B;
    fn or(&self, x: Self::B, y: Self::B) -> Self::B;
    fn not(&self, x: Self::B) -> Self::B;
    /// `x` where `m` holds, otherwise `y`. `∂`'s recorder runs a recipe on whole values, where `m` and each array case must have one layout: build them from the same arguments.
    fn select(&self, m: Self::B, x: Self::V, y: Self::V) -> Self::V;
    /// 1 where `m` holds, otherwise 0.
    #[inline(always)]
    fn one_where(&self, m: Self::B) -> Self::V { self.select(m, self.number(F::one()), self.number(F::zero())) }
    /// `v` where it's one number that every item shares, known before any item is read, such as the 3 of `x^3`. A recipe can pick a cheaper formula by it. `None` where `v` varies by item.
    #[inline(always)]
    fn constant(&self, _v: Self::V) -> Option<F> { None }
    /// Whether values are integers, as a compiler's values at an integer type are. A recipe for an integer object asks before taking an integer-only step (`Binary::BitAnd`). Floats never are.
    #[inline(always)]
    fn integral(&self) -> bool { false }
    /// Captured tolerance; recording backends return its runtime setting slot.
    #[inline(always)]
    fn tolerance(&self, fallback: F) -> Self::V { self.number(fallback) }
    /// The sign of `x`, as [`sign`] gives it.
    #[inline(always)]
    fn sign(&self, x: Self::V) -> Self::V { sign(self, x) }
    /// `1 ÷ x`, as [`reciprocal`] gives it.
    #[inline(always)]
    fn reciprocal(&self, x: Self::V) -> Self::V { reciprocal(self, x) }
    /// `x` raised to `k`, a constant that `Exponent::of` routes off the general route, as [`power`] gives it.
    #[inline(always)]
    fn power(&self, k: F, x: Self::V) -> Self::V { power(self, Exponent::of(k.wide()), x).expect("an exponent off the general route") }
    /// The real part of `x^y` through the existing power formula.
    #[inline(always)]
    fn real_power(&self, x: Self::V, y: Self::V) -> Self::V { crate::pow::real_formula(self, x, y) }
    /// The partial in `x` of `x^y = r`; `halves` permits constant half-integer powers.
    #[inline(always)]
    fn power_slope(&self, x: Self::V, y: Self::V, r: Self::V, halves: bool) -> Self::V { crate::derivative::power_slope(self, x, y, r, halves) }
    named! {
        unary: negate Negate, magnitude Magnitude, sqrt Sqrt, floor Floor, ln Ln, sin Sin, cos Cos, sinh Sinh, cosh Cosh, sqrt1mx2 Sqrt1mx2, sqrt1px2 Sqrt1px2, digamma Digamma, exp Exp, lgamma Lgamma, complement Not;
        binary: add Add, subtract Subtract, multiply Multiply, divide Divide, larger Larger, smaller Smaller, binom Binom, binom_dn BinomDn, binom_dk BinomDk, polygamma Polygamma, bit_and BitAnd;
        compare: less Less, less_equal LessEqual, equal Equal
    }
}

/// Conservative range interpretation of the shared predicates.
pub(crate) struct RangeOps(pub crate::buffer::DType);
impl RangeOps {
    fn compare(op: Cmp, x: crate::facts::Range, y: crate::facts::Range) -> Option<bool> {
        if x.is_empty() || y.is_empty() { return None; }
        match op {
            Cmp::Less if x.hi < y.lo => Some(true), Cmp::Less if x.lo >= y.hi => Some(false), Cmp::LessEqual if x.hi <= y.lo => Some(true),
            Cmp::LessEqual if x.lo > y.hi => Some(false), Cmp::Equal if x.constant().is_some() && x == y => Some(true),
            Cmp::Equal if x.hi < y.lo || y.hi < x.lo => Some(false), _ => None
        }
    }
    fn selected(test: Option<bool>, x: crate::facts::Range, y: crate::facts::Range) -> crate::facts::Range {
        match test { Some(true) => x, Some(false) => y, None => x.join(y) }
    }
}
impl<F: Float> Ops<F> for RangeOps {
    type V = crate::facts::Range;
    type B = Option<bool>;
    type I = crate::facts::Range;
    type W = crate::facts::Range;
    fn width(&self) -> crate::buffer::FloatWidth { crate::operation::numpy_float(self.0) }
    fn fused(&self) -> bool { crate::FUSED }
    fn integral(&self) -> bool { self.0.is_integral() }
    fn widen(&self, _: Self::V) -> Self::W { crate::facts::Range::ANY }
    fn wide_number(&self, x: f64) -> Self::W { crate::facts::Range::of(Some(x)) }
    fn wide_multiply_add(&self, _: Self::W, _: Self::W, _: Self::W) -> Self::W { crate::facts::Range::ANY }
    fn narrow(&self, _: Self::W) -> Self::V { crate::facts::Range::ANY }
    fn integer(&self, x: i64) -> Self::I { crate::facts::Range::of((x.unsigned_abs() <= 1 << 53).then_some(x as f64)) }
    fn integer_op(&self, _: Integer, _: Self::I, _: Self::I) -> Self::I { crate::facts::Range::ANY }
    fn integer_compare(&self, op: Cmp, x: Self::I, y: Self::I) -> Self::B { Self::compare(op, x, y) }
    fn integer_select(&self, test: Self::B, x: Self::I, y: Self::I) -> Self::I { Self::selected(test, x, y) }
    fn to_integer(&self, _: Self::V) -> Self::I { crate::facts::Range::ANY }
    fn from_integer(&self, _: Self::I) -> Self::V { crate::facts::Range::ANY }
    fn to_bits(&self, _: Self::V) -> Self::I { crate::facts::Range::ANY }
    fn from_bits(&self, _: Self::I) -> Self::V { crate::facts::Range::ANY }
    fn number(&self, x: F) -> Self::V { crate::facts::Range::of(Some(x.wide())) }
    fn constant(&self, x: Self::V) -> Option<F> { x.constant().map(F::narrow) }
    fn unary(&self, op: Unary, x: Self::V) -> Self::V { if op == Unary::Magnitude { x.magnitude() } else { crate::facts::Range::ANY } }
    fn binary(&self, _: Binary, _: Self::V, _: Self::V) -> Self::V { crate::facts::Range::ANY }
    fn compare(&self, op: Cmp, x: Self::V, y: Self::V) -> Self::B { Self::compare(op, x, y) }
    fn multiply_add(&self, _: Self::V, _: Self::V, _: Self::V) -> Self::V { crate::facts::Range::ANY }
    /// No overflow (`Some(false)`) where the operation's range, as its formula's facts give it, lies within the computing dtype's values. Past 2⁵³ a bound and the result's ends round, so a result stays strictly inside such a bound.
    fn overflow(&self, op: Binary, x: Self::V, y: Self::V) -> Self::B {
        use crate::facts::Formula;
        let result = match op {
            Binary::Add => crate::Add.range([x, y]), Binary::Subtract => crate::Subtract.range([x, y]), Binary::Multiply => crate::Multiply.range([x, y]),
            _ => return None,
        };
        let fits = crate::facts::Range::of_dtype(self.0);
        let exact = |bound: f64| bound.abs() < 2f64.powi(53);
        let above = if exact(fits.lo) { result.lo >= fits.lo } else { result.lo > fits.lo };
        let below = if exact(fits.hi) { result.hi <= fits.hi } else { result.hi < fits.hi };
        (above && below).then_some(false)
    }
    fn and(&self, x: Self::B, y: Self::B) -> Self::B {
        match (x, y) { (Some(false), _) | (_, Some(false)) => Some(false), (Some(true), Some(true)) => Some(true), _ => None }
    }
    fn or(&self, x: Self::B, y: Self::B) -> Self::B {
        match (x, y) { (Some(true), _) | (_, Some(true)) => Some(true), (Some(false), Some(false)) => Some(false), _ => None }
    }
    fn not(&self, x: Self::B) -> Self::B { x.map(|x| !x) }
    fn select(&self, test: Self::B, x: Self::V, y: Self::V) -> Self::V { Self::selected(test, x, y) }
}

/// Scalar instantiation of the same formulae used by slice kernels and graph lowering.
#[derive(Clone, Copy, Debug, Default)]
pub struct NativeOps<const FUSED: bool> { pub tolerance: Option<f64> }
impl<const FUSED: bool> NativeOps<FUSED> { pub const DEFAULT: Self = Self { tolerance: None }; }
pub type Floats = NativeOps<{ crate::FUSED }>;
#[allow(non_upper_case_globals)]
pub const Floats: Floats = NativeOps::DEFAULT;

impl<F: Float, const FUSED: bool> Ops<F> for NativeOps<FUSED> {
    type V = F;
    type B = bool;
    type I = u64;
    type W = f64;
    #[inline(always)]
    fn fused(&self) -> bool { FUSED }
    #[inline(always)]
    fn widen(&self, x: F) -> f64 { x.wide() }
    #[inline(always)]
    fn wide_number(&self, x: f64) -> f64 { x }
    #[inline(always)]
    fn wide_multiply_add(&self, x: f64, y: f64, z: f64) -> f64 { if FUSED { x.mul_add(y, z) } else { x * y + z } }
    #[inline(always)]
    fn narrow(&self, x: f64) -> F { F::narrow(x) }
    #[inline(always)]
    fn width(&self) -> crate::buffer::FloatWidth { if F::Working::WIDE { crate::buffer::FloatWidth::F64 } else { crate::buffer::FloatWidth::F32 } }
    #[inline(always)]
    fn integer(&self, x: i64) -> u64 { x as u64 }
    #[inline(always)]
    fn integer_op(&self, op: Integer, x: u64, y: u64) -> u64 {
        match op {
            Integer::Add => x.wrapping_add(y), Integer::Subtract => x.wrapping_sub(y), Integer::And => x & y, Integer::Or => x | y, Integer::Xor => x ^ y,
            Integer::ShiftLeft => x.wrapping_shl(y as u32), Integer::ShiftRight => x.wrapping_shr(y as u32),
            Integer::SignedShiftRight => (x as i64).wrapping_shr(y as u32) as u64
        }
    }
    #[inline(always)]
    fn integer_compare(&self, op: Cmp, x: u64, y: u64) -> bool {
        match op { Cmp::Less => (x as i64) < y as i64, Cmp::LessEqual => (x as i64) <= y as i64, Cmp::Equal => x == y }
    }
    #[inline(always)]
    fn integer_select(&self, test: bool, x: u64, y: u64) -> u64 { if test { x } else { y } }
    #[inline(always)]
    fn to_integer(&self, x: F) -> u64 { (x.wide() as i64) as u64 }
    #[inline(always)]
    fn from_integer(&self, x: u64) -> F { F::narrow((x as i64) as f64) }
    #[inline(always)]
    fn to_bits(&self, x: F) -> u64 { if F::Working::WIDE { x.wide().to_bits() } else { (x.wide() as f32).to_bits() as u64 } }
    #[inline(always)]
    fn from_bits(&self, x: u64) -> F { F::narrow(if F::Working::WIDE { f64::from_bits(x) } else { f32::from_bits(x as u32) as f64 }) }
    #[inline(always)]
    fn number(&self, x: F) -> F { x }
    #[inline(always)]
    fn unary(&self, op: Unary, x: F) -> F {
        match op {
            Unary::Negate => -x, Unary::Magnitude => x.abs(), Unary::Sqrt => x.sqrt(), Unary::Floor => x.floor(), Unary::Ln => crate::ln(x),
            Unary::Sin => crate::sin(x), Unary::Cos => crate::cos(x), Unary::Sinh => crate::sinh(x), Unary::Cosh => crate::cosh(x),
            Unary::Sqrt1mx2 => crate::sqrt1mx2(x), Unary::Sqrt1px2 => crate::sqrt1px2(x), Unary::Digamma => crate::digamma(x), Unary::Exp => crate::exp(x),
            Unary::Lgamma => crate::lgamma(x), Unary::Not => F::one() - x
        }
    }
    #[inline(always)]
    fn binary(&self, op: Binary, x: F, y: F) -> F {
        match op {
            Binary::Add => x + y, Binary::Subtract => x - y, Binary::Multiply => x * y, Binary::Divide => x / y, Binary::Larger => { if x > y { x } else { y } }
            Binary::Smaller => { if x < y { x } else { y } } Binary::Binom => crate::binom(x, y), Binary::BinomDn => crate::binom_dn(x, y),
            Binary::Power => crate::pow::formula(self, x, y), Binary::BinomDk => crate::binom_dk(x, y), Binary::Polygamma => crate::polygamma(x, y),
            Binary::BitAnd => unreachable!("floats take no bitwise and: a recipe asks for one only where `Ops::integral` holds")
        }
    }
    #[inline(always)]
    fn compare(&self, c: Cmp, x: F, y: F) -> bool { match c { Cmp::Less => x < y, Cmp::LessEqual => x <= y, Cmp::Equal => x == y } }
    #[inline(always)]
    fn multiply_add(&self, x: F, y: F, z: F) -> F { F::from_working(x.working().mla::<FUSED>(y.working(), z.working())) }
    #[inline(always)]
    fn overflow(&self, _op: Binary, _x: F, _y: F) -> bool { false }
    #[inline(always)]
    fn and(&self, x: bool, y: bool) -> bool { x & y }
    #[inline(always)]
    fn or(&self, x: bool, y: bool) -> bool { x | y }
    #[inline(always)]
    fn not(&self, x: bool) -> bool { !x }
    #[inline(always)]
    fn select(&self, m: bool, x: F, y: F) -> F { if m { x } else { y } }
    #[inline(always)]
    fn constant(&self, value: F) -> Option<F> { Some(value) }
    #[inline(always)]
    fn tolerance(&self, fallback: F) -> F { self.tolerance.map_or(fallback, F::narrow) }
}

/// Whether `x` and `y` lie within tolerance `t` of each other: `|x-y| < t·max(|x|, |y|)`. With a strict `<`, an infinity is close to no number without a finiteness test, because `inf < inf` is false. NaN is close to nothing.
#[inline(always)]
pub fn close<F: Float, O: Ops<F> + ?Sized>(o: &O, t: O::V, x: O::V, y: O::V) -> O::B {
    let gap = o.subtract(x, y);
    let gap = o.magnitude(gap);
    let (a, b) = (o.magnitude(x), o.magnitude(y));
    let size = o.larger(a, b);
    let bound = o.multiply(t, size);
    o.less(gap, bound)
}

/// Whether `x` equals `y`: `x == y`, or `close`, which gives whether they're close. Each of these compositions runs `close` after its plain comparison: in the kernels' loops, that order is what lets LLVM vectorize 16 items at a time rather than 8.
#[inline(always)]
pub fn equal<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V, close: impl FnOnce(&O) -> O::B) -> O::B {
    if o.integral() { return o.equal(x, y); }
    let same = o.equal(x, y);
    let close = close(o);
    o.or(same, close)
}

/// Whether `x` is below `y` and doesn't equal it, where `close` gives whether they're close. `x < y` already implies `x != y`, so only `close` is tested.
#[inline(always)]
pub fn less<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V, close: impl FnOnce(&O) -> O::B) -> O::B {
    if o.integral() { return o.less(x, y); }
    let below = o.less(x, y);
    let close = close(o);
    let apart = o.not(close);
    o.and(below, apart)
}

/// Whether `x` is below `y` or equals it, where `close` gives whether they're close. `x <= y` already holds where `x == y`, so only `close` is added.
#[inline(always)]
pub fn less_equal<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V, close: impl FnOnce(&O) -> O::B) -> O::B {
    if o.integral() { return o.less_equal(x, y); }
    let below = o.less_equal(x, y);
    let close = close(o);
    o.or(below, close)
}

/// The floor of `y` within tolerance `t`. `y` equals the whole number above it when raising `y` by `t·|y|` reaches that number. Capping the raise just below a half keeps a whole number from reaching the next one where `t·|y|` exceeds a half. Where floats are 1 or more apart, the sum rounds back to `y`. A zero result is `0`, since `-0` plus a zero raise is `0`.
#[inline(always)]
pub fn floor<F: Float, O: Ops<F> + ?Sized>(o: &O, t: O::V, y: O::V) -> O::V {
    if o.integral() { return y; }
    let size = o.magnitude(y);
    let raise = o.multiply(t, size);
    let most = o.number(F::narrow(0.5).next_down());
    let raise = o.smaller(raise, most);
    let raised = o.add(y, raise);
    o.floor(raised)
}

#[inline(always)]
pub(crate) fn integer_floor_divide<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V {
    let q = o.divide(x, y);
    let r = o.subtract(x, o.multiply(q, y));
    let zero = o.number(F::zero());
    let lower = o.or(o.and(o.less(r, zero), o.less(zero, y)), o.and(o.less(zero, r), o.less(y, zero)));
    o.subtract(q, o.one_where(lower))
}

/// The exact floor of `y`. Adding 0 turns IEEE's `-0` into `0`.
#[inline(always)]
pub fn exact_floor<F: Float, O: Ops<F> + ?Sized>(o: &O, y: O::V) -> O::V {
    let whole = o.floor(y);
    let zero = o.number(F::zero());
    o.add(whole, zero)
}

/// The ceiling of `y` through `floor`, the floor at the same tolerance: `0 - floor(-y)`.
#[inline(always)]
pub fn ceil<F: Float, O: Ops<F> + ?Sized>(o: &O, y: O::V, floor: impl FnOnce(&O, O::V) -> O::V) -> O::V {
    if o.integral() { return y; }
    let negated = o.negate(y);
    let whole = floor(o, negated);
    let zero = o.number(F::zero());
    o.subtract(zero, whole)
}

/// `y` modulo `x`: `y - x·q`, for `q` the floor of `y÷x` that `floor` gives, rounded once where the target fuses multiply-add. Within a tolerance `t`, a product `x·q` that equals `y` leaves no remainder. `x` = 0 gives `y`. At no tolerance, where `y÷x` passes the float's precision, the rounded quotient has lost its low digits: `tolerant::Residue` recomputes those items.
#[inline(always)]
pub fn residue<F: Float, O: Ops<F> + ?Sized>(o: &O, t: Option<O::V>, x: O::V, y: O::V, floor: impl FnOnce(&O, O::V) -> O::V) -> O::V {
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
pub fn root2<F: Float, O: Ops<F> + ?Sized>(o: &O, y: O::V) -> O::V {
    let root = o.sqrt(y);
    let zero = o.number(F::zero());
    let root = o.add(root, zero);
    let low = o.number(F::neg_infinity());
    let infinite = o.equal(y, low);
    let flipped = o.negate(y);
    o.select(infinite, flipped, root)
}

/// The real part of √y: √y where y is positive, 0 at or below 0, and NaN at NaN, which `larger` keeps.
#[inline(always)]
pub fn real_sqrt<F: Float, O: Ops<F> + ?Sized>(o: &O, y: O::V) -> O::V {
    let zero = o.number(F::zero());
    let clamped = o.larger(zero, y);
    o.sqrt(clamped)
}

/// (sin πx, cos πx), exact at whole and half x: x = n/2 + r with n the whole number nearest 2x and |r| up to ¼, where n mod 4 picks the quadrant.
#[inline(always)]
pub fn sincospi<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> (O::V, O::V) {
    let (half, quarter) = (o.number(F::narrow(0.5)), o.number(F::narrow(0.25)));
    let n = o.floor(o.add(o.add(x, x), half));
    let a = o.multiply(o.subtract(x, o.multiply(n, half)), o.number(F::narrow(core::f64::consts::PI)));
    let (s, k) = (o.sin(a), o.cos(a));
    let q = o.subtract(n, o.multiply(o.number(F::narrow(4.0)), o.floor(o.multiply(n, quarter))));
    let [q0, q1, q2] = [0.0, 1.0, 2.0].map(|m| o.equal(q, o.number(F::narrow(m))));
    // 0 - v negates v exactly, and gives +0, never ¯0, at an exact half step.
    let zero = o.number(F::zero());
    let (ns, nk) = (o.subtract(zero, s), o.subtract(zero, k));
    (o.select(q0, s, o.select(q1, k, o.select(q2, ns, nk))), o.select(q0, k, o.select(q1, ns, o.select(q2, nk, s))))
}

/// Whether `v` lies in `band` against the range `lo` to `hi` of floats equal to one number (`Tolerant::range`), by one or two plain comparisons.
#[inline(always)]
pub fn in_band<F: Float, O: Ops<F> + ?Sized>(o: &O, band: Band, lo: O::V, hi: O::V, v: O::V) -> O::B {
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
pub fn reciprocal<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V { let one = o.number(F::one()); o.divide(one, x) }

/// `π × x`, with π at `F`'s width.
#[inline(always)]
pub fn pi_times<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V { let pi = o.number(F::narrow(std::f64::consts::PI)); o.multiply(x, pi) }

/// `π × (x ÷ y)`, an angle of `x` over `y` turns of half a circle.
#[inline(always)]
pub fn pi_ratio<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V { let ratio = o.divide(x, y); pi_times(o, ratio) }

/// `x ÷ y`, or `x` where both are zero, as Mathish division gives `0÷0`.
#[inline(always)]
pub fn mathish_divide<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> O::V {
    let quotient = o.divide(x, y);
    let zero = o.number(F::zero());
    let (a, b) = (o.equal(x, zero), o.equal(y, zero));
    let both = o.and(a, b);
    o.select(both, x, quotient)
}

/// The sign of `x`: `-1`, `1`, or `x` itself where `x` is a zero or NaN.
#[inline(always)]
pub fn sign<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> O::V {
    let (zero, one, minus) = (o.number(F::zero()), o.number(F::one()), o.number(-F::one()));
    let (above, below) = (o.less(zero, x), o.less(x, zero));
    o.select(above, one, o.select(below, minus, x))
}

/// `x` raised to a power that `e` routes off the general route: a whole exponent up to `MULTIPLIED` through [`multiplied`], a larger one through [`whole`], and halves as [`multiplied`] of `root2` of `x`. `None` for `Exponent::General`.
#[inline(always)]
pub fn power<F: Float, O: Ops<F> + ?Sized>(o: &O, e: Exponent, x: O::V) -> Option<O::V> {
    match e {
        Exponent::Whole { m, negative } if !o.integral() && m > crate::pow::MULTIPLIED => Some(whole(o, m, negative, x)),
        Exponent::Whole { m, negative } => Some(multiplied(o, m, negative, x)),
        Exponent::Halves { m, negative } => Some(multiplied(o, m, negative, root2(o, x))), Exponent::General => None
    }
}

/// x^m, or x^-m where `negative`, for a whole m up to `MULTIPLIED`: single words, squaring from m's lowest set bit, so x^6 is x²·x⁴, then the reciprocal for a negative power. A rounding at the step that gives x^k carries into x^m about m/k times. x^7, x^8, x^-5 to x^-8 and, through halves, 1/√x^3 pass 4 ulps (rustymath's accuracy page). A result that the float holds exactly comes out exactly. Where x^m overflows, x^-m is 0, though the true result can be a subnormal.
#[inline(always)]
fn multiplied<F: Float, O: Ops<F> + ?Sized>(o: &O, m: u32, negative: bool, x: O::V) -> O::V {
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
pub(crate) fn whole<F: Float, O: Ops<F> + ?Sized>(o: &O, m: u32, negative: bool, x: O::V) -> O::V {
    let (b, mut error) = base(o, negative, x);
    let mut w = (b, None);
    for i in (0..u32::BITS - 1 - m.leading_zeros()).rev() { w = step(o, w, b, m, i, &mut error); }
    w.0
}

/// The number that [`whole`] raises to m: x, or q = 1/x with its relative error ε = 1 - x·q where `negative`, since 1/x is q·(1 + ε) to within ε².
#[inline(always)]
pub(crate) fn base<F: Float, O: Ops<F> + ?Sized>(o: &O, negative: bool, x: O::V) -> (O::V, Option<O::V>) {
    if !negative { return (x, None); }
    let q = reciprocal(o, x);
    let one = o.number(F::one());
    (q, Some(o.multiply_add(o.negate(x), q, one)))
}

/// The steps of bit position `i` of [`whole`]'s m: `w` squared, then times the base `b` where that bit is set.
#[inline(always)]
pub(crate) fn step<F: Float, O: Ops<F> + ?Sized>(o: &O, w: Word<O::V>, b: O::V, m: u32, i: u32, error: &mut Option<O::V>) -> Word<O::V> {
    let w = times(o, w, None, m, i, error);
    if (m >> i) & 1 == 1 { times(o, w, Some(b), m, i, error) } else { w }
}

/// One step of [`whole`] at bit position `i`: `w` squared, or times `b` for `by` = `Some(b)`. A rounding at the step that gives x^k carries into x^m about m/k times. Each step at a position above 0 runs in double words, taking its product's rounding error exactly from a fused multiply-add. The first step after them rounds the double word to one word. That step also adds the base's error `error`, times m and halved once for each square from that step on. This turns a power of q into a power of 1/x. The steps after it run in single words. The roundings left keep x^m within 4 ulps.
#[inline(always)]
fn times<F: Float, O: Ops<F> + ?Sized>(o: &O, (h, l): Word<O::V>, by: Option<O::V>, m: u32, i: u32, error: &mut Option<O::V>) -> Word<O::V> {
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
        (None, None) => return (o.multiply(h, y), None), (Some(l), None) => l, (None, Some(c)) => o.multiply(h, c),
        (Some(l), Some(c)) => o.multiply_add(h, c, l)
    };
    let k = o.multiply(twice(o), l);
    // k is infinite or NaN only where h is, or where h·y overflows. h·y alone is the power there.
    let k = o.select(finite(o, k), k, o.number(F::zero()));
    (o.multiply_add(h, y, k), None)
}

/// Whether `v` is finite: `v - v` is 0 there, and NaN at an infinity or NaN.
#[inline(always)]
fn finite<F: Float, O: Ops<F> + ?Sized>(o: &O, v: O::V) -> O::B { o.equal(o.subtract(v, v), o.number(F::zero())) }

#[inline(always)]
pub fn horner<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V, coefficients: &[f64]) -> O::V {
    let mut result = o.number(F::narrow(coefficients[0]));
    for &coefficient in &coefficients[1..] { result = o.multiply_add(result, x, o.number(F::narrow(coefficient))); }
    result
}

#[inline(always)]
pub fn ln2<F: Float, O: Ops<F> + ?Sized>(o: &O) -> (O::V, O::V) {
    let (hi, lo) = if o.width() == crate::buffer::FloatWidth::F64 {
        (0.69314718055966295651160180568695068359375, 0.28235290563031577122588448175013436025525412068e-12)
    } else { (0.693145751953125, 1.428606765330187e-06) };
    (o.number(F::narrow(hi)), o.number(F::narrow(lo)))
}

#[inline(always)]
pub fn rounding_shift<F: Float, O: Ops<F> + ?Sized>(o: &O) -> O::V {
    o.number(F::narrow(if o.width() == crate::buffer::FloatWidth::F64 { 6755399441055744.0 } else { 12582912.0 }))
}

/// 2^n from z = rounding_shift + n, where n lies within the normal exponent range.
#[inline(always)]
pub fn pow2<F: Float, O: Ops<F> + ?Sized>(o: &O, z: O::V) -> O::V {
    let (mantissa, bias) = if o.width() == crate::buffer::FloatWidth::F64 { (52, 1023) } else { (23, 127) };
    let bits = o.integer_op(Integer::Add, o.to_bits(z), o.integer(bias));
    o.from_bits(o.integer_op(Integer::ShiftLeft, bits, o.integer(mantissa)))
}

/// y times 2^n, with one rounding when the result is subnormal.
#[inline(always)]
pub fn ldexp<F: Float, O: Ops<F> + ?Sized>(o: &O, y: O::V, n: O::V) -> O::V {
    let shift = rounding_shift(o);
    let half = o.multiply_add(n, o.number(F::narrow(0.5)), shift);
    let rest = o.add(o.subtract(n, o.subtract(half, shift)), shift);
    o.multiply(o.multiply(y, pow2(o, half)), pow2(o, rest))
}

/// Positive normal x = m times 2^e, with m in [0.75, 1.5].
#[inline(always)]
pub fn split<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> (O::V, O::V) {
    let (mantissa, bias) = if o.width() == crate::buffer::FloatWidth::F64 { (52, 1023) } else { (23, 127) };
    let scaled = o.multiply(x, o.number(F::narrow(1.0 / 0.75)));
    let exponent = o.integer_op(Integer::ShiftRight, o.to_bits(scaled), o.integer(mantissa));
    let exponent = o.integer_op(Integer::And, exponent, o.integer(2 * bias + 1));
    let unbiased = o.integer_op(Integer::Subtract, exponent, o.integer(bias));
    let offset = o.integer_op(Integer::ShiftLeft, unbiased, o.integer(mantissa));
    (o.from_bits(o.integer_op(Integer::Subtract, o.to_bits(x), offset)), o.from_integer(unbiased))
}

/// Positive x = m times 2^e, including subnormals.
#[inline(always)]
pub fn frexp<F: Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> (O::V, O::V) {
    let smallest = if o.width() == crate::buffer::FloatWidth::F64 { f64::MIN_POSITIVE } else { f32::MIN_POSITIVE as f64 };
    let small = o.less(x, o.number(F::narrow(smallest)));
    let scale = o.select(small, o.number(F::narrow(18446744073709551616.0)), o.number(F::one()));
    let (m, e) = split(o, o.multiply(x, scale));
    (m, o.subtract(e, o.select(small, o.number(F::narrow(64.0)), o.number(F::zero()))))
}
