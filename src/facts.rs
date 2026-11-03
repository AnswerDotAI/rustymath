//! Shared function definitions for numerical frontends and graph planners. Each [`Formula`] row owns its kind and character policies, numerical body, validity and repair predicates, ranges, identities, partial derivatives and exact cases. [`Kernel`](crate::operation::Kernel) erases the arity for execution. [`facts!`](crate::facts) lists typed product pairs and Python names.
use crate::buffer::{DType, FloatWidth, View};
use crate::pow::parity;
use crate::float::sealed::Core;
use crate::recipe::Ops;
use crate::reduce::{Maximum, Minimum, Reduce};
use crate::Flipped;
use crate::tolerant::{Tolerance, Tolerant, TolerantComplex, lcm_from_gcd};
use std::f64::consts::PI;
use num_complex::Complex;
use num_traits::{One, Signed, Zero};

/// The values other than NaN that the items of an argument or a result can take: each lies from `lo` to `hi`. ¯0 and +0 count as one value, because the sign of a zero isn't promised and can differ by route. Any range can also hold NaN. A rule's range is sound where it holds every result that the object's kernels give, within their accuracy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range { pub lo: f64, pub hi: f64 }

impl Range {
    /// Every value.
    pub const ANY: Self = Self { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
    /// Every value from 0 up.
    pub const NONNEGATIVE: Self = Self { lo: 0.0, hi: f64::INFINITY };
    /// No value, as an array of no items holds. It's `join`'s identity: its `lo` is ∞ and its `hi` ¯∞.
    pub const EMPTY: Self = Self { lo: f64::INFINITY, hi: f64::NEG_INFINITY };
    /// The range of an argument that is the one number `c`, or `ANY` where `c` is NaN or `None`, as for an argument that isn't one number.
    #[inline]
    pub fn of(c: Option<f64>) -> Self { c.filter(|c| !c.is_nan()).map_or(Self::ANY, |c| Self { lo: c, hi: c }) }
    /// The values that items of `dtype` hold: 0 and 1 for Booleans, from each integer width's least integer to its greatest, and any value for other dtypes.
    #[inline]
    pub fn of_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Boolean => Self { lo: 0.0, hi: 1.0 },
            DType::Integer(width) => with_width!(width, T => Self { lo: T::MIN as f64, hi: T::MAX as f64 }),
            _ => Self::ANY,
        }
    }
    /// The range of `view`'s items, from the least to the greatest, for Booleans, integers and floats. A float's NaN items take no part, as a range holds every value but NaN. An empty view, or one of NaN items alone, holds no value (`EMPTY`). Other items take `ANY`.
    pub fn of_view(view: View<'_>) -> Self {
        match view {
            View::Booleans([]) => Self::EMPTY,
            View::Booleans(x) => Self { lo: if x.contains(&false) { 0.0 } else { 1.0 }, hi: if x.contains(&true) { 1.0 } else { 0.0 } },
            View::Integers(x) => with_ints!(x, |x| match x { [] => Self::EMPTY, x => { let (lo, hi) = crate::reduce::extremes(x); Self { lo: lo as f64, hi: hi as f64 } } }),
            View::Floats(x) => with_floats!(x, |x| Self { lo: crate::reduce::min(x).into(), hi: crate::reduce::max(x).into() }),
            _ => Self::ANY,
        }
    }
    /// The one number that the range holds, if it holds only one.
    #[inline]
    pub fn constant(self) -> Option<f64> { (self.lo == self.hi).then_some(self.lo) }
    /// Whether the range holds no negative number.
    #[inline]
    pub fn nonnegative(self) -> bool { self.lo >= 0.0 }
    /// Whether the range holds no value (`EMPTY`).
    #[inline]
    pub fn is_empty(self) -> bool { self.lo > self.hi }
    /// Whether every value lies from `lo` to `hi`.
    #[inline]
    pub fn within(self, lo: f64, hi: f64) -> bool { lo <= self.lo && self.hi <= hi }
    /// Whether no value equals `c`.
    #[inline]
    pub fn excludes(self, c: f64) -> bool { c < self.lo || self.hi < c }
    /// The values that both ranges hold.
    #[inline]
    pub fn meet(self, other: Self) -> Self { Self { lo: self.lo.max(other.lo), hi: self.hi.min(other.hi) } }
    /// The least range that holds both ranges.
    #[inline]
    pub fn join(self, other: Self) -> Self { Self { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) } }
    /// The range of |x| for the values x of this range.
    pub fn magnitude(self) -> Self {
        if self.lo >= 0.0 { self } else if self.hi <= 0.0 { Self { lo: -self.hi, hi: -self.lo } } else { Self { lo: 0.0, hi: self.hi.max(-self.lo) } }
    }
    /// The range of x × x for the values x of this range, as `⊼x` squares them.
    pub fn square(self) -> Self { let m = self.magnitude(); Self { lo: m.lo * m.lo, hi: m.hi * m.hi } }
    /// The range of a fold's results, from its items' range, its seed's, if it has one, and `step`, which gives the range of a step's result from the ranges of its two arguments. Every lane holds an item. One bound holds for any fold: the least range that holds the items and the seed, where a step's results stay in it. Where they don't, as a sum's don't, each bound widens to an infinity, except that a bound on the far side of 0 stops at 0: [2, 3] widens to [0, ∞]. Where a step's results leave that range too, that bound is `ANY`.
    ///
    /// Where `lane` gives the most items that a lane holds and `step` is an associative function's, as `+ × ⌈ ⌊ ∧ ∨` are, a second bound holds, and the range is the meet of both. It starts at the items' and seed's range, and joins in `step` of the range on itself while the lane length that it covers doubles, up to `lane`: [2, 3] over lanes of up to 4 items gives [2, 12]. It holds because a lane of up to 2m items splits into two lanes of up to m items whose results, by associativity, one step combines, and `step` bounds that step for any arguments within its two ranges. A scan's running results are the results of shorter lanes. A seed counts as one more item.
    pub fn folded(items: Self, seed: Option<Self>, lane: Option<usize>, step: impl Fn([Self; 2]) -> Self) -> Self {
        let start = seed.map_or(items, |s| s.join(items));
        let (lo, hi) = (if start.lo >= 0.0 { 0.0 } else { f64::NEG_INFINITY }, if start.hi <= 0.0 { 0.0 } else { f64::INFINITY });
        let kept = [start, Self { lo, hi }].into_iter().find(|&r| step([r, r]).inside(r)).unwrap_or(Self::ANY);
        let Some(lane) = lane else { return kept };
        let (mut covered, mut m, items) = (start, 1, lane.saturating_add(usize::from(seed.is_some())));
        while m < items { covered = covered.join(step([covered, covered])); m = m.saturating_mul(2); }
        covered.meet(kept)
    }
    /// Whether `other` holds every value of this range.
    fn inside(self, other: Self) -> bool { other.lo <= self.lo && self.hi <= other.hi }
    /// This range with each bound moved `ULPS` units in the last place outward, for bounds that a kernel computes within its accuracy, as the transcendental functions do. A bound never moves past 0, and 0 or an infinite bound stays.
    fn loose(self) -> Self { Self { lo: outward(self.lo, false, FloatWidth::F64), hi: outward(self.hi, true, FloatWidth::F64) } }
    /// This range as a kernel's results of `dtype` hold it. The bounds are f64, and a kernel at a narrower float rounds each result to that width, within its accuracy there: each bound moves `ULPS` units in that width's last place outward, as `loose` moves it at f64. Other dtypes keep the range.
    pub fn at_width(self, dtype: DType) -> Self {
        match dtype {
            DType::Float(width @ (FloatWidth::F32 | FloatWidth::F16)) => Self { lo: outward(self.lo, false, width), hi: outward(self.hi, true, width) },
            _ => self,
        }
    }
}

/// How far a computed bound can lie from a kernel's result on the same argument, in units in the last place: a kernel lies within 4 of the correctly rounded result, and so does the function that computes the bound.
const ULPS: u64 = 8;
/// From ¯1 to 1: the values of the sine and the cosine, and the arguments on which the inverse sine, the inverse cosine and the inverse hyperbolic tangent are real.
const UNIT: (f64, f64) = (-1.0, 1.0);

/// `x` moved `ULPS` units in the last place of floats of `width` up, or down, but not past 0. A zero or an infinity stays. A bound between two floats of `width` starts from the nearer, which the move covers.
fn outward(x: f64, up: bool, width: FloatWidth) -> f64 {
    if x == 0.0 || x.is_infinite() { return x; }
    let (bits, infinity): (u64, u64) = match width {
        FloatWidth::F64 => (x.abs().to_bits(), f64::INFINITY.to_bits()),
        FloatWidth::F32 => ((x.abs() as f32).to_bits().into(), f32::INFINITY.to_bits().into()),
        FloatWidth::F16 => (half::f16::from_f64(x.abs()).to_bits().into(), half::f16::INFINITY.to_bits().into()),
    };
    let moved = if up == x.is_sign_positive() { (bits + ULPS).min(infinity) } else { bits.saturating_sub(ULPS) };
    let moved = match width { FloatWidth::F64 => f64::from_bits(moved), FloatWidth::F32 => f32::from_bits(moved as u32).into(), FloatWidth::F16 => half::f16::from_bits(moved as u16).to_f64() };
    moved.copysign(x)
}

/// The range from `lo` to `hi`, bounds that correctly rounded arithmetic gives on the arguments' bounds, as `+` and `√` give them. A NaN bound, as ∞ - ∞ gives, becomes an infinity.
fn exact(lo: f64, hi: f64) -> Range { Range { lo: if lo.is_nan() { f64::NEG_INFINITY } else { lo }, hi: if hi.is_nan() { f64::INFINITY } else { hi } } }
/// The range of `f`, which never decreases, on the values of `x`, for a kernel within its accuracy.
fn increasing(x: Range, f: impl Fn(f64) -> f64) -> Range { exact(f(x.lo), f(x.hi)).loose() }
/// The range of `f`, which never increases, on the values of `x`, for a kernel within its accuracy.
fn decreasing(x: Range, f: impl Fn(f64) -> f64) -> Range { exact(f(x.hi), f(x.lo)).loose() }
/// The range of `f`, a function of |x| that never decreases, on the values of `x`, for a kernel within its accuracy.
fn even(x: Range, f: impl Fn(f64) -> f64) -> Range { increasing(x.magnitude(), f) }
/// The range of `op` on one value of `x` and one of `y`, where its least and greatest results lie among its results on their bounds, as a product's do, and a quotient's where `y` doesn't hold 0. A NaN there, as 0 × ∞ gives, stands for both 0 and an infinity. The infinity's sign is that of the results near those bounds: a bound's own, or for 0, that of the side of 0 on which its range lies.
fn corners(x: Range, y: Range, op: impl Fn(f64, f64) -> f64) -> Range {
    let side = |v: f64, inward: f64| if v == 0.0 { inward } else { v.signum() };
    let mut r = Range { lo: f64::INFINITY, hi: f64::NEG_INFINITY };
    for (a, da) in [(x.lo, 1.0), (x.hi, -1.0)] {
        for (b, db) in [(y.lo, 1.0), (y.hi, -1.0)] {
            let v = op(a, b);
            for v in if v.is_nan() { [0.0, f64::INFINITY * side(a, da) * side(b, db)] } else { [v, v] } { r = r.join(Range { lo: v, hi: v }); }
        }
    }
    r
}
/// The range of x ÷ y for one value of `x` and one of `y`: `ANY` where `y` holds 0.
fn quotient(x: Range, y: Range) -> Range { if y.excludes(0.0) { corners(x, y, |a, b| a / b) } else { Range::ANY } }
/// The range of π × x for the values x of `x`.
fn pi_times(x: Range) -> Range { corners(x, Range::of(Some(PI)), |a, b| a * b).loose() }
/// `Range::NONNEGATIVE` where every one of `args` is non-negative, and otherwise `ANY`.
fn signs<const N: usize>(args: [Range; N]) -> Range { if args.iter().all(|a| a.nonnegative()) { Range::NONNEGATIVE } else { Range::ANY } }
/// `domain`, a box of argument ranges on which a function's results never fail, where every one of `args` lies in it, and otherwise `None`.
fn inside<const N: usize>(args: [Range; N], domain: [Range; N]) -> Option<[Range; N]> { args.iter().zip(&domain).all(|(a, d)| a.within(d.lo, d.hi)).then_some(domain) }
/// The box around `n` and `y` on which `n√y` never fails: `n` finite and of one sign, with `y` from 0 up, or, where `y` holds a negative number, `n` one odd whole number, with any `y`.
fn root_domain(n: Range, y: Range) -> Option<[Range; 2]> {
    let tiny = f64::from_bits(1);
    match n.constant() {
        Some(c) if y.lo < 0.0 && parity(c).1 => inside([n, y], [n, Range::ANY]),
        _ => inside([n, y], [if n.lo > 0.0 { exact(tiny, f64::MAX) } else { exact(f64::MIN, -tiny) }, Range::NONNEGATIVE]),
    }
}

/// Each object's range, domain, fold facts and executable formulae.
pub trait Formula<const N: usize>: Clone + Send + Sync + 'static {
    fn result_dtype(&self, input: DType) -> DType { input }
    fn computing_dtype(&self, input: DType) -> Option<DType> {
        input.computes_in(crate::buffer::Width::I64, crate::operation::numpy_float(input), |d| self.supports(d))
    }
    fn character_rule(&self) -> Option<crate::characters::Rule> { None }
    fn tolerance(&self) -> Option<f64> { None }
    fn ordered_kinds(&self, [x, y]: [i32; 2]) -> Option<i32> {
        let ops = crate::recipe::IntegerOps::<i32>::new(Variant::checked(DType::Integer(crate::buffer::Width::I32)));
        self.integer(&ops, std::array::from_fn(|i| if i == 0 { x } else { y }))
    }
    fn character_policy(&self, types: [DType; 2]) -> Result<Option<crate::characters::Policy>, crate::error::Error> {
        crate::characters::policy(self.character_rule(), types, self.result_dtype(DType::Integer(crate::buffer::Width::I32)), |x| self.ordered_kinds(x))
    }
    fn exact(&self, _args: [&num_rational::BigRational; N]) -> Result<Option<num_rational::BigRational>, crate::error::Error> { Ok(None) }
    fn has_exact(&self) -> bool { false }
    fn range(&self, _args: [Range; N]) -> Range { Range::ANY }
    fn integer_domain(&self, _args: [Range; N]) -> bool { true }
    fn typed_domain(&self, _types: [DType; N]) -> bool { false }
    /// A box containing args on which graph-numeric results do not raise a data error.
    fn domain(&self, _args: [Range; N]) -> Option<[Range; N]> { None }
    fn has_domain(&self) -> bool { false }
    fn identity(&self) -> [Option<f64>; N] { [None; N] }
    fn fold(&self) -> Option<Reduce> { None }
    fn truth_table(&self) -> Option<u8> { None }
    fn truth(&self) -> Option<Reduce> {
        match self.truth_table()? { 8 => Some(Reduce::Min), 14 => Some(Reduce::Max), 6 => Some(Reduce::Xor), 9 => Some(Reduce::Xnor), _ => None }
    }
    fn same(&self, _arg: Range) -> Option<Range> { None }
    fn reads(&self) -> Reads { Reads::Items }
    fn kernel(&self, variant: Variant) -> Option<Kernel> {
        self.supports(variant.dtype).then(|| Kernel::new::<Self, N>(self.clone(), variant))
    }
    fn resolve(&self, variant: Variant, types: [DType; N]) -> Result<Kernel, crate::error::Error> { Kernel::resolve(self.clone(), variant, types) }
    fn supports(&self, _dtype: DType) -> bool { false }
    fn integer_formula(&self) -> bool { false }
    fn emit<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _x: [O::V; N]) -> Option<O::V> { None }
    fn emit_test<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _x: [O::V; N]) -> Option<O::B> { None }
    fn emit_repair<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _x: [O::V; N]) -> Option<O::B> { None }
    fn repair_routine(&self) -> Option<Repair> { None }
    fn fails_nan(&self) -> bool { false }
    fn emit_valid<F: crate::Float, O: Ops<F> + ?Sized>(&self, _variant: Variant, o: &O, x: [O::V; N], result: O::V) -> Option<(crate::error::Kind, O::B)> {
        self.fails_nan().then(|| (crate::error::Kind::Domain, nan_valid(o, x, [result])))
    }
    fn boolean(&self) -> bool { self.result_dtype(DType::Float(crate::buffer::FloatWidth::F64)) == DType::Boolean }
    fn real<F: crate::float::Native>(&self, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [F; N]) -> Option<F> {
        if self.boolean() { self.emit_test(ops, x).map(|x| if x { F::one() } else { F::zero() }) } else { self.emit(ops, x) }
    }
    fn integer<I: crate::recipe::Overflowing>(&self, _ops: &crate::recipe::IntegerOps<I>, _x: [I; N]) -> Option<I> { None }
    fn complex<F: crate::float::Native>(&self, _ops: &crate::recipe::NativeOps<{ crate::FUSED }>, _x: [num_complex::Complex<F>; N]) -> Option<num_complex::Complex<F>> { None }
    fn native_slices(&self) -> Option<NativeSlices<Self, N>> { None }
    fn emit_partials<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(
        &self, _o: &O, _x: [O::V; N], _result: O::V, _needed: [bool; N],
    ) -> [Option<O::V>; N] { [None; N] }
}

pub struct NativeSlices<K, const N: usize> {
    pub single: fn(&K, [crate::Arg<'_, f32>; N], usize, &mut [std::mem::MaybeUninit<f32>]) -> bool,
    pub double: fn(&K, [crate::Arg<'_, f64>; N], usize, &mut [std::mem::MaybeUninit<f64>]) -> bool,
}
impl<K> NativeSlices<K, 1> where K: crate::Unary<f32, Out = f32> + crate::Unary<f64, Out = f64> {
    fn unary() -> Self {
        Self { single: |k, x, n, out| crate::Unary::<f32>::slice(k, x, n, out), double: |k, x, n, out| crate::Unary::<f64>::slice(k, x, n, out) }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Repair {
    pub single: extern "C" fn(f32) -> f32,
    pub double: extern "C" fn(f64) -> f64,
}

pub(crate) fn nan_valid<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, inputs: impl IntoIterator<Item = O::V>, results: impl IntoIterator<Item = O::V>) -> O::B {
    let mut valid = o.equal(o.number(F::zero()), o.number(F::zero()));
    for result in results { valid = o.and(valid, o.equal(result, result)); }
    for input in inputs { valid = o.or(valid, o.not(o.equal(input, input))); }
    valid
}

impl Repair {
    pub fn apply<F: crate::Float>(self, x: F) -> F {
        F::narrow(if F::Working::WIDE { (self.double)(x.wide()) } else { (self.single)(x.wide() as f32) as f64 })
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reads { Items, Whole }


#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot(pub u32);

/// Policies captured when the call is built; never read from force-time preferences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variant {
    pub dtype: DType,
    pub checked: bool,
    pub mathish: bool,
    pub wrapping: bool,
}

impl Variant {
    pub const fn checked(dtype: DType) -> Self { Self { dtype, checked: true, mathish: false, wrapping: false } }
}

pub use crate::operation::Kernel;

/// `k`'s scalar real result, including repairs, computed at the argument's working precision.
#[inline(always)]
pub fn formula_at<K: Formula<N>, F: crate::Float, const N: usize>(k: &K, x: [F; N]) -> F {
    let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
    let x = x.map(F::working);
    let result = if k.emit_repair(&o, x) == Some(true) { k.repair_routine().expect("a repair predicate has its routine").apply(x[0]) }
        else { k.real(&o, x).expect("a real body that gives every result") };
    F::from_working(result)
}

/// `x` to the power `y`, where `y` is one number (`Ops::constant`) whose `Exponent` route isn't the general one, through `recipe::power`, as `Pow`'s and `WholePow`'s kernels compute a constant exponent.
fn constant_power<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> Option<O::V> {
    crate::recipe::power(o, crate::Exponent::of(o.constant(y)?.wide()), x)
}

/// Each object's facts, formula and partials have one definition.
macro_rules! objects {
    (@partials 1, slope |$o:tt, $x:tt, $r:tt| $slope:expr) => {
        #[inline(always)]
        fn emit_partials<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, $o: &O, [$x]: [O::V; 1], $r: O::V, [need]: [bool; 1]) -> [Option<O::V>; 1] {
            [need.then(|| $slope)]
        }
    };
    (@partials 2, slopes |$o:tt, $x:tt, $y:tt, $r:tt| [$dx:expr, $dy:expr]) => {
        #[inline(always)]
        fn emit_partials<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, $o: &O, [$x, $y]: [O::V; 2], $r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
            [nx.then(|| $dx), ny.then(|| $dy)]
        }
    };
    (@partials 2, shared $slopes:ident) => {
        #[inline(always)]
        fn emit_partials<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, o: &O, x: [O::V; 2], r: O::V, need: [bool; 2]) -> [Option<O::V>; 2] {
            crate::derivative::$slopes(o, x, r, need)
        }
    };
    (@partials $n:tt,) => {};
    (@present $body:expr) => { true };
    (@test_output $body:expr) => { fn result_dtype(&self, _: DType) -> DType { DType::Boolean } };
    (@output $dtype:ident, real) => { match $dtype { DType::Complex(w) => DType::Float(w.into()), _ => $dtype } };
    (@output $dtype:ident, complex) => {
        match $dtype.inexact(crate::buffer::FloatWidth::F64) {
            Some((DType::Float(w), _)) => DType::Complex(w.into()), Some((d, _)) => d, None => $dtype,
        }
    };
    (@computing real) => {};
    (@computing complex) => { fn computing_dtype(&self, input: DType) -> Option<DType> { let dtype = input.complex(); self.supports(dtype).then_some(dtype) } };
    (@checks nan) => { fn fails_nan(&self) -> bool { true } };
    (@only bits) => { true };
    (@boolean bits) => { false };
    (@boolean arithmetic) => { false };
    (@boolean $kind:ident) => { true };
    (@integers $n:tt, bits) => { objects!(@integers $n, formula); };
    (@only formula) => { false };
    (@only arithmetic) => { false };
    (@integers $n:tt, arithmetic) => { objects!(@integers $n, formula); };
    (@only only) => { true };
    (@integers $n:tt, only) => { objects!(@integers $n, formula); };
    (@integers $n:tt, formula) => {
        fn integer_formula(&self) -> bool { true }
        fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, x: [I; $n]) -> Option<I> {
            if self.boolean() { self.emit_test(o, x).map(|x| I::low(i64::from(x))) } else { self.emit(o, x) }
        }
    };
    ($n:tt: $($object:ty {
        $(range $args:pat => $range:expr;)? $(domain $targs:pat => $domain:expr;)? $(integer_domain $iargs:pat => $integer_domain:expr;)? $(typed_domain $kargs:pat => $typed_domain:expr;)? $(identity $identity:expr;)? $(fold $fold:ident;)? $(truth $truth:ident;)? $(table $table:expr;)?
        $(output $output:ident;)? $(characters $rule:ident;)? $(tolerance |$tolk:pat_param| $tolerance:expr;)? $(same |$sx:pat_param| $same:expr;)? $(formula |$k:pat_param, $o:tt, $fargs:tt| $formula:expr;)? $(test |$tk:pat_param, $to:tt, $targs2:tt| $test:expr;)?
        $(slope |$so:tt, $sx1:tt, $sr:tt| $slope:expr;)? $(slopes |$s2o:tt, $s2x:tt, $s2y:tt, $s2r:tt| [$dx:expr, $dy:expr];)? $(shared $shared:ident;)?
        $(repair $repair:expr;)?
        $(integers $integer:ident;)?
        $(native_integer |$nik:pat_param, $nio:tt, $nix:tt| $native_integer:expr;)?
        $(native_complex |$nfk:pat_param, $nfo:tt, $nfx:tt| $native_complex:expr;)?
        $(native_real |$nrk:pat_param, $nro:tt, $nrx:tt| $native_real:expr;)?
        $(slices $slices:ident;)?
        $(exact |$ek:pat_param, $ex:tt| $exact:expr;)?
        $(checks $checks:ident;)?
        $(valid |$vk:pat_param, $vv:tt, $vo:tt, $vx:tt, $vr:tt| $valid:expr;)?
    })+) => {$(
        impl Formula<$n> for $object {
            $(fn exact(&self, $ex: [&num_rational::BigRational; $n]) -> Result<Option<num_rational::BigRational>, crate::error::Error> { let $ek = self; $exact } fn has_exact(&self) -> bool { true })?
            $(fn character_rule(&self) -> Option<crate::characters::Rule> { Some(crate::characters::Rule::$rule) })?
            $(fn tolerance(&self) -> Option<f64> { let $tolk = self; Some($tolerance) })?
            $(fn result_dtype(&self, dtype: DType) -> DType { objects!(@output dtype, $output) } objects!(@computing $output);)?
            $(objects!(@test_output $test);)?
            $(fn range(&self, $args: [Range; $n]) -> Range { $range })?
            $(fn domain(&self, $targs: [Range; $n]) -> Option<[Range; $n]> { $domain } fn has_domain(&self) -> bool { true })?
            $(fn integer_domain(&self, $iargs: [Range; $n]) -> bool { $integer_domain })?
            $(fn typed_domain(&self, $kargs: [DType; $n]) -> bool { $typed_domain })?
            $(fn identity(&self) -> [Option<f64>; $n] { $identity })?
            $(fn fold(&self) -> Option<crate::reduce::Reduce> { Some(crate::reduce::Reduce::$fold) })?
            $(fn truth(&self) -> Option<crate::reduce::Reduce> { Some(crate::reduce::Reduce::$truth) })?
            $(fn truth_table(&self) -> Option<u8> { Some($table) })?
            $(fn same(&self, $sx: Range) -> Option<Range> { Some($same) })?
            fn supports(&self, dtype: DType) -> bool {
                match dtype {
                    DType::Boolean => (false $(|| objects!(@present $integer))? $(|| objects!(@present $native_integer))?) $(&& objects!(@boolean $integer))?,
                    DType::Integer(_) => false $(|| objects!(@present $integer))? $(|| objects!(@present $native_integer))?,
                    DType::Float(_) => (false $(|| objects!(@present $formula))? $(|| objects!(@present $test))? $(|| objects!(@present $native_real))?) $(&& !objects!(@only $integer))?,
                    DType::Complex(_) => false $(|| objects!(@present $native_complex))?,
                    DType::Character => false,
                }
            }
            $(#[inline(always)] fn emit<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, $o: &O, $fargs: [O::V; $n]) -> Option<O::V> { let $k = self; $formula })?
            $(#[inline(always)] fn emit_test<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, $to: &O, $targs2: [O::V; $n]) -> Option<O::B> { let $tk = self; $test })?
            $(fn emit_repair<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, o: &O, x: [O::V; $n]) -> Option<O::B> { Some(($repair).predicate(o, x[0])) }
            fn repair_routine(&self) -> Option<Repair> { Some(($repair).routine()) })?
            objects!(@partials $n, $(slope |$so, $sx1, $sr| $slope)? $(slopes |$s2o, $s2x, $s2y, $s2r| [$dx, $dy])? $(shared $shared)?);
            $(objects!(@integers $n, $integer);)?
            $(objects!(@checks $checks);)?
            $(fn emit_valid<F: crate::Float, O: Ops<F> + ?Sized>(&self, $vv: Variant, $vo: &O, $vx: [O::V; $n], $vr: O::V) -> Option<(crate::error::Kind, O::B)> { let $vk = self; $valid })?
            $(fn integer<I: crate::recipe::Overflowing>(&self, $nio: &crate::recipe::IntegerOps<I>, $nix: [I; $n]) -> Option<I> { let $nik = self; $native_integer })?
            $(fn complex<F: crate::float::Native>(&self, $nfo: &crate::recipe::NativeOps<{ crate::FUSED }>, $nfx: [num_complex::Complex<F>; $n]) -> Option<num_complex::Complex<F>> { let $nfk = self; $native_complex })?
            $(fn real<F: crate::float::Native>(&self, $nro: &crate::recipe::NativeOps<{ crate::FUSED }>, $nrx: [F; $n]) -> Option<F> { let $nrk = self; $native_real })?
            $(fn native_slices(&self) -> Option<NativeSlices<Self, $n>> { Some(NativeSlices::$slices()) })?
        }
    )+};
}

#[derive(Clone, Copy)]
pub(crate) struct Comparison(pub crate::recipe::Cmp);
#[derive(Clone, Copy)]
pub(crate) struct IntegerOperation(pub crate::recipe::Integer);
#[derive(Clone, Copy)]
pub(crate) struct IntegerComparison(pub crate::recipe::Cmp);

impl Formula<2> for IntegerOperation {
    fn supports(&self, dtype: DType) -> bool { dtype.is_integral() }
    fn integer_formula(&self) -> bool { true }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::V> {
        Some(o.from_bits(o.integer_op(self.0, o.to_bits(x), o.to_bits(y))))
    }
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, x: [I; 2]) -> Option<I> { self.emit(o, x) }
}
impl Formula<2> for IntegerComparison {
    fn supports(&self, dtype: DType) -> bool { dtype.is_integral() }
    fn result_dtype(&self, _: DType) -> DType { DType::Boolean }
    fn integer_formula(&self) -> bool { true }
    fn emit_test<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::B> {
        Some(o.integer_compare(self.0, o.to_integer(x), o.to_integer(y)))
    }
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, x: [I; 2]) -> Option<I> {
        self.emit_test(o, x).map(|value| I::low(i64::from(value)))
    }
}
#[derive(Clone, Copy)]
pub(crate) struct Complement;
objects!(1:
    Complement { formula |_, o, [x]| Some(o.subtract(o.number(F::one()), x)); integers formula; exact |_, [x]| Ok(Some(num_rational::BigRational::one() - x)); valid |_, v, o, [x], _| arithmetic_valid(v, o, crate::recipe::Binary::Subtract, o.number(F::one()), x); }
    crate::Abs { range [x] => x.magnitude(); output real; formula |_, o, [x]| Some(o.magnitude(x)); slope |o, x, _| o.sign(x); integers formula; native_complex |_, _, [x]| Some(num_complex::Complex::new(crate::complex::modulus::<F, { crate::FUSED }>(x.re, x.im), F::zero())); exact |_, [x]| Ok(Some(x.abs())); valid |_, v, o, [x], _| negation_valid(v, o, x, true); }
    crate::Negate { range [x] => Range { lo: -x.hi, hi: -x.lo }; formula |_, o, [x]| Some(o.negate(x)); slope |o, _, _| o.number(-F::one()); integers arithmetic; native_complex |_, _, [x]| Some(-x); exact |_, [x]| Ok(Some(-x)); valid |_, v, o, [x], _| negation_valid(v, o, x, false); }
    crate::Conj { range [x] => x; characters Identity; formula |_, _, [x]| Some(x); slope |o, _, _| o.number(F::one()); integers formula; native_complex |_, _, [x]| Some(x.conj()); exact |_, [x]| Ok(Some(x.clone())); }
    crate::Real { range [x] => x; output real; formula |_, _, [x]| Some(x); slope |o, _, _| o.number(F::one()); integers formula; native_complex |_, _, [x]| Some(num_complex::Complex::new(x.re, F::zero())); exact |_, [x]| Ok(Some(x.clone())); }
    crate::Imag { range _ => exact(0.0, 0.0); output real; formula |_, o, [_]| Some(o.number(F::zero())); slope |o, _, _| o.number(F::zero()); integers formula; native_complex |_, _, [x]| Some(num_complex::Complex::new(x.im, F::zero())); exact |_, [_]| Ok(Some(num_rational::BigRational::zero())); }
    crate::Angle { range _ => exact(-PI, PI); output real; slope |o, _, _| o.number(F::zero()); native_complex |_, _, [x]| Some(num_complex::Complex::new(crate::arc::atan2::<F, { crate::FUSED }>(x.im, x.re), F::zero())); native_real |_, _, [x]| Some(crate::arc::real_angle(x)); }
    crate::Isnan { test |_, o, [x]| Some(o.not(o.equal(x, x))); integers formula; native_complex |_, _, [x]| Some(num_complex::Complex::new(if x.re.is_nan() || x.im.is_nan() { F::one() } else { F::zero() }, F::zero())); }
    crate::Isfinite { test |_, o, [x]| Some(if o.integral() { o.equal(x, x) } else { o.equal(o.subtract(x, x), o.number(F::zero())) }); integers formula; native_complex |_, _, [x]| Some(num_complex::Complex::new(if x.re.is_finite() && x.im.is_finite() { F::one() } else { F::zero() }, F::zero())); }
    crate::Isinf { test |_, o, [x]| Some(if o.integral() { o.not(o.equal(x, x)) } else { o.equal(o.magnitude(x), o.number(F::infinity())) }); integers formula; native_complex |_, _, [x]| Some(num_complex::Complex::new(if x.re.is_infinite() || x.im.is_infinite() { F::one() } else { F::zero() }, F::zero())); }
    crate::Trunc { native_integer |_, _, [x]| Some(x); native_real |_, _, [x]| Some(x.trunc()); }
    crate::Rint { native_integer |_, _, [x]| Some(x); native_complex |_, _, [x]| Some(num_complex::Complex::new(x.re.round_ties_even(), x.im.round_ties_even())); native_real |_, _, [x]| Some(x.round_ties_even()); }
    crate::Scale { formula |k, o, [x]| Some(o.multiply(x, o.number(F::narrow(k.0)))); }
    crate::BitNot { native_integer |_, o, [x]| Some(o.result(if o.variant.dtype == DType::Boolean { i64::from(x == I::low(0)) } else { !Into::<i64>::into(x) })); }
    crate::Round { native_integer |k, o, [x]| crate::parts::rounded_integer::<i64>(x.into(), k.scale).map(|x| o.result(x)); native_complex |k, _, [x]| Some(num_complex::Complex::new(crate::parts::rounded(x.re, k.scale), crate::parts::rounded(x.im, k.scale))); native_real |k, _, [x]| Some(crate::parts::rounded(x, k.scale)); }
    crate::Sqrt { range [x] => exact(x.lo.max(0.0).sqrt(), x.hi.sqrt()); domain [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(o.sqrt(x)); slope |o, _, r| o.reciprocal(o.add(r, r)); native_complex |_, _, [x]| Some(crate::complex::sqrt::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Cbrt { range [x] => increasing(x, f64::cbrt); formula |_, o, [x]| Some(crate::pow::cbrt_formula(o, x)); slope |o, _, r| o.reciprocal(o.multiply(o.number(F::narrow(3.0)), o.multiply(r, r))); }
    crate::Exp { range [x] => increasing(x, f64::exp).meet(Range::NONNEGATIVE); formula |_, o, [x]| Some(crate::exp::formula(o, x)); slope |_, _, r| r; native_complex |_, _, [x]| Some(crate::complex::exp::<F, { crate::FUSED }>(x)); }
    crate::Expm1 { range [x] => increasing(x, f64::exp_m1); formula |_, o, [x]| Some(crate::exp::minus_one_formula(o, x, o.number(F::zero()))); slope |o, _, r| o.add(r, o.number(F::one())); }
    crate::Ln { range [x] => increasing(x, |v| if v < 0.0 { f64::NEG_INFINITY } else { v.ln() }); domain [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(crate::ln::formula(o, x)); slope |o, x, _| o.reciprocal(x); native_complex |_, _, [x]| Some(crate::complex::ln::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Log2 { range [x] => increasing(x, f64::log2); domain [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(crate::ln::log_with(o, x, crate::ln::base2(o))); slope |o, x, _| o.divide(o.number(F::narrow(std::f64::consts::LOG2_E)), x); slices unary; checks nan; }
    crate::Log10 { range [x] => increasing(x, f64::log10); domain [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(crate::ln::log_with(o, x, crate::ln::base10(o))); slope |o, x, _| o.divide(o.number(F::narrow(std::f64::consts::LOG10_E)), x); slices unary; checks nan; }
    crate::RealLn { range [x] => increasing(x.magnitude(), f64::ln); output real; formula |_, o, [x]| Some(crate::ln::formula(o, o.magnitude(x))); slope |o, x, _| o.reciprocal(x); native_complex |_, _, [x]| Some(crate::complex::ln::<F, { crate::FUSED }>(x)); }
    crate::RealSqrt {
        range [x] => exact(x.lo.max(0.0).sqrt(), x.hi.max(0.0).sqrt()); output real;
        formula |_, o, [x]| Some(crate::recipe::real_sqrt(o, x));
        slope |o, x, r| {
            let (zero, inside) = (o.number(F::zero()), o.less(o.number(F::zero()), x));
            // r is 0 outside the domain, where 1 stands in for it, so that no slope of the case that the select drops is infinite (JAX's double `where`).
            let r = o.select(inside, r, o.number(F::one()));
            o.select(inside, o.reciprocal(o.add(r, r)), zero)
        };
        native_complex |_, _, [x]| Some(crate::complex::sqrt::<F, { crate::FUSED }>(x));
    }
    crate::RealSqrt1mx2 {
        range [x] => decreasing(x.magnitude(), |v| (1.0 - v.min(1.0) * v.min(1.0)).sqrt()).meet(Range::NONNEGATIVE); output real;
        // √(1 - x²)'s slope inside -1..1, -x ÷ r, and 0 beyond, where 1 stands in for r so that the dropped case's slope stays finite (JAX's double `where`).
        slope |o, x, r| {
            let (one, inside) = (o.number(F::one()), o.less(o.magnitude(x), o.number(F::one())));
            o.select(inside, o.negate(o.divide(x, o.select(inside, r, one))), o.number(F::zero()))
        };
        native_complex |_, _, [x]| Some(crate::complex::sqrt1mx2::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(if x.abs() > F::one() { F::zero() } else { crate::sqrts::sqrt1mx2::<F, { crate::FUSED }>(x) });
    }
    crate::RealSqrtx2m1 {
        range [x] => increasing(x, |v| if v.abs() < 1.0 { 0.0 } else { v.signum() * (v * v - 1.0).sqrt() }); output real;
        // The slope of x's sign times √(x² - 1) beyond -1..1, |x| ÷ |r|, and 0 inside, where 1 stands in for |r|.
        slope |o, x, r| {
            let (one, outside) = (o.number(F::one()), o.less(o.number(F::one()), o.magnitude(x)));
            o.select(outside, o.divide(o.magnitude(x), o.select(outside, o.magnitude(r), one)), o.number(F::zero()))
        };
        native_complex |_, _, [x]| Some(crate::complex::sqrtx2m1::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(if x.abs() < F::one() { F::zero() } else { x.signum() * crate::sqrts::sqrtx2m1::<F, { crate::FUSED }>(x.abs()) });
    }
    crate::RealAsin {
        range [x] => increasing(x, |v| v.clamp(UNIT.0, UNIT.1).asin()); output real;
        // asin's and acos's slopes inside -1..1, ±1 ÷ √(1 - x²), and 0 beyond, where 0 stands in for x.
        slope |o, x, _| {
            let inside = o.less(o.magnitude(x), o.number(F::one()));
            o.select(inside, o.reciprocal(o.sqrt1mx2(o.select(inside, x, o.number(F::zero())))), o.number(F::zero()))
        };
        native_complex |_, _, [x]| Some(crate::complex::asin::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::arc::asin::<F, { crate::FUSED }>(if x > F::one() { F::one() } else if x < -F::one() { -F::one() } else { x }));
    }
    crate::RealAcos {
        range [x] => decreasing(x, |v| v.clamp(UNIT.0, UNIT.1).acos()).meet(Range::NONNEGATIVE); output real;
        slope |o, x, _| {
            let inside = o.less(o.magnitude(x), o.number(F::one()));
            o.select(inside, o.negate(o.reciprocal(o.sqrt1mx2(o.select(inside, x, o.number(F::zero()))))), o.number(F::zero()))
        };
        native_complex |_, _, [x]| Some(crate::complex::acos::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::arc::acos::<F, { crate::FUSED }>(if x > F::one() { F::one() } else if x < -F::one() { -F::one() } else { x }));
    }
    crate::RealAcosh {
        range [x] => even(x, |v| v.abs().max(1.0).acosh()).meet(Range::NONNEGATIVE); output real;
        // acosh |x|'s slope beyond -1..1, x's sign ÷ sinh r, and 0 inside, where 1 stands in for sinh r.
        slope |o, x, r| {
            let (one, outside) = (o.number(F::one()), o.less(o.number(F::one()), o.magnitude(x)));
            o.select(outside, o.divide(o.sign(x), o.select(outside, o.sinh(r), one)), o.number(F::zero()))
        };
        native_complex |_, _, [x]| Some(crate::complex::acosh::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::hyperbolic::acosh::<F, { crate::FUSED }>(if x.abs() < F::one() { F::one() } else { x.abs() }));
    }
    crate::RealSqrtm1mx2 { range _ => exact(0.0, 0.0); output real; slope |o, _, _| o.number(F::zero()); native_complex |k, _, [x]| Some(crate::complex::sqrtm1mx2::<F, { crate::FUSED }>(x, k.0)); native_real |_, _, [x]| Some(if x.is_nan() { x } else { F::zero() }); }
    crate::sqrts::Sqrtm1mx2 { output complex; native_complex |k, _, [x]| Some(crate::complex::sqrtm1mx2::<F, { crate::FUSED }>(x, k.0)); }
    crate::Log1p { range [x] => increasing(x, |v| if v < -1.0 { f64::NEG_INFINITY } else { v.ln_1p() }); domain [x] => inside([x], [exact(-1.0, f64::INFINITY)]); formula |_, o, [x]| Some(crate::ln::log1p_formula(o, x)); slope |o, x, _| o.reciprocal(o.add(o.number(F::one()), x)); checks nan; }
    crate::Sinh { range [x] => increasing(x, f64::sinh); formula |_, o, [x]| Some(crate::hyperbolic::sinh_formula(o, x)); slope |o, x, _| o.cosh(x); native_complex |_, _, [x]| Some(crate::complex::sinh::<F, { crate::FUSED }>(x)); }
    crate::Cosh { range [x] => even(x, f64::cosh).meet(Range::NONNEGATIVE); formula |_, o, [x]| Some(crate::hyperbolic::cosh_formula(o, x)); slope |o, x, _| o.sinh(x); native_complex |_, _, [x]| Some(crate::complex::cosh::<F, { crate::FUSED }>(x)); }
    crate::Tanh { range [x] => increasing(x, f64::tanh); formula |_, o, [x]| Some(crate::hyperbolic::tanh_formula(o, x)); slope |o, _, r| o.subtract(o.number(F::one()), o.multiply(r, r)); native_complex |_, _, [x]| Some(crate::complex::tanh::<F, { crate::FUSED }>(x)); }
    crate::Asinh { range [x] => increasing(x, f64::asinh); slope |o, x, _| o.reciprocal(o.sqrt1px2(x)); native_complex |_, _, [x]| Some(crate::complex::asinh::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::hyperbolic::asinh::<F, { crate::FUSED }>(x)); }
    // sinh(acosh x) is √(x² - 1).
    crate::Acosh { range [x] => increasing(x, |v| v.max(1.0).acosh()).meet(Range::NONNEGATIVE); domain [x] => inside([x], [exact(1.0, f64::INFINITY)]); slope |o, _, r| o.reciprocal(o.sinh(r)); native_complex |_, _, [x]| Some(crate::complex::acosh::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::hyperbolic::acosh::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Atanh { range [x] => increasing(x, |v| v.clamp(UNIT.0, UNIT.1).atanh()); domain [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, _| o.reciprocal(o.subtract(o.number(F::one()), o.multiply(x, x))); native_complex |_, _, [x]| Some(crate::complex::atanh::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::hyperbolic::atanh::<F, { crate::FUSED }>(x)); checks nan; }
    crate::RealAtanh { output real; slope |o, x, _| o.reciprocal(o.subtract(o.number(F::one()), o.multiply(x, x))); native_complex |_, _, [x]| Some(crate::complex::atanh::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::hyperbolic::atanh::<F, { crate::FUSED }>(if x.abs() > F::one() { F::one()/x } else { x })); }
    crate::Sin { range _ => exact(UNIT.0, UNIT.1).loose(); formula |_, o, [x]| Some(crate::trig::formula(o, x, crate::trig::Kind::Sin)); slope |o, x, _| o.cos(x); repair crate::trig::Kind::Sin; native_complex |_, _, [x]| Some(crate::complex::sin::<F, { crate::FUSED }>(x)); }
    crate::Cos { range _ => exact(UNIT.0, UNIT.1).loose(); formula |_, o, [x]| Some(crate::trig::formula(o, x, crate::trig::Kind::Cos)); slope |o, x, _| o.negate(o.sin(x)); repair crate::trig::Kind::Cos; native_complex |_, _, [x]| Some(crate::complex::cos::<F, { crate::FUSED }>(x)); }
    crate::Tan { formula |_, o, [x]| Some(crate::trig::formula(o, x, crate::trig::Kind::Tan)); slope |o, _, r| o.add(o.number(F::one()), o.multiply(r, r)); repair crate::trig::Kind::Tan; native_complex |_, _, [x]| Some(crate::complex::tan::<F, { crate::FUSED }>(x)); }
    crate::Asin { range [x] => increasing(x, |v| v.clamp(UNIT.0, UNIT.1).asin()); domain [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, _| o.reciprocal(o.sqrt1mx2(x)); native_complex |_, _, [x]| Some(crate::complex::asin::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::arc::asin::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Acos { range [x] => decreasing(x, |v| v.clamp(UNIT.0, UNIT.1).acos()).meet(Range::NONNEGATIVE); domain [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, _| o.negate(o.reciprocal(o.sqrt1mx2(x))); native_complex |_, _, [x]| Some(crate::complex::acos::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::arc::acos::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Atan { range [x] => increasing(x, f64::atan); slope |o, x, _| o.reciprocal(o.add(o.number(F::one()), o.multiply(x, x))); native_complex |_, _, [x]| Some(crate::complex::atan::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::arc::atan::<F, { crate::FUSED }>(x)); }
    crate::Sqrt1mx2 { range [x] => decreasing(x.magnitude(), |v| (1.0 - v.min(1.0) * v.min(1.0)).sqrt()).meet(Range::NONNEGATIVE); domain [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, r| o.negate(o.divide(x, r)); native_complex |_, _, [x]| Some(crate::complex::sqrt1mx2::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::sqrts::sqrt1mx2::<F, { crate::FUSED }>(x)); }
    crate::Sqrt1px2 { range [x] => even(x, |v| (1.0 + v * v).sqrt()).meet(Range::NONNEGATIVE); slope |o, x, r| o.divide(x, r); native_complex |_, _, [x]| Some(crate::complex::sqrt1px2::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::sqrts::sqrt1px2::<F, { crate::FUSED }>(x)); }
    crate::Sqrtx2m1 {
        range [x] => increasing(x, |v| v.signum() * (v * v - 1.0).max(0.0).sqrt());
        domain [x] => inside([x], [if x.lo >= 1.0 { exact(1.0, f64::INFINITY) } else { exact(f64::NEG_INFINITY, -1.0) }]);
        slope |o, x, r| o.divide(x, r);
        native_complex |_, _, [x]| Some(crate::complex::sqrtx2m1::<F, { crate::FUSED }>(x));
        native_real |_, _, [x]| Some(crate::sqrts::sqrtx2m1::<F, { crate::FUSED }>(x)); checks nan;
    }
    crate::Reciprocal { range [x] => quotient(Range::of(Some(1.0)), x).loose(); formula |_, o, [x]| Some(crate::recipe::reciprocal(o, x)); slope |o, _, r| o.negate(o.multiply(r, r)); native_complex |_, _, [x]| Some(crate::complex::divide::<F>(num_complex::Complex::new(F::one(), F::zero()), x)); exact |_, [x]| crate::exact::divide(&num_rational::BigRational::one(), x, false); }
    crate::PiTimes { range [x] => pi_times(x); formula |_, o, [x]| Some(crate::recipe::pi_times(o, x)); slope |o, _, _| o.number(F::narrow(std::f64::consts::PI)); native_complex |_, _, [x]| Some(x * F::narrow(std::f64::consts::PI)); }
    // Γ(x + 1)'s slope is Γ(x + 1) ψ(x + 1).
    crate::Factorial { range args => signs(args); slope |o, x, r| o.multiply(r, o.digamma(o.add(x, o.number(F::one())))); native_complex |_, _, [x]| Some(crate::complex::factorial::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::special::factorial::<F, { crate::FUSED }>(x)); exact |_, [x]| crate::exact::factorial(x); checks nan; }
    crate::Gamma {
        range args => signs(args); slope |o, x, r| o.multiply(r, o.digamma(x));
        native_complex |_, _, [x]| Some(crate::complex::gamma::<F, { crate::FUSED }>(x));
        native_real |_, _, [x]| Some(crate::special::gamma::<F, { crate::FUSED }>(x)); checks nan;
    }
    crate::Sign { formula |_, o, [x]| Some(crate::recipe::sign(o, x)); slope |o, _, _| o.number(F::zero()); integers arithmetic; native_complex |_, _, [x]| Some(crate::complex::sign::<F, { crate::FUSED }>(x)); exact |_, [x]| Ok(Some(x.signum())); }
    crate::tolerant::Floor { formula |k, o, [x]| Some(crate::with_tolerance!(k.0, |t| t.floor_in(o, x))); slope |o, _, _| o.number(F::zero()); integers formula; native_complex |k, o, [x]| Some(crate::tolerant::TolerantComplex::complex_floor(native_tolerance(o.tolerance, k.0), x)); exact |_, [x]| Ok(Some(x.floor())); }
    crate::tolerant::Ceil { formula |k, o, [x]| Some(crate::with_tolerance!(k.0, |t| t.ceil_in(o, x))); slope |o, _, _| o.number(F::zero()); integers formula; native_complex |k, o, [x]| Some(crate::tolerant::TolerantComplex::complex_ceil(native_tolerance(o.tolerance, k.0), x)); exact |_, [x]| Ok(Some(x.ceil())); }
    crate::Not { test |_, o, [x]| Some(o.equal(x, o.number(F::zero()))); integers only; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::Cis { output complex; native_complex |_, _, [x]| Some(crate::complex::exp::<F, { crate::FUSED }>(num_complex::Complex::new(-x.im, x.re))); }
    crate::LambertW { native_complex |_, _, [x]| Some(crate::lambert::complex(x)); native_real |_, _, [x]| Some(crate::lambert::lambertw::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Lgamma { slope |o, x, _| o.digamma(x); native_complex |_, _, [x]| Some(crate::complex::lgamma::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::special::lgamma::<F, { crate::FUSED }>(x)); checks nan; }
    // ψ's slope is ψ⁽¹⁾.
    crate::Digamma { slope |o, x, _| o.polygamma(o.number(F::one()), x); native_real |_, _, [x]| Some(crate::special::digamma::<F, { crate::FUSED }>(x)); }
    // 2/√π e^-x², and its negative for erfc.
    crate::Erf { slope |o, x, _| o.multiply(o.number(F::narrow(std::f64::consts::FRAC_2_SQRT_PI)), o.exp(o.negate(o.multiply(x, x)))); native_real |_, _, [x]| Some(crate::special::erf::<F, { crate::FUSED }>(x)); }
    crate::Erfc { slope |o, x, _| o.multiply(o.number(F::narrow(-std::f64::consts::FRAC_2_SQRT_PI)), o.exp(o.negate(o.multiply(x, x)))); native_real |_, _, [x]| Some(crate::special::erfc::<F, { crate::FUSED }>(x)); }
    // √π/2 e^(erfinv x)², from the result.
    crate::Erfinv { slope |o, _, r| o.multiply(o.number(F::narrow(std::f64::consts::FRAC_2_SQRT_PI.recip())), o.exp(o.multiply(r, r))); native_real |_, _, [x]| Some(crate::special::erfinv::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Erfcinv { slope |o, _, r| o.multiply(o.number(F::narrow(-std::f64::consts::FRAC_2_SQRT_PI.recip())), o.exp(o.multiply(r, r))); native_real |_, _, [x]| Some(crate::special::erfcinv::<F, { crate::FUSED }>(x)); slices unary; checks nan; }
);
objects!(2:
    Comparison { test |k, o, [x, y]| Some(o.compare(k.0, x, y)); integers formula; exact |k, [x, y]| Ok(Some(crate::exact::boolean(match k.0 { crate::recipe::Cmp::Less => x < y, crate::recipe::Cmp::LessEqual => x <= y, crate::recipe::Cmp::Equal => x == y }))); }
    crate::Add { range [x, y] => exact(x.lo + y.lo, x.hi + y.hi); identity [Some(0.0), Some(0.0)]; fold Sum; truth Sum; characters Shift; formula |_, o, [x, y]| Some(o.add(x, y)); slopes |o, _, _, _| [o.number(F::one()), o.number(F::one())]; integers arithmetic; native_complex |_, _, [x, y]| Some(x+y); exact |_, [x, y]| Ok(Some(x + y)); valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Add, x, y); }
    crate::Subtract { range [x, y] => exact(x.lo - y.hi, x.hi - y.lo); identity [None, Some(0.0)]; characters Difference; formula |_, o, [x, y]| Some(o.subtract(x, y)); slopes |o, _, _, _| [o.number(F::one()), o.number(-F::one())]; integers arithmetic; native_complex |_, _, [x, y]| Some(x-y); exact |_, [x, y]| Ok(Some(x - y)); valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Subtract, x, y); }
    crate::Multiply { range [x, y] => corners(x, y, |a, b| a * b); identity [Some(1.0), Some(1.0)]; fold Product; table 8; same |x| x.square(); formula |_, o, [x, y]| Some(o.multiply(x, y)); slopes |_, x, y, _| [y, x]; integers arithmetic; native_complex |_, _, [x, y]| Some(x*y); exact |_, [x, y]| Ok(Some(x * y)); valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Multiply, x, y); }
    crate::Divide { range [x, y] => quotient(x, y); identity [None, Some(1.0)]; formula |_, o, [x, y]| Some(o.divide(x, y)); slopes |o, _, y, r| [o.reciprocal(y), o.negate(o.divide(r, y))]; native_complex |_, _, [x, y]| Some(crate::complex::divide::<F>(x, y)); exact |_, [x, y]| crate::exact::divide(x, y, false); }
    // A nonzero divisor that every item shares divides plainly, since 0 ÷ 0 can't arise.
    crate::MathishDivide {
        range [x, y] => quotient(x, y); identity [None, Some(1.0)];
        formula |_, o, [x, y]| Some(match o.constant(y) { Some(d) if d != F::zero() => o.divide(x, y), _ => crate::recipe::mathish_divide(o, x, y) });
        slopes |o, _, y, r| [o.reciprocal(y), o.negate(o.divide(r, y))];
        native_complex |_, _, [x, y]| Some(if crate::arith::both_zero(x, y) { x } else { crate::complex::divide::<F>(x, y) });
        exact |_, [x, y]| crate::exact::divide(x, y, true);
    }
    // π(x ÷ y).
    crate::PiTimes { range [x, y] => pi_times(quotient(x, y)); formula |_, o, [x, y]| Some(crate::recipe::pi_ratio(o, x, y)); slopes |o, _, y, r| [crate::recipe::pi_ratio(o, o.number(F::one()), y), o.negate(o.divide(r, y))]; native_complex |_, _, [x, y]| Some(crate::complex::divide::<F>(x, y) * F::narrow(std::f64::consts::PI)); }
    // A tie goes to the left argument, as in Futhark.
    Maximum {
        range [x, y] => Range { lo: x.lo.max(y.lo), hi: x.hi.max(y.hi) }; identity [Some(f64::NEG_INFINITY), Some(f64::NEG_INFINITY)]; fold Max; table 14; characters Ordered;
        formula |_, o, [x, y]| Some(o.larger(x, y)); slopes |o, x, y, _| [o.one_where(o.less_equal(y, x)), o.one_where(o.less(x, y))]; integers formula; exact |_, [x, y]| Ok(Some(x.max(y).clone()));
    }
    Minimum {
        range [x, y] => Range { lo: x.lo.min(y.lo), hi: x.hi.min(y.hi) }; identity [Some(f64::INFINITY), Some(f64::INFINITY)]; fold Min; table 8; characters Ordered;
        formula |_, o, [x, y]| Some(o.smaller(x, y)); slopes |o, x, y, _| [o.one_where(o.less_equal(x, y)), o.one_where(o.less(y, x))]; integers formula; exact |_, [x, y]| Ok(Some(x.min(y).clone()));
    }
    crate::Hypot { range [x, y] => { let (a, b) = (x.magnitude(), y.magnitude()); exact(a.lo.max(b.lo), a.hi.hypot(b.hi)).loose() }; native_real |_, _, [x, y]| Some(crate::pow::hypot::<F, { crate::FUSED }>(x, y)); }
    crate::Atan2 { range _ => exact(-PI, PI).loose(); native_real |_, _, [y, x]| Some(crate::arc::atan2::<F, { crate::FUSED }>(y, x)); }
    // B(a, b): B(a, b)(ψ(a) - ψ(a + b)) in a and B(a, b)(ψ(b) - ψ(a + b)) in b, from the result.
    crate::Beta { range args => signs(args); slopes |o, a, b, r| [o.multiply(r, o.subtract(o.digamma(a), o.digamma(o.add(a, b)))), o.multiply(r, o.subtract(o.digamma(b), o.digamma(o.add(a, b))))]; native_real |_, _, [a, b]| Some(crate::special::beta::<F, { crate::FUSED }>(a, b)); checks nan; }
    // A whole exponent that every item shares gives a real power of any base, which the kernel computes through its whole-exponent loop (`WholePow`).
    crate::Pow {
        range [x, y] => powered(x, y); domain [x, y] => if whole(y) { Some([x, y]) } else { inside([x, y], [Range::NONNEGATIVE, Range::ANY]) };
        integer_domain [_, y] => y.lo >= 0.0;
        typed_domain [_, y] => y.is_integral();
        identity [None, Some(1.0)]; formula |_, o, [x, y]| if o.integral() { constant_power(o, x, y) } else { Some(crate::pow::formula(o, x, y)) }; shared power;
        native_integer |_, o, [x, y]| crate::integer::native_power(o.variant, x.into(), y.into()).map(I::low);
        native_complex |_, _, [x, y]| Some(crate::complex::pow::<F, { crate::FUSED }>(x, y)); exact |_, [x, y]| crate::exact::power(x, y); checks nan;
    }
    crate::RealPow { range [x, _] => signs([x]); identity [None, Some(1.0)]; output real; formula |_, o, [x, y]| Some(crate::pow::real_formula(o, x, y)); shared real_power; native_complex |_, _, [x, y]| Some(crate::complex::pow::<F, { crate::FUSED }>(x, y)); }
    crate::WholePow { range [x, y] => powered(x, y); identity [None, Some(1.0)]; formula |_, o, [x, y]| if o.integral() { constant_power(o, x, y) } else { Some(crate::pow::formula(o, x, y)) }; shared power; native_integer |_, o, [x, y]| crate::integer::native_power(o.variant, x.into(), y.into()).map(I::low); native_complex |_, _, [x, y]| Some(crate::complex::pow::<F, { crate::FUSED }>(x, y)); exact |_, [x, y]| crate::exact::power(x, y); checks nan; }
    crate::WrappingPow { identity [None, Some(1.0)]; shared power; native_integer |_, o, [x, y]| crate::integer::native_power(crate::facts::Variant { wrapping: true, ..o.variant }, x.into(), y.into()).map(I::low); }
    // The yth root of x, x^(1/y).
    crate::Root { range [_, y] => signs([y]); domain [n, y] => root_domain(n, y); formula |_, o, [n, y]| Some(crate::pow::root_formula(o, n, y)); slopes |o, x, y, r| [o.negate(o.divide(o.multiply(r, o.ln(y)), o.multiply(x, x))), o.divide(r, o.multiply(x, y))]; native_complex |_, _, [n, y]| Some(crate::complex::nth_root::<F, { crate::FUSED }>(n, y)); exact |_, [n, y]| { if num_traits::Zero::is_zero(n) { return Err(crate::error::Error::new(crate::error::Kind::Domain, "root degree is zero")); } crate::exact::power(y, &n.recip()) }; checks nan; }
    crate::Logb { domain [x, y] => inside([x, y], [Range::NONNEGATIVE; 2]); shared logb; native_complex |_, _, [b, x]| Some(crate::complex::divide(crate::complex::ln::<F, { crate::FUSED }>(x), crate::complex::ln::<F, { crate::FUSED }>(b))); native_real |_, _, [b, x]| Some(crate::ln::logb::<F, { crate::FUSED }>(b, x)); checks nan; }
    crate::MathishLogb { domain [x, y] => inside([x, y], [Range::NONNEGATIVE; 2]); shared logb; native_complex |_, _, [b, x]| { let one = num_complex::Complex::new(F::one(), F::zero()); Some(if b == one && x == one { num_complex::Complex::new(F::zero(), F::zero()) } else { crate::complex::divide(crate::complex::ln::<F, { crate::FUSED }>(x), crate::complex::ln::<F, { crate::FUSED }>(b)) }) }; native_real |_, _, [b, x]| Some(if b == F::one() && x == F::one() { F::zero() } else { crate::ln::logb::<F, { crate::FUSED }>(b, x) }); checks nan; }
    // `y` modulo `x`. Its slope in `x` is minus the quotient's floor, (y - r) ÷ x from the result, and 0 where `x` is 0, where the residue is `y` itself. Integers take a positive power of two that every item shares as a mask (`masked`).
    crate::tolerant::Residue {
        identity [Some(0.0), None];
        formula |k, o, [x, y]| if o.integral() { masked(o, x, y) } else { (!k.0.is_exact()).then(|| k.0.residue_in(o, x, y)) };
        slopes |o, x, y, r| [{
            let at_zero = o.equal(x, o.number(F::zero()));
            o.negate(o.multiply(o.one_where(o.not(at_zero)), o.divide(o.subtract(y, r), o.add(x, o.one_where(at_zero)))))
        }, o.number(F::one())];
        native_integer |_, _, [x, y]| Some(I::low(crate::integer::residue(x.into(), y.into())));
        native_complex |k, o, [x, y]| Some(crate::tolerant::TolerantComplex::complex_residue(native_tolerance(o.tolerance, k.0), x, y));
        native_real |k, o, [x, y]| {
            let t = native_tolerance(o.tolerance, k.0);
            Some(if t.is_exact() { crate::tolerant::exact_residue(x, y) } else { crate::tolerant::Tolerant::residue(t, x, y) })
        };
        exact |_, [x, y]| Ok(Some(crate::exact::residue(x, y)));
    }
    crate::WholeDivide { identity [None, Some(1.0)]; native_integer |_, o, [x, y]| { let (x, y) = (x.into(), y.into()); if x.checked_rem(y) == Some(0) { x.checked_div(y).map(|x| o.result(x)) } else { None } }; exact |_, [x, y]| { if num_traits::Zero::is_zero(y) { return Ok(None); } let q = x / y; Ok(q.is_integer().then_some(q)) }; }
    // C(n, k), with n first.
    crate::Binom {
        slopes |o, n, k, _| [o.binom_dn(n, k), o.binom_dk(n, k)];
        native_integer |_, o, [n, k]| {
            let (n, k) = (n.into(), k.into());
            if k < 0 { return None; }
            let value = crate::integer::binom(n, k).unwrap_or_else(|| {
                if o.variant.checked { (crate::error::Kind::Overflow.reporter())(); }
                0
            });
            Some(o.result(value))
        };
        native_complex |_, _, [n, k]| Some(crate::complex::binom::<F, { crate::FUSED }>(n, k));
        native_real |_, _, [n, k]| Some(crate::special::binom::<F, { crate::FUSED }>(n, k));
        exact |_, [n, k]| {
            if !k.is_integer() { return Ok(None); }
            let k = num_traits::ToPrimitive::to_i64(k).ok_or_else(|| crate::error::Error::new(crate::error::Kind::Limit, "exact binomial count is too large"))?;
            crate::exact::binom(n, k).map(Some)
        }; checks nan;
    }
    crate::tolerant::FloorDivide {
        formula |k, o, [x, y]| Some(if o.integral() { crate::recipe::integer_floor_divide(o, x, y) }
            else { crate::with_tolerance!(k.0, |t| t.floor_in(o, o.divide(x, y))) });
        slopes |o, _, _, _| [o.number(F::zero()), o.number(F::zero())];
        integers arithmetic;
        exact |_, [x, y]| crate::exact::floor_divide(x, y, false);
        valid |_, v, o, [x, y], _| integer_division_valid(v, o, x, y, false);
    }
    crate::tolerant::MathishFloorDivide {
        formula |k, o, [x, y]| Some(if o.integral() { crate::recipe::integer_floor_divide(o, x, y) }
            else { crate::with_tolerance!(k.0, |t| t.floor_in(o, crate::recipe::mathish_divide(o, x, y))) });
        slopes |o, _, _, _| [o.number(F::zero()), o.number(F::zero())];
        integers arithmetic;
        exact |_, [x, y]| crate::exact::floor_divide(x, y, true);
        valid |_, v, o, [x, y], _| integer_division_valid(v, o, x, y, true);
    }
    crate::tolerant::Gcd {
        identity [Some(0.0), Some(0.0)]; tolerance |k| k.0.value();
        native_integer |_, o, [x, y]| crate::integer::gcd(x.into(), y.into()).map(|x| o.result(x));
        native_complex |k, o, [x, y]| Some(native_tolerance(o.tolerance, k.0).complex_gcd(x, y));
        native_real |k, o, [x, y]| Some(native_tolerance(o.tolerance, k.0).gcd(x, y));
        exact |_, [x, y]| Ok(Some(crate::exact::gcd(x, y))); checks nan;
    }
    crate::tolerant::Lcm {
        identity [Some(1.0), Some(1.0)]; tolerance |k| k.0.value();
        native_integer |_, o, [x, y]| crate::integer::lcm(x.into(), y.into()).map(|x| o.result(x));
        native_complex |k, o, [x, y]| Some(lcm_from_gcd(x, y, native_tolerance(o.tolerance, k.0).complex_gcd(x, y)));
        native_real |k, o, [x, y]| Some(lcm_from_gcd(x, y, native_tolerance(o.tolerance, k.0).gcd(x, y)));
        exact |_, [x, y]| Ok(Some(crate::exact::lcm(x, y))); checks nan;
    }
    crate::Nand { table 7; test |_, o, [x, y]| Some(o.not(o.and(o.equal(x, o.number(F::one())), o.equal(y, o.number(F::one()))))); integers only; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::Nor { table 1; test |_, o, [x, y]| Some(o.and(o.equal(x, o.number(F::zero())), o.equal(y, o.number(F::zero())))); integers only; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::BitAnd { fold And; table 8; formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::And, o.to_bits(x), o.to_bits(y)))); integers only; }
    crate::BitOr { fold Or; table 14; formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::Or, o.to_bits(x), o.to_bits(y)))); integers only; }
    crate::BitXor { fold Xor; table 6; formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::Xor, o.to_bits(x), o.to_bits(y)))); integers only; }
    crate::LeftShift { formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::ShiftLeft, o.to_bits(x), o.to_integer(y)))); integers bits; valid |_, v, o, [_, y], _| shift_valid(v, o, y); }
    crate::RightShift {
        formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::SignedShiftRight, o.to_bits(x), o.to_integer(y))));
        integers bits; valid |_, v, o, [_, y], _| shift_valid(v, o, y);
    }
    crate::reduce::Xor { fold Xor; table 6; test |_, o, [x, y]| Some(o.not(o.equal(x, y))); integers only; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::reduce::Xnor { fold Xnor; table 9; test |_, o, [x, y]| Some(o.equal(x, y)); integers only; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::tolerant::Equal {
        identity [Some(1.0), Some(1.0)]; table 9; characters Ordered; tolerance |k| k.0.value();
        test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); integers formula;
        native_complex |k, o, [x, y]| Some(if native_tolerance(o.tolerance, k.0).complex_equal(x, y) { Complex::one() } else { Complex::zero() });
        exact |_, [x, y]| Ok(Some(crate::exact::boolean(x == y)));
    }
    crate::tolerant::NotEqual {
        identity [Some(0.0), Some(0.0)]; table 6; characters Ordered; tolerance |k| k.0.value();
        test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); integers formula;
        native_complex |k, o, [x, y]| Some(if !native_tolerance(o.tolerance, k.0).complex_equal(x, y) { Complex::one() } else { Complex::zero() });
        exact |_, [x, y]| Ok(Some(crate::exact::boolean(x != y)));
    }
    crate::tolerant::Less { identity [Some(0.0), None]; table 2; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x < y))); }
    crate::tolerant::LessEqual { identity [Some(1.0), None]; table 11; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x <= y))); }
    crate::tolerant::Greater { identity [None, Some(0.0)]; table 4; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x > y))); }
    crate::tolerant::GreaterEqual { identity [None, Some(1.0)]; table 13; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x >= y))); }
    crate::BinomDk { shared binom_dk_slopes; native_real |_, _, [n, k]| Some(crate::special::binom_dk::<F, { crate::FUSED }>(n, k)); checks nan; }
    crate::BinomDn { shared binom_dn_slopes; native_real |_, _, [n, k]| Some(crate::special::binom_dn::<F, { crate::FUSED }>(n, k)); checks nan; }
    crate::Gammainc { shared gammainc_slopes; native_real |_, _, [a, x]| Some(crate::special::gammainc::<F, { crate::FUSED }>(a, x)); checks nan; }
    crate::Gammaincc { native_real |_, _, [a, x]| Some(crate::special::gammaincc::<F, { crate::FUSED }>(a, x)); checks nan; }
    crate::Gammaincinv { native_real |_, _, [a, p]| Some(crate::special::gammaincinv::<F, { crate::FUSED }>(a, p)); checks nan; }
    crate::Polygamma { shared polygamma_slopes; native_real |_, _, [n, x]| Some(crate::special::polygamma::<F, { crate::FUSED }>(n, x)); checks nan; }
);
objects!(3:
    crate::Clip { formula |_, o, [x, lo, hi]| Some(o.smaller(hi, o.larger(lo, x))); integers formula; }
    crate::Choose { formula |_, o, [x, y, mask]| Some(o.select(o.not(o.equal(mask, o.number(F::zero()))), x, y)); integers formula; native_complex |_, _, [x, y, mask]| Some(if mask.re != F::zero() { x } else { y }); }
    crate::Betainc {
        range _ => exact(0.0, 1.0);
        native_real |_, _, [a, b, x]| Some(crate::special::betainc::<F, { crate::FUSED }>(a, b, x)); checks nan;
    }
    crate::Betaincinv {
        range _ => exact(0.0, 1.0);
        native_real |_, _, [a, b, p]| Some(crate::special::betaincinv::<F, { crate::FUSED }>(a, b, p)); checks nan;
    }
);

/// Whether `y` is one whole number, as a constant whole exponent is.
fn whole(y: Range) -> bool { y.constant().is_some_and(|k| k.fract() == 0.0) }

/// The range of `x` to the power `y`: nonnegative where `y` is one even whole number or `x` is nonnegative.
fn powered(x: Range, y: Range) -> Range { if y.constant().is_some_and(|k| matches!(parity(k), (true, false))) { Range::NONNEGATIVE } else { signs([x]) } }

/// `y` modulo `x` for integers, where `x` is a positive power of two that every item shares (`Ops::constant`): the bitwise and of `y` with rustymath's mask for it (`integer::mask`), as compiled code takes it. `None` for any other divisor.
fn masked<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, x: O::V, y: O::V) -> Option<O::V> {
    let d = o.constant(x)?.wide();
    let mask = crate::integer::mask(d as i64).filter(|_| d.fract() == 0.0)?;
    Some(o.bit_and(y, o.number(F::narrow(mask as f64))))
}

#[inline]
fn native_tolerance(value: Option<f64>, fallback: Tolerance) -> Tolerance { value.and_then(Tolerance::new).unwrap_or(fallback) }

/// Comparison `k`'s truth value through `o`, as its kernel computes it: against a number that one argument holds (`Ops::constant`), through the range of floats equal to that number at tolerance `t` (`tolerant::against_steps`), and otherwise step by step (`tolerant::Stepped`).
fn tested<K, F, O>(k: K, t: crate::tolerant::Tolerance, o: &O, x: O::V, y: O::V) -> O::B
where K: crate::tolerant::Stepped + crate::tolerant::Banding + Copy, F: crate::Float, O: crate::recipe::Ops<F> + ?Sized {
    let against = match (o.constant(x), o.constant(y)) { (_, Some(c)) => Some((c, x, true)), (Some(c), None) => Some((c, y, false)), _ => None };
    let tolerance = o.tolerance(F::narrow(t.value()));
    let known = o.constant(tolerance).and_then(|t| crate::tolerant::Tolerance::new(t.wide()));
    against.and_then(|(c, v, left)| crate::tolerant::against_steps(k, known?, o, c, v, left)).unwrap_or_else(|| k.steps::<F, O>(o, x, y))
}

/// A bitwise function's facts, which its table decides: the tables that fold (and, or, xor, xnor) for numbers, and the same tables' reductions for Booleans.
impl Formula<1> for crate::Bitwise {
    fn integer_formula(&self) -> bool { true }
    fn supports(&self, dtype: DType) -> bool { <Self as Formula<2>>::supports(self, dtype) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [y]: [O::V; 1]) -> Option<O::V> { <Self as Formula<2>>::emit(self, o, [o.number(F::zero()), y]) }
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, [y]: [I; 1]) -> Option<I> { <Self as Formula<1>>::emit(self, o, [y]) }
}
impl Formula<2> for crate::Bitwise {
    fn fold(&self) -> Option<crate::reduce::Reduce> { use crate::reduce::Reduce::*; match self.0 { 8 => Some(And), 14 => Some(Or), 6 => Some(Xor), 9 => Some(Xnor), _ => None } }
    fn truth_table(&self) -> Option<u8> { Some(self.0 as u8) }
    fn integer_formula(&self) -> bool { true }
    fn supports(&self, dtype: DType) -> bool { dtype.is_integral() && (dtype != DType::Integer(crate::buffer::Width::U8) || self.0 & 1 == 0) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::V> {
        use crate::recipe::Integer;
        o.integral().then(|| o.from_bits(crate::arith::bitwise_with(self.0, o.to_bits(x), o.to_bits(y), o.integer(-1),
            |a, b| o.integer_op(Integer::And, a, b), |a, b| o.integer_op(Integer::Xor, a, b))))
    }
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, x: [I; 2]) -> Option<I> { <Self as Formula<2>>::emit(self, o, x) }
}

impl<K: Formula<2>> Formula<2> for Flipped<K> {
    fn truth_table(&self) -> Option<u8> { self.0.truth_table().map(|t| (t & 9) | (t & 2) << 1 | (t & 4) >> 1) }
    fn tolerance(&self) -> Option<f64> { self.0.tolerance() }
    fn character_policy(&self, [x, y]: [DType; 2]) -> Result<Option<crate::characters::Policy>, crate::error::Error> {
        use crate::characters::{Policy, ByKinds};
        self.0.character_policy([y, x]).map(|p| p.map(|p| match p {
            Policy::Kinds(ByKinds::Left) => Policy::Kinds(ByKinds::Right), Policy::Kinds(ByKinds::Right) => Policy::Kinds(ByKinds::Left), p => p,
        }))
    }
    fn exact(&self, [x, y]: [&num_rational::BigRational; 2]) -> Result<Option<num_rational::BigRational>, crate::error::Error> { self.0.exact([y, x]) }
    fn has_exact(&self) -> bool { self.0.has_exact() }
    fn character_rule(&self) -> Option<crate::characters::Rule> { self.0.character_rule() }
    fn result_dtype(&self, input: DType) -> DType { self.0.result_dtype(input) }
    fn range(&self, [x, y]: [Range; 2]) -> Range { self.0.range([y, x]) }
    fn domain(&self, [x, y]: [Range; 2]) -> Option<[Range; 2]> { self.0.domain([y, x]).map(|[y, x]| [x, y]) }
    fn integer_domain(&self, [x, y]: [Range; 2]) -> bool { self.0.integer_domain([y, x]) }
    fn typed_domain(&self, [x, y]: [DType; 2]) -> bool { self.0.typed_domain([y, x]) }
    fn has_domain(&self) -> bool { self.0.has_domain() }
    fn identity(&self) -> [Option<f64>; 2] { let [x, y] = self.0.identity(); [y, x] }
    fn integer_formula(&self) -> bool { self.0.integer_formula() }
    fn supports(&self, dtype: DType) -> bool { self.0.supports(dtype) }
    fn boolean(&self) -> bool { self.0.boolean() }
    fn fails_nan(&self) -> bool { self.0.fails_nan() }
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, [x, y]: [I; 2]) -> Option<I> { self.0.integer(o, [y, x]) }
    fn real<F: crate::float::Native>(&self, o: &crate::recipe::NativeOps<{ crate::FUSED }>, [x, y]: [F; 2]) -> Option<F> { self.0.real(o, [y, x]) }
    fn complex<F: crate::float::Native>(&self, o: &crate::recipe::NativeOps<{ crate::FUSED }>, [x, y]: [num_complex::Complex<F>; 2]) -> Option<num_complex::Complex<F>> { self.0.complex(o, [y, x]) }
    fn emit_valid<F: crate::Float, O: Ops<F> + ?Sized>(&self, v: Variant, o: &O, [x, y]: [O::V; 2], r: O::V) -> Option<(crate::error::Kind, O::B)> { self.0.emit_valid(v, o, [y, x], r) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::V> { self.0.emit(o, [y, x]) }
    fn emit_test<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::B> { self.0.emit_test(o, [y, x]) }
    fn emit_partials<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
        let [dy, dx] = self.0.emit_partials(o, [y, x], r, [ny, nx]);
        [dx, dy]
    }
}


impl<K: Formula<1>> Formula<2> for crate::derivative::Slope<K> {
    fn supports(&self, dtype: DType) -> bool { matches!(dtype, DType::Float(_)) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, r]: [O::V; 2]) -> Option<O::V> { Some(self.0.emit_partials(o, [x], r, [true])[0].unwrap_or_else(|| o.number(F::narrow(f64::NAN)))) }
}

impl<K: Formula<2>, const SIDE: usize> Formula<3> for crate::derivative::Slope<K, SIDE> {
    fn supports(&self, dtype: DType) -> bool { matches!(dtype, DType::Float(_)) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y, r]: [O::V; 3]) -> Option<O::V> {
        Some(self.0.emit_partials(o, [x, y], r, std::array::from_fn(|i| i == SIDE))[SIDE].unwrap_or_else(|| o.number(F::narrow(f64::NAN))))
    }
}


/// Calls `$then!` with the fold and product tables after the supplied arguments in parentheses.
/// `folds` and `booleans` give Reduce variants and Python names for numerical and Boolean folds.
/// Each `products` or `boolean_products` row gives map-object/fold pairs, native item types and Python names. An optional `wrapping` list uses the pair's existing fixed-width implementation. Boolean matrix products use OR-AND.
#[macro_export]
macro_rules! facts {
    ($then:ident $(, $arg:tt)*) => { $then! { ($($arg),*)
        folds {
            Sum: ["add"];
            Product: ["multiply"];
            Max: ["maximum"];
            Min: ["minimum"];
            And: ["bitwise_and"];
            Or: ["bitwise_or"];
            Xor: ["bitwise_xor"];
            Xnor: [];
        }
        booleans {
            Sum: [];
            Max: ["logical_or"];
            Min: ["logical_and"];
            Xor: ["logical_xor"];
            Xnor: ["logical_xnor"];
        }
        products {
            PlusTimes: [(crate::Multiply, Sum)] at [i64, half::f16, f32, f64, num_complex::Complex32, num_complex::Complex64] wrapping [u8, i16, i32] => [("add", "multiply")];
            MaxPlus: [(crate::Add, Max)] at [i64, f32, f64] => [("maximum", "add")];
            MinPlus: [(crate::Add, Min)] at [i64, f32, f64] => [("minimum", "add")];
            PlusEqual: [(crate::tolerant::Equal, Sum)] at [i16, i32, i64, f32, f64] => [("add", "equal")];
            PlusNotEqual: [(crate::tolerant::NotEqual, Sum)] at [i16, i32, i64, f32, f64] => [("add", "not_equal")];
        }
        boolean_products {
            OrAnd: [(crate::Multiply, Sum), (crate::BitAnd, Max), (crate::BitAnd, Or), (crate::Multiply, Max)] at [bool] => [("add", "multiply"), ("logical_or", "logical_and")];
            AndOr: [(crate::BitOr, Min), (crate::BitOr, And)] at [bool] => [("logical_and", "logical_or")];
        }
    } };
}

fn arithmetic_valid<F: crate::Float, O: Ops<F> + ?Sized>(variant: Variant, o: &O, op: crate::recipe::Binary, x: O::V, y: O::V) -> Option<(crate::error::Kind, O::B)> {
    (variant.checked && variant.dtype.is_integral() && !variant.wrapping).then(|| (crate::error::Kind::Overflow, o.not(o.overflow(op, x, y))))
}

fn negation_valid<F: crate::Float, O: Ops<F> + ?Sized>(variant: Variant, o: &O, x: O::V, magnitude: bool) -> Option<(crate::error::Kind, O::B)> {
    let (kind, valid) = arithmetic_valid(variant, o, crate::recipe::Binary::Subtract, o.number(F::zero()), x)?;
    Some((kind, if magnitude { o.or(o.less_equal(o.number(F::zero()), x), valid) } else { valid }))
}

fn shift_valid<F: crate::Float, O: Ops<F> + ?Sized>(variant: Variant, o: &O, count: O::V) -> Option<(crate::error::Kind, O::B)> {
    variant.checked.then(|| (crate::error::Kind::Domain, o.and(o.less_equal(o.number(F::zero()), count), o.less(count, o.number(F::narrow((variant.dtype.bytes() * 8) as f64))))))
}

fn truth_valid<F: crate::Float, O: Ops<F> + ?Sized, const N: usize>(variant: Variant, o: &O, args: [O::V; N]) -> Option<(crate::error::Kind, O::B)> {
    variant.checked.then(|| {
        let valid = args.into_iter().fold(o.equal(o.number(F::zero()), o.number(F::zero())), |valid, x| {
            o.and(valid, o.or(o.equal(x, o.number(F::zero())), o.equal(x, o.number(F::one()))))
        });
        (crate::error::Kind::Domain, valid)
    })
}

fn integer_division_valid<F: crate::Float, O: Ops<F> + ?Sized>(variant: Variant, o: &O, x: O::V, y: O::V, mathish: bool) -> Option<(crate::error::Kind, O::B)> {
    if !variant.checked || !o.integral() { return None; }
    let zero = o.number(F::zero());
    let nonzero = o.not(o.equal(y, zero));
    Some((crate::error::Kind::Domain, if mathish { o.or(nonzero, o.equal(x, zero)) } else { nonzero }))
}
