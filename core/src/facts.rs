//! Shared function definitions for numerical frontends. Each [`Formula`] row owns its kind and character policies, numerical body, validity and repair predicates, ranges, identities, partial derivatives and exact cases. `rustymath::operation::Kernel` erases the arity for execution. [`facts!`](crate::facts) lists typed product pairs and Python names.
use crate::buffer::{DType, FloatWidth, Source};
use crate::domain::{Range, Variant};
use crate::float::sealed::Core;
use crate::pow::parity;
use crate::recipe::{NativeOps, Ops};
use crate::reduce::{Maximum, Minimum, Product, Reduce, Reduction};
use crate::tolerant::{Residue, Tolerance, Tolerant, TolerantComplex, lcm_from_gcd};
use crate::{Arg, Betainc, BetaincAt, FUSED, Flipped, Float, Native};
use num_complex::Complex;
use num_traits::{One, Signed, Zero};
use std::f64::consts::PI;

/// From ¯1 to 1: the values of the sine and the cosine, and the arguments on which the inverse sine, the inverse cosine and the inverse hyperbolic tangent are real.
const UNIT: (f64, f64) = (-1.0, 1.0);

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
fn inside<const N: usize>(args: [Range; N], domain: [Range; N]) -> Option<[Range; N]> {
    args.iter().zip(&domain).all(|(a, d)| a.within(d.lo, d.hi)).then_some(domain)
}
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
    const BOOLEAN: bool = false;
    /// Whether a NaN result from arguments that aren't NaN is a Domain error, which the default `emit_valid` checks (`nan_valid`). A constant, so a checked loop over items reads it at compile time and never calls a method for it.
    const FAILS_NAN: bool = false;
    /// Whether `exact` computes this formula on rationals.
    const HAS_EXACT: bool = false;
    /// Whether `integer` runs this formula's `emit` on integers, so that range arithmetic over `emit_valid` bounds an integer kernel's failures (`may_raise`). A row with a `native_integer` body supports integers and leaves this false.
    const INTEGER_FORMULA: bool = false;
    /// Whether kernels of this formula only map. Modified writes, folds, scans and walks of adjacent items build no code for it: each refuses it at compile time, as it refuses a formula of another arity. A derivative's slope (`derivative::Slope`) only maps.
    const MAP_ONLY: bool = false;
    fn result_dtype(&self, input: DType) -> DType { if Self::BOOLEAN { DType::Boolean } else { input } }
    fn computing_dtype(&self, input: DType) -> Option<DType> { input.computes_in(crate::buffer::Width::I64, input.numpy_float(), |d| self.supports(d)) }
    fn character_rule(&self) -> Option<crate::characters::Rule> { None }
    fn ordered_kinds(&self, [x, y]: [i32; 2]) -> Option<i32> {
        let ops = crate::recipe::IntegerOps::<i32>::new(Variant::checked(DType::Integer(crate::buffer::Width::I32)));
        self.integer(&ops, std::array::from_fn(|i| if i == 0 { x } else { y }))
    }
    fn character_policy(&self, types: [DType; 2]) -> Result<Option<crate::characters::Policy>, crate::error::Error> {
        crate::characters::policy(self.character_rule(), types, self.result_dtype(DType::Integer(crate::buffer::Width::I32)), |x| self.ordered_kinds(x))
    }
    fn exact(&self, _args: [&num_rational::BigRational; N]) -> Result<Option<num_rational::BigRational>, crate::error::Error> { Ok(None) }
    fn range(&self, _args: [Range; N]) -> Range { Range::ANY }
    /// A box at the computing dtype containing args on which graph-numeric results do not raise a data error.
    fn domain(&self, _dtype: DType, _args: [Range; N]) -> Option<[Range; N]> { None }
    fn identity(&self) -> [Option<f64>; N] { [None; N] }
    fn fold(&self) -> Option<Reduce> { None }
    fn inverse_fold(&self) -> Option<Reduce> { None }
    /// The function's results on Booleans, as bits: bit `2x+y` for two arguments, bit `x` for one. A Boolean kernel computes by it (`operation::map_truth`).
    fn truth_table(&self) -> Option<u8> { None }
    fn truth(&self) -> Option<Reduce> {
        match self.truth_table()? { 8 => Some(Reduce::Min), 14 => Some(Reduce::Max), 6 => Some(Reduce::Xor), 9 => Some(Reduce::Xnor), _ => None }
    }
    /// Refines the real result interval when both operands denote the same value.
    fn same(&self, _arg: Range) -> Option<Range> { None }
    fn reads(&self) -> Reads { Reads::Items }
    fn supports(&self, _dtype: DType) -> bool { false }
    fn emit<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _x: [O::V; N]) -> Option<O::V> { None }
    fn emit_test<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _x: [O::V; N]) -> Option<O::B> { None }
    /// This comparison of `v` with the one number `c` through `o`, where that has a cheaper recipe than comparing item by item, as a held constant has. `left` says whether `v` is the left argument. `None` for other formulas.
    fn emit_against<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _c: F, _v: O::V, _left: bool) -> Option<O::B> { None }
    /// This comparison's tolerance, and the bands that an item on the left and on the right of one number must lie in (`tolerant::Banding`), through which it compares many items with one number (`tolerant::against`). `None` for other formulas.
    fn bands(&self) -> Option<(crate::tolerant::Tolerance, crate::tolerant::Band, crate::tolerant::Band)> { None }
    fn emit_repair<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _x: [O::V; N]) -> Option<O::B> { None }
    fn repair_routine(&self) -> Option<Repair> { None }
    fn emit_valid<F: crate::Float, O: Ops<F> + ?Sized>(&self, _variant: Variant, o: &O, x: [O::V; N], result: O::V) -> Option<(crate::error::Kind, O::B)> {
        Self::FAILS_NAN.then(|| (crate::error::Kind::Domain, nan_valid(o, x, [result])))
    }
    #[inline]
    fn real<F: crate::float::Native>(&self, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, x: [F; N]) -> Option<F> {
        if const { Self::BOOLEAN } { self.emit_test(ops, x).map(|x| if x { F::one() } else { F::zero() }) } else { self.emit(ops, x) }
    }
    fn integer<I: crate::recipe::Overflowing>(&self, _ops: &crate::recipe::IntegerOps<I>, _x: [I; N]) -> Option<I> { None }
    fn complex<F: crate::float::Native>(
        &self, _ops: &crate::recipe::NativeOps<{ crate::FUSED }>, _x: [num_complex::Complex<F>; N]) -> Option<num_complex::Complex<F>> { None }
    /// Whether `real_complex` gives this formula's complex results from real arguments.
    const REAL_COMPLEX: bool = false;
    /// The complex result from real arguments, where that costs less than from the arguments as complex numbers. `None` sends the item to `complex`.
    fn real_complex<F: crate::float::Native>(&self, _ops: &crate::recipe::NativeOps<{ crate::FUSED }>, _x: [F; N]) -> Option<num_complex::Complex<F>> { None }
    fn native_slices(&self) -> Option<NativeSlices<Self, N>> { None }
    fn emit_partials<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _o: &O, _x: [O::V; N], _result: O::V, _needed: [bool; N]) -> [Option<O::V>; N] {
        [None; N]
    }
}

/// A use of a formula object by code generic over formulas, as [`Op::visit`] hands it over. `unary` and `binary` receive the one- and two-argument objects, through `visit` unless a visitor wraps them, as `rustymath::operation::Kernel::slope` does.
pub trait Visit: Sized {
    type Out;
    fn visit<K: Formula<N>, const N: usize>(self, object: K) -> Self::Out;
    fn unary<K: Formula<1>>(self, object: K) -> Self::Out { self.visit(object) }
    fn binary<K: Formula<2>>(self, object: K) -> Self::Out { self.visit(object) }
}

/// Writes `Op` and `Op::visit` from one list: objects without fields by arity (`reduce`'s are binary), then objects that hold a value, with the arity and constructor of each.
macro_rules! ops {
    (unary [$($u:ident)*] binary [$($b:ident)*] reduce [$($r:ident)*] ternary [$($t:ident)*] held [$($h:ident($ty:ty) => $how:ident $make:expr),* $(,)?]) => {
        /// A formula object by name, for a front end that picks its formula at run time, as BPL's primitives and rustymath's Python functions do: `rustymath::operation::Kernel::of` resolves one, so each formula's kernel code compiles once in rustymath-kernels, and [`Op::visit`] hands its object to code generic over formulas.
        #[derive(Clone, Copy, Debug)]
        pub enum Op { $($u,)* $($b,)* $($r,)* $($t,)* $($h($ty),)* PiTimes, Bitwise(u8), BetaincAt(f64, f64), FlippedBinom }
        impl Op {
            /// `v` on this formula's object. `arity` picks between the two forms of `PiTimes` and of `Bitwise`, which take one argument or two; every other object has one form.
            pub fn visit<V: Visit>(self, arity: usize, v: V) -> V::Out {
                match self {
                    $(Op::$u => v.unary(crate::$u),)* $(Op::$b => v.binary(crate::$b),)* $(Op::$r => v.binary(crate::reduce::$r),)* $(Op::$t => v.visit(crate::$t),)*
                    $(Op::$h(x) => v.$how($make(x)),)*
                    Op::PiTimes if arity == 1 => v.unary(crate::PiTimes), Op::PiTimes => v.binary(crate::PiTimes),
                    Op::Bitwise(table) if arity == 1 => v.unary(crate::Bitwise(table)), Op::Bitwise(table) => v.binary(crate::Bitwise(table)),
                    Op::BetaincAt(a, b) => v.unary(BetaincAt(a, b)), Op::FlippedBinom => v.binary(Flipped(crate::Binom)),
                }
            }
        }
    };
}
ops! {
    unary [Conj Negate Sign Reciprocal Abs Exp Ln Log2 Log10 Log1p Expm1 Cis Sqrt Cbrt RealSqrt RealLn Factorial Gamma Lgamma Digamma LambertW Erf Erfc Erfinv Erfcinv
        Not Isnan Isfinite Isinf Real Imag Angle Trunc Rint Sin Cos Tan Asin Acos Atan Sinh Cosh Tanh Asinh Acosh Atanh Sqrt1mx2 Sqrt1px2 Sqrtx2m1]
    binary [Add Subtract Multiply Divide MathishDivide Pow Root RealPow Logb MathishLogb Nand Nor Hypot Atan2 Binom BinomDn BinomDk Polygamma Beta Gammainc Gammaincc
        Gammaincinv LeftShift RightShift]
    reduce [Minimum Maximum Xor Xnor]
    ternary [Clip Betainc Betaincinv]
    held [
        Floor(Tolerance) => unary crate::tolerant::Floor, Ceil(Tolerance) => unary crate::tolerant::Ceil, Residue(Tolerance) => binary crate::tolerant::Residue,
        FlippedResidue(Tolerance) => binary |t| Flipped(crate::tolerant::Residue(t)), FloorDivide(Tolerance) => binary crate::tolerant::FloorDivide,
        MathishFloorDivide(Tolerance) => binary crate::tolerant::MathishFloorDivide, Gcd(Tolerance) => binary crate::tolerant::Gcd, Lcm(Tolerance) => binary crate::tolerant::Lcm,
        Equal(Tolerance) => binary crate::tolerant::Equal, NotEqual(Tolerance) => binary crate::tolerant::NotEqual, Less(Tolerance) => binary crate::tolerant::Less,
        LessEqual(Tolerance) => binary crate::tolerant::LessEqual, Greater(Tolerance) => binary crate::tolerant::Greater,
        GreaterEqual(Tolerance) => binary crate::tolerant::GreaterEqual, Scale(f64) => unary crate::Scale, Round(crate::Round) => unary std::convert::identity,
        Pdf(crate::distribution::Distribution) => unary crate::distribution::Pdf, Cdf(crate::distribution::Distribution) => unary crate::distribution::Cdf,
        Quantile(crate::distribution::Distribution) => unary crate::distribution::Quantile,
    ]
}
impl Op {
    /// The dtype this formula computes in for arguments of `input` (`Formula::computing_dtype`), with `arity` as [`Op::visit`] takes it.
    pub fn computing_dtype(self, arity: usize, input: DType) -> Option<DType> {
        struct Of(DType);
        impl Visit for Of {
            type Out = Option<DType>;
            fn visit<K: Formula<N>, const N: usize>(self, object: K) -> Option<DType> { object.computing_dtype(self.0) }
        }
        self.visit(arity, Of(input))
    }
    /// Whether this formula has a body at `dtype` (`Formula::supports`), with `arity` as [`Op::visit`] takes it.
    pub fn supports(self, arity: usize, dtype: DType) -> bool {
        struct Of(DType);
        impl Visit for Of {
            type Out = bool;
            fn visit<K: Formula<N>, const N: usize>(self, object: K) -> bool { object.supports(self.0) }
        }
        self.visit(arity, Of(dtype))
    }
}
/// A formula's native slice bodies. Real and integer variants run the body at their dtype (f32 for f16). Complex variants keep the real-input shortcut and final validity check at the map boundary; an absent body uses the scalar formula.
pub struct NativeSlices<K, const N: usize> {
    pub single: Option<Slice<K, f32, N>>, pub double: Option<Slice<K, f64, N>>, pub byte: Option<Slice<K, u8, N>>, pub short: Option<Slice<K, i16, N>>,
    pub int: Option<Slice<K, i32, N>>, pub long: Option<Slice<K, i64, N>>, pub complex_single: Option<Slice<K, num_complex::Complex32, N, ()>>,
    pub complex_double: Option<Slice<K, num_complex::Complex64, N, ()>>, pub real_complex_single: Option<Slice<K, f32, N, (), num_complex::Complex32>>,
    pub real_complex_double: Option<Slice<K, f64, N, (), num_complex::Complex64>>,
    pub complex_test_single: Option<Slice<K, num_complex::Complex32, N, (), bool>>,
    pub complex_test_double: Option<Slice<K, num_complex::Complex64, N, (), bool>>
}
/// A slice form writes all `n` results at `U` from arguments at `T`. The caller reports errors from its verdict `R`: `bool` for real/integer forms, `()` for complex forms.
pub type Slice<K, T, const N: usize, R = bool, U = T> = fn(&K, &NativeOps<{ FUSED }>, [crate::Arg<'_, T>; N], usize, &mut [std::mem::MaybeUninit<U>]) -> R;
impl<K, const N: usize> NativeSlices<K, N> {
    const NONE: Self = Self { single: None, double: None, byte: None, short: None, int: None, long: None, complex_single: None, complex_double: None,
        real_complex_single: None, real_complex_double: None, complex_test_single: None, complex_test_double: None };
    fn with_complex(self, single: Slice<K, num_complex::Complex32, N, ()>, double: Slice<K, num_complex::Complex64, N, ()>) -> Self {
        Self { complex_single: Some(single), complex_double: Some(double), ..self }
    }
    fn with_real_complex(self, single: Slice<K, f32, N, (), num_complex::Complex32>, double: Slice<K, f64, N, (), num_complex::Complex64>) -> Self {
        Self { real_complex_single: Some(single), real_complex_double: Some(double), ..self }
    }
    fn with_complex_tests(self, single: Slice<K, num_complex::Complex32, N, (), bool>, double: Slice<K, num_complex::Complex64, N, (), bool>) -> Self {
        Self { complex_test_single: Some(single), complex_test_double: Some(double), ..self }
    }
}
impl<K> NativeSlices<K, 1> {
    pub(crate) fn unary(single: Slice<K, f32, 1>, double: Slice<K, f64, 1>) -> Self { Self { single: Some(single), double: Some(double), ..Self::NONE } }
}
impl NativeSlices<crate::Pow, 2> {
    fn power() -> Self {
        Self { single: Some(|k, _, x, n, out| crate::pow::power_slice::<f32>(k, x, n, out)),
            double: Some(|k, _, x, n, out| crate::pow::power_slice::<f64>(k, x, n, out)), ..Self::NONE }
    }
}
impl NativeSlices<crate::Root, 2> {
    fn root() -> Self {
        Self { single: Some(|k, _, x, n, out| crate::pow::root_slice::<f32>(k, x, n, out)),
            double: Some(|k, _, x, n, out| crate::pow::root_slice::<f64>(k, x, n, out)), ..Self::NONE }
    }
}
impl NativeSlices<Residue, 2> {
    fn residue() -> Self {
        Self { single: Some(crate::tolerant::residue_slice), double: Some(crate::tolerant::residue_slice), byte: Some(crate::integer::residue_byte),
            short: Some(crate::integer::residue_short), int: Some(crate::integer::residue_int), long: Some(crate::integer::residue_long), ..Self::NONE }
    }
}

impl NativeSlices<crate::Logb, 2> {
    fn logarithm() -> Self {
        Self { single: Some(|_, o, x, n, out| logarithm_slice::<f32, false>(o, x, n, out)),
            double: Some(|_, o, x, n, out| logarithm_slice::<f64, false>(o, x, n, out)), ..Self::NONE }
    }
}
impl NativeSlices<crate::MathishLogb, 2> {
    fn mathish_logarithm() -> Self {
        Self { single: Some(|_, o, x, n, out| logarithm_slice::<f32, true>(o, x, n, out)),
            double: Some(|_, o, x, n, out| logarithm_slice::<f64, true>(o, x, n, out)), ..Self::NONE }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Repair { pub single: extern "C" fn(f32) -> f32, pub double: extern "C" fn(f64) -> f64 }

#[inline(always)]
pub fn nan_valid<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, inputs: impl IntoIterator<Item = O::V>, results: impl IntoIterator<Item = O::V>) -> O::B {
    let mut valid = o.equal(o.number(F::zero()), o.number(F::zero()));
    for result in results { valid = o.and(valid, o.equal(result, result)); }
    for input in inputs { valid = o.or(valid, o.not(o.equal(input, input))); }
    valid
}

fn logarithm_valid<F: Float, O: Ops<F> + ?Sized>(o: &O, x: [O::V; 2], result: O::V) -> Option<(crate::error::Kind, O::B)> {
    let zero = o.number(F::zero());
    let nonnegative = o.and(o.less_equal(zero, x[0]), o.less_equal(zero, x[1]));
    Some((crate::error::Kind::Domain, o.or(nonnegative, nan_valid(o, x, [result]))))
}

fn logarithm_slice<F: Native, const MATHISH: bool>(ops: &NativeOps<{ FUSED }>, args: [Arg<'_, F>; 2], n: usize, out: &mut [std::mem::MaybeUninit<F>]) -> bool {
    crate::ln::logb_many::<F, MATHISH>(args, n, out) || {
        // SAFETY: logb_many initializes every output, even when it reports a NaN result.
        let results = unsafe { crate::slice::written(&mut out[..n]) };
        crate::slice::judged(&args, results, |x, r| logarithm_valid(ops, x, r).unwrap().1)
    }
}

impl Repair {
    pub fn apply<F: crate::Float>(self, x: F) -> F { F::narrow(if F::Working::WIDE { (self.double)(x.wide()) } else { (self.single)(x.wide() as f32) as f64 }) }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Reads { Items, Whole }

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Slot(pub u32);

/// `k`'s scalar real result, including repairs, computed at the argument's working precision.
#[inline(always)]
pub fn formula_at<K: Formula<N>, F: crate::Float, const N: usize>(k: &K, x: [F; N]) -> F {
    let o = crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT;
    let x = x.map(F::working);
    let result = if k.emit_repair(&o, x) == Some(true) { k.repair_routine().expect("a repair predicate has its routine").apply(x[0]) }
    else { k.real(&o, x).expect("a real body that gives every result") };
    F::from_working(result)
}

/// A real formula's slice body and rare-item repairs, without recording errors from intermediate values.
pub fn real_slice<K: Formula<N>, F: Native + Source<O>, O: Float, const N: usize>(
    object: &K, ops: &NativeOps<{ FUSED }>, args: [Arg<'_, F>; N], n: usize, out: &mut [std::mem::MaybeUninit<O>]) -> bool {
    let body = |x| object.real(ops, x).expect("a resolved object has a real body");
    if let Some(repair) = object.repair_routine() {
        let needs = |x| object.emit_repair(ops, x).unwrap_or(false);
        crate::slice::two_pass(args, n, out, |x| (!needs(x)).then(|| body(x)), |x| if needs(x) { repair.apply(x[0]) } else { body(x) })
    } else { crate::slice::each(args, n, out, #[inline(always)] |x| Source::<O>::read(body(x))) }
}

/// Bit `2x+y` of `table`, or bit `x` for one argument: a Boolean function's result by its truth table.
#[inline(always)]
pub fn truth_bit<const N: usize>(table: u8, x: [bool; N]) -> bool { (table >> x.iter().fold(0, |at, &b| 2 * at + u8::from(b))) & 1 != 0 }

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
    (@test_output $body:expr) => { const BOOLEAN: bool = true; };
    (@output $dtype:ident, real) => { match $dtype { DType::Complex(w) => DType::Float(w.into()), _ => $dtype } };
    (@output $dtype:ident, integer) => { match $dtype { DType::Boolean => DType::Integer(crate::buffer::Width::I64), _ => $dtype } };
    (@output $dtype:ident, complex) => {
        match $dtype.inexact(crate::buffer::FloatWidth::F64) {
            Some((DType::Float(w), _)) => DType::Complex(w.into()), Some((d, _)) => d, None => $dtype,
        }
    };
    (@computing real) => {};
    (@computing integer) => {};
    (@computing complex) => { fn computing_dtype(&self, input: DType) -> Option<DType> { let dtype = input.complex(); self.supports(dtype).then_some(dtype) } };
    (@checks nan) => { const FAILS_NAN: bool = true; };
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
        const INTEGER_FORMULA: bool = true;
        #[inline(always)]
        fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, x: [I; $n]) -> Option<I> {
            if const { Self::BOOLEAN } { self.emit_test(o, x).map(|x| I::low(i64::from(x))) } else { self.emit(o, x) }
        }
    };
    ($n:tt: $($object:ty {
        $(range $args:pat => $range:expr;)? $(domain $domain_dtype:pat_param, $targs:pat => $domain:expr;)? $(identity $identity:expr;)? $(fold $fold:ident;)? $(inverse_fold $inverse_fold:ident;)? $(truth $truth:ident;)? $(table $table:expr;)?
        $(output $output:ident;)? $(characters $rule:ident;)? $(same |$sx:pat_param| $same:expr;)? $(formula |$k:pat_param, $o:tt, $fargs:tt| $formula:expr;)? $(test |$tk:pat_param, $to:tt, $targs2:tt| $test:expr;)? $(against |$ak:pat_param| $atol:expr;)?
        $(slope |$so:tt, $sx1:tt, $sr:tt| $slope:expr;)? $(slopes |$s2o:tt, $s2x:tt, $s2y:tt, $s2r:tt| [$dx:expr, $dy:expr];)? $(shared $shared:ident;)?
        $(repair $repair:expr;)?
        $(integers $integer:ident;)?
        $(native_integer |$nik:pat_param, $nio:tt, $nix:tt| $native_integer:expr;)?
        $(native_complex |$nfk:pat_param, $nfo:tt, $nfx:tt| $native_complex:expr;)?
        $(complex_slices $complex_slices:ident;)?
        $(complex_tests $complex_tests:ident;)?
        $(real_complex_slices $real_complex_slices:ident;)?
        $(native_real |$nrk:pat_param, $nro:tt, $nrx:tt| $native_real:expr;)?
        $(real_complex |$rck:pat_param, $rco:tt, $rcx:tt| $real_complex:expr;)?
        $(slices $slices:expr;)?
        $(exact |$ek:pat_param, $ex:tt| $exact:expr;)?
        $(checks $checks:ident;)?
        $(valid |$vk:pat_param, $vv:tt, $vo:tt, $vx:tt, $vr:tt| $valid:expr;)?
    })+) => {$(
        impl Formula<$n> for $object {
            $(fn exact(&self, $ex: [&num_rational::BigRational; $n]) -> Result<Option<num_rational::BigRational>, crate::error::Error> { let $ek = self; $exact } const HAS_EXACT: bool = true;)?
            $(fn character_rule(&self) -> Option<crate::characters::Rule> { Some(crate::characters::Rule::$rule) })?
            $(fn result_dtype(&self, dtype: DType) -> DType { objects!(@output dtype, $output) } objects!(@computing $output);)?
            $(objects!(@test_output $test);)?
            $(fn range(&self, $args: [Range; $n]) -> Range { $range })?
            $(fn domain(&self, $domain_dtype: DType, $targs: [Range; $n]) -> Option<[Range; $n]> { $domain })?
            $(fn identity(&self) -> [Option<f64>; $n] { $identity })?
            $(fn fold(&self) -> Option<crate::reduce::Reduce> { Some(crate::reduce::Reduce::$fold) })?
            $(fn inverse_fold(&self) -> Option<crate::reduce::Reduce> { Some(crate::reduce::Reduce::$inverse_fold) })?
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
            $(#[inline(always)] fn emit_against<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, o: &O, c: F, v: O::V, left: bool) -> Option<O::B> { let $ak = self; against(*self, $atol, o, c, v, left) }
            fn bands(&self) -> Option<(crate::tolerant::Tolerance, crate::tolerant::Band, crate::tolerant::Band)> { let $ak = self; let (left, right) = <Self as crate::tolerant::Banding>::BANDS; Some(($atol, left, right)) })?
            $(fn emit_repair<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(&self, o: &O, x: [O::V; $n]) -> Option<O::B> { Some(($repair).predicate(o, x[0])) }
            fn repair_routine(&self) -> Option<Repair> { Some(($repair).routine()) })?
            objects!(@partials $n, $(slope |$so, $sx1, $sr| $slope)? $(slopes |$s2o, $s2x, $s2y, $s2r| [$dx, $dy])? $(shared $shared)?);
            $(objects!(@integers $n, $integer);)?
            $(objects!(@checks $checks);)?
            $(#[inline(always)] fn emit_valid<F: crate::Float, O: Ops<F> + ?Sized>(&self, $vv: Variant, $vo: &O, $vx: [O::V; $n], $vr: O::V) -> Option<(crate::error::Kind, O::B)> { let $vk = self; $valid })?
            $(#[inline(always)] fn integer<I: crate::recipe::Overflowing>(&self, $nio: &crate::recipe::IntegerOps<I>, $nix: [I; $n]) -> Option<I> { let $nik = self; $native_integer })?
            $(#[inline(always)] fn complex<F: crate::float::Native>(&self, $nfo: &crate::recipe::NativeOps<{ crate::FUSED }>, $nfx: [num_complex::Complex<F>; $n]) -> Option<num_complex::Complex<F>> { let $nfk = self; $native_complex })?
            $(#[inline(always)] fn real<F: crate::float::Native>(&self, $nro: &crate::recipe::NativeOps<{ crate::FUSED }>, $nrx: [F; $n]) -> Option<F> { let $nrk = self; $native_real })?
            $(const REAL_COMPLEX: bool = true; #[inline(always)] fn real_complex<F: crate::float::Native>(&self, $rco: &crate::recipe::NativeOps<{ crate::FUSED }>, $rcx: [F; $n]) -> Option<num_complex::Complex<F>> { let $rck = self; $real_complex })?
            fn native_slices(&self) -> Option<NativeSlices<Self, $n>> {
                None $(.or_else(|| Some($slices)))?
                    $(.or_else(|| Some(NativeSlices::NONE)).map(|s| s.with_complex(crate::complex::bulk::$complex_slices, crate::complex::bulk::$complex_slices)))?
                    $(.or_else(|| Some(NativeSlices::NONE)).map(|s| s.with_real_complex(crate::complex::bulk::$real_complex_slices, crate::complex::bulk::$real_complex_slices)))?
                    $(.or_else(|| Some(NativeSlices::NONE)).map(|s| s.with_complex_tests(crate::tolerant::$complex_tests, crate::tolerant::$complex_tests)))?
            }
        }
    )+};
}

objects!(1:
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
    crate::Round { native_integer |k, o, [x]| { let wide = crate::parts::rounded_wide(Into::<i64>::into(x).into(), k.scale); Some(o.result(crate::integer::overflowed(o.variant, i64::try_from(wide).ok(), || wide as i64))) }; native_complex |k, _, [x]| Some(num_complex::Complex::new(crate::parts::rounded(x.re, k.scale), crate::parts::rounded(x.im, k.scale))); native_real |k, _, [x]| Some(crate::parts::rounded(x, k.scale)); }
    crate::Sqrt { range [x] => exact(x.lo.max(0.0).sqrt(), x.hi.sqrt()); domain _, [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(o.sqrt(x)); slope |o, _, r| o.reciprocal(o.add(r, r)); native_complex |_, _, [x]| Some(crate::complex::sqrt::<F, { crate::FUSED }>(x)); complex_slices sqrt; real_complex |_, o, [x]| { let r = o.sqrt(o.magnitude(x)); Some(if x < F::zero() { num_complex::Complex::new(F::zero(), r) } else { num_complex::Complex::new(r, F::zero()) }) }; checks nan; }
    crate::Cbrt { range [x] => increasing(x, f64::cbrt); formula |_, o, [x]| Some(crate::pow::cbrt_formula(o, x)); slope |o, _, r| o.reciprocal(o.multiply(o.number(F::narrow(3.0)), o.multiply(r, r))); }
    crate::Exp { range [x] => increasing(x, f64::exp).meet(Range::NONNEGATIVE); formula |_, o, [x]| Some(crate::exp::formula(o, x)); slope |_, _, r| r; native_complex |_, _, [x]| Some(crate::complex::exp::<F, { crate::FUSED }>(x)); complex_slices exp; }
    crate::Expm1 { range [x] => increasing(x, f64::exp_m1); formula |_, o, [x]| Some(crate::exp::minus_one_formula(o, x, o.number(F::zero()))); slope |o, _, r| o.add(r, o.number(F::one())); native_complex |_, _, [x]| Some(crate::complex::exp::<F, { crate::FUSED }>(x) - Complex::new(F::one(), F::zero())); }
    crate::Ln { range [x] => increasing(x, |v| if v < 0.0 { f64::NEG_INFINITY } else { v.ln() }); domain _, [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(crate::ln::formula(o, x)); slope |o, x, _| o.reciprocal(x); native_complex |_, _, [x]| Some(crate::complex::ln::<F, { crate::FUSED }>(x)); complex_slices ln; real_complex |_, o, [x]| Some(num_complex::Complex::new(crate::ln::formula(o, o.magnitude(x)), if x < F::zero() { F::narrow(PI) } else { F::zero() })); checks nan; }
    crate::Log2 { range [x] => increasing(x, f64::log2); domain _, [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(crate::ln::log_with(o, x, crate::ln::base2(o))); slope |o, x, _| o.divide(o.number(F::narrow(std::f64::consts::LOG2_E)), x); slices NativeSlices::unary(crate::ln::log2_slice, crate::ln::log2_slice); checks nan; }
    crate::Log10 { range [x] => increasing(x, f64::log10); domain _, [x] => inside([x], [Range::NONNEGATIVE]); formula |_, o, [x]| Some(crate::ln::log_with(o, x, crate::ln::base10(o))); slope |o, x, _| o.divide(o.number(F::narrow(std::f64::consts::LOG10_E)), x); slices NativeSlices::unary(crate::ln::log10_slice, crate::ln::log10_slice); checks nan; }
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
    crate::Log1p { range [x] => increasing(x, |v| if v < -1.0 { f64::NEG_INFINITY } else { v.ln_1p() }); domain _, [x] => inside([x], [exact(-1.0, f64::INFINITY)]); formula |_, o, [x]| Some(crate::ln::log1p_formula(o, x)); slope |o, x, _| o.reciprocal(o.add(o.number(F::one()), x)); native_complex |_, _, [x]| Some(crate::complex::ln1p::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Sinh { range [x] => increasing(x, f64::sinh); formula |_, o, [x]| Some(crate::hyperbolic::sinh_formula(o, x)); slope |o, x, _| o.cosh(x); native_complex |_, _, [x]| Some(crate::complex::sinh::<F, { crate::FUSED }>(x)); complex_slices sinh; }
    crate::Cosh { range [x] => even(x, f64::cosh).meet(Range::NONNEGATIVE); formula |_, o, [x]| Some(crate::hyperbolic::cosh_formula(o, x)); slope |o, x, _| o.sinh(x); native_complex |_, _, [x]| Some(crate::complex::cosh::<F, { crate::FUSED }>(x)); complex_slices cosh; }
    crate::Tanh { range [x] => increasing(x, f64::tanh); formula |_, o, [x]| Some(crate::hyperbolic::tanh_formula(o, x)); slope |o, _, r| o.subtract(o.number(F::one()), o.multiply(r, r)); native_complex |_, _, [x]| Some(crate::complex::tanh::<F, { crate::FUSED }>(x)); complex_slices tanh; }
    crate::Asinh { range [x] => increasing(x, f64::asinh); slope |o, x, _| o.reciprocal(o.sqrt1px2(x)); native_complex |_, _, [x]| Some(crate::complex::asinh::<F, { crate::FUSED }>(x)); complex_slices asinh; native_real |_, _, [x]| Some(crate::hyperbolic::asinh::<F, { crate::FUSED }>(x)); }
    // sinh(acosh x) is √(x² - 1).
    crate::Acosh { range [x] => increasing(x, |v| v.max(1.0).acosh()).meet(Range::NONNEGATIVE); domain _, [x] => inside([x], [exact(1.0, f64::INFINITY)]); slope |o, _, r| o.reciprocal(o.sinh(r)); native_complex |_, _, [x]| Some(crate::complex::acosh::<F, { crate::FUSED }>(x)); complex_slices acosh; native_real |_, _, [x]| Some(crate::hyperbolic::acosh::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Atanh { range [x] => increasing(x, |v| v.clamp(UNIT.0, UNIT.1).atanh()); domain _, [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, _| o.reciprocal(o.subtract(o.number(F::one()), o.multiply(x, x))); native_complex |_, _, [x]| Some(crate::complex::atanh::<F, { crate::FUSED }>(x)); complex_slices atanh; native_real |_, _, [x]| Some(crate::hyperbolic::atanh::<F, { crate::FUSED }>(x)); checks nan; }
    crate::RealAtanh { output real; slope |o, x, _| o.reciprocal(o.subtract(o.number(F::one()), o.multiply(x, x))); native_complex |_, _, [x]| Some(crate::complex::atanh::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::hyperbolic::atanh::<F, { crate::FUSED }>(if x.abs() > F::one() { F::one()/x } else { x })); }
    crate::Sin { range _ => exact(UNIT.0, UNIT.1).loose(); formula |_, o, [x]| Some(crate::trig::formula(o, x, crate::trig::Kind::Sin)); slope |o, x, _| o.cos(x); repair crate::trig::Kind::Sin; native_complex |_, _, [x]| Some(crate::complex::sin::<F, { crate::FUSED }>(x)); complex_slices sin; }
    crate::Cos { range _ => exact(UNIT.0, UNIT.1).loose(); formula |_, o, [x]| Some(crate::trig::formula(o, x, crate::trig::Kind::Cos)); slope |o, x, _| o.negate(o.sin(x)); repair crate::trig::Kind::Cos; native_complex |_, _, [x]| Some(crate::complex::cos::<F, { crate::FUSED }>(x)); complex_slices cos; }
    crate::Tan { formula |_, o, [x]| Some(crate::trig::formula(o, x, crate::trig::Kind::Tan)); slope |o, _, r| o.add(o.number(F::one()), o.multiply(r, r)); repair crate::trig::Kind::Tan; native_complex |_, _, [x]| Some(crate::complex::tan::<F, { crate::FUSED }>(x)); complex_slices tan; }
    crate::Asin { range [x] => increasing(x, |v| v.clamp(UNIT.0, UNIT.1).asin()); domain _, [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, _| o.reciprocal(o.sqrt1mx2(x)); native_complex |_, _, [x]| Some(crate::complex::asin::<F, { crate::FUSED }>(x)); complex_slices asin; native_real |_, _, [x]| Some(crate::arc::asin::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Acos { range [x] => decreasing(x, |v| v.clamp(UNIT.0, UNIT.1).acos()).meet(Range::NONNEGATIVE); domain _, [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, _| o.negate(o.reciprocal(o.sqrt1mx2(x))); native_complex |_, _, [x]| Some(crate::complex::acos::<F, { crate::FUSED }>(x)); complex_slices acos; native_real |_, _, [x]| Some(crate::arc::acos::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Atan { range [x] => increasing(x, f64::atan); slope |o, x, _| o.reciprocal(o.add(o.number(F::one()), o.multiply(x, x))); native_complex |_, _, [x]| Some(crate::complex::atan::<F, { crate::FUSED }>(x)); complex_slices atan; native_real |_, _, [x]| Some(crate::arc::atan::<F, { crate::FUSED }>(x)); }
    crate::Sqrt1mx2 { range [x] => decreasing(x.magnitude(), |v| (1.0 - v.min(1.0) * v.min(1.0)).sqrt()).meet(Range::NONNEGATIVE); domain _, [x] => inside([x], [exact(UNIT.0, UNIT.1)]); slope |o, x, r| o.negate(o.divide(x, r)); native_complex |_, _, [x]| Some(crate::complex::sqrt1mx2::<F, { crate::FUSED }>(x)); complex_slices sqrt1mx2; native_real |_, _, [x]| Some(crate::sqrts::sqrt1mx2::<F, { crate::FUSED }>(x)); }
    crate::Sqrt1px2 { range [x] => even(x, |v| (1.0 + v * v).sqrt()).meet(Range::NONNEGATIVE); slope |o, x, r| o.divide(x, r); native_complex |_, _, [x]| Some(crate::complex::sqrt1px2::<F, { crate::FUSED }>(x)); complex_slices sqrt1px2; native_real |_, _, [x]| Some(crate::sqrts::sqrt1px2::<F, { crate::FUSED }>(x)); }
    crate::Sqrtx2m1 {
        range [x] => increasing(x, |v| v.signum() * (v * v - 1.0).max(0.0).sqrt());
        domain _, [x] => inside([x], [if x.lo >= 1.0 { exact(1.0, f64::INFINITY) } else { exact(f64::NEG_INFINITY, -1.0) }]);
        slope |o, x, r| o.divide(x, r);
        native_complex |_, _, [x]| Some(crate::complex::sqrtx2m1::<F, { crate::FUSED }>(x));
        complex_slices sqrtx2m1;
        native_real |_, _, [x]| Some(crate::sqrts::sqrtx2m1::<F, { crate::FUSED }>(x)); checks nan;
    }
    crate::Reciprocal { range [x] => quotient(Range::of(Some(1.0)), x).loose(); formula |_, o, [x]| Some(if o.integral() { crate::recipe::integer_floor_divide(o, o.number(F::one()), x) } else { crate::recipe::reciprocal(o, x) }); slope |o, _, r| o.negate(o.multiply(r, r)); integers arithmetic; native_complex |_, _, [x]| Some(crate::complex::divide::<F>(num_complex::Complex::new(F::one(), F::zero()), x)); exact |_, [x]| crate::exact::divide(&num_rational::BigRational::one(), x, false); }
    crate::PiTimes { range [x] => pi_times(x); formula |_, o, [x]| Some(crate::recipe::pi_times(o, x)); slope |o, _, _| o.number(F::narrow(std::f64::consts::PI)); native_complex |_, _, [x]| Some(x * F::narrow(std::f64::consts::PI)); }
    // Γ(x + 1)'s slope is Γ(x + 1) ψ(x + 1).
    crate::Factorial {
        range args => signs(args); output integer;
        slope |o, x, r| o.multiply(r, o.digamma(o.add(x, o.number(F::one()))));
        native_integer |_, o, [x]| {
            let n: i64 = x.into();
            // Γ has a pole at each negative integer, which the greatest value stands for, as for a quotient by 0: i64's under dynamic kinds, which a narrower width reports as an overflow (`IntegerOps::result`) so the call widens, and otherwise the width's.
            if n < 0 { return Some(if o.variant.fails_on_overflow() { o.result(i64::MAX) } else { I::limit(1) }); }
            // 66! contains 64 factors of two, so larger factorials have zero low 64 bits.
            let (p, valid) = (2..=n.min(66)).fold((1, true), |(p, valid), k| {
                let (q, fits) = <Product as Reduction<i64>>::chain(p, k);
                (q, valid && fits)
            });
            if o.variant.fails_on_overflow() && !valid { (crate::error::Kind::Overflow.reporter())(); }
            Some(o.result(p))
        };
        native_complex |_, _, [x]| Some(crate::complex::factorial::<F, { crate::FUSED }>(x));
        native_real |_, _, [x]| Some(crate::special::factorial::<F, { crate::FUSED }>(x));
        exact |_, [x]| crate::exact::factorial(x);
    }
    crate::Gamma {
        range args => signs(args); slope |o, x, r| o.multiply(r, o.digamma(x));
        native_complex |_, _, [x]| Some(crate::complex::gamma::<F, { crate::FUSED }>(x));
        native_real |_, _, [x]| Some(crate::special::gamma::<F, { crate::FUSED }>(x)); checks nan;
    }
    crate::Sign { formula |_, o, [x]| Some(crate::recipe::sign(o, x)); slope |o, _, _| o.number(F::zero()); integers arithmetic; native_complex |_, _, [x]| Some(crate::complex::sign::<F, { crate::FUSED }>(x)); exact |_, [x]| Ok(Some(x.signum())); }
    crate::tolerant::Floor { formula |k, o, [x]| Some(crate::with_tolerance!(k.0, |t| t.floor_in(o, x))); slope |o, _, _| o.number(F::zero()); integers formula; native_complex |k, o, [x]| Some(crate::tolerant::TolerantComplex::complex_floor(k.0.overridden(o.tolerance), x)); exact |_, [x]| Ok(Some(x.floor())); }
    crate::tolerant::Ceil { formula |k, o, [x]| Some(crate::with_tolerance!(k.0, |t| t.ceil_in(o, x))); slope |o, _, _| o.number(F::zero()); integers formula; native_complex |k, o, [x]| Some(crate::tolerant::TolerantComplex::complex_ceil(k.0.overridden(o.tolerance), x)); exact |_, [x]| Ok(Some(x.ceil())); }
    crate::Not { table 1; test |_, o, [x]| Some(o.equal(x, o.number(F::zero()))); integers formula; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::Cis { output complex; native_complex |_, _, [x]| Some(crate::complex::exp::<F, { crate::FUSED }>(num_complex::Complex::new(-x.im, x.re))); complex_slices cis; real_complex_slices real_cis; real_complex |_, o, [x]| (!crate::trig::Kind::Cos.predicate(o, x)).then(|| { let (c, s) = crate::complex::quarter(x).unwrap_or_else(|| (crate::trig::formula(o, x, crate::trig::Kind::Cos), crate::trig::formula(o, x, crate::trig::Kind::Sin))); num_complex::Complex::new(c, s) }); }
    crate::LambertW { domain dtype, [x] => crate::lambert::real_domain(dtype, x); native_complex |_, _, [x]| Some(crate::lambert::complex(x)); native_real |_, _, [x]| Some(crate::lambert::lambertw::<F, { crate::FUSED }>(x)); checks nan; }
    crate::Lgamma { slope |o, x, _| o.digamma(x); native_complex |_, _, [x]| Some(crate::complex::lgamma::<F, { crate::FUSED }>(x)); native_real |_, _, [x]| Some(crate::special::lgamma::<F, { crate::FUSED }>(x)); checks nan; }
    // ψ's slope is ψ⁽¹⁾.
    crate::Digamma { slope |o, x, _| o.polygamma(o.number(F::one()), x); native_real |_, _, [x]| Some(crate::special::digamma::<F, { crate::FUSED }>(x)); }
    // 2/√π e^-x², and its negative for erfc.
    crate::Erf { slope |o, x, _| o.multiply(o.number(F::narrow(std::f64::consts::FRAC_2_SQRT_PI)), o.exp(o.negate(o.multiply(x, x)))); native_real |_, _, [x]| Some(crate::special::erf::<F, { crate::FUSED }>(x)); }
    crate::Erfc { slope |o, x, _| o.multiply(o.number(F::narrow(-std::f64::consts::FRAC_2_SQRT_PI)), o.exp(o.negate(o.multiply(x, x)))); native_real |_, _, [x]| Some(crate::special::erfc::<F, { crate::FUSED }>(x)); slices NativeSlices::unary(crate::special::erfc_slice, crate::special::erfc_slice); }
    // √π/2 e^(erfinv x)², from the result.
    crate::Erfinv { slope |o, _, r| o.multiply(o.number(F::narrow(std::f64::consts::FRAC_2_SQRT_PI.recip())), o.exp(o.multiply(r, r))); native_real |_, _, [x]| Some(crate::special::erfinv::<F, { crate::FUSED }>(x)); slices NativeSlices::unary(crate::special::erfinv_slice, crate::special::erfinv_slice); checks nan; }
    crate::Erfcinv { slope |o, _, r| o.multiply(o.number(F::narrow(-std::f64::consts::FRAC_2_SQRT_PI.recip())), o.exp(o.multiply(r, r))); native_real |_, _, [x]| Some(crate::special::erfcinv::<F, { crate::FUSED }>(x)); slices NativeSlices::unary(crate::special::erfcinv_slice, crate::special::erfcinv_slice); checks nan; }
);
objects!(2:
    crate::Add { range [x, y] => exact(x.lo + y.lo, x.hi + y.hi); identity [Some(0.0), Some(0.0)]; fold Sum; truth Sum; characters Shift; formula |_, o, [x, y]| Some(o.add(x, y)); slopes |o, _, _, _| [o.number(F::one()), o.number(F::one())]; integers arithmetic; native_complex |_, _, [x, y]| Some(x+y); exact |_, [x, y]| Ok(Some(x + y)); valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Add, x, y); }
    crate::Subtract { range [x, y] => exact(x.lo - y.hi, x.hi - y.lo); identity [None, Some(0.0)]; inverse_fold Sum; characters Difference; formula |_, o, [x, y]| Some(o.subtract(x, y)); slopes |o, _, _, _| [o.number(F::one()), o.number(-F::one())]; integers arithmetic; native_complex |_, _, [x, y]| Some(x-y); exact |_, [x, y]| Ok(Some(x - y)); valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Subtract, x, y); }
    crate::Multiply { range [x, y] => corners(x, y, |a, b| a * b); identity [Some(1.0), Some(1.0)]; fold Product; table 8; same |x| x.square(); formula |_, o, [x, y]| Some(o.multiply(x, y)); slopes |_, x, y, _| [y, x]; integers arithmetic; native_complex |_, _, [x, y]| Some(x*y); exact |_, [x, y]| Ok(Some(x * y)); valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Multiply, x, y); }
    crate::Divide { range [x, y] => quotient(x, y); identity [None, Some(1.0)]; inverse_fold Product; formula |_, o, [x, y]| Some(o.divide(x, y)); slopes |o, _, y, r| [o.reciprocal(y), o.negate(o.divide(r, y))]; native_complex |_, _, [x, y]| Some(crate::complex::divide::<F>(x, y)); exact |_, [x, y]| crate::exact::divide(x, y, false); }
    // A nonzero divisor that every item shares divides plainly, since 0 ÷ 0 can't arise.
    crate::MathishDivide {
        range [x, y] => quotient(x, y); identity [None, Some(1.0)]; inverse_fold Product;
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
    // A whole exponent that every item shares gives a real power of any base (`domain`).
    crate::Pow {
        range [x, y] => powered(x, y); domain _, [x, y] => if whole(y) { Some([x, y]) } else { inside([x, y], [Range::NONNEGATIVE, Range::ANY]) };
        identity [None, Some(1.0)]; formula |_, o, [x, y]| Some(crate::pow::formula(o, x, y)); shared power;
        native_integer |_, o, [x, y]| Some(I::low(crate::integer::native_power(o.variant, x.into(), y.into())));
        native_complex |_, _, [x, y]| Some(crate::complex::pow::<F, { crate::FUSED }>(x, y)); complex_slices pow; slices NativeSlices::power(); exact |_, [x, y]| crate::exact::power(x, y); checks nan;
        valid |_, v, o, x, r| power_valid(v, o, x, r);
    }
    crate::RealPow { range [x, _] => signs([x]); identity [None, Some(1.0)]; output real; formula |_, o, [x, y]| Some(crate::pow::real_formula(o, x, y)); shared real_power; native_complex |_, _, [x, y]| Some(crate::complex::pow::<F, { crate::FUSED }>(x, y)); }
    // The yth root of x, x^(1/y).
    crate::Root { range [_, y] => signs([y]); domain _, [n, y] => root_domain(n, y); formula |_, o, [n, y]| crate::pow::root_formula(o, n, y); slopes |o, x, y, r| [o.negate(o.divide(o.multiply(r, o.ln(y)), o.multiply(x, x))), o.divide(r, o.multiply(x, y))]; native_complex |_, _, [n, y]| Some(crate::complex::nth_root::<F, { crate::FUSED }>(n, y)); native_real |_, _, [n, y]| Some(crate::pow::root::<F, { crate::FUSED }>(n, y)); slices NativeSlices::root(); exact |_, [n, y]| { if num_traits::Zero::is_zero(n) { return Err(crate::error::Error::new(crate::error::Kind::Domain, "root degree is zero")); } crate::exact::power(y, &n.recip()) }; checks nan; }
    crate::Logb { domain _, [x, y] => inside([x, y], [Range::NONNEGATIVE; 2]); shared logb; native_complex |_, _, [b, x]| Some(crate::complex::divide(crate::complex::ln::<F, { crate::FUSED }>(x), crate::complex::ln::<F, { crate::FUSED }>(b))); native_real |_, _, [b, x]| Some(crate::ln::logb::<F, { crate::FUSED }>(b, x)); slices NativeSlices::logarithm(); valid |_, _, o, x, r| logarithm_valid(o, x, r); }
    crate::MathishLogb { domain _, [x, y] => inside([x, y], [Range::NONNEGATIVE; 2]); shared logb; native_complex |_, _, [b, x]| { let one = num_complex::Complex::new(F::one(), F::zero()); Some(if b == one && x == one { num_complex::Complex::new(F::zero(), F::zero()) } else { crate::complex::divide(crate::complex::ln::<F, { crate::FUSED }>(x), crate::complex::ln::<F, { crate::FUSED }>(b)) }) }; native_real |_, _, [b, x]| Some(crate::ln::logb_mode::<F, true>(b, x)); slices NativeSlices::mathish_logarithm(); valid |_, _, o, x, r| logarithm_valid(o, x, r); }
    // `y` modulo `x`. Its slope in `x` is minus the quotient's floor, (y - r) ÷ x from the result, and 0 where `x` is 0, where the residue is `y` itself. Integers take a positive power of two that every item shares as a mask (`masked`). Integer kernels run its slice forms (`integer.rs`). Where a whole block shares one divisor, those take a mask or a reciprocal in place of a division.
    crate::tolerant::Residue {
        identity [Some(0.0), None];
        formula |k, o, [x, y]| if o.integral() { masked(o, x, y) } else { (!k.0.is_exact()).then(|| k.0.residue_in(o, x, y)) };
        slopes |o, x, y, r| [{
            let at_zero = o.equal(x, o.number(F::zero()));
            o.negate(o.multiply(o.one_where(o.not(at_zero)), o.divide(o.subtract(y, r), o.add(x, o.one_where(at_zero)))))
        }, o.number(F::one())];
        native_integer |_, _, [x, y]| Some(I::low(crate::integer::residue(x.into(), y.into())));
        native_complex |k, o, [x, y]| Some(crate::tolerant::TolerantComplex::complex_residue(k.0.overridden(o.tolerance), x, y));
        native_real |k, o, [x, y]| {
            let t = k.0.overridden(o.tolerance);
            Some(if t.is_exact() { crate::tolerant::exact_residue(x, y) } else { crate::tolerant::Tolerant::residue(t, x, y) })
        };
        slices NativeSlices::residue();
        exact |_, [x, y]| Ok(Some(crate::exact::residue(x, y)));
    }
    // C(n, k), with n first.
    crate::Binom {
        identity [None, Some(1.0)];
        slopes |o, n, k, _| [o.binom_dn(n, k), o.binom_dk(n, k)];
        native_integer |_, o, [n, k]| {
            let (n, k) = (n.into(), k.into());
            let value = crate::integer::overflowed(o.variant, crate::integer::binom(n, k), || crate::integer::binom_wrapped(n, k).unwrap_or(0));
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
        identity [None, Some(1.0)];
        formula |k, o, [x, y]| Some(if o.integral() { crate::recipe::integer_floor_divide(o, x, y) }
            else { crate::with_tolerance!(k.0, |t| t.floor_in(o, o.divide(x, y))) });
        slopes |o, _, _, _| [o.number(F::zero()), o.number(F::zero())];
        integers arithmetic;
        exact |_, [x, y]| crate::exact::floor_divide(x, y, false);
        valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Divide, x, y);
    }
    crate::tolerant::MathishFloorDivide {
        identity [None, Some(1.0)];
        formula |k, o, [x, y]| Some(if o.integral() { crate::recipe::integer_floor_divide(o, x, y) }
            else { crate::with_tolerance!(k.0, |t| t.floor_in(o, crate::recipe::mathish_divide(o, x, y))) });
        slopes |o, _, _, _| [o.number(F::zero()), o.number(F::zero())];
        integers arithmetic;
        exact |_, [x, y]| crate::exact::floor_divide(x, y, true);
        valid |_, v, o, [x, y], _| arithmetic_valid(v, o, crate::recipe::Binary::Divide, x, y);
    }
    crate::tolerant::Gcd {
        identity [Some(0.0), Some(0.0)]; table 14;
        native_integer |_, o, [x, y]| { let (x, y) = (x.into(), y.into()); Some(o.result(crate::integer::overflowed(o.variant, crate::integer::gcd(x, y), || crate::integer::gcd_wrapped(x, y)))) };
        native_complex |k, o, [x, y]| Some(k.0.overridden(o.tolerance).complex_gcd(x, y));
        native_real |k, o, [x, y]| Some(k.0.overridden(o.tolerance).gcd(x, y));
        exact |_, [x, y]| Ok(Some(crate::exact::gcd(x, y))); checks nan;
    }
    crate::tolerant::Lcm {
        identity [Some(1.0), Some(1.0)]; table 8;
        native_integer |_, o, [x, y]| { let (x, y) = (x.into(), y.into()); Some(o.result(crate::integer::overflowed(o.variant, crate::integer::lcm(x, y), || crate::integer::lcm_wrapped(x, y)))) };
        native_complex |k, o, [x, y]| Some(lcm_from_gcd(x, y, k.0.overridden(o.tolerance).complex_gcd(x, y)));
        native_real |k, o, [x, y]| Some(lcm_from_gcd(x, y, k.0.overridden(o.tolerance).gcd(x, y)));
        exact |_, [x, y]| Ok(Some(crate::exact::lcm(x, y))); checks nan;
    }
    crate::Nand { table 7; test |_, o, [x, y]| Some(o.not(o.and(boolean_one(o, x), boolean_one(o, y)))); integers formula; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::Nor { table 1; test |_, o, [x, y]| Some(o.and(o.equal(x, o.number(F::zero())), o.equal(y, o.number(F::zero())))); integers formula; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::BitAnd { fold And; table 8; formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::And, o.to_bits(x), o.to_bits(y)))); integers only; }
    crate::BitOr { fold Or; table 14; formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::Or, o.to_bits(x), o.to_bits(y)))); integers only; }
    crate::LeftShift { formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::ShiftLeft, o.to_bits(x), o.to_integer(y)))); integers bits; }
    crate::RightShift {
        formula |_, o, [x, y]| Some(o.from_bits(o.integer_op(crate::recipe::Integer::SignedShiftRight, o.to_bits(x), o.to_integer(y))));
        integers bits;
    }
    crate::reduce::Xor { fold Xor; table 6; test |_, o, [x, y]| Some(o.not(o.equal(x, y))); integers only; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::reduce::Xnor { fold Xnor; table 9; test |_, o, [x, y]| Some(o.equal(x, y)); integers only; valid |_, v, o, x, _| truth_valid(v, o, x); }
    crate::tolerant::Equal {
        identity [Some(1.0), Some(1.0)]; table 9; characters Ordered;
        test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); against |k| k.0; integers formula;
        native_complex |k, o, [x, y]| Some(if k.0.overridden(o.tolerance).complex_equal(x, y) { Complex::one() } else { Complex::zero() });
        complex_tests complex_equal_slice;
        exact |_, [x, y]| Ok(Some(crate::exact::boolean(x == y)));
    }
    crate::tolerant::NotEqual {
        identity [Some(0.0), Some(0.0)]; table 6; characters Ordered;
        test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); against |k| k.0; integers formula;
        native_complex |k, o, [x, y]| Some(if !k.0.overridden(o.tolerance).complex_equal(x, y) { Complex::one() } else { Complex::zero() });
        complex_tests complex_not_equal_slice;
        exact |_, [x, y]| Ok(Some(crate::exact::boolean(x != y)));
    }
    crate::tolerant::Less { identity [Some(0.0), None]; table 2; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); against |k| k.0; integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x < y))); }
    crate::tolerant::LessEqual { identity [Some(1.0), None]; table 11; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); against |k| k.0; integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x <= y))); }
    crate::tolerant::Greater { identity [None, Some(0.0)]; table 4; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); against |k| k.0; integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x > y))); }
    crate::tolerant::GreaterEqual { identity [None, Some(1.0)]; table 13; characters Ordered; test |k, o, [x, y]| Some(tested(*k, k.0, o, x, y)); against |k| k.0; integers formula; exact |_, [x, y]| Ok(Some(crate::exact::boolean(x >= y))); }
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

impl BetaincAt {
    fn parameters(&self, width: FloatWidth) -> [f64; 2] { with_float_width!(width, T => [self.0, self.1].map(|x| <T as Float>::Working::narrow(x).wide())) }
}

impl Formula<1> for BetaincAt {
    fn supports(&self, dtype: DType) -> bool { Betainc.supports(dtype) }
    fn range(&self, [x]: [Range; 1]) -> Range { Betainc.range([Range::of(Some(self.0)), Range::of(Some(self.1)), x]) }
    fn real<F: Native>(&self, o: &NativeOps<{ FUSED }>, [x]: [F; 1]) -> Option<F> {
        let [a, b] = self.parameters(Ops::<F>::width(o));
        Betainc.real(o, [F::narrow(a), F::narrow(b), x])
    }
    fn emit_partials<F: Float, O: Ops<F> + ?Sized>(&self, o: &O, [x]: [O::V; 1], _: O::V, [need]: [bool; 1]) -> [Option<O::V>; 1] {
        [need.then(|| {
            let ([a, b], one) = (self.parameters(o.width()), o.number(F::one()));
            let n = |v: f64| o.number(F::narrow(v));
            let mut e = o.negate(o.subtract(o.add(o.lgamma(n(a)), o.lgamma(n(b))), o.lgamma(n(a + b))));
            if a != 1.0 { e = o.add(e, o.multiply(n(a - 1.0), o.ln(x))); }
            if b != 1.0 { e = o.add(e, o.multiply(n(b - 1.0), o.ln(o.subtract(one, x)))); }
            o.exp(e)
        })]
    }
}

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

/// The tolerance at which `o` compares at tolerance `t`: the one that `o` sets, otherwise `t` (`Ops::tolerance`). `None` where it isn't known when the recipe is built.
pub fn run_tolerance<F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, t: crate::tolerant::Tolerance) -> Option<crate::tolerant::Tolerance> {
    crate::tolerant::Tolerance::new(o.constant(o.tolerance(F::narrow(t.value())))?.wide())
}
/// Comparison `k` of `v` with the one number `c`, as `k`'s kernel compares items with a number: the band that `v` must lie in against the range of floats equal to `c` at tolerance `t`, or at the tolerance that `o` sets (`tolerant::against_steps`). `left` says whether `v` is the left argument. `None` where the tolerance isn't known when the recipe is built, or `c` is NaN.
fn against<K: crate::tolerant::Banding, F: crate::Float, O: crate::recipe::Ops<F> + ?Sized>(
    k: K, t: crate::tolerant::Tolerance, o: &O, c: F, v: O::V, left: bool) -> Option<O::B> {
    crate::tolerant::against_steps(k, run_tolerance(o, t)?, o, c, v, left)
}

/// Comparison `k`'s truth value through `o`, as its kernel computes it: against a number that exactly one argument holds (`Ops::constant`) through `against`, and otherwise step by step (`tolerant::Stepped`). Native ops hold every item as a number, so each item compares step by step rather than building the range of floats equal to the other.
#[inline]
fn tested<K, F, O>(k: K, t: crate::tolerant::Tolerance, o: &O, x: O::V, y: O::V) -> O::B where K: crate::tolerant::Stepped + crate::tolerant::Banding + Copy,
    F: crate::Float, O: crate::recipe::Ops<F> + ?Sized {
    let side = match (o.constant(x), o.constant(y)) {
        (Some(_), Some(_)) => None, (None, Some(c)) => Some((c, x, true)), (Some(c), None) => Some((c, y, false)), _ => None
    };
    side.and_then(|(c, v, left)| against(k, t, o, c, v, left)).unwrap_or_else(|| k.steps::<F, O>(o, x, y))
}

/// A bitwise function's facts, which its table decides: the tables that fold (and, or, xor, xnor) for numbers, and the same tables' reductions for Booleans.
impl Formula<1> for crate::Bitwise {
    const HAS_EXACT: bool = true;
    fn exact(&self, [y]: [&num_rational::BigRational; 1]) -> Result<Option<num_rational::BigRational>, crate::error::Error> {
        <Self as Formula<2>>::exact(self, [&num_rational::BigRational::zero(), y])
    }
    const INTEGER_FORMULA: bool = true;
    fn supports(&self, dtype: DType) -> bool { <Self as Formula<2>>::supports(self, dtype) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [y]: [O::V; 1]) -> Option<O::V> { <Self as Formula<2>>::emit(self, o, [o.number(F::zero()), y]) }
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, [y]: [I; 1]) -> Option<I> { <Self as Formula<1>>::emit(self, o, [y]) }
}
impl Formula<2> for crate::Bitwise {
    fn fold(&self) -> Option<crate::reduce::Reduce> {
        use crate::reduce::Reduce::*;
        match self.0 { 8 => Some(And), 14 => Some(Or), 6 => Some(Xor), 9 => Some(Xnor), _ => None }
    }
    const HAS_EXACT: bool = true;
    fn exact(&self, [x, y]: [&num_rational::BigRational; 2]) -> Result<Option<num_rational::BigRational>, crate::error::Error> {
        if !x.is_integer() || !y.is_integer() { return Err(crate::error::Error::new(crate::error::Kind::Domain, "bitwise needs whole numbers")); }
        Ok(Some(num_rational::BigRational::from_integer(crate::bitwise(self.0, x.to_integer(), y.to_integer()))))
    }
    fn truth_table(&self) -> Option<u8> { Some(self.0 as u8) }
    const INTEGER_FORMULA: bool = true;
    fn supports(&self, dtype: DType) -> bool { dtype.is_integral() && (dtype != DType::Integer(crate::buffer::Width::U8) || self.0 & 1 == 0) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::V> {
        use crate::recipe::Integer;
        o.integral().then(|| {
            o.from_bits(crate::arith::bitwise_with(
                self.0, o.to_bits(x), o.to_bits(y), o.integer(-1), |a, b| o.integer_op(Integer::And, a, b), |a, b| o.integer_op(Integer::Xor, a, b)))
        })
    }
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, x: [I; 2]) -> Option<I> { <Self as Formula<2>>::emit(self, o, x) }
}

impl<K: Formula<2>> Formula<2> for Flipped<K> {
    fn truth_table(&self) -> Option<u8> { self.0.truth_table().map(|t| (t & 9) | (t & 2) << 1 | (t & 4) >> 1) }
    fn character_policy(&self, [x, y]: [DType; 2]) -> Result<Option<crate::characters::Policy>, crate::error::Error> {
        use crate::characters::{ByKinds, Policy};
        self.0.character_policy([y, x]).map(|p| {
            p.map(|p| match p {
                Policy::Kinds(ByKinds::Left) => Policy::Kinds(ByKinds::Right), Policy::Kinds(ByKinds::Right) => Policy::Kinds(ByKinds::Left), p => p
            })
        })
    }
    fn exact(&self, [x, y]: [&num_rational::BigRational; 2]) -> Result<Option<num_rational::BigRational>, crate::error::Error> { self.0.exact([y, x]) }
    const HAS_EXACT: bool = K::HAS_EXACT;
    fn character_rule(&self) -> Option<crate::characters::Rule> { self.0.character_rule() }
    fn result_dtype(&self, input: DType) -> DType { self.0.result_dtype(input) }
    fn range(&self, [x, y]: [Range; 2]) -> Range { self.0.range([y, x]) }
    fn same(&self, x: Range) -> Option<Range> { self.0.same(x) }
    fn domain(&self, dtype: DType, [x, y]: [Range; 2]) -> Option<[Range; 2]> { self.0.domain(dtype, [y, x]).map(|[y, x]| [x, y]) }
    fn identity(&self) -> [Option<f64>; 2] { let [x, y] = self.0.identity(); [y, x] }
    const INTEGER_FORMULA: bool = K::INTEGER_FORMULA;
    fn supports(&self, dtype: DType) -> bool { self.0.supports(dtype) }
    const BOOLEAN: bool = K::BOOLEAN;
    const FAILS_NAN: bool = K::FAILS_NAN;
    #[inline]
    fn integer<I: crate::recipe::Overflowing>(&self, o: &crate::recipe::IntegerOps<I>, [x, y]: [I; 2]) -> Option<I> { self.0.integer(o, [y, x]) }
    #[inline]
    fn real<F: crate::float::Native>(&self, o: &crate::recipe::NativeOps<{ crate::FUSED }>, [x, y]: [F; 2]) -> Option<F> { self.0.real(o, [y, x]) }
    #[inline]
    fn complex<F: crate::float::Native>(
        &self, o: &crate::recipe::NativeOps<{ crate::FUSED }>, [x, y]: [num_complex::Complex<F>; 2]) -> Option<num_complex::Complex<F>> {
        self.0.complex(o, [y, x])
    }
    fn emit_valid<F: crate::Float, O: Ops<F> + ?Sized>(&self, v: Variant, o: &O, [x, y]: [O::V; 2], r: O::V) -> Option<(crate::error::Kind, O::B)> {
        self.0.emit_valid(v, o, [y, x], r)
    }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::V> { self.0.emit(o, [y, x]) }
    fn emit_test<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2]) -> Option<O::B> { self.0.emit_test(o, [y, x]) }
    fn emit_partials<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y]: [O::V; 2], r: O::V, [nx, ny]: [bool; 2]) -> [Option<O::V>; 2] {
        let [dy, dx] = self.0.emit_partials(o, [y, x], r, [ny, nx]);
        [dx, dy]
    }
}

impl<K: Formula<1>> Formula<2> for crate::derivative::Slope<K> {
    const MAP_ONLY: bool = true;
    fn supports(&self, dtype: DType) -> bool { matches!(dtype, DType::Float(_)) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, r]: [O::V; 2]) -> Option<O::V> {
        Some(self.0.emit_partials(o, [x], r, [true])[0].unwrap_or_else(|| o.number(F::narrow(f64::NAN))))
    }
}

impl<K: Formula<2>, const SIDE: usize> Formula<3> for crate::derivative::Slope<K, SIDE> {
    const MAP_ONLY: bool = true;
    fn supports(&self, dtype: DType) -> bool { matches!(dtype, DType::Float(_)) }
    fn emit<F: crate::Float, O: Ops<F> + ?Sized>(&self, o: &O, [x, y, r]: [O::V; 3]) -> Option<O::V> {
        Some(self.0.emit_partials(o, [x, y], r, std::array::from_fn(|i| i == SIDE))[SIDE].unwrap_or_else(|| o.number(F::narrow(f64::NAN))))
    }
}

#[inline(always)]
fn arithmetic_valid<F: crate::Float, O: Ops<F> + ?Sized>(
    variant: Variant, o: &O, op: crate::recipe::Binary, x: O::V, y: O::V) -> Option<(crate::error::Kind, O::B)> {
    (variant.fails_on_overflow() && variant.dtype.is_integral()).then(|| (crate::error::Kind::Overflow, o.not(o.overflow(op, x, y))))
}

#[inline(always)]
fn negation_valid<F: crate::Float, O: Ops<F> + ?Sized>(variant: Variant, o: &O, x: O::V, magnitude: bool) -> Option<(crate::error::Kind, O::B)> {
    let (kind, valid) = arithmetic_valid(variant, o, crate::recipe::Binary::Subtract, o.number(F::zero()), x)?;
    Some((kind, if magnitude { o.or(o.less_equal(o.number(F::zero()), x), valid) } else { valid }))
}

/// Numeric Boolean true: 1 within the run's comparison tolerance (`Ops::tolerance`), which BPL makes nonzero only under Mathish. The default tolerance where a run sets none.
#[inline]
fn boolean_one<F: crate::Float, O: Ops<F> + ?Sized>(o: &O, x: O::V) -> O::B {
    let one = o.number(F::one());
    crate::recipe::equal(o, x, one, |o| crate::recipe::close(o, o.tolerance(F::narrow(Tolerance::DEFAULT.value())), x, one))
}

#[inline(always)]
fn truth_valid<F: crate::Float, O: Ops<F> + ?Sized, const N: usize>(variant: Variant, o: &O, args: [O::V; N]) -> Option<(crate::error::Kind, O::B)> {
    variant.checked.then(|| {
        let valid = args.into_iter()
            .fold(o.equal(o.number(F::zero()), o.number(F::zero())), |valid, x| o.and(valid, o.or(o.equal(x, o.number(F::zero())), boolean_one(o, x))));
        (crate::error::Kind::Domain, valid)
    })
}

#[inline(always)]
fn power_valid<F: Float, O: Ops<F> + ?Sized>(variant: Variant, o: &O, [x, y]: [O::V; 2], result: O::V) -> Option<(crate::error::Kind, O::B)> {
    (variant.checked && !o.integral()).then(|| (crate::error::Kind::Domain, nan_valid(o, [x, y], [result])))
}
