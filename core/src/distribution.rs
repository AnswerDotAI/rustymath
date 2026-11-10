//! Probability distributions. Each has a density, a cumulative distribution, a quantile function and a sampler. The first three are also function objects, `Pdf`, `Cdf` and `Quantile`, which hold the distribution. Their native bulk routines choose the distribution's formulas outside the item loop. Closed-form distributions then vectorize. Scalar evaluation runs the same routine on one item. An item fails where its result is NaN and its argument isn't, as for a quantile of p outside 0 to 1. A discrete distribution's density is its mass at a whole x, and its quantile function and sampler give whole numbers.
use crate::{
    Arg, Atan, Erfc, Erfcinv, Float, Ln, Log1p, Native, betainc, betaincinv, cos, erfcinv, exp, expm1, gammainc, gammaincc, gammaincinv, lgamma, ln, log1p, pow,
    random::Items, sin, slice::{CHUNK, kept, written}
};
use rand::{Rng, distr::{Distribution as _, Open01, StandardUniform}};
use std::{f64::consts::{FRAC_1_SQRT_2, PI, SQRT_2, TAU}, mem::MaybeUninit};

/// √(2π).
const SQRT_TAU: f64 = 2.5066282746310002;

/// A probability distribution, whose constructor checked its parameters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Distribution(Law);

/// A parameter outside its distribution's domain, with a message that says which.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Invalid(pub &'static str);

impl std::fmt::Display for Invalid { fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result { f.write_str(self.0) } }

impl std::error::Error for Invalid {}

#[derive(Clone, Copy, Debug, PartialEq)]
enum Law {
    Normal(Normal), Uniform(Uniform), Beta(Beta), Cauchy(Cauchy), Exponential(Exponential), Fisher(Fisher), Gamma(Gamma), InverseGamma(InverseGamma),
    Laplace(Laplace), Logistic(Logistic), LogNormal(LogNormal), StudentT(StudentT), Weibull(Weibull), Binomial(Binomial), Poisson(Poisson)
}

/// `$body` with `$d` bound to the formulas of the distribution that `$law` holds.
macro_rules! each_law {
    ($law:expr, $d:ident => $body:expr) => {
        match $law {
            Law::Normal($d) => $body,
            Law::Uniform($d) => $body,
            Law::Beta($d) => $body,
            Law::Cauchy($d) => $body,
            Law::Exponential($d) => $body,
            Law::Fisher($d) => $body,
            Law::Gamma($d) => $body,
            Law::InverseGamma($d) => $body,
            Law::Laplace($d) => $body,
            Law::Logistic($d) => $body,
            Law::LogNormal($d) => $body,
            Law::StudentT($d) => $body,
            Law::Weibull($d) => $body,
            Law::Binomial($d) => $body,
            Law::Poisson($d) => $body,
        }
    };
}

/// One distribution's formulas. `quantile`, `quantile_map` and `quantile_many` take p from 0 to 1. The `_many` methods are the native bulk routines at float type `F`, which `Pdf`, `Cdf` and `Quantile` run for one item as for many. By default a native bulk routine runs its `_map` in a loop. A map gives one item's result at `F` and holds its constants, so that the loop computes them once. By default it computes in `f64` and rounds to `F`. A law whose formulas need nothing wider computes in `F` itself, so that its native bulk routines run at the arguments' width. A formula that takes a function from that function's native bulk routine overrides its `_many` method, through [`composed`].
trait Formulas {
    fn pdf(&self, x: f64) -> f64;
    fn cdf(&self, x: f64) -> f64;
    fn quantile(&self, p: f64) -> f64;
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { self.quantile(open(rng)) }
    /// The words that one attempt of a draw reads, which an item packs before its region's (`random::Items`).
    fn words(&self) -> usize { 1 }
    fn samples(&self, items: &mut Items, out: &mut [f64]) { for o in out { *o = self.sample(&mut items.item()); } }
    #[inline(always)]
    fn pdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy { move |x| narrowed(x, |x| self.pdf(x)) }
    #[inline(always)]
    fn cdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy { move |x| narrowed(x, |x| self.cdf(x)) }
    #[inline(always)]
    fn quantile_map<F: Float>(&self) -> impl Fn(F) -> F + Copy { move |p| narrowed(p, |p| self.quantile(p)) }
    #[inline(always)]
    fn pdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let pdf = self.pdf_map();
        each!([x], n, out, |[x]| pdf(x))
    }
    #[inline(always)]
    fn cdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let cdf = self.cdf_map();
        each!([x], n, out, |[x]| cdf(x))
    }
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let q = self.quantile_map(); each!([p], n, out, |[p]| q(p))
    }
}

/// `Formulas`' `f64` methods for a law whose generic methods hold its formulas. Each runs its native bulk routine on one item.
macro_rules! through_slices {
    () => {
        fn pdf(&self, x: f64) -> f64 {
            one_item(|out| self.pdf_many(Arg::One(x), 1, out))
        }
        fn cdf(&self, x: f64) -> f64 {
            one_item(|out| self.cdf_many(Arg::One(x), 1, out))
        }
        fn quantile(&self, p: f64) -> f64 {
            one_item(|out| self.quantile_many(Arg::One(p), 1, out))
        }
    };
}

/// The result that `many`, a native bulk routine, writes into its one slot.
#[inline(always)]
fn one_item<F: Float>(many: impl FnOnce(&mut [MaybeUninit<F>]) -> bool) -> F {
    let mut out = [MaybeUninit::uninit()];
    many(&mut out);
    (unsafe { written(&mut out) })[0]
}

/// `f` of `x`, computed in `f64` and rounded to `x`'s type.
#[inline(always)]
fn narrowed<F: Float>(x: F, f: impl Fn(f64) -> f64) -> F { F::narrow(f(x.wide())) }

/// [`Distribution::pdf`] as a function object. The log-normal distribution's native bulk routine takes ln from [`Ln`]'s native bulk routine, which vectorizes where a loop of whole densities doesn't. The Weibull distribution uses the shared [`Pow`] formula.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pdf(pub Distribution);
fn pdf_slice<F: Native>(k: &Pdf, _: &crate::recipe::NativeOps<{ crate::FUSED }>, [x]: [Arg<'_, F>; 1], n: usize, out: &mut [MaybeUninit<F>]) -> bool {
    each_law!(&k.0 .0, d => d.pdf_many(x, n, out))
}

/// [`Distribution::cdf`] as a function object. The normal and log-normal distributions' native bulk routines take erfc from [`Erfc`]'s native bulk routine, which computes only the formulas that each block of arguments needs. The Cauchy distribution's takes atan from [`Atan`]'s, and the Weibull distribution uses the shared [`Pow`] formula.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Cdf(pub Distribution);
fn cdf_slice<F: Native>(k: &Cdf, _: &crate::recipe::NativeOps<{ crate::FUSED }>, [x]: [Arg<'_, F>; 1], n: usize, out: &mut [MaybeUninit<F>]) -> bool {
    each_law!(&k.0 .0, d => d.cdf_many(x, n, out))
}

/// [`Distribution::quantile`] as a function object, through [`quantiles`]. The normal and log-normal distributions' native bulk routines take erfcinv(2p) from [`Erfcinv`]'s native bulk routine, whose loops don't branch on each item's region. Logarithms use [`Log1p`] and [`Ln`] native bulk routines; tangent and root use the shared [`Tan`] and [`Root`] formulas.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Quantile(pub Distribution);
fn quantile_slice<F: Native>(k: &Quantile, _: &crate::recipe::NativeOps<{ crate::FUSED }>, [p]: [Arg<'_, F>; 1], n: usize, out: &mut [MaybeUninit<F>]) -> bool {
    quantiles(p, n, out, |p, n, out| each_law!(&k.0 .0, d => d.quantile_many(p, n, out)))
}

macro_rules! native_law {
    ($( $object:ident => $slice:ident $(valid $valid:ident)? ),+) => {$(
        impl crate::facts::Formula<1> for $object {
            fn supports(&self, dtype: crate::buffer::DType) -> bool { matches!(dtype, crate::buffer::DType::Float(_)) }
            const FAILS_NAN: bool = true;
            $(fn emit_valid<F: Float, O: crate::recipe::Ops<F> + ?Sized>(&self, _: crate::domain::Variant, o: &O, [p]: [O::V; 1], _: O::V) -> Option<(crate::error::Kind, O::B)> { Some((crate::error::Kind::Domain, $valid(o, p))) })?
            fn real<F: Native>(&self, ops: &crate::recipe::NativeOps<{ crate::FUSED }>, [x]: [F; 1]) -> Option<F> {
                Some(one_item(|out| $slice(self, ops, [Arg::One(x)], 1, out)))
            }
            fn native_slices(&self) -> Option<crate::facts::NativeSlices<Self, 1>> { Some(crate::facts::NativeSlices::unary($slice, $slice)) }
        }
    )+};
}
native_law!(Pdf => pdf_slice, Cdf => cdf_slice, Quantile => quantile_slice valid probability);

fn probability<F: Float, O: crate::recipe::Ops<F> + ?Sized>(o: &O, p: O::V) -> O::B {
    o.and(o.less_equal(o.number(F::zero()), p), o.less_equal(p, o.number(F::one())))
}

/// Writes quantiles from `law`. Probabilities outside 0..=1, including NaN, produce NaN and mark the call invalid. The law receives 0.5 in those lanes.
fn quantiles<F: Float>(p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>], law: impl Fn(Arg<'_, F>, usize, &mut [MaybeUninit<F>]) -> bool) -> bool {
    let inside = |p: F| probability(&crate::recipe::Floats, p);
    let all = match p { Arg::Many(p) => p[..n].iter().fold(true, |all, &p| all & inside(p)), Arg::One(p) => inside(p) };
    if all { return law(p, n, out); }
    let within: Vec<F> = (0..n).map(|i| if inside(p.nth(i)) { p.nth(i) } else { F::narrow(0.5) }).collect();
    let ok = law(Arg::Many(&within), n, out);
    (0..n).filter(|&i| !inside(p.nth(i))).fold(ok, |_, i| { out[i].write(F::nan()); false })
}

/// `finish(x, v)` for each of the `n` items x of `x`, written into `out`. `values` writes a block's values v into the block's slots of `out`, from the block's arguments `inner(x)`, usually through one native bulk routine, as [`through`] gives it. It may overwrite the arguments. A `One` argument computes its result once. Each step runs over up to `CHUNK` items at a time, so the arguments and values between the steps stay in cache, and the arguments' buffer holds no more items than `x`. An item fails where its result is NaN and its argument isn't, as in every native bulk routine. Gives whether every item succeeded.
fn composed<F: Float>(
    x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>], inner: impl Fn(F) -> F, values: impl Fn(&mut [F], &mut [MaybeUninit<F>]), finish: impl Fn(F, F) -> F
) -> bool {
    let one;
    let (x, copies) = match x { Arg::Many(x) => (&x[..n], 0), Arg::One(v) => { one = [v]; (&one[..n.min(1)], n.saturating_sub(1)) } };
    let mut args = Vec::with_capacity(x.len().min(CHUNK));
    let mut ok = true;
    for (x, out) in x.chunks(CHUNK).zip(out.chunks_mut(CHUNK)) {
        args.clear();
        args.extend(x.iter().map(|&v| inner(v)));
        values(&mut args, &mut out[..x.len()]);
        let results = unsafe { written(&mut out[..x.len()]) };
        let mut clean = true;
        // The flag reads only the result, as `slice::each`'s does. x86-64 leaves this loop scalar when the flag also reads the argument.
        for (r, &v) in results.iter_mut().zip(x) {
            *r = finish(v, *r);
            clean &= !r.is_nan();
        }
        ok &= clean || results.iter().zip(x).all(|(&r, &v)| kept(r, [v]));
    }
    if copies > 0 {
        let r = unsafe { written(&mut out[..1]) }[0];
        out[1..n].fill(MaybeUninit::new(r));
    }
    ok
}

/// The values of `g`'s native bulk routine at a block's arguments, for [`composed`].
fn through<F: Native, K>(g: K, slice: crate::facts::Slice<K, F, 1>) -> impl Fn(&mut [F], &mut [MaybeUninit<F>]) {
    move |a, v| { slice(&g, &crate::recipe::NativeOps::DEFAULT, [Arg::Many(a)], a.len(), v); }
}

/// Each argument of a block raised to `k`, with `k`'s route chosen once (`pow::powers`), for [`composed`].
fn raised<F: Float>(k: F) -> impl Fn(&mut [F], &mut [MaybeUninit<F>]) {
    let e = crate::pow::Exponent::of(k.wide(), crate::FUSED);
    move |a, v| crate::pow::powers(e, k, a, v)
}

impl Distribution {
    /// The density at x, or for a discrete distribution the probability of x.
    pub fn pdf(&self, x: f64) -> f64 { one_item(|out| pdf_slice(&Pdf(*self), &crate::recipe::NativeOps::DEFAULT, [Arg::One(x)], 1, out)) }
    /// The probability of a value at most x.
    pub fn cdf(&self, x: f64) -> f64 { one_item(|out| cdf_slice(&Cdf(*self), &crate::recipe::NativeOps::DEFAULT, [Arg::One(x)], 1, out)) }
    /// The least x whose `cdf` is at least p. p outside 0 to 1 gives NaN.
    pub fn quantile(&self, p: f64) -> f64 { one_item(|out| quantile_slice(&Quantile(*self), &crate::recipe::NativeOps::DEFAULT, [Arg::One(p)], 1, out)) }
    /// The words that one attempt of a draw reads, which each item of `samples` packs (`random::Items`).
    pub fn words(&self) -> usize { each_law!(&self.0, d => d.words()) }
    /// Fills `out` with values drawn from the distribution, one for each item of `items`, which packs this distribution's `words`. A normal distribution's item gives two values, from one pair of uniform words.
    pub fn samples(&self, items: &mut Items, out: &mut [f64]) { each_law!(&self.0, d => d.samples(items, out)) }
    /// Whether the distribution's values are whole numbers.
    pub fn is_discrete(&self) -> bool { matches!(self.0, Law::Binomial(_) | Law::Poisson(_)) }
}

/// `Ok` when every parameter is finite and `ok` holds, or else `Invalid(message)`.
fn checked(parameters: &[f64], ok: bool, message: &'static str) -> Result<(), Invalid> {
    if !parameters.iter().all(|p| p.is_finite()) { Err(Invalid("distribution parameters must be finite")) } else if ok { Ok(()) } else { Err(Invalid(message)) }
}

/// The largest binomial trial count or Poisson rate: f64 holds every whole number up to it. Past it, `btrd`'s ratio walk (`i += 1.0`) and `least`'s halving stop advancing.
const WHOLE: u64 = 1 << 53;

impl Distribution {
    /// The normal distribution with this mean and standard deviation.
    pub fn normal(mean: f64, sd: f64) -> Result<Self, Invalid> {
        checked(&[mean, sd], sd > 0.0, "the standard deviation must be positive")?;
        Ok(Self(Law::Normal(Normal { mean, sd })))
    }
    /// The uniform distribution from `min` to `max`.
    pub fn uniform(min: f64, max: f64) -> Result<Self, Invalid> {
        checked(&[min, max, max - min], min < max, "the minimum must be below the maximum")?;
        Ok(Self(Law::Uniform(Uniform { min, max })))
    }
    /// The beta distribution with shapes a and b.
    pub fn beta(a: f64, b: f64) -> Result<Self, Invalid> {
        checked(&[a, b], a > 0.0 && b > 0.0, "the shapes must be positive")?;
        Ok(Self(Law::Beta(Beta::new(a, b))))
    }
    /// The Cauchy distribution with this location and scale.
    pub fn cauchy(location: f64, scale: f64) -> Result<Self, Invalid> {
        checked(&[location, scale], scale > 0.0, "the scale must be positive")?;
        Ok(Self(Law::Cauchy(Cauchy { location, scale })))
    }
    /// The chi-squared distribution with `df` degrees of freedom: the gamma distribution with shape df/2 and scale 2.
    pub fn chi_squared(df: f64) -> Result<Self, Invalid> {
        checked(&[df], df > 0.0, "the degrees of freedom must be positive")?;
        Ok(Self(Law::Gamma(Gamma::new(df / 2.0, 2.0))))
    }
    /// The exponential distribution with this rate.
    pub fn exponential(rate: f64) -> Result<Self, Invalid> {
        checked(&[rate], rate > 0.0, "the rate must be positive")?;
        Ok(Self(Law::Exponential(Exponential { rate })))
    }
    /// The F distribution with `df1` and `df2` degrees of freedom.
    pub fn fisher(df1: f64, df2: f64) -> Result<Self, Invalid> {
        checked(&[df1, df2], df1 > 0.0 && df2 > 0.0, "the degrees of freedom must be positive")?;
        Ok(Self(Law::Fisher(Fisher { df1, df2, beta: Beta::new(df1 / 2.0, df2 / 2.0) })))
    }
    /// The gamma distribution with this shape and scale.
    pub fn gamma(shape: f64, scale: f64) -> Result<Self, Invalid> {
        checked(&[shape, scale, 1.0 / scale], shape > 0.0 && scale > 0.0, "the shape and scale must be positive")?;
        Ok(Self(Law::Gamma(Gamma::new(shape, scale))))
    }
    /// The inverse gamma distribution with this shape and scale.
    pub fn inverse_gamma(shape: f64, scale: f64) -> Result<Self, Invalid> {
        checked(&[shape, scale], shape > 0.0 && scale > 0.0, "the shape and scale must be positive")?;
        Ok(Self(Law::InverseGamma(InverseGamma { shape, scale, ln_gamma: lgamma(shape) })))
    }
    /// The Laplace distribution with this location and scale.
    pub fn laplace(location: f64, scale: f64) -> Result<Self, Invalid> {
        checked(&[location, scale], scale > 0.0, "the scale must be positive")?;
        Ok(Self(Law::Laplace(Laplace { location, scale })))
    }
    /// The logistic distribution with this location and scale.
    pub fn logistic(location: f64, scale: f64) -> Result<Self, Invalid> {
        checked(&[location, scale], scale > 0.0, "the scale must be positive")?;
        Ok(Self(Law::Logistic(Logistic { location, scale })))
    }
    /// The log-normal distribution whose logarithm has this mean, the location, and this standard deviation, the scale.
    pub fn log_normal(location: f64, scale: f64) -> Result<Self, Invalid> {
        checked(&[location, scale], scale > 0.0, "the scale must be positive")?;
        Ok(Self(Law::LogNormal(LogNormal { location, scale })))
    }
    /// Student's t distribution with `df` degrees of freedom.
    pub fn student_t(df: f64) -> Result<Self, Invalid> {
        checked(&[df], df > 0.0, "the degrees of freedom must be positive")?;
        let factor = exp(lgamma((df + 1.0) / 2.0) - lgamma(df / 2.0)) / (df * PI).sqrt();
        Ok(Self(Law::StudentT(StudentT { df, factor })))
    }
    /// The Weibull distribution with this shape and scale.
    pub fn weibull(shape: f64, scale: f64) -> Result<Self, Invalid> {
        checked(&[shape, scale], shape > 0.0 && scale > 0.0, "the shape and scale must be positive")?;
        Ok(Self(Law::Weibull(Weibull { shape, scale })))
    }
    /// The binomial distribution of the successes in n trials, each with probability p.
    pub fn binomial(n: u64, p: f64) -> Result<Self, Invalid> {
        checked(&[p], (0.0..=1.0).contains(&p), "the probability must be from 0 to 1")?;
        if n > WHOLE { return Err(Invalid("binomial trials exceed the sampler's whole-number range")); }
        Ok(Self(Law::Binomial(Binomial { n: n as f64, p })))
    }
    /// The Poisson distribution with this rate.
    pub fn poisson(rate: f64) -> Result<Self, Invalid> {
        checked(&[rate], rate >= 0.0, "the rate must not be negative")?;
        if rate > WHOLE as f64 { return Err(Invalid("the Poisson rate exceeds the sampler's whole-number range")); }
        Ok(Self(Law::Poisson(Poisson { rate })))
    }
}

/// The standard parameters of the distributions that have them, in the order their constructors take them.
pub mod standard {
    pub const NORMAL: [f64; 2] = [0.0, 1.0];
    pub const UNIFORM: [f64; 2] = [0.0, 1.0];
    pub const CAUCHY: [f64; 2] = [0.0, 1.0];
    pub const EXPONENTIAL: [f64; 1] = [1.0];
    pub const LAPLACE: [f64; 2] = [0.0, 1.0];
    pub const LOGISTIC: [f64; 2] = [0.0, 1.0];
    pub const LOG_NORMAL: [f64; 2] = [0.0, 1.0];
}

/// c × l, or 0 when c is 0, so that a power x^0 is 1 at x = 0.
#[inline(always)]
fn times<F: Float>(c: F, l: F) -> F { if c == F::zero() { F::zero() } else { c * l } }

/// A value drawn uniformly from the open interval 0 to 1, whose logarithm is finite.
fn open<R: Rng + ?Sized>(rng: &mut R) -> f64 { Open01.sample(rng) }

/// A value drawn from the standard normal distribution, by the Box–Muller method.
fn standard_normal<R: Rng + ?Sized>(rng: &mut R) -> f64 {
    let (u, v): (f64, f64) = (open(rng), StandardUniform.sample(rng));
    (-2.0 * ln(u)).sqrt() * cos(TAU * v)
}

/// The words that one attempt of `standard_gamma` reads at `shape`: the normal draw's two and the test's one, and one more below a shape of 1.
fn gamma_words(shape: f64) -> usize { if shape < 1.0 { 4 } else { 3 } }

/// A value drawn from the gamma distribution with this shape and scale 1, by Marsaglia and Tsang's method. A shape below 1 draws with shape + 1 and scales by u^(1/shape).
fn standard_gamma<R: Rng + ?Sized>(shape: f64, rng: &mut R) -> f64 {
    if shape < 1.0 { return standard_gamma(shape + 1.0, rng) * exp(ln(open(rng)) / shape); }
    let d = shape - 1.0 / 3.0;
    let c = 1.0 / (9.0 * d).sqrt();
    loop {
        let x = standard_normal(rng);
        let v = 1.0 + c * x;
        if v <= 0.0 { continue; }
        let v = v * v * v;
        let u = open(rng);
        if u < 1.0 - 0.0331 * x * x * x * x || ln(u) < 0.5 * x * x + d * (1.0 - v + ln(v)) { return d * v; }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Normal { mean: f64, sd: f64 }
impl Normal {
    /// The argument of `erfc` in the cdf, (mean - x)/(sd√2), at float type `F`, as a map that multiplies by the reciprocal it computes when it's made. A loop over a closure that read `sd` through a reference divided for each item in x86-64 builds.
    #[inline(always)]
    fn erfc_line<F: Float>(&self) -> impl Fn(F) -> F + Copy + use<F> {
        let (mean, scale) = (F::narrow(self.mean), F::narrow(FRAC_1_SQRT_2 / self.sd));
        move |x| (mean - x) * scale
    }
    /// The quantile from e = erfcinv(2p), mean - sd·√2·e, at float type `F`, as a map that holds its constants.
    #[inline(always)]
    fn from_erfcinv<F: Float>(&self) -> impl Fn(F) -> F + Copy + use<F> {
        let (mean, step) = (F::narrow(self.mean), F::narrow(self.sd * SQRT_2));
        move |e| mean - step * e
    }
    /// The density at float type `F`, as a map that holds its constants, so that a loop over items computes them once.
    #[inline(always)]
    fn density<F: Float>(&self) -> impl Fn(F) -> F + Copy + use<F> {
        let (mean, rsd, half, height) = (F::narrow(self.mean), F::narrow(self.sd.recip()), F::narrow(-0.5), F::narrow((self.sd * SQRT_TAU).recip()));
        move |x| { let z = (x - mean) * rsd; exp(half * z * z) * height }
    }
}

impl Formulas for Normal {
    through_slices!();
    #[inline(always)]
    fn pdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy { self.density() }
    #[inline(always)]
    fn cdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let half = F::narrow(0.5);
        composed(x, n, out, self.erfc_line(), through(Erfc, crate::special::erfc_slice), #[inline(always)] move |_, e| half * e)
    }
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (two, from) = (F::narrow(2.0), self.from_erfcinv());
        composed(p, n, out, #[inline(always)] move |p| two * p, through(Erfcinv, crate::special::erfcinv_slice), #[inline(always)] move |_, e| from(e))
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { self.mean + self.sd * standard_normal(rng) }
    /// An item, a pair of uniform words, gives two values, from the cosine and the sine of one angle.
    fn words(&self) -> usize { 2 }
    fn samples(&self, items: &mut Items, out: &mut [f64]) {
        for pair in out.chunks_mut(2) {
            let mut item = items.item();
            let (u, v): (f64, f64) = (open(&mut item), StandardUniform.sample(&mut item));
            let r = self.sd * (-2.0 * ln(u)).sqrt();
            pair[0] = self.mean + r * cos(TAU * v);
            if let Some(second) = pair.get_mut(1) { *second = self.mean + r * sin(TAU * v); }
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Uniform { min: f64, max: f64 }

impl Uniform {
    /// The bounds and the width between them, at float type `F`.
    #[inline(always)]
    fn bounds<F: Float>(&self) -> (F, F, F) { let (min, max) = (F::narrow(self.min), F::narrow(self.max)); (min, max, max - min) }
}

impl Formulas for Uniform {
    through_slices!();
    #[inline(always)]
    fn pdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let (min, max, width) = self.bounds::<F>();
        let height = width.recip();
        move |x| if x >= min && x <= max { height } else { F::zero() }
    }
    #[inline(always)]
    fn cdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let (min, _, width) = self.bounds::<F>();
        move |x| {
            let c = (x - min) / width;
            if c < F::zero() { F::zero() } else if c > F::one() { F::one() } else { c }
        }
    }
    #[inline(always)]
    fn quantile_map<F: Float>(&self) -> impl Fn(F) -> F + Copy { let (min, _, width) = self.bounds::<F>(); move |p| min + p * width }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { let u: f64 = StandardUniform.sample(rng); self.min + u * (self.max - self.min) }
}

/// The beta distribution, with ln B(a, b) computed once.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Beta { a: f64, b: f64, ln_beta: f64 }

impl Beta { fn new(a: f64, b: f64) -> Self { Self { a, b, ln_beta: lgamma(a) + lgamma(b) - lgamma(a + b) } } }

impl Formulas for Beta {
    fn pdf(&self, x: f64) -> f64 {
        if !(0.0..=1.0).contains(&x) { return 0.0; }
        exp(times(self.a - 1.0, ln(x)) + times(self.b - 1.0, log1p(-x)) - self.ln_beta)
    }
    fn cdf(&self, x: f64) -> f64 { betainc(self.a, self.b, x.clamp(0.0, 1.0)) }
    fn quantile(&self, p: f64) -> f64 { betaincinv(self.a, self.b, p) }
    fn words(&self) -> usize { gamma_words(self.a) + gamma_words(self.b) }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { let x = standard_gamma(self.a, rng); x / (x + standard_gamma(self.b, rng)) }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Cauchy { location: f64, scale: f64 }

impl Formulas for Cauchy {
    through_slices!();
    #[inline(always)]
    fn pdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let (location, rscale, height) = (F::narrow(self.location), F::narrow(self.scale.recip()), F::narrow((PI * self.scale).recip()));
        move |x| { let z = (x - location) * rscale; height / (F::one() + z * z) }
    }
    /// The arctangent, from `Atan`'s native bulk routine, is of z near the centre and of -1/z beyond |z| = 1. From that arctangent a, the cdf is 1/2 + a/π near the centre, a/π below -1 and 1 + a/π above 1. There 1/2 + atan(z)/π would lose the lower tail's digits.
    #[inline(always)]
    fn cdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (location, rscale, rpi, half) = (F::narrow(self.location), F::narrow(self.scale.recip()), F::narrow(PI.recip()), F::narrow(0.5));
        composed(x, n, out, move |x| {
            let z = (x - location) * rscale;
            let (num, den) = if z.abs() <= F::one() { (z, F::one()) } else { (-F::one(), z) };
            num / den
        }, through(Atan, crate::arc::atan_slice), move |x, a| {
            let (z, a) = ((x - location) * rscale, a * rpi);
            if z.abs() <= F::one() { half + a } else if z < F::zero() { a } else { F::one() + a }
        })
    }
    /// The tangent, from the shared `Tan` formula, is of π·a, where a is p below p = 1/4, 1 - p above 3/4, and p - 1/2 between, so its argument stays within ±π/4. Near 0 and 1 the quantile is location ∓ scale/tan(π·a), which keeps p's digits there.
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (location, scale, pi, low, high, half) =
            (F::narrow(self.location), F::narrow(self.scale), F::narrow(PI), F::narrow(0.25), F::narrow(0.75), F::narrow(0.5));
        composed(p, n, out, move |p| {
            pi * if p < low { p } else if p > high { F::one() - p } else { p - half }
        }, |a, v| {
            each!([Arg::Many(a)], a.len(), v, |[x]| F::from_working(crate::trig::formula(
                &crate::recipe::NativeOps::<{ crate::FUSED }>::DEFAULT, F::working(x), crate::trig::Kind::Tan
            )));
        }, move |p, t| {
            let (num, den) = if p < low { (-scale, t) } else if p > high { (scale, t) } else { (scale * t, F::one()) };
            location + num / den
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Exponential { rate: f64 }

impl Formulas for Exponential {
    through_slices!();
    #[inline(always)]
    fn pdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let rate = F::narrow(self.rate);
        move |x| { let d = rate * exp(-rate * x); if x < F::zero() { F::zero() } else { d } }
    }
    #[inline(always)]
    fn cdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let rate = F::narrow(self.rate);
        move |x| { let c = -expm1(-rate * x); if x <= F::zero() { F::zero() } else { c } }
    }
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let mean = F::narrow(self.rate.recip());
        composed(p, n, out, |p| -p, through(Log1p, crate::ln::log1p_slice), move |_, l| -l * mean)
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { -ln(open(rng)) / self.rate }
}

/// The F distribution, through the beta distribution of y = df1 x / (df1 x + df2).
#[derive(Clone, Copy, Debug, PartialEq)]
struct Fisher { df1: f64, df2: f64, beta: Beta }

impl Fisher { fn y(&self, x: f64) -> f64 { self.df1 * x / (self.df1 * x + self.df2) } }

impl Formulas for Fisher {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x.is_infinite() { return 0.0; }
        let d = self.df1 * x + self.df2;
        self.beta.pdf(self.y(x)) * self.df1 * self.df2 / (d * d)
    }
    fn cdf(&self, x: f64) -> f64 { if x <= 0.0 { 0.0 } else if x.is_infinite() { 1.0 } else { self.beta.cdf(self.y(x)) } }
    fn quantile(&self, p: f64) -> f64 { let y = self.beta.quantile(p); self.df2 * y / (self.df1 * (1.0 - y)) }
    fn words(&self) -> usize { gamma_words(self.df1 / 2.0) + gamma_words(self.df2 / 2.0) }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let x = standard_gamma(self.df1 / 2.0, rng) / self.df1;
        x / (standard_gamma(self.df2 / 2.0, rng) / self.df2)
    }
}

/// The gamma distribution, with ln Γ(shape) computed once.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Gamma { shape: f64, scale: f64, ln_gamma: f64 }

impl Gamma { fn new(shape: f64, scale: f64) -> Self { Self { shape, scale, ln_gamma: lgamma(shape) } } }

impl Formulas for Gamma {
    fn pdf(&self, x: f64) -> f64 {
        if x < 0.0 || x.is_infinite() { return 0.0; }
        exp(times(self.shape - 1.0, ln(x)) - x / self.scale - self.ln_gamma - self.shape * ln(self.scale))
    }
    fn cdf(&self, x: f64) -> f64 { if x <= 0.0 { 0.0 } else { gammainc(self.shape, x / self.scale) } }
    fn quantile(&self, p: f64) -> f64 { self.scale * gammaincinv(self.shape, p) }
    fn words(&self) -> usize { gamma_words(self.shape) }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { self.scale * standard_gamma(self.shape, rng) }
}

/// The inverse gamma distribution: 1/x has a gamma distribution with scale 1/scale.
#[derive(Clone, Copy, Debug, PartialEq)]
struct InverseGamma { shape: f64, scale: f64, ln_gamma: f64 }

impl Formulas for InverseGamma {
    fn pdf(&self, x: f64) -> f64 {
        if x <= 0.0 { return 0.0; }
        exp(self.shape * ln(self.scale) - self.ln_gamma - (self.shape + 1.0) * ln(x) - self.scale / x)
    }
    fn cdf(&self, x: f64) -> f64 { if x <= 0.0 { 0.0 } else { gammaincc(self.shape, self.scale / x) } }
    fn quantile(&self, p: f64) -> f64 { self.scale / gammaincinv(self.shape, 1.0 - p) }
    fn words(&self) -> usize { gamma_words(self.shape) }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { self.scale / standard_gamma(self.shape, rng) }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Laplace { location: f64, scale: f64 }

impl Formulas for Laplace {
    through_slices!();
    #[inline(always)]
    fn pdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let (location, rscale, height) = (F::narrow(self.location), F::narrow(self.scale.recip()), F::narrow((2.0 * self.scale).recip()));
        move |x| exp(-((x - location) * rscale).abs()) * height
    }
    /// Both halves share one exponential, of -|z|, which keeps the loop free of branches.
    #[inline(always)]
    fn cdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let (location, rscale, half) = (F::narrow(self.location), F::narrow(self.scale.recip()), F::narrow(0.5));
        move |x| {
            let z = (x - location) * rscale;
            let e = half * exp(-z.abs());
            if z < F::zero() { e } else { F::one() - e }
        }
    }
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (location, scale, half) = (F::narrow(self.location), F::narrow(self.scale), F::narrow(0.5));
        composed(p, n, out, move |p| {
            let r = if p < half { p } else { F::one() - p };
            r + r
        }, through(Ln, crate::ln::ln_slice), move |p, l| if p < half { location + scale * l } else { location - scale * l })
    }
}

/// The logistic distribution. Its formulas use e^-|z|, which can't overflow, so its tails stay exact.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Logistic { location: f64, scale: f64 }

impl Formulas for Logistic {
    through_slices!();
    #[inline(always)]
    fn pdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let (location, rscale) = (F::narrow(self.location), F::narrow(self.scale.recip()));
        move |x| {
            let e = exp(-((x - location) * rscale).abs());
            let d = F::one() + e;
            e / (d * d) * rscale
        }
    }
    #[inline(always)]
    fn cdf_map<F: Float>(&self) -> impl Fn(F) -> F + Copy {
        let (location, rscale) = (F::narrow(self.location), F::narrow(self.scale.recip()));
        move |x| {
            let z = (x - location) * rscale;
            let e = exp(-z.abs());
            let r = (F::one() + e).recip();
            if z < F::zero() { e * r } else { r }
        }
    }
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (location, scale) = (F::narrow(self.location), F::narrow(self.scale));
        composed(p, n, out, |p| p / (F::one() - p), through(Ln, crate::ln::ln_slice), move |_, l| location + scale * l)
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct LogNormal { location: f64, scale: f64 }
impl LogNormal {
    /// The normal distribution that ln x follows. The log-normal density at x is its density at ln x over x, the log-normal cdf is its cdf at ln x, and the log-normal quantile and samples are exponentials of its own.
    #[inline(always)]
    fn normal(&self) -> Normal { Normal { mean: self.location, sd: self.scale } }
}

impl Formulas for LogNormal {
    through_slices!();
    /// A nonpositive or NaN x goes to `Ln` as 1. Its density is then 0, or NaN for NaN.
    #[inline(always)]
    fn pdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let pdf = self.normal().density();
        composed(x, n, out, #[inline(always)] |x| if x > F::zero() { x } else { F::one() }, through(Ln, crate::ln::ln_slice), #[inline(always)] move |x, l| {
            let p = pdf(l) / x;
            if x <= F::zero() { F::zero() } else { p }
        })
    }
    #[inline(always)]
    fn cdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (line, half) = (self.normal().erfc_line(), F::narrow(0.5));
        composed(
            x, n, out, #[inline(always)] move |x| line(ln(x)), through(Erfc, crate::special::erfc_slice), #[inline(always)] move |x, e| if x <= F::zero() { F::zero() } else { half * e })
    }
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (two, from) = (F::narrow(2.0), self.normal().from_erfcinv());
        composed(p, n, out, #[inline(always)] move |p| two * p, through(Erfcinv, crate::special::erfcinv_slice), #[inline(always)] move |_, e| exp(from(e)))
    }
    fn words(&self) -> usize { 2 }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { exp(self.normal().sample(rng)) }
}

/// Student's t distribution, with its density's constant factor computed once. Its cdf comes from the incomplete beta function: of y = t²/(df + t²) near the centre, and of 1 - y = df/(df + t²) in the tails, whichever keeps its digits.
#[derive(Clone, Copy, Debug, PartialEq)]
struct StudentT { df: f64, factor: f64 }

impl Formulas for StudentT {
    fn pdf(&self, t: f64) -> f64 { self.factor * exp(-(self.df + 1.0) / 2.0 * log1p(t * t / self.df)) }
    fn cdf(&self, t: f64) -> f64 {
        let (h, d) = (self.df / 2.0, self.df + t * t);
        let y = t * t / d;
        // The probability between -|t| and |t|, or beyond them.
        let (inside, outside) = if y < 0.5 { let i = betainc(0.5, h, y); (i, 1.0 - i) } else { let o = betainc(h, 0.5, self.df / d); (1.0 - o, o) };
        if t < 0.0 { 0.5 * outside } else { 0.5 + 0.5 * inside }
    }
    fn quantile(&self, p: f64) -> f64 {
        let h = self.df / 2.0;
        // The probability outside -|t| to |t|, and inside it, where 1 - 2q is exact.
        let q = 2.0 * p.min(1.0 - p);
        let t = if q > 0.5 { let y = betaincinv(0.5, h, 1.0 - q); (self.df * y / (1.0 - y)).sqrt() }
        else { let x = betaincinv(h, 0.5, q); (self.df * (1.0 - x) / x).sqrt() };
        if p < 0.5 { -t } else { t }
    }
    fn words(&self) -> usize { 2 + gamma_words(self.df / 2.0) }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { standard_normal(rng) / (standard_gamma(self.df / 2.0, rng) * 2.0 / self.df).sqrt() }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Weibull { shape: f64, scale: f64 }

impl Formulas for Weibull {
    through_slices!();
    /// shape/scale·z^(shape-1)·e^(-z^shape) for z = x/scale. The shared `Pow` formula gives z^(shape-1), from a base of 1 for a negative x. The density is 0 where e^(-z^shape) underflows, even where z^(shape-1) overflows, and for a negative or infinite x.
    #[inline(always)]
    fn pdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (rscale, height) = (F::narrow(self.scale.recip()), F::narrow(self.shape / self.scale));
        composed(x, n, out, move |x| if x < F::zero() { F::one() } else { x * rscale }, raised(F::narrow(self.shape - 1.0)), move |x, w| {
            let e = exp(-times(x * rscale, w));
            let d = if e == F::zero() { e } else { height * w * e };
            if x < F::zero() || x.is_infinite() { F::zero() } else { d }
        })
    }
    /// 1 - e^(-z^shape) for z = x/scale. The shared `Pow` formula gives z^shape, from a base of 1 for a nonpositive x, whose cdf is 0.
    #[inline(always)]
    fn cdf_many<F: Native>(&self, x: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let rscale = F::narrow(self.scale.recip());
        composed(x, n, out, move |x| if x <= F::zero() { F::one() } else { x * rscale }, raised(F::narrow(self.shape)), |x, w| {
            let c = -expm1(-w);
            if x <= F::zero() { F::zero() } else { c }
        })
    }
    /// scale·(-log1p(-p))^(1/shape), from `Log1p`'s native bulk routine and the shared `Root` formula. `Root` takes shape itself as its degree, where `Pow` would need 1/shape rounded.
    #[inline(always)]
    fn quantile_many<F: Native>(&self, p: Arg<'_, F>, n: usize, out: &mut [MaybeUninit<F>]) -> bool {
        let (degree, scale) = (F::narrow(self.shape), F::narrow(self.scale));
        composed(p, n, out, |p| -p, move |a, v| {
            crate::ln::log1p_slice(&Log1p, &crate::recipe::NativeOps::DEFAULT, [Arg::Many(a)], a.len(), v);
            for (x, &l) in a.iter_mut().zip(unsafe { written(v) }.iter()) { *x = -l; }
            crate::pow::roots(degree, a, v);
        }, move |_, w| scale * w)
    }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 { self.scale * pow(-ln(open(rng)), 1.0 / self.shape) }
}

/// Whether x is a whole number from 0 to `max`.
fn whole(x: f64, max: f64) -> bool { (0.0..=max).contains(&x) && x.fract() == 0.0 }

/// The least whole k from 0 to `max` whose `cdf` reaches `target`, searched outward from `guess` and then by halving. A cdf within 1e-14 below `target` reaches it, so a probability that the cdf of a whole k gives only after rounding maps back to that k.
fn least(target: f64, guess: f64, max: f64, cdf: impl Fn(f64) -> f64) -> f64 {
    let reached = |k: f64| cdf(k) >= target * (1.0 - 1e-14);
    let start = if guess.is_nan() { 0.0 } else { guess.floor().clamp(0.0, max) };
    let (mut below, mut above) = (start, start);
    let mut step = 1.0;
    if reached(start) {
        // Step down until the cdf no longer reaches the target. -1 stands below 0, where it never does.
        loop {
            below = (above - step).max(-1.0);
            if below < 0.0 || !reached(below) { break; }
            above = below;
            step *= 2.0;
        }
    } else {
        loop {
            if below >= max { return max; }
            above = (below + step).min(max);
            if reached(above) { break; }
            below = above;
            step *= 2.0;
        }
    }
    while above - below > 1.0 { let middle = ((below + above) / 2.0).floor(); if reached(middle) { above = middle } else { below = middle } }
    above
}

/// A normal approximation of the quantile, as the starting point of `least`: mean + sd z, for the standard normal quantile z of p.
fn normal_guess(mean: f64, sd: f64, p: f64) -> f64 { mean - sd * SQRT_2 * erfcinv(2.0 * p) }

#[derive(Clone, Copy, Debug, PartialEq)]
struct Binomial { n: f64, p: f64 }

impl Formulas for Binomial {
    fn pdf(&self, x: f64) -> f64 {
        if !whole(x, self.n) { return 0.0; }
        let (n, p) = (self.n, self.p);
        exp(lgamma(n + 1.0) - lgamma(x + 1.0) - lgamma(n - x + 1.0) + times(x, ln(p)) + times(n - x, log1p(-p)))
    }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { return 0.0; }
        if x >= self.n || self.p == 0.0 { return 1.0; }
        if self.p == 1.0 { return 0.0; }
        let k = x.floor();
        betainc(self.n - k, k + 1.0, 1.0 - self.p)
    }
    /// A distribution with p = 0 or 1 has one value, which is every quantile.
    fn quantile(&self, q: f64) -> f64 {
        let (n, p) = (self.n, self.p);
        if p == 0.0 { 0.0 } else if p == 1.0 { n } else { least(q, normal_guess(n * p, (n * p * (1.0 - p)).sqrt(), q), n, |k| self.cdf(k)) }
    }
    /// Inversion when the smaller of np and n(1-p) is below 10, and Hörmann's BTRD method otherwise. Both draw with p up to 1/2 and reflect. Inversion reads one uniform draw and BTRD two.
    fn words(&self) -> usize { if self.n * self.p.min(1.0 - self.p) < 10.0 { 1 } else { 2 } }
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let flip = self.p > 0.5;
        let p = if flip { 1.0 - self.p } else { self.p };
        let k = if self.n * p < 10.0 { binomial_inversion(self.n, p, rng) } else { btrd(self.n, p, rng) };
        if flip { self.n - k } else { k }
    }
}

/// A binomial draw by inversion: the cdf's terms in turn, until they pass a uniform draw. Past np + 10 √(np(1-p) + 1), which a draw almost never reaches, it starts again.
fn binomial_inversion<R: Rng + ?Sized>(n: f64, p: f64, rng: &mut R) -> f64 {
    let q = 1.0 - p;
    let first = exp(n * log1p(-p));
    let bound = n.min(n * p + 10.0 * (n * p * q + 1.0).sqrt());
    'draw: loop {
        let (mut u, mut mass, mut k): (f64, f64, f64) = (StandardUniform.sample(rng), first, 0.0);
        while u > mass {
            u -= mass;
            k += 1.0;
            if k > bound { continue 'draw; }
            mass *= (n - k + 1.0) * p / (k * q);
        }
        return k;
    }
}

/// The correction ln k! - ((k + 1/2) ln(k + 1) - (k + 1) + ln √(2π)) to Stirling's formula, from a table below 10 and its series above.
fn stirling(k: f64) -> f64 {
    const TABLE: [f64; 10] = [
        0.08106146679532726, 0.0413406959554093, 0.02767792568499834, 0.020790672103765093, 0.016644691189821193, 0.013876128823070748, 0.01189670994589177,
        0.010411265261972096, 0.009255462182712733, 0.00833056343336287,
    ];
    if k < 10.0 { return TABLE[k as usize]; }
    let r = 1.0 / (k + 1.0);
    let r2 = r * r;
    (1.0 / 12.0 - (1.0 / 360.0 - r2 / 1260.0) * r2) * r
}

/// A binomial draw by Hörmann's BTRD method (transformed rejection with decomposition), for p up to 1/2 and np from 10 up.
fn btrd<R: Rng + ?Sized>(n: f64, p: f64, rng: &mut R) -> f64 {
    let q = 1.0 - p;
    let (m, r) = (((n + 1.0) * p).floor(), p / q);
    let (nr, npq) = ((n + 1.0) * r, n * p * q);
    let sq = npq.sqrt();
    let b = 1.15 + 2.53 * sq;
    let a = -0.0873 + 0.0248 * b + 0.01 * p;
    let c = n * p + 0.5;
    let alpha = (2.83 + 5.1 / b) * sq;
    let vr = 0.92 - 4.2 / b;
    let urvr = 0.86 * vr;
    loop {
        let mut v: f64 = StandardUniform.sample(rng);
        if v <= urvr { let u = v / vr - 0.43; return ((2.0 * a / (0.5 - u.abs()) + b) * u + c).floor(); }
        let u = if v >= vr { let u: f64 = StandardUniform.sample(rng); u - 0.5 } else {
            let w = v / vr - 0.93;
            let w2: f64 = StandardUniform.sample(rng);
            v = w2 * vr;
            0.5f64.copysign(w) - w
        };
        let us = 0.5 - u.abs();
        let k = ((2.0 * a / us + b) * u + c).floor();
        if k < 0.0 || k > n { continue; }
        v *= alpha / (a / (us * us) + b);
        let km = (k - m).abs();
        if km <= 15.0 {
            // f(k)/f(m) by the ratio of successive terms.
            let mut f = 1.0;
            let mut i = m.min(k);
            while i < m.max(k) { i += 1.0; if m < k { f *= nr / i - r } else { v *= nr / i - r } }
            if v <= f { return k; }
            continue;
        }
        // Bounds on ln f(k)/f(m) accept or reject most draws before the exact test.
        let v = ln(v);
        let rho = (km / npq) * (((km / 3.0 + 0.625) * km + 1.0 / 6.0) / npq + 0.5);
        let t = -km * km / (2.0 * npq);
        if v < t - rho { return k; }
        if v > t + rho { continue; }
        let (nm, nk) = (n - m + 1.0, n - k + 1.0);
        let h = (m + 0.5) * ln((m + 1.0) / (r * nm)) + stirling(m) + stirling(n - m);
        if v <= h + (n + 1.0) * ln(nm / nk) + (k + 0.5) * ln(nk * r / (k + 1.0)) - stirling(k) - stirling(n - k) { return k; }
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Poisson { rate: f64 }

impl Formulas for Poisson {
    fn pdf(&self, x: f64) -> f64 { if !whole(x, f64::INFINITY) { return 0.0; } exp(times(x, ln(self.rate)) - self.rate - lgamma(x + 1.0)) }
    fn cdf(&self, x: f64) -> f64 {
        if x < 0.0 { 0.0 } else if self.rate == 0.0 || x.is_infinite() { 1.0 } else { gammaincc(x.floor() + 1.0, self.rate) }
    }
    /// No whole number has cdf 1, so the quantile of 1 is ∞.
    fn quantile(&self, q: f64) -> f64 {
        if self.rate == 0.0 { 0.0 }
        else if q == 1.0 { f64::INFINITY }
        else { least(q, normal_guess(self.rate, self.rate.sqrt(), q), f64::INFINITY, |k| self.cdf(k)) }
    }
    /// Below a rate of 10, an attempt multiplies uniform draws until their product passes e^-rate: the expected count, rate + 1, is packed. PTRS reads two.
    fn words(&self) -> usize { if self.rate < 10.0 { (self.rate + 1.0).ceil() as usize } else { 2 } }
    /// Multiplying uniform draws below a rate of 10, and Hörmann's PTRS method (transformed rejection with squeeze) above.
    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> f64 {
        let rate = self.rate;
        if rate < 10.0 {
            let limit = exp(-rate);
            let (mut k, mut product) = (0.0, 1.0);
            loop {
                let u: f64 = StandardUniform.sample(rng);
                product *= u;
                if product <= limit { return k; }
                k += 1.0;
            }
        }
        let (root, ln_rate) = (rate.sqrt(), ln(rate));
        let b = 0.931 + 2.53 * root;
        let a = -0.059 + 0.02483 * b;
        let ln_alpha = ln(1.1239 + 1.1328 / (b - 3.4));
        let vr = 0.9277 - 3.6224 / (b - 2.0);
        loop {
            let (u, v): (f64, f64) = (StandardUniform.sample(rng), StandardUniform.sample(rng));
            let u = u - 0.5;
            let us = 0.5 - u.abs();
            let k = ((2.0 * a / us + b) * u + rate + 0.43).floor();
            if us >= 0.07 && v <= vr { return k; }
            if k < 0.0 || (us < 0.013 && v > us) { continue; }
            if ln(v) + ln_alpha - ln(a / (us * us) + b) <= -rate + k * ln_rate - lgamma(k + 1.0) { return k; }
        }
    }
}
