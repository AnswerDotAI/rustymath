//! Minimizing a function of several variables, and finding a zero of a function of one.
//!
//! [`minimize`] runs one of five methods on an [`Objective`]. [`Options`] gives the method, the bounds and when to stop. Where it
//! gives no method, [`Method::choose`] picks one from the number of variables, whether the objective has a gradient and whether
//! there are bounds. Every method starts from the start point moved inside the bounds, evaluates the objective only inside them, and
//! returns the best point it evaluated. A NaN value counts as +∞. An error from the objective stops the search and comes back as
//! `minimize`'s error. A search that uses up its evaluations returns its best point with `converged` false.
//!
//! [`root`] finds a zero of a function of one variable between two points where its signs differ, by Brent's method.
use crate::{linalg::{axpy, dot}, random::SplitMix64, Native};
use rand::RngExt;
use std::collections::VecDeque;

/// A function of a slice of `F` to minimize. A closure that returns a `Result` is one. [`Differentiable`] adds a gradient.
pub trait Objective<F> {
    /// What stops an evaluation, such as an error in the caller's function or an interrupt.
    type Error;
    /// The value at `x`.
    fn value(&mut self, x: &[F]) -> Result<F, Self::Error>;
    /// The value at each row of `xs`, into `out`, with `xs.len() / out.len()` items to a row. Nelder–Mead evaluates its first simplex
    /// and each shrink through one call, and differential evolution each generation. By default it calls `value` on each row.
    fn values(&mut self, xs: &[F], out: &mut [F]) -> Result<(), Self::Error> {
        let n = xs.len() / out.len();
        for (y, x) in out.iter_mut().zip(xs.chunks_exact(n)) { *y = self.value(x)? }
        Ok(())
    }
    /// Whether `gradient` gives the gradient. L-BFGS needs one, and [`Method::choose`] picks L-BFGS for an objective that has one.
    fn has_gradient(&self) -> bool { false }
    /// The gradient at `x`, into `g`. Methods call it only where `has_gradient` is true.
    fn gradient(&mut self, x: &[F], g: &mut [F]) -> Result<(), Self::Error> {
        let _ = (x, g);
        unreachable!("this objective has no gradient")
    }
}

impl<F, E, V: FnMut(&[F]) -> Result<F, E>> Objective<F> for V {
    type Error = E;
    fn value(&mut self, x: &[F]) -> Result<F, E> { self(x) }
}

/// An objective from a closure that gives its value and one that writes its gradient.
pub struct Differentiable<V, G> {
    pub value: V,
    pub gradient: G,
}

impl<F, E, V: FnMut(&[F]) -> Result<F, E>, G: FnMut(&[F], &mut [F]) -> Result<(), E>> Objective<F> for Differentiable<V, G> {
    type Error = E;
    fn value(&mut self, x: &[F]) -> Result<F, E> { (self.value)(x) }
    fn has_gradient(&self) -> bool { true }
    fn gradient(&mut self, x: &[F], g: &mut [F]) -> Result<(), E> { (self.gradient)(x, g) }
}

/// The methods [`minimize`] runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Brent's method, for one variable: golden-section steps, and parabolic steps where they shrink the interval fast enough. It
    /// first brackets a minimum with growing steps from the start, which stop at a bound.
    Brent,
    /// Nelder–Mead's simplex, with Gao and Han's coefficients for the number of variables. It restarts from its best point until a
    /// restart no longer lowers the value.
    NelderMead,
    /// Powell's 1964 method of conjugate directions: Brent's method along each direction in turn, then along the step they made
    /// together, which replaces the direction of largest decrease where Powell's test accepts it.
    Powell,
    /// L-BFGS with the last 10 steps and a strong Wolfe line search, for an objective with a gradient. It takes no bounds.
    Lbfgs,
    /// Differential evolution within finite bounds, rand/1/bin with crossover 0.7 and a mutation factor drawn from 0.5 to 1 each
    /// generation. The start is one member, and the rest are uniform within the bounds. It stops when the members' values lie within
    /// 1% of the best, or their points within 1% of the bounds' width in every variable. Powell's method then polishes the best point.
    Evolution,
}

impl Method {
    /// Each method's name, which `FromStr` reads, ignoring case, and `Display` writes.
    pub const NAMES: [(&'static str, Method); 5] = [
        ("brent", Method::Brent),
        ("nelder-mead", Method::NelderMead),
        ("powell", Method::Powell),
        ("l-bfgs", Method::Lbfgs),
        ("differential-evolution", Method::Evolution),
    ];

    /// The method [`minimize`] runs when [`Options::method`] is `None`: Brent's method for one variable, L-BFGS for an objective with
    /// a gradient and no bounds, Powell's method within bounds, and Nelder–Mead otherwise.
    pub fn choose(variables: usize, gradient: bool, bounded: bool) -> Method {
        if variables == 1 { Method::Brent }
        else if gradient && !bounded { Method::Lbfgs }
        else if bounded { Method::Powell }
        else { Method::NelderMead }
    }
}

impl std::str::FromStr for Method {
    type Err = &'static str;
    fn from_str(name: &str) -> Result<Method, &'static str> {
        let found = Method::NAMES.iter().find(|(key, _)| key.eq_ignore_ascii_case(name));
        found.map(|&(_, method)| method).ok_or("unknown method: use brent, nelder-mead, powell, l-bfgs or differential-evolution")
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(Method::NAMES.iter().find(|&&(_, method)| method == *self).expect("every method has a name").0)
    }
}

/// The least and greatest value of each variable. An infinite limit leaves that side open.
#[derive(Clone, Copy, Debug)]
pub struct Bounds<'a, F> {
    pub lower: &'a [F],
    pub upper: &'a [F],
}

impl<F: Native> Bounds<'_, F> {
    /// Moves `x` inside the bounds.
    fn clamp(&self, x: &mut [F]) {
        for ((x, &l), &u) in x.iter_mut().zip(self.lower).zip(self.upper) { *x = x.max(l).min(u) }
    }

    /// The range of `t` that keeps `x + t d` inside the bounds.
    fn along(&self, x: &[F], d: &[F]) -> (F, F) {
        let mut range = (F::neg_infinity(), F::infinity());
        for i in (0..x.len()).filter(|&i| d[i] != F::zero()) {
            let (p, q) = ((self.lower[i] - x[i]) / d[i], (self.upper[i] - x[i]) / d[i]);
            range = (range.0.max(p.min(q)), range.1.min(p.max(q)));
        }
        range
    }
}

/// What [`minimize`] runs and when it stops. `Options::default()` lets [`Method::choose`] pick the method and leaves every
/// variable unbounded.
#[derive(Clone, Copy, Debug, Default)]
pub struct Options<'a, F> {
    /// The method, or `None` for the one [`Method::choose`] picks.
    pub method: Option<Method>,
    /// Limits on the variables. L-BFGS takes none, and differential evolution needs finite ones.
    pub bounds: Option<Bounds<'a, F>>,
    /// The relative change at which a method stops, or `None` for √ε of `F`.
    pub tolerance: Option<F>,
    /// The most evaluations of the objective, or `None` for 1000 for each variable, plus 1000, and for differential evolution another
    /// 1000 for each member of its population.
    pub evaluations: Option<usize>,
    /// Differential evolution's population, or `None` for 15 for each variable, and at least 5.
    pub population: Option<usize>,
    /// The seed of differential evolution's generator.
    pub seed: u64,
}

impl<F: Native> Options<'_, F> {
    /// The method [`minimize`] runs for `variables` variables and an objective with or without a gradient: `method`, or the one
    /// [`Method::choose`] picks.
    pub fn method_for(&self, variables: usize, gradient: bool) -> Method {
        self.method.unwrap_or_else(|| Method::choose(variables, gradient, self.bounds.is_some()))
    }

    /// Why [`minimize`] would reject these options for `variables` variables and an objective with or without a gradient. A binding
    /// calls it first and reports the message as an error.
    pub fn check(&self, variables: usize, gradient: bool) -> Result<(), &'static str> {
        let bounds = self.bounds;
        if bounds.is_some_and(|b| b.lower.len() != variables || b.upper.len() != variables) {
            return Err("bounds need a lower and an upper limit for each variable");
        }
        if bounds.is_some_and(|b| b.lower.iter().zip(b.upper).any(|(l, u)| !(l <= u))) {
            return Err("each lower limit must be at most its upper limit");
        }
        if self.tolerance.is_some_and(|t| !(t > F::zero() && t.is_finite())) { return Err("the tolerance must be a positive number") }
        match self.method_for(variables, gradient) {
            Method::Brent if variables > 1 => Err("Brent's method takes one variable"),
            Method::Lbfgs if !gradient => Err("L-BFGS needs an objective with a gradient"),
            Method::Lbfgs if bounds.is_some() => Err("L-BFGS takes no bounds"),
            Method::Evolution if !bounds.is_some_and(|b| b.lower.iter().chain(b.upper).all(|v| v.is_finite())) => {
                Err("differential evolution needs finite bounds")
            }
            Method::Evolution if self.population.is_some_and(|p| p < 4) => Err("differential evolution needs a population of at least 4"),
            _ => Ok(()),
        }
    }
}

/// What [`minimize`] or [`root`] found.
#[derive(Clone, Debug, PartialEq)]
pub struct Found<X, F> {
    /// The best point evaluated. For `root`, it's the end of the last bracket nearer zero.
    pub x: X,
    /// The function's value at `x`.
    pub value: F,
    /// How many times the function was evaluated.
    pub evaluations: usize,
    /// Whether the method's stopping test passed before the evaluations ran out.
    pub converged: bool,
}

/// The minimum of `objective` from `start`, by the method [`Options::method_for`] gives.
///
/// # Panics
///
/// If [`Options::check`] rejects the options.
pub fn minimize<F: Native, O: Objective<F>>(objective: &mut O, start: &[F], options: &Options<'_, F>) -> Result<Found<Vec<F>, F>, O::Error> {
    let (n, gradient) = (start.len(), objective.has_gradient());
    if let Err(problem) = options.check(n, gradient) { panic!("{problem}") }
    if n == 0 {
        let value = objective.value(start)?;
        return Ok(Found { x: Vec::new(), value: if value.is_nan() { F::infinity() } else { value }, evaluations: 1, converged: true });
    }
    let method = options.method_for(n, gradient);
    let tolerance = options.tolerance.unwrap_or_else(|| F::epsilon().sqrt());
    let population = options.population.unwrap_or((15 * n).max(5));
    let limit = options.evaluations.unwrap_or(1000 * (n + 1) + if method == Method::Evolution { 1000 * population } else { 0 });
    let mut search = Search::new(objective, start, options.bounds, limit);
    let outcome = match method {
        Method::Brent => brent_method(&mut search, tolerance),
        Method::NelderMead => nelder_mead(&mut search, tolerance),
        Method::Powell => powell(&mut search, tolerance),
        Method::Lbfgs => lbfgs(&mut search, tolerance),
        Method::Evolution => evolution(&mut search, population, tolerance, options.seed),
    };
    match outcome {
        Ok(converged) => Ok(search.found(converged)),
        Err(Halt::Spent) => Ok(search.found(false)),
        Err(Halt::Error(e)) => Err(e),
    }
}

/// Why a method stopped before its stopping test passed: the objective's error, or the evaluations ran out.
enum Halt<E> {
    Error(E),
    Spent,
}

/// The objective as every method sees it: each value counted against the limit, taken at the caller's point after moving it inside
/// the bounds, with NaN as +∞, and the best point so far kept.
struct Search<'a, F, O> {
    objective: &'a mut O,
    bounds: Option<Bounds<'a, F>>,
    limit: usize,
    evaluations: usize,
    best: Vec<F>,
    least: F,
}

impl<'a, F: Native, O: Objective<F>> Search<'a, F, O> {
    fn new(objective: &'a mut O, start: &[F], bounds: Option<Bounds<'a, F>>, limit: usize) -> Self {
        let mut best = start.to_vec();
        if let Some(b) = bounds { b.clamp(&mut best) }
        Search { objective, bounds, limit, evaluations: 0, best, least: F::infinity() }
    }

    /// Counts `k` evaluations, and moves each row of `xs` inside the bounds.
    fn prepare(&mut self, xs: &mut [F], k: usize) -> Result<(), Halt<O::Error>> {
        if self.limit - self.evaluations < k { return Err(Halt::Spent) }
        self.evaluations += k;
        if let Some(b) = self.bounds {
            for x in xs.chunks_exact_mut(self.best.len()) { b.clamp(x) }
        }
        Ok(())
    }

    /// `y` with NaN as +∞, after keeping `x` as the best point if `y` is the least value so far.
    fn record(&mut self, y: F, x: &[F]) -> F {
        let y = if y.is_nan() { F::infinity() } else { y };
        if y < self.least {
            self.least = y;
            self.best.copy_from_slice(x);
        }
        y
    }

    /// The value at `x`, after moving `x` inside the bounds.
    fn value(&mut self, x: &mut [F]) -> Result<F, Halt<O::Error>> {
        self.prepare(x, 1)?;
        let y = self.objective.value(x).map_err(Halt::Error)?;
        Ok(self.record(y, x))
    }

    /// The values at the rows of `xs`, into `out`, after moving each row inside the bounds.
    fn values(&mut self, xs: &mut [F], out: &mut [F]) -> Result<(), Halt<O::Error>> {
        self.prepare(xs, out.len())?;
        self.objective.values(xs, out).map_err(Halt::Error)?;
        for (y, x) in out.iter_mut().zip(xs.chunks_exact(self.best.len())) { *y = self.record(*y, x) }
        Ok(())
    }

    fn gradient(&mut self, x: &[F], g: &mut [F]) -> Result<(), Halt<O::Error>> { self.objective.gradient(x, g).map_err(Halt::Error) }

    fn found(self, converged: bool) -> Found<Vec<F>, F> {
        Found { x: self.best, value: self.least, evaluations: self.evaluations, converged }
    }
}

/// The largest magnitude in `x`.
fn largest<F: Native>(x: &[F]) -> F { x.iter().fold(F::zero(), |m, v| m.max(v.abs())) }

/// The first step a method takes in a variable at `x`: a tenth of its magnitude, and at least 0.1.
fn first_step<F: Native>(x: F) -> F { F::narrow(0.1) * x.abs().max(F::one()) }

/// Minimizes along `d` from `x`, where the value is `fx`, moving `x` to the minimum and returning its value. The first step is `d`
/// itself, and every step stays inside the bounds. `trial` is scratch of `x`'s length.
fn line<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, x: &mut [F], d: &[F], fx: F, tolerance: F, trial: &mut [F]) -> Result<F, Halt<O::Error>> {
    let (lo, hi) = search.bounds.map_or((F::neg_infinity(), F::infinity()), |b| b.along(x, d));
    let floor = tolerance * (largest(x) + tolerance) / largest(d);
    let (t, ft) = {
        let mut phi = |t: F| {
            trial.copy_from_slice(x);
            axpy(trial, t, d);
            search.value(trial)
        };
        least_along(&mut phi, fx, lo, hi, tolerance, floor)?
    };
    axpy(x, t, d);
    Ok(ft)
}

/// The least value of `phi` for t from `lo` to `hi`, which contain 0, where `phi` is `f0`. Steps that grow by the golden ratio go
/// downhill from 0, first by 1, until `phi` rises or a step reaches a limit. Brent's method then searches the bracket. Returns t and
/// `phi(t)`.
fn least_along<F: Native, H>(phi: &mut impl FnMut(F) -> Result<F, H>, f0: F, lo: F, hi: F, tolerance: F, floor: F) -> Result<(F, F), H> {
    let within = |t: F| t.max(lo).min(hi);
    let mut a = F::zero();
    let mut b = within(F::one());
    if b == a { b = within(-F::one()) }
    if b == a { return Ok((a, f0)) }
    let mut fb = phi(b)?;
    if fb > f0 { (a, b, fb) = (b, a, f0) }
    loop {
        let c = within(b + F::narrow(1.618_033_988_749_895) * (b - a));
        if c == b || !c.is_finite() { return Ok((b, fb)) }
        let fc = phi(c)?;
        if fc >= fb { return brent(phi, a.min(c), a.max(c), b, fb, tolerance, floor) }
        (a, b, fb) = (b, c, fc);
    }
}

/// Brent's minimization of `phi` between `a` and `b`, from `x` between them, where `phi` is `fx`: golden-section steps, and
/// parabolic steps where they fall inside the interval and shrink it fast enough. It stops when the interval around the best point is
/// within `tolerance` of that point, relative, plus `floor`.
#[allow(clippy::too_many_arguments)]
fn brent<F: Native, H>(phi: &mut impl FnMut(F) -> Result<F, H>, mut a: F, mut b: F, mut x: F, mut fx: F, tolerance: F, floor: F) -> Result<(F, F), H> {
    let (half, two, golden) = (F::narrow(0.5), F::narrow(2.0), F::narrow(0.381_966_011_250_105_1));
    let (mut v, mut w, mut fv, mut fw) = (x, x, fx, fx);
    let (mut d, mut e) = (F::zero(), F::zero());
    loop {
        let m = half * (a + b);
        let tol1 = tolerance * x.abs() + floor;
        let tol2 = two * tol1;
        if (x - m).abs() <= tol2 - half * (b - a) { return Ok((x, fx)) }
        let mut parabolic = false;
        if e.abs() > tol1 {
            let r = (x - w) * (fx - fv);
            let q = (x - v) * (fx - fw);
            let mut p = (x - v) * q - (x - w) * r;
            let mut q = two * (q - r);
            if q > F::zero() { p = -p } else { q = -q }
            if p.abs() < (half * q * e).abs() && p > q * (a - x) && p < q * (b - x) {
                e = d;
                d = p / q;
                if x + d - a < tol2 || b - (x + d) < tol2 { d = tol1.copysign(m - x) }
                parabolic = true;
            }
        }
        if !parabolic {
            e = if x >= m { a - x } else { b - x };
            d = golden * e;
        }
        let u = if d.abs() >= tol1 { x + d } else { x + tol1.copysign(d) };
        let fu = phi(u)?;
        if fu <= fx {
            if u >= x { a = x } else { b = x }
            (v, fv, w, fw, x, fx) = (w, fw, x, fx, u, fu);
        } else {
            if u < x { a = u } else { b = u }
            if fu <= fw || w == x { (v, fv, w, fw) = (w, fw, u, fu) }
            else if fu <= fv || v == x || v == w { (v, fv) = (u, fu) }
        }
    }
}

/// Brent's method on one variable: one line search from the start.
fn brent_method<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, tolerance: F) -> Result<bool, Halt<O::Error>> {
    let mut x = search.best.clone();
    let fx = search.value(&mut x)?;
    let d = [first_step(x[0])];
    line(search, &mut x, &d, fx, tolerance, &mut [F::zero()])?;
    Ok(true)
}

/// Nelder–Mead from the best point so far, restarting from its best point until a restart no longer lowers the value by more than
/// `tolerance`, relative.
fn nelder_mead<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, tolerance: F) -> Result<bool, Halt<O::Error>> {
    let mut x = search.best.clone();
    simplex(search, &x, tolerance)?;
    loop {
        let before = search.least;
        x.copy_from_slice(&search.best);
        simplex(search, &x, tolerance)?;
        if before - search.least <= tolerance * (before.abs() + tolerance) { return Ok(true) }
    }
}

/// One run of Nelder–Mead's simplex from `x`, until its values lie within `tolerance` of the best value and its points within
/// `tolerance` of the best point, relative. Every move puts the worst vertex at a point on the line from the centroid of the others
/// through it.
fn simplex<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, x: &[F], tolerance: F) -> Result<(), Halt<O::Error>> {
    let n = x.len();
    let (one, k) = (F::one(), F::narrow(n.max(2) as f64));
    let (expand, contract, shrink) = (one + F::narrow(2.0) / k, F::narrow(0.75) - one / (F::narrow(2.0) * k), one - one / k);
    let mut points = Vec::with_capacity((n + 1) * n);
    for i in 0..=n {
        points.extend_from_slice(x);
        if i > 0 {
            let (j, s) = (i - 1, first_step(x[i - 1]));
            let down = search.bounds.is_some_and(|b| x[j] + s > b.upper[j] && b.upper[j] - x[j] < x[j] - b.lower[j]);
            points[i * n + j] = if down { x[j] - s } else { x[j] + s };
        }
    }
    let mut values = vec![F::zero(); n + 1];
    search.values(&mut points, &mut values)?;
    let (mut centroid, mut trial, mut other) = (vec![F::zero(); n], vec![F::zero(); n], vec![F::zero(); n]);
    loop {
        let (b, s, w) = ranks(&values);
        let best = &points[b * n..][..n];
        let spread = (0..=n).map(|i| (0..n).fold(F::zero(), |m, j| m.max((points[i * n + j] - best[j]).abs()))).fold(F::zero(), F::max);
        if values[w] - values[b] <= tolerance * (values[b].abs() + tolerance) && spread <= tolerance * (largest(best) + tolerance) {
            return Ok(());
        }
        centroid.fill(F::zero());
        for i in (0..=n).filter(|&i| i != w) { axpy(&mut centroid, one / F::narrow(n as f64), &points[i * n..][..n]) }
        let worst = w * n..(w + 1) * n;
        let mut moved = |out: &mut [F], t: F| {
            for j in 0..n { out[j] = centroid[j] + t * (points[w * n + j] - centroid[j]) }
            search.value(out)
        };
        let fr = moved(&mut trial, -one)?;
        let (point, value) = if fr < values[b] {
            let fe = moved(&mut other, -expand)?;
            if fe < fr { (&other, fe) } else { (&trial, fr) }
        } else if fr < values[s] {
            (&trial, fr)
        } else {
            let fc = moved(&mut other, if fr < values[w] { -contract } else { contract })?;
            if fc >= fr.min(values[w]) {
                swap_rows(&mut points, b, 0, n);
                values.swap(b, 0);
                let (first, rest) = points.split_at_mut(n);
                for row in rest.chunks_exact_mut(n) {
                    for j in 0..n { row[j] = first[j] + shrink * (row[j] - first[j]) }
                }
                search.values(&mut points[n..], &mut values[1..])?;
                continue;
            }
            (&other, fc)
        };
        points[worst].copy_from_slice(point);
        values[w] = value;
    }
}

/// The positions of the least, second greatest and greatest of `values`, which has at least two.
fn ranks<F: Native>(values: &[F]) -> (usize, usize, usize) {
    let (mut b, mut w) = (0, 0);
    for (i, &v) in values.iter().enumerate() {
        if v < values[b] { b = i }
        if v > values[w] { w = i }
    }
    if b == w { w = (b + 1) % values.len() }
    let s = (0..values.len()).filter(|&i| i != w).fold(b, |s, i| if values[i] > values[s] { i } else { s });
    (b, s, w)
}

/// Swaps rows `i` and `j` of `n` items each.
fn swap_rows<F>(x: &mut [F], i: usize, j: usize, n: usize) {
    if i != j { for k in 0..n { x.swap(i * n + k, j * n + k) } }
}

/// Powell's conjugate directions from the best point so far: a line search along each direction in turn, then along the step they
/// made together where Powell's test accepts it, which replaces the direction of largest decrease. A replaced direction can point out
/// of an active bound, so a round along replaced directions that lowers the value by less than `tolerance`, relative, restarts from
/// the coordinate directions. Such a round along the coordinate directions ends the search.
fn powell<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, tolerance: F) -> Result<bool, Halt<O::Error>> {
    let n = search.best.len();
    let two = F::narrow(2.0);
    let mut x = search.best.clone();
    let mut fx = search.value(&mut x)?;
    let coordinates = |directions: &mut [F], x: &[F]| {
        directions.fill(F::zero());
        for i in 0..n { directions[i * n + i] = first_step(x[i]) }
    };
    let mut directions = vec![F::zero(); n * n];
    coordinates(&mut directions, &x);
    let mut coordinate = true;
    let (mut start, mut step, mut trial) = (x.clone(), vec![F::zero(); n], vec![F::zero(); n]);
    loop {
        start.copy_from_slice(&x);
        let f0 = fx;
        let (mut largest_drop, mut drop) = (0, F::zero());
        for i in 0..n {
            let before = fx;
            fx = line(search, &mut x, &directions[i * n..][..n], fx, tolerance, &mut trial)?;
            if before - fx > drop { (largest_drop, drop) = (i, before - fx) }
        }
        if two * (f0 - fx) <= tolerance * (f0.abs() + fx.abs() + tolerance) {
            if coordinate { return Ok(true) }
            coordinates(&mut directions, &x);
            coordinate = true;
            continue;
        }
        for j in 0..n { (step[j], trial[j]) = (x[j] - start[j], two * x[j] - start[j]) }
        let fe = search.value(&mut trial)?;
        let square = |v: F| v * v;
        if fe < f0 && two * (f0 - two * fx + fe) * square(f0 - fx - drop) < drop * square(f0 - fe) {
            fx = line(search, &mut x, &step, fx, tolerance, &mut trial)?;
            directions.copy_within((largest_drop + 1) * n.., largest_drop * n);
            directions[(n - 1) * n..].copy_from_slice(&step);
            coordinate = false;
        }
    }
}

/// L-BFGS from the best point so far. It stops when a step lowers the value by less than `tolerance`, relative, when the gradient is
/// zero, or when no step along steepest descent lowers the value.
fn lbfgs<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, tolerance: F) -> Result<bool, Halt<O::Error>> {
    const MEMORY: usize = 10;
    let n = search.best.len();
    let mut x = search.best.clone();
    let mut fx = search.value(&mut x)?;
    let mut g = vec![F::zero(); n];
    search.gradient(&x, &mut g)?;
    let mut history: VecDeque<(Vec<F>, Vec<F>, F)> = VecDeque::with_capacity(MEMORY);
    let (mut d, mut xn, mut gn, mut weights) = (vec![F::zero(); n], vec![F::zero(); n], vec![F::zero(); n], [F::zero(); MEMORY]);
    loop {
        if largest(&g) == F::zero() { return Ok(true) }
        d.copy_from_slice(&g);
        for (k, (s, y, rho)) in history.iter().enumerate().rev() {
            weights[k] = *rho * dot(s, &d);
            axpy(&mut d, -weights[k], y);
        }
        let scale = history.back().map_or_else(|| F::one() / largest(&g), |(s, y, _)| dot(s, y) / dot(y, y));
        d.iter_mut().for_each(|v| *v = -*v * scale);
        for (k, (s, y, rho)) in history.iter().enumerate() {
            let b = *rho * dot(y, &d);
            axpy(&mut d, -weights[k] - b, s);
        }
        let mut slope = dot(&g, &d);
        if slope >= F::zero() {
            history.clear();
            let scale = F::one() / largest(&g);
            for j in 0..n { d[j] = -g[j] * scale }
            slope = dot(&g, &d);
        }
        let Some(fnew) = wolfe(search, &x, fx, &d, slope, &mut xn, &mut gn)? else {
            if history.is_empty() { return Ok(true) }
            history.clear();
            continue;
        };
        let (mut s, mut y, _) = if history.len() == MEMORY { history.pop_front().unwrap() } else { (vec![F::zero(); n], vec![F::zero(); n], F::zero()) };
        for j in 0..n { (s[j], y[j]) = (xn[j] - x[j], gn[j] - g[j]) }
        let sy = dot(&s, &y);
        if sy > F::epsilon() * dot(&y, &y) { history.push_back((s, y, F::one() / sy)) }
        let decrease = fx - fnew;
        (x, xn, g, gn, fx) = (xn, x, gn, g, fnew);
        if decrease <= tolerance * (fx.abs() + tolerance) { return Ok(true) }
    }
}

/// A step α along `d` from `x`, where the value is `fx` and its slope along `d` is `slope`, that meets the strong Wolfe conditions:
/// it lowers the value by at least 10⁻⁴ α `slope`, and leaves at most 0.9 of the slope's magnitude. Nocedal and Wright's algorithms
/// 3.5 and 3.6 find it, with steps that double and then a zoom by safeguarded quadratic interpolation. The new value is returned,
/// with the point in `xn` and its gradient in `gn`. `None` means no step lowered the value.
#[allow(clippy::too_many_arguments)]
fn wolfe<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, x: &[F], fx: F, d: &[F], slope: F, xn: &mut [F], gn: &mut [F]) -> Result<Option<F>, Halt<O::Error>> {
    const STEPS: usize = 30;
    let (c1, c2, half, tenth) = (F::narrow(1e-4), F::narrow(0.9), F::narrow(0.5), F::narrow(0.1));
    let probe = |search: &mut Search<'_, F, O>, alpha: F, xn: &mut [F]| {
        xn.copy_from_slice(x);
        axpy(xn, alpha, d);
        search.value(xn)
    };
    let sufficient = |alpha: F, value: F| value <= fx + c1 * alpha * slope;
    let (mut lo, mut flo, mut slo) = (F::zero(), fx, slope);
    let mut bracket = None;
    let mut alpha = F::one();
    for i in 0..STEPS {
        let fa = probe(search, alpha, xn)?;
        if !sufficient(alpha, fa) || (i > 0 && fa >= flo) {
            bracket = Some((alpha, fa));
            break;
        }
        search.gradient(xn, gn)?;
        let sa = dot(gn, d);
        if sa.abs() <= -c2 * slope { return Ok(Some(fa)) }
        if sa >= F::zero() {
            bracket = Some((lo, flo));
            (lo, flo, slo) = (alpha, fa, sa);
            break;
        }
        (lo, flo, slo) = (alpha, fa, sa);
        alpha = alpha + alpha;
    }
    if let Some((mut hi, mut fhi)) = bracket {
        for _ in 0..STEPS {
            let width = hi - lo;
            if width.abs() <= F::epsilon() * lo.abs() { break }
            let curve = fhi - flo - slo * width;
            let guess = if curve > F::zero() { lo - slo * width * width / (curve + curve) } else { lo + half * width };
            let (p, q) = (lo + tenth * width, hi - tenth * width);
            let alpha = guess.max(p.min(q)).min(p.max(q));
            let fa = probe(search, alpha, xn)?;
            if !sufficient(alpha, fa) || fa >= flo {
                (hi, fhi) = (alpha, fa);
                continue;
            }
            search.gradient(xn, gn)?;
            let sa = dot(gn, d);
            if sa.abs() <= -c2 * slope { return Ok(Some(fa)) }
            if sa * width >= F::zero() { (hi, fhi) = (lo, flo) }
            (lo, flo, slo) = (alpha, fa, sa);
        }
    }
    if lo == F::zero() { return Ok(None) }
    let flo = probe(search, lo, xn)?;
    search.gradient(xn, gn)?;
    Ok(Some(flo))
}

/// Differential evolution within the bounds, then Powell's method from its best point. [`Method::Evolution`] describes it.
fn evolution<F: Native, O: Objective<F>>(search: &mut Search<'_, F, O>, population: usize, tolerance: F, seed: u64) -> Result<bool, Halt<O::Error>> {
    const SETTLED: f64 = 0.01;
    let Some(Bounds { lower, upper }) = search.bounds else { unreachable!() };
    let n = search.best.len();
    let mut rng = SplitMix64(seed);
    let mut uniform = move || F::narrow(rng.random::<f64>());
    let mut members = Vec::with_capacity(population * n);
    members.extend_from_slice(&search.best);
    for _ in 1..population { for j in 0..n { members.push(lower[j] + (upper[j] - lower[j]) * uniform()) } }
    let mut values = vec![F::zero(); population];
    search.values(&mut members, &mut values)?;
    let (mut trials, mut tried) = (members.clone(), values.clone());
    let (half, settled) = (F::narrow(0.5), F::narrow(SETTLED));
    let pick = |u: F, below: usize| ((u * F::narrow(below as f64)).to_usize().unwrap_or(0)).min(below - 1);
    loop {
        let (least, most) = values.iter().fold((F::infinity(), F::neg_infinity()), |(l, m), &v| (l.min(v), m.max(v)));
        let gathered = (0..n).all(|j| {
            let (l, m) = (0..population).fold((F::infinity(), F::neg_infinity()), |(l, m), i| (l.min(members[i * n + j]), m.max(members[i * n + j])));
            m - l <= settled * (upper[j] - lower[j])
        });
        if most - least <= settled * least.abs() || gathered { break }
        let factor = half + half * uniform();
        for i in 0..population {
            let mut chosen = [i; 4];
            for c in 1..4 {
                chosen[c] = loop {
                    let r = pick(uniform(), population);
                    if !chosen[..c].contains(&r) { break r }
                };
            }
            let [_, p, q, r] = chosen;
            let crossed = pick(uniform(), n);
            for j in 0..n {
                let parent = members[i * n + j];
                trials[i * n + j] = if j == crossed || uniform() < F::narrow(0.7) {
                    let v = members[p * n + j] + factor * (members[q * n + j] - members[r * n + j]);
                    if v < lower[j] { half * (parent + lower[j]) } else if v > upper[j] { half * (parent + upper[j]) } else { v }
                } else {
                    parent
                };
            }
        }
        search.values(&mut trials, &mut tried)?;
        for i in 0..population {
            if tried[i] <= values[i] {
                values[i] = tried[i];
                members[i * n..][..n].copy_from_slice(&trials[i * n..][..n]);
            }
        }
    }
    powell(search, tolerance)
}

/// A zero of `f` between the ends of `bracket`, where `f`'s signs differ, by Brent's method: inverse quadratic interpolation and
/// secant steps, with bisection wherever they don't shrink the bracket fast enough. It stops when the bracket is within `tolerance` of
/// its end nearer zero, relative, or within `tolerance`² near zero, with `None` for √ε of `F`. It also stops after `evaluations`
/// values, with `None` for 1000. Ends whose signs agree give the one nearer zero, with `converged` false.
pub fn root<F: Native, E>(mut f: impl FnMut(F) -> Result<F, E>, bracket: [F; 2], tolerance: Option<F>, evaluations: Option<usize>) -> Result<Found<F, F>, E> {
    let (tolerance, limit) = (tolerance.unwrap_or_else(|| F::epsilon().sqrt()), evaluations.unwrap_or(1000).max(2));
    let (half, two, three) = (F::narrow(0.5), F::narrow(2.0), F::narrow(3.0));
    let [mut a, mut b] = bracket;
    let (mut fa, mut fb) = (f(a)?, f(b)?);
    let mut count = 2;
    let found = |x, value, evaluations, converged| Found { x, value, evaluations, converged };
    if fa.abs() < fb.abs() { (a, b, fa, fb) = (b, a, fb, fa) }
    if fb == F::zero() { return Ok(found(b, fb, count, true)) }
    if (fa > F::zero()) == (fb > F::zero()) { return Ok(found(b, fb, count, false)) }
    let (mut c, mut fc) = (a, fa);
    let (mut d, mut e) = (b - a, b - a);
    loop {
        if (fb > F::zero()) == (fc > F::zero()) { (c, fc, d, e) = (a, fa, b - a, b - a) }
        if fc.abs() < fb.abs() { (a, b, c, fa, fb, fc) = (b, c, b, fb, fc, fb) }
        let tol1 = half * tolerance * (b.abs() + tolerance);
        let m = half * (c - b);
        if m.abs() <= tol1 || fb == F::zero() { return Ok(found(b, fb, count, true)) }
        if count == limit { return Ok(found(b, fb, count, false)) }
        if e.abs() >= tol1 && fa.abs() > fb.abs() {
            let s = fb / fa;
            let (mut p, mut q) = if a == c {
                (two * m * s, F::one() - s)
            } else {
                let (q, r) = (fa / fc, fb / fc);
                (s * (two * m * q * (q - r) - (b - a) * (r - F::one())), (q - F::one()) * (r - F::one()) * (s - F::one()))
            };
            if p > F::zero() { q = -q } else { p = -p }
            if two * p < (three * m * q - (tol1 * q).abs()).min((e * q).abs()) { (e, d) = (d, p / q) } else { (d, e) = (m, m) }
        } else {
            (d, e) = (m, m);
        }
        (a, fa) = (b, fb);
        b = b + if d.abs() > tol1 { d } else { tol1.copysign(m) };
        fb = f(b)?;
        count += 1;
    }
}
