//! Minimizing a function of several variables, and solving equations, nonlinear or linear.
//!
//! [`minimize`] runs one of five methods on an [`Objective`]. [`solve`] finds a zero of a [`Mapping`], or its least-squares solution
//! where it gives more values than it has variables: by Brent's method for one equation in one variable, and by Levenberg–Marquardt
//! otherwise. [`solve_linear`] solves A x = b by a Krylov method, where a [`Mapping`] gives A's products with vectors, and
//! [`solve_affine`] solves f(x) = 0 the same way for an affine `f`, through f's values alone. [`Options`] gives the method, the bounds
//! and when to stop. Where it gives no method, [`Method::choose`] picks one for the [`Task`].
//!
//! [`minimize`] and [`solve`] start from the start point moved inside the bounds, evaluate the function only inside them, and return
//! the best point they evaluated. A NaN value counts as +∞. An error from the function stops the search and comes back as the front's
//! error. A search that uses up its evaluations returns its best point with `converged` false.
//!
//! The Krylov methods use A only through its products. Conjugate gradients (CG) solves a symmetric positive definite system, MINRES a
//! symmetric one, and GMRES any square one. Least squares runs CG or MINRES on the normal equations AᵀA x = Aᵀb, with one product
//! with A and one with Aᵀ for each product with AᵀA. MINRES there gives the iterates of Fong and Saunders' LSMR, and CG those of CGLS,
//! which match Paige and Saunders' LSQR. Running on AᵀA squares A's condition number. On ill-conditioned problems that costs some
//! accuracy next to LSMR's and LSQR's own Golub–Kahan bidiagonalization, which works on A itself. A mapping's
//! [`Mapping::precondition`] gives a preconditioner, an approximation to the inverse of the matrix the method runs on, which cuts the
//! steps a solve needs. CG and MINRES apply it on each step, and GMRES on the right. CG and MINRES both build on the Lanczos process.
//! MINRES and GMRES both track the residual through plane rotations, the same ones the tridiagonal and bidiagonal iterations use.
use crate::{linalg::{axpy, dot, givens, lstsq, Factor}, random::SplitMix64, Native};
use rand::RngExt;
use std::collections::VecDeque;

/// A function of a slice of `F` to minimize. A closure that returns a `Result` is one. [`Differentiable`] adds a gradient.
pub trait Objective<F> {
    /// What stops an evaluation, such as an error in the caller's function or an interrupt.
    type Error;
    /// The value at `x`.
    fn value(&mut self, x: &[F]) -> Result<F, Self::Error>;
    /// Whether `values` evaluates a batch of points in one call. Nelder–Mead evaluates its first simplex and each shrink as a batch,
    /// and differential evolution each generation. Without `batch`, a search calls `value` on each point.
    fn batch(&self) -> bool { false }
    /// The value at each row of `xs`, into `out`, with `xs.len() / out.len()` items to a row. Methods call it only where `batch` is
    /// true.
    fn values(&mut self, xs: &[F], out: &mut [F]) -> Result<(), Self::Error> {
        let _ = (xs, out);
        unreachable!("this objective evaluates no batches")
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

/// A function from a slice of `F` to a slice of `F`, for [`solve`] and [`solve_linear`]. A closure that writes the values into its
/// second argument and returns a `Result` is one. [`Differentiable`] adds vector–Jacobian products.
pub trait Mapping<F> {
    /// What stops an evaluation, as for [`Objective`].
    type Error;
    /// The values at `x`, into `out`.
    fn value(&mut self, x: &[F], out: &mut [F]) -> Result<(), Self::Error>;
    /// Whether `vjp` gives vector–Jacobian products. Least squares in [`solve_linear`] needs them.
    fn has_vjp(&self) -> bool { false }
    /// uᵀJ, the product of `u` with the Jacobian at `x`, into `out`. For a linear mapping it's the product of the transpose with `u`,
    /// at any `x`. Methods call it only where `has_vjp` is true.
    fn vjp(&mut self, u: &[F], x: &[F], out: &mut [F]) -> Result<(), Self::Error> {
        let _ = (u, x, out);
        unreachable!("this mapping has no vector–Jacobian product")
    }
    /// The Jacobian at `x`, with a row of `x.len()` items for each value, into `j`, or `false` where Levenberg–Marquardt should take
    /// forward differences instead. By default, a mapping with `vjp` builds it one row per product when it has no more values than
    /// variables, which takes no more calls than differences do, and otherwise gives `false`.
    fn jacobian(&mut self, x: &[F], j: &mut [F]) -> Result<bool, Self::Error>
    where
        F: Native,
    {
        let n = x.len();
        if !self.has_vjp() || n == 0 || j.len() > n * n { return Ok(false) }
        let mut u = vec![F::zero(); j.len() / n];
        for (i, row) in j.chunks_exact_mut(n).enumerate() {
            u[i] = F::one();
            self.vjp(&u, x, row)?;
            u[i] = F::zero();
        }
        Ok(true)
    }
    /// Whether `precondition` gives a preconditioner's products. [`solve_linear`] runs without one otherwise.
    fn has_preconditioner(&self) -> bool { false }
    /// The product with `v` of a preconditioner for [`solve_linear`], into `out`: M, an approximation to the inverse of the matrix,
    /// or of AᵀA for least squares. CG and MINRES need M symmetric positive definite. Methods call it only where
    /// `has_preconditioner` is true.
    fn precondition(&mut self, v: &[F], out: &mut [F]) -> Result<(), Self::Error> {
        let _ = (v, out);
        unreachable!("this mapping has no preconditioner")
    }
}

impl<F, E, V: FnMut(&[F], &mut [F]) -> Result<(), E>> Mapping<F> for V {
    type Error = E;
    fn value(&mut self, x: &[F], out: &mut [F]) -> Result<(), E> { self(x, out) }
}

/// A function and its derivative, from two closures. As an [`Objective`], `value` gives the value and `derivative` writes the
/// gradient. As a [`Mapping`], `value` writes the values and `derivative` takes `(u, x, out)` and writes the vector–Jacobian product.
pub struct Differentiable<V, G> {
    pub value: V,
    pub derivative: G,
}

impl<F, E, V: FnMut(&[F]) -> Result<F, E>, G: FnMut(&[F], &mut [F]) -> Result<(), E>> Objective<F> for Differentiable<V, G> {
    type Error = E;
    fn value(&mut self, x: &[F]) -> Result<F, E> { (self.value)(x) }
    fn has_gradient(&self) -> bool { true }
    fn gradient(&mut self, x: &[F], g: &mut [F]) -> Result<(), E> { (self.derivative)(x, g) }
}

impl<F, E, V: FnMut(&[F], &mut [F]) -> Result<(), E>, G: FnMut(&[F], &[F], &mut [F]) -> Result<(), E>> Mapping<F> for Differentiable<V, G> {
    type Error = E;
    fn value(&mut self, x: &[F], out: &mut [F]) -> Result<(), E> { (self.value)(x, out) }
    fn has_vjp(&self) -> bool { true }
    fn vjp(&mut self, u: &[F], x: &[F], out: &mut [F]) -> Result<(), E> { (self.derivative)(u, x, out) }
}

/// The methods [`minimize`], [`solve`] and [`solve_linear`] run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Brent's method, for one variable. Minimizing, it brackets a minimum with growing steps from the start, which stop at a bound,
    /// then takes golden-section steps, and parabolic steps where they shrink the interval fast enough. Solving one equation, it
    /// takes finite bounds as its bracket. Otherwise it steps outward from the start, extending the end whose value is nearer zero by
    /// 1.6 times the bracket's width, until the values' signs differ. A step that lands where the value is NaN moves halfway back
    /// toward its end until the value is defined. A side stops stepping at a bound, or where it can't step to a defined value.
    /// Inverse quadratic interpolation and secant steps then shrink the bracket, with bisection wherever they don't shrink it fast
    /// enough.
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
    /// Levenberg–Marquardt, for a zero or least-squares solution of nonlinear equations. Each step solves the linear least-squares
    /// problem of the Jacobian, damped with Marquardt's scale for each variable, through `linalg::lstsq`. Nielsen's rule adjusts the
    /// damping after each step. The Jacobian comes from [`Mapping::jacobian`], or else from forward differences, at one evaluation
    /// for each variable. It stops when a step moves the point by less than `tolerance`, relative, or when the values are orthogonal
    /// to every column of the Jacobian within `tolerance`. It fails where the values are infinite or the Jacobian isn't finite. It
    /// takes no bounds.
    LevenbergMarquardt,
    /// Conjugate gradients, for a symmetric positive definite matrix. On the normal equations of least squares it gives the iterates
    /// of CGLS, which match Paige and Saunders' LSQR.
    Cg,
    /// MINRES, for a symmetric matrix, definite or not. On the normal equations of least squares it gives the iterates of Fong and
    /// Saunders' LSMR.
    Minres,
    /// GMRES, restarted every [`GMRES_RESTART`] steps, for any square matrix.
    Gmres,
}

impl Method {
    /// Each method's name, which `FromStr` reads, ignoring case, and `Display` writes.
    pub const NAMES: [(&'static str, Method); 9] = [
        ("brent", Method::Brent),
        ("nelder-mead", Method::NelderMead),
        ("powell", Method::Powell),
        ("l-bfgs", Method::Lbfgs),
        ("differential-evolution", Method::Evolution),
        ("levenberg-marquardt", Method::LevenbergMarquardt),
        ("cg", Method::Cg),
        ("minres", Method::Minres),
        ("gmres", Method::Gmres),
    ];

    /// The method a front runs for `task` on `variables` variables when [`Options::method`] is `None`. Minimizing, that's Brent's
    /// method for one variable, L-BFGS for an objective with a gradient and no bounds, Powell's method within bounds, and Nelder–Mead
    /// otherwise. Solving, it's Brent's method for one equation in one variable, and Levenberg–Marquardt otherwise. Solving linear
    /// equations, it's MINRES for least squares, and GMRES for a square system.
    pub fn choose(task: Task, variables: usize, bounded: bool) -> Method {
        match task {
            Task::Minimize { gradient } => {
                if variables == 1 { Method::Brent }
                else if gradient && !bounded { Method::Lbfgs }
                else if bounded { Method::Powell }
                else { Method::NelderMead }
            }
            Task::Solve { residuals } => if variables == 1 && residuals == 1 { Method::Brent } else { Method::LevenbergMarquardt },
            Task::Linear { .. } => if task.least_squares(variables) { Method::Minres } else { Method::Gmres },
        }
    }
}

impl std::str::FromStr for Method {
    type Err = String;
    fn from_str(name: &str) -> Result<Method, String> {
        let found = Method::NAMES.iter().find(|(key, _)| key.eq_ignore_ascii_case(name));
        found.map(|&(_, method)| method).ok_or_else(|| format!("unknown method: use one of {}", Method::NAMES.map(|(key, _)| key).join(", ")))
    }
}

impl std::fmt::Display for Method {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(Method::NAMES.iter().find(|&&(_, method)| method == *self).expect("every method has a name").0)
    }
}

/// What a front is for, which decides the methods that suit it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Task {
    /// [`minimize`], for an objective with or without a gradient.
    Minimize { gradient: bool },
    /// [`solve`], for a mapping that gives `residuals` values.
    Solve { residuals: usize },
    /// [`solve_linear`], for `equations` equations, with or without products with the matrix's transpose.
    Linear { equations: usize, adjoint: bool },
}

impl Task {
    /// Whether [`solve_linear`] solves least squares for `variables` unknowns: for a system that isn't square, or whose matrix gives
    /// products with its transpose.
    pub fn least_squares(&self, variables: usize) -> bool {
        matches!(*self, Task::Linear { equations, adjoint } if equations != variables || adjoint)
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

/// What a front runs and when it stops. `Options::default()` lets [`Method::choose`] pick the method and leaves every variable
/// unbounded.
#[derive(Clone, Copy, Debug)]
pub struct Options<'a, F> {
    /// The method, or `None` for the one [`Method::choose`] picks.
    pub method: Option<Method>,
    /// Limits on the variables. L-BFGS, Levenberg–Marquardt and the Krylov methods take none, and differential evolution needs
    /// finite ones. For Brent's method solving an equation, finite limits are the bracket.
    pub bounds: Option<Bounds<'a, F>>,
    /// The relative change at which a method stops, or `None` for √ε of `F`. A Krylov method stops when its residual's norm falls to
    /// `tolerance` times its right-hand side's.
    pub tolerance: Option<F>,
    /// The most evaluations of the function, or `None` for 1000 for each variable, plus 1000, and for differential evolution another
    /// 1000 for each member of its population. For [`solve_linear`], the most products with the matrix and its transpose, or `None`
    /// for 20 for each unknown, plus 20.
    pub evaluations: Option<usize>,
    /// Differential evolution's population, or `None` for 15 for each variable, and at least 5.
    pub population: Option<usize>,
    /// The seed of differential evolution's generator.
    pub seed: u64,
}

impl<F> Default for Options<'_, F> {
    fn default() -> Self { Options { method: None, bounds: None, tolerance: None, evaluations: None, population: None, seed: 0 } }
}

impl<F: Native> Options<'_, F> {
    /// The method a front runs for `task` on `variables` variables: `method`, or the one [`Method::choose`] picks.
    pub fn method_for(&self, task: Task, variables: usize) -> Method {
        self.method.unwrap_or_else(|| Method::choose(task, variables, self.bounds.is_some()))
    }

    /// Why a front would reject these options for `task` on `variables` variables. A binding calls it first and reports the message
    /// as an error.
    pub fn check(&self, task: Task, variables: usize) -> Result<(), &'static str> {
        use Method::*;
        let bounds = self.bounds;
        if bounds.is_some_and(|b| b.lower.len() != variables || b.upper.len() != variables) {
            return Err("bounds need a lower and an upper limit for each variable");
        }
        if bounds.is_some_and(|b| b.lower.iter().zip(b.upper).any(|(l, u)| !(l <= u))) {
            return Err("each lower limit must be at most its upper limit");
        }
        if self.tolerance.is_some_and(|t| !(t > F::zero() && t.is_finite())) { return Err("the tolerance must be a positive number") }
        let method = self.method_for(task, variables);
        match task {
            Task::Minimize { gradient } => match method {
                Brent if variables > 1 => Err("Brent's method takes one variable"),
                Lbfgs if !gradient => Err("L-BFGS needs an objective with a gradient"),
                Lbfgs if bounds.is_some() => Err("L-BFGS takes no bounds"),
                Evolution if !bounds.is_some_and(|b| b.lower.iter().chain(b.upper).all(|v| v.is_finite())) => {
                    Err("differential evolution needs finite bounds")
                }
                Evolution if self.population.is_some_and(|p| p < 4) => Err("differential evolution needs a population of at least 4"),
                Brent | NelderMead | Powell | Lbfgs | Evolution => Ok(()),
                _ => Err("that method solves equations, and doesn't minimize"),
            },
            Task::Solve { residuals } => match method {
                Brent if variables != 1 || residuals != 1 => Err("Brent's method solves one equation in one variable"),
                LevenbergMarquardt if bounds.is_some() => Err("Levenberg–Marquardt takes no bounds"),
                Brent | LevenbergMarquardt => Ok(()),
                _ => Err("that method doesn't solve nonlinear equations"),
            },
            Task::Linear { equations, adjoint } => match method {
                Cg | Minres | Gmres if bounds.is_some() => Err("linear solving takes no bounds"),
                Cg | Minres | Gmres if equations != variables && !adjoint => Err("least squares needs products with the transpose"),
                Cg | Minres | Gmres => Ok(()),
                _ => Err("that method doesn't solve linear equations"),
            },
        }
    }

    /// The method and the tolerance a front runs with for `task` on `variables` variables. It panics where [`Options::check`] rejects
    /// the options.
    fn settings(&self, task: Task, variables: usize) -> (Method, F) {
        if let Err(problem) = self.check(task, variables) { panic!("{problem}") }
        (self.method_for(task, variables), self.tolerance.unwrap_or_else(|| F::epsilon().sqrt()))
    }
}

/// What a front found.
#[derive(Clone, Debug, PartialEq)]
pub struct Found<X, F> {
    /// The best point evaluated. For [`solve_linear`], the last iterate.
    pub x: X,
    /// The function's value at `x`. For [`solve`] and [`solve_linear`], the norm of the residuals there.
    pub value: F,
    /// How many times the function was evaluated. For [`solve_linear`], how many products with the matrix and its transpose,
    /// including one for the residual at `x`.
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
    let (n, task) = (start.len(), Task::Minimize { gradient: objective.has_gradient() });
    let (method, tolerance) = options.settings(task, n);
    let population = options.population.unwrap_or((15 * n).max(5));
    let limit = options.evaluations.unwrap_or(1000 * (n + 1) + if method == Method::Evolution { 1000 * population } else { 0 });
    let mut search = Search::new(objective, start, options.bounds, limit);
    let outcome = match method {
        _ if n == 0 => search.value(&mut []).map(|_| true),
        Method::Brent => brent_method(&mut search, tolerance),
        Method::NelderMead => nelder_mead(&mut search, tolerance),
        Method::Powell => powell(&mut search, tolerance),
        Method::Lbfgs => lbfgs(&mut search, tolerance),
        Method::Evolution => evolution(&mut search, population, tolerance, options.seed),
        _ => unreachable!("`check` accepts only methods that minimize"),
    };
    passed(outcome).map(|converged| search.found(converged))
}

/// A zero of `mapping` from `start`, by the method [`Options::method_for`] gives. `first` holds `mapping`'s values at `start`, which
/// the caller computes. They give the number of values, and count as the search's first evaluation. Where `mapping` gives more values
/// than it has variables, the result is the least-squares solution, and where it gives fewer, a nearby solution.
///
/// # Panics
///
/// If [`Options::check`] rejects the options.
pub fn solve<F: Native + Factor, M: Mapping<F>>(mapping: &mut M, start: &[F], first: &[F], options: &Options<'_, F>) -> Result<Found<Vec<F>, F>, M::Error> {
    let (n, task) = (start.len(), Task::Solve { residuals: first.len() });
    let (method, tolerance) = options.settings(task, n);
    let mut residual = Norm { mapping, r: first.to_vec() };
    let mut search = Search::new(&mut residual, start, options.bounds, options.evaluations.unwrap_or(1000 * (n + 1)));
    search.seed(start, norm(first));
    let outcome = match method {
        _ if n == 0 => Ok(true),
        Method::Brent => root(&mut search, (start[0], first[0]), tolerance),
        _ => levenberg_marquardt(&mut search, tolerance),
    };
    passed(outcome).map(|converged| search.found(converged))
}

/// The solution of A x = `b` from `start`, by the method [`Options::method_for`] gives. `a` gives A's products with vectors, and
/// `b` has one item for each equation. A system that isn't square, or whose `a` gives products with A's transpose through
/// [`Mapping::vjp`], gets its least-squares solution, from the normal equations AᵀA x = Aᵀb. A right-hand side of zeros gives zeros.
/// `a`'s [`Mapping::precondition`] gives the products of a preconditioner M, which approximates A⁻¹, or (AᵀA)⁻¹ for least squares.
/// CG and MINRES need M symmetric positive definite, and GMRES applies it on the right. A good M cuts the products a solve needs.
///
/// # Panics
///
/// If [`Options::check`] rejects the options.
pub fn solve_linear<F: Native, M: Mapping<F>>(a: &mut M, b: &[F], start: &[F], options: &Options<'_, F>) -> Result<Found<Vec<F>, F>, M::Error> {
    let (m, n) = (b.len(), start.len());
    let task = Task::Linear { equations: m, adjoint: a.has_vjp() };
    let (method, tolerance) = options.settings(task, n);
    let limit = options.evaluations.unwrap_or(20 * (n + 1));
    let mut op = Products { a, at: start, normal: task.least_squares(n), inner: vec![F::zero(); m], products: 0, limit };
    let mut x = start.to_vec();
    let outcome = op.rhs(b).and_then(|rhs| {
        if norm(&rhs) == F::zero() {
            x.fill(F::zero());
            return Ok(true);
        }
        match method {
            Method::Cg => cg(&mut op, &rhs, &mut x, tolerance),
            Method::Minres => minres(&mut op, &rhs, &mut x, tolerance),
            _ => gmres(&mut op, &rhs, &mut x, tolerance),
        }
    });
    let converged = passed(outcome)?;
    op.a.value(&x, &mut op.inner)?;
    for (r, &v) in op.inner.iter_mut().zip(b) { *r = v - *r }
    Ok(Found { value: norm(&op.inner), x, evaluations: op.products + 1, converged })
}

/// The solution of f(x) = 0 for an affine `f`, f(x) = A x − b, from `start`, by [`solve_linear`] on products that it computes from
/// `f`'s values alone. `first` holds f(`start`), which the caller computes. It gives the number of residuals, and serves as f(0) where
/// `start` is zero. f(0) gives −b. Each product A v is (f(t v) − f(0)) ÷ t, for the power of two t that makes t |A v| about 16 |b|.
/// The subtraction of f(0) then loses no digits, however large b is. `f`'s `vjp` gives the transpose's products, and its
/// `precondition` the preconditioner's. A product evaluates `f` once, or twice where the ratio |A v| ÷ |v| from the last product gave
/// too small a t. `evaluations` counts products, as for `solve_linear`.
///
/// # Panics
///
/// If [`Options::check`] rejects the options.
pub fn solve_affine<F: Native, M: Mapping<F>>(f: &mut M, start: &[F], first: &[F], options: &Options<'_, F>) -> Result<Found<Vec<F>, F>, M::Error> {
    let f0 = if start.iter().all(|&x| x == F::zero()) { first.to_vec() } else {
        let mut f0 = vec![F::zero(); first.len()];
        f.value(&vec![F::zero(); start.len()], &mut f0)?;
        f0
    };
    let b: Vec<F> = f0.iter().map(|&v| -v).collect();
    let mut a = Affine { f, size: norm(&b), f0, ratio: None, scaled: Vec::with_capacity(start.len()) };
    solve_linear(&mut a, &b, start, options)
}

/// Why a method stopped before its stopping test passed: the function's error, or the evaluations ran out.
enum Halt<E> {
    Error(E),
    Spent,
}

/// Whether a method's stopping test passed, from how it ended: `false` where the evaluations ran out, or the error that stopped it.
fn passed<E>(outcome: Result<bool, Halt<E>>) -> Result<bool, E> {
    match outcome {
        Ok(converged) => Ok(converged),
        Err(Halt::Spent) => Ok(false),
        Err(Halt::Error(e)) => Err(e),
    }
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
        if self.evaluations + k > self.limit { return Err(Halt::Spent) }
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

    /// Counts the caller's evaluation at `start`, which gave `y`, and keeps `start` as the best point where it lies within the bounds.
    fn seed(&mut self, start: &[F], y: F) {
        self.evaluations += 1;
        if start == self.best { self.record(y, start); }
    }

    /// The value at `x`, after moving `x` inside the bounds.
    fn value(&mut self, x: &mut [F]) -> Result<F, Halt<O::Error>> {
        self.prepare(x, 1)?;
        let y = self.objective.value(x).map_err(Halt::Error)?;
        Ok(self.record(y, x))
    }

    /// The values at the rows of `xs`, into `out`, after moving each row inside the bounds: through one call of `values` where the
    /// objective evaluates batches, and otherwise through `value` on each row.
    fn values(&mut self, xs: &mut [F], out: &mut [F]) -> Result<(), Halt<O::Error>> {
        self.prepare(xs, out.len())?;
        let n = self.best.len();
        if self.objective.batch() { self.objective.values(xs, out).map_err(Halt::Error)? }
        else { for (y, x) in out.iter_mut().zip(xs.chunks_exact(n)) { *y = self.objective.value(x).map_err(Halt::Error)? } }
        for (y, x) in out.iter_mut().zip(xs.chunks_exact(n)) { *y = self.record(*y, x) }
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

/// A [`Mapping`] as an [`Objective`] whose value is the norm of the mapping's values, which it keeps in `r`. [`solve`]'s methods run
/// on a [`Search`] of it.
struct Norm<'m, F, M> {
    mapping: &'m mut M,
    r: Vec<F>,
}

impl<F: Native, M: Mapping<F>> Objective<F> for Norm<'_, F, M> {
    type Error = M::Error;
    fn value(&mut self, x: &[F]) -> Result<F, M::Error> {
        self.mapping.value(x, &mut self.r)?;
        Ok(norm(&self.r))
    }
}

/// The Euclidean norm of `x`, with no scaling.
fn norm<F: Native>(x: &[F]) -> F { dot(x, x).sqrt() }

/// The point `x`, moved inside the bounds, and the value there of a mapping of one variable to one value.
fn at<F: Native, M: Mapping<F>>(search: &mut Search<'_, F, Norm<'_, F, M>>, x: F) -> Result<(F, F), Halt<M::Error>> {
    let mut p = [x];
    search.value(&mut p)?;
    Ok((p[0], search.objective.r[0]))
}

/// Brent's method for a zero of one equation in one variable, which [`Method::Brent`] describes, from `start`, a point and its value.
/// It stops when the bracket is within `tolerance` of its end nearer zero, relative, or within `tolerance`² near zero. It fails where
/// it finds no change of sign.
fn root<F: Native, M: Mapping<F>>(search: &mut Search<'_, F, Norm<'_, F, M>>, start: (F, F), tolerance: F) -> Result<bool, Halt<M::Error>> {
    let (half, two, three, grow) = (F::narrow(0.5), F::narrow(2.0), F::narrow(3.0), F::narrow(1.6));
    let differ = |fa: F, fb: F| (fa > F::zero()) != (fb > F::zero());
    let ends = search.bounds.map(|b| (b.lower[0], b.upper[0])).filter(|(l, u)| l.is_finite() && u.is_finite());
    // `start` holds the caller's evaluation, which `solve` has counted. A point that equals it needs no evaluation of its own.
    let ((mut a, mut fa), (mut b, mut fb)) = match ends {
        Some((l, u)) => (if l == start.0 { start } else { at(search, l)? }, if u == start.0 { start } else { at(search, u)? }),
        None => {
            let from = if search.best[0] == start.0 { start } else { at(search, search.best[0])? };
            (from, from)
        }
    };
    // Whether each end can still step outward.
    let (mut low, mut high) = (ends.is_none(), ends.is_none());
    while fa != F::zero() && fb != F::zero() && !differ(fa, fb) {
        if !low && !high { return Ok(false) }
        let down = low && (!high || fa.abs() < fb.abs());
        let width = if b > a { b - a } else { first_step(a) };
        let end = if down { a } else { b };
        let mut target = if down { a - grow * width } else { b + grow * width };
        let (x, fx) = loop {
            let (x, fx) = at(search, target)?;
            let inward = end + half * (target - end);
            if (!fx.is_nan() && x.is_finite()) || inward == end || inward == target { break (x, fx) }
            target = inward;
        };
        let undefined = fx.is_nan() || !x.is_finite();
        if !undefined { if down { (a, fa) = (x, fx) } else { (b, fb) = (x, fx) } }
        if undefined || x != target { if down { low = false } else { high = false } }
    }
    if fa == F::zero() || fb == F::zero() { return Ok(true) }
    if fa.abs() < fb.abs() { (a, b, fa, fb) = (b, a, fb, fa) }
    let (mut c, mut fc) = (a, fa);
    let (mut d, mut e) = (b - a, b - a);
    loop {
        if !differ(fb, fc) { (c, fc, d, e) = (a, fa, b - a, b - a) }
        if fc.abs() < fb.abs() { (a, b, c, fa, fb, fc) = (b, c, b, fb, fc, fb) }
        let tol1 = half * tolerance * (b.abs() + tolerance);
        let m = half * (c - b);
        if m.abs() <= tol1 || fb == F::zero() { return Ok(true) }
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
        fb = at(search, b)?.1;
    }
}

/// Levenberg–Marquardt from the start, whose values `solve` has recorded, which [`Method::LevenbergMarquardt`] describes.
fn levenberg_marquardt<F: Native + Factor, M: Mapping<F>>(search: &mut Search<'_, F, Norm<'_, F, M>>, tolerance: F) -> Result<bool, Halt<M::Error>> {
    let (n, m) = (search.best.len(), search.objective.r.len());
    let (one, two) = (F::one(), F::narrow(2.0));
    let mut x = search.best.clone();
    let mut fx = search.least;
    let mut r = search.objective.r.clone();
    // The Jacobian, and the damped problem [J; √λ D] δ = [-r; 0] that each step solves.
    let (mut j, mut damped, mut target) = (vec![F::zero(); m * n], vec![F::zero(); (m + n) * n], vec![F::zero(); m + n]);
    let (mut scale, mut trial, mut fit) = (vec![F::zero(); n], vec![F::zero(); n], vec![F::zero(); m]);
    // D holds each column's largest norm so far, so λ is relative to JᵀJ's diagonal and starts at Marquardt's 10⁻³.
    let (mut damping, mut growth) = (F::narrow(1e-3), two);
    loop {
        if fx == F::zero() { return Ok(true) }
        jacobian(search, &x, &r, &mut j)?;
        let mut cosine = F::zero();
        for c in 0..n {
            let (mut length, mut product) = (F::zero(), F::zero());
            for i in 0..m { (length, product) = (length + j[i * n + c] * j[i * n + c], product + j[i * n + c] * r[i]) }
            let length = length.sqrt();
            // Infinite residuals, or a Jacobian with NaN or infinite items, leave no step to take and no gradient to test.
            if !(length.is_finite() && product.is_finite()) { return Ok(false) }
            if length > F::zero() { cosine = cosine.max(product.abs() / (length * fx)) }
            scale[c] = scale[c].max(length);
        }
        if cosine <= tolerance { return Ok(true) }
        loop {
            damped[..m * n].copy_from_slice(&j);
            damped[m * n..].fill(F::zero());
            for c in 0..n { damped[(m + c) * n + c] = damping.sqrt() * scale[c] }
            for i in 0..m { target[i] = -r[i] }
            target[m..].fill(F::zero());
            let (step, _) = lstsq(&damped, &target, [m + n, n], 1);
            for c in 0..n { trial[c] = x[c] + step[c] }
            let ft = search.value(&mut trial)?;
            let small = norm(&step) <= tolerance * (norm(&x) + tolerance);
            if ft < fx {
                for i in 0..m { fit[i] = r[i] + dot(&j[i * n..][..n], &step) }
                let predicted = fx * fx - dot(&fit, &fit);
                let ratio = if predicted > F::zero() { (fx * fx - ft * ft) / predicted } else { one };
                let swing = two * ratio - one;
                damping = damping * F::narrow(1.0 / 3.0).max(one - swing * swing * swing);
                growth = two;
                x.copy_from_slice(&trial);
                r.copy_from_slice(&search.objective.r);
                fx = ft;
                if small { return Ok(true) }
                break;
            }
            if small { return Ok(true) }
            damping = damping * growth;
            growth = growth * two;
        }
    }
}

/// The Jacobian at `x`, where the values are `r`, into `j`: the mapping's own, or forward differences, which take one evaluation for
/// each variable.
fn jacobian<F: Native, M: Mapping<F>>(search: &mut Search<'_, F, Norm<'_, F, M>>, x: &[F], r: &[F], j: &mut [F]) -> Result<(), Halt<M::Error>> {
    if search.objective.mapping.jacobian(x, j).map_err(Halt::Error)? { return Ok(()) }
    let n = x.len();
    let mut shifted = x.to_vec();
    for c in 0..n {
        shifted[c] = x[c] + F::epsilon().sqrt() * x[c].abs().max(F::one());
        let h = shifted[c] - x[c];
        search.value(&mut shifted)?;
        for (i, (&after, &before)) in search.objective.r.iter().zip(r).enumerate() { j[i * n + c] = (after - before) / h }
        shifted[c] = x[c];
    }
    Ok(())
}

/// The products A v of an affine `f`, f(x) = A x − b, from `f`'s values, as [`solve_affine`] computes them. It passes on `f`'s
/// vector–Jacobian products and preconditioner.
struct Affine<'m, F, M> {
    f: &'m mut M,
    /// f(0), which is −b.
    f0: Vec<F>,
    /// The norm of b.
    size: F,
    /// |A v| ÷ |v| at the last product, which gives the next t.
    ratio: Option<F>,
    /// t v, the point at which `f` is evaluated.
    scaled: Vec<F>,
}

impl<F: Native, M: Mapping<F>> Affine<'_, F, M> {
    /// The power of two t that makes t |A v| about 16 |b|, for a `v` of norm `nv` where |A v| ÷ |v| is `ratio`, or 1 where b is zero.
    fn scale(&self, nv: F, ratio: F) -> F {
        if self.size == F::zero() { return F::one() }
        let t = (F::narrow(16.0) * self.size / (ratio * nv)).log2().round().exp2();
        if t.is_finite() && t > F::zero() { t } else { F::one() }
    }

    /// (f(t v) − f(0)) ÷ t, into `out`.
    fn at(&mut self, v: &[F], t: F, out: &mut [F]) -> Result<(), M::Error> {
        self.scaled.clear();
        self.scaled.extend(v.iter().map(|&x| x * t));
        self.f.value(&self.scaled, out)?;
        for (o, &z) in out.iter_mut().zip(&self.f0) { *o = (*o - z) / t }
        Ok(())
    }
}

impl<F: Native, M: Mapping<F>> Mapping<F> for Affine<'_, F, M> {
    type Error = M::Error;
    /// A v, computed again with a corrected t where the t from the last product's ratio made t |A v| smaller than |b|.
    fn value(&mut self, v: &[F], out: &mut [F]) -> Result<(), M::Error> {
        let nv = norm(v);
        if nv == F::zero() {
            out.fill(F::zero());
            return Ok(());
        }
        let t = self.scale(nv, self.ratio.unwrap_or(F::one()));
        self.at(v, t, out)?;
        let mut av = norm(out);
        if av > F::zero() && t * av < self.size {
            self.at(v, self.scale(nv, av / nv), out)?;
            av = norm(out);
        }
        if av > F::zero() { self.ratio = Some(av / nv) }
        Ok(())
    }
    fn has_vjp(&self) -> bool { self.f.has_vjp() }
    fn vjp(&mut self, u: &[F], x: &[F], out: &mut [F]) -> Result<(), M::Error> { self.f.vjp(u, x, out) }
    fn has_preconditioner(&self) -> bool { self.f.has_preconditioner() }
    fn precondition(&mut self, v: &[F], out: &mut [F]) -> Result<(), M::Error> { self.f.precondition(v, out) }
}

/// The operator a Krylov method runs on, the matrix or AᵀA for least squares, with each product counted against the limit.
struct Products<'a, F, M> {
    a: &'a mut M,
    /// The point at which `vjp` gives the transpose's products.
    at: &'a [F],
    normal: bool,
    /// The matrix's product inside a product with AᵀA.
    inner: Vec<F>,
    products: usize,
    limit: usize,
}

impl<F: Native, M: Mapping<F>> Products<'_, F, M> {
    /// Counts `k` products against the limit.
    fn count(&mut self, k: usize) -> Result<(), Halt<M::Error>> {
        if self.products + k > self.limit { return Err(Halt::Spent) }
        self.products += k;
        Ok(())
    }

    /// The operator's product with `v`, into `out`.
    fn apply(&mut self, v: &[F], out: &mut [F]) -> Result<(), Halt<M::Error>> {
        if !self.normal {
            self.count(1)?;
            return self.a.value(v, out).map_err(Halt::Error);
        }
        self.count(2)?;
        self.a.value(v, &mut self.inner).map_err(Halt::Error)?;
        self.a.vjp(&self.inner, self.at, out).map_err(Halt::Error)
    }

    /// M·v, the preconditioner's product with `v`, written into `out`, or `v` itself where the mapping has no preconditioner. It
    /// counts no product.
    fn precondition<'v>(&mut self, v: &'v [F], out: &'v mut [F]) -> Result<&'v [F], Halt<M::Error>> {
        if !self.a.has_preconditioner() { return Ok(v) }
        self.a.precondition(v, out).map_err(Halt::Error)?;
        Ok(out)
    }

    /// The right-hand side of the system the method runs on: `b`, or Aᵀb for least squares.
    fn rhs(&mut self, b: &[F]) -> Result<Vec<F>, Halt<M::Error>> {
        if !self.normal { return Ok(b.to_vec()) }
        self.count(1)?;
        let mut v = vec![F::zero(); self.at.len()];
        self.a.vjp(b, self.at, &mut v).map_err(Halt::Error)?;
        Ok(v)
    }

    /// `rhs` minus the operator's product with `x`, into `r`.
    fn residual(&mut self, rhs: &[F], x: &[F], r: &mut [F]) -> Result<(), Halt<M::Error>> {
        self.apply(x, r)?;
        for (r, &b) in r.iter_mut().zip(rhs) { *r = b - *r }
        Ok(())
    }
}

/// Conjugate gradients on `op` from `x`, with `op`'s preconditioner. It stops when the residual's norm falls to `tolerance` times
/// `rhs`'s, or where the operator or the preconditioner turns out not to be positive definite.
fn cg<F: Native, M: Mapping<F>>(op: &mut Products<'_, F, M>, rhs: &[F], x: &mut [F], tolerance: F) -> Result<bool, Halt<M::Error>> {
    let (n, zero, goal) = (x.len(), F::zero(), tolerance * norm(rhs));
    let (mut r, mut z, mut q) = (vec![zero; n], vec![zero; n], vec![zero; n]);
    op.residual(rhs, x, &mut r)?;
    let mut p = op.precondition(&r, &mut z)?.to_vec();
    let mut rz = dot(&r, &p);
    loop {
        let size = norm(&r);
        if !(size > goal) { return Ok(size <= goal) }
        if !(rz > zero) { return Ok(false) }
        op.apply(&p, &mut q)?;
        let pq = dot(&p, &q);
        if !(pq > zero) { return Ok(false) }
        let alpha = rz / pq;
        axpy(x, alpha, &p);
        axpy(&mut r, -alpha, &q);
        let mz = op.precondition(&r, &mut z)?;
        let next = dot(&r, mz);
        let beta = next / rz;
        rz = next;
        for i in 0..n { p[i] = mz[i] + beta * p[i] }
    }
}

/// MINRES on the symmetric `op` from `x`, with `op`'s preconditioner, as Paige and Saunders give it: the Lanczos process, with plane
/// rotations that track the residual's norm. That norm is the preconditioner's, √(rᵀMr), which is the residual's own without one.
/// It stops when that norm falls to `tolerance` times `rhs`'s in the same norm.
fn minres<F: Native, M: Mapping<F>>(op: &mut Products<'_, F, M>, rhs: &[F], x: &mut [F], tolerance: F) -> Result<bool, Halt<M::Error>> {
    let (n, zero) = (x.len(), F::zero());
    let (mut r1, mut y, mut v) = (vec![zero; n], vec![zero; n], vec![zero; n]);
    let goal = tolerance * dot(rhs, op.precondition(rhs, &mut v)?).sqrt();
    op.residual(rhs, x, &mut r1)?;
    let my = op.precondition(&r1, &mut y)?;
    let mut beta = dot(&r1, my).sqrt();
    if !(beta > goal) { return Ok(beta <= goal) }
    for i in 0..n { v[i] = my[i] / beta }
    let mut r2 = r1.clone();
    let (mut w, mut w1, mut w2) = (vec![zero; n], vec![zero; n], vec![zero; n]);
    let (mut oldb, mut dbar, mut epsln, mut phibar, mut cs, mut sn) = (zero, zero, zero, beta, -F::one(), zero);
    loop {
        op.apply(&v, &mut y)?;
        if oldb > zero { axpy(&mut y, -beta / oldb, &r1) }
        let alfa = dot(&v, &y);
        axpy(&mut y, -alfa / beta, &r2);
        std::mem::swap(&mut r1, &mut r2);
        r2.copy_from_slice(&y);
        let my = op.precondition(&r2, &mut y)?;
        (oldb, beta) = (beta, dot(&r2, my).sqrt());
        let oldeps = epsln;
        let delta = cs * dbar + sn * alfa;
        let gbar = sn * dbar - cs * alfa;
        (epsln, dbar) = (sn * beta, -cs * beta);
        let (c, s, gamma) = givens(gbar, beta);
        if gamma == zero { return Ok(false) }
        (cs, sn) = (c, s);
        let phi = cs * phibar;
        phibar = sn * phibar;
        std::mem::swap(&mut w1, &mut w2);
        std::mem::swap(&mut w2, &mut w);
        for i in 0..n { w[i] = (v[i] - oldeps * w1[i] - delta * w2[i]) / gamma }
        axpy(x, phi, &w);
        if !(phibar > goal) { return Ok(phibar <= goal) }
        for i in 0..n { v[i] = my[i] / beta }
    }
}

/// The most steps GMRES takes between restarts. A system with fewer unknowns restarts after one step for each unknown.
pub const GMRES_RESTART: usize = 20;

/// GMRES on `op` from `x`, restarted every [`GMRES_RESTART`] steps, with `op`'s preconditioner on the right: it runs on the operator
/// times the preconditioner, which leaves the residual the operator's own. That's the Arnoldi process with modified Gram–Schmidt,
/// and plane rotations that turn its Hessenberg matrix into a triangle and track the residual's norm. It stops when that norm falls
/// to `tolerance` times `rhs`'s, or where the Krylov space stops growing.
fn gmres<F: Native, M: Mapping<F>>(op: &mut Products<'_, F, M>, rhs: &[F], x: &mut [F], tolerance: F) -> Result<bool, Halt<M::Error>> {
    let (n, zero, goal) = (x.len(), F::zero(), tolerance * norm(rhs));
    let k = GMRES_RESTART.min(n);
    let (mut basis, mut h) = (vec![zero; (k + 1) * n], vec![zero; k * k]);
    let (mut turns, mut g, mut y) = (vec![(zero, zero); k], vec![zero; k + 1], vec![zero; k]);
    let (mut w, mut z) = (vec![zero; n], vec![zero; n]);
    loop {
        op.residual(rhs, x, &mut basis[..n])?;
        let beta = norm(&basis[..n]);
        if !(beta > goal) { return Ok(beta <= goal) }
        for v in &mut basis[..n] { *v = *v / beta }
        g.fill(zero);
        g[0] = beta;
        let (mut steps, mut ended) = (0, false);
        while steps < k && !ended {
            let j = steps;
            let mv = op.precondition(&basis[j * n..][..n], &mut z)?;
            op.apply(mv, &mut w)?;
            for i in 0..=j {
                let hij = dot(&w, &basis[i * n..][..n]);
                h[i * k + j] = hij;
                axpy(&mut w, -hij, &basis[i * n..][..n]);
            }
            let next = norm(&w);
            for (i, &(c, s)) in turns[..j].iter().enumerate() {
                let (p, q) = (h[i * k + j], h[(i + 1) * k + j]);
                (h[i * k + j], h[(i + 1) * k + j]) = (c * p + s * q, c * q - s * p);
            }
            let (c, s, r) = givens(h[j * k + j], next);
            if r == zero { break }
            (turns[j], h[j * k + j]) = ((c, s), r);
            (g[j], g[j + 1]) = (c * g[j], -s * g[j]);
            steps += 1;
            ended = !(g[steps].abs() > goal) || next == zero;
            if !ended { for i in 0..n { basis[steps * n + i] = w[i] / next } }
        }
        for i in (0..steps).rev() {
            let t = (i + 1..steps).fold(g[i], |t, l| t - h[i * k + l] * y[l]);
            y[i] = t / h[i * k + i];
        }
        z.fill(zero);
        for i in 0..steps { axpy(&mut z, y[i], &basis[i * n..][..n]) }
        let step = op.precondition(&z, &mut w)?;
        axpy(x, F::one(), step);
        if ended || steps < k { return Ok(g[steps].abs() <= goal) }
    }
}
