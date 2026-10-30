//! Polynomials of one variable. A polynomial is the slice of its coefficients in ascending order, so `c` stands for `c[0] + c[1]x + c[2]x² + …`. Trailing zero coefficients don't count towards the degree. `evaluate` therefore never multiplies an infinite x by one of them. `from_roots`, `derivative`, `value` and `factored` check for an interrupt between roots, or between groups of [`BETWEEN_CHECKS`](crate::BETWEEN_CHECKS) coefficients or roots, and return its error as soon as they get one, as [`linalg::exact`](crate::linalg::exact)'s functions do. A call too short for a second group makes no check, so evaluating a short polynomial at each of many points costs no checks.
use crate::{BETWEEN_CHECKS, error::Error, field::{minus, times}, float::{FUSED, Float, sealed::Core}, interrupt::check};
use num_complex::Complex;
use num_traits::{NumAssignRef, Zero};

/// `c` without its trailing zeros.
pub fn trimmed<T: Zero>(c: &[T]) -> &[T] { &c[..c.iter().rposition(|k| !k.is_zero()).map_or(0, |i| i + 1)] }

/// Appends to `out` the value of the polynomial `c` at each item of `x`, by Horner's rule. An f16 polynomial computes in f32 and rounds each result once.
pub fn evaluate<F: Float>(c: &[F], x: &[F], out: &mut Vec<F>) {
    let Some((&top, rest)) = trimmed(c).split_last() else { return out.resize(out.len() + x.len(), F::zero()) };
    crate::slice::appended(out, x.len(), |out| {
        if crate::slice::running(size_of::<F::Working>()) == 16 { blocks::<F, 16>(top, rest, x, out) } else { blocks::<F, 32>(top, rest, x, out) }
    })
}

/// `evaluate` through `horner` on blocks of `L` items, as many as `slice::running` gives.
#[inline]
fn blocks<F: Float, const L: usize>(top: F, rest: &[F], x: &[F], out: &mut [std::mem::MaybeUninit<F>]) {
    let ((blocks, tail), (outs, last)) = (x.as_chunks::<L>(), out.as_chunks_mut::<L>());
    for (xs, ys) in blocks.iter().zip(outs) { horner(top, rest, xs, ys) }
    let mut padded = [F::zero(); L];
    padded[..tail.len()].copy_from_slice(tail);
    let mut values = [std::mem::MaybeUninit::uninit(); L];
    horner(top, rest, &padded, &mut values);
    last.copy_from_slice(&values[..tail.len()]);
}

/// Writes into `out` the value at each of `xs` of the polynomial whose last coefficient is `top` and whose others are `rest`. The items' running values stay in registers, and each coefficient takes one multiply-add per item. The loop over the items vectorizes, and the items' operations don't wait for each other. The block's fixed size keeps its loops free of calls to `memcpy`, which a variable length brings in.
#[inline]
fn horner<F: Float, const L: usize>(top: F, rest: &[F], xs: &[F; L], out: &mut [std::mem::MaybeUninit<F>; L]) {
    let mut y = [top.working(); L];
    for &k in rest.iter().rev() { let k = k.working(); for i in 0..L { y[i] = y[i].mla::<FUSED>(xs[i].working(), k); } }
    for i in 0..L { out[i].write(F::from_working(y[i])); }
}

/// The roots of the polynomial `c`, as the eigenvalues of its companion matrix, computed in f64 by [`eigenvalues`](crate::linalg::eigenvalues), sorted by real part, then imaginary part. Real coefficients take the real solver, so a real root has an imaginary part of exactly 0. `None` when the eigenvalue solver doesn't converge.
pub fn roots<F: Float>(c: &[Complex<F>]) -> Option<Vec<Complex<F>>> {
    let c: Vec<Complex<f64>> = trimmed(c).iter().map(|z| Complex::new(z.re.wide(), z.im.wide())).collect();
    let n = c.len().saturating_sub(1);
    if n == 0 { return Some(Vec::new()); }
    // The companion matrix has ones below its diagonal and -c[i]/c[n] down its last column.
    let a: Vec<Complex<f64>> = c[..n].iter().map(|&k| -crate::complex::divide(k, c[n])).collect();
    let companion: Vec<Complex<f64>> = (0..n * n).map(|k| if k % n == n - 1 { a[k / n] } else { Complex::new(f64::from(k / n == k % n + 1), 0.0) }).collect();
    let mut values = if a.iter().all(|z| z.im == 0.0) { crate::linalg::eigenvalues(&companion.iter().map(|z| z.re).collect::<Vec<f64>>(), n)? }
    else { crate::linalg::eigenvalues(&companion, n)? };
    values.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
    Some(values.into_iter().map(|z| Complex::new(F::narrow(z.re), F::narrow(z.im))).collect())
}

/// The polynomial `m` times the product of `x - r` over the roots `r`. It checks for an interrupt between roots.
pub fn from_roots<T: Clone + NumAssignRef>(m: T, roots: &[T]) -> Result<Vec<T>, Error> {
    let mut c = vec![m];
    for (i, r) in roots.iter().enumerate() {
        if i > 0 { check()?; }
        // `a - b r`
        let less = |a: &T, b: &T| minus(a, &times(b, r));
        c.push(T::zero());
        for i in (1..c.len()).rev() { c[i] = less(&c[i - 1], &c[i]); }
        c[0] = less(&T::zero(), &c[0]);
    }
    Ok(c)
}

/// The derivative of the polynomial `c`. It checks for an interrupt between groups of `BETWEEN_CHECKS` coefficients.
pub fn derivative<T: Clone + NumAssignRef>(c: &[T]) -> Result<Vec<T>, Error> {
    let (mut n, mut d) = (T::zero(), Vec::with_capacity(c.len().saturating_sub(1)));
    for (i, group) in c.get(1..).unwrap_or_default().chunks(BETWEEN_CHECKS).enumerate() {
        if i > 0 { check()?; }
        d.extend(group.iter().map(|k| { n += T::one(); times(&n, k) }));
    }
    Ok(d)
}

/// The value of the polynomial `c` at `x`, by Horner's rule, for any `T` with the field operations, such as an exact rational. [`evaluate`] computes the values of a float polynomial at many points at once. It checks for an interrupt between groups of `BETWEEN_CHECKS` coefficients.
pub fn value<T: Clone + NumAssignRef>(c: &[T], x: &T) -> Result<T, Error> {
    let Some((top, rest)) = trimmed(c).split_last() else { return Ok(T::zero()) };
    let mut y = top.clone();
    for (i, group) in rest.rchunks(BETWEEN_CHECKS).enumerate() {
        if i > 0 { check()?; }
        for k in group.iter().rev() {
            y *= x;
            y += k;
        }
    }
    Ok(y)
}

/// The value at `x` of `m` times the product of `x - r` over the roots `r`. It checks for an interrupt between groups of `BETWEEN_CHECKS` roots.
pub fn factored<T: Clone + NumAssignRef>(m: &T, roots: &[T], x: &T) -> Result<T, Error> {
    let mut y = m.clone();
    for (i, group) in roots.chunks(BETWEEN_CHECKS).enumerate() { if i > 0 { check()?; } for r in group { y *= &minus(x, r); } }
    Ok(y)
}

/// The value of each cell of `bases.len()` digits in `digits`, in the mixed radix of `bases`, as APL's `⊥` gives it: Horner's rule, with each step's own base. The first base is never used. `integer::decode` gives `i64`s, checked for overflow.
pub fn decode<T: Clone + NumAssignRef>(bases: &[T], digits: &[T]) -> Vec<T> {
    digits.chunks(bases.len()).map(|d| {
        d[1..].iter().zip(&bases[1..]).fold(d[0].clone(), |mut v, (k, b)| {
            v *= b;
            v += k;
            v
        })
    }).collect()
}

/// The digits of each of `values` in the mixed radix of `bases`, as APL's `⊤` gives them: each value's digits together, the most significant first. `residue(b, v)` is `v`'s residue by the base `b`, which for a base of 0 is `v`, so a base of 0 takes all of what remains as its digit. `integer::encode` gives `i64`s, checked for overflow.
pub fn encode<T: Clone + NumAssignRef>(bases: &[T], values: &[T], residue: impl Fn(&T, &T) -> T) -> Vec<T> {
    encoded(bases, values, |b, mut v| {
        let digit = residue(b, &v);
        if b.is_zero() { v.set_zero() } else {
            v -= &digit;
            v /= b;
        }
        Some((v, digit))
    }).expect("a step in a field gives a value")
}

/// `encode`'s rule, where `div_mod(base, value)` gives the value that the next base divides and the digit, and `None` where it gives `None`. `integer::encode` divides through its `Divisor`s this way.
pub(crate) fn encoded<B, T: Clone + Zero>(bases: &[B], values: &[T], div_mod: impl Fn(&B, T) -> Option<(T, T)>) -> Option<Vec<T>> {
    let mut digits = vec![T::zero(); bases.len() * values.len()];
    for (out, value) in digits.chunks_mut(bases.len().max(1)).zip(values) {
        let mut value = value.clone();
        for (digit, base) in out.iter_mut().zip(bases).rev() { (value, *digit) = div_mod(base, value)?; }
    }
    Some(digits)
}

/// The polynomial of degree `degree` that fits the points `(x[i], y[i])` by least squares, which [`lstsq`](crate::linalg::lstsq) solves in f64. Where fewer than `degree + 1` distinct points fix the polynomial, it's the fit whose coefficients have the least norm. Panics if `x` and `y` differ in length.
pub fn fit<F: Float>(x: &[F], y: &[F], degree: usize) -> Vec<F> {
    assert_eq!(x.len(), y.len(), "fit needs one y for each x");
    let powers: Vec<f64> = x.iter().flat_map(|x| { let x = x.wide(); std::iter::successors(Some(1.0), move |p| Some(p * x)).take(degree + 1) }).collect();
    let y: Vec<f64> = y.iter().map(|y| y.wide()).collect();
    crate::linalg::lstsq(&powers, &y, [x.len(), degree + 1], 1).0.into_iter().map(F::narrow).collect()
}
