//! Polynomials of one variable. A polynomial is the slice of its coefficients in ascending order, so `c` stands for
//! `c[0] + c[1]x + c[2]x² + …`. Trailing zero coefficients don't count towards the degree. `evaluate` therefore never multiplies an
//! infinite x by one of them. `from_roots`, `derivative`, `value` and `factored` have `_with` forms that call `check` once for each
//! root, or before each group of [`BETWEEN_CHECKS`](crate::BETWEEN_CHECKS) coefficients or roots, and return the check's error as soon
//! as they get one, as [`linalg::exact`](crate::linalg::exact)'s functions do.
use crate::{field::{minus, times}, float::{sealed::Core, Float, FUSED}, ok, BETWEEN_CHECKS};
use num_complex::Complex;
use num_traits::{NumAssignRef, Zero};

/// `c` without its trailing zeros.
pub fn trimmed<T: Zero>(c: &[T]) -> &[T] { &c[..c.iter().rposition(|k| !k.is_zero()).map_or(0, |i| i + 1)] }

/// Replaces each `x[i]` with the value of the polynomial `c` at `x[i]`, by Horner's rule. An f16 polynomial computes in f32 and rounds
/// each result once.
pub fn evaluate<F: Float>(c: &[F], x: &mut [F]) {
    let Some((&top, rest)) = trimmed(c).split_last() else { return x.fill(F::zero()) };
    if crate::slice::running(size_of::<F::Working>()) == 16 { blocks::<F, 16>(top, rest, x) } else { blocks::<F, 32>(top, rest, x) }
}

/// `evaluate` through `horner` on blocks of `L` items, as many as `slice::running` gives.
#[inline]
fn blocks<F: Float, const L: usize>(top: F, rest: &[F], x: &mut [F]) {
    let (blocks, tail) = x.as_chunks_mut::<L>();
    for xs in blocks { horner(top, rest, xs); }
    let mut padded = [F::zero(); L];
    padded[..tail.len()].copy_from_slice(tail);
    horner(top, rest, &mut padded);
    tail.copy_from_slice(&padded[..tail.len()]);
}

/// Replaces each of `xs` with the value there of the polynomial whose last coefficient is `top` and whose others are `rest`. The items'
/// running values stay in registers, and each coefficient takes one multiply-add per item. The loop over the items vectorizes, and the
/// items' operations don't wait for each other. The block's fixed size keeps its loops free of calls to `memcpy`, which a variable
/// length brings in.
#[inline]
fn horner<F: Float, const L: usize>(top: F, rest: &[F], xs: &mut [F; L]) {
    let mut y = [top.working(); L];
    for &k in rest.iter().rev() {
        let k = k.working();
        for i in 0..L { y[i] = y[i].mla::<FUSED>(xs[i].working(), k); }
    }
    for i in 0..L { xs[i] = F::from_working(y[i]); }
}

/// The roots of the polynomial `c`, as the eigenvalues of its companion matrix, computed in f64 by
/// [`eigenvalues`](crate::linalg::eigenvalues), sorted by real part, then imaginary part. Real coefficients take the real solver, so a
/// real root has an imaginary part of exactly 0. `None` when the eigenvalue solver doesn't converge.
pub fn roots<F: Float>(c: &[Complex<F>]) -> Option<Vec<Complex<F>>> {
    let c: Vec<Complex<f64>> = trimmed(c).iter().map(|z| Complex::new(z.re.wide(), z.im.wide())).collect();
    let n = c.len().saturating_sub(1);
    if n == 0 { return Some(Vec::new()); }
    // The companion matrix has ones below its diagonal and -c[i]/c[n] down its last column.
    let a: Vec<Complex<f64>> = c[..n].iter().map(|&k| -crate::complex::divide(k, c[n])).collect();
    let companion: Vec<Complex<f64>> = (0..n * n).map(|k| if k % n == n - 1 { a[k / n] } else { Complex::new(f64::from(k / n == k % n + 1), 0.0) }).collect();
    let mut values = if a.iter().all(|z| z.im == 0.0) {
        crate::linalg::eigenvalues(&companion.iter().map(|z| z.re).collect::<Vec<f64>>(), n)?
    } else {
        crate::linalg::eigenvalues(&companion, n)?
    };
    values.sort_by(|x, y| x.re.total_cmp(&y.re).then(x.im.total_cmp(&y.im)));
    Some(values.into_iter().map(|z| Complex::new(F::narrow(z.re), F::narrow(z.im))).collect())
}

/// The polynomial `m` times the product of `x - r` over the roots `r`.
pub fn from_roots<T: Clone + NumAssignRef>(m: T, roots: &[T]) -> Vec<T> {
    let Ok(c) = from_roots_with(m, roots, ok);
    c
}

/// [`from_roots`], calling `check` once for each root.
pub fn from_roots_with<T: Clone + NumAssignRef, E>(m: T, roots: &[T], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<T>, E> {
    let mut c = vec![m];
    for r in roots {
        check()?;
        // `a - b r`
        let less = |a: &T, b: &T| minus(a, &times(b, r));
        c.push(T::zero());
        for i in (1..c.len()).rev() { c[i] = less(&c[i - 1], &c[i]); }
        c[0] = less(&T::zero(), &c[0]);
    }
    Ok(c)
}

/// The derivative of the polynomial `c`.
pub fn derivative<T: Clone + NumAssignRef>(c: &[T]) -> Vec<T> {
    let Ok(d) = derivative_with(c, ok);
    d
}

/// [`derivative`], calling `check` before each group of `BETWEEN_CHECKS` coefficients.
pub fn derivative_with<T: Clone + NumAssignRef, E>(c: &[T], mut check: impl FnMut() -> Result<(), E>) -> Result<Vec<T>, E> {
    let (mut n, mut d) = (T::zero(), Vec::with_capacity(c.len().saturating_sub(1)));
    for group in c.get(1..).unwrap_or_default().chunks(BETWEEN_CHECKS) {
        check()?;
        d.extend(group.iter().map(|k| { n += T::one(); times(&n, k) }));
    }
    Ok(d)
}

/// The value of the polynomial `c` at `x`, by Horner's rule, for any `T` with the field operations, such as an exact rational.
/// [`evaluate`] computes the values of a float polynomial at many points at once.
pub fn value<T: Clone + NumAssignRef>(c: &[T], x: &T) -> T {
    let Ok(y) = value_with(c, x, ok);
    y
}

/// [`value`], calling `check` before each group of `BETWEEN_CHECKS` coefficients.
pub fn value_with<T: Clone + NumAssignRef, E>(c: &[T], x: &T, mut check: impl FnMut() -> Result<(), E>) -> Result<T, E> {
    let Some((top, rest)) = trimmed(c).split_last() else { return Ok(T::zero()) };
    let mut y = top.clone();
    for group in rest.rchunks(BETWEEN_CHECKS) {
        check()?;
        for k in group.iter().rev() {
            y *= x;
            y += k;
        }
    }
    Ok(y)
}

/// The value at `x` of `m` times the product of `x - r` over the roots `r`.
pub fn factored<T: Clone + NumAssignRef>(m: &T, roots: &[T], x: &T) -> T {
    let Ok(y) = factored_with(m, roots, x, ok);
    y
}

/// [`factored`], calling `check` before each group of `BETWEEN_CHECKS` roots.
pub fn factored_with<T: Clone + NumAssignRef, E>(m: &T, roots: &[T], x: &T, mut check: impl FnMut() -> Result<(), E>) -> Result<T, E> {
    let mut y = m.clone();
    for group in roots.chunks(BETWEEN_CHECKS) {
        check()?;
        for r in group { y *= &minus(x, r); }
    }
    Ok(y)
}

/// The value of each cell of `bases.len()` digits in `digits`, in the mixed radix of `bases`, as APL's `⊥` gives it: Horner's rule,
/// with each step's own base. The first base is never used. `integer::decode` gives `i64`s, checked for overflow.
pub fn decode<T: Clone + NumAssignRef>(bases: &[T], digits: &[T]) -> Vec<T> {
    digits.chunks(bases.len()).map(|d| d[1..].iter().zip(&bases[1..]).fold(d[0].clone(), |mut v, (k, b)| { v *= b; v += k; v })).collect()
}

/// The polynomial of degree `degree` that fits the points `(x[i], y[i])` by least squares, which [`lstsq`](crate::linalg::lstsq) solves
/// in f64. Where fewer than `degree + 1` distinct points fix the polynomial, it's the fit whose coefficients have the least norm. Panics if
/// `x` and `y` differ in length.
pub fn fit<F: Float>(x: &[F], y: &[F], degree: usize) -> Vec<F> {
    assert_eq!(x.len(), y.len(), "fit needs one y for each x");
    let powers: Vec<f64> = x.iter().flat_map(|x| (0..=degree).map(move |j| x.wide().powi(j as i32))).collect();
    let y: Vec<f64> = y.iter().map(|y| y.wide()).collect();
    crate::linalg::lstsq(&powers, &y, [x.len(), degree + 1], 1).0.into_iter().map(F::narrow).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn polynomials() {
        // Mixed radix: each step multiplies by its own base, and the first base goes unused.
        assert_eq!(decode(&[9.0, 2.0, 2.0], &[1.0, 0.0, 1.5, 1.0, 1.0, 1.0]), [5.5, 7.0]);
        // (x - 1)(x - 2), with a trailing zero that infinity never meets.
        let c = [2.0, -3.0, 1.0, 0.0];
        let mut x = [0.0, 1.5, 3.0, f64::INFINITY];
        evaluate(&c, &mut x);
        assert_eq!(x, [2.0, -0.25, 2.0, f64::INFINITY]);
        let mut x = [1.0, 2.0];
        evaluate(&[0.0], &mut x);
        assert_eq!(x, [0.0; 2]);

        // Across block boundaries, at f32 and f16, against f64.
        let k = [0.5, -1.25, 0.75, 2.0, -0.125];
        let x: Vec<f64> = (0..1000).map(|i| i as f64 / 250.0 - 2.0).collect();
        let mut want = x.clone();
        evaluate(&k, &mut want);
        let mut got: Vec<f32> = x.iter().map(|&x| x as f32).collect();
        evaluate(&k.map(|k| k as f32), &mut got);
        assert!(got.iter().zip(&want).all(|(&g, &w)| (g as f64 - w).abs() <= 1e-5 * w.abs().max(1.0)));
        let mut half: Vec<f16> = x.iter().map(|&x| f16::from_f64(x)).collect();
        evaluate(&k.map(f16::from_f64), &mut half);
        assert!(half.iter().zip(&want).all(|(&g, &w)| (g.to_f64() - w).abs() <= 1e-2 * w.abs().max(1.0)));

        let r = roots(&c.map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r[0].im == 0.0 && r[1].im == 0.0 && (r[0].re - 1.0).abs() < 1e-12 && (r[1].re - 2.0).abs() < 1e-12);
        // x² + 1, and a polynomial with complex coefficients, (x - i)(x - 2).
        let r = roots(&[1.0f32, 0.0, 1.0].map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r.iter().all(|z| z.re.abs() < 1e-6 && (z.im.abs() - 1.0).abs() < 1e-6));
        let r = roots(&from_roots(Complex::new(1.0, 0.0), &[Complex::new(0.0, 1.0), Complex::new(2.0, 0.0)])).unwrap();
        assert!((r[0] - Complex::new(0.0, 1.0)).norm() < 1e-12 && (r[1] - Complex::new(2.0, 0.0)).norm() < 1e-12);
        assert!(roots::<f64>(&[Complex::new(3.0, 0.0)]).unwrap().is_empty());
        // 1 + ∞x has the root -1/∞, which is -0.
        let r = roots(&[1.0, f64::INFINITY].map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r.len() == 1 && r[0].re == 0.0 && r[0].re.is_sign_negative() && r[0].im == 0.0);

        assert_eq!(from_roots(1.0, &[1.0, 2.0]), [2.0, -3.0, 1.0]);
        assert_eq!(derivative(&[2.0, -3.0, 1.0]), [-3.0, 2.0]);
        let f = fit(&[0.0f64, 1.0, 2.0, 3.0, 4.0], &[2.0, 0.0, 0.0, 2.0, 6.0], 2);
        assert!(f.iter().zip([2.0, -3.0, 1.0]).all(|(&a, b)| (a - b).abs() < 1e-12));
    }
}
