//! Polynomials of one variable. A polynomial is the slice of its coefficients in ascending order, so `c` stands for
//! `c[0] + c[1]x + c[2]x² + …`. Trailing zero coefficients don't count towards the degree. `evaluate` therefore never multiplies an
//! infinite x by one of them.
//!
//! For example, the roots of (x - 1)(x - 2):
//!
//! ```
//! use num_complex::Complex;
//! let c = [2.0f64, -3.0, 1.0].map(|k| Complex::new(k, 0.0));
//! let r = rustymath::polynomial::roots(&c).unwrap();
//! assert!(r.iter().all(|z| z.im == 0.0 && ((z.re - 1.0).abs() < 1e-12 || (z.re - 2.0).abs() < 1e-12)));
//! ```
use crate::float::{sealed::Core, Float, FUSED};
use faer::{prelude::*, Mat};
use num_complex::Complex;
use num_traits::{Num, Zero};

/// `c` without its trailing zeros.
fn trimmed<T: Zero>(c: &[T]) -> &[T] { &c[..c.iter().rposition(|k| !k.is_zero()).map_or(0, |i| i + 1)] }

/// Replaces each `x[i]` with the value of the polynomial `c` at `x[i]`, by Horner's rule. An f16 polynomial computes in f32 and rounds
/// each result once.
pub fn evaluate<F: Float>(c: &[F], x: &mut [F]) {
    let Some((&top, rest)) = trimmed(c).split_last() else { return x.fill(F::zero()) };
    let (blocks, tail) = x.as_chunks_mut::<LANES>();
    for xs in blocks { horner(top, rest, xs); }
    let mut padded = [F::zero(); LANES];
    padded[..tail.len()].copy_from_slice(tail);
    horner(top, rest, &mut padded);
    tail.copy_from_slice(&padded[..tail.len()]);
}

/// The number of items that `horner` works on at once.
const LANES: usize = 32;

/// Replaces each of `xs` with the value there of the polynomial whose last coefficient is `top` and whose others are `rest`. The items'
/// running values stay in registers, and each coefficient takes one multiply-add per item. The loop over the items vectorizes, and the
/// items' operations don't wait for each other. The block's fixed size keeps its loops free of calls to `memcpy`, which a variable
/// length brings in.
#[inline]
fn horner<F: Float>(top: F, rest: &[F], xs: &mut [F; LANES]) {
    let mut y = [top.working(); LANES];
    for &k in rest.iter().rev() {
        let k = k.working();
        for i in 0..LANES { y[i] = y[i].mla::<FUSED>(xs[i].working(), k); }
    }
    for i in 0..LANES { xs[i] = F::from_working(y[i]); }
}

/// The roots of the polynomial `c`, as the eigenvalues of its companion matrix, which `faer` computes in f64. When every coefficient
/// is real, a real root has an imaginary part of exactly 0. `None` when the eigenvalue solver doesn't converge.
pub fn roots<F: Float>(c: &[Complex<F>]) -> Option<Vec<Complex<F>>> {
    let c: Vec<Complex<f64>> = trimmed(c).iter().map(|z| Complex::new(z.re.wide(), z.im.wide())).collect();
    let n = c.len().saturating_sub(1);
    if n == 0 { return Some(Vec::new()); }
    // The companion matrix has ones below its diagonal and -c[i]/c[n] down its last column. Real coefficients divide as reals: complex
    // division gives NaN for 1/∞, where real division gives 0.
    let a: Vec<Complex<f64>> = if c.iter().all(|z| z.im == 0.0) {
        c[..n].iter().map(|k| Complex::new(-k.re / c[n].re, 0.0)).collect()
    } else {
        c[..n].iter().map(|&k| -k / c[n]).collect()
    };
    let values = if a.iter().all(|z| z.im == 0.0) {
        Mat::from_fn(n, n, |i, j| if j == n - 1 { a[i].re } else { f64::from(i == j + 1) }).eigenvalues()
    } else {
        Mat::from_fn(n, n, |i, j| if j == n - 1 { a[i] } else { Complex::new(f64::from(i == j + 1), 0.0) }).eigenvalues()
    };
    Some(values.ok()?.into_iter().map(|z| Complex::new(F::narrow(z.re), F::narrow(z.im))).collect())
}

/// The polynomial `m` times the product of `x - r` over the roots `r`.
pub fn from_roots<T: Copy + Num>(m: T, roots: &[T]) -> Vec<T> {
    let mut c = vec![m];
    for &r in roots {
        c.push(T::zero());
        for i in (1..c.len()).rev() { c[i] = c[i - 1] - r * c[i]; }
        c[0] = T::zero() - r * c[0];
    }
    c
}

/// The derivative of the polynomial `c`.
pub fn derivative<T: Copy + Num>(c: &[T]) -> Vec<T> {
    let mut n = T::zero();
    c.iter().skip(1).map(|&k| { n = n + T::one(); n * k }).collect()
}

/// The polynomial of degree `degree` that fits the points `(x[i], y[i])` by least squares, which `faer` solves in f64 through a QR
/// decomposition. Panics if `x` and `y` differ in length or hold fewer than `degree + 1` points.
pub fn fit<F: Float>(x: &[F], y: &[F], degree: usize) -> Vec<F> {
    assert_eq!(x.len(), y.len(), "fit needs one y for each x");
    let powers = Mat::from_fn(x.len(), degree + 1, |i, j| x[i].wide().powi(j as i32));
    let solution = powers.qr().solve_lstsq(Mat::from_fn(y.len(), 1, |i, _| y[i].wide()));
    (0..=degree).map(|i| F::narrow(solution[(i, 0)])).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;

    #[test]
    fn polynomials() {
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

        let mut r = roots(&c.map(|k| Complex::new(k, 0.0))).unwrap();
        r.sort_by(|a, b| a.re.total_cmp(&b.re));
        assert!(r[0].im == 0.0 && r[1].im == 0.0 && (r[0].re - 1.0).abs() < 1e-12 && (r[1].re - 2.0).abs() < 1e-12);
        // x² + 1, and a polynomial with complex coefficients, (x - i)(x - 2).
        let r = roots(&[1.0f32, 0.0, 1.0].map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r.iter().all(|z| z.re.abs() < 1e-6 && (z.im.abs() - 1.0).abs() < 1e-6));
        let r = roots(&from_roots(Complex::new(1.0, 0.0), &[Complex::new(0.0, 1.0), Complex::new(2.0, 0.0)])).unwrap();
        assert!(r.iter().any(|z| (z - Complex::new(0.0, 1.0)).norm() < 1e-12) && r.iter().any(|z| (z - Complex::new(2.0, 0.0)).norm() < 1e-12));
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
