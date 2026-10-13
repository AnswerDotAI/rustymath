//! Decompositions of real and complex matrices held row-major in slices: LU with partial pivoting, Cholesky, Householder QR and the
//! SVD, with the solves, inverses and least-squares solutions built from them, at `f32`, `f64` and complex numbers with parts of either
//! type. The child module `svd` computes the SVD, `eigen` the Hermitian eigen decomposition, and `general` the eigen decomposition of
//! any square matrix.
//!
//! Each decomposition is written once, generic over [`Scalar`]. A complex factorization conjugates where the real one transposes.
//! Cholesky gives L Lᴴ, and QR's reflections take the form `I - τ v vᴴ`, leaving R with a real diagonal. Singular values and the
//! eigenvalues of a Hermitian matrix are real, of the parts' type. `general` uses single-shift QR iterations for a complex matrix, in
//! place of a real one's double-shift iterations.
//!
//! LU, Cholesky and QR work in blocks of columns, wider for a larger matrix. Each block's own columns factor with the unblocked
//! algorithm, along rows, so every inner loop reads contiguous memory. The rest of the matrix then updates through the tiled `+.×`
//! kernels of `linalg.rs`, where most of a large factorization's time goes. QR keeps each block's reflections in the compact WY
//! form `I - V T Vᴴ`, so that applying them takes three matrix products. Dot
//! products keep eight running sums, and each multiply-add is fused where the target has an instruction for it, so results can
//! differ from LAPACK's in their last bits.
//!
//! QR and the SVD first scale a matrix by a power of two when its largest item lies where sums of squares would overflow or lose
//! digits to underflow. Their factors are those of the scaled matrix, scaled back exactly. LU and Cholesky divide by a subnormal
//! pivot, whose reciprocal would overflow.
use super::{identity, transpose_into, Scalar, Tiled};
use crate::{float::{sealed::Core, FUSED}, Native};
use num_complex::Complex;
use num_traits::{Float as _, One as _, ToPrimitive as _, Zero as _};
use std::{borrow::Cow, cmp::Ordering};

mod eigen;
mod general;
mod secular;
mod svd;
pub use eigen::{eigh, hermitian_eigenvalues, Eigh};
pub use general::{eig, eigenvalues, Eig};

/// Columns in each block of a blocked LU or Cholesky factorization of an `n`×`n` matrix. A larger matrix takes wider blocks, whose
/// trailing updates are longer matrix products. On the Mac this measured fastest from 256 to 1024 rows.
fn lu_block(n: usize) -> usize { (n / 8).clamp(32, 128) }
/// Columns in each block of a blocked QR factorization with `p` reflections. QR measured fastest with narrower blocks than LU's.
fn qr_block(p: usize) -> usize { (p / 16).clamp(32, 64) }
/// Columns up to which `Reflections::q` applies one reflection at a time.
const ONE_AT_A_TIME: usize = 64;
/// Right-hand sides up to which `Reflections::apply_adjoint` applies one reflection at a time. For a 4096×16 `lstsq`, blocks measured
/// as fast at 12 and faster from 16.
const FEW_COLUMNS: usize = 12;
/// Columns at which a block stops halving and factors one column at a time.
const LEAF: usize = 16;
/// Rows in each block of a solve with more than one right-hand side. A block's own rows substitute with vector operations, and the rows
/// below it update through a matrix product.
const NS: usize = 32;

/// An item type whose matrices rustymath factors: `f32`, `f64`, or a complex number with parts of either type. Only rustymath implements
/// it, and the functions of this module call it. Its methods compile in rustymath.
pub trait Factor: Scalar {
    /// The LU factorization of the `n`×`n` matrix `a`.
    fn matrix_lu(a: &[Self], n: usize) -> Lu<Self>;
    /// The solution of `a x = b` for the matrix that `lu` factors and the `n`×`k` matrix `b`.
    fn matrix_lu_solve(lu: &Lu<Self>, b: &[Self], k: usize) -> Vec<Self>;
    /// The inverse of the matrix that `lu` factors.
    fn matrix_lu_inverse(lu: &Lu<Self>) -> Vec<Self>;
    /// The Cholesky factor of the `n`×`n` matrix `a`, or `None` where `a` isn't positive definite.
    fn matrix_cholesky(a: &[Self], n: usize) -> Option<Vec<Self>>;
    /// The QR factorization of the matrix `a` of `size` `[m, n]`: thin, or complete where `complete` is true.
    fn matrix_qr(a: &[Self], size: [usize; 2], complete: bool) -> (Vec<Self>, Vec<Self>);
    /// The thin SVD of the matrix `a` of `size` `[m, n]`, with its singular vectors only where `vectors` is true.
    fn matrix_svd(a: &[Self], size: [usize; 2], vectors: bool) -> Svd<Self>;
    /// The least-squares solution of `a x = b` for the matrix that `svd` factors and the `m`×`k` matrix `b`, or without `b` its `n`×`m`
    /// pseudoinverse.
    fn matrix_svd_solve(svd: &Svd<Self>, b: Option<&[Self]>, k: usize) -> Vec<Self>;
    /// The least-squares solution of `a x = b`, and `a`'s singular values, for the matrix `a` of `size` `[m, n]` and the `m`×`k` matrix `b`.
    fn matrix_lstsq(a: &[Self], b: &[Self], size: [usize; 2], k: usize) -> (Vec<Self>, Vec<Self::Real>);
    /// The eigen decomposition of the Hermitian `n`×`n` matrix `a`, from its lower triangle, with its eigenvectors only where `vectors`
    /// is true, or `None` where it doesn't converge.
    fn matrix_eigh(a: &[Self], n: usize, vectors: bool) -> Option<Eigh<Self>>;
    /// The eigen decomposition of the `n`×`n` matrix `a`, with its eigenvectors only where `vectors` is true, or `None` where it
    /// doesn't converge.
    fn matrix_eig(a: &[Self], n: usize, vectors: bool) -> Option<Eig<Self::Real>>;
}

macro_rules! factor {
    ($($t:ty),+) => {$(
        impl Factor for $t {
            fn matrix_lu(a: &[$t], n: usize) -> Lu<$t> { lu_factor(a, n) }
            fn matrix_lu_solve(lu: &Lu<$t>, b: &[$t], k: usize) -> Vec<$t> { lu_solve(lu, b, k) }
            fn matrix_lu_inverse(lu: &Lu<$t>) -> Vec<$t> { lu_inverse(lu) }
            fn matrix_cholesky(a: &[$t], n: usize) -> Option<Vec<$t>> { cholesky_factor(a, n) }
            fn matrix_qr(a: &[$t], size: [usize; 2], complete: bool) -> (Vec<$t>, Vec<$t>) { qr_factor(a, size, complete) }
            fn matrix_svd(a: &[$t], size: [usize; 2], vectors: bool) -> Svd<$t> { svd::svd_factor(a, size, vectors) }
            fn matrix_svd_solve(svd: &Svd<$t>, b: Option<&[$t]>, k: usize) -> Vec<$t> { svd_solve(svd, b, k) }
            fn matrix_lstsq(a: &[$t], b: &[$t], size: [usize; 2], k: usize) -> (Vec<$t>, Vec<<$t as Scalar>::Real>) { lstsq_solve(a, b, size, k) }
            fn matrix_eigh(a: &[$t], n: usize, vectors: bool) -> Option<Eigh<$t>> { eigen::eigh_factor(a, n, vectors) }
            fn matrix_eig(a: &[$t], n: usize, vectors: bool) -> Option<Eig<<$t as Scalar>::Real>> { general::eig_factor(a, n, vectors) }
        }
    )+};
}
factor!(f32, f64, Complex<f32>, Complex<f64>);

/// An LU factorization with partial pivoting of an `n`×`n` matrix. `factors` holds L below its diagonal, whose own diagonal is 1 and
/// isn't stored, and U on and above it. Row `i` of L U is row `perm[i]` of the factored matrix.
#[derive(Clone, Debug, PartialEq)]
pub struct Lu<T> {
    pub n: usize,
    pub factors: Vec<T>,
    pub perm: Vec<usize>,
}

impl<T: Factor> Lu<T> {
    /// The solution `x` of `a x = b`, for the factored `a` and the `n`×`k` matrix `b`, as an `n`×`k` matrix.
    pub fn solve(&self, b: &[T], k: usize) -> Vec<T> { T::matrix_lu_solve(self, b, k) }
    /// The inverse of the factored matrix.
    pub fn inverse(&self) -> Vec<T> { T::matrix_lu_inverse(self) }
    /// U's diagonal.
    pub fn diagonal(&self) -> impl Iterator<Item = T> + '_ { (0..self.n).map(|i| self.factors[i * self.n + i]) }
    /// L as an `n`×`n` matrix, with ones on its diagonal and zeros above it.
    pub fn lower(&self) -> Vec<T> {
        let n = self.n;
        (0..n * n).map(|k| match (k / n).cmp(&(k % n)) { Ordering::Greater => self.factors[k], Ordering::Equal => T::one(), Ordering::Less => T::zero() }).collect()
    }
    /// U as an `n`×`n` matrix, with zeros below its diagonal.
    pub fn upper(&self) -> Vec<T> {
        let n = self.n;
        (0..n * n).map(|k| if k % n >= k / n { self.factors[k] } else { T::zero() }).collect()
    }
    /// The factored matrix's determinant: the product of U's diagonal, negated where `perm` is odd.
    pub fn determinant(&self) -> T {
        let product = self.diagonal().fold(T::one(), |p, d| p * d);
        if odd(&self.perm) { -product } else { product }
    }
    /// The factored matrix's determinant as a sign and the natural logarithm of its magnitude, which can't overflow. The sign has
    /// magnitude 1, or is 0 with a logarithm of -∞ for a singular matrix.
    pub fn log_determinant(&self) -> (T, T::Real) {
        let start = (if odd(&self.perm) { -T::one() } else { T::one() }, T::Real::zero());
        self.diagonal().fold(start, |(sign, log), d| {
            let m = d.modulus();
            if m == T::Real::zero() { (T::zero(), T::Real::neg_infinity()) } else { (sign * d.mul_real(m.recip()), log + m.ln()) }
        })
    }
}

/// Whether the permutation `perm` of `0..perm.len()` is odd: whether an odd number of swaps puts it in order.
fn odd(perm: &[usize]) -> bool {
    let mut seen = vec![false; perm.len()];
    let mut swaps = 0;
    for start in 0..perm.len() {
        if seen[start] { continue }
        let mut i = start;
        while !seen[i] {
            seen[i] = true;
            i = perm[i];
            swaps += 1;
        }
        swaps -= 1;
    }
    swaps % 2 == 1
}

/// The thin SVD of an `m`×`n` matrix: `a = U diag(s) Vᴴ`, where `p` is the smaller of `m` and `n`, U is `m`×`p` and V is `n`×`p`, both
/// with orthonormal columns. `s` is real and in descending order. Without singular vectors, `u` and `v` are empty.
#[derive(Clone, Debug, PartialEq)]
pub struct Svd<T: Scalar> {
    pub size: [usize; 2],
    pub u: Vec<T>,
    pub s: Vec<T::Real>,
    pub v: Vec<T>,
}

impl<T: Factor> Svd<T> {
    /// The least-squares solution `x` of `a x = b` with the least norm, for the factored `a` and the `m`×`k` matrix `b`, as an `n`×`k`
    /// matrix. Singular values below `ε` times the larger of `m` and `n` times the largest singular value count as zero, as in NumPy's
    /// `lstsq`.
    pub fn solve(&self, b: &[T], k: usize) -> Vec<T> { T::matrix_svd_solve(self, Some(b), k) }
    /// The `n`×`m` pseudoinverse of the factored matrix, with the cutoff that `solve` uses.
    pub fn pseudoinverse(&self) -> Vec<T> { T::matrix_svd_solve(self, None, self.size[0]) }
}

/// The LU factorization of the `n`×`n` matrix `a`, with partial pivoting. A zero pivot leaves its column unreduced, so check U's
/// diagonal before solving.
pub fn lu<T: Factor>(a: &[T], n: usize) -> Lu<T> { T::matrix_lu(a, n) }
/// The lower triangular L with `L Lᴴ = a`, for the Hermitian `n`×`n` matrix `a`, or `None` where `a` isn't positive definite. L's
/// diagonal is real. It reads only `a`'s lower triangle.
pub fn cholesky<T: Factor>(a: &[T], n: usize) -> Option<Vec<T>> { T::matrix_cholesky(a, n) }
/// The thin QR factorization `(Q, R)` of the matrix `a` of `size` `[m, n]`. Where `p` is the smaller of `m` and `n`, Q is `m`×`p` with
/// orthonormal columns, and R is `p`×`n` and upper triangular, with a real diagonal.
pub fn qr<T: Factor>(a: &[T], size: [usize; 2]) -> (Vec<T>, Vec<T>) { T::matrix_qr(a, size, false) }
/// The complete QR factorization `(Q, R)` of the matrix `a` of `size` `[m, n]`: Q is `m`×`m` and unitary, and R is `m`×`n`. Q's first
/// columns and R's first rows are `qr`'s, and R's other rows are zero.
pub fn qr_complete<T: Factor>(a: &[T], size: [usize; 2]) -> (Vec<T>, Vec<T>) { T::matrix_qr(a, size, true) }
/// The thin SVD of the matrix `a` of `size` `[m, n]`.
pub fn svd<T: Factor>(a: &[T], size: [usize; 2]) -> Svd<T> { T::matrix_svd(a, size, true) }
/// The singular values of the matrix `a` of `size` `[m, n]`, in descending order.
pub fn singular_values<T: Factor>(a: &[T], size: [usize; 2]) -> Vec<T::Real> { T::matrix_svd(a, size, false).s }
/// The least-squares solution `x` of `a x = b` with the least norm, and `a`'s singular values in descending order, for the matrix `a` of
/// `size` `[m, n]` and the `m`×`k` matrix `b`. `x` is `n`×`k`. Singular values below `ε` times the larger of `m` and `n` times the
/// largest count as zero, as in NumPy's `lstsq`.
pub fn lstsq<T: Factor>(a: &[T], b: &[T], size: [usize; 2], k: usize) -> (Vec<T>, Vec<T::Real>) { T::matrix_lstsq(a, b, size, k) }
/// The solution `x` of `a x = b` for the matrix `a` of `size` `[m, n]` and the `m`×`k` matrix `b`, as an `n`×`k` matrix: `lstsq`'s
/// least-squares solution with the least norm. A square `a` with no negligible pivot solves through its LU factorization.
pub fn solve<T: Factor>(a: &[T], b: &[T], size: [usize; 2], k: usize) -> Vec<T> {
    match regular(a, size) {
        Some(lu) => lu.solve(b, k),
        None => lstsq(a, b, size, k).0,
    }
}
/// The `n`×`m` pseudoinverse of the matrix `a` of `size` `[m, n]`, with `lstsq`'s cutoff. A square `a` with no negligible pivot gives
/// its inverse through its LU factorization.
pub fn pinv<T: Factor>(a: &[T], size: [usize; 2]) -> Vec<T> {
    match regular(a, size) {
        Some(lu) => lu.inverse(),
        None => svd(a, size).pseudoinverse(),
    }
}
/// The rank of the matrix `a` of `size` `[m, n]`: how many of its singular values exceed `lstsq`'s cutoff.
pub fn rank<T: Factor>(a: &[T], size: [usize; 2]) -> usize {
    let s = singular_values(a, size);
    let floor = cutoff(&s, size[0].max(size[1]));
    s.iter().filter(|&&x| x > floor).count()
}
/// The LU factorization of a square, nonempty `a` whose pivots all exceed `ε` times `n` times `a`'s largest magnitude. A NaN pivot
/// doesn't count as negligible, so NaN reaches the results.
fn regular<T: Factor>(a: &[T], [m, n]: [usize; 2]) -> Option<Lu<T>> {
    if m != n || n == 0 { return None }
    let largest = a.iter().fold(T::Real::zero(), |s, x| s.max(x.modulus()));
    let negligible = T::Real::epsilon() * <T::Real as crate::Float>::narrow(n as f64) * largest;
    let lu = lu(a, n);
    let regular = lu.diagonal().all(|d| !(d.modulus() <= negligible));
    regular.then_some(lu)
}

/// `y += a x`, item by item. `axpy`, `axpyc` and `rotate` loop by index over equal lengths. A `zip` in them leaves a call to `Zip::new`
/// in large callers, such as the SVD's, which LLVM doesn't inline.
#[inline(always)]
fn axpy<T: Scalar>(y: &mut [T], a: T, x: &[T]) {
    let n = y.len().min(x.len());
    let (y, x) = (&mut y[..n], &x[..n]);
    for i in 0..n { y[i] = x[i].mul_acc(a, y[i]) }
}

/// `y += a conj(x)`, item by item, which for real items is `axpy`.
#[inline(always)]
fn axpyc<T: Scalar>(y: &mut [T], a: T, x: &[T]) {
    let n = y.len().min(x.len());
    let (y, x) = (&mut y[..n], &x[..n]);
    for i in 0..n { y[i] = x[i].conj().mul_acc(a, y[i]) }
}

/// The sum of `x[i] y[i]`.
#[inline(always)]
fn dot<T: Scalar>(x: &[T], y: &[T]) -> T { sum_of_products(x, y, |a| a) }

/// The sum of `conj(x[i]) y[i]`, the inner product of two complex vectors. For real items it is `dot`.
#[inline(always)]
fn dotc<T: Scalar>(x: &[T], y: &[T]) -> T { sum_of_products(x, y, T::conj) }

/// The sum of `f(x[i]) y[i]`, in eight running sums, which vectorize.
#[inline(always)]
fn sum_of_products<T: Scalar>(x: &[T], y: &[T], f: impl Fn(T) -> T) -> T {
    let mut s = [T::zero(); 8];
    let ((xc, xr), (yc, yr)) = (x.as_chunks::<8>(), y.as_chunks::<8>());
    for (a, b) in xc.iter().zip(yc) {
        for k in 0..8 { s[k] = f(a[k]).mul_acc(b[k], s[k]) }
    }
    xr.iter().zip(yr).fold(s.iter().fold(T::zero(), |t, &v| t + v), |t, (&a, &b)| f(a).mul_acc(b, t))
}

/// The position of the item of `x` with the greatest `abs1`, the first of any equal ones, or 0 where every item is NaN. Eight running
/// maximums find the greatest, which vectorizes, then a scan finds its position.
fn largest<T: Scalar>(x: &[T]) -> usize {
    let mut m = [T::Real::zero(); 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for c in chunks {
        for k in 0..8 { m[k] = m[k].max(c[k].abs1()) }
    }
    let top = rest.iter().fold(m.iter().fold(T::Real::zero(), |a, &b| a.max(b)), |a, &b| a.max(b.abs1()));
    x.iter().position(|v| v.abs1() == top).unwrap_or(0)
}

/// The power of two that brings the greatest `abs1` in `a` to between 1 and 2, when that magnitude lies outside the range where sums
/// of squares neither overflow nor lose digits to underflow, from √(least normal)/ε to its reciprocal. `None` inside that range, and
/// for a matrix of zeros or one holding an infinity. Scaling by a power of two is exact.
fn safe_scale<T: Scalar>(a: &[T]) -> Option<T::Real> {
    let top = a.get(largest(a))?.abs1();
    let low = T::Real::min_positive_value().sqrt() / T::Real::epsilon();
    if !top.is_finite() || top == T::Real::zero() || (low..=low.recip()).contains(&top) { return None }
    let limit = T::Real::max_value().log2().floor();
    Some((T::Real::one() + T::Real::one()).powi((-top.log2().floor()).max(-limit).min(limit).to_i32()?))
}

/// Divides each item of `x` by `d`. It multiplies by the reciprocal, except for a subnormal `d`, whose reciprocal overflows.
fn divide<T: Scalar>(x: &mut [T], d: T) {
    if d.abs1() >= T::Real::min_positive_value() {
        let r = d.reciprocal();
        for v in x { *v = *v * r }
    } else {
        for v in x { *v = v.quotient(d) }
    }
}

/// The transpose of the `m`×`n` matrix `a`, with each item converted by `f`.
fn transpose_by<T: Copy>(a: &[T], m: usize, n: usize, f: impl Fn(T) -> T) -> Vec<T> {
    let mut t = a.to_vec();
    transpose_into(a, n, &mut t, m, [m, n], f);
    t
}

/// The transpose of the `m`×`n` matrix `a`.
fn transpose<T: Copy>(a: &[T], m: usize, n: usize) -> Vec<T> { transpose_by(a, m, n, |x| x) }

/// The conjugate transpose of the `m`×`n` matrix `a`, which for real items is its transpose.
fn adjoint<T: Scalar>(a: &[T], m: usize, n: usize) -> Vec<T> { transpose_by(a, m, n, T::conj) }

/// Rows `p` and `q` of `a`, whose rows hold `width` items, with `p` before `q`.
#[inline(always)]
fn two_rows<T>(a: &mut [T], width: usize, p: usize, q: usize) -> (&mut [T], &mut [T]) {
    let (head, tail) = a.split_at_mut(q * width);
    (&mut head[p * width..(p + 1) * width], &mut tail[..width])
}

/// `x ← c x - s y` and `y ← s x + c y`, item by item, for real `c` and `s`.
#[inline(always)]
fn rotate<T: Scalar>(x: &mut [T], y: &mut [T], c: T::Real, s: T::Real) {
    let n = x.len().min(y.len());
    let (x, y) = (&mut x[..n], &mut y[..n]);
    for i in 0..n { (x[i], y[i]) = (x[i].mul_real(c) - y[i].mul_real(s), x[i].mul_real(s) + y[i].mul_real(c)) }
}

/// Columns `k0..k1` of `f`, a matrix with rows of `n` items, from row `k0` down, transposed so that each column is contiguous.
fn transposed<T: Scalar>(f: &[T], n: usize, k0: usize, k1: usize) -> Vec<T> {
    let (rows, nb) = (f.len() / n - k0, k1 - k0);
    let mut panel = vec![T::zero(); nb * rows];
    transpose_into(&f[k0 * n + k0..], n, &mut panel, rows, [rows, nb], |x| x);
    panel
}

/// Factors columns `c0..c1` of the LU block that `panel` holds transposed, with rows of `rows` items. It halves the columns until `LEAF` or
/// fewer remain: the left half factors first, then its rows of U and a matrix product update the right half, which factors last.
/// Each pivot's row swap applies to all of the block's columns, and through `swap` to the rest of the matrix.
fn lu_panel<T: Scalar + Tiled>(panel: &mut [T], rows: usize, [c0, c1]: [usize; 2], swap: &mut impl FnMut(usize, usize)) {
    if c1 - c0 > LEAF {
        let mid = (c0 + c1) / 2;
        lu_panel(panel, rows, [c0, mid], swap);
        let (left, right) = panel.split_at_mut(mid * rows);
        let (k, half) = (mid - c0, &mut right[..(c1 - mid) * rows]);
        // The right half's rows of U, by substitution with the left half's L, then the rows below them.
        for column in half.chunks_exact_mut(rows) {
            for p in c0..mid {
                let u = column[p];
                axpy(&mut column[p + 1..mid], -u, &left[p * rows + p + 1..p * rows + mid]);
            }
        }
        let u12: Vec<T> = half.chunks_exact(rows).flat_map(|column| column[c0..mid].to_vec()).collect();
        T::add_product(&mut half[mid..], rows, |j, p| -u12[j * k + p], |p, r| left[(c0 + p) * rows + mid + r], [c1 - mid, k, rows - mid]);
        return lu_panel(panel, rows, [mid, c1], swap);
    }
    let nb = panel.len() / rows;
    for c in c0..c1 {
        let column = &panel[c * rows..(c + 1) * rows];
        let p = c + largest(&column[c..]);
        if p != c {
            for j in 0..nb { panel.swap(j * rows + c, j * rows + p) }
            swap(c, p);
        }
        let pivot = panel[c * rows + c];
        if pivot == T::zero() { continue }
        let (left, right) = panel.split_at_mut((c + 1) * rows);
        let l = &mut left[c * rows + c + 1..];
        divide(l, pivot);
        for column in right[..(c1 - c - 1) * rows].chunks_exact_mut(rows) {
            let u = column[c];
            axpy(&mut column[c + 1..], -u, l);
        }
    }
}
/// `lu` for `T`, by blocks of `lu_block(n)` columns. Each block's columns factor in a transposed copy, where each column is contiguous. Each
/// pivot's row swap applies across the whole matrix.
fn lu_factor<T: Scalar + Tiled>(a: &[T], n: usize) -> Lu<T> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    let (mut f, mut perm) = (a.to_vec(), (0..n).collect::<Vec<_>>());
    let width = lu_block(n);
    for k0 in (0..n).step_by(width) {
        let k1 = (k0 + width).min(n);
        let (nb, rows, w) = (k1 - k0, n - k0, n - k1);
        let mut panel = transposed(&f, n, k0, k1);
        let mut swap = |c: usize, p: usize| {
            let (head, tail) = f.split_at_mut((k0 + p) * n);
            let (rc, rp) = (&mut head[(k0 + c) * n..(k0 + c + 1) * n], &mut tail[..n]);
            rc[..k0].swap_with_slice(&mut rp[..k0]);
            rc[k1..].swap_with_slice(&mut rp[k1..]);
            perm.swap(k0 + c, k0 + p);
        };
        lu_panel(&mut panel, rows, [0, nb], &mut swap);
        transpose_into(&panel, rows, &mut f[k0 * n + k0..], n, [nb, rows], |x| x);
        if w == 0 { break }
        // The block's rows of U to its right: U12 ← L11⁻¹ A12, a block of 16 rows at a time.
        forward(|i, j| panel[j * rows + i], nb, &mut f[k0 * n + k1..], [n, w], 16, false);
        // The rest of the matrix: A22 ← A22 - L21 U12.
        let (top, bottom) = f.split_at_mut(k1 * n);
        T::add_product(&mut bottom[k1..], n, |i, p| -panel[p * rows + nb + i], |p, j| top[(k0 + p) * n + k1 + j], [w, nb, w]);
    }
    Lu { n, factors: f, perm }
}

/// `Lu::solve` for `T`. One right-hand side substitutes with dot products along L's and U's rows. More substitute a block of rows at a
/// time, after one matrix product subtracts the rows already solved.
fn lu_solve<T: Scalar + Tiled>(lu: &Lu<T>, b: &[T], k: usize) -> Vec<T> {
    let (n, f) = (lu.n, &lu.factors);
    assert_eq!(b.len(), n * k, "`b` must hold `n`×`k` items");
    let mut x: Vec<T> = lu.perm.iter().flat_map(|&p| b[p * k..(p + 1) * k].iter().copied()).collect();
    if k == 1 {
        for i in 0..n { x[i] = x[i] - dot(&f[i * n..i * n + i], &x[..i]) }
        for i in (0..n).rev() { x[i] = (x[i] - dot(&f[i * n + i + 1..(i + 1) * n], &x[i + 1..])).quotient(f[i * n + i]) }
    } else {
        forward(|i, j| f[i * n + j], n, &mut x, [k, k], NS, false);
        backward(f, n, &mut x, k);
    }
    x
}

/// `Lu::inverse` for `T`: `U⁻¹ L⁻¹`, with its columns permuted to undo the row swaps. Its substitution with L skips the zeros above the
/// diagonal of `L⁻¹`.
fn lu_inverse<T: Scalar + Tiled>(lu: &Lu<T>) -> Vec<T> {
    let n = lu.n;
    let mut x = identity(n);
    forward(|i, j| lu.factors[i * n + j], n, &mut x, [n, n], NS, true);
    backward(&lu.factors, n, &mut x, n);
    let mut inverse = vec![T::zero(); n * n];
    for r in 0..n {
        for (c, &p) in lu.perm.iter().enumerate() { inverse[r * n + p] = x[r * n + c] }
    }
    inverse
}

/// Solves `L y = x` in place, a block of `block` rows at a time, where L is the `n`×`n` unit lower triangular matrix whose items below
/// its diagonal `l(i, j)` gives, and `x` has `n` rows of `k` items, which start `ldx` items apart. With `lower`, `x` starts as the
/// identity, and only the columns up to each row's own can be nonzero. Each block's rows substitute with vector operations, then one
/// matrix product removes them from every row below, so each solved row is packed for a product only once.
fn forward<T: Scalar + Tiled>(l: impl Fn(usize, usize) -> T, n: usize, x: &mut [T], [ldx, k]: [usize; 2], block: usize, lower: bool) {
    for i0 in (0..n).step_by(block) {
        let i1 = (i0 + block).min(n);
        let width = if lower { i1 } else { k };
        let (done, rest) = x.split_at_mut(i1 * ldx);
        let rows = &mut done[i0 * ldx..];
        for i in i0 + 1..i1 {
            let (head, tail) = rows.split_at_mut((i - i0) * ldx);
            for j in i0..i { axpy(&mut tail[..width], -l(i, j), &head[(j - i0) * ldx..(j - i0) * ldx + width]) }
        }
        let rows = &*rows;
        T::add_product(rest, ldx, |i, p| -l(i1 + i, i0 + p), |p, j| rows[p * ldx + j], [n - i1, i1 - i0, width]);
    }
}

/// Solves `U x = y` in place for the `n`×`k` matrix `x`, where U is the upper triangle of `f`, from the last block of rows up. Each
/// block's rows substitute with vector operations, then one matrix product removes them from every row above.
fn backward<T: Scalar + Tiled>(f: &[T], n: usize, x: &mut [T], k: usize) {
    for i0 in (0..n.div_ceil(NS)).rev().map(|block| block * NS) {
        let i1 = (i0 + NS).min(n);
        let (above, rows) = x.split_at_mut(i0 * k);
        let rows = &mut rows[..(i1 - i0) * k];
        for i in (i0..i1).rev() {
            let (mid, below) = rows.split_at_mut((i - i0 + 1) * k);
            let xi = &mut mid[(i - i0) * k..];
            for j in i + 1..i1 { axpy(xi, -f[i * n + j], &below[(j - i - 1) * k..(j - i) * k]) }
            divide(xi, f[i * n + i]);
        }
        let rows = &*rows;
        T::add_product(above, k, |i, p| -f[i * n + i0 + p], |p, j| rows[p * k + j], [i0, i1 - i0, k]);
    }
}

/// Factors columns `c0..c1` of the Cholesky block that `panel` holds transposed, with rows of `rows` items, halving the columns as
/// `lu_panel` does. It gives false where the matrix isn't positive definite.
fn cholesky_panel<T: Scalar + Tiled>(panel: &mut [T], rows: usize, [c0, c1]: [usize; 2]) -> bool {
    if c1 - c0 > LEAF {
        let mid = (c0 + c1) / 2;
        if !cholesky_panel(panel, rows, [c0, mid]) { return false }
        let (left, right) = panel.split_at_mut(mid * rows);
        let l = |r: usize, p: usize| left[(c0 + p) * rows + mid + r];
        T::add_product(&mut right[mid..], rows, |j, p| -l(j, p).conj(), |p, r| l(r, p), [c1 - mid, mid - c0, rows - mid]);
        return cholesky_panel(panel, rows, [mid, c1]);
    }
    for c in c0..c1 {
        let d = panel[c * rows + c];
        if !(d.re() > T::Real::zero()) { return false }
        let (left, right) = panel.split_at_mut((c + 1) * rows);
        let d = T::from_real(d.re().sqrt());
        left[c * rows + c] = d;
        let lc = &mut left[c * rows + c + 1..];
        divide(lc, d);
        // Each later column j loses conj(L[j][c]) times column c, from row j down.
        for (j, column) in right[..(c1 - c - 1) * rows].chunks_exact_mut(rows).enumerate() {
            let below = &lc[j..];
            axpy(&mut column[c + 1 + j..], -below[0].conj(), below);
        }
    }
    true
}
/// `cholesky` for `T`, by blocks of `lu_block(n)` columns. Each block's columns factor in a transposed copy, where each column is contiguous.
/// One matrix product for each later block of columns then updates the lower triangle of the rest.
fn cholesky_factor<T: Scalar + Tiled>(a: &[T], n: usize) -> Option<Vec<T>> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    let mut l = a.to_vec();
    let width = lu_block(n);
    for k0 in (0..n).step_by(width) {
        let k1 = (k0 + width).min(n);
        let (nb, rows, w) = (k1 - k0, n - k0, n - k1);
        let mut panel = transposed(&l, n, k0, k1);
        if !cholesky_panel(&mut panel, rows, [0, nb]) { return None }
        transpose_into(&panel, rows, &mut l[k0 * n + k0..], n, [nb, rows], |x| x);
        if w == 0 { break }
        let l21 = |i: usize, p: usize| panel[p * rows + nb + i];
        T::add_lower_product(&mut l[k1 * n + k1..], n, |i, p| -l21(i, p), |p, j| l21(j, p).conj(), [w, nb, w]);
    }
    for i in 0..n { l[i * n + i + 1..(i + 1) * n].fill(T::zero()) }
    Some(l)
}

/// Turns `x` into a Householder vector `v` with `v[0] = 1`, and gives `τ`, so that `I - τ̄ v vᴴ` maps `x` to a real multiple of its
/// first unit vector. `x[0]` keeps that multiple and the rest of `x` keeps `v`, as LAPACK's `zlarfg` leaves them. Where `x` is zero
/// after its first item, and that item is real, `τ` is 0.
fn reflector<T: Scalar>(x: &mut [T]) -> T {
    let (head, rest) = x.split_at_mut(1);
    let (x0, below) = (head[0], dotc(rest, rest).re());
    let zero = T::Real::zero();
    if below == zero && x0.im() == zero { return T::zero() }
    let (re, im) = (x0.re(), x0.im());
    let norm = re.mla::<FUSED>(re, im.mla::<FUSED>(im, below)).sqrt();
    let beta = T::from_real(if re >= zero { -norm } else { norm });
    let scale = (x0 - beta).reciprocal();
    for v in rest.iter_mut() { *v = *v * scale }
    head[0] = beta;
    (beta - x0).div_real(beta.re())
}

/// Applies the reflection `I - τ̄ v vᴴ` to `x`, where `v` is 1 at `x[0]` and `tail` after it.
#[inline(always)]
fn apply_reflector<T: Scalar>(x: &mut [T], tau: T, tail: &[T]) {
    let tw = tau.conj() * (x[0] + dotc(tail, &x[1..]));
    x[0] = x[0] - tw;
    axpy(&mut x[1..], -tw, tail);
}

/// Row `r` of the Householder vector in column `c` of a QR block that `panel` holds transposed, with rows of `rows` items: 1 at row `c`,
/// the panel's items below it, and 0 above it.
fn householder<T: Scalar>(panel: &[T], rows: usize, c: usize, r: usize) -> T {
    match r.cmp(&c) {
        Ordering::Less => T::zero(),
        Ordering::Equal => T::one(),
        Ordering::Greater => panel[c * rows + r],
    }
}

/// The upper triangular T of the compact WY form `I - V T Vᴴ` of a QR block's reflections, applied first to last, as LAPACK's `zlarft`
/// gives it. `panel` holds the block transposed, with rows of `rows` items, and `taus` gives each reflection's `τ`.
fn triangular<T: Scalar>(panel: &[T], taus: &[T], rows: usize) -> Vec<T> {
    let nb = taus.len();
    let mut t = vec![T::zero(); nb * nb];
    for c in 0..nb {
        let tau = taus[c];
        t[c * nb + c] = tau;
        if tau == T::zero() { continue }
        let vc = &panel[c * rows + c + 1..(c + 1) * rows];
        // Each earlier vector j against v_c, which is 1 at row c: v_jᴴ v_c.
        let s: Vec<T> = (0..c).map(|j| panel[j * rows + c].conj() + dotc(&panel[j * rows + c + 1..(j + 1) * rows], vc)).collect();
        for i in 0..c { t[i * nb + c] = -tau * dot(&t[i * nb + i..i * nb + c], &s[i..c]) }
    }
    t
}

/// One block of a product of Householder reflections, `I - V T Vᴴ`, for reflections `k0` to `k0 + nb - 1`, acting on rows `k0` on.
/// Column `c` of V is reflection `k0 + c`, which `householder` reads from row `c` of `panel`. T is `triangular`'s.
struct Block<T> {
    k0: usize,
    nb: usize,
    panel: Vec<T>,
    t: Vec<T>,
}

impl<T: Scalar + Tiled> Block<T> {
    /// The block of reflections `k0` on, with the vectors in the rows of `panel` and the τs `taus`.
    fn new(k0: usize, panel: Vec<T>, taus: &[T]) -> Self {
        let nb = taus.len();
        let t = triangular(&panel, taus, panel.len() / nb);
        Self { k0, nb, panel, t }
    }

    /// Applies `I - V T Vᴴ`, or its adjoint `I - V Tᴴ Vᴴ` where `adjoint` is true, to the columns from `c0` on of the matrix `x`, whose
    /// rows hold `ld` items.
    fn apply(&self, x: &mut [T], ld: usize, c0: usize, adjoint: bool) {
        let (k0, nb, rows, cols) = (self.k0, self.nb, self.panel.len() / self.nb, ld - c0);
        // `triangular` stores T's zeros, so neither T nor Tᴴ needs a mask.
        let t = if adjoint { Cow::Owned(transpose_by(&self.t, nb, nb, T::conj)) } else { Cow::Borrowed(&self.t[..]) };
        let v = |r: usize, c: usize| householder(&self.panel, rows, c, r);
        let mut w = vec![T::zero(); nb * cols];
        T::add_product(&mut w, cols, |i, r| v(r, i).conj(), |r, j| x[(k0 + r) * ld + c0 + j], [nb, rows, cols]);
        let mut tw = vec![T::zero(); nb * cols];
        T::add_product(&mut tw, cols, |i, c| t[i * nb + c], |c, j| w[c * cols + j], [nb, nb, cols]);
        T::add_product(&mut x[k0 * ld + c0..], ld, |r, c| -v(r, c), |c, j| tw[c * cols + j], [rows, nb, cols]);
    }
}

/// Householder reflections of `m`-item columns, in blocks, as the `m`×`m` product Q of their matrices, first to last.
struct Reflections<T> {
    m: usize,
    blocks: Vec<Block<T>>,
}

/// The QR factorization of `a`: its reflections, by blocks of `qr_block(p)` columns, and the `p`×`n` upper triangular R, where `p` is the
/// smaller of `m` and `n`. Each block's columns factor in a transposed copy, where each column is contiguous. The block's reflections then
/// update the columns after it, in the WY form.
fn reflect<T: Scalar + Tiled>(a: &[T], [m, n]: [usize; 2]) -> (Reflections<T>, Vec<T>) {
    assert_eq!(a.len(), m * n, "`a` must hold `m`×`n` items");
    let p = m.min(n);
    let mut f = a.to_vec();
    let scale = safe_scale(a);
    if let Some(s) = scale { for x in &mut f { *x = x.mul_real(s) } }
    let mut blocks = Vec::new();
    let width = qr_block(p);
    for k0 in (0..p).step_by(width) {
        let k1 = (k0 + width).min(p);
        let (nb, rows) = (k1 - k0, m - k0);
        let mut panel = transposed(&f, n, k0, k1);
        let mut taus = Vec::with_capacity(nb);
        for c in 0..nb {
            let (left, right) = panel.split_at_mut((c + 1) * rows);
            let v = &mut left[c * rows + c..];
            let tau = reflector(v);
            taus.push(tau);
            if tau == T::zero() { continue }
            // The block's later columns: A ← A - τ̄ v (vᴴ A), where v is 1 at row c.
            for column in right.chunks_exact_mut(rows) {
                apply_reflector(&mut column[c..], tau, &v[1..]);
            }
        }
        for r in 0..nb {
            for c in r..nb { f[(k0 + r) * n + k0 + c] = panel[c * rows + r] }
        }
        let block = Block::new(k0, panel, &taus);
        // The columns after the block: A2 ← (I - V Tᴴ Vᴴ) A2.
        if k1 < n { block.apply(&mut f, n, k1, true) }
        blocks.push(block);
    }
    // R returns to `a`'s scale. The reflections don't depend on the scale.
    let r = (0..p * n).map(|k| if k % n < k / n { T::zero() } else { scale.map_or(f[k], |s| f[k].div_real(s)) }).collect();
    (Reflections { m, blocks }, r)
}

impl<T: Scalar + Tiled> Reflections<T> {
    /// Each reflection, first to last: the row it acts from, its τ, and its vector after the 1 at that row.
    fn each(&self) -> impl DoubleEndedIterator<Item = (usize, T, &[T])> {
        self.blocks.iter().flat_map(move |b| {
            let rows = self.m - b.k0;
            (0..b.nb).map(move |c| (b.k0 + c, b.t[c * b.nb + c], &b.panel[c * rows + c + 1..(c + 1) * rows]))
        })
    }

    /// Q's first `p` columns, where `p` is the number of reflections: the reflections applied, last block first, to those columns of the
    /// identity. Before a block starting at k0, columns before k0 are still columns of the identity, which it leaves unchanged. Up to
    /// `ONE_AT_A_TIME` columns, each reflection applies alone to the rows of Qᵀ, which avoids the products and their packed buffers that a
    /// block needs.
    fn q(&self) -> Vec<T> {
        let (m, p) = (self.m, self.blocks.last().map_or(0, |b| b.k0 + b.nb));
        if p <= ONE_AT_A_TIME {
            let mut qt: Vec<T> = (0..p * m).map(|k| if k / m == k % m { T::one() } else { T::zero() }).collect();
            for (row, tau, v) in self.each().rev() {
                if tau == T::zero() { continue }
                // `apply_reflector` applies I - τ̄ v vᴴ, and Q's reflections are I - τ v vᴴ.
                for column in qt[row * m..].chunks_exact_mut(m) { apply_reflector(&mut column[row..], tau.conj(), v) }
            }
            return transpose(&qt, p, m);
        }
        let mut q: Vec<T> = (0..m * p).map(|k| if k / p == k % p { T::one() } else { T::zero() }).collect();
        for block in self.blocks.iter().rev() { block.apply(&mut q, p, block.k0, false) }
        q
    }

    /// `Q x` for the `m`×`k` matrix `x`, in place.
    fn apply(&self, x: &mut [T], k: usize) {
        for block in self.blocks.iter().rev() { block.apply(x, k, 0, false) }
    }

    /// `Qᴴ x` for the `m`×`k` matrix `x`, in place: the blocks' adjoints, first block first. Up to `FEW_COLUMNS` columns, each reflection
    /// applies alone to each column of a transposed copy, where each column is contiguous.
    fn apply_adjoint(&self, x: &mut [T], k: usize) {
        if k > FEW_COLUMNS {
            for block in &self.blocks { block.apply(x, k, 0, true) }
            return;
        }
        let m = self.m;
        let mut xt = transpose(x, m, k);
        for (row, tau, v) in self.each() {
            if tau == T::zero() { continue }
            for column in xt.chunks_exact_mut(m) { apply_reflector(&mut column[row..], tau, v) }
        }
        transpose_into(&xt, m, x, k, [k, m], |x| x);
    }
}

/// `qr` and `qr_complete` for `T`: Q and R from the reflections. A complete Q applies the reflections to the `m`×`m` identity, and R
/// gains rows of zeros to be `m`×`n`.
fn qr_factor<T: Scalar + Tiled>(a: &[T], [m, n]: [usize; 2], complete: bool) -> (Vec<T>, Vec<T>) {
    let (reflections, mut r) = reflect(a, [m, n]);
    if !complete { return (reflections.q(), r) }
    let mut q = identity(m);
    reflections.apply(&mut q, m);
    r.resize(m * n, T::zero());
    (q, r)
}

/// `Svd::solve` and `Svd::pseudoinverse` for `T`.
fn svd_solve<T: Scalar + Tiled>(svd: &Svd<T>, b: Option<&[T]>, k: usize) -> Vec<T> {
    let ([m, n], p) = (svd.size, svd.s.len());
    if let Some(b) = b { assert_eq!(b.len(), m * k, "`b` must hold `m`×`k` items") }
    assert!(svd.u.len() == m * p && svd.v.len() == n * p, "the SVD must hold its singular vectors");
    pseudo_solve(svd, b, k, m.max(n))
}

/// `V diag(1/s) Uᴴ b` for the factors that `svd` holds and a `b` of `k` columns, or `V diag(1/s) Uᴴ` without `b`, where `1/s` is zero
/// for each singular value no greater than `ε` times `scale` times the largest.
fn pseudo_solve<T: Scalar + Tiled>(svd: &Svd<T>, b: Option<&[T]>, k: usize, scale: usize) -> Vec<T> {
    let ([m, n], p) = (svd.size, svd.s.len());
    if k == 0 { return Vec::new() }
    let floor = cutoff(&svd.s, scale);
    let mut y = match b {
        Some(b) => {
            let mut y = vec![T::zero(); p * k];
            T::add_product(&mut y, k, |i, r| svd.u[r * p + i].conj(), |r, j| b[r * k + j], [p, m, k]);
            y
        }
        None => adjoint(&svd.u, m, p),
    };
    for (row, &s) in y.chunks_exact_mut(k).zip(&svd.s) {
        let d = if s > floor { s.recip() } else { T::Real::zero() };
        for x in row { *x = x.mul_real(d) }
    }
    let mut x = vec![T::zero(); n * k];
    T::add_product(&mut x, k, |i, r| svd.v[i * p + r], |r, j| y[r * k + j], [n, p, k]);
    x
}

/// The size below which a singular value counts as zero, for a matrix whose singular values `s` descend and whose larger side has `scale`
/// items: `ε` times `scale` times the largest, as in NumPy's `lstsq`.
fn cutoff<R: crate::Float>(s: &[R], scale: usize) -> R { R::epsilon() * R::narrow(scale as f64) * s.first().copied().unwrap_or(R::zero()) }

/// `lstsq` for `T`. A matrix with at least as many rows as columns reduces to the R of its QR factorization, and `b` to `Qᴴ b`, without
/// forming Q. A wider matrix solves through its own SVD.
fn lstsq_solve<T: Scalar + Tiled>(a: &[T], b: &[T], [m, n]: [usize; 2], k: usize) -> (Vec<T>, Vec<T::Real>)
where
    T::Real: Tiled,
{
    assert_eq!(b.len(), m * k, "`b` must hold `m`×`k` items");
    if m < n {
        let svd = svd::svd_factor(a, [m, n], true);
        return (pseudo_solve(&svd, Some(b), k, n), svd.s);
    }
    let (reflections, r) = reflect(a, [m, n]);
    let mut y = b.to_vec();
    reflections.apply_adjoint(&mut y, k);
    let svd = svd::svd_factor(&r, [n, n], true);
    (pseudo_solve(&svd, Some(&y[..n * k]), k, m), svd.s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A real item type of the tests: `f32` or `f64`, whose real quantities, such as singular values, have its own type.
    pub(super) trait RealItem: Factor + Native + Scalar<Real = Self> + Tiled + std::fmt::Debug {}
    impl<T: Factor + Native + Scalar<Real = T> + Tiled + std::fmt::Debug> RealItem for T {}

    /// An `m`×`n` matrix of numbers from -1 to 1, from a linear congruential generator.
    pub(super) fn random<T: RealItem>(m: usize, n: usize, seed: u64) -> Vec<T> {
        let mut x = seed;
        (0..m * n)
            .map(|_| {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                T::narrow((x >> 11) as f64 / (1u64 << 52) as f64 - 1.0)
            })
            .collect()
    }

    pub(super) fn wide<T: RealItem>(x: &[T]) -> Vec<f64> { x.iter().map(|v| v.wide()).collect() }

    /// The product of the `m`×`k` matrix `a` and the `k`×`n` matrix `b`, in f64, item by item.
    pub(super) fn product(a: &[f64], b: &[f64], [m, k, n]: [usize; 3]) -> Vec<f64> {
        (0..m * n).map(|ij| (0..k).map(|p| a[ij / n * k + p] * b[p * n + ij % n]).sum()).collect()
    }

    pub(super) fn norm(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum::<f64>().sqrt() }

    /// The norm of `x - y`, relative to `scale`.
    pub(super) fn distance(x: &[f64], y: &[f64], scale: f64) -> f64 { norm(&x.iter().zip(y).map(|(a, b)| a - b).collect::<Vec<_>>()) / scale }

    /// How far the `rows`×`cols` matrix `q` is from having orthonormal columns.
    pub(super) fn orthogonality(q: &[f64], rows: usize, cols: usize) -> f64 {
        let qt = transpose(q, rows, cols);
        distance(&product(&qt, q, [cols, rows, cols]), &identity(cols), 1.0)
    }

    /// Checks each decomposition at `T` on matrices either side of the block width, where `tolerance` is a few times `T`'s epsilon.
    fn decompositions<T: RealItem>(tolerance: f64) {
        let nb = lu_block(0);
        for n in [0, 1, 2, 5, nb - 1, nb, nb + 1, 2 * nb + 7] {
            let bound = tolerance * (n as f64).max(1.0);
            let a = random::<T>(n, n, n as u64 + 1);
            let aw = wide(&a);
            // L U is A with its rows permuted.
            let f = lu(&a, n);
            let factors = wide(&f.factors);
            let l: Vec<f64> = (0..n * n).map(|k| match (k / n).cmp(&(k % n)) { Ordering::Greater => factors[k], Ordering::Equal => 1.0, Ordering::Less => 0.0 }).collect();
            let u: Vec<f64> = (0..n * n).map(|k| if k % n >= k / n { factors[k] } else { 0.0 }).collect();
            let pa: Vec<f64> = f.perm.iter().flat_map(|&p| aw[p * n..(p + 1) * n].to_vec()).collect();
            assert!(distance(&product(&l, &u, [n; 3]), &pa, norm(&aw).max(1.0)) < bound, "lu at {n}");
            // Solutions have small residuals for their size.
            for k in [1, 3] {
                let b = random::<T>(n, k, 99);
                let x = wide(&f.solve(&b, k));
                assert!(distance(&product(&aw, &x, [n, n, k]), &wide(&b), (norm(&aw) * norm(&x)).max(1.0)) < bound, "solve at {n}, {k}");
            }
            let inverse = wide(&f.inverse());
            assert!(distance(&product(&aw, &inverse, [n; 3]), &identity(n), (norm(&aw) * norm(&inverse)).max(1.0)) < bound, "inverse at {n}");
            // Cholesky of a positive definite matrix, and none of one that isn't.
            let spd: Vec<T> = product(&aw, &transpose(&aw, n, n), [n; 3]).iter().enumerate().map(|(k, &v)| T::narrow(v + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
            let l = wide(&cholesky(&spd, n).expect("a positive definite matrix"));
            assert!((0..n * n).all(|k| k % n <= k / n || l[k] == 0.0), "cholesky's upper triangle at {n}");
            assert!(distance(&product(&l, &transpose(&l, n, n), [n; 3]), &wide(&spd), norm(&wide(&spd)).max(1.0)) < bound, "cholesky at {n}");
            if n > 0 { assert_eq!(cholesky(&spd.iter().map(|&v| -v).collect::<Vec<_>>(), n), None, "cholesky of a negative definite matrix") }
            // QR and the SVD, of square, tall and wide matrices.
            for (m, c) in [(n, n), (n + 5, n), (n, n + 5)] {
                let a = random::<T>(m, c, (m * 31 + c) as u64);
                let (aw, p, scale) = (wide(&a), m.min(c), norm(&wide(&a)).max(1.0));
                let (q, r) = qr(&a, [m, c]);
                let (q, r) = (wide(&q), wide(&r));
                assert!(distance(&product(&q, &r, [m, p, c]), &aw, scale) < bound, "qr at {m}×{c}");
                assert!(orthogonality(&q, m, p) < bound, "qr's q at {m}×{c}");
                assert!((0..p * c).all(|k| k % c >= k / c || r[k] == 0.0), "qr's r at {m}×{c}");
                let f = svd(&a, [m, c]);
                let (u, s, v) = (wide(&f.u), wide(&f.s), wide(&f.v));
                assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0), "svd's s at {m}×{c}");
                let us: Vec<f64> = (0..m * p).map(|k| u[k] * s[k % p]).collect();
                assert!(distance(&product(&us, &transpose(&v, c, p), [m, p, c]), &aw, scale) < bound, "svd at {m}×{c}");
                // The accumulated rotations' rounding errors add up over the sweeps, in V, or in U for a wide matrix.
                let (ou, ov) = (orthogonality(&u, m, p), orthogonality(&v, c, p));
                assert!(ou < 4.0 * bound && ov < 4.0 * bound, "svd's u and v at {m}×{c}: {ou:e} and {ov:e}, against {bound:e}");
                assert!(distance(&wide(&singular_values(&a, [m, c])), &wide(&f.s), scale) < bound, "singular values alone at {m}×{c}");
                // A least-squares solution leaves a residual orthogonal to A's columns, and the pseudoinverse inverts A on its range.
                if m >= c && c > 0 {
                    let b = random::<T>(m, 2, 7);
                    let x = wide(&f.solve(&b, 2));
                    let residual: Vec<f64> = product(&aw, &x, [m, c, 2]).iter().zip(wide(&b)).map(|(ax, b)| ax - b).collect();
                    let normal = product(&transpose(&aw, m, c), &residual, [c, m, 2]);
                    assert!(norm(&normal) / (scale * (scale * norm(&x) + norm(&wide(&b)))) < bound, "svd's solve at {m}×{c}");
                }
                // `lstsq` leaves its residual orthogonal to A's columns too, and gives the SVD's singular values.
                if m > 0 && c > 0 {
                    let b = random::<T>(m, 2, 11);
                    let (x, s2) = lstsq(&a, &b, [m, c], 2);
                    let x = wide(&x);
                    let residual: Vec<f64> = product(&aw, &x, [m, c, 2]).iter().zip(wide(&b)).map(|(ax, b)| ax - b).collect();
                    let normal = product(&transpose(&aw, m, c), &residual, [c, m, 2]);
                    assert!(norm(&normal) / (scale * (scale * norm(&x) + norm(&wide(&b)))) < bound, "lstsq at {m}×{c}");
                    assert!(distance(&wide(&s2), &s, s[0]) < bound, "lstsq's singular values at {m}×{c}");
                }
                let pinv = wide(&f.pseudoinverse());
                assert!(distance(&product(&product(&aw, &pinv, [m, c, m]), &aw, [m, m, c]), &aw, scale) < bound, "pseudoinverse at {m}×{c}");
            }
        }
        // A zero column gives a singular value within rounding of zero, and U stays orthonormal.
        let n = 9;
        let a: Vec<T> = random::<T>(n, n, 5).iter().enumerate().map(|(k, &v)| if k % n == 3 { T::zero() } else { v }).collect();
        let f = svd(&a, [n, n]);
        assert!(f.s[n - 1].wide() <= tolerance * f.s[0].wide(), "a zero column's singular value");
        assert!(orthogonality(&wide(&f.u), n, n) < tolerance * n as f64, "u with a zero column");
        // A bidiagonal matrix with a zero on its diagonal reaches the QR algorithm unchanged, which rotates that row's superdiagonal
        // item away before it iterates.
        let (d, e) = ([3.0, 0.0, 2.0, 0.5, 1.0], [1.0, 1.0, 0.25, 2.0]);
        let b: Vec<T> = (0..25).map(|k| T::narrow(match (k / 5, k % 5) { (i, j) if i == j => d[i], (i, j) if j == i + 1 => e[i], _ => 0.0 })).collect();
        let f = svd(&b, [5, 5]);
        let us: Vec<f64> = (0..25).map(|k| wide(&f.u)[k] * wide(&f.s)[k % 5]).collect();
        assert!(distance(&product(&us, &transpose(&wide(&f.v), 5, 5), [5; 3]), &wide(&b), norm(&wide(&b))) < 8.0 * tolerance, "svd of a bidiagonal with a zero");
        assert!(orthogonality(&wide(&f.u), 5, 5) < 8.0 * tolerance, "u of a bidiagonal with a zero");
    }

    #[test]
    fn decompositions_have_small_residuals() {
        decompositions::<f64>(4.0 * f64::EPSILON);
        decompositions::<f32>(4.0 * f32::EPSILON as f64);
    }

    /// Checks the decompositions at `T` on LAPACK's hard matrix types, where `tolerance` is a few times `T`'s epsilon. A matrix scaled
    /// by a power of two near underflow or overflow gives factors scaled exactly in step. A zero column gives finite LU factors with a
    /// zero pivot. A subnormal pivot gives finite multipliers.
    fn hard_matrices<T: RealItem>(tolerance: f64) {
        let n = 40;
        let tiny = T::min_positive_value() / T::epsilon();
        let (a, b) = (random::<T>(n, n, 17), random::<T>(n, 3, 19));
        let aw = wide(&a);
        let spd: Vec<T> = product(&aw, &transpose(&aw, n, n), [n; 3]).iter().enumerate().map(|(k, &v)| T::narrow(v + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
        for s in [tiny, tiny.recip()] {
            let times = |x: &[T], c: T| x.iter().map(|&v| v * c).collect::<Vec<T>>();
            let (f, g) = (lu(&a, n), lu(&times(&a, s), n));
            assert_eq!(g.perm, f.perm, "lu's permutation scaled by {s:?}");
            assert_eq!(g.factors, f.factors.iter().enumerate().map(|(k, &v)| if k % n >= k / n { v * s } else { v }).collect::<Vec<T>>(), "lu scaled by {s:?}");
            assert_eq!(g.solve(&times(&b, s), 3), f.solve(&b, 3), "lu's solve scaled by {s:?}");
            assert_eq!(times(&g.inverse(), s), f.inverse(), "lu's inverse scaled by {s:?}");
            // An even power of two has an exact square root.
            let c = if s.sqrt() * s.sqrt() == s { s } else { s + s };
            assert_eq!(cholesky(&times(&spd, c), n), cholesky(&spd, n).map(|l| times(&l, c.sqrt())), "cholesky scaled by {c:?}");
            for (rows, cols) in [(n, n), (n + 5, n), (n, n + 5)] {
                let a = random::<T>(rows, cols, (rows * 31 + cols) as u64);
                let b = random::<T>(rows, 2, 23);
                let ((q, r), (qs, rs)) = (qr(&a, [rows, cols]), qr(&times(&a, s), [rows, cols]));
                assert_eq!((qs, rs), (q, times(&r, s)), "qr scaled by {s:?} at {rows}×{cols}");
                let (f, g) = (svd(&a, [rows, cols]), svd(&times(&a, s), [rows, cols]));
                assert_eq!((g.u, g.s, g.v), (f.u, times(&f.s, s), f.v), "svd scaled by {s:?} at {rows}×{cols}");
                let ((x, sv), (xs, svs)) = (lstsq(&a, &b, [rows, cols], 2), lstsq(&times(&a, s), &times(&b, s), [rows, cols], 2));
                assert_eq!((xs, svs), (x, times(&sv, s)), "lstsq scaled by {s:?} at {rows}×{cols}");
            }
        }
        let n = 2 * lu_block(0) + 7;
        for zero in [0, n / 2, n - 1] {
            let a: Vec<T> = random::<T>(n, n, 29).iter().enumerate().map(|(k, &v)| if k % n == zero { T::zero() } else { v }).collect();
            let (aw, f) = (wide(&a), lu(&a, n));
            let factors = wide(&f.factors);
            assert!(factors.iter().all(|v| v.is_finite()) && f.diagonal().any(|d| d == T::zero()), "lu with zero column {zero}");
            let l: Vec<f64> = (0..n * n).map(|k| match (k / n).cmp(&(k % n)) { Ordering::Greater => factors[k], Ordering::Equal => 1.0, Ordering::Less => 0.0 }).collect();
            let u: Vec<f64> = (0..n * n).map(|k| if k % n >= k / n { factors[k] } else { 0.0 }).collect();
            let pa: Vec<f64> = f.perm.iter().flat_map(|&p| aw[p * n..(p + 1) * n].to_vec()).collect();
            assert!(distance(&product(&l, &u, [n; 3]), &pa, norm(&aw)) < tolerance * n as f64, "lu's residual with zero column {zero}");
        }
        let (p, q) = (T::min_positive_value() / T::narrow(1024.0), T::min_positive_value() / T::narrow(1048576.0));
        assert_eq!(lu(&[p, T::one(), q, T::one()], 2).factors, [p, T::one(), q / p, T::one() - q / p], "lu with a subnormal pivot");
    }

    #[test]
    fn decompositions_survive_hard_matrices() {
        hard_matrices::<f64>(4.0 * f64::EPSILON);
        hard_matrices::<f32>(4.0 * f32::EPSILON as f64);
    }

    /// Checks `solve`, `pinv`, `qr_complete`, the determinants and `hermitian_eigenvalues` at `T`, where `tolerance` is a few times
    /// `T`'s epsilon. The singular matrices repeat columns, so their rank is exact.
    fn solves_and_determinants<T: RealItem>(tolerance: f64) {
        let bound = 64.0 * tolerance;
        let base = random::<T>(9, 4, 7);
        // Columns 4 to 6 repeat columns 0 to 2, so each matrix has rank 4: tall, square, wide, and a regular square one.
        let repeat = |rows: usize| -> Vec<T> { (0..rows * 7).map(|k| base[k / 7 * 4 + k % 7 % 4]).collect() };
        let regular = random::<T>(6, 6, 11);
        for (a, [m, n], r) in [(repeat(9), [9, 7], 4), (repeat(7), [7, 7], 4), (transpose(&repeat(9), 9, 7), [7, 9], 4), (regular.clone(), [6, 6], 6)] {
            let (aw, scale) = (wide(&a), norm(&wide(&a)));
            assert_eq!(rank(&a, [m, n]), r, "rank at {m}×{n}");
            // The pseudoinverse meets the Moore-Penrose conditions A A⁺ A = A and A⁺ A A⁺ = A⁺.
            let p = wide(&pinv(&a, [m, n]));
            let ap = product(&aw, &p, [m, n, m]);
            assert!(distance(&product(&ap, &aw, [m, m, n]), &aw, scale) < bound, "A A⁺ A at {m}×{n}");
            assert!(distance(&product(&p, &ap, [n, m, m]), &p, norm(&p)) < bound, "A⁺ A A⁺ at {m}×{n}");
            // `solve` gives the least-norm least-squares solution, A⁺ b.
            let b = random::<T>(m, 2, 13);
            let x = wide(&solve(&a, &b, [m, n], 2));
            let expected = product(&p, &wide(&b), [n, m, 2]);
            assert!(distance(&x, &expected, norm(&expected)) < bound, "solve at {m}×{n}");
            // The complete QR: Q is square and orthogonal, R's rows below the thin ones are zero, and Q R is A.
            let (q, r) = qr_complete(&a, [m, n]);
            let (q, r, thin) = (wide(&q), wide(&r), m.min(n));
            assert!(orthogonality(&q, m, m) < bound, "complete Q at {m}×{n}");
            assert!(r[thin * n..].iter().all(|&v| v == 0.0), "complete R's lower rows at {m}×{n}");
            assert!(distance(&product(&q, &r, [m, m, n]), &aw, scale) < bound, "complete Q R at {m}×{n}");
        }
        // A regular matrix's determinant times its inverse's is 1, and the log-determinant agrees with it.
        let det = |a: &[T], n: usize| lu(a, n).determinant().wide();
        let inverse = pinv(&regular, [6, 6]);
        assert!((det(&regular, 6) * det(&inverse, 6) - 1.0).abs() < bound, "determinants of a matrix and its inverse");
        let (sign, log) = lu(&regular, 6).log_determinant();
        assert!((sign.wide() * log.wide().exp() / det(&regular, 6) - 1.0).abs() < bound, "log-determinant");
        // diag(1, 2, 3, 4) with two rows swapped has determinant -24, and with a zero column, 0 and a logarithm of -∞.
        let swapped: Vec<T> = [0., 2., 0., 0., 1., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.].map(T::narrow).to_vec();
        assert_eq!(det(&swapped, 4), -24.0, "an odd permutation's sign");
        let singular: Vec<T> = [1., 0., 3., 0.].map(T::narrow).to_vec();
        let (sign, log) = lu(&singular, 2).log_determinant();
        assert_eq!((det(&singular, 2), sign.wide(), log.wide()), (0.0, 0.0, f64::NEG_INFINITY), "a singular matrix");
        // Eigenvalues alone match `eigh`'s, either side of the size where `eigh` switches to divide and conquer.
        for n in [6, 70] {
            let a = random::<T>(n, n, n as u64);
            let s: Vec<T> = (0..n * n).map(|k| a[k] + a[k % n * n + k / n]).collect();
            let (values, expected) = (wide(&hermitian_eigenvalues(&s, n).unwrap()), wide(&eigh(&s, n).unwrap().values));
            assert!(distance(&values, &expected, norm(&expected)) < bound, "eigenvalues alone at {n}");
        }
    }

    #[test]
    fn solves_and_determinants_never_fail() {
        solves_and_determinants::<f64>(4.0 * f64::EPSILON);
        solves_and_determinants::<f32>(4.0 * f32::EPSILON as f64);
    }

    /// Checks LU, Cholesky and QR at `T` on matrices large enough for the wider blocks, each leaving a partial last block, where
    /// `tolerance` is a few times `T`'s epsilon. Products with random vectors keep each check to O(n²) time.
    fn blocked<T: RealItem>(tolerance: f64) {
        let times = |a: &[f64], v: &[f64]| product(a, v, [a.len() / v.len(), v.len(), 1]);
        for n in [530, 1030] {
            let bound = tolerance * n as f64;
            let a = random::<T>(n, n, n as u64);
            let (aw, x) = (wide(&a), wide(&random::<T>(n, 1, 3)));
            let f = lu(&a, n);
            let (l, u) = (wide(&f.lower()), wide(&f.upper()));
            let ax = times(&aw, &x);
            let pax: Vec<f64> = f.perm.iter().map(|&p| ax[p]).collect();
            assert!(distance(&times(&l, &times(&u, &x)), &pax, norm(&aw) * norm(&x)) < bound, "lu at {n}");
            let b = random::<T>(n, 3, 5);
            let solved = wide(&f.solve(&b, 3));
            assert!(distance(&product(&aw, &solved, [n, n, 3]), &wide(&b), norm(&aw) * norm(&solved)) < bound, "solve at {n}");
            let inverse = wide(&f.inverse());
            assert!(distance(&times(&aw, &times(&inverse, &x)), &x, norm(&aw) * norm(&inverse) * norm(&x)) < bound, "inverse at {n}");
            // (A + Aᵀ)/2 + n I is diagonally dominant with a positive diagonal, so it's positive definite.
            let spd: Vec<T> = (0..n * n).map(|k| T::narrow((aw[k] + aw[k % n * n + k / n]) / 2.0 + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
            let (sw, l) = (wide(&spd), wide(&cholesky(&spd, n).expect("a positive definite matrix")));
            assert!((0..n * n).all(|k| k % n <= k / n || l[k] == 0.0), "cholesky's upper triangle at {n}");
            assert!(distance(&times(&l, &times(&transpose(&l, n, n), &x)), &times(&sw, &x), norm(&sw) * norm(&x)) < bound, "cholesky at {n}");
            for rows in [n, n + 5] {
                let a = random::<T>(rows, n, rows as u64 + 7);
                let (aw, (reflections, r)) = (wide(&a), reflect(&a, [rows, n]));
                let (q, r) = (wide(&reflections.q()), wide(&r));
                assert!(distance(&times(&q, &times(&r, &x)), &times(&aw, &x), norm(&aw) * norm(&x)) < bound, "qr at {rows}×{n}");
                assert!((0..n * n).all(|k| k % n >= k / n || r[k] == 0.0), "qr's r at {rows}×{n}");
                let (y, qt) = (wide(&random::<T>(n, 1, 9)), transpose(&q, rows, n));
                assert!(distance(&times(&qt, &times(&q, &y)), &y, norm(&y)) < bound, "qr's q at {rows}×{n}");
                for k in [2, FEW_COLUMNS + 1] {
                    let b = random::<T>(rows, k, 13);
                    let mut qtb = b.clone();
                    reflections.apply_adjoint(&mut qtb, k);
                    let bw = wide(&b);
                    assert!(distance(&wide(&qtb)[..n * k], &product(&qt, &bw, [n, rows, k]), norm(&bw)) < bound, "qr's Qᴴ b at {rows}×{n}×{k}");
                }
            }
        }
    }

    #[test]
    fn blocked_decompositions_have_small_residuals() {
        blocked::<f64>(4.0 * f64::EPSILON);
        blocked::<f32>(4.0 * f32::EPSILON as f64);
    }

    type C64 = num_complex::Complex<f64>;

    /// An `m`×`n` matrix of complex numbers whose parts lie between -1 and 1.
    pub(super) fn complex_random<R: RealItem>(m: usize, n: usize, seed: u64) -> Vec<num_complex::Complex<R>> {
        random::<R>(m, n, seed).into_iter().zip(random::<R>(m, n, seed + 1000)).map(|(re, im)| num_complex::Complex::new(re, im)).collect()
    }

    pub(super) fn complex_wide<R: RealItem>(x: &[num_complex::Complex<R>]) -> Vec<C64> { x.iter().map(|z| C64::new(z.re.wide(), z.im.wide())).collect() }

    /// The product of the `m`×`k` matrix `a` and the `k`×`n` matrix `b`, item by item.
    fn complex_product(a: &[C64], b: &[C64], [m, k, n]: [usize; 3]) -> Vec<C64> {
        (0..m * n).map(|ij| (0..k).map(|p| a[ij / n * k + p] * b[p * n + ij % n]).sum()).collect()
    }

    fn complex_norm(x: &[C64]) -> f64 { x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() }

    /// The norm of `x - y`, relative to `scale`.
    fn complex_distance(x: &[C64], y: &[C64], scale: f64) -> f64 { x.iter().zip(y).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt() / scale }

    /// Checks LU with its solves and inverse, Cholesky and QR at complex items with parts of type `R`, on the matrices `decompositions`
    /// takes, where `tolerance` is a few times `R`'s epsilon.
    fn complex_decompositions<R: RealItem>(tolerance: f64) {
        let (zero, one) = (C64::new(0.0, 0.0), C64::new(1.0, 0.0));
        let nb = lu_block(0);
        for n in [0, 1, 2, 5, nb - 1, nb, nb + 1, 2 * nb + 7] {
            let bound = tolerance * (n as f64).max(1.0);
            let a = complex_random::<R>(n, n, n as u64 + 1);
            let (aw, i) = (complex_wide(&a), identity::<C64>(n));
            let scale = complex_norm(&aw).max(1.0);
            // L U is A with its rows permuted.
            let f = lu_factor(&a, n);
            let factors = complex_wide(&f.factors);
            let l: Vec<C64> = (0..n * n).map(|k| match (k / n).cmp(&(k % n)) { Ordering::Greater => factors[k], Ordering::Equal => one, Ordering::Less => zero }).collect();
            let u: Vec<C64> = (0..n * n).map(|k| if k % n >= k / n { factors[k] } else { zero }).collect();
            let pa: Vec<C64> = f.perm.iter().flat_map(|&p| aw[p * n..(p + 1) * n].to_vec()).collect();
            assert!(complex_distance(&complex_product(&l, &u, [n; 3]), &pa, scale) < bound, "complex lu at {n}");
            for k in [1, 3] {
                let b = complex_random::<R>(n, k, 99);
                let x = complex_wide(&lu_solve(&f, &b, k));
                assert!(complex_distance(&complex_product(&aw, &x, [n, n, k]), &complex_wide(&b), (scale * complex_norm(&x)).max(1.0)) < bound, "complex solve at {n}, {k}");
            }
            let inverse = complex_wide(&lu_inverse(&f));
            assert!(complex_distance(&complex_product(&aw, &inverse, [n; 3]), &i, (scale * complex_norm(&inverse)).max(1.0)) < bound, "complex inverse at {n}");
            // Cholesky of a Hermitian positive definite matrix gives L with a real, positive diagonal. A negative definite one has none.
            let hermitian: Vec<C64> = complex_product(&aw, &adjoint(&aw, n, n), [n; 3]).iter().zip(&i).map(|(&v, &d)| v + d * n as f64).collect();
            let hpd: Vec<num_complex::Complex<R>> = hermitian.iter().map(|v| num_complex::Complex::new(R::narrow(v.re), R::narrow(v.im))).collect();
            let hw = complex_wide(&hpd);
            let l = complex_wide(&cholesky_factor(&hpd, n).expect("a Hermitian positive definite matrix"));
            assert!((0..n * n).all(|k| k % n <= k / n || l[k] == zero), "complex cholesky's upper triangle at {n}");
            assert!((0..n).all(|j| l[j * (n + 1)].im == 0.0 && l[j * (n + 1)].re > 0.0), "complex cholesky's diagonal at {n}");
            assert!(complex_distance(&complex_product(&l, &adjoint(&l, n, n), [n; 3]), &hw, complex_norm(&hw).max(1.0)) < bound, "complex cholesky at {n}");
            if n > 0 { assert!(cholesky_factor(&hpd.iter().map(|&v| -v).collect::<Vec<_>>(), n).is_none(), "complex cholesky of a negative definite matrix") }
            // QR of square, tall and wide matrices gives orthonormal columns of Q, and R with a real, non-negative diagonal.
            for (m, c) in [(n, n), (n + 5, n), (n, n + 5)] {
                let a = complex_random::<R>(m, c, (m * 31 + c) as u64);
                let (aw, p) = (complex_wide(&a), m.min(c));
                let (q, r) = qr_factor(&a, [m, c], false);
                let (q, r) = (complex_wide(&q), complex_wide(&r));
                assert!(complex_distance(&complex_product(&q, &r, [m, p, c]), &aw, complex_norm(&aw).max(1.0)) < bound, "complex qr at {m}×{c}");
                assert!(complex_distance(&complex_product(&adjoint(&q, m, p), &q, [p, m, p]), &identity::<C64>(p), 1.0) < bound, "complex qr's q at {m}×{c}");
                assert!((0..p * c).all(|k| k % c >= k / c || r[k] == zero), "complex qr's r at {m}×{c}");
                assert!((0..p).all(|j| r[j * c + j].im == 0.0), "complex qr's diagonal at {m}×{c}");
            }
        }
    }

    #[test]
    fn complex_decompositions_have_small_residuals() {
        complex_decompositions::<f64>(4.0 * f64::EPSILON);
        complex_decompositions::<f32>(4.0 * f32::EPSILON as f64);
    }


    /// Checks the SVD and the Hermitian eigen decomposition at complex items with parts of type `R`, either side of each block width,
    /// where `tolerance` is a few times `R`'s epsilon.
    fn complex_svd_and_eigen<R: RealItem>(tolerance: f64)
    where
        Complex<R>: Factor + Scalar<Real = R>,
    {
        type C<R> = num_complex::Complex<R>;
        for n in [0, 1, 2, 5, 31, 64, 129] {
            let bound = tolerance * (n as f64).max(1.0);
            // Square, tall, very tall and wide matrices: U Σ Vᴴ is A, U and V have orthonormal columns, and the singular values descend.
            for (m, c) in [(n, n), (n + 5, n), (2 * n + 5, n), (n, n + 5)] {
                let a = complex_random::<R>(m, c, (m * 37 + c) as u64);
                let (aw, p) = (complex_wide(&a), m.min(c));
                let Svd { u, s, v, .. } = svd(&a, [m, c]);
                assert!(distance(&wide(&singular_values(&a, [m, c])), &wide(&s), complex_norm(&aw).max(1.0)) < bound, "complex singular values alone at {m}×{c}");
                let (u, v, s) = (complex_wide(&u), complex_wide(&v), wide(&s));
                let us: Vec<C64> = (0..m * p).map(|k| u[k] * s[k % p]).collect();
                assert!(complex_distance(&complex_product(&us, &adjoint(&v, c, p), [m, p, c]), &aw, complex_norm(&aw).max(1.0)) < bound, "complex svd at {m}×{c}");
                assert!(complex_distance(&complex_product(&adjoint(&u, m, p), &u, [p, m, p]), &identity::<C64>(p), 1.0) < bound, "complex svd's u at {m}×{c}");
                assert!(complex_distance(&complex_product(&adjoint(&v, c, p), &v, [p, c, p]), &identity::<C64>(p), 1.0) < bound, "complex svd's v at {m}×{c}");
                assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0), "complex singular values descend at {m}×{c}");
            }
            // A Hermitian matrix, read from its lower triangle: H V is V Λ, V is unitary, and the eigenvalues ascend.
            let b = complex_random::<R>(n, n, n as u64 + 7);
            let half = R::narrow(0.5);
            let h: Vec<C<R>> = (0..n * n).map(|k| (b[k] + b[k % n * n + k / n].conj()).mul_real(half)).collect();
            let junk = C::new(R::narrow(9.0), R::narrow(-9.0));
            let lower: Vec<C<R>> = (0..n * n).map(|k| if k % n > k / n { junk } else { h[k] }).collect();
            let Eigh { values, vectors } = eigh(&lower, n).expect("a Hermitian matrix converges");
            let (hw, vectors, values) = (complex_wide(&h), complex_wide(&vectors), wide(&values));
            assert!(values.windows(2).all(|w| w[0] <= w[1]), "complex eigenvalues ascend at {n}");
            let vl: Vec<C64> = (0..n * n).map(|k| vectors[k] * values[k % n]).collect();
            assert!(complex_distance(&complex_product(&hw, &vectors, [n; 3]), &vl, complex_norm(&hw).max(1.0)) < bound, "complex hermitian eigen at {n}");
            assert!(complex_distance(&complex_product(&adjoint(&vectors, n, n), &vectors, [n; 3]), &identity::<C64>(n), 1.0) < bound, "unitary eigenvectors at {n}");
            // A complex matrix with real items gives the real code's singular values and eigenvalues, within rounding.
            let re: Vec<R> = h.iter().map(|z| z.re).collect();
            let real: Vec<C<R>> = re.iter().map(|&x| C::new(x, R::zero())).collect();
            let close = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(a, b)| (a - b).abs() <= bound * x.iter().fold(1.0, |m: f64, v| m.max(v.abs())));
            assert!(close(&wide(&singular_values(&real, [n, n])), &wide(&singular_values(&re, [n, n]))), "real singular values at {n}");
            assert!(close(&wide(&eigh(&real, n).unwrap().values), &wide(&eigh(&re, n).unwrap().values)), "real eigenvalues at {n}");
            if n > 1 {
                // A NaN gives NaN for every singular value and no eigen decomposition.
                let mut nan = h.clone();
                nan[n] = C::new(R::nan(), R::zero());
                assert!(svd(&nan, [n, n]).s.iter().all(|x| x.is_nan()), "complex svd of a matrix holding NaN at {n}");
                assert!(eigh(&nan, n).is_none(), "complex hermitian eigen of a matrix holding NaN at {n}");
            }
        }
    }

    #[test]
    fn complex_svd_and_eigen_have_small_residuals() {
        complex_svd_and_eigen::<f64>(4.0 * f64::EPSILON);
        complex_svd_and_eigen::<f32>(4.0 * f32::EPSILON as f64);
    }
}
