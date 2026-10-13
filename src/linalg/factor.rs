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
//! A thin QR factorization of a tall matrix, and least squares with one, first try CholeskyQR2. It forms `AᴴA` and its Cholesky factor,
//! solves for a first Q, then repeats both steps on that Q. Every step is a matrix product or a triangular solve, run on several threads in
//! pieces of rows. A matrix too ill-conditioned for that to match Householder QR's accuracy falls back to the reflections. R's diagonal is
//! positive on the CholeskyQR2 path, and of either sign from the reflections.
//!
//! QR and the SVD first scale a matrix by a power of two when its largest item lies where sums of squares would overflow or lose
//! digits to underflow. Their factors are those of the scaled matrix, scaled back exactly. LU and Cholesky divide by a subnormal
//! pivot, whose reciprocal would overflow.
use super::{filled, identity, identity_into, transpose_into, Rows, Scalar, Tiled};
use crate::{float::{sealed::Core, FUSED}, parallel, slice::appended, Native};
use num_complex::Complex;
use num_traits::{Float as _, One as _, Zero as _};
use std::{borrow::Cow, cmp::Ordering, marker::PhantomData, mem::MaybeUninit};

mod eigen;
mod general;
mod secular;
mod svd;
pub use eigen::{eigh, hermitian_eigenvalues, Eigh};
pub use general::{eig, eigenvalues, Eig};
#[doc(hidden)]
pub use eigen::CONQUER as EIGH_CONQUER;
#[doc(hidden)]
pub use svd::SPLIT as SVD_SPLIT;

/// Columns in each block of a blocked Cholesky factorization of an `n`×`n` matrix. A larger matrix takes wider blocks, whose trailing
/// updates are longer matrix products. On the Mac this measured fastest from 256 to 1024 rows.
fn cholesky_block(n: usize) -> usize { (n / 8).clamp(32, 128) }

/// Columns in each block of the reflections that the Hermitian eigensolver and the SVD group into `Block`s.
fn qr_block(p: usize) -> usize { (p / 16).clamp(32, 64) }
/// Columns in each block of the QR factorization in `reflect`, with `p` reflections. On an M5 Max, 128-column blocks made QR of a
/// 2048×2048 matrix 10% faster on one thread and 21% faster on 18 threads than 64-column blocks. 256 columns made a 1024×1024 QR 15%
/// slower on one thread.
fn qr_width(p: usize) -> usize { (p / 8).clamp(32, 128) }
/// The row length of a working copy of a matrix with `n` columns. A multiple of 128 items puts the rows of a product's tiles in the same
/// cache sets. On an M5 Max, eight more items made QR of a 2048×2048 matrix 10% faster on one thread.
fn stride(n: usize) -> usize { if n > 0 && n % 128 == 0 { n + 8 } else { n } }
/// Columns up to which `Reflections::q` applies one reflection at a time.
const ONE_AT_A_TIME: usize = 64;
/// Right-hand sides up to which `Reflections::apply_adjoint` applies one reflection at a time. For a 4096×16 `lstsq`, blocks measured
/// as fast at 12 and faster from 16.
#[doc(hidden)]
pub const FEW_COLUMNS: usize = 12;
/// Columns at which a block stops halving and factors one column at a time.
const LEAF: usize = 16;
/// Rows in each block of a solve with more than one right-hand side. A block's own rows substitute with vector operations, and the rows
/// below it update through a matrix product.
const NS: usize = 32;

/// An item type whose matrices rustymath factors: `f32`, `f64`, or a complex number with parts of either type. Only rustymath implements
/// it, and the functions of this module call it. Its methods compile in rustymath.
pub trait Factor: Scalar {
    /// The LU factorization of the `n`×`n` matrix `a`, into `work`'s buffers.
    fn matrix_lu_into(a: &[Self], n: usize, work: &mut LuWork<Self>);
    /// The solution of `a x = b` for the matrix that `lu` factors and the `n`×`k` matrix `b`.
    fn matrix_lu_solve(lu: LuRef<'_, Self>, b: &[Self], k: usize) -> Vec<Self>;
    /// The inverse of the matrix that `lu` factors, written into `out`, with `x` as scratch.
    fn matrix_lu_inverse_into(lu: LuRef<'_, Self>, x: &mut Vec<Self>, out: &mut [Self]);
    /// The Cholesky factor of the `n`×`n` matrix `a`, or `None` where `a` isn't positive definite.
    fn matrix_cholesky(a: &[Self], n: usize) -> Option<Vec<Self>>;
    /// The QR factorization of the matrix `a` of `size` `[m, n]`: thin, or complete where `complete` is true.
    fn matrix_qr(a: &[Self], size: [usize; 2], complete: bool) -> (Vec<Self>, Vec<Self>);
    /// The thin SVD of the matrix `a` of `size` `[m, n]`, with its singular vectors only where `vectors` is true.
    fn matrix_svd(a: &[Self], size: [usize; 2], vectors: bool) -> Svd<Self>;
    /// An approximation to the thin SVD's first `k` singular values and vectors of the matrix `a` of `size` `[m, n]`.
    fn matrix_truncated_svd(a: &[Self], size: [usize; 2], k: usize) -> Svd<Self>;
    /// The least-squares solution of `a x = b` for the matrix that `svd` factors and the `m`×`k` matrix `b`, or without `b` its `n`×`m`
    /// pseudoinverse.
    fn matrix_svd_solve(svd: SvdRef<'_, Self>, b: Option<&[Self]>, k: usize) -> Vec<Self>;
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
            fn matrix_lu_into(a: &[$t], n: usize, work: &mut LuWork<$t>) { lu_factor(a, n, work) }
            fn matrix_lu_solve(lu: LuRef<'_, $t>, b: &[$t], k: usize) -> Vec<$t> { lu_solve(lu, b, k) }
            fn matrix_lu_inverse_into(lu: LuRef<'_, $t>, x: &mut Vec<$t>, out: &mut [$t]) { lu_inverse(lu, x, out) }
            fn matrix_cholesky(a: &[$t], n: usize) -> Option<Vec<$t>> { cholesky_factor(a, n) }
            fn matrix_qr(a: &[$t], size: [usize; 2], complete: bool) -> (Vec<$t>, Vec<$t>) { qr_factor(a, size, complete) }
            fn matrix_svd(a: &[$t], size: [usize; 2], vectors: bool) -> Svd<$t> { svd::svd_factor(a, size, vectors) }
            fn matrix_truncated_svd(a: &[$t], size: [usize; 2], k: usize) -> Svd<$t> { svd::truncated_factor(a, size, k) }
            fn matrix_svd_solve(svd: SvdRef<'_, $t>, b: Option<&[$t]>, k: usize) -> Vec<$t> { svd_solve(svd, b, k) }
            fn matrix_lstsq(a: &[$t], b: &[$t], size: [usize; 2], k: usize) -> (Vec<$t>, Vec<<$t as Scalar>::Real>) { lstsq_solve(a, b, size, k) }
            fn matrix_eigh(a: &[$t], n: usize, vectors: bool) -> Option<Eigh<$t>> { eigen::eigh_factor(a, n, vectors) }
            fn matrix_eig(a: &[$t], n: usize, vectors: bool) -> Option<Eig<<$t as Scalar>::Real>> { general::eig_factor(a, n, vectors) }
        }
    )+};
}
factor!(f32, f64, Complex<f32>, Complex<f64>);

/// An LU factorization with partial pivoting of an `n`×`n` matrix. `factors` holds L below its diagonal, whose own diagonal is 1 and
/// isn't stored, and U on and above it. Row `i` of L U is row `perm[i]` of the factored matrix. [`lu`] returns one that owns its
/// factors. [`LuRef`] borrows them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Lu<T, F = Vec<T>, P = Vec<usize>> {
    pub n: usize,
    pub factors: F,
    pub perm: P,
    items: PhantomData<T>,
}

/// An [`Lu`] that borrows its factors and row permutation.
pub type LuRef<'a, T> = Lu<T, &'a [T], &'a [usize]>;

impl<T, F: AsRef<[T]>, P: AsRef<[usize]>> Lu<T, F, P> {
    /// The factorization of an `n`×`n` matrix with these packed factors and row permutation.
    pub fn new(n: usize, factors: F, perm: P) -> Self { Lu { n, factors, perm, items: PhantomData } }
    /// A view of the factors, with no copies.
    pub fn as_ref(&self) -> LuRef<'_, T> { Lu::new(self.n, self.factors.as_ref(), self.perm.as_ref()) }
}

impl<T: Factor, F: AsRef<[T]>, P: AsRef<[usize]>> Lu<T, F, P> {
    /// The solution `x` of `a x = b`, for the factored `a` and the `n`×`k` matrix `b`, as an `n`×`k` matrix.
    pub fn solve(&self, b: &[T], k: usize) -> Vec<T> { T::matrix_lu_solve(self.as_ref(), b, k) }
    /// The inverse of the factored matrix.
    pub fn inverse(&self) -> Vec<T> {
        let mut out = vec![T::zero(); self.n * self.n];
        T::matrix_lu_inverse_into(self.as_ref(), &mut Vec::new(), &mut out);
        out
    }
    /// U's diagonal.
    pub fn diagonal(&self) -> impl Iterator<Item = T> + '_ {
        let (n, factors) = (self.n, self.factors.as_ref());
        (0..n).map(move |i| factors[i * n + i])
    }
    /// L as an `n`×`n` matrix, with ones on its diagonal and zeros above it.
    pub fn lower(&self) -> Vec<T> {
        let (n, factors) = (self.n, self.factors.as_ref());
        (0..n * n).map(|k| match (k / n).cmp(&(k % n)) { Ordering::Greater => factors[k], Ordering::Equal => T::one(), Ordering::Less => T::zero() }).collect()
    }
    /// U as an `n`×`n` matrix, with zeros below its diagonal.
    pub fn upper(&self) -> Vec<T> {
        let (n, factors) = (self.n, self.factors.as_ref());
        (0..n * n).map(|k| if k % n >= k / n { factors[k] } else { T::zero() }).collect()
    }
    /// The factored matrix's determinant: the product of U's diagonal, negated where `perm` is odd.
    pub fn determinant(&self) -> T {
        let product = self.diagonal().fold(T::one(), |p, d| p * d);
        if odd(self.perm.as_ref()) { -product } else { product }
    }
    /// The factored matrix's determinant as a sign and the natural logarithm of its magnitude, which can't overflow. The sign has
    /// magnitude 1, or is 0 with a logarithm of -∞ for a singular matrix.
    pub fn log_determinant(&self) -> (T, T::Real) {
        let start = (if odd(self.perm.as_ref()) { -T::one() } else { T::one() }, T::Real::zero());
        self.diagonal().fold(start, |(sign, log), d| {
            let m = d.modulus();
            if m == T::Real::zero() { (T::zero(), T::Real::neg_infinity()) } else { (sign * d.mul_real(m.recip()), log + m.ln()) }
        })
    }
}

/// Whether the permutation `perm` of `0..perm.len()` is odd: whether an odd number of swaps puts it in order.
fn odd(perm: &[usize]) -> bool {
    // Its parity is that of its length minus its number of cycles. Each cycle counts once, at its smallest position.
    let cycles = (0..perm.len()).filter(|&start| {
        let mut i = perm[start];
        while i > start { i = perm[i] }
        i == start
    }).count();
    (perm.len() - cycles) % 2 == 1
}

/// Buffers that LU factorizations reuse. A run of calls on matrices of one size allocates only in its first call.
pub struct LuWork<T> {
    lu: Lu<T>,
    scratch: Vec<T>,
    pivots: Vec<usize>,
    x: Vec<T>,
}

impl<T> Default for LuWork<T> {
    fn default() -> Self {
        LuWork { lu: Lu::new(0, Vec::new(), Vec::new()), scratch: Vec::new(), pivots: Vec::new(), x: Vec::new() }
    }
}

impl<T: Factor> LuWork<T> {
    /// [`lu`] of the `n`×`n` matrix `a`, held in these buffers until the next call.
    pub fn factor(&mut self, a: &[T], n: usize) -> LuRef<'_, T> {
        T::matrix_lu_into(a, n, self);
        self.lu.as_ref()
    }
    /// [`solve`] for the matrix `a` of `size` `[m, n]` and the `m`×`k` matrix `b`, using these buffers.
    pub fn solve(&mut self, a: &[T], b: &[T], size: [usize; 2], k: usize) -> Vec<T> {
        if self.regular(a, size) { self.lu.as_ref().solve(b, k) } else { lstsq(a, b, size, k).0 }
    }
    /// [`pinv`] of the matrix `a` of `size` `[m, n]`, written into `out`, which holds `n`×`m` items.
    pub fn pinv_into(&mut self, a: &[T], size: [usize; 2], out: &mut [T]) {
        if self.regular(a, size) { T::matrix_lu_inverse_into(self.lu.as_ref(), &mut self.x, out) } else { out.copy_from_slice(&svd(a, size).pseudoinverse()) }
    }
    /// Whether `a` is square and nonempty, with pivots that all exceed `ε` times `n` times `a`'s largest magnitude, leaving its LU
    /// factorization in these buffers. A NaN pivot doesn't count as negligible, so NaN reaches the results.
    fn regular(&mut self, a: &[T], [m, n]: [usize; 2]) -> bool {
        if m != n || n == 0 { return false }
        let largest = greatest(a, T::modulus);
        let negligible = T::Real::epsilon() * <T::Real as crate::Float>::narrow(n as f64) * largest;
        self.factor(a, n).diagonal().all(|d| !(d.modulus() <= negligible))
    }
}

/// The thin SVD of an `m`×`n` matrix: `a = U diag(s) Vᴴ`, where `p` is the smaller of `m` and `n`, U is `m`×`p` and V is `n`×`p`, both
/// with orthonormal columns. `s` is real and in descending order. Without singular vectors, `u` and `v` are empty. [`svd`] returns one
/// that owns its factors. [`SvdRef`] borrows them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Svd<T: Scalar, U = Vec<T>, S = Vec<<T as Scalar>::Real>> {
    pub size: [usize; 2],
    pub u: U,
    pub s: S,
    pub v: U,
    items: PhantomData<T>,
}

/// An [`Svd`] that borrows its factors.
pub type SvdRef<'a, T> = Svd<T, &'a [T], &'a [<T as Scalar>::Real]>;

impl<T: Scalar, U: AsRef<[T]>, S: AsRef<[T::Real]>> Svd<T, U, S> {
    /// The thin SVD of a matrix of `size` `[m, n]` with these factors.
    pub fn new(size: [usize; 2], u: U, s: S, v: U) -> Self { Svd { size, u, s, v, items: PhantomData } }
    /// A view of the factors, with no copies.
    pub fn as_ref(&self) -> SvdRef<'_, T> { Svd::new(self.size, self.u.as_ref(), self.s.as_ref(), self.v.as_ref()) }
}

impl<T: Factor, U: AsRef<[T]>, S: AsRef<[T::Real]>> Svd<T, U, S> {
    /// The least-squares solution `x` of `a x = b` with the least norm, for the factored `a` and the `m`×`k` matrix `b`, as an `n`×`k`
    /// matrix. Singular values below `ε` times the larger of `m` and `n` times the largest singular value count as zero, as in NumPy's
    /// `lstsq`.
    pub fn solve(&self, b: &[T], k: usize) -> Vec<T> { T::matrix_svd_solve(self.as_ref(), Some(b), k) }
    /// The `n`×`m` pseudoinverse of the factored matrix, with the cutoff that `solve` uses.
    pub fn pseudoinverse(&self) -> Vec<T> { T::matrix_svd_solve(self.as_ref(), None, self.size[0]) }
}

/// The LU factorization of the `n`×`n` matrix `a`, with partial pivoting. A zero pivot leaves its column unreduced, so check U's
/// diagonal before solving.
pub fn lu<T: Factor>(a: &[T], n: usize) -> Lu<T> {
    let mut work = LuWork::default();
    work.factor(a, n);
    work.lu
}
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
/// An approximation to the thin SVD's first `k` singular values and vectors of the matrix `a` of `size` `[m, n]`, by Halko, Martinsson
/// and Tropp's randomized range finder with two power iterations. Its error depends on how fast `a`'s singular values decay beyond the
/// `k`th. It draws its test matrix from a fixed seed. `u` has `k` columns, as `v` does, where `k`
/// is no larger than `m` and `n`.
pub fn truncated_svd<T: Factor>(a: &[T], size: [usize; 2], k: usize) -> Svd<T> { T::matrix_truncated_svd(a, size, k) }
/// The least-squares solution `x` of `a x = b` with the least norm, and `a`'s singular values in descending order, for the matrix `a` of
/// `size` `[m, n]` and the `m`×`k` matrix `b`. `x` is `n`×`k`. Singular values below `ε` times the larger of `m` and `n` times the
/// largest count as zero, as in NumPy's `lstsq`.
pub fn lstsq<T: Factor>(a: &[T], b: &[T], size: [usize; 2], k: usize) -> (Vec<T>, Vec<T::Real>) { T::matrix_lstsq(a, b, size, k) }
/// The solution `x` of `a x = b` for the matrix `a` of `size` `[m, n]` and the `m`×`k` matrix `b`, as an `n`×`k` matrix. A square `a`
/// whose LU pivots all exceed `ε` times `n` times its largest magnitude solves through its LU factorization, with no cutoff,
/// as `inv` and `solve` do in NumPy. Any other `a` gives `lstsq`'s least-squares solution with the least norm.
pub fn solve<T: Factor>(a: &[T], b: &[T], size: [usize; 2], k: usize) -> Vec<T> {
    LuWork::default().solve(a, b, size, k)
}
/// The `n`×`m` pseudoinverse of the matrix `a` of `size` `[m, n]`, under [`solve`]'s rule. A square `a` with no negligible pivot gives
/// its inverse through its LU factorization, with no cutoff. Any other `a` gives the pseudoinverse with `lstsq`'s cutoff.
pub fn pinv<T: Factor>(a: &[T], [m, n]: [usize; 2]) -> Vec<T> {
    let mut out = vec![T::zero(); m * n];
    LuWork::default().pinv_into(a, [m, n], &mut out);
    out
}
/// √(least normal)/ε, the bottom of the range where sums of squares neither overflow nor lose digits to underflow. Its reciprocal is the
/// top.
fn safe_low<R: Native>() -> R { R::min_positive_value().sqrt() / R::epsilon() }
/// The rank of the matrix `a` of `size` `[m, n]`: how many of its singular values exceed `lstsq`'s cutoff. A square matrix that
/// [`solve`] and [`pinv`] treat as regular can still have a lower rank. `[1 1e8; 0 1]` at f64 has no negligible pivot, but its smaller
/// singular value, about 1e-8, falls below the cutoff of about 4.4e-8.
pub fn rank<T: Factor>(a: &[T], size: [usize; 2]) -> usize {
    let s = singular_values(a, size);
    let floor = cutoff(&s, size[0].max(size[1]));
    s.iter().filter(|&&x| x > floor).count()
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

/// The position of the item of `x` with the greatest `abs1`, the first of any equal ones, or 0 where every item is NaN. `greatest` finds
/// that `abs1`, then a scan finds its position.
fn largest<T: Scalar>(x: &[T]) -> usize {
    let top = greatest(x, T::abs1);
    x.iter().position(|v| v.abs1() == top).unwrap_or(0)
}

/// `√(x² + y²)` with no scaling, for the rotations of the tridiagonal and bidiagonal iterations. It overflows where `x²` or `y²` does.
fn plain_hypot<R: Native>(x: R, y: R) -> R { x.mla::<FUSED>(x, y * y).sqrt() }

/// The greatest `f(v)` over the items `v` of `x`, ignoring NaN, or 0 for an empty `x`. Eight running maximums find it, which vectorizes.
fn greatest<T: Copy, R: Native>(x: &[T], f: impl Fn(T) -> R) -> R {
    let mut m = [R::zero(); 8];
    let (chunks, rest) = x.as_chunks::<8>();
    for c in chunks {
        for k in 0..8 { m[k] = m[k].max(f(c[k])) }
    }
    rest.iter().fold(m.iter().fold(R::zero(), |a, &b| a.max(b)), |a, &b| a.max(f(b)))
}

/// The power of two that brings the greatest `abs1` in `a` to between 1 and 2, when that magnitude lies outside the range from
/// `safe_low` to its reciprocal. `None` inside that range, for a matrix of zeros, and where an item has an infinite part. Scaling by
/// a power of two is exact. Where `abs1` overflows, a second pass finds the greatest larger part, at least half the largest float,
/// which gives the same power of two.
fn safe_scale<T: Scalar>(a: &[T]) -> Option<T::Real> {
    let top = greatest(a, T::abs1);
    let top = if top.is_finite() { top } else { greatest(a, |x| x.re().abs().max(x.im().abs())) };
    let low = safe_low::<T::Real>();
    if !top.is_finite() || top == T::Real::zero() || (low..=low.recip()).contains(&top) { return None }
    unit_scale(top)
}

/// The power of two that brings the positive, finite `top` to between 1 and 2, within the range of the type's exponents.
fn unit_scale<R: Native>(top: R) -> Option<R> {
    let limit = R::max_value().log2().floor();
    Some((R::one() + R::one()).powi((-top.log2().floor()).max(-limit).min(limit).to_i32()?))
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

/// The transpose of the `rows`×`cols` matrix whose rows start `ls` items apart in `a`, with each item converted by `f`. It writes each item
/// once, into the new vector's spare capacity.
fn transpose_by<T: Copy + Send + Sync>(a: &[T], ls: usize, size: [usize; 2], f: impl Fn(T) -> T + Sync) -> Vec<T> {
    let mut t = Vec::new();
    transpose_by_into(a, ls, size, f, &mut t);
    t
}
/// [`transpose_by`], written into `t` in place of its old items. Rows of the transpose split across threads.
fn transpose_by_into<T: Copy + Send + Sync>(a: &[T], ls: usize, [rows, cols]: [usize; 2], f: impl Fn(T) -> T + Sync, t: &mut Vec<T>) {
    t.clear();
    if rows * cols == 0 { return }
    appended(t, rows * cols, |slots| parallel::split(slots, rows, &|start, part| {
        transpose_into(&a[start / rows..], ls, part, rows, [rows, part.len() / rows], |x| MaybeUninit::new(f(x)));
        true
    }));
}

/// The transpose of the `m`×`n` matrix `a`.
fn transpose<T: Copy + Send + Sync>(a: &[T], m: usize, n: usize) -> Vec<T> { transpose_by(a, n, [m, n], |x| x) }

/// The conjugate transpose of the `m`×`n` matrix `a`, which for real items is its transpose.
fn adjoint<T: Scalar>(a: &[T], m: usize, n: usize) -> Vec<T> { transpose_by(a, n, [m, n], T::conj) }

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
    transpose_by(&f[k0 * n + k0..], n, [f.len() / n - k0, k1 - k0], |x| x)
}

/// Swaps rows `first + i` and `pivots[i]` of each column of `columns`, in turn for each `i`. Each column holds `n` items. The columns
/// split across threads.
fn swap_rows<T: Send>(columns: &mut [T], n: usize, pivots: &[usize], first: usize) {
    parallel::split(columns, n, &|_, columns| {
        for column in columns.chunks_exact_mut(n) {
            for i in 0..pivots.len() { column.swap(first + i, pivots[i]) }
        }
        true
    });
}

/// Transposes the `n`×`n` matrix `a` in place. It swaps each pair of 16×16 tiles across the diagonal through two buffers, reading and
/// writing runs of adjacent items.
fn transpose_square<T: Copy + Default>(a: &mut [T], n: usize) {
    const B: usize = 16;
    let (mut p, mut q) = ([[T::default(); B]; B], [[T::default(); B]; B]);
    for i0 in (0..n).step_by(B) {
        let h = B.min(n - i0);
        for j0 in (i0..n).step_by(B) {
            let w = B.min(n - j0);
            for r in 0..h { p[r][..w].copy_from_slice(&a[(i0 + r) * n + j0..][..w]) }
            for r in 0..w { q[r][..h].copy_from_slice(&a[(j0 + r) * n + i0..][..h]) }
            for r in 0..h {
                let row = &mut a[(i0 + r) * n + j0..][..w];
                for c in 0..w { row[c] = q[c][r] }
            }
            for r in 0..w {
                let row = &mut a[(j0 + r) * n + i0..][..h];
                for c in 0..h { row[c] = p[c][r] }
            }
        }
    }
}

/// Factors columns `c0..c1` of the matrix that `f` holds column by column, with columns of `n` items, from row `c0` down. It halves the
/// columns until `LEAF` or fewer remain. The left half factors first, and its row swaps apply to the right half. The right half's rows
/// of U come from substitution with the left half's L, on a row-major copy in `scratch`, and a matrix product updates the rows below
/// them. The right half factors last, and its row swaps apply to the left half. A leaf's row swaps apply to its own columns. Column
/// `c`'s pivot comes from row `pivots[c]`.
fn lu_panel<T: Scalar + Tiled>(f: &mut [T], n: usize, [c0, c1]: [usize; 2], pivots: &mut [usize], scratch: &mut Vec<T>) {
    if c1 - c0 > LEAF {
        let mid = (c0 + c1) / 2;
        lu_panel(f, n, [c0, mid], pivots, scratch);
        let (left, right) = f.split_at_mut(mid * n);
        let (left, k, w) = (&*left, mid - c0, c1 - mid);
        let half = &mut right[..w * n];
        swap_rows(half, n, &pivots[c0..mid], c0);
        transpose_by_into(&half[c0..], n, [w, k], |x| x, scratch);
        forward(move |i, j| left[(c0 + j) * n + c0 + i], [0, k], scratch, [w, w], LEAF, false);
        let u12 = &scratch[..];
        parallel::split(half, n, &|start, columns| {
            let count = columns.len() / n;
            transpose_into(&u12[start / n..], w, &mut columns[c0..], n, [k, count], |x| x);
            true
        });
        T::add_product(&mut half[mid..], n, move |j, s| -u12[s * w + j], Rows(&left[c0 * n + mid..], n), [w, k, n - mid]);
        lu_panel(f, n, [mid, c1], pivots, scratch);
        return swap_rows(&mut f[c0 * n..mid * n], n, &pivots[mid..c1], mid);
    }
    for c in c0..c1 {
        let p = c + largest(&f[c * n + c..(c + 1) * n]);
        pivots[c] = p;
        if p != c {
            for j in c0..c1 { f.swap(j * n + c, j * n + p) }
        }
        let pivot = f[c * n + c];
        if pivot == T::zero() { continue }
        let (left, right) = f.split_at_mut((c + 1) * n);
        let l = &mut left[c * n + c + 1..];
        divide(l, pivot);
        for column in right[..(c1 - c - 1) * n].chunks_exact_mut(n) {
            let u = column[c];
            axpy(&mut column[c + 1..], -u, l);
        }
    }
}
/// `lu` for `T`, into `work`'s buffers. It factors a transposed copy of `a`, where each column is contiguous, then transposes the
/// factors back in place.
fn lu_factor<T: Scalar + Tiled>(a: &[T], n: usize, work: &mut LuWork<T>) {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    let LuWork { lu, scratch, pivots, .. } = work;
    lu.n = n;
    transpose_by_into(a, n, [n, n], |x| x, &mut lu.factors);
    pivots.clear();
    pivots.resize(n, 0);
    lu_panel(&mut lu.factors, n, [0, n], pivots, scratch);
    transpose_square(&mut lu.factors, n);
    lu.perm.clear();
    lu.perm.extend(0..n);
    for c in 0..n { lu.perm.swap(c, pivots[c]) }
}

/// `Lu::solve` for `T`. One right-hand side substitutes with dot products along L's and U's rows. More substitute a block of rows at a
/// time, after one matrix product subtracts the rows already solved.
fn lu_solve<T: Scalar + Tiled>(lu: LuRef<'_, T>, b: &[T], k: usize) -> Vec<T> {
    let (n, f) = (lu.n, lu.factors);
    assert_eq!(b.len(), n * k, "`b` must hold `n`×`k` items");
    let mut x: Vec<T> = lu.perm.iter().flat_map(|&p| b[p * k..(p + 1) * k].iter().copied()).collect();
    if k == 1 {
        for i in 0..n { x[i] = x[i] - dot(&f[i * n..i * n + i], &x[..i]) }
        for i in (0..n).rev() { x[i] = (x[i] - dot(&f[i * n + i + 1..(i + 1) * n], &x[i + 1..])).quotient(f[i * n + i]) }
    } else {
        forward(|i, j| f[i * n + j], [0, n], &mut x, [k, k], NS, false);
        backward(f, n, &mut x, k);
    }
    x
}

/// `Lu::inverse` for `T`, into `out`: `U⁻¹ L⁻¹`, with its columns permuted to undo the row swaps. It works in `x`. Its substitution with L
/// skips the zeros above the diagonal of `L⁻¹`.
fn lu_inverse<T: Scalar + Tiled>(lu: LuRef<'_, T>, x: &mut Vec<T>, out: &mut [T]) {
    let n = lu.n;
    identity_into(x, n);
    forward(|i, j| lu.factors[i * n + j], [0, n], x, [n, n], NS, true);
    backward(lu.factors, n, x, n);
    for r in 0..n {
        for (c, &p) in lu.perm.iter().enumerate() { out[r * n + p] = x[r * n + c] }
    }
}

/// Solves `L y = x` in place for rows `r0..r1`, where L is the unit lower triangular matrix whose items below its diagonal `l(i, j)`
/// gives, rows `0..r0` of y are solved and already removed from these rows, and `x` holds rows `r0..r1` of `k` items, which start `ldx`
/// items apart. Above `block` rows, the top half solves first, one matrix product removes it from the bottom half, and the bottom half
/// solves last. With `lower`, `x` starts as the identity, only the columns up to each row's own can be nonzero, and it solves a block of
/// `block` rows at a time instead, with a matrix product removing each block from every row below. A block's rows substitute with
/// vector operations.
fn forward<T: Scalar + Tiled>(l: impl Fn(usize, usize) -> T + Sync + Copy, [r0, r1]: [usize; 2], x: &mut [T], [ldx, k]: [usize; 2], block: usize, lower: bool) {
    if !lower && r1 - r0 > block {
        let mid = (r0 + r1) / 2;
        let (top, bottom) = x.split_at_mut((mid - r0) * ldx);
        forward(l, [r0, mid], top, [ldx, k], block, false);
        T::add_product(bottom, ldx, move |i, p| -l(mid + i, r0 + p), Rows(top, ldx), [r1 - mid, mid - r0, k]);
        return forward(l, [mid, r1], bottom, [ldx, k], block, false);
    }
    for i0 in (r0..r1).step_by(block) {
        let i1 = (i0 + block).min(r1);
        let width = if lower { i1 } else { k };
        let (done, rest) = x.split_at_mut((i1 - r0) * ldx);
        let rows = &mut done[(i0 - r0) * ldx..];
        for i in i0 + 1..i1 {
            let (head, tail) = rows.split_at_mut((i - i0) * ldx);
            for j in i0..i { axpy(&mut tail[..width], -l(i, j), &head[(j - i0) * ldx..(j - i0) * ldx + width]) }
        }
        let rows = &*rows;
        T::add_product(rest, ldx, move |i, p| -l(i1 + i, i0 + p), Rows(rows, ldx), [r1 - i1, i1 - i0, width]);
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
        T::add_product(above, k, |i, p| -f[i * n + i0 + p], Rows(rows, k), [i0, i1 - i0, k]);
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
        T::add_product(&mut right[mid..], rows, |j, p| -l(j, p).conj(), Rows(&left[c0 * rows + mid..], rows), [c1 - mid, mid - c0, rows - mid]);
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
/// `cholesky` for `T`, by blocks of `cholesky_block(n)` columns. Each block's columns factor in a transposed copy, where each column is
/// contiguous. One matrix product for each later block of columns then updates the lower triangle of the rest.
fn cholesky_factor<T: Scalar + Tiled>(a: &[T], n: usize) -> Option<Vec<T>> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    let mut l = a.to_vec();
    let width = cholesky_block(n);
    for k0 in (0..n).step_by(width) {
        let k1 = (k0 + width).min(n);
        let (nb, rows, w) = (k1 - k0, n - k0, n - k1);
        let mut panel = transposed(&l, n, k0, k1);
        if !cholesky_panel(&mut panel, rows, [0, nb]) { return None }
        transpose_into(&panel, rows, &mut l[k0 * n + k0..], n, [nb, rows], |x| x);
        if w == 0 { break }
        let l21 = |i: usize, p: usize| panel[p * rows + nb + i];
        let mut l21_conj = filled(nb * w, T::zero());
        for (column, conj) in panel.chunks_exact(rows).zip(l21_conj.chunks_exact_mut(w)) { for j in 0..w { conj[j] = column[nb + j].conj() } }
        T::add_lower_product(&mut l[k1 * n + k1..], n, |i, p| -l21(i, p), Rows(&l21_conj, w), [w, nb, w]);
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
#[inline]
fn householder<T: Scalar>(panel: &[T], rows: usize, c: usize, r: usize) -> T {
    match r.cmp(&c) {
        Ordering::Less => T::zero(),
        Ordering::Equal => T::one(),
        Ordering::Greater => panel[c * rows + r],
    }
}

/// Fills the diagonal block of rows and columns `c0..c1` of T, the upper triangular factor of the compact WY form `I - V T Vᴴ` of a QR
/// block's reflections, applied first to last, as LAPACK's `zlarft` gives it. That block is the T of reflections `c0` to `c1 - 1` alone.
/// `panel` holds the block transposed, with rows of `rows` items. `t` holds T, `nb`×`nb`, with each reflection's `τ` already on its
/// diagonal and zeros below it. It halves the columns until `LEAF` or fewer remain, and `joined` fills the block between the halves.
fn triangular<T: Scalar + Tiled>(panel: &[T], rows: usize, [c0, c1]: [usize; 2], t: &mut [T]) {
    if c1 - c0 > LEAF {
        let mid = (c0 + c1) / 2;
        triangular(panel, rows, [c0, mid], t);
        triangular(panel, rows, [mid, c1], t);
        return joined(panel, rows, [c0, mid, c1], t);
    }
    let nb = panel.len() / rows;
    for c in c0..c1 {
        let tau = t[c * nb + c];
        if tau == T::zero() { continue }
        let vc = &panel[c * rows + c + 1..(c + 1) * rows];
        // Each earlier vector j against v_c, which is 1 at row c: v_jᴴ v_c.
        let s: Vec<T> = (c0..c).map(|j| panel[j * rows + c].conj() + dotc(&panel[j * rows + c + 1..(j + 1) * rows], vc)).collect();
        for i in c0..c { t[i * nb + c] = -tau * dot(&t[i * nb + i..i * nb + c], &s[i - c0..]) }
    }
}

/// Fills T's block for rows `c0..mid` and columns `mid..c1` as `-T₁ S T₂`, from T's diagonal blocks `T₁` and `T₂` for the two halves,
/// where `S = V₁ᴴ V₂`. V₂ is zero above row `mid`, so S sums over the rows from `mid`.
fn joined<T: Scalar + Tiled>(panel: &[T], rows: usize, [c0, mid, c1]: [usize; 3], t: &mut [T]) {
    let ([k1, k2], nb, below) = ([mid - c0, c1 - mid], panel.len() / rows, rows - mid);
    let v1 = transpose_by(&panel[c0 * rows + mid..], rows, [k1, below], T::conj);
    let mut st = vec![T::zero(); k2 * k1];
    T::add_product(&mut st, k1, |j, q| householder(panel, rows, mid + j, mid + q), Rows(&v1, k1), [k2, below, k1]);
    let mut s2 = vec![T::zero(); k1 * k2];
    T::add_product(&mut s2, k2, |i, l| st[l * k1 + i], Rows(&t[mid * nb + mid..], nb), [k1, k2, k2]);
    let mut t12 = vec![T::zero(); k1 * k2];
    T::add_product(&mut t12, k2, |i, l| -t[(c0 + i) * nb + c0 + l], Rows(&s2, k2), [k1, k1, k2]);
    for i in 0..k1 { t[(c0 + i) * nb + mid..][..k2].copy_from_slice(&t12[i * k2..][..k2]) }
}

/// The vectors of reflections `c0..c1` in a QR block that `panel` holds transposed, with rows of `rows` items: one vector to a row, each
/// from row `c0` on, with its zeros and its 1 written out.
fn vectors<T: Scalar>(panel: &[T], rows: usize, [c0, c1]: [usize; 2]) -> Vec<T> {
    let mut v = Vec::with_capacity((c1 - c0) * (rows - c0));
    for c in c0..c1 {
        v.resize(v.len() + c - c0, T::zero());
        v.push(T::one());
        v.extend_from_slice(&panel[c * rows + c + 1..(c + 1) * rows]);
    }
    v
}

/// Factors columns `c0..c1` of the QR block that `panel` holds transposed, with rows of `rows` items. Each column's reflection vector
/// replaces it below the diagonal, and R replaces it on and above. Its τ goes on the diagonal of `t`, which holds T, `nb`×`nb`, and the
/// rest of T's rows and columns `c0..c1` above it. It halves the columns until `LEAF` or fewer remain. The left half's reflections reach
/// the right half through three products, and `joined` fills T's block between the halves.
fn panel_qr<T: Scalar + Tiled>(panel: &mut [T], rows: usize, [c0, c1]: [usize; 2], t: &mut [T]) {
    let nb = panel.len() / rows;
    if c1 - c0 <= LEAF {
        for c in c0..c1 {
            let (left, right) = panel.split_at_mut((c + 1) * rows);
            let v = &mut left[c * rows + c..];
            let tau = reflector(v);
            t[c * nb + c] = tau;
            if tau == T::zero() { continue }
            // The leaf's later columns: A ← A - τ̄ v (vᴴ A), where v is 1 at row c.
            for column in right[..(c1 - c - 1) * rows].chunks_exact_mut(rows) { apply_reflector(&mut column[c..], tau, &v[1..]) }
        }
        return triangular(panel, rows, [c0, c1], t);
    }
    let mid = (c0 + c1) / 2;
    let ([k1, k2], r) = ([mid - c0, c1 - mid], rows - c0);
    panel_qr(panel, rows, [c0, mid], t);
    let vt = vectors(panel, rows, [c0, mid]);
    let vc = transpose_by(&vt, r, [k1, r], T::conj);
    let t1: Vec<T> = (0..k1 * k1).map(|q| t[(c0 + q / k1) * nb + c0 + q % k1].conj()).collect();
    // The right half, X ← (I - V₁ T₁ᴴ V₁ᴴ) X, as Xᵀ ← Xᵀ - Xᵀ conj(V₁) conj(T₁) V₁ᵀ in the transposed layout.
    let x = &mut panel[mid * rows..c1 * rows];
    let mut w = vec![T::zero(); k2 * k1];
    T::add_product(&mut w, k1, |i, q| x[i * rows + c0 + q], Rows(&vc, k1), [k2, r, k1]);
    let mut z = vec![T::zero(); k2 * k1];
    T::add_product(&mut z, k1, |i, q| w[i * k1 + q], Rows(&t1, k1), [k2, k1, k1]);
    T::add_product(&mut x[c0..], rows, |i, q| -z[i * k1 + q], Rows(&vt, r), [k2, k1, r]);
    panel_qr(panel, rows, [mid, c1], t);
    joined(panel, rows, [c0, mid, c1], t)
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
        let mut t = vec![T::zero(); nb * nb];
        for (c, &tau) in taus.iter().enumerate() { t[c * nb + c] = tau }
        triangular(&panel, panel.len() / nb, [0, nb], &mut t);
        Self { k0, nb, panel, t }
    }

    /// Applies `I - V T Vᴴ`, or its adjoint `I - V Tᴴ Vᴴ` where `adjoint` is true, to columns `c0..c1` of the matrix `x`, whose rows start
    /// `ld` items apart.
    fn apply(&self, x: &mut [T], ld: usize, [c0, c1]: [usize; 2], adjoint: bool) {
        let (k0, nb, rows, cols) = (self.k0, self.nb, self.panel.len() / self.nb, c1 - c0);
        // T is stored with its zeros, so neither T nor Tᴴ needs a mask.
        let t = if adjoint { Cow::Owned(transpose_by(&self.t, nb, [nb, nb], T::conj)) } else { Cow::Borrowed(&self.t[..]) };
        let v = |r: usize, c: usize| householder(&self.panel, rows, c, r);
        let mut w = vec![T::zero(); nb * cols];
        T::add_product(&mut w, cols, |i, r| v(r, i).conj(), Rows(&x[k0 * ld + c0..], ld), [nb, rows, cols]);
        let mut tw = vec![T::zero(); nb * cols];
        T::add_product(&mut tw, cols, |i, c| t[i * nb + c], Rows(&w, cols), [nb, nb, cols]);
        T::add_product(&mut x[k0 * ld + c0..], ld, |r, c| -v(r, c), Rows(&tw, cols), [rows, nb, cols]);
    }
}

/// Householder reflections of `m`-item columns, in blocks, as the `m`×`m` product Q of their matrices, first to last.
struct Reflections<T> {
    m: usize,
    blocks: Vec<Block<T>>,
}

/// The QR factorization of `a`: its reflections, by blocks of `qr_width(p)` columns, and the `p`×`n` upper triangular R, where `p` is the
/// smaller of `m` and `n`. Each block's columns factor through `panel_qr` in a transposed copy, where each column is contiguous. The block's
/// reflections then update the columns after it, in the WY form.
fn reflect<T: Scalar + Tiled>(a: &[T], [m, n]: [usize; 2]) -> (Reflections<T>, Vec<T>) {
    assert_eq!(a.len(), m * n, "`a` must hold `m`×`n` items");
    let p = m.min(n);
    let (ld, scale) = (stride(n), safe_scale(a));
    let mut f = Vec::with_capacity(m * ld);
    for i in 0..m {
        f.extend(a[i * n..(i + 1) * n].iter().map(|&x| scale.map_or(x, |s| x.mul_real(s))));
        f.resize((i + 1) * ld, T::zero());
    }
    let mut blocks = Vec::new();
    let width = qr_width(p);
    for k0 in (0..p).step_by(width) {
        let k1 = (k0 + width).min(p);
        let (nb, rows) = (k1 - k0, m - k0);
        let mut panel = transposed(&f, ld, k0, k1);
        let mut t = vec![T::zero(); nb * nb];
        panel_qr(&mut panel, rows, [0, nb], &mut t);
        for r in 0..nb {
            for c in r..nb { f[(k0 + r) * ld + k0 + c] = panel[c * rows + r] }
        }
        let block = Block { k0, nb, panel, t };
        // The columns after the block: A2 ← (I - V Tᴴ Vᴴ) A2.
        if k1 < n { block.apply(&mut f, ld, [k1, n], true) }
        blocks.push(block);
    }
    // R returns to `a`'s scale. The reflections don't depend on the scale.
    let r = (0..p * n).map(|k| {
        let (i, j) = (k / n, k % n);
        if j < i { T::zero() } else { scale.map_or(f[i * ld + j], |s| f[i * ld + j].div_real(s)) }
    }).collect();
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
        let ld = stride(p);
        let mut q = filled(m * ld, T::zero());
        for i in 0..p { q[i * ld + i] = T::one() }
        for block in self.blocks.iter().rev() { block.apply(&mut q, ld, [block.k0, p], false) }
        if ld == p { return q }
        let mut out = Vec::with_capacity(m * p);
        for row in q.chunks_exact(ld) { out.extend_from_slice(&row[..p]) }
        out
    }

    /// `Q x` for the `m`×`k` matrix `x`, in place.
    fn apply(&self, x: &mut [T], k: usize) {
        for block in self.blocks.iter().rev() { block.apply(x, k, [0, k], false) }
    }

    /// `Qᴴ x` for the `m`×`k` matrix `x`, in place: the blocks' adjoints, first block first. Up to `FEW_COLUMNS` columns, each reflection
    /// applies alone to each column of a transposed copy, where each column is contiguous.
    fn apply_adjoint(&self, x: &mut [T], k: usize) {
        if k > FEW_COLUMNS {
            for block in &self.blocks { block.apply(x, k, [0, k], true) }
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

/// `qr` and `qr_complete` for `T`. A thin factorization of a tall matrix comes from `tall_qr` where that succeeds. Otherwise Q and R come
/// from the reflections. A complete Q applies the reflections to the `m`×`m` identity, and R gains rows of zeros to be `m`×`n`.
fn qr_factor<T: Scalar + Tiled>(a: &[T], [m, n]: [usize; 2], complete: bool) -> (Vec<T>, Vec<T>) {
    if !complete && tall_for_qr([m, n]) && let Some((r, q)) = tall_qr(a, [m, n], Tall::Q) { return (q, r) }
    let (reflections, mut r) = reflect(a, [m, n]);
    if !complete { return (reflections.q(), r) }
    let mut q = identity(m);
    reflections.apply(&mut q, m);
    r.resize(m * n, T::zero());
    (q, r)
}

/// `Svd::solve` and `Svd::pseudoinverse` for `T`.
fn svd_solve<T: Scalar + Tiled>(svd: SvdRef<'_, T>, b: Option<&[T]>, k: usize) -> Vec<T> {
    let ([m, n], p) = (svd.size, svd.s.len());
    if let Some(b) = b { assert_eq!(b.len(), m * k, "`b` must hold `m`×`k` items") }
    assert!(svd.u.len() == m * p && svd.v.len() == n * p, "the SVD must hold its singular vectors");
    pseudo_solve(svd, b, k, m.max(n))
}

/// `V diag(1/s) Uᴴ b` for the factors that `svd` holds and a `b` of `k` columns, or `V diag(1/s) Uᴴ` without `b`, where `1/s` is zero
/// for each singular value no greater than `ε` times `scale` times the largest.
fn pseudo_solve<T: Scalar + Tiled>(svd: SvdRef<'_, T>, b: Option<&[T]>, k: usize, scale: usize) -> Vec<T> {
    let ([m, n], p) = (svd.size, svd.s.len());
    if k == 0 { return Vec::new() }
    let floor = cutoff(&svd.s, scale);
    let mut y = match b {
        Some(b) => {
            let mut y = vec![T::zero(); p * k];
            T::add_product(&mut y, k, |i, r| svd.u[r * p + i].conj(), Rows(b, k), [p, m, k]);
            y
        }
        None => adjoint(&svd.u, m, p),
    };
    for (row, &s) in y.chunks_exact_mut(k).zip(svd.s) {
        let d = if s > floor { s.recip() } else { T::Real::zero() };
        for x in row { *x = x.mul_real(d) }
    }
    let mut x = vec![T::zero(); n * k];
    T::add_product(&mut x, k, |i, r| svd.v[i * p + r], Rows(&y, k), [n, p, k]);
    x
}

/// The size below which a singular value counts as zero, for a matrix whose singular values `s` descend and whose larger side has `scale`
/// items: `ε` times `scale` times the largest, as in NumPy's `lstsq`.
fn cutoff<R: crate::Float>(s: &[R], scale: usize) -> R { R::epsilon() * R::narrow(scale as f64) * s.first().copied().unwrap_or(R::zero()) }

/// `lstsq` for `T`. A matrix with at least as many rows as columns reduces to the R of its QR factorization, and `b` to `Qᴴ b`, without
/// forming Q: through `tall_qr` where that succeeds, and otherwise through the reflections. A wider matrix solves through its own SVD.
fn lstsq_solve<T: Scalar + Tiled>(a: &[T], b: &[T], [m, n]: [usize; 2], k: usize) -> (Vec<T>, Vec<T::Real>)
where
    T::Real: Tiled,
{
    assert_eq!(b.len(), m * k, "`b` must hold `m`×`k` items");
    if m < n {
        let svd = svd::svd_factor(a, [m, n], true);
        return (pseudo_solve(svd.as_ref(), Some(b), k, n), svd.s);
    }
    let reduced = tall_for_lstsq([m, n]).then(|| tall_qr(a, [m, n], Tall::Project(b, k))).flatten();
    let (r, y) = reduced.unwrap_or_else(|| {
        let (reflections, r) = reflect(a, [m, n]);
        let mut y = b.to_vec();
        reflections.apply_adjoint(&mut y, k);
        y.truncate(n * k);
        (r, y)
    });
    let svd = svd::svd_factor(&r, [n, n], true);
    (pseudo_solve(svd.as_ref(), Some(&y), k, m), svd.s)
}

/// Items in each block of rows that `tall_qr`'s passes work through at a time, small enough to stay in cache.
const TALL_BLOCK: usize = 1 << 18;
/// The largest Frobenius norm of `Q₁ᴴQ₁ - I` at which `tall_qr` keeps its result.
const NEARLY_ORTHONORMAL: f64 = 0.25;

/// Whether `qr` tries `tall_qr` for a matrix of `size` `[m, n]`. The first test admits the sizes where CholeskyQR2 measured faster on one
/// thread on the Mac. The second admits larger ones, where its threaded passes are faster on the default thread count.
fn tall_for_qr([m, n]: [usize; 2]) -> bool { (n >= 16 && m >= (4 * n).max(n * n / 4)) || threaded_work(m, n, 1 << 21) }

/// Whether `lstsq` tries `tall_qr` for a matrix of `size` `[m, n]`, with the same two tests as `tall_for_qr`. Householder least squares
/// forms no Q. CholeskyQR2 therefore needs taller matrices than for `qr` before it's faster.
fn tall_for_lstsq([m, n]: [usize; 2]) -> bool { (n >= 20 && m >= 2 * n * n) || threaded_work(m, n, 1 << 23) }

/// Whether a matrix of `size` `[m, n]` holds enough work for CholeskyQR2's threaded passes to be faster than Householder on the Mac's
/// default 18 threads: at least 16 rows per column, and `m n²` of at least `work`. Narrow matrices such as 1000000×10 pass. On one thread
/// they take up to 1.7 times as long as Householder.
fn threaded_work(m: usize, n: usize, work: usize) -> bool { m >= 16 * n && m.saturating_mul(n.saturating_mul(n)) >= work }

/// What `tall_qr` gives beside R: Q, nothing, or `Qᴴ b` for the `m`×`k` matrix `b`.
enum Tall<'a, T> { Q, R, Project(&'a [T], usize) }

/// The thin QR factorization of a tall `a` by CholeskyQR2: R, with Q or `Qᴴ b` as `want` asks. It scales
/// `a` as `reflect` does. `None` where `a` is too ill-conditioned for the result to be as accurate as Householder QR's: where `AᴴA` has no
/// Cholesky factor, or where the first pass's Q₁ is too far from orthonormal for the second pass to correct.
///
/// Pass one factors `AᴴA = L₁ L₁ᴴ`. Pass two solves `Q₁ L₁ᴴ = A` and factors `Q₁ᴴQ₁ = L₂ L₂ᴴ`. Then `Q = Q₁ L₂⁻ᴴ`, `Qᴴ b = L₂⁻¹ Q₁ᴴ b` and
/// `R = (L₁ L₂)ᴴ`, whose diagonal is real and positive. Each pass gives every thread a contiguous range of `a`'s rows and its own partial
/// Gram matrix, and works through the range in blocks of `TALL_BLOCK` items. Pass two stores Q₁ only for `Tall::Q`.
fn tall_qr<T: Scalar + Tiled>(a: &[T], [m, n]: [usize; 2], want: Tall<'_, T>) -> Option<(Vec<T>, Vec<T>)> {
    let scale = safe_scale(a);
    let scaled = match scale {
        Some(s) => Cow::Owned(a.iter().map(|x| x.mul_real(s)).collect()),
        None => Cow::Borrowed(a),
    };
    let (a, rows, k) = (&scaled[..], (TALL_BLOCK / n).max(1), if let Tall::Project(_, k) = want { k } else { 0 });
    let count = parallel::threads().min(m.div_ceil(rows));
    let range = |p: usize| p * m / count..(p + 1) * m / count;
    let mut grams: Vec<_> = (0..count).map(|_| filled(n * n, T::zero())).collect();
    parallel::pieces_of(&mut grams, 1, usize::MAX, &|p, g| {
        let (start, end) = (range(p).start, range(p).end);
        for x in (start..end).step_by(rows).map(|s| &a[s * n..end.min(s + rows) * n]) {
            T::add_lower_product(&mut g[0], n, |i, j| x[j * n + i].conj(), Rows(x, n), [n, x.len() / n, n]);
        }
        true
    });
    let l1 = cholesky_factor(&summed(grams.iter()), n)?;
    let (u1, d1) = unit_lower(&l1, n);
    let mut q = if matches!(want, Tall::Q) { filled(m * n, T::zero()) } else { Vec::new() };
    let mut rest = &mut q[..];
    let outs: Vec<&mut [T]> = (0..count).map(|p| {
        let (out, tail) = std::mem::take(&mut rest).split_at_mut(if matches!(want, Tall::Q) { range(p).len() * n } else { 0 });
        rest = tail;
        out
    }).collect();
    let mut parts: Vec<_> = outs.into_iter().map(|out| (out, filled(n * n, T::zero()), filled(n * k, T::zero()))).collect();
    parallel::pieces_of(&mut parts, 1, usize::MAX, &|p, part| {
        let (out, gram, y) = &mut part[0];
        let (start, end) = (range(p).start, range(p).end);
        for r in (start..end).step_by(rows).map(|s| s..end.min(s + rows)) {
            let h = r.len();
            let qt = solved(&a[r.start * n..r.end * n], n, &u1, &d1);
            let q1: Cow<[T]> = if out.is_empty() { Cow::Owned(adjoint(&qt, n, h)) } else {
                let band = &mut out[(r.start - start) * n..(r.end - start) * n];
                transpose_into(&qt, h, band, n, [n, h], T::conj);
                Cow::Borrowed(&*band)
            };
            T::add_lower_product(gram, n, |i, j| qt[i * h + j], Rows(&q1, n), [n, h, n]);
            if let Tall::Project(b, k) = want { T::add_product(y, k, |i, j| qt[i * h + j], Rows(&b[r.start * k..r.end * k], k), [n, h, k]) }
        }
        true
    });
    let (g2, mut y) = (summed(parts.iter().map(|part| &part.1)), summed(parts.iter().map(|part| &part.2)));
    let far = (0..n * n).filter(|&c| c % n <= c / n).fold(T::Real::zero(), |sum, c| {
        let (diagonal, v) = (c % n == c / n, g2[c].modulus());
        let v = if diagonal { (g2[c] - T::one()).modulus() } else { v };
        sum + if diagonal { v * v } else { v * v + v * v }
    });
    if !(far.sqrt() <= <T::Real as crate::Float>::narrow(NEARLY_ORTHONORMAL)) { return None }
    let l2 = cholesky_factor(&g2, n)?;
    let (u2, d2) = unit_lower(&l2, n);
    let mut l = vec![T::zero(); n * n];
    T::add_product(&mut l, n, |i, j| l1[i * n + j], Rows(&l2, n), [n, n, n]);
    let mut r = adjoint(&l, n, n);
    if let Some(s) = scale { for v in &mut r { *v = v.div_real(s) } }
    if !matches!(want, Tall::Q) {
        lower_solve(&u2, &d2, n, &mut y, k);
        return Some((r, y));
    }
    parallel::pieces_of(&mut q, rows * n, usize::MAX, &|_, band| {
        let h = band.len() / n;
        let qt = solved(band, n, &u2, &d2);
        transpose_into(&qt, h, band, n, [n, h], T::conj);
        true
    });
    Some((r, q))
}

/// The sum of `parts`, item by item, added in order.
fn summed<'a, T: Scalar + 'a>(mut parts: impl Iterator<Item = &'a Vec<T>>) -> Vec<T> {
    let mut sum = parts.next().cloned().unwrap_or_default();
    for part in parts {
        for (s, &v) in sum.iter_mut().zip(part) { *s = *s + v }
    }
    sum
}

/// The lower triangular `n`×`n` matrix `l`, which has a real, positive diagonal, as `L̃ D`: the unit lower triangular L̃, of which only the
/// items below the diagonal are set, and the diagonal D.
fn unit_lower<T: Scalar>(l: &[T], n: usize) -> (Vec<T>, Vec<T::Real>) {
    let d: Vec<T::Real> = (0..n).map(|j| l[j * n + j].re()).collect();
    (l.iter().enumerate().map(|(c, &v)| v.div_real(d[c % n])).collect(), d)
}

/// Solves `L̃ D x = y` in place for the `n`×`k` matrix `x`, where `unit_lower` gives L̃ as `u` and D as `d`.
fn lower_solve<T: Scalar + Tiled>(u: &[T], d: &[T::Real], n: usize, x: &mut [T], k: usize) {
    if k == 0 { return }
    forward(|i, j| u[i * n + j], [0, n], x, [k, k], NS, false);
    for (row, &di) in x.chunks_exact_mut(k).zip(d) { divide(row, T::from_real(di)) }
}

/// `(x L⁻ᴴ)ᴴ` for the `h`×`n` matrix `x`, as an `n`×`h` matrix, where `unit_lower` gives L as `u` and `d`: the conjugate transpose of the
/// solution of `X Lᴴ = x`, from substitution along the rows of `xᴴ`.
fn solved<T: Scalar + Tiled>(x: &[T], n: usize, u: &[T], d: &[T::Real]) -> Vec<T> {
    let h = x.len() / n;
    let mut xt = adjoint(x, h, n);
    lower_solve(u, d, n, &mut xt, h);
    xt
}

