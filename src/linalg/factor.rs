//! Decompositions of real and complex matrices held row-major in slices: LU with partial pivoting, Cholesky, Householder QR and the
//! SVD, with the solves, inverses and least-squares solutions built from them, at `f32`, `f64` and complex numbers with parts of either
//! type. On macOS, Accelerate's LAPACK computes the SVD, the Hermitian eigen decomposition and a real matrix's general one. Elsewhere
//! the child module `svd` computes the SVD and `eigen` the Hermitian eigen decomposition. The child module `general` computes a complex
//! matrix's eigen decomposition on every target, and a real one's off macOS.
//!
//! Each decomposition is written once, generic over [`Scalar`]. A complex factorization conjugates where the real one transposes.
//! Cholesky gives L Lᴴ, and QR's reflections take the form `I - τ v vᴴ`, leaving R with a real diagonal. Singular values and the
//! eigenvalues of a Hermitian matrix are real, of the parts' type. `general` uses single-shift QR iterations for a complex matrix, in
//! place of a real one's double-shift iterations.
//!
//! LU, Cholesky and QR work in blocks of columns, wider for a larger matrix. Each block's own columns factor with the unblocked
//! algorithm, along rows, so every inner loop reads contiguous memory. The rest of the matrix then updates through `Tiled`'s matrix
//! products, where most of a large factorization's time goes. QR keeps each block's reflections in the compact WY
//! form `I - V T Vᴴ`, so that applying them takes three matrix products. Dot
//! products keep eight running sums, and each multiply-add is fused where the target has an instruction for it, so results can
//! differ from LAPACK's in their last bits.
//!
//! A thin QR factorization of a tall matrix, and least squares with one, first try CholeskyQR2. It forms `AᴴA` and its Cholesky factor,
//! solves for a first Q, then repeats both steps on that Q. Every step is a matrix product or a triangular solve, run on several threads in
//! pieces of rows. A matrix too ill-conditioned for that to match Householder QR's accuracy falls back to the reflections. R's diagonal is
//! positive on the CholeskyQR2 path, and of either sign from the reflections.
//!
//! QR and the native SVD first scale a matrix by a power of two when its largest item lies where sums of squares would overflow or lose
//! digits to underflow. Their factors are those of the scaled matrix, scaled back exactly. LU and Cholesky divide by a subnormal
//! pivot, whose reciprocal would overflow.
use super::{filled, identity, transpose_into, Matrix, Scalar, Side, Tiled, Triangle};
use crate::{float::{sealed::Core, FUSED}, parallel, slice::appended, Native};
use num_complex::Complex;
use num_traits::{Float as _, One as _, Zero as _};
use std::{borrow::Cow, cmp::Ordering, marker::PhantomData, mem::MaybeUninit};

#[cfg(native)]
pub(super) mod eigen;
pub(super) mod general;
#[cfg(native)]
mod secular;
#[cfg(native)]
pub(super) mod svd;
mod truncated;
/// Above this many rows, the native Hermitian eigensolver's divide and conquer splits a tridiagonal matrix's eigen problem in two.
/// Implicit QL solves smaller ones. Tests size their matrices around it.
#[doc(hidden)]
pub const EIGH_CONQUER: usize = 64;
/// Above this many columns, the native SVD's divide and conquer splits a bidiagonal matrix's SVD in two. The QR algorithm solves
/// smaller ones. Tests size their matrices around it.
#[doc(hidden)]
pub const SVD_SPLIT: usize = 64;

/// Columns in each block of a blocked Cholesky factorization of an `n`×`n` matrix. A larger matrix takes wider blocks, whose trailing
/// updates are longer matrix products. On the Mac this measured fastest from 256 to 1024 rows.
fn cholesky_block(n: usize) -> usize { (n / 8).clamp(32, 128) }

/// Columns in each block of the reflections that the eigensolvers and the SVD group into `Block`s.
fn qr_block(p: usize) -> usize { (p / 16).clamp(32, 64) }

/// Rows at or below which `reflections` applies each reflection on its own, where building the WY form's blocks costs more than
/// its products save.
const SMALL: usize = 96;

/// Qᵀ, the transpose without conjugation, for the reflections that `tridiagonal` or `hessenberg` leaves in `w` and `taus`. Reflection `k` acts on rows
/// and columns from `k + 1` on, as reflection `k` of a QR factorization acts on a matrix without the first row and column.
/// `Reflections::q` therefore forms Q's last `n - 1` rows and columns, by blocks of reflections in the WY form, through the tiled `+.×`
/// kernels.
fn reflections<T: Tiled>(w: &[T], taus: &[T], n: usize) -> Vec<T> {
    let mut qt = identity(n);
    if n < 2 { return qt }
    if n <= SMALL {
        // A small matrix applies each reflection's transpose, `I - τ v̄ vᵀ`, from the right on its own. Before reflection `k`, Qᵀ
        // differs from the identity only in rows and columns after `k + 1`.
        let mut v = filled(n, T::zero());
        for (k, &tau) in taus.iter().enumerate().rev() {
            if tau == T::zero() { continue }
            let v = &mut v[..n - k - 1];
            v[0] = T::one();
            v[1..].copy_from_slice(&w[k * n + k + 2..(k + 1) * n]);
            for row in qt[(k + 1) * n..].chunks_exact_mut(n) {
                let row = &mut row[k + 1..];
                let s = tau * dotc(v, row);
                axpy(row, -s, v);
            }
        }
        return qt;
    }
    let p = n - 1;
    transpose_into(&blocked(w, taus, n).q(), p, &mut qt[n + 1..], n, [p, p], |x| x);
    qt
}

/// The reflections that `reflections` multiplies, in blocks, for a matrix of at least 2 rows. They act on rows and columns 1 to `n - 1`.
/// Row `c` of a block's panel holds reflection `k0 + c` from column `k0 + 1` of `w`.
fn blocked<T: Tiled>(w: &[T], taus: &[T], n: usize) -> Reflections<T> {
    let p = n - 1;
    let blocks = blocks_of(&taus[..p], qr_block(p), |k0, k1| (k0..k1).flat_map(|k| w[k * n + k0 + 1..(k + 1) * n].iter().copied()).collect());
    Reflections { m: p, blocks }
}

/// Columns in each block of the QR factorization in `reflect`, with `p` reflections. On an M5 Max, 128-column blocks made QR of a
/// 2048×2048 matrix 10% faster on one thread and 21% faster on 18 threads than 64-column blocks. 256 columns made a 1024×1024 QR 15%
/// slower on one thread.
fn qr_width(p: usize) -> usize { (p / 8).clamp(32, 128) }
/// The row length of a working copy of a matrix with `n` columns. A multiple of 128 items puts the rows of a product's tiles in the same
/// cache sets. On an M5 Max, eight more items made QR of a 2048×2048 matrix 10% faster on one thread.
pub(super) fn stride(n: usize) -> usize { if n > 0 && n % 128 == 0 { n + 8 } else { n } }
/// Columns up to which `Reflections::q` applies one reflection at a time.
const ONE_AT_A_TIME: usize = 64;
/// Right-hand sides up to which `Reflections::apply_adjoint` applies one reflection at a time. For a 4096×16 `lstsq`, blocks measured
/// as fast at 12 and faster from 16.
#[doc(hidden)]
pub const FEW_COLUMNS: usize = 12;
/// Columns at which a block stops halving and factors one column at a time. A native lower triangular solve whose right-hand side is at
/// least as wide as the matrix also stops halving at this many rows.
const LEAF: usize = 16;
/// Rows at which other native triangular solves stop halving. A leaf's own rows substitute with vector operations, and the rows below
/// it update through a matrix product.
#[cfg(native)]
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
            fn matrix_svd(a: &[$t], size: [usize; 2], vectors: bool) -> Svd<$t> { svd_factor(a, size, vectors) }
            fn matrix_truncated_svd(a: &[$t], size: [usize; 2], k: usize) -> Svd<$t> { truncated::truncated_factor(a, size, k) }
            fn matrix_svd_solve(svd: SvdRef<'_, $t>, b: Option<&[$t]>, k: usize) -> Vec<$t> { svd_solve(svd, b, k) }
            fn matrix_lstsq(a: &[$t], b: &[$t], size: [usize; 2], k: usize) -> (Vec<$t>, Vec<<$t as Scalar>::Real>) { lstsq_solve(a, b, size, k) }
            fn matrix_eigh(a: &[$t], n: usize, vectors: bool) -> Option<Eigh<$t>> { eigh_factor(a, n, vectors) }
            fn matrix_eig(a: &[$t], n: usize, vectors: bool) -> Option<Eig<<$t as Scalar>::Real>> { eig_factor(a, n, vectors) }
        }
    )+};
}
factor!(f32, f64, Complex<f32>, Complex<f64>);

/// An LU factorization with partial pivoting of an `n`×`n` matrix. `factors` packs L below the diagonal and U on and above it. L's
/// diagonal is 1 and isn't stored. Unlike the crate's other matrices, `factors` is column-major: item `(i, j)` is `factors[j * n + i]`.
/// Row `i` of L U is row `perm[i]` of the factored matrix. [`lu`] returns one that owns its factors. [`LuRef`] borrows them.
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
        let mut out = filled(self.n * self.n, T::zero());
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
        (0..n * n).map(|k| {
            let (i, j) = (k / n, k % n);
            match i.cmp(&j) { Ordering::Greater => factors[j * n + i], Ordering::Equal => T::one(), Ordering::Less => T::zero() }
        }).collect()
    }
    /// U as an `n`×`n` matrix, with zeros below its diagonal.
    pub fn upper(&self) -> Vec<T> {
        let (n, factors) = (self.n, self.factors.as_ref());
        (0..n * n).map(|k| { let (i, j) = (k / n, k % n); if j >= i { factors[j * n + i] } else { T::zero() } }).collect()
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
    let mut out = filled(m * n, T::zero());
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

/// The eigen decomposition of a Hermitian `n`×`n` matrix: `a = V diag(values) Vᴴ`. `values` is real and in ascending order. Column
/// `j` of the `n`×`n` matrix `vectors` is a unit eigenvector for `values[j]`. Without eigenvectors, `vectors` is empty.
#[derive(Clone, Debug, PartialEq)]
pub struct Eigh<T: Scalar> {
    pub values: Vec<T::Real>,
    pub vectors: Vec<T>,
}

/// The eigen decomposition of the Hermitian `n`×`n` matrix `a`, reading only its lower triangle. `None` where the iterations don't
/// converge, as for a matrix holding NaN or an infinity.
pub fn eigh<T: Factor>(a: &[T], n: usize) -> Option<Eigh<T>> { T::matrix_eigh(a, n, true) }
/// The eigenvalues of the Hermitian `n`×`n` matrix `a`, in ascending order, reading only its lower triangle. `None` where `eigh` gives
/// `None`.
pub fn hermitian_eigenvalues<T: Factor>(a: &[T], n: usize) -> Option<Vec<T::Real>> { T::matrix_eigh(a, n, false).map(|f| f.values) }

/// The eigen decomposition of an `n`×`n` matrix: `a V = V diag(values)`, with complex values and vectors whose parts have type `T`.
/// The eigenvalues come in the order the Schur form holds them. Column `j` of the `n`×`n` matrix `vectors` is an eigenvector for
/// `values[j]`, of unit length. For a real matrix, a real eigenvalue has an imaginary part of exactly 0, and a real eigenvector. A complex
/// pair comes as the eigenvalue with the positive imaginary part, then its conjugate. Their eigenvectors are conjugates too.
#[derive(Clone, Debug, PartialEq)]
pub struct Eig<T> {
    pub values: Vec<Complex<T>>,
    pub vectors: Vec<Complex<T>>,
}

/// The eigen decomposition of the real or complex `n`×`n` matrix `a`. `None` where the iterations don't converge, as for a matrix
/// holding NaN or an infinity.
pub fn eig<T: Factor>(a: &[T], n: usize) -> Option<Eig<T::Real>> { T::matrix_eig(a, n, true) }

/// The eigenvalues of the `n`×`n` matrix `a`, as `eig` gives them. Without eigenvectors to build, each iteration transforms only the
/// rows and columns it still acts on.
pub fn eigenvalues<T: Factor>(a: &[T], n: usize) -> Option<Vec<Complex<T::Real>>> { T::matrix_eig(a, n, false).map(|f| f.values) }

/// Whether every item of `x` is finite, in both parts where it's complex.
fn finite<T: Scalar>(x: &[T]) -> bool { x.iter().all(|x| x.re().is_finite() && x.im().is_finite()) }

/// `svd` and `singular_values` for `T`, through `Tiled::svd`. A matrix holding a NaN or an infinity, or one whose iterations don't
/// converge, gives NaN for every result.
fn svd_factor<T: Tiled>(a: &[T], [m, n]: [usize; 2], vectors: bool) -> Svd<T> {
    assert_eq!(a.len(), m * n, "`a` must hold `m`×`n` items");
    let p = m.min(n);
    if p == 0 { return Svd::new([m, n], Vec::new(), Vec::new(), Vec::new()) }
    let (u, s, v) = finite(a).then(|| T::svd(a, [m, n], vectors)).flatten().unwrap_or_else(|| {
        let nan = T::Real::nan();
        let filled = |len| if vectors { vec![T::from_real(nan); len] } else { Vec::new() };
        (filled(m * p), vec![nan; p], filled(n * p))
    });
    Svd::new([m, n], u, s, v)
}

/// `eigh` and `hermitian_eigenvalues` for `T`, through `Tiled::eigh`. A NaN or an infinity in `a`'s lower triangle gives `None`.
fn eigh_factor<T: Tiled>(a: &[T], n: usize, vectors: bool) -> Option<Eigh<T>> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    if n == 0 { return Some(Eigh { values: Vec::new(), vectors: Vec::new() }) }
    (0..n).all(|i| finite(&a[i * n..=i * n + i])).then(|| T::eigh(a, n, vectors)).flatten()
}

/// `eig` and `eigenvalues` for `T`, through `Tiled::eig`. A matrix holding a NaN or an infinity gives `None`.
fn eig_factor<T: Tiled>(a: &[T], n: usize, vectors: bool) -> Option<Eig<T::Real>> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    if n == 0 { return Some(Eig { values: Vec::new(), vectors: Vec::new() }) }
    finite(a).then(|| T::eig(a, n, vectors)).flatten()
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
#[cfg(native)]
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
pub(super) fn transpose_by<T: Copy + num_traits::Zero + Send + Sync>(a: &[T], ls: usize, size: [usize; 2], f: impl Fn(T) -> T + Sync) -> Vec<T> {
    let mut t = Vec::new();
    transpose_by_into(a, [ls, size[0]], size, f, &mut t);
    t
}
/// [`transpose_by`], written into `t` in place of its old items, with the transpose's rows `lt` items apart. Zeros fill each row past its
/// `rows` items. Rows of the transpose split across threads.
pub(super) fn transpose_by_into<T: Copy + num_traits::Zero + Send + Sync>(a: &[T], [ls, lt]: [usize; 2], [rows, cols]: [usize; 2], f: impl Fn(T) -> T + Sync, t: &mut Vec<T>) {
    t.clear();
    if rows * cols == 0 { return }
    appended(t, lt * cols, |slots| parallel::split(slots, lt, &|start, part| {
        transpose_into(&a[start / lt..], ls, part, lt, [rows, part.len() / lt], |x| MaybeUninit::new(f(x)));
        for row in part.chunks_exact_mut(lt) { row[rows..].fill(MaybeUninit::new(T::zero())) }
        true
    }));
}

/// The transpose of the `m`×`n` matrix `a`.
fn transpose<T: Copy + num_traits::Zero + Send + Sync>(a: &[T], m: usize, n: usize) -> Vec<T> { transpose_by(a, n, [m, n], |x| x) }

/// The `rows`×`cols` matrix `a`, copied with its rows `ld` items apart and zeros after each row's `cols` items.
pub(super) fn spaced<T: Copy + num_traits::Zero>(a: &[T], [rows, cols]: [usize; 2], ld: usize) -> Vec<T> {
    if ld == cols { return a[..rows * cols].to_vec() }
    let mut x = Vec::with_capacity(rows * ld);
    for row in a[..rows * cols].chunks_exact(cols) {
        x.extend_from_slice(row);
        x.resize(x.len() + ld - cols, T::zero());
    }
    x
}

/// Closes up the `rows` runs of `cols` items that start `ld` items apart in `x`, as [`spaced`] leaves them, so that they're contiguous.
pub(super) fn compact<T: Copy>(x: &mut Vec<T>, [rows, cols]: [usize; 2], ld: usize) {
    if ld != cols { for i in 1..rows { x.copy_within(i * ld..i * ld + cols, i * cols) } }
    x.truncate(rows * cols);
}

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

/// Swaps rows `first + i` and `pivots[i]` of each column of `columns`, in turn for each `i`. The columns start `ld` items apart and split
/// across threads.
fn swap_rows<T: Send>(columns: &mut [T], ld: usize, pivots: &[usize], first: usize) {
    parallel::split(columns, ld, &|_, columns| {
        for column in columns.chunks_exact_mut(ld) {
            for i in 0..pivots.len() { column.swap(first + i, pivots[i]) }
        }
        true
    });
}

/// Factors columns `c0..c1` of the `n`-row matrix that `f` holds column by column, with columns `ld` items apart, from row `c0` down. It
/// halves the columns until `LEAF` or fewer remain. The left half factors first, and its row swaps apply to the right half. The right
/// half's rows of U come from substitution with the left half's L, on a row-major copy in `scratch` with rows `stride(w)` items apart, and
/// a matrix product updates the rows below them. The right half factors last, and its row swaps apply to the left half. A leaf's row
/// swaps apply to its own columns. Column `c`'s pivot comes from row `pivots[c]`.
fn lu_panel<T: Tiled>(f: &mut [T], [n, ld]: [usize; 2], [c0, c1]: [usize; 2], pivots: &mut [usize], scratch: &mut Vec<T>) {
    if c1 - c0 > LEAF {
        let mid = (c0 + c1) / 2;
        lu_panel(f, [n, ld], [c0, mid], pivots, scratch);
        let (left, right) = f.split_at_mut(mid * ld);
        let (left, k, w) = (&*left, mid - c0, c1 - mid);
        let lw = stride(w);
        let half = &mut right[..w * ld];
        swap_rows(half, ld, &pivots[c0..mid], c0);
        transpose_by_into(&half[c0..], [ld, lw], [w, k], |x| x, scratch);
        T::triangular(scratch, lw, Matrix::rows(&left[c0 * ld + c0..], ld).transpose(), [k, w], Side::Left, Triangle::UnitUpper);
        let u12 = &scratch[..];
        parallel::split(half, ld, &|start, columns| {
            let count = columns.len() / ld;
            transpose_into(&u12[start / ld..], lw, &mut columns[c0..], ld, [k, count], |x| x);
            true
        });
        T::gemm(&mut half[mid..], ld, Matrix::rows(u12, lw).transpose(), Matrix::rows(&left[c0 * ld + mid..], ld), [w, k, n - mid], -T::one(), T::one());
        lu_panel(f, [n, ld], [mid, c1], pivots, scratch);
        return swap_rows(&mut f[c0 * ld..mid * ld], ld, &pivots[mid..c1], mid);
    }
    for c in c0..c1 {
        let p = c + largest(&f[c * ld + c..c * ld + n]);
        pivots[c] = p;
        if p != c {
            for j in c0..c1 { f.swap(j * ld + c, j * ld + p) }
        }
        let pivot = f[c * ld + c];
        if pivot == T::zero() { continue }
        let (left, right) = f.split_at_mut((c + 1) * ld);
        let l = &mut left[c * ld + c + 1..c * ld + n];
        divide(l, pivot);
        for column in right[..(c1 - c - 1) * ld].chunks_exact_mut(ld) {
            let u = column[c];
            axpy(&mut column[c + 1..n], -u, l);
        }
    }
}
/// `lu` for `T`, into `work`'s buffers. It factors a column-by-column copy of `a` in place, with columns `stride(n)` items apart so that
/// a power-of-two `n` doesn't map every column to the same cache sets. It then closes the columns up to `n` items apart.
fn lu_factor<T: Tiled>(a: &[T], n: usize, work: &mut LuWork<T>) {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    let LuWork { lu, scratch, pivots, .. } = work;
    let ld = stride(n);
    lu.n = n;
    transpose_by_into(a, [n, ld], [n, n], |x| x, &mut lu.factors);
    pivots.clear();
    pivots.resize(n, 0);
    lu_panel(&mut lu.factors, [n, ld], [0, n], pivots, scratch);
    compact(&mut lu.factors, [n, n], ld);
    lu.perm.clear();
    lu.perm.extend(0..n);
    for c in 0..n { lu.perm.swap(c, pivots[c]) }
}

/// `Lu::solve` for `T`: a solve with L, then one with U.
fn lu_solve<T: Tiled>(lu: LuRef<'_, T>, b: &[T], k: usize) -> Vec<T> {
    let (n, f) = (lu.n, lu.factors);
    assert_eq!(b.len(), n * k, "`b` must hold `n`×`k` items");
    let mut x: Vec<T> = lu.perm.iter().flat_map(|&p| b[p * k..(p + 1) * k].iter().copied()).collect();
    let p = Matrix::rows(f, n).transpose();
    T::triangular(&mut x, k, p, [n, k], Side::Left, Triangle::UnitUpper);
    T::triangular(&mut x, k, p, [n, k], Side::Left, Triangle::Lower);
    x
}

/// `Lu::inverse` for `T`, into `out`: `U⁻¹ L⁻¹ P`, where `P` undoes the row swaps. Accelerate solves from `P` in `out` itself. Rustymath's
/// own kernels solve from the identity in `x`, with rows `stride(n)` items apart, skipping the zeros above the diagonal of `L⁻¹`, then
/// permute the result's columns into `out`.
#[cfg_attr(not(native), allow(unused_variables))]
fn lu_inverse<T: Tiled>(lu: LuRef<'_, T>, x: &mut Vec<T>, out: &mut [T]) {
    let (n, p) = (lu.n, Matrix::rows(lu.factors, lu.n).transpose());
    #[cfg(accelerate)]
    if super::accelerated() {
        out.fill(T::zero());
        for (c, &r) in lu.perm.iter().enumerate() { out[c * n + r] = T::one() }
        T::triangular(out, n, p, [n, n], Side::Left, Triangle::UnitUpper);
        return T::triangular(out, n, p, [n, n], Side::Left, Triangle::Lower);
    }
    #[cfg(native)]
    {
        let ld = stride(n);
        x.clear();
        x.resize(n * ld, T::zero());
        for i in 0..n { x[i * ld + i] = T::one() }
        forward(p, [0, n], x, [ld, n], NS, true, true);
        T::triangular(x, ld, p, [n, n], Side::Left, Triangle::Lower);
        for r in 0..n {
            for (c, &q) in lu.perm.iter().enumerate() { out[r * n + q] = x[r * ld + c] }
        }
    }
}

/// Solves `L y = x` in place for rows `r0..r1`, where L is lower triangular, rows `0..r0` of y are solved and already removed
/// from these rows, and `x` holds rows `r0..r1` of `k` items, which start `ldx`
/// items apart. Above `block` rows, the top half solves first, one matrix product removes it from the bottom half, and the bottom half
/// solves last. With `lower`, `x` starts as the identity, only the columns up to each row's own can be nonzero, and it solves a block of
/// `block` rows at a time instead, with a matrix product removing each block from every row below. A block's rows substitute with
/// vector operations. With `unit`, L's diagonal is implicitly one.
#[cfg(native)]
fn forward<T: Tiled>(l: Matrix<'_, T>, [r0, r1]: [usize; 2], x: &mut [T], [ldx, k]: [usize; 2], block: usize, lower: bool, unit: bool) {
    if !lower && r1 - r0 > block {
        let mid = (r0 + r1) / 2;
        let (top, bottom) = x.split_at_mut((mid - r0) * ldx);
        forward(l, [r0, mid], top, [ldx, k], block, false, unit);
        T::gemm(bottom, ldx, l.offset(mid, r0), Matrix::rows(top, ldx), [r1 - mid, mid - r0, k], -T::one(), T::one());
        return forward(l, [mid, r1], bottom, [ldx, k], block, false, unit);
    }
    for i0 in (r0..r1).step_by(block) {
        let i1 = (i0 + block).min(r1);
        let width = if lower { i1 } else { k };
        let (done, rest) = x.split_at_mut(((i1 - r0) * ldx).min(x.len()));
        let rows = &mut done[(i0 - r0) * ldx..];
        for i in i0..i1 {
            let (head, tail) = rows.split_at_mut((i - i0) * ldx);
            for j in i0..i { axpy(&mut tail[..width], -l.at(i, j), &head[(j - i0) * ldx..(j - i0) * ldx + width]) }
            if !unit { divide(&mut tail[..width], l.at(i, i)) }
        }
        let rows = &*rows;
        if i1 < r1 { T::gemm(rest, ldx, l.offset(i1, i0), Matrix::rows(rows, ldx), [r1 - i1, i1 - i0, width], -T::one(), T::one()) }
    }
}

/// Solves `U x = y` in place for rows `r0..r1` of the upper triangular U. `x` holds those rows, `k` items each, starting `ldx` items
/// apart, with the solved rows from `r1` on already removed. Above `NS` rows, the bottom half solves first, one matrix product removes
/// it from the top half, and the top half solves last. Up to `NS` rows substitute with vector operations, from the last row up. With
/// `unit`, U's diagonal is implicitly one.
#[cfg(native)]
fn backward<T: Tiled>(u: Matrix<'_, T>, [r0, r1]: [usize; 2], x: &mut [T], [ldx, k]: [usize; 2], unit: bool) {
    if r1 - r0 > NS {
        let mid = (r0 + r1) / 2;
        let (top, bottom) = x.split_at_mut((mid - r0) * ldx);
        backward(u, [mid, r1], bottom, [ldx, k], unit);
        T::gemm(top, ldx, u.offset(r0, mid), Matrix::rows(bottom, ldx), [mid - r0, r1 - mid, k], -T::one(), T::one());
        return backward(u, [r0, mid], top, [ldx, k], unit);
    }
    for i in (r0..r1).rev() {
        let (done, below) = x.split_at_mut(((i - r0 + 1) * ldx).min(x.len()));
        let xi = &mut done[(i - r0) * ldx..][..k];
        for j in i + 1..r1 { axpy(xi, -u.at(i, j), &below[(j - i - 1) * ldx..][..k]) }
        if !unit { divide(xi, u.at(i, i)) }
    }
}

/// Native triangular solves share the substitutions above. A right solve adjoints its right-hand side once to reuse the left solve. One
/// contiguous right-hand side goes through `single`. A lower solve whose right-hand side is at least as wide as the matrix, such as an
/// LU panel's, stops halving at `LEAF` rows, and a narrower one at `NS`.
#[cfg(native)]
pub(super) fn triangular_native<T: Tiled>(x: &mut [T], ldx: usize, a: Matrix<'_, T>, [m, n]: [usize; 2], side: Side, triangle: Triangle) {
    if m == 0 || n == 0 { return }
    if matches!(side, Side::Right) {
        let mut xt = transpose_by(x, ldx, [m, n], T::conj);
        triangular_native(&mut xt, m, a.adjoint(), [n, m], Side::Left, triangle);
        return transpose_into(&xt, m, x, ldx, [n, m], T::conj);
    }
    let (lower, unit) = (matches!(triangle, Triangle::Lower) != a.op.transposed(), matches!(triangle, Triangle::UnitUpper));
    if n == 1 && ldx == 1 && single(x, a, m, lower, unit) { return }
    if lower { forward(a, [0, m], x, [ldx, n], if n >= m { LEAF } else { NS }, false, unit) } else { backward(a, [0, m], x, [ldx, n], unit) }
}

/// Solves for one contiguous right-hand side `x`, along the lines that `a` stores contiguously. Where its view transposes, those are its
/// logical columns, and each solved item leaves the rest through `axpy`. Otherwise they're its rows, and each item takes a `dot`. A
/// conjugating view solves nothing and gives false.
#[cfg(native)]
fn single<T: Scalar>(x: &mut [T], a: Matrix<'_, T>, m: usize, lower: bool, unit: bool) -> bool {
    let (d, ld) = (a.data, a.ld);
    let solved = |v: T, i: usize| if unit { v } else { v.quotient(d[i * ld + i]) };
    match a.op {
        super::Op::Transpose if lower => for j in 0..m {
            x[j] = solved(x[j], j);
            let xj = x[j];
            axpy(&mut x[j + 1..], -xj, &d[j * ld + j + 1..j * ld + m]);
        },
        super::Op::Transpose => for j in (0..m).rev() {
            x[j] = solved(x[j], j);
            let xj = x[j];
            axpy(&mut x[..j], -xj, &d[j * ld..j * ld + j]);
        },
        super::Op::Normal if lower => for i in 0..m { x[i] = solved(x[i] - dot(&d[i * ld..i * ld + i], &x[..i]), i) },
        super::Op::Normal => for i in (0..m).rev() { x[i] = solved(x[i] - dot(&d[i * ld + i + 1..i * ld + m], &x[i + 1..]), i) },
        _ => return false,
    }
    true
}

/// Factors columns `c0..c1` of the `rows`-row Cholesky block that `panel` holds column by column, with columns `ld` items apart, halving
/// the columns as `lu_panel` does. It gives false where the matrix isn't positive definite.
fn cholesky_panel<T: Tiled>(panel: &mut [T], [rows, ld]: [usize; 2], [c0, c1]: [usize; 2]) -> bool {
    if c1 - c0 > LEAF {
        let mid = (c0 + c1) / 2;
        if !cholesky_panel(panel, [rows, ld], [c0, mid]) { return false }
        let (left, right) = panel.split_at_mut(mid * ld);
        let l = Matrix::rows(&left[c0 * ld + mid..], ld);
        T::gemm(&mut right[mid..], ld, l.adjoint(), l, [c1 - mid, mid - c0, rows - mid], -T::one(), T::one());
        return cholesky_panel(panel, [rows, ld], [mid, c1]);
    }
    for c in c0..c1 {
        let d = panel[c * ld + c];
        if !(d.re() > T::Real::zero()) { return false }
        let (left, right) = panel.split_at_mut((c + 1) * ld);
        let d = T::from_real(d.re().sqrt());
        left[c * ld + c] = d;
        let lc = &mut left[c * ld + c + 1..c * ld + rows];
        divide(lc, d);
        // Each later column j loses conj(L[j][c]) times column c, from row j down.
        for (j, column) in right[..(c1 - c - 1) * ld].chunks_exact_mut(ld).enumerate() {
            let below = &lc[j..];
            axpy(&mut column[c + 1 + j..rows], -below[0].conj(), below);
        }
    }
    true
}
/// `cholesky` for `T`, by blocks of `cholesky_block(n)` columns, in a copy of `a` with rows `stride(n)` items apart. Each block's columns
/// factor in a transposed copy, with columns `stride(rows)` items apart. One matrix product for each later block of columns then updates
/// the lower triangle of the rest. It then closes the rows up to `n` items apart.
fn cholesky_factor<T: Tiled>(a: &[T], n: usize) -> Option<Vec<T>> {
    assert_eq!(a.len(), n * n, "`a` must hold `n`×`n` items");
    let ld = stride(n);
    let mut l = spaced(a, [n, n], ld);
    let (width, mut panel) = (cholesky_block(n), Vec::new());
    for k0 in (0..n).step_by(width) {
        let k1 = (k0 + width).min(n);
        let (nb, rows, w) = (k1 - k0, n - k0, n - k1);
        let pl = stride(rows);
        transpose_by_into(&l[k0 * ld + k0..], [ld, pl], [rows, nb], |x| x, &mut panel);
        if !cholesky_panel(&mut panel, [rows, pl], [0, nb]) { return None }
        transpose_into(&panel, pl, &mut l[k0 * ld + k0..], ld, [nb, rows], |x| x);
        if w == 0 { break }
        T::gram(&mut l[k1 * ld + k1..], ld, Matrix::rows(&panel[nb..], pl).transpose(), [w, nb], -T::Real::one(), T::Real::one());
    }
    compact(&mut l, [n, n], ld);
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

/// Fills the diagonal block of rows and columns `c0..c1` of T, the upper triangular factor of the compact WY form `I - V T Vᴴ` of a QR
/// block's reflections, applied first to last, as LAPACK's `zlarft` gives it. That block is the T of reflections `c0` to `c1 - 1` alone.
/// `panel` holds the block transposed, with rows of `rows` items. `t` holds T, `nb`×`nb`, with each reflection's `τ` already on its
/// diagonal and zeros below it. It halves the columns until `LEAF` or fewer remain, and `joined` fills the block between the halves.
fn triangular<T: Tiled>(panel: &[T], rows: usize, [c0, c1]: [usize; 2], t: &mut [T]) {
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
fn joined<T: Tiled>(panel: &[T], rows: usize, [c0, mid, c1]: [usize; 3], t: &mut [T]) {
    let ([k1, k2], nb, below) = ([mid - c0, c1 - mid], panel.len() / rows, rows - mid);
    let v2 = vectors(panel, rows, [mid, c1]);
    let s = T::multiply(Matrix::rows(&panel[c0 * rows + mid..], rows).conjugate(), Matrix::rows(&v2, below).transpose(), [k1, below, k2]);
    let s2 = T::multiply(Matrix::rows(&s, k2), Matrix::rows(&t[mid * nb + mid..], nb), [k1, k2, k2]);
    let t12 = T::multiply(Matrix::rows(&t[c0 * nb + c0..], nb), Matrix::rows(&s2, k2), [k1, k1, k2]);
    for i in 0..k1 {
        for j in 0..k2 { t[(c0 + i) * nb + mid + j] = -t12[i * k2 + j] }
    }
}

/// Writes out the zeros and unit diagonal of transposed Householder vectors, replacing the panel's R entries.
fn explicit_vectors<T: Scalar>(panel: &mut [T], rows: usize) {
    for (c, v) in panel.chunks_exact_mut(rows).enumerate() {
        v[..c].fill(T::zero());
        v[c] = T::one();
    }
}

/// The vectors of reflections `c0..c1` in a QR block that `panel` holds transposed, with rows of `rows` items: one vector to a row, each
/// from row `c0` on, with its zeros and its 1 written out.
fn vectors<T: Scalar>(panel: &[T], rows: usize, [c0, c1]: [usize; 2]) -> Vec<T> {
    let mut v = Vec::with_capacity((c1 - c0) * (rows - c0));
    for column in panel[c0 * rows..c1 * rows].chunks_exact(rows) { v.extend_from_slice(&column[c0..]) }
    explicit_vectors(&mut v, rows - c0);
    v
}

/// Factors columns `c0..c1` of the QR block that `panel` holds transposed, with rows of `rows` items. Each column's reflection vector
/// replaces it below the diagonal, and R replaces it on and above. Its τ goes on the diagonal of `t`, which holds T, `nb`×`nb`, and the
/// rest of T's rows and columns `c0..c1` above it. It halves the columns until `LEAF` or fewer remain. The left half's reflections reach
/// the right half through three products, and `joined` fills T's block between the halves.
fn panel_qr<T: Tiled>(panel: &mut [T], rows: usize, [c0, c1]: [usize; 2], t: &mut [T]) {
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
    // The right half, X ← (I - V₁ T₁ᴴ V₁ᴴ) X, as Xᵀ ← Xᵀ - Xᵀ conj(V₁) conj(T₁) V₁ᵀ in the transposed layout.
    let x = &mut panel[mid * rows..c1 * rows];
    let w = T::multiply(Matrix::rows(&x[c0..], rows), Matrix::rows(&vt, r).adjoint(), [k2, r, k1]);
    let z = T::multiply(Matrix::rows(&w, k1), Matrix::rows(&t[c0 * nb + c0..], nb).conjugate(), [k2, k1, k1]);
    T::gemm(&mut x[c0..], rows, Matrix::rows(&z, k1), Matrix::rows(&vt, r), [k2, k1, r], -T::one(), T::one());
    panel_qr(panel, rows, [mid, c1], t);
    joined(panel, rows, [c0, mid, c1], t)
}

/// One block of a product of Householder reflections, `I - V T Vᴴ`, for reflections `k0` to `k0 + nb - 1`, acting on rows `k0` on.
/// Row `c` of `vt` holds column `c` of V, reflection `k0 + c`, with its zeros and unit diagonal explicit. T is `triangular`'s.
struct Block<T> {
    k0: usize,
    nb: usize,
    vt: Vec<T>,
    t: Vec<T>,
}

impl<T: Tiled> Block<T> {
    /// The block of reflections `k0` on, with the vectors in the rows of `panel` and the τs `taus`.
    fn new(k0: usize, panel: Vec<T>, taus: &[T]) -> Self {
        let nb = taus.len();
        let mut t = filled(nb * nb, T::zero());
        for (c, &tau) in taus.iter().enumerate() { t[c * nb + c] = tau }
        triangular(&panel, panel.len() / nb, [0, nb], &mut t);
        Self::from_panel(k0, nb, panel, t)
    }

    /// Reuses a finished panel as Vᵀ. Its R entries are no longer needed after T and R have been read.
    fn from_panel(k0: usize, nb: usize, mut panel: Vec<T>, t: Vec<T>) -> Self {
        let rows = panel.len() / nb;
        explicit_vectors(&mut panel, rows);
        Self { k0, nb, vt: panel, t }
    }

    /// Applies `I - V T Vᴴ`, or its adjoint `I - V Tᴴ Vᴴ` where `adjoint` is true, to columns `c0..c1` of the matrix `x`, whose rows start
    /// `ld` items apart.
    fn apply(&self, x: &mut [T], ld: usize, [c0, c1]: [usize; 2], adjoint: bool) {
        let (k0, nb, rows, cols) = (self.k0, self.nb, self.vt.len() / self.nb, c1 - c0);
        // Keep both intermediates adjointed: Xᴴ V, then Xᴴ V Tᴴ (or T). This needs no conjugate-only view of the large V.
        let t = Matrix::rows(&self.t, nb);
        let t = if adjoint { t } else { t.adjoint() };
        let v = Matrix::rows(&self.vt, rows).transpose();
        let wt = T::multiply(Matrix::rows(&x[k0 * ld + c0..], ld).adjoint(), v, [cols, rows, nb]);
        let twt = T::multiply(Matrix::rows(&wt, nb), t, [cols, nb, nb]);
        T::gemm(&mut x[k0 * ld + c0..], ld, v, Matrix::rows(&twt, nb).adjoint(), [rows, nb, cols], -T::one(), T::one());
    }
}

/// The blocks of `width` reflections that `taus` holds. `panel(k0, k1)` gives reflections `k0..k1` as the rows of a transposed panel,
/// which `Block::new` takes.
fn blocks_of<T: Tiled>(taus: &[T], width: usize, panel: impl Fn(usize, usize) -> Vec<T>) -> Vec<Block<T>> {
    let p = taus.len();
    (0..p).step_by(width).map(|k0| { let k1 = (k0 + width).min(p); Block::new(k0, panel(k0, k1), &taus[k0..k1]) }).collect()
}

/// Householder reflections of `m`-item columns, in blocks, as the `m`×`m` product Q of their matrices, first to last.
struct Reflections<T> {
    m: usize,
    blocks: Vec<Block<T>>,
}

/// The QR factorization of `a`: its reflections, by blocks of `qr_width(p)` columns, and the `p`×`n` upper triangular R, where `p` is the
/// smaller of `m` and `n`. Each block's columns factor through `panel_qr` in a transposed copy, where each column is contiguous. The block's
/// reflections then update the columns after it, in the WY form.
fn reflect<T: Tiled>(a: &[T], [m, n]: [usize; 2]) -> (Reflections<T>, Vec<T>) {
    assert_eq!(a.len(), m * n, "`a` must hold `m`×`n` items");
    let p = m.min(n);
    let (ld, scale) = (stride(n), safe_scale(a));
    let mut f = spaced(a, [m, n], ld);
    if let Some(s) = scale { for x in &mut f { *x = x.mul_real(s) } }
    let mut blocks = Vec::new();
    let width = qr_width(p);
    for k0 in (0..p).step_by(width) {
        let k1 = (k0 + width).min(p);
        let (nb, rows) = (k1 - k0, m - k0);
        let mut panel = transposed(&f, ld, k0, k1);
        let mut t = filled(nb * nb, T::zero());
        panel_qr(&mut panel, rows, [0, nb], &mut t);
        for r in 0..nb {
            for c in r..nb { f[(k0 + r) * ld + k0 + c] = panel[c * rows + r] }
        }
        let block = Block::from_panel(k0, nb, panel, t);
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

impl<T: Tiled> Reflections<T> {
    /// Each reflection, first to last: the row it acts from, its τ, and its vector after the 1 at that row.
    fn each(&self) -> impl DoubleEndedIterator<Item = (usize, T, &[T])> {
        self.blocks.iter().flat_map(move |b| {
            let rows = self.m - b.k0;
            (0..b.nb).map(move |c| (b.k0 + c, b.t[c * b.nb + c], &b.vt[c * rows + c + 1..(c + 1) * rows]))
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
        compact(&mut q, [m, p], ld);
        q
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
fn qr_factor<T: Tiled>(a: &[T], [m, n]: [usize; 2], complete: bool) -> (Vec<T>, Vec<T>) {
    if !complete && tall_for_qr([m, n]) && let Some((r, q)) = tall_qr(a, [m, n], Tall::Q) { return (q, r) }
    let (reflections, mut r) = reflect(a, [m, n]);
    if !complete { return (reflections.q(), r) }
    let mut q = identity(m);
    reflections.apply(&mut q, m);
    r.resize(m * n, T::zero());
    (q, r)
}

/// `Svd::solve` and `Svd::pseudoinverse` for `T`.
fn svd_solve<T: Tiled>(svd: SvdRef<'_, T>, b: Option<&[T]>, k: usize) -> Vec<T> {
    let ([m, n], p) = (svd.size, svd.s.len());
    if let Some(b) = b { assert_eq!(b.len(), m * k, "`b` must hold `m`×`k` items") }
    assert!(svd.u.len() == m * p && svd.v.len() == n * p, "the SVD must hold its singular vectors");
    pseudo_solve(svd, b, k, m.max(n))
}

/// `V diag(1/s) Uᴴ b` for the factors that `svd` holds and a `b` of `k` columns, or `V diag(1/s) Uᴴ` without `b`, where `1/s` is zero
/// for each singular value no greater than `ε` times `scale` times the largest.
fn pseudo_solve<T: Tiled>(svd: SvdRef<'_, T>, b: Option<&[T]>, k: usize, scale: usize) -> Vec<T> {
    let ([m, n], p) = (svd.size, svd.s.len());
    if k == 0 { return Vec::new() }
    let floor = cutoff(&svd.s, scale);
    let mut y = match b {
        Some(b) => T::multiply(Matrix::rows(svd.u, p).adjoint(), Matrix::rows(b, k), [p, m, k]),
        None => adjoint(&svd.u, m, p),
    };
    for (row, &s) in y.chunks_exact_mut(k).zip(svd.s) {
        let d = if s > floor { s.recip() } else { T::Real::zero() };
        for x in row { *x = x.mul_real(d) }
    }
    T::multiply(Matrix::rows(svd.v, p), Matrix::rows(&y, k), [n, p, k])
}

/// The size below which a singular value counts as zero, for a matrix whose singular values `s` descend and whose larger side has `scale`
/// items: `ε` times `scale` times the largest, as in NumPy's `lstsq`.
fn cutoff<R: crate::Float>(s: &[R], scale: usize) -> R { R::epsilon() * R::narrow(scale as f64) * s.first().copied().unwrap_or(R::zero()) }

/// `lstsq` for `T`. A matrix with at least as many rows as columns reduces to the R of its QR factorization, and `b` to `Qᴴ b`, without
/// forming Q: through `tall_qr` where that succeeds, and otherwise through the reflections. A wider matrix solves through its own SVD.
fn lstsq_solve<T: Tiled>(a: &[T], b: &[T], [m, n]: [usize; 2], k: usize) -> (Vec<T>, Vec<T::Real>) {
    assert_eq!(b.len(), m * k, "`b` must hold `m`×`k` items");
    if m < n {
        let svd = svd_factor(a, [m, n], true);
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
    let svd = svd_factor(&r, [n, n], true);
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

/// Whether a matrix of `size` `[m, n]` holds enough work for CholeskyQR2's threaded passes to be faster than Householder: at least 16
/// rows per column, and `m n²` of at least `work`. Narrow matrices such as 1000000×10 pass. Where Accelerate runs the products, CholeskyQR2
/// is faster for them on one thread too. Elsewhere the call must be able to split across threads: in wasm and on one thread, Householder is
/// faster for them.
fn threaded_work(m: usize, n: usize, work: usize) -> bool {
    (super::accelerated() || !parallel::serial()) && m >= 16 * n && m.saturating_mul(n.saturating_mul(n)) >= work
}

/// What `tall_qr` gives beside R: Q, nothing, or `Qᴴ b` for the `m`×`k` matrix `b`.
enum Tall<'a, T> { Q, #[cfg(native)] R, Project(&'a [T], usize) }

/// The thin QR factorization of a tall `a` by CholeskyQR2: R, with Q or `Qᴴ b` as `want` asks. It scales
/// `a` as `reflect` does. `None` where `a` is too ill-conditioned for the result to be as accurate as Householder QR's: where `AᴴA` has no
/// Cholesky factor, or where the first pass's Q₁ is too far from orthonormal for the second pass to correct.
///
/// Pass one factors `AᴴA = L₁ L₁ᴴ`. Pass two solves `Q₁ L₁ᴴ = A` and factors `Q₁ᴴQ₁ = L₂ L₂ᴴ`. Then `Q = Q₁ L₂⁻ᴴ`, `Qᴴ b = L₂⁻¹ Q₁ᴴ b` and
/// `R = (L₁ L₂)ᴴ`, whose diagonal is real and positive. Each pass gives every thread a contiguous range of `a`'s rows and its own partial
/// Gram matrix, and works through the range in blocks of `TALL_BLOCK` items. Pass two stores Q₁ only for `Tall::Q`.
fn tall_qr<T: Tiled>(a: &[T], [m, n]: [usize; 2], want: Tall<'_, T>) -> Option<(Vec<T>, Vec<T>)> {
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
            T::gram(&mut g[0], n, Matrix::rows(x, n).adjoint(), [n, x.len() / n], T::Real::one(), T::Real::one());
        }
        true
    });
    let l1 = cholesky_factor(&summed(grams.iter()), n)?;
    let mut q = if matches!(want, Tall::Q) { a.to_vec() } else { Vec::new() };
    let mut rest = &mut q[..];
    let outs: Vec<&mut [T]> = (0..count).map(|p| {
        let (out, tail) = std::mem::take(&mut rest).split_at_mut(if matches!(want, Tall::Q) { range(p).len() * n } else { 0 });
        rest = tail;
        out
    }).collect();
    let mut parts: Vec<_> = outs.into_iter().map(|out| (out, filled(n * n, T::zero()), filled(n * k, T::zero()), Vec::new())).collect();
    parallel::pieces_of(&mut parts, 1, usize::MAX, &|p, part| {
        let (out, gram, y, scratch) = &mut part[0];
        let (start, end) = (range(p).start, range(p).end);
        for r in (start..end).step_by(rows).map(|s| s..end.min(s + rows)) {
            let h = r.len();
            let src = &a[r.start * n..r.end * n];
            let q1 = if out.is_empty() {
                scratch.clear();
                scratch.extend_from_slice(src);
                &mut scratch[..]
            } else {
                &mut out[(r.start - start) * n..(r.end - start) * n]
            };
            T::triangular(q1, n, Matrix::rows(&l1, n).adjoint(), [h, n], Side::Right, Triangle::Lower);
            let qt = Matrix::rows(q1, n).adjoint();
            T::gram(gram, n, qt, [n, h], T::Real::one(), T::Real::one());
            if let Tall::Project(b, k) = want {
                T::gemm(y, k, qt, Matrix::rows(&b[r.start * k..r.end * k], k), [n, h, k], T::one(), T::one());
            }
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
    let l = T::multiply(Matrix::rows(&l1, n), Matrix::rows(&l2, n), [n, n, n]);
    let mut r = adjoint(&l, n, n);
    if let Some(s) = scale { for v in &mut r { *v = v.div_real(s) } }
    if !matches!(want, Tall::Q) {
        T::triangular(&mut y, k, Matrix::rows(&l2, n), [n, k], Side::Left, Triangle::Lower);
        return Some((r, y));
    }
    parallel::pieces_of(&mut q, rows * n, usize::MAX, &|_, band| {
        let h = band.len() / n;
        T::triangular(band, n, Matrix::rows(&l2, n).adjoint(), [h, n], Side::Right, Triangle::Lower);
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

