//! Accelerate's BLAS and LAPACK serve every plus-times product, triangular solve, SVD and Hermitian eigen decomposition on macOS, and the general eigen problem of real matrices. A complex matrix's general eigen problem takes the native solver, which ran 2.4 to 2.9 times as fast as Accelerate's `?geev`. The ILP64 interface takes 64-bit sizes, so no matrix is too large for it. LAPACK stores a matrix column by column, and reads a row-major one as its transpose. `?heevd`, and `?gesdd` on a wide matrix, work with that transpose. `?geev`, and `?gesdd` on a tall matrix, take a transposed copy. Each working copy's leading dimension comes from `stride`. Unpadded, a 512×512 matrix took twice as long in `?geev` as one of 511 or 513, 27% longer in `?gesdd` and 8% longer in `?syevd`.
use super::{Eig, Eigh, Matrix, Op, Part, Scalar, Side, Triangle, factor::{spaced, stride, transpose_by, transpose_by_into}, filled};
use crate::{Float, Native};
use num_complex::Complex;
use num_traits::Zero;
use std::{borrow::Cow, cell::Cell, ptr};

/// The ILP64 interface's integer, for sizes, leading dimensions and LAPACK's `info`.
type Int = i64;

/// CBLAS's code for row-major storage, which every BLAS call passes.
const ROW_MAJOR: i32 = 101;

#[link(name = "Accelerate", kind = "framework")]
unsafe extern "C" { fn BLASSetThreading(threading: u32) -> i32; }

thread_local! {
    /// Whether this thread last selected single-threaded BLAS. `None` until its first call.
    static SERIAL: Cell<Option<bool>> = const { Cell::new(None) };
}

/// Sets this thread's BLAS threading from rustymath's thread setting, only when that choice changes. Accelerate runs on one thread where rustymath runs serially, as on a batch's worker threads or after `set_threads(1)`, and on its own threads otherwise. NumPy's calls into Accelerate on the same thread use the same setting.
fn follow_threads() {
    let serial = crate::parallel::serial();
    SERIAL.with(|last| {
        if last.get() != Some(serial) {
            // SAFETY: 0 is `BLAS_THREADING_MULTI_THREADED` and 1 is `BLAS_THREADING_SINGLE_THREADED`.
            unsafe { BLASSetThreading(u32::from(serial)) };
            last.set(Some(serial));
        }
    })
}

/// Whether `T` is complex, with parts of half its width.
fn complex<T: Scalar>() -> bool { size_of::<T>() != size_of::<T::Real>() }

/// `a`, a `rows`×`cols` matrix in its own orientation, as BLAS's data, leading dimension and operation code. Only an operation that the BLAS function lacks makes a row-major copy. No function takes a conjugate alone. SYRK and HERK take the matrix as it is, or SYRK its transpose and HERK its adjoint, which `gram` selects.
fn prepared<T: Scalar>(a: Matrix<'_, T>, [rows, cols]: [usize; 2], gram: bool) -> (Cow<'_, [T]>, Int, i32) {
    let adjoint = if complex::<T>() { 113 } else { 112 };
    let op = match a.op { Op::Normal => 111, Op::Transpose => 112, Op::Adjoint => adjoint, Op::Conjugate => { if complex::<T>() { 0 } else { 111 } } };
    if op == 0 || (gram && op != 111 && op != adjoint) {
        let data = if a.op.transposed() { transpose_by(a.data, a.ld, [cols, rows], |x| x) } else {
            let mut data = Vec::with_capacity(rows * cols);
            for i in 0..rows { data.extend(a.data[i * a.ld..][..cols].iter().map(|&x| x.conj())) }
            data
        };
        return (Cow::Owned(data), cols as Int, 111);
    }
    (Cow::Borrowed(a.data), a.ld as Int, op)
}

/// `c = alpha a b + beta c` for the items of `c` in `part`, with `size` `[m, k, n]`, through `?gemm`. `Part::Lower` takes `b` as the adjoint of `a`, and changes only the lower triangle of `c`, through `?syrk` or `?herk`.
pub(super) fn product<T: Accelerated>(c: &mut [T], ldc: usize, a: Matrix<'_, T>, b: Matrix<'_, T>, [m, k, n]: [usize; 3], alpha: T, beta: T, part: Part) {
    follow_threads();
    match part {
        Part::Whole => {
            let ((a, lda, ta), (b, ldb, tb)) = (prepared(a, [m, k], false), prepared(b, [k, n], false));
            // SAFETY: `Tiled::gemm` checked that each matrix holds its rows and columns.
            unsafe { T::xgemm([ta, tb], [m, k, n].map(|d| d as Int), alpha, (a.as_ptr(), lda), (b.as_ptr(), ldb), beta, (c.as_mut_ptr(), ldc as Int)) }
        }
        Part::Lower => {
            let (x, ldx, trans) = prepared(a, [m, k], true);
            // SAFETY: as for `Part::Whole`.
            unsafe { T::xherk(trans, [m as Int, k as Int], alpha.re(), (x.as_ptr(), ldx), beta.re(), (c.as_mut_ptr(), ldc as Int)) }
        }
    }
}

/// Solves `a x = b` or `x a = b` in place through `?trsm`, with `triangle` describing the stored square matrix `a`.
pub(super) fn triangular<T: Accelerated>(x: &mut [T], ldx: usize, a: Matrix<'_, T>, [m, n]: [usize; 2], side: Side, triangle: Triangle) {
    let (order, side) = match side { Side::Left => (m, 141), Side::Right => (n, 142) };
    let (a, lda, trans) = prepared(a, [order, order], false);
    let (uplo, diag) = match triangle { Triangle::Lower => (122, 131), Triangle::UnitUpper => (121, 132) };
    follow_threads();
    // SAFETY: `Tiled::triangular` checked that both matrices hold their rows and columns.
    unsafe { T::xtrsm([side, uplo, trans, diag], [m as Int, n as Int], (a.as_ptr(), lda), (x.as_mut_ptr(), ldx as Int)) }
}

/// A workspace size that LAPACK gives as a float, rounded up to a whole number of items, and at least one.
fn workspace(size: f64) -> usize { (size.ceil() as usize).max(1) }

/// The thin SVD of the nonempty row-major `m`×`n` matrix `a` as `(U, s, V)`, with the singular vectors only where `vectors` is true, or `None` where `?gesdd` doesn't converge. A tall matrix factors through its adjoint, whose SVD is `V Σ Uᴴ`. LAPACK reads a tall row-major matrix as a wide one. Accelerate's `?gesdd` took nine times as long on a 4096×16 matrix that it saw as wide.
pub(super) fn svd<T: Accelerated>(a: &[T], [m, n]: [usize; 2], vectors: bool) -> Option<(Vec<T>, Vec<T::Real>, Vec<T>)> {
    if m <= n { return gesdd(spaced(a, [m, n], stride(n)), stride(n), [m, n], vectors); }
    let ld = stride(m);
    let mut x = Vec::new();
    transpose_by_into(a, [n, ld], [m, n], T::conj, &mut x);
    let (u, s, v) = gesdd(x, ld, [n, m], vectors)?;
    Some((v, s, u))
}

/// `svd` of the row-major `m`×`n` matrix in `x`, whose rows lie `ldx` items apart, where `m` is at most `n`. `?gesdd` factors the column-major `n`×`m` matrix that `x`'s items form, `xᵀ = W Σ Zᴴ`, so `x = conj(Z) Σ Wᵀ`. Zᴴ's column-major items are U's row-major ones. V is W conjugated.
fn gesdd<T: Accelerated>(mut x: Vec<T>, ldx: usize, [m, n]: [usize; 2], vectors: bool) -> Option<(Vec<T>, Vec<T::Real>, Vec<T>)> {
    let p = m.min(n);
    let (job, [ldw, ldu]) = if vectors { (b'S', [stride(n), p]) } else { (b'N', [1, 1]) };
    let mut s = filled(p, T::Real::zero());
    let (mut w, mut u) = if vectors { (filled(ldw * p, T::zero()), filled(p * m, T::zero())) } else { (vec![T::zero()], vec![T::zero()]) };
    let rwork = if !complex::<T>() { 0 } else if vectors { (5 * p * p + 5 * p).max(2 * m.max(n) * p + 2 * p * p + p) } else { 7 * p };
    let (mut rwork, mut iwork) = (filled(rwork, T::Real::zero()), filled(8 * p, 0));
    // SAFETY: each array holds what `?gesdd` reads and writes for this job and size, and `work` holds `lwork` items.
    let mut call = |work: &mut [T], lwork: Int| unsafe {
        T::xgesdd(
            job, [n as Int, m as Int], (x.as_mut_ptr(), ldx as Int), s.as_mut_ptr(), (w.as_mut_ptr(), ldw as Int), (u.as_mut_ptr(), ldu as Int),
            (work.as_mut_ptr(), lwork), rwork.as_mut_ptr(), iwork.as_mut_ptr())
    };
    follow_threads();
    let mut size = [T::zero()];
    if call(&mut size, -1) != 0 { return None; }
    let mut work = filled(workspace(size[0].re().wide()), T::zero());
    let lwork = work.len() as Int;
    if call(&mut work, lwork) != 0 { return None; }
    if !vectors { return Some((Vec::new(), s, Vec::new())); }
    let v = transpose_by(&w, ldw, [p, n], T::conj);
    Some((u, s, v))
}

/// The eigen decomposition of the nonempty Hermitian `n`×`n` matrix whose lower triangle `a` holds, through `?syevd` or `?heevd`, or `None` where it doesn't converge. Those read the upper triangle of the column-major matrix that `a`'s items form, which is `a`'s lower triangle. They factor `conj(a)`, whose eigenvalues are `a`'s and whose eigenvectors are `a`'s conjugated.
pub(super) fn eigh<T: Accelerated>(a: &[T], n: usize, vectors: bool) -> Option<Eigh<T>> {
    let job = if vectors { b'V' } else { b'N' };
    let ld = stride(n);
    let (mut x, mut values) = (spaced(a, [n, n], ld), filled(n, T::Real::zero()));
    // SAFETY: `x` holds `n` rows `ld` items apart, `values` holds `n` items, and each workspace holds its length.
    let mut heevd = |work: &mut [T], rwork: &mut [T::Real], iwork: &mut [Int], [lwork, lrwork, liwork]: [Int; 3]| unsafe {
        T::xheevd(
            job, n as Int, (x.as_mut_ptr(), ld as Int), values.as_mut_ptr(), (work.as_mut_ptr(), lwork), (rwork.as_mut_ptr(), lrwork),
            (iwork.as_mut_ptr(), liwork))
    };
    follow_threads();
    let (mut work, mut rwork, mut iwork) = (vec![T::zero()], vec![T::Real::zero()], vec![0]);
    if heevd(&mut work, &mut rwork, &mut iwork, [-1; 3]) != 0 { return None; }
    let (mut work, mut rwork, mut iwork) =
        (filled(workspace(work[0].re().wide()), T::zero()), filled(workspace(rwork[0].wide()), T::Real::zero()), filled(iwork[0].max(1) as usize, 0));
    let lengths = [work.len(), rwork.len(), iwork.len()].map(|l| l as Int);
    if heevd(&mut work, &mut rwork, &mut iwork, lengths) != 0 { return None; }
    let vectors = if vectors { transpose_by(&x, ld, [n, n], T::conj) } else { Vec::new() };
    Some(Eigh { values, vectors })
}

/// The order from which real matrices' general eigen problem takes `?geev`. Up to 128 rows, `?geev` took 1.2 to 2.7 times as long as the native solver with Accelerate's products, on one thread of an M5 Max. At 256, `?geev` was level for eigenvalues alone and 1.2 times as fast with eigenvectors. From 512, with its copy padded, it was about twice as fast.
pub(super) const GEEV_FROM: usize = 256;

/// The eigen decomposition of the nonempty row-major real `n`×`n` matrix `a`, with its eigenvectors only where `vectors` is true, or `None` where `?geev` doesn't converge.
pub(super) fn eig<T: Geev>(a: &[T], n: usize, vectors: bool) -> Option<Eig<T>> { follow_threads(); T::xgeev(a, n, vectors) }

/// `Geev::xgeev`. `geev` calls `?geev` with the job for right eigenvectors, a column-major copy of `a` with its leading dimension, `[wr, wi]`, the right eigenvectors with their leading dimension, and the workspace with its length. `?geev` gives unit eigenvectors. It gives a complex pair's as `x + iy` and `x - iy` in the columns `x` and `y`, with the eigenvalue whose imaginary part is positive first.
fn real_eig<R: Native>(
    a: &[R], n: usize, vectors: bool, mut geev: impl FnMut(u8, (&mut [R], Int), [&mut [R]; 2], (&mut [R], Int), (&mut [R], Int)) -> Int) -> Option<Eig<R>> {
    let (job, ld) = (if vectors { b'V' } else { b'N' }, stride(n));
    let (mut x, mut wr, mut wi) = (Vec::new(), filled(n, R::zero()), filled(n, R::zero()));
    transpose_by_into(a, [n, ld], [n, n], |v| v, &mut x);
    let (mut vr, ldvr) = if vectors { (filled(n * ld, R::zero()), ld) } else { (vec![R::zero()], 1) };
    let mut size = [R::zero()];
    if geev(job, (&mut x, ld as Int), [&mut wr, &mut wi], (&mut vr, ldvr as Int), (&mut size, -1)) != 0 { return None; }
    let mut work = filled(workspace(size[0].wide()), R::zero());
    let lwork = work.len() as Int;
    if geev(job, (&mut x, ld as Int), [&mut wr, &mut wi], (&mut vr, ldvr as Int), (&mut work, lwork)) != 0 { return None; }
    let values = wr.iter().zip(&wi).map(|(&re, &im)| Complex::new(re, im)).collect();
    if !vectors { return Some(Eig { values, vectors: Vec::new() }); }
    let mut v = filled(n * n, Complex::zero());
    let mut j = 0;
    while j < n {
        let pair = wi[j] != R::zero();
        for r in 0..n {
            let x = vr[j * ldvr + r];
            if !pair { v[r * n + j] = Complex::new(x, R::zero()); continue; }
            let y = vr[(j + 1) * ldvr + r];
            (v[r * n + j], v[r * n + j + 1]) = (Complex::new(x, y), Complex::new(x, -y));
        }
        j += if pair { 2 } else { 1 };
    }
    Some(Eig { values, vectors: v })
}

/// `?geev` for a real type. A complex matrix's general eigen problem takes the native solver instead.
pub(crate) trait Geev: Sized {
    /// The eigen decomposition of the nonempty row-major `n`×`n` matrix `a` through `?geev`, with its eigenvectors only where `vectors` is true, or `None` where it doesn't converge.
    fn xgeev(a: &[Self], n: usize, vectors: bool) -> Option<Eig<Self>>;
}

/// Implements `Geev` for each real type from its `?geev` symbol.
macro_rules! geev {
    ($($t:ty: $geev:literal;)+) => {$(
        impl Geev for $t {
            fn xgeev(a: &[$t], n: usize, vectors: bool) -> Option<Eig<$t>> {
                unsafe extern "C" {
                    #[link_name = $geev]
                    fn f(jobvl: *const u8, jobvr: *const u8, n: *const Int, a: *mut $t, lda: *const Int, wr: *mut $t, wi: *mut $t, vl: *mut $t, ldvl: *const Int, vr: *mut $t, ldvr: *const Int, work: *mut $t, lwork: *const Int, info: *mut Int);
                }
                let size = n as Int;
                real_eig(a, n, vectors, |job, (x, lda), [wr, wi], (vr, ldvr), (work, lwork)| {
                    let mut info = 0;
                    // SAFETY: `real_eig` sizes each array for `n` and the job. `?geev` doesn't read `vl` without left eigenvectors.
                    unsafe { f(&b'N', &job, &size, x.as_mut_ptr(), &lda, wr.as_mut_ptr(), wi.as_mut_ptr(), ptr::null_mut(), &1, vr.as_mut_ptr(), &ldvr, work.as_mut_ptr(), &lwork, &mut info) };
                    info
                })
            }
        }
    )+};
}
geev! {
    f32: "sgeev$NEWLAPACK$ILP64";
    f64: "dgeev$NEWLAPACK$ILP64";
}

/// Accelerate's BLAS and LAPACK functions for one item type: BLAS row-major, LAPACK column-major. The table at the end of this file implements it. Each pointer comes with its leading dimension or length. A real type ignores `rwork`. Each LAPACK function gives its `info`, and a length of -1 asks it for its workspace sizes instead, in the first item of each workspace.
pub(crate) trait Accelerated: Scalar {
    /// `?gemm`: `c = alpha op(a) op(b) + beta c`, with `size` `[m, k, n]`.
    unsafe fn xgemm(trans: [i32; 2], size: [Int; 3], alpha: Self, a: (*const Self, Int), b: (*const Self, Int), beta: Self, c: (*mut Self, Int));
    /// `?syrk` or `?herk`: the lower triangle of `c = alpha op(x) op(x)ᴴ + beta c`, with `size` `[n, k]`.
    unsafe fn xherk(trans: i32, size: [Int; 2], alpha: Self::Real, x: (*const Self, Int), beta: Self::Real, c: (*mut Self, Int));
    /// `?trsm` with `alpha` 1, with the codes for its side, triangle, operation and diagonal.
    unsafe fn xtrsm(codes: [i32; 4], size: [Int; 2], a: (*const Self, Int), x: (*mut Self, Int));
    /// `?gesdd` on the `m`×`n` matrix `a`.
    unsafe fn xgesdd(
        job: u8, size: [Int; 2], a: (*mut Self, Int), s: *mut Self::Real, u: (*mut Self, Int), vt: (*mut Self, Int), work: (*mut Self, Int),
        rwork: *mut Self::Real, iwork: *mut Int) -> Int;
    /// `?syevd` or `?heevd` on the upper triangle of the `n`×`n` matrix `a`.
    unsafe fn xheevd(
        job: u8, n: Int, a: (*mut Self, Int), w: *mut Self::Real, work: (*mut Self, Int), rwork: (*mut Self::Real, Int), iwork: (*mut Int, Int)) -> Int;
}

/// The LAPACK functions of `Accelerated`. The complex `?gesdd` and `?heevd` take an extra `rwork` workspace. Real types ignore that argument.
macro_rules! lapack {
    (real $($args:tt)*) => { lapack!(@ $($args)*); };
    (complex $($args:tt)*) => { lapack!(@ $($args)*, rwork); };
    (@ $t:ty, $r:ty, $gesdd:literal, $heevd:literal $(, $rwork:ident)?) => {
        unsafe fn xgesdd(job: u8, [m, n]: [Int; 2], (a, lda): (*mut $t, Int), s: *mut $r, (u, ldu): (*mut $t, Int), (vt, ldvt): (*mut $t, Int), (work, lwork): (*mut $t, Int), $($rwork @)? _: *mut $r, iwork: *mut Int) -> Int {
            unsafe extern "C" {
                #[link_name = $gesdd]
                fn f(job: *const u8, m: *const Int, n: *const Int, a: *mut $t, lda: *const Int, s: *mut $r, u: *mut $t, ldu: *const Int, vt: *mut $t, ldvt: *const Int, work: *mut $t, lwork: *const Int, $($rwork: *mut $r,)? iwork: *mut Int, info: *mut Int);
            }
            let mut info = 0;
            unsafe { f(&job, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, $($rwork,)? iwork, &mut info) };
            info
        }
        unsafe fn xheevd(job: u8, n: Int, (a, lda): (*mut $t, Int), w: *mut $r, (work, lwork): (*mut $t, Int), $($rwork @)? _: (*mut $r, Int), (iwork, liwork): (*mut Int, Int)) -> Int {
            unsafe extern "C" {
                #[link_name = $heevd]
                fn f(job: *const u8, uplo: *const u8, n: *const Int, a: *mut $t, lda: *const Int, w: *mut $r, work: *mut $t, lwork: *const Int, $($rwork: *mut $r, lrwork: *const Int,)? iwork: *mut Int, liwork: *const Int, info: *mut Int);
            }
            let mut info = 0;
            unsafe { f(&job, &b'U', &n, a, &lda, w, work, &lwork, $($rwork.0, &$rwork.1,)? iwork, &liwork, &mut info) };
            info
        }
    };
}

/// Implements `Accelerated` for each item type from its BLAS and LAPACK symbols. Real BLAS scalars pass by value and complex ones by pointer, which `$pass` converts.
macro_rules! accelerated {
    ($($kind:ident $t:ty, $r:ty, $scalar:ty, $pass:expr, [$gemm:literal, $gram:literal, $trsm:literal, $gesdd:literal, $heevd:literal];)+) => {$(
        impl Accelerated for $t {
            unsafe fn xgemm([ta, tb]: [i32; 2], [m, k, n]: [Int; 3], alpha: $t, (a, lda): (*const $t, Int), (b, ldb): (*const $t, Int), beta: $t, (c, ldc): (*mut $t, Int)) {
                unsafe extern "C" {
                    #[link_name = $gemm]
                    fn f(order: i32, ta: i32, tb: i32, m: Int, n: Int, k: Int, alpha: $scalar, a: *const $t, lda: Int, b: *const $t, ldb: Int, beta: $scalar, c: *mut $t, ldc: Int);
                }
                unsafe { f(ROW_MAJOR, ta, tb, m, n, k, ($pass)(&alpha), a, lda, b, ldb, ($pass)(&beta), c, ldc) }
            }
            unsafe fn xherk(trans: i32, [n, k]: [Int; 2], alpha: $r, (x, ldx): (*const $t, Int), beta: $r, (c, ldc): (*mut $t, Int)) {
                unsafe extern "C" {
                    #[link_name = $gram]
                    fn f(order: i32, uplo: i32, trans: i32, n: Int, k: Int, alpha: $r, x: *const $t, ldx: Int, beta: $r, c: *mut $t, ldc: Int);
                }
                unsafe { f(ROW_MAJOR, 122, trans, n, k, alpha, x, ldx, beta, c, ldc) }
            }
            unsafe fn xtrsm([side, uplo, trans, diag]: [i32; 4], [m, n]: [Int; 2], (a, lda): (*const $t, Int), (x, ldx): (*mut $t, Int)) {
                unsafe extern "C" {
                    #[link_name = $trsm]
                    fn f(order: i32, side: i32, uplo: i32, trans: i32, diag: i32, m: Int, n: Int, alpha: $scalar, a: *const $t, lda: Int, x: *mut $t, ldx: Int);
                }
                let one = <$t>::from_real(1.0);
                unsafe { f(ROW_MAJOR, side, uplo, trans, diag, m, n, ($pass)(&one), a, lda, x, ldx) }
            }
            lapack!($kind $t, $r, $gesdd, $heevd);
        }
    )+};
}
accelerated! {
    real f32, f32, f32, |x: &f32| *x, ["cblas_sgemm$NEWLAPACK$ILP64", "cblas_ssyrk$NEWLAPACK$ILP64", "cblas_strsm$NEWLAPACK$ILP64",
        "sgesdd$NEWLAPACK$ILP64", "ssyevd$NEWLAPACK$ILP64"];
    real f64, f64, f64, |x: &f64| *x, ["cblas_dgemm$NEWLAPACK$ILP64", "cblas_dsyrk$NEWLAPACK$ILP64", "cblas_dtrsm$NEWLAPACK$ILP64",
        "dgesdd$NEWLAPACK$ILP64", "dsyevd$NEWLAPACK$ILP64"];
    complex Complex<f32>, f32, *const Complex<f32>, |x: &Complex<f32>| x as *const _, ["cblas_cgemm$NEWLAPACK$ILP64",
        "cblas_cherk$NEWLAPACK$ILP64", "cblas_ctrsm$NEWLAPACK$ILP64", "cgesdd$NEWLAPACK$ILP64", "cheevd$NEWLAPACK$ILP64"];
    complex Complex<f64>, f64, *const Complex<f64>, |x: &Complex<f64>| x as *const _, ["cblas_zgemm$NEWLAPACK$ILP64",
        "cblas_zherk$NEWLAPACK$ILP64", "cblas_ztrsm$NEWLAPACK$ILP64", "zgesdd$NEWLAPACK$ILP64", "zheevd$NEWLAPACK$ILP64"];
}
