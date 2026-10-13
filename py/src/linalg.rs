//! rustymath's `linalg` module as NumPy generalized ufuncs, which `python/rustymath/linalg.py` wraps. A loop runs once for each matrix
//! of a stack. It copies each argument's core matrix into a row-major buffer in its working type, `f32` for `float16`, calls rustymath,
//! then writes the results through the outputs' strides, rounded to the loop's type. complex64 and complex128 matrices compute at their
//! own width. Results that are always real, such as singular values, have the real type of the loop's width. A matrix that fails gets
//! NaN results and makes the loop report NumPy's "invalid value", as a failing item of any rustymath ufunc does.
//!
//! A core dimension can't be the smaller of two others. Each function whose results have `min(m, n)` rows or items therefore
//! registers twice: once for `m ≤ n`, with a name ending `_m`, and once for `m ≥ n`, ending `_n`. One loop serves both.
use std::ffi::{c_char, c_void};
use std::slice;

use half::f16;
use num_complex::Complex;
use num_traits::Float as _;
use numpy::npyffi::{npy_intp, NPY_TYPES};
use pyo3::prelude::*;
use rustymath::linalg::{self, Factor, Inner, MaxPlus, MinPlus, OrAnd, PlusEqual, PlusNotEqual, PlusTimes, Scalar};
use rustymath::Float;

use crate::{fe, register, Code, Loop, NONE};

/// A generalized ufunc loop.
type Ufunc = unsafe extern "C" fn(*mut *mut c_char, *mut npy_intp, *mut npy_intp, *mut c_void);

/// A loop's element type, and `W`, the type rustymath computes it in.
trait Stored: Code {
    type W: Copy;
    fn load(self) -> Self::W;
    fn store(w: Self::W) -> Self;
}

impl Stored for f16 {
    type W = f32;
    fn load(self) -> f32 { self.working() }
    fn store(w: f32) -> f16 { f16::from_working(w) }
}

macro_rules! itself {
    ($($t:ty),+) => {$(impl Stored for $t { type W = $t; fn load(self) -> $t { self } fn store(w: $t) -> $t { w } })+};
}
itself!(f32, f64, Complex<f32>, Complex<f64>, bool, i64);

/// The real type that the item type `E` computes in, or computes its parts in: `f32` for `float16`, `float32` and `complex64`.
type Part<E> = <<E as Stored>::W as Scalar>::Real;

/// A float loop's element type: `float16`, `float32`, `float64`, `complex64` or `complex128`, which rustymath factors in its working
/// type.
trait Item: Stored<W: Factor> {
    /// The element type of results that are always real, such as singular values: the real type of the same width.
    type Real: Stored<W = Part<Self>>;
    /// The element type of results that are always complex, such as a general matrix's eigenvalues: the complex type of the same width.
    type Complex: Code;
}
macro_rules! items {
    ($($t:ty: $real:ty, $complex:ty;)+) => {$(impl Item for $t { type Real = $real; type Complex = $complex; })+};
}
items! {
    f16: f16, Complex<f32>;
    f32: f32, Complex<f32>;
    f64: f64, Complex<f64>;
    Complex<f32>: f32, Complex<f32>;
    Complex<f64>: f64, Complex<f64>;
}

/// Runs `each` on each matrix of the stack that a generalized ufunc loop with `nargs` arguments, `ndims` core dimensions and
/// `nstrides` core strides gets. `each` takes each argument's pointer at that matrix, the core dimensions, and the core strides in
/// argument order, and gives whether the matrix succeeded. Where one fails, the loop reports "invalid value".
unsafe fn stack(
    args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, [nargs, ndims, nstrides]: [usize; 3],
    mut each: impl FnMut(&[*mut c_char], &[usize], &[npy_intp]) -> bool,
) {
    let (n, outer, core) = unsafe { (*dims as usize, slice::from_raw_parts(steps, nargs), slice::from_raw_parts(steps.add(nargs), nstrides)) };
    let dims: Vec<usize> = unsafe { slice::from_raw_parts(dims.add(1), ndims) }.iter().map(|&d| d as usize).collect();
    let mut p = unsafe { slice::from_raw_parts(args, nargs) }.to_vec();
    let mut ok = true;
    for _ in 0..n {
        ok &= each(&p, &dims, core);
        for (p, &step) in p.iter_mut().zip(outer) { *p = p.wrapping_offset(step) }
    }
    unsafe {
        fe::feclearexcept(fe::REPORTED);
        if !ok { fe::feraiseexcept(fe::INVALID); }
    }
}

/// The `rows`×`cols` matrix at `p`, whose rows lie `rs` bytes apart and items `cs` bytes apart, row by row in `E`'s working type.
unsafe fn read<E: Stored>(p: *mut c_char, [rs, cs]: [npy_intp; 2], [rows, cols]: [usize; 2]) -> Vec<E::W> {
    let mut x = Vec::with_capacity(rows * cols);
    for i in 0..rows as isize {
        for j in 0..cols as isize { x.push(unsafe { (p.offset(i * rs + j * cs) as *const E).read_unaligned() }.load()) }
    }
    x
}

/// Writes the `rows`×`cols` matrix `x`, held row by row, through `f` to `p`, whose rows lie `rs` bytes apart and items `cs` bytes
/// apart. A vector is a matrix of one column, and a scalar one of one item.
unsafe fn write<W: Copy, O>(x: &[W], p: *mut c_char, [rs, cs]: [npy_intp; 2], [rows, cols]: [usize; 2], f: impl Fn(W) -> O) {
    for i in 0..rows {
        for j in 0..cols { unsafe { (p.offset(i as isize * rs + j as isize * cs) as *mut O).write_unaligned(f(x[i * cols + j])) } }
    }
}

/// `n` NaNs, the results of a matrix that fails.
fn nans<W: Scalar>(n: usize) -> Vec<W> { vec![W::from_real(W::Real::nan()); n] }

/// `n` complex NaNs, the complex results of a matrix of item type `E` that fails.
fn complex_nans<E: Item>(n: usize) -> Vec<Complex<Part<E>>> { vec![Complex::new(Part::<E>::nan(), Part::<E>::nan()); n] }

/// `solve`: rustymath's `solve` of `a x = b`, for an `m`×`n` matrix `a` and an `m`×`k` matrix `b`.
unsafe extern "C" fn solve<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [3, 3, 6], |p, d, s| {
            let [m, n, k] = [d[0], d[1], d[2]];
            let x = linalg::solve(&read::<E>(p[0], [s[0], s[1]], [m, n]), &read::<E>(p[1], [s[2], s[3]], [m, k]), [m, n], k);
            write(&x, p[2], [s[4], s[5]], [n, k], E::store);
            true
        })
    }
}

/// `pinv`: rustymath's `pinv` of an `m`×`n` matrix, which is `n`×`m`.
unsafe extern "C" fn pinv<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [2, 2, 4], |p, d, s| {
            let [m, n] = [d[0], d[1]];
            write(&linalg::pinv(&read::<E>(p[0], [s[0], s[1]], [m, n]), [m, n]), p[1], [s[2], s[3]], [n, m], E::store);
            true
        })
    }
}

/// `det`: the determinant of an `m`×`m` matrix, from its LU factorization.
unsafe extern "C" fn det<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [2, 1, 2], |p, d, s| {
            let lu = linalg::lu(&read::<E>(p[0], [s[0], s[1]], [d[0], d[0]]), d[0]);
            (p[1] as *mut E).write_unaligned(E::store(lu.determinant()));
            true
        })
    }
}

/// `slogdet`: the sign of the determinant of an `m`×`m` matrix and the natural logarithm of its magnitude, as `Lu::log_determinant`
/// gives them.
unsafe extern "C" fn slogdet<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [3, 1, 2], |p, d, s| {
            let (sign, log) = linalg::lu(&read::<E>(p[0], [s[0], s[1]], [d[0], d[0]]), d[0]).log_determinant();
            (p[1] as *mut E).write_unaligned(E::store(sign));
            (p[2] as *mut E::Real).write_unaligned(E::Real::store(log));
            true
        })
    }
}

/// `lu`: L and U of an `m`×`m` matrix's LU factorization, and the positions of its rows: row `i` of L U is row `p[i]` of the matrix.
unsafe extern "C" fn lu<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [4, 1, 7], |p, d, s| {
            let m = d[0];
            let f = linalg::lu(&read::<E>(p[0], [s[0], s[1]], [m, m]), m);
            write(&f.lower(), p[1], [s[2], s[3]], [m, m], E::store);
            write(&f.upper(), p[2], [s[4], s[5]], [m, m], E::store);
            write(&f.perm, p[3], [s[6], 0], [m, 1], |i| i as i64);
            true
        })
    }
}

/// `cholesky`: the lower triangular L with L Lᴴ = a, for a positive definite `m`×`m` matrix `a`, reading only its lower triangle.
unsafe extern "C" fn cholesky<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [2, 1, 4], |p, d, s| {
            let m = d[0];
            let l = linalg::cholesky(&read::<E>(p[0], [s[0], s[1]], [m, m]), m);
            let ok = l.is_some();
            write(&l.unwrap_or_else(|| nans(m * m)), p[1], [s[2], s[3]], [m, m], E::store);
            ok
        })
    }
}

/// The QR factorization of the `m`×`n` matrix at `p[0]`, written as Q to `p[1]` and R to `p[2]`: complete, with Q `m`×`m`, or thin,
/// with Q `m`×`k` where `k` is the smaller of `m` and `n`. `s` holds the three matrices' strides.
unsafe fn factored_qr<E: Item>(p: &[*mut c_char], [m, n]: [usize; 2], s: &[npy_intp], complete: bool) -> bool {
    let a = unsafe { read::<E>(p[0], [s[0], s[1]], [m, n]) };
    let ((q, r), k) = if complete { (linalg::qr_complete(&a, [m, n]), m) } else { (linalg::qr(&a, [m, n]), m.min(n)) };
    unsafe {
        write(&q, p[1], [s[2], s[3]], [m, k], E::store);
        write(&r, p[2], [s[4], s[5]], [k, n], E::store);
    }
    true
}

/// `qr_m` and `qr_n`: the thin QR factorization of an `m`×`n` matrix.
unsafe extern "C" fn qr<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe { stack(args, dims, steps, [3, 2, 6], |p, d, s| factored_qr::<E>(p, [d[0], d[1]], s, false)) }
}

/// `qr_complete`: the complete QR factorization of an `m`×`n` matrix.
unsafe extern "C" fn qr_complete<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe { stack(args, dims, steps, [3, 2, 6], |p, d, s| factored_qr::<E>(p, [d[0], d[1]], s, true)) }
}

/// `svd_m` and `svd_n`: the thin SVD of an `m`×`n` matrix, as U, which is `m`×`p`, the `p` singular values in descending order, and
/// V, which is `n`×`p`, where `p` is the smaller of `m` and `n`.
unsafe extern "C" fn svd<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [4, 2, 7], |p, d, s| {
            let ([m, n], k) = ([d[0], d[1]], d[0].min(d[1]));
            let f = linalg::svd(&read::<E>(p[0], [s[0], s[1]], [m, n]), [m, n]);
            write(&f.u, p[1], [s[2], s[3]], [m, k], E::store);
            write(&f.s, p[2], [s[4], 0], [k, 1], E::Real::store);
            write(&f.v, p[3], [s[5], s[6]], [n, k], E::store);
            true
        })
    }
}

/// `svdvals_m` and `svdvals_n`: the singular values of an `m`×`n` matrix, in descending order.
unsafe extern "C" fn svdvals<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [2, 2, 3], |p, d, s| {
            let sv = linalg::singular_values(&read::<E>(p[0], [s[0], s[1]], [d[0], d[1]]), [d[0], d[1]]);
            write(&sv, p[1], [s[2], 0], [sv.len(), 1], E::Real::store);
            true
        })
    }
}

/// `rank`: rustymath's `rank` of an `m`×`n` matrix.
unsafe extern "C" fn rank<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [2, 2, 2], |p, d, s| {
            let r = linalg::rank(&read::<E>(p[0], [s[0], s[1]], [d[0], d[1]]), [d[0], d[1]]);
            (p[1] as *mut i64).write_unaligned(r as i64);
            true
        })
    }
}

/// `lstsq_m` and `lstsq_n`: the least-squares solution `x` of `a x = b` with the least norm, for an `m`×`n` matrix `a` and an `m`×`k`
/// matrix `b`, and `a`'s singular values in descending order.
unsafe extern "C" fn lstsq<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [4, 3, 7], |p, d, s| {
            let [m, n, k] = [d[0], d[1], d[2]];
            let (a, b) = (read::<E>(p[0], [s[0], s[1]], [m, n]), read::<E>(p[1], [s[2], s[3]], [m, k]));
            let (x, sv) = linalg::lstsq(&a, &b, [m, n], k);
            write(&x, p[2], [s[4], s[5]], [n, k], E::store);
            write(&sv, p[3], [s[6], 0], [sv.len(), 1], E::Real::store);
            true
        })
    }
}

/// `eigh`: the eigenvalues of a symmetric or Hermitian `m`×`m` matrix in ascending order, and its unit eigenvectors in columns, reading
/// only its lower triangle.
unsafe extern "C" fn eigh<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [3, 1, 5], |p, d, s| {
            let m = d[0];
            let f = linalg::eigh(&read::<E>(p[0], [s[0], s[1]], [m, m]), m);
            let ok = f.is_some();
            let (w, v) = f.map_or_else(|| (nans(m), nans(m * m)), |f| (f.values, f.vectors));
            write(&w, p[1], [s[2], 0], [m, 1], E::Real::store);
            write(&v, p[2], [s[3], s[4]], [m, m], E::store);
            ok
        })
    }
}

/// `eigvalsh`: the eigenvalues of a symmetric or Hermitian `m`×`m` matrix in ascending order, reading only its lower triangle.
unsafe extern "C" fn eigvalsh<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [2, 1, 3], |p, d, s| {
            let m = d[0];
            let w = linalg::hermitian_eigenvalues(&read::<E>(p[0], [s[0], s[1]], [m, m]), m);
            let ok = w.is_some();
            write(&w.unwrap_or_else(|| nans(m)), p[1], [s[2], 0], [m, 1], E::Real::store);
            ok
        })
    }
}

/// `eig`: the eigenvalues of an `m`×`m` matrix, and its unit eigenvectors in columns, both complex.
unsafe extern "C" fn eig<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [3, 1, 5], |p, d, s| {
            let m = d[0];
            let f = linalg::eig(&read::<E>(p[0], [s[0], s[1]], [m, m]), m);
            let ok = f.is_some();
            let (w, v) = f.map_or_else(|| (complex_nans::<E>(m), complex_nans::<E>(m * m)), |f| (f.values, f.vectors));
            write(&w, p[1], [s[2], 0], [m, 1], |z| z);
            write(&v, p[2], [s[3], s[4]], [m, m], |z| z);
            ok
        })
    }
}

/// `eigvals`: the eigenvalues of an `m`×`m` matrix, as complex numbers.
unsafe extern "C" fn eigvals<E: Item>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [2, 1, 3], |p, d, s| {
            let m = d[0];
            let w = linalg::eigenvalues(&read::<E>(p[0], [s[0], s[1]], [m, m]), m);
            let ok = w.is_some();
            write(&w.unwrap_or_else(|| complex_nans::<E>(m)), p[1], [s[2], 0], [m, 1], |z| z);
            ok
        })
    }
}

/// A rustymath inner product, as the value of its unit type.
trait Pair { const PAIR: Self; }
impl Pair for PlusTimes { const PAIR: Self = PlusTimes; }
impl Pair for MaxPlus { const PAIR: Self = MaxPlus; }
impl Pair for MinPlus { const PAIR: Self = MinPlus; }
impl Pair for OrAnd { const PAIR: Self = OrAnd; }
impl Pair for PlusEqual { const PAIR: Self = PlusEqual; }
impl Pair for PlusNotEqual { const PAIR: Self = PlusNotEqual; }

/// `matmul` and the other matrix products: `P`'s inner product of an `n`×`k` matrix and a `k`×`m` matrix.
unsafe extern "C" fn product<P: Pair + Inner<E::W, Out = E::W>, E: Stored>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [3, 3, 6], |p, d, s| {
            let [n, k, m] = [d[0], d[1], d[2]];
            let c = P::PAIR.inner(&read::<E>(p[0], [s[0], s[1]], [n, k]), &read::<E>(p[1], [s[2], s[3]], [k, m]), [n, k, m]);
            write(&c, p[2], [s[4], s[5]], [n, m], E::store);
            true
        })
    }
}

/// `add_equal` and `add_not_equal`: `P`'s count over each row of an `n`×`k` matrix and column of a `k`×`m` matrix, as int64.
unsafe extern "C" fn count<P: Pair + Inner<E::W, Out: Copy + Into<i64>>, E: Stored>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void) {
    unsafe {
        stack(args, dims, steps, [3, 3, 6], |p, d, s| {
            let [n, k, m] = [d[0], d[1], d[2]];
            let c = P::PAIR.inner(&read::<E>(p[0], [s[0], s[1]], [n, k]), &read::<E>(p[1], [s[2], s[3]], [k, m]), [n, k, m]);
            write(&c, p[2], [s[4], s[5]], [n, m], Into::<i64>::into);
            true
        })
    }
}

/// The type of one argument of a loop: the loop's element type, the real or complex type of the same width, or int64.
#[derive(Clone, Copy)]
enum Kind { E, R, C, I }

/// The loop `f` at element type `E`, whose arguments have the types `kinds`.
fn entry<E: Item>(f: Ufunc, kinds: &[Kind]) -> Loop {
    (Some(f), kinds.iter().map(|k| (match k { Kind::E => E::CODE, Kind::R => E::Real::CODE, Kind::C => E::Complex::CODE, Kind::I => i64::CODE }) as c_char).collect())
}

/// `P`'s product loops at float16, float32 and float64.
fn float_products<P: Pair + Inner<f32, Out = f32> + Inner<f64, Out = f64>>() -> Vec<Loop> {
    let kinds = [Kind::E; 3];
    vec![entry::<f16>(product::<P, f16>, &kinds), entry::<f32>(product::<P, f32>, &kinds), entry::<f64>(product::<P, f64>, &kinds)]
}

/// `P`'s product loop at int64, whose arithmetic wraps.
fn integer_product<P: Pair + Inner<i64, Out = i64>>() -> Loop { (Some(product::<P, i64> as Ufunc), vec![i64::CODE as c_char; 3]) }

/// `P`'s count loops at int64, float16, float32 and float64, each giving int64 counts.
fn counts<P: Pair + Inner<i64, Out = i64> + Inner<f32, Out = i32> + Inner<f64, Out = i64>>() -> Vec<Loop> {
    let entry = |f: Ufunc, t: NPY_TYPES| (Some(f), vec![t as c_char, t as c_char, i64::CODE as c_char]);
    vec![entry(count::<P, i64>, i64::CODE), entry(count::<P, f16>, f16::CODE), entry(count::<P, f32>, f32::CODE), entry(count::<P, f64>, f64::CODE)]
}

/// Registers each generalized ufunc in `m`, with loops at float16, float32, float64, complex64 and complex128. `$nin` counts its
/// inputs, and `$kind` gives the type of each argument, inputs first.
macro_rules! gufuncs {
    ($m:expr; $($name:literal $signature:literal: $nin:literal => [$($kind:ident),+] = $f:ident, $doc:literal;)+) => {$({
        let kinds = [$(Kind::$kind),+];
        let loops = vec![
            entry::<f16>($f::<f16>, &kinds),
            entry::<f32>($f::<f32>, &kinds),
            entry::<f64>($f::<f64>, &kinds),
            entry::<Complex<f32>>($f::<Complex<f32>>, &kinds),
            entry::<Complex<f64>>($f::<Complex<f64>>, &kinds),
        ];
        register($m, $name, $doc, [$nin, kinds.len() - $nin], NONE, Some($signature), loops)?;
    })+};
}

/// The signature of a matrix product. As in `numpy.matmul`, a vector on the left is one row, and a vector on the right one column.
const PRODUCT: &str = "(n?,k),(k,m?)->(n?,m?)";

/// Adds the generalized ufuncs to the submodule `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    gufuncs! { m;
        "solve" "(m,n),(m,k)->(n,k)": 2 => [E, E, E] = solve, "x with a x = b for each matrix a and matrix b, through LU with no cutoff where a is square with no negligible pivot, and otherwise the least-squares solution with the least norm.";
        "pinv" "(m,n)->(n,m)": 1 => [E, E] = pinv, "The pseudoinverse of each matrix. A square matrix with no negligible LU pivot gives its inverse through LU, with no cutoff.";
        "det" "(m,m)->()": 1 => [E, E] = det, "The determinant of each square matrix.";
        "slogdet" "(m,m)->(),()": 1 => [E, E, R] = slogdet, "The sign of each square matrix's determinant, and the logarithm of its magnitude.";
        "lu" "(m,m)->(m,m),(m,m),(m)": 1 => [E, E, E, I] = lu, "L, U and the row positions of each square matrix's LU factorization.";
        "cholesky" "(m,m)->(m,m)": 1 => [E, E] = cholesky, "The lower triangular Cholesky factor of each positive definite matrix.";
        "qr_m" "(m,n)->(m,m),(m,n)": 1 => [E, E, E] = qr, "Q and R of each matrix with no more rows than columns.";
        "qr_n" "(m,n)->(m,n),(n,n)": 1 => [E, E, E] = qr, "Q and R of each matrix with no more columns than rows.";
        "qr_complete" "(m,n)->(m,m),(m,n)": 1 => [E, E, E] = qr_complete, "The complete Q and R of each matrix.";
        "svd_m" "(m,n)->(m,m),(m),(n,m)": 1 => [E, E, R, E] = svd, "The thin SVD of each matrix with no more rows than columns.";
        "svd_n" "(m,n)->(m,n),(n),(n,n)": 1 => [E, E, R, E] = svd, "The thin SVD of each matrix with no more columns than rows.";
        "svdvals_m" "(m,n)->(m)": 1 => [E, R] = svdvals, "The singular values of each matrix with no more rows than columns.";
        "svdvals_n" "(m,n)->(n)": 1 => [E, R] = svdvals, "The singular values of each matrix with no more columns than rows.";
        "rank" "(m,n)->()": 1 => [E, I] = rank, "The rank of each matrix.";
        "lstsq_m" "(m,n),(m,k)->(n,k),(m)": 2 => [E, E, E, R] = lstsq, "The least-norm least-squares solution for each matrix with no more rows than columns.";
        "lstsq_n" "(m,n),(m,k)->(n,k),(n)": 2 => [E, E, E, R] = lstsq, "The least-norm least-squares solution for each matrix with no more columns than rows.";
        "eigh" "(m,m)->(m),(m,m)": 1 => [E, R, E] = eigh, "The eigenvalues and eigenvectors of each symmetric or Hermitian matrix.";
        "eigvalsh" "(m,m)->(m)": 1 => [E, R] = eigvalsh, "The eigenvalues of each symmetric or Hermitian matrix.";
        "eig" "(m,m)->(m),(m,m)": 1 => [E, C, C] = eig, "The complex eigenvalues and eigenvectors of each square matrix.";
        "eigvals" "(m,m)->(m)": 1 => [E, C] = eigvals, "The complex eigenvalues of each square matrix.";
    }
    let or_and = (Some(product::<OrAnd, bool> as Ufunc), vec![NPY_TYPES::NPY_BOOL as c_char; 3]);
    let complex = [Kind::E; 3];
    let complex = vec![entry::<Complex<f32>>(product::<PlusTimes, Complex<f32>>, &complex), entry::<Complex<f64>>(product::<PlusTimes, Complex<f64>>, &complex)];
    register(m, "matmul", "The matrix product of each pair of matrices, or for Booleans, whether some k has both a[i,k] and b[k,j]. Integers wrap on overflow.", [2, 1], NONE, Some(PRODUCT), [vec![or_and, integer_product::<PlusTimes>()], float_products::<PlusTimes>(), complex].concat())?;
    register(m, "maximum_add", "The max-plus product of each pair of real matrices: the greatest a[i,k] + b[k,j].", [2, 1], NONE, Some(PRODUCT), [vec![integer_product::<MaxPlus>()], float_products::<MaxPlus>()].concat())?;
    register(m, "minimum_add", "The min-plus product of each pair of real matrices: the least a[i,k] + b[k,j].", [2, 1], NONE, Some(PRODUCT), [vec![integer_product::<MinPlus>()], float_products::<MinPlus>()].concat())?;
    register(m, "add_equal", "The number of k where a[i,k] equals b[k,j], for each pair of real matrices.", [2, 1], NONE, Some(PRODUCT), counts::<PlusEqual>())?;
    register(m, "add_not_equal", "The number of k where a[i,k] differs from b[k,j], for each pair of real matrices.", [2, 1], NONE, Some(PRODUCT), counts::<PlusNotEqual>())
}
