//! rustymath's `linalg` module as NumPy generalized ufuncs, which `python/rustymath/linalg.py` wraps. A loop runs once for each matrix of a stack. It borrows contiguous core matrices and copies strided or float16 matrices into worker-local row-major buffers, calls rustymath, then writes the results through the outputs' strides, rounded to the loop's type. complex64 and complex128 matrices compute at their own width. Results that are always real, such as singular values, have the real type of the loop's width. A matrix that fails gets NaN results and makes the loop report NumPy's "invalid value", as a failing item of any rustymath ufunc does.
//!
//! A core dimension can't be the smaller of two others. Each function whose results have `min(m, n)` rows or items therefore registers twice: once for `m ≤ n`, with a name ending `_m`, and once for `m ≥ n`, ending `_n`. One loop serves both.
use std::slice;
use std::{borrow::Cow, convert::Infallible, ffi::{c_char, c_void}, marker::PhantomData, mem::{align_of, size_of}};

use half::f16;
use num_complex::Complex;
use num_traits::{Float as _, NumAssignRef, Zero as _};
use numpy::npyffi::{NPY_TYPES, npy_intp};
use numpy::{PyArray1, PyArrayDescrMethods, PyArrayMethods, PyReadonlyArray2, PyUntypedArray, PyUntypedArrayMethods};
use pyo3::prelude::*;
use rustymath::Float;
use rustymath::linalg::{self, AndOr, Factor, Inner, LuRef, LuWork, MaxPlus, MinPlus, OrAnd, PlusEqual, PlusNotEqual, PlusTimes, Scalar, SvdRef};

use crate::{Code, Loop, NONE, fe, parallel::{overlaps, separate}};

/// A generalized ufunc loop.
type Ufunc = unsafe extern "C" fn(*mut *mut c_char, *mut npy_intp, *mut npy_intp, *mut c_void);

/// One argument of a loop: its NumPy type, its width in bytes, the positions of its core dimensions among the loop's, and the position of its first core stride among the loop's.
struct Arg { code: c_char, width: usize, shape: Vec<usize>, at: usize }

/// A loop's number of inputs, its number of core dimensions and its arguments, derived once from the registered dtype loop and signature.
struct MatrixLoop { nin: usize, ndims: usize, args: Vec<Arg> }

impl MatrixLoop {
    /// The number of core strides that NumPy passes: one for each core dimension of each argument.
    fn strides(&self) -> usize { self.args.iter().map(|a| a.shape.len()).sum() }

    /// Whether conservative byte bounds prove that no two of the `n` matrices write the same bytes: each output's matrices are separate, and so are different outputs. Inputs need no check, because NumPy copies any input that an output overlaps before it calls the loop.
    fn independent(&self, pointers: &[usize], dims: &[usize], outer: &[npy_intp], core: &[npy_intp], n: usize) -> bool {
        let spans: Option<Vec<_>> = self.args.iter().zip(pointers).zip(outer).skip(self.nin).map(|((arg, &p), &outer)| {
            let axes: Vec<_> = arg.shape.iter().zip(&core[arg.at..]).map(|(&d, &s)| (dims[d], s)).collect();
            separate(p, arg.width, &axes, n, outer)
        }).collect();
        spans.is_some_and(|s| (1..s.len()).all(|i| (0..i).all(|j| !overlaps(&s[i], &s[j]))))
    }
}

/// Registers the generalized ufunc `name`, with `counts` inputs and outputs, and gives each of its `loops` the `MatrixLoop` that the loop's types and `signature` describe.
fn register(m: &Bound<'_, PyModule>, name: &str, doc: &str, counts: [usize; 2], signature: &str, loops: Vec<Loop>) -> PyResult<()> {
    let mut dimensions = Vec::new();
    let shapes: Vec<Vec<usize>> = signature.split('(').skip(1).map(|s| {
        s.split(')').next().unwrap().split(',').filter(|d| !d.is_empty()).map(|d| {
            let d = d.trim_end_matches('?');
            if let Some(i) = dimensions.iter().position(|&v| v == d) { i } else { dimensions.push(d); dimensions.len() - 1 }
        }).collect()
    }).collect();
    fn code_width<T: Code>() -> (c_char, usize) { (T::CODE as c_char, size_of::<T>()) }
    let widths = [
        code_width::<f16>(), code_width::<f32>(), code_width::<f64>(), code_width::<Complex<f32>>(), code_width::<Complex<f64>>(), code_width::<bool>(),
        code_width::<i64>(),
    ];
    let data = loops.iter().map(|(_, codes)| {
        let mut at = 0;
        let args = codes.iter().zip(&shapes).map(|(&code, shape)| {
            let width = widths.iter().find(|&&(t, _)| t == code).unwrap().1;
            at += shape.len();
            Arg { code, width, shape: shape.clone(), at: at - shape.len() }
        }).collect();
        Box::into_raw(Box::new(MatrixLoop { nin: counts[0], ndims: dimensions.len(), args })).cast::<c_void>()
    }).collect();
    crate::register(m, name, doc, counts, NONE, Some(signature), loops, Some(data))
}

/// A loop's element type, and `W`, the type rustymath computes it in.
trait Stored: Code + numpy::Element {
    type W: Copy;
    fn load(self) -> Self::W;
    fn store(w: Self::W) -> Self;
    fn borrow(_: &[Self]) -> Option<&[Self::W]> { None }
}

impl Stored for f16 {
    type W = f32;
    fn load(self) -> f32 { self.working() }
    fn store(w: f32) -> f16 { f16::from_working(w) }
}

macro_rules! itself {
    ($($t:ty),+) => {$(impl Stored for $t {
        type W = $t;
        fn load(self) -> $t { self }
        fn store(w: $t) -> $t { w }
        fn borrow(x: &[Self]) -> Option<&[Self]> { Some(x) }
    })+};
}
itself!(f32, f64, Complex<f32>, Complex<f64>, bool, i64);

/// The real type that the item type `E` computes in, or computes its parts in: `f32` for `float16`, `float32` and `complex64`.
type Part<E> = <<E as Stored>::W as Scalar>::Real;

/// A float loop's element type: `float16`, `float32`, `float64`, `complex64` or `complex128`, which rustymath factors in its working type.
trait Item: Stored<W: Factor> {
    /// The element type of results that are always real, such as singular values: the real type of the same width.
    type Real: Stored<W = Part<Self>>;
    /// The element type of results that are always complex, such as a general matrix's eigenvalues: the complex type of the same width.
    type Complex: Stored<W = Complex<Part<Self>>>;
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

/// One matrix of a loop's stack: the core matrix of each argument at position `i`.
struct Cell<'a> { layout: &'a MatrixLoop, pointers: &'a [usize], outer: &'a [npy_intp], core: &'a [npy_intp], dims: &'a [usize], i: usize }

impl Cell<'_> {
    /// The loop's core dimensions, in the order that the signature first names them.
    fn dims<const N: usize>(&self) -> [usize; N] { self.dims.try_into().expect("a loop's number of core dimensions") }

    /// Argument `k`, which holds items of type `code`: its address, its row and item strides in bytes, and its rows and columns. A vector is a matrix of one column, and a scalar one of one item.
    fn matrix(&self, k: usize, code: NPY_TYPES) -> (*mut c_char, [npy_intp; 2], [usize; 2]) {
        let arg = &self.layout.args[k];
        assert_eq!(arg.code, code as c_char, "argument {k} holds the loop's type");
        let (p, s) = ((self.pointers[k] as *mut c_char).wrapping_offset(self.i as isize * self.outer[k]), &self.core[arg.at..]);
        match arg.shape[..] { [r, c] => (p, [s[0], s[1]], [self.dims[r], self.dims[c]]), [r] => (p, [s[0], 0], [self.dims[r], 1]), _ => (p, [0, 0], [1, 1]) }
    }

    /// Input `k`, row by row in `T`'s working type.
    fn read<'c, T: Stored + 'c>(&'c self, k: usize) -> Cow<'c, [T::W]> {
        assert!(k < self.layout.nin, "argument {k} is an input");
        let (p, s, size) = self.matrix(k, T::CODE);
        // SAFETY: NumPy passes each argument's matrices at these addresses and strides, with items of the registered types.
        unsafe { read::<T>(p, s, size) }
    }

    /// Writes `x`, which holds output `k` row by row, through `f`.
    fn write<W: Copy, O: Code>(&self, k: usize, x: &[W], f: impl Fn(W) -> O) {
        assert!(k >= self.layout.nin, "argument {k} is an output");
        let (p, s, size) = self.matrix(k, O::CODE);
        // SAFETY: as for `read`. No input overlaps this matrix, and no other matrix writes to it.
        unsafe { write(x, p, s, size, f) }
    }
}

/// A generalized ufunc's work on each matrix of a stack, at the loop's element type `E`.
trait Gufunc<E> {
    /// The scratch value that the matrices which run in turn on one thread share, such as an `LuWork`.
    type S: Default;
    /// Reads `c`'s inputs, writes its outputs, and gives whether the matrix succeeded.
    fn each(c: &Cell<'_>, s: &mut Self::S) -> bool;
}

/// The loop that runs `G` on each matrix of a stack. Where no two matrices write the same bytes, rustymath's `stack_with` splits the stack across threads. Otherwise the matrices run in turn on this thread.
unsafe extern "C" fn looped<G: Gufunc<E>, E>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, data: *mut c_void) {
    let incoming = unsafe { fe::fetestexcept(fe::REPORTED) };
    let layout = unsafe { &*data.cast::<MatrixLoop>() };
    let nargs = layout.args.len();
    let (n, outer, core) = unsafe { (*dims as usize, slice::from_raw_parts(steps, nargs), slice::from_raw_parts(steps.add(nargs), layout.strides())) };
    let dims: Vec<usize> = unsafe { slice::from_raw_parts(dims.add(1), layout.ndims) }.iter().map(|&d| d as usize).collect();
    let pointers: Vec<usize> = unsafe { slice::from_raw_parts(args, nargs) }.iter().map(|&p| p as usize).collect();
    let each = |i, _: &mut [()], s: &mut G::S| G::each(&Cell { layout, pointers: &pointers, outer, core, dims: &dims, i }, s);
    let ok = if n > 1 && layout.independent(&pointers, &dims, outer, core, n) {
        let Ok((_, ok)) = rustymath::parallel::stack_with(n, 0, &each, || Ok::<_, Infallible>(()));
        ok
    } else { let mut s = G::S::default(); (0..n).fold(true, |ok, i| each(i, &mut [], &mut s) & ok) };
    unsafe { fe::finish(incoming, if ok { 0 } else { fe::INVALID }); }
}

/// The `rows`×`cols` matrix at `p`, whose rows lie `rs` bytes apart and items `cs` bytes apart, row by row in `E`'s working type.
unsafe fn read<'a, E: Stored + 'a>(p: *mut c_char, [rs, cs]: [npy_intp; 2], [rows, cols]: [usize; 2]) -> Cow<'a, [E::W]> {
    let len = rows * cols;
    if len == 0 { return Cow::Borrowed(&[]); }
    if (cols == 1 || cs == size_of::<E>() as isize) && (rows == 1 || rs == (cols * size_of::<E>()) as isize) && (p as usize).is_multiple_of(align_of::<E>()) {
        if let Some(x) = E::borrow(unsafe { slice::from_raw_parts(p.cast::<E>(), len) }) { return Cow::Borrowed(x); }
    }
    let mut x = Vec::with_capacity(rows * cols);
    for i in 0..rows as isize { for j in 0..cols as isize { x.push(unsafe { (p.offset(i * rs + j * cs) as *const E).read_unaligned() }.load()) } }
    Cow::Owned(x)
}

/// Writes the `rows`×`cols` matrix `x`, held row by row, through `f` to `p`, whose rows lie `rs` bytes apart and items `cs` bytes apart. A vector is a matrix of one column, and a scalar one of one item.
unsafe fn write<W: Copy, O>(x: &[W], p: *mut c_char, [rs, cs]: [npy_intp; 2], [rows, cols]: [usize; 2], f: impl Fn(W) -> O) {
    for i in 0..rows {
        for j in 0..cols { unsafe { (p.offset(i as isize * rs + j as isize * cs) as *mut O).write_unaligned(f(x[i * cols + j])) } }
    }
}

/// `n` NaNs, the results of a matrix that fails.
fn nans<W: Scalar>(n: usize) -> Vec<W> { vec![W::from_real(W::Real::nan()); n] }

/// `n` complex NaNs, the complex results of a matrix of item type `E` that fails.
fn complex_nans<E: Item>(n: usize) -> Vec<Complex<Part<E>>> { vec![Complex::new(Part::<E>::nan(), Part::<E>::nan()); n] }

/// Runs `f` on the LU factorization in the first two inputs: the `n`×`n` packed factors, transposed as the `lu` gufunc writes them, and the positions of the `n` rows. `None` where a position is out of range.
fn with_lu<E: Item, R>(c: &Cell<'_>, n: usize, f: impl FnOnce(LuRef<'_, E::W>) -> R) -> Option<R> {
    let (factors, perm) = (c.read::<E>(0), c.read::<i64>(1));
    let perm: Vec<usize> = perm.iter().map(|&i| i as usize).collect();
    perm.iter().all(|&i| i < n).then(|| f(LuRef::new(n, &factors, &perm)))
}

/// Runs `f` on the thin SVD of an `m`×`n` matrix in the first three inputs: U, the singular values and V.
fn with_svd<E: Item, R>(c: &Cell<'_>, [m, n]: [usize; 2], f: impl FnOnce(SvdRef<'_, E::W>) -> R) -> R {
    f(SvdRef::new([m, n], &c.read::<E>(0), &c.read::<E::Real>(1), &c.read::<E>(2)))
}

/// Defines each listed generalized ufunc's work on one matrix as a unit type, for every float and complex element type `E`. `Name(c, s: S) { body }` runs `body` with the matrix's `Cell` `c` and the shared scratch value `s` of type `S`.
macro_rules! gufunc_types {
    ($($(#[$doc:meta])* $name:ident($c:ident, $s:tt: $S:ty) $body:block)+) => {$(
        $(#[$doc])*
        struct $name;
        impl<E: Item> Gufunc<E> for $name {
            type S = $S;
            fn each($c: &Cell<'_>, $s: &mut $S) -> bool $body
        }
    )+};
}

gufunc_types! {
    /// `solve`: rustymath's `solve` of `a x = b`, for an `m`×`n` matrix `a` and an `m`×`k` matrix `b`.
    Solve(c, work: LuWork<E::W>) {
        let [m, n, k] = c.dims();
        c.write(2, &work.solve(&c.read::<E>(0), &c.read::<E>(1), [m, n], k), E::store);
        true
    }
    /// `pinv`: rustymath's `pinv` of an `m`×`n` matrix, which is `n`×`m`.
    Pinv(c, scratch: (LuWork<E::W>, Vec<E::W>)) {
        let ([m, n], (work, x)) = (c.dims(), scratch);
        x.resize(m * n, E::W::zero());
        work.pinv_into(&c.read::<E>(0), [m, n], x);
        c.write(1, x, E::store);
        true
    }
    /// `det`: the determinant of an `m`×`m` matrix, from its LU factorization.
    Det(c, work: LuWork<E::W>) {
        let [m] = c.dims();
        c.write(1, &[work.factor(&c.read::<E>(0), m).determinant()], E::store);
        true
    }
    /// `slogdet`: the sign of the determinant of an `m`×`m` matrix and the natural logarithm of its magnitude, as `Lu::log_determinant` gives them.
    Slogdet(c, work: LuWork<E::W>) {
        let [m] = c.dims();
        let (sign, log) = work.factor(&c.read::<E>(0), m).log_determinant();
        c.write(1, &[sign], E::store);
        c.write(2, &[log], E::Real::store);
        true
    }
    /// `lu`: the packed LU factors of an `m`×`m` matrix and the positions of its rows. NumPy holds the factors transposed: their rows are rustymath's columns.
    Lu(c, work: LuWork<E::W>) {
        let [m] = c.dims();
        let f = work.factor(&c.read::<E>(0), m);
        c.write(1, f.factors, E::store);
        c.write(2, f.perm, |i| i as i64);
        true
    }
    /// `lu_solve`: `x` with `a x = b`, from the packed LU factors and row positions of an `m`×`m` matrix `a`.
    LuSolve(c, _: ()) {
        let [n, k] = c.dims();
        let x = with_lu::<E, _>(c, n, |lu| lu.solve(&c.read::<E>(2), k));
        let ok = x.is_some();
        c.write(3, &x.unwrap_or_else(|| nans(n * k)), E::store);
        ok
    }
    /// `lu_inverse`: the inverse of an `m`×`m` matrix, from its packed LU factors and row positions.
    LuInverse(c, _: ()) {
        let [n] = c.dims();
        let x = with_lu::<E, _>(c, n, |lu| lu.inverse());
        let ok = x.is_some();
        c.write(2, &x.unwrap_or_else(|| nans(n * n)), E::store);
        ok
    }
    /// `lu_slogdet`: the sign of an `m`×`m` matrix's determinant and the natural logarithm of its magnitude, from its packed LU factors and row positions.
    LuSlogdet(c, _: ()) {
        let [n] = c.dims();
        let r = with_lu::<E, _>(c, n, |lu| lu.log_determinant());
        let (sign, log) = r.unwrap_or((E::W::from_real(Part::<E>::nan()), Part::<E>::nan()));
        c.write(2, &[sign], E::store);
        c.write(3, &[log], E::Real::store);
        r.is_some()
    }
    /// `lu_det`: the determinant of an `m`×`m` matrix, from its packed LU factors and row positions.
    LuDet(c, _: ()) {
        let [n] = c.dims();
        let det = with_lu::<E, _>(c, n, |lu| lu.determinant());
        c.write(2, &[det.unwrap_or(E::W::from_real(Part::<E>::nan()))], E::store);
        det.is_some()
    }
    /// `cholesky`: the lower triangular L with L Lᴴ = a, for a positive definite `m`×`m` matrix `a`, reading only its lower triangle.
    Cholesky(c, _: ()) {
        let [m] = c.dims();
        let l = linalg::cholesky(&c.read::<E>(0), m);
        let ok = l.is_some();
        c.write(1, &l.unwrap_or_else(|| nans(m * m)), E::store);
        ok
    }
    /// `qr_m` and `qr_n`: the thin QR factorization of an `m`×`n` matrix.
    Qr(c, _: ()) {
        let [m, n] = c.dims();
        let (q, r) = linalg::qr(&c.read::<E>(0), [m, n]);
        c.write(1, &q, E::store);
        c.write(2, &r, E::store);
        true
    }
    /// `qr_complete`: the complete QR factorization of an `m`×`n` matrix.
    QrComplete(c, _: ()) {
        let [m, n] = c.dims();
        let (q, r) = linalg::qr_complete(&c.read::<E>(0), [m, n]);
        c.write(1, &q, E::store);
        c.write(2, &r, E::store);
        true
    }
    /// `svd_m` and `svd_n`: the thin SVD of an `m`×`n` matrix, as U, which is `m`×`p`, the `p` singular values in descending order, and V, which is `n`×`p`, where `p` is the smaller of `m` and `n`.
    Svd(c, _: ()) {
        let [m, n] = c.dims();
        let f = linalg::svd(&c.read::<E>(0), [m, n]);
        c.write(1, &f.u, E::store);
        c.write(2, &f.s, E::Real::store);
        c.write(3, &f.v, E::store);
        true
    }
    /// `truncated_svd`: an approximation to the first `k` singular values of an `m`×`n` matrix and their vectors, as U, which is `m`×`k`, the values in descending order, and V, which is `n`×`k`. The outputs set `k`. A `k` above `m` or `n` gives NaN results.
    TruncatedSvd(c, _: ()) {
        let [m, n, k] = c.dims();
        let ok = k <= m.min(n);
        let (u, values, v) = if ok {
            let f = linalg::truncated_svd(&c.read::<E>(0), [m, n], k);
            (f.u, f.s, f.v)
        } else { (nans(m * k), nans(k), nans(n * k)) };
        c.write(1, &u, E::store);
        c.write(2, &values, E::Real::store);
        c.write(3, &v, E::store);
        ok
    }
    /// `svd_solve`: the least-norm `x` with `a x = b`, from the thin SVD factors of an `m`×`n` matrix `a`.
    SvdSolve(c, _: ()) {
        let [m, _, n, k] = c.dims();
        c.write(4, &with_svd::<E, _>(c, [m, n], |svd| svd.solve(&c.read::<E>(3), k)), E::store);
        true
    }
    /// `svd_pinv`: the pseudoinverse of an `m`×`n` matrix, from its thin SVD factors.
    SvdPinv(c, _: ()) {
        let [m, _, n] = c.dims();
        c.write(3, &with_svd::<E, _>(c, [m, n], |svd| svd.pseudoinverse()), E::store);
        true
    }
    /// `svdvals_m` and `svdvals_n`: the singular values of an `m`×`n` matrix, in descending order.
    Svdvals(c, _: ()) {
        let [m, n] = c.dims();
        c.write(1, &linalg::singular_values(&c.read::<E>(0), [m, n]), E::Real::store);
        true
    }
    /// `rank`: rustymath's `rank` of an `m`×`n` matrix.
    Rank(c, _: ()) {
        let [m, n] = c.dims();
        c.write(1, &[linalg::rank(&c.read::<E>(0), [m, n]) as i64], |r| r);
        true
    }
    /// `lstsq_m` and `lstsq_n`: the least-squares solution `x` of `a x = b` with the least norm, for an `m`×`n` matrix `a` and an `m`×`k` matrix `b`, and `a`'s singular values in descending order.
    Lstsq(c, _: ()) {
        let [m, n, k] = c.dims();
        let (x, sv) = linalg::lstsq(&c.read::<E>(0), &c.read::<E>(1), [m, n], k);
        c.write(2, &x, E::store);
        c.write(3, &sv, E::Real::store);
        true
    }
    /// `eigh`: the eigenvalues of a symmetric or Hermitian `m`×`m` matrix in ascending order, and its unit eigenvectors in columns, reading only its lower triangle.
    Eigh(c, _: ()) {
        let [m] = c.dims();
        let f = linalg::eigh(&c.read::<E>(0), m);
        let ok = f.is_some();
        let (w, v) = f.map_or_else(|| (nans(m), nans(m * m)), |f| (f.values, f.vectors));
        c.write(1, &w, E::Real::store);
        c.write(2, &v, E::store);
        ok
    }
    /// `eigvalsh`: the eigenvalues of a symmetric or Hermitian `m`×`m` matrix in ascending order, reading only its lower triangle.
    Eigvalsh(c, _: ()) {
        let [m] = c.dims();
        let w = linalg::hermitian_eigenvalues(&c.read::<E>(0), m);
        let ok = w.is_some();
        c.write(1, &w.unwrap_or_else(|| nans(m)), E::Real::store);
        ok
    }
    /// `eig`: the eigenvalues of an `m`×`m` matrix, and its unit eigenvectors in columns, both complex.
    Eig(c, _: ()) {
        let [m] = c.dims();
        let f = linalg::eig(&c.read::<E>(0), m);
        let ok = f.is_some();
        let (w, v) = f.map_or_else(|| (complex_nans::<E>(m), complex_nans::<E>(m * m)), |f| (f.values, f.vectors));
        c.write(1, &w, E::Complex::store);
        c.write(2, &v, E::Complex::store);
        ok
    }
    /// `eigvals`: the eigenvalues of an `m`×`m` matrix, as complex numbers.
    Eigvals(c, _: ()) {
        let [m] = c.dims();
        let w = linalg::eigenvalues(&c.read::<E>(0), m);
        let ok = w.is_some();
        c.write(1, &w.unwrap_or_else(|| complex_nans::<E>(m)), E::Complex::store);
        ok
    }
}

/// `permanent`: Ryser's permanent for floating and complex matrices.
struct Permanent;
impl<E: Item> Gufunc<E> for Permanent where E::W: NumAssignRef {
    type S = ();
    fn each(c: &Cell<'_>, _: &mut ()) -> bool {
        let [n] = c.dims();
        let value = if n < 64 { linalg::exact::permanent(&c.read::<E>(0), n) } else { E::W::from_real(Part::<E>::nan()) };
        c.write(1, &[value], E::store);
        n < 64
    }
}

fn row_reduce<'py, E: Item>(py: Python<'py>, a: PyReadonlyArray2<'py, E>, columns: Option<usize>) -> PyResult<Bound<'py, PyAny>> where E::W: NumAssignRef {
    let shape = a.shape();
    let f = linalg::exact::echelon_with(a.as_array().iter().map(|&x| x.load()).collect(), [shape[0], shape[1]], columns.unwrap_or(shape[1]), || {
        py.check_signals()
    })?;
    let rows = PyArray1::from_vec(py, f.rows.into_iter().map(E::store).collect()).reshape([f.pivots.len(), shape[1]])?;
    let pivots = PyArray1::from_vec(py, f.pivots.into_iter().map(|i| i as isize).collect());
    let determinant = PyArray1::from_vec(py, vec![E::store(f.determinant)]).get_item(0)?;
    Ok((rows, pivots, determinant).into_pyobject(py)?.into_any())
}

/// Row reduction of one matrix. Pivoting uses the first nonzero item, as in the exact-field algorithm.
#[pyfunction]
#[pyo3(signature = (a, columns=None))]
fn echelon<'py>(py: Python<'py>, a: &Bound<'py, PyUntypedArray>, columns: Option<usize>) -> PyResult<Bound<'py, PyAny>> {
    match a.dtype().char() {
        b'e' => row_reduce(py, a.extract::<PyReadonlyArray2<f16>>()?, columns), b'f' => row_reduce(py, a.extract::<PyReadonlyArray2<f32>>()?, columns),
        b'd' => row_reduce(py, a.extract::<PyReadonlyArray2<f64>>()?, columns), b'F' => row_reduce(py, a.extract::<PyReadonlyArray2<Complex<f32>>>()?, columns),
        b'D' => row_reduce(py, a.extract::<PyReadonlyArray2<Complex<f64>>>()?, columns),
        _ => Err(pyo3::exceptions::PyTypeError::new_err("echelon takes real or complex matrices"))
    }
}

/// `matmul` and the other matrix products: `P`'s inner product of an `n`×`k` matrix and a `k`×`m` matrix.
struct Product<P>(PhantomData<P>);
impl<P: Default + Inner<E::W, Out = E::W>, E: Stored> Gufunc<E> for Product<P> {
    type S = ();
    fn each(c: &Cell<'_>, _: &mut ()) -> bool {
        let [n, k, m] = c.dims();
        c.write(2, &P::default().inner(&c.read::<E>(0), &c.read::<E>(1), [n, k, m]), E::store);
        true
    }
}

/// `add_equal` and `add_not_equal`: `P`'s count over each row of an `n`×`k` matrix and column of a `k`×`m` matrix, as int64.
struct Count<P>(PhantomData<P>);
impl<P: Default + Inner<E::W, Out: Copy + Into<i64>>, E: Stored> Gufunc<E> for Count<P> {
    type S = ();
    fn each(c: &Cell<'_>, _: &mut ()) -> bool {
        let [n, k, m] = c.dims();
        c.write(2, &P::default().inner(&c.read::<E>(0), &c.read::<E>(1), [n, k, m]), Into::<i64>::into);
        true
    }
}

/// The type of one argument of a loop: the loop's element type, the real or complex type of the same width, or int64.
#[derive(Clone, Copy)]
enum Kind { E, R, C, I }

/// The loop `f` at element type `E`, whose arguments have the types `kinds`.
fn entry<E: Item>(f: Ufunc, kinds: &[Kind]) -> Loop {
    (Some(f), kinds.iter().map(|k| {
        (match k { Kind::E => E::CODE, Kind::R => E::Real::CODE, Kind::C => E::Complex::CODE, Kind::I => i64::CODE }) as c_char
    }).collect())
}

/// `P`'s product loops at float16, float32 and float64.
fn float_products<P: Default + Inner<f32, Out = f32> + Inner<f64, Out = f64>>() -> Vec<Loop> {
    let kinds = [Kind::E; 3];
    vec![entry::<f16>(looped::<Product<P>, f16>, &kinds), entry::<f32>(looped::<Product<P>, f32>, &kinds), entry::<f64>(looped::<Product<P>, f64>, &kinds)]
}

/// `P`'s product loop at int64, whose arithmetic wraps.
fn integer_product<P: Default + Inner<i64, Out = i64>>() -> Loop { (Some(looped::<Product<P>, i64> as Ufunc), vec![i64::CODE as c_char; 3]) }

/// `P`'s count loops at int64, float16, float32 and float64, each giving int64 counts.
fn counts<P: Default + Inner<i64, Out = i64> + Inner<f32, Out = i32> + Inner<f64, Out = i64>>() -> Vec<Loop> {
    let entry = |f: Ufunc, t: NPY_TYPES| (Some(f), vec![t as c_char, t as c_char, i64::CODE as c_char]);
    vec![
        entry(looped::<Count<P>, i64>, i64::CODE), entry(looped::<Count<P>, f16>, f16::CODE), entry(looped::<Count<P>, f32>, f32::CODE),
        entry(looped::<Count<P>, f64>, f64::CODE),
    ]
}

/// Registers each generalized ufunc in `m`, with loops at float16, float32, float64, complex64 and complex128. `$nin` counts its inputs, and `$kind` gives the type of each argument, inputs first.
macro_rules! gufuncs {
    ($m:expr; $($name:literal $signature:literal: $nin:literal => [$($kind:ident),+] = $f:ident, $doc:literal;)+) => {$({
        let kinds = [$(Kind::$kind),+];
        let loops = vec![
            entry::<f16>(looped::<$f, f16>, &kinds), entry::<f32>(looped::<$f, f32>, &kinds), entry::<f64>(looped::<$f, f64>, &kinds),
            entry::<Complex<f32>>(looped::<$f, Complex<f32>>, &kinds), entry::<Complex<f64>>(looped::<$f, Complex<f64>>, &kinds),
        ];
        register($m, $name, $doc, [$nin, kinds.len() - $nin], $signature, loops)?;
    })+};
}

/// The signature of a matrix product. As in `numpy.matmul`, a vector on the left is one row, and a vector on the right one column.
const PRODUCT: &str = "(n?,k),(k,m?)->(n?,m?)";

/// Adds the generalized ufuncs to the submodule `m`.
pub fn add(m: &Bound<'_, PyModule>) -> PyResult<()> {
    gufuncs! { m;
        "solve" "(m,n),(m,k)->(n,k)": 2 => [E, E, E] = Solve, "x with a x = b for each matrix a and matrix b, through LU with no cutoff where a is square with no negligible pivot, and otherwise the least-squares solution with the least norm.";
        "pinv" "(m,n)->(n,m)": 1 => [E, E] = Pinv, "The pseudoinverse of each matrix. A square matrix with no negligible LU pivot gives its inverse through LU, with no cutoff.";
        "det" "(m,m)->()": 1 => [E, E] = Det, "The determinant of each square matrix.";
        "slogdet" "(m,m)->(),()": 1 => [E, E, R] = Slogdet, "The sign of each square matrix's determinant, and the logarithm of its magnitude.";
        "lu" "(m,m)->(m,m),(m)": 1 => [E, E, I] = Lu, "Packed LU factors and the row positions of each square matrix. The factors are transposed, because rustymath stores them column by column.";
        "lu_solve" "(m,m),(m),(m,k)->(m,k)": 3 => [E, I, E, E] = LuSolve, "Solve from transposed packed LU factors, as `lu` gives them, without refactoring.";
        "lu_inverse" "(m,m),(m)->(m,m)": 2 => [E, I, E] = LuInverse, "Inverse from transposed packed LU factors, as `lu` gives them, without refactoring.";
        "lu_det" "(m,m),(m)->()": 2 => [E, I, E] = LuDet, "Determinant from transposed packed LU factors, as `lu` gives them.";
        "lu_slogdet" "(m,m),(m)->(),()": 2 => [E, I, E, R] = LuSlogdet, "Sign and log magnitude of a determinant from transposed packed LU factors, as `lu` gives them.";
        "cholesky" "(m,m)->(m,m)": 1 => [E, E] = Cholesky, "The lower triangular Cholesky factor of each positive definite matrix.";
        "qr_m" "(m,n)->(m,m),(m,n)": 1 => [E, E, E] = Qr, "Q and R of each matrix with no more rows than columns.";
        "qr_n" "(m,n)->(m,n),(n,n)": 1 => [E, E, E] = Qr, "Q and R of each matrix with no more columns than rows.";
        "qr_complete" "(m,n)->(m,m),(m,n)": 1 => [E, E, E] = QrComplete, "The complete Q and R of each matrix.";
        "svd_m" "(m,n)->(m,m),(m),(n,m)": 1 => [E, E, R, E] = Svd, "The thin SVD of each matrix with no more rows than columns.";
        "svd_n" "(m,n)->(m,n),(n),(n,n)": 1 => [E, E, R, E] = Svd, "The thin SVD of each matrix with no more columns than rows.";
        "truncated_svd" "(m,n)->(m,k),(k),(n,k)": 1 => [E, E, R, E] = TruncatedSvd, "An approximation to the first k singular values and vectors of each matrix, by a randomized range finder. The outputs set k, which can't exceed either side.";
        "svd_solve" "(m,p),(p),(n,p),(m,k)->(n,k)": 4 => [E, R, E, E, E] = SvdSolve, "Least-norm solve from thin SVD factors, without refactoring.";
        "svd_pinv" "(m,p),(p),(n,p)->(n,m)": 3 => [E, R, E, E] = SvdPinv, "Pseudoinverse from thin SVD factors, without refactoring.";
        "svdvals_m" "(m,n)->(m)": 1 => [E, R] = Svdvals, "The singular values of each matrix with no more rows than columns.";
        "svdvals_n" "(m,n)->(n)": 1 => [E, R] = Svdvals, "The singular values of each matrix with no more columns than rows.";
        "rank" "(m,n)->()": 1 => [E, I] = Rank, "The rank of each matrix.";
        "lstsq_m" "(m,n),(m,k)->(n,k),(m)": 2 => [E, E, E, R] = Lstsq, "The least-norm least-squares solution for each matrix with no more rows than columns.";
        "lstsq_n" "(m,n),(m,k)->(n,k),(n)": 2 => [E, E, E, R] = Lstsq, "The least-norm least-squares solution for each matrix with no more columns than rows.";
        "eigh" "(m,m)->(m),(m,m)": 1 => [E, R, E] = Eigh, "The eigenvalues and eigenvectors of each symmetric or Hermitian matrix.";
        "eigvalsh" "(m,m)->(m)": 1 => [E, R] = Eigvalsh, "The eigenvalues of each symmetric or Hermitian matrix.";
        "eig" "(m,m)->(m),(m,m)": 1 => [E, C, C] = Eig, "The complex eigenvalues and eigenvectors of each square matrix.";
        "eigvals" "(m,m)->(m)": 1 => [E, C] = Eigvals, "The complex eigenvalues of each square matrix.";
        "permanent" "(m,m)->()": 1 => [E, E] = Permanent, "The permanent of each square matrix, by Ryser's exponential-time formula.";
    }
    m.add_function(wrap_pyfunction!(echelon, m)?)?;
    let or_and = (Some(looped::<Product<OrAnd>, bool> as Ufunc), vec![NPY_TYPES::NPY_BOOL as c_char; 3]);
    let and_or = (Some(looped::<Product<AndOr>, bool> as Ufunc), vec![NPY_TYPES::NPY_BOOL as c_char; 3]);
    register(m, "logical_and_or", "Whether every k has either a[i,k] or b[k,j].", [2, 1], PRODUCT, vec![and_or])?;
    let complex = [Kind::E; 3];
    let complex = vec![
        entry::<Complex<f32>>(looped::<Product<PlusTimes>, Complex<f32>>, &complex),
        entry::<Complex<f64>>(looped::<Product<PlusTimes>, Complex<f64>>, &complex),
    ];
    register(
        m, "matmul", "The matrix product of each pair of matrices, or for Booleans, whether some k has both a[i,k] and b[k,j]. Integers wrap on overflow.",
        [2, 1], PRODUCT, [vec![or_and, integer_product::<PlusTimes>()], float_products::<PlusTimes>(), complex].concat())?;
    register(
        m, "maximum_add", "The max-plus product of each pair of real matrices: the greatest a[i,k] + b[k,j].", [2, 1], PRODUCT,
        [vec![integer_product::<MaxPlus>()], float_products::<MaxPlus>()].concat())?;
    register(
        m, "minimum_add", "The min-plus product of each pair of real matrices: the least a[i,k] + b[k,j].", [2, 1], PRODUCT,
        [vec![integer_product::<MinPlus>()], float_products::<MinPlus>()].concat())?;
    register(m, "add_equal", "The number of k where a[i,k] equals b[k,j], for each pair of real matrices.", [2, 1], PRODUCT, counts::<PlusEqual>())?;
    register(m, "add_not_equal", "The number of k where a[i,k] differs from b[k,j], for each pair of real matrices.", [2, 1], PRODUCT, counts::<PlusNotEqual>())
}
