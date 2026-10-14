//! rustymath's functions as NumPy ufuncs. Independent ranges split across threads, each running a function object's slice form on
//! chunks of `CHUNK` items. An
//! input with stride 0 is one item beside the others, and a strided input is copied into a buffer first. Through `numpy.errstate`,
//! NumPy reports "invalid value" for a NaN result from arguments that aren't NaN, and "overflow" for an integer result that doesn't
//! fit. Each loop clears the other floating-point status flags, which rustymath's branch-free code sets in lanes it discards.
use std::ffi::{c_char, c_int, c_void, CStr, CString};
use std::{array, ops::Range, ptr, slice};

use half::f16;
use numpy::npyffi::{npy_intp, PyUFuncGenericFunction, NPY_TYPES, PY_UFUNC_API};
use numpy::{Complex32, Complex64, PyArray1, PyArrayMethods, PyReadonlyArrayDyn, PyUntypedArrayMethods};
use pyo3::prelude::*;
use pyo3::types::PyCapsule;
use rustymath::tolerant::{Ceil, Equal, Floor, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, NotEqual, Residue, Tolerance, Tolerant};
use rustymath::{reduce::{self, Reduction}, Arg, Binary, Flipped, Ternary, Unary, CHUNK};

/// Runs `$body` with `$t` as the element type of the float array `$a`.
macro_rules! floats {
    ($a:expr, $t:ident => $body:expr) => {
        match $a.dtype().char() {
            b'e' => { type $t = f16; $body }
            b'f' => { type $t = f32; $body }
            _ => { type $t = f64; $body }
        }
    };
}

mod distribution;
mod exact;
mod integer;
mod linalg;
mod number_theory;
mod parallel;
mod polynomial;
mod search;

/// The tolerance that `ct` gives, or `Tolerance::EXACT` where it's out of range. The Python wrappers check the range first.
fn tolerance(ct: f64) -> Tolerance { Tolerance::new(ct).unwrap_or(Tolerance::EXACT) }

/// Reports NumPy's "invalid value" for the function `name`, as NumPy does after a ufunc loop with a failed item. `numpy.errstate` then
/// decides whether it warns, raises or does nothing. It calls `PyUFunc_GiveFloatingpointErrors`, slot 46 of NumPy 2's ufunc API. The
/// `numpy` crate doesn't bind that slot.
fn report_invalid(py: Python<'_>, name: &CStr) -> PyResult<()> {
    const NPY_FPE_INVALID: c_int = 8;
    let api = py.import("numpy._core.umath")?.getattr("_UFUNC_API")?.cast_into::<PyCapsule>()?.pointer_checked(None)?;
    // SAFETY: slot 46 holds `int PyUFunc_GiveFloatingpointErrors(const char *, int)`.
    let give: unsafe extern "C" fn(*const c_char, c_int) -> c_int = unsafe { std::mem::transmute(*api.cast::<*const c_void>().as_ptr().add(46)) };
    if unsafe { give(name.as_ptr(), NPY_FPE_INVALID) } < 0 { Err(PyErr::fetch(py)) } else { Ok(()) }
}

/// The C library's floating-point status flags, as each platform's `fenv.h` numbers them.
mod fe {
    use std::ffi::c_int;
    #[cfg(windows)]
    pub const INVALID: c_int = 0x10;
    #[cfg(windows)]
    pub const OVERFLOW: c_int = 0x04;
    /// The four flags that NumPy reports: invalid value, divide by zero, overflow and underflow.
    #[cfg(windows)]
    pub const REPORTED: c_int = 0x1e;
    #[cfg(all(not(windows), target_arch = "aarch64"))]
    pub const INVALID: c_int = 0x01;
    #[cfg(all(not(windows), target_arch = "aarch64"))]
    pub const OVERFLOW: c_int = 0x04;
    #[cfg(all(not(windows), target_arch = "aarch64"))]
    pub const REPORTED: c_int = 0x0f;
    #[cfg(all(not(windows), not(target_arch = "aarch64")))]
    pub const INVALID: c_int = 0x01;
    #[cfg(all(not(windows), not(target_arch = "aarch64")))]
    pub const OVERFLOW: c_int = 0x08;
    #[cfg(all(not(windows), not(target_arch = "aarch64")))]
    pub const REPORTED: c_int = 0x1d;
    unsafe extern "C" {
        pub fn feclearexcept(excepts: c_int) -> c_int;
        pub fn feraiseexcept(excepts: c_int) -> c_int;
        pub fn fetestexcept(excepts: c_int) -> c_int;
    }

    /// Discards incidental arithmetic flags, retaining failures from earlier callbacks and this one.
    pub unsafe fn finish(previous: c_int, failed: c_int) {
        unsafe {
            feclearexcept(REPORTED);
            if previous | failed != 0 { feraiseexcept(previous | failed); }
        }
    }
}

/// An element type of a ufunc loop.
trait Code: Copy + Default + Send + Sync {
    /// Its NumPy type number.
    const CODE: NPY_TYPES;
    /// The status flag for a loop with a failed item. NumPy then reports "invalid value", as it does for a NaN from arguments that
    /// aren't NaN.
    const FLAG: c_int = fe::INVALID;
}

macro_rules! codes {
    (overflow: $($t:ty => $code:ident),+) => {$(impl Code for $t { const CODE: NPY_TYPES = NPY_TYPES::$code; const FLAG: c_int = fe::OVERFLOW; })+};
    ($($t:ty => $code:ident),+) => {$(impl Code for $t { const CODE: NPY_TYPES = NPY_TYPES::$code; })+};
}
codes!(f16 => NPY_HALF, f32 => NPY_FLOAT, f64 => NPY_DOUBLE, Complex32 => NPY_CFLOAT, Complex64 => NPY_CDOUBLE, bool => NPY_BOOL);
// An integer item fails where its result doesn't fit, and NumPy reports "overflow".
codes!(overflow: i8 => NPY_BYTE, u8 => NPY_UBYTE, i16 => NPY_SHORT, u16 => NPY_USHORT, i32 => NPY_INT, u32 => NPY_UINT);
#[cfg(not(windows))]
codes!(overflow: i64 => NPY_LONG, u64 => NPY_ULONG);
#[cfg(windows)]
codes!(overflow: i64 => NPY_LONGLONG, u64 => NPY_ULONGLONG);

/// A ufunc's function at element type `E`, with `N` inputs.
trait Kernel<E, const N: usize>: Sync {
    /// The type of each result.
    type Out: Code;
    /// The running total that `reduce` and `accumulate` carry from chunk to chunk.
    type Total: Copy;
    /// Whether `reduce` and `accumulate` run `total`, `reduce`, `scan` and `result`. Otherwise they run `many` item by item.
    const FOLDS: bool = false;
    /// Appends a result for each of `n` items to `out`, and gives whether every item succeeded.
    fn many(&self, x: [Arg<'_, E>; N], n: usize, out: &mut Vec<Self::Out>) -> bool;
    /// The running total that starts from the result `acc`.
    fn total(&self, _acc: Self::Out) -> Self::Total { unreachable!("only a kernel that folds has total") }
    /// `t` combined with a whole strided lane, in bounded pieces that may run across threads.
    unsafe fn reduce(&self, _t: Self::Total, _n: usize, _p: *mut c_char, _step: npy_intp) -> Self::Total { unreachable!("only a kernel that folds has reduce") }
    /// Appends the running results from `t` over `x` to `out`, and gives the last running total.
    fn scan(&self, _t: Self::Total, _x: &[E], _out: &mut Vec<Self::Out>) -> Self::Total { unreachable!("only a kernel that folds has scan") }
    /// The result that the running total `t` gives.
    fn result(&self, _t: Self::Total) -> Self::Out { unreachable!("only a kernel that folds has result") }
}

/// A rustymath function object as a kernel.
#[derive(Clone, Copy)]
struct Obj<K>(K);

impl<E: Copy, K: Unary<E, Out: Code> + Sync> Kernel<E, 1> for Obj<K> {
    type Out = K::Out;
    type Total = K::Out;
    fn many(&self, [x]: [Arg<'_, E>; 1], n: usize, out: &mut Vec<K::Out>) -> bool { self.0.many(x, n, out) }
}

impl<E: Copy, K: Binary<E, Out: Code> + Sync> Kernel<E, 2> for Obj<K> {
    type Out = K::Out;
    type Total = K::Out;
    fn many(&self, [x, y]: [Arg<'_, E>; 2], n: usize, out: &mut Vec<K::Out>) -> bool { self.0.many(x, y, n, out) }
}

impl<E: Copy, K: Ternary<E, Out: Code> + Sync> Kernel<E, 3> for Obj<K> {
    type Out = K::Out;
    type Total = K::Out;
    fn many(&self, [x, y, z]: [Arg<'_, E>; 3], n: usize, out: &mut Vec<K::Out>) -> bool { self.0.many(x, y, z, n, out) }
}

/// NumPy's `add` and `multiply`, which run rustymath's `Add` and `Multiply`. On Booleans they run `Maximum` and `Minimum`, the `or` and
/// `and` that NumPy gives. On floats, complex numbers and 64-bit integers, their `reduce` and `accumulate` run rustymath's sums and
/// products. Narrower integers fold item by item.
#[derive(Clone, Copy)]
struct Add;
#[derive(Clone, Copy)]
struct Multiply;

/// Writes each type's kernel for `$k`, which runs the function object `$obj`. With a bracketed reduction `$r`, `reduce` folds each
/// lane by `$r`, then combines that with the running total, and `accumulate` runs `$r`'s scan. An integer fold or scan
/// wraps, as `$obj` does, so the kernel ignores whether it fits. `@half` writes the `f16` kernel, whose running total is `$r`'s `f32`
/// total across every chunk. `reduce` rounds that total once at the end. `accumulate` rounds each running result once. `@fold` writes the fold methods of a kernel whose running total is its item. `Logical` uses it too.
macro_rules! arithmetic {
    (@one $k:ty, $e:ty, $obj:expr, [$($r:ty)?]) => {
        impl Kernel<$e, 2> for $k {
            type Out = $e;
            type Total = $e;
            const FOLDS: bool = arithmetic!(@folds $($r)?);
            fn many(&self, [x, y]: [Arg<'_, $e>; 2], n: usize, out: &mut Vec<$e>) -> bool { Binary::<$e>::many(&$obj, x, y, n, out) }
            $(arithmetic!(@fold $e, $r);)?
        }
    };
    (@half $k:ty, $obj:expr, $r:ty) => {
        impl Kernel<f16, 2> for $k {
            type Out = f16;
            type Total = f32;
            const FOLDS: bool = true;
            fn many(&self, [x, y]: [Arg<'_, f16>; 2], n: usize, out: &mut Vec<f16>) -> bool { Binary::<f16>::many(&$obj, x, y, n, out) }
            fn total(&self, acc: f16) -> f32 { acc.to_f32() }
            unsafe fn reduce(&self, t: f32, n: usize, p: *mut c_char, step: npy_intp) -> f32 { unsafe { fold_lane::<f16, $r>(t, n, p, step) } }
            fn scan(&self, t: f32, x: &[f16], out: &mut Vec<f16>) -> f32 {
                let mut totals = Vec::with_capacity(x.len());
                <$r as Reduction<f16>>::scan(x, Some(t), &mut totals);
                out.extend(totals.iter().map(|&t| f16::from_f32(t)));
                totals[x.len() - 1]
            }
            fn result(&self, t: f32) -> f16 { f16::from_f32(t) }
        }
    };
    (@fold $e:ty, $r:ty) => {
        fn total(&self, acc: $e) -> $e { acc }
        unsafe fn reduce(&self, t: $e, n: usize, p: *mut c_char, step: npy_intp) -> $e { unsafe { fold_lane::<$e, $r>(t, n, p, step) } }
        fn scan(&self, t: $e, x: &[$e], out: &mut Vec<$e>) -> $e {
            <$r as Reduction<$e>>::scan(x, Some(t), out);
            out[out.len() - 1]
        }
        fn result(&self, t: $e) -> $e { t }
    };
    (@folds) => { false };
    (@folds $r:ty) => { true };
    ($k:ty = $obj:expr, $folds:tt; $($e:ty),+) => { $(arithmetic!(@one $k, $e, $obj, $folds);)+ };
}

arithmetic!(Add = rustymath::Add, [reduce::Sum]; f32, f64, Complex32, Complex64, i64, u64);
arithmetic!(Multiply = rustymath::Multiply, [reduce::Product]; f32, f64, Complex32, Complex64, i64, u64);
arithmetic!(@half Add, rustymath::Add, reduce::Sum);
arithmetic!(@half Multiply, rustymath::Multiply, reduce::Product);
arithmetic!(Add = rustymath::Add, []; i8, i16, i32, u8, u16, u32);
arithmetic!(Multiply = rustymath::Multiply, []; i8, i16, i32, u8, u16, u32);
arithmetic!(Add = reduce::Maximum, [reduce::Max]; bool);
arithmetic!(Multiply = reduce::Minimum, [reduce::Min]; bool);
arithmetic!(reduce::Maximum = reduce::Maximum, [reduce::Max]; bool, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);
arithmetic!(reduce::Minimum = reduce::Minimum, [reduce::Min]; bool, f16, f32, f64, i8, i16, i32, i64, u8, u16, u32, u64);

/// The kernel of `logical_xor` and `logical_xnor`, whose step, fold and scan come from the Boolean reduction `R`.
struct Logical<R>(std::marker::PhantomData<R>);
impl<R: Reduction<bool, Total = bool> + Sync> Kernel<bool, 2> for Logical<R> {
    type Out = bool;
    type Total = bool;
    const FOLDS: bool = true;
    fn many(&self, [x, y]: [Arg<'_, bool>; 2], n: usize, out: &mut Vec<bool>) -> bool {
        rustymath::pairs(x, y, n, out, |a, b| (R::step(a, b), true))
    }
    arithmetic!(@fold bool, R);
}

/// Division has a free complex helper rather than a function object.
struct Divide;
macro_rules! division {
    ($($e:ty),+; |$x:ident, $y:ident| $body:expr) => {$(
        impl Kernel<$e, 2> for Divide {
            type Out = $e;
            type Total = $e;
            fn many(&self, [x, y]: [Arg<'_, $e>; 2], n: usize, out: &mut Vec<$e>) -> bool {
                rustymath::pairs(x, y, n, out, |$x, $y| {
                    let r = $body;
                    (r, !r.is_nan() | $x.is_nan() | $y.is_nan())
                })
            }
        }
    )+};
}
division!(f16, f32, f64; |x, y| x / y);
division!(Complex32, Complex64; |x, y| rustymath::divide(x, y));

/// Whether the array at `p` with stride `step` holds contiguous, aligned items of type `T`.
fn dense<T>(p: *mut c_char, step: npy_intp) -> bool { step == size_of::<T>() as npy_intp && p as usize % align_of::<T>() == 0 }

/// Items `start..start + len` of the input at `p` with stride `step`: one item where the stride is 0, the items themselves where they
/// are contiguous and aligned, and otherwise a copy in `buf`.
unsafe fn input<'a, E: Code>(p: *mut c_char, step: npy_intp, start: usize, len: usize, buf: &'a mut Vec<E>) -> Arg<'a, E> {
    let at = |j: usize| unsafe { p.offset((start + j) as isize * step) as *const E };
    if step == 0 { return Arg::One(unsafe { at(0).read_unaligned() }) }
    if dense::<E>(p, step) { return Arg::Many(unsafe { slice::from_raw_parts(at(0), len) }) }
    buf.clear();
    buf.extend((0..len).map(|j| unsafe { at(j).read_unaligned() }));
    Arg::Many(buf)
}

/// Writes `out` to the output at `p` with stride `step`, from item `start`.
unsafe fn write<O: Code>(out: &[O], p: *mut c_char, step: npy_intp, start: usize) {
    let at = |j: usize| unsafe { p.offset((start + j) as isize * step) as *mut O };
    if dense::<O>(p, step) { unsafe { slice::from_raw_parts_mut(at(0), out.len()) }.copy_from_slice(out) }
    else { for (j, &r) in out.iter().enumerate() { unsafe { at(j).write_unaligned(r) } } }
}

/// Whether different items write disjoint bytes and cannot change any other item's inputs. Exact in-place mappings are safe:
/// each chunk finishes reading its inputs before writing its results. Any other input/output overlap stays serial.
fn independent<E: Code, O: Code>(n: usize, x: &[*mut c_char], xs: &[npy_intp], o: *mut c_char, os: npy_intp) -> bool {
    let Some(out) = parallel::separate(o as usize, size_of::<O>(), &[], n, os) else { return false };
    x.iter().zip(xs).all(|(&p, &step)| {
        (p == o && step == os && size_of::<E>() == size_of::<O>()) ||
            parallel::span(p as usize, size_of::<E>(), &[(n, step)]).is_some_and(|s| !parallel::overlaps(&s, &out))
    })
}

/// `k` on `n` items of the inputs `x` with strides `xs`, written to the output `o` with stride `os`, giving whether every item
/// succeeded. `reduce` and `accumulate` pass the output as the first input, which then changes from item to item. Otherwise `many`
/// runs on independent ranges, or serially where the arrays overlap in another way.
unsafe fn run<K: Kernel<E, N>, E: Code, const N: usize>(k: &K, n: usize, x: &[*mut c_char], xs: &[npy_intp], o: *mut c_char, os: npy_intp) -> bool {
    let reducing = N == 2 && os == 0 && xs[0] == 0 && o == x[0];
    let accumulating = N == 2 && os != 0 && xs[0] == os && o == x[0].wrapping_offset(os);
    if reducing || accumulating {
        return unsafe { if K::FOLDS && xs[1] != 0 { folded::<K, E, N>(k, n, x, xs, o, os, reducing) } else { serial::<K, E, N>(k, n, x, xs, o, os) } };
    }
    let addresses = array::from_fn::<_, N, _>(|i| x[i] as usize);
    let out = o as usize;
    let write = |range| unsafe { chunked::<K, E, N>(k, range, &addresses, xs, out, os) };
    // SAFETY: the NumPy callback owns these arrays until `ranges` returns. `independent` proves that distinct ranges cannot race.
    if independent::<E, K::Out>(n, x, xs, o, os) { rustymath::parallel::ranges(n, CHUNK, &write) }
    else { write(0..n) }
}

/// One independent range, with bounded input and output scratch buffers local to this worker.
unsafe fn chunked<K: Kernel<E, N>, E: Code, const N: usize>(
    k: &K, range: Range<usize>, x: &[usize; N], xs: &[npy_intp], o: usize, os: npy_intp,
) -> bool {
    let (mut bufs, mut out, mut ok): ([Vec<E>; N], _, _) = (array::from_fn(|_| Vec::new()), Vec::with_capacity(CHUNK.min(range.len())), true);
    for start in (range.start..range.end).step_by(CHUNK) {
        let len = CHUNK.min(range.end - start);
        let mut buf = bufs.iter_mut();
        let args = array::from_fn(|i| unsafe { input(x[i] as *mut c_char, xs[i], start, len, buf.next().unwrap()) });
        out.clear();
        ok &= k.many(args, len, &mut out);
        unsafe { write(&out, o as *mut c_char, os, start) }
    }
    ok
}

/// `reduce` or `accumulate` one item at a time, each reading the result that the one before it wrote.
unsafe fn serial<K: Kernel<E, N>, E: Code, const N: usize>(k: &K, n: usize, x: &[*mut c_char], xs: &[npy_intp], o: *mut c_char, os: npy_intp) -> bool {
    let (mut out, mut ok) = (Vec::with_capacity(1), true);
    for j in 0..n as isize {
        let args = array::from_fn(|i| Arg::One(unsafe { (x[i].offset(j * xs[i]) as *const E).read_unaligned() }));
        out.clear();
        ok &= k.many(args, 1, &mut out);
        unsafe { (o.offset(j * os) as *mut K::Out).write_unaligned(out[0]) }
    }
    ok
}

/// A whole strided lane folded in bounded pieces, without writing to any aliased input until all the pieces finish. The running total
/// stays in `R::Total`, including `f32` for half sums/products. NumPy integer reductions wrap, so overflow status is ignored.
unsafe fn fold_lane<E: Code, R: Reduction<E>>(t: R::Total, n: usize, p: *mut c_char, step: npy_intp) -> R::Total {
    let (address, mut total) = (p as usize, [R::unit()]);
    reduce::fold_pieces::<E, R>(&mut total, [1, n, 1], &|range, fold| {
        let mut buf = Vec::new();
        for start in (range.start..range.end).step_by(CHUNK) {
            let len = CHUNK.min(range.end - start);
            let Arg::Many(items) = (unsafe { input(address as *mut c_char, step, start, len, &mut buf) }) else { unreachable!("the lane has a stride") };
            fold(start, items);
        }
    });
    R::merge(t, total[0]).0
}

/// `reduce` through whole-lane pieces or `accumulate` through serial chunks. Scans retain the kernel's `Total` across chunks.
unsafe fn folded<K: Kernel<E, N>, E: Code, const N: usize>(
    k: &K, n: usize, x: &[*mut c_char], xs: &[npy_intp], o: *mut c_char, os: npy_intp, reducing: bool,
) -> bool {
    let mut t = k.total(unsafe { (x[0] as *const K::Out).read_unaligned() });
    if reducing {
        t = unsafe { k.reduce(t, n, x[1], xs[1]) };
        unsafe { (o as *mut K::Out).write_unaligned(k.result(t)) }
        return true;
    }
    let (mut buf, mut out) = (Vec::new(), Vec::new());
    for start in (0..n).step_by(CHUNK) {
        let len = CHUNK.min(n - start);
        let Arg::Many(items) = (unsafe { input(x[1], xs[1], start, len, &mut buf) }) else { unreachable!("the second input has a stride") };
        out.clear();
        t = k.scan(t, items, &mut out);
        unsafe { write(&out, o, os, start) }
    }
    true
}

/// A ufunc's kernel, made from the tolerance `ct` where the ufunc takes one as its last input.
trait Make {
    type K;
    const CT: bool;
    fn make(ct: f64) -> Self::K;
}

/// The ufunc loop for `M`'s kernel at element type `E`. It clears the floating-point status flags that rustymath's branch-free code
/// sets in lanes whose results it discards, then raises the result type's `FLAG` if an item failed.
unsafe extern "C" fn inner<M: Make, E: Code, const N: usize>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void)
where
    M::K: Kernel<E, N>,
{
    let n = unsafe { *dims } as usize;
    if n == 0 { return; }
    let previous = unsafe { fe::fetestexcept(fe::REPORTED) };
    let count = N + M::CT as usize + 1;
    let (args, steps) = unsafe { (slice::from_raw_parts(args, count), slice::from_raw_parts(steps, count)) };
    let ct = if M::CT { unsafe { (args[N] as *const f64).read_unaligned() } } else { 0.0 };
    let ok = unsafe { run::<M::K, E, N>(&M::make(ct), n, &args[..N], &steps[..N], args[count - 1], steps[count - 1]) };
    unsafe { fe::finish(previous, if ok { 0 } else { <<M::K as Kernel<E, N>>::Out as Code>::FLAG }); }
}

/// One loop of a ufunc: its function, and the type numbers of its inputs and output.
type Loop = (PyUFuncGenericFunction, Vec<c_char>);

/// The loop for `M`'s kernel at element type `E`.
fn entry<M: Make, E: Code, const N: usize>() -> Loop
where
    M::K: Kernel<E, N>,
{
    let ct = M::CT.then_some(NPY_TYPES::NPY_DOUBLE);
    let codes = [E::CODE; N].into_iter().chain(ct).chain([<<M::K as Kernel<E, N>>::Out as Code>::CODE]).map(|c| c as c_char).collect();
    (Some(inner::<M, E, N>), codes)
}

/// The loop at `bool`. NumPy picks the first loop whose types its inputs cast to safely, so it comes before every other loop.
fn boolean<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<bool, N>,
{
    vec![entry::<M, bool, N>()]
}

/// The loops at every integer type, narrower before wider and signed before unsigned, as NumPy orders its own. They come before the
/// float loops.
fn int<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<i8, N> + Kernel<u8, N> + Kernel<i16, N> + Kernel<u16, N> + Kernel<i32, N> + Kernel<u32, N> + Kernel<i64, N> + Kernel<u64, N>,
{
    vec![
        entry::<M, i8, N>(),
        entry::<M, u8, N>(),
        entry::<M, i16, N>(),
        entry::<M, u16, N>(),
        entry::<M, i32, N>(),
        entry::<M, u32, N>(),
        entry::<M, i64, N>(),
        entry::<M, u64, N>(),
    ]
}

/// The loops at `float16`, `float32` and `float64`.
fn real<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<f16, N> + Kernel<f32, N> + Kernel<f64, N>,
{
    let mut loops = vec![entry::<M, f16, N>()];
    loops.extend(native::<M, N>());
    loops
}

/// The native float loops, for functions such as `Cis` whose output has no float16 complex dtype.
fn native<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<f32, N> + Kernel<f64, N>,
{
    vec![entry::<M, f32, N>(), entry::<M, f64, N>()]
}

/// The loops at `complex64` and `complex128`.
fn complex<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<Complex32, N> + Kernel<Complex64, N>,
{
    vec![entry::<M, Complex32, N>(), entry::<M, Complex64, N>()]
}

/// Adds the ufunc `name` with `nin` inputs and `nout` outputs, NumPy identity `identity` and `loops` to `m`.
/// A `signature`, such as `"(m,n),(n,p)->(m,p)"`, makes it a generalized ufunc. Each of its loops then takes whole core matrices.
fn register(m: &Bound<'_, PyModule>, name: &str, doc: &str, [nin, nout]: [usize; 2], identity: c_int, signature: Option<&str>, loops: Vec<Loop>, data: Option<Vec<*mut c_void>>) -> PyResult<()> {
    let (funcs, codes): (Vec<_>, Vec<Vec<c_char>>) = loops.into_iter().unzip();
    // NumPy keeps these pointers for the life of the ufunc, which is the life of the process.
    let (count, funcs, types) = (funcs.len(), Vec::leak(funcs), Vec::leak(codes.concat()));
    let data = Vec::leak(data.unwrap_or_else(|| vec![ptr::null_mut(); count]));
    let (name_c, doc_c) = (CString::new(name)?.into_raw(), CString::new(doc)?.into_raw());
    let signature_c = signature.map(CString::new).transpose()?.map_or(ptr::null(), |s| s.into_raw().cast_const());
    let ufunc = unsafe {
        PY_UFUNC_API.PyUFunc_FromFuncAndDataAndSignature(
            m.py(), funcs.as_mut_ptr(), data.as_mut_ptr(), types.as_mut_ptr(), count as c_int, nin as c_int, nout as c_int, identity, name_c,
            doc_c, 0, signature_c,
        )
    };
    m.add(name, unsafe { Bound::from_owned_ptr_or_err(m.py(), ufunc)? })
}

/// NumPy's identities: `PyUFunc_Zero`, `PyUFunc_One`, `PyUFunc_None` and `PyUFunc_ReorderableNone`. A ufunc with an identity, or
/// a reorderable one, reduces over several axes at once.
const ZERO: c_int = 0;
const ONE: c_int = 1;
const NONE: c_int = -1;
const REORDERABLE: c_int = -2;

/// A generalized ufunc core loop computing both bounds of each lane.
unsafe extern "C" fn minmax_inner<E: Code>(args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void)
where
    reduce::Min: Reduction<E, Total = E>,
    reduce::Max: Reduction<E, Total = E>,
{
    let previous = unsafe { fe::fetestexcept(fe::REPORTED) };
    let (args, dims, steps) = unsafe { (slice::from_raw_parts(args, 3), slice::from_raw_parts(dims, 2), slice::from_raw_parts(steps, 4)) };
    let (n, mut buf) = (dims[1] as usize, Vec::new());
    for i in 0..dims[0] {
        let (lo, hi) = if n == 0 { reduce::extremes::<E>(&[]) } else {
            match unsafe { input(args[0].offset(i * steps[0]), steps[3], 0, n, &mut buf) } {
                Arg::One(v) => (v, v),
                Arg::Many(items) => reduce::extremes(items),
            }
        };
        unsafe {
            (args[1].offset(i * steps[1]) as *mut E).write_unaligned(lo);
            (args[2].offset(i * steps[2]) as *mut E).write_unaligned(hi);
        }
    }
    unsafe { fe::finish(previous, 0); }
}

fn minmax_entry<E: Code>() -> Loop
where
    reduce::Min: Reduction<E, Total = E>,
    reduce::Max: Reduction<E, Total = E>,
{
    (Some(minmax_inner::<E>), vec![E::CODE as c_char; 3])
}

/// Scans a Boolean array by `table[r][x]`, where `r` is the running result and `x` the next item.
#[pyfunction]
#[pyo3(signature = (x, table, axis=-1, start=None))]
fn logical_scan<'py>(py: Python<'py>, x: PyReadonlyArrayDyn<'py, bool>, table: [[bool; 2]; 2], axis: isize, start: Option<bool>) -> PyResult<Bound<'py, PyAny>> {
    let shape = x.shape();
    let axis = if axis < 0 { axis + shape.len() as isize } else { axis };
    if axis < 0 || axis >= shape.len() as isize { return Err(pyo3::exceptions::PyValueError::new_err("axis is out of bounds")); }
    let axis = axis as usize;
    let dims = [shape[..axis].iter().product(), shape[axis], shape[axis + 1..].iter().product()];
    let data = x.as_array();
    let data = data.as_standard_layout();
    let mut out = Vec::with_capacity(data.len());
    reduce::Logical(table).scan_along(data.as_slice().unwrap(), dims, start, &mut out);
    Ok(PyArray1::from_vec(py, out).reshape(shape)?.into_any())
}

/// Sets the thread limit for calls made on this thread, returning its previous setting. 0 restores the default, 1 runs serially.
#[pyfunction]
fn set_threads(n: usize) -> usize { rustymath::parallel::set_threads(n) }
/// The thread limit explicitly set on this thread, or 0 for the default.
#[pyfunction]
fn thread_setting() -> usize { rustymath::parallel::thread_setting() }
/// The effective thread count for calls made on this thread.
#[pyfunction]
fn threads() -> usize { rustymath::parallel::threads() }

/// For each entry, adds a ufunc to `m` whose kernel has type `$k`, made by `$make`. A `; ct` after the inputs makes the tolerance
/// the ufunc's last input, named `ct` in `$make`. The groups `int`, `real` and `complex` choose the loops.
macro_rules! ufuncs {
    ($m:expr; $($name:ident($($x:ident),+ $(; $ct:ident)?) $(identity $id:ident)?: $k:ty = $make:expr, $($group:ident),+: $doc:literal;)+) => {$({
        #[allow(non_camel_case_types)]
        struct $name;
        impl Make for $name {
            type K = $k;
            const CT: bool = ufuncs!(@ct $($ct)?);
            #[allow(unused_variables)]
            fn make(ct: f64) -> $k { $(let $ct = ct;)? $make }
        }
        const N: usize = [$(stringify!($x)),+].len();
        let identity = [$($id,)? NONE][0];
        register($m, stringify!($name), $doc, [N + <$name as Make>::CT as usize, 1], identity, None, [$($group::<$name, N>()),+].concat(), None)?;
    })+};
    (@ct) => { false };
    (@ct $ct:ident) => { true };
}

/// Entries of `ufuncs!` for rustymath function objects that take no settings: `$obj` is both the object's type and its value.
macro_rules! objects {
    ($m:expr; $($name:ident($($x:ident),+) = $obj:ident, $($group:ident),+: $doc:literal;)+) => {
        ufuncs! { $m; $($name($($x),+): Obj<rustymath::$obj> = Obj(rustymath::$obj), $($group),+: $doc;)+ }
    };
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    objects! { m;
        exp(x) = Exp, real, complex: "e^x.";
        expm1(x) = Expm1, real: "e^x - 1, without the loss of digits near 0 that `exp(x) - 1` has.";
        log(x) = Ln, real, complex: "The natural logarithm of x.";
        log1p(x) = Log1p, real: "ln(1 + x), without the loss of digits near 0 that `log(1 + x)` has.";
        log2(x) = Log2, real: "The logarithm of x to base 2.";
        log10(x) = Log10, real: "The logarithm of x to base 10.";
        logb(base, x) = Logb, real, complex: "The logarithm of x to the given base, with the base first.";
        sqrt(x) = Sqrt, real, complex: "The square root of x, the principal one for a complex x.";
        cbrt(x) = Cbrt, real: "The cube root of x.";
        sin(x) = Sin, real, complex: "sin x.";
        cos(x) = Cos, real, complex: "cos x.";
        cis(x) = Cis, native, complex: "exp(i x). A real float16 input promotes to complex64.";
        tan(x) = Tan, real, complex: "tan x.";
        arcsin(x) = Asin, real, complex: "asin x, from -π/2 to π/2 for a real x.";
        arccos(x) = Acos, real, complex: "acos x, from 0 to π for a real x.";
        arctan(x) = Atan, real, complex: "atan x, from -π/2 to π/2 for a real x.";
        sinh(x) = Sinh, real, complex: "sinh x.";
        cosh(x) = Cosh, real, complex: "cosh x.";
        tanh(x) = Tanh, real, complex: "tanh x.";
        arcsinh(x) = Asinh, real, complex: "asinh x.";
        arccosh(x) = Acosh, real, complex: "acosh x.";
        arctanh(x) = Atanh, real, complex: "atanh x.";
        absolute(x) = Abs, boolean, int, real, complex: "|x|, a real number for a complex x. A signed integer type's least value gives itself, as in NumPy.";
        sign(x) = Sign, int, real, complex: "-1, 0 or 1 for a real x, and x/|x| for a complex x.";
        angle(x) = Angle, real, complex: "The angle of x from the positive real axis, from -π to π.";
        sqrt1mx2(x) = Sqrt1mx2, real, complex: "√(1 - x²), the principal root for a complex x.";
        sqrt1px2(x) = Sqrt1px2, real, complex: "√(1 + x²), the principal root for a complex x.";
        sqrtx2m1(x) = Sqrtx2m1, real, complex: "√(x - 1) √(x + 1): √(x² - 1) with the sign of a real x.";
        gamma(x) = Gamma, real, complex: "Γ(x).";
        factorial(x) = Factorial, real: "Γ(x + 1), the floating-point extension of factorial.";
        gammaln(x) = Lgamma, real: "ln |Γ(x)|.";
        loggamma(x) = Lgamma, complex: "The principal branch of ln Γ(z), as SciPy's `loggamma` gives it for complex z.";
        lambertw(x) = LambertW, real, complex: "The principal branch W₀ of the Lambert W function. A real x below -1/e gives NaN.";
        erf(x) = Erf, real: "The error function, 2/√π times the integral of e^-t² from 0 to x.";
        erfc(x) = Erfc, real: "1 - erf(x), without the loss of digits for large x that `1 - erf(x)` has.";
        erfinv(p) = Erfinv, real: "The x with erf(x) = p.";
        erfcinv(q) = Erfcinv, real: "The x with erfc(x) = q.";
        power(x, y) = Pow, int, real, complex: "x^y. An integer power gives 0 and NumPy's overflow warning where it doesn't fit its type, or where y is negative and x isn't 1 or -1.";
        subtract(x, y) = Subtract, int, real, complex: "x - y. Integer arithmetic wraps, as in NumPy.";
        root(n, y) = Root, real: "The real nth root of y, computed without rounding 1/n first.";
        hypot(x, y) = Hypot, real: "√(x² + y²), with no overflow or underflow in between.";
        arctan2(y, x) = Atan2, real: "The angle of the point (x, y) from the positive x axis, from -π to π.";
        binom(n, k) = Binom, int, real, complex: "The binomial coefficient of n and k. Integer results keep their dtype and report overflow where they don't fit.";
        gammainc(a, x) = Gammainc, real: "The regularized lower incomplete gamma function P(a, x).";
        gammaincc(a, x) = Gammaincc, real: "The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).";
        gammaincinv(a, p) = Gammaincinv, real: "The x with P(a, x) = p.";
        betainc(a, b, x) = Betainc, real: "The regularized incomplete beta function I_x(a, b).";
        betaincinv(a, b, p) = Betaincinv, real: "The x with I_x(a, b) = p.";
    }
    ufuncs! { m;
        add(x, y) identity ZERO: Add = Add, boolean, int, real, complex: "x + y, which for Booleans is their `or`. Its `reduce` and `accumulate` run rustymath's sums.";
        multiply(x, y) identity ONE: Multiply = Multiply, boolean, int, real, complex: "x × y, which for Booleans is their `and`. Its `reduce` and `accumulate` run rustymath's products.";
        divide(x, y): Divide = Divide, real, complex: "x / y, preserving IEEE results. Complex division uses rustymath's fast, narrower-range formula.";
        maximum(x, y) identity REORDERABLE: reduce::Maximum = reduce::Maximum, boolean, int, real: "The larger of x and y, or y where either is NaN. A reduction beside NaN gives NaN or the largest other item, depending on the target.";
        minimum(x, y) identity REORDERABLE: reduce::Minimum = reduce::Minimum, boolean, int, real: "The smaller of x and y, or y where either is NaN. A reduction beside NaN gives NaN or the smallest other item, depending on the target.";
        logical_and(x, y) identity ONE: reduce::Minimum = reduce::Minimum, boolean: "Boolean and, with optimized reduce and accumulate.";
        logical_or(x, y) identity ZERO: reduce::Maximum = reduce::Maximum, boolean: "Boolean or, with optimized reduce and accumulate.";
        logical_xor(x, y) identity ZERO: Logical<reduce::Xor> = Logical(std::marker::PhantomData), boolean: "Boolean exclusive or, with optimized reduce and accumulate.";
        logical_xnor(x, y) identity ONE: Logical<reduce::Xnor> = Logical(std::marker::PhantomData), boolean: "Boolean equivalence, with optimized reduce and accumulate.";
        equal(x, y; ct): Obj<Equal> = Obj(Equal(tolerance(ct))), int, real, complex: "Whether x equals y, within the tolerance for floats.";
        not_equal(x, y; ct): Obj<NotEqual> = Obj(NotEqual(tolerance(ct))), int, real, complex: "Whether x doesn't equal y, within the tolerance for floats.";
        less(x, y; ct): Obj<Less> = Obj(Less(tolerance(ct))), int, real: "Whether x is below y and doesn't equal it, within the tolerance for floats.";
        less_equal(x, y; ct): Obj<LessEqual> = Obj(LessEqual(tolerance(ct))), int, real: "Whether x is below y or equals it, within the tolerance for floats.";
        greater(x, y; ct): Obj<Greater> = Obj(Greater(tolerance(ct))), int, real: "Whether x is above y and doesn't equal it, within the tolerance for floats.";
        greater_equal(x, y; ct): Obj<GreaterEqual> = Obj(GreaterEqual(tolerance(ct))), int, real: "Whether x is above y or equals it, within the tolerance for floats.";
        floor(x; ct): Obj<Floor> = Obj(Floor(tolerance(ct))), boolean, int, real, complex: "The tolerant floor of x. Integers and Booleans keep their values.";
        ceil(x; ct): Obj<Ceil> = Obj(Ceil(tolerance(ct))), boolean, int, real, complex: "The tolerant ceiling of x. Integers and Booleans keep their values.";
        remainder(x, y; ct): Obj<Flipped<Residue>> = Obj(Flipped(Residue(tolerance(ct)))), int, real, complex:
            "The remainder of x divided by y, with the sign of y. A zero y gives x, by APL's rule. The tolerance applies to floats.";
        gcd(x, y; ct): Obj<Gcd> = Obj(Gcd(tolerance(ct))), int, real, complex: "The greatest common divisor of x and y. The tolerance applies to floats.";
        lcm(x, y; ct): Obj<Lcm> = Obj(Lcm(tolerance(ct))), int, real: "The least common multiple of x and y, with the sign of x y. The tolerance applies to floats.";
    }
    register(m, "minmax", "The minimum and maximum along an axis, preserving the input dtype. Empty lanes give the dtype's extrema identities.", [1, 2], NONE, Some("(n)->(),()"), vec![
        minmax_entry::<bool>(), minmax_entry::<i8>(), minmax_entry::<u8>(), minmax_entry::<i16>(), minmax_entry::<u16>(),
        minmax_entry::<i32>(), minmax_entry::<u32>(), minmax_entry::<i64>(), minmax_entry::<u64>(), minmax_entry::<f16>(),
        minmax_entry::<f32>(), minmax_entry::<f64>(),
    ], None)?;
    for f in [wrap_pyfunction!(logical_scan, m)?, wrap_pyfunction!(set_threads, m)?, wrap_pyfunction!(thread_setting, m)?, wrap_pyfunction!(threads, m)?] {
        m.add_function(f)?;
    }
    polynomial::add(m)?;
    search::add(m)?;
    integer::add(m)?;
    for (name, add) in [("number_theory", number_theory::add as fn(&Bound<'_, PyModule>) -> PyResult<()>), ("distribution", distribution::add), ("linalg", linalg::add), ("exact", exact::add)] {
        let sub = PyModule::new(m.py(), name)?;
        add(&sub)?;
        m.add_submodule(&sub)?;
    }
    m.add("CT", Tolerance::DEFAULT.value())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
