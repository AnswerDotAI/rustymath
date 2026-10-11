//! rustymath's functions as NumPy ufuncs. Each loop runs a rustymath function object's slice form on chunks of `CHUNK` items. An
//! input with stride 0 is one item beside the others, and a strided input is copied into a buffer first. Through `numpy.errstate`,
//! NumPy reports "invalid value" for a NaN result from arguments that aren't NaN, and "overflow" for an integer result that doesn't
//! fit. Each loop clears the other floating-point status flags, which rustymath's branch-free code sets in lanes it discards.
use std::ffi::{c_char, c_int, c_void, CString};
use std::{array, ptr, slice};

use half::f16;
use numpy::npyffi::{npy_intp, PyUFuncGenericFunction, NPY_TYPES, PY_UFUNC_API};
use numpy::{Complex32, Complex64};
use pyo3::prelude::*;
use rustymath::tolerant::{Ceil, Equal, Floor, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, NotEqual, Residue, Tolerance, Tolerant};
use rustymath::{reduce, Arg, Binary, Ternary, Unary, CHUNK};

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
mod number_theory;
mod polynomial;
mod search;

/// The tolerance that `ct` gives, or `Tolerance::EXACT` where it's out of range. The Python wrappers check the range first.
fn tolerance(ct: f64) -> Tolerance { Tolerance::new(ct).unwrap_or(Tolerance::EXACT) }

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
    }
}

/// An element type of a ufunc loop.
trait Code: Copy + Default {
    /// Its NumPy type number.
    const CODE: NPY_TYPES;
    /// The status flag for a loop with a failed item. NumPy then reports "invalid value", as it does for a NaN from arguments that
    /// aren't NaN.
    const FLAG: c_int = fe::INVALID;
}

macro_rules! codes {
    ($($t:ty => $code:ident),+) => {$(impl Code for $t { const CODE: NPY_TYPES = NPY_TYPES::$code; })+};
}
codes!(f16 => NPY_HALF, f32 => NPY_FLOAT, f64 => NPY_DOUBLE, Complex32 => NPY_CFLOAT, Complex64 => NPY_CDOUBLE, bool => NPY_BOOL);

impl Code for i64 {
    #[cfg(not(windows))]
    const CODE: NPY_TYPES = NPY_TYPES::NPY_LONG;
    #[cfg(windows)]
    const CODE: NPY_TYPES = NPY_TYPES::NPY_LONGLONG;
    /// An integer item fails where its result doesn't fit, and NumPy reports "overflow".
    const FLAG: c_int = fe::OVERFLOW;
}

/// A ufunc's function at element type `E`, with `N` inputs.
trait Kernel<E, const N: usize> {
    /// The type of each result.
    type Out: Code;
    /// Whether `reduce` and `accumulate` run `fold` and `scan`. Otherwise they run `many` item by item.
    const FOLDS: bool = false;
    /// Appends a result for each of `n` items to `out`, and gives whether every item succeeded.
    fn many(&self, x: [Arg<'_, E>; N], n: usize, out: &mut Vec<Self::Out>) -> bool;
    /// `acc` combined with each item of `x` in turn.
    fn fold(&self, _acc: Self::Out, _x: &[E]) -> Self::Out { unreachable!("only a kernel that folds has fold") }
    /// The running results of `fold` from `start`, appended to `out`.
    fn scan(&self, _start: Self::Out, _x: &[E], _out: &mut Vec<Self::Out>) { unreachable!("only a kernel that folds has scan") }
}

/// A rustymath function object as a kernel.
#[derive(Clone, Copy)]
struct Obj<K>(K);

impl<E: Copy, K: Unary<E, Out: Code>> Kernel<E, 1> for Obj<K> {
    type Out = K::Out;
    fn many(&self, [x]: [Arg<'_, E>; 1], n: usize, out: &mut Vec<K::Out>) -> bool { self.0.many(x, n, out) }
}

impl<E: Copy, K: Binary<E, Out: Code>> Kernel<E, 2> for Obj<K> {
    type Out = K::Out;
    fn many(&self, [x, y]: [Arg<'_, E>; 2], n: usize, out: &mut Vec<K::Out>) -> bool { self.0.many(x, y, n, out) }
}

impl<E: Copy, K: Ternary<E, Out: Code>> Kernel<E, 3> for Obj<K> {
    type Out = K::Out;
    fn many(&self, [x, y, z]: [Arg<'_, E>; 3], n: usize, out: &mut Vec<K::Out>) -> bool { self.0.many(x, y, z, n, out) }
}

/// `K` with its two arguments in the other order, for a ufunc whose NumPy order differs from rustymath's.
#[derive(Clone, Copy)]
struct Swapped<K>(K);

impl<E, K: Kernel<E, 2>> Kernel<E, 2> for Swapped<K> {
    type Out = K::Out;
    fn many(&self, [x, y]: [Arg<'_, E>; 2], n: usize, out: &mut Vec<K::Out>) -> bool { self.0.many([y, x], n, out) }
}

/// A result without its sign.
trait Magnitude { fn magnitude(self) -> Self; }
impl Magnitude for f16 { fn magnitude(self) -> Self { f16::from_bits(self.to_bits() & 0x7fff) } }
impl Magnitude for f32 { fn magnitude(self) -> Self { self.abs() } }
impl Magnitude for f64 { fn magnitude(self) -> Self { self.abs() } }
impl Magnitude for i64 { fn magnitude(self) -> Self { self.wrapping_abs() } }

/// `K` with each result's sign dropped, for NumPy's `lcm`, which is never negative.
#[derive(Clone, Copy)]
struct Absolute<K>(K);

impl<E, K: Kernel<E, N, Out: Magnitude>, const N: usize> Kernel<E, N> for Absolute<K> {
    type Out = K::Out;
    fn many(&self, x: [Arg<'_, E>; N], n: usize, out: &mut Vec<K::Out>) -> bool {
        let start = out.len();
        let ok = self.0.many(x, n, out);
        for r in &mut out[start..] { *r = r.magnitude() }
        ok
    }
}

/// NumPy's `add` and `multiply`. Their `reduce` and `accumulate` run `rustymath::reduce` where it has the type.
#[derive(Clone, Copy)]
struct Add;
#[derive(Clone, Copy)]
struct Multiply;

/// The running values of `f` from `start` over `x`, appended to `out`.
fn running<T: Copy>(start: T, x: &[T], out: &mut Vec<T>, f: impl Fn(T, T) -> T) {
    out.extend(x.iter().scan(start, |s, &v| { *s = f(*s, v); Some(*s) }));
}

/// Writes each type's kernel for `$k`, whose items combine by the closure after `:` or by the function object after `=`. The
/// bracketed `fold` and `scan`, where given, give `reduce` and `accumulate`.
macro_rules! arithmetic {
    (@one $k:ty, $e:ty, $many:tt, [$(fold $fold:expr, scan $scan:expr)?]) => {
        impl Kernel<$e, 2> for $k {
            type Out = $e;
            const FOLDS: bool = arithmetic!(@folds $($fold)?);
            fn many(&self, [x, y]: [Arg<'_, $e>; 2], n: usize, out: &mut Vec<$e>) -> bool { arithmetic!(@many $many, $e, x, y, n, out) }
            $(
                fn fold(&self, acc: $e, x: &[$e]) -> $e { let fold: fn($e, &[$e]) -> $e = $fold; fold(acc, x) }
                fn scan(&self, start: $e, x: &[$e], out: &mut Vec<$e>) { let scan: fn($e, &[$e], &mut Vec<$e>) = $scan; scan(start, x, out) }
            )?
        }
    };
    (@many (|$x:ident, $y:ident| $f:expr), $e:ty, $a:ident, $b:ident, $n:ident, $out:ident) => {
        rustymath::pairs($a, $b, $n, $out, |$x: $e, $y: $e| ($f, true))
    };
    (@many ($obj:expr), $e:ty, $a:ident, $b:ident, $n:ident, $out:ident) => { Binary::<$e>::many(&$obj, $a, $b, $n, $out) };
    (@folds) => { false };
    (@folds $fold:expr) => { true };
    ($k:ident: |$x:ident, $y:ident| $f:expr, $folds:tt; $($e:ty),+) => { $(arithmetic!(@one $k, $e, (|$x, $y| $f), $folds);)+ };
    ($k:ty = $obj:expr, $folds:tt; $($e:ty),+) => { $(arithmetic!(@one $k, $e, ($obj), $folds);)+ };
}

arithmetic!(Add: |x, y| x + y, [fold |acc, x| acc + reduce::sum(x).expect("a float sum always exists"),
    scan |start, x, out| { reduce::cumsum(x, Some(start), out); }]; f16, f32, f64);
arithmetic!(Multiply: |x, y| x * y, [fold |acc, x| acc * reduce::prod(x).expect("a float product always exists"),
    scan |start, x, out| { reduce::cumprod(x, Some(start), out); }]; f16, f32, f64);
arithmetic!(Add: |x, y| x.wrapping_add(y), [fold |acc, x| reduce::sum(x).map_or_else(|| x.iter().fold(acc, |a, &v| a.wrapping_add(v)), |s| acc.wrapping_add(s)),
    scan |start, x, out| running(start, x, out, i64::wrapping_add)]; i64);
arithmetic!(Multiply: |x, y| x.wrapping_mul(y), [fold |acc, x| reduce::prod(x).map_or_else(|| x.iter().fold(acc, |a, &v| a.wrapping_mul(v)), |p| acc.wrapping_mul(p)),
    scan |start, x, out| running(start, x, out, i64::wrapping_mul)]; i64);
arithmetic!(Add: |x, y| x + y, []; Complex32, Complex64);
arithmetic!(Multiply: |x, y| x * y, []; Complex32, Complex64);
arithmetic!(reduce::Fmax = reduce::Fmax, [fold |acc, x| reduce::Fmax.one(acc, reduce::max(x)),
    scan |start, x, out| reduce::cummax(x, Some(start), out)]; f16, f32, f64, i64);
arithmetic!(reduce::Fmin = reduce::Fmin, [fold |acc, x| reduce::Fmin.one(acc, reduce::min(x)),
    scan |start, x, out| reduce::cummin(x, Some(start), out)]; f16, f32, f64, i64);

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

/// `k` on `n` items of the inputs `x` with strides `xs`, written to the output `o` with stride `os`, giving whether every item
/// succeeded. `reduce` and `accumulate` pass the output as the first input, which then changes from item to item. Otherwise `many`
/// runs on chunks.
unsafe fn run<K: Kernel<E, N>, E: Code, const N: usize>(k: &K, n: usize, x: &[*mut c_char], xs: &[npy_intp], o: *mut c_char, os: npy_intp) -> bool {
    let reducing = N == 2 && os == 0 && xs[0] == 0 && o == x[0];
    let accumulating = N == 2 && os != 0 && xs[0] == os && o == x[0].wrapping_offset(os);
    if reducing || accumulating {
        return unsafe { if K::FOLDS && xs[1] != 0 { folded::<K, E, N>(k, n, x, xs, o, os, reducing) } else { serial::<K, E, N>(k, n, x, xs, o, os) } };
    }
    let (mut bufs, mut out, mut ok): ([Vec<E>; N], _, _) = (array::from_fn(|_| Vec::new()), Vec::with_capacity(CHUNK.min(n)), true);
    for start in (0..n).step_by(CHUNK) {
        let len = CHUNK.min(n - start);
        let mut buf = bufs.iter_mut();
        let args = array::from_fn(|i| unsafe { input(x[i], xs[i], start, len, buf.next().unwrap()) });
        out.clear();
        ok &= k.many(args, len, &mut out);
        unsafe { write(&out, o, os, start) }
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

/// `reduce` or `accumulate` through the kernel's `fold` or `scan`, on chunks of the second input. Neither can fail.
unsafe fn folded<K: Kernel<E, N>, E: Code, const N: usize>(
    k: &K, n: usize, x: &[*mut c_char], xs: &[npy_intp], o: *mut c_char, os: npy_intp, reducing: bool,
) -> bool {
    let mut acc = unsafe { (x[0] as *const K::Out).read_unaligned() };
    let (mut buf, mut out) = (Vec::new(), Vec::new());
    for start in (0..n).step_by(CHUNK) {
        let len = CHUNK.min(n - start);
        let Arg::Many(items) = (unsafe { input(x[1], xs[1], start, len, &mut buf) }) else { unreachable!("the second input has a stride") };
        if reducing {
            acc = k.fold(acc, items);
        } else {
            out.clear();
            k.scan(acc, items, &mut out);
            acc = out[len - 1];
            unsafe { write(&out, o, os, start) }
        }
    }
    if reducing { unsafe { (o as *mut K::Out).write_unaligned(acc) } }
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
    let count = N + M::CT as usize + 1;
    let (args, steps) = unsafe { (slice::from_raw_parts(args, count), slice::from_raw_parts(steps, count)) };
    let ct = if M::CT { unsafe { (args[N] as *const f64).read_unaligned() } } else { 0.0 };
    let ok = unsafe { run::<M::K, E, N>(&M::make(ct), n, &args[..N], &steps[..N], args[count - 1], steps[count - 1]) };
    unsafe {
        fe::feclearexcept(fe::REPORTED);
        if !ok { fe::feraiseexcept(<<M::K as Kernel<E, N>>::Out as Code>::FLAG); }
    }
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

/// The loops at `i64`. NumPy picks the first loop whose types its inputs cast to safely, so integer loops come before float ones.
fn int<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<i64, N>,
{
    vec![entry::<M, i64, N>()]
}

/// The loops at `float16`, `float32` and `float64`.
fn real<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<f16, N> + Kernel<f32, N> + Kernel<f64, N>,
{
    vec![entry::<M, f16, N>(), entry::<M, f32, N>(), entry::<M, f64, N>()]
}

/// The loops at `complex64` and `complex128`.
fn complex<M: Make, const N: usize>() -> Vec<Loop>
where
    M::K: Kernel<Complex32, N> + Kernel<Complex64, N>,
{
    vec![entry::<M, Complex32, N>(), entry::<M, Complex64, N>()]
}

/// Adds the ufunc `name` with `nin` inputs, NumPy identity `identity` and `loops` to `m`.
fn register(m: &Bound<'_, PyModule>, name: &str, doc: &str, nin: usize, identity: c_int, loops: Vec<Loop>) -> PyResult<()> {
    let (funcs, codes): (Vec<_>, Vec<Vec<c_char>>) = loops.into_iter().unzip();
    // NumPy keeps these pointers for the life of the ufunc, which is the life of the process.
    let (count, funcs, types) = (funcs.len(), Vec::leak(funcs), Vec::leak(codes.concat()));
    let data = Vec::leak(vec![ptr::null_mut(); count]);
    let (name_c, doc_c) = (CString::new(name)?.into_raw(), CString::new(doc)?.into_raw());
    let ufunc = unsafe {
        PY_UFUNC_API.PyUFunc_FromFuncAndData(
            m.py(), funcs.as_mut_ptr(), data.as_mut_ptr(), types.as_mut_ptr(), count as c_int, nin as c_int, 1, identity, name_c, doc_c, 0,
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
        register($m, stringify!($name), $doc, N + <$name as Make>::CT as usize, identity, [$($group::<$name, N>()),+].concat())?;
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
        sqrt(x) = Sqrt, real, complex: "The square root of x, the principal one for a complex x.";
        cbrt(x) = Cbrt, real: "The cube root of x.";
        sin(x) = Sin, real, complex: "sin x.";
        cos(x) = Cos, real, complex: "cos x.";
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
        absolute(x) = Abs, real, complex: "|x|, a real number for a complex x.";
        sign(x) = Sign, real, complex: "-1, 0 or 1 for a real x, and x/|x| for a complex x.";
        angle(x) = Angle, real, complex: "The angle of x from the positive real axis, from -π to π.";
        sqrt1mx2(x) = Sqrt1mx2, real, complex: "√(1 - x²), the principal root for a complex x.";
        sqrt1px2(x) = Sqrt1px2, real, complex: "√(1 + x²), the principal root for a complex x.";
        sqrtx2m1(x) = Sqrtx2m1, real, complex: "√(x - 1) √(x + 1): √(x² - 1) with the sign of a real x.";
        gamma(x) = Gamma, real, complex: "Γ(x).";
        gammaln(x) = Lgamma, real: "ln |Γ(x)|.";
        loggamma(x) = Lgamma, complex: "The principal branch of ln Γ(z), as SciPy's `loggamma` gives it for complex z.";
        lambertw(x) = LambertW, real, complex: "The principal branch W₀ of the Lambert W function. A real x below -1/e gives NaN.";
        erf(x) = Erf, real: "The error function, 2/√π times the integral of e^-t² from 0 to x.";
        erfc(x) = Erfc, real: "1 - erf(x), without the loss of digits for large x that `1 - erf(x)` has.";
        erfinv(p) = Erfinv, real: "The x with erf(x) = p.";
        erfcinv(q) = Erfcinv, real: "The x with erfc(x) = q.";
        power(x, y) = Pow, int, real, complex: "x^y. An integer power that doesn't fit int64 gives 0 and NumPy's overflow warning.";
        root(n, y) = Root, real: "The real nth root of y, computed without rounding 1/n first.";
        hypot(x, y) = Hypot, real: "√(x² + y²), with no overflow or underflow in between.";
        arctan2(y, x) = Atan2, real: "The angle of the point (x, y) from the positive x axis, from -π to π.";
        binom(n, k) = Binom, real, complex: "The binomial coefficient of n and k.";
        gammainc(a, x) = Gammainc, real: "The regularized lower incomplete gamma function P(a, x).";
        gammaincc(a, x) = Gammaincc, real: "The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).";
        gammaincinv(a, p) = Gammaincinv, real: "The x with P(a, x) = p.";
        betainc(a, b, x) = Betainc, real: "The regularized incomplete beta function I_x(a, b).";
        betaincinv(a, b, p) = Betaincinv, real: "The x with I_x(a, b) = p.";
    }
    ufuncs! { m;
        add(x, y) identity ZERO: Add = Add, int, real, complex: "x + y. Its `reduce` and `accumulate` run rustymath's sums.";
        multiply(x, y) identity ONE: Multiply = Multiply, int, real, complex: "x × y. Its `reduce` and `accumulate` run rustymath's products.";
        fmax(x, y) identity REORDERABLE: reduce::Fmax = reduce::Fmax, int, real: "The larger of x and y, or the other one where one is NaN.";
        fmin(x, y) identity REORDERABLE: reduce::Fmin = reduce::Fmin, int, real: "The smaller of x and y, or the other one where one is NaN.";
        equal(x, y; ct): Obj<Equal> = Obj(Equal(tolerance(ct))), real, complex: "Whether x equals y within the tolerance.";
        not_equal(x, y; ct): Obj<NotEqual> = Obj(NotEqual(tolerance(ct))), real, complex: "Whether x doesn't equal y within the tolerance.";
        less(x, y; ct): Obj<Less> = Obj(Less(tolerance(ct))), real: "Whether x is below y and doesn't equal it within the tolerance.";
        less_equal(x, y; ct): Obj<LessEqual> = Obj(LessEqual(tolerance(ct))), real: "Whether x is below y or equals it within the tolerance.";
        greater(x, y; ct): Obj<Greater> = Obj(Greater(tolerance(ct))), real: "Whether x is above y and doesn't equal it within the tolerance.";
        greater_equal(x, y; ct): Obj<GreaterEqual> = Obj(GreaterEqual(tolerance(ct))), real: "Whether x is above y or equals it within the tolerance.";
        floor(x; ct): Obj<Floor> = Obj(Floor(tolerance(ct))), real, complex: "The tolerant floor of x.";
        ceil(x; ct): Obj<Ceil> = Obj(Ceil(tolerance(ct))), real, complex: "The tolerant ceiling of x.";
        remainder(x, y; ct): Swapped<Obj<Residue>> = Swapped(Obj(Residue(tolerance(ct)))), int, real, complex:
            "The remainder of x divided by y, with the sign of y. The tolerance applies to floats.";
        gcd(x, y; ct): Obj<Gcd> = Obj(Gcd(tolerance(ct))), int, real, complex: "The greatest common divisor of x and y. The tolerance applies to floats.";
        lcm(x, y; ct): Absolute<Obj<Lcm>> = Absolute(Obj(Lcm(tolerance(ct)))), int, real: "The least common multiple of x and y, never negative. The tolerance applies to floats.";
    }
    polynomial::add(m)?;
    search::add(m)?;
    for (name, add) in [("number_theory", number_theory::add as fn(&Bound<'_, PyModule>) -> PyResult<()>), ("distribution", distribution::add)] {
        let sub = PyModule::new(m.py(), name)?;
        add(&sub)?;
        m.add_submodule(&sub)?;
    }
    m.add("CT", Tolerance::DEFAULT.value())?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
