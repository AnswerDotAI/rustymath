//! rustymath's functions as NumPy ufuncs, each with a loop for float16, float32 and float64.
use std::ffi::{c_char, c_int, c_void, CString};
use std::{array, ptr, slice};

use half::f16;
use numpy::npyffi::{npy_intp, PyUFuncGenericFunction, NPY_TYPES, PY_UFUNC_API};
use pyo3::prelude::*;
use rustymath::tolerant::{Exactly, Tolerance, Tolerant};
use rustymath::{with_tolerance, Arg, Binary, Float, CHUNK};

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

/// The NumPy type number of a loop's input or output element.
trait Code: Copy { const CODE: NPY_TYPES; }
impl Code for f16 { const CODE: NPY_TYPES = NPY_TYPES::NPY_HALF; }
impl Code for f32 { const CODE: NPY_TYPES = NPY_TYPES::NPY_FLOAT; }
impl Code for f64 { const CODE: NPY_TYPES = NPY_TYPES::NPY_DOUBLE; }
impl Code for bool { const CODE: NPY_TYPES = NPY_TYPES::NPY_BOOL; }

/// Items `start..` of `a`.
fn from<F: Copy>(a: Arg<'_, F>, start: usize) -> Arg<'_, F> { match a { Arg::One(v) => Arg::One(v), Arg::Many(s) => Arg::Many(&s[start..]) } }

/// `many` on `CHUNK` items at a time, appending each chunk to one reused vector, which then copies into `out`.
fn chunked<F: Copy>(x: Arg<'_, F>, y: Arg<'_, F>, out: &mut [F], many: impl Fn(Arg<'_, F>, Arg<'_, F>, usize, &mut Vec<F>)) {
    let mut buf = Vec::with_capacity(CHUNK.min(out.len()));
    for (k, o) in out.chunks_mut(CHUNK).enumerate() {
        buf.clear();
        many(from(x, k * CHUNK), from(y, k * CHUNK), o.len(), &mut buf);
        o.copy_from_slice(&buf);
    }
}

/// One ufunc's item function, of `N` floats. Tolerant functions compare within `t`, and the others ignore it.
trait Kernel<const N: usize> {
    type Out<F: Float + Code>: Code;
    fn f<F: Float + Code>(t: impl Tolerant, x: [F; N]) -> Self::Out<F>;
    /// `f`, or `None` where `f` takes a slow branch, which would stop a slice loop vectorizing.
    #[inline(always)]
    fn checked<F: Float + Code>(t: impl Tolerant, x: [F; N]) -> Option<Self::Out<F>> { Some(Self::f(t, x)) }
    /// `f` for each item of `out`, where each argument is one item or one item for each result. A kernel whose function has a slice
    /// form in rustymath runs that. Over slices, the default runs `checked`, then `f` where `checked` gave `None`. Where an argument is
    /// one item, the default runs `f` item by item.
    #[inline(always)]
    fn many<F: Float + Code>(t: impl Tolerant, x: [Arg<'_, F>; N], out: &mut [Self::Out<F>]) {
        let at = |a: Arg<'_, F>, j: usize| match a { Arg::One(v) => v, Arg::Many(s) => s[j] };
        if x.iter().any(|a| matches!(a, Arg::One(_))) {
            for (j, y) in out.iter_mut().enumerate() { *y = Self::f(t, array::from_fn(|i| at(x[i], j))) }
            return;
        }
        let xs: [&[F]; N] = x.map(|a| match a { Arg::Many(s) => s, Arg::One(_) => &[] });
        let mut slow = false;
        for (j, y) in out.iter_mut().enumerate() {
            let r = Self::checked(t, array::from_fn(|i| xs[i][j]));
            slow |= r.is_none();
            *y = r.unwrap_or(*y);
        }
        if !slow { return; }
        for (j, y) in out.iter_mut().enumerate() {
            let a = array::from_fn(|i| xs[i][j]);
            if Self::checked(t, a).is_none() { *y = Self::f(t, a); }
        }
    }
}

/// The ufunc loop for kernel `K` at type `F`, with `CT` saying whether the ufunc takes a tolerance as its last input.
unsafe extern "C" fn inner<K: Kernel<N>, F: Float + Code, const N: usize, const CT: bool>(
    args: *mut *mut c_char, dims: *mut npy_intp, steps: *mut npy_intp, _: *mut c_void,
) {
    let n = unsafe { *dims } as usize;
    if n == 0 { return; }
    let (args, steps) = unsafe { (slice::from_raw_parts(args, N + CT as usize + 1), slice::from_raw_parts(steps, N + CT as usize + 1)) };
    if !CT { return unsafe { run::<K, F, N>(Exactly, n, args, steps) }; }
    let ct = unsafe { (args[N] as *const f64).read_unaligned() };
    with_tolerance!(tolerance(ct), |t| unsafe { run::<K, F, N>(t, n, args, steps) })
}

/// `K` on `n` items. An input with stride 0 is one item, and an aligned input with a stride of one item is a slice. With every input
/// one of those and an aligned output with a stride of one item, `K::many` runs. Otherwise `K::f` runs item by item.
unsafe fn run<K: Kernel<N>, F: Float + Code, const N: usize>(t: impl Tolerant, n: usize, args: &[*mut c_char], steps: &[npy_intp]) {
    let out = args.len() - 1;
    let dense = |p: *mut c_char, step: npy_intp, size: usize| step == size as npy_intp && p as usize % size == 0;
    let arg = |i: usize| match steps[i] {
        0 => Some(Arg::One(unsafe { (args[i] as *const F).read_unaligned() })),
        _ if dense(args[i], steps[i], size_of::<F>()) => Some(Arg::Many(unsafe { slice::from_raw_parts(args[i] as *const F, n) })),
        _ => None,
    };
    let x: [Option<Arg<'_, F>>; N] = array::from_fn(arg);
    if x.iter().all(Option::is_some) && dense(args[out], steps[out], size_of::<K::Out<F>>()) {
        let ys = unsafe { slice::from_raw_parts_mut(args[out] as *mut K::Out<F>, n) };
        K::many(t, x.map(Option::unwrap), ys);
    } else {
        for j in 0..n as isize {
            let x = array::from_fn(|i| unsafe { (args[i].offset(j * steps[i]) as *const F).read_unaligned() });
            unsafe { (args[out].offset(j * steps[out]) as *mut K::Out<F>).write_unaligned(K::f(t, x)) };
        }
    }
}

/// Adds `K` to `m` as the ufunc `name`.
fn add<K: Kernel<N>, const N: usize, const CT: bool>(m: &Bound<'_, PyModule>, name: &str, doc: &str) -> PyResult<()> {
    fn codes<K: Kernel<N>, F: Float + Code, const N: usize, const CT: bool>() -> impl Iterator<Item = c_char> {
        let ct = CT.then_some(f64::CODE);
        [F::CODE; N].into_iter().chain(ct).chain([K::Out::<F>::CODE]).map(|c| c as c_char)
    }
    // NumPy keeps these pointers for the life of the ufunc, which is the life of the process.
    let funcs: &mut [PyUFuncGenericFunction] =
        Vec::leak(vec![Some(inner::<K, f16, N, CT>), Some(inner::<K, f32, N, CT>), Some(inner::<K, f64, N, CT>)]);
    let types = Vec::leak(codes::<K, f16, N, CT>().chain(codes::<K, f32, N, CT>()).chain(codes::<K, f64, N, CT>()).collect());
    let data = Vec::leak(vec![ptr::null_mut(); funcs.len()]);
    let (name_c, doc_c) = (CString::new(name)?.into_raw(), CString::new(doc)?.into_raw());
    let nin = (N + CT as usize) as c_int;
    let ufunc = unsafe {
        PY_UFUNC_API.PyUFunc_FromFuncAndData(m.py(), funcs.as_mut_ptr(), data.as_mut_ptr(), types.as_mut_ptr(), funcs.len() as c_int, nin, 1, -1, name_c, doc_c, 0)
    };
    m.add(name, unsafe { Bound::from_owned_ptr_or_err(m.py(), ufunc)? })
}

/// For each entry, defines a kernel type and adds it to `m` as a ufunc. `$t` names the tolerance in the bodies, and `$ct` says
/// whether the ufunc takes it as an extra input. A body of type `F` gives an array of the inputs' type, and `bool` a bool array.
/// `checked` names a unary function that gives `None` where the body takes a slow branch.
/// `many` names the function object in rustymath whose slice form runs over slices and broadcast items.
macro_rules! ufuncs {
    ($m:expr, $ct:literal, $t:ident; $($name:ident($($x:ident),+) -> $out:ident = $body:expr, $(checked $checked:ident,)? $(many $many:ident,)? $doc:literal;)+) => {$({
        #[allow(non_camel_case_types)]
        struct $name;
        impl $name { const N: usize = [$(stringify!($x)),+].len(); }
        impl Kernel<{ $name::N }> for $name {
            type Out<F: Float + Code> = $out;
            #[inline(always)]
            fn f<F: Float + Code>($t: impl Tolerant, [$($x),+]: [F; $name::N]) -> $out { $body }
            $(
                #[inline(always)]
                fn checked<F: Float + Code>(_: impl Tolerant, [x]: [F; 1]) -> Option<$out> { rustymath::$checked(x) }
            )?
            $(
                #[inline(always)]
                fn many<F: Float + Code>(_: impl Tolerant, [x, y]: [Arg<'_, F>; $name::N], out: &mut [$out]) { chunked(x, y, out, |x, y, n, buf| { rustymath::$many.many(x, y, n, buf); }) }
            )?
        }
        add::<$name, { $name::N }, $ct>($m, stringify!($name), $doc)?;
    })+};
}

#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    ufuncs! { m, false, _t;
        exp(x) -> F = rustymath::exp(x), "e^x.";
        expm1(x) -> F = rustymath::expm1(x), "e^x - 1, without the loss of digits near 0 that `exp(x) - 1` has.";
        log(x) -> F = rustymath::ln(x), "The natural logarithm of x.";
        log1p(x) -> F = rustymath::log1p(x), "ln(1 + x), without the loss of digits near 0 that `log(1 + x)` has.";
        log2(x) -> F = rustymath::log2(x), "The logarithm of x to base 2.";
        log10(x) -> F = rustymath::log10(x), "The logarithm of x to base 10.";
        sin(x) -> F = rustymath::sin(x), checked checked_sin, "sin x.";
        cos(x) -> F = rustymath::cos(x), checked checked_cos, "cos x.";
        tan(x) -> F = rustymath::tan(x), checked checked_tan, "tan x.";
        arcsin(x) -> F = rustymath::asin(x), "asin x, from -π/2 to π/2.";
        arccos(x) -> F = rustymath::acos(x), "acos x, from 0 to π.";
        arctan(x) -> F = rustymath::atan(x), "atan x, from -π/2 to π/2.";
        sinh(x) -> F = rustymath::sinh(x), "sinh x.";
        cosh(x) -> F = rustymath::cosh(x), "cosh x.";
        tanh(x) -> F = rustymath::tanh(x), "tanh x.";
        arcsinh(x) -> F = rustymath::asinh(x), "asinh x.";
        arccosh(x) -> F = rustymath::acosh(x), "acosh x.";
        arctanh(x) -> F = rustymath::atanh(x), "atanh x.";
        power(x, y) -> F = rustymath::pow(x, y), many Pow, "x^y.";
        root(n, y) -> F = rustymath::root(n, y), "The real nth root of y, computed without rounding 1/n first.";
        cbrt(x) -> F = rustymath::cbrt(x), "The cube root of x.";
        hypot(x, y) -> F = rustymath::hypot(x, y), "√(x² + y²), with no overflow or underflow in between.";
        gamma(x) -> F = rustymath::gamma(x), "Γ(x).";
        gammaln(x) -> F = rustymath::lgamma(x), "ln |Γ(x)|.";
        erf(x) -> F = rustymath::erf(x), "The error function, 2/√π times the integral of e^-t² from 0 to x.";
        erfc(x) -> F = rustymath::erfc(x), "1 - erf(x), without the loss of digits for large x that `1 - erf(x)` has.";
        erfinv(p) -> F = rustymath::erfinv(p), "The x with erf(x) = p.";
        erfcinv(q) -> F = rustymath::erfcinv(q), "The x with erfc(x) = q.";
        gammainc(a, x) -> F = rustymath::gammainc(a, x), "The regularized lower incomplete gamma function P(a, x).";
        gammaincc(a, x) -> F = rustymath::gammaincc(a, x), "The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).";
        gammaincinv(a, p) -> F = rustymath::gammaincinv(a, p), "The x with P(a, x) = p.";
        betainc(a, b, x) -> F = rustymath::betainc(a, b, x), "The regularized incomplete beta function I_x(a, b).";
        betaincinv(a, b, p) -> F = rustymath::betaincinv(a, b, p), "The x with I_x(a, b) = p.";
    }
    ufuncs! { m, true, t;
        equal(x, y) -> bool = t.equal(x, y), "Whether x equals y within the tolerance.";
        floor(x) -> F = t.floor(x), "The tolerant floor of x.";
        ceil(x) -> F = t.ceil(x), "The tolerant ceiling of x.";
        remainder(x, y) -> F = t.residue(y, x), "The tolerant remainder of x divided by y, with the sign of y.";
        gcd(x, y) -> F = t.gcd(x, y), "The tolerant greatest common divisor of x and y.";
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
