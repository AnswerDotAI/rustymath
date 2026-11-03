//! Python argument conversion, promotion, signatures and result wrapping for rustymath functions. Numerical calls use the shared Kernel and Read interfaces with the GIL released.
use crate::array::{give, give_items};
use crate::convert::{Call, convert, convert_kinds, one, released};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustymath::agreement::Agreement;
use rustymath::array::{Array as Core, ArrayView};
use rustymath::buffer::{Buffer, ComplexWidth, DType, Element, FloatWidth, Items, View};
use rustymath::error::{Error, Kind};
use rustymath::layout::{Lanes, Layout, position};
use rustymath::linalg::Scalar;
use rustymath::stats;
use rustymath::tolerant::{Ceil, Equal, Floor, FloorDivide, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, MathishFloorDivide, NotEqual, Residue, Tolerance};
use rustymath::{Flipped, reduce, with_ints};
use rustymath_buffer::py_err;

/// The error for a function with no kernel for `dtype`.
pub fn missing(function: &str, dtype: DType) -> PyErr { py_err(Error::new(Kind::Domain, format!("{function} doesn't take {}", dtype.name()))) }

pub(crate) fn comparison(mathish: bool) -> Tolerance { if mathish { Tolerance::DEFAULT } else { Tolerance::EXACT } }
pub(crate) fn mathish_at(dtype: Option<DType>) -> bool { matches!(dtype, Some(DType::Float(FloatWidth::F64) | DType::Complex(ComplexWidth::F64))) }

/// An axis, given as a position, negative from the end, or as an axis name.
#[derive(FromPyObject)]
pub enum Axis { #[pyo3(annotation = "int")] Position(i64), #[pyo3(annotation = "str")] Name(String) }

impl Axis {
    /// The axis of `layout` that this names. An IndexError where it names none.
    pub fn of(&self, layout: &Layout) -> PyResult<usize> {
        let rank = layout.shape().len();
        match self {
            Axis::Name(s) => layout.axis_named(s).ok_or_else(|| py_err(Error::new(Kind::Index, format!("no axis is named {s:?}")))),
            &Axis::Position(a) => position(a, rank).ok_or_else(|| py_err(Error::new(Kind::Index, format!("axis {a} is out of range for rank {rank}")))),
        }
    }
}

/// The axes a reduction runs over: one axis, or a tuple of them.
#[derive(FromPyObject)]
pub enum Axes { #[pyo3(annotation = "int | str")] One(Axis), #[pyo3(annotation = "tuple")] Many(Vec<Axis>) }

impl Axes {
    fn positions(&self, layout: &Layout) -> PyResult<Vec<usize>> {
        let mut axes = match self { Self::One(a) => vec![a.of(layout)?], Self::Many(axes) => axes.iter().map(|a| a.of(layout)).collect::<PyResult<_>>()? };
        axes.sort_unstable();
        axes.dedup();
        Ok(axes)
    }
}

fn axes_of(axis: Option<&Axes>, layout: &Layout) -> PyResult<Vec<usize>> {
    axis.map_or_else(|| Ok((0..layout.shape().len()).collect()), |a| a.positions(layout))
}

/// Python's promotion policy; the shared formula declares which kinds it computes.
fn computes<K: rustymath::facts::Formula<N>, const N: usize>(function: &str, object: &K, dtype: DType, only: Option<&[DType]>) -> PyResult<DType> {
    match only {
        None => object.computing_dtype(dtype),
        Some(allowed) => {
            dtype.computes_in(rustymath::buffer::Width::I64, rustymath::operation::numpy_float(dtype), |at| allowed.contains(&at) && object.supports(at))
        }
    }.ok_or_else(|| missing(function, dtype))
}
pub(crate) fn select<K: rustymath::facts::Formula<N>, const N: usize>(
    function: &str, object: K, dtype: DType, types: [DType; N], mathish: bool, wrapping: bool, only: Option<&[DType]>
) -> PyResult<rustymath::operation::Kernel> {
    let dtype = if only.is_some() { computes(function, &object, dtype, only)? } else { dtype };
    let mut kernel = object.resolve(rustymath::facts::Variant { dtype, checked: true, mathish, wrapping }, types).map_err(py_err)?;
    kernel.variant.checked = kernel.variant.dtype.is_integral();
    Ok(kernel)
}
fn call_dtype(call: &Call) -> DType {
    if call.args.iter().any(|x| x.view().dtype() == DType::Character) { DType::Character } else { call.args[0].view().dtype() }
}
/// Conversion and wrapping surround one shared mapped execution, for every arity and kind.
pub(crate) fn apply<'py>(py: Python<'py>, kernel: &rustymath::operation::Kernel, call: &Call) -> PyResult<Bound<'py, PyAny>> {
    let views: Vec<_> = call.args.iter().map(|a| a.view()).collect();
    let (layout, inputs) = match views.as_slice() {
        [x] => (x.layout().clone(), vec![x.clone()]),
        [x, y] => {
            let pair = Agreement::new(x.layout(), y.layout()).map_err(py_err)?;
            let inputs = vec![x.remapped(pair.left, pair.len), y.remapped(pair.right, pair.len)];
            (pair.layout, inputs)
        }
        [x, y, z] => {
            let pair = Agreement::new(x.layout(), y.layout()).map_err(py_err)?;
            let whole = Agreement::new(&pair.layout, z.layout()).map_err(py_err)?;
            let inputs = vec![
                x.remapped(pair.left, pair.len).remapped(whole.left.clone(), whole.len), y.remapped(pair.right, pair.len).remapped(whole.left, whole.len),
                z.remapped(whole.right, whole.len),
            ];
            (whole.layout, inputs)
        }
        _ => return Err(PyValueError::new_err("an elementwise call needs one, two or three arguments")),
    };
    let mut buffer = Buffer::with_capacity(kernel.result_dtype(), layout.count());
    let reads: Vec<_> = inputs.iter().map(ArrayView::read).collect();
    released(py, || kernel.map(&reads, 0..layout.count(), rustymath::operation::Settings::default(), &mut buffer)).map_err(py_err)?;
    give_items(py, layout, buffer, call.scalar())
}

/// Folds use the same mapped reader and reduction driver as graph execution.
fn totals<K: rustymath::facts::Formula<2>>(
    function: &str, object: &K, x: ArrayView<'_>, axis: Option<&Axes>, only: Option<&[DType]>) -> PyResult<Option<Core>> {
    let mut axes = match axis { None => vec![None], Some(a) => a.positions(x.layout())?.into_iter().map(Some).collect() };
    axes.sort_unstable_by(|a, b| b.cmp(a));
    let mut out: Option<Core> = None;
    for axis in axes {
        let view = out.as_ref().map_or_else(|| x.clone(), Core::view);
        let layout = view.layout();
        let next = axis.map_or(Layout::EMPTY, |a| layout.axes((0..layout.shape().len()).filter(|&i| i != a)));
        let flat = [view.len()];
        let shape = axis.map_or(flat.as_slice(), |_| layout.shape());
        let kernel = select(function, object.clone(), view.dtype(), [view.dtype(); 2], false, true, only)?;
        let mut buffer = Buffer::with_capacity(kernel.fold_dtype(view.dtype(), shape[axis.unwrap_or(0)], None), next.count());
        kernel.fold_read(
                shape, axis.unwrap_or(0), rustymath::operation::Seed::Identity, rustymath::operation::Direction::Right,
                rustymath::operation::Settings::default(), view.read(), &mut buffer).map_err(py_err)?;
        out = Some(Core::new(next, buffer).map_err(py_err)?);
    }
    Ok(out)
}
fn fold<'py, K: rustymath::facts::Formula<2>>(
    py: Python<'py>, function: &str, object: &K, x: &Bound<'py, PyAny>, axis: Option<Axes>, only: Option<&[DType]>) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let view = x.view();
    let core = match released(py, || totals(function, object, view, axis.as_ref(), only))? { Some(core) => core, None => x.into_core()? };
    let scalar = core.layout().shape().is_empty();
    give(py, core, scalar)
}
pub(crate) fn along_axis(layout: Layout, len: usize, axis: Option<Axis>) -> PyResult<([usize; 3], Layout, Option<usize>)> {
    match axis {
        None => Ok(([1, len, 1], Layout::from([len]), None)),
        Some(a) => { let a = a.of(&layout)?; Ok((Lanes::new(layout.shape(), a).map_err(py_err)?.dims(), layout, Some(a))) }
    }
}
fn scan<'py, K: rustymath::facts::Formula<2>>(
    py: Python<'py>, function: &str, object: K, x: &Bound<'py, PyAny>, axis: Option<Axis>, only: Option<&[DType]>) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let view = x.view();
    let (dims, layout, _) = along_axis(x.layout(), view.len(), axis)?;
    let kernel = select(function, object, view.dtype(), [view.dtype(); 2], false, true, only)?;
    let mut buffer = Buffer::with_capacity(kernel.scan_dtype(), view.len());
    released(py, || {
        kernel.scan(
            dims, rustymath::operation::Seed::Identity, rustymath::operation::Direction::Left, rustymath::operation::Settings::default(), view.read(),
            &mut buffer)
    }).map_err(py_err)?;
    give_items(py, layout, buffer, false)
}

pub(crate) struct NativeBinary { pub kernel: fn(DType, bool) -> PyResult<rustymath::operation::Kernel> }
pub(crate) type NativeFunction<'a> = (&'a str, bool);
type Slopes = for<'py> fn(Python<'py>, &[Bound<'py, PyAny>], &Bound<'py, PyAny>, bool) -> PyResult<Vec<Bound<'py, PyAny>>>;
pub(crate) fn native_binary(name: &str) -> PyResult<&'static NativeBinary> {
    NATIVE_BINARIES.iter().flatten().find(|(n, _)| *n == name).map(|(_, b)| b)
        .ok_or_else(|| PyValueError::new_err(format!("{name:?} isn't a native binary function")))
}

/// Public names and signatures select objects; formulas, validity, reductions and partials stay in rustymath.
macro_rules! functions {
    ($($(#[doc = $($doc:tt)+])+ $name:ident($($x:ident),+ $(; $ct:ident)?) = $k:expr $(=> $mathish:expr)? $(, only $only:expr)? $(, wrapping $wrapping:expr)?;)+) => {
        $(functions!(@define false; $(#[doc = $($doc)+])+ $name($($x),+ $(; $ct)?) = $k $(=> $mathish)? $(, only $only)? $(, wrapping $wrapping)?);)+
        pub(crate) mod mathish {
            use super::*;
            $(functions!(@affected [$($ct)?], [$($mathish)?]; $(#[doc = $($doc)+])+ $name($($x),+ $(; $ct)?) = $k $(=> $mathish)? $(, only $only)? $(, wrapping $wrapping)?);)+
        }
        static NATIVE_BINARIES: &[Option<(&str, NativeBinary)>] = &[
            $(functions!(@native $name, [$($x),+], [$($ct)?], [$($mathish)?], $k $(, only $only)? $(, wrapping $wrapping)?)),+
        ];
        static NATIVE_SLOPES: &[(&str, Slopes)] = &[
            $((stringify!($name), functions!(@slopes $name, [$($x),+], [$($ct)?], [$($mathish)?], $k $(, only $only)?))),+
        ];
        fn add_elementwise(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
            m.add("_binary_names", NATIVE_BINARIES.iter().flatten().map(|(name, _)| *name).collect::<Vec<_>>())?;
            m.add("_elementwise_names", NATIVE_SLOPES.iter().map(|(name, _)| *name).collect::<Vec<_>>())?;
            Ok(())
        }
        pub(crate) fn add_mathish(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(functions!(@add [$($ct)?], [$($mathish)?]; m, mathish::$name);)+
            Ok(())
        }
    };
    (@affected [], []; $($rest:tt)*) => {};
    (@affected $ct:tt, $mathish:tt; $($rest:tt)*) => { functions!(@define true; $($rest)*); };
    (@add [], []; $m:ident, $f:path) => {};
    (@add $ct:tt, $mathish:tt; $m:ident, $f:path) => { $m.add_function(wrap_pyfunction!($f, $m)?)?; };
    (@define $on:expr; $(#[doc = $($doc:tt)+])+ $name:ident($($x:ident),+ $(; $ct:ident)?) = $k:expr $(=> $mathish:expr)? $(, only $only:expr)? $(, wrapping $wrapping:expr)?) => {
        $(#[doc = $($doc)+])+
        #[pyfunction]
        #[pyo3(signature = ($($x),+))]
        pub(crate) fn $name<'py>(py: Python<'py>, $($x: &Bound<'py, PyAny>,)+) -> PyResult<Bound<'py, PyAny>> {
            let call = functions!(@convert [$($x),+])?;
            let dtype = call_dtype(&call);
            let types: [DType; { functions!(@arity [$($x),+]) }] = ::std::array::from_fn(|i| call.args[i].view().dtype());
            let mode = functions!(@mode [$($x),+], [$($ct)?], $on, $k, dtype, None$(.or(Some($only)))?);
            functions!(@object [$($ct)?], [$($mathish)?], mode, $k, |k| {
                let kernel = select::<_, { functions!(@arity [$($x),+]) }>(stringify!($name), k, dtype, types, mode, false$(|| $wrapping)?, None$(.or(Some($only)))?)?;
                apply(py, &kernel, &call)
            })
        }
    };
    (@native $name:ident, [$x:ident, $y:ident], [$($ct:ident)?], [$($mathish:expr)?], $k:expr $(, only $only:expr)? $(, wrapping $wrapping:expr)?) => {
        Some((stringify!($name), NativeBinary { kernel: |dtype, on| {
            let mode = functions!(@mode [$x, $y], [$($ct)?], on, $k, dtype, None$(.or(Some($only)))?);
            functions!(@object [$($ct)?], [$($mathish)?], mode, $k, |k| {
                select::<_, 2>(stringify!($name), k, dtype, [dtype; 2], mode, false$(|| $wrapping)?, None$(.or(Some($only)))?)
            })
        } }))
    };
    (@native $name:ident, [$($x:ident),+], $($rest:tt)*) => { None };
    (@slopes $name:ident, [$x:ident], [$($ct:ident)?], [$($mathish:expr)?], $k:expr $(, only $only:expr)?) => {
        |py, args, r, on| {
            let [x] = args else { return Err(PyValueError::new_err(concat!(stringify!($name), " takes one argument"))) };
            functions!(@object [$($ct)?], [$($mathish)?], on, $k, |k| {
                let call = convert(&[x, r])?;
                let kernel = select::<_, 2>(stringify!($name), rustymath::derivative::Slope(k), call_dtype(&call), [call_dtype(&call); 2], on, false, None$(.or(Some($only)))?)?;
                Ok(vec![apply(py, &kernel, &call)?])
            })
        }
    };
    (@slopes $name:ident, [$x:ident, $y:ident], [$($ct:ident)?], [$($mathish:expr)?], $k:expr $(, only $only:expr)?) => {
        |py, args, r, on| {
            let [x, y] = args else { return Err(PyValueError::new_err(concat!(stringify!($name), " takes two arguments"))) };
            functions!(@object [$($ct)?], [$($mathish)?], on, $k, |k| {
                let call = convert(&[x, y, r])?;
                let dtype = call_dtype(&call);
                let kernels = [select::<_, 3>(stringify!($name), rustymath::derivative::Slope::<_, 0>(k.clone()), dtype, [dtype; 3], on, false, None$(.or(Some($only)))?)?,
                    select::<_, 3>(stringify!($name), rustymath::derivative::Slope::<_, 1>(k), dtype, [dtype; 3], on, false, None$(.or(Some($only)))?)?];
                kernels.iter().map(|kernel| apply(py, kernel, &call)).collect()
            })
        }
    };
    (@slopes $name:ident, [$($x:ident),+], $($rest:tt)*) => { |_, _, _, _| Err(PyValueError::new_err(concat!(stringify!($name), " has no derivative"))) };
    (@mode [$($x:ident),+], [$($ct:ident)?], $on:expr, $ordinary:expr, $dtype:expr, $only:expr) => { $on && $dtype != DType::Character && {
        $(let $ct = Tolerance::EXACT;)?
        mathish_at(Some(computes::<_, { functions!(@arity [$($x),+]) }>("function", &$ordinary, $dtype, $only)?))
    }};
    (@object [$($ct:ident)?], [], $mode:expr, $ordinary:expr, |$k:ident| $body:block) => {{
        $(let $ct = comparison($mode);)?
        let $k = $ordinary;
        $body
    }};
    (@object [$($ct:ident)?], [$mathish:expr], $mode:expr, $ordinary:expr, |$k:ident| $body:block) => {{
        let mode = $mode;
        $(let $ct = comparison(mode);)?
        if mode { let $k = $mathish; $body } else { let $k = $ordinary; $body }
    }};
    (@convert [$x:ident, $y:ident]) => { convert_kinds(&[$x, $y]) };
    (@convert [$($x:ident),+]) => { convert(&[$($x),+]) };
    (@arity [$x:ident]) => { 1 };
    (@arity [$x:ident, $y:ident]) => { 2 };
    (@arity [$x:ident, $y:ident, $z:ident]) => { 3 };
}

/// The real and complex dtypes, for an object whose forms serve two Python functions.
const REAL: &[DType] = &[DType::Float(FloatWidth::F16), DType::Float(FloatWidth::F32), DType::Float(FloatWidth::F64)];
const COMPLEX: &[DType] = &[DType::Complex(ComplexWidth::F32), DType::Complex(ComplexWidth::F64)];

functions! {
    /// Bitwise AND, logical for Booleans.
    bitwise_and(x, y) = rustymath::BitAnd;
    /// Bitwise OR, logical for Booleans.
    bitwise_or(x, y) = rustymath::BitOr;
    /// Bitwise XOR, logical for Booleans.
    bitwise_xor(x, y) = rustymath::BitXor;
    /// Bitwise complement, logical negation for Booleans.
    bitwise_not(x) = rustymath::BitNot;
    /// Shift left, discarding high bits. The count must be below the item width.
    left_shift(x, y) = rustymath::LeftShift;
    /// Shift right, preserving signed integers' sign bit. The count must be below the item width.
    right_shift(x, y) = rustymath::RightShift;
    /// Convert radians to degrees.
    degrees(x) = rustymath::Scale(180.0 / ::std::f64::consts::PI);
    /// Convert degrees to radians.
    radians(x) = rustymath::Scale(::std::f64::consts::PI / 180.0);
    /// Whether x has a NaN part.
    isnan(x) = rustymath::Isnan;
    /// Whether every part of x is finite.
    isfinite(x) = rustymath::Isfinite;
    /// Whether x has an infinite part.
    isinf(x) = rustymath::Isinf;
    /// The real part of x, with real numbers unchanged.
    real(x) = rustymath::Real;
    /// The imaginary part of x, zero for real numbers.
    imag(x) = rustymath::Imag;
    /// The complex conjugate of x, with real numbers unchanged.
    conj(x) = rustymath::Conj;
    /// Rounding toward zero, with whole numbers unchanged.
    trunc(x) = rustymath::Trunc;
    /// Rounding to the nearest whole number, with ties to even.
    rint(x) = rustymath::Rint;
    /// Clip x between lower and upper bounds, in one pass. The bounds broadcast as other arguments do.
    clip(x, lower, upper) = rustymath::Clip;
    #[doc = rustymath::function_doc!(exp)]
    exp(x) = rustymath::Exp;
    #[doc = rustymath::function_doc!(expm1)]
    expm1(x) = rustymath::Expm1;
    #[doc = rustymath::function_doc!(ln)]
    log(x) = rustymath::Ln;
    #[doc = rustymath::function_doc!(log1p)]
    log1p(x) = rustymath::Log1p;
    #[doc = rustymath::function_doc!(log2)]
    log2(x) = rustymath::Log2;
    #[doc = rustymath::function_doc!(log10)]
    log10(x) = rustymath::Log10;
    #[doc = rustymath::function_doc!(logb)]
    /// Mathish mode gives zero for base and argument one at float64 and complex128.
    logb(base, x) = rustymath::Logb => rustymath::MathishLogb;
    /// The square root of x, the principal one for a complex x.
    sqrt(x) = rustymath::Sqrt;
    #[doc = rustymath::function_doc!(cbrt)]
    cbrt(x) = rustymath::Cbrt;
    #[doc = rustymath::function_doc!(sin)]
    sin(x) = rustymath::Sin;
    #[doc = rustymath::function_doc!(cos)]
    cos(x) = rustymath::Cos;
    /// exp(i x). A real float16 x gives complex64.
    cis(x) = rustymath::Cis;
    #[doc = rustymath::function_doc!(tan)]
    tan(x) = rustymath::Tan;
    #[doc = rustymath::function_doc!(asin)]
    arcsin(x) = rustymath::Asin;
    #[doc = rustymath::function_doc!(acos)]
    arccos(x) = rustymath::Acos;
    #[doc = rustymath::function_doc!(atan)]
    arctan(x) = rustymath::Atan;
    #[doc = rustymath::function_doc!(sinh)]
    sinh(x) = rustymath::Sinh;
    #[doc = rustymath::function_doc!(cosh)]
    cosh(x) = rustymath::Cosh;
    #[doc = rustymath::function_doc!(tanh)]
    tanh(x) = rustymath::Tanh;
    #[doc = rustymath::function_doc!(asinh)]
    arcsinh(x) = rustymath::Asinh;
    #[doc = rustymath::function_doc!(acosh)]
    arccosh(x) = rustymath::Acosh;
    #[doc = rustymath::function_doc!(atanh)]
    arctanh(x) = rustymath::Atanh;
    /// |x|, a real number for a complex x. The magnitude of a signed integer type's least value wraps to that value, as NumPy's does.
    absolute(x) = rustymath::Abs, wrapping true;
    /// -1, 0 or 1 for a real x, and x/|x| for a complex x.
    sign(x) = rustymath::Sign;
    /// The angle of x from the positive real axis, from -π to π.
    angle(x) = rustymath::Angle;
    #[doc = rustymath::function_doc!(sqrt1mx2)]
    /// The principal root for a complex x.
    sqrt1mx2(x) = rustymath::Sqrt1mx2;
    #[doc = rustymath::function_doc!(sqrt1px2)]
    /// The principal root for a complex x.
    sqrt1px2(x) = rustymath::Sqrt1px2;
    #[doc = rustymath::function_doc!(sqrtx2m1)]
    sqrtx2m1(x) = rustymath::Sqrtx2m1;
    #[doc = rustymath::function_doc!(gamma)]
    gamma(x) = rustymath::Gamma;
    #[doc = rustymath::function_doc!(factorial)]
    factorial(x) = rustymath::Factorial;
    #[doc = rustymath::function_doc!(lgamma)]
    gammaln(x) = rustymath::Lgamma, only REAL;
    #[doc = rustymath::function_doc!(digamma)]
    digamma(x) = rustymath::Digamma, only REAL;
    #[doc = rustymath::function_doc!(polygamma)]
    polygamma(n, x) = rustymath::Polygamma;
    /// The principal branch of ln Γ(z), as SciPy's `loggamma` gives it for complex z. A real z computes as complex.
    loggamma(x) = rustymath::Lgamma, only COMPLEX;
    #[doc = rustymath::function_doc!(lambertw)]
    lambertw(x) = rustymath::LambertW;
    #[doc = rustymath::function_doc!(erf)]
    erf(x) = rustymath::Erf;
    #[doc = rustymath::function_doc!(erfc)]
    erfc(x) = rustymath::Erfc;
    #[doc = rustymath::function_doc!(erfinv)]
    erfinv(p) = rustymath::Erfinv;
    #[doc = rustymath::function_doc!(erfcinv)]
    erfcinv(q) = rustymath::Erfcinv;
    /// x + y. A character plus an integer, on either side, is a character. An integer result that doesn't fit its dtype wraps, as NumPy's does.
    add(x, y) = rustymath::Add, wrapping true;
    /// x - y. A character minus an integer is a character, and a character minus a character is an integer. An integer result that doesn't fit its dtype wraps, as NumPy's does.
    subtract(x, y) = rustymath::Subtract, wrapping true;
    /// x × y. An integer result that doesn't fit its dtype wraps, as NumPy's does.
    multiply(x, y) = rustymath::Multiply, wrapping true;
    /// x / y. Integers compute as floats. Mathish mode defines zero divided by zero as zero for float64 and complex128.
    divide(x, y) = rustymath::Divide => rustymath::MathishDivide;
    /// x^y. An integer power that doesn't fit its dtype wraps, as NumPy's does. A negative integer y beside an x other than 1 or -1 raises ValueError.
    power(x, y) = rustymath::Pow, wrapping true;
    #[doc = rustymath::function_doc!(root)]
    root(n, y) = rustymath::Root;
    #[doc = rustymath::function_doc!(hypot)]
    hypot(x, y) = rustymath::Hypot;
    #[doc = rustymath::function_doc!(atan2)]
    arctan2(y, x) = rustymath::Atan2;
    #[doc = rustymath::function_doc!(binom)]
    /// An integer result raises OverflowError where it doesn't fit its dtype, and ValueError for a negative k.
    binom(n, k) = rustymath::Binom;
    #[doc = rustymath::function_doc!(binom_dn)]
    binom_dn(n, k) = rustymath::BinomDn, only REAL;
    #[doc = rustymath::function_doc!(binom_dk)]
    binom_dk(n, k) = rustymath::BinomDk, only REAL;
    /// The larger of x and y, or y where either is NaN. Characters compare by code point, and a number is below a character.
    maximum(x, y) = reduce::Maximum;
    /// The smaller of x and y, or y where either is NaN. Characters compare by code point, and a number is below a character.
    minimum(x, y) = reduce::Minimum;
    /// Boolean and.
    logical_and(x, y) = reduce::Minimum, only &[DType::Boolean];
    /// Boolean or.
    logical_or(x, y) = reduce::Maximum, only &[DType::Boolean];
    /// Boolean exclusive or.
    logical_xor(x, y) = reduce::Xor;
    /// Boolean equivalence.
    logical_xnor(x, y) = reduce::Xnor;
    #[doc = rustymath::function_doc!(beta)]
    beta(a, b) = rustymath::Beta;
    #[doc = rustymath::function_doc!(gammainc)]
    gammainc(a, x) = rustymath::Gammainc;
    #[doc = rustymath::function_doc!(gammaincc)]
    gammaincc(a, x) = rustymath::Gammaincc;
    #[doc = rustymath::function_doc!(gammaincinv)]
    gammaincinv(a, p) = rustymath::Gammaincinv;
    #[doc = rustymath::function_doc!(betainc)]
    betainc(a, b, x) = rustymath::Betainc;
    #[doc = rustymath::function_doc!(betaincinv)]
    betaincinv(a, b, p) = rustymath::Betaincinv;
    /// Whether x equals y. Mathish mode compares floats within relative tolerance 1e-14. Integers compare exactly, and characters by code point. A number never equals a character.
    equal(x, y; ct) = Equal(ct);
    /// Whether x doesn't equal y, with floats compared under Mathish mode.
    not_equal(x, y; ct) = NotEqual(ct);
    /// Whether x is below y and doesn't equal it, with floats compared under Mathish mode. Characters compare by code point, and a number is below a character.
    less(x, y; ct) = Less(ct);
    /// Whether x is below y or equals it, with floats compared under Mathish mode. Characters compare by code point, and a number is below a character.
    less_equal(x, y; ct) = LessEqual(ct);
    /// Whether x is above y and doesn't equal it, with floats compared under Mathish mode. Characters compare by code point, and a number is below a character.
    greater(x, y; ct) = Greater(ct);
    /// Whether x is above y or equals it, with floats compared under Mathish mode. Characters compare by code point, and a number is below a character.
    greater_equal(x, y; ct) = GreaterEqual(ct);
    /// The largest whole number at most x. Mathish mode rounds a float within relative tolerance 1e-14 of the next whole number to it. Integers and Booleans keep their values.
    floor(x; ct) = Floor(ct);
    /// The smallest whole number at least x. Mathish mode rounds a float within relative tolerance 1e-14 of the whole number below to it. Integers and Booleans keep their values.
    ceil(x; ct) = Ceil(ct);
    /// The remainder of x divided by y. A zero y gives x, by APL's rule. Floats use the floored quotient and native fused multiply-add where available. Mathish mode snaps the remainder to zero when the product equals x within tolerance.
    remainder(x, y; ct) = Flipped(Residue(ct));
    /// The floor of rounded x / y. Mathish mode uses tolerant flooring and defines zero divided by zero as zero for float64. Integer zero divisors and overflowing quotients fail.
    floor_divide(x, y; ct) = FloorDivide(ct) => MathishFloorDivide(ct);
    /// The greatest common divisor of x and y, never negative for integers and floats. Float and complex remainder steps compare under Mathish mode.
    gcd(x, y; ct) = Gcd(ct);
    /// The least common multiple of x and y, with the sign of x y. Float remainder steps compare under Mathish mode.
    lcm(x, y; ct) = Lcm(ct);
}

/// Each reduction's fold or scan as a Python function of an array and its axes, with `only` limiting its dtypes.
macro_rules! reductions {
    ($($(#[doc = $doc:literal])+ $name:ident = $how:ident $r:expr $(, only $only:expr)?;)+) => {
        $(
            $(#[doc = $doc])+
            #[pyfunction]
            #[pyo3(signature = (x, axis=None))]
            pub(crate) fn $name<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<reductions!(@axis $how)>) -> PyResult<Bound<'py, PyAny>> {
                let r = $r;
                reductions!(@$how py, stringify!($name), x, axis, r $(, $only)?)
            }
        )+
        /// Adds the reductions and scans to `m`.
        fn add_reductions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
            Ok(())
        }
    };
    (@Fold $py:ident, $name:expr, $x:ident, $axis:ident, $r:ident $(, $only:expr)?) => { fold($py, $name, &$r, $x, $axis, None$(.or(Some($only)))?) };
    (@Scan $py:ident, $name:expr, $x:ident, $axis:ident, $r:ident $(, $only:expr)?) => { scan($py, $name, $r, $x, $axis, None$(.or(Some($only)))?) };
    (@axis Fold) => { Axes };
    (@axis Scan) => { Axis };
}

reductions! {
    /// The sum of x over `axis`: every axis for None, or one axis, by position or name, or a tuple of them. Integers and Booleans sum as int64, and float16 as float32. An integer sum that doesn't fit int64 wraps.
    sum = Fold rustymath::Add;
    /// The product of x over `axis`, which `sum` describes. Integers and Booleans multiply as int64, and float16 as float32. An integer product that doesn't fit int64 wraps.
    prod = Fold rustymath::Multiply;
    /// The largest item of x over `axis`, which `sum` describes. An empty lane gives the dtype's least value. Beside NaN, a lane gives NaN or its greatest other item, depending on the processor and the lane's length.
    max = Fold reduce::Maximum;
    /// The smallest item of x over `axis`, which `sum` describes. An empty lane gives the dtype's greatest value. Beside NaN, a lane gives NaN or its least other item, depending on the processor and the lane's length.
    min = Fold reduce::Minimum;
    /// Whether every Boolean of x over `axis` is true, with `axis` as `sum` describes.
    all = Fold reduce::Minimum, only &[DType::Boolean];
    /// Whether any Boolean of x over `axis` is true, with `axis` as `sum` describes.
    any = Fold reduce::Maximum, only &[DType::Boolean];
    /// The running sums of x along `axis`, by position or name, or along the flattened x for None. Integers and Booleans sum as int64, and float16 as float32. An integer sum that doesn't fit int64 wraps.
    cumsum = Scan rustymath::Add;
    /// The running products of x along `axis`, as `cumsum` takes it. Integers and Booleans multiply as int64, and float16 as float32. An integer product that doesn't fit int64 wraps.
    cumprod = Scan rustymath::Multiply;
}

/// The smallest and largest items of `x` over `axis`, as `sum` takes it.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn minmax<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let x = one(x)?;
    let view = x.view();
    let computed = released(py, || -> PyResult<Option<[Core; 2]>> {
        if axis.is_none() && let Some(items) = view.contiguous() && let Some(ends) = extremes(items) {
            return Ok(Some(ends.map(|b| Core::new(Layout::EMPTY, b).expect("one item has a rank-0 layout"))));
        }
        let lo = totals("minmax", &reduce::Minimum, view.clone(), axis.as_ref(), None)?;
        let hi = totals("minmax", &reduce::Maximum, view, axis.as_ref(), None)?;
        Ok(lo.zip(hi).map(Into::into))
    })?;
    let [lo, hi] = match computed { Some(ends) => ends, None => { let core = x.into_core()?; [core.clone(), core] } };
    let scalar = lo.layout().shape().is_empty();
    Ok((give(py, lo, scalar)?, give(py, hi, scalar)?))
}

/// The least and the greatest item of `x`, in one pass, where its items are Booleans or integers. Floats take less time in two folds, because each runs `fmin` or `fmax`.
fn extremes(x: View<'_>) -> Option<[Buffer; 2]> {
    fn ends<T: Element>(x: &[T]) -> [Buffer; 2] where reduce::Min: reduce::Reduction<T, Total = T>, reduce::Max: reduce::Reduction<T, Total = T> {
        let (lo, hi) = reduce::extremes(x);
        [lo, hi].map(|e| T::buffer(vec![e]))
    }
    match x { View::Booleans(b) => Some(ends(b)), View::Integers(i) => Some(with_ints!(i, |v| ends(v))), _ => None }
}

/// The running results along `axis` of the Boolean function that `table` gives as table[running_result][next_item], starting from `start` where it's given. With no axis, the scan runs along the flattened x.
#[pyfunction]
#[pyo3(signature = (x, table, axis=None, start=None))]
fn logical_scan<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, table: [[bool; 2]; 2], axis: Option<Axis>, start: Option<bool>) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let input = x.view();
    if input.dtype() != DType::Boolean { return Err(missing("logical_scan", input.dtype())); }
    let (dims, layout, _) = along_axis(x.layout(), input.len(), axis)?;
    let code = table.into_iter().flatten().enumerate().fold(0, |code, (i, bit)| code | (u8::from(bit) << i));
    let kernel = rustymath::operation::Kernel::new::<_, 2>(rustymath::Bitwise(code), rustymath::facts::Variant::checked(DType::Boolean));
    let start = start.map(|s| Buffer::Booleans(vec![s]));
    let start = start.as_ref().map(|s| ArrayView::compact(&Layout::EMPTY, s.view()));
    let seed = start.as_ref().map_or(rustymath::operation::Seed::Identity, |s| rustymath::operation::Seed::Initial(s.read()));
    let mut buffer = Buffer::with_capacity(DType::Boolean, input.len());
    released(py, || kernel.scan(dims, seed, rustymath::operation::Direction::Left, rustymath::operation::Settings::default(), input.read(), &mut buffer))
        .map_err(py_err)?;
    give_items(py, layout, buffer, false)
}

#[pyfunction]
#[pyo3(signature = (function, x, axis=None, start=None))]
fn _accumulate<'py>(
    py: Python<'py>, function: NativeFunction<'_>, x: &Bound<'py, PyAny>, axis: Option<Axis>, start: Option<&Bound<'py, PyAny>>
) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[Some(x), start].into_iter().flatten().collect::<Vec<_>>())?;
    let x = &call.args[0];
    let start = call.args.get(1).map(|s| s.view());
    if start.as_ref().is_some_and(|s| !s.layout().shape().is_empty()) { return Err(PyValueError::new_err("start must be a scalar")); }
    let view = x.view();
    let (dims, layout, _) = along_axis(x.layout(), view.len(), axis)?;
    let (function, mode) = function;
    let kernel = (native_binary(function)?.kernel)(view.dtype(), mode)?;
    let seed = start.as_ref().map_or(rustymath::operation::Seed::Identity, |s| rustymath::operation::Seed::Initial(s.read()));
    let mut buffer = Buffer::with_capacity(kernel.scan_dtype(), view.len());
    released(py, || kernel.scan(dims, seed, rustymath::operation::Direction::Left, rustymath::operation::Settings::default(), view.read(), &mut buffer))
        .map_err(py_err)?;
    give_items(py, layout, buffer, false)
}

/// The slopes of the native elementwise `function` at `args`, where it gave `value`: one array for a function of one argument, and a tuple of its partial derivatives for a function of two. Each comes from rustymath's recipe for the function ([`Slope`](rustymath::derivative::Slope)), which reads `value` where its formula needs the result, as tanh's slope is 1 - tanh².
#[pyfunction]
#[pyo3(signature = (function, value, *args))]
fn _slopes<'py>(
    py: Python<'py>, function: NativeFunction<'_>, value: &Bound<'py, PyAny>, args: &Bound<'py, pyo3::types::PyTuple>) -> PyResult<Bound<'py, PyAny>> {
    let (name, mathish) = function;
    let slopes = NATIVE_SLOPES.iter().find(|(n, _)| *n == name).map(|(_, s)| s)
        .ok_or_else(|| PyValueError::new_err(format!("{name:?} isn't a native elementwise function")))?;
    let mut slopes = slopes(py, &args.iter().collect::<Vec<_>>(), value, mathish)?;
    if slopes.len() == 1 { return Ok(slopes.remove(0)); }
    Ok(pyo3::types::PyTuple::new(py, slopes)?.into_any())
}

/// The folds by the native binary `function` of each window of `width` neighbouring items of `x` along `axis`, by position or name. `function` must have an associative reduction. A lane of n items gives n - width + 1 folds, each keyed as the item that starts its window.
#[pyfunction]
#[pyo3(signature = (function, x, width, axis=Axis::Position(-1)))]
fn _sliding<'py>(py: Python<'py>, function: NativeFunction<'_>, x: &Bound<'py, PyAny>, width: usize, axis: Axis) -> PyResult<Bound<'py, PyAny>> {
    let (function, mode) = function;
    let x = one(x)?;
    let view = x.view();
    let (lanes, layout) = windowed(&x.layout(), width, &axis)?;
    let kernel = (native_binary(function)?.kernel)(view.dtype(), mode)?;
    if kernel.fold().is_none() { return Err(PyValueError::new_err(format!("{function:?} has no associative reduction to slide"))); }
    let flat = Layout::from(lanes.dims());
    let windows = rustymath::structure::Windows::new(&[1, width as i64], &[1, 1], |_| None, flat.shape()).map_err(py_err)?;
    let plan = windows.plan(&flat, flat.shape()).map_err(py_err)?;
    let input = view.remapped(rustymath::agreement::Mapping::from(plan.steps), plan.layout.count());
    let dims = [lanes.outer * (lanes.len + 1 - width), width, lanes.inner];
    let mut buffer = Buffer::with_capacity(kernel.fold_dtype(view.dtype(), width, None), layout.count());
    released(py, || {
        kernel.fold_read(
            &dims, 1, rustymath::operation::Seed::Identity, rustymath::operation::Direction::Right, rustymath::operation::Settings::default(), input.read(),
            &mut buffer)
    }).map_err(py_err)?;
    give_items(py, layout, buffer, false)
}

/// The position along `axis` of the first maximum of each window of `width` neighbouring items of `x`, or of the first minimum where `max` is false, as int64.
#[pyfunction]
fn _sliding_pick<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, width: usize, axis: Axis, max: bool) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let items = x.row_major()?;
    let (lanes, layout) = windowed(&x.layout(), width, &axis)?;
    let rows = released(py, || reduce::view_picks(items.view(), lanes.dims(), reduce::Picks::Windows(width), max))
        .ok_or_else(|| missing(if max { "sliding_argmax" } else { "sliding_argmin" }, items.view().dtype()))?;
    give_items(py, layout, i64::buffer(rows.into_iter().map(|r| r as i64).collect()), false)
}

/// `a`, with `b` broadcast over the selection that `index` makes, as brackets take it, and combined by `function` into each item it selects, in turn. A target selected more than once combines every item that selects it. `function` is add, subtract, multiply, divide, maximum or minimum. A failed integer result follows [`fitted`]'s rule.
#[pyfunction]
fn _at<'py>(
    py: Python<'py>, function: NativeFunction<'_>, a: &Bound<'py, PyAny>, index: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (function, mode) = function;
    if !matches!(function, "add" | "subtract" | "multiply" | "divide" | "maximum" | "minimum") {
        return Err(PyValueError::new_err(format!("at takes add, subtract, multiply, divide, maximum or minimum, not {function}")));
    }
    let mut args = convert(&[a, b])?.args.into_iter();
    let (a, b) = (args.next().expect("two arguments give two"), args.next().expect("two arguments give two"));
    let kernel = (native_binary(function)?.kernel)(a.view().dtype(), mode)?;
    let mut out = if a.view().dtype() == kernel.result_dtype() { a.into_core()? } else {
        let items = a.at(kernel.result_dtype())?;
        let buffer = match items { Items::Owned(b) => b, Items::Borrowed(v) => Buffer::copied(v, 0..v.len()) };
        Core::new(a.layout(), buffer).map_err(py_err)?
    };
    let plan = crate::structure::selection(out.layout(), index, true)?;
    let b = b.into_core()?;
    released(py, || out.combine_at(&kernel, &plan, &b)).map_err(py_err)?;
    give(py, out, false)
}

/// The lanes along `axis` of an array of `layout`, for windows of `width` items, and the layout of the results, one for each window. A ValueError where a window doesn't fit the axis.
fn windowed(layout: &Layout, width: usize, axis: &Axis) -> PyResult<(Lanes, Layout)> {
    let a = axis.of(layout)?;
    let lanes = Lanes::new(layout.shape(), a).map_err(py_err)?;
    if !(1..=lanes.len).contains(&width) { return Err(PyValueError::new_err(format!("a window of {width} items doesn't fit an axis of {}", lanes.len))); }
    let starts = rustymath::structure::select_axis(layout, a, rustymath::structure::Selector::Range { start: 0, len: lanes.len + 1 - width, step: 1 })
        .map_err(py_err)?;
    Ok((lanes, starts.layout))
}

enum Summary { Mean, Variance(usize), Std(usize), Norm(f64) }

/// Shared conversion, axis preparation and wrapping for statistical reductions. Numerical loops live in `rustymath::stats`.
fn summary<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>, how: Summary) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let layout = x.layout();
    let axes = stats::Axes::new(&layout, &axes_of(axis.as_ref(), &layout)?).map_err(py_err)?;
    let compute = x.view().dtype().inexact(FloatWidth::F64).ok_or_else(|| missing("statistics", x.view().dtype()))?.0;
    let items = x.at(compute)?;
    let out = released(py, || {
        rustymath::with_inexact!(compute, T => {
            let data = axes.read(items.of::<T>());
            match how {
                Summary::Mean => T::buffer(stats::mean(&data, axes.dims)),
                Summary::Variance(ddof) => <T as Scalar>::Real::buffer(stats::variance(&data, axes.dims, ddof)),
                Summary::Std(ddof) => <T as Scalar>::Real::buffer(stats::stddev(&data, axes.dims, ddof)),
                Summary::Norm(p) => <T as Scalar>::Real::buffer(stats::norm(&data, axes.dims, p)),
            }
        })
    });
    let scalar = axes.layout.shape().is_empty();
    give_items(py, axes.layout, out, scalar)
}

/// Arithmetic mean over every axis, or the axes given by positions or names. Float16 accumulates and returns float32.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
pub(crate) fn mean<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>) -> PyResult<Bound<'py, PyAny>> { summary(py, x, axis, Summary::Mean) }

/// Two-pass variance, real for complex data, with denominator item count minus ddof. A nonpositive denominator gives NaN.
#[pyfunction]
#[pyo3(signature = (x, axis=None, ddof=0))]
pub(crate) fn var<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>, ddof: usize) -> PyResult<Bound<'py, PyAny>> {
    summary(py, x, axis, Summary::Variance(ddof))
}

/// The square root of var, over the same axes and with the same ddof.
#[pyfunction]
#[pyo3(signature = (x, axis=None, ddof=0))]
pub(crate) fn std<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>, ddof: usize) -> PyResult<Bound<'py, PyAny>> {
    summary(py, x, axis, Summary::Std(ddof))
}

/// Elementwise p-norm over the selected axes. ord=0 counts nonzeros; ±inf gives the greatest/least magnitude. This is not a matrix-induced norm.
#[pyfunction]
#[pyo3(signature = (x, axis=None, ord=2.0))]
pub(crate) fn norm<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>, ord: f64) -> PyResult<Bound<'py, PyAny>> {
    if ord.is_nan() { return Err(PyValueError::new_err("norm's ord cannot be NaN")); }
    summary(py, x, axis, Summary::Norm(ord))
}

/// Weighted average over the selected axes. Weights broadcast by the ordinary leading-axis/keyed-axis rules. Without weights, this is mean.
#[pyfunction]
#[pyo3(signature = (x, axis=None, weights=None))]
pub(crate) fn average<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>, weights: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let Some(weights) = weights else { return mean(py, x, axis) };
    let call = convert(&[x, weights])?;
    let (x, w) = (&call.args[0], &call.args[1]);
    let agreement = Agreement::new(&x.layout(), &w.layout()).map_err(py_err)?;
    let axes = stats::Axes::new(&agreement.layout, &axes_of(axis.as_ref(), &agreement.layout)?).map_err(py_err)?;
    let compute = x.view().dtype().inexact(FloatWidth::F64).ok_or_else(|| missing("average", x.view().dtype()))?.0;
    let (x, w) = (x.at(compute)?, w.at(compute)?);
    let out = released(
        py, || rustymath::with_inexact!(compute, T => T::buffer(stats::average(x.of::<T>(), w.of::<T>(), &axes, [&agreement.left, &agreement.right]))));
    let scalar = axes.layout.shape().is_empty();
    give_items(py, axes.layout, out, scalar)
}

fn covariance<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, rowvar: bool, ddof: usize, correlation: bool) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let layout = x.layout();
    let (size, rowvar, output) = match *layout.shape() {
        [] => ([1, 1], true, Layout::EMPTY),
        [n] => ([1, n], true, Layout::EMPTY),
        [rows, cols] => {
            let variables = layout.axes([usize::from(!rowvar)].into_iter());
            ([rows, cols], rowvar, variables.concat(&variables).unique_names())
        }
        _ => return Err(PyValueError::new_err("covariance takes a vector or a matrix")),
    };
    let compute = x.view().dtype().inexact(FloatWidth::F64).ok_or_else(|| missing("covariance", x.view().dtype()))?.0;
    let items = x.at(compute)?;
    let out = released(py, || rustymath::with_inexact!(compute, T => T::buffer(stats::covariance(items.of::<T>(), size, rowvar, ddof, correlation))));
    let scalar = output.shape().is_empty();
    give_items(py, output, out, scalar)
}

/// Covariance of variables in the rows of a matrix, or columns with rowvar=False. A vector gives its variance.
#[pyfunction]
#[pyo3(signature = (x, rowvar=true, ddof=1))]
pub(crate) fn cov<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, rowvar: bool, ddof: usize) -> PyResult<Bound<'py, PyAny>> {
    covariance(py, x, rowvar, ddof, false)
}

/// Correlation of variables in the rows of a matrix, or columns with rowvar=False. Constant variables give NaN.
#[pyfunction]
#[pyo3(signature = (x, rowvar=true))]
pub(crate) fn corrcoef<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, rowvar: bool) -> PyResult<Bound<'py, PyAny>> { covariance(py, x, rowvar, 0, true) }

/// Histogram counts and edges. bins is a positive count (10 by default), or strictly increasing edges. `range` sets the span of a bin count. Explicit edges take no `range`. The final bin includes its upper edge; out-of-range values and NaNs are ignored.
#[pyfunction]
#[pyo3(signature = (x, bins=None, range=None))]
pub(crate) fn histogram<'py>(
    py: Python<'py>, x: &Bound<'py, PyAny>, bins: Option<&Bound<'py, PyAny>>, range: Option<(f64, f64)>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let count = bins.map_or(Ok(10), |b| b.extract::<usize>());
    let edges = if count.is_err() { bins } else { None };
    if edges.is_some() && range.is_some() { return Err(PyValueError::new_err("histogram's range applies to a bin count, not to explicit edges")); }
    let call = convert(&[Some(x), edges].into_iter().flatten().collect::<Vec<_>>())?;
    let dtype = call.args[0].view().dtype();
    let compute = dtype.inexact(FloatWidth::F64).filter(|(d, _)| !matches!(d, DType::Complex(_))).ok_or_else(|| missing("histogram", dtype))?.0;
    let x = call.args[0].at(compute)?;
    let explicit = call.args.get(1).map(|a| a.at(compute)).transpose()?;
    let (counts, edges) = released(py, || -> PyResult<_> {
        Ok(rustymath::with_float_width!(kernel match compute { DType::Float(w) => w, _ => unreachable!() }, F => Some({
            let data = F::of(x.view()).expect("histogram computes at a float width");
            let edges = match explicit.as_ref() {
                Some(e) => e.of::<F>().to_vec(),
                None => stats::histogram_edges(data, count.unwrap_or(10), range).map_err(py_err)?,
            };
            (stats::histogram(data, &edges).map_err(py_err)?, F::buffer(edges))
        })).expect("histogram computes in float32 or float64"))
    })?;
    let counts = give_items(py, Layout::from([counts.len()]), i64::buffer(counts), false)?;
    let edges = give_items(py, Layout::from([edges.len()]), edges, false)?;
    Ok((counts, edges))
}

/// Decimal rounding to nearest, with ties to even. Negative precisions round to tens, hundreds and so on.
#[pyfunction]
#[pyo3(signature = (x, decimals=0))]
pub(crate) fn round<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, decimals: i32) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[x])?;
    let kernel = select::<_, 1>("round", rustymath::Round::new(decimals), call_dtype(&call), [call_dtype(&call); 1], false, false, None)?;
    apply(py, &kernel, &call)
}

/// Adds the elementwise functions, reductions and scans to `m`.
pub fn add_to(m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_elementwise(m)?;
    add_reductions(m)?;
    m.add_function(wrap_pyfunction!(minmax, m)?)?;
    m.add_function(wrap_pyfunction!(logical_scan, m)?)?;
    m.add_function(wrap_pyfunction!(_accumulate, m)?)?;
    m.add_function(wrap_pyfunction!(_slopes, m)?)?;
    m.add_function(wrap_pyfunction!(_sliding, m)?)?;
    m.add_function(wrap_pyfunction!(_sliding_pick, m)?)?;
    m.add_function(wrap_pyfunction!(_at, m)?)?;
    m.add_function(wrap_pyfunction!(mean, m)?)?;
    m.add_function(wrap_pyfunction!(average, m)?)?;
    m.add_function(wrap_pyfunction!(var, m)?)?;
    m.add_function(wrap_pyfunction!(self::std, m)?)?;
    m.add_function(wrap_pyfunction!(norm, m)?)?;
    m.add_function(wrap_pyfunction!(cov, m)?)?;
    m.add_function(wrap_pyfunction!(corrcoef, m)?)?;
    m.add_function(wrap_pyfunction!(histogram, m)?)?;
    m.add_function(wrap_pyfunction!(round, m)?)
}
