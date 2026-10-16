//! rustymath's elementwise functions, reductions and scans as Python functions, and the steps that every Python function of arrays shares:
//!
//! 1. [`convert`] turns the data arguments into views of one dtype, and says whether every one was a Python scalar.
//! 2. The function runs on the views. An elementwise function runs its function object through [`Apply`], at the dtype that its arguments compute in, as [`rustymath::operation`] gives it. Two or three arguments pair their axes as rustymath broadcasts them: axes with the same name first, then the rest from the front.
//! 3. [`give`] returns the result as a rustymath array, or as a Python number where every argument was a Python scalar.
//!
//! [`fitted`] raises `OverflowError` for integer results that don't fit their dtype, and [`py_err`] turns any other rustymath error into a Python exception. Every step holds the GIL, so no Python code can change a borrowed buffer while a function reads it.
use crate::array::give;
use crate::convert::{convert, one};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustymath::agreement::{Agreement, Mapping};
use rustymath::array::Array as Core;
use rustymath::buffer::{Buffer, ComplexWidth, DType, Element, FloatWidth, Source, View};
use rustymath::error::{Error, Kind};
use rustymath::layout::{position, Lanes, Layout};
use rustymath::operation::{BinaryKernel, FoldKernel, Kernels, ScanKernel, TernaryKernel, UnaryKernel};
use rustymath::tolerant::{Ceil, Equal, Floor, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, NotEqual, Residue, Tolerance};
use rustymath::{Flipped, kernels, reduce, with_ints};
use rustymath_buffer::{name, py_err};

/// The error for a function with no kernel for `dtype`.
pub fn missing(function: &str, dtype: DType) -> PyErr { py_err(Error::new(Kind::Domain, format!("{function} doesn't take {}", name(dtype)))) }

/// The results of a kernel, which gives them with whether every one succeeded. A failed integer result is an Overflow error. A float or complex result fails only where it's NaN, and stays NaN.
pub fn fitted(function: &str, (buffer, ok): (Buffer, bool)) -> PyResult<Buffer> {
    if ok || !matches!(buffer.dtype(), DType::Integer(_)) { return Ok(buffer); }
    Err(py_err(Error::new(Kind::Overflow, format!("an integer result of {function} doesn't fit {}", name(buffer.dtype())))))
}

/// Whether `x` holds a negative integer.
fn negative(x: View<'_>) -> bool {
    match x { View::Integers(ints) => with_ints!(ints, |v| v.iter().any(|&i| Source::<i64>::read(i) < 0)), _ => false }
}

/// The tolerance `ct`, or `Tolerance::DEFAULT` for `None`. A ValueError outside 0 to below 1.
pub(crate) fn tolerance(ct: Option<f64>) -> PyResult<Tolerance> {
    ct.map_or(Ok(Tolerance::DEFAULT), |c| Tolerance::new(c).ok_or_else(|| PyValueError::new_err(format!("ct must be at least 0 and below 1, not {c}"))))
}

/// An axis, given as a position, negative from the end, or as an axis name.
#[derive(FromPyObject)]
pub enum Axis {
    #[pyo3(annotation = "int")]
    Position(i64),
    #[pyo3(annotation = "str")]
    Name(String),
}

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
pub enum Axes {
    #[pyo3(annotation = "int | str")]
    One(Axis),
    #[pyo3(annotation = "tuple")]
    Many(Vec<Axis>),
}

/// Running a function object's kernels on a call's Python arguments, for each arity.
pub trait Apply<K> {
    fn apply<'py>(&self, py: Python<'py>, function: &str, k: &K, args: &[&Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>>;
}

impl<K> Apply<K> for Kernels<UnaryKernel<K>> {
    fn apply<'py>(&self, py: Python<'py>, function: &str, k: &K, args: &[&Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> {
        let call = convert(args)?;
        let (x, layout) = (call.args[0].view(), call.args[0].layout());
        let result = self.unary(k, x, &Mapping::Linear(1), x.len()).ok_or_else(|| missing(function, x.dtype()))?;
        give(py, Core::new(layout, fitted(function, result)?).map_err(py_err)?, call.scalar())
    }
}

impl<K> Apply<K> for Kernels<BinaryKernel<K>> {
    fn apply<'py>(&self, py: Python<'py>, function: &str, k: &K, args: &[&Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> { binary(self, py, function, k, args, None) }
}

/// [`Apply`] for two arguments. `domain` is the message of a Domain error for a failed integer result beside a negative second argument, as a negative exponent of an integer power gives.
fn binary<'py, K>(kernels: &Kernels<BinaryKernel<K>>, py: Python<'py>, function: &str, k: &K, args: &[&Bound<'py, PyAny>], domain: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(args)?;
    let (x, y) = (call.args[0].view(), call.args[1].view());
    let agreement = Agreement::new(&call.args[0].layout(), &call.args[1].layout()).map_err(py_err)?;
    let (buffer, ok) = kernels.binary(k, x, y, &agreement).ok_or_else(|| missing(function, x.dtype()))?;
    if let Some(message) = domain.filter(|_| !ok && negative(y)) { return Err(py_err(Error::new(Kind::Domain, message.to_owned()))); }
    give(py, Core::new(agreement.layout, fitted(function, (buffer, ok))?).map_err(py_err)?, call.scalar())
}

impl<K> Apply<K> for Kernels<TernaryKernel<K>> {
    fn apply<'py>(&self, py: Python<'py>, function: &str, k: &K, args: &[&Bound<'py, PyAny>]) -> PyResult<Bound<'py, PyAny>> {
        let call = convert(args)?;
        let views = [0, 1, 2].map(|i| call.args[i].view());
        let first = Agreement::new(&call.args[0].layout(), &call.args[1].layout()).map_err(py_err)?;
        let second = Agreement::new(&first.layout, &call.args[2].layout()).map_err(py_err)?;
        let result = self.ternary(k, views, &first, &second).ok_or_else(|| missing(function, views[0].dtype()))?;
        give(py, Core::new(second.layout, fitted(function, result)?).map_err(py_err)?, call.scalar())
    }
}

/// `r`'s totals of `x` over the axes that `axis` names: every axis for `None`, or one axis or a tuple of them. A total over every axis is a Python number.
fn fold<'py, R>(py: Python<'py>, function: &str, kernels: &Kernels<FoldKernel<R>>, r: &R, x: &Bound<'py, PyAny>, axis: Option<Axes>) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let totals = totals(function, kernels, r, x.view(), x.layout(), axis.as_ref())?;
    let core = match totals { Some(core) => core, None => x.into_core()? };
    let scalar = core.layout().shape().is_empty();
    give(py, core, scalar)
}

/// `r`'s totals of `view`, which has `layout`, over the axes that `axis` names, as [`fold`] takes them. `None` where `axis` is an empty tuple, which leaves the argument as it is.
fn totals<R>(function: &str, kernels: &Kernels<FoldKernel<R>>, r: &R, view: View<'_>, mut layout: Layout, axis: Option<&Axes>) -> PyResult<Option<Core>> {
    let mut steps: Vec<Option<usize>> = match axis {
        None => vec![None],
        Some(Axes::One(a)) => vec![Some(a.of(&layout)?)],
        Some(Axes::Many(axes)) => axes.iter().map(|a| a.of(&layout).map(Some)).collect::<PyResult<_>>()?,
    };
    // The last axis first, so that each later position still names its axis.
    steps.sort_unstable_by(|a, b| b.cmp(a));
    steps.dedup();
    let mut buffer: Option<Buffer> = None;
    for step in steps {
        let view = buffer.as_ref().map_or(view, Buffer::view);
        let (along, next) = match step {
            None => ([1, view.len(), 1], Layout::EMPTY),
            Some(a) => (Lanes::new(layout.shape(), a).map_err(py_err)?.dims(), layout.axes((0..layout.shape().len()).filter(|&i| i != a))),
        };
        let result = kernels.fold(r, view, along).ok_or_else(|| missing(function, view.dtype()))?;
        buffer = Some(fitted(function, result)?);
        layout = next;
    }
    buffer.map(|b| Core::new(layout, b).map_err(py_err)).transpose()
}

/// The lanes along `axis` of an argument with `layout` and `len` items, as rustymath's axis functions take them, and the result's layout: `layout` itself, or the flattened argument for `None`.
pub(crate) fn along_axis(layout: Layout, len: usize, axis: Option<Axis>) -> PyResult<([usize; 3], Layout)> {
    match axis {
        None => Ok(([1, len, 1], Layout::from([len].as_slice()))),
        Some(a) => Ok((Lanes::new(layout.shape(), a.of(&layout)?).map_err(py_err)?.dims(), layout)),
    }
}

/// `r`'s running totals of `x` along `axis`, or along the flattened `x` for `None`.
fn scan<'py, R>(py: Python<'py>, function: &str, kernels: &Kernels<ScanKernel<R>>, r: &R, x: &Bound<'py, PyAny>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[x])?;
    let x = call.args[0].view();
    let (along, layout) = along_axis(call.args[0].layout(), x.len(), axis)?;
    let result = kernels.scan(r, x, along).ok_or_else(|| missing(function, x.dtype()))?;
    give(py, Core::new(layout, fitted(function, result)?).map_err(py_err)?, false)
}

/// Each function object as a Python function, in one entry: its name, its arguments, with `; ct` for a tolerance keyword, and the object, which `ct` can make. `only` limits its kernels to the dtypes it lists. With two arguments, `negative` gives the message that [`binary`] raises for a failed integer result beside a negative second argument.
macro_rules! functions {
    ($($(#[doc = $doc:literal])+ $name:ident($($x:ident),+ $(; $ct:ident)?) = $k:expr $(, only $only:expr)? $(, negative $negative:literal)?;)+) => {
        $(
            $(#[doc = $doc])+
            #[pyfunction]
            #[pyo3(signature = ($($x),+ $(, $ct = None)?))]
            pub(crate) fn $name<'py>(py: Python<'py>, $($x: &Bound<'py, PyAny>,)+ $($ct: Option<f64>)?) -> PyResult<Bound<'py, PyAny>> {
                $(let $ct = tolerance($ct)?;)?
                let k = $k;
                let kernels = functions!(@kernels [$($x),+] k)$(.only($only))?;
                functions!(@apply kernels, py, $name, k, [$($x),+] $(, $negative)?)
            }
        )+
        /// Adds the elementwise functions to `m`.
        fn add_elementwise(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
            Ok(())
        }
    };
    (@apply $kernels:ident, $py:ident, $name:ident, $k:ident, $args:tt) => { $kernels.apply($py, stringify!($name), &$k, &$args) };
    (@apply $kernels:ident, $py:ident, $name:ident, $k:ident, $args:tt, $negative:literal) => { binary(&$kernels, $py, stringify!($name), &$k, &$args, Some($negative)) };
    (@kernels [$x:ident] $k:ident) => { kernels!(Unary of $k) };
    (@kernels [$x:ident, $y:ident] $k:ident) => { kernels!(Binary of $k) };
    (@kernels [$x:ident, $y:ident, $z:ident] $k:ident) => { kernels!(Ternary of $k) };
}

/// The real and complex dtypes, for an object whose forms serve two Python functions.
const REAL: &[DType] = &[DType::Float(FloatWidth::F16), DType::Float(FloatWidth::F32), DType::Float(FloatWidth::F64)];
const COMPLEX: &[DType] = &[DType::Complex(ComplexWidth::F32), DType::Complex(ComplexWidth::F64)];

functions! {
    /// e^x.
    exp(x) = rustymath::Exp;
    /// e^x - 1, without the loss of digits near 0 that `exp(x) - 1` has.
    expm1(x) = rustymath::Expm1;
    /// The natural logarithm of x.
    log(x) = rustymath::Ln;
    /// ln(1 + x), without the loss of digits near 0 that `log(1 + x)` has.
    log1p(x) = rustymath::Log1p;
    /// The logarithm of x to base 2.
    log2(x) = rustymath::Log2;
    /// The logarithm of x to base 10.
    log10(x) = rustymath::Log10;
    /// The logarithm of x to the given base, with the base first.
    logb(base, x) = rustymath::Logb;
    /// The square root of x, the principal one for a complex x.
    sqrt(x) = rustymath::Sqrt;
    /// The cube root of x.
    cbrt(x) = rustymath::Cbrt;
    /// sin x.
    sin(x) = rustymath::Sin;
    /// cos x.
    cos(x) = rustymath::Cos;
    /// exp(i x). A real float16 x gives complex64.
    cis(x) = rustymath::Cis;
    /// tan x.
    tan(x) = rustymath::Tan;
    /// asin x, from -π/2 to π/2 for a real x.
    arcsin(x) = rustymath::Asin;
    /// acos x, from 0 to π for a real x.
    arccos(x) = rustymath::Acos;
    /// atan x, from -π/2 to π/2 for a real x.
    arctan(x) = rustymath::Atan;
    /// sinh x.
    sinh(x) = rustymath::Sinh;
    /// cosh x.
    cosh(x) = rustymath::Cosh;
    /// tanh x.
    tanh(x) = rustymath::Tanh;
    /// asinh x.
    arcsinh(x) = rustymath::Asinh;
    /// acosh x.
    arccosh(x) = rustymath::Acosh;
    /// atanh x.
    arctanh(x) = rustymath::Atanh;
    /// |x|, a real number for a complex x. A signed integer type's least value has no magnitude of its type, and raises OverflowError.
    absolute(x) = rustymath::Abs;
    /// -1, 0 or 1 for a real x, and x/|x| for a complex x.
    sign(x) = rustymath::Sign;
    /// The angle of x from the positive real axis, from -π to π.
    angle(x) = rustymath::Angle;
    /// √(1 - x²), the principal root for a complex x.
    sqrt1mx2(x) = rustymath::Sqrt1mx2;
    /// √(1 + x²), the principal root for a complex x.
    sqrt1px2(x) = rustymath::Sqrt1px2;
    /// √(x - 1) √(x + 1): √(x² - 1) with the sign of a real x.
    sqrtx2m1(x) = rustymath::Sqrtx2m1;
    /// Γ(x).
    gamma(x) = rustymath::Gamma;
    /// Γ(x + 1), the floating-point extension of factorial.
    factorial(x) = rustymath::Factorial;
    /// ln |Γ(x)|.
    gammaln(x) = rustymath::Lgamma, only REAL;
    /// The principal branch of ln Γ(z), as SciPy's `loggamma` gives it for complex z. A real z computes as complex.
    loggamma(x) = rustymath::Lgamma, only COMPLEX;
    /// The principal branch W₀ of the Lambert W function. A real x below -1/e gives NaN.
    lambertw(x) = rustymath::LambertW;
    /// The error function, 2/√π times the integral of e^-t² from 0 to x.
    erf(x) = rustymath::Erf;
    /// 1 - erf(x), without the loss of digits for large x that `1 - erf(x)` has.
    erfc(x) = rustymath::Erfc;
    /// The x with erf(x) = p.
    erfinv(p) = rustymath::Erfinv;
    /// The x with erfc(x) = q.
    erfcinv(q) = rustymath::Erfcinv;
    /// x + y.
    add(x, y) = rustymath::Add;
    /// x - y.
    subtract(x, y) = rustymath::Subtract;
    /// x × y.
    multiply(x, y) = rustymath::Multiply;
    /// x / y. Integers compute as floats.
    divide(x, y) = rustymath::Divide;
    /// x^y. An integer power raises OverflowError where it doesn't fit its dtype, and ValueError for a negative y beside an x other than 1 or -1.
    power(x, y) = rustymath::Pow, negative "an integer power with a negative exponent isn't an integer";
    /// The real nth root of y, computed without rounding 1/n first.
    root(n, y) = rustymath::Root;
    /// √(x² + y²), with no overflow or underflow in between.
    hypot(x, y) = rustymath::Hypot;
    /// The angle of the point (x, y) from the positive x axis, from -π to π.
    arctan2(y, x) = rustymath::Atan2;
    /// The binomial coefficient of n and k. An integer result raises OverflowError where it doesn't fit its dtype, and ValueError for a negative k.
    binom(n, k) = rustymath::Binom, negative "an integer binom takes no negative k";
    /// The larger of x and y, or y where either is NaN.
    maximum(x, y) = reduce::Maximum;
    /// The smaller of x and y, or y where either is NaN.
    minimum(x, y) = reduce::Minimum;
    /// Boolean and.
    logical_and(x, y) = reduce::Minimum, only &[DType::Boolean];
    /// Boolean or.
    logical_or(x, y) = reduce::Maximum, only &[DType::Boolean];
    /// Boolean exclusive or.
    logical_xor(x, y) = reduce::Xor;
    /// Boolean equivalence.
    logical_xnor(x, y) = reduce::Xnor;
    /// The regularized lower incomplete gamma function P(a, x).
    gammainc(a, x) = rustymath::Gammainc;
    /// The regularized upper incomplete gamma function Q(a, x) = 1 - P(a, x).
    gammaincc(a, x) = rustymath::Gammaincc;
    /// The x with P(a, x) = p.
    gammaincinv(a, p) = rustymath::Gammaincinv;
    /// The regularized incomplete beta function I_x(a, b).
    betainc(a, b, x) = rustymath::Betainc;
    /// The x with I_x(a, b) = p.
    betaincinv(a, b, p) = rustymath::Betaincinv;
    /// Whether x equals y. Floats are equal when |x - y| is at most `ct` times the larger magnitude, as APL compares them. Integers compare exactly.
    equal(x, y; ct) = Equal(ct);
    /// Whether x doesn't equal y, within the tolerance `ct` for floats.
    not_equal(x, y; ct) = NotEqual(ct);
    /// Whether x is below y and doesn't equal it, within the tolerance `ct` for floats.
    less(x, y; ct) = Less(ct);
    /// Whether x is below y or equals it, within the tolerance `ct` for floats.
    less_equal(x, y; ct) = LessEqual(ct);
    /// Whether x is above y and doesn't equal it, within the tolerance `ct` for floats.
    greater(x, y; ct) = Greater(ct);
    /// Whether x is above y or equals it, within the tolerance `ct` for floats.
    greater_equal(x, y; ct) = GreaterEqual(ct);
    /// The largest whole number at most x. An item within relative tolerance `ct` of the next whole number rounds to it. Integers and Booleans keep their values.
    floor(x; ct) = Floor(ct);
    /// The smallest whole number at least x. An item within relative tolerance `ct` of the whole number below rounds to it. Integers and Booleans keep their values.
    ceil(x; ct) = Ceil(ct);
    /// The remainder of x divided by y, with the sign of y. A zero y gives x, by APL's rule. For floats, a quotient within relative tolerance `ct` of a whole number counts as whole, giving 0.
    remainder(x, y; ct) = Flipped(Residue(ct));
    /// The greatest common divisor of x and y, never negative for integers and floats. Each remainder step of floats and complex numbers uses relative tolerance `ct`.
    gcd(x, y; ct) = Gcd(ct);
    /// The least common multiple of x and y, with the sign of x y. Each remainder step of floats uses relative tolerance `ct`.
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
                reductions!(@$how py, stringify!($name), &kernels!($how of r)$(.only($only))?, &r, x, axis)
            }
        )+
        /// Adds the reductions and scans to `m`.
        fn add_reductions(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
            Ok(())
        }
    };
    (@Fold $($args:tt)*) => { fold($($args)*) };
    (@Scan $($args:tt)*) => { scan($($args)*) };
    (@axis Fold) => { Axes };
    (@axis Scan) => { Axis };
}

reductions! {
    /// The sum of x over `axis`: every axis for None, or one axis, by position or name, or a tuple of them. Integers and Booleans sum as int64, and float16 as float32.
    sum = Fold reduce::Sum;
    /// The product of x over `axis`, which `sum` describes. Integers and Booleans multiply as int64, and float16 as float32.
    prod = Fold reduce::Product;
    /// The largest item of x over `axis`, which `sum` describes. An empty lane gives the dtype's least value.
    max = Fold reduce::Max;
    /// The smallest item of x over `axis`, which `sum` describes. An empty lane gives the dtype's greatest value.
    min = Fold reduce::Min;
    /// Whether every Boolean of x over `axis` is true, with `axis` as `sum` describes.
    all = Fold reduce::Min, only &[DType::Boolean];
    /// Whether any Boolean of x over `axis` is true, with `axis` as `sum` describes.
    any = Fold reduce::Max, only &[DType::Boolean];
    /// The running sums of x along `axis`, by position or name, or along the flattened x for None. Integers and Booleans sum as int64, and float16 as float32.
    cumsum = Scan reduce::Sum;
    /// The running products of x along `axis`, as `cumsum` takes it. Integers and Booleans multiply as int64, and float16 as float32.
    cumprod = Scan reduce::Product;
}

/// The smallest and largest items of `x` over `axis`, as `sum` takes it.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn minmax<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let x = one(x)?;
    let (view, layout) = (x.view(), x.layout());
    let ends = match axis { None => extremes(view), Some(_) => None };
    let [lo, hi] = match ends {
        Some(ends) => ends.map(|b| Core::new(Layout::EMPTY, b).expect("one item has a rank-0 layout")),
        None => {
            let lo = totals("minmax", &kernels!(Fold reduce::Min), &reduce::Min, view, layout.clone(), axis.as_ref())?;
            let hi = totals("minmax", &kernels!(Fold reduce::Max), &reduce::Max, view, layout, axis.as_ref())?;
            match lo.zip(hi) { Some(ends) => ends.into(), None => { let core = x.into_core()?; [core.clone(), core] } }
        }
    };
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
    let call = convert(&[x])?;
    let x = call.args[0].view();
    let View::Booleans(items) = x else { return Err(missing("logical_scan", x.dtype())) };
    let (along, layout) = along_axis(call.args[0].layout(), items.len(), axis)?;
    let mut out = Vec::with_capacity(items.len());
    reduce::Logical(table).scan_along(items, along, start, &mut out);
    give(py, Core::new(layout, Buffer::Booleans(out)).map_err(py_err)?, false)
}

/// Adds the elementwise functions, reductions and scans to `m`.
pub fn add_to(m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_elementwise(m)?;
    add_reductions(m)?;
    m.add_function(wrap_pyfunction!(minmax, m)?)?;
    m.add_function(wrap_pyfunction!(logical_scan, m)?)
}
