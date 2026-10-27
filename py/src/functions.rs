//! rustymath's elementwise functions, reductions and scans as Python functions, and the steps that every Python function of arrays shares:
//!
//! 1. [`convert`] turns the data arguments into views of one dtype, and says whether every one was a Python scalar.
//! 2. The function runs on the views, with the GIL released. An elementwise function runs its function object through [`Apply`], at the dtype that its arguments compute in, as [`rustymath::operation`] gives it. Two or three arguments pair their axes as rustymath broadcasts them: axes with the same name first, then the rest from the front.
//! 3. [`give`] returns the result as a rustymath array, or as a Python number where every argument was a Python scalar.
//!
//! [`checked`] keeps a failed integer result that its kernel wrapped, as NumPy does. It raises `OverflowError` for any other integer result that doesn't fit its dtype. [`py_err`] turns every other rustymath error into a Python exception. Step 2 runs with the GIL released ([`released`](crate::convert::released)), so other Python threads run meanwhile. A Python thread that writes to an argument's items during a call leaves the call's results undefined.
use crate::array::{give, give_items};
use crate::convert::{convert, convert_kinds, one, released, Call};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use rustymath::agreement::{Agreement, Mapping};
use rustymath::array::{Array as Core, ArrayView};
use rustymath::buffer::{Buffer, ComplexWidth, DType, Element, Fill, FloatWidth, Items, View};
use rustymath::error::{Error, Kind};
use rustymath::gather::Steps;
use rustymath::layout::{position, Lanes, Layout};
use rustymath::operation::{AccumulationKernel, BinaryKernel, Kernels, MappedFoldKernel, SlidingKernel, TernaryKernel, UnaryKernel};
use rustymath::linalg::Scalar;
use rustymath::stats;
use rustymath::structure::{walked_source, Plan, Slide};
use rustymath::tolerant::{Ceil, Equal, Floor, FloorDivide, MathishFloorDivide, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, NotEqual, Residue, Tolerance};
use rustymath::{Flipped, kernels, reduce, with_ints, Wrapped};
use rustymath_buffer::py_err;

/// The error for a function with no kernel for `dtype`.
pub fn missing(function: &str, dtype: DType) -> PyErr { py_err(Error::new(Kind::Domain, format!("{function} doesn't take {}", dtype.name()))) }

/// A kernel's results for arguments that compute at `dtype`, which it gives with whether every one succeeded. `None` gives the error for a dtype that the function doesn't take. A failed result follows [`fitted`]'s rule.
fn checked<F: Copy>(function: &str, kernels: &Kernels<F>, dtype: DType, result: Option<(Buffer, bool)>) -> PyResult<Buffer> {
    let (buffer, ok) = result.ok_or_else(|| missing(function, dtype))?;
    fitted(function, kernels, dtype, buffer.dtype(), ok)?;
    Ok(buffer)
}

/// Checks results of `out`'s dtype that computed at `dtype`, where `ok` says whether every one succeeded. A failed integer result keeps its wrapped value where [`Kernels::wraps`] says that the kernel wrote one, as integer arithmetic and reductions do. Any other failed integer result is an Overflow error. A float or complex result fails only where it's NaN, and stays NaN.
fn fitted<F: Copy>(function: &str, kernels: &Kernels<F>, dtype: DType, out: DType, ok: bool) -> PyResult<()> {
    if ok || kernels.wraps(dtype) || !matches!(out, DType::Integer(_)) { return Ok(()); }
    Err(py_err(Error::new(Kind::Overflow, format!("an integer result of {function} doesn't fit {}", out.name()))))
}

pub(crate) fn comparison(mathish: bool) -> Tolerance { if mathish { Tolerance::DEFAULT } else { Tolerance::EXACT } }
pub(crate) fn mathish_at(dtype: Option<DType>) -> bool {
    matches!(dtype, Some(DType::Float(FloatWidth::F64) | DType::Complex(ComplexWidth::F64)))
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

impl Axes {
    fn positions(&self, layout: &Layout) -> PyResult<Vec<usize>> {
        let mut axes = match self {
            Self::One(a) => vec![a.of(layout)?],
            Self::Many(axes) => axes.iter().map(|a| a.of(layout)).collect::<PyResult<_>>()?,
        };
        axes.sort_unstable();
        axes.dedup();
        Ok(axes)
    }
}

fn axes_of(axis: Option<&Axes>, layout: &Layout) -> PyResult<Vec<usize>> {
    axis.map_or_else(|| Ok((0..layout.shape().len()).collect()), |a| a.positions(layout))
}

/// Running a function object's kernels on a call's Python arguments, for each arity.
pub trait Apply<K> {
    fn apply<'py>(&self, py: Python<'py>, function: &str, k: &K, call: Call) -> PyResult<Bound<'py, PyAny>>;
}

impl<K: Sync> Apply<K> for Kernels<UnaryKernel<K>> {
    fn apply<'py>(&self, py: Python<'py>, function: &str, k: &K, call: Call) -> PyResult<Bound<'py, PyAny>> {
        let x = call.args[0].view();
        let buffer = released(py, || checked(function, self, x.dtype(), self.unary(k, x)))?;
        give_items(py, x.layout().clone(), buffer, call.scalar())
    }
}

/// `k` on two arguments. `domain` is the message of a Domain error for a failed integer result from a pair with a negative integer exponent and a base other than 1 or -1, as an integer power gives. Where `characters` gives the object's [`characters::binary`](rustymath::characters::binary), a call with a character argument runs it, and raises a Domain error where an offset isn't whole or a result isn't a Unicode scalar value.
fn binary<'py, K: Sync>(kernels: &Kernels<BinaryKernel<K>>, py: Python<'py>, function: &str, k: &K, call: Call, domain: Option<&str>, characters: Option<rustymath::characters::Kernel<K>>) -> PyResult<Bound<'py, PyAny>> {
    let Some(characters) = characters else { return binary_call(kernels, py, function, k, call, domain) };
    let (x, y) = (call.args[0].view(), call.args[1].view());
    if x.dtype() != DType::Character && y.dtype() != DType::Character { return binary_call(kernels, py, function, k, call, domain); }
    let agreement = Agreement::new(x.layout(), y.layout()).map_err(py_err)?;
    let (buffer, ok) = released(py, || characters(k, x, y, agreement.layout.shape(), &agreement)).ok_or_else(|| missing(function, DType::Character))?;
    if !ok { return Err(py_err(Error::new(Kind::Domain, format!("{function} gives a character only for a whole offset and a result in Unicode")))); }
    give_items(py, agreement.layout, buffer, call.scalar())
}

fn convert_binary<K>(_: &K, characters: Option<rustymath::characters::Kernel<K>>, args: &[&Bound<'_, PyAny>]) -> PyResult<Call> {
    if characters.is_some() { convert_kinds(args) } else { convert(args) }
}

/// [`binary`] on a call that [`convert`] or [`convert_kinds`] gave.
fn binary_call<'py, K: Sync>(kernels: &Kernels<BinaryKernel<K>>, py: Python<'py>, function: &str, k: &K, call: Call, domain: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
    let (x, y) = (call.args[0].view(), call.args[1].view());
    let agreement = Agreement::new(x.layout(), y.layout()).map_err(py_err)?;
    let buffer = released(py, || checked_binary(function, kernels, kernels.binary(k, x, y, &agreement), x.dtype(), y, &agreement, domain))?;
    give_items(py, agreement.layout, buffer, call.scalar())
}

/// Whether an integer that `y` gives at some position of `agreement` is negative, a fill included: the agreement's fill where it pairs no item of `y`, and `y`'s own where its mapping reaches no stored item.
fn negative_right(y: ArrayView<'_>, agreement: &Agreement) -> bool {
    let View::Integers(ints) = y.storage() else { return false };
    let (outer, inner) = (&agreement.right, y.mapping());
    with_ints!(ints, |items| (0..agreement.len).any(|i| {
        let x = match outer.get(i) { None => Fill::item(outer.chosen()), Some(j) => inner.get(j).map_or_else(|| Fill::item(inner.chosen()), |o| items[o]) };
        i64::from(x) < 0
    }))
}

/// [`checked`] after raising `domain` for a failed integer call with a negative right argument.
fn checked_binary<F: Copy>(function: &str, kernels: &Kernels<F>, result: Option<(Buffer, bool)>, dtype: DType, right: ArrayView<'_>, agreement: &Agreement, domain: Option<&str>) -> PyResult<Buffer> {
    if let (Some((buffer, false)), Some(message)) = (&result, domain) && matches!(buffer.dtype(), DType::Integer(_)) && negative_right(right, agreement) {
        return Err(py_err(Error::new(Kind::Domain, message.to_owned())));
    }
    checked(function, kernels, dtype, result)
}

impl<K: Sync> Apply<K> for Kernels<TernaryKernel<K>> {
    fn apply<'py>(&self, py: Python<'py>, function: &str, k: &K, call: Call) -> PyResult<Bound<'py, PyAny>> {
        let views = [0, 1, 2].map(|i| call.args[i].view());
        let first = Agreement::new(views[0].layout(), views[1].layout()).map_err(py_err)?;
        let second = Agreement::new(&first.layout, views[2].layout()).map_err(py_err)?;
        let buffer = released(py, || checked(function, self, views[0].dtype(), self.ternary(k, views, &first, &second)))?;
        give_items(py, second.layout, buffer, call.scalar())
    }
}

/// The kernel tables that fold by one reduction: folds of lanes ([`ArrayView::fold_items`]), folds through a view's mapping ([`reduce::along_mapped`]) and sliding folds of windows ([`slid`]). `folds!` builds them.
struct Folds<R> { lanes: Kernels<AccumulationKernel<R>>, mapped: Kernels<MappedFoldKernel<R>>, sliding: Kernels<SlidingKernel<R>> }

/// The [`Folds`] of the reduction `r`, limited to the dtypes that `only` names where it's given.
macro_rules! folds {
    ($r:expr $(, $only:expr)?) => { Folds { lanes: kernels!(Fold of $r)$(.only($only))?, mapped: kernels!(MappedFold of $r)$(.only($only))?, sliding: kernels!(Sliding of $r)$(.only($only))? } };
}

/// `r`'s totals of `x` over the axes that `axis` names: every axis for `None`, or one axis or a tuple of them. A total over every axis is a Python number.
fn fold<'py, R: Sync>(py: Python<'py>, function: &str, folds: &Folds<R>, r: &R, x: &Bound<'py, PyAny>, axis: Option<Axes>) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let view = x.view();
    let core = match released(py, || totals(function, folds, r, view, axis.as_ref()))? { Some(core) => core, None => x.into_core()? };
    let scalar = core.layout().shape().is_empty();
    give(py, core, scalar)
}

/// `r`'s totals of `x` over the axes that `axis` names, as [`fold`] takes them. `None` where `axis` is an empty tuple, which leaves the argument as it is. A fold along one axis of a view of windows slides over the view's storage without building the windows ([`slid`]). A view that reads a block of its storage folds the block in place, and a view that keeps what it gathers ([`ArrayView::keeps`]) gathers once where it reads runs that [`reduce::rereads_cheaply`] rejects, both through [`ArrayView::fold_items`]. Any other view folds through its mapping without building it ([`reduce::along_mapped`]). The totals keep `x`'s order.
fn totals<R>(function: &str, folds: &Folds<R>, r: &R, x: ArrayView<'_>, axis: Option<&Axes>) -> PyResult<Option<Core>> {
    let mut steps: Vec<Option<usize>> = match axis {
        None => vec![None],
        Some(axes) => axes.positions(x.layout())?.into_iter().map(Some).collect(),
    };
    // The last axis first, so that each later position still names its axis.
    steps.sort_unstable_by(|a, b| b.cmp(a));
    let mut out: Option<Core> = None;
    for step in steps {
        let view = out.as_ref().map_or(x, Core::view);
        let layout = view.layout();
        let next = match step { None => Layout::EMPTY, Some(a) => layout.axes((0..layout.shape().len()).filter(|&i| i != a)) };
        let (shape, axis) = match step { None => (vec![view.len()], 0), Some(a) => (layout.shape().to_vec(), a) };
        let through = view.mapping().block(layout.shape()).is_none() && !(view.keeps() && !reduce::rereads_cheaply(view.mapping(), layout.shape()));
        let (buffer, order) = if let Some(k) = step && let Some(slid) = slid(function, &folds.sliding, r, view, k)? {
            slid
        } else if through && let Some((buffer, fits, order)) = folds.mapped.mapped_fold(r, view.storage(), view.mapping(), &shape, axis) {
            (checked(function, &folds.mapped, view.dtype(), Some((buffer, fits)))?, order)
        } else {
            let (items, dims, order) = view.fold_items(step).map_err(py_err)?;
            (checked(function, &folds.lanes, items.view().dtype(), folds.lanes.run(r, items.view(), dims, None))?, order)
        };
        let totals = match order {
            None => Core::new(next, buffer),
            Some(steps) => Core::new(Layout::from([buffer.len()].as_slice()), buffer).and_then(|t| t.selected(Plan { layout: next, steps }, None)),
        };
        out = Some(totals.map_err(py_err)?);
    }
    Ok(out)
}

/// `r`'s folds along axis `k` of a view of windows, slid over the view's storage without building the windows, as rustymath's [`Slide`] classifies the view, with the tables that read them in the order of the view's other axes where they come in another order. `None` for a view that `Slide` doesn't classify.
fn slid<R>(function: &str, sliding: &Kernels<SlidingKernel<R>>, r: &R, view: ArrayView<'_>, k: usize) -> PyResult<Option<(Buffer, Option<Vec<Steps>>)>> {
    let Mapping::Steps { base, steps } = view.mapping() else { return Ok(None) };
    let shape = view.layout().shape();
    let tables = Steps::with_base(*base, steps.to_vec());
    let Some(s) = walked_source(&tables, shape, view.storage().len()).and_then(|source| Slide::new(&tables, shape, &source, &[k])) else { return Ok(None) };
    let &[outer, len, inner] = &s.dims[..] else { return Ok(None) };
    let storage = view.storage().slice(s.region.clone());
    let buffer = checked(function, sliding, storage.dtype(), sliding.sliding(r, storage, [outer, len, inner], s.widths[0]))?;
    Ok(Some((buffer, (!Steps::in_order(&s.tables)).then_some(s.tables))))
}

/// The lanes along `axis` of an argument with `layout` and `len` items, as rustymath's axis functions take them, and the result's layout: `layout` itself, or the flattened argument for `None`. Last comes the position of `axis` in `layout`, or `None` for no axis.
pub(crate) fn along_axis(layout: Layout, len: usize, axis: Option<Axis>) -> PyResult<([usize; 3], Layout, Option<usize>)> {
    match axis {
        None => Ok(([1, len, 1], Layout::from([len].as_slice()), None)),
        Some(a) => { let a = a.of(&layout)?; Ok((Lanes::new(layout.shape(), a).map_err(py_err)?.dims(), layout, Some(a))) }
    }
}

/// `r`'s running totals of `x` along `axis`, or along the flattened `x` for `None`.
fn scan<'py, R: Sync>(py: Python<'py>, function: &str, kernels: &Kernels<AccumulationKernel<R>>, r: &R, x: &Bound<'py, PyAny>, axis: Option<Axis>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[x])?;
    let items = call.args[0].at(call.args[0].view().dtype())?;
    let x = items.view();
    let (along, layout, _) = along_axis(call.args[0].layout(), x.len(), axis)?;
    let buffer = released(py, || checked(function, kernels, x.dtype(), kernels.run(r, x, along, None)))?;
    give_items(py, layout, buffer, false)
}

/// Bulk products, folds and scans at the same dtypes as their elementwise functions. A tolerant function compares within the tolerance that each call passes.
pub(crate) struct NativeBinary {
    pub apply: for<'a, 'b> fn(View<'a>, View<'b>, &Agreement, bool) -> PyResult<Buffer>,
    pub fold: for<'a, 'b> fn(View<'a>, [usize; 3], Option<View<'b>>, bool) -> PyResult<Buffer>,
    pub scan: for<'a, 'b> fn(View<'a>, [usize; 3], Option<View<'b>>, bool) -> PyResult<Buffer>,
}

pub(crate) type NativeFunction<'a> = (&'a str, bool);

pub(crate) fn native_binary(name: &str) -> PyResult<&'static NativeBinary> {
    NATIVE_BINARIES.iter().flatten().find(|(n, _)| *n == name).map(|(_, b)| b)
        .ok_or_else(|| PyValueError::new_err(format!("{name:?} isn't a native binary function")))
}

/// Runs `$run!` with the reduction by which the native function `$name` folds, from rustymath's table of facts ([`facts!`](rustymath::facts)), where a function of Booleans folds Booleans only. `None` for a function with no specialized reduction.
macro_rules! known_reduction {
    (($name:expr, $run:ident) folds { $($fold:ident: [$($p:tt)*] [$($n:literal),*];)* } booleans { $($truth:ident: [$($tp:tt)*] [$($tn:literal),*];)* } comparisons $comparisons:tt characters $characters:tt products $products:tt boolean_products $boolean_products:tt) => {
        match $name { $($($n => $run!(reduce::$fold),)*)* $($($tn => $run!(reduce::$truth, only &[DType::Boolean]),)*)* _ => None }
    };
}

/// Known associative functions keep their bulk reductions and their existing empty identities.
fn reduction_fold(name: &str, x: View<'_>, dims: [usize; 3], tail: Option<View<'_>>) -> Option<PyResult<Buffer>> {
    macro_rules! run { ($r:expr $(, only $only:expr)?) => {{
        let r = $r;
        let kernels = kernels!(Fold of r)$(.only($only))?;
        Some(checked(name, &kernels, x.dtype(), kernels.run(&r, x, dims, tail)))
    }}; }
    rustymath::facts!(known_reduction, name, run)
}

fn reduction_scan(name: &str, x: View<'_>, dims: [usize; 3], start: Option<View<'_>>) -> Option<PyResult<Buffer>> {
    macro_rules! run { ($r:expr $(, only $only:expr)?) => {{
        let r = $r;
        let kernels = kernels!(Scan of r)$(.only($only))?;
        Some(checked(name, &kernels, x.dtype(), kernels.run(&r, x, dims, start)))
    }}; }
    rustymath::facts!(known_reduction, name, run)
}

/// Each function object as a Python function and native callback. `; ct` binds the mode's comparison tolerance internally; an object after `=>` replaces the ordinary one in Mathish mode. `only` limits dtypes. `negative` gives a failed integer result's error beside a negative second argument. A character [`Rule`](rustymath::characters::Rule) supplies character arithmetic.
macro_rules! functions {
    ($($(#[doc = $doc:literal])+ $name:ident($($x:ident),+ $(; $ct:ident)?) = $k:expr $(=> $mathish:expr)? $(, only $only:expr)? $(, negative $negative:literal)?;)+) => {
        $(functions!(@define false; $(#[doc = $doc])+ $name($($x),+ $(; $ct)?) = $k $(=> $mathish)? $(, only $only)? $(, negative $negative)?);)+
        pub(crate) mod mathish {
            use super::*;
            $(functions!(@affected [$($ct)?], [$($mathish)?]; $(#[doc = $doc])+ $name($($x),+ $(; $ct)?) = $k $(=> $mathish)? $(, only $only)? $(, negative $negative)?);)+
        }
        static NATIVE_BINARIES: &[Option<(&str, NativeBinary)>] = &[
            $(functions!(@native $name, [$($x),+], [$($ct)?], [$($mathish)?], $k $(, only $only)? $(, negative $negative)?)),+
        ];
        fn add_elementwise(m: &Bound<'_, PyModule>) -> PyResult<()> {
            $(m.add_function(wrap_pyfunction!($name, m)?)?;)+
            m.add("_binary_names", NATIVE_BINARIES.iter().flatten().map(|(name, _)| *name).collect::<Vec<_>>())?;
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
    (@define $on:expr; $(#[doc = $doc:literal])+ $name:ident($($x:ident),+ $(; $ct:ident)?) = $k:expr $(=> $mathish:expr)? $(, only $only:expr)? $(, negative $negative:literal)?) => {
        $(#[doc = $doc])+
        #[pyfunction]
        #[pyo3(signature = ($($x),+))]
        pub(crate) fn $name<'py>(py: Python<'py>, $($x: &Bound<'py, PyAny>,)+) -> PyResult<Bound<'py, PyAny>> {
            $(let $ct = Tolerance::EXACT;)?
            let ordinary = $k;
            let call = functions!(@convert ordinary, [$($x),+])?;
            let _mode = $on && {
                let raw = functions!(@kernels [$($x),+] ordinary)$(.only($only))?;
                mathish_at(raw.computes(call.args[0].view().dtype()))
            };
            functions!(@object [$($ct)?], [$($mathish)?], _mode, $k, |k| {
                let kernels = functions!(@kernels [$($x),+] k)$(.only($only))?;
                functions!(@apply kernels, py, $name, k, call, [$($x),+] $(, $negative)?)
            })
        }
    };
    (@native $name:ident, [$x:ident, $y:ident], [$($ct:ident)?], [$($mathish:expr)?], $k:expr $(, only $only:expr)? $(, negative $negative:literal)?) => {
        Some((stringify!($name), NativeBinary {
            apply: |x, y, agreement, on| {
                let _mode = functions!(@mode [$($ct)?], on, $k, x.dtype());
                functions!(@object [$($ct)?], [$($mathish)?], _mode, $k, |k| {
                    let (xl, yl) = (Layout::from([x.len()]), Layout::from([y.len()]));
                    let kernels = kernels!(Binary of k)$(.only($only))?;
                    let (xv, yv) = (ArrayView::compact(&xl, x), ArrayView::compact(&yl, y));
                    checked_binary(stringify!($name), &kernels, kernels.binary(&k, xv, yv, agreement), x.dtype(), yv, agreement, None$(.or(Some($negative)))?)
                })
            },
            fold: |x, dims, tail, on| {
                if let Some(result) = reduction_fold(stringify!($name), x, dims, tail) { return result; }
                let _mode = functions!(@mode [$($ct)?], on, $k, x.dtype());
                functions!(@object [$($ct)?], [$($mathish)?], _mode, $k, |k| {
                    if dims[1] == 0 { return Err(PyValueError::new_err(format!("{} has no identity for an empty inner product", stringify!($name)))); }
                    let kernels = kernels!(RightFold of k)$(.only($only))?;
                    checked(stringify!($name), &kernels, x.dtype(), kernels.run(&k, x, dims, tail))
                })
            },
            scan: |x, dims, start, on| {
                if let Some(result) = reduction_scan(stringify!($name), x, dims, start) { return result; }
                let _mode = functions!(@mode [$($ct)?], on, $k, x.dtype());
                functions!(@object [$($ct)?], [$($mathish)?], _mode, $k, |k| {
                    let kernels = kernels!(LeftScan of k)$(.only($only))?;
                    checked(stringify!($name), &kernels, x.dtype(), kernels.run(&k, x, dims, start))
                })
            },
        }))
    };
    (@native $name:ident, [$($x:ident),+], $($rest:tt)*) => { None };
    (@mode [$($ct:ident)?], $on:expr, $ordinary:expr, $dtype:expr) => { $on && {
        $(let $ct = Tolerance::EXACT;)?
        let k = $ordinary;
        mathish_at(kernels!(Binary of k).computes($dtype))
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
    (@convert $k:ident, [$x:ident, $y:ident]) => {
        convert_binary(&$k, rustymath::characters!(of $k), &[$x, $y])
    };
    (@convert $k:ident, [$($x:ident),+]) => { convert(&[$($x),+]) };
    (@apply $kernels:ident, $py:ident, $name:ident, $k:ident, $call:ident, [$x:ident, $y:ident] $(, $negative:literal)?) => {
        binary(&$kernels, $py, stringify!($name), &$k, $call, None$(.or(Some($negative)))?, rustymath::characters!(of $k))
    };
    (@apply $kernels:ident, $py:ident, $name:ident, $k:ident, $call:ident, $args:tt) => { $kernels.apply($py, stringify!($name), &$k, $call) };
    (@kernels [$x:ident] $k:ident) => { kernels!(Unary of $k) };
    (@kernels [$x:ident, $y:ident] $k:ident) => { kernels!(Binary of $k) };
    (@kernels [$x:ident, $y:ident, $z:ident] $k:ident) => { kernels!(Ternary of $k) };
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
    /// The logarithm of x to the given base, with the base first. Mathish mode gives zero for base and argument one at float64 and complex128.
    logb(base, x) = rustymath::Logb => rustymath::MathishLogb;
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
    /// |x|, a real number for a complex x. The magnitude of a signed integer type's least value wraps to that value, as NumPy's does.
    absolute(x) = Wrapped(rustymath::Abs);
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
    /// ψ(x), the digamma function: the derivative of ln Γ(x).
    digamma(x) = rustymath::Digamma, only REAL;
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
    /// x + y. A character plus an integer, on either side, is a character. An integer result that doesn't fit its dtype wraps, as NumPy's does.
    add(x, y) = Wrapped(rustymath::Add);
    /// x - y. A character minus an integer is a character, and a character minus a character is an integer. An integer result that doesn't fit its dtype wraps, as NumPy's does.
    subtract(x, y) = Wrapped(rustymath::Subtract);
    /// x × y. An integer result that doesn't fit its dtype wraps, as NumPy's does.
    multiply(x, y) = Wrapped(rustymath::Multiply);
    /// x / y. Integers compute as floats. Mathish mode defines zero divided by zero as zero for float64 and complex128.
    divide(x, y) = rustymath::Divide => rustymath::MathishDivide;
    /// x^y. An integer power that doesn't fit its dtype wraps, as NumPy's does. A negative integer y beside an x other than 1 or -1 raises ValueError.
    power(x, y) = rustymath::WrappingPow, negative "an integer power with a negative exponent isn't an integer";
    /// The real nth root of y, computed without rounding 1/n first.
    root(n, y) = rustymath::Root;
    /// √(x² + y²), with no overflow or underflow in between.
    hypot(x, y) = rustymath::Hypot;
    /// The angle of the point (x, y) from the positive x axis, from -π to π.
    arctan2(y, x) = rustymath::Atan2;
    /// The binomial coefficient of n and k. An integer result raises OverflowError where it doesn't fit its dtype, and ValueError for a negative k. For a non-whole n with a whole k up to 64, a float result is within 2k ulps.
    binom(n, k) = rustymath::Binom, negative "an integer binom takes no negative k";
    /// ∂C(n, k)/∂n, the binomial coefficient's slope in n. For a non-whole n with a whole k up to 64, it's within 2k ulps.
    binom_dn(n, k) = rustymath::BinomDn, only REAL;
    /// ∂C(n, k)/∂k, the binomial coefficient's slope in k. Near n = 2k, where its two ψ terms cancel, its relative error has no bound.
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
    (@Fold $py:ident, $name:expr, $x:ident, $axis:ident, $r:ident $(, $only:expr)?) => { fold($py, $name, &folds!($r $(, $only)?), &$r, $x, $axis) };
    (@Scan $py:ident, $name:expr, $x:ident, $axis:ident, $r:ident $(, $only:expr)?) => { scan($py, $name, &kernels!(Scan of $r)$(.only($only))?, &$r, $x, $axis) };
    (@axis Fold) => { Axes };
    (@axis Scan) => { Axis };
}

reductions! {
    /// The sum of x over `axis`: every axis for None, or one axis, by position or name, or a tuple of them. Integers and Booleans sum as int64, and float16 as float32. An integer sum that doesn't fit int64 wraps.
    sum = Fold reduce::Sum;
    /// The product of x over `axis`, which `sum` describes. Integers and Booleans multiply as int64, and float16 as float32. An integer product that doesn't fit int64 wraps.
    prod = Fold reduce::Product;
    /// The largest item of x over `axis`, which `sum` describes. An empty lane gives the dtype's least value. Beside NaN, a lane gives NaN or its greatest other item, depending on the processor and the lane's length.
    max = Fold reduce::Max;
    /// The smallest item of x over `axis`, which `sum` describes. An empty lane gives the dtype's greatest value. Beside NaN, a lane gives NaN or its least other item, depending on the processor and the lane's length.
    min = Fold reduce::Min;
    /// Whether every Boolean of x over `axis` is true, with `axis` as `sum` describes.
    all = Fold reduce::Min, only &[DType::Boolean];
    /// Whether any Boolean of x over `axis` is true, with `axis` as `sum` describes.
    any = Fold reduce::Max, only &[DType::Boolean];
    /// The running sums of x along `axis`, by position or name, or along the flattened x for None. Integers and Booleans sum as int64, and float16 as float32. An integer sum that doesn't fit int64 wraps.
    cumsum = Scan reduce::Sum;
    /// The running products of x along `axis`, as `cumsum` takes it. Integers and Booleans multiply as int64, and float16 as float32. An integer product that doesn't fit int64 wraps.
    cumprod = Scan reduce::Product;
}

/// The smallest and largest items of `x` over `axis`, as `sum` takes it.
#[pyfunction]
#[pyo3(signature = (x, axis=None))]
fn minmax<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let x = one(x)?;
    let view = x.view();
    let computed = released(py, || -> PyResult<Option<[Core; 2]>> {
        if axis.is_none() && let Some(ends) = extremes(view.fold_items(None).map_err(py_err)?.0.view()) {
            return Ok(Some(ends.map(|b| Core::new(Layout::EMPTY, b).expect("one item has a rank-0 layout"))));
        }
        let lo = totals("minmax", &folds!(reduce::Min), &reduce::Min, view, axis.as_ref())?;
        let hi = totals("minmax", &folds!(reduce::Max), &reduce::Max, view, axis.as_ref())?;
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
    let call = convert(&[x])?;
    let items = call.args[0].at(call.args[0].view().dtype())?;
    let x = items.view();
    let View::Booleans(items) = x else { return Err(missing("logical_scan", x.dtype())) };
    let (along, layout, _) = along_axis(call.args[0].layout(), items.len(), axis)?;
    let mut out = Vec::with_capacity(items.len());
    released(py, || reduce::Logical(table).scan_along(items, along, start, &mut out));
    give_items(py, layout, Buffer::Booleans(out), false)
}

#[pyfunction]
#[pyo3(signature = (function, x, axis=None, start=None))]
fn _accumulate<'py>(py: Python<'py>, function: NativeFunction<'_>, x: &Bound<'py, PyAny>, axis: Option<Axis>, start: Option<&Bound<'py, PyAny>>) -> PyResult<Bound<'py, PyAny>> {
    let call = convert(&[Some(x), start].into_iter().flatten().collect::<Vec<_>>())?;
    let x = &call.args[0];
    let start = call.args.get(1);
    if start.is_some_and(|s| !s.layout().shape().is_empty()) { return Err(PyValueError::new_err("start must be a scalar")); }
    let (dims, layout, _) = along_axis(x.layout(), x.view().len(), axis)?;
    let (items, start) = (x.row_major()?, start.map(|s| s.row_major()).transpose()?);
    let (function, mode) = function;
    let scan = native_binary(function)?.scan;
    let result = released(py, || scan(items.view(), dims, start.as_ref().map(|s| s.view()), mode))?;
    give_items(py, layout, result, false)
}

/// The folds by the native binary `function` of each window of `width` neighbouring items of `x` along `axis`, by position or name. `function` must have an associative reduction. A lane of n items gives n - width + 1 folds, each keyed as the item that starts its window.
#[pyfunction]
#[pyo3(signature = (function, x, width, axis=Axis::Position(-1)))]
fn _sliding<'py>(py: Python<'py>, function: NativeFunction<'_>, x: &Bound<'py, PyAny>, width: usize, axis: Axis) -> PyResult<Bound<'py, PyAny>> {
    let (function, _) = function;
    let x = one(x)?;
    let items = x.row_major()?;
    let view = items.view();
    let (lanes, layout) = windowed(&x.layout(), width, &axis)?;
    macro_rules! run { ($r:expr $(, only $only:expr)?) => {{
        let r = $r;
        let kernels = kernels!(Sliding of r)$(.only($only))?;
        Some(checked(function, &kernels, view.dtype(), kernels.sliding(&r, view, lanes.dims(), width)))
    }}; }
    let folds = released(py, || rustymath::facts!(known_reduction, function, run)).ok_or_else(|| PyValueError::new_err(format!("{function:?} has no associative reduction to slide")))??;
    give_items(py, layout, folds, false)
}

/// The position along `axis` of the first maximum of each window of `width` neighbouring items of `x`, or of the first minimum where `max` is false, as int64.
#[pyfunction]
fn _sliding_pick<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, width: usize, axis: Axis, max: bool) -> PyResult<Bound<'py, PyAny>> {
    let x = one(x)?;
    let items = x.row_major()?;
    let (lanes, layout) = windowed(&x.layout(), width, &axis)?;
    let rows = released(py, || reduce::view_picks(items.view(), lanes.dims(), reduce::Picks::Windows(width), max)).ok_or_else(|| missing(if max { "sliding_argmax" } else { "sliding_argmin" }, items.view().dtype()))?;
    give_items(py, layout, i64::buffer(rows.into_iter().map(|r| r as i64).collect()), false)
}

/// `a`, with `b` broadcast over the selection that `index` makes, as brackets take it, and combined by `function` into each item it selects, in turn. A target selected more than once combines every item that selects it. `function` is add, subtract, multiply, divide, maximum or minimum. A failed integer result follows [`fitted`]'s rule.
#[pyfunction]
fn _at<'py>(py: Python<'py>, function: NativeFunction<'_>, a: &Bound<'py, PyAny>, index: &Bound<'py, PyAny>, b: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let (function, mode) = function;
    let mut args = convert(&[a, b])?.args.into_iter();
    let (a, b) = (args.next().expect("two arguments give two"), args.next().expect("two arguments give two"));
    // A maximum or minimum folds each place's values as its reduction folds them (`Combine`), so a NaN value never takes the place of the values before it.
    macro_rules! run { ($kind:ident, $k:expr) => {{
        let k = $k;
        let kernels = kernels!($kind of k);
        let dtype = kernels.computes(a.view().dtype()).ok_or_else(|| missing(function, a.view().dtype()))?;
        let mut out = if a.view().dtype() == dtype { a.into_core()? } else {
            let Items::Owned(buffer) = a.view().cast(dtype).map_err(py_err)? else { unreachable!("different dtypes need a cast") };
            Core::new(a.layout(), buffer).map_err(py_err)?
        };
        let plan = crate::structure::selection(out.layout(), index)?;
        let b = b.into_core()?;
        let ok = released(py, || out.combine_at(&k, &kernels, &plan, &b)).map_err(py_err)?;
        fitted(function, &kernels, dtype, dtype, ok)?;
        out
    }}; }
    let out = match function {
        "add" => run!(Scatter, Wrapped(rustymath::Add)), "subtract" => run!(Scatter, Wrapped(rustymath::Subtract)), "multiply" => run!(Scatter, Wrapped(rustymath::Multiply)),
        "divide" if mode && mathish_at(kernels!(Scatter of rustymath::Divide).computes(a.view().dtype())) => run!(Scatter, rustymath::MathishDivide),
        "divide" => run!(Scatter, rustymath::Divide),
        "maximum" => run!(Combine, reduce::Max), "minimum" => run!(Combine, reduce::Min),
        _ => return Err(PyValueError::new_err(format!("at takes add, subtract, multiply, divide, maximum or minimum, not {function}"))),
    };
    give(py, out, false)
}

/// The lanes along `axis` of an array of `layout`, for windows of `width` items, and the layout of the results, one for each window. A ValueError where a window doesn't fit the axis.
fn windowed(layout: &Layout, width: usize, axis: &Axis) -> PyResult<(Lanes, Layout)> {
    let a = axis.of(layout)?;
    let lanes = Lanes::new(layout.shape(), a).map_err(py_err)?;
    if !(1..=lanes.len).contains(&width) { return Err(PyValueError::new_err(format!("a window of {width} items doesn't fit an axis of {}", lanes.len))); }
    let starts = rustymath::structure::select_axis(layout, a, rustymath::structure::Selector::Range { start: 0, len: lanes.len + 1 - width, step: 1 }).map_err(py_err)?;
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
    let out = released(py, || rustymath::with_inexact!(compute, T => {
        let data = axes.read(items.of::<T>());
        match how {
            Summary::Mean => T::buffer(stats::mean(&data, axes.dims)),
            Summary::Variance(ddof) => <T as Scalar>::Real::buffer(stats::variance(&data, axes.dims, ddof)),
            Summary::Std(ddof) => <T as Scalar>::Real::buffer(stats::stddev(&data, axes.dims, ddof)),
            Summary::Norm(p) => <T as Scalar>::Real::buffer(stats::norm(&data, axes.dims, p)),
        }
    }));
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
pub(crate) fn var<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>, ddof: usize) -> PyResult<Bound<'py, PyAny>> { summary(py, x, axis, Summary::Variance(ddof)) }

/// The square root of var, over the same axes and with the same ddof.
#[pyfunction]
#[pyo3(signature = (x, axis=None, ddof=0))]
pub(crate) fn std<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, axis: Option<Axes>, ddof: usize) -> PyResult<Bound<'py, PyAny>> { summary(py, x, axis, Summary::Std(ddof)) }

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
    let out = released(py, || rustymath::with_inexact!(compute, T => T::buffer(stats::average(x.of::<T>(), w.of::<T>(), &axes, [&agreement.left, &agreement.right]))));
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
        },
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
pub(crate) fn cov<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, rowvar: bool, ddof: usize) -> PyResult<Bound<'py, PyAny>> { covariance(py, x, rowvar, ddof, false) }

/// Correlation of variables in the rows of a matrix, or columns with rowvar=False. Constant variables give NaN.
#[pyfunction]
#[pyo3(signature = (x, rowvar=true))]
pub(crate) fn corrcoef<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, rowvar: bool) -> PyResult<Bound<'py, PyAny>> { covariance(py, x, rowvar, 0, true) }

/// Histogram counts and edges. bins is a positive count (10 by default), or strictly increasing edges. `range` sets the span of a bin count. Explicit edges take no `range`. The final bin includes its upper edge; out-of-range values and NaNs are ignored.
#[pyfunction]
#[pyo3(signature = (x, bins=None, range=None))]
pub(crate) fn histogram<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, bins: Option<&Bound<'py, PyAny>>, range: Option<(f64, f64)>) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let count = bins.map_or(Ok(10), |b| b.extract::<usize>());
    let edges = if count.is_err() { bins } else { None };
    if edges.is_some() && range.is_some() { return Err(PyValueError::new_err("histogram's range applies to a bin count, not to explicit edges")); }
    let call = convert(&[Some(x), edges].into_iter().flatten().collect::<Vec<_>>())?;
    let dtype = call.args[0].view().dtype();
    let compute = dtype.inexact(FloatWidth::F64).filter(|(d, _)| !matches!(d, DType::Complex(_))).ok_or_else(|| missing("histogram", dtype))?.0;
    let x = call.args[0].at(compute)?;
    let explicit = call.args.get(1).map(|a| a.at(compute)).transpose()?;
    let (counts, edges) = released(py, || -> PyResult<_> { Ok(rustymath::with_float_width!(kernel match compute { DType::Float(w) => w, _ => unreachable!() }, F => Some({
        let data = F::of(x.view()).expect("histogram computes at a float width");
        let edges = match explicit.as_ref() {
            Some(e) => e.of::<F>().to_vec(),
            None => stats::histogram_edges(data, count.unwrap_or(10), range).map_err(py_err)?,
        };
        (stats::histogram(data, &edges).map_err(py_err)?, F::buffer(edges))
    })).expect("histogram computes in float32 or float64")) })?;
    let counts = give_items(py, Layout::from([counts.len()]), i64::buffer(counts), false)?;
    let edges = give_items(py, Layout::from([edges.len()]), edges, false)?;
    Ok((counts, edges))
}

/// Decimal rounding to nearest, with ties to even. Negative precisions round to tens, hundreds and so on.
#[pyfunction]
#[pyo3(signature = (x, decimals=0))]
pub(crate) fn round<'py>(py: Python<'py>, x: &Bound<'py, PyAny>, decimals: i32) -> PyResult<Bound<'py, PyAny>> {
    let k = rustymath::Round::new(decimals);
    kernels!(Unary of k).apply(py, "round", &k, convert(&[x])?)
}

/// Adds the elementwise functions, reductions and scans to `m`.
pub fn add_to(m: &Bound<'_, PyModule>) -> PyResult<()> {
    add_elementwise(m)?;
    add_reductions(m)?;
    m.add_function(wrap_pyfunction!(minmax, m)?)?;
    m.add_function(wrap_pyfunction!(logical_scan, m)?)?;
    m.add_function(wrap_pyfunction!(_accumulate, m)?)?;
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
