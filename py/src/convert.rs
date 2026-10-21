//! Turns a call's data arguments into views of one dtype. Arrays of different dtypes take the dtype that NumPy promotes them to, which [`DType::common`] gives, and characters beside numbers raise. An array or NumPy scalar inside a list counts as an array. The Python values in a call convert together. Beside an array they follow NEP 50: a value of the array's kind or a lower one takes the array's dtype, and a value of a higher kind promotes the array. On their own, they take the dtype of their highest kind: `bool`, `i64`, `f64`, then `c128`.
use crate::array::Array;
use rustymath::half::f16;
use rustymath::num_complex::{Complex32, Complex64};
use pyo3::exceptions::{PySystemError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyFloat, PyInt, PyList, PyString};
use rustymath::array::{Array as Core, ArrayView};
use rustymath::buffer::{Buffer, ComplexWidth, DType, Element, FloatWidth, Items, Width};
use rustymath::error::{Error, Kind};
use rustymath::layout::Layout;
use rustymath::structure::transpose;
use rustymath::with_dtype;
use rustymath_buffer::{Read, frame::{Frame, frame}, py_err, read, read_array, walk::{Collect, walked}};
use std::borrow::Cow;

/// The shared compact dtype named `name`, or a Python argument error.
pub fn dtype(name: &str) -> PyResult<DType> {
    DType::named(name).ok_or_else(|| PyValueError::new_err(format!("unknown dtype {name:?}")))
}

/// A Python exception carried through a rustymath function that's generic over its error, as [`Stacked`](rustymath::linalg::Stacked) is: a rustymath error, or the exception that [`signals`] raised.
pub struct Raised(PyErr);

impl From<Error> for Raised { fn from(e: Error) -> Self { Self(py_err(e)) } }

impl From<Raised> for PyErr { fn from(r: Raised) -> Self { r.0 } }

/// Runs `work`, a call's computation in Rust, with the GIL released, so other Python threads run meanwhile, as they do during NumPy's computations. Every binding converts its arguments before `work` and gives its results after it, holding the GIL. A call reads each `rustymath.Array` argument as it was when converted: the argument shares the array's storage, and another thread's write to the array during the call copies the storage first. Another thread's write to a buffer argument, such as a NumPy array, during the call leaves the call's results undefined.
pub fn released<R: Send>(py: Python<'_>, work: impl FnOnce() -> R + Send) -> R { py.detach(work) }

/// Raises the exception for a pending signal, such as Ctrl-C, holding the GIL only for the check. Work that [`released`] runs checks with it between steps.
pub fn signals() -> Result<(), Raised> { Python::attach(|py| py.check_signals()).map_err(Raised) }

/// One data argument, converted, and whether it was a Python scalar, as `scalar` tests.
pub struct Arg { source: Source, pub scalar: bool }

/// Where an argument's items are.
enum Source {
    /// The items of an object that exports a buffer, such as a NumPy array.
    Read { read: Read, layout: Layout },
    /// Python values, copied foreign storage, an array converted to another dtype, or a `rustymath.Array`'s items, shared with it.
    Owned(Core),
}

impl Source {
    /// The source of items read from a buffer: borrowed in place where the read lent them, and owned where it copied them.
    fn read(read: Read, layout: Layout) -> Self {
        if read.is_owned() { Self::Owned(Core::new(layout, read.into_buffer()).expect("a read's layout fits its items")) } else { Self::Read { read, layout } }
    }
}

impl From<Read> for Arg {
    fn from(read: Read) -> Self { let layout = Layout::from(read.shape()); Self { source: Source::read(read, layout), scalar: false } }
}

impl Arg {
    pub fn view(&self) -> ArrayView<'_> {
        match &self.source { Source::Read { read, layout } => ArrayView::compact(layout, read.view()), Source::Owned(c) => c.view() }
    }
    pub fn layout(&self) -> Layout { self.view().layout().clone() }
    /// The argument as an owned array. Only borrowed items are copied.
    pub fn into_core(self) -> PyResult<Core> {
        match self.source {
            Source::Read { read, layout } => Core::new(layout, read.into_buffer()).map_err(py_err),
            Source::Owned(c) => Ok(c),
        }
    }
    /// Selects before taking ownership, with the GIL released: foreign items gather directly, while Rustymath arrays can share affine views.
    pub fn selected(&self, py: Python<'_>, plan: rustymath::structure::Plan) -> PyResult<Core> {
        match &self.source {
            Source::Read { read, .. } => { let items = read.view(); released(py, || Core::gathered(items, &plan)) }
            Source::Owned(c) => released(py, || c.selected(plan)),
        }.map_err(py_err)
    }
    /// The items at `dtype`, borrowed where the argument already has it, and otherwise widened. An empty argument converts to any dtype. It raises where the argument's dtype doesn't widen to `dtype`, as [`DType::widens_to`] decides.
    pub fn at(&self, dtype: DType) -> PyResult<Items<'_>> {
        let view = self.view();
        if view.is_empty() && view.dtype() != dtype { return Ok(Items::Owned(Buffer::with_capacity(dtype, 0))); }
        if !view.is_empty() && !view.dtype().widens_to(dtype) { return Err(py_err(Error::new(Kind::Domain, format!("{} doesn't widen to {}", view.dtype().name(), dtype.name())))); }
        view.cast(dtype).map_err(py_err)
    }
    /// The items in row-major order at the argument's own dtype, borrowed where they're stored that way.
    pub fn row_major(&self) -> PyResult<Items<'_>> { self.at(self.view().dtype()) }
    /// The items in row-major order as `T`, widened as [`Arg::at`] widens them: borrowed where the argument already has `T`'s dtype, and otherwise the widened copy.
    pub fn items<T: Element>(&self) -> PyResult<Cow<'_, [T]>> {
        Ok(self.at(T::DTYPE)?.items())
    }
    /// The items in row-major order as `T`, converted as [`Buffer::cast`] converts them, narrowing included.
    pub fn cast<T: Element>(&self) -> PyResult<Vec<T>> { Ok(self.view().items::<T>().map_err(py_err)?.into_owned()) }
}

/// A call's data arguments, at one dtype.
pub struct Call { pub args: Vec<Arg> }

impl Call {
    /// Whether every argument was a Python scalar, so that the result is a Python number.
    pub fn scalar(&self) -> bool { self.args.iter().all(|a| a.scalar) }
}

/// A Python value that the walk found, before its dtype is known. `Big` is an `int` that doesn't fit 64 bits.
#[derive(Clone, Copy)]
enum Leaf { Bool(bool), Int(i64), Big(f64), Float(f64), Complex(Complex64), Char(char) }

impl Leaf {
    /// The dtype that the leaf takes on its own.
    fn dtype(self) -> DType {
        match self {
            Self::Bool(_) => DType::Boolean,
            Self::Int(_) | Self::Big(_) => DType::Integer(Width::I64),
            Self::Float(_) => DType::Float(FloatWidth::F64),
            Self::Complex(_) => DType::Complex(ComplexWidth::F64),
            Self::Char(_) => DType::Character,
        }
    }
}

fn mixed() -> PyErr { py_err(Error::new(Kind::Domain, "characters and numbers can't share an array")) }

/// Adds an array's dtype to the call's, promoting the two as [`DType::common`] does. Characters beside numbers raise.
fn join(strong: &mut Option<DType>, dtype: DType) -> PyResult<()> {
    *strong = Some(match *strong { Some(s) => s.common(dtype).ok_or_else(mixed)?, None => dtype });
    Ok(())
}

/// The dtype of a call whose arrays have `strong` and whose Python values reach `weak`, as [`DType::common_weak`] promotes them. A call with neither takes `float64`.
fn target(strong: Option<DType>, weak: Option<DType>) -> PyResult<DType> {
    Ok(match (strong, weak) {
        (Some(s), Some(w)) => s.common_weak(w).ok_or_else(mixed)?,
        (Some(d), None) | (None, Some(d)) => d,
        (None, None) => DType::Float(FloatWidth::F64),
    })
}

/// An item of the dtype that a Python value converts to.
trait FromLeaf: Sized {
    fn from_leaf(leaf: Leaf, dtype: DType) -> Result<Self, Error>;
}

fn unfit(leaf: Leaf, dtype: DType) -> Error {
    let name = dtype.name();
    match leaf {
        Leaf::Int(n) => Error::new(Kind::Overflow, format!("{n} doesn't fit {name}")),
        Leaf::Big(x) => Error::new(Kind::Overflow, format!("{x:e} doesn't fit {name}")),
        _ => Error::new(Kind::Domain, format!("a Python value can't convert to {name}")),
    }
}

impl FromLeaf for bool {
    fn from_leaf(leaf: Leaf, dtype: DType) -> Result<Self, Error> { if let Leaf::Bool(b) = leaf { Ok(b) } else { Err(unfit(leaf, dtype)) } }
}
impl FromLeaf for char {
    fn from_leaf(leaf: Leaf, dtype: DType) -> Result<Self, Error> { if let Leaf::Char(c) = leaf { Ok(c) } else { Err(unfit(leaf, dtype)) } }
}
macro_rules! integers {
    ($($t:ty),+) => {$(
        impl FromLeaf for $t {
            fn from_leaf(leaf: Leaf, dtype: DType) -> Result<Self, Error> {
                match leaf {
                    Leaf::Bool(b) => Ok(b.into()),
                    Leaf::Int(n) => <$t>::try_from(n).map_err(|_| unfit(leaf, dtype)),
                    _ => Err(unfit(leaf, dtype)),
                }
            }
        }
    )+};
}
integers!(u8, i16, i32, i64);

/// The real value of a numeric leaf below the complex kind.
fn real(leaf: Leaf) -> Option<f64> {
    match leaf { Leaf::Bool(b) => Some(f64::from(u8::from(b))), Leaf::Int(n) => Some(n as f64), Leaf::Big(x) | Leaf::Float(x) => Some(x), _ => None }
}
macro_rules! reals {
    ($($t:ty: |$x:ident| $from:expr),+) => {$(
        impl FromLeaf for $t {
            fn from_leaf(leaf: Leaf, dtype: DType) -> Result<Self, Error> { real(leaf).map(|$x| $from).ok_or_else(|| unfit(leaf, dtype)) }
        }
    )+};
}
reals!(f16: |x| f16::from_f64(x), f32: |x| x as f32, f64: |x| x);
macro_rules! complexes {
    ($($t:ty: |$z:ident| $from:expr),+) => {$(
        impl FromLeaf for $t {
            fn from_leaf(leaf: Leaf, dtype: DType) -> Result<Self, Error> {
                let $z = match leaf { Leaf::Complex(z) => z, _ => Complex64::new(real(leaf).ok_or_else(|| unfit(leaf, dtype))?, 0.0) };
                Ok($from)
            }
        }
    )+};
}
complexes!(Complex32: |z| Complex32::new(z.re as f32, z.im as f32), Complex64: |z| z);

/// A part of nested values in row-major order: one Python value, or the items of a buffer.
enum Piece { Leaf(Leaf), Block(Read) }

/// The collector for compact values: a [`Piece`] for each Python value or buffer.
struct Pieces;

impl Collect for Pieces {
    type Piece = Piece;
    fn int(&mut self, n: i128) -> Piece { Piece::Leaf(i64::try_from(n).map_or(Leaf::Big(n as f64), Leaf::Int)) }
    /// A buffer is tested first, because NumPy's `float64` and `complex128` scalars are also Python floats and complex numbers.
    fn piece(&mut self, obj: &Bound<'_, PyAny>) -> PyResult<Option<(Piece, Vec<usize>)>> {
        if let Some(r) = read(obj)? { let shape = r.shape().to_vec(); return Ok(Some((Piece::Block(r), shape))); }
        Ok(leaf(obj)?.map(|l| (Piece::Leaf(l), vec![])))
    }
}

/// The value of a Python scalar, a one-character `str`, or `None`, which reads as NaN. `Ok(None)` for any other object.
fn leaf(obj: &Bound<'_, PyAny>) -> PyResult<Option<Leaf>> {
    Ok(Some(if obj.is_instance_of::<PyBool>() {
        Leaf::Bool(obj.extract()?)
    } else if obj.is_instance_of::<PyInt>() {
        obj.extract().map(Leaf::Int).or_else(|_| obj.extract().map(Leaf::Big))?
    } else if obj.is_instance_of::<PyFloat>() {
        Leaf::Float(obj.extract()?)
    } else if let Ok(z) = obj.cast::<PyComplex>() {
        Leaf::Complex(Complex64::new(z.real(), z.imag()))
    } else if let Ok(s) = obj.cast::<PyString>() {
        let mut chars = s.to_str()?.chars();
        match (chars.next(), chars.next()) {
            (Some(c), None) => Leaf::Char(c),
            _ => return Err(py_err(Error::new(Kind::Domain, "a str inside a sequence must be one character; use list(s) for its characters"))),
        }
    } else if obj.is_none() {
        Leaf::Float(f64::NAN)
    } else {
        return Ok(None);
    }))
}

/// An argument before the call's dtype is known: an array, or nested values with their layout and whether they were a Python scalar.
enum Pending { Array(Arg), Values(Layout, Vec<Piece>, bool) }

/// Whether `obj` is a Python scalar: a `bool`, `int`, `float` or `complex`, not a subclass such as NumPy's `float64`.
fn scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_exact_instance_of::<PyBool>() || obj.is_exact_instance_of::<PyInt>() || obj.is_exact_instance_of::<PyFloat>() || obj.is_exact_instance_of::<PyComplex>()
}

/// Classifies one data argument, adding the dtype of any array it holds to `strong`.
fn classify(obj: &Bound<'_, PyAny>, strong: &mut Option<DType>) -> PyResult<Pending> {
    if scalar(obj) { return Ok(Pending::Values(Layout::EMPTY, vec![Piece::Leaf(leaf(obj)?.expect("a Python scalar is a leaf"))], true)); }
    // `None` is a missing value only as an item, so an operator given a bare `None` returns `NotImplemented`.
    if obj.is_none() { return Err(PyTypeError::new_err("rustymath can't convert NoneType")); }
    if let Some(frame) = frame(obj)? { return framed(obj.py(), frame, strong); }
    let source = if let Ok(a) = obj.cast::<Array>() {
        Some(Source::Owned(a.try_borrow()?.core().clone()))
    } else {
        let read = match read_array(obj)? {
            None if obj.hasattr("__array__")? => Some(read_array(&obj.call_method0("__array__")?)?.ok_or_else(|| PyTypeError::new_err("__array__ gave an object without a buffer"))?),
            read => read,
        };
        read.map(|(read, layout)| Source::read(read, layout))
    };
    if let Some(source) = source {
        let a = Arg { source, scalar: false };
        join(strong, a.view().dtype())?;
        return Ok(Pending::Array(a));
    }
    if let Ok(s) = obj.cast::<PyString>() {
        let chars: Vec<Piece> = s.to_str()?.chars().map(|c| Piece::Leaf(Leaf::Char(c))).collect();
        return Ok(Pending::Values(Layout::from([chars.len()]), chars, false));
    }
    let (layout, pieces) = walked(obj, &mut Pieces)?;
    for p in &pieces { if let Piece::Block(r) = p { join(strong, r.dtype())?; } }
    Ok(Pending::Values(layout, pieces, false))
}

/// A frame as a matrix of rows by columns, keyed and named by its labels. Its columns convert together, as a list of them would, and the result is their transpose, a view.
fn framed(py: Python<'_>, frame: Frame<'_>, strong: &mut Option<DType>) -> PyResult<Pending> {
    // rustymath holds characters but not strings, so a column of strings fails here, naming its label.
    let text = |o: Bound<'_, PyAny>| o.cast::<PyString>().is_ok_and(|s| s.to_str().is_ok_and(|t| t.chars().count() != 1));
    if let Some(i) = frame.columns.iter().position(|c| c.cast::<PyList>().is_ok_and(|items| items.iter().any(text))) {
        let label = frame.keys[1].as_ref().and_then(|k| k.names()[i].as_deref().map(|n| format!("{n:?}"))).unwrap_or_else(|| i.to_string());
        return Err(py_err(Error::new(Kind::Domain, format!("column {label} holds text, which a rustymath array can't hold"))));
    }
    let columns = one(&PyList::new(py, &frame.columns)?.into_any())?.into_core()?;
    let mut core = if frame.columns.is_empty() { columns.reshape(Layout::from([frame.rows, 0])) } else { columns.selected(transpose(columns.layout(), &[1, 0]).map_err(py_err)?) }.map_err(py_err)?;
    core.set_keys(frame.keys.to_vec()).map_err(py_err)?;
    core.set_names(frame.names.to_vec()).map_err(py_err)?;
    join(strong, core.dtype())?;
    Ok(Pending::Array(Arg { source: Source::Owned(core), scalar: false }))
}

/// Nested values at `dtype` under `layout`.
fn build(layout: Layout, pieces: &[Piece], dtype: DType) -> PyResult<Core> {
    let buffer = with_dtype!(dtype, T => {
        let mut out: Vec<T> = Vec::new();
        for p in pieces {
            match p {
                Piece::Leaf(l) => out.push(T::from_leaf(*l, dtype).map_err(py_err)?),
                Piece::Block(r) => match T::of(r.view()) {
                    Some(items) => out.extend_from_slice(items),
                    None => out.extend_from_slice(T::of(Buffer::cast(r.view(), dtype).map_err(py_err)?.view()).expect("a cast gives its dtype")),
                },
            }
        }
        T::buffer(out)
    });
    Core::new(layout, buffer).map_err(py_err)
}

/// Converts arguments together, optionally at a constructor's explicit dtype.
fn convert_at(args: &[&Bound<'_, PyAny>], forced: Option<DType>) -> PyResult<Call> {
    let mut strong = None;
    let pending = args.iter().map(|a| classify(a, &mut strong)).collect::<PyResult<Vec<_>>>()?;
    let mut weak: Option<DType> = None;
    for p in &pending {
        let Pending::Values(_, pieces, _) = p else { continue };
        for p in pieces {
            let Piece::Leaf(l) = p else { continue };
            let d = l.dtype();
            if weak.is_some_and(|w| (w == DType::Character) != (d == DType::Character)) { return Err(mixed()); }
            if weak.is_none_or(|w| d.kind() > w.kind()) { weak = Some(d); }
        }
    }
    let inferred = target(strong, weak)?;
    let target = forced.unwrap_or(inferred);
    // Every array and nested buffer widens to `strong`, so this one check guards the casts here and in `build`.
    if forced.is_none() && let Some(s) = strong.filter(|&s| !s.widens_to(target)) {
        return Err(PySystemError::new_err(format!("internal error: promotion would narrow {} to {}", s.name(), target.name())));
    }
    let args = pending.into_iter().map(|p| {
        let a = match p {
            Pending::Array(a) => a,
            Pending::Values(layout, pieces, scalar) => {
                // Leaf readers narrow within a kind. Lowering a kind uses the shared whole-buffer cast's truncation/validation rules.
                let at = if target.kind() < inferred.kind() { inferred } else { target };
                Arg { source: Source::Owned(build(layout, &pieces, at)?), scalar }
            }
        };
        if a.view().dtype() == target { return Ok(a); }
        let items = a.view().cast(target).map_err(py_err)?;
        let Items::Owned(buffer) = items else { unreachable!("different dtypes need a cast") };
        Core::new(a.layout(), buffer).map(|c| Arg { source: Source::Owned(c), scalar: a.scalar }).map_err(py_err)
    }).collect::<PyResult<Vec<_>>>()?;
    Ok(Call { args })
}

/// Converts a call's data arguments to their promoted dtype.
pub fn convert(args: &[&Bound<'_, PyAny>]) -> PyResult<Call> { convert_at(args, None) }

/// Converts a call's data arguments as [`convert`] does, except that characters beside numbers keep their dtype: the character arguments convert together, and so do the numeric ones. Characters and numbers inside one argument still raise.
pub fn convert_kinds(args: &[&Bound<'_, PyAny>]) -> PyResult<Call> {
    let joint = convert(args);
    if joint.is_ok() { return joint; }
    let characters = args.iter().map(|a| Ok(convert(&[a])?.args[0].view().dtype() == DType::Character)).collect::<PyResult<Vec<_>>>()?;
    if characters.iter().all(|&c| c == characters[0]) { return joint; }
    let group = |kind: bool| convert(&args.iter().zip(&characters).filter(|&(_, &c)| c == kind).map(|(a, _)| *a).collect::<Vec<_>>());
    let (mut chars, mut numbers) = (group(true)?.args.into_iter(), group(false)?.args.into_iter());
    Ok(Call { args: characters.iter().map(|&c| if c { chars.next() } else { numbers.next() }.expect("one converted argument for each")).collect() })
}

/// Converts one data argument.
pub fn one(x: &Bound<'_, PyAny>) -> PyResult<Arg> { Ok(convert(&[x])?.args.into_iter().next().expect("one argument gives one")) }

/// Converts one argument to an explicit dtype, constructing Python values there and borrowing matching arrays.
pub fn one_at(x: &Bound<'_, PyAny>, dtype: DType) -> PyResult<Arg> {
    Ok(convert_at(&[x], Some(dtype))?.args.into_iter().next().expect("one argument gives one"))
}
