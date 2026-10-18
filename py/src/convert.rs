//! Turns a call's data arguments into views of one dtype. Arrays of different dtypes take the dtype that NumPy promotes them to, which [`DType::common`] gives, and characters beside numbers raise. An array or NumPy scalar inside a list counts as an array. The Python values in a call convert together. Beside an array they follow NEP 50: a value of the array's kind or a lower one takes the array's dtype, and a value of a higher kind promotes the array. On their own, they take the dtype of their highest kind: `bool`, `i64`, `f64`, then `c128`.
use crate::array::Array;
use rustymath::half::f16;
use rustymath::num_complex::{Complex32, Complex64};
use pyo3::exceptions::{PySystemError, PyTypeError, PyValueError};
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyComplex, PyDict, PyFloat, PyInt, PyList, PyRange, PyString, PyTuple};
use rustymath::array::{Array as Core, ArrayView};
use rustymath::buffer::{Buffer, ComplexWidth, DType, Element, FloatWidth, Items, Width};
use rustymath::error::{Error, Kind};
use rustymath::layout::{Keys, Layout, element_count, generated_len};
use rustymath::with_dtype;
use rustymath_buffer::{Read, py_err, read};
use std::borrow::Cow;
use std::sync::Arc;

/// The shared compact dtype named `name`, or a Python argument error.
pub fn dtype(name: &str) -> PyResult<DType> {
    DType::named(name).ok_or_else(|| PyValueError::new_err(format!("unknown dtype {name:?}")))
}

/// One data argument, converted, and whether it was a Python scalar, as `scalar` tests.
pub struct Arg<'py> { source: Source<'py>, pub scalar: bool }

/// Where an argument's items are.
enum Source<'py> {
    /// The items of an object that exports a buffer, such as a NumPy array.
    Read { read: Read, layout: Layout },
    /// A `rustymath.Array`, borrowed.
    Array(PyRef<'py, Array>),
    /// Python values, copied foreign storage, or an array converted to another dtype.
    Owned(Core),
}

impl From<Read> for Arg<'_> {
    fn from(read: Read) -> Self {
        let layout = Layout::from(read.shape());
        let source = if read.is_owned() { Source::Owned(Core::new(layout, read.into_buffer()).expect("buffer shape")) } else { Source::Read { read, layout } };
        Self { source, scalar: false }
    }
}

impl Arg<'_> {
    pub fn view(&self) -> ArrayView<'_> {
        match &self.source { Source::Read { read, layout } => ArrayView::compact(layout, read.view()), Source::Array(a) => a.core().view(), Source::Owned(c) => c.view() }
    }
    pub fn layout(&self) -> Layout { self.view().layout().clone() }
    /// The argument as an owned array. Only borrowed items are copied.
    pub fn into_core(self) -> PyResult<Core> {
        match self.source {
            Source::Read { read, layout } => Core::new(layout, read.into_buffer()).map_err(py_err),
            Source::Array(a) => Ok(a.core().clone()),
            Source::Owned(c) => Ok(c),
        }
    }
    /// Selects before taking ownership: foreign items gather directly, while Rustymath arrays can share affine views.
    pub fn selected(&self, plan: rustymath::structure::Plan) -> PyResult<Core> {
        match &self.source {
            Source::Read { read, .. } => Core::gathered(read.view(), &plan),
            Source::Array(a) => a.core().selected(plan),
            Source::Owned(c) => c.selected(plan),
        }.map_err(py_err)
    }
    /// The items at `dtype`, borrowed where the argument already has it, and otherwise widened. An empty argument converts to any dtype. It raises where the argument's dtype doesn't widen to `dtype`, as [`DType::widens_to`] decides.
    pub fn at(&self, dtype: DType) -> PyResult<Items<'_>> {
        let view = self.view();
        if view.is_empty() && view.dtype() != dtype { return Ok(Items::Owned(Buffer::with_capacity(dtype, 0))); }
        if !view.is_empty() && !view.dtype().widens_to(dtype) { return Err(py_err(Error::new(Kind::Domain, format!("{} doesn't widen to {}", view.dtype().name(), dtype.name())))); }
        view.cast(dtype).map_err(py_err)
    }
    /// The items in row-major order as `T`, widened as [`Arg::at`] widens them: borrowed where the argument already has `T`'s dtype, and otherwise the widened copy.
    pub fn items<T: Element>(&self) -> PyResult<Cow<'_, [T]>> {
        Ok(self.at(T::DTYPE)?.items())
    }
    /// The items in row-major order as `T`, converted as [`Buffer::cast`] converts them, narrowing included.
    pub fn cast<T: Element>(&self) -> PyResult<Vec<T>> { Ok(self.view().items::<T>().map_err(py_err)?.into_owned()) }
}

/// A call's data arguments, at one dtype.
pub struct Call<'py> { pub args: Vec<Arg<'py>> }

impl Call<'_> {
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

/// What a walk of nested Python values collects, in row-major order. Compact values collect a [`Piece`] for each Python value or buffer, and `exact` collects rationals.
pub(crate) trait Collect: Sized {
    /// The piece for the item `n` of a `range`.
    fn int(n: i128) -> Self;
    /// The piece for `obj`, with the shape of its items, which is empty for one value. `None` for an object that holds no values of this kind.
    fn piece(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Self, Vec<usize>)>>;
}

impl Collect for Piece {
    fn int(n: i128) -> Self { Self::Leaf(i64::try_from(n).map_or(Leaf::Big(n as f64), Leaf::Int)) }
    /// A buffer is tested first, because NumPy's `float64` and `complex128` scalars are also Python floats and complex numbers.
    fn piece(obj: &Bound<'_, PyAny>) -> PyResult<Option<(Self, Vec<usize>)>> {
        if let Some(r) = read(obj)? { let shape = r.shape().to_vec(); return Ok(Some((Self::Block(r), shape))); }
        Ok(leaf(obj)?.map(|l| (Self::Leaf(l), vec![])))
    }
}

/// Nested values walked into a layout, with their pieces.
struct Walk<P> { shape: Vec<usize>, names: Vec<Option<Vec<Option<Arc<str>>>>>, depth: Option<usize>, pieces: Vec<P> }

/// The nested values of `obj`: their layout, with the keys of any dicts, and their pieces.
pub(crate) fn walked<P: Collect>(obj: &Bound<'_, PyAny>) -> PyResult<(Layout, Vec<P>)> {
    let mut walk = Walk { shape: vec![], names: vec![], depth: None, pieces: vec![] };
    walk.walk(obj, 0)?;
    Ok((walk.layout()?, walk.pieces))
}

fn ragged() -> PyErr { py_err(Error::new(Kind::Length, "ragged nested sequence")) }

/// The value of a Python scalar or a one-character `str`, or `None` for any other object.
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
    } else {
        return Ok(None);
    }))
}

impl<P: Collect> Walk<P> {
    /// Records an axis of `len` positions with `names` at `depth`, which must match every other axis there.
    fn axis(&mut self, depth: usize, len: usize, names: Option<Vec<Option<Arc<str>>>>) -> PyResult<()> {
        if self.depth == Some(depth) { return Err(ragged()); }
        if depth == self.shape.len() {
            self.shape.push(len);
            self.names.push(names);
            element_count(&self.shape).map_err(py_err)?;
        } else if self.shape[depth] != len || self.names[depth] != names {
            return Err(ragged());
        }
        Ok(())
    }
    /// Adds a piece whose items sit `depth` levels down.
    fn push(&mut self, depth: usize, piece: P) -> PyResult<()> {
        if depth < self.shape.len() || *self.depth.get_or_insert(depth) != depth { return Err(ragged()); }
        self.pieces.push(piece);
        Ok(())
    }
    fn walk(&mut self, obj: &Bound<'_, PyAny>, depth: usize) -> PyResult<()> {
        if let Ok(d) = obj.cast::<PyDict>() {
            let names = d.keys().iter().enumerate().map(|(i, k)| match k.cast::<PyString>() {
                Ok(s) => Ok(Some(Arc::from(s.to_str()?))),
                Err(_) if k.extract::<usize>().is_ok_and(|n| n == i) => Ok(None),
                Err(_) => Err(py_err(Error::new(Kind::Domain, "a dict's keys must be strings, or ints equal to their positions"))),
            }).collect::<PyResult<Vec<_>>>()?;
            self.axis(depth, d.len(), Some(names))?;
            for v in d.values() { self.walk(&v, depth + 1)?; }
        } else if obj.is_instance_of::<PyList>() || obj.is_instance_of::<PyTuple>() {
            self.axis(depth, obj.len()?, None)?;
            for v in obj.try_iter()? { self.walk(&v?, depth + 1)?; }
        } else if obj.is_instance_of::<PyRange>() {
            let len = generated_len(&[obj.len()?]).map_err(py_err)?;
            let (start, step): (i128, i128) = (obj.getattr("start")?.extract()?, obj.getattr("step")?.extract()?);
            self.axis(depth, len, None)?;
            for i in 0..len as i128 {
                self.push(depth + 1, P::int(start + i * step))?;
            }
        } else if let Some((piece, shape)) = P::piece(obj)? {
            for (a, &len) in shape.iter().enumerate() { self.axis(depth + a, len, None)?; }
            self.push(depth + shape.len(), piece)?;
        } else {
            return Err(PyTypeError::new_err(format!("rustymath can't convert {}", obj.get_type().name()?)));
        }
        Ok(())
    }
    /// The walked values' layout, with the keys of their dicts.
    fn layout(&self) -> PyResult<Layout> {
        let layout = Layout::from(self.shape.as_slice());
        if self.names.iter().all(Option::is_none) { return Ok(layout); }
        let keys = self.names.iter().map(|n| n.clone().map(Keys::partial).transpose()).collect::<Result<Vec<_>, _>>().map_err(py_err)?;
        layout.with_keys(keys).map_err(py_err)
    }
}

/// An argument before the call's dtype is known: an array, or nested values with their layout and whether they were a Python scalar.
enum Pending<'py> { Array(Arg<'py>), Values(Layout, Vec<Piece>, bool) }

/// Whether `obj` is a Python scalar: a `bool`, `int`, `float` or `complex`, not a subclass such as NumPy's `float64`.
fn scalar(obj: &Bound<'_, PyAny>) -> bool {
    obj.is_exact_instance_of::<PyBool>() || obj.is_exact_instance_of::<PyInt>() || obj.is_exact_instance_of::<PyFloat>() || obj.is_exact_instance_of::<PyComplex>()
}

/// Classifies one data argument, adding the dtype of any array it holds to `strong`.
fn classify<'py>(obj: &Bound<'py, PyAny>, strong: &mut Option<DType>) -> PyResult<Pending<'py>> {
    if scalar(obj) { return Ok(Pending::Values(Layout::EMPTY, vec![Piece::Leaf(leaf(obj)?.expect("a Python scalar is a leaf"))], true)); }
    let source = if let Ok(a) = obj.cast::<Array>() {
        Some(Source::Array(a.try_borrow()?))
    } else if let Some(r) = read(obj)? {
        Some(Arg::from(r).source)
    } else if obj.hasattr("__array__")? {
        let a = obj.call_method0("__array__")?;
        Some(Arg::from(read(&a)?.ok_or_else(|| PyTypeError::new_err("__array__ gave an object without a buffer"))?).source)
    } else {
        None
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
    let (layout, pieces) = walked::<Piece>(obj)?;
    for p in &pieces { if let Piece::Block(r) = p { join(strong, r.dtype())?; } }
    Ok(Pending::Values(layout, pieces, false))
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
fn convert_at<'py>(args: &[&Bound<'py, PyAny>], forced: Option<DType>) -> PyResult<Call<'py>> {
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
pub fn convert<'py>(args: &[&Bound<'py, PyAny>]) -> PyResult<Call<'py>> { convert_at(args, None) }

/// Converts one data argument.
pub fn one<'py>(x: &Bound<'py, PyAny>) -> PyResult<Arg<'py>> { Ok(convert(&[x])?.args.into_iter().next().expect("one argument gives one")) }

/// Converts one argument to an explicit dtype, constructing Python values there and borrowing matching arrays.
pub fn one_at<'py>(x: &Bound<'py, PyAny>, dtype: DType) -> PyResult<Arg<'py>> {
    Ok(convert_at(&[x], Some(dtype))?.args.into_iter().next().expect("one argument gives one"))
}
