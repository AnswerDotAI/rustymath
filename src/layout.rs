//! Array shapes, axis keys and names, and their limits.
use crate::error::{Error, Kind};
use crate::hashing::{HashMap, HashMapExt, HashSet, HashSetExt};
use std::{fmt, sync::Arc};

/// The most axes an array can have.
pub const MAX_RANK: usize = 128;
/// The most items in an array that a function makes.
pub const MAX_GENERATED_ELEMENTS: usize = 1_000_000_000;

/// The number of items in an array of `shape`. A Limit error when `shape` has more than `MAX_RANK` axes, or more items than a `usize` counts.
#[inline]
pub fn element_count(shape: &[usize]) -> Result<usize, Error> {
    if shape.len() > MAX_RANK { return Err(Error::new(Kind::Limit, format!("an array has at most {MAX_RANK} axes"))); }
    if shape.contains(&0) { return Ok(0); }
    shape.iter().try_fold(1usize, |n, &d| n.checked_mul(d)).ok_or_else(|| Error::new(Kind::Limit, "an array of this shape has too many items to count"))
}

/// The number of items in an array of `shape` that a function makes. A Limit error past `MAX_GENERATED_ELEMENTS`, or where the array would take the process past its memory guard's limit ([`memory::admit`](crate::memory::admit)).
#[inline]
pub fn generated_len(shape: &[usize]) -> Result<usize, Error> {
    let len = element_count(shape)?;
    if len > MAX_GENERATED_ELEMENTS { return Err(Error::new(Kind::Limit, format!("an array has at most {MAX_GENERATED_ELEMENTS} items"))); }
    crate::memory::admit(len)?;
    Ok(len)
}

/// The distance between neighbouring items along each axis of an array of `shape`.
#[inline]
pub fn strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for a in (1..shape.len()).rev() { strides[a - 1] = strides[a] * shape[a]; }
    strides
}

/// Each axis's coordinate of flat position `flat`, from the last axis to the first. `flat` must be within the shape.
#[inline]
pub fn coordinates(mut flat: usize, shape: &[usize]) -> impl Iterator<Item = (usize, usize)> + '_ {
    shape.iter().enumerate().rev().map(move |(axis, &len)| {
        let coordinate = flat % len;
        flat /= len;
        (axis, coordinate)
    })
}

/// The position among `len` that `n` gives, counting from the end when negative. `None` outside the range.
#[inline]
pub fn position(n: i64, len: usize) -> Option<usize> {
    let i = if n < 0 { len as i64 + n } else { n };
    usize::try_from(i).ok().filter(|&i| i < len)
}

/// An array's items as lanes along one axis: `outer` lanes before it, each `len` items long, with `inner` items between neighbours along it.
#[derive(Clone, Copy, Debug)]
pub struct Lanes { pub outer: usize, pub len: usize, pub inner: usize }

impl Lanes {
    /// The lanes along `axis` of an array of `shape`. A Rank error when `axis` is outside the rank, and a Limit error where `element_count` gives one.
    #[inline]
    pub fn new(shape: &[usize], axis: usize) -> Result<Self, Error> {
        let &len = shape.get(axis).ok_or_else(|| Error::new(Kind::Rank, format!("axis {axis} is outside an array of rank {}", shape.len())))?;
        Ok(Self { outer: element_count(&shape[..axis])?, len, inner: element_count(&shape[axis + 1..])? })
    }
    /// The offset of item `j` along the axis, in the lane at outer index `i` and inner index `k`.
    #[inline]
    pub fn offset(&self, i: usize, j: usize, k: usize) -> usize { (i * self.len + j) * self.inner + k }
    /// The lanes as rustymath's axis functions take them: `[outer, len, inner]`.
    #[inline]
    pub fn dims(&self) -> [usize; 3] { [self.outer, self.len, self.inner] }
}

/// The value that every present item of `values` equals. Absent items don't count. Two different values give none.
pub fn agreed<'a, T: PartialEq + Clone + 'a>(values: impl IntoIterator<Item = Option<&'a T>>) -> Option<T> {
    let mut found = None;
    for value in values.into_iter().flatten() {
        if found.is_some_and(|f| f != value) { return None; }
        found = Some(value);
    }
    found.cloned()
}

/// The first name that appears more than once in `names`.
pub fn repeated_name(names: &[Option<Arc<str>>]) -> Option<&Arc<str>> {
    let mut seen = HashSet::new();
    names.iter().flatten().find(|n| !seen.insert(*n))
}

/// Whether an axis name is hidden: it starts with `_`. Hidden and visible names are separate namespaces.
#[inline]
pub fn hidden(name: &str) -> bool { name.starts_with('_') }

/// Whether two axis names conflict: both are names, they differ, and they're in the same namespace. A name meets a name from the other namespace as if that axis were unnamed.
#[inline]
pub fn names_conflict(a: Option<&str>, b: Option<&str>) -> bool { matches!((a, b), (Some(a), Some(b)) if a != b && hidden(a) == hidden(b)) }

/// An array's shape: the length of each axis. Ranks 0 to 2 are held inline with no allocation, and higher ranks in one allocation of exactly their length. Convert a `Vec`, an array or a slice of lengths into one, or collect it from lengths.
#[derive(Clone)]
pub struct Shape(Lengths);

#[derive(Clone)]
enum Lengths { Small(u8, [usize; 2]), Large(Box<[usize]>) }

impl Shape {
    /// The shape of a rank-0 array.
    pub const EMPTY: Self = Self(Lengths::Small(0, [0; 2]));
}
impl Default for Shape {
    #[inline]
    fn default() -> Self { Self::EMPTY }
}
impl std::ops::Deref for Shape {
    type Target = [usize];
    #[inline]
    fn deref(&self) -> &[usize] { match &self.0 { Lengths::Small(n, lengths) => &lengths[..usize::from(*n)], Lengths::Large(lengths) => &lengths[..] } }
}
impl std::ops::DerefMut for Shape {
    #[inline]
    fn deref_mut(&mut self) -> &mut [usize] {
        match &mut self.0 { Lengths::Small(n, lengths) => &mut lengths[..usize::from(*n)], Lengths::Large(lengths) => &mut lengths[..] }
    }
}
impl PartialEq for Shape {
    #[inline]
    fn eq(&self, other: &Self) -> bool { **self == **other }
}
impl fmt::Debug for Shape { fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result { fmt::Debug::fmt(&**self, f) } }
impl From<&[usize]> for Shape {
    #[inline]
    fn from(lengths: &[usize]) -> Self {
        match *lengths {
            [] => Self::EMPTY, [a] => Self(Lengths::Small(1, [a, 0])), [a, b] => Self(Lengths::Small(2, [a, b])), _ => Self(Lengths::Large(lengths.into()))
        }
    }
}
impl From<Vec<usize>> for Shape {
    #[inline]
    fn from(lengths: Vec<usize>) -> Self { if lengths.len() > 2 { Self(Lengths::Large(lengths.into_boxed_slice())) } else { Self::from(lengths.as_slice()) } }
}
impl<const N: usize> From<[usize; N]> for Shape { fn from(lengths: [usize; N]) -> Self { Self::from(lengths.as_slice()) } }
impl FromIterator<usize> for Shape {
    fn from_iter<I: IntoIterator<Item = usize>>(lengths: I) -> Self {
        let mut lengths = lengths.into_iter();
        let mut small = [0; 2];
        for n in 0..2 {
            let Some(len) = lengths.next() else { return Self(Lengths::Small(n, small)) };
            small[usize::from(n)] = len;
        }
        match lengths.next() {
            None => Self(Lengths::Small(2, small)), Some(third) => Self(Lengths::Large(small.into_iter().chain([third]).chain(lengths).collect()))
        }
    }
}

/// Names for the positions on one axis. A position may have no name. The names that are present are unique.
#[derive(Clone, Debug, Default)]
pub struct Keys { names: Vec<Option<Arc<str>>>, index: HashMap<Arc<str>, usize> }

// Two keys are equal when they hold the same names in the same order.
impl PartialEq for Keys { fn eq(&self, other: &Self) -> bool { self.names == other.names } }
impl Eq for Keys {}

impl Keys {
    /// Keys from `names`, where `None` marks a position with no name. A Domain error when a name appears twice.
    pub fn partial(names: Vec<Option<Arc<str>>>) -> Result<Arc<Self>, Error> {
        let mut index = HashMap::with_capacity(names.len());
        for (i, name) in names.iter().enumerate() {
            let Some(name) = name else { continue };
            if index.insert(name.clone(), i).is_some() { return Err(duplicate(name)); }
        }
        Ok(Arc::new(Self { names, index }))
    }
    #[inline]
    pub fn names(&self) -> &[Option<Arc<str>>] { &self.names }
    #[inline]
    pub fn len(&self) -> usize { self.names.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.names.is_empty() }
    /// Whether every position has a name.
    #[inline]
    pub fn complete(&self) -> bool { self.index.len() == self.names.len() }
    /// Whether no position has a name. An axis like that carries no keys.
    #[inline]
    pub fn blank(&self) -> bool { self.index.is_empty() && !self.is_empty() }
    #[inline]
    pub fn position(&self, name: &str) -> Option<usize> { self.index.get(name).copied() }
    /// Adds positions at the end, named by `names`. A Domain error, with no position added, when a name is already here or appears twice in `names`.
    pub fn extend(&mut self, names: &[Option<Arc<str>>]) -> Result<(), Error> {
        let mut new = HashSet::new();
        if let Some(name) = names.iter().flatten().find(|n| self.position(n).is_some() || !new.insert(*n)) { return Err(duplicate(name)); }
        for name in names {
            if let Some(name) = name { self.index.insert(name.clone(), self.names.len()); }
            self.names.push(name.clone());
        }
        Ok(())
    }
    /// For each position of `wanted`, the matching position here. Named positions match by name. Unnamed positions match in order.
    pub fn align(&self, wanted: &Self) -> Vec<Option<usize>> {
        let mut gaps = self.names.iter().enumerate().filter(|(_, n)| n.is_none()).map(|(i, _)| i);
        wanted.names.iter().map(|n| match n { Some(n) => self.position(n), None => gaps.next() }).collect()
    }
    /// For each position of `wanted`, the matching position here. With `subset`, `wanted` may have fewer positions, and one with no match is an Index error. Otherwise both need the same positions, and a mismatch is a Length error.
    pub fn positions(&self, wanted: &Self, subset: bool) -> Result<Vec<usize>, Error> {
        let kind = if subset { Kind::Index } else { Kind::Length };
        if !subset && self.len() != wanted.len() { return Err(Error::new(kind, format!("keyed axes of lengths {} and {} don't pair", self.len(), wanted.len()))); }
        self.align(wanted).into_iter().zip(&wanted.names).map(|(p, name)| p.ok_or_else(|| match name {
            Some(name) => Error::new(kind, format!("missing key: {name}")),
            None => Error::new(kind, "too few positions without keys"),
        })).collect()
    }
    /// Keys at source positions, in the order given. A position with no source has no name.
    pub fn select(&self, positions: impl IntoIterator<Item = Option<usize>>) -> Result<Arc<Self>, Error> {
        Self::partial(positions.into_iter().map(|i| i.and_then(|i| self.names[i].clone())).collect())
    }
    /// These positions, then the named positions of `other` that these lack. Unnamed positions pair in order. Both sides need the same number of them.
    pub fn union(self: &Arc<Self>, other: &Arc<Self>) -> Result<Arc<Self>, Error> {
        if self == other { return Ok(self.clone()); }
        let (a, b) = (self.len() - self.index.len(), other.len() - other.index.len());
        if a != b { return Err(Error::new(Kind::Length, format!("axes with {a} and {b} positions without keys don't align"))); }
        let mut extra = other.names.iter().filter(|k| k.as_ref().is_some_and(|k| self.position(k).is_none())).peekable();
        if extra.peek().is_none() { return Ok(self.clone()); }
        Self::partial(self.names.iter().chain(extra).cloned().collect())
    }
}

/// The Domain error for a key that appears twice on one axis.
fn duplicate(key: &str) -> Error { Error::new(Kind::Domain, format!("duplicate key: {key}")) }

/// An array's shape, with optional keys and a name for each axis. A layout with no keys or names holds no `Meta`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Layout { shape: Shape, meta: Option<Box<Meta>> }

/// The keys and names of a layout's axes. An empty list means none on any axis.
#[derive(Clone, Debug, Default, PartialEq)]
struct Meta { keys: Vec<Option<Arc<Keys>>>, names: Vec<Option<Arc<str>>> }

impl<S: Into<Shape>> From<S> for Layout {
    #[inline]
    fn from(shape: S) -> Self { Self { shape: shape.into(), meta: None } }
}
impl Layout {
    /// The layout of a rank-0 array.
    pub const EMPTY: Self = Self { shape: Shape::EMPTY, meta: None };
    #[inline]
    pub fn shape(&self) -> &[usize] { &self.shape }
    /// The name of each axis, with `None` for an axis without one. Empty when no axis has a name (`has_names`).
    #[inline]
    pub fn names(&self) -> &[Option<Arc<str>>] { self.meta.as_ref().map_or(&[], |l| &l.names) }
    #[inline]
    pub fn name(&self, axis: usize) -> Option<&Arc<str>> { self.names().get(axis).and_then(Option::as_ref) }
    /// The axis named `name`, if there is one.
    #[inline]
    pub fn axis_named(&self, name: &str) -> Option<usize> { self.names().iter().position(|n| n.as_deref() == Some(name)) }
    fn meta_mut(&mut self) -> &mut Meta { self.meta.get_or_insert_with(Default::default) }
    /// The layout without an empty `Meta`.
    fn tidy(mut self) -> Self { if self.meta.as_deref() == Some(&Meta::default()) { self.meta = None; } self }
    /// The layout with `set` applied to its `Meta`. `empty` says that the new entry needs no `Meta`.
    fn edit(mut self, empty: bool, set: impl FnOnce(&mut Meta)) -> Self {
        if empty && self.meta.is_none() { return self; }
        set(self.meta_mut());
        self.tidy()
    }
    /// The layout with `names` on its axes. A Rank error when the count differs from the rank, and a Domain error when a name repeats.
    pub fn with_names(self, names: Vec<Option<Arc<str>>>) -> Result<Self, Error> {
        if !names.is_empty() && names.len() != self.shape.len() { return Err(Error::new(Kind::Rank, format!("{} names for {} axes", names.len(), self.shape.len()))); }
        if let Some(name) = repeated_name(&names) { return Err(Error::new(Kind::Domain, format!("the result would have two axes named {name}"))); }
        Ok(self.inherit_names(names))
    }
    /// The layout with `names`, which the caller has checked: one for each axis, or none.
    pub fn inherit_names(self, names: Vec<Option<Arc<str>>>) -> Self {
        assert!(names.is_empty() || names.len() == self.shape.len());
        let empty = names.iter().all(Option::is_none);
        self.edit(empty, |m| m.names = if empty { vec![] } else { names })
    }
    /// The layout without any name that appears on more than one axis.
    pub fn unique_names(mut self) -> Self {
        let Some(meta) = self.meta.as_mut() else { return self };
        let mut counts = HashMap::new();
        for name in meta.names.iter().flatten() { *counts.entry(name.clone()).or_insert(0) += 1; }
        for name in &mut meta.names { if name.as_ref().is_some_and(|n| counts[n] > 1) { *name = None; } }
        if meta.names.iter().all(Option::is_none) { meta.names.clear(); }
        self.tidy()
    }
    #[inline]
    fn key_list(&self) -> &[Option<Arc<Keys>>] { self.meta.as_ref().map_or(&[], |l| &l.keys) }
    #[inline]
    pub fn has_keys(&self) -> bool { !self.key_list().is_empty() }
    /// Whether any axis has a name.
    #[inline]
    pub fn has_names(&self) -> bool { !self.names().is_empty() }
    /// Whether the layout has no keys or names.
    #[inline]
    pub fn plain(&self) -> bool { self.meta.is_none() }
    #[inline]
    pub fn keys(&self, axis: usize) -> Option<&Arc<Keys>> { self.key_list().get(axis).and_then(Option::as_ref) }
    /// The keys of every axis, with `None` for an axis without them.
    pub fn all_keys(&self) -> Vec<Option<Arc<Keys>>> { (0..self.shape.len()).map(|a| self.keys(a).cloned()).collect() }
    /// Whether shapes, assigned axis names and assigned keys match. An unlabelled axis matches either label set.
    pub fn matches(&self, other: &Self) -> bool {
        self.shape == other.shape && (0..self.shape.len()).all(|a| {
            !names_conflict(self.name(a).map(AsRef::as_ref), other.name(a).map(AsRef::as_ref)) &&
                match (self.keys(a), other.keys(a)) { (Some(x), Some(y)) => !y.align(x).contains(&None), _ => true }
        })
    }
    /// The layout without keys on `axis`.
    pub fn without_keys(self, axis: usize) -> Self {
        if self.keys(axis).is_none() { return self; }
        let mut keys = self.all_keys();
        keys[axis] = None;
        self.with_keys(keys).unwrap()
    }
    /// The layout with `keys` on its axes. A Rank error when the count differs from the rank, and a Length error when an axis's keys differ from its length.
    pub fn with_keys(self, keys: Vec<Option<Arc<Keys>>>) -> Result<Self, Error> {
        if !keys.is_empty() && keys.len() != self.shape.len() { return Err(Error::new(Kind::Rank, format!("keys for {} axes of an array with {} axes", keys.len(), self.shape.len()))); }
        if let Some((k, n)) = keys.iter().zip(self.shape.iter()).find_map(|(k, &n)| k.as_ref().filter(|k| k.len() != n).map(|k| (k.len(), n))) {
            return Err(Error::new(Kind::Length, format!("{k} keys for an axis of length {n}")));
        }
        let keys: Vec<_> = keys.into_iter().map(|k| k.filter(|k| !k.blank())).collect();
        let empty = keys.iter().all(Option::is_none);
        Ok(self.edit(empty, |m| m.keys = if empty { vec![] } else { keys }))
    }
    /// A layout with one axis for each of `sources`: a layout and one of its axes, whose length, keys and name the new axis takes.
    fn gathered<'a>(sources: impl Iterator<Item = (&'a Self, usize)> + Clone) -> Self {
        let shape: Shape = sources.clone().map(|(l, a)| l.shape[a]).collect();
        if sources.clone().all(|(l, _)| l.plain()) { return Self::from(shape); }
        let keys = sources.clone().map(|(l, a)| l.keys(a).cloned()).collect();
        let names = sources.map(|(l, a)| l.name(a).cloned()).collect();
        Self::from(shape).with_keys(keys).unwrap().inherit_names(names)
    }
    /// The layout of `axes`, in that order.
    pub fn axes(&self, axes: impl Iterator<Item = usize> + Clone) -> Self { Self::gathered(axes.map(|a| (self, a))) }
    /// This layout's axes, then `other`'s.
    pub fn concat(&self, other: &Self) -> Self { Self::gathered((0..self.shape.len()).map(|a| (self, a)).chain((0..other.shape.len()).map(|a| (other, a)))) }
    /// The layout of a product such as `+.×`, which contracts the last axis of a left argument of this layout with the first axis of a right argument of `right`.
    pub fn contracted(&self, right: &Self) -> Self { self.axes(0..self.shape.len().saturating_sub(1)).concat(&right.axes(1..right.shape.len())) }
    /// The layout with `other`'s axes in place of `axes`.
    pub fn replace(&self, axes: std::ops::Range<usize>, other: &Self) -> Self {
        self.axes(0..axes.start).concat(other).concat(&self.axes(axes.end..self.shape.len()))
    }
    /// The layout with `axis` holding the positions it reads at `positions`, keeping their keys. `None` reads no position and has no key.
    pub fn select(&self, axis: usize, positions: impl ExactSizeIterator<Item = Option<usize>>) -> Result<Self, Error> {
        let mut layout = self.clone();
        layout.shape[axis] = positions.len();
        let Some(keys) = self.keys(axis) else { return Ok(layout) };
        let mut all = self.all_keys();
        all[axis] = Some(keys.select(positions)?);
        layout.with_keys(all)
    }
    /// Grows the first axis by `len` positions. `names` gives their keys. A Domain error, with the layout unchanged, when one of them is already on that axis or appears twice.
    pub fn append(&mut self, len: usize, names: Option<&[Option<Arc<str>>]>) -> Result<(), Error> {
        let (old, rank) = (self.shape[0], self.shape.len());
        if let Some(names) = names {
            match self.meta.as_mut().and_then(|m| m.keys.first_mut()).and_then(Option::as_mut) {
                Some(keys) => Arc::make_mut(keys).extend(names)?,
                None => {
                    let keys = Keys::partial(std::iter::repeat_n(None, old).chain(names.iter().cloned()).collect())?;
                    let meta = self.meta_mut();
                    if meta.keys.is_empty() { meta.keys = vec![None; rank]; }
                    meta.keys[0] = Some(keys);
                }
            }
        }
        self.shape[0] += len;
        Ok(())
    }
}
