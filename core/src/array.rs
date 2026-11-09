//! Borrowed array storage, layouts and positional reads.
use crate::agreement::{Binding, Mapping, Missing, Read, ReadPlan};
use crate::buffer::{Buffer, DType, Element, Items, View};
use crate::error::{Error, Kind};
use crate::layout::Layout;
use std::borrow::Cow;

/// Sources lent for one operation. The plan carries positions, never data or a materialization cache.
#[derive(Clone)]
pub struct ArrayView<'a> { layout: &'a Layout, sources: Vec<Binding<'a>>, plan: ReadPlan }

impl<'a> ArrayView<'a> {
    pub fn new(layout: &'a Layout, sources: Vec<Binding<'a>>, plan: ReadPlan) -> Result<Self, Error> {
        if layout.count() != plan.len() { return Err(Error::new(Kind::Length, "read plan doesn't fit its layout")); }
        Ok(Self { layout, sources, plan })
    }
    pub fn compact(layout: &'a Layout, storage: View<'a>) -> Self {
        Self { layout, sources: vec![Binding::full(storage)], plan: ReadPlan::input(0, storage.len()) }
    }
    pub fn layout(&self) -> &'a Layout { self.layout }
    pub fn dtype(&self) -> DType { self.sources[0].view.dtype() }
    pub fn len(&self) -> usize { self.plan.len() }
    pub fn is_empty(&self) -> bool { self.plan.is_empty() }
    pub fn read(&self) -> Read<'_> { Read { sources: &self.sources, plan: &self.plan } }
    pub fn plan(&self) -> &ReadPlan { &self.plan }
    pub fn sources(&self) -> &[Binding<'a>] { &self.sources }
    /// Changes metadata, retaining the logical sequence.
    pub fn with_layout<'b>(&self, layout: &'b Layout) -> Result<ArrayView<'b>, Error> where 'a: 'b {
        ArrayView::new(layout, self.sources.clone(), self.plan.clone())
    }
    /// Composes a call's positional read. Missing positions read the prototype (`Missing::prototype`).
    pub fn remapped(&self, mapping: Mapping, items: usize) -> Self {
        let mut sources = self.sources.clone();
        let missing = Missing::prototype(self.dtype(), &mut sources);
        Self { layout: self.layout, sources, plan: self.plan.remapped(mapping, items, missing) }
    }
    pub fn contiguous(&self) -> Option<View<'a>> {
        if self.is_empty() { return Some(self.sources[0].view.slice(0..0)); }
        let (source, range) = self.plan.contiguous(0..self.len())?;
        self.sources.get(source)?.borrow(range)
    }
    pub fn affine(&self) -> Option<(View<'a>, usize, Vec<isize>)> {
        let (source, mapping) = self.plan.positional(self.layout.shape())?;
        let (base, strides) = mapping.affine(self.layout.shape())?;
        let (view, base) = self.sources.get(source)?.affine(base, &strides, self.layout.shape())?;
        Some((view, base, strides))
    }
    /// Borrows a contiguous read at its dtype, otherwise prepares exactly these logical items.
    pub fn cast(&self, dtype: DType) -> Result<Items<'a>, Error> {
        if dtype == self.dtype() && let Some(view) = self.contiguous() { return Ok(Items::Borrowed(view)); }
        let mut buffer = Buffer::with_capacity(dtype, self.len());
        self.read().read_into(0..self.len(), dtype, true, &mut buffer)?;
        Ok(Items::Owned(buffer))
    }
    pub fn items<T: Element>(&self) -> Result<Cow<'a, [T]>, Error> { Ok(self.cast(T::DTYPE)?.items()) }
}
