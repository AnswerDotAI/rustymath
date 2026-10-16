//! Rustymath's array: a [`Layout`] paired with a [`Buffer`] that holds one item for each of its positions, in row-major order.
use crate::buffer::{Buffer, DType, View};
use crate::error::{Error, Kind};
use crate::layout::{Layout, element_count};
use std::sync::Arc;

/// A layout and a buffer with one item for each of its positions, in row-major order.
#[derive(Clone, Debug)]
pub struct Array { layout: Layout, buffer: Buffer }

impl Array {
    /// The array of `buffer` under `layout`. A Length error when the buffer doesn't hold one item for each position.
    pub fn new(layout: Layout, buffer: Buffer) -> Result<Self, Error> {
        let n = element_count(layout.shape())?;
        if buffer.len() != n { return Err(Error::new(Kind::Length, format!("{} items can't fill {n} positions", buffer.len()))); }
        Ok(Self { layout, buffer })
    }
    #[inline]
    pub fn layout(&self) -> &Layout { &self.layout }
    /// Names the axes through [`Layout::with_names`]: one name or `None` for each axis, or none at all to remove them.
    pub fn set_names(&mut self, names: Vec<Option<Arc<str>>>) -> Result<(), Error> { self.layout = self.layout.clone().with_names(names)?; Ok(()) }
    #[inline]
    pub fn view(&self) -> View<'_> { self.buffer.view() }
    #[inline]
    pub fn dtype(&self) -> DType { self.buffer.dtype() }
    #[inline]
    pub fn len(&self) -> usize { self.buffer.len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.buffer.is_empty() }
    pub fn into_parts(self) -> (Layout, Buffer) { (self.layout, self.buffer) }
    /// The address of the first item, for writing through, as an exported buffer may.
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.buffer.as_mut_ptr() }
}
