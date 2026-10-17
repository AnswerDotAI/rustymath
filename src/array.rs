//! Value-semantic arrays: a layout and a flat mapping into shared compact storage. Structural views share storage; mutation detaches it.
use crate::buffer::{Buffer, DType, Element, Items, View};
use crate::agreement::Mapping;
use crate::error::{Error, Kind};
use crate::layout::{Layout, element_count};
use std::sync::Arc;
use std::borrow::Cow;

/// Borrowed logical items: a layout and a mapping into compact storage.
#[derive(Clone, Copy)]
pub struct ArrayView<'a> { layout: &'a Layout, storage: View<'a>, mapping: &'a Mapping }

impl<'a> ArrayView<'a> {
    /// Borrows storage through `mapping`, whose offsets must fit the storage or give fills.
    pub fn new(layout: &'a Layout, storage: View<'a>, mapping: &'a Mapping) -> Self { Self { layout, storage, mapping } }
    /// Borrows compact items in row-major order.
    pub fn compact(layout: &'a Layout, storage: View<'a>) -> Self { Self::new(layout, storage, &Mapping::Linear(1)) }
    pub fn layout(self) -> &'a Layout { self.layout }
    pub fn storage(self) -> View<'a> { self.storage }
    pub fn mapping(self) -> &'a Mapping { self.mapping }
    pub fn dtype(self) -> DType { self.storage.dtype() }
    pub fn len(self) -> usize { self.layout.shape().iter().product() }
    pub fn is_empty(self) -> bool { self.len() == 0 }
    /// The logical sequence as one borrowed slice, including a contiguous subrange of larger storage.
    pub fn contiguous(self) -> Option<View<'a>> {
        let n = self.len();
        if n == 0 { return Some(self.storage.slice(0..0)); }
        let start = match self.mapping {
            Mapping::Linear(1) => 0,
            _ if n == 1 => self.mapping.get(0)?,
            _ => { let (start, step) = self.mapping.run(0, n)?; if step != 1 { return None; } start },
        };
        Some(self.storage.slice(start..start + n))
    }
    /// Logical items at `dtype`, borrowed when already contiguous at that dtype, otherwise prepared in one buffer.
    pub fn cast(self, dtype: DType) -> Result<Items<'a>, Error> {
        if let Some(view) = self.contiguous() {
            return if view.dtype() == dtype { Ok(Items::Borrowed(view)) } else { Buffer::cast(view, dtype).map(Items::Owned) };
        }
        Buffer::cast_mapped(self.storage, self.mapping, self.len(), dtype).map(Items::Owned)
    }
    /// Typed spelling of [`Self::cast`], borrowing or moving its prepared items without another copy.
    pub fn items<T: Element>(self) -> Result<Cow<'a, [T]>, Error> { Ok(self.cast(T::DTYPE)?.items()) }
    /// Composes an input with its call's mapping, or prepares it once when conversion or an unsupported composition needs compact items.
    pub(crate) fn mapped(self, outer: &Mapping, frame: &[usize], dtype: DType) -> (Items<'a>, Option<Mapping>) {
        if self.dtype() == dtype {
            if let Some(view) = self.contiguous() { return (Items::Borrowed(view), None); }
            if let Some(mapping) = self.mapping.compose(outer, self.layout.shape(), frame) { return (Items::Borrowed(self.storage), Some(mapping)); }
        }
        (self.cast(dtype).expect("computing dtype"), None)
    }
}

/// A layout and a mapping into shared storage. A small view retains its whole base until [`Self::compact`].
#[derive(Clone, Debug)]
pub struct Array { layout: Layout, storage: Arc<Buffer>, mapping: Mapping }

impl Array {
    /// The array of `buffer` under `layout`. A Length error when the buffer doesn't hold one item for each position.
    pub fn new(layout: Layout, buffer: Buffer) -> Result<Self, Error> {
        let n = element_count(layout.shape())?;
        if buffer.len() != n { return Err(Error::new(Kind::Length, format!("{} items can't fill {n} positions", buffer.len()))); }
        Ok(Self { layout, storage: Arc::new(buffer), mapping: Mapping::Linear(1) })
    }
    #[inline]
    pub fn layout(&self) -> &Layout { &self.layout }
    /// Names the axes through [`Layout::with_names`]: one name or `None` for each axis, or none at all to remove them.
    pub fn set_names(&mut self, names: Vec<Option<Arc<str>>>) -> Result<(), Error> { self.layout = self.layout.clone().with_names(names)?; Ok(()) }
    #[inline]
    pub fn view(&self) -> ArrayView<'_> { ArrayView::new(&self.layout, self.storage.view(), &self.mapping) }
    #[inline]
    pub fn dtype(&self) -> DType { self.storage.dtype() }
    #[inline]
    pub fn len(&self) -> usize { self.view().len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.view().is_empty() }
    fn detach(&mut self) {
        if self.len() != self.storage.len() || self.view().contiguous().is_none() {
            self.storage = Arc::new(Buffer::cast_mapped(self.storage.view(), &self.mapping, self.len(), self.dtype()).expect("same dtype"));
        }
        self.mapping = Mapping::Linear(1);
    }
    /// Compact storage with no retained items outside this array, reusing a uniquely held full allocation.
    pub fn into_compact(mut self) -> (Layout, Buffer) { self.detach(); (self.layout, Arc::unwrap_or_clone(self.storage)) }
    /// An explicit retention escape: even a contiguous subrange stops retaining its larger base.
    pub fn compact(&self) -> Self { let mut out = self.clone(); out.detach(); out }
    /// Writable compact items, detached from views, exports and any other owners.
    pub fn make_mut(&mut self) -> &mut Buffer { self.detach(); Arc::make_mut(&mut self.storage) }
    /// Shares an affine structural selection, or gathers through its composed mapping. Unsupported compositions compact first.
    pub fn selected(&self, plan: &crate::structure::Plan) -> Result<Self, Error> {
        if let Some(mapping) = self.mapping.through(&plan.steps, self.layout.shape()) {
            if mapping.affine(plan.layout.shape()).is_some() { return Ok(Self { layout: plan.layout.clone(), storage: self.storage.clone(), mapping }); }
            let buffer = Buffer::cast_mapped(self.storage.view(), &mapping, element_count(plan.layout.shape())?, self.dtype())?;
            return Self::new(plan.layout.clone(), buffer);
        }
        Self::gathered(self.view().cast(self.dtype())?.view(), plan)
    }
    /// Changes the layout without copying compact items. A noncontiguous sequence compacts once.
    pub fn reshape(&self, layout: Layout) -> Result<Self, Error> {
        if element_count(layout.shape())? != self.len() { return Err(Error::new(Kind::Length, "reshape must retain the number of items")); }
        let mut out = if self.view().contiguous().is_some() { self.clone() } else { self.compact() };
        let base = if out.is_empty() { 0 } else { out.mapping.get(0).expect("compact items") };
        out.mapping = if base == 0 { Mapping::Linear(1) } else { Mapping::Steps { base, steps: vec![crate::gather::Steps::along(out.len(), 1)] } };
        out.layout = layout;
        Ok(out)
    }
    /// Copies a structural plan from borrowed compact items. Missing positions take the element type's fill.
    pub fn gathered(source: View<'_>, plan: &crate::structure::Plan) -> Result<Self, Error> {
        let buffer = Buffer::walked(source, 0, &plan.steps, element_count(plan.layout.shape())?);
        Self::new(plan.layout.clone(), buffer)
    }
    /// Joins arrays of one dtype along an axis, retaining the layout rules of [`crate::structure::catenate`].
    pub fn concatenate(arrays: &[ArrayView<'_>], axis: usize) -> Result<Self, Error> {
        let Some(&source) = arrays.first() else { return Err(Error::new(Kind::Length, "concatenate needs an array")) };
        let mut layout = source.layout.clone();
        for &view in &arrays[1..] {
            if view.dtype() != source.dtype() { return Err(Error::new(Kind::Domain, "concatenate items must have one dtype")); }
            layout = crate::structure::catenate(&layout, view.layout, axis)?;
        }
        let mut buffer = Buffer::with_capacity(source.dtype(), crate::layout::generated_len(layout.shape())?);
        let lanes = arrays.iter().map(|a| crate::layout::Lanes::new(a.layout.shape(), axis)).collect::<Result<Vec<_>, _>>()?;
        let items = arrays.iter().map(|a| a.cast(source.dtype())).collect::<Result<Vec<_>, _>>()?;
        for outer in 0..lanes[0].outer {
            for (items, lane) in items.iter().zip(&lanes) {
                let n = lane.len * lane.inner;
                buffer.extend_from(items.view(), outer * n..(outer + 1) * n);
            }
        }
        Self::new(layout, buffer)
    }
    /// Chooses between two arrays by a Boolean mask, using ordinary named/keyed broadcasting for all three arguments.
    pub fn choose(condition: ArrayView<'_>, x: ArrayView<'_>, y: ArrayView<'_>) -> Result<Self, Error> {
        use crate::agreement::Agreement;
        let mask = bool::of(condition.storage).ok_or_else(|| Error::new(Kind::Domain, "condition must be Boolean"))?;
        if x.dtype() != y.dtype() { return Err(Error::new(Kind::Domain, "chosen items must have one dtype")); }
        let pair = Agreement::new(x.layout, y.layout)?;
        let whole = Agreement::new(condition.layout, &pair.layout)?;
        let buffer = with_dtype!(x.dtype(), T => {
            let (xs, ys) = (T::of(x.storage).expect("matching dtype"), T::of(y.storage).expect("matching dtype"));
            let mut out = Vec::with_capacity(whole.len);
            crate::parallel::extended(&mut out, whole.len, crate::CHUNK, &|range, slots| slots.fill(range.len(), |slots| {
                for (i, slot) in range.zip(slots) {
                    let take_x = whole.left.get(i).and_then(|j| condition.mapping.get(j)).is_some_and(|j| mask[j]);
                    slot.write(whole.right.get(i).and_then(|j| if take_x { pair.left.get(j).and_then(|j| x.mapping.get(j)).map(|j| xs[j]) }
                        else { pair.right.get(j).and_then(|j| y.mapping.get(j)).map(|j| ys[j]) }).unwrap_or(T::FILL));
                }
                true
            }));
            T::buffer(out)
        });
        Self::new(whole.layout, buffer)
    }
    /// Broadcasts borrowed items into a compact array of `shape`, pairing axes from the front and retaining matching metadata.
    pub fn broadcasted(source: ArrayView<'_>, shape: &[usize]) -> Result<Self, Error> {
        let agreement = crate::agreement::Agreement::new(&Layout::from(shape), source.layout)?;
        if agreement.layout.shape() != shape { return Err(Error::new(Kind::Length, "source doesn't broadcast to this shape")); }
        let (items, mapping) = source.mapped(&agreement.right, agreement.layout.shape(), source.dtype());
        let buffer = with_dtype!(source.dtype(), T => {
            let data = T::of(items.view()).expect("matching dtype");
            let mut out = Vec::with_capacity(agreement.len);
            crate::agreement::block(data, mapping.as_ref().unwrap_or(&agreement.right), agreement.len, &mut out, |items, n, slots| {
                match items {
                    crate::Arg::One(item) => { slots.fill(n, |slots| { slots.fill(std::mem::MaybeUninit::new(item)); true }); }
                    crate::Arg::Many(items) => slots.copy(items),
                }
                true
            });
            T::buffer(out)
        });
        Self::new(agreement.layout, buffer)
    }
    /// Adjacent differences along one axis, through the ordinary subtraction kernels without two gathered copies. Booleans give XOR. Results retain the following positions' keys.
    pub fn differences(source: ArrayView<'_>, axis: usize) -> Result<Self, Error> {
        use crate::{agreement::{Agreement, Mapping}, structure::{self, Selector}};
        let layout = source.layout;
        let &size = layout.shape().get(axis).ok_or_else(|| Error::new(Kind::Index, "difference axis is out of range"))?;
        let n = size.saturating_sub(1);
        let select = |start| structure::select_axis(layout, axis, Selector::Range { start, len: n, step: 1 });
        let (next, previous) = (select(usize::from(size > 0))?, select(0)?);
        let len = crate::layout::element_count(next.layout.shape())?;
        let agreement = Agreement { layout: next.layout, len, left: Mapping::Steps { base: 0, steps: next.steps }, right: Mapping::Steps { base: 0, steps: previous.steps } };
        let result = if source.dtype() == DType::Boolean { let k = crate::reduce::Xor; crate::kernels!(Binary of k).binary(&k, source, source, &agreement) }
            else { let k = crate::Subtract; crate::kernels!(Binary of k).binary(&k, source, source, &agreement) };
        let (buffer, ok) = result.ok_or_else(|| Error::new(Kind::Domain, "differences need numerical items"))?;
        if !ok && source.dtype().is_integral() { return Err(Kind::Overflow.into()); }
        Self::new(agreement.layout, buffer)
    }
    /// Writes `source`, broadcast over a selection, detaching shared storage first. Later repeated targets win. Conversion and shape checks finish before any item changes.
    pub fn assign(&mut self, plan: &crate::structure::Plan, source: &Self) -> Result<(), Error> {
        let scalar = Layout::EMPTY;
        let agreement = crate::agreement::Agreement::new(&plan.layout, if source.len() == 1 { &scalar } else { source.layout() })?;
        if agreement.layout.shape() != plan.layout.shape() { return Err(Error::new(Kind::Length, "assignment must fit the selected layout")); }
        let offsets = (0..agreement.len).map(|i| crate::gather::Steps::reach(0, &plan.steps, i).filter(|&o| o < self.len())).collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::new(Kind::Index, "assignment position is outside its array"))?;
        let items = source.view().cast(self.dtype())?;
        let values = items.view();
        with_dtype!(self.dtype(), T => {
            let (out, values) = (T::of_mut(self.make_mut()).expect("matching dtype"), T::of(values).expect("matching dtype"));
            for (i, &o) in offsets.iter().enumerate() { out[o] = agreement.right.get(i).map_or(T::FILL, |j| values[j]); }
        });
        Ok(())
    }
}
