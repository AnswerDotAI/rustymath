//! Rustymath's array: a [`Layout`] paired with a [`Buffer`] that holds one item for each of its positions, in row-major order.
use crate::buffer::{Buffer, DType, Element, View};
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
    /// Copies the items of a structural plan, including its keys and names. Missing positions read the element type's fill.
    pub fn gather(&self, plan: &crate::structure::Plan) -> Result<Self, Error> {
        Self::gathered(self.view(), plan)
    }
    /// [`Self::gather`] from borrowed compact items, without first copying the input.
    pub fn gathered(source: View<'_>, plan: &crate::structure::Plan) -> Result<Self, Error> {
        let buffer = Buffer::walked(source, &plan.steps);
        Self::new(plan.layout.clone(), buffer)
    }
    /// Joins compact arrays of one dtype along an axis, retaining the layout rules of [`crate::structure::catenate`].
    pub fn concatenate(arrays: &[(&Layout, View<'_>)], axis: usize) -> Result<Self, Error> {
        let Some(&(first, source)) = arrays.first() else { return Err(Error::new(Kind::Length, "concatenate needs an array")) };
        let mut layout = first.clone();
        for &(next, view) in &arrays[1..] {
            if view.dtype() != source.dtype() { return Err(Error::new(Kind::Domain, "concatenate items must have one dtype")); }
            layout = crate::structure::catenate(&layout, next, axis)?;
        }
        let mut buffer = Buffer::with_capacity(source.dtype(), crate::layout::generated_len(layout.shape())?);
        let lanes = arrays.iter().map(|(l, _)| crate::layout::Lanes::new(l.shape(), axis)).collect::<Result<Vec<_>, _>>()?;
        for outer in 0..lanes[0].outer {
            for ((_, view), lane) in arrays.iter().zip(&lanes) {
                let n = lane.len * lane.inner;
                buffer.extend_from(*view, outer * n..(outer + 1) * n);
            }
        }
        Self::new(layout, buffer)
    }
    /// Chooses between two arrays by a Boolean mask, using ordinary named/keyed broadcasting for all three arguments.
    pub fn choose(condition: (&Layout, &[bool]), x: (&Layout, View<'_>), y: (&Layout, View<'_>)) -> Result<Self, Error> {
        use crate::agreement::Agreement;
        if x.1.dtype() != y.1.dtype() { return Err(Error::new(Kind::Domain, "chosen items must have one dtype")); }
        let pair = Agreement::new(x.0, y.0)?;
        let whole = Agreement::new(condition.0, &pair.layout)?;
        let buffer = with_dtype!(x.1.dtype(), T => {
            let (xs, ys) = (T::of(x.1).expect("matching dtype"), T::of(y.1).expect("matching dtype"));
            let mut out = Vec::with_capacity(whole.len);
            crate::parallel::extended(&mut out, whole.len, crate::CHUNK, &|range, slots| slots.fill(range.len(), |slots| {
                for (i, slot) in range.zip(slots) {
                    let take_x = whole.left.get(i).is_some_and(|j| condition.1[j]);
                    slot.write(whole.right.get(i).and_then(|j| if take_x { pair.left.get(j).map(|j| xs[j]) } else { pair.right.get(j).map(|j| ys[j]) }).unwrap_or(T::FILL));
                }
                true
            }));
            T::buffer(out)
        });
        Self::new(whole.layout, buffer)
    }
    /// Broadcasts borrowed items into a compact array of `shape`, pairing axes from the front and retaining matching metadata.
    pub fn broadcasted(layout: &Layout, source: View<'_>, shape: &[usize]) -> Result<Self, Error> {
        let agreement = crate::agreement::Agreement::new(&Layout::from(shape), layout)?;
        if agreement.layout.shape() != shape { return Err(Error::new(Kind::Length, "source doesn't broadcast to this shape")); }
        let buffer = with_dtype!(source.dtype(), T => {
            let data = T::of(source).expect("matching dtype");
            let mut out = Vec::with_capacity(agreement.len);
            crate::agreement::block(data, &agreement.right, agreement.len, &mut out, |items, n, slots| {
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
    pub fn differences(layout: &Layout, source: View<'_>, axis: usize) -> Result<Self, Error> {
        use crate::{agreement::{Agreement, Mapping}, structure::{self, Selector}};
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
    /// Writes `source`, broadcast over a selection, without changing this array's allocation or layout. Later repeated targets win. Conversion and shape checks finish before any item changes.
    pub fn assign(&mut self, plan: &crate::structure::Plan, source: &Self) -> Result<(), Error> {
        let scalar = Layout::EMPTY;
        let agreement = crate::agreement::Agreement::new(&plan.layout, if source.len() == 1 { &scalar } else { source.layout() })?;
        if agreement.layout.shape() != plan.layout.shape() { return Err(Error::new(Kind::Length, "assignment must fit the selected layout")); }
        let offsets = (0..agreement.len).map(|i| crate::gather::Steps::reach(0, &plan.steps, i).filter(|&o| o < self.len())).collect::<Option<Vec<_>>>()
            .ok_or_else(|| Error::new(Kind::Index, "assignment position is outside its array"))?;
        let cast = (source.dtype() != self.dtype()).then(|| Buffer::cast(source.view(), self.dtype())).transpose()?;
        let values = cast.as_ref().map_or_else(|| source.view(), Buffer::view);
        with_dtype!(self.dtype(), T => {
            let (out, values) = (T::of_mut(&mut self.buffer).expect("matching dtype"), T::of(values).expect("matching dtype"));
            for (i, &o) in offsets.iter().enumerate() { out[o] = agreement.right.get(i).map_or(T::FILL, |j| values[j]); }
        });
        Ok(())
    }
    /// The address of the first item, for writing through, as an exported buffer may.
    pub fn as_mut_ptr(&mut self) -> *mut u8 { self.buffer.as_mut_ptr() }
}
