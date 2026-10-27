//! Value-semantic arrays: a layout and a flat mapping into shared compact storage. Structural views share storage; mutation detaches it.
use crate::buffer::{Buffer, DType, Element, Fill, Items, View};
use crate::agreement::{Mapping, small_read};
use crate::error::{Error, Kind};
use crate::gather::{Steps, Writes};
use crate::layout::{Keys, Lanes, Layout, element_count};
use std::sync::{Arc, OnceLock};
use std::borrow::Cow;

/// Borrowed logical items: a layout and a mapping into compact storage, with the cache of an [`Array`] view that keeps the items it gathers ([`ArrayView::keeps`]).
#[derive(Clone, Copy)]
pub struct ArrayView<'a> { layout: &'a Layout, storage: View<'a>, mapping: &'a Mapping, kept: Option<&'a Kept> }

impl<'a> ArrayView<'a> {
    /// Borrows storage through `mapping`, whose offsets must fit the storage or give fills.
    pub fn new(layout: &'a Layout, storage: View<'a>, mapping: &'a Mapping) -> Self { Self { layout, storage, mapping, kept: None } }
    /// Borrows compact items in row-major order.
    pub fn compact(layout: &'a Layout, storage: View<'a>) -> Self { Self::new(layout, storage, &Mapping::Linear(1)) }
    pub fn layout(self) -> &'a Layout { self.layout }
    pub fn storage(self) -> View<'a> { self.storage }
    pub fn mapping(self) -> &'a Mapping { self.mapping }
    pub fn dtype(self) -> DType { self.storage.dtype() }
    pub fn len(self) -> usize { self.layout.count() }
    pub fn is_empty(self) -> bool { self.len() == 0 }
    /// Whether a whole read at this view's own dtype keeps the items it gathers for later reads: true for a view of an [`Array`] that isn't affine, until its first such read.
    pub fn keeps(self) -> bool { self.kept.is_some() }
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
    /// Logical items at `dtype`, borrowed when already contiguous at that dtype, otherwise prepared in one buffer. A view that keeps what it gathers ([`Self::keeps`]) gathers into its cache at its own dtype, once, and lends the cache.
    pub fn cast(self, dtype: DType) -> Result<Items<'a>, Error> {
        if let Some(view) = self.contiguous() {
            return if view.dtype() == dtype { Ok(Items::Borrowed(view)) } else { Buffer::cast(view, dtype).map(Items::Owned) };
        }
        if let Some(kept) = self.kept && dtype == self.dtype() { return Ok(Items::Borrowed(self.gather(kept, true).view())); }
        Buffer::cast_mapped(self.storage, self.mapping, self.len(), dtype).map(Items::Owned)
    }
    /// This view's items at its own dtype, gathered into `kept` by the first read that needs them, and shared by later reads. A `reader`'s gather is a decline of reading in place (`events::declined`). A copy that a call asks for, as `compact()` and writes make, isn't.
    fn gather(self, kept: &'a Kept, reader: bool) -> &'a Arc<Buffer> {
        kept.get_or_init(|| {
            if reader { crate::events::declined(crate::events::Route::InPlace, "a view that gathers", self.len()); }
            Arc::new(Buffer::cast_mapped(self.storage, self.mapping, self.len(), self.dtype()).expect("same dtype"))
        })
    }
    /// Typed spelling of [`Self::cast`], borrowing or moving its prepared items without another copy.
    pub fn items<T: Element>(self) -> Result<Cow<'a, [T]>, Error> { Ok(self.cast(T::DTYPE)?.items()) }
    /// The items that a fold over axis `axis`, or over every item for `None`, reads, with the `[outer, len, inner]` dimensions to fold them along their middle, and the tables that read the totals in the order of the other axes, where they come in another order. Where the array reads a block of its storage, as [`Mapping::block`] finds for compact arrays, transposes, reversals and windows that don't overlap, it lends the block, which folds in its own order ([`Block::fold`](crate::agreement::Block::fold)). Any other view gathers its items at its own dtype.
    pub fn fold_items(self, axis: Option<usize>) -> Result<(Items<'a>, [usize; 3], Option<Vec<Steps>>), Error> {
        let shape = self.layout.shape();
        if let Some(block) = self.mapping.block(shape) {
            let (dims, order) = block.fold(axis);
            return Ok((Items::Borrowed(self.storage.slice(block.range)), dims, order));
        }
        let dims = match axis { None => [1, self.len(), 1], Some(a) => Lanes::new(shape, a)?.dims() };
        Ok((self.cast(self.dtype())?, dims, None))
    }
    /// Composes an input with its call's mapping, or prepares it once when conversion or an unsupported composition needs compact items.
    pub(crate) fn mapped(self, outer: &Mapping, frame: &[usize], dtype: DType) -> (Items<'a>, Option<Mapping>) {
        if self.dtype() == dtype {
            if let Some(view) = self.contiguous() { return (Items::Borrowed(view), None); }
            if let Some(mapping) = self.mapping.compose(outer, self.layout.shape(), frame) { return (Items::Borrowed(self.storage), Some(mapping)); }
        }
        (self.cast(dtype).expect("computing dtype"), outer.cast_fill(dtype).expect("computing fill dtype"))
    }
}

/// A layout and a mapping into shared storage. A small view retains its whole base until [`Self::compact`]. A view that isn't affine also keeps the items that its first whole read gathers, and lends them to later reads ([`ArrayView::keeps`]).
#[derive(Clone, Debug)]
pub struct Array { layout: Layout, storage: Arc<Buffer>, mapping: Mapping, kept: Option<Arc<Kept>> }

/// The items that a view gathers, kept for its later reads.
type Kept = OnceLock<Arc<Buffer>>;

impl Array {
    /// The array of `buffer` under `layout`. A Length error when the buffer doesn't hold one item for each position.
    pub fn new(layout: Layout, buffer: Buffer) -> Result<Self, Error> {
        let n = element_count(layout.shape())?;
        if buffer.len() != n { return Err(Error::new(Kind::Length, format!("{} items can't fill {n} positions", buffer.len()))); }
        Ok(Self { layout, storage: Arc::new(buffer), mapping: Mapping::Linear(1), kept: None })
    }
    #[inline]
    pub fn layout(&self) -> &Layout { &self.layout }
    /// Names the axes through [`Layout::with_names`]: one name or `None` for each axis, or none at all to remove them.
    pub fn set_names(&mut self, names: Vec<Option<Arc<str>>>) -> Result<(), Error> { self.layout = self.layout.clone().with_names(names)?; Ok(()) }
    /// Labels axis positions through [`Layout::with_keys`], without changing the values or their order.
    pub fn set_keys(&mut self, keys: Vec<Option<Arc<Keys>>>) -> Result<(), Error> { self.layout = self.layout.clone().with_keys(keys)?; Ok(()) }
    /// The items as a borrowed view, which reads the compact items that a view has gathered once it has them.
    #[inline]
    pub fn view(&self) -> ArrayView<'_> {
        let (storage, mapping) = self.parts();
        ArrayView { layout: &self.layout, storage: storage.view(), mapping, kept: self.kept.as_deref().filter(|kept| kept.get().is_none()) }
    }
    /// This array's storage and mapping, or the compact items that a view has gathered, once it has them.
    #[inline]
    fn parts(&self) -> (&Arc<Buffer>, &Mapping) {
        match self.kept.as_deref().and_then(OnceLock::get) { Some(items) => (items, &Mapping::Linear(1)), None => (&self.storage, &self.mapping) }
    }
    #[inline]
    pub fn dtype(&self) -> DType { self.storage.dtype() }
    #[inline]
    pub fn len(&self) -> usize { self.view().len() }
    #[inline]
    pub fn is_empty(&self) -> bool { self.view().is_empty() }
    /// The items in order, as compact storage of their own: the storage itself where it holds just these items in order, the items that a view keeps, gathered once, or a new copy.
    fn compacted(&self) -> Arc<Buffer> {
        let (storage, _) = self.parts();
        let view = self.view();
        if let Some(kept) = view.kept { return view.gather(kept, false).clone(); }
        if view.len() == storage.len() && view.contiguous().is_some() { return storage.clone(); }
        Arc::new(Buffer::cast_mapped(view.storage, view.mapping, view.len(), view.dtype()).expect("same dtype"))
    }
    fn detach(&mut self) { (self.storage, self.mapping, self.kept) = (self.compacted(), Mapping::Linear(1), None); }
    /// Compact storage with no retained items outside this array, reusing a uniquely held full allocation.
    pub fn into_compact(mut self) -> (Layout, Buffer) { self.detach(); (self.layout, Arc::unwrap_or_clone(self.storage)) }
    /// An explicit retention escape: even a contiguous subrange stops retaining its larger base.
    pub fn compact(&self) -> Self { let mut out = self.clone(); out.detach(); out }
    /// Writable compact items, detached from views, exports and any other owners.
    pub fn make_mut(&mut self) -> &mut Buffer { self.detach(); Arc::make_mut(&mut self.storage) }
    /// Selects by `plan`, reading `fill`, or the item type's fill, where the plan reaches no item. A selection by strides, padding or rotations is a view of this array's storage, and one by a table of positions copies its items. Where the plan doesn't compose with this view's mapping, a small selection lists its offsets ([`small_read`]), and a larger one gathers this view's items first.
    pub fn selected(&self, plan: crate::structure::Plan, fill: Option<Arc<Fill>>) -> Result<Self, Error> {
        let n = element_count(plan.layout.shape())?;
        let outer = Mapping::from(plan.steps).filled(fill);
        let (storage, mapping) = self.parts();
        let (storage, mapping) = match mapping.composed(&outer, self.layout.shape(), plan.layout.shape(), small_read(n, self.len())) {
            Some(mapping) => (storage.clone(), mapping),
            None => { crate::events::declined(crate::events::Route::InPlace, "a selection that doesn't compose", n); (self.compacted(), outer) }
        };
        if mapping.tabled() { return Self::new(plan.layout, Buffer::cast_mapped(storage.view(), &mapping, n, self.dtype())?); }
        Ok(Self::viewed(plan.layout, storage, mapping))
    }
    /// Changes the layout without copying. Compact items keep one plain run, and a view keeps its mapping, which reads the same items in order under any layout.
    pub fn reshape(&self, layout: Layout) -> Result<Self, Error> {
        if element_count(layout.shape())? != self.len() { return Err(Error::new(Kind::Length, "reshape must retain the number of items")); }
        let (storage, mapping) = self.parts();
        if self.view().contiguous().is_some() {
            let base = if self.is_empty() { 0 } else { mapping.get(0).expect("compact items") };
            let mapping = if base == 0 { Mapping::Linear(1) } else { Mapping::Steps { base, steps: vec![Steps::along(self.len(), 1)].into() } };
            return Ok(Self { layout, storage: storage.clone(), mapping, kept: None });
        }
        let kept = mapping.affine(layout.shape()).is_none().then(|| self.kept.clone().unwrap_or_default());
        Ok(Self { layout, storage: storage.clone(), mapping: mapping.clone(), kept })
    }
    /// A view of `storage` through `mapping` under `layout`, which keeps the items its first whole read gathers where the mapping isn't affine.
    fn viewed(layout: Layout, storage: Arc<Buffer>, mapping: Mapping) -> Self {
        let kept = mapping.affine(layout.shape()).is_none().then(Arc::default);
        Self { layout, storage, mapping, kept }
    }
    /// The items through a view that strides describe, with no fills, as a buffer export reads them: this array's own view where it's compact or affine, and otherwise the items that it keeps, gathered once.
    pub fn strided(&self) -> ArrayView<'_> {
        let view = self.view();
        let Some(kept) = view.kept else { return view };
        view.gather(kept, true);
        self.view()
    }
    /// Copies a structural plan from borrowed compact items. Missing positions take `fill`, or the item type's fill.
    pub fn gathered(source: View<'_>, plan: &crate::structure::Plan, fill: Option<Arc<Fill>>) -> Result<Self, Error> {
        let buffer = Buffer::cast_mapped(source, &Mapping::from(plan.steps.clone()).filled(fill), element_count(plan.layout.shape())?, source.dtype())?;
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
        let lanes = arrays.iter().map(|a| Lanes::new(a.layout.shape(), axis)).collect::<Result<Vec<_>, _>>()?;
        let lens: Vec<usize> = lanes.iter().map(|l| l.len * l.inner).collect();
        let (outer, stride) = (lanes[0].outer, lens.iter().sum::<usize>());
        let buffer = with_dtype!(source.dtype(), T => {
            let parts = arrays.iter().map(|a| a.items::<T>()).collect::<Result<Vec<_>, _>>()?;
            // The result starts as copies of one item. Each array's lanes then replace them, in one typed loop for each array.
            let mut out = parts.iter().find_map(|p| p.first()).map_or_else(Vec::new, |&x| vec![x; outer * stride]);
            let mut at = 0;
            for (part, &n) in parts.iter().zip(&lens) {
                crate::gather::scatter_walk(&mut out, part, at, &[Steps::along(outer, stride), Steps::along(n, 1)], crate::gather::Same);
                at += n;
            }
            T::buffer(out)
        });
        Self::new(layout, buffer)
    }
    /// Chooses between two arrays by a Boolean mask, through [`Choose`](crate::Choose). `x` and `y` agree by the usual rules for names and keys, and the mask lays onto their result through [`Agreement::onto`](crate::agreement::Agreement::onto): a key that only the mask has is ignored, and a key that the mask lacks picks `y`.
    pub fn choose(condition: ArrayView<'_>, x: ArrayView<'_>, y: ArrayView<'_>) -> Result<Self, Error> {
        use crate::agreement::Agreement;
        if condition.dtype() != DType::Boolean { return Err(Error::new(Kind::Domain, "condition must be Boolean")); }
        if x.dtype() != y.dtype() { return Err(Error::new(Kind::Domain, "chosen items must have one dtype")); }
        let pair = Agreement::new(x.layout, y.layout)?;
        let whole = Agreement::onto(&pair.layout, condition.layout)?;
        let k = crate::Choose;
        let (buffer, _) = crate::kernels!(Ternary of k).ternary(&k, [x, y, condition], &pair, &whole).expect("`Choose` has a kernel at every dtype");
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
    /// Adjacent differences along one axis, through the ordinary subtraction kernels without two gathered copies. Booleans give XOR. Integer differences wrap to their type where they don't fit. Results retain the following positions' keys.
    pub fn differences(source: ArrayView<'_>, axis: usize) -> Result<Self, Error> {
        use crate::{agreement::{Agreement, Mapping}, structure::{self, Selector}};
        let layout = source.layout;
        let &size = layout.shape().get(axis).ok_or_else(|| Error::new(Kind::Index, "difference axis is out of range"))?;
        let n = size.saturating_sub(1);
        let select = |start| structure::select_axis(layout, axis, Selector::Range { start, len: n, step: 1 });
        let (next, previous) = (select(usize::from(size > 0))?, select(0)?);
        let len = crate::layout::element_count(next.layout.shape())?;
        let agreement = Agreement { layout: next.layout, len, left: Mapping::Steps { base: 0, steps: next.steps.into() }, right: Mapping::Steps { base: 0, steps: previous.steps.into() } };
        let result = if source.dtype() == DType::Boolean { let k = crate::reduce::Xor; crate::kernels!(Binary of k).binary(&k, source, source, &agreement) }
            else { let k = crate::Subtract; crate::kernels!(Binary of k).binary(&k, source, source, &agreement) };
        let (buffer, _) = result.ok_or_else(|| Error::new(Kind::Domain, "differences need numerical items"))?;
        Self::new(agreement.layout, buffer)
    }
    /// Where a selection writes, as `plan`'s walk, and `source`'s items at this array's dtype: one item for every write, or `source` broadcast over the selection, one item for each write in the same order. A fill from broadcasting reads the dtype's fill. Shape, position and conversion checks all finish here, before any item changes.
    fn targets<'p, 's>(&self, plan: &'p crate::structure::Plan, source: &'s Self) -> Result<(Writes<'p>, Items<'s>), Error> {
        let scalar = Layout::EMPTY;
        let agreement = crate::agreement::Agreement::new(&plan.layout, if source.len() == 1 { &scalar } else { source.layout() })?;
        if agreement.layout.shape() != plan.layout.shape() { return Err(Error::new(Kind::Length, "assignment must fit the selected layout")); }
        let writes = Writes::Walk { base: 0, steps: &plan.steps };
        let items = source.view().cast(self.dtype())?;
        if matches!(agreement.right, Mapping::Linear(1) | Mapping::Single) { return Ok((writes, items)); }
        Ok((writes, Items::Owned(Buffer::cast_mapped(items.view(), &agreement.right, agreement.len, self.dtype())?)))
    }
    /// Writes `source`, broadcast over a selection, detaching shared storage first. Later repeated targets win. `plan` must come from [`structure::select`](crate::structure::select) on this array's layout, with selectors whose positions were checked as they were built, as [`structure::index_table`](crate::structure::index_table) checks them. Those position checks, and the conversion and shape checks here, finish before any item changes. A plan from another array panics at its first write outside the array.
    pub fn assign(&mut self, plan: &crate::structure::Plan, source: &Self) -> Result<(), Error> {
        let (writes, items) = self.targets(plan, source)?;
        let values = items.view();
        with_dtype!(self.dtype(), T => {
            let (out, values) = (T::of_mut(self.make_mut()).expect("matching dtype"), T::of(values).expect("matching dtype"));
            writes.write(out, values, crate::gather::Same);
        });
        Ok(())
    }
    /// Combines `source`, broadcast over a selection, into the items it selects by `k`, in turn, detaching shared storage first. A target selected more than once combines every item that selects it, as NumPy's `ufunc.at` does. `kernels` are `k`'s, from [`kernels!`](crate::kernels)`(Scatter of k)`. It gives whether every result succeeded, as [`Kernels::scatter`](crate::operation::Kernels::scatter) says. Its `plan` and checks are as [`Array::assign`] gives them. A Domain error where `k` has no kernel at this array's dtype.
    pub fn combine_at<K>(&mut self, k: &K, kernels: &crate::operation::Kernels<crate::operation::ScatterKernel<K>>, plan: &crate::structure::Plan, source: &Self) -> Result<bool, Error> {
        let dtype = self.dtype();
        if kernels.at(dtype).is_none() { return Err(Error::new(Kind::Domain, format!("this function doesn't take {}", dtype.name()))); }
        let (writes, items) = self.targets(plan, source)?;
        Ok(kernels.scatter(k, self.make_mut(), writes, items.view()).expect("the function has a kernel at the array's dtype"))
    }
}

/// A contiguous array of whole numbers serves as index values that a selection's table reads in place, as [`Array::compact`] gives one.
impl crate::gather::IndexSource for Array {
    fn view(&self) -> View<'_> { Array::view(self).contiguous().expect("index values are contiguous") }
}
