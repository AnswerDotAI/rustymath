//! Value-semantic arrays: a layout and a read plan over shared compact sources. Views compose their reads; explicit copies and writes detach storage.
use crate::agreement::{Agreement, Binding, Mapping, Missing, Read, ReadPlan, ReadRoute};
use crate::buffer::{Buffer, DType, Element, Items, View};
use crate::error::{Error, Kind};
use crate::gather::Steps;
use crate::layout::{Keys, Lanes, Layout, element_count};
use crate::operation::{Kernel, Positions, Settings, WriteRoot};
use std::borrow::Cow;
use std::sync::Arc;

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
        let (source, base, strides) = self.plan.affine(self.layout.shape())?;
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

/// A value keeps its sources alive; reading a view never replaces its plan or caches a gathered copy.
#[derive(Clone, Debug)]
pub struct Array { layout: Layout, sources: Arc<[Arc<Buffer>]>, plan: ReadPlan }

impl Array {
    pub fn new(layout: Layout, buffer: Buffer) -> Result<Self, Error> {
        let n = element_count(layout.shape())?;
        if buffer.len() != n { return Err(Error::new(Kind::Length, format!("{} items can't fill {n} positions", buffer.len()))); }
        Ok(Self { layout, sources: Arc::from([Arc::new(buffer)]), plan: ReadPlan::input(0, n) })
    }
    pub fn layout(&self) -> &Layout { &self.layout }
    pub fn set_names(&mut self, names: Vec<Option<Arc<str>>>) -> Result<(), Error> { self.layout = self.layout.clone().with_names(names)?; Ok(()) }
    pub fn set_keys(&mut self, keys: Vec<Option<Arc<Keys>>>) -> Result<(), Error> { self.layout = self.layout.clone().with_keys(keys)?; Ok(()) }
    pub fn view(&self) -> ArrayView<'_> {
        ArrayView { layout: &self.layout, sources: self.sources.iter().map(|s| Binding::full(s.view())).collect(), plan: self.plan.clone() }
    }
    pub fn dtype(&self) -> DType { self.sources[0].dtype() }
    pub fn len(&self) -> usize { self.plan.len() }
    pub fn is_empty(&self) -> bool { self.plan.is_empty() }
    fn is_compact(&self) -> bool { self.sources.len() == 1 && self.sources[0].len() == self.len() && matches!(self.plan.route(), ReadRoute::Input(0)) }
    fn detach(&mut self) {
        if self.is_compact() { return; }
        let items = self.view().cast(self.dtype()).expect("an array reads at its own dtype");
        let buffer = match items { Items::Owned(buffer) => buffer, Items::Borrowed(view) => Buffer::copied(view, 0..view.len()) };
        self.sources = Arc::from([Arc::new(buffer)]);
        self.plan = ReadPlan::input(0, self.len());
    }
    /// Compact storage with no retained items outside this array, reusing a uniquely held allocation.
    pub fn into_compact(mut self) -> (Layout, Buffer) {
        self.detach();
        let dtype = self.dtype();
        let buffer = match Arc::get_mut(&mut self.sources).and_then(|sources| Arc::get_mut(&mut sources[0])) {
            Some(source) => std::mem::replace(source, Buffer::with_capacity(dtype, 0)), None => (*self.sources[0]).clone()
        };
        (self.layout, buffer)
    }
    pub fn compact(&self) -> Self {
        let mut out = self.clone();
        out.detach();
        out
    }
    pub fn make_mut(&mut self) -> &mut Buffer { self.detach(); Arc::make_mut(&mut Arc::make_mut(&mut self.sources)[0]) }
    /// Every selection remains a view, including tables and padding. A chosen fill is one more scalar source.
    pub fn selected(&self, plan: crate::structure::Plan, fill: Option<Arc<Buffer>>) -> Result<Self, Error> {
        let n = element_count(plan.layout.shape())?;
        let mut sources = self.sources.to_vec();
        let fill = match fill {
            Some(fill) => {
                if fill.len() != 1 { return Err(Error::new(Kind::Length, "a fill must contain one item")); }
                Some(Arc::new(Buffer::kept(fill.view(), self.dtype())?))
            }
            None if self.dtype() == DType::Character => Some(Arc::new(Buffer::Characters(vec![char::FILL]))),
            None => None,
        };
        let missing = fill.map_or(Missing::Default, |fill| {
            let source = sources.len();
            sources.push(fill);
            Missing::Scalar(source)
        });
        Ok(Self { layout: plan.layout, sources: sources.into(), plan: self.plan.remapped(Mapping::from(plan.steps), n, missing) })
    }
    pub fn reshape(&self, layout: Layout) -> Result<Self, Error> {
        if element_count(layout.shape())? != self.len() { return Err(Error::new(Kind::Length, "reshape must retain the number of items")); }
        Ok(Self { layout, sources: self.sources.clone(), plan: self.plan.clone() })
    }
    /// Copies a structural plan from borrowed compact items through the shared reader.
    pub fn gathered(source: View<'_>, plan: &crate::structure::Plan, fill: Option<Arc<Buffer>>) -> Result<Self, Error> {
        let fill = fill.map(|f| Buffer::kept(f.view(), source.dtype())).transpose()?;
        if fill.as_ref().is_some_and(|f| f.len() != 1) { return Err(Error::new(Kind::Length, "a fill must contain one item")); }
        let mut sources = vec![Binding::full(source)];
        let missing = if let Some(fill) = &fill { sources.push(Binding::full(fill.view())); Missing::Scalar(1) } else { Missing::Default };
        let n = element_count(plan.layout.shape())?;
        let read = ReadPlan::copy(0, source.len(), Mapping::from(plan.steps.clone()), n, missing);
        let mut buffer = Buffer::with_capacity(source.dtype(), n);
        Read { sources: &sources, plan: &read }.read_into(0..n, source.dtype(), true, &mut buffer)?;
        Self::new(plan.layout.clone(), buffer)
    }
    /// Joins arrays of one dtype along an axis, retaining the usual layout rules.
    pub fn concatenate(arrays: &[ArrayView<'_>], axis: usize) -> Result<Self, Error> {
        let Some(source) = arrays.first() else { return Err(Error::new(Kind::Length, "concatenate needs an array")) };
        let mut layout = source.layout.clone();
        for view in &arrays[1..] {
            if view.dtype() != source.dtype() { return Err(Error::new(Kind::Domain, "concatenate items must have one dtype")); }
            layout = crate::structure::catenate(&layout, view.layout, axis)?;
        }
        let lanes = arrays.iter().map(|a| Lanes::new(a.layout.shape(), axis)).collect::<Result<Vec<_>, _>>()?;
        let lens: Vec<usize> = lanes.iter().map(|l| l.len * l.inner).collect();
        let (outer, stride) = (lanes[0].outer, lens.iter().sum::<usize>());
        let buffer = with_dtype!(source.dtype(), T => {
            let parts = arrays.iter().map(|a| a.items::<T>()).collect::<Result<Vec<_>, _>>()?;
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
    pub fn choose(condition: ArrayView<'_>, x: ArrayView<'_>, y: ArrayView<'_>) -> Result<Self, Error> {
        if condition.dtype() != DType::Boolean { return Err(Error::new(Kind::Domain, "condition must be Boolean")); }
        if x.dtype() != y.dtype() { return Err(Error::new(Kind::Domain, "chosen items must have one dtype")); }
        let pair = Agreement::new(x.layout, y.layout)?;
        let whole = Agreement::onto(&pair.layout, condition.layout)?;
        let x = x.remapped(pair.left, pair.len).remapped(whole.left.clone(), whole.len);
        let y = y.remapped(pair.right, pair.len).remapped(whole.left, whole.len);
        let condition = condition.remapped(whole.right, whole.len);
        let mut buffer = Buffer::with_capacity(x.dtype(), whole.len);
        crate::operation::choose((condition.read(), DType::Boolean), &[y.read(), x.read()], 0..whole.len, x.dtype(), true, &mut buffer)?;
        Self::new(whole.layout, buffer)
    }
    pub fn broadcasted(source: ArrayView<'_>, shape: &[usize]) -> Result<Self, Error> {
        let agreement = Agreement::new(&Layout::from(shape), source.layout)?;
        if agreement.layout.shape() != shape { return Err(Error::new(Kind::Length, "source doesn't broadcast to this shape")); }
        let source = source.remapped(agreement.right, agreement.len);
        let buffer = source.cast(source.dtype())?;
        Self::new(agreement.layout, match buffer { Items::Owned(b) => b, Items::Borrowed(v) => Buffer::copied(v, 0..v.len()) })
    }
    /// Adjacent differences read two selections through the ordinary kernel. Integer differences wrap at the input width.
    pub fn differences(source: ArrayView<'_>, axis: usize) -> Result<Self, Error> {
        let layout = source.layout;
        let &size = layout.shape().get(axis).ok_or_else(|| Error::new(Kind::Index, "difference axis is out of range"))?;
        if source.dtype() == DType::Character { return Err(Error::new(Kind::Domain, "differences need numerical items")); }
        let n = size.saturating_sub(1);
        let select = |start| crate::structure::select_axis(layout, axis, crate::structure::Selector::Range { start, len: n, step: 1 });
        let (next, previous) = (select(usize::from(size > 0))?, select(0)?);
        let len = next.layout.count();
        let x = source.remapped(Mapping::from(next.steps), len);
        let y = source.remapped(Mapping::from(previous.steps), len);
        let variant = crate::domain::Variant { dtype: source.dtype(), checked: true, wrapping: true };
        let k = if source.dtype() == DType::Boolean { Kernel::new(crate::reduce::Xor, variant) } else { Kernel::new(crate::Subtract, variant) };
        let mut buffer = Buffer::with_capacity(source.dtype(), len);
        k.map(&[x.read(), y.read()], 0..len, Settings { tolerance: crate::tolerant::Tolerance::EXACT }, &mut buffer)?;
        Self::new(next.layout, buffer)
    }
    fn updated(&self, combine: Option<&Kernel>, plan: &crate::structure::Plan, source: &Self) -> Result<Buffer, Error> {
        let scalar = Layout::EMPTY;
        let agreement = Agreement::new(&plan.layout, if source.len() == 1 { &scalar } else { source.layout() })?;
        if agreement.layout.shape() != plan.layout.shape() { return Err(Error::new(Kind::Length, "assignment must fit the selected layout")); }
        let root = self.view();
        let source = source.view().remapped(agreement.right, agreement.len);
        let positions = ReadPlan::input(0, self.len()).remapped(Mapping::from(plan.steps.clone()), agreement.len, Missing::Default);
        let mut buffer = Buffer::with_capacity(self.dtype(), self.len());
        crate::operation::at(
            combine, WriteRoot::Read(root.read()), source.read(), Positions::Route(&positions, 0), true,
            Settings { tolerance: crate::tolerant::Tolerance::EXACT }, &mut buffer)?;
        Ok(buffer)
    }
    /// Writes a broadcast source into a selection. Checks finish before replacing the array's storage.
    pub fn assign(&mut self, plan: &crate::structure::Plan, source: &Self) -> Result<(), Error> {
        let buffer = self.updated(None, plan, source)?;
        *self = Self::new(self.layout.clone(), buffer)?;
        Ok(())
    }
    pub fn combine_at(&mut self, kernel: &Kernel, plan: &crate::structure::Plan, source: &Self) -> Result<(), Error> {
        let buffer = self.updated(Some(kernel), plan, source)?;
        *self = Self::new(self.layout.clone(), buffer)?;
        Ok(())
    }
}
