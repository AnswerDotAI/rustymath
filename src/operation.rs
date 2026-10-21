//! Running a function object on compact arrays, at the dtype that each argument computes in. [`kernels!`](crate::kernels) finds an object's kernels by testing its [`Unary`], [`Binary`], [`Ternary`], [`Reduction`] or [`Inner`] impl at each element type, so only the object's forms say which dtypes it takes. [`Kernels::computes`] gives the dtype that an argument computes in, and the method for each arity casts an argument there where its own dtype differs, then runs the kernel.
//!
//! An argument computes in the dtype that [`DType::computes_in`] gives for the object's kernels, under NumPy's policy for exact numbers: a Boolean or an integer without a kernel at its own dtype tries `i64`, then the float that [`DType::inexact`] gives it under `f64`. A character, or a number with no kernel on that ladder, has no kernel.
use crate::agreement::{Agreement, Frames, Mapping, Read, Side, expanded, map1, map2, read_runs};
use crate::array::ArrayView;
use crate::buffer::{Buffer, DType, Element, FloatWidth, Source, View, Width};
use crate::reduce::{Reduction, scan_along};
use crate::slice::{Arg, Binary, Results, Ternary, Unary};
use crate::linalg::Inner;
use crate::parallel::{stack_with, Slots};

/// An object's kernel at one dtype for one argument: its result at each of `len` positions, which read the argument through the mapping, and whether every result succeeded.
pub type UnaryKernel<K> = for<'a> fn(&K, View<'a>, &Mapping, usize) -> (Buffer, bool);
/// An object's kernel at one dtype for two arguments, which pair as the agreement says.
pub type BinaryKernel<K> = for<'a> fn(&K, View<'a>, View<'a>, &Agreement) -> (Buffer, bool);
/// An object's kernel at one dtype for three arguments. The first agreement pairs the first two arguments, and the second pairs the first agreement's layout with the third.
pub type TernaryKernel<K> = for<'a> fn(&K, [View<'a>; 3], &Agreement, &Agreement) -> (Buffer, bool);
/// An accumulation along the middle of `[outer, len, inner]`. The probe selects `Fold`, `Scan`, `RightFold` or `LeftScan`; the optional argument is a right tail per lane for folds or a scalar start for scans. The selected kernel reads it in its total or computing dtype.
pub type AccumulationKernel<K> = for<'a> fn(&K, View<'a>, [usize; 3], Option<View<'a>>) -> (Buffer, bool);
/// A reduction's folds of each window of a number of neighbouring rows along the middle of `[outer, len, inner]`, with that number.
pub type SlidingKernel<K> = for<'a> fn(&K, View<'a>, [usize; 3], usize) -> (Buffer, bool);
/// A matrix product's kernel at one dtype: the product of each pair of cells that the frames pair, each of sizes `[n, k, m]`, and whether every product fit. `check` runs between rounds of cells, and a `false` from it stops the products, which then give `None`.
pub type ProductKernel<P> = for<'a> fn(&P, [View<'a>; 2], &Frames, [usize; 3], &mut dyn FnMut() -> bool) -> Option<(Buffer, bool)>;
/// An object's kernel at one dtype for combining items into an array in place, as [`Kernels::scatter`] describes.
pub type ScatterKernel<K> = for<'a> fn(&K, &mut Buffer, &[usize], View<'a>) -> bool;

/// An object's kernel at each dtype where it has one, in the order of [`DType::ALL`], with whether it leaves a failed result wrapped to its type.
pub struct Kernels<F>([Option<(F, bool)>; DType::ALL.len()]);

/// The float width that a Boolean or integer computes in under rustymath's own policy, NumPy's: the float that [`DType::inexact`] gives it under `f64`. `f64` for any other dtype.
pub fn numpy_float(dtype: DType) -> FloatWidth { match dtype.inexact(FloatWidth::F64) { Some((DType::Float(w), _)) => w, _ => FloatWidth::F64 } }

impl<F: Copy> Kernels<F> {
    /// The kernels that [`kernels!`](crate::kernels) finds.
    #[doc(hidden)]
    pub const fn new(at: [Option<(F, bool)>; DType::ALL.len()]) -> Self { Self(at) }
    /// The kernel at `dtype`, where the object has one.
    pub fn at(&self, dtype: DType) -> Option<F> { self.0[dtype.index()].map(|(f, _)| f) }
    /// These kernels without any at a dtype outside `dtypes`.
    #[inline]
    pub fn only(mut self, dtypes: &[DType]) -> Self {
        let kept = dtypes.iter().fold(0u32, |kept, d| kept | 1 << d.index());
        for (i, k) in self.0.iter_mut().enumerate() { if kept >> i & 1 == 0 { *k = None; } }
        self
    }
    /// The dtype that an argument of `dtype` computes in, as the module describes, or `None` where the object has no kernel for it.
    pub fn computes(&self, dtype: DType) -> Option<DType> {
        dtype.computes_in(Width::I64, numpy_float(dtype), |d| self.at(d).is_some())
    }
    /// Whether the kernel that an argument of `dtype` computes with leaves a failed result wrapped to its type, as integer arithmetic and every [`Reduction`] do. False where the object has no kernel for it.
    pub fn wraps(&self, dtype: DType) -> bool { self.computes(dtype).and_then(|d| self.0[d.index()]).is_some_and(|(_, wraps)| wraps) }
    /// The kernel for arguments of `dtype`, and their dtype if they need casting first.
    fn chosen(&self, dtype: DType) -> Option<(F, DType)> {
        let at = self.computes(dtype)?;
        Some((self.at(at).expect("computes gives a dtype with a kernel"), at))
    }
}

/// `x` at `dtype`: itself where it already has `dtype`, and otherwise a cast that `cast` holds.
#[inline]
fn cast<'a>(x: View<'a>, dtype: DType, cast: &'a mut Option<Buffer>) -> View<'a> {
    if x.dtype() == dtype { return x; }
    cast.insert(Buffer::cast(x, dtype).expect("an argument casts to the dtype it computes in")).view()
}

impl<K> Kernels<UnaryKernel<K>> {
    /// `k` on the logical items of `x`, and whether every result succeeded. `None` where `k` has no kernel for `x`'s dtype.
    pub fn unary(&self, k: &K, x: ArrayView<'_>) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        if at == x.dtype() { return Some(kernel(k, x.storage(), x.mapping(), x.len())); }
        Some(kernel(k, x.cast(at).expect("computing dtype").view(), &Mapping::Linear(1), x.len()))
    }
}

/// Keeps an ordinary compact call's agreement borrowed; only changed input mappings need a new one.
fn remapped(a: &Agreement, left: Option<Mapping>, right: Option<Mapping>) -> Option<Agreement> {
    (left.is_some() || right.is_some()).then(|| Agreement {
        layout: a.layout.clone(), len: a.len, left: left.unwrap_or_else(|| a.left.clone()), right: right.unwrap_or_else(|| a.right.clone()),
    })
}

impl<K> Kernels<BinaryKernel<K>> {
    /// `k` on `x` and `y`, which share a dtype and pair as `agreement` says, and whether every result succeeded. `None` where `k` has no kernel for their dtype.
    pub fn binary(&self, k: &K, x: ArrayView<'_>, y: ArrayView<'_>, agreement: &Agreement) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        let (x, left) = x.mapped(&agreement.left, agreement.layout.shape(), at);
        let (y, right) = y.mapped(&agreement.right, agreement.layout.shape(), at);
        let changed = remapped(agreement, left, right);
        Some(kernel(k, x.view(), y.view(), changed.as_ref().unwrap_or(agreement)))
    }
}

impl<K> Kernels<TernaryKernel<K>> {
    /// `k` on three arguments of one dtype, or on two of one dtype and a Boolean mask, paired as [`TernaryKernel`] describes, and whether every result succeeded. `None` where `k` has no kernel for their dtype.
    pub fn ternary(&self, k: &K, [x, y, z]: [ArrayView<'_>; 3], first: &Agreement, second: &Agreement) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        let (x, left) = x.mapped(&first.left, first.layout.shape(), at);
        let (y, right) = y.mapped(&first.right, first.layout.shape(), at);
        // A Boolean third argument keeps its dtype: `ternary_at` turns it into items of `at` a chunk at a time, through `Element::truth`.
        let (z, last) = z.mapped(&second.right, second.layout.shape(), if z.dtype() == DType::Boolean { DType::Boolean } else { at });
        let (one, two) = (remapped(first, left, right), remapped(second, None, last));
        Some(kernel(k, [x.view(), y.view(), z.view()], one.as_ref().unwrap_or(first), two.as_ref().unwrap_or(second)))
    }
}

impl<K> Kernels<AccumulationKernel<K>> {
    /// Runs the accumulation selected by the probe, and reports whether every result succeeded. `None` where there is no kernel for `x`'s dtype. Folds take an optional right tail per lane; scans take an optional scalar start. `RightFold` needs two items per lane without a tail, because a singleton retains its input dtype and is handled by the caller.
    pub fn run(&self, k: &K, x: View<'_>, dims: [usize; 3], start: Option<View<'_>>) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(k, cast(x, at, &mut None), dims, start))
    }
}

impl<K> Kernels<SlidingKernel<K>> {
    /// The folds of each window of `width` rows along the middle of `dims`, as [`reduce::sliding`](crate::reduce::sliding) gives them, and whether every fold fits. `None` where there is no kernel for `x`'s dtype.
    pub fn sliding(&self, k: &K, x: View<'_>, dims: [usize; 3], width: usize) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(k, cast(x, at, &mut None), dims, width))
    }
}

impl<P> Kernels<ProductKernel<P>> {
    /// `p`'s products of the cells of `x` and `y`, which share a dtype and pair as `frames` says, each of sizes `[n, k, m]`, and whether every product fit. `None` where `p` has no kernel for their dtype, or where `check` stopped them.
    pub fn product(&self, p: &P, [x, y]: [View<'_>; 2], frames: &Frames, size: [usize; 3], check: &mut dyn FnMut() -> bool) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        kernel(p, [cast(x, at, &mut None), cast(y, at, &mut None)], frames, size, check)
    }
}

impl<K> Kernels<ScatterKernel<K>> {
    /// Combines each item of `values`, or its only item, into the item of `out` at the matching offset by `k`, in turn. It gives whether every result succeeded. A failed result is written as `k`'s slice form writes it: wrapped to its type, or NaN. `values` reads at `out`'s dtype. `None` where `k` has no kernel at `out`'s dtype.
    pub fn scatter(&self, k: &K, out: &mut Buffer, offsets: &[usize], values: View<'_>) -> Option<bool> {
        let kernel = self.at(out.dtype())?;
        Some(kernel(k, out, offsets, cast(values, out.dtype(), &mut None)))
    }
}

/// The items of `x`, which has the kernel's dtype.
fn items<T: Element>(x: View<'_>) -> &[T] { T::of(x).expect("an argument has its kernel's dtype") }

fn unary_at<K: Unary<T, Out: Element> + Sync, T: Element>(k: &K, x: View<'_>, mapping: &Mapping, len: usize) -> (Buffer, bool) {
    let mut out = Vec::with_capacity(len);
    let ok = map1(items::<T>(x), mapping, len, k, &mut out);
    (<<K as Results<T, 1>>::Out as Element>::buffer(out), ok)
}

pub(crate) fn binary_at<K: Binary<T, Out: Element> + Sync, T: Element, L>(k: &K, x: View<'_>, y: View<'_>, agreement: &Agreement<L>) -> (Buffer, bool) {
    let mut out = Vec::with_capacity(agreement.len);
    let ok = map2(items::<T>(x), items::<T>(y), agreement, k, &mut out);
    (<<K as Results<T, 2>>::Out as Element>::buffer(out), ok)
}

/// A contiguous or constant argument through two mappings, without an expansion buffer.
#[inline]
fn direct<'a, T: Element>(data: &'a [T], mapping: &Mapping, parent: &Mapping) -> Option<Arg<'a, T>> {
    match (parent, mapping) {
        (Mapping::Single, _) => Some(Arg::One(mapping.get(0).map_or(T::FILL, |i| data[i]))),
        (Mapping::Linear(1), Mapping::Single) => Some(Arg::One(data[0])),
        (Mapping::Linear(1), Mapping::Linear(1)) => Some(Arg::Many(data)),
        _ => None,
    }
}

/// A third argument at a kernel's item type `T`, or a Boolean mask that [`Kernels::ternary`] leaves uncast, which becomes items of `T` through [`Element::truth`].
#[derive(Clone, Copy)]
enum Third<'a, T> { Items(&'a [T]), Mask(&'a [bool]) }

impl<'a, T: Element> Third<'a, T> {
    fn of(z: View<'a>) -> Self { match bool::of(z) { Some(mask) if T::DTYPE != DType::Boolean => Third::Mask(mask), _ => Third::Items(items::<T>(z)) } }
}

/// Contiguous or constant arguments go to the slice form directly, and a mask through [`masked`](crate::slice::masked). Otherwise each argument reads through its two mappings composed into one, as [`read_runs`] reads it. Where the two don't compose structurally, they compose into a table of offsets.
fn ternary_at<K: Ternary<T, Out: Element> + Sync, T: Element>(k: &K, [x, y, z]: [View<'_>; 3], first: &Agreement, second: &Agreement) -> (Buffer, bool) {
    let (x, y, z) = (items::<T>(x), items::<T>(y), Third::<T>::of(z));
    let mut out = Vec::with_capacity(second.len);
    let pair = if second.len == 0 { None } else { direct(x, &first.left, &second.left).zip(direct(y, &first.right, &second.left)) };
    let done = pair.and_then(|(xa, ya)| match z {
        Third::Items(z) => direct(z, &second.right, &Mapping::Linear(1)).map(|za| k.many(xa, ya, za, second.len, &mut out)),
        Third::Mask(m) => direct(m, &second.right, &Mapping::Linear(1)).map(|ma| crate::slice::masked(k, xa, ya, ma, second.len, &mut out)),
    });
    let ok = done.unwrap_or_else(|| {
        let through = |m: &Mapping| m.composed(&second.left, first.layout.shape(), second.layout.shape(), true).expect("a table always composes");
        let (xm, ym) = (through(&first.left), through(&first.right));
        let (xs, ys) = (Side::new(x, &xm), Side::new(y, &ym));
        let leaf = |[a, b, c]: [Arg<'_, T>; 3], n: usize, slots: &mut Slots<'_, _>| k.many_into(a, b, c, n, slots);
        match z {
            Third::Items(z) => read_runs([&xs, &ys, &Side::new(z, &second.right)], second.len, &mut out, &leaf),
            Third::Mask(mask) => read_runs([&xs, &ys, &Truths { mask, mapping: &second.right }], second.len, &mut out, &leaf),
        }
    });
    (<<K as Results<T, 3>>::Out as Element>::buffer(out), ok)
}

/// A Boolean mask read as items of `T` through [`Element::truth`], as a ternary's third argument.
struct Truths<'a> { mask: &'a [bool], mapping: &'a Mapping }

impl<T: Element> Read<T> for Truths<'_> {
    fn read<'b>(&'b self, start: usize, n: usize, buf: &'b mut Vec<T>) -> Arg<'b, T> {
        expanded(self.mapping, self.mask, start, n, &mut Vec::new()).converted(0, n, buf, T::truth)
    }
    fn constant(&self) -> Option<T> { matches!(self.mapping, Mapping::Single).then(|| T::truth(self.mask[0])) }
    fn reads(&self) -> (&Mapping, usize) { (self.mapping, 1) }
}


fn fold_at<R: Reduction<T>, T: Element>(_: &R, x: View<'_>, dims: [usize; 3], tail: Option<View<'_>>) -> (Buffer, bool) where R::Total: Element {
    let mut tc = None;
    let tail = tail.map(|tail| cast(tail, R::Total::DTYPE, &mut tc));
    let (totals, ok) = crate::reduce::along_with_tail::<T, R>(items::<T>(x), dims, tail.map(items::<R::Total>));
    (R::Total::buffer(totals), ok)
}

fn scan_at<R: Reduction<T>, T: Element>(_: &R, x: View<'_>, dims: [usize; 3], start: Option<View<'_>>) -> (Buffer, bool) where R::Total: Element {
    let mut sc = None;
    let start = start.map(|s| items::<R::Total>(cast(s, R::Total::DTYPE, &mut sc))[0]);
    let mut out = Vec::with_capacity(dims.iter().product());
    let ok = scan_along::<T, R>(items::<T>(x), dims, start, &mut out);
    (R::Total::buffer(out), ok)
}

fn sliding_at<R: Reduction<T>, T: Element>(_: &R, x: View<'_>, dims: [usize; 3], width: usize) -> (Buffer, bool) where R::Total: Element {
    let (totals, ok) = crate::reduce::sliding::<T, R>(items::<T>(x), dims, width);
    (R::Total::buffer(totals), ok)
}

fn right_fold_at<K: Binary<T, Out: Element + Source<T>>, T: Element>(k: &K, x: View<'_>, dims: [usize; 3], tail: Option<View<'_>>) -> (Buffer, bool) {
    let ok = std::cell::Cell::new(true);
    let mut tc = None;
    let tail = tail.map(|t| cast(t, T::DTYPE, &mut tc));
    let out = crate::reduce::right_fold(items::<T>(x), dims, tail.map(items::<T>), |a, b| Some(checked_step(k, a, b, &ok)), Source::read)
        .expect("every step writes its value");
    (K::Out::buffer(out), ok.get())
}

fn checked_step<K: Binary<T>, T: Copy>(k: &K, a: T, b: T, ok: &std::cell::Cell<bool>) -> K::Out {
    k.checked(a, b).unwrap_or_else(|| {
        ok.set(false);
        // `checked` omits a failed value. Its slice form retains it, including NaN, without discarding other lanes.
        let mut one = Vec::with_capacity(1);
        k.many(Arg::One(a), Arg::One(b), 1, &mut one);
        one.pop().expect("one result was written")
    })
}

fn scatter_at<K: Binary<T, Out = T>, T: Element>(k: &K, out: &mut Buffer, offsets: &[usize], values: View<'_>) -> bool {
    let ok = std::cell::Cell::new(true);
    let (out, values) = (T::of_mut(out).expect("the output has its kernel's dtype"), items::<T>(values));
    let step = |a, y| checked_step(k, a, y, &ok);
    crate::gather::scatter_with(out, crate::gather::targets(offsets, Arg::of(values)), step);
    ok.get()
}

fn left_scan_at<K: Binary<T, Out: Element + Source<T>>, T: Element>(k: &K, x: View<'_>, dims: [usize; 3], seed: Option<View<'_>>) -> (Buffer, bool) {
    let ok = std::cell::Cell::new(true);
    let mut sc = None;
    let seed = seed.map(|s| cast(s, T::DTYPE, &mut sc));
    let mut out = Vec::with_capacity(x.len());
    crate::reduce::left_scan(items::<T>(x), dims, seed.map(|s| items::<T>(s)[0]), |a, b| Some(Source::read(checked_step(k, a, b, &ok))), &mut out);
    (T::buffer(out), ok.get())
}

fn product_at<P: Inner<T, Out: Element + Default> + Sync, T: Element>(p: &P, [x, y]: [View<'_>; 2], frames: &Frames, size @ [n, k, m]: [usize; 3], check: &mut dyn FnMut() -> bool) -> Option<(Buffer, bool)> {
    let (x, y) = (items::<T>(x), items::<T>(y));
    let write = |i: usize, out: &mut [P::Out], _: &mut ()| p.checked_into(&x[frames.items(0, i, n * k)], &y[frames.items(1, i, k * m)], size, out);
    let (out, ok) = stack_with(frames.count, n * m, &write, || if check() { Ok(()) } else { Err(()) }).ok()?;
    Some((P::Out::buffer(out), ok))
}

/// The machinery of [`kernels!`](crate::kernels), which tests a trait impl at each element type by autoref specialization. [`Found`] gives the kernel on `Probe` where the impl holds. Method lookup tries `Probe` before `&Probe`, so [`Missing`]'s `None` on `&Probe` applies only where it doesn't.
#[doc(hidden)]
pub mod probe {
    use super::*;
    use std::marker::PhantomData;

    pub struct Probe<K, T, A>(PhantomData<(K, T, A)>);
    impl<K, T, A> Probe<K, T, A> {
        #[inline(always)]
        pub const fn new() -> Self { Self(PhantomData) }
        /// The probe for `k`'s type, for a [`kernels!`](crate::kernels) given a value.
        #[inline(always)]
        pub const fn of(_: &K) -> Self { Self(PhantomData) }
    }
    /// The operations that `kernels!` names.
    pub enum Unary {}
    pub enum Binary {}
    pub enum Ternary {}
    pub enum Fold {}
    pub enum Scan {}
    pub enum RightFold {}
    pub enum LeftScan {}
    pub enum Sliding {}
    pub enum Product {}
    pub enum Scatter {}
    /// The test that [`characters!`](crate::characters!) makes.
    pub enum Characters {}

    /// A kernel where the impl holds, with whether it leaves a failed result wrapped to its type.
    pub trait Found<F> {
        fn kernel(&self) -> Option<(F, bool)>;
    }
    pub trait Missing<F> {
        #[inline(always)]
        fn kernel(&self) -> Option<(F, bool)> { None }
    }
    impl<K, T, A, F> Missing<F> for &Probe<K, T, A> {}

    impl<K: crate::Unary<T, Out: Element> + Sync, T: Element> Found<UnaryKernel<K>> for Probe<K, T, Unary> {
        #[inline(always)]
        fn kernel(&self) -> Option<(UnaryKernel<K>, bool)> { Some((unary_at::<K, T>, K::WRAPS)) }
    }
    impl<K: crate::Binary<T, Out: Element> + Sync, T: Element> Found<BinaryKernel<K>> for Probe<K, T, Binary> {
        #[inline(always)]
        fn kernel(&self) -> Option<(BinaryKernel<K>, bool)> { Some((binary_at::<K, T, crate::layout::Layout>, K::WRAPS)) }
    }
    impl<K: crate::Ternary<T, Out: Element> + Sync, T: Element> Found<TernaryKernel<K>> for Probe<K, T, Ternary> {
        #[inline(always)]
        fn kernel(&self) -> Option<(TernaryKernel<K>, bool)> { Some((ternary_at::<K, T>, K::WRAPS)) }
    }
    // Every reduction wraps a total that doesn't fit, as `Reduction` says.
    impl<R: Reduction<T>, T: Element> Found<AccumulationKernel<R>> for Probe<R, T, Fold> where R::Total: Element {
        #[inline(always)]
        fn kernel(&self) -> Option<(AccumulationKernel<R>, bool)> { Some((fold_at::<R, T>, true)) }
    }
    impl<R: Reduction<T>, T: Element> Found<AccumulationKernel<R>> for Probe<R, T, Scan> where R::Total: Element {
        #[inline(always)]
        fn kernel(&self) -> Option<(AccumulationKernel<R>, bool)> { Some((scan_at::<R, T>, true)) }
    }
    impl<R: Reduction<T>, T: Element> Found<SlidingKernel<R>> for Probe<R, T, Sliding> where R::Total: Element {
        #[inline(always)]
        fn kernel(&self) -> Option<(SlidingKernel<R>, bool)> { Some((sliding_at::<R, T>, true)) }
    }
    impl<K: crate::Binary<T, Out: Element + Source<T>>, T: Element> Found<AccumulationKernel<K>> for Probe<K, T, RightFold> {
        #[inline(always)]
        fn kernel(&self) -> Option<(AccumulationKernel<K>, bool)> { Some((right_fold_at::<K, T>, K::WRAPS)) }
    }
    impl<K: crate::Binary<T, Out: Element + Source<T>>, T: Element> Found<AccumulationKernel<K>> for Probe<K, T, LeftScan> {
        #[inline(always)]
        fn kernel(&self) -> Option<(AccumulationKernel<K>, bool)> { Some((left_scan_at::<K, T>, K::WRAPS)) }
    }
    impl<P: crate::linalg::Inner<T, Out: Element> + Sync, T: Element> Found<ProductKernel<P>> for Probe<P, T, Product> {
        #[inline(always)]
        fn kernel(&self) -> Option<(ProductKernel<P>, bool)> { Some((product_at::<P, T>, false)) }
    }
    impl<K: crate::Binary<T, Out = T>, T: Element> Found<ScatterKernel<K>> for Probe<K, T, Scatter> {
        #[inline(always)]
        fn kernel(&self) -> Option<(ScatterKernel<K>, bool)> { Some((scatter_at::<K, T>, K::WRAPS)) }
    }
    impl<K: crate::characters::Characters + crate::Binary<i32, Out: Element> + crate::Binary<i64, Out: Element> + Sync> Found<crate::characters::Kernel<K>> for Probe<K, (), Characters> {
        #[inline(always)]
        fn kernel(&self) -> Option<(crate::characters::Kernel<K>, bool)> { Some((crate::characters::binary::<K, crate::layout::Layout>, false)) }
    }
}

/// The [`Kernels`](crate::operation::Kernels) of a function object or reduction: `Unary`, `Binary` or `Ternary` for that arity, `RightFold` or `LeftScan` for ordered accumulation by a binary function, `Scatter` for combining a binary function's results into an array in place, `Fold`, `Scan` or `Sliding` for a [`Reduction`](crate::reduce::Reduction), and `Product` for a matrix product's [`Inner`](crate::linalg::Inner) pair. Name the object by its type, or by a value after `of`. Each kernel's test resolves where the macro expands, so the type must be concrete there. In generic code the tests see only the type parameter's bounds, not the impls of the type it later stands for, so a kernel that the bounds don't promise comes out as `None`, with no error.
#[macro_export]
macro_rules! kernels {
    ($arity:ident of $k:expr) => {{
        let k = &$k;
        $crate::kernels!(@all $arity [of k])
    }};
    ($arity:ident $k:ty) => { $crate::kernels!(@all $arity [type $k]) };
    (@all $arity:ident $how:tt) => {{
        #[allow(unused_imports)]
        use $crate::operation::probe::{Found as _, Missing as _};
        $crate::element_types!(($crate::kernels) @list $arity $how)
    }};
    (@list $arity:ident $how:tt [$($name:ident: $d:tt $t:ty),+]) => {
        $crate::operation::Kernels::new([$($crate::kernels!(@at $arity $how $t)),+])
    };
    (@at $arity:ident [type $k:ty] $t:ty) => {
        (&$crate::operation::probe::Probe::<$k, $t, $crate::operation::probe::$arity>::new()).kernel()
    };
    (@at $arity:ident [of $k:ident] $t:ty) => {
        (&$crate::operation::probe::Probe::<_, $t, $crate::operation::probe::$arity>::of($k)).kernel()
    };
}
