//! Running a function object on compact arrays, at the dtype that each argument computes in. [`kernels!`](crate::kernels) finds an object's kernels by testing its [`Unary`], [`Binary`], [`Ternary`], [`Reduction`] or [`Inner`] impl at each element type, so only the object's forms say which dtypes it takes. [`Kernels::computes`] gives the dtype that an argument computes in, and the method for each arity casts an argument there where its own dtype differs, then runs the kernel.
//!
//! An argument computes in its own dtype where the object has a kernel there. Otherwise it computes in the first of these that the object has a kernel for:
//! - a Boolean or an integer: `i64`, then the float that [`DType::inexact`] gives it, then each wider float, then each complex dtype whose parts are at least as wide
//! - a float: each wider float, narrowest first, then each complex dtype at least as wide
//! - a `Complex32`: `Complex64`
//!
//! A character, or a number with none of these, has no kernel.
use crate::agreement::{Agreement, Frames, Mapping, chunks, map1, map2};
use crate::buffer::{Buffer, DType, Element, Source, View, Width};
use crate::reduce::{Reduction, scan_along};
use crate::slice::{Arg, Binary, CHUNK, Ternary, Unary};
use crate::linalg::Inner;
use crate::parallel::{extended, stack_with};

/// An object's kernel at one dtype for one argument: its result at each of `len` positions, which read the argument through the mapping, and whether every result succeeded.
pub type UnaryKernel<K> = for<'a> fn(&K, View<'a>, &Mapping, usize) -> (Buffer, bool);
/// An object's kernel at one dtype for two arguments, which pair as the agreement says.
pub type BinaryKernel<K> = for<'a> fn(&K, View<'a>, View<'a>, &Agreement) -> (Buffer, bool);
/// An object's kernel at one dtype for three arguments. The first agreement pairs the first two arguments, and the second pairs the first agreement's layout with the third.
pub type TernaryKernel<K> = for<'a> fn(&K, [View<'a>; 3], &Agreement, &Agreement) -> (Buffer, bool);
/// An accumulation along the middle of `[outer, len, inner]`. The probe selects `Fold`, `Scan`, `RightFold` or `LeftScan`; the optional argument is a right tail per lane for folds or a scalar start for scans. The selected kernel reads it in its total or computing dtype.
pub type AccumulationKernel<K> = for<'a> fn(&K, View<'a>, [usize; 3], Option<View<'a>>) -> (Buffer, bool);
/// A matrix product's kernel at one dtype: the product of each pair of cells that the frames pair, each of sizes `[n, k, m]`, and whether every product fit. `check` runs between rounds of cells, and a `false` from it stops the products, which then give `None`.
pub type ProductKernel<P> = for<'a> fn(&P, [View<'a>; 2], &Frames, [usize; 3], &mut dyn FnMut() -> bool) -> Option<(Buffer, bool)>;

/// An object's kernel at each dtype where it has one, in the order of [`DType::ALL`].
pub struct Kernels<F>([Option<F>; DType::ALL.len()]);

impl<F: Copy> Kernels<F> {
    /// The kernels that [`kernels!`](crate::kernels) finds.
    #[doc(hidden)]
    pub const fn new(at: [Option<F>; DType::ALL.len()]) -> Self { Self(at) }
    /// The kernel at `dtype`, where the object has one.
    pub fn at(&self, dtype: DType) -> Option<F> { self.0[dtype.index()] }
    /// These kernels without any at a dtype outside `dtypes`.
    pub fn only(mut self, dtypes: &[DType]) -> Self {
        for (k, d) in self.0.iter_mut().zip(DType::ALL) { if !dtypes.contains(&d) { *k = None; } }
        self
    }
    /// The dtype that an argument of `dtype` computes in, as the module describes, or `None` where the object has no kernel for it.
    pub fn computes(&self, dtype: DType) -> Option<DType> {
        if self.at(dtype).is_some() { return Some(dtype); }
        let integer = dtype.is_integral().then_some(DType::Integer(Width::I64));
        let float = dtype.inexact()?.0;
        integer.into_iter().chain(DType::ALL.into_iter().filter(|&d| float.widens_to(d))).find(|&d| self.at(d).is_some())
    }
    /// The kernel for arguments of `dtype`, and their dtype if they need casting first.
    fn chosen(&self, dtype: DType) -> Option<(F, DType)> {
        let at = self.computes(dtype)?;
        Some((self.at(at).expect("computes gives a dtype with a kernel"), at))
    }
}

/// `x` at `dtype`: itself where it already has `dtype`, and otherwise a cast that `cast` holds.
fn cast<'a>(x: View<'a>, dtype: DType, cast: &'a mut Option<Buffer>) -> View<'a> {
    if x.dtype() == dtype { return x; }
    cast.insert(Buffer::cast(x, dtype).expect("an argument casts to the dtype it computes in")).view()
}

impl<K> Kernels<UnaryKernel<K>> {
    /// `k` at each of `len` positions, which read `x` through `mapping`, and whether every result succeeded. `None` where `k` has no kernel for `x`'s dtype.
    pub fn unary(&self, k: &K, x: View<'_>, mapping: &Mapping, len: usize) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(k, cast(x, at, &mut None), mapping, len))
    }
}

impl<K> Kernels<BinaryKernel<K>> {
    /// `k` on `x` and `y`, which share a dtype and pair as `agreement` says, and whether every result succeeded. `None` where `k` has no kernel for their dtype.
    pub fn binary(&self, k: &K, x: View<'_>, y: View<'_>, agreement: &Agreement) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(k, cast(x, at, &mut None), cast(y, at, &mut None), agreement))
    }
}

impl<K> Kernels<TernaryKernel<K>> {
    /// `k` on three arguments of one dtype, paired as [`TernaryKernel`] describes, and whether every result succeeded. `None` where `k` has no kernel for their dtype.
    pub fn ternary(&self, k: &K, [x, y, z]: [View<'_>; 3], first: &Agreement, second: &Agreement) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(k, [cast(x, at, &mut None), cast(y, at, &mut None), cast(z, at, &mut None)], first, second))
    }
}

impl<K> Kernels<AccumulationKernel<K>> {
    /// Runs the accumulation selected by the probe, and reports whether every result succeeded. `None` where there is no kernel for `x`'s dtype. Folds take an optional right tail per lane; scans take an optional scalar start. `RightFold` needs two items per lane without a tail, because a singleton retains its input dtype and is handled by the caller.
    pub fn run(&self, k: &K, x: View<'_>, dims: [usize; 3], start: Option<View<'_>>) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(k, cast(x, at, &mut None), dims, start))
    }
}

impl<P> Kernels<ProductKernel<P>> {
    /// `p`'s products of the cells of `x` and `y`, which share a dtype and pair as `frames` says, each of sizes `[n, k, m]`, and whether every product fit. `None` where `p` has no kernel for their dtype, or where `check` stopped them.
    pub fn product(&self, p: &P, [x, y]: [View<'_>; 2], frames: &Frames, size: [usize; 3], check: &mut dyn FnMut() -> bool) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        kernel(p, [cast(x, at, &mut None), cast(y, at, &mut None)], frames, size, check)
    }
}

/// The items of `x`, which has the kernel's dtype.
fn items<T: Element>(x: View<'_>) -> &[T] { T::of(x).expect("an argument has its kernel's dtype") }

fn unary_at<K: Unary<T, Out: Element> + Sync, T: Element>(k: &K, x: View<'_>, mapping: &Mapping, len: usize) -> (Buffer, bool) {
    let mut out = Vec::with_capacity(len);
    let ok = map1(items::<T>(x), mapping, len, k, &mut out);
    (<<K as Unary<T>>::Out as Element>::buffer(out), ok)
}

fn binary_at<K: Binary<T, Out: Element> + Sync, T: Element>(k: &K, x: View<'_>, y: View<'_>, agreement: &Agreement) -> (Buffer, bool) {
    let mut out = Vec::with_capacity(agreement.len);
    let ok = map2(items::<T>(x), items::<T>(y), agreement, k, &mut out);
    (<<K as Binary<T>>::Out as Element>::buffer(out), ok)
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

/// Contiguous/constant arguments go to the slice form directly. Other mappings expand in chunks within parallel ranges.
fn ternary_at<K: Ternary<T, Out: Element> + Sync, T: Element>(k: &K, [x, y, z]: [View<'_>; 3], first: &Agreement, second: &Agreement) -> (Buffer, bool) {
    let (x, y, z) = (items::<T>(x), items::<T>(y), items::<T>(z));
    let read = |data: &[T], at: Option<usize>| at.map_or(T::FILL, |j| data[j]);
    let mut out = Vec::with_capacity(second.len);
    let direct = if second.len == 0 { None } else {
        direct(x, &first.left, &second.left).zip(direct(y, &first.right, &second.left))
            .zip(direct(z, &second.right, &Mapping::Linear(1)))
    };
    let ok = if let Some(((x, y), z)) = direct { k.many(x, y, z, second.len, &mut out) }
    else { extended(&mut out, second.len, CHUNK, &|range, slots| {
        let mut bufs = [Vec::new(), Vec::new(), Vec::new()];
        chunks(range, |start, n| {
            bufs.iter_mut().for_each(Vec::clear);
            for i in start..start + n {
                let pair = second.left.get(i);
                bufs[0].push(read(x, pair.and_then(|j| first.left.get(j))));
                bufs[1].push(read(y, pair.and_then(|j| first.right.get(j))));
                bufs[2].push(read(z, second.right.get(i)));
            }
            k.many_into(Arg::Many(&bufs[0]), Arg::Many(&bufs[1]), Arg::Many(&bufs[2]), n, slots)
        })
    }) };
    (<<K as Ternary<T>>::Out as Element>::buffer(out), ok)
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
    let write = |i: usize, out: &mut [P::Out], _: &mut ()| match p.checked(&x[frames.cell(0, i) * n * k..][..n * k], &y[frames.cell(1, i) * k * m..][..k * m], size) {
        Some(r) => { out.copy_from_slice(&r); true }
        None => false,
    };
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
        pub const fn new() -> Self { Self(PhantomData) }
        /// The probe for `k`'s type, for a [`kernels!`](crate::kernels) given a value.
        pub const fn of(_: &K) -> Self { Self(PhantomData) }
    }
    /// The arities that `kernels!` names.
    pub enum Unary {}
    pub enum Binary {}
    pub enum Ternary {}
    pub enum Fold {}
    pub enum Scan {}
    pub enum RightFold {}
    pub enum LeftScan {}
    pub enum Product {}

    pub trait Found<F> {
        fn kernel(&self) -> Option<F>;
    }
    pub trait Missing<F> {
        fn kernel(&self) -> Option<F> { None }
    }
    impl<K, T, A, F> Missing<F> for &Probe<K, T, A> {}

    impl<K: crate::Unary<T, Out: Element> + Sync, T: Element> Found<UnaryKernel<K>> for Probe<K, T, Unary> {
        fn kernel(&self) -> Option<UnaryKernel<K>> { Some(unary_at::<K, T>) }
    }
    impl<K: crate::Binary<T, Out: Element> + Sync, T: Element> Found<BinaryKernel<K>> for Probe<K, T, Binary> {
        fn kernel(&self) -> Option<BinaryKernel<K>> { Some(binary_at::<K, T>) }
    }
    impl<K: crate::Ternary<T, Out: Element> + Sync, T: Element> Found<TernaryKernel<K>> for Probe<K, T, Ternary> {
        fn kernel(&self) -> Option<TernaryKernel<K>> { Some(ternary_at::<K, T>) }
    }
    impl<R: Reduction<T>, T: Element> Found<AccumulationKernel<R>> for Probe<R, T, Fold> where R::Total: Element {
        fn kernel(&self) -> Option<AccumulationKernel<R>> { Some(fold_at::<R, T>) }
    }
    impl<R: Reduction<T>, T: Element> Found<AccumulationKernel<R>> for Probe<R, T, Scan> where R::Total: Element {
        fn kernel(&self) -> Option<AccumulationKernel<R>> { Some(scan_at::<R, T>) }
    }
    impl<K: crate::Binary<T, Out: Element + Source<T>>, T: Element> Found<AccumulationKernel<K>> for Probe<K, T, RightFold> {
        fn kernel(&self) -> Option<AccumulationKernel<K>> { Some(right_fold_at::<K, T>) }
    }
    impl<K: crate::Binary<T, Out: Element + Source<T>>, T: Element> Found<AccumulationKernel<K>> for Probe<K, T, LeftScan> {
        fn kernel(&self) -> Option<AccumulationKernel<K>> { Some(left_scan_at::<K, T>) }
    }
    impl<P: crate::linalg::Inner<T, Out: Element + Default> + Sync, T: Element> Found<ProductKernel<P>> for Probe<P, T, Product> {
        fn kernel(&self) -> Option<ProductKernel<P>> { Some(product_at::<P, T>) }
    }
}

/// The [`Kernels`](crate::operation::Kernels) of a function object or reduction: `Unary`, `Binary` or `Ternary` for that arity, `RightFold` or `LeftScan` for ordered accumulation by a binary function, `Fold` or `Scan` for a [`Reduction`](crate::reduce::Reduction), and `Product` for a matrix product's [`Inner`](crate::linalg::Inner) pair. Name the object by its type, or by a value after `of`. Each kernel's test resolves where the macro expands, so the type must be concrete there. In generic code the tests see only the type parameter's bounds, not the impls of the type it later stands for, so a kernel that the bounds don't promise comes out as `None`, with no error.
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
