//! Running a function object on compact arrays, at the dtype that each argument computes in. [`kernels!`](crate::kernels) finds an object's kernels by testing its [`Unary`], [`Binary`], [`Ternary`], [`Reduction`] or [`Inner`] impl at each element type, so only the object's forms say which dtypes it takes. [`Kernels::computes`] gives the dtype that an argument computes in, and the method for each arity casts an argument there where its own dtype differs, then runs the kernel.
//!
//! An argument computes in its own dtype where the object has a kernel there. Otherwise it computes in the first of these that the object has a kernel for:
//! - a Boolean or an integer: `i64`, then the float that [`DType::inexact`] gives it, then each wider float, then each complex dtype whose parts are at least as wide
//! - a float: each wider float, narrowest first, then each complex dtype at least as wide
//! - a `Complex32`: `Complex64`
//!
//! A character, or a number with none of these, has no kernel.
use crate::agreement::{Agreement, Frames, Mapping, chunks, map1, map2};
use crate::buffer::{Buffer, DType, Element, View, Width};
use crate::reduce::{Reduction, along, scan_along};
use crate::slice::{Arg, Binary, CHUNK, Ternary, Unary};
use crate::linalg::Inner;
use crate::parallel::{extended, stack_with};

/// An object's kernel at one dtype for one argument: its result at each of `len` positions, which read the argument through the mapping, and whether every result succeeded.
pub type UnaryKernel<K> = for<'a> fn(&K, View<'a>, &Mapping, usize) -> (Buffer, bool);
/// An object's kernel at one dtype for two arguments, which pair as the agreement says.
pub type BinaryKernel<K> = for<'a> fn(&K, View<'a>, View<'a>, &Agreement) -> (Buffer, bool);
/// An object's kernel at one dtype for three arguments. The first agreement pairs the first two arguments, and the second pairs the first agreement's layout with the third.
pub type TernaryKernel<K> = for<'a> fn(&K, [View<'a>; 3], &Agreement, &Agreement) -> (Buffer, bool);
/// A reduction's fold at one dtype, along the middle of the dimensions `[outer, len, inner]`, giving one total for each outer and inner position.
pub type FoldKernel<R> = for<'a> fn(&R, View<'a>, [usize; 3]) -> (Buffer, bool);
/// A reduction's scan at one dtype, along the middle of the dimensions `[outer, len, inner]`, giving a running total at each position.
pub type ScanKernel<R> = for<'a> fn(&R, View<'a>, [usize; 3]) -> (Buffer, bool);
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

impl<R> Kernels<FoldKernel<R>> {
    /// `r`'s totals of `x` along the middle of `dims`, and whether every total succeeded. `None` where `r` has no fold for `x`'s dtype.
    pub fn fold(&self, r: &R, x: View<'_>, dims: [usize; 3]) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(r, cast(x, at, &mut None), dims))
    }
}

impl<R> Kernels<ScanKernel<R>> {
    /// `r`'s running totals of `x` along the middle of `dims`, and whether every total succeeded. `None` where `r` has no scan for `x`'s dtype.
    pub fn scan(&self, r: &R, x: View<'_>, dims: [usize; 3]) -> Option<(Buffer, bool)> {
        let (kernel, at) = self.chosen(x.dtype())?;
        Some(kernel(r, cast(x, at, &mut None), dims))
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

fn fold_at<R: Reduction<T>, T: Element>(_: &R, x: View<'_>, dims: [usize; 3]) -> (Buffer, bool) where R::Total: Element {
    let (totals, ok) = along::<T, R>(items::<T>(x), dims);
    (R::Total::buffer(totals), ok)
}

fn scan_at<R: Reduction<T>, T: Element>(_: &R, x: View<'_>, dims: [usize; 3]) -> (Buffer, bool) where R::Total: Element {
    let mut out = Vec::with_capacity(dims.iter().product());
    let ok = scan_along::<T, R>(items::<T>(x), dims, None, &mut out);
    (R::Total::buffer(out), ok)
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
    impl<R: Reduction<T>, T: Element> Found<FoldKernel<R>> for Probe<R, T, Fold> where R::Total: Element {
        fn kernel(&self) -> Option<FoldKernel<R>> { Some(fold_at::<R, T>) }
    }
    impl<R: Reduction<T>, T: Element> Found<ScanKernel<R>> for Probe<R, T, Scan> where R::Total: Element {
        fn kernel(&self) -> Option<ScanKernel<R>> { Some(scan_at::<R, T>) }
    }
    impl<P: crate::linalg::Inner<T, Out: Element + Default> + Sync, T: Element> Found<ProductKernel<P>> for Probe<P, T, Product> {
        fn kernel(&self) -> Option<ProductKernel<P>> { Some(product_at::<P, T>) }
    }
}

/// The [`Kernels`](crate::operation::Kernels) of a function object or reduction: `Unary`, `Binary` or `Ternary` for a function object of that arity, `Fold` or `Scan` for a [`Reduction`](crate::reduce::Reduction), and `Product` for a matrix product's [`Inner`](crate::linalg::Inner) pair. Name the object by its type, or by a value after `of`. Each kernel's test resolves where the macro expands, so the type must be concrete there. In generic code the tests see only the type parameter's bounds, not the impls of the type it later stands for, so a kernel that the bounds don't promise comes out as `None`, with no error.
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
