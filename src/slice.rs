//! Slice forms of rustymath's functions. Each function object, such as [`Pow`], has a scalar form `one` and a slice form `many`.
//! Both run code that rustymath compiles once for each item type, inside its own crate. A caller's build settings can't change that
//! code, and rustymath's speed test times what callers run. An edit to a function's body rebuilds only rustymath.
use crate::float::{
    sealed::{Forms, Item},
    Float,
};
use num_complex::Complex;
use half::f16;
use std::mem::MaybeUninit;

/// How many items a caller handles at a time where it converts or expands arguments into buffers for a slice form. Buffers of this
/// many items stay in cache.
pub const CHUNK: usize = 4096;

/// How many independent running values a loop keeps, each of `size` bytes: a fold's partial results, or a block of items under
/// Horner's rule. 32 measured faster on the Mac than 8 or 16 for items of up to 8 bytes. A 16-byte item, such as a `Complex<f64>`, keeps
/// 16, because 32 fill every vector register, and a complex sum then took 1.7 times as long. In wasm, LLVM leaves 32 values of 8 bytes,
/// or 16 of 16 bytes, in linear memory, where each step loads and stores them. It keeps half as many in locals.
pub(crate) const fn running(size: usize) -> usize {
    let wasm = cfg!(target_arch = "wasm32");
    if size >= 16 { if wasm { 8 } else { 16 } } else if wasm && size == 8 { 16 } else { 32 }
}

/// One argument of a slice function: one item that every result pairs with, or one item for each result.
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a, T> {
    One(T),
    Many(&'a [T]),
}

impl<T: Copy> Arg<'_, T> {
    /// Items `start..start + n` of this argument, each converted by `f`. A `Many` argument converts into `buf`.
    #[inline]
    pub fn converted<'b, A: Send>(self, start: usize, n: usize, buf: &'b mut Vec<A>, f: impl Fn(T) -> A + Sync) -> Arg<'b, A>
    where
        T: Sync,
    {
        match self.part(start, n) {
            Arg::One(v) => Arg::One(f(v)),
            many => {
                buf.clear();
                pairs(many, Arg::One(()), n, buf, |v, ()| (f(v), true));
                Arg::Many(buf)
            }
        }
    }

    /// Items `start..start + n` of this argument. A `One` argument stays as it is.
    #[inline(always)]
    pub(crate) fn part(self, start: usize, n: usize) -> Self {
        match self {
            Arg::One(v) => Arg::One(v),
            Arg::Many(s) => Arg::Many(&s[start..start + n]),
        }
    }
}

/// Defines each arity's public trait, with a blanket impl at each item type that reaches the forms `forms!` writes. The types pass
/// through `@concrete` one at a time, so that the argument names don't repeat inside the types' repetition.
macro_rules! arities {
    ($($(#[$doc:meta])* $name:ident<$n:literal>($($x:ident),+);)+) => {$(
        $(#[$doc])*
        pub trait $name<T> {
            /// The type of each result.
            type Out;
            /// The function of one item of each argument.
            fn one(&self, $($x: T),+) -> Self::Out;
            /// Appends `one` for each of `n` results to `out`, and gives whether every result succeeded. A `Many` argument has at least
            /// `n` items, and a `One` argument pairs its item with every result. A float or complex result fails where it is NaN and no
            /// argument is, because rustymath gives NaN for an argument outside a function's domain. An integer result fails where it
            /// doesn't fit its type.
            fn many(&self, $($x: Arg<'_, T>,)+ n: usize, out: &mut Vec<Self::Out>) -> bool;
            /// `one`, or `None` where the item fails, as `many` reports.
            fn checked(&self, $($x: T),+) -> Option<Self::Out>;
        }
        arities!(@concrete $name, $n, ($($x),+): f16, f32, f64, Complex<f32>, Complex<f64>, i8, i16, i32, i64, u8, u16, u32, u64, bool);
    )+};
    (@concrete $name:ident, $n:literal, $args:tt: $e:ty $(, $rest:ty)*) => {
        arities!(@impl $name, $n, $args, $e);
        arities!(@concrete $name, $n, $args: $($rest),*);
    };
    (@concrete $name:ident, $n:literal, $args:tt:) => {};
    (@impl $name:ident, $n:literal, ($($x:ident),+), $e:ty) => {
        impl<K: Forms<$e, $n>> $name<$e> for K {
            type Out = <K as Forms<$e, $n>>::Out;
            #[inline]
            fn one(&self, $($x: $e),+) -> Self::Out { self.scalar([$($x),+]) }
            #[inline]
            fn many(&self, $($x: Arg<'_, $e>,)+ n: usize, out: &mut Vec<Self::Out>) -> bool {
                crate::slice::appended(out, n, |slots| crate::parallel::elementwise(slots, &|start, slots| {
                    crate::parallel::in_chunks(start, slots, |at, part| self.slice([$($x.part(at, part.len())),+], part.len(), part))
                }))
            }
            #[inline]
            fn checked(&self, $($x: $e),+) -> Option<Self::Out> { self.checked_scalar([$($x),+]) }
        }
    };
}

arities! {
    /// A function of one item as a value, with a scalar form and a slice form. rustymath's function objects, such as [`Exp`](crate::Exp),
    /// implement it at each item type they take: every [`Float`] type for a float function, and complex numbers, integers or Booleans for others.
    /// `Out` is the type of each result, which is the item type unless the function gives another, as [`Abs`](crate::Abs) gives the
    /// real magnitude of a complex number. Its methods take `&self`, so a function object can hold settings.
    Unary<1>(x);
    /// A function of two items as a value, as [`Unary`] is for one.
    Binary<2>(x, y);
    /// A function of three items as a value, as [`Unary`] is for one.
    Ternary<3>(x, y, z);
}

/// `K` with its two arguments in the other order, as a [`Binary`] at each item type where `K` is one: `Flipped(k).one(x, y)` is
/// `k.one(y, x)`, and likewise for `many` and `checked`.
#[derive(Clone, Copy, Debug)]
pub struct Flipped<K>(pub K);

impl<E: Item, K: Forms<E, 2>> Forms<E, 2> for Flipped<K> {
    type Out = K::Out;
    #[inline]
    fn scalar(&self, [x, y]: [E; 2]) -> K::Out { self.0.scalar([y, x]) }
    #[inline]
    fn checked_scalar(&self, [x, y]: [E; 2]) -> Option<K::Out> { self.0.checked_scalar([y, x]) }
    #[inline]
    fn slice(&self, [x, y]: [Arg<'_, E>; 2], n: usize, out: &mut [MaybeUninit<K::Out>]) -> bool { self.0.slice([y, x], n, out) }
}

/// Whether `r`, the result for arguments `args`, keeps the rule that a NaN result needs a NaN argument.
#[inline(always)]
pub(crate) fn kept<O: Item, E: Item, const N: usize>(r: O, args: [E; N]) -> bool { !r.has_nan() | args.iter().any(|a| a.has_nan()) }

/// What a function object's item body gives: an output, which fails where it is NaN and no argument is, an `Option` of one, which
/// fails where it is `None`, or a [`Total`] output, which never fails. An integer function, whose result can overflow, gives an
/// `Option`. An output that is never NaN, such as `bool`, fails only through `None`.
pub(crate) trait Outcome<O: Item>: Sized {
    /// Whether an output that `split` fails can still succeed, as a NaN output does where an argument is NaN. Only a plain output can.
    const RESCUE: bool;
    /// The output, as a scalar form gives it. `None` panics, because a scalar form has no other way to report it.
    fn value(self) -> O;
    /// The output, and whether it succeeded, judged by the output alone, so that a NaN output fails. That costs one test for each item,
    /// where testing each argument too would cost one more for each argument.
    fn split(self) -> (O, bool);
    /// The output, and whether it succeeded for arguments `args`.
    #[inline(always)]
    fn judged<E: Item, const N: usize>(self, args: [E; N]) -> (O, bool) {
        let (r, ok) = self.split();
        (r, ok || (Self::RESCUE && kept(r, args)))
    }
}

impl<O: Item> Outcome<O> for O {
    const RESCUE: bool = true;
    #[inline(always)]
    fn value(self) -> O { self }
    #[inline(always)]
    fn split(self) -> (O, bool) { (self, !self.has_nan()) }
}

impl<O: Item + Default> Outcome<O> for Option<O> {
    const RESCUE: bool = false;
    #[inline(always)]
    fn value(self) -> O { self.expect("the result doesn't fit the output type") }
    #[inline(always)]
    fn split(self) -> (O, bool) { (self.unwrap_or_default(), self.is_some()) }
}

/// The output of an item body that never fails: one that never gives NaN from arguments that aren't NaN, such as `exp`'s, or one whose
/// NaN is the function's value, such as that of `∞ - ∞`. The slice loop tests none of them.
#[derive(Clone, Copy)]
pub(crate) struct Total<O>(pub O);

impl<O: Item> Outcome<O> for Total<O> {
    const RESCUE: bool = false;
    #[inline(always)]
    fn value(self) -> O { self.0 }
    #[inline(always)]
    fn split(self) -> (O, bool) { (self.0, true) }
}

/// The items of `x` that result `i` takes.
#[inline(always)]
fn items<E: Copy, const N: usize>(x: &[Arg<'_, E>; N], i: usize) -> [E; N] { std::array::from_fn(|k| match x[k] { Arg::One(v) => v, Arg::Many(s) => s[i] }) }

/// Writes `f` for each of `n` results into `out`, for arguments `x`, and gives whether `f` said `true` for every result. One and two
/// arguments run through [`pairs_into`], whose loops vectorize.
#[inline(always)]
fn mapped<E: Copy, O, const N: usize>(x: &[Arg<'_, E>; N], n: usize, out: &mut [MaybeUninit<O>], f: impl Fn([E; N]) -> (O, bool)) -> bool {
    match x.as_slice() {
        &[a] => pairs_into(a, Arg::One(()), n, out, #[inline(always)] |a, ()| f([a; N])),
        &[a, b] => pairs_into(a, b, n, out, #[inline(always)] |a, b| f(std::array::from_fn(|k| if k == 0 { a } else { b }))),
        _ => {
            let mut ok = true;
            for (i, slot) in out[..n].iter_mut().enumerate() {
                let (r, good) = f(items(x, i));
                slot.write(r);
                ok &= good;
            }
            ok
        }
    }
}

/// Writes `f` for each of `n` results into `out`, for arguments `x`, and gives whether every result succeeded, as [`Outcome`] says.
/// Slice forms call it through [`each!`]. The loop judges each output alone, and a [`Total`] output not at all. Only where some output
/// is NaN does a second pass look at the arguments of the NaN outputs, which is rare.
#[inline(always)]
pub(crate) fn each<E: Item, O: Item, R: Outcome<O>, const N: usize>(
    x: [Arg<'_, E>; N], n: usize, out: &mut [MaybeUninit<O>], f: impl Fn([E; N]) -> R,
) -> bool {
    mapped(&x, n, out, #[inline(always)] |a| f(a).split())
        || (R::RESCUE && written(&mut out[..n]).iter().enumerate().all(|(i, &r)| kept(r, items(&x, i))))
}

/// [`each`] with the item body marked `#[inline(always)]`. Every slice form runs its loop this way: `forms!`'s default slice form, and
/// a custom slice form once it has read its settings. LLVM leaves a large unmarked body out of line, and its call stops the loop
/// vectorizing.
macro_rules! each {
    ($x:expr, $n:expr, $out:expr, |$args:pat_param| $body:expr $(,)?) => {
        crate::slice::each($x, $n, $out, #[inline(always)] |$args| $body)
    };
}

/// Writes a function object's forms, compiled inside rustymath. The list names the item types, such as `Complex<f32>` or `i64`. `real`
/// stands for `f32` and `f64`, and also writes an `f16` form, which runs the `f32` form and narrows its outputs. `=> T` makes the
/// output type `T` in place of the item type: for each listed type, or after `real` for every float type. The object is the first
/// argument of the item body, and its items the bracketed ones. The body gives an output, an `Option` of one, or a [`Total`] output,
/// as [`Outcome`] says. A `many` body replaces the default slice form, which runs the item body through [`each!`]. Its arguments are
/// the object, the bracketed `Arg`s, the number of results, and the slots it writes them into.
///
/// A function object is a concrete type, never a generic one, so that its forms compile here. Settings that choose a loop, such as
/// whether a tolerance is 0, are read once per call inside `many`, which then runs `each!`.
///
/// ```ignore
/// forms!(Sqrt at real: |_, [x]| x.sqrt());
/// forms!(Abs at Complex<f32> => f32, Complex<f64> => f64: |_, [z]| p::hypot::<_, { crate::FUSED }>(z.re, z.im));
/// forms!(Gcd at i64: |_, [x, y]| gcd(x, y));
/// forms!(Less at real => bool: |k, [x, y]| with_tolerance!(k.0, |t| t.less(x, y)),
///     many |k, [x, y], n, out| with_tolerance!(k.0, |t| each!([x, y], n, out, |[x, y]| t.less(x, y))));
/// ```
macro_rules! forms {
    ($object:ident at real $(=> $o:ty)? : |$k:pat_param, $args:tt| $body:expr $(, many |$mk:pat_param, $margs:tt, $n:ident, $out:ident| $many:expr)?) => {
        forms!(@types $object, $k, $args, $body, [$((|$mk, $margs, $n, $out| $many))?], [f32 $(=> $o)?, f64 $(=> $o)?]);
        forms!(@half $object, forms!(@out half::f16 $(, $o)?), forms!(@narrow $($o)?), $args);
    };
    ($object:ident at $($e:ty $(=> $o:ty)?),+ : |$k:pat_param, $args:tt| $body:expr $(, many |$mk:pat_param, $margs:tt, $n:ident, $out:ident| $many:expr)?) => {
        forms!(@types $object, $k, $args, $body, [$((|$mk, $margs, $n, $out| $many))?], [$($e $(=> $o)?),+]);
    };
    // One type at a time, so that the optional `many` group, which repeats on its own, needn't sit inside the types' repetition.
    (@types $object:ident, $k:pat_param, $args:tt, $body:expr, $many:tt, [$e:ty $(=> $o:ty)? $(, $($rest:tt)+)?]) => {
        forms!(@concrete $object, $e, forms!(@out $e $(, $o)?), $k, $args, $body, $many);
        $(forms!(@types $object, $k, $args, $body, $many, [$($rest)+]);)?
    };
    (@concrete $object:ident, $e:ty, $o:ty, $k:pat_param, [$($x:ident),+], $body:expr, $many:tt) => {
        impl crate::float::sealed::Forms<$e, { forms!(@n $($x),+) }> for $object {
            type Out = $o;
            fn scalar(&self, [$($x),+]: [$e; forms!(@n $($x),+)]) -> $o {
                let $k = self;
                crate::slice::Outcome::<$o>::value($body)
            }
            fn checked_scalar(&self, [$($x),+]: [$e; forms!(@n $($x),+)]) -> Option<$o> {
                let $k = self;
                let (r, ok) = crate::slice::Outcome::<$o>::judged($body, [$($x),+]);
                ok.then_some(r)
            }
            forms!(@slice $e => $o, $k, [$($x),+], $body, $many);
        }
    };
    (@slice $e:ty => $o:ty, $k:pat_param, [$($x:ident),+], $body:expr, []) => {
        fn slice(&self, args: [crate::Arg<'_, $e>; forms!(@n $($x),+)], n: usize, out: &mut [std::mem::MaybeUninit<$o>]) -> bool {
            let $k = self;
            each!(args, n, out, |[$($x),+]| $body)
        }
    };
    (@slice $e:ty => $o:ty, $k:pat_param, $args:tt, $body:expr, [(|$mk:pat_param, [$($mx:ident),+], $n:ident, $out:ident| $many:expr)]) => {
        fn slice(&self, [$($mx),+]: [crate::Arg<'_, $e>; forms!(@n $($mx),+)], $n: usize, $out: &mut [std::mem::MaybeUninit<$o>]) -> bool {
            let $mk = self;
            $many
        }
    };
    (@half $object:ident, $o:ty, $narrow:expr, [$($x:ident),+]) => {
        impl crate::float::sealed::Forms<half::f16, { forms!(@n $($x),+) }> for $object {
            type Out = $o;
            fn scalar(&self, x: [half::f16; forms!(@n $($x),+)]) -> $o {
                ($narrow)(crate::float::sealed::Forms::<f32, { forms!(@n $($x),+) }>::scalar(self, x.map(half::f16::to_f32)))
            }
            fn checked_scalar(&self, x: [half::f16; forms!(@n $($x),+)]) -> Option<$o> {
                crate::float::sealed::Forms::<f32, { forms!(@n $($x),+) }>::checked_scalar(self, x.map(half::f16::to_f32)).map($narrow)
            }
            fn slice(&self, x: [crate::Arg<'_, half::f16>; forms!(@n $($x),+)], n: usize, out: &mut [std::mem::MaybeUninit<$o>]) -> bool {
                crate::slice::working(x, n, out, |x, m, o| crate::float::sealed::Forms::<f32, { forms!(@n $($x),+) }>::slice(self, x, m, o), $narrow)
            }
        }
    };
    (@out $e:ty) => { $e };
    (@out $e:ty, $o:ty) => { $o };
    (@narrow) => { half::f16::from_f32 };
    (@narrow $o:ty) => { std::convert::identity };
    (@n $($x:ident),+) => { [$(stringify!($x)),+].len() };
}

/// Runs `$body` for each item `$i` of a result of `$n` items, with `$a` and `$b` the items of `$x` and `$y` that pair with it. Each
/// pairing of `One` and `Many` has a loop of its own. The loops index slices of the result's length with no iterator adapters, and
/// vectorize at any build settings.
macro_rules! each_pair {
    ($x:expr, $y:expr, $n:expr, |$i:ident, $a:ident, $b:ident| $body:expr) => {{
        let n = $n;
        match ($x, $y) {
            (Arg::One($a), Arg::One($b)) => for $i in 0..n { $body },
            (Arg::One($a), Arg::Many(ys)) => {
                let ys = &ys[..n];
                for $i in 0..n { let $b = ys[$i]; $body }
            }
            (Arg::Many(xs), Arg::One($b)) => {
                let xs = &xs[..n];
                for $i in 0..n { let $a = xs[$i]; $body }
            }
            (Arg::Many(xs), Arg::Many(ys)) => {
                let (xs, ys) = (&xs[..n], &ys[..n]);
                for $i in 0..n { let ($a, $b) = (xs[$i], ys[$i]); $body }
            }
        }
    }};
}

/// Appends `f` on each of `n` pairs of items to `out`, and gives whether `f` said `true` for every pair. An `Arg::One` pairs with
/// every item of the other argument. The loops write straight into `out`'s spare capacity, which is never zeroed first. This is the
/// loop that rustymath's slice forms run, and a caller's own kernels can run it too, as basedpl's arithmetic does. A long call splits
/// across threads, as [`parallel`](crate::parallel) describes.
#[inline(never)]
pub fn pairs<X: Copy + Sync, Y: Copy + Sync, B: Send>(x: Arg<'_, X>, y: Arg<'_, Y>, n: usize, out: &mut Vec<B>, f: impl Fn(X, Y) -> (B, bool) + Sync) -> bool {
    appended(out, n, |slots| crate::parallel::elementwise(slots, &|start, slots| pairs_into(x.part(start, slots.len()), y.part(start, slots.len()), slots.len(), slots, |a, b| f(a, b))))
}

/// [`pairs`], writing into the `n` slots of `out` in place of appending to a vector.
#[inline]
pub(crate) fn pairs_into<X: Copy, Y: Copy, B>(x: Arg<'_, X>, y: Arg<'_, Y>, n: usize, out: &mut [MaybeUninit<B>], f: impl Fn(X, Y) -> (B, bool)) -> bool {
    let (out, mut ok) = (&mut out[..n], true);
    each_pair!(x, y, n, |i, a, b| {
        let (r, good) = f(a, b);
        out[i].write(r);
        ok &= good;
    });
    ok
}

/// Appends `n` items to `out`, which `write` writes into its spare capacity, never zeroed first, and gives `write`'s result. `write`
/// must write every slot it gets. Each caller's loop does, by its construction.
#[inline(always)]
pub(crate) fn appended<B, T>(out: &mut Vec<B>, n: usize, write: impl FnOnce(&mut [MaybeUninit<B>]) -> T) -> T {
    out.reserve(n);
    let result = write(&mut out.spare_capacity_mut()[..n]);
    // SAFETY: `write` wrote all `n` slots.
    unsafe { out.set_len(out.len() + n) }
    result
}

/// `out`'s slots as items. Each caller passes only slots that its loop has just written.
#[inline(always)]
pub(crate) fn written<B>(out: &mut [MaybeUninit<B>]) -> &mut [B] {
    // SAFETY: the caller has written every slot.
    unsafe { &mut *(std::ptr::from_mut(out) as *mut [B]) }
}

/// Writes `f` for each of `n` results into `out`, for arguments `x`, for a function `f` with a fast form that holds for all but rare items.
/// `fast` computes every item with no branch, and gives `None` for the rare items. The items run in two passes. The first writes NaN
/// for each rare item and judges each result alone, as [`each`] does. The second runs only if a result is NaN. It recomputes each NaN
/// item with `f` and judges it by its arguments too. A rare item costs one more scan of at most one `CHUNK`, the most that
/// `parallel::in_chunks` hands a slice form. Slice forms call it through [`two_pass!`].
#[inline(always)]
pub(crate) fn two_pass<R: crate::Float, const N: usize>(
    x: [Arg<'_, R>; N], n: usize, out: &mut [MaybeUninit<R>], fast: impl Fn([R; N]) -> Option<R>, f: impl Fn([R; N]) -> R,
) -> bool {
    let out = &mut out[..n];
    let first = mapped(&x, n, out, #[inline(always)] |a| {
        let r = fast(a).unwrap_or(R::nan());
        (r, !r.is_nan())
    });
    if first { return true; }
    let mut ok = true;
    for (i, r) in written(out).iter_mut().enumerate().filter(|(_, r)| r.is_nan()) {
        let a = items(&x, i);
        *r = f(a);
        ok &= kept(*r, a);
    }
    ok
}

/// [`two_pass`] with both item bodies marked `#[inline(always)]`, as [`each!`] marks its body.
macro_rules! two_pass {
    ($x:expr, $n:expr, $out:expr, |$fa:pat_param| $fast:expr, |$sa:pat_param| $f:expr $(,)?) => {
        crate::slice::two_pass($x, $n, $out, #[inline(always)] |$fa| $fast, #[inline(always)] |$sa| $f)
    };
}

/// Writes `f` for each of `n` results into `out`, for complex arguments `x`, through real slice forms. The items split into a slice of
/// real parts and a slice of imaginary parts for each argument, with a `One` argument repeated. `fast` computes the results' real
/// and imaginary parts from those slices, through real slice forms, and appends them. Each result that `rare` picks, from its
/// arguments and its fast value, then takes `f`, the scalar function, which handles what the fast formula doesn't, such as tiny or
/// infinite parts. The results succeed as [`each`] judges them. Slice forms call it through [`split!`].
pub(crate) fn split<R: crate::Native, const N: usize>(
    x: [Arg<'_, Complex<R>>; N], n: usize, out: &mut [MaybeUninit<Complex<R>>], fast: impl Fn([(&[R], &[R]); N], &mut Vec<R>, &mut Vec<R>),
    rare: impl Fn([Complex<R>; N], Complex<R>) -> bool, f: impl Fn([Complex<R>; N]) -> Complex<R>,
) -> bool
where
    Complex<R>: Item,
{
    let (mut parts, mut re, mut im) = ([(); N].map(|()| (Vec::new(), Vec::new())), Vec::new(), Vec::new());
    let out = &mut out[..n];
    for ((xs, ys), a) in parts.iter_mut().zip(x) {
        match a {
            Arg::One(z) => {
                xs.resize(n, z.re);
                ys.resize(n, z.im);
            }
            Arg::Many(s) => {
                xs.extend(s[..n].iter().map(|z| z.re));
                ys.extend(s[..n].iter().map(|z| z.im));
            }
        }
    }
    fast(std::array::from_fn(|k| (&parts[k].0[..], &parts[k].1[..])), &mut re, &mut im);
    assert!(re.len() == n && im.len() == n, "`fast` gives the parts of every result");
    for ((slot, &a), &b) in out.iter_mut().zip(&re).zip(&im) { slot.write(Complex::new(a, b)); }
    let zs = written(out);
    if (0..n).fold(false, |any, i| any | rare(items(&x, i), zs[i])) {
        for i in 0..n {
            let a = items(&x, i);
            if rare(a, zs[i]) { zs[i] = f(a); }
        }
    }
    zs.iter().all(|r| !r.has_nan()) || zs.iter().enumerate().all(|(i, &r)| kept(r, items(&x, i)))
}

/// [`split`] with `rare` marked `#[inline(always)]`, as [`each!`] marks its body, because the scan for rare items runs on every item.
macro_rules! split {
    ($x:expr, $n:expr, $out:expr, |$pa:pat_param, $re:pat_param, $im:pat_param| $fast:expr, |$ra:pat_param, $rr:pat_param| $rare:expr, |$fa:pat_param| $f:expr $(,)?) => {
        crate::slice::split($x, $n, $out, |$pa, $re, $im| $fast, #[inline(always)] |$ra, $rr| $rare, |$fa| $f)
    };
}

/// `f`, a slice form at `F`'s working type, for narrower floats, as `forms!`'s `f16` forms run it: the items convert to the working
/// type and run, and `narrow` turns each output into the output at `F`.
pub(crate) fn working<F: Float, O: Copy, P, const N: usize>(
    x: [Arg<'_, F>; N], n: usize, out: &mut [MaybeUninit<P>], f: impl Fn([Arg<'_, F::Working>; N], usize, &mut [MaybeUninit<O>]) -> bool,
    narrow: impl Fn(O) -> P,
) -> bool {
    let (mut bufs, mut rs) = ([(); N].map(|()| Vec::new()), Vec::new());
    let mut next = bufs.each_mut().into_iter();
    let ok = appended(&mut rs, n, |o| f(x.map(|a| a.converted(0, n, next.next().unwrap(), F::working)), n, o));
    pairs_into(Arg::Many(&rs), Arg::One(()), n, &mut out[..n], |r, ()| (narrow(r), true));
    ok
}

