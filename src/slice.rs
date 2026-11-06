//! Slice forms of rustymath's functions. Each function object, such as [`Pow`], has a scalar form `one` and a slice form `many`. Both run code that rustymath compiles once for each item type, inside its own crate. A caller's build settings can't change that code, and rustymath's speed test times what callers run. An edit to a function's body rebuilds only rustymath.
use crate::float::Float;
use num_complex::Complex;
use std::mem::MaybeUninit;

/// How many items a caller handles at a time where it converts or expands arguments into buffers for a slice form. Buffers of this many items stay in cache.
pub const CHUNK: usize = 4096;

/// How many independent running values a loop keeps, each of `size` bytes: a fold's partial results, or a block of items under Horner's rule. 32 measured faster on the Mac than 8 or 16 for items of up to 8 bytes. A 16-byte item, such as a `Complex<f64>`, keeps 16, because 32 fill every vector register, and a complex sum then took 1.7 times as long. In wasm, LLVM leaves 32 values of 8 bytes, or 16 of 16 bytes, in linear memory, where each step loads and stores them. It keeps half as many in locals.
pub const fn running(size: usize) -> usize {
    let wasm = cfg!(target_arch = "wasm32");
    if size >= 16 { if wasm { 8 } else { 16 } } else if wasm && size == 8 { 16 } else { 32 }
}

/// One argument of a slice function: one item that every result pairs with, or one item for each result.
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a, T> { One(T), Many(&'a [T]) }

impl<'a, T: Copy> Arg<'a, T> {
    /// The argument that `items` gives: its only item for every result when it holds one, otherwise one item for each result.
    pub fn of(items: &'a [T]) -> Self { if let [x] = *items { Arg::One(x) } else { Arg::Many(items) } }
    /// Item `i` of this argument, or its one item.
    #[inline(always)]
    pub fn nth(self, i: usize) -> T { match self { Arg::One(v) => v, Arg::Many(s) => s[i] } }
    /// The one item of an argument that holds one for every result.
    #[inline(always)]
    pub fn one(self) -> Option<T> { match self { Arg::One(v) => Some(v), Arg::Many(_) => None } }
    /// Items `start..start + n` of this argument, each converted by `f`. A `Many` argument converts into `buf`.
    #[inline]
    pub fn converted<'b, A>(self, start: usize, n: usize, buf: &'b mut Vec<A>, f: impl Fn(T) -> A) -> Arg<'b, A> {
        match self.part(start, n) {
            Arg::One(v) => Arg::One(f(v)),
            Arg::Many(items) => {
                buf.clear();
                buf.extend(items.iter().map(|&v| f(v)));
                Arg::Many(buf)
            }
        }
    }

    /// Items `start..start + n` of this argument. A `One` argument stays as it is.
    #[inline(always)]
    pub(crate) fn part(self, start: usize, n: usize) -> Self { match self { Arg::One(v) => Arg::One(v), Arg::Many(s) => Arg::Many(&s[start..start + n]) } }
}

/// A type that function objects take or give: a float, a complex number, an integer, a Boolean, a character or an ordering. Pieces on several threads read and write items.
pub trait KernelItem: Copy + Send + Sync {
    /// Whether the type has NaN, as floats and complex numbers do.
    const NAN: bool;
    /// Whether this item is NaN, or for a complex number has a NaN part. An integer or Boolean never is.
    fn has_nan(self) -> bool;
}

macro_rules! nan_items { ($($t:ty),+) => {$( impl KernelItem for $t { const NAN: bool = true; #[inline(always)] fn has_nan(self) -> bool { self.is_nan() } } )+} }
nan_items!(f32, f64, half::f16, Complex<f32>, Complex<f64>);
macro_rules! exact_items { ($($t:ty),+) => {$( impl KernelItem for $t { const NAN: bool = false; #[inline(always)] fn has_nan(self) -> bool { false } } )+} }
exact_items!(u8, i16, i32, i64, bool, char, std::cmp::Ordering);

/// The results of a function object of `N` items of type `T`: their type and whether any can fail. [`Unary`], [`Binary`] and [`Ternary`] each require it, so code that takes a function object of any arity reads these through one bound.
pub trait Results<T, const N: usize> {
    /// The type of each result.
    type Out: KernelItem;
    /// Whether no result fails: `checked` always gives `Some`, and `many` and `many_into` always give true. A total float function gives NaN from arguments that aren't NaN only where NaN is its value, as [`Subtract`](crate::Subtract) gives for ∞ - ∞ and [`Sin`](crate::Sin) for ±∞.
    const TOTAL: bool;
}

/// Native scalar and slice interfaces. The slice body initializes every output slot even when an item fails.
macro_rules! arities {
    ($($(#[$doc:meta])* $name:ident<$n:literal>($($x:ident),+);)+) => {$(
        $(#[$doc])*
        ///
        /// # Safety
        /// The slice body initializes every slot in its output before returning.
        pub unsafe trait $name<T>: Results<T, $n> + Sync {
            fn one(&self, $($x: T),+) -> Self::Out;
            fn judged(&self, $($x: T),+) -> (Self::Out, bool);
            fn slice(&self, args: [Arg<'_, T>; $n], n: usize, out: &mut [MaybeUninit<Self::Out>]) -> bool;
            #[inline]
            fn checked(&self, $($x: T),+) -> Option<Self::Out> { let (r, ok) = self.judged($($x),+); ok.then_some(r) }
            fn many(&self, $($x: Arg<'_, T>,)+ n: usize, out: &mut Vec<Self::Out>) -> bool where T: Copy + Sync {
                appended(out, n, |slots| crate::parallel::split(slots, CHUNK, &|start, slots| {
                    crate::parallel::in_chunks(start, slots, |at, part| self.slice([$($x.part(at, part.len())),+], part.len(), part))
                }))
            }
            fn many_into(&self, $($x: Arg<'_, T>,)+ n: usize, out: &mut crate::parallel::Slots<'_, Self::Out>) -> bool where T: Copy {
                out.fill(n, |slots| crate::parallel::in_chunks(0, slots, |at, part| self.slice([$($x.part(at, part.len())),+], part.len(), part)))
            }
        }
    )+};
}

arities! {
    /// A function of one item as a value, with a scalar form and a slice form. rustymath's function objects, such as [`Exp`](crate::Exp), implement it at each item type they take: every [`Float`] type for a float function, and complex numbers, integers or Booleans for others. `Out` is the type of each result, which is the item type unless the function gives another, as [`Abs`](crate::Abs) gives the real magnitude of a complex number. Its methods take `&self`, so a function object can hold settings.
    Unary<1>(x);
    /// A function of two items as a value, as [`Unary`] is for one.
    Binary<2>(x, y);
    /// A function of three items as a value, as [`Unary`] is for one.
    Ternary<3>(x, y, z);
}

/// `K` with its two arguments in the other order, as a [`Binary`] at each item type where `K` is one: `Flipped(k).one(x, y)` is `k.one(y, x)`, and likewise for `many` and `checked`.
#[derive(Clone, Copy, Debug)]
pub struct Flipped<K>(pub K);

impl<T, K: Results<T, 2>> Results<T, 2> for Flipped<K> {
    type Out = K::Out;
    const TOTAL: bool = K::TOTAL;
}
unsafe impl<T, K: Binary<T>> Binary<T> for Flipped<K> {
    fn one(&self, x: T, y: T) -> K::Out { self.0.one(y, x) }
    fn judged(&self, x: T, y: T) -> (K::Out, bool) { self.0.judged(y, x) }
    fn slice(&self, [x, y]: [Arg<'_, T>; 2], n: usize, out: &mut [MaybeUninit<K::Out>]) -> bool { self.0.slice([y, x], n, out) }
}

/// `K` with no result failing. Each result is `K`'s, wrapped to its type, as static kinds and NumPy keep integer results. LLVM removes `K`'s test of whether a result fits, because the loop never reads it. The adapter suits an object whose scalar form is its whole computation, as integer arithmetic's is.
#[derive(Clone, Copy, Debug)]
pub struct Wrapped<K>(pub K);

impl<T, K: Results<T, N>, const N: usize> Results<T, N> for Wrapped<K> {
    type Out = K::Out;
    const TOTAL: bool = true;
}
macro_rules! wrapped {
    ($arity:ident, $n:literal, $($x:ident),+) => {
        unsafe impl<T: KernelItem, K: $arity<T>> $arity<T> for Wrapped<K> {
            fn one(&self, $($x: T),+) -> K::Out { self.0.one($($x),+) }
            fn judged(&self, $($x: T),+) -> (K::Out, bool) { (self.0.one($($x),+), true) }
            fn slice(&self, x: [Arg<'_, T>; $n], n: usize, out: &mut [MaybeUninit<K::Out>]) -> bool {
                each(x, n, out, #[inline(always)] |[$($x),+]| Total(self.0.one($($x),+)))
            }
        }
    };
}
wrapped!(Unary, 1, x);
wrapped!(Binary, 2, x, y);
wrapped!(Ternary, 3, x, y, z);

/// Whether `r`, the result for arguments `args`, keeps the rule that a NaN result needs a NaN argument.
#[inline(always)]
pub(crate) fn kept<O: KernelItem, E: KernelItem, const N: usize>(r: O, args: [E; N]) -> bool { !r.has_nan() | args.iter().any(|a| a.has_nan()) }

/// What a function object's item body gives: an output, which fails where it is NaN and no argument is, an `Option` of one, which fails where it is `None`, a wrapped output paired with whether it's exact, which fails where it isn't, or a [`Total`] output, which never fails. An integer power, whose result can overflow, gives a wrapped output. An integer function with no wrapped result to give, such as a binomial coefficient, gives an `Option`. An output that is never NaN, such as `bool`, fails only through `None` or an inexact pair.
pub(crate) trait Outcome<O: KernelItem>: Sized {
    /// Whether an output that `split` fails can still succeed, as a NaN output does where an argument is NaN. Only a plain output can.
    const RESCUE: bool;
    /// Whether no output fails: a [`Total`] output, or a plain output of a type without NaN.
    const TOTAL: bool;
    /// The output, as a scalar form gives it. `None` panics, because a scalar form has no other way to report it.
    fn value(self) -> O;
    /// The output, and whether it succeeded, judged by the output alone, so that a NaN output fails. That costs one test for each item, where testing each argument too would cost one more for each argument.
    fn split(self) -> (O, bool);
    /// The output, and whether it succeeded for arguments `args`.
    #[inline(always)]
    fn judged<E: KernelItem, const N: usize>(self, args: [E; N]) -> (O, bool) { let (r, ok) = self.split(); (r, ok || (Self::RESCUE && kept(r, args))) }
}

impl<O: KernelItem> Outcome<O> for O {
    const RESCUE: bool = true;
    const TOTAL: bool = !O::NAN;
    #[inline(always)]
    fn value(self) -> O { self }
    #[inline(always)]
    fn split(self) -> (O, bool) { (self, !self.has_nan()) }
}

impl<O: KernelItem + Default> Outcome<O> for Option<O> {
    const RESCUE: bool = false;
    const TOTAL: bool = false;
    #[inline(always)]
    fn value(self) -> O { self.expect("the result doesn't fit the output type") }
    #[inline(always)]
    fn split(self) -> (O, bool) { (self.unwrap_or_default(), self.is_some()) }
}

/// An output wrapped to its type, and whether it's the exact result, as an integer power gives it. A result that isn't exact fails, and its slot holds the wrapped output. That output is NumPy's result for fixed-width integers.
impl<O: KernelItem> Outcome<O> for (O, bool) {
    const RESCUE: bool = false;
    const TOTAL: bool = false;
    #[inline(always)]
    fn value(self) -> O { self.0 }
    #[inline(always)]
    fn split(self) -> (O, bool) { self }
}

/// The output of an item body that never fails: one that never gives NaN from arguments that aren't NaN, such as `exp`'s, or one whose NaN is the function's value, such as that of `∞ - ∞`. The slice loop tests none of them.
#[derive(Clone, Copy)]
pub(crate) struct Total<O>(pub O);

impl<O: KernelItem> Outcome<O> for Total<O> {
    const RESCUE: bool = false;
    const TOTAL: bool = true;
    #[inline(always)]
    fn value(self) -> O { self.0 }
    #[inline(always)]
    fn split(self) -> (O, bool) { (self.0, true) }
}

/// Whether an item body never fails, as its outcome type `R` says. `forms!` passes the body as a closure, which names that type without running it.
pub(crate) const fn total<K, E, O: KernelItem, R: Outcome<O>, const N: usize>(_: fn(&K, [E; N]) -> R) -> bool { R::TOTAL }

/// The items of `x` that result `i` takes.
#[inline(always)]
fn items<E: Copy, const N: usize>(x: &[Arg<'_, E>; N], i: usize) -> [E; N] { std::array::from_fn(|k| x[k].nth(i)) }

/// Writes `f` for each of `n` results into `out`, for arguments `x`, and gives whether `f` said `true` for every result. One and two arguments run through [`pairs_into`], and three through `triples_into`, whose loops vectorize.
#[inline(always)]
pub(crate) fn mapped<E: Copy, O, const N: usize>(x: &[Arg<'_, E>; N], n: usize, out: &mut [MaybeUninit<O>], f: impl Fn([E; N]) -> (O, bool)) -> bool {
    if const { N == 1 } { pairs_into(x[0], Arg::One(()), n, out, #[inline(always)] |a, ()| f([a; N])) }
    else if const { N == 2 } {
        pairs_into(x[0], x[1], n, out, #[inline(always)] |a, b| f(std::array::from_fn(|k| if k == 0 { a } else { b })))
    } else if const { N == 3 } {
        triples_into(x[0], x[1], x[2], n, out, #[inline(always)] |a, b, c| {
            f(std::array::from_fn(|k| match k { 0 => a, 1 => b, _ => c }))
        })
    } else { unreachable!("function objects take one to three arguments") }
}

/// Writes `f` for each of `n` results into `out`, for arguments `x`, and gives whether every result succeeded, as [`Outcome`] says. Slice forms call it through [`each!`]. The loop judges each output alone, and a [`Total`] output not at all. Only where some output is NaN does a second pass look at the arguments of the NaN outputs, which is rare.
#[inline(always)]
pub(crate) fn each<E: KernelItem, O: KernelItem, R: Outcome<O>, const N: usize>(
    x: [Arg<'_, E>; N], n: usize, out: &mut [MaybeUninit<O>], f: impl Fn([E; N]) -> R) -> bool {
    mapped(&x, n, out, #[inline(always)] |a| f(a).split()) || (R::RESCUE && written(&mut out[..n]).iter().enumerate().all(|(i, &r)| kept(r, items(&x, i))))
}

/// A rare form for `each_usual` that no block takes, for a function with no rare form.
pub(crate) fn none<E, O>() -> (impl Fn(E) -> bool, impl Fn(E) -> O) { (|_| false, |_| unreachable!("`none` accepts no item")) }

/// Writes a form of each item of `xs` into `out`, a block of `BLOCK` items at a time, and gives whether `mixed` said `true` for every block it wrote. `uniform` tests a whole block and gives whether every item of it takes `first`'s form, then whether every item takes `second`'s. A block that neither covers, and the last part block, goes to `mixed`. A test for each block keeps rare items, such as a function's special values, out of the loop over a uniform block. Each caller's test suits its forms: `each_usual` folds two tests over the items, and `erfc` folds a minimum and a maximum of their magnitudes.
#[inline(always)]
pub(crate) fn blocks<E: Copy, O, const BLOCK: usize>(
    xs: &[E], out: &mut [MaybeUninit<O>], uniform: impl Fn(&[E; BLOCK]) -> (bool, bool), first: impl Fn(E) -> O, second: impl Fn(E) -> O,
    mut mixed: impl FnMut(&[E], &mut [MaybeUninit<O>]) -> bool) -> bool {
    let (blocks, rest) = xs.as_chunks::<BLOCK>();
    let (slots, last) = out[..xs.len()].split_at_mut(blocks.len() * BLOCK);
    let mut ok = true;
    for (b, o) in blocks.iter().zip(slots.as_chunks_mut::<BLOCK>().0) {
        let (one, two) = uniform(b);
        if one { for k in 0..BLOCK { o[k].write(first(b[k])); } } else if two { for k in 0..BLOCK { o[k].write(second(b[k])); } } else { ok &= mixed(b, o) }
    }
    ok & mixed(rest, last)
}

/// [`each`] for one argument, through [`blocks`] of 64 items. A block whose every item `usual`'s test accepts writes `usual`'s form of each item. Failing that, a block whose every item `rare`'s test accepts writes `rare`'s form, as `exp` writes 0 for a block below its underflow bound. Any other block writes `body` of each item, judged as `each` judges it. The two forms never fail.
#[inline(always)]
pub(crate) fn each_usual<E: KernelItem, O: KernelItem, R: Outcome<O>>(
    x: Arg<'_, E>, n: usize, out: &mut [MaybeUninit<O>], usual: (impl Fn(E) -> bool, impl Fn(E) -> O), rare: (impl Fn(E) -> bool, impl Fn(E) -> O),
    body: impl Fn(E) -> R) -> bool {
    let Arg::Many(xs) = x else { return each([x], n, out, #[inline(always)] |[x]| body(x)); };
    // With blocks of 16, f64's test and the fast form's constants outgrew AArch64's vector registers, and the loop rebuilt the constants for each block.
    blocks::<_, _, 64>(
        &xs[..n], out, #[inline(always)] |b| b.iter().fold((true, true), |(one, two), &v| (one & (usual.0)(v), two & (rare.0)(v))), usual.1, rare.1,
        #[inline(always)] |b, o| {
            each([Arg::Many(b)], b.len(), o, #[inline(always)] |[x]| body(x))
        })
}

/// Declares a function object of one argument whose slice form runs `each_usual`. `$usual` says whether an item needs none of the function's special values, `$fast` computes such an item, and `$name` is the full function. An optional `$rare => $value` gives the form for a block whose items all take one rare value, as `exp` gives 0 below its underflow bound. A trailing `; total` wraps the full function's results in [`Total`].
macro_rules! usual_object {
    ($(#[doc = $doc:literal])* $object:ident: $name:ident, $usual:expr => $fast:expr $(, $rare:expr => $value:expr)? $(; $outcome:ident)?) => {
        $(#[doc = $doc])*
        #[derive(Clone, Copy, Debug)]
        pub struct $object;
        forms!($object at real: |_, [x]| usual_object!(@outcome [$($outcome)?], $name::<_, { crate::FUSED }>(x)),
            many |_, [x], n, out| crate::slice::each_usual(x, n, out, ($usual, $fast), usual_object!(@rare $($rare, $value)?),
                #[inline(always)] |x| usual_object!(@outcome [$($outcome)?], $name::<_, { crate::FUSED }>(x))));
    };
    (@rare) => { crate::slice::none() };
    (@rare $rare:expr, $value:expr) => { ($rare, $value) };
    (@outcome [], $e:expr) => { $e };
    (@outcome [total], $e:expr) => { crate::slice::Total($e) };
}

/// [`each`] with the item body marked `#[inline(always)]`. Every slice form runs its loop this way: `forms!`'s default slice form, and a custom slice form once it has read its settings. LLVM leaves a large unmarked body out of line, and its call stops the loop vectorizing.
macro_rules! each {
    ($x:expr, $n:expr, $out:expr, |$args:pat_param| $body:expr $(,)?) => {
        crate::slice::each(
            $x, $n, $out, #[inline(always)] |$args| $body,
        )
    };
}

/// Writes a function object's forms, compiled inside rustymath. The list names the item types, such as `Complex<f32>` or `i64`. `real` stands for `f32` and `f64`, and also writes an `f16` form, which runs the `f32` form and narrows its outputs. `=> T` makes the output type `T` in place of the item type: for each listed type, or after `real` for every float type. The object is the first argument of the item body, and its items the bracketed ones. The body gives an output, an `Option` of one, or a [`Total`] output, as [`Outcome`] says. A `many` body replaces the default slice form, which runs the item body through [`each!`]. Its arguments are the object, the bracketed `Arg`s, the number of results, and the slots it writes them into. Attributes before the object, such as `#[inline]`, go on its scalar forms, `one` and `judged`, so that a caller in another crate can inline a small form into its own loop over items.
///
/// A function object is a concrete type, never a generic one, so that its forms compile here. Settings that choose a loop, such as whether a tolerance is 0, are read once per call inside `many`, which then runs `each!`.
///
/// ```text
/// forms!(Choose at real: |_, [x, y, m]| Total(if m != 0.0 { x } else { y }));
/// forms!(Choose at bool: |_, [x, y, m]| (m & x) | (!m & y));
/// forms!(Gcd at i64: |_, [x, y]| gcd(x, y));
/// forms!(Less at real => bool: |k, [x, y]| k.0.less(x, y),
///     many |k, [x, y], n, out| with_tolerance!(k.0, |t| each!([x, y], n, out, |[x, y]| t.less(x, y))));
/// ```
macro_rules! forms {
    ($(#[$($m:tt)+])* $object:ident at real $(=> $o:ty)? : |$k:pat_param, $args:tt| $body:expr $(, many |$mk:pat_param, $margs:tt, $n:ident, $out:ident| $many:expr)?) => {
        forms!(@types [$(#[$($m)+])*], $object, $k, $args, $body, [$((|$mk, $margs, $n, $out| $many))?], [f32 $(=> $o)?, f64 $(=> $o)?]);
        forms!(@half [$(#[$($m)+])*], $object, forms!(@out half::f16 $(, $o)?), forms!(@narrow $($o)?), $args);
    };
    ($(#[$($m:tt)+])* $object:ident at $($e:ty $(=> $o:ty)?),+ : |$k:pat_param, $args:tt| $body:expr $(, many |$mk:pat_param, $margs:tt, $n:ident, $out:ident| $many:expr)?) => {
        forms!(@types [$(#[$($m)+])*], $object, $k, $args, $body, [$((|$mk, $margs, $n, $out| $many))?], [$($e $(=> $o)?),+]);
    };
    // A generic type, such as `Indicator<Less>`, goes in brackets, because a type can't be followed by `at`.
    ($(#[$($m:tt)+])* [$object:ty] at $($e:ty $(=> $o:ty)?),+ : |$k:pat_param, $args:tt| $body:expr $(, many |$mk:pat_param, $margs:tt, $n:ident, $out:ident| $many:expr)?) => {
        forms!(@types [$(#[$($m)+])*], $object, $k, $args, $body, [$((|$mk, $margs, $n, $out| $many))?], [$($e $(=> $o)?),+]);
    };
    // One type at a time, so that the optional `many` group, which repeats on its own, needn't sit inside the types' repetition.
    (@types $attrs:tt, $object:ty, $k:pat_param, $args:tt, $body:expr, $many:tt, [$e:ty $(=> $o:ty)? $(, $($rest:tt)+)?]) => {
        forms!(@concrete $attrs, $object, $e, forms!(@out $e $(, $o)?), $k, $args, $body, $many);
        $(forms!(@types $attrs, $object, $k, $args, $body, $many, [$($rest)+]);)?
    };
    (@concrete $attrs:tt, $object:ty, $e:ty, $o:ty, $k:pat_param, [$x:ident], $body:expr, $many:tt) => {
        forms!(@impl $attrs, $object, Unary, 1, $e, $o, $k, [$x], $body, $many);
    };
    (@concrete $attrs:tt, $object:ty, $e:ty, $o:ty, $k:pat_param, [$x:ident, $y:ident], $body:expr, $many:tt) => {
        forms!(@impl $attrs, $object, Binary, 2, $e, $o, $k, [$x, $y], $body, $many);
    };
    (@concrete $attrs:tt, $object:ty, $e:ty, $o:ty, $k:pat_param, [$x:ident, $y:ident, $z:ident], $body:expr, $many:tt) => {
        forms!(@impl $attrs, $object, Ternary, 3, $e, $o, $k, [$x, $y, $z], $body, $many);
    };
    (@impl [$(#[$($m:tt)+])*], $object:ty, $arity:ident, $arity_n:literal, $e:ty, $o:ty, $k:pat_param, [$($x:ident),+], $body:expr, $many:tt) => {
        impl crate::slice::Results<$e, $arity_n> for $object {
            type Out = $o;
            const TOTAL: bool = crate::slice::total::<Self, $e, $o, _, $arity_n>(|$k, [$($x),+]| $body);
        }
        unsafe impl crate::slice::$arity<$e> for $object {
            $(#[$($m)+])*
            fn one(&self, $($x: $e),+) -> $o {
                let $k = self;
                crate::slice::Outcome::<$o>::value($body)
            }
            $(#[$($m)+])*
            fn judged(&self, $($x: $e),+) -> ($o, bool) {
                let $k = self;
                crate::slice::Outcome::<$o>::judged($body, [$($x),+])
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
    (@half $attrs:tt, $object:ty, $o:ty, $narrow:expr, [$x:ident]) => { forms!(@half_impl $attrs, $object, Unary, 1, $o, $narrow, [$x]); };
    (@half $attrs:tt, $object:ty, $o:ty, $narrow:expr, [$x:ident, $y:ident]) => { forms!(@half_impl $attrs, $object, Binary, 2, $o, $narrow, [$x, $y]); };
    (@half $attrs:tt, $object:ty, $o:ty, $narrow:expr, [$x:ident, $y:ident, $z:ident]) => { forms!(@half_impl $attrs, $object, Ternary, 3, $o, $narrow, [$x, $y, $z]); };
    (@half_impl [$(#[$($m:tt)+])*], $object:ty, $arity:ident, $arity_n:literal, $o:ty, $narrow:expr, [$($x:ident),+]) => {
        impl crate::slice::Results<half::f16, $arity_n> for $object {
            type Out = $o;
            const TOTAL: bool = <Self as crate::slice::Results<f32, $arity_n>>::TOTAL;
        }
        unsafe impl crate::slice::$arity<half::f16> for $object {
            $(#[$($m)+])*
            fn one(&self, $($x: half::f16),+) -> $o { ($narrow)(crate::slice::$arity::<f32>::one(self, $($x.to_f32()),+)) }
            $(#[$($m)+])*
            fn judged(&self, $($x: half::f16),+) -> ($o, bool) {
                let (r, ok) = crate::slice::$arity::<f32>::judged(self, $($x.to_f32()),+);
                (($narrow)(r), ok)
            }
            fn slice(&self, x: [crate::Arg<'_, half::f16>; $arity_n], n: usize, out: &mut [std::mem::MaybeUninit<$o>]) -> bool {
                crate::slice::working(x, n, out, |x, m, o| crate::slice::$arity::<f32>::slice(self, x, m, o), $narrow)
            }
        }
    };
    (@out $e:ty) => { $e };
    (@out $e:ty, $o:ty) => { $o };
    (@narrow) => { half::f16::from_f32 };
    (@narrow $o:ty) => { std::convert::identity };
    (@n $($x:ident),+) => { [$(stringify!($x)),+].len() };
}

/// Runs `$body` for each item `$i` of a result of `$n` items, with `$a` and `$b` the items of `$x` and `$y` that pair with it. Each pairing of `One` and `Many` has a loop of its own. The loops index slices of the result's length with no iterator adapters, and vectorize at any build settings.
macro_rules! each_pair {
    ($x:expr, $y:expr, $n:expr, |$i:ident, $a:ident, $b:ident| $body:expr) => {{
        let n = $n;
        match ($x, $y) {
            (Arg::One($a), Arg::One($b)) => {
                for $i in 0..n {
                    $body
                }
            }
            (Arg::One($a), Arg::Many(ys)) => {
                let ys = &ys[..n];
                for $i in 0..n {
                    let $b = ys[$i];
                    $body
                }
            }
            (Arg::Many(xs), Arg::One($b)) => {
                let xs = &xs[..n];
                for $i in 0..n {
                    let $a = xs[$i];
                    $body
                }
            }
            (Arg::Many(xs), Arg::Many(ys)) => {
                let (xs, ys) = (&xs[..n], &ys[..n]);
                for $i in 0..n {
                    let ($a, $b) = (xs[$i], ys[$i]);
                    $body
                }
            }
        }
    }};
}

/// Runs `$body` for each item `$i` of a result of `$n` items, with `$a`, `$b` and `$c` the items of `$x`, `$y` and `$z` that go with it, as [`each_pair!`] does for two arguments. Each pairing of `One` and `Many` has a loop of its own.
macro_rules! each_triple {
    ($x:expr, $y:expr, $z:expr, $n:expr, |$i:ident, $a:ident, $b:ident, $c:ident| $body:expr) => {{
        let n = $n;
        match $x {
            Arg::One($a) => each_pair!($y, $z, n, |$i, $b, $c| $body),
            Arg::Many(firsts) => {
                let firsts = &firsts[..n];
                each_pair!($y, $z, n, |$i, $b, $c| {
                    let $a = firsts[$i];
                    $body
                })
            }
        }
    }};
}

/// Appends `f` on each of `n` pairs of items to `out`, and gives whether `f` said `true` for every pair. An `Arg::One` pairs with every item of the other argument. The loops write straight into `out`'s spare capacity, which is never zeroed first. This is the loop that rustymath's slice forms run, and a caller's own kernels can run it too, as basedpl's arithmetic does. A long call splits across threads, as [`parallel`](crate::parallel) describes.
#[inline(never)]
pub fn pairs<X: Copy + Sync, Y: Copy + Sync, B: Send>(x: Arg<'_, X>, y: Arg<'_, Y>, n: usize, out: &mut Vec<B>, f: impl Fn(X, Y) -> (B, bool) + Sync) -> bool {
    appended(out, n, |slots| {
        crate::parallel::split(slots, CHUNK, &|start, slots| {
            pairs_into(x.part(start, slots.len()), y.part(start, slots.len()), slots.len(), slots, |a, b| f(a, b))
        })
    })
}

/// [`pairs`], writing into the `n` slots of `out` in place of appending to a vector. It runs [`triples_into`]'s loops with a unit first argument, which go straight to [`each_pair!`]'s.
#[inline]
pub(crate) fn pairs_into<X: Copy, Y: Copy, B>(x: Arg<'_, X>, y: Arg<'_, Y>, n: usize, out: &mut [MaybeUninit<B>], f: impl Fn(X, Y) -> (B, bool)) -> bool {
    triples_into(Arg::One(()), x, y, n, out, #[inline(always)] |(), a, b| f(a, b))
}

/// [`pairs_into`] for three arguments, which `mapped` runs for a function object of three arguments.
#[inline]
pub(crate) fn triples_into<X: Copy, Y: Copy, Z: Copy, B>(
    x: Arg<'_, X>, y: Arg<'_, Y>, z: Arg<'_, Z>, n: usize, out: &mut [MaybeUninit<B>], f: impl Fn(X, Y, Z) -> (B, bool)) -> bool {
    let (out, mut ok) = (&mut out[..n], true);
    each_triple!(x, y, z, n, |i, a, b, c| {
        let (r, good) = f(a, b, c);
        out[i].write(r);
        ok &= good;
    });
    ok
}

/// Whether `valid` holds for every result in `out` with the items of `x` that gave it, paired as [`mapped`] pairs them. These loops only judge: a loop that also computes each result stops vectorizing for some bodies, such as `Ln`'s. Failures gather as bits, with no branch.
#[inline(always)]
pub(crate) fn judged<E: Copy, O: Copy, const N: usize>(x: &[Arg<'_, E>; N], out: &[O], valid: impl Fn([E; N], O) -> bool) -> bool {
    let (n, mut bad) = (out.len(), 0u8);
    if const { N == 1 } { each_pair!(x[0], Arg::Many(out), n, |_i, a, r| bad |= u8::from(!valid([a; N], r))); }
    else if const { N == 2 } {
        each_triple!(x[0], x[1], Arg::Many(out), n, |_i, a, b, r| bad |= u8::from(!valid(std::array::from_fn(|k| if k == 0 { a } else { b }), r)));
    } else { for i in 0..n { bad |= u8::from(!valid(items(x, i), out[i])); } }
    bad == 0
}

/// Appends `n` items to `out`, which `write` writes into its spare capacity, never zeroed first, and gives `write`'s result. `write` must write every slot it gets. Each caller's loop does, by its construction.
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

/// Writes `f` for each of `n` results into `out`, for arguments `x`, for a function `f` with a fast form that holds for all but rare items. `fast` computes every item with no branch, and gives `None` for the rare items. The items run in two passes. The first writes NaN for each rare item and judges each result alone, as [`each`] does. The second runs only if a result is NaN. It recomputes each NaN item with `f` and judges it by its arguments too. A rare item costs one more scan of at most one `CHUNK`, the most that `parallel::in_chunks` hands a slice form. Slice forms call it through [`two_pass!`].
#[inline(always)]
pub(crate) fn two_pass<R, O, const N: usize>(
    x: [Arg<'_, R>; N], n: usize, out: &mut [MaybeUninit<O>], fast: impl Fn([R; N]) -> Option<R>, f: impl Fn([R; N]) -> R) -> bool where
    R: crate::Float + crate::buffer::Source<O>, O: crate::Float {
    let out = &mut out[..n];
    let first = mapped(&x, n, out, #[inline(always)] |a| {
        let r = <R as crate::buffer::Source<O>>::read(fast(a).unwrap_or(R::nan()));
        (r, !r.is_nan())
    });
    if first { return true; }
    let mut ok = true;
    for (i, r) in written(out).iter_mut().enumerate().filter(|(_, r)| r.is_nan()) {
        let a = items(&x, i);
        *r = <R as crate::buffer::Source<O>>::read(f(a));
        ok &= kept(*r, a);
    }
    ok
}

/// `f`, a slice form at `F`'s working type, for narrower floats, as `forms!`'s `f16` forms run it: the items convert to the working type and run, and `narrow` turns each output into the output at `F`.
pub(crate) fn working<F: Float, O: Copy, P, const N: usize>(
    x: [Arg<'_, F>; N], n: usize, out: &mut [MaybeUninit<P>], f: impl Fn([Arg<'_, F::Working>; N], usize, &mut [MaybeUninit<O>]) -> bool,
    narrow: impl Fn(O) -> P) -> bool {
    let mut bufs = [(); N].map(|()| Vec::new());
    let mut next = bufs.each_mut().into_iter();
    buffered(n, out, |o| f(x.map(|a| a.converted(0, n, next.next().unwrap(), F::working)), n, o), narrow)
}

/// `f`, a slice form writing `n` outputs of type `O`, run into a buffer, with `each` turning each output into the `P` that `out` takes. It gives `f`'s result.
pub(crate) fn buffered<O: Copy, P>(n: usize, out: &mut [MaybeUninit<P>], f: impl FnOnce(&mut [MaybeUninit<O>]) -> bool, each: impl Fn(O) -> P) -> bool {
    let mut rs = Vec::new();
    let ok = appended(&mut rs, n, f);
    pairs_into(Arg::Many(&rs), Arg::One(()), n, &mut out[..n], |r, ()| (each(r), true));
    ok
}
