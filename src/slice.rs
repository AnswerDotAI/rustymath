//! Slice forms of rustymath's functions. Each function object, such as [`Pow`], has a scalar form `one` and a slice form `many`.
//! Both run code that rustymath compiles once for each item type, inside its own crate. A caller's build settings can't change that
//! code, and rustymath's speed test times what callers run. An edit to a function's body rebuilds only rustymath.
use crate::float::{
    sealed::{Forms, Item},
    Float,
};
use num_complex::Complex;
use half::f16;

/// How many items a caller handles at a time where it converts or expands arguments into buffers for a slice form. Buffers of this
/// many items stay in cache.
pub const CHUNK: usize = 4096;

/// One argument of a slice function: one item that every result pairs with, or one item for each result.
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a, T> {
    One(T),
    Many(&'a [T]),
}

impl<T: Copy> Arg<'_, T> {
    /// Items `start..start + n` of this argument, each converted by `f`. A `Many` argument converts into `buf`.
    #[inline]
    pub fn converted<'b, A>(self, start: usize, n: usize, buf: &'b mut Vec<A>, f: impl Fn(T) -> A) -> Arg<'b, A> {
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
        arities!(@concrete $name, $n, ($($x),+): f16, f32, f64, Complex<f32>, Complex<f64>, i8, i16, i32, i64, u8, u16, u32, u64);
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
            fn many(&self, $($x: Arg<'_, $e>,)+ n: usize, out: &mut Vec<Self::Out>) -> bool { self.slice([$($x),+], n, out) }
            #[inline]
            fn checked(&self, $($x: $e),+) -> Option<Self::Out> { self.checked_scalar([$($x),+]) }
        }
    };
}

arities! {
    /// A function of one item as a value, with a scalar form and a slice form. rustymath's function objects, such as [`Exp`](crate::Exp),
    /// implement it at each item type they take: every [`Float`] type for a float function, and complex numbers or integers for others.
    /// `Out` is the type of each result, which is the item type unless the function gives another, as [`Abs`](crate::Abs) gives the
    /// real magnitude of a complex number. Its methods take `&self`, so a function object can hold settings.
    Unary<1>(x);
    /// A function of two items as a value, as [`Unary`] is for one.
    Binary<2>(x, y);
    /// A function of three items as a value, as [`Unary`] is for one.
    Ternary<3>(x, y, z);
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

/// The output of an item body that never gives NaN from arguments that aren't NaN, such as `exp`'s. Such an output never fails. The
/// slice loop tests none of them.
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

/// Appends `f` for each of `n` results to `out`, for arguments `x`, and gives whether `f` said `true` for every result. One and two
/// arguments run through [`pairs`], whose loops vectorize.
#[inline(always)]
fn mapped<E: Copy, O, const N: usize>(x: &[Arg<'_, E>; N], n: usize, out: &mut Vec<O>, f: impl Fn([E; N]) -> (O, bool)) -> bool {
    match x.as_slice() {
        &[a] => pairs(a, Arg::One(()), n, out, #[inline(always)] |a, ()| f([a; N])),
        &[a, b] => pairs(a, b, n, out, #[inline(always)] |a, b| f(std::array::from_fn(|k| if k == 0 { a } else { b }))),
        _ => {
            let mut ok = true;
            out.extend((0..n).map(|i| {
                let (r, good) = f(items(x, i));
                ok &= good;
                r
            }));
            ok
        }
    }
}

/// Appends `f` for each of `n` results to `out`, for arguments `x`, and gives whether every result succeeded, as [`Outcome`] says.
/// Slice forms call it through [`each!`]. The loop judges each output alone, and a [`Total`] output not at all. Only where some output
/// is NaN does a second pass look at the arguments of the NaN outputs, which is rare.
#[inline(always)]
pub(crate) fn each<E: Item, O: Item, R: Outcome<O>, const N: usize>(
    x: [Arg<'_, E>; N], n: usize, out: &mut Vec<O>, f: impl Fn([E; N]) -> R,
) -> bool {
    let start = out.len();
    mapped(&x, n, out, #[inline(always)] |a| f(a).split())
        || (R::RESCUE && out[start..].iter().enumerate().all(|(i, &r)| kept(r, items(&x, i))))
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
/// the object, the bracketed `Arg`s, the number of results, and the vector they append to.
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
        fn slice(&self, args: [crate::Arg<'_, $e>; forms!(@n $($x),+)], n: usize, out: &mut Vec<$o>) -> bool {
            let $k = self;
            each!(args, n, out, |[$($x),+]| $body)
        }
    };
    (@slice $e:ty => $o:ty, $k:pat_param, $args:tt, $body:expr, [(|$mk:pat_param, [$($mx:ident),+], $n:ident, $out:ident| $many:expr)]) => {
        fn slice(&self, [$($mx),+]: [crate::Arg<'_, $e>; forms!(@n $($mx),+)], $n: usize, $out: &mut Vec<$o>) -> bool {
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
            fn slice(&self, x: [crate::Arg<'_, half::f16>; forms!(@n $($x),+)], n: usize, out: &mut Vec<$o>) -> bool {
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
/// loop that rustymath's slice forms run, and a caller's own kernels can run it too, as basedpl's arithmetic does.
#[inline]
pub fn pairs<X: Copy, Y: Copy, B>(x: Arg<'_, X>, y: Arg<'_, Y>, n: usize, out: &mut Vec<B>, f: impl Fn(X, Y) -> (B, bool)) -> bool {
    appended(out, n, |slots| {
        let mut ok = true;
        each_pair!(x, y, n, |i, a, b| {
            let (r, good) = f(a, b);
            slots[i].write(r);
            ok &= good;
        });
        ok
    })
}

/// Appends `n` items to `out`, which `write` writes into its spare capacity, never zeroed first, and gives `write`'s result. `write`
/// must write every slot it gets. Each caller's loop does, by its construction.
#[inline(always)]
pub(crate) fn appended<B, T>(out: &mut Vec<B>, n: usize, write: impl FnOnce(&mut [std::mem::MaybeUninit<B>]) -> T) -> T {
    out.reserve(n);
    let result = write(&mut out.spare_capacity_mut()[..n]);
    // SAFETY: `write` wrote all `n` slots.
    unsafe { out.set_len(out.len() + n) }
    result
}

/// Appends `f` for each of `n` results, for arguments `x`, for a function `f` with a fast form that holds for all but rare items.
/// `fast` computes every item with no branch, and gives `None` for the rare items. Each `CHUNK` of items runs in two passes. The first
/// writes NaN for each rare item and judges each result alone, as [`each`] does. The second runs only if the chunk holds a NaN. It
/// recomputes each NaN item with `f` and judges it by its arguments too. A rare item costs a scan of its own chunk only. Slice forms
/// call it through [`two_pass!`].
#[inline(always)]
pub(crate) fn two_pass<R: crate::Float, const N: usize>(
    x: [Arg<'_, R>; N], n: usize, out: &mut Vec<R>, fast: impl Fn([R; N]) -> Option<R>, f: impl Fn([R; N]) -> R,
) -> bool {
    let mut ok = true;
    for start in (0..n).step_by(CHUNK) {
        let (m, from) = (CHUNK.min(n - start), out.len());
        let x = x.map(|a| a.part(start, m));
        let first = mapped(&x, m, out, #[inline(always)] |a| {
            let r = fast(a).unwrap_or(R::nan());
            (r, !r.is_nan())
        });
        if first { continue; }
        for (i, r) in out[from..].iter_mut().enumerate().filter(|(_, r)| r.is_nan()) {
            let a = items(&x, i);
            *r = f(a);
            ok &= kept(*r, a);
        }
    }
    ok
}

/// [`two_pass`] with both item bodies marked `#[inline(always)]`, as [`each!`] marks its body.
macro_rules! two_pass {
    ($x:expr, $n:expr, $out:expr, |$fa:pat_param| $fast:expr, |$sa:pat_param| $f:expr $(,)?) => {
        crate::slice::two_pass($x, $n, $out, #[inline(always)] |$fa| $fast, #[inline(always)] |$sa| $f)
    };
}

/// Appends `f` for each of `n` results, for complex arguments `x`, through real slice forms. Each `CHUNK` of items splits into a slice
/// of real parts and a slice of imaginary parts for each argument, with a `One` argument repeated. `fast` computes the results' real
/// and imaginary parts from those slices, through real slice forms, and appends them. Each result that `rare` picks, from its
/// arguments and its fast value, then takes `f`, the scalar function, which handles what the fast formula doesn't, such as tiny or
/// infinite parts. The results succeed as [`each`] judges them. Slice forms call it through [`split!`].
pub(crate) fn split<R: crate::Native, const N: usize>(
    x: [Arg<'_, Complex<R>>; N], n: usize, out: &mut Vec<Complex<R>>, fast: impl Fn([(&[R], &[R]); N], &mut Vec<R>, &mut Vec<R>),
    rare: impl Fn([Complex<R>; N], Complex<R>) -> bool, f: impl Fn([Complex<R>; N]) -> Complex<R>,
) -> bool
where
    Complex<R>: Item,
{
    let (start, mut parts, mut re, mut im) = (out.len(), [(); N].map(|()| (Vec::new(), Vec::new())), Vec::new(), Vec::new());
    for from in (0..n).step_by(CHUNK) {
        let (m, base) = (CHUNK.min(n - from), out.len());
        let x = x.map(|a| a.part(from, m));
        for ((xs, ys), a) in parts.iter_mut().zip(x) {
            xs.clear();
            ys.clear();
            match a {
                Arg::One(z) => {
                    xs.resize(m, z.re);
                    ys.resize(m, z.im);
                }
                Arg::Many(s) => {
                    xs.extend(s.iter().map(|z| z.re));
                    ys.extend(s.iter().map(|z| z.im));
                }
            }
        }
        re.clear();
        im.clear();
        fast(std::array::from_fn(|k| (&parts[k].0[..], &parts[k].1[..])), &mut re, &mut im);
        out.extend(re.iter().zip(&im).map(|(&a, &b)| Complex::new(a, b)));
        if (0..m).fold(false, |any, i| any | rare(items(&x, i), out[base + i])) {
            for i in 0..m {
                let a = items(&x, i);
                if rare(a, out[base + i]) { out[base + i] = f(a); }
            }
        }
    }
    out[start..].iter().all(|r| !r.has_nan()) || out[start..].iter().enumerate().all(|(i, &r)| kept(r, items(&x, i)))
}

/// [`split`] with `rare` marked `#[inline(always)]`, as [`each!`] marks its body, because the scan for rare items runs on every item.
macro_rules! split {
    ($x:expr, $n:expr, $out:expr, |$pa:pat_param, $re:pat_param, $im:pat_param| $fast:expr, |$ra:pat_param, $rr:pat_param| $rare:expr, |$fa:pat_param| $f:expr $(,)?) => {
        crate::slice::split($x, $n, $out, |$pa, $re, $im| $fast, #[inline(always)] |$ra, $rr| $rare, |$fa| $f)
    };
}

/// `f`, a slice form at `F`'s working type, for narrower floats, as `forms!`'s `f16` forms run it: each `CHUNK` items convert to the
/// working type and run, and `narrow` turns each output into the output at `F`.
pub(crate) fn working<F: Float, O: Copy, P, const N: usize>(
    x: [Arg<'_, F>; N], n: usize, out: &mut Vec<P>, f: impl Fn([Arg<'_, F::Working>; N], usize, &mut Vec<O>) -> bool,
    narrow: impl Fn(O) -> P,
) -> bool {
    let (mut bufs, mut rs) = ([(); N].map(|()| Vec::new()), Vec::new());
    let mut ok = true;
    for start in (0..n).step_by(CHUNK) {
        let m = CHUNK.min(n - start);
        let mut bufs = bufs.each_mut().into_iter();
        rs.clear();
        ok &= f(x.map(|a| a.converted(start, m, bufs.next().unwrap(), F::working)), m, &mut rs);
        pairs(Arg::Many(&rs), Arg::One(()), m, out, |r, ()| (narrow(r), true));
    }
    ok
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{reduce::{Fmax, Fmin}, *};
    use std::fmt::Debug;

    /// Whether `a` and `b` agree: both NaN, the same value with the same sign, or finite and nonzero within 8 units in the last place
    /// of each other. A scalar form and its slice form can round differently, as `Pow`'s do at f32, where only the scalar form reads
    /// tables. Zeros, infinities and signs must match exactly.
    fn same<F: Float>(a: F, b: F) -> bool {
        if a.is_nan() || b.is_nan() { return a.is_nan() && b.is_nan(); }
        if a == b { return a.is_sign_negative() == b.is_sign_negative(); }
        let near = (a - b).abs() <= F::narrow(8.0) * F::epsilon() * a.abs().max(b.abs()).max(F::min_positive_value());
        a.is_finite() && b.is_finite() && a != F::zero() && b != F::zero() && near
    }

    /// Items with every special value, both signs, a subnormal, and magnitudes beyond the trig functions' fast reduction.
    fn items<F: Float>() -> Vec<F> {
        let wide = [0.0, 1.0, 0.5, 2.0, 3.0, 0.1, 7.25, 1e-310, 1e-30, 30.0, 700.0, 2e6, 1e10, 3e8, f64::INFINITY, f64::NAN];
        wide.iter().flat_map(|&x| [x, -x]).map(F::narrow).collect()
    }

    /// `k.many` against `k.one` for every pairing of `One` and `Many` among `items`. It must append after what `out` holds, and say
    /// whether every item succeeded, as `checked` does item by item. The NaN rule must agree too. That catches a function marked total
    /// that gives NaN from arguments that aren't NaN.
    fn binary<F: Float + Debug, K: Binary<F, Out = F>>(k: K, name: &str) {
        let items = items::<F>();
        let check = |x: Arg<'_, F>, y: Arg<'_, F>, n: usize| {
            let at = |a: Arg<'_, F>, i| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
            let mut out = vec![F::one()];
            let ok = k.many(x, y, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()), "{name}");
            for i in 0..n { assert!(same(out[i + 1], k.one(at(x, i), at(y, i))), "{name}({:?}, {:?})", at(x, i), at(y, i)) }
            assert_eq!(ok, (0..n).all(|i| k.checked(at(x, i), at(y, i)).is_some()), "{name}");
            assert_eq!(ok, (0..n).all(|i| kept(out[i + 1], [at(x, i), at(y, i)])), "{name} breaks the NaN rule");
        };
        let (xs, ys): (Vec<F>, Vec<F>) = items.iter().flat_map(|&x| items.iter().map(move |&y| (x, y))).unzip();
        check(Arg::Many(&xs), Arg::Many(&ys), xs.len());
        for &v in &items {
            check(Arg::One(v), Arg::Many(&items), items.len());
            check(Arg::Many(&items), Arg::One(v), items.len());
        }
    }

    /// `k.many` against `k.one` for each of `items`, and for each item as `Arg::One`, as for `binary`.
    fn unary<F: Float + Debug, K: Unary<F, Out = F>>(k: K, name: &str) {
        let items = items::<F>();
        let check = |x: Arg<'_, F>, n: usize| {
            let at = |i| match x { Arg::One(v) => v, Arg::Many(s) => s[i] };
            let mut out = vec![F::one()];
            let ok = k.many(x, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()), "{name}");
            for i in 0..n { assert!(same(out[i + 1], k.one(at(i))), "{name}({:?})", at(i)) }
            assert_eq!(ok, (0..n).all(|i| k.checked(at(i)).is_some()), "{name}");
            assert_eq!(ok, (0..n).all(|i| kept(out[i + 1], [at(i)])), "{name} breaks the NaN rule");
        };
        check(Arg::Many(&items), items.len());
        for &v in &items { check(Arg::One(v), 3) }
    }

    /// `k.many` against `k.one` for every triple of `items`, as `Many` arguments and with each argument in turn as `One`, as for
    /// `binary`.
    fn ternary<F: Float + Debug, K: Ternary<F, Out = F>>(k: K, name: &str) {
        let items = items::<F>();
        let at = |a: Arg<'_, F>, i| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
        let check = |[x, y, z]: [Arg<'_, F>; 3], n: usize| {
            let mut out = vec![F::one()];
            let ok = k.many(x, y, z, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()), "{name}");
            for i in 0..n { assert!(same(out[i + 1], k.one(at(x, i), at(y, i), at(z, i))), "{name}({:?}, {:?}, {:?})", at(x, i), at(y, i), at(z, i)) }
            assert_eq!(ok, (0..n).all(|i| k.checked(at(x, i), at(y, i), at(z, i)).is_some()), "{name}");
            assert_eq!(ok, (0..n).all(|i| kept(out[i + 1], [at(x, i), at(y, i), at(z, i)])), "{name} breaks the NaN rule");
        };
        let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
        for &x in &items { for &y in &items { for &z in &items { xs.push(x); ys.push(y); zs.push(z); } } }
        check([Arg::Many(&xs), Arg::Many(&ys), Arg::Many(&zs)], xs.len());
        for &v in &items {
            check([Arg::One(v), Arg::Many(&items), Arg::Many(&items)], items.len());
            check([Arg::Many(&items), Arg::One(v), Arg::Many(&items)], items.len());
            check([Arg::Many(&items), Arg::Many(&items), Arg::One(v)], items.len());
        }
    }

    #[test]
    fn slice_forms_match_scalar_forms() {
        macro_rules! each_width {
            (unary: $($u:ident),+; binary: $($b:ident),+; ternary: $($t:ident),+) => {
                $(unary::<f64, _>($u, stringify!($u)); unary::<f32, _>($u, stringify!($u)); unary::<f16, _>($u, stringify!($u));)+
                $(binary::<f64, _>($b, stringify!($b)); binary::<f32, _>($b, stringify!($b)); binary::<f16, _>($b, stringify!($b));)+
                $(ternary::<f64, _>($t, stringify!($t)); ternary::<f32, _>($t, stringify!($t)); ternary::<f16, _>($t, stringify!($t));)+
            };
        }
        each_width!(
            unary: Exp, Expm1, Ln, Log1p, Log2, Log10, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Cbrt, Gamma,
                Lgamma, Erf, Erfc, Erfinv, Erfcinv, LambertW, Sqrt, Sqrt1mx2, Sqrt1px2, Sqrtx2m1, Abs, Sign, Angle;
            binary: Pow, Root, Hypot, Atan2, Gammainc, Gammaincc, Gammaincinv, Binom, Fmax, Fmin;
            ternary: Betainc, Betaincinv
        );
    }

    #[derive(Clone, Copy)]
    struct Positive;
    forms!(Positive at real => bool: |_, [x]| x > 0.0);
    #[derive(Clone, Copy)]
    struct Norm;
    forms!(Norm at Complex<f32> => f32, Complex<f64> => f64: |_, [z]| z.norm());

    /// An output type other than the item type, at each float type, f16 included, and at concrete complex types. A `bool` output is
    /// never NaN, so it never fails.
    #[test]
    fn outputs_can_differ_from_items() {
        fn positive<F: Float>(x: &[F]) -> Vec<bool> where Positive: Unary<F, Out = bool> {
            let mut out = Vec::new();
            assert!(Positive.many(Arg::Many(x), x.len(), &mut out));
            out
        }
        assert_eq!(positive(&[1.0f64, -1.0, f64::NAN]), [true, false, false]);
        assert_eq!(positive(&[f16::from_f32(2.0), f16::NAN]), [true, false]);
        assert_eq!(Positive.checked(f32::NAN), Some(false));
        let mut out = Vec::new();
        assert!(Norm.many(Arg::Many(&[Complex::new(3.0f32, 4.0), Complex::new(f32::INFINITY, f32::NAN)]), 2, &mut out));
        assert_eq!(out, [5.0, f32::INFINITY]);
    }
}
