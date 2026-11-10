//! Shared slice loops, output storage and native bulk arguments.
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
    pub fn part(self, start: usize, n: usize) -> Self { match self { Arg::One(v) => Arg::One(v), Arg::Many(s) => Arg::Many(&s[start..start + n]) } }
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

/// A binary formula with its arguments in the other order.
#[derive(Clone, Copy, Debug)]
pub struct Flipped<K>(pub K);

/// An integer reduction or product whose results wrap to its output type.
#[derive(Clone, Copy, Debug)]
pub struct Wrapped<K>(pub K);

/// Whether `r`, the result for arguments `args`, keeps the rule that a NaN result needs a NaN argument.
#[inline(always)]
pub(crate) fn kept<O: KernelItem, E: KernelItem, const N: usize>(r: O, args: [E; N]) -> bool { !r.has_nan() | args.iter().any(|a| a.has_nan()) }

/// What a function object's item body gives: an output, which fails where it is NaN and no argument is, an `Option` of one, which fails where it is `None`, a wrapped output paired with whether it's exact, which fails where it isn't, or a [`Total`] output, which never fails. An integer power, whose result can overflow, gives a wrapped output. An integer function with no wrapped result to give, such as a binomial coefficient, gives an `Option`. An output that is never NaN, such as `bool`, fails only through `None` or an inexact pair.
pub(crate) trait Outcome<O: KernelItem>: Sized {
    /// Whether an output that `split` fails can still succeed, as a NaN output does where an argument is NaN. Only a plain output can.
    const RESCUE: bool;
    /// The output, and whether it succeeded, judged by the output alone, so that a NaN output fails. That costs one test for each item, where testing each argument too would cost one more for each argument.
    fn split(self) -> (O, bool);
}

impl<O: KernelItem> Outcome<O> for O {
    const RESCUE: bool = true;
    #[inline(always)]
    fn split(self) -> (O, bool) { (self, !self.has_nan()) }
}

impl<O: KernelItem + Default> Outcome<O> for Option<O> {
    const RESCUE: bool = false;
    #[inline(always)]
    fn split(self) -> (O, bool) { (self.unwrap_or_default(), self.is_some()) }
}

/// An output wrapped to its type, and whether it's the exact result, as an integer power gives it. A result that isn't exact fails, and its slot holds the wrapped output. That output is NumPy's result for fixed-width integers.
impl<O: KernelItem> Outcome<O> for (O, bool) {
    const RESCUE: bool = false;
    #[inline(always)]
    fn split(self) -> (O, bool) { self }
}

/// The output of an item body that never fails: one that never gives NaN from arguments that aren't NaN, such as `exp`'s, or one whose NaN is the function's value, such as that of `∞ - ∞`. The slice loop tests none of them.
#[derive(Clone, Copy)]
pub(crate) struct Total<O>(pub O);

impl<O: KernelItem> Outcome<O> for Total<O> {
    const RESCUE: bool = false;
    #[inline(always)]
    fn split(self) -> (O, bool) { (self.0, true) }
}


/// The items of `x` that result `i` takes.
#[inline(always)]
fn items<E: Copy, const N: usize>(x: &[Arg<'_, E>; N], i: usize) -> [E; N] { std::array::from_fn(|k| x[k].nth(i)) }

/// Writes `f` for each of `n` results into `out`, for arguments `x`, and gives whether `f` said `true` for every result. One and two arguments run through [`pairs_into`], and three through `triples_into`, whose loops vectorize.
#[inline(always)]
pub fn mapped<E: Copy, O, const N: usize>(x: &[Arg<'_, E>; N], n: usize, out: &mut [MaybeUninit<O>], f: impl Fn([E; N]) -> (O, bool)) -> bool {
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
    mapped(
        &x, n, out, #[inline(always)] |a| f(a).split()
    ) || (R::RESCUE && unsafe { written(&mut out[..n]) }.iter().enumerate().all(|(i, &r)| kept(r, items(&x, i))))
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


/// [`each`] with the item body marked `#[inline(always)]`, so native bulk loops inline their scalar body.
macro_rules! each {
    ($x:expr, $n:expr, $out:expr, |$args:pat_param| $body:expr $(,)?) => {
        crate::slice::each(
            $x, $n, $out, #[inline(always)] |$args| $body,
        )
    };
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
    unsafe {
        appended(out, n, |slots| {
            crate::parallel::split(slots, CHUNK, &|start, slots| {
                pairs_into(x.part(start, slots.len()), y.part(start, slots.len()), slots.len(), slots, |a, b| f(a, b))
            })
        })
    }
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
pub fn judged<E: Copy, O: Copy, const N: usize>(x: &[Arg<'_, E>; N], out: &[O], valid: impl Fn([E; N], O) -> bool) -> bool {
    let (n, mut bad) = (out.len(), 0u8);
    if const { N == 1 } { each_pair!(x[0], Arg::Many(out), n, |_i, a, r| bad |= u8::from(!valid([a; N], r))); }
    else if const { N == 2 } {
        each_triple!(x[0], x[1], Arg::Many(out), n, |_i, a, b, r| bad |= u8::from(!valid(std::array::from_fn(|k| if k == 0 { a } else { b }), r)));
    } else { for i in 0..n { bad |= u8::from(!valid(items(x, i), out[i])); } }
    bad == 0
}

/// Appends `n` items to `out`, which `write` writes into its spare capacity, never zeroed first, and gives `write`'s result. `write` must write every slot it gets. Each caller's loop does, by its construction.
///
/// # Safety
/// `write` must initialize every slot it receives before returning.
#[inline(always)]
pub unsafe fn appended<B, T>(out: &mut Vec<B>, n: usize, write: impl FnOnce(&mut [MaybeUninit<B>]) -> T) -> T {
    out.reserve(n);
    let result = write(&mut out.spare_capacity_mut()[..n]);
    // SAFETY: `write` wrote all `n` slots.
    unsafe { out.set_len(out.len() + n) }
    result
}

/// `out`'s slots as items. Each caller passes only slots that its loop has just written.
///
/// # Safety
/// Every slot in `out` must hold an initialized `B`.
#[inline(always)]
pub unsafe fn written<B>(out: &mut [MaybeUninit<B>]) -> &mut [B] {
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
    for (i, r) in unsafe { written(out) }.iter_mut().enumerate().filter(|(_, r)| r.is_nan()) {
        let a = items(&x, i);
        *r = <R as crate::buffer::Source<O>>::read(f(a));
        ok &= kept(*r, a);
    }
    ok
}


/// `f`, a slice form writing `n` outputs of type `O`, run into a buffer, with `each` turning each output into the `P` that `out` takes. It gives `f`'s result.
///
/// # Safety
/// `f` must initialize every output slot it receives before returning.
pub unsafe fn buffered<O: Copy, P>(n: usize, out: &mut [MaybeUninit<P>], f: impl FnOnce(&mut [MaybeUninit<O>]) -> bool, each: impl Fn(O) -> P) -> bool {
    let mut rs = Vec::new();
    let ok = unsafe { appended(&mut rs, n, f) };
    pairs_into(Arg::Many(&rs), Arg::One(()), n, &mut out[..n], |r, ()| (each(r), true));
    ok
}
