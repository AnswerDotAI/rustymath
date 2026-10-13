//! Inner products of row-major matrices. The inner product `f.g` of an `m`×`k` matrix `a` and a `k`×`n` matrix `b` is the `m`×`n`
//! matrix whose item at row `i` and column `j` folds `f` over `g(a[i][p], b[p][j])` for each `p`. `+.×` is the matrix product. The
//! `f` of each pair is associative, and the loops combine its steps in whatever order suits them. A float result can therefore differ
//! in its last bits from a fold taken in order.
//!
//! The loops follow GotoBLAS and BLIS. Packing copies a block of each argument into a contiguous buffer sized for a cache level. The
//! blocking reuses each buffer while it is in cache. A micro-kernel computes a small tile of the result in registers, one step of `p`
//! at a time. A micro-kernel in plain Rust vectorizes through LLVM's SLP vectorizer, which the builds of rustymath and basedpl turn on
//! at opt-level 2. It vectorizes only tiles of up to 32 items, too few to keep the multiply-add units busy. Plain Rust also
//! takes four operations for a maximum or minimum that gives NaN beside NaN, where AArch64's `fmax` and wasm's `max` take one. On
//! AArch64, in wasm and on x86-64 with AVX2 and FMA, the float pairs run micro-kernels on explicit vectors instead. AVX2's extremes
//! compare and select, because its own maximum and minimum skip NaN. Other targets run the plain kernels.
//!
//! The private module `factor` holds decompositions of real and complex matrices, whose blocked updates run on these loops. This module
//! re-exports its public items, and [`Scalar`], the trait of the item types they take. [`transpose_into`] transposes a matrix through 8×8
//! tiles.
use crate::{convert, float::{sealed::Core, FUSED}, Float, Native};
use num_complex::Complex;

mod factor;
mod scalar;
pub mod exact;
pub use factor::*;
pub use scalar::Scalar;

/// The `n`×`n` identity matrix.
pub(crate) fn identity<T: num_traits::Zero + num_traits::One>(n: usize) -> Vec<T> {
    (0..n * n).map(|k| if k % (n + 1) == 0 { T::one() } else { T::zero() }).collect()
}

/// Writes the transpose of the `rows`×`cols` matrix in `src`, whose rows start `ls` items apart, into `dst`, whose rows start `ld`
/// items apart. Each item passes through `f`: `dst[c * ld + r]` is `f(src[r * ls + c])`. Each 8×8 tile reads eight runs of 8
/// adjacent items and writes eight such runs.
#[inline]
pub fn transpose_into<S: Copy, D>(src: &[S], ls: usize, dst: &mut [D], ld: usize, [rows, cols]: [usize; 2], f: impl Fn(S) -> D) {
    let (whole_rows, whole_cols) = (rows / 8 * 8, cols / 8 * 8);
    for r0 in (0..whole_rows).step_by(8) {
        for c0 in (0..whole_cols).step_by(8) {
            let tile: [&[S; 8]; 8] = std::array::from_fn(|i| src[(r0 + i) * ls + c0..].first_chunk().expect("a whole tile lies in `src`"));
            for j in 0..8 {
                let row: &mut [D; 8] = dst[(c0 + j) * ld + r0..].first_chunk_mut().expect("a whole tile lies in `dst`");
                for i in 0..8 { row[i] = f(tile[i][j]) }
            }
        }
    }
    if whole_cols < cols {
        for r in 0..whole_rows { for c in whole_cols..cols { dst[c * ld + r] = f(src[r * ls + c]) } }
    }
    for r in whole_rows..rows {
        for c in 0..cols { dst[c * ld + r] = f(src[r * ls + c]) }
    }
}

/// An inner product `f.g` at item type `T`. Each pair object implements it at the item types listed here. Its loops compile in
/// rustymath, as slice forms do.
pub trait Inner<T> {
    /// The type of each result.
    type Out;
    /// The `m`×`n` inner product of `a`, an `m`×`k` matrix, and `b`, a `k`×`n` matrix, both row-major, as a row-major matrix. `size`
    /// is `[m, k, n]`. Where `k` is 0, each result is the identity of `f`. Integer arithmetic wraps, as NumPy's does.
    fn inner(&self, a: &[T], b: &[T], size: [usize; 3]) -> Vec<Self::Out>;
    /// `inner`, or `None` where an integer result could overflow, for callers that keep integers exact. A result it gives is the exact
    /// value. Floats never give `None`.
    fn checked(&self, a: &[T], b: &[T], size: [usize; 3]) -> Option<Vec<Self::Out>> { Some(self.inner(a, b, size)) }
}

/// `+.×`, the matrix product, at `i64`, `f32` and `f64`, and complex numbers with parts of either type. Each float step is a fused
/// multiply-add where the target has one. A complex product sums three real products of the parts, as the 3M method does. Its real
/// parts are as accurate as four real products make them. Its normwise error is about 4 units of roundoff, against about 2.5 for four
/// real products. An imaginary part's error scales with the size of the whole product. An imaginary part a millionth the size of its
/// real part keeps about 10 of f64's 16 digits and about 1 of f32's 7. A result whose 3M imaginary part isn't finite takes four real
/// products instead.
pub struct PlusTimes;
/// `⌈.+`, the max-plus product of longest-path problems, at `i64`, `f32` and `f64`. Its maximum is NaN beside NaN, as
/// [`reduce::maximum`](crate::reduce::maximum) is.
pub struct MaxPlus;
/// `⌊.+`, the min-plus product of shortest-path problems, at `i64`, `f32` and `f64`. Its minimum is NaN beside NaN.
pub struct MinPlus;
/// `∨.∧` on Booleans: whether some `p` has both `a[i][p]` and `b[p][j]`.
pub struct OrAnd;
/// `∧.∨` on Booleans: whether every `p` has `a[i][p]` or `b[p][j]`.
pub struct AndOr;
/// `+.=`, at `i16`, `i32`, `i64`, `f32` and `f64`: the number of `p` where `a[i][p]` equals `b[p][j]`. The count has the width of the
/// items, as `i16`, `i32` or `i64`, so an `i16` count needs `k` below 2^15 and an `i32` count below 2^31. Floats compare exactly, so NaN
/// equals nothing.
pub struct PlusEqual;
/// `+.≠`, at `i16`, `i32`, `i64`, `f32` and `f64`: the number of `p` where `a[i][p]` differs from `b[p][j]`, with `+.=`'s count type.
pub struct PlusNotEqual;

/// The scalar operations of a pair at item type `T`.
trait Pair<T> {
    type Out: Copy;
    /// The identity of `f`: the result of folding no steps.
    const UNIT: Self::Out;
    fn g(a: T, b: T) -> Self::Out;
    fn f(x: Self::Out, y: Self::Out) -> Self::Out;
    /// `f(acc, g(a, b))`, which `+.×` computes as one fused multiply-add.
    #[inline(always)]
    fn step(acc: Self::Out, a: T, b: T) -> Self::Out { Self::f(acc, Self::g(a, b)) }
}

macro_rules! floats {
    ($($t:ty),+) => {$(
        impl Pair<$t> for PlusTimes {
            type Out = $t;
            const UNIT: $t = 0.0;
            #[inline(always)]
            fn g(a: $t, b: $t) -> $t { a * b }
            #[inline(always)]
            fn f(x: $t, y: $t) -> $t { x + y }
            #[inline(always)]
            fn step(acc: $t, a: $t, b: $t) -> $t { a.mla::<FUSED>(b, acc) }
        }
        impl Pair<$t> for MaxPlus {
            type Out = $t;
            const UNIT: $t = <$t>::NEG_INFINITY;
            #[inline(always)]
            fn g(a: $t, b: $t) -> $t { a + b }
            #[inline(always)]
            fn f(x: $t, y: $t) -> $t { Core::maximum(x, y) }
        }
        impl Pair<$t> for MinPlus {
            type Out = $t;
            const UNIT: $t = <$t>::INFINITY;
            #[inline(always)]
            fn g(a: $t, b: $t) -> $t { a + b }
            #[inline(always)]
            fn f(x: $t, y: $t) -> $t { Core::minimum(x, y) }
        }
    )+};
}
floats!(f32, f64);

impl Pair<bool> for OrAnd {
    type Out = bool;
    const UNIT: bool = false;
    #[inline(always)]
    fn g(a: bool, b: bool) -> bool { a & b }
    #[inline(always)]
    fn f(x: bool, y: bool) -> bool { x | y }
}

impl Pair<bool> for AndOr {
    type Out = bool;
    const UNIT: bool = true;
    #[inline(always)]
    fn g(a: bool, b: bool) -> bool { a | b }
    #[inline(always)]
    fn f(x: bool, y: bool) -> bool { x & y }
}

impl Inner<bool> for OrAnd {
    type Out = bool;
    fn inner(&self, a: &[bool], b: &[bool], size: [usize; 3]) -> Vec<bool> { packed::<false>(a, b, size) }
}

impl Inner<bool> for AndOr {
    type Out = bool;
    fn inner(&self, a: &[bool], b: &[bool], size: [usize; 3]) -> Vec<bool> { packed::<true>(a, b, size) }
}

/// `∨.∧` of `a` and `b`, or with `NOT` set, `∧.∨`, which is `~(~a)∨.∧(~b)`. `NOT` flips each item of `a` and `b` as it is read, and
/// each result as it is written. Each row of the result ORs together the rows of `b` where the row of `a` is true. The rows of `b`
/// are packed 64 items to a word, in blocks of `WORDS` words, and each block of a row's results folds in a local array that stays in
/// registers. An item of `a` becomes a mask of all ones or none, so the loop over it has no branch.
fn packed<const NOT: bool>(a: &[bool], b: &[bool], [m, k, n]: [usize; 3]) -> Vec<bool> {
    const WORDS: usize = 4;
    assert!(a.len() == m * k && b.len() == k * n, "`a` must hold `m`×`k` items and `b` `k`×`n`");
    let mut c = vec![NOT; m * n];
    if k == 0 || n == 0 { return c }
    let blocks = n.div_ceil(64 * WORDS);
    let mut packed = vec![[0u64; WORDS]; k * blocks];
    for (row, bits) in b.chunks_exact(n).zip(packed.chunks_exact_mut(blocks)) {
        for (j, chunk) in row.chunks(64).enumerate() {
            bits[j / WORDS][j % WORDS] = chunk.iter().enumerate().fold(0, |w, (t, &v)| w | u64::from(v ^ NOT) << t);
        }
    }
    for (row, out) in a.chunks_exact(k).zip(c.chunks_exact_mut(n)) {
        for (block, out) in out.chunks_mut(64 * WORDS).enumerate() {
            let mut acc = [0u64; WORDS];
            for (&x, bits) in row.iter().zip(packed[block..].iter().step_by(blocks)) {
                let mask = u64::from(x ^ NOT).wrapping_neg();
                for t in 0..WORDS { acc[t] |= bits[t] & mask }
            }
            for (j, o) in out.iter_mut().enumerate() { *o = (acc[j / 64] >> (j % 64) & 1 != 0) ^ NOT }
        }
    }
    c
}
impl Pair<i64> for PlusTimes {
    type Out = i64;
    const UNIT: i64 = 0;
    #[inline(always)]
    fn g(a: i64, b: i64) -> i64 { a.wrapping_mul(b) }
    #[inline(always)]
    fn f(x: i64, y: i64) -> i64 { x.wrapping_add(y) }
}
impl Pair<i64> for MaxPlus {
    type Out = i64;
    const UNIT: i64 = i64::MIN;
    #[inline(always)]
    fn g(a: i64, b: i64) -> i64 { a.wrapping_add(b) }
    #[inline(always)]
    fn f(x: i64, y: i64) -> i64 { x.max(y) }
}
impl Pair<i64> for MinPlus {
    type Out = i64;
    const UNIT: i64 = i64::MAX;
    #[inline(always)]
    fn g(a: i64, b: i64) -> i64 { a.wrapping_add(b) }
    #[inline(always)]
    fn f(x: i64, y: i64) -> i64 { x.min(y) }
}
macro_rules! counts {
    ($($t:ty => $out:ty),+) => {$(
        impl Pair<$t> for PlusEqual {
            type Out = $out;
            const UNIT: $out = 0;
            #[inline(always)]
            fn g(a: $t, b: $t) -> $out { <$out>::from(a == b) }
            #[inline(always)]
            fn f(x: $out, y: $out) -> $out { x + y }
        }
    )+};
}
counts!(i64 => i64, i32 => i32, i16 => i16, f64 => i64, f32 => i32);

/// Steps of `p` in each packed block.
#[doc(hidden)]
pub const KC: usize = 256;
/// Rows of `a` in each packed block.
#[doc(hidden)]
pub const MC: usize = 128;
/// Columns of `b` in each packed block.
#[doc(hidden)]
pub const NC: usize = 2048;

/// The tile shapes, `[MR, NR]`, and the bytes in a vector. A tile holds the rows and columns of the result that a micro-kernel keeps in
/// registers. AArch64 has 32 vector registers of 16 bytes, and AVX2 has 16 registers of 32 bytes. wasm has 16-byte vectors, and V8
/// gives it 16 registers on x86-64. The extremes and the counts compare at each step, and on AVX2 they need registers for those
/// comparisons, so their tiles have fewer rows. A plain kernel's tile holds at most 32 items. On the Mac, plain tiles of 48 or more
/// floats compiled to scalar code and ran 4 to 5 times slower.
#[cfg(target_arch = "aarch64")]
mod tiles {
    pub(super) const BYTES: usize = 16;
    pub(super) const TIMES64: [usize; 2] = [8, 4];
    pub(super) const TIMES32: [usize; 2] = [8, 8];
    pub(super) const COMPARES64: [usize; 2] = [6, 8];
    pub(super) const COMPARES32: [usize; 2] = [6, 16];
    pub(super) const COMPARES16: [usize; 2] = [6, 32];
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
mod tiles {
    pub(super) const BYTES: usize = 32;
    pub(super) const TIMES64: [usize; 2] = [6, 8];
    pub(super) const TIMES32: [usize; 2] = [6, 16];
    pub(super) const COMPARES64: [usize; 2] = [4, 8];
    pub(super) const COMPARES32: [usize; 2] = [4, 16];
    pub(super) const COMPARES16: [usize; 2] = [4, 32];
}
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))))]
mod tiles {
    pub(super) const BYTES: usize = 16;
    pub(super) const TIMES64: [usize; 2] = [4, 4];
    pub(super) const TIMES32: [usize; 2] = [4, 8];
    pub(super) const COMPARES64: [usize; 2] = [4, 4];
    pub(super) const COMPARES32: [usize; 2] = [4, 8];
    pub(super) const COMPARES16: [usize; 2] = [4, 16];
}

/// The tile of the plain kernel on 64-bit integers, which only products too large for f64's 53 bits reach.
const PLAIN: [usize; 2] = [4, 8];

macro_rules! inner {
    ($($object:ident at $t:ty => $out:ty: $tile:expr, $kernel:ident;)+) => {$(
        impl Inner<$t> for $object {
            type Out = $out;
            fn inner(&self, a: &[$t], b: &[$t], size: [usize; 3]) -> Vec<Self::Out> {
                const MR: usize = $tile[0];
                const NR: usize = $tile[1];
                tiled_inner::<$t, $object, MR, NR>(a, b, size, $kernel::<MR, NR, { NR * size_of::<$t>() / tiles::BYTES }>)
            }
        }
    )+};
}
/// `P`'s inner product of `a` and `b`, as `Inner::inner` gives it, through `kernel` on tiles of `MR`×`NR`.
#[inline(always)]
fn tiled_inner<T: Copy + Default, P: Pair<T>, const MR: usize, const NR: usize>(
    a: &[T],
    b: &[T],
    [m, k, n]: [usize; 3],
    kernel: impl Fn(&[T], &[T], &mut [[P::Out; NR]; MR], bool),
) -> Vec<P::Out> {
    assert!(a.len() == m * k && b.len() == k * n, "`a` must hold `m`×`k` items and `b` `k`×`n`");
    let mut c = vec![P::UNIT; m * n];
    product::<T, P, MR, NR>(&mut c, n, |i, p| a[i * k + p], |p, j| b[p * n + j], [m, k, n], true, false, kernel);
    c
}
inner! {
    PlusTimes at f64 => f64: tiles::TIMES64, times64;
    PlusTimes at f32 => f32: tiles::TIMES32, times32;
    MaxPlus at f64 => f64: tiles::COMPARES64, max_plus64;
    MaxPlus at f32 => f32: tiles::COMPARES32, max_plus32;
    MinPlus at f64 => f64: tiles::COMPARES64, min_plus64;
    MinPlus at f32 => f32: tiles::COMPARES32, min_plus32;
    PlusEqual at i32 => i32: tiles::COMPARES32, equal_i32;
    PlusEqual at i16 => i16: tiles::COMPARES16, equal_i16;
    PlusEqual at f64 => i64: tiles::COMPARES64, equal64;
    PlusEqual at f32 => i32: tiles::COMPARES32, equal32;
}

/// `+.=` at `i64` counts at `i16` where `windowed` maps both arguments into it, and otherwise at `i32` where every item fits in 32
/// bits. A narrower count puts more items in each vector.
impl Inner<i64> for PlusEqual {
    type Out = i64;
    fn inner(&self, a: &[i64], b: &[i64], size: [usize; 3]) -> Vec<i64> {
        if let Some(c) = windowed(a, b, size) { return c.into_iter().map(i64::from).collect() }
        let narrow = |x: &[i64]| x.iter().map(|&v| i32::try_from(v).ok()).collect::<Option<Vec<_>>>();
        if let (Some(a), Some(b)) = (narrow(a), narrow(b)) { return self.inner(&a, &b, size).into_iter().map(i64::from).collect() }
        const MR: usize = tiles::COMPARES64[0];
        const NR: usize = tiles::COMPARES64[1];
        tiled_inner::<i64, PlusEqual, MR, NR>(a, b, size, equal_i64::<MR, NR, { NR * size_of::<i64>() / tiles::BYTES }>)
    }
}

/// `+.=` of `a` and `b` counted at `i16`, where `k` is below 2^15 and every item of both lies in one window of 2^16 values. Each item
/// becomes its distance from the window's start, read as an `i16`, so two items are equal exactly where they were before.
fn windowed(a: &[i64], b: &[i64], size: [usize; 3]) -> Option<Vec<i16>> {
    if size[1] > i16::MAX as usize { return None }
    let ((la, ha), (lb, hb)) = (crate::reduce::extremes(a), crate::reduce::extremes(b));
    let (lo, hi) = (la.min(lb), ha.max(hb));
    if i128::from(hi) - i128::from(lo) > i128::from(u16::MAX) { return None }
    let shifted = |x: &[i64]| x.iter().map(|&v| v.wrapping_sub(lo) as u16 as i16).collect::<Vec<_>>();
    Some(PlusEqual.inner(&shifted(a), &shifted(b), size))
}

/// `+.≠` as `k` less `+.=`.
macro_rules! differences {
    ($($t:ty => $out:ty),+) => {$(
        impl Inner<$t> for PlusNotEqual {
            type Out = $out;
            fn inner(&self, a: &[$t], b: &[$t], size: [usize; 3]) -> Vec<$out> {
                let k = size[1] as $out;
                PlusEqual.inner(a, b, size).into_iter().map(|equal| k - equal).collect()
            }
        }
    )+};
}
differences!(i64 => i64, i32 => i32, i16 => i16, f64 => i64, f32 => i32);

/// The largest magnitude among the items of `x`.
fn magnitude(x: &[i64]) -> u128 { u128::from(x.iter().fold(0, |m, v| m.max(v.unsigned_abs()))) }

/// The integer inner products of the float pairs. Each computes at the narrowest item type that holds every value the product forms
/// exactly: `f32` where none exceeds 2^24 in magnitude, `f64` where all lie below 2^51, and otherwise `i64` with wrapping arithmetic. The
/// float paths run the float kernels and give the integers that wrapping arithmetic gives. `$bound` gives the largest magnitude that a
/// product can form, from the largest magnitudes in `a` and `b` and from `k`.
macro_rules! integers {
    ($($object:ident: $bound:expr;)+) => {$(
        impl Inner<i64> for $object {
            type Out = i64;
            fn inner(&self, a: &[i64], b: &[i64], size: [usize; 3]) -> Vec<i64> {
                tiered(self, a, b, size, $bound(magnitude(a), magnitude(b), size[1] as u128))
            }
            fn checked(&self, a: &[i64], b: &[i64], size: [usize; 3]) -> Option<Vec<i64>> {
                let bound = $bound(magnitude(a), magnitude(b), size[1] as u128);
                (bound <= i64::MAX as u128).then(|| tiered(self, a, b, size, bound))
            }
        }
    )+};
}
integers! {
    PlusTimes: |a: u128, b: u128, k: u128| a.saturating_mul(b).saturating_mul(k);
    MaxPlus: |a: u128, b: u128, _| a + b;
    MinPlus: |a: u128, b: u128, _| a + b;
}

/// `integers!`'s inner product of `pair`, whose values never exceed `bound` in magnitude.
fn tiered<P>(pair: &P, a: &[i64], b: &[i64], [m, k, n]: [usize; 3], bound: u128) -> Vec<i64>
where
    P: Inner<f32, Out = f32> + Inner<f64, Out = f64> + Pair<i64, Out = i64>,
{
    assert!(a.len() == m * k && b.len() == k * n, "`a` must hold `m`×`k` items and `b` `k`×`n`");
    if k == 0 { return vec![<P as Pair<i64>>::UNIT; m * n]; }
    if bound <= 1 << 24 { return exactly::<f32, P>(pair, a, b, [m, k, n]); }
    if bound <= convert::WIDE_LIMIT as u128 { return exactly::<f64, P>(pair, a, b, [m, k, n]); }
    tiled_inner::<i64, P, { PLAIN[0] }, { PLAIN[1] }>(a, b, [m, k, n], micro::<i64, P, { PLAIN[0] }, { PLAIN[1] }>)
}

/// `pair`'s inner product of `a` and `b` at the float type `F`, which holds every value it forms exactly. The conversions go
/// through `convert`, because `as` between `f64` and `i64` converts one item at a time on AVX2 and in wasm.
fn exactly<F: Float, P: Inner<F, Out = F>>(pair: &P, a: &[i64], b: &[i64], size: [usize; 3]) -> Vec<i64> {
    let floats = |x: &[i64]| x.iter().map(|&v| convert::float::<F, i64>(v)).collect::<Vec<F>>();
    pair.inner(&floats(a), &floats(b), size).into_iter().map(|x| convert::shifted(x.wide())).collect()
}

/// Item types whose matrix products run on the tiled `+.×` kernels: `f32` and `f64`, and complex numbers with parts of either type.
/// The decompositions in `factor` build on them.
pub(crate) trait Tiled: Copy + Default {
    /// `c += a b`, where `c` is an `m`×`n` matrix whose rows start `ldc` items apart, and `a(i, p)` and `b(p, j)` give the items of
    /// the `m`×`k` matrix `a` and the `k`×`n` matrix `b`. `size` is `[m, k, n]`.
    fn add_product(c: &mut [Self], ldc: usize, a: impl Fn(usize, usize) -> Self, b: impl Fn(usize, usize) -> Self, size: [usize; 3]);
    /// `add_product` for the items of `c` on and below its diagonal. Items above it change too, where their tile reaches the diagonal.
    fn add_lower_product(c: &mut [Self], ldc: usize, a: impl Fn(usize, usize) -> Self, b: impl Fn(usize, usize) -> Self, size: [usize; 3]);
}
macro_rules! tiled {
    ($($t:ty: $tile:ident, $kernel:ident;)+) => {$(
        impl Tiled for $t {
            #[inline]
            fn add_product(c: &mut [$t], ldc: usize, a: impl Fn(usize, usize) -> $t, b: impl Fn(usize, usize) -> $t, size: [usize; 3]) {
                const MR: usize = tiles::$tile[0];
                const NR: usize = tiles::$tile[1];
                product::<$t, PlusTimes, MR, NR>(c, ldc, a, b, size, false, false, $kernel::<MR, NR, { NR * size_of::<$t>() / tiles::BYTES }>)
            }
            #[inline]
            fn add_lower_product(c: &mut [$t], ldc: usize, a: impl Fn(usize, usize) -> $t, b: impl Fn(usize, usize) -> $t, size: [usize; 3]) {
                const MR: usize = tiles::$tile[0];
                const NR: usize = tiles::$tile[1];
                product::<$t, PlusTimes, MR, NR>(c, ldc, a, b, size, false, true, $kernel::<MR, NR, { NR * size_of::<$t>() / tiles::BYTES }>)
            }
        }
    )+};
}
tiled! {
    f64: TIMES64, times64;
    f32: TIMES32, times32;
}
/// A complex product sums real products of the parts into `c`, each on the real kernels. With `p = aᵣ bᵣ`, `q = aᵢ bᵢ` and
/// `s = (aᵣ + aᵢ)(bᵣ + bᵢ)`, the real part of `c` gains `p - q` and the imaginary part gains `s - p - q`, as the 3M method forms them
/// from three real products. Where a result's `s - p - q` isn't finite, through an overflow or a non-finite item, its imaginary part
/// gains `aᵣ bᵢ + aᵢ bᵣ` instead, from two more real products, as four real products give it. `$method` names the real product. With
/// `$lower`, only the items on and below the diagonal gain the sums.
macro_rules! parts {
    ($c:ident, $ldc:ident, $a:ident, $b:ident, $size:ident, $method:ident, $lower:expr) => {{
        let [m, k, n] = $size;
        if m == 0 || k == 0 || n == 0 { return }
        let width = |i: usize| if $lower { (i + 1).min(n) } else { n };
        let (mut p, mut q, mut s) = (vec![R::zero(); m * n], vec![R::zero(); m * n], vec![R::zero(); m * n]);
        R::$method(&mut p, n, |i, l| $a(i, l).re, |l, j| $b(l, j).re, $size);
        R::$method(&mut q, n, |i, l| $a(i, l).im, |l, j| $b(l, j).im, $size);
        R::$method(&mut s, n, |i, l| { let x = $a(i, l); x.re + x.im }, |l, j| { let y = $b(l, j); y.re + y.im }, $size);
        let mut finite = true;
        for i in 0..m {
            for j in 0..width(i) {
                let (t, x) = (i * n + j, &mut $c[i * $ldc + j]);
                let im = s[t] - p[t] - q[t];
                finite &= im.is_finite();
                *x = Complex::new(x.re + (p[t] - q[t]), if im.is_finite() { x.im + im } else { x.im });
            }
        }
        if !finite {
            let mut four = vec![R::zero(); m * n];
            R::$method(&mut four, n, |i, l| $a(i, l).re, |l, j| $b(l, j).im, $size);
            R::$method(&mut four, n, |i, l| $a(i, l).im, |l, j| $b(l, j).re, $size);
            for i in 0..m {
                for j in 0..width(i) {
                    let (t, x) = (i * n + j, &mut $c[i * $ldc + j]);
                    if !(s[t] - p[t] - q[t]).is_finite() { x.im = x.im + four[t] }
                }
            }
        }
    }};
}
impl<R: Native + Tiled> Tiled for Complex<R> {
    fn add_product(c: &mut [Self], ldc: usize, a: impl Fn(usize, usize) -> Self, b: impl Fn(usize, usize) -> Self, size: [usize; 3]) {
        parts!(c, ldc, a, b, size, add_product, false)
    }
    fn add_lower_product(c: &mut [Self], ldc: usize, a: impl Fn(usize, usize) -> Self, b: impl Fn(usize, usize) -> Self, size: [usize; 3]) {
        parts!(c, ldc, a, b, size, add_lower_product, true)
    }
}
/// `+.×` on complex matrices, through `Tiled`'s three real products. Each item type has its own impl, so the loops compile in rustymath.
macro_rules! complex_inner {
    ($($r:ty),+) => {$(
        impl Inner<Complex<$r>> for PlusTimes {
            type Out = Complex<$r>;
            fn inner(&self, a: &[Complex<$r>], b: &[Complex<$r>], [m, k, n]: [usize; 3]) -> Vec<Self::Out> {
                assert!(a.len() == m * k && b.len() == k * n, "`a` must hold `m`×`k` items and `b` `k`×`n`");
                let mut c = vec![Complex::default(); m * n];
                Complex::<$r>::add_product(&mut c, n, |i, p| a[i * k + p], |p, j| b[p * n + j], [m, k, n]);
                c
            }
        }
    )+};
}
complex_inner!(f32, f64);

/// The micro-kernels in plain Rust, for targets without the explicit ones. Each takes the number of vectors in a tile row, `V`, as
/// the explicit ones do, and ignores it.
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"), all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))))]
mod plain {
    use super::*;
    macro_rules! plain {
        ($($name:ident: $t:ty, $object:ident;)+) => {$(
            #[inline(always)]
            pub(super) fn $name<const MR: usize, const NR: usize, const V: usize>(
                a: &[$t],
                b: &[$t],
                t: &mut [[<$object as Pair<$t>>::Out; NR]; MR],
                first: bool,
            ) {
                micro::<$t, $object, MR, NR>(a, b, t, first)
            }
        )+};
    }
    plain! {
        times64: f64, PlusTimes;
        times32: f32, PlusTimes;
        max_plus64: f64, MaxPlus;
        max_plus32: f32, MaxPlus;
        min_plus64: f64, MinPlus;
        min_plus32: f32, MinPlus;
        equal64: f64, PlusEqual;
        equal32: f32, PlusEqual;
        equal_i64: i64, PlusEqual;
        equal_i32: i32, PlusEqual;
        equal_i16: i16, PlusEqual;
    }
}
#[cfg(not(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"), all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))))]
use plain::*;

/// Micro-kernels on explicit vectors: 16 bytes on AArch64 and in wasm, and 32 with AVX2. Each entry gives its first step from `a`
/// and `b`, then each later step from the tile's running value `c`, `a` and `b`.
#[cfg(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"), all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
mod lanes {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;
    #[cfg(target_arch = "wasm32")]
    use core::arch::wasm32::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    macro_rules! lanes {
        ($($name:ident: $t:ty, $lanes:literal, $splat:ident, $load:ident, $store:ident, |$a:ident, $b:ident| $first:expr, |$c:ident| $step:expr;)+) => {$(
            /// `super::micro` on vectors of `$lanes` items. Each row of the tile is `V` vectors.
            #[inline(always)]
            pub(super) fn $name<const MR: usize, const NR: usize, const V: usize>(a: &[$t], b: &[$t], t: &mut [[$t; NR]; MR], first: bool) {
                assert!(NR == $lanes * V && a.len() / MR == b.len() / NR);
                let steps = a.len() / MR;
                // SAFETY: this module compiles only for targets with these instructions. Each load and store moves `$lanes` items
                // inside `a`, `b` or a row of `t`.
                unsafe {
                    let (ap, bp) = (a.as_ptr(), b.as_ptr());
                    let mut acc = [[$splat(0.0); V]; MR];
                    let mut p = 0;
                    if first {
                        for r in 0..MR {
                            let $a = $splat(*ap.add(r));
                            for v in 0..V { acc[r][v] = { let $b = $load(bp.add($lanes * v).cast()); $first } }
                        }
                        p = 1;
                    } else {
                        for r in 0..MR { for v in 0..V { acc[r][v] = $load(t[r].as_ptr().add($lanes * v).cast()) } }
                    }
                    while p < steps {
                        let y: [_; V] = core::array::from_fn(|v| $load(bp.add(p * NR + $lanes * v).cast()));
                        for r in 0..MR {
                            let $a = $splat(*ap.add(p * MR + r));
                            for v in 0..V { acc[r][v] = { let ($c, $b) = (acc[r][v], y[v]); $step } }
                        }
                        p += 1;
                    }
                    for r in 0..MR { for v in 0..V { $store(t[r].as_mut_ptr().add($lanes * v).cast(), acc[r][v]) } }
                }
            }
        )+};
    }
    #[cfg(target_arch = "aarch64")]
    lanes! {
        times64: f64, 2, vdupq_n_f64, vld1q_f64, vst1q_f64, |a, b| vmulq_f64(a, b), |c| vfmaq_f64(c, a, b);
        times32: f32, 4, vdupq_n_f32, vld1q_f32, vst1q_f32, |a, b| vmulq_f32(a, b), |c| vfmaq_f32(c, a, b);
        max_plus64: f64, 2, vdupq_n_f64, vld1q_f64, vst1q_f64, |a, b| vaddq_f64(a, b), |c| vmaxq_f64(c, vaddq_f64(a, b));
        max_plus32: f32, 4, vdupq_n_f32, vld1q_f32, vst1q_f32, |a, b| vaddq_f32(a, b), |c| vmaxq_f32(c, vaddq_f32(a, b));
        min_plus64: f64, 2, vdupq_n_f64, vld1q_f64, vst1q_f64, |a, b| vaddq_f64(a, b), |c| vminq_f64(c, vaddq_f64(a, b));
        min_plus32: f32, 4, vdupq_n_f32, vld1q_f32, vst1q_f32, |a, b| vaddq_f32(a, b), |c| vminq_f32(c, vaddq_f32(a, b));
    }
    // Baseline wasm has no fused multiply-add.
    #[cfg(target_arch = "wasm32")]
    lanes! {
        times64: f64, 2, f64x2_splat, v128_load, v128_store, |a, b| f64x2_mul(a, b), |c| f64x2_add(c, f64x2_mul(a, b));
        times32: f32, 4, f32x4_splat, v128_load, v128_store, |a, b| f32x4_mul(a, b), |c| f32x4_add(c, f32x4_mul(a, b));
        max_plus64: f64, 2, f64x2_splat, v128_load, v128_store, |a, b| f64x2_add(a, b), |c| f64x2_max(c, f64x2_add(a, b));
        max_plus32: f32, 4, f32x4_splat, v128_load, v128_store, |a, b| f32x4_add(a, b), |c| f32x4_max(c, f32x4_add(a, b));
        min_plus64: f64, 2, f64x2_splat, v128_load, v128_store, |a, b| f64x2_add(a, b), |c| f64x2_min(c, f64x2_add(a, b));
        min_plus32: f32, 4, f32x4_splat, v128_load, v128_store, |a, b| f32x4_add(a, b), |c| f32x4_min(c, f32x4_add(a, b));
    }
    // AVX2's `maxpd` and `minpd` give their second argument beside NaN. These steps compare and select as `Core::maximum` does.
    #[cfg(target_arch = "x86_64")]
    lanes! {
        times64: f64, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b| _mm256_mul_pd(a, b), |c| _mm256_fmadd_pd(a, b, c);
        times32: f32, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b| _mm256_mul_ps(a, b), |c| _mm256_fmadd_ps(a, b, c);
        max_plus64: f64, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b| _mm256_add_pd(a, b), |c| {
            let s = _mm256_add_pd(a, b);
            _mm256_blendv_pd(s, c, _mm256_or_pd(_mm256_cmp_pd::<_CMP_GT_OQ>(c, s), _mm256_cmp_pd::<_CMP_UNORD_Q>(c, c)))
        };
        max_plus32: f32, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b| _mm256_add_ps(a, b), |c| {
            let s = _mm256_add_ps(a, b);
            _mm256_blendv_ps(s, c, _mm256_or_ps(_mm256_cmp_ps::<_CMP_GT_OQ>(c, s), _mm256_cmp_ps::<_CMP_UNORD_Q>(c, c)))
        };
        min_plus64: f64, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b| _mm256_add_pd(a, b), |c| {
            let s = _mm256_add_pd(a, b);
            _mm256_blendv_pd(s, c, _mm256_or_pd(_mm256_cmp_pd::<_CMP_LT_OQ>(c, s), _mm256_cmp_pd::<_CMP_UNORD_Q>(c, c)))
        };
        min_plus32: f32, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b| _mm256_add_ps(a, b), |c| {
            let s = _mm256_add_ps(a, b);
            _mm256_blendv_ps(s, c, _mm256_or_ps(_mm256_cmp_ps::<_CMP_LT_OQ>(c, s), _mm256_cmp_ps::<_CMP_UNORD_Q>(c, c)))
        };
    }
    /// Micro-kernels whose tile starts at zero, for `PlusEqual`, as `super::micro` on vectors of `$lanes` items, with each row of the tile
    /// `V` vectors. Where items are equal, a comparison gives a lane of all ones, which is -1 as an integer. Subtracting it adds 1 to the
    /// count.
    macro_rules! equalities {
        ($($name:ident: $t:ty => $out:ty, $lanes:literal, $splat:ident, $load:ident, $zero:expr, $load_count:ident, $store_count:ident, |$a:ident, $b:ident, $c:ident| $step:expr;)+) => {$(
            #[inline(always)]
            pub(super) fn $name<const MR: usize, const NR: usize, const V: usize>(a: &[$t], b: &[$t], t: &mut [[$out; NR]; MR], first: bool) {
                assert!(NR == $lanes * V && a.len() / MR == b.len() / NR);
                let steps = a.len() / MR;
                // SAFETY: as in `lanes!`.
                unsafe {
                    let (ap, bp) = (a.as_ptr(), b.as_ptr());
                    let mut acc = [[$zero; V]; MR];
                    if !first { for r in 0..MR { for v in 0..V { acc[r][v] = $load_count(t[r].as_ptr().add($lanes * v).cast()) } } }
                    for p in 0..steps {
                        let y: [_; V] = core::array::from_fn(|v| $load(bp.add(p * NR + $lanes * v).cast()));
                        for r in 0..MR {
                            let $a = $splat(*ap.add(p * MR + r) as _);
                            for v in 0..V { acc[r][v] = { let ($c, $b) = (acc[r][v], y[v]); $step } }
                        }
                    }
                    for r in 0..MR { for v in 0..V { $store_count(t[r].as_mut_ptr().add($lanes * v).cast(), acc[r][v]) } }
                }
            }
        )+};
    }
    #[cfg(target_arch = "aarch64")]
    equalities! {
        equal64: f64 => i64, 2, vdupq_n_f64, vld1q_f64, vdupq_n_s64(0), vld1q_s64, vst1q_s64, |a, b, c| vsubq_s64(c, vreinterpretq_s64_u64(vceqq_f64(a, b)));
        equal32: f32 => i32, 4, vdupq_n_f32, vld1q_f32, vdupq_n_s32(0), vld1q_s32, vst1q_s32, |a, b, c| vsubq_s32(c, vreinterpretq_s32_u32(vceqq_f32(a, b)));
        equal_i64: i64 => i64, 2, vdupq_n_s64, vld1q_s64, vdupq_n_s64(0), vld1q_s64, vst1q_s64, |a, b, c| vsubq_s64(c, vreinterpretq_s64_u64(vceqq_s64(a, b)));
        equal_i32: i32 => i32, 4, vdupq_n_s32, vld1q_s32, vdupq_n_s32(0), vld1q_s32, vst1q_s32, |a, b, c| vsubq_s32(c, vreinterpretq_s32_u32(vceqq_s32(a, b)));
        equal_i16: i16 => i16, 8, vdupq_n_s16, vld1q_s16, vdupq_n_s16(0), vld1q_s16, vst1q_s16, |a, b, c| vsubq_s16(c, vreinterpretq_s16_u16(vceqq_s16(a, b)));
    }
    #[cfg(target_arch = "wasm32")]
    equalities! {
        equal64: f64 => i64, 2, f64x2_splat, v128_load, i64x2_splat(0), v128_load, v128_store, |a, b, c| i64x2_sub(c, f64x2_eq(a, b));
        equal32: f32 => i32, 4, f32x4_splat, v128_load, i32x4_splat(0), v128_load, v128_store, |a, b, c| i32x4_sub(c, f32x4_eq(a, b));
        equal_i64: i64 => i64, 2, i64x2_splat, v128_load, i64x2_splat(0), v128_load, v128_store, |a, b, c| i64x2_sub(c, i64x2_eq(a, b));
        equal_i32: i32 => i32, 4, i32x4_splat, v128_load, i32x4_splat(0), v128_load, v128_store, |a, b, c| i32x4_sub(c, i32x4_eq(a, b));
        equal_i16: i16 => i16, 8, i16x8_splat, v128_load, i16x8_splat(0), v128_load, v128_store, |a, b, c| i16x8_sub(c, i16x8_eq(a, b));
    }
    #[cfg(target_arch = "x86_64")]
    equalities! {
        equal64: f64 => i64, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_setzero_si256(), _mm256_loadu_si256, _mm256_storeu_si256, |a, b, c| {
            _mm256_sub_epi64(c, _mm256_castpd_si256(_mm256_cmp_pd::<_CMP_EQ_OQ>(a, b)))
        };
        equal32: f32 => i32, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_setzero_si256(), _mm256_loadu_si256, _mm256_storeu_si256, |a, b, c| {
            _mm256_sub_epi32(c, _mm256_castps_si256(_mm256_cmp_ps::<_CMP_EQ_OQ>(a, b)))
        };
        equal_i64: i64 => i64, 4, _mm256_set1_epi64x, _mm256_loadu_si256, _mm256_setzero_si256(), _mm256_loadu_si256, _mm256_storeu_si256, |a, b, c| {
            _mm256_sub_epi64(c, _mm256_cmpeq_epi64(a, b))
        };
        equal_i32: i32 => i32, 8, _mm256_set1_epi32, _mm256_loadu_si256, _mm256_setzero_si256(), _mm256_loadu_si256, _mm256_storeu_si256, |a, b, c| {
            _mm256_sub_epi32(c, _mm256_cmpeq_epi32(a, b))
        };
        equal_i16: i16 => i16, 16, _mm256_set1_epi16, _mm256_loadu_si256, _mm256_setzero_si256(), _mm256_loadu_si256, _mm256_storeu_si256, |a, b, c| {
            _mm256_sub_epi16(c, _mm256_cmpeq_epi16(a, b))
        };
    }
}
#[cfg(any(target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"), all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")))]
use lanes::*;

/// Folds the inner product of `P` into `c`, an `m`×`n` matrix whose rows start `ldc` items apart, with `kernel` folding each `MR`×`NR`
/// tile as `micro` does. `a(i, p)` and `b(p, j)` give the items of the `m`×`k` matrix `a` and the `k`×`n` matrix `b`. With `start`, the
/// product replaces `c`, which must then hold `f`'s identity where `k` is 0. Otherwise each result folds into `c`'s item with `f`. With
/// `lower`, only tiles that reach the diagonal or below it fold, which leaves `c`'s upper triangle partly folded.
#[inline(always)]
fn product<T: Copy + Default, P: Pair<T>, const MR: usize, const NR: usize>(
    c: &mut [P::Out],
    ldc: usize,
    a: impl Fn(usize, usize) -> T,
    b: impl Fn(usize, usize) -> T,
    [m, k, n]: [usize; 3],
    start: bool,
    lower: bool,
    kernel: impl Fn(&[T], &[T], &mut [[P::Out; NR]; MR], bool),
) {
    if m == 0 || k == 0 || n == 0 { return; }
    assert!(ldc >= n && c.len() >= (m - 1) * ldc + n, "`c` must hold `m` rows of `n` items, `ldc` apart");
    let mut ap = vec![T::default(); MC.min(m).next_multiple_of(MR) * KC.min(k)];
    let mut bp = vec![T::default(); NC.min(n).next_multiple_of(NR) * KC.min(k)];
    for jc in (0..n).step_by(NC) {
        let nc = NC.min(n - jc);
        for pc in (0..k).step_by(KC) {
            let kc = KC.min(k - pc);
            let load = pc > 0 || !start;
            pack::<_, NR>(&mut bp, kc, nc, |p, j| b(pc + p, jc + j));
            for ic in (0..m).step_by(MC) {
                let mc = MC.min(m - ic);
                if lower && ic + mc <= jc { continue }
                pack::<_, MR>(&mut ap, kc, mc, |p, i| a(ic + i, pc + p));
                for jr in (0..nc).step_by(NR) {
                    for ir in (0..mc).step_by(MR) {
                        if lower && ic + ir + MR.min(mc - ir) <= jc + jr { continue }
                        let (rows, cols) = (MR.min(mc - ir), NR.min(nc - jr));
                        let at = |r: usize, s: usize| (ic + ir + r) * ldc + jc + jr + s;
                        let mut t = [[P::UNIT; NR]; MR];
                        if load { for r in 0..rows { for s in 0..cols { t[r][s] = c[at(r, s)] } } }
                        kernel(&ap[ir * kc..(ir + MR) * kc], &bp[jr * kc..(jr + NR) * kc], &mut t, !load);
                        for r in 0..rows { for s in 0..cols { c[at(r, s)] = t[r][s] } }
                    }
                }
            }
        }
    }
}

/// Packs `len` lines of `kc` steps into strips of `W` lines. Each strip holds its `W` items for step 0, then those for step 1, and so
/// on. `item(p, l)` gives line `l` at step `p`. Lines past `len` hold the default, and the results computed from them are never stored.
fn pack<T: Copy + Default, const W: usize>(buffer: &mut [T], kc: usize, len: usize, item: impl Fn(usize, usize) -> T) {
    for (s, strip) in buffer.chunks_exact_mut(W * kc).take(len.div_ceil(W)).enumerate() {
        let (first, lines) = (s * W, W.min(len - s * W));
        for (p, step) in strip.as_chunks_mut::<W>().0.iter_mut().enumerate() {
            if lines == W {
                for l in 0..W { step[l] = item(p, first + l) }
            } else {
                for l in 0..W { step[l] = if l < lines { item(p, first + l) } else { T::default() } }
            }
        }
    }
}

/// Folds the steps of the strips `a` and `b`, which `pack` wrote, into the tile `t`. With `first`, the first step starts the tile.
#[inline(always)]
fn micro<T: Copy, P: Pair<T>, const MR: usize, const NR: usize>(a: &[T], b: &[T], t: &mut [[P::Out; NR]; MR], first: bool) {
    let mut acc = *t;
    let mut steps = a.as_chunks::<MR>().0.iter().zip(b.as_chunks::<NR>().0);
    if first {
        let (a, b) = steps.next().expect("a block has at least one step");
        for r in 0..MR { for s in 0..NR { acc[r][s] = P::g(a[r], b[s]) } }
    }
    for (a, b) in steps {
        for r in 0..MR { for s in 0..NR { acc[r][s] = P::step(acc[r][s], a[r], b[s]) } }
    }
    *t = acc;
}

