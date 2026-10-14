//! Inner products of row-major matrices. The inner product `f.g` of an `m`×`k` matrix `a` and a `k`×`n` matrix `b` is the `m`×`n`
//! matrix whose item at row `i` and column `j` folds `f` over `g(a[i][p], b[p][j])` for each `p`. `+.×` is the matrix product. The
//! `f` of each pair is associative, and the loops combine its steps in whatever order suits them. A float result can therefore differ
//! in its last bits from a fold taken in order.
//!
//! The loops follow GotoBLAS and BLIS. Packing copies a block of each argument into a contiguous buffer sized for a cache level. The
//! blocking reuses each buffer while it is in cache. A micro-kernel computes a small tile of the result in registers, one step of `p`
//! at a time. A micro-kernel in plain Rust vectorizes through LLVM's SLP vectorizer, which the builds of rustymath and basedpl turn on
//! at opt-level 2. It vectorizes only tiles of up to 32 items, too few to keep the multiply-add units busy. On AArch64, in wasm
//! and on x86-64 with AVX2 and FMA, the float pairs run micro-kernels on explicit vectors instead. Their maximum and minimum are each
//! target's own instruction: `fmax` on AArch64, `max` in wasm and `maxpd` on x86-64. On AArch64 and in wasm a NaN term makes a
//! result NaN. On x86-64 a NaN term is skipped. Targets without these vectors run the plain kernels.
//!
//! A product large enough splits its rows across the threads that [`crate::parallel::set_threads`] sets.
//! On macOS, Accelerate computes real and complex plus-times products instead. `Tiled` also sends the decompositions' triangular
//! solves, SVDs and Hermitian eigen decompositions to it, and the general eigen decompositions of real matrices from
//! `accelerate::GEEV_FROM` rows. macOS builds with the `native` feature also compile rustymath's own kernels, and `RUSTYMATH_NATIVE=1`
//! selects them for the whole process.
//!
//! The private module `factor` holds decompositions of real and complex matrices, whose blocked updates run on these loops. This module
//! re-exports its public items, and [`Scalar`], the trait of the item types they take. [`transpose_into`] transposes a matrix through 8×8
//! tiles.
use crate::{convert, float::{sealed::Core, FUSED}, parallel, slice, Float};
use num_complex::Complex;
use num_traits::{One, Zero};
use std::{mem::MaybeUninit, sync::LazyLock};

mod factor;
mod scalar;
#[cfg(accelerate)]
mod accelerate;
// The platform's own BLAS and LAPACK, which every `Tiled` type implements: Accelerate on macOS, and none elsewhere.
#[cfg(accelerate)]
pub(crate) use accelerate::Accelerated as Vendor;
#[cfg(not(accelerate))]
pub(crate) trait Vendor {}
#[cfg(not(accelerate))]
impl<T> Vendor for T {}

/// Whether this process selects rustymath's own kernels where Accelerate is also compiled in: in macOS builds with the `native`
/// feature, when `RUSTYMATH_NATIVE=1`. It's read once.
fn switched() -> bool {
    static NATIVE: LazyLock<bool> = LazyLock::new(|| std::env::var_os("RUSTYMATH_NATIVE").is_some_and(|v| v == "1"));
    cfg!(all(accelerate, native)) && *NATIVE
}

/// Whether Accelerate serves this process's products, triangular solves and decompositions.
pub(crate) fn accelerated() -> bool { cfg!(accelerate) && !switched() }

/// `$vendor` where Accelerate serves this process, and `$native` where rustymath's own kernels do. Each compiles only where its kernels
/// do.
macro_rules! dispatch {
    ($vendor:expr, $native:expr) => {{
        #[cfg(all(accelerate, native))]
        let r = if accelerated() { $vendor } else { $native };
        #[cfg(all(accelerate, not(native)))]
        let r = $vendor;
        #[cfg(not(accelerate))]
        let r = $native;
        r
    }};
}
pub mod exact;
pub use factor::*;
pub use scalar::Scalar;

/// The `n`×`n` identity matrix.
pub(crate) fn identity<T: num_traits::Zero + num_traits::One>(n: usize) -> Vec<T> {
    let mut x = Vec::new();
    identity_into(&mut x, n);
    x
}

/// [`identity`], written into `x` in place of its old items.
pub(crate) fn identity_into<T: num_traits::Zero + num_traits::One>(x: &mut Vec<T>, n: usize) {
    x.clear();
    x.extend(std::iter::repeat_with(T::zero).take(n * n));
    for i in 0..n { x[i * (n + 1)] = T::one() }
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
    type Out: Copy;
    /// The `m`×`n` inner product of `a`, an `m`×`k` matrix, and `b`, a `k`×`n` matrix, both row-major, as a row-major matrix. `size`
    /// is `[m, k, n]`. Where `k` is 0, each result is the identity of `f`. Integer arithmetic wraps, as NumPy's does.
    fn inner(&self, a: &[T], b: &[T], size: [usize; 3]) -> Vec<Self::Out>;
    /// `inner`, or `None` where an integer result could overflow, for callers that keep integers exact. A result it gives is the exact
    /// value. Floats never give `None`.
    fn checked(&self, a: &[T], b: &[T], size: [usize; 3]) -> Option<Vec<Self::Out>> { Some(self.inner(a, b, size)) }
    /// `inner`, written into `out`, which holds `m`×`n` items.
    fn inner_into(&self, a: &[T], b: &[T], size: [usize; 3], out: &mut [Self::Out]) { out.copy_from_slice(&self.inner(a, b, size)) }
}

/// `+.×`, the matrix product, at `i64`, `f32` and `f64`, and complex numbers with parts of either type. Native float steps use fused
/// multiply-add where the target has one; Accelerate computes them on macOS.
/// Nonfinite complex results need not match scalar arithmetic. Native complex products sum three real products of the parts, as the
/// 3M method does. Their real parts are as accurate as four real products make them. Its normwise error is about 4 units of roundoff,
/// against about 2.5 for four real products. An imaginary part's error scales with the size of the whole product. An imaginary part a
/// millionth the size of its real part keeps about 10 of f64's 16 digits and about 1 of f32's 7. A result whose 3M imaginary part isn't
/// finite takes four real products instead.
#[derive(Default)]
pub struct PlusTimes;
/// `⌈.+`, the max-plus product of longest-path problems, at `i64`, `f32` and `f64`. On AArch64 a NaN term makes its result NaN.
/// Other targets skip NaN terms.
#[derive(Default)]
pub struct MaxPlus;
/// `⌊.+`, the min-plus product of shortest-path problems, at `i64`, `f32` and `f64`. Its minimum treats NaN as `MaxPlus`'s maximum does.
#[derive(Default)]
pub struct MinPlus;
/// `∨.∧` on Booleans: whether some `p` has both `a[i][p]` and `b[p][j]`.
#[derive(Default)]
pub struct OrAnd;
/// `∧.∨` on Booleans: whether every `p` has `a[i][p]` or `b[p][j]`.
#[derive(Default)]
pub struct AndOr;
/// `+.=`, at `i16`, `i32`, `i64`, `f32` and `f64`: the number of `p` where `a[i][p]` equals `b[p][j]`. The count has the width of the
/// items, as `i16`, `i32` or `i64`, so an `i16` count needs `k` below 2^15 and an `i32` count below 2^31. Floats compare exactly, so NaN
/// equals nothing.
#[derive(Default)]
pub struct PlusEqual;
/// `+.≠`, at `i16`, `i32`, `i64`, `f32` and `f64`: the number of `p` where `a[i][p]` differs from `b[p][j]`, with `+.=`'s count type.
#[derive(Default)]
pub struct PlusNotEqual;

/// The scalar operations of a pair at item type `T`.
trait Pair<T> {
    type Out: Copy + Send;
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
            fn f(acc: $t, term: $t) -> $t { Core::maximum(term, acc) }
        }
        impl Pair<$t> for MinPlus {
            type Out = $t;
            const UNIT: $t = <$t>::INFINITY;
            #[inline(always)]
            fn g(a: $t, b: $t) -> $t { a + b }
            #[inline(always)]
            fn f(acc: $t, term: $t) -> $t { Core::minimum(term, acc) }
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
    #[cfg(native)]
    pub(super) const TIMES64: [usize; 2] = [8, 4];
    #[cfg(native)]
    pub(super) const TIMES32: [usize; 2] = [8, 8];
    pub(super) const COMPARES64: [usize; 2] = [6, 8];
    pub(super) const COMPARES32: [usize; 2] = [6, 16];
    pub(super) const COMPARES16: [usize; 2] = [6, 32];
}
#[cfg(all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma"))]
mod tiles {
    pub(super) const BYTES: usize = 32;
    #[cfg(native)]
    pub(super) const TIMES64: [usize; 2] = [6, 8];
    #[cfg(native)]
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
                let mut c = filled(size[0] * size[2], <$object as Pair<$t>>::UNIT);
                self.inner_into(a, b, size, &mut c);
                c
            }
            fn inner_into(&self, a: &[$t], b: &[$t], size: [usize; 3], out: &mut [Self::Out]) {
                const MR: usize = $tile[0];
                const NR: usize = $tile[1];
                tiled_into::<$t, $object, MR, NR>(a, b, size, out, $kernel::<MR, NR, { NR * size_of::<$t>() / tiles::BYTES }>)
            }
        }
    )+};
}
/// `n` copies of `x`. `vec!` of zeros allocates through calloc, whose `madvise` on a large block costs more than writing its items.
fn filled<T: Clone>(n: usize, x: T) -> Vec<T> {
    let mut v = Vec::with_capacity(n);
    v.resize(n, x);
    v
}
/// `P`'s inner product of `a` and `b`, as `Inner::inner` gives it, through `kernel` on tiles of `MR`×`NR`.
#[inline(always)]
fn tiled_inner<T: Copy + Default + Send + Sync, P: Pair<T>, const MR: usize, const NR: usize>(
    a: &[T],
    b: &[T],
    size: [usize; 3],
    kernel: impl Fn(&[T], &[T], &mut [[P::Out; NR]; MR], bool) + Sync,
) -> Vec<P::Out> {
    let mut c = filled(size[0] * size[2], P::UNIT);
    tiled_into::<T, P, MR, NR>(a, b, size, &mut c, kernel);
    c
}
/// `tiled_inner`, written into `c`, which holds `m`×`n` items.
#[inline(always)]
fn tiled_into<T: Copy + Default + Send + Sync, P: Pair<T>, const MR: usize, const NR: usize>(
    a: &[T],
    b: &[T],
    [m, k, n]: [usize; 3],
    c: &mut [P::Out],
    kernel: impl Fn(&[T], &[T], &mut [[P::Out; NR]; MR], bool) + Sync,
) {
    assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `c` `m`×`n`");
    if k == 0 { c.fill(P::UNIT) }
    product::<T, P, MR, NR>(c, n, |i, p| a[i * k + p], |p, j, out| row_run(b, n, p, j, out), [m, k, n], true, Part::Whole, kernel);
}
inner! {
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

/// A strided matrix borrowed in its stored, transposed, conjugated or adjoint orientation.
#[derive(Clone, Copy)]
pub(crate) struct Matrix<'a, T> { pub data: &'a [T], pub ld: usize, pub op: Op }
#[derive(Clone, Copy)]
pub(crate) enum Op { Normal, Transpose, Adjoint, Conjugate }
#[derive(Clone, Copy)]
pub(crate) enum Side { Left, Right }
/// The stored triangle, before applying the matrix view's transpose or adjoint.
#[derive(Clone, Copy)]
pub(crate) enum Triangle { Lower, UnitUpper }
impl Op {
    #[inline(always)]
    fn transposed(self) -> bool { matches!(self, Self::Transpose | Self::Adjoint) }
}
impl<'a, T: Copy> Matrix<'a, T> {
    pub fn rows(data: &'a [T], ld: usize) -> Self { Self { data, ld, op: Op::Normal } }
    pub fn transpose(self) -> Self {
        Self { op: match self.op { Op::Normal => Op::Transpose, Op::Transpose => Op::Normal, Op::Adjoint => Op::Conjugate, Op::Conjugate => Op::Adjoint }, ..self }
    }
    /// Starts at a row and column in the logical orientation.
    #[cfg(native)]
    pub fn offset(self, row: usize, col: usize) -> Self {
        let at = if self.op.transposed() { col * self.ld + row } else { row * self.ld + col };
        Self { data: &self.data[at..], ..self }
    }
    /// Checks the stored extent once, before any foreign call.
    fn check(self, mut rows: usize, mut cols: usize) {
        if self.op.transposed() { std::mem::swap(&mut rows, &mut cols) }
        if rows == 0 || cols == 0 { return }
        let end = (rows - 1).checked_mul(self.ld).and_then(|n| n.checked_add(cols));
        assert!(self.ld >= cols && end.is_some_and(|n| n <= self.data.len()), "matrix view must hold its rows and columns");
    }
}
impl<T: Scalar> Matrix<'_, T> {
    pub fn conjugate(self) -> Self {
        // Conjugation is an identity for real scalars; don't make their BLAS views more restrictive.
        if size_of::<T>() == size_of::<T::Real>() { return self }
        Self { op: match self.op { Op::Normal => Op::Conjugate, Op::Transpose => Op::Adjoint, Op::Adjoint => Op::Transpose, Op::Conjugate => Op::Normal }, ..self }
    }
    pub fn adjoint(self) -> Self { self.transpose().conjugate() }
    #[inline(always)]
    pub fn at(self, i: usize, j: usize) -> T {
        let x = self.data[if self.op.transposed() { j * self.ld + i } else { i * self.ld + j }];
        if matches!(self.op, Op::Adjoint | Op::Conjugate) { x.conj() } else { x }
    }
    #[cfg(native)]
    #[inline(always)]
    fn run(self, p: usize, j: usize, out: &mut [MaybeUninit<T>]) {
        if matches!(self.op, Op::Normal) { row_run(self.data, self.ld, p, j, out) }
        else { for (l, slot) in out.iter_mut().enumerate() { *slot = MaybeUninit::new(self.at(p, j + l)) } }
    }
}
/// Copies a contiguous run without repeating its bounds check for each packed item.
#[inline(always)]
fn row_run<T: Copy>(data: &[T], ld: usize, p: usize, j: usize, out: &mut [MaybeUninit<T>]) {
    let items = &data[p * ld + j..][..out.len()];
    for l in 0..out.len() { out[l] = MaybeUninit::new(items[l]) }
}
/// Item types whose matrix products run on the tiled `+.×` kernels: `f32` and `f64`, and complex numbers with parts of either type.
/// The decompositions in `factor` build on them.
pub(crate) trait Tiled: Scalar + Default + Vendor {
    /// The row-major matrix product, allocated once through the same GEMM operation as updates.
    #[inline]
    fn multiply(a: Matrix<'_, Self>, b: Matrix<'_, Self>, size: [usize; 3]) -> Vec<Self> {
        let mut c = filled(size[0] * size[2], Self::zero());
        Self::gemm(&mut c, size[2], a, b, size, Self::one(), Self::zero());
        c
    }
    /// `c = alpha a b + beta c`, with `size = [m, k, n]`. Dispatch precedes native packing and row splitting.
    #[inline]
    fn gemm(c: &mut [Self], ldc: usize, a: Matrix<'_, Self>, b: Matrix<'_, Self>, size: [usize; 3], alpha: Self, beta: Self) {
        gemm(c, ldc, a, b, size, alpha, beta, Part::Whole)
    }
    /// The lower triangle of `c = alpha x xᴴ + beta c`, for `x` of size `[n, k]`. Native diagonal tiles may also change upper items.
    #[inline]
    fn gram(c: &mut [Self], ldc: usize, x: Matrix<'_, Self>, [n, k]: [usize; 2], alpha: Self::Real, beta: Self::Real) {
        gemm(c, ldc, x, x.adjoint(), [n, k, n], Self::from_real(alpha), Self::from_real(beta), Part::Lower)
    }
    /// Solves `a x = b` or `x a = b` in place, with `triangle` describing the stored square matrix `a`.
    fn triangular(x: &mut [Self], ldx: usize, a: Matrix<'_, Self>, [m, n]: [usize; 2], side: Side, triangle: Triangle) {
        let order = match side { Side::Left => m, Side::Right => n };
        a.check(order, order);
        Matrix::rows(x, ldx).check(m, n);
        if m == 0 || n == 0 { return }
        dispatch!(accelerate::triangular(x, ldx, a, [m, n], side, triangle), factor::triangular_native(x, ldx, a, [m, n], side, triangle))
    }
    /// Native packing also serves computed real parts inside the complex 3M implementation.
    #[cfg(native)]
    fn native_product(c: &mut [Self], ldc: usize, a: impl Fn(usize, usize) -> Self + Sync, b: Matrix<'_, Self>, size: [usize; 3], start: bool, part: Part);
    /// The thin SVD of the finite, nonempty `m`×`n` matrix `a` as `(U, s, V)`, with its singular vectors only where `vectors` is true, or
    /// `None` where it doesn't converge.
    fn svd(a: &[Self], size: [usize; 2], vectors: bool) -> Option<(Vec<Self>, Vec<Self::Real>, Vec<Self>)>;
    /// The eigen decomposition of the nonempty Hermitian `n`×`n` matrix whose finite lower triangle `a` holds, with its eigenvectors only
    /// where `vectors` is true, or `None` where it doesn't converge.
    fn eigh(a: &[Self], n: usize, vectors: bool) -> Option<Eigh<Self>>;
    /// The eigen decomposition of the finite, nonempty `n`×`n` matrix `a`, with its eigenvectors only where `vectors` is true, or `None`
    /// where it doesn't converge.
    fn eig(a: &[Self], n: usize, vectors: bool) -> Option<Eig<Self::Real>>;
}

fn gemm<T: Tiled>(c: &mut [T], ldc: usize, a: Matrix<'_, T>, b: Matrix<'_, T>, [m, k, n]: [usize; 3], alpha: T, beta: T, part: Part) {
    a.check(m, k);
    b.check(k, n);
    Matrix::rows(c, ldc).check(m, n);
    if m == 0 || n == 0 { return }
    if k == 0 || alpha == T::zero() { return scale(c, ldc, [m, n], beta, part) }
    dispatch!(accelerate::product(c, ldc, a, b, [m, k, n], alpha, beta, part), native_gemm(c, ldc, a, b, [m, k, n], alpha, beta, part))
}

/// `c = beta c` for the items of `c` in `part`, where `c` is an `m`×`n` matrix whose rows start `ldc` items apart. A zero `beta` writes
/// zeros, whatever `c` held.
fn scale<T: Tiled>(c: &mut [T], ldc: usize, [m, n]: [usize; 2], beta: T, part: Part) {
    if beta == T::one() { return }
    for i in 0..m {
        for j in part.columns(i, n) {
            let x = &mut c[i * ldc + j];
            *x = if beta == T::zero() { T::zero() } else { beta * *x };
        }
    }
}

/// `gemm` on rustymath's own packed kernels, for a nonzero `k` and `alpha`.
#[cfg(native)]
fn native_gemm<T: Tiled>(c: &mut [T], ldc: usize, a: Matrix<'_, T>, b: Matrix<'_, T>, [m, k, n]: [usize; 3], alpha: T, beta: T, part: Part) {
    let start = beta == T::zero();
    if !start { scale(c, ldc, [m, n], beta, part) }
    T::native_product(c, ldc, |i, p| if alpha == T::one() { a.at(i, p) } else { alpha * a.at(i, p) }, b, [m, k, n], start, part)
}
/// The items of a product's result that it must fold: all of them, or those on and below its diagonal.
#[derive(Clone, Copy)]
pub(crate) enum Part { Whole, Lower }
impl Part {
    /// The columns of row `i` that this part holds, in a row of `n` items.
    #[inline(always)]
    fn columns(self, i: usize, n: usize) -> std::ops::Range<usize> {
        match self { Part::Whole => 0..n, Part::Lower => 0..(i + 1).min(n) }
    }
}
macro_rules! tiled {
    (@real $t:ty, $tile:ident, $kernel:ident; $c:ident, $ldc:ident, $a:ident, $b:ident, $size:ident, $start:ident, $part:ident) => {{
        const MR: usize = tiles::$tile[0];
        const NR: usize = tiles::$tile[1];
        product::<$t, PlusTimes, MR, NR>($c, $ldc, $a, |p, j, out| $b.run(p, j, out), $size, $start, $part, $kernel::<MR, NR, { NR * size_of::<$t>() / tiles::BYTES }>)
    }};
    (@complex $r:ty; $c:ident, $ldc:ident, $a:ident, $b:ident, $size:ident, $start:ident, $part:ident) => {
        complex_product::<$r>($c, $ldc, $a, $b, $size, $start, $part)
    };
    (@eig real; $a:ident, $n:ident, $vectors:ident) => {{
        #[cfg(accelerate)]
        if $n >= accelerate::GEEV_FROM && accelerated() { return accelerate::eig($a, $n, $vectors) }
        factor::general::native($a, $n, $vectors)
    }};
    (@eig complex; $a:ident, $n:ident, $vectors:ident) => { factor::general::native($a, $n, $vectors) };
    ($($t:ty: $kind:ident($($args:tt)*);)+) => {$(
        impl Tiled for $t {
            #[cfg(native)]
            #[inline]
            fn native_product(c: &mut [$t], ldc: usize, a: impl Fn(usize, usize) -> $t + Sync, b: Matrix<'_, $t>, size: [usize; 3], start: bool, part: Part) {
                tiled!(@$kind $($args)*; c, ldc, a, b, size, start, part)
            }
            fn svd(a: &[$t], size: [usize; 2], vectors: bool) -> Option<(Vec<$t>, Vec<<$t as Scalar>::Real>, Vec<$t>)> {
                dispatch!(accelerate::svd(a, size, vectors), factor::svd::native(a, size, vectors))
            }
            fn eigh(a: &[$t], n: usize, vectors: bool) -> Option<Eigh<$t>> {
                dispatch!(accelerate::eigh(a, n, vectors), factor::eigen::native(a, n, vectors))
            }
            fn eig(a: &[$t], n: usize, vectors: bool) -> Option<Eig<<$t as Scalar>::Real>> {
                tiled!(@eig $kind; a, n, vectors)
            }
        }
        impl Inner<$t> for PlusTimes {
            type Out = $t;
            fn inner(&self, a: &[$t], b: &[$t], [m, k, n]: [usize; 3]) -> Vec<$t> {
                assert!(a.len() == m * k && b.len() == k * n, "`a` must hold `m`×`k` items and `b` `k`×`n`");
                <$t>::multiply(Matrix::rows(a, k), Matrix::rows(b, n), [m, k, n])
            }
            fn inner_into(&self, a: &[$t], b: &[$t], [m, k, n]: [usize; 3], c: &mut [$t]) {
                assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `c` `m`×`n`");
                <$t>::gemm(c, n, Matrix::rows(a, k), Matrix::rows(b, n), [m, k, n], <$t>::one(), <$t>::zero())
            }
        }
    )+};
}
tiled! {
    f32: real(f32, TIMES32, times32);
    f64: real(f64, TIMES64, times64);
    Complex<f32>: complex(f32);
    Complex<f64>: complex(f64);
}
/// Native complex 3M: `p = aᵣ bᵣ`, `q = aᵢ bᵢ`, `s = (aᵣ + aᵢ)(bᵣ + bᵢ)` gives real `p-q` and imaginary `s-p-q`.
/// Nonfinite imaginary parts use the two ordinary cross-products instead.
#[cfg(native)]
fn complex_product<R: crate::Native + Tiled>(c: &mut [Complex<R>], ldc: usize, a: impl Fn(usize, usize) -> Complex<R> + Sync, b: Matrix<'_, Complex<R>>, size: [usize; 3], start: bool, part: Part) {
    let [m, k, n] = size;
    if m == 0 || k == 0 || n == 0 { return }
    let (mut p, mut q, mut s) = (filled(m * n, R::zero()), filled(m * n, R::zero()), filled(m * n, R::zero()));
    let (mut br, mut bi, mut bs) = (filled(k * n, R::zero()), filled(k * n, R::zero()), filled(k * n, R::zero()));
    for l in 0..k {
        let (re, im, sum) = (&mut br[l * n..][..n], &mut bi[l * n..][..n], &mut bs[l * n..][..n]);
        for j in 0..n { let x = b.at(l, j); (re[j], im[j], sum[j]) = (x.re, x.im, x.re + x.im) }
    }
    R::native_product(&mut p, n, |i, l| a(i, l).re, Matrix::rows(&br, n), size, true, part);
    R::native_product(&mut q, n, |i, l| a(i, l).im, Matrix::rows(&bi, n), size, true, part);
    R::native_product(&mut s, n, |i, l| { let x = a(i, l); x.re + x.im }, Matrix::rows(&bs, n), size, true, part);
    let mut finite = true;
    for i in 0..m {
        for j in part.columns(i, n) {
            let (t, x) = (i * n + j, &mut c[i * ldc + j]);
            let im = s[t] - p[t] - q[t];
            finite &= im.is_finite();
            if start { *x = Complex::default() }
            *x = Complex::new(x.re + (p[t] - q[t]), if im.is_finite() { x.im + im } else { x.im });
        }
    }
    if !finite {
        let mut four = filled(m * n, R::zero());
        R::native_product(&mut four, n, |i, l| a(i, l).re, Matrix::rows(&bi, n), size, true, part);
        R::native_product(&mut four, n, |i, l| a(i, l).im, Matrix::rows(&br, n), size, false, part);
        for i in 0..m {
            for j in part.columns(i, n) {
                let (t, x) = (i * n + j, &mut c[i * ldc + j]);
                if !(s[t] - p[t] - q[t]).is_finite() { x.im = x.im + four[t] }
            }
        }
    }
}

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
    // rustymath's own plus-times products, which macOS compiles only with the `native` feature.
    #[cfg(all(target_arch = "aarch64", native))]
    lanes! {
        times64: f64, 2, vdupq_n_f64, vld1q_f64, vst1q_f64, |a, b| vmulq_f64(a, b), |c| vfmaq_f64(c, a, b);
        times32: f32, 4, vdupq_n_f32, vld1q_f32, vst1q_f32, |a, b| vmulq_f32(a, b), |c| vfmaq_f32(c, a, b);
    }
    #[cfg(target_arch = "aarch64")]
    lanes! {
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
    // rustymath's own plus-times products, which macOS compiles only with the `native` feature.
    #[cfg(all(target_arch = "x86_64", native))]
    lanes! {
        times64: f64, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b| _mm256_mul_pd(a, b), |c| _mm256_fmadd_pd(a, b, c);
        times32: f32, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b| _mm256_mul_ps(a, b), |c| _mm256_fmadd_ps(a, b, c);
    }
    // AVX2's `maxpd` and `minpd` give their second argument beside NaN.
    #[cfg(target_arch = "x86_64")]
    lanes! {
        max_plus64: f64, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b| _mm256_add_pd(a, b), |c| _mm256_max_pd(_mm256_add_pd(a, b), c);
        max_plus32: f32, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b| _mm256_add_ps(a, b), |c| _mm256_max_ps(_mm256_add_ps(a, b), c);
        min_plus64: f64, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b| _mm256_add_pd(a, b), |c| _mm256_min_pd(_mm256_add_pd(a, b), c);
        min_plus32: f32, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b| _mm256_add_ps(a, b), |c| _mm256_min_ps(_mm256_add_ps(a, b), c);
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

/// The multiply-adds in each band of rows that `product` hands to a thread. On an M5 Max, 2^19 made 512×512 QR and inverses
/// 5 to 13% faster on 4 threads than 2^20 did. 2^18 made `+.×`, Cholesky and QR 10 to 40% slower on 18 threads than 2^19.
const BAND_WORK: usize = 1 << 19;

/// The multiply-adds that waking one more thread costs a product. A panel of `w` multiply-adds splits across at most √(w / `THREAD_WORK`)
/// threads, the count that minimises w / t + t × `THREAD_WORK`. On an M5 Max, `+.×` of two 256×256 matrices took 21.6 copies on 8
/// threads and 28.8 on 18, and larger products were fastest on all 18.
const THREAD_WORK: usize = 1 << 17;

/// The most items of `b` that `product` packs into one panel. It splits each panel's bands across threads in one call, however many
/// steps of `KC` the panel holds.
const PANEL: usize = 1 << 22;
/// The most bytes of `b` that `product` packs into one panel when this thread's calls run serially, as on a pool thread. Larger panels
/// outgrow the cache: on an M5 Max, a one-thread Gram product of a 16384×256 f64 matrix took 3% longer with 8 MB panels and 10% longer
/// with 32 MB ones, and a threaded tall QR whose threads each ran one such product was 2% slower with 4 MB panels than with 2 MB ones.
const SERIAL_PANEL: usize = 2 << 20;
/// Folds the inner product of `P` into `c`, an `m`×`n` matrix whose rows start `ldc` items apart, with `kernel` folding each `MR`×`NR`
/// tile as `micro` does. `a(i, p)` gives the items of the `m`×`k` matrix `a`, and `b` holds the `k`×`n` matrix `b`. With `start`, the
/// product replaces `c`, which must then hold `f`'s identity where `k` is 0. Otherwise each result folds into `c`'s item with `f`. Only
/// tiles that hold items of `part` fold. Items outside `part` in those tiles fold too. Each block of `NC`
/// columns packs `b` in panels of at most [`PANEL`] items, or [`SERIAL_PANEL`] bytes on one thread, across threads as [`parallel::split`] chooses. Then it runs bands of rows
/// across at most √(w / [`THREAD_WORK`]) of this thread's [`parallel::threads`], for a panel of `w` multiply-adds: about one band for each
/// [`BAND_WORK`] multiply-adds, up to four bands for each thread, each at most `MC` rows. Each band runs the panel's steps of `KC` in
/// turn, packing its own block of `a` for each.
#[inline(always)]
fn product<T: Copy + Default + Send + Sync, P: Pair<T>, const MR: usize, const NR: usize>(
    c: &mut [P::Out],
    ldc: usize,
    a: impl Fn(usize, usize) -> T + Sync,
    b: impl Fn(usize, usize, &mut [MaybeUninit<T>]) + Sync,
    [m, k, n]: [usize; 3],
    start: bool,
    part: Part,
    kernel: impl Fn(&[T], &[T], &mut [[P::Out; NR]; MR], bool) + Sync,
) {
    if m == 0 || k == 0 || n == 0 { return; }
    assert!(ldc >= n && c.len() >= (m - 1) * ldc + n, "`c` must hold `m` rows of `n` items, `ldc` apart");
    let c = &mut c[..(m - 1) * ldc + n];
    let width = NC.min(n).next_multiple_of(NR);
    let panel = if parallel::threads() == 1 { SERIAL_PANEL / size_of::<T>() } else { PANEL };
    let depth = (panel / (width * KC)).clamp(1, k.div_ceil(KC)) * KC;
    let mut bp = Vec::with_capacity(width * depth.min(k));
    for jc in (0..n).step_by(NC) {
        let nc = NC.min(n - jc);
        for pp in (0..k).step_by(depth) {
            let kp = depth.min(k - pp);
            let unit = NR * kp;
            let work = m * kp * nc;
            let threads = if work < 2 * BAND_WORK { 1 } else { parallel::threads().min((work / THREAD_WORK).isqrt()) };
            let band_rows = if threads == 1 { m } else { m.div_ceil((work / BAND_WORK).min(4 * threads)).next_multiple_of(MR).min(MC) };
            let pack_b = |at: usize, units: &mut [MaybeUninit<T>]| {
                for (u, strip) in units.chunks_exact_mut(unit).enumerate() {
                    let s = at / unit + u;
                    let (j, lines) = (jc + s * NR, NR.min(nc - s * NR));
                    for (step, part) in strip.chunks_mut(NR * KC).enumerate() {
                        let pc = pp + step * KC;
                        pack::<_, NR>(part, part.len() / NR, lines, |p, l, out| b(pc + p, j + l, out));
                    }
                }
                true
            };
            bp.clear();
            slice::appended(&mut bp, nc.div_ceil(NR) * unit, |slots| if threads == 1 { pack_b(0, slots) } else { parallel::split(slots, unit, &pack_b) });
            let (a, bp, kernel) = (&a, &bp[..], &kernel);
            let lower = matches!(part, Part::Lower);
            parallel::pieces_of(c, band_rows * ldc, threads, &move |first, band| {
                let (i0, i1) = (first / ldc, (first / ldc + band_rows).min(m));
                let mut slots = Box::<[T]>::new_uninit_slice(MC.min(i1 - i0).next_multiple_of(MR) * KC.min(kp));
                for step in 0..kp.div_ceil(KC) {
                    let pc = pp + step * KC;
                    let (kc, load) = (KC.min(k - pc), pc > 0 || !start);
                    for ic in (i0..i1).step_by(MC) {
                        let mc = MC.min(i1 - ic);
                        if lower && ic + mc <= jc { continue }
                        let ap = pack::<_, MR>(&mut slots, kc, mc, |p, i, out| for l in 0..out.len() { out[l] = MaybeUninit::new(a(ic + i + l, pc + p)) });
                        for jr in (0..nc).step_by(NR) {
                            let bj = &bp[jr / NR * unit + step * NR * KC..][..NR * kc];
                            for ir in (0..mc).step_by(MR) {
                                if lower && ic + ir + MR.min(mc - ir) <= jc + jr { continue }
                                let (rows, cols) = (MR.min(mc - ir), NR.min(nc - jr));
                                let at = |r: usize, s: usize| (ic - i0 + ir + r) * ldc + jc + jr + s;
                                let mut t = [[P::UNIT; NR]; MR];
                                if load { for r in 0..rows { for s in 0..cols { t[r][s] = band[at(r, s)] } } }
                                kernel(&ap[ir * kc..(ir + MR) * kc], bj, &mut t, !load);
                                for r in 0..rows { for s in 0..cols { band[at(r, s)] = t[r][s] } }
                            }
                        }
                    }
                }
                true
            });
        }
    }
}

/// Packs `len` lines of `kc` steps into strips of `W` lines at the start of `buffer`, and gives the items it wrote. Each strip holds its
/// `W` items for step 0, then those for step 1, and so on. `run(p, l, out)` writes `out.len()` items of step `p` into `out`, from line
/// `l` on. Lines past `len` hold the default, and the results computed from them are never stored.
fn pack<T: Copy + Default, const W: usize>(buffer: &mut [MaybeUninit<T>], kc: usize, len: usize, run: impl Fn(usize, usize, &mut [MaybeUninit<T>])) -> &[T] {
    let strips = len.div_ceil(W);
    for (s, strip) in buffer.chunks_exact_mut(W * kc).take(strips).enumerate() {
        let (first, lines) = (s * W, W.min(len - s * W));
        for (p, step) in strip.as_chunks_mut::<W>().0.iter_mut().enumerate() {
            if lines == W {
                run(p, first, step)
            } else {
                run(p, first, &mut step[..lines]);
                for slot in &mut step[lines..] { *slot = MaybeUninit::new(T::default()) }
            }
        }
    }
    slice::written(&mut buffer[..strips * W * kc])
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

