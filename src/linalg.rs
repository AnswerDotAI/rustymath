//! Inner products of row-major matrices. The inner product `f.g` of an `m`×`k` matrix `a` and a `k`×`n` matrix `b` is the `m`×`n` matrix whose item at row `i` and column `j` folds `f` over `g(a[i][p], b[p][j])` for each `p`. `+.×` is the matrix product. The specialized pairs have associative `f`, and their loops combine its steps in whatever order suits them. A float result can therefore differ in its last bits from a fold taken in order. [`inner_with`] runs other native pairs through bounded term blocks and a caller's right fold, which preserves the association of non-associative functions.
//!
//! The loops follow GotoBLAS and BLIS. Packing copies a block of each argument into a contiguous buffer sized for a cache level. The blocking reuses each buffer while it is in cache. A micro-kernel computes a small tile of the result in registers, one step of `p` at a time. A micro-kernel in plain Rust vectorizes through LLVM's SLP vectorizer, which the builds of rustymath and basedpl turn on at opt-level 2. It vectorizes only tiles of up to 32 items, too few to keep the multiply-add units busy. On AArch64, in wasm and on x86-64 with AVX2 and FMA, the float pairs run micro-kernels on explicit vectors instead. Their maximum and minimum are AArch64's `fmaxnm`, wasm's `pmax` and x86-64's `maxpd`, with the running value as the argument each gives beside NaN. A tile starts at `-∞` or `∞`, so its running value is never NaN and a NaN term leaves it unchanged, as `Core::maximum` does. Targets without these vectors run the plain kernels.
//!
//! A product large enough splits its rows across the threads that [`crate::parallel::set_threads`] sets. On macOS, Accelerate computes real and complex plus-times products instead. `Tiled` also sends the decompositions' triangular solves, SVDs and Hermitian eigen decompositions to it, and the general eigen decompositions of real matrices from `accelerate::GEEV_FROM` rows. macOS builds with the `native` feature also compile rustymath's own kernels, and `RUSTYMATH_NATIVE=1` selects them for the whole process.
//!
//! The private module `factor` holds decompositions of real and complex matrices, whose blocked updates run on these loops. This module re-exports its public items, and [`Scalar`], the trait of the item types they take. [`transpose_into`] transposes a matrix through 8×8 tiles.
use crate::{Float, agreement::{Frames, Mapping, expanded_items}, buffer::{Buffer, DType, Element, FloatWidth, Source, View, Width, read_all}, convert, error::Error, float::{FUSED, sealed::Core}, parallel, slice};
use num_complex::Complex;
use num_traits::{One, Zero};
use std::{mem::MaybeUninit, sync::LazyLock};

#[cfg(accelerate)]
mod accelerate;
mod factor;
mod inner;
mod scalar;
mod stacked;
// The platform's own BLAS and LAPACK, which every `Tiled` type implements: Accelerate on macOS, and none elsewhere.
#[cfg(accelerate)]
pub(crate) use accelerate::Accelerated as Vendor;
#[cfg(not(accelerate))]
pub(crate) trait Vendor {}
#[cfg(not(accelerate))]
impl<T> Vendor for T {}

/// Whether this process selects rustymath's own kernels where Accelerate is also compiled in: in macOS builds with the `native` feature, when `RUSTYMATH_NATIVE=1`. It's read once.
fn switched() -> bool {
    static NATIVE: LazyLock<bool> = LazyLock::new(|| std::env::var_os("RUSTYMATH_NATIVE").is_some_and(|v| v == "1"));
    cfg!(all(accelerate, native)) && *NATIVE
}

/// Whether Accelerate serves this process's products, triangular solves and decompositions.
pub(crate) fn accelerated() -> bool { cfg!(accelerate) && !switched() }

/// `$vendor` where Accelerate serves this process, and `$native` where rustymath's own kernels do. Each compiles only where its kernels do.
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
/// `$vendor` where Accelerate serves this process and the matrix's order `$n` is at least `$from`, and `$native` otherwise. Accelerate's LAPACK pays a fixed cost for each call, which makes the native solver faster for small matrices. Use it only in a function body: it returns `$vendor`'s result.
macro_rules! from_order {
    ($n:expr, $from:expr, $vendor:expr, $native:expr) => {{
        #[cfg(accelerate)]
        if $n >= $from && accelerated() { return $vendor }
        $native
    }};
}
pub mod exact;
pub use factor::*;
pub use inner::inner_with;
pub use scalar::Scalar;
pub use stacked::{cells, paired, stored, Stacked};

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

/// Writes the transpose of the `rows`×`cols` matrix in `src`, whose rows start `ls` items apart, into `dst`, whose rows start `ld` items apart. Each item passes through `f`: `dst[c * ld + r]` is `f(src[r * ls + c])`. Whole 8×8 tiles go through [`tiles`], and the items past the last whole tile one at a time.
#[inline]
pub fn transpose_into<S: Copy, D>(src: &[S], ls: usize, dst: &mut [D], ld: usize, [rows, cols]: [usize; 2], f: impl Fn(S) -> D) {
    if rows == 0 || cols == 0 { return; }
    assert!(src.len() >= (rows - 1) * ls + cols && dst.len() >= (cols - 1) * ld + rows, "both matrices lie in their slices");
    let (whole_rows, whole_cols) = (rows / 8 * 8, cols / 8 * 8);
    // Each write moves 16 bytes where the items allow, which is at most 8 items, as the registers that the compiler shuffles hold.
    match 16 / size_of::<S>().max(1) {
        8.. => tiles::<8, S, D>(src, ls, dst, ld, [whole_rows, whole_cols], &f),
        4.. => tiles::<4, S, D>(src, ls, dst, ld, [whole_rows, whole_cols], &f),
        2.. => tiles::<2, S, D>(src, ls, dst, ld, [whole_rows, whole_cols], &f),
        _ => tiles::<1, S, D>(src, ls, dst, ld, [whole_rows, whole_cols], &f),
    }
    if whole_cols < cols { for r in 0..whole_rows { for c in whole_cols..cols { dst[c * ld + r] = f(src[r * ls + c]) } } }
    for r in whole_rows..rows { for c in 0..cols { dst[c * ld + r] = f(src[r * ls + c]) } }
}

/// The whole 8×8 tiles of a transpose, the first `rows` rows and `cols` columns of each matrix, which its caller checked lie in their slices. Each tile reads `P` of its runs of 8 adjacent items at a time, and writes `P` adjacent items of each result run, one from each run read. The compiler shuffles the runs' lanes in registers, as `zip` instructions do for two runs of 8-byte items.
#[inline(always)]
fn tiles<const P: usize, S: Copy, D>(src: &[S], ls: usize, dst: &mut [D], ld: usize, [rows, cols]: [usize; 2], f: &impl Fn(S) -> D) {
    let (from, to) = (src.as_ptr(), dst.as_mut_ptr());
    for r0 in (0..rows).step_by(8) {
        for c0 in (0..cols).step_by(8) {
            for i in (0..8).step_by(P) {
                // SAFETY: the tile's rows and result rows lie in `src` and `dst`, as the caller checked.
                let read = |k: usize| unsafe { from.add((r0 + i + k) * ls + c0).cast::<[S; 8]>().read() };
                let mut runs = [read(0); P];
                for k in 1..P { runs[k] = read(k) }
                for j in 0..8 {
                    let out = unsafe { &mut *to.add((c0 + j) * ld + r0 + i).cast::<[D; P]>() };
                    for k in 0..P { out[k] = f(runs[k][j]) }
                }
            }
        }
    }
}

/// An inner product `f.g` at item type `T`. Each pair object implements it at the item types listed here. Its loops compile in rustymath, as slice forms do.
pub trait Inner<T> {
    /// The type of each result.
    type Out: Copy + Default;
    /// The `m`×`n` inner product of `a`, an `m`×`k` matrix, and `b`, a `k`×`n` matrix, both row-major, written row-major into `out`, which holds `m`×`n` items. `size` is `[m, k, n]`. Where `k` is 0, each result is the identity of `f`. An integer result that doesn't fit its type panics, where `checked_into` gives `false`.
    fn inner_into(&self, a: &[T], b: &[T], size: [usize; 3], out: &mut [Self::Out]);
    /// `inner_into`'s results, as a new row-major matrix.
    fn inner(&self, a: &[T], b: &[T], size @ [m, _, n]: [usize; 3]) -> Vec<Self::Out> {
        let mut c = filled(m * n, Self::Out::default());
        self.inner_into(a, b, size, &mut c);
        c
    }
    /// `inner_into`, or `false` where an integer result doesn't fit its type, for callers that keep integers exact or report overflow. Where it gives `true`, `out` holds the exact results. Floats always give `true`.
    #[inline]
    fn checked_into(&self, a: &[T], b: &[T], size: [usize; 3], out: &mut [Self::Out]) -> bool { self.inner_into(a, b, size, out); true }
    /// `inner` of the matrices that `a` and `b` give in the orientations given, with `size` `[m, k, n]` for those matrices. A transposed `a` holds a `k`×`m` matrix, row-major, and gives its `m`×`k` transpose. This default copies a transposed operand into its orientation. Float and complex `PlusTimes` read it in place.
    fn inner_ops(&self, a: &[T], b: &[T], [oa, ob]: [Orientation; 2], [m, k, n]: [usize; 3]) -> Vec<Self::Out> where T: Copy {
        fn oriented<T: Copy>(x: &[T], o: Orientation, [rows, cols]: [usize; 2]) -> std::borrow::Cow<'_, [T]> {
            match o { Orientation::Normal => std::borrow::Cow::Borrowed(x), Orientation::Transpose => (0..rows * cols).map(|ij| x[ij % cols * rows + ij / cols]).collect() }
        }
        self.inner(&oriented(a, oa, [m, k]), &oriented(b, ob, [k, n]), [m, k, n])
    }
    /// `inner_ops` of operands whose items come through mappings, as [`Mapped`] describes. This default reads each operand whole, then calls `inner_ops`. Float and complex `PlusTimes` read an operand that isn't held in order one block of its stored rows at a time and multiply each block as it's read. Only one block of that operand then exists at once.
    fn inner_mapped(&self, a: Mapped<'_, T>, b: Mapped<'_, T>, orientations: [Orientation; 2], size @ [m, k, n]: [usize; 3]) -> Vec<Self::Out> where T: Element {
        let (mut ab, mut bb) = (Vec::new(), Vec::new());
        self.inner_ops(a.read(0, m * k, &mut ab), b.read(0, k * n, &mut bb), orientations, size)
    }
}

/// `+.×`, the matrix product, at `i64`, `f32` and `f64`, and complex numbers with parts of either type. Native float steps use fused multiply-add where the target has one; Accelerate computes them on macOS. Nonfinite complex results need not match scalar arithmetic. Native complex products form each part from two real products of the parts, as scalar multiplication does. A small imaginary part beside a large real part keeps its own relative accuracy.
#[derive(Default)]
pub struct PlusTimes;
/// `⌈.+`, the max-plus product of longest-path problems, at `i64`, `f32` and `f64`. The module doc says how each target treats a NaN term.
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
/// `+.=`, at `i16`, `i32`, `i64`, `f32` and `f64`: the number of `p` where `a[i][p]` equals `b[p][j]`. The count has the width of the items, as `i16`, `i32` or `i64`, so an `i16` count needs `k` below 2^15 and an `i32` count below 2^31. Floats compare exactly, so NaN equals nothing.
#[derive(Default)]
pub struct PlusEqual;
/// `+.≠`, at `i16`, `i32`, `i64`, `f32` and `f64`: the number of `p` where `a[i][p]` differs from `b[p][j]`, with `+.=`'s count type.
#[derive(Default)]
pub struct PlusNotEqual;

/// One of the pairs above, for code that chooses a pair at run time, as a frontend does from its own names for functions. [`Product::run`] is where each pair finds its kernel for a dtype.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Product { PlusTimes, MaxPlus, MinPlus, OrAnd, AndOr, PlusEqual, PlusNotEqual }

impl Product {
    /// The pair's products of the cells of `x` and `y`, which share a dtype and pair as `frames` says, each of sizes `[n, k, m]`, with whether every product fits, as [`Kernels::product`](crate::operation::Kernels::product) gives them. `None` where the pair has no kernel for their dtype, and an error where an interrupt stopped them. A count computes at `i64` or at a float, and reads characters as their code points. An `i16` or `i32` count could overflow, and the `i64` kernel counts at those widths only where that's safe.
    pub fn run(self, xy: [View<'_>; 2], frames: &Frames, size: [usize; 3]) -> Option<Result<(Buffer, bool), Error>> {
        const COUNTS: [DType; 3] = [DType::Integer(Width::I64), DType::Float(FloatWidth::F32), DType::Float(FloatWidth::F64)];
        match self {
            Self::PlusTimes => crate::kernels!(Product PlusTimes).product(&PlusTimes, xy, frames, size),
            Self::MaxPlus => crate::kernels!(Product MaxPlus).product(&MaxPlus, xy, frames, size),
            Self::MinPlus => crate::kernels!(Product MinPlus).product(&MinPlus, xy, frames, size),
            Self::OrAnd => crate::kernels!(Product OrAnd).product(&OrAnd, xy, frames, size),
            Self::AndOr => crate::kernels!(Product AndOr).product(&AndOr, xy, frames, size),
            Self::PlusEqual => crate::kernels!(Product PlusEqual).only(&COUNTS).product(&PlusEqual, xy.map(crate::characters::as_codes), frames, size),
            Self::PlusNotEqual => crate::kernels!(Product PlusNotEqual).only(&COUNTS).product(&PlusNotEqual, xy.map(crate::characters::as_codes), frames, size),
        }
    }
    /// The products of [`run`](Self::run) with wrapping steps: `+.×` on `u8`, `i16` or `i32` items gives each result's exact sum of products wrapped to 32 bits, at `i32`, as static kinds and fixed widths keep integer results (`Wrapped<PlusTimes>`). `None` for any other pair or dtype, which `run` then takes.
    pub fn run_wrapped(self, xy: [View<'_>; 2], frames: &Frames, size: [usize; 3]) -> Option<Result<(Buffer, bool), Error>> {
        match self {
            Self::PlusTimes => crate::kernels!(Product crate::Wrapped<PlusTimes>).product(&crate::Wrapped(PlusTimes), xy, frames, size),
            _ => None,
        }
    }
    /// Whether the pair's kernel compares items exactly, as the counts `+.=` and `+.≠` do. A frontend sends approximate items that compare within a tolerance to another route.
    pub fn compares_exactly(self) -> bool { matches!(self, Self::PlusEqual | Self::PlusNotEqual) }
}

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
macro_rules! complex_pairs {
    ($($t:ty),+) => {$(
        impl Pair<Complex<$t>> for PlusTimes {
            type Out = Complex<$t>;
            const UNIT: Complex<$t> = Complex::new(0.0, 0.0);
            #[inline(always)]
            fn g(a: Complex<$t>, b: Complex<$t>) -> Complex<$t> { a * b }
            #[inline(always)]
            fn f(x: Complex<$t>, y: Complex<$t>) -> Complex<$t> { x + y }
            #[inline(always)]
            fn step(acc: Complex<$t>, a: Complex<$t>, b: Complex<$t>) -> Complex<$t> { a.mul_acc(b, acc) }
        }
    )+};
}
complex_pairs!(f32, f64);

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
    #[inline]
    fn inner_into(&self, a: &[bool], b: &[bool], size: [usize; 3], out: &mut [bool]) {
        if size[0] * size[1] * size[2] <= DIRECT_WORK { direct::<bool, bool, OrAnd>(a, b, size, out) } else { packed::<false>(a, b, size, out) }
    }
}

impl Inner<bool> for AndOr {
    type Out = bool;
    #[inline]
    fn inner_into(&self, a: &[bool], b: &[bool], size: [usize; 3], out: &mut [bool]) {
        if size[0] * size[1] * size[2] <= DIRECT_WORK { direct::<bool, bool, AndOr>(a, b, size, out) } else { packed::<true>(a, b, size, out) }
    }
}

/// `∨.∧` of `a` and `b`, or with `NOT` set, `∧.∨`, which is `~(~a)∨.∧(~b)`, written into `c`. `NOT` flips each item of `a` and `b` as it is read, and each result as it is written. Each row of the result ORs together the rows of `b` where the row of `a` is true. The rows of `b` are packed 64 items to a word, in blocks of `WORDS` words, and each block of a row's results folds in a local array that stays in registers. An item of `a` becomes a mask of all ones or none, so the loop over it has no branch.
fn packed<const NOT: bool>(a: &[bool], b: &[bool], [m, k, n]: [usize; 3], c: &mut [bool]) {
    const WORDS: usize = 4;
    assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `c` `m`×`n`");
    if k == 0 || n == 0 { return c.fill(NOT); }
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
}
/// The pairs at `i64` and `+.×`'s pair at `i32`. `tiered` runs the pairs at `i64` only where every value that a product forms fits `i64`. Their wrapping operations then never wrap. The pair at `i32` wraps, as the products of [`Wrapped<PlusTimes>`] do.
macro_rules! wrapping_times {
    ($($t:ty),+) => {$(
        impl Pair<$t> for PlusTimes {
            type Out = $t;
            const UNIT: $t = 0;
            #[inline(always)]
            fn g(a: $t, b: $t) -> $t { a.wrapping_mul(b) }
            #[inline(always)]
            fn f(x: $t, y: $t) -> $t { x.wrapping_add(y) }
        }
    )+};
}
wrapping_times!(i64, i32);
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
/// The pairs at `i128`, which `checked_into` uses where a result could pass `i64`. Each step of `+.×` checks that its running total fits 128 bits.
impl Pair<i128> for PlusTimes {
    type Out = Option<i128>;
    const UNIT: Option<i128> = Some(0);
    fn g(a: i128, b: i128) -> Option<i128> { a.checked_mul(b) }
    fn f(x: Option<i128>, y: Option<i128>) -> Option<i128> { x?.checked_add(y?) }
}
impl Pair<i128> for MaxPlus {
    type Out = Option<i128>;
    const UNIT: Option<i128> = Some(i128::MIN);
    fn g(a: i128, b: i128) -> Option<i128> { a.checked_add(b) }
    fn f(x: Option<i128>, y: Option<i128>) -> Option<i128> { Some(x?.max(y?)) }
}
impl Pair<i128> for MinPlus {
    type Out = Option<i128>;
    const UNIT: Option<i128> = Some(i128::MAX);
    fn g(a: i128, b: i128) -> Option<i128> { a.checked_add(b) }
    fn f(x: Option<i128>, y: Option<i128>) -> Option<i128> { Some(x?.min(y?)) }
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

/// The tile shapes, `[MR, NR]`, and the bytes in a vector. A tile holds the rows and columns of the result that a micro-kernel keeps in registers. AArch64 has 32 vector registers of 16 bytes, and AVX2 has 16 registers of 32 bytes. wasm has 16-byte vectors, and V8 gives it 16 registers on x86-64. The extremes and the counts compare at each step, and on AVX2 they need registers for those comparisons, so their tiles have fewer rows. `TIMES32` serves both `f32` and `i32` products, whose items take 4 bytes. A plain kernel's tile holds at most 32 items. On the Mac, plain tiles of 48 or more floats compiled to scalar code and ran 4 to 5 times slower.
#[cfg(target_arch = "aarch64")]
mod tiles {
    pub(super) const BYTES: usize = 16;
    #[cfg(native)]
    pub(super) const TIMES64: [usize; 2] = [8, 4];
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
            fn inner_into(&self, a: &[$t], b: &[$t], size: [usize; 3], out: &mut [Self::Out]) {
                const MR: usize = $tile[0];
                const NR: usize = $tile[1];
                tiled_into::<$t, $t, $object, MR, NR>(a, b, size, out, $kernel::<MR, NR, { NR * size_of::<$t>() / tiles::BYTES }>)
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
/// `P`'s inner product of `a` and `b`, as `Inner::inner_into` gives it, written into `c`, which holds `m`×`n` items, through `kernel` on tiles of `MR`×`NR`. Each item reads as the kernel's type `T` through [`Source`] as it's packed, so an operand of narrower items is never copied whole.
#[inline(always)]
fn tiled_into<S: Source<T> + Copy + Sync, T: Copy + Default + Send + Sync, P: Pair<T>, const MR: usize, const NR: usize>(
    a: &[S], b: &[S], [m, k, n]: [usize; 3], c: &mut [P::Out], kernel: impl Fn(&[T], &[T], &mut [[P::Out; NR]; MR], bool) + Sync) {
    assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `c` `m`×`n`");
    if m * k * n <= DIRECT_WORK { return direct::<S, T, P>(a, b, [m, k, n], c) }
    let f = <S as Source<T>>::read;
    product::<T, P, MR, NR>(
        c, n, |rows, p0, out| strip(a, k, false, rows, p0, out, f), |columns, p0, out| strip(b, n, true, columns, p0, out, f), [m, k, n], true,
        Part::Whole, kernel);
}

/// The most steps, `m`×`k`×`n`, of a product that [`direct`] computes, where packing the operands for the tiled kernels, or calling Accelerate, costs more than the steps. Measured on 22 October 2026 on an M5 Max, a stack of 4,096 products of 4×4 matrices cost 9.3 copies this way, against 25 on Accelerate and 38 on the packed kernels. A stack of 1,024 products of 8×8 matrices cost 10.1 copies this way, against 10.3 on Accelerate and 9.7 on the packed kernels, so the threshold stays below them.
const DIRECT_WORK: usize = 64;

/// `P`'s inner product of `a` and `b`, written into `c`, which holds `m`×`n` items, by a plain loop. Each item reads as `T` through [`Source`]. Each result folds its row of `a` with its column of `b` in order, in a register, as the tiled kernels do on every target, NaN included. Four results of a row fold side by side, which vectorises, and the row's last columns fold one at a time. Nothing is packed or allocated.
// A stack of tiny products calls this once for each cell. Once the loop vectorizer ran before thin LTO's import step, this body grew too large to import into `product_at`, and the call cost a stack of 64 2×2 Boolean products about 600 instructions.
#[inline]
fn direct<S: Source<T> + Copy, T: Copy, P: Pair<T>>(a: &[S], b: &[S], [m, k, n]: [usize; 3], c: &mut [P::Out]) {
    let f = <S as Source<T>>::read;
    const SIDE: usize = 4;
    assert!(a.len() >= m * k && b.len() >= k * n && c.len() >= m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `c` `m`×`n`");
    // SAFETY: every index below is `i * k + p`, `p * n + j` or `i * n + j` with `i < m`, `p < k` and `j < n`, which the assert bounds. Checking each one cost about half of `direct`'s instructions on a 4×4 product.
    for i in 0..m {
        let mut j = 0;
        while j + SIDE <= n {
            let mut acc = [P::UNIT; SIDE];
            for p in 0..k {
                let x = f(unsafe { *a.get_unchecked(i * k + p) });
                for s in 0..SIDE { acc[s] = P::step(acc[s], x, f(unsafe { *b.get_unchecked(p * n + j + s) })) }
            }
            for s in 0..SIDE { unsafe { *c.get_unchecked_mut(i * n + j + s) = acc[s] } }
            j += SIDE;
        }
        for j in j..n {
            let mut acc = P::UNIT;
            for p in 0..k { acc = P::step(acc, f(unsafe { *a.get_unchecked(i * k + p) }), f(unsafe { *b.get_unchecked(p * n + j) })) }
            unsafe { *c.get_unchecked_mut(i * n + j) = acc }
        }
    }
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

/// `+.=` at `i64` counts at `i16` where `windowed` maps both arguments into it, and otherwise at `i32` where every item fits in 32 bits. A narrower count puts more items in each vector.
impl Inner<i64> for PlusEqual {
    type Out = i64;
    fn inner_into(&self, a: &[i64], b: &[i64], size: [usize; 3], out: &mut [i64]) {
        if let Some(c) = windowed(a, b, size) { return out.iter_mut().zip(&c).for_each(|(o, &x)| *o = i64::from(x)); }
        let narrow = |x: &[i64]| x.iter().map(|&v| i32::try_from(v).ok()).collect::<Option<Vec<_>>>();
        if let (Some(a), Some(b)) = (narrow(a), narrow(b)) { return out.iter_mut().zip(&self.inner(&a, &b, size)).for_each(|(o, &x)| *o = i64::from(x)); }
        const MR: usize = tiles::COMPARES64[0];
        const NR: usize = tiles::COMPARES64[1];
        tiled_into::<i64, i64, PlusEqual, MR, NR>(a, b, size, out, equal_i64::<MR, NR, { NR * size_of::<i64>() / tiles::BYTES }>)
    }
}

/// `+.=` of `a` and `b` counted at `i16`, where `k` is below 2^15 and every item of both lies in one window of 2^16 values. Each item becomes its distance from the window's start, read as an `i16`, so two items are equal exactly where they were before.
fn windowed(a: &[i64], b: &[i64], size: [usize; 3]) -> Option<Vec<i16>> {
    if size[1] > i16::MAX as usize { return None; }
    let ((la, ha), (lb, hb)) = (crate::reduce::extremes(a), crate::reduce::extremes(b));
    let (lo, hi) = (la.min(lb), ha.max(hb));
    if i128::from(hi) - i128::from(lo) > i128::from(u16::MAX) { return None; }
    let shifted = |x: &[i64]| x.iter().map(|&v| v.wrapping_sub(lo) as u16 as i16).collect::<Vec<_>>();
    Some(PlusEqual.inner(&shifted(a), &shifted(b), size))
}

/// `+.≠` as `k` less `+.=`.
macro_rules! differences {
    ($($t:ty => $out:ty),+) => {$(
        impl Inner<$t> for PlusNotEqual {
            type Out = $out;
            fn inner_into(&self, a: &[$t], b: &[$t], size: [usize; 3], out: &mut [$out]) {
                PlusEqual.inner_into(a, b, size, out);
                let k = size[1] as $out;
                for x in out { *x = k - *x }
            }
        }
    )+};
}
differences!(i64 => i64, i32 => i32, i16 => i16, f64 => i64, f32 => i32);

/// The largest magnitude among the items of `x`.
fn magnitude<T: Copy>(x: &[T]) -> u128 where i64: From<T> { u128::from(x.iter().fold(0, |m, &v| m.max(i64::from(v).unsigned_abs()))) }

/// The integer inner products of the float pairs. Each computes at the narrowest item type that holds every value the product forms exactly: `f32` where none exceeds 2^24 in magnitude, `f64` where all lie below 2^51, `i64` where all fit it, and otherwise 128 bits through `wide`. `$bound` gives the largest magnitude that a product can form, from the largest magnitudes in `a` and `b` and from `k`. `checked_into` gives `false` where a result doesn't fit `i64`, and `inner` and `inner_into` panic there.
macro_rules! integers {
    ($($object:ident: $bound:expr;)+) => {$(
        impl Inner<i64> for $object {
            type Out = i64;
            fn inner_into(&self, a: &[i64], b: &[i64], size: [usize; 3], out: &mut [i64]) {
                assert!(self.checked_into(a, b, size, out), "an integer inner product fits i64");
            }
            fn checked_into(&self, a: &[i64], b: &[i64], size: [usize; 3], out: &mut [i64]) -> bool {
                let bound = $bound(magnitude(a), magnitude(b), size[1] as u128);
                if bound <= i64::MAX as u128 { tiered(self, a, b, size, bound, out); return true; }
                wide::<Self>(a, b, size).map(|c| out.copy_from_slice(&c)).is_some()
            }
        }
    )+};
}
integers! {
    PlusTimes: |a: u128, b: u128, k: u128| a.saturating_mul(b).saturating_mul(k);
    MaxPlus: |a: u128, b: u128, _| a + b;
    MinPlus: |a: u128, b: u128, _| a + b;
}

/// `P`'s inner product of `a` and `b`, computed in 128 bits through `P`'s [`Pair`] at `i128`. `None` where a result doesn't fit `i64`, or where a running total doesn't fit 128 bits.
fn wide<P: Pair<i128, Out = Option<i128>>>(a: &[i64], b: &[i64], [m, k, n]: [usize; 3]) -> Option<Vec<i64>> {
    (0..m * n).map(|c| {
        let (i, j) = (c / n, c % n);
        let total = (0..k).fold(P::UNIT, |t, p| P::step(t, a[i * k + p].into(), b[p * n + j].into()))?;
        i64::try_from(total).ok()
    }).collect()
}

/// `integers!`'s inner product of `pair`, whose values never exceed `bound` in magnitude, written into `out`.
fn tiered<P>(pair: &P, a: &[i64], b: &[i64], [m, k, n]: [usize; 3], bound: u128, out: &mut [i64]) where
    P: Inner<f32, Out = f32> + Inner<f64, Out = f64> + Pair<i64, Out = i64> {
    assert!(a.len() == m * k && b.len() == k * n, "`a` must hold `m`×`k` items and `b` `k`×`n`");
    if k == 0 { return out.fill(<P as Pair<i64>>::UNIT); }
    if exactly(pair, a, b, [m, k, n], bound, true, out, |x| x) { return; }
    tiled_into::<i64, i64, P, { PLAIN[0] }, { PLAIN[1] }>(a, b, [m, k, n], out, micro::<i64, P, { PLAIN[0] }, { PLAIN[1] }>)
}

/// `pair`'s inner product of `a` and `b`, whose values never exceed `bound` in magnitude, in a float that holds each of them exactly: `f32` up to 2^24, and `f64` up to `WIDE_LIMIT` where `wide` holds. Each item reads through [`Source`], and each result, a whole number, converts from `i64` to the type of `out` through `whole`. `false`, with `out` untouched, where neither float holds every value.
fn exactly<P, S, O>(pair: &P, a: &[S], b: &[S], size: [usize; 3], bound: u128, wide: bool, out: &mut [O], whole: impl Fn(i64) -> O) -> bool where
    P: Inner<f32, Out = f32> + Inner<f64, Out = f64>, S: Source<f32> + Source<f64> {
    fn at<F: Float + Element, P: Inner<F, Out = F>, S: Source<F>, O>(pair: &P, a: &[S], b: &[S], size: [usize; 3], out: &mut [O], whole: impl Fn(i64) -> O) -> bool {
        for (o, x) in out.iter_mut().zip(pair.inner(&read_all(a), &read_all(b), size)) { *o = whole(convert::shifted(x.wide())) }
        true
    }
    if bound <= 1 << 24 { at::<f32, P, S, O>(pair, a, b, size, out, whole) } else { wide && bound <= convert::WIDE_LIMIT as u128 && at::<f64, P, S, O>(pair, a, b, size, out, whole) }
}

/// `+.×` of `a` and `b` with wrapping `i32` steps, written into `out`, on the `i32` kernel. The kernel converts each item to `i32` through [`Source`] as it packs it.
fn times_i32_into<S: Source<i32>>(a: &[S], b: &[S], size: [usize; 3], out: &mut [i32]) {
    const MR: usize = tiles::TIMES32[0];
    const NR: usize = tiles::TIMES32[1];
    tiled_into::<S, i32, PlusTimes, MR, NR>(a, b, size, out, times_i32::<MR, NR, { NR * size_of::<i32>() / tiles::BYTES }>)
}

/// [`Wrapped<PlusTimes>`] on `u8`, `i16` and `i32` items. Each result is its exact sum of products wrapped to 32 bits, as static kinds and fixed widths keep integer results. Nothing fails. [`Product::run_wrapped`] runs these products. `$largest` gives the largest magnitude that a product of two items can form: from the item type for `u8` and `i16`, and from the items' largest magnitudes for `i32`. Where `k` such products sum exactly in `f32`, the product computes in `f32` and wraps each whole result, as `integers!`'s products do. Where Accelerate serves the product, a sum that `f64` holds exactly computes in `f64` the same way. Every other product runs on the `i32` kernel, whose wrapping steps are exact to 32 bits whatever the items. On an M5 Max on 25 October 2026, against the `i32` kernel's time, a product of two 512×512 `u8` matrices took a fifth through Accelerate's `f64` product and 1.6 times as long through rustymath's own. A 512×256 by 256×512 product took 0.85 of the `i32` kernel's time through rustymath's own `f32` product.
macro_rules! wrapped_narrow {
    ($($t:ty: $largest:expr;)+) => {$(
        impl Inner<$t> for crate::Wrapped<PlusTimes> {
            type Out = i32;
            fn inner_into(&self, a: &[$t], b: &[$t], size @ [m, k, n]: [usize; 3], out: &mut [i32]) {
                assert!(a.len() == m * k && b.len() == k * n && out.len() == m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `out` `m`×`n`");
                if m * k * n > DIRECT_WORK && exactly(&PlusTimes, a, b, size, $largest(a, b) * k as u128, accelerated(), out, |x| x as i32) { return }
                times_i32_into(a, b, size, out)
            }
        }
    )+};
}
wrapped_narrow! {
    u8: |_, _| 255 * 255;
    i16: |_, _| 1 << 30;
    i32: |a, b| magnitude(a) * magnitude(b);
}

/// `+.×` of f16 matrices, summed and returned at f32. rustymath's own kernels convert each item to f32 as they pack it, so neither operand is copied whole. Accelerate has no f16 product, so where it serves the process, both operands convert to f32 copies for its f32 product.
impl Inner<half::f16> for PlusTimes {
    type Out = f32;
    fn inner_into(&self, a: &[half::f16], b: &[half::f16], size @ [m, k, n]: [usize; 3], c: &mut [f32]) {
        assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `c` `m`×`n`");
        if m * k * n <= DIRECT_WORK { return direct::<half::f16, f32, PlusTimes>(a, b, size, c) }
        dispatch!(half_products_accelerated(a, b, size, c), half_products_native(a, b, size, c))
    }
}

/// [`PlusTimes`] at f16 on Accelerate's f32 product, through f32 copies of both operands.
#[cfg(accelerate)]
fn half_products_accelerated(a: &[half::f16], b: &[half::f16], size @ [_, k, n]: [usize; 3], c: &mut [f32]) {
    f32::gemm(c, n, Matrix::rows(&read_all(a), k), Matrix::rows(&read_all(b), n), size, 1.0, 0.0)
}

/// [`PlusTimes`] at f16 on rustymath's own f32 kernels, which convert each item as they pack it.
#[cfg(native)]
fn half_products_native(a: &[half::f16], b: &[half::f16], size: [usize; 3], c: &mut [f32]) {
    const MR: usize = tiles::TIMES32[0];
    const NR: usize = tiles::TIMES32[1];
    tiled_into::<half::f16, f32, PlusTimes, MR, NR>(a, b, size, c, times32::<MR, NR, { NR * size_of::<f32>() / tiles::BYTES }>)
}

/// A strided matrix borrowed in its stored, transposed, conjugated or adjoint orientation.
#[derive(Clone, Copy)]
pub(crate) struct Matrix<'a, T> { pub data: &'a [T], pub ld: usize, pub op: Op }
#[derive(Clone, Copy)]
pub(crate) enum Op { Normal, Transpose, Adjoint, Conjugate }
/// Whether an operand of [`Inner::inner_ops`] holds its matrix as stored or transposed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Orientation { Normal, Transpose }
/// A row-major matrix whose items `mapping` reads from `data`, as [`expanded_items`] lays an argument over a frame. [`Inner::inner_mapped`] takes its operands this way. A matrix held in order has `Mapping::Linear(1)`.
#[derive(Clone, Copy)]
pub struct Mapped<'a, T> { pub data: &'a [T], pub mapping: &'a Mapping }

impl<'a, T: Element> Mapped<'a, T> {
    /// The matrix in orientation `o`, with no copy, where the mapping reads its stored matrix at positive strides, one of them 1: row by row, or column by column as a transpose. `None` for any other mapping.
    fn matrix(&self, o: Orientation, [rows, cols]: [usize; 2]) -> Option<Matrix<'a, T>> {
        let [r, c] = if o == Orientation::Normal { [rows, cols] } else { [cols, rows] };
        let (first, strides) = self.mapping.affine(&[r, c])?;
        // An axis of length 1 reads one item, so any stride serves it. A single row whose columns are strided reads as a transposed column.
        let sc = if c == 1 { 1 } else { strides[1] };
        let sr = if r > 1 { strides[0] } else if sc == 1 { c as isize } else { 1 };
        let data = self.data.get(first..)?;
        let stored = match (sr, sc) {
            (s, 1) if s >= c as isize => Matrix::rows(data, s as usize),
            (1, s) if s >= r as isize => Matrix { data, ld: s as usize, op: Op::Transpose },
            _ => return None,
        };
        Some(if o == Orientation::Normal { stored } else { stored.transpose() })
    }
    /// Whether [`Inner::inner_mapped`] reads this matrix where it lies, in orientation `o` with `size` rows and columns, rather than one block at a time.
    pub fn in_place(&self, o: Orientation, size: [usize; 2]) -> bool { self.matrix(o, size).is_some() }
    /// Items `start..start + n` of the matrix, as [`expanded_items`] reads them.
    fn read<'b>(&'b self, start: usize, n: usize, buf: &'b mut Vec<T>) -> &'b [T] { expanded_items(self.mapping, self.data, start, n, buf) }
}
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
        Self { op: match self.op {
            Op::Normal => Op::Transpose, Op::Transpose => Op::Normal, Op::Adjoint => Op::Conjugate, Op::Conjugate => Op::Adjoint
        }, ..self }
    }
    /// Starts at a row and column in the logical orientation.
    pub fn offset(self, row: usize, col: usize) -> Self {
        let at = if self.op.transposed() { col * self.ld + row } else { row * self.ld + col };
        Self { data: &self.data[at..], ..self }
    }
    /// The `rows`×`cols` matrix that an operand of [`Inner::inner_ops`] gives in orientation `o`. A transposed operand stores the `cols`×`rows` matrix, row-major.
    pub fn oriented(data: &'a [T], o: Orientation, [rows, cols]: [usize; 2]) -> Self {
        match o { Orientation::Normal => Self::rows(data, cols), Orientation::Transpose => Self { data, ld: rows, op: Op::Transpose } }
    }
    /// Checks the stored extent once, before any foreign call.
    fn check(self, mut rows: usize, mut cols: usize) {
        if self.op.transposed() { std::mem::swap(&mut rows, &mut cols) }
        if rows == 0 || cols == 0 { return; }
        let end = (rows - 1).checked_mul(self.ld).and_then(|n| n.checked_add(cols));
        assert!(self.ld >= cols && end.is_some_and(|n| n <= self.data.len()), "matrix view must hold its rows and columns");
    }
}
impl<T: Scalar> Matrix<'_, T> {
    pub fn conjugate(self) -> Self {
        // Conjugation is an identity for real scalars; don't make their BLAS views more restrictive.
        if !T::COMPLEX { return self; }
        Self { op: match self.op {
            Op::Normal => Op::Conjugate, Op::Transpose => Op::Adjoint, Op::Adjoint => Op::Transpose, Op::Conjugate => Op::Normal
        }, ..self }
    }
    pub fn adjoint(self) -> Self { self.transpose().conjugate() }
    #[cfg(native)]
    #[inline(always)]
    pub fn at(self, i: usize, j: usize) -> T {
        let x = self.data[if self.op.transposed() { j * self.ld + i } else { i * self.ld + j }];
        if matches!(self.op, Op::Adjoint | Op::Conjugate) { x.conj() } else { x }
    }
    /// Writes a packing strip of this view, times `scale`, as the free function `strip` does: item `[p][l]` is item `(r0 + l, p0 + p)`. A product packs strips of `a` from `a`, and of `b` from `b`'s transpose.
    #[cfg(native)]
    #[inline(always)]
    fn strip<const W: usize>(self, rows: [usize; 2], p0: usize, scale: T, out: &mut [[MaybeUninit<T>; W]]) where T: Default {
        let (transposed, one) = (self.op.transposed(), scale == T::one());
        if !T::COMPLEX || !matches!(self.op, Op::Adjoint | Op::Conjugate) {
            if one { strip(self.data, self.ld, transposed, rows, p0, out, |x| x) } else { strip(self.data, self.ld, transposed, rows, p0, out, |x| scale * x) }
        } else if one { strip(self.data, self.ld, transposed, rows, p0, out, |x: T| x.conj()) }
        else { strip(self.data, self.ld, transposed, rows, p0, out, |x: T| scale * x.conj()) }
    }
}
/// Writes all `W` items of each step of a packing strip, through `f`. Item `[p][l]` is item `(r0 + l, p0 + p)` of the matrix that `data` holds at `r * ld + p`, or at `p * ld + r` where `transposed`, for `l` below `lines`, and the default after that. Transposed data gives each step from one run, and other data from `lines` rows read side by side. Every step writes a fixed `W` items, which keeps short strips from calling `memcpy` or `memset` at each step.
#[inline(always)]
fn strip<S: Copy, T: Default, const W: usize>(
    data: &[S], ld: usize, transposed: bool, [r0, lines]: [usize; 2], p0: usize, out: &mut [[MaybeUninit<T>; W]], f: impl Fn(S) -> T) {
    assert!(lines <= W);
    let kc = out.len();
    let rows: [&[S]; W] = std::array::from_fn(|l| if transposed || l >= lines { &[] } else { &data[(r0 + l) * ld + p0..][..kc] });
    match (transposed, lines == W) {
        (true, true) => {
            for (p, step) in out.iter_mut().enumerate() {
                let run: &[S; W] = data[(p0 + p) * ld + r0..].first_chunk().expect("a whole step lies in `data`");
                for l in 0..W { step[l] = MaybeUninit::new(f(run[l])) }
            }
        }
        (true, false) => {
            for (p, step) in out.iter_mut().enumerate() {
                let run = &data[(p0 + p) * ld + r0..][..lines];
                for l in 0..W { step[l] = MaybeUninit::new(if l < lines { f(run[l]) } else { T::default() }) }
            }
        }
        (false, true) => { for (p, step) in out.iter_mut().enumerate() { for l in 0..W { step[l] = MaybeUninit::new(f(rows[l][p])) } } }
        (false, false) => {
            for (p, step) in out.iter_mut().enumerate() { for l in 0..W { step[l] = MaybeUninit::new(if l < lines { f(rows[l][p]) } else { T::default() }) } }
        }
    }
}
/// Item types whose matrix products run on the tiled `+.×` kernels: `f32` and `f64`, and complex numbers with parts of either type. The decompositions in `factor` build on them.
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
        if m == 0 || n == 0 { return; }
        dispatch!(accelerate::triangular(x, ldx, a, [m, n], side, triangle), factor::triangular_native(x, ldx, a, [m, n], side, triangle))
    }
    /// `alpha a b` on rustymath's own packed kernels, replacing `c` with `start` and added to it otherwise. Complex items run four real products through `complex_product`.
    #[cfg(native)]
    fn native_product(c: &mut [Self], ldc: usize, a: Matrix<'_, Self>, alpha: Self, b: Matrix<'_, Self>, size: [usize; 3], start: bool, part: Part);
    /// The thin SVD of the finite, nonempty `m`×`n` matrix `a` as `(U, s, V)`, with its singular vectors only where `vectors` is true, or `None` where it doesn't converge.
    fn svd(a: &[Self], size: [usize; 2], vectors: bool) -> Option<(Vec<Self>, Vec<Self::Real>, Vec<Self>)>;
    /// The eigen decomposition of the nonempty Hermitian `n`×`n` matrix whose finite lower triangle `a` holds, with its eigenvectors only where `vectors` is true, or `None` where it doesn't converge.
    fn eigh(a: &[Self], n: usize, vectors: bool) -> Option<Eigh<Self>>;
    /// The eigen decomposition of the finite, nonempty `n`×`n` matrix `a`, with its eigenvectors only where `vectors` is true, or `None` where it doesn't converge.
    fn eig(a: &[Self], n: usize, vectors: bool) -> Option<Eig<Self::Real>>;
}

fn gemm<T: Tiled>(c: &mut [T], ldc: usize, a: Matrix<'_, T>, b: Matrix<'_, T>, [m, k, n]: [usize; 3], alpha: T, beta: T, part: Part) {
    a.check(m, k);
    b.check(k, n);
    Matrix::rows(c, ldc).check(m, n);
    if m == 0 || n == 0 { return; }
    if k == 0 || alpha == T::zero() { return scale(c, ldc, [m, n], beta, part); }
    dispatch!(accelerate::product(c, ldc, a, b, [m, k, n], alpha, beta, part), native_gemm(c, ldc, a, b, [m, k, n], alpha, beta, part))
}

/// How many items [`blocked`] reads at a time from an operand that isn't held in order, rounded to whole stored rows. A block this size stays in cache while its product runs.
const BLOCK_ITEMS: usize = 1 << 18;

/// Calls `f` on each block of an operand's stored rows, `[rows, cols]` as stored, with the block's first row, its number of rows and its items. A block holds about [`BLOCK_ITEMS`] items, in whole rows.
fn blocks<T: Element>(op: Mapped<'_, T>, [rows, cols]: [usize; 2], mut f: impl FnMut(usize, usize, &[T])) {
    let (step, mut block) = ((BLOCK_ITEMS / cols).max(1), Vec::new());
    for first in (0..rows).step_by(step) {
        let len = step.min(rows - first);
        f(first, len, op.read(first * cols, len * cols, &mut block));
    }
}

/// The most rows of the result in one piece of [`by_columns`]'s split across threads.
const COLUMN_ROWS: usize = 4096;
/// The results that [`scaled_runs`] keeps in registers while every column adds into them.
const COLUMN_STRIP: usize = 32;
/// The fewest adjacent rows of `a` that [`by_columns`] reads at once. Shorter runs cost more in loop overhead than gathering saves.
const COLUMN_RUN: usize = 64;

/// `+.×` of `a` and a vector `b`, where each row of `a` starts one item after the row before it in storage, as the windows of a vector or a single-channel image do. Each column of `a` is then a run of storage, which [`scaled_runs`] reads where it lies and scales by its item of `b`. `a` is never gathered. `None` where `a`'s mapping chose a fill, where it doesn't split into entries for its rows and entries for its columns, or where its rows aren't adjacent in runs of at least [`COLUMN_RUN`]. The loop reads a fill as 0, the item type's own fill.
fn by_columns<T: Tiled + Element>(a: Mapped<'_, T>, b: Mapped<'_, T>, oa: Orientation, [m, k]: [usize; 2]) -> Option<Vec<T>> {
    use crate::gather::Steps;
    let (Mapping::Steps { base, steps }, None) = (a.mapping, a.mapping.chosen()) else { return None };
    if m * k == 0 { return None; }
    // The entries of the stored matrix's leading axis come first, and give its first `lead` items.
    let lead = if oa == Orientation::Normal { m } else { k };
    let (mut split, mut size) = (0, 1);
    while size < lead && split < steps.len() { size *= steps[split].len(); split += 1; }
    if size != lead { return None; }
    let (outer, inner) = steps.split_at(split);
    let (rows, cols) = if oa == Orientation::Normal { (outer, inner) } else { (inner, outer) };
    let Steps::Stride { step: 1, len: width, .. } = *rows.iter().rev().find(|s| s.len() > 1)? else { return None };
    if width < COLUMN_RUN { return None; }
    let mut whole = Vec::new();
    let weights = b.read(0, k, &mut whole);
    let columns: Vec<(T, usize)> = (0..k).filter_map(|p| Some((weights[p], Steps::reach(0, cols, p)?))).collect();
    let mut c = filled(m, T::zero());
    parallel::split(&mut c, COLUMN_ROWS, &|first, part| {
        let mut r = first;
        while r < first + part.len() {
            let len = (width - r % width).min(COLUMN_ROWS).min(first + part.len() - r);
            // A row whose outer entries give a fill reads only fills, and its result stays 0.
            if let Some(at) = Steps::reach(*base, rows, r) { scaled_runs(&mut part[r - first..r - first + len], a.data, at, &columns) }
            r += len;
        }
        true
    });
    Some(c)
}

/// Adds into `out`'s results, for adjacent rows of a matrix whose first row starts at `data[at]` and each next row one item later, the item at each row's start plus each column's offset, times the column's weight, for each of `columns`. A strip of [`COLUMN_STRIP`] results stays in registers while every column adds into it.
fn scaled_runs<T: Tiled>(out: &mut [T], data: &[T], at: usize, columns: &[(T, usize)]) {
    let (strips, tail) = out.as_chunks_mut::<COLUMN_STRIP>();
    for (s, strip) in strips.iter_mut().enumerate() {
        let (at, mut acc) = (at + s * COLUMN_STRIP, *strip);
        for &(w, o) in columns {
            let x = &data[at + o..][..COLUMN_STRIP];
            for t in 0..COLUMN_STRIP { acc[t] = w.mul_acc(x[t], acc[t]) }
        }
        *strip = acc;
    }
    let at = at + strips.len() * COLUMN_STRIP;
    for (t, y) in tail.iter_mut().enumerate() { *y = columns.iter().fold(*y, |acc, &(w, o)| w.mul_acc(data[at + o + t], acc)) }
}

/// Writes `out`'s results for stored rows `first..` of a matrix whose rows start `ld` items apart in `data`: each the sum of products of its row with `x`. Four rows at a time share each read of `x`, each row in eight running sums, which vectorize and then add pairwise.
#[cfg(native)]
fn dots<T: Tiled>(out: &mut [T], data: &[T], ld: usize, first: usize, x: &[T]) {
    let k = x.len();
    let (quads, tail) = out.as_chunks_mut::<4>();
    let (xc, xr) = x.as_chunks::<8>();
    let sum = |s: &[T; 8]| ((s[0] + s[4]) + (s[2] + s[6])) + ((s[1] + s[5]) + (s[3] + s[7]));
    for (q, quad) in quads.iter_mut().enumerate() {
        let rows: [&[T]; 4] = std::array::from_fn(|r| &data[(first + 4 * q + r) * ld..][..k]);
        let mut s = [[T::zero(); 8]; 4];
        for (c, xs) in xc.iter().enumerate() {
            for r in 0..4 { let a = &rows[r][c * 8..][..8]; for l in 0..8 { s[r][l] = a[l].mul_acc(xs[l], s[r][l]) } }
        }
        let at = xc.len() * 8;
        for r in 0..4 { quad[r] = xr.iter().enumerate().fold(sum(&s[r]), |t, (p, &b)| rows[r][at + p].mul_acc(b, t)) }
    }
    let first = first + 4 * quads.len();
    for (i, y) in tail.iter_mut().enumerate() { *y = factor::dot(&data[(first + i) * ld..][..k], x) }
}

/// `+.×` of mapped operands on the tiled kernels, for [`Inner::inner_mapped`]. An operand whose mapping reads it at positive strides, one of them 1, is used where it lies, row by row or as a transpose. Any other operand is read a block of its stored rows at a time. Each block's product goes straight into the result. A block whose rows are the result's rows or columns writes them. A block whose rows run along the contraction adds to the whole result. Where neither operand can be used where it lies, `b` is read whole. A vector `b` beside an `a` whose rows are adjacent in storage goes to [`by_columns`], which reads `a` where it lies.
fn blocked<T: Tiled + Element>(a: Mapped<'_, T>, b: Mapped<'_, T>, [oa, ob]: [Orientation; 2], [m, k, n]: [usize; 3]) -> Vec<T> {
    let (ma, mb) = (a.matrix(oa, [m, k]), b.matrix(ob, [k, n]));
    if let (Some(a), Some(b)) = (ma, mb) { return T::multiply(a, b, [m, k, n]); }
    if ma.is_none() && n == 1 && let Some(c) = by_columns(a, b, oa, [m, k]) { return c; }
    let mut c = filled(m * n, T::zero());
    if m == 0 || k == 0 || n == 0 { return c; }
    let (one, zero) = (T::one(), T::zero());
    let added = |first: usize| if first == 0 { zero } else { one };
    let mut whole = Vec::new();
    match ma {
        None => {
            let b = match mb { Some(b) => b, None => Matrix::oriented(b.read(0, k * n, &mut whole), ob, [k, n]) };
            blocks(a, if oa == Orientation::Normal { [m, k] } else { [k, m] }, |first, len, part| match oa {
                Orientation::Normal => T::gemm(&mut c[first * n..], n, Matrix::rows(part, k), b, [len, k, n], one, zero),
                Orientation::Transpose => T::gemm(&mut c, n, Matrix::oriented(part, oa, [m, len]), b.offset(first, 0), [m, len, n], one, added(first)),
            });
        }
        Some(a) => blocks(b, if ob == Orientation::Normal { [k, n] } else { [n, k] }, |first, len, part| match ob {
            Orientation::Normal => T::gemm(&mut c, n, a.offset(0, first), Matrix::rows(part, n), [m, len, n], one, added(first)),
            Orientation::Transpose => T::gemm(&mut c[first..], n, a, Matrix::oriented(part, ob, [k, len]), [m, k, len], one, zero),
        }),
    }
    c
}

/// `c = beta c` for the items of `c` in `part`, where `c` is an `m`×`n` matrix whose rows start `ldc` items apart. A zero `beta` writes zeros, whatever `c` held.
fn scale<T: Tiled>(c: &mut [T], ldc: usize, [m, n]: [usize; 2], beta: T, part: Part) {
    if beta == T::one() { return; }
    for i in 0..m {
        for j in part.columns(i, n) {
            let x = &mut c[i * ldc + j];
            *x = if beta == T::zero() { T::zero() } else { beta * *x };
        }
    }
}

/// The multiply-adds in each unit of [`by_vector`]'s split across threads. `parallel::split` times the first unit to choose the pieces.
#[cfg(native)]
const VECTOR_WORK: usize = 1 << 15;
/// The results that [`by_vector`] keeps in the first-level cache while the stored rows of a transposed matrix add into them.
#[cfg(native)]
const VECTOR_STRIP: usize = 2048;
/// The stored rows of a transposed matrix that [`by_vector`] adds into its results together, read side by side in storage order.
#[cfg(native)]
const VECTOR_ROWS: usize = 8;
/// The fewest multiply-adds in each piece of [`by_vector`]'s split of stored rows across threads: twice what waking one more thread costs ([`THREAD_WORK`]), so that every split gains. At 4 threads on an M5 Max, a split cost about 10 to 12 µs whether it made two pieces or four, and pieces of 2^15 to 2^17 made a vector times a matrix of 1024 columns up to 76% slower than one piece where their work came to less than that. With pieces of 2^18, 128 and 256 rows ran as one piece, and 512 and 1024 rows ran 20 and 28% faster than before in two and four pieces.
#[cfg(native)]
const VECTOR_PIECE: usize = 2 * THREAD_WORK;

/// `alpha a b`, where `m` or `n` is 1, replacing `c` with `start` and added to it otherwise, as [`Tiled::native_product`] gives it. The packed kernels would pack the whole matrix and pad the vector to a tile. As a matrix times a vector, `a b` or the transpose of `bᵀ aᵀ`, each result is the sum of products of a stored row with the vector ([`dots`]) where the matrix's stored rows run along the contraction. Otherwise the stored rows, scaled by the vector's items, add into a strip of [`VECTOR_STRIP`] results [`VECTOR_ROWS`] rows at a time ([`scaled_runs`]), which reads each block of rows in storage order. The results split across threads, or, where they fit one strip of a transposed matrix, the stored rows do, each piece adding into results of its own. `false` for any other size, and for `Part::Lower`.
#[cfg(native)]
fn by_vector<T: Tiled>(c: &mut [T], ldc: usize, a: Matrix<'_, T>, alpha: T, b: Matrix<'_, T>, [m, k, n]: [usize; 3], start: bool, part: Part) -> bool {
    let (matrix, vector, rows, step) = match (m, n, part) {
        (_, 1, Part::Whole) => (a, b.transpose(), m, ldc),
        (1, _, Part::Whole) => (b.transpose(), a, n, 1),
        _ => return false,
    };
    // A conjugated matrix's results are the conjugates of its stored items' results with the conjugated vector.
    let conj = matches!(matrix.op, Op::Adjoint | Op::Conjugate);
    let x: Vec<T> = (0..k).map(|p| if conj { vector.at(0, p).conj() } else { vector.at(0, p) }).collect();
    let (data, ld, unit) = (matrix.data, matrix.ld, (VECTOR_WORK / k).max(1).next_multiple_of(4));
    let mut y = filled(rows, T::zero());
    if matrix.op.transposed() {
        // Stored row `p` holds the matrix's column `p`, so the matrix's row `i` starts at `data[i]`. A strip of results takes the stored rows a block at a time, reading the block's rows side by side in storage order.
        let mut columns: Vec<(T, usize)> = x.iter().enumerate().map(|(p, &w)| (w, p * ld)).collect();
        if rows <= VECTOR_STRIP {
            // Results that fit one strip leave each thread a narrow part of every stored row, so the stored rows split instead. Each piece reads its rows whole into results of its own, which add into `y` ([`parallel::summed`]).
            parallel::summed(&mut columns, 1, (VECTOR_PIECE / rows).max(1), &mut y, &mut Vec::new(), &|_, columns, part| {
                for block in columns.chunks(VECTOR_ROWS) { scaled_runs(part, data, 0, block) }
            });
        } else {
            parallel::split(&mut y, unit, &|first, part| {
                for (s, strip) in part.chunks_mut(VECTOR_STRIP).enumerate() {
                    for block in columns.chunks(VECTOR_ROWS) { scaled_runs(strip, data, first + s * VECTOR_STRIP, block) }
                }
                true
            });
        }
    } else {
        parallel::split(&mut y, unit, &|first, part| { dots(part, data, ld, first, &x); true });
    }
    // As the packed kernels do, `alpha` scales only where it isn't 1, which keeps a complex result's infinite part from making NaN.
    let one = alpha == T::one();
    for (i, y) in y.into_iter().enumerate() {
        let y = if conj { y.conj() } else { y };
        let (y, z) = (if one { y } else { alpha * y }, &mut c[i * step]);
        *z = if start { y } else { *z + y };
    }
    true
}

/// `gemm` on rustymath's own kernels, for a nonzero `k` and `alpha`: [`by_vector`] for a product with a vector, and the packed kernels otherwise.
#[cfg(native)]
fn native_gemm<T: Tiled>(c: &mut [T], ldc: usize, a: Matrix<'_, T>, b: Matrix<'_, T>, [m, k, n]: [usize; 3], alpha: T, beta: T, part: Part) {
    let start = beta == T::zero();
    if !start { scale(c, ldc, [m, n], beta, part) }
    if !by_vector(c, ldc, a, alpha, b, [m, k, n], start, part) { T::native_product(c, ldc, a, alpha, b, [m, k, n], start, part) }
}
/// The items of a product's result that it must fold: all of them, or those on and below its diagonal.
#[derive(Clone, Copy)]
pub(crate) enum Part { Whole, Lower }
impl Part {
    /// The columns of row `i` that this part holds, in a row of `n` items.
    #[inline(always)]
    fn columns(self, i: usize, n: usize) -> std::ops::Range<usize> { match self { Part::Whole => 0..n, Part::Lower => 0..(i + 1).min(n) } }
}
macro_rules! tiled {
    (@real $t:ty, $tile:ident, $kernel:ident; $c:ident, $ldc:ident, $a:ident, $alpha:ident, $b:ident, $size:ident, $start:ident, $part:ident) => {{
        const MR: usize = tiles::$tile[0];
        const NR: usize = tiles::$tile[1];
        let one = <$t>::one();
        product::<$t, PlusTimes, MR, NR>($c, $ldc, |rows, p0, out| $a.strip(rows, p0, $alpha, out), |columns, p0, out| $b.transpose().strip(columns, p0, one, out), $size, $start, $part, $kernel::<MR, NR, { NR * size_of::<$t>() / tiles::BYTES }>)
    }};
    (@complex $r:ty; $c:ident, $ldc:ident, $a:ident, $alpha:ident, $b:ident, $size:ident, $start:ident, $part:ident) => {
        complex_product::<$r>($c, $ldc, $a, $alpha, $b, $size, $start, $part)
    };
    (@eig real; $a:ident, $n:ident, $vectors:ident) => { from_order!($n, accelerate::GEEV_FROM, accelerate::eig($a, $n, $vectors), factor::general::native($a, $n, $vectors)) };
    (@eig complex; $a:ident, $n:ident, $vectors:ident) => { factor::general::native($a, $n, $vectors) };
    ($($t:ty: $kind:ident($($args:tt)*);)+) => {$(
        impl Tiled for $t {
            #[cfg(native)]
            #[inline]
            fn native_product(c: &mut [$t], ldc: usize, a: Matrix<'_, $t>, alpha: $t, b: Matrix<'_, $t>, size: [usize; 3], start: bool, part: Part) {
                tiled!(@$kind $($args)*; c, ldc, a, alpha, b, size, start, part)
            }
            fn svd(a: &[$t], size: [usize; 2], vectors: bool) -> Option<(Vec<$t>, Vec<<$t as Scalar>::Real>, Vec<$t>)> {
                from_order!(size[0].max(size[1]), accelerate::GESDD_FROM, accelerate::svd(a, size, vectors), factor::svd::native(a, size, vectors))
            }
            fn eigh(a: &[$t], n: usize, vectors: bool) -> Option<Eigh<$t>> {
                from_order!(n, accelerate::HEEVD_FROM, accelerate::eigh(a, n, vectors), factor::eigen::native(a, n, vectors))
            }
            fn eig(a: &[$t], n: usize, vectors: bool) -> Option<Eig<<$t as Scalar>::Real>> {
                tiled!(@eig $kind; a, n, vectors)
            }
        }
        impl Inner<$t> for PlusTimes {
            type Out = $t;
            fn inner_ops(&self, a: &[$t], b: &[$t], [oa, ob]: [Orientation; 2], [m, k, n]: [usize; 3]) -> Vec<$t> {
                assert!(a.len() == m * k && b.len() == k * n, "`a` must hold `m`×`k` items and `b` `k`×`n`");
                <$t>::multiply(Matrix::oriented(a, oa, [m, k]), Matrix::oriented(b, ob, [k, n]), [m, k, n])
            }
            #[inline]
            fn inner_into(&self, a: &[$t], b: &[$t], [m, k, n]: [usize; 3], c: &mut [$t]) {
                assert!(a.len() == m * k && b.len() == k * n && c.len() == m * n, "`a` must hold `m`×`k` items, `b` `k`×`n` and `c` `m`×`n`");
                if m * k * n <= DIRECT_WORK { return direct::<$t, $t, PlusTimes>(a, b, [m, k, n], c) }
                <$t>::gemm(c, n, Matrix::rows(a, k), Matrix::rows(b, n), [m, k, n], <$t>::one(), <$t>::zero())
            }
            fn inner_mapped(&self, a: Mapped<'_, $t>, b: Mapped<'_, $t>, orientations: [Orientation; 2], size: [usize; 3]) -> Vec<$t> { blocked(a, b, orientations, size) }
        }
    )+};
}
tiled! {
    f32: real(f32, TIMES32, times32);
    f64: real(f64, TIMES64, times64);
    Complex<f32>: complex(f32);
    Complex<f64>: complex(f64);
}
/// `alpha a b`, from four real products of the parts: real parts `aᵣbᵣ - aᵢbᵢ` and imaginary parts `aᵣbᵢ + aᵢbᵣ`, as scalar multiplication forms them. `parts` splits each operand once, before the real products.
#[cfg(native)]
fn complex_product<R: crate::Native + Tiled>(
    c: &mut [Complex<R>], ldc: usize, a: Matrix<'_, Complex<R>>, alpha: Complex<R>, b: Matrix<'_, Complex<R>>, size: [usize; 3], start: bool, part: Part) {
    let [m, k, n] = size;
    if m == 0 || k == 0 || n == 0 { return; }
    let ([ar, ai], [br, bi]) = (parts(a, [m, k], alpha), parts(b, [k, n], Complex::new(R::one(), R::zero())));
    let real = |c: &mut [R], x: &[R], y: &[R], scale: R, start: bool| R::native_product(c, n, Matrix::rows(x, k), scale, Matrix::rows(y, n), size, start, part);
    let (mut re, mut im) = (filled(m * n, R::zero()), filled(m * n, R::zero()));
    real(&mut re, &ar, &br, R::one(), true);
    real(&mut re, &ai, &bi, -R::one(), false);
    real(&mut im, &ar, &bi, R::one(), true);
    real(&mut im, &ai, &br, R::one(), false);
    for i in 0..m {
        for j in part.columns(i, n) {
            let (t, x) = (i * n + j, &mut c[i * ldc + j]);
            *x = if start { Complex::new(re[t], im[t]) } else { Complex::new(x.re + re[t], x.im + im[t]) };
        }
    }
}

/// The real parts and the imaginary parts of `scale` times the `rows`×`cols` matrix that `x` views, each row by row.
#[cfg(native)]
fn parts<R: crate::Native + Tiled>(x: Matrix<'_, Complex<R>>, [rows, cols]: [usize; 2], scale: Complex<R>) -> [Vec<R>; 2] {
    let (one, conj) = (scale == Complex::new(R::one(), R::zero()), matches!(x.op, Op::Adjoint | Op::Conjugate));
    let value = |v: Complex<R>| { let v = if conj { v.conj() } else { v }; if one { v } else { scale * v } };
    let mut z = filled(rows * cols, Complex::default());
    if x.op.transposed() { transpose_into(x.data, x.ld, &mut z, cols, [cols, rows], value) } else {
        for (row, out) in z.chunks_exact_mut(cols).enumerate() { for (o, &v) in out.iter_mut().zip(&x.data[row * x.ld..][..cols]) { *o = value(v) } }
    }
    [z.iter().map(|v| v.re).collect(), z.iter().map(|v| v.im).collect()]
}

/// The micro-kernels in plain Rust, for targets without the explicit ones. Each takes the number of vectors in a tile row, `V`, as the explicit ones do, and ignores it.
#[cfg(not(any(
    target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"),
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
)))]
mod plain {
    use super::*;
    macro_rules! plain {
        ($($name:ident: $t:ty, $object:ident;)+) => {$(
            #[inline(always)]
            pub(super) fn $name<const MR: usize, const NR: usize, const V: usize>(
                a: &[$t], b: &[$t], t: &mut [[<$object as Pair<$t>>::Out; NR]; MR], first: bool,
            ) {
                micro::<$t, $object, MR, NR>(a, b, t, first)
            }
        )+};
    }
    plain! {
        times64: f64, PlusTimes;
        times32: f32, PlusTimes;
        times_i32: i32, PlusTimes;
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
#[cfg(not(any(
    target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"),
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
)))]
use plain::*;

/// Micro-kernels on explicit vectors: 16 bytes on AArch64 and in wasm, and 32 with AVX2. Each is `super::micro` on vectors, with each row of the tile `V` vectors, and its entry gives one step from the tile's running value `c`, `a` and `b`. A tile starts where its entry says, at the pair's unit for a tile of items, as `micro`'s does, so the first step drops a NaN term as the later steps do.
#[cfg(any(
    target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"),
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
))]
mod lanes {
    #[cfg(target_arch = "aarch64")]
    use core::arch::aarch64::*;
    #[cfg(target_arch = "wasm32")]
    use core::arch::wasm32::*;
    #[cfg(target_arch = "x86_64")]
    use core::arch::x86_64::*;
    macro_rules! kernels {
        ($($name:ident: $t:ty => $out:ty, $lanes:literal, $splat:ident, $load:ident, $start:expr, $load_out:ident, $store_out:ident, |$a:ident, $b:ident, $c:ident| $step:expr;)+) => {$(
            #[inline(always)]
            pub(super) fn $name<const MR: usize, const NR: usize, const V: usize>(a: &[$t], b: &[$t], t: &mut [[$out; NR]; MR], first: bool) {
                assert!(NR == $lanes * V && a.len() / MR == b.len() / NR);
                let steps = a.len() / MR;
                // SAFETY: this module compiles only for targets with these instructions. Each load and store moves `$lanes` items inside `a`, `b` or a row of `t`.
                unsafe {
                    let (ap, bp) = (a.as_ptr(), b.as_ptr());
                    let mut acc = [[$start; V]; MR];
                    if !first { for r in 0..MR { for v in 0..V { acc[r][v] = $load_out(t[r].as_ptr().add($lanes * v).cast()) } } }
                    for p in 0..steps {
                        let y: [_; V] = core::array::from_fn(|v| $load(bp.add(p * NR + $lanes * v).cast()));
                        for r in 0..MR {
                            let $a = $splat(*ap.add(p * MR + r) as _);
                            for v in 0..V { acc[r][v] = { let ($c, $b) = (acc[r][v], y[v]); $step } }
                        }
                    }
                    for r in 0..MR { for v in 0..V { $store_out(t[r].as_mut_ptr().add($lanes * v).cast(), acc[r][v]) } }
                }
            }
        )+};
    }
    /// `kernels!` for tiles of items, which start at the entry's pair's unit.
    macro_rules! lanes {
        ($($name:ident: $t:ty, $pair:ident, $lanes:literal, $splat:ident, $load:ident, $store:ident, |$a:ident, $b:ident, $c:ident| $step:expr;)+) => {
            kernels! { $($name: $t => $t, $lanes, $splat, $load, $splat(<super::$pair as super::Pair<$t>>::UNIT), $load, $store, |$a, $b, $c| $step;)+ }
        };
    }
    // rustymath's own plus-times products, which macOS compiles only with the `native` feature.
    #[cfg(all(target_arch = "aarch64", native))]
    lanes! {
        times64: f64, PlusTimes, 2, vdupq_n_f64, vld1q_f64, vst1q_f64, |a, b, c| vfmaq_f64(c, a, b);
        times32: f32, PlusTimes, 4, vdupq_n_f32, vld1q_f32, vst1q_f32, |a, b, c| vfmaq_f32(c, a, b);
    }
    #[cfg(target_arch = "aarch64")]
    lanes! {
        // Integer plus-times products have no Accelerate form, so every target compiles them.
        times_i32: i32, PlusTimes, 4, vdupq_n_s32, vld1q_s32, vst1q_s32, |a, b, c| vmlaq_s32(c, a, b);
        max_plus64: f64, MaxPlus, 2, vdupq_n_f64, vld1q_f64, vst1q_f64, |a, b, c| vmaxnmq_f64(c, vaddq_f64(a, b));
        max_plus32: f32, MaxPlus, 4, vdupq_n_f32, vld1q_f32, vst1q_f32, |a, b, c| vmaxnmq_f32(c, vaddq_f32(a, b));
        min_plus64: f64, MinPlus, 2, vdupq_n_f64, vld1q_f64, vst1q_f64, |a, b, c| vminnmq_f64(c, vaddq_f64(a, b));
        min_plus32: f32, MinPlus, 4, vdupq_n_f32, vld1q_f32, vst1q_f32, |a, b, c| vminnmq_f32(c, vaddq_f32(a, b));
    }
    // Baseline wasm has no fused multiply-add.
    #[cfg(target_arch = "wasm32")]
    lanes! {
        times64: f64, PlusTimes, 2, f64x2_splat, v128_load, v128_store, |a, b, c| f64x2_add(c, f64x2_mul(a, b));
        times32: f32, PlusTimes, 4, f32x4_splat, v128_load, v128_store, |a, b, c| f32x4_add(c, f32x4_mul(a, b));
        times_i32: i32, PlusTimes, 4, i32x4_splat, v128_load, v128_store, |a, b, c| i32x4_add(c, i32x4_mul(a, b));
        max_plus64: f64, MaxPlus, 2, f64x2_splat, v128_load, v128_store, |a, b, c| f64x2_pmax(c, f64x2_add(a, b));
        max_plus32: f32, MaxPlus, 4, f32x4_splat, v128_load, v128_store, |a, b, c| f32x4_pmax(c, f32x4_add(a, b));
        min_plus64: f64, MinPlus, 2, f64x2_splat, v128_load, v128_store, |a, b, c| f64x2_pmin(c, f64x2_add(a, b));
        min_plus32: f32, MinPlus, 4, f32x4_splat, v128_load, v128_store, |a, b, c| f32x4_pmin(c, f32x4_add(a, b));
    }
    // rustymath's own plus-times products, which macOS compiles only with the `native` feature.
    #[cfg(all(target_arch = "x86_64", native))]
    lanes! {
        times64: f64, PlusTimes, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b, c| _mm256_fmadd_pd(a, b, c);
        times32: f32, PlusTimes, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b, c| _mm256_fmadd_ps(a, b, c);
    }
    #[cfg(target_arch = "x86_64")]
    lanes! {
        // Integer plus-times products have no Accelerate form, so every target compiles them. AVX2 has no multiply-add on 32-bit integers.
        times_i32: i32, PlusTimes, 8, _mm256_set1_epi32, _mm256_loadu_si256, _mm256_storeu_si256, |a, b, c| _mm256_add_epi32(c, _mm256_mullo_epi32(a, b));
        // AVX2's `maxpd` and `minpd` give their second argument beside NaN: the running value `c`.
        max_plus64: f64, MaxPlus, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b, c| _mm256_max_pd(_mm256_add_pd(a, b), c);
        max_plus32: f32, MaxPlus, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b, c| _mm256_max_ps(_mm256_add_ps(a, b), c);
        min_plus64: f64, MinPlus, 4, _mm256_set1_pd, _mm256_loadu_pd, _mm256_storeu_pd, |a, b, c| _mm256_min_pd(_mm256_add_pd(a, b), c);
        min_plus32: f32, MinPlus, 8, _mm256_set1_ps, _mm256_loadu_ps, _mm256_storeu_ps, |a, b, c| _mm256_min_ps(_mm256_add_ps(a, b), c);
    }
    // `PlusEqual`'s tiles count equal pairs. Where items are equal, a comparison gives a lane of all ones, which is -1 as an integer, so subtracting it adds 1 to the count.
    #[cfg(target_arch = "aarch64")]
    kernels! {
        equal64: f64 => i64, 2, vdupq_n_f64, vld1q_f64, vdupq_n_s64(0), vld1q_s64, vst1q_s64, |a, b, c| vsubq_s64(c, vreinterpretq_s64_u64(vceqq_f64(a, b)));
        equal32: f32 => i32, 4, vdupq_n_f32, vld1q_f32, vdupq_n_s32(0), vld1q_s32, vst1q_s32, |a, b, c| vsubq_s32(c, vreinterpretq_s32_u32(vceqq_f32(a, b)));
        equal_i64: i64 => i64, 2, vdupq_n_s64, vld1q_s64, vdupq_n_s64(0), vld1q_s64, vst1q_s64, |a, b, c| vsubq_s64(c, vreinterpretq_s64_u64(vceqq_s64(a, b)));
        equal_i32: i32 => i32, 4, vdupq_n_s32, vld1q_s32, vdupq_n_s32(0), vld1q_s32, vst1q_s32, |a, b, c| vsubq_s32(c, vreinterpretq_s32_u32(vceqq_s32(a, b)));
        equal_i16: i16 => i16, 8, vdupq_n_s16, vld1q_s16, vdupq_n_s16(0), vld1q_s16, vst1q_s16, |a, b, c| vsubq_s16(c, vreinterpretq_s16_u16(vceqq_s16(a, b)));
    }
    #[cfg(target_arch = "wasm32")]
    kernels! {
        equal64: f64 => i64, 2, f64x2_splat, v128_load, i64x2_splat(0), v128_load, v128_store, |a, b, c| i64x2_sub(c, f64x2_eq(a, b));
        equal32: f32 => i32, 4, f32x4_splat, v128_load, i32x4_splat(0), v128_load, v128_store, |a, b, c| i32x4_sub(c, f32x4_eq(a, b));
        equal_i64: i64 => i64, 2, i64x2_splat, v128_load, i64x2_splat(0), v128_load, v128_store, |a, b, c| i64x2_sub(c, i64x2_eq(a, b));
        equal_i32: i32 => i32, 4, i32x4_splat, v128_load, i32x4_splat(0), v128_load, v128_store, |a, b, c| i32x4_sub(c, i32x4_eq(a, b));
        equal_i16: i16 => i16, 8, i16x8_splat, v128_load, i16x8_splat(0), v128_load, v128_store, |a, b, c| i16x8_sub(c, i16x8_eq(a, b));
    }
    #[cfg(target_arch = "x86_64")]
    kernels! {
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
#[cfg(any(
    target_arch = "aarch64", all(target_arch = "wasm32", target_feature = "simd128"),
    all(target_arch = "x86_64", target_feature = "avx2", target_feature = "fma")
))]
use lanes::*;

/// The multiply-adds in each band of rows that `product` hands to a thread. On an M5 Max, 2^19 made 512×512 QR and inverses 5 to 13% faster on 4 threads than 2^20 did. 2^18 made `+.×`, Cholesky and QR 10 to 40% slower on 18 threads than 2^19.
const BAND_WORK: usize = 1 << 19;

/// The multiply-adds that waking one more thread costs a product. A panel of `w` multiply-adds splits across at most √(w / `THREAD_WORK`) threads, the count that minimises w / t + t × `THREAD_WORK`. On an M5 Max, `+.×` of two 256×256 matrices took 21.6 copies on 8 threads and 28.8 on 18, and larger products were fastest on all 18.
const THREAD_WORK: usize = 1 << 17;

/// The most items of `b` that `product` packs into one panel. It splits each panel's bands across threads in one call, however many steps of `KC` the panel holds.
const PANEL: usize = 1 << 22;
/// The most bytes of `b` that `product` packs into one panel when this thread's calls run serially, as on a pool thread. Larger panels outgrow the cache: on an M5 Max, a one-thread Gram product of a 16384×256 f64 matrix took 3% longer with 8 MB panels and 10% longer with 32 MB ones, and a threaded tall QR whose threads each ran one such product was 2% slower with 4 MB panels than with 2 MB ones.
const SERIAL_PANEL: usize = 2 << 20;
/// The steps of `b` that `product` packs across all of a panel's strips before packing the next steps. Packing a strip reads `NR` items from each row of a row-major `b`. Unless those items fill whole cache lines, the next strip reads the rest of each line. When `b`'s rows were a power of two apart, packing one strip at a time had already evicted those lines: on an M5 Max, a 64×256×16384 f64 product took 15 to 28% longer than with 16 steps at a time.
const ROWS: usize = 16;
/// The columns of the result that `product` fills one row of tiles at a time, before the next row of tiles. Filling each row of tiles in turn writes each row of `c` in order, whatever `ldc` is. Sweeping the whole width instead wrote the rows of a column of tiles together, and a power-of-two `ldc` maps those rows to the same cache sets: on an M5 Max, a 64×32×16384 f64 product took 2.3 times as long as a 64×32×16000 one. 128 columns of a packed `KC`-step f64 block of `b` hold 256 KB, which stays in a Zen 4 core's 1 MB second-level cache while each row of tiles rereads it.
const SWEEP: usize = 128;
/// Folds the inner product of `P` into `c`, an `m`×`n` matrix whose rows start `ldc` items apart, with `kernel` folding each `MR`×`NR` tile as `micro` does. `a(rows, p0, steps)` writes a strip of the `m`×`k` matrix `a`, and `b(columns, p0, steps)` a strip of the transpose of the `k`×`n` matrix `b`, as `Matrix::strip` does. With `start`, the product replaces `c`, which must then hold `f`'s identity where `k` is 0. Otherwise each result folds into `c`'s item with `f`. Only tiles that hold items of `part` fold. Items outside `part` in those tiles fold too. Each block of `NC` columns packs `b` in panels of at most [`PANEL`] items, or [`SERIAL_PANEL`] bytes on one thread, [`ROWS`] steps at a time across the panel's strips, across threads as [`parallel::split`] chooses. Then it runs bands of rows across at most √(w / [`THREAD_WORK`]) of this thread's [`parallel::threads`], for a panel of `w` multiply-adds: about one band for each [`BAND_WORK`] multiply-adds, up to four bands for each thread, each at most `MC` rows. Each band runs the panel's steps of `KC` in turn, packing its own block of `a` for each. Within a block of `a`, it fills one row of tiles at a time across [`SWEEP`] columns of the result.
#[inline(always)]
fn product<T: Copy + Default + Send + Sync, P: Pair<T>, const MR: usize, const NR: usize>(
    c: &mut [P::Out], ldc: usize, a: impl Fn([usize; 2], usize, &mut [[MaybeUninit<T>; MR]]) + Sync,
    b: impl Fn([usize; 2], usize, &mut [[MaybeUninit<T>; NR]]) + Sync, [m, k, n]: [usize; 3], start: bool, part: Part,
    kernel: impl Fn(&[T], &[T], &mut [[P::Out; NR]; MR], bool) + Sync) {
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
                for p0 in (0..kp).step_by(ROWS) {
                    let rows = ROWS.min(kp - p0);
                    for (u, strip) in units.chunks_exact_mut(unit).enumerate() {
                        let s = at / unit + u;
                        let (j, lines) = (jc + s * NR, NR.min(nc - s * NR));
                        pack::<_, NR>(&mut strip[p0 * NR..][..rows * NR], rows, lines, |first, ln, steps| b([j + first, ln], pp + p0, steps));
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
                        if lower && ic + mc <= jc { continue; }
                        let ap = pack::<_, MR>(&mut slots, kc, mc, |first, lines, steps| a([ic + first, lines], pc, steps));
                        for jb in (0..nc).step_by(SWEEP) {
                            for ir in (0..mc).step_by(MR) {
                                for jr in (jb..nc.min(jb + SWEEP)).step_by(NR) {
                                    let bj = &bp[jr / NR * unit + step * NR * KC..][..NR * kc];
                                    if lower && ic + ir + MR.min(mc - ir) <= jc + jr { continue; }
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
                }
                true
            });
        }
    }
}

/// Packs `len` lines of `kc` steps into strips of `W` lines at the start of `buffer`, and gives the items it wrote. Each strip holds its `W` items for step 0, then those for step 1, and so on. `strip(first, lines, steps)` writes all `W` items of each of the strip's `kc` steps: `lines` items for lines `first` on, then the default. The results computed from the defaults are never stored.
fn pack<T: Copy + Default, const W: usize>(
    buffer: &mut [MaybeUninit<T>], kc: usize, len: usize, strip: impl Fn(usize, usize, &mut [[MaybeUninit<T>; W]])) -> &[T] {
    let strips = len.div_ceil(W);
    for (s, chunk) in buffer.chunks_exact_mut(W * kc).take(strips).enumerate() { strip(s * W, W.min(len - s * W), chunk.as_chunks_mut::<W>().0); }
    slice::written(&mut buffer[..strips * W * kc])
}

/// Folds the steps of the strips `a` and `b`, which `pack` wrote, into the tile `t`. With `first`, the tile starts at the pair's unit, as `direct`'s sums do.
#[inline(always)]
fn micro<T: Copy, P: Pair<T>, const MR: usize, const NR: usize>(a: &[T], b: &[T], t: &mut [[P::Out; NR]; MR], first: bool) {
    let mut acc = if first { [[P::UNIT; NR]; MR] } else { *t };
    for (a, b) in a.as_chunks::<MR>().0.iter().zip(b.as_chunks::<NR>().0) { for r in 0..MR { for s in 0..NR { acc[r][s] = P::step(acc[r][s], a[r], b[s]) } } }
    *t = acc;
}
