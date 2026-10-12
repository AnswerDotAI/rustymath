//! How fast each function object's slice form runs on 65,536 inputs. The slice form is the code that callers such as basedpl and the
//! Python package run. Each function has a budget at f64 and at f32, in copies of the same number of items: the most time it may take,
//! as a multiple of the time to copy those items. Each budget is twice the higher of the function's cost on the Mac and on CI. A loop
//! that stops vectorizing takes several times as long and goes over budget. Where the standard library has the function, its time
//! item by item is measured too, for the README's table, but sets no limit. The rows `Pow.one` and `Root.one` time the scalar forms
//! pair by pair, as a caller with one value at a time runs them.
//! The inner products multiply 256×256 matrices, whose results hold the same number of items. The decompositions factor 256×256 real
//! matrices and 128×128 complex ones, and `lstsq` solves a 4096×16 real one. Each of these counts copies of as many items as its
//! matrix holds. None of them has CI figures yet. Their budgets are four times their cost on the Mac.
//! `RUSTYMATH_TIMES=1 cargo test --test speed -- --nocapture` prints the README's table, then every function's cost in copies.
//! The test holds the machine's timing lock while it runs, as `timing_lock` describes.
use rustymath::*;
use std::{hint::black_box, ops::Neg, time::Instant};

const N: usize = 65_536;
/// A function's name, its budgets at f64 and at f32 in copies, and its times in ns per item: its own at f64, the standard library's
/// at f64, its own at f32, and the standard library's at f32.
type Compared = (&'static str, [f64; 2], [f64; 4]);
/// A function's name, its budgets at f64 and at f32 in copies, and its costs in copies.
type Budgeted = (&'static str, [f64; 2], [f64; 2]);

/// The fastest of a few runs of `run`, in ns for each of `n` items.
fn fastest(n: usize, mut run: impl FnMut()) -> f64 {
    (0..5)
        .map(|_| {
            let start = Instant::now();
            run();
            start.elapsed().as_secs_f64() * 1e9 / n as f64
        })
        .fold(f64::INFINITY, f64::min)
}

/// The fastest of a few runs of `f`, which writes one output for each of `inputs`, in ns per item.
fn time_slice<A, T: Copy + Default>(inputs: &[A], f: impl Fn(&[A], &mut [T])) -> f64 {
    let mut out = vec![T::default(); inputs.len()];
    fastest(inputs.len(), || { f(black_box(inputs), &mut out); black_box(&mut out); })
}

/// The fastest of a few runs of `f`, which appends one output for each of `inputs` to an empty vector, in ns per item.
fn time_appended<A, T>(inputs: &[A], f: impl Fn(&[A], &mut Vec<T>)) -> f64 {
    let mut out = Vec::with_capacity(inputs.len());
    fastest(inputs.len(), || { out.clear(); f(black_box(inputs), &mut out); black_box(&mut out); })
}

/// The fastest of a few runs of `f` on each of `inputs`, in ns per item.
fn time<A: Copy, T: Copy + Default>(inputs: &[A], f: impl Fn(A) -> T) -> f64 {
    time_slice(inputs, |xs, out| for (o, &x) in out.iter_mut().zip(xs) { *o = f(x); })
}

/// The time per item of `k`'s slice form on `x`.
fn time_unary<F, K: Unary<F>>(x: &[F], k: &K) -> f64 { time_appended(x, |x, out| { k.many(Arg::Many(x), x.len(), out); }) }

/// The time per item of `k`'s slice form on the pairs of `x` and `y`.
fn time_binary<F: Copy, K: Binary<F>>(x: &[F], y: &[F], k: &K) -> f64 {
    time_appended(x, |x, out| { k.many(Arg::Many(x), Arg::Many(y), x.len(), out); })
}

/// The time per item of `k`'s slice form on the triples of `x`, `y` and `z`.
fn time_ternary<F: Copy, K: Ternary<F>>(x: &[F], y: &[F], z: &[F], k: &K) -> f64 {
    time_appended(x, |x, out| { k.many(Arg::Many(x), Arg::Many(y), Arg::Many(z), x.len(), out); })
}

/// Horner's rule for one item, as a loop over the items would call it with the standard library's `mul_add`.
fn horner<F: Float>(c: &[F], x: F) -> F { c.iter().rev().fold(F::zero(), |y, &k| y.mul_add(x, k)) }

/// `N` inputs from `lo` to `hi`, from the fractional parts of multiples of `step`. Their order has no pattern that a branch predictor
/// could learn.
fn inputs(lo: f64, hi: f64, step: f64) -> Vec<f64> { (0..N).map(|i| lo + (hi - lo) * (i as f64 * step).fract()).collect() }
const GOLDEN: f64 = 0.618_033_988_749_894_9;
const SILVER: f64 = 0.414_213_562_373_095_1;
const BRONZE: f64 = 0.302_775_637_731_994_6;

/// Inputs from `lo` to `hi`, at f64 and narrowed to f32.
fn widths(lo: f64, hi: f64, step: f64) -> (Vec<f64>, Vec<f32>) {
    let x = inputs(lo, hi, step);
    let narrowed = x.iter().map(|&x| x as f32).collect();
    (x, narrowed)
}

/// The budgets of `$object`, then the times of its slice form and of the standard library's `$std` item by item, at f64 and then f32,
/// on inputs from `$lo` to `$hi`.
macro_rules! unary {
    ($($object:ident within $budget:expr, vs $std:ident on $lo:literal to $hi:literal;)+) => {
        [$({
            let (x64, x32) = widths($lo, $hi, GOLDEN);
            (stringify!($object), $budget, [time_unary(&x64, &$object), time(&x64, f64::$std), time_unary(&x32, &$object), time(&x32, f32::$std)])
        }),+]
    };
}

/// The time per item of `f` on each pair of `x` and `y`, called one pair at a time.
fn time_pairs<F: Copy, T: Copy + Default>(x: &[F], y: &[F], f: impl Fn(F, F) -> T) -> f64 {
    time_slice(x, |x, out| for i in 0..x.len() { out[i] = f(x[i], y[i]) })
}

/// The budgets of `k`, then the times of its slice form and of a reference version pair by pair, on pairs from the ranges `xs` and
/// `ys`, at f64 and then f32. With `one`, it times `k`'s scalar form pair by pair in place of its slice form.
fn binary<K: Binary<f64, Out = f64> + Binary<f32, Out = f32>>(
    name: &'static str,
    k: K,
    one: bool,
    budget: [f64; 2],
    (xs, ys): ((f64, f64), (f64, f64)),
    std64: impl Fn(f64, f64) -> f64,
    std32: impl Fn(f32, f32) -> f32,
) -> Compared {
    let ((x64, x32), (y64, y32)) = (widths(xs.0, xs.1, GOLDEN), widths(ys.0, ys.1, SILVER));
    let (ours64, ours32) = if one {
        (time_pairs(&x64, &y64, |a, b| k.one(a, b)), time_pairs(&x32, &y32, |a, b| k.one(a, b)))
    } else {
        (time_binary(&x64, &y64, &k), time_binary(&x32, &y32, &k))
    };
    (name, budget, [ours64, time_pairs(&x64, &y64, std64), ours32, time_pairs(&x32, &y32, std32)])
}

/// The budgets and times of a reduction or scan of `rustymath::reduce`, and the times of the same taken item by item in order, which
/// the compiler can't vectorize for floats.
fn reductions() -> [Compared; 4] {
    let (x64, x32) = widths(-1.0, 1.0, GOLDEN);
    let fold = |run: &dyn Fn() -> f64| fastest(N, || { black_box(run()); });
    let mut s64 = Vec::with_capacity(N);
    let mut s32 = Vec::with_capacity(N);
    let scan64 = |x: &[f64], out: &mut Vec<f64>| { let mut t = 0.0; out.extend(x.iter().map(|&v| { t += v; t })); };
    let scan32 = |x: &[f32], out: &mut Vec<f32>| { let mut t = 0.0; out.extend(x.iter().map(|&v| { t += v; t })); };
    let max64 = |x: &[f64], out: &mut Vec<f64>| { let mut t = f64::NEG_INFINITY; out.extend(x.iter().map(|&v| { t = t.max(v); t })); };
    let max32 = |x: &[f32], out: &mut Vec<f32>| { let mut t = f32::NEG_INFINITY; out.extend(x.iter().map(|&v| { t = t.max(v); t })); };
    [
        ("reduce::sum", [1.0, 1.0], [
            fold(&|| reduce::sum(black_box(&x64)).0),
            fold(&|| black_box(&x64).iter().sum::<f64>()),
            fold(&|| f64::from(reduce::sum(black_box(&x32)).0)),
            fold(&|| f64::from(black_box(&x32).iter().sum::<f32>())),
        ]),
        ("reduce::max", [1.0, 1.6], [
            fold(&|| reduce::max(black_box(&x64))),
            fold(&|| black_box(&x64).iter().copied().fold(f64::NEG_INFINITY, f64::max)),
            fold(&|| f64::from(reduce::max(black_box(&x32)))),
            fold(&|| f64::from(black_box(&x32).iter().copied().fold(f32::NEG_INFINITY, f32::max))),
        ]),
        ("reduce::cumsum", [3.4, 7.8], [
            time_appended(&x64, |x, out| { reduce::cumsum(x, None, out); }),
            fastest(N, || { s64.clear(); scan64(black_box(&x64), &mut s64); black_box(&s64); }),
            time_appended(&x32, |x, out| { reduce::cumsum(x, None, out); }),
            fastest(N, || { s32.clear(); scan32(black_box(&x32), &mut s32); black_box(&s32); }),
        ]),
        ("reduce::cummax", [6.0, 14.0], [
            time_appended(&x64, |x, out| reduce::cummax(x, None, out)),
            fastest(N, || { s64.clear(); max64(black_box(&x64), &mut s64); black_box(&s64); }),
            time_appended(&x32, |x, out| reduce::cummax(x, None, out)),
            fastest(N, || { s32.clear(); max32(black_box(&x32), &mut s32); black_box(&s32); }),
        ]),
    ]
}

/// Writes each item of `x`, negated, to `out`. It moves the data that a copy does, in a vector loop as the functions do. The compiler
/// would turn a plain copy into a call to `memcpy`. The speed of `memcpy` varies with the CPU and with how the two buffers align.
fn negate<T: Copy + Neg<Output = T>>(x: &[T], out: &mut [T]) { for (o, &v) in out.iter_mut().zip(x) { *o = -v } }

/// The time to `negate` `N` items, each `v`, in ns per item: the unit of every budget, called a copy. A nonzero `v` makes `vec!` write
/// every item. Zeros could come from untouched pages. Every read of an untouched page finds the same page of zeros in the cache.
fn copy<T: Copy + Default + Neg<Output = T>>(v: T) -> f64 { time_slice(&vec![v; N], negate) }

/// The functions that the standard library lacks, with their budgets and costs in copies of floats, `unit` at f64 and at f32. The
/// incomplete gamma and beta functions iterate, and run on 1,024 inputs. The others run on 16,384.
fn special(unit: [f64; 2]) -> [Budgeted; 11] {
    let (many, few) = (N / 4, N / 64);
    let one = |k: &dyn Fn(&[f64], &[f32]) -> [f64; 2], (lo, hi): (f64, f64), n: usize| {
        let (x64, x32) = widths(lo, hi, GOLDEN);
        let t = k(&x64[..n], &x32[..n]);
        [t[0] / unit[0], t[1] / unit[1]]
    };
    macro_rules! unary_copies {
        ($k:expr, $lo:literal to $hi:literal) => { one(&|x64, x32| [time_unary(x64, &$k), time_unary(x32, &$k)], ($lo, $hi), many) };
    }
    let two = |k: &dyn Fn(&[f64], &[f64], &[f32], &[f32]) -> [f64; 2], xs: (f64, f64), ys: (f64, f64)| {
        let ((x64, x32), (y64, y32)) = (widths(xs.0, xs.1, GOLDEN), widths(ys.0, ys.1, SILVER));
        let t = k(&x64[..few], &y64[..few], &x32[..few], &y32[..few]);
        [t[0] / unit[0], t[1] / unit[1]]
    };
    macro_rules! binary_copies {
        ($k:expr, $xs:expr, $ys:expr) => { two(&|x64, y64, x32, y32| [time_binary(x64, y64, &$k), time_binary(x32, y32, &$k)], $xs, $ys) };
    }
    let three = |k: &dyn Fn(&[f64], &[f64], &[f64], &[f32], &[f32], &[f32]) -> [f64; 2], zs: (f64, f64)| {
        let ((x64, x32), (y64, y32), (z64, z32)) = (widths(0.5, 20.0, GOLDEN), widths(0.5, 20.0, SILVER), widths(zs.0, zs.1, BRONZE));
        let t = k(&x64[..few], &y64[..few], &z64[..few], &x32[..few], &y32[..few], &z32[..few]);
        [t[0] / unit[0], t[1] / unit[1]]
    };
    macro_rules! ternary_copies {
        ($k:expr, $zs:expr) => {
            three(&|x64, y64, z64, x32, y32, z32| [time_ternary(x64, y64, z64, &$k), time_ternary(x32, y32, z32, &$k)], $zs)
        };
    }
    [
        ("Gamma", [330.0, 210.0], unary_copies!(Gamma, 0.1 to 20.0)),
        ("Lgamma", [170.0, 170.0], unary_copies!(Lgamma, 0.1 to 1000.0)),
        ("Erf", [61.0, 36.0], unary_copies!(Erf, -5.0 to 5.0)),
        ("Erfc", [250.0, 130.0], unary_copies!(Erfc, -5.0 to 10.0)),
        ("Erfinv", [440.0, 72.0], unary_copies!(Erfinv, -0.999 to 0.999)),
        ("Erfcinv", [440.0, 75.0], unary_copies!(Erfcinv, 0.001 to 1.999)),
        ("Gammainc", [1400.0, 1200.0], binary_copies!(Gammainc, (0.5, 20.0), (0.0, 30.0))),
        ("Gammaincc", [1400.0, 1200.0], binary_copies!(Gammaincc, (0.5, 20.0), (0.0, 30.0))),
        ("Gammaincinv", [5100.0, 12000.0], binary_copies!(Gammaincinv, (0.5, 20.0), (0.001, 0.999))),
        ("Betainc", [3700.0, 8400.0], ternary_copies!(Betainc, (0.0, 1.0))),
        ("Betaincinv", [13000.0, 30000.0], ternary_copies!(Betaincinv, (0.001, 0.999))),
    ]
}

/// Complex function objects, with their budgets and costs in copies of the same number of complex items.
fn complex() -> [Budgeted; 12] {
    use num_complex::Complex;
    let n = N / 4;
    let unit = [copy(Complex::new(1.0f64, 1.0)), copy(Complex::new(1.0f32, 1.0))];
    let zs = |re: (f64, f64), im: (f64, f64), step: f64| {
        let z64: Vec<_> = inputs(re.0, re.1, step).into_iter().zip(inputs(im.0, im.1, SILVER)).take(n).map(|(x, y)| Complex::new(x, y)).collect();
        let z32: Vec<_> = z64.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect();
        (z64, z32)
    };
    macro_rules! unary_copies {
        ($k:expr, $re:expr, $im:expr) => {{
            let (z64, z32) = zs($re, $im, GOLDEN);
            [time_unary(&z64, &$k) / unit[0], time_unary(&z32, &$k) / unit[1]]
        }};
    }
    let square = (-2.0, 2.0);
    let (w64, w32) = zs(square, square, BRONZE);
    let (z64, z32) = zs(square, square, GOLDEN);
    [
        ("complex Exp", [38.0, 46.0], unary_copies!(Exp, square, square)),
        ("complex Ln", [67.0, 76.0], unary_copies!(Ln, square, square)),
        ("complex Sqrt", [22.0, 25.0], unary_copies!(Sqrt, square, square)),
        ("complex Sin", [51.0, 56.0], unary_copies!(Sin, square, square)),
        ("complex Tan", [160.0, 260.0], unary_copies!(Tan, square, square)),
        ("complex Asin", [260.0, 420.0], unary_copies!(Asin, square, square)),
        ("complex Atanh", [130.0, 230.0], unary_copies!(Atanh, square, square)),
        ("complex Lgamma", [290.0, 500.0], unary_copies!(Lgamma, (0.1, 20.0), (-10.0, 10.0))),
        ("complex Gamma", [420.0, 720.0], unary_copies!(Gamma, (0.1, 20.0), (-10.0, 10.0))),
        ("complex Abs", [5.2, 5.2], unary_copies!(Abs, square, square)),
        ("complex Sqrt1mx2", [110.0, 170.0], unary_copies!(Sqrt1mx2, square, square)),
        ("complex Pow", [120.0, 130.0], [time_binary(&z64, &w64, &Pow) / unit[0], time_binary(&z32, &w32, &Pow) / unit[1]]),
    ]
}

/// Inner products of 256×256 matrices, whose results hold `N` items. In the integer rows, the 64-bit column times items large enough
/// to need the 64-bit kernels, and the 32-bit column items small enough for the 32-bit kernels. Both count copies of 64-bit items.
fn inner_products(unit: [f64; 2]) -> [Budgeted; 6] {
    use linalg::*;
    let size = [256; 3];
    let ((a64, a32), (b64, b32)) = (widths(-1.0, 1.0, GOLDEN), widths(-1.0, 1.0, SILVER));
    let run = |k: &dyn Fn() -> usize| fastest(N, || { black_box(k()); });
    macro_rules! copies {
        ($k:expr) => { [run(&|| $k.inner(black_box(&a64), black_box(&b64), size).len()) / unit[0], run(&|| $k.inner(black_box(&a32), black_box(&b32), size).len()) / unit[1]] };
    }
    let whole = |x: &[f64], scale: f64| x.iter().map(|&v| (v * scale) as i64).collect::<Vec<_>>();
    let (large, small) = ([whole(&a64, 1e6), whole(&b64, 1e6)], [whole(&a64, 100.0), whole(&b64, 100.0)]);
    let wide = large.clone().map(|x| x.into_iter().map(|v| v << 20).collect::<Vec<_>>());
    macro_rules! integer_copies {
        ($k:expr, $large:expr, $small:expr) => {
            [$large, $small].map(|[a, b]: &[Vec<i64>; 2]| run(&|| $k.inner(black_box(a.as_slice()), black_box(b.as_slice()), size).len()) / unit[0])
        };
    }
    [
        ("linalg::PlusTimes", [290.0, 340.0], copies!(PlusTimes)),
        ("linalg::MaxPlus", [610.0, 660.0], copies!(MaxPlus)),
        ("linalg::MinPlus", [610.0, 660.0], copies!(MinPlus)),
        ("linalg::PlusEqual", [620.0, 640.0], copies!(PlusEqual)),
        ("linalg::PlusTimes at i64", [320.0, 190.0], integer_copies!(PlusTimes, &large, &small)),
        ("linalg::PlusEqual at i64", [590.0, 420.0], integer_copies!(PlusEqual, &wide, &small)),
    ]
}

/// Decompositions of 256×256 real matrices and 128×128 complex ones, and a least-squares solution for a 4096×16 real matrix. Each cost
/// counts copies of as many items as the matrix holds. `s` is symmetric positive definite, and `h` is Hermitian.
fn decompositions(unit: [f64; 2]) -> Vec<Budgeted> {
    use linalg::*;
    use num_complex::Complex;
    let mut seed = 1u64;
    let mut random = |len: usize| -> Vec<f64> {
        (0..len)
            .map(|_| {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
            })
            .collect()
    };
    let (a64, t64, b64, c64) = (random(N), random(N), random(256), random(4096));
    let at: Vec<f64> = (0..N).map(|k| a64[k % 256 * 256 + k / 256]).collect();
    let mut s64 = linalg::PlusTimes.inner(&a64, &at, [256; 3]);
    for i in 0..256 { s64[i * 257] += 256.0 }
    let narrow = |x: &[f64]| x.iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let (a32, t32, b32, c32, s32) = (narrow(&a64), narrow(&t64), narrow(&b64), narrow(&c64), narrow(&s64));
    let run = |items: usize, k: &dyn Fn() -> usize| fastest(items, || { black_box(k()); });
    let real = |k64: &dyn Fn() -> usize, k32: &dyn Fn() -> usize| [run(N, k64) / unit[0], run(N, k32) / unit[1]];
    let n = 128;
    let complex_unit = [copy(Complex::new(1.0f64, 1.0)), copy(Complex::new(1.0f32, 1.0))];
    let z64: Vec<Complex<f64>> = a64.iter().zip(&t64).take(n * n).map(|(&re, &im)| Complex::new(re, im)).collect();
    let h64: Vec<Complex<f64>> = (0..n * n).map(|k| z64[k] + z64[k % n * n + k / n].conj()).collect();
    let zb64: Vec<Complex<f64>> = b64.iter().zip(&c64).take(n).map(|(&re, &im)| Complex::new(re, im)).collect();
    let narrowed = |x: &[Complex<f64>]| x.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect::<Vec<_>>();
    let (z32, h32, zb32) = (narrowed(&z64), narrowed(&h64), narrowed(&zb64));
    let complex = |k64: &dyn Fn() -> usize, k32: &dyn Fn() -> usize| [run(n * n, k64) / complex_unit[0], run(n * n, k32) / complex_unit[1]];
    vec![
        ("linalg::Lu::solve", [220.0, 290.0], real(&|| lu(black_box(&a64), 256).solve(&b64, 1).len(), &|| lu(black_box(&a32), 256).solve(&b32, 1).len())),
        ("linalg::Lu::inverse", [620.0, 780.0], real(&|| lu(black_box(&a64), 256).inverse().len(), &|| lu(black_box(&a32), 256).inverse().len())),
        ("linalg::cholesky", [120.0, 150.0], real(&|| cholesky(black_box(&s64), 256).map_or(0, |l| l.len()), &|| cholesky(black_box(&s32), 256).map_or(0, |l| l.len()))),
        ("linalg::qr", [960.0, 1300.0], real(&|| qr(black_box(&a64), [256, 256]).0.len(), &|| qr(black_box(&a32), [256, 256]).0.len())),
        ("linalg::svd", [6300.0, 7300.0], real(&|| svd(black_box(&a64), [256, 256]).u.len(), &|| svd(black_box(&a32), [256, 256]).u.len())),
        ("linalg::eigh", [2900.0, 3800.0], real(&|| eigh(black_box(&s64), 256).map_or(0, |f| f.vectors.len()), &|| eigh(black_box(&s32), 256).map_or(0, |f| f.vectors.len()))),
        ("linalg::eig", [11000.0, 13000.0], real(&|| eig(black_box(&a64), 256).map_or(0, |f| f.vectors.len()), &|| eig(black_box(&a32), 256).map_or(0, |f| f.vectors.len()))),
        ("linalg::eigenvalues", [5800.0, 7500.0], real(&|| eigenvalues(black_box(&a64), 256).map_or(0, |v| v.len()), &|| eigenvalues(black_box(&a32), 256).map_or(0, |v| v.len()))),
        ("linalg::lstsq", [150.0, 210.0], real(&|| lstsq(black_box(&t64), &c64, [4096, 16], 1).0.len(), &|| lstsq(black_box(&t32), &c32, [4096, 16], 1).0.len())),
        ("complex linalg::Lu::solve", [240.0, 330.0], complex(&|| lu(black_box(&z64), n).solve(&zb64, 1).len(), &|| lu(black_box(&z32), n).solve(&zb32, 1).len())),
        ("complex linalg::qr", [1300.0, 1800.0], complex(&|| qr(black_box(&z64), [n, n]).0.len(), &|| qr(black_box(&z32), [n, n]).0.len())),
        ("complex linalg::svd", [4400.0, 5400.0], complex(&|| svd(black_box(&z64), [n, n]).u.len(), &|| svd(black_box(&z32), [n, n]).u.len())),
        ("complex linalg::eigh", [2000.0, 2700.0], complex(&|| eigh(black_box(&h64), n).map_or(0, |f| f.vectors.len()), &|| eigh(black_box(&h32), n).map_or(0, |f| f.vectors.len()))),
        ("complex linalg::eig", [10000.0, 12000.0], complex(&|| eig(black_box(&z64), n).map_or(0, |f| f.vectors.len()), &|| eig(black_box(&z32), n).map_or(0, |f| f.vectors.len()))),
    ]
}

/// The file whose lock keeps timings and builds apart on this machine, as DEV.md describes. Timings take it exclusively. Builds take
/// it shared.
const TIMING_LOCK: &str = "/tmp/aai-ws-timing.lock";

/// Takes the timing lock exclusively, and returns the file that holds it until dropped. It waits up to 5 minutes while a build or
/// another timing holds the lock. The operating system releases the lock when the test ends, even by a crash.
fn timing_lock() -> std::fs::File {
    let file = std::fs::File::options().create(true).append(true).open(TIMING_LOCK).expect("the timing lock's file opens");
    let (start, mut waiting) = (Instant::now(), false);
    loop {
        match file.try_lock() {
            Ok(()) => return file,
            Err(std::fs::TryLockError::WouldBlock) => {}
            Err(std::fs::TryLockError::Error(e)) => panic!("{TIMING_LOCK}: {e}"),
        }
        assert!(start.elapsed().as_secs() < 300, "a build or another timing has held {TIMING_LOCK} for 5 minutes");
        if !waiting { eprintln!("waiting for a build or another timing to release {TIMING_LOCK}"); }
        waiting = true;
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}
#[test]
fn speed() {
    let _lock = timing_lock();
    let unit = [copy(1.0f64), copy(1.0f32)];
    let mut compared = unary! {
        Exp within [15.0, 12.0], vs exp on -20.0 to 20.0;
        Expm1 within [20.0, 16.0], vs exp_m1 on -1.0 to 1.0;
        Ln within [24.0, 25.0], vs ln on 0.001 to 1000.0;
        Log1p within [35.0, 33.0], vs ln_1p on -0.5 to 1000.0;
        Log2 within [25.0, 21.0], vs log2 on 0.001 to 1000.0;
        Log10 within [25.0, 22.0], vs log10 on 0.001 to 1000.0;
        Sin within [16.0, 15.0], vs sin on -10.0 to 10.0;
        Cos within [16.0, 16.0], vs cos on -10.0 to 10.0;
        Tan within [21.0, 18.0], vs tan on -10.0 to 10.0;
        Asin within [17.0, 12.0], vs asin on -1.0 to 1.0;
        Acos within [19.0, 14.0], vs acos on -1.0 to 1.0;
        Atan within [17.0, 12.0], vs atan on -10.0 to 10.0;
        Sinh within [26.0, 21.0], vs sinh on -10.0 to 10.0;
        Cosh within [17.0, 14.0], vs cosh on -10.0 to 10.0;
        Tanh within [25.0, 20.0], vs tanh on -10.0 to 10.0;
        Asinh within [28.0, 26.0], vs asinh on -1000.0 to 1000.0;
        Acosh within [34.0, 29.0], vs acosh on 1.0 to 1000.0;
        Atanh within [28.0, 25.0], vs atanh on -0.999 to 0.999;
        Cbrt within [20.0, 33.0], vs cbrt on -1000.0 to 1000.0;
    }
    .to_vec();
    let (powers, roots) = (((0.0, 4.0), (-30.0, 30.0)), ((5.0, 5.0), (0.0, 1000.0)));
    let nth = |n: f64, y: f64| y.powf(1.0 / n);
    let nth32 = |n: f32, y: f32| y.powf(1.0 / n);
    compared.push(binary("Pow", Pow, false, [80.0, 68.0], powers, f64::powf, f32::powf));
    compared.push(binary("Pow.one", Pow, true, [110.0, 130.0], powers, f64::powf, f32::powf));
    compared.push(binary("Hypot", Hypot, false, [9.6, 10.0], ((-1000.0, 1000.0), (-1000.0, 1000.0)), f64::hypot, f32::hypot));
    compared.push(binary("Root", Root, false, [86.0, 81.0], roots, nth, nth32));
    compared.push(binary("Root.one", Root, true, [120.0, 140.0], roots, nth, nth32));
    compared.push(binary("Maximum", reduce::Maximum, false, [3.2, 3.6], ((-1.0, 1.0), (-1.0, 1.0)), f64::max, f32::max));
    compared.extend(reductions());
    // A polynomial of degree 7 at the negative of each input, against `horner` for one input at a time. `black_box` hides the degree,
    // which a caller rarely knows when it compiles.
    let c64 = [0.5, -1.25, 0.75, 2.0, -0.125, 0.0625, -1.5, 0.25];
    let c32 = c64.map(|k| k as f32);
    let (c64, c32): (&[f64], &[f32]) = (black_box(&c64), black_box(&c32));
    let (x64, x32) = widths(-2.0, 2.0, GOLDEN);
    compared.push((
        "polynomial::evaluate",
        [8.8, 7.4],
        [
            time_slice(&x64, |x, out| { negate(x, out); polynomial::evaluate(c64, out) }),
            time(&x64, |x| horner(c64, x)),
            time_slice(&x32, |x, out| { negate(x, out); polynomial::evaluate(c32, out) }),
            time(&x32, |x| horner(c32, x)),
        ],
    ));
    let budgeted: Vec<Budgeted> = compared
        .iter()
        .map(|&(name, budget, t)| (name, budget, [t[0] / unit[0], t[2] / unit[1]]))
        .chain(special(unit))
        .chain(complex())
        .chain(inner_products(unit))
        .chain(decompositions(unit))
        .collect();
    if std::env::var_os("RUSTYMATH_TIMES").is_some() {
        println!("| Function | f64 | std f64 | f32 | std f32 |\n|---|---|---|---|---|");
        for (name, _, t) in compared.iter() { println!("| `{name}` | {:.2} | {:.2} | {:.2} | {:.2} |", t[0], t[1], t[2], t[3]); }
        println!("\n| Function | f64 copies | f32 copies |\n|---|---|---|");
        for (name, _, c) in budgeted.iter() { println!("| `{name}` | {:.1} | {:.1} |", c[0], c[1]); }
    }
    let over: Vec<_> =
        budgeted.iter().flat_map(|&(name, b, c)| [(name, "f64", c[0], b[0]), (name, "f32", c[1], b[1])]).filter(|r| r.2 > r.3).collect();
    assert!(over.is_empty(), "over budget, in copies: {over:?}");
}
