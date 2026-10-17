//! How fast each function object's slice form runs on 65,536 inputs. The slice form is the code that callers such as basedpl and the Python package run. `speed.txt` gives each function a budget at f64 and at f32, in copies of the same number of items: the most time it may take, as a multiple of the time to copy those items. Each row's copy is timed just before the row, as basedpl's speed test times each case's copy. A loop that stops vectorizing takes several times as long and goes over budget. Where the standard library has the function, its time item by item is measured too, for the comparison table in `src/lib.md`, but sets no limit. The rows `Pow.one` and `Root.one` time the scalar forms pair by pair, as a caller with one value at a time runs them. The inner products multiply 256×256 matrices, whose results hold the same number of items. The decompositions factor 256×256 real matrices and 128×128 complex ones, and `lstsq` solves a 4096×16 real one. Sorts, partitions, medians and variances run along the rows of a 256×256 matrix. One sort runs along two lanes of 32,768. A weighted average reduces the first and last axes of a 16×64×64 array. Each of these counts copies of as many items as its matrix holds. The test fails listing every cost over its budget, as its width, name, cost and budget, in the form that `basedpl.bpltests.logged_costs` reads. With `RUSTYMATH_COSTS` set, as in `RUSTYMATH_COSTS=1 cargo test --test speed`, the test writes that table, then every cost in that form, straight to stderr. The test harness captures `println!` output from a passing test, but not these lines. CI's `Test` and `Speed` workflows set it. The test holds the machine's timing lock while it runs, as `timing_lock` describes.
//! Array guards time f32 differences on length-2 lanes and on one long vector, and a compact broadcast repeating each input 64 times. Safe i32-to-i64 widening counts i64 copies. These rows use only the demonstrated dtypes.
use rand::seq::SliceRandom;
use rustymath::*;
use std::{collections::HashMap, hint::black_box, io::Write, ops::Neg, time::Instant};
pub mod support;

const N: usize = 65_536;
/// A function's name, its times in ns per item (its own at f64, the standard library's at f64, its own at f32, and the standard library's at f32), and its own costs in copies at f64 and at f32.
type Compared = (&'static str, [f64; 4], [f64; 2]);
/// A function's name and its costs in copies, at f64 and at f32.
type Costed = (&'static str, [f64; 2]);

/// How many runs each timing takes. Each run reads inputs that no earlier run has read. A branch predictor learns the outcomes of up to tens of thousands of items it has seen. On repeated inputs, a function that branches on its data runs as fast as on sorted data.
const RUNS: usize = 5;

/// The fastest of `RUNS` runs of `run`, in ns for each of `n` items. Run `k` is passed `k`, so it can read its own inputs.
fn fastest(n: usize, mut run: impl FnMut(usize)) -> f64 {
    (0..RUNS).map(|k| {
        let start = Instant::now();
        run(k);
        start.elapsed().as_secs_f64() * 1e9 / n as f64
    }).fold(f64::INFINITY, f64::min)
}

/// Run `k`'s inputs, from `x`, which holds `RUNS` sets of them one after another.
fn set<A>(x: &[A], k: usize) -> &[A] { let n = x.len() / RUNS; &x[k * n..][..n] }

/// The fastest of `RUNS` runs of `f`, which writes `n` outputs into a slice, in ns per item.
fn time_slice<T: Copy + Default>(n: usize, f: impl Fn(usize, &mut [T])) -> f64 {
    let mut out = vec![T::default(); n];
    fastest(n, |k| {
        f(k, &mut out);
        black_box(&mut out);
    })
}

/// The fastest of `RUNS` runs of `f`, which appends `n` outputs to an empty vector, in ns per item.
fn time_appended<T>(n: usize, f: impl Fn(usize, &mut Vec<T>)) -> f64 {
    let mut out = Vec::with_capacity(n);
    fastest(n, |k| {
        out.clear();
        f(k, &mut out);
        black_box(&mut out);
    })
}

/// The time per item of `f` on each of the inputs in `x`.
fn time<A: Copy, T: Copy + Default>(x: &[A], f: impl Fn(A) -> T) -> f64 {
    time_slice(x.len() / RUNS, |k, out| { for (o, &v) in out.iter_mut().zip(black_box(set(x, k))) { *o = f(v); } })
}

/// The time per item of `k`'s slice form on the inputs in `x`.
fn time_unary<F, K: Unary<F>>(x: &[F], k: &K) -> f64 { let n = x.len() / RUNS; time_appended(n, |r, out| { k.many(Arg::Many(black_box(set(x, r))), n, out); }) }

/// The time per item of `k`'s slice form on the pairs of inputs in `x` and `y`.
fn time_binary<F: Copy, K: Binary<F>>(x: &[F], y: &[F], k: &K) -> f64 {
    let n = x.len() / RUNS;
    time_appended(n, |r, out| { k.many(Arg::Many(black_box(set(x, r))), Arg::Many(set(y, r)), n, out); })
}

/// The time per item of `k`'s slice form on the triples of inputs in `x`, `y` and `z`.
fn time_ternary<F: Copy, K: Ternary<F>>(x: &[F], y: &[F], z: &[F], k: &K) -> f64 {
    let n = x.len() / RUNS;
    time_appended(n, |r, out| {
        k.many(Arg::Many(black_box(set(x, r))), Arg::Many(set(y, r)), Arg::Many(set(z, r)), n, out);
    })
}

/// Horner's rule for one item, as a loop over the items would call it with the standard library's `mul_add`.
fn horner<F: Float>(c: &[F], x: F) -> F { c.iter().rev().fold(F::zero(), |y, &k| y.mul_add(x, k)) }

/// `RUNS` sets of `len` inputs from `lo` to `hi`, from the fractional parts of multiples of `step`, in an order shuffled with a fixed seed. Every call shuffles with the same seed, so inputs made together keep their pairs.
fn inputs(lo: f64, hi: f64, step: f64, len: usize) -> Vec<f64> {
    let mut x: Vec<f64> = (0..RUNS * len).map(|i| lo + (hi - lo) * (i as f64 * step).fract()).collect();
    x.shuffle(&mut support::seeded(1));
    x
}
const GOLDEN: f64 = 0.618_033_988_749_894_9;
const SILVER: f64 = 0.414_213_562_373_095_1;
const BRONZE: f64 = 0.302_775_637_731_994_6;

/// `RUNS` sets of `len` inputs from `lo` to `hi`, at f64 and narrowed to f32.
fn widths(lo: f64, hi: f64, step: f64, len: usize) -> (Vec<f64>, Vec<f32>) {
    let x = inputs(lo, hi, step, len);
    let narrowed = x.iter().map(|&x| x as f32).collect();
    (x, narrowed)
}

/// A row of the comparison table in `src/lib.md`, `name`: the times of its own slice form and of the standard library's version, at f64 and then f32, and its own costs in copies. Each width's copy is timed just before its own slice form.
fn compare(
    units: &Units<f64, f32>, name: &'static str, ours64: impl FnOnce() -> f64, std64: impl FnOnce() -> f64, ours32: impl FnOnce() -> f64,
    std32: impl FnOnce() -> f64) -> Compared {
    let ((t64, c64), s64) = (units.0.beside(ours64), std64());
    let ((t32, c32), s32) = (units.1.beside(ours32), std32());
    (name, [t64, s64, t32, s32], [c64, c32])
}

/// The times of `$object`'s slice form and of the standard library's `$std` item by item, at f64 and then f32, on inputs from `$lo` to `$hi`, with its costs against `$units`.
macro_rules! unary {
    ($units:expr; $($object:ident vs $std:ident on $lo:literal to $hi:literal;)+) => {
        [$({
            let (x64, x32) = widths($lo, $hi, GOLDEN, N);
            compare($units, stringify!($object), || time_unary(&x64, &$object), || time(&x64, f64::$std), || time_unary(&x32, &$object), || time(&x32, f32::$std))
        }),+]
    };
}

/// The time per item of `f` on each pair of `x` and `y`, called one pair at a time.
fn time_pairs<F: Copy, T: Copy + Default>(x: &[F], y: &[F], f: impl Fn(F, F) -> T) -> f64 {
    let n = x.len() / RUNS;
    time_slice(n, |r, out| { let (x, y) = (black_box(set(x, r)), set(y, r)); for i in 0..n { out[i] = f(x[i], y[i]) } })
}

/// The times of its slice form and of a reference version pair by pair, on pairs from the ranges `xs` and `ys`, at f64 and then f32. With `one`, it times `k`'s scalar form pair by pair in place of its slice form.
fn binary<K: Binary<f64, Out = f64> + Binary<f32, Out = f32>>(
    units: &Units<f64, f32>, name: &'static str, k: K, one: bool, (xs, ys): ((f64, f64), (f64, f64)), std64: impl Fn(f64, f64) -> f64,
    std32: impl Fn(f32, f32) -> f32) -> Compared {
    let ((x64, x32), (y64, y32)) = (widths(xs.0, xs.1, GOLDEN, N), widths(ys.0, ys.1, SILVER, N));
    let ours64 = || if one { time_pairs(&x64, &y64, |a, b| k.one(a, b)) } else { time_binary(&x64, &y64, &k) };
    let ours32 = || if one { time_pairs(&x32, &y32, |a, b| k.one(a, b)) } else { time_binary(&x32, &y32, &k) };
    compare(units, name, ours64, || time_pairs(&x64, &y64, std64), ours32, || time_pairs(&x32, &y32, std32))
}

/// The times of a reduction or scan of `rustymath::reduce`, and the times of the same taken item by item in order, which the compiler can't vectorize for floats.
fn reductions(units: &Units<f64, f32>) -> [Compared; 4] {
    let (x64, x32) = widths(-1.0, 1.0, GOLDEN, N);
    let fold = |run: &dyn Fn(usize) -> f64| { fastest(N, |k| { black_box(run(k)); }) };
    let mut s64 = Vec::with_capacity(N);
    let mut s32 = Vec::with_capacity(N);
    let scan64 = |x: &[f64], out: &mut Vec<f64>| {
        let mut t = 0.0;
        out.extend(x.iter().map(|&v| { t += v; t }));
    };
    let scan32 = |x: &[f32], out: &mut Vec<f32>| {
        let mut t = 0.0;
        out.extend(x.iter().map(|&v| { t += v; t }));
    };
    let max64 = |x: &[f64], out: &mut Vec<f64>| {
        let mut t = f64::NEG_INFINITY;
        out.extend(x.iter().map(|&v| { t = t.max(v); t }));
    };
    let max32 = |x: &[f32], out: &mut Vec<f32>| {
        let mut t = f32::NEG_INFINITY;
        out.extend(x.iter().map(|&v| { t = t.max(v); t }));
    };
    [
        compare(
            units, "reduce::sum", || fold(&|k| reduce::sum(black_box(set(&x64, k))).0), || fold(&|k| black_box(set(&x64, k)).iter().sum::<f64>()),
            || fold(&|k| f64::from(reduce::sum(black_box(set(&x32, k))).0)), || fold(&|k| f64::from(black_box(set(&x32, k)).iter().sum::<f32>()))), compare(
            units, "reduce::max", || fold(&|k| reduce::max(black_box(set(&x64, k)))),
            || fold(&|k| black_box(set(&x64, k)).iter().copied().fold(f64::NEG_INFINITY, f64::max)),
            || fold(&|k| f64::from(reduce::max(black_box(set(&x32, k))))),
            || fold(&|k| f64::from(black_box(set(&x32, k)).iter().copied().fold(f32::NEG_INFINITY, f32::max)))), compare(units, "reduce::cumsum", || {
                time_appended(N, |k, out| { reduce::cumsum(set(&x64, k), None, out); })
            }, || {
                fastest(N, |k| {
                    s64.clear();
                    scan64(black_box(set(&x64, k)), &mut s64);
                    black_box(&s64);
                })
            }, || {
                time_appended(N, |k, out| { reduce::cumsum(set(&x32, k), None, out); })
            }, || {
                fastest(N, |k| {
                    s32.clear();
                    scan32(black_box(set(&x32, k)), &mut s32);
                    black_box(&s32);
                })
            }), compare(units, "reduce::cummax", || time_appended(N, |k, out| reduce::cummax(set(&x64, k), None, out)), || {
                fastest(N, |k| {
                    s64.clear();
                    max64(black_box(set(&x64, k)), &mut s64);
                    black_box(&s64);
                })
            }, || time_appended(N, |k, out| reduce::cummax(set(&x32, k), None, out)), || {
                fastest(N, |k| {
                    s32.clear();
                    max32(black_box(set(&x32, k)), &mut s32);
                    black_box(&s32);
                })
            }),
    ]
}

/// Writes each item of `x`, negated, to `out`. It moves the data that a copy does, in a vector loop as the functions do. The compiler would turn a plain copy into a call to `memcpy`. The speed of `memcpy` varies with the CPU and with how the two buffers align.
fn negate<T: Copy + Neg<Output = T>>(x: &[T], out: &mut [T]) { for (o, &v) in out.iter_mut().zip(x) { *o = -v } }

/// The unit of every budget, called a copy: the time to `negate` `N` items of type `T`, each `v`, in ns per item. A nonzero `v` makes `vec!` write every item. Zeros could come from untouched pages. Every read of an untouched page finds the same page of zeros in the cache.
struct Unit<T>(Vec<T>);

impl<T: Copy + Default + Neg<Output = T>> Unit<T> {
    fn new(v: T) -> Self { Self(vec![v; RUNS * N]) }

    /// The time per item that `run` gives, in ns and in copies. The copy is timed just before `run`, as basedpl's speed test times each case's copy.
    fn beside(&self, run: impl FnOnce() -> f64) -> (f64, f64) {
        let copy = time_slice(N, |k, out| negate(black_box(set(&self.0, k)), out));
        let t = run();
        (t, t / copy)
    }
}

/// A unit at f64 and one at f32, for real or for complex items.
struct Units<A, B>(Unit<A>, Unit<B>);

impl<A: Copy + Default + Neg<Output = A>, B: Copy + Default + Neg<Output = B>> Units<A, B> {
    /// The costs in copies of `run64` and `run32`, each a time per item, against the unit at its own width.
    fn costs(&self, run64: impl FnOnce() -> f64, run32: impl FnOnce() -> f64) -> [f64; 2] { [self.0.beside(run64).1, self.1.beside(run32).1] }
}

/// The functions that the standard library lacks, with their costs in copies of floats against `units`. The incomplete gamma and beta functions iterate, and run on 1,024 inputs. The others run on 16,384.
fn special(units: &Units<f64, f32>) -> [Costed; 11] {
    let (many, few) = (N / 4, N / 64);
    macro_rules! unary_copies {
        ($k:expr, $lo:literal to $hi:literal) => {{
            let (x64, x32) = widths($lo, $hi, GOLDEN, many);
            units.costs(|| time_unary(&x64, &$k), || time_unary(&x32, &$k))
        }};
    }
    macro_rules! binary_copies {
        ($k:expr, $xs:expr, $ys:expr) => {{
            let (xs, ys): ((f64, f64), (f64, f64)) = ($xs, $ys);
            let ((x64, x32), (y64, y32)) = (widths(xs.0, xs.1, GOLDEN, few), widths(ys.0, ys.1, SILVER, few));
            units.costs(|| time_binary(&x64, &y64, &$k), || time_binary(&x32, &y32, &$k))
        }};
    }
    macro_rules! ternary_copies {
        ($k:expr, $zs:expr) => {{
            let zs: (f64, f64) = $zs;
            let ((x64, x32), (y64, y32), (z64, z32)) = (widths(0.5, 20.0, GOLDEN, few), widths(0.5, 20.0, SILVER, few), widths(zs.0, zs.1, BRONZE, few));
            units.costs(|| time_ternary(&x64, &y64, &z64, &$k), || time_ternary(&x32, &y32, &z32, &$k))
        }};
    }
    [
        ("Gamma", unary_copies!(Gamma, 0.1 to 20.0)), ("Lgamma", unary_copies!(Lgamma, 0.1 to 1000.0)), ("Erf", unary_copies!(Erf, -5.0 to 5.0)),
        ("Erfc", unary_copies!(Erfc, -5.0 to 10.0)), ("Erfinv", unary_copies!(Erfinv, -0.999 to 0.999)), ("Erfcinv", unary_copies!(Erfcinv, 0.001 to 1.999)),
        ("Gammainc", binary_copies!(Gammainc, (0.5, 20.0), (0.0, 30.0))), ("Gammaincc", binary_copies!(Gammaincc, (0.5, 20.0), (0.0, 30.0))),
        ("Gammaincinv", binary_copies!(Gammaincinv, (0.5, 20.0), (0.001, 0.999))), ("Betainc", ternary_copies!(Betainc, (0.0, 1.0))),
        ("Betaincinv", ternary_copies!(Betaincinv, (0.001, 0.999))),
    ]
}

/// Complex function objects, with their costs in copies of the same number of complex items.
fn complex(units: &Units<num_complex::Complex<f64>, num_complex::Complex<f32>>) -> [Costed; 12] {
    use num_complex::Complex;
    let n = N / 4;
    let zs = |re: (f64, f64), im: (f64, f64), step: f64| {
        let z64: Vec<_> = inputs(re.0, re.1, step, n).into_iter().zip(inputs(im.0, im.1, SILVER, n)).map(|(x, y)| Complex::new(x, y)).collect();
        let z32: Vec<_> = z64.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect();
        (z64, z32)
    };
    macro_rules! unary_copies {
        ($k:expr, $re:expr, $im:expr) => {{
            let (z64, z32) = zs($re, $im, GOLDEN);
            units.costs(|| time_unary(&z64, &$k), || time_unary(&z32, &$k))
        }};
    }
    let square = (-2.0, 2.0);
    let (w64, w32) = zs(square, square, BRONZE);
    let (z64, z32) = zs(square, square, GOLDEN);
    [
        ("complex Exp", unary_copies!(Exp, square, square)), ("complex Ln", unary_copies!(Ln, square, square)),
        ("complex Sqrt", unary_copies!(Sqrt, square, square)), ("complex Sin", unary_copies!(Sin, square, square)),
        ("complex Tan", unary_copies!(Tan, square, square)), ("complex Asin", unary_copies!(Asin, square, square)),
        ("complex Atanh", unary_copies!(Atanh, square, square)), ("complex Lgamma", unary_copies!(Lgamma, (0.1, 20.0), (-10.0, 10.0))),
        ("complex Gamma", unary_copies!(Gamma, (0.1, 20.0), (-10.0, 10.0))), ("complex Abs", unary_copies!(Abs, square, square)),
        ("complex Sqrt1mx2", unary_copies!(Sqrt1mx2, square, square)),
        ("complex Pow", units.costs(|| time_binary(&z64, &w64, &Pow), || time_binary(&z32, &w32, &Pow))),
    ]
}

/// Inner products of 256×256 matrices, whose results hold `N` items. In the integer rows, the 64-bit column times items large enough to need the 64-bit kernels, and the 32-bit column items small enough for the 32-bit kernels. Both count copies of 64-bit items.
fn inner_products(units: &Units<f64, f32>) -> [Costed; 6] {
    use linalg::*;
    let size = [256; 3];
    let first = |(x, y): (Vec<f64>, Vec<f32>)| (set(&x, 0).to_vec(), set(&y, 0).to_vec());
    let ((a64, a32), (b64, b32)) = (first(widths(-1.0, 1.0, GOLDEN, N)), first(widths(-1.0, 1.0, SILVER, N)));
    let run = |k: &dyn Fn() -> usize| { fastest(N, |_| { black_box(k()); }) };
    macro_rules! copies {
        ($k:expr) => {
            units.costs(|| run(&|| $k.inner(black_box(&a64), black_box(&b64), size).len()), || run(&|| $k.inner(black_box(&a32), black_box(&b32), size).len()))
        };
    }
    let whole = |x: &[f64], scale: f64| x.iter().map(|&v| (v * scale) as i64).collect::<Vec<_>>();
    let (large, small) = ([whole(&a64, 1e6), whole(&b64, 1e6)], [whole(&a64, 100.0), whole(&b64, 100.0)]);
    let wide = large.clone().map(|x| x.into_iter().map(|v| v << 20).collect::<Vec<_>>());
    macro_rules! integer_copies {
        ($k:expr, $large:expr, $small:expr) => {
            [$large, $small].map(|[a, b]: &[Vec<i64>; 2]| units.0.beside(|| run(&|| $k.inner(black_box(a.as_slice()), black_box(b.as_slice()), size).len())).1)
        };
    }
    [
        ("linalg::PlusTimes", copies!(PlusTimes)), ("linalg::MaxPlus", copies!(MaxPlus)), ("linalg::MinPlus", copies!(MinPlus)),
        ("linalg::PlusEqual", copies!(PlusEqual)), ("linalg::PlusTimes at i64", integer_copies!(PlusTimes, &large, &small)),
        ("linalg::PlusEqual at i64", integer_copies!(PlusEqual, &wide, &small)),
    ]
}

/// Orderings and statistics along the rows of a 256×256 matrix, a sort along two lanes of 32,768, and a weighted average over the first and last axes of a 16×64×64 array.
fn orderings(units: &Units<f64, f32>) -> [Costed; 6] {
    use buffer::{Floats, View};
    let (x64, x32) = widths(-1.0, 1.0, GOLDEN, N);
    let (w64, w32) = widths(0.0, 1.0, SILVER, N);
    let rows = [256, 256, 1];
    let run = |k: &dyn Fn(usize)| fastest(N, |r| k(r));
    macro_rules! copies {
        (|$x:ident| $body:expr) => {
            units.costs(|| run(&|r| { let $x = View::Floats(Floats::F64(set(&x64, r))); black_box($body); }),
                || run(&|r| { let $x = View::Floats(Floats::F32(set(&x32, r))); black_box($body); }))
        };
    }
    let layout = layout::Layout::from([16, 64, 64]);
    let axes = stats::Axes::new(&layout, &[0, 2]).unwrap();
    let agreement = agreement::Agreement::new(&layout, &layout).unwrap();
    let mappings = [&agreement.left, &agreement.right];
    [
        ("search::view_sort_along", copies!(|x| search::view_sort_along(x, rows, false))),
        ("search::view_sort_along long lanes", copies!(|x| search::view_sort_along(x, [2, N / 2, 1], false))),
        ("search::view_partition_along", copies!(|x| search::view_partition_along(x, rows, 128, false))),
        ("search::view_quantiles_along", copies!(|x| search::view_quantiles_along(x, rows, &[0.5]).unwrap())),
        ("stats::variance", units.costs(|| run(&|r| { black_box(stats::variance(set(&x64, r), rows, 0)); }),
            || run(&|r| { black_box(stats::variance(set(&x32, r), rows, 0)); }))),
        ("stats::average", units.costs(|| run(&|r| { black_box(stats::average(set(&x64, r), set(&w64, r), &axes, mappings)); }),
            || run(&|r| { black_box(stats::average(set(&x32, r), set(&w32, r), &axes, mappings)); }))),
    ]
}

/// Allocated array results at the demonstrated dtypes. Differences and widening count copies of `N` input items; broadcasting counts `N` output items.
fn arrays(unit: &Unit<f32>) -> [(&'static str, &'static str, f64); 4] {
    use array::{Array, ArrayView};
    use buffer::{Buffer, DType, Floats, Ints, View, Width};
    use layout::Layout;
    let x: Vec<f32> = (0..RUNS * N).map(|i| i as f32).collect();
    let integers: Vec<i32> = (0..RUNS * N).map(|i| i as i32).collect();
    let differences = |layout: &Layout, axis| unit.beside(|| fastest(N, |r| {
        black_box(Array::differences(ArrayView::compact(layout, View::Floats(Floats::F32(black_box(set(&x, r))))), axis).unwrap());
    })).1;
    let source = Layout::from([N / 64]);
    [
        ("f32", "Array::differences short lanes", differences(&Layout::from([N / 2, 2]), 1)),
        ("f32", "Array::differences", differences(&Layout::from([N]), 0)),
        ("f32", "Array::broadcasted", unit.beside(|| fastest(N, |r| {
            black_box(Array::broadcasted(ArrayView::compact(&source, View::Floats(Floats::F32(black_box(&set(&x, r)[..N / 64])))), &[N / 64, 64]).unwrap());
        })).1),
        ("i64", "Buffer::cast i32", Unit::new(1i64).beside(|| fastest(N, |r| {
            black_box(Buffer::cast(View::Integers(Ints::I32(black_box(set(&integers, r)))), DType::Integer(Width::I64)).unwrap());
        })).1),
    ]
}

/// Decompositions of 256×256 real matrices and 128×128 complex ones, and a least-squares solution for a 4096×16 real matrix. Each cost counts copies of as many items as the matrix holds. `s` is symmetric positive definite, and `h` is Hermitian.
fn decompositions(units: &Units<f64, f32>, complex_units: &Units<num_complex::Complex<f64>, num_complex::Complex<f32>>) -> Vec<Costed> {
    use linalg::*;
    use num_complex::Complex;
    let mut seed = 1u64;
    let mut random = |len: usize| -> Vec<f64> {
        (0..len).map(|_| {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (seed >> 11) as f64 / (1u64 << 52) as f64 - 1.0
        }).collect()
    };
    let (a64, t64, b64, c64) = (random(N), random(N), random(256), random(4096));
    let at: Vec<f64> = (0..N).map(|k| a64[k % 256 * 256 + k / 256]).collect();
    let mut s64 = linalg::PlusTimes.inner(&a64, &at, [256; 3]);
    for i in 0..256 { s64[i * 257] += 256.0 }
    let narrow = |x: &[f64]| x.iter().map(|&v| v as f32).collect::<Vec<f32>>();
    let (a32, t32, b32, c32, s32) = (narrow(&a64), narrow(&t64), narrow(&b64), narrow(&c64), narrow(&s64));
    let run = |items: usize, k: &dyn Fn() -> usize| { fastest(items, |_| { black_box(k()); }) };
    let real = |k64: &dyn Fn() -> usize, k32: &dyn Fn() -> usize| units.costs(|| run(N, k64), || run(N, k32));
    let n = 128;
    let z64: Vec<Complex<f64>> = a64.iter().zip(&t64).take(n * n).map(|(&re, &im)| Complex::new(re, im)).collect();
    let h64: Vec<Complex<f64>> = (0..n * n).map(|k| z64[k] + z64[k % n * n + k / n].conj()).collect();
    let zb64: Vec<Complex<f64>> = b64.iter().zip(&c64).take(n).map(|(&re, &im)| Complex::new(re, im)).collect();
    let narrowed = |x: &[Complex<f64>]| x.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect::<Vec<_>>();
    let (z32, h32, zb32) = (narrowed(&z64), narrowed(&h64), narrowed(&zb64));
    let complex = |k64: &dyn Fn() -> usize, k32: &dyn Fn() -> usize| complex_units.costs(|| run(n * n, k64), || run(n * n, k32));
    vec![
        ("linalg::Lu::solve", real(&|| lu(black_box(&a64), 256).solve(&b64, 1).len(), &|| lu(black_box(&a32), 256).solve(&b32, 1).len())),
        ("linalg::Lu::inverse", real(&|| lu(black_box(&a64), 256).inverse().len(), &|| lu(black_box(&a32), 256).inverse().len())),
        ("linalg::cholesky", real(&|| cholesky(black_box(&s64), 256).map_or(0, |l| l.len()), &|| cholesky(black_box(&s32), 256).map_or(0, |l| l.len()))),
        ("linalg::qr", real(&|| qr(black_box(&a64), [256, 256]).0.len(), &|| qr(black_box(&a32), [256, 256]).0.len())),
        ("linalg::svd", real(&|| svd(black_box(&a64), [256, 256]).u.len(), &|| svd(black_box(&a32), [256, 256]).u.len())),
        ("linalg::eigh", real(&|| eigh(black_box(&s64), 256).map_or(0, |f| f.vectors.len()), &|| eigh(black_box(&s32), 256).map_or(0, |f| f.vectors.len()))),
        ("linalg::eig", real(&|| eig(black_box(&a64), 256).map_or(0, |f| f.vectors.len()), &|| eig(black_box(&a32), 256).map_or(0, |f| f.vectors.len()))), (
            "linalg::eigenvalues",
            real(&|| eigenvalues(black_box(&a64), 256).map_or(0, |v| v.len()), &|| eigenvalues(black_box(&a32), 256).map_or(0, |v| v.len())),
        ), ("linalg::lstsq", real(&|| lstsq(black_box(&t64), &c64, [4096, 16], 1).0.len(), &|| lstsq(black_box(&t32), &c32, [4096, 16], 1).0.len())),
        ("complex linalg::Lu::solve", complex(&|| lu(black_box(&z64), n).solve(&zb64, 1).len(), &|| lu(black_box(&z32), n).solve(&zb32, 1).len())),
        ("complex linalg::qr", complex(&|| qr(black_box(&z64), [n, n]).0.len(), &|| qr(black_box(&z32), [n, n]).0.len())),
        ("complex linalg::svd", complex(&|| svd(black_box(&z64), [n, n]).u.len(), &|| svd(black_box(&z32), [n, n]).u.len())), (
            "complex linalg::eigh",
            complex(&|| eigh(black_box(&h64), n).map_or(0, |f| f.vectors.len()), &|| eigh(black_box(&h32), n).map_or(0, |f| f.vectors.len())),
        ), (
            "complex linalg::eig",
            complex(&|| eig(black_box(&z64), n).map_or(0, |f| f.vectors.len()), &|| eig(black_box(&z32), n).map_or(0, |f| f.vectors.len())),
        ),
    ]
}

/// The budget of each function at each item type, keyed by `(type, name)`. Each section of `text` starts with a type in brackets, and each line under it holds a name and its budget, separated by a tab. Lines that start with `#` are comments.
fn budgets(text: &str) -> HashMap<(&str, &str), f64> {
    let mut width = "";
    text.lines().filter(|line| !line.is_empty() && !line.starts_with('#')).filter_map(|line| {
        if let Some(w) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) { width = w; return None; }
        let (name, budget) = line.split_once('\t').expect("a budget line holds a name, a tab and a budget");
        Some(((width, name), budget.parse().expect("a budget is a number")))
    }).collect()
}

/// The file whose lock keeps timings and builds apart on this machine, as DEV.md describes. Timings take it exclusively. Builds take it shared.
const TIMING_LOCK: &str = "/tmp/aai-ws-timing.lock";

/// Takes the timing lock exclusively, and returns the file that holds it until dropped. It waits up to 5 minutes while a build or another timing holds the lock. The operating system releases the lock when the test ends, even by a crash. On macOS it also sets the calling thread's QoS class to user-interactive. macOS then keeps the thread, which runs the timing, on the performance cores. The thread's calls also run on it alone, because the budgets are for one thread. After waiting 30 seconds it prints, once, the other processes that have the lock's file open.
fn timing_lock() -> std::fs::File {
    #[cfg(target_vendor = "apple")]
    unsafe { libc::pthread_set_qos_class_self_np(libc::qos_class_t::QOS_CLASS_USER_INTERACTIVE, 0) };
    rustymath::parallel::set_threads(1);
    let file = std::fs::File::options().create(true).append(true).open(TIMING_LOCK).expect("the timing lock's file opens");
    let (start, mut waiting, mut reported) = (Instant::now(), false, false);
    loop {
        match file.try_lock() {
            Ok(()) => return file, Err(std::fs::TryLockError::WouldBlock) => {} Err(std::fs::TryLockError::Error(e)) => panic!("{TIMING_LOCK}: {e}")
        }
        assert!(start.elapsed().as_secs() < 300, "a build or another timing has held {TIMING_LOCK} for 5 minutes");
        if !waiting { writeln!(std::io::stderr(), "waiting for a build or another timing to release {TIMING_LOCK}").unwrap(); }
        if !reported && start.elapsed().as_secs() >= 30 {
            writeln!(std::io::stderr(), "{TIMING_LOCK} is still taken after 30 seconds. Processes with it open: {}", lock_users()).unwrap();
            reported = true;
        }
        waiting = true;
        std::thread::sleep(std::time::Duration::from_millis(250));
    }
}

/// The processes other than this one that have the timing lock's file open, as `pid command` pairs from `lsof`. On macOS, where `lsof` doesn't show `flock` holds, the list also includes any other process waiting for the lock.
fn lock_users() -> String {
    let Ok(out) = std::process::Command::new("lsof").args(["+c", "0", "-F", "pc", TIMING_LOCK]).output() else {
        return "unknown, because lsof didn't run".into();
    };
    let (text, me) = (String::from_utf8_lossy(&out.stdout), std::process::id().to_string());
    let (mut users, mut pid) = (Vec::new(), "");
    for line in text.lines() {
        match line.split_at_checked(1) { Some(("p", p)) => pid = p, Some(("c", c)) if pid != me => users.push(format!("{pid} {c}")), _ => {} }
    }
    if users.is_empty() { "none".into() } else { users.join(", ") }
}
#[test]
fn speed() {
    let _lock = timing_lock();
    let units = Units(Unit::new(1.0f64), Unit::new(1.0f32));
    let complex_units = Units(Unit::new(num_complex::Complex::new(1.0f64, 1.0)), Unit::new(num_complex::Complex::new(1.0f32, 1.0)));
    let mut compared = unary! {
        &units;
        Exp vs exp on -20.0 to 20.0;
        Expm1 vs exp_m1 on -1.0 to 1.0;
        Ln vs ln on 0.001 to 1000.0;
        Log1p vs ln_1p on -0.5 to 1000.0;
        Log2 vs log2 on 0.001 to 1000.0;
        Log10 vs log10 on 0.001 to 1000.0;
        Sin vs sin on -10.0 to 10.0;
        Cos vs cos on -10.0 to 10.0;
        Tan vs tan on -10.0 to 10.0;
        Asin vs asin on -1.0 to 1.0;
        Acos vs acos on -1.0 to 1.0;
        Atan vs atan on -10.0 to 10.0;
        Sinh vs sinh on -10.0 to 10.0;
        Cosh vs cosh on -10.0 to 10.0;
        Tanh vs tanh on -10.0 to 10.0;
        Asinh vs asinh on -1000.0 to 1000.0;
        Acosh vs acosh on 1.0 to 1000.0;
        Atanh vs atanh on -0.999 to 0.999;
        Cbrt vs cbrt on -1000.0 to 1000.0;
    }.to_vec();
    let (powers, roots) = (((0.0, 4.0), (-30.0, 30.0)), ((5.0, 5.0), (0.0, 1000.0)));
    let nth = |n: f64, y: f64| y.powf(1.0 / n);
    let nth32 = |n: f32, y: f32| y.powf(1.0 / n);
    compared.push(binary(&units, "Pow", Pow, false, powers, f64::powf, f32::powf));
    compared.push(binary(&units, "Pow.one", Pow, true, powers, f64::powf, f32::powf));
    compared.push(binary(&units, "Hypot", Hypot, false, ((-1000.0, 1000.0), (-1000.0, 1000.0)), f64::hypot, f32::hypot));
    compared.push(binary(&units, "Root", Root, false, roots, nth, nth32));
    compared.push(binary(&units, "Root.one", Root, true, roots, nth, nth32));
    compared.push(binary(&units, "Maximum", reduce::Maximum, false, ((-1.0, 1.0), (-1.0, 1.0)), f64::max, f32::max));
    compared.extend(reductions(&units));
    // A polynomial of degree 7 at the negative of each input, against `horner` for one input at a time. `black_box` hides the degree, which a caller rarely knows when it compiles.
    let c64 = [0.5, -1.25, 0.75, 2.0, -0.125, 0.0625, -1.5, 0.25];
    let c32 = c64.map(|k| k as f32);
    let (c64, c32): (&[f64], &[f32]) = (black_box(&c64), black_box(&c32));
    let (x64, x32) = widths(-2.0, 2.0, GOLDEN, N);
    compared.push(compare(&units, "polynomial::evaluate", || {
        time_slice(N, |k, out| { negate(black_box(set(&x64, k)), out); polynomial::evaluate(c64, out) })
    }, || time(&x64, |x| horner(c64, x)), || {
        time_slice(N, |k, out| { negate(black_box(set(&x32, k)), out); polynomial::evaluate(c32, out) })
    }, || time(&x32, |x| horner(c32, x))));
    let costed: Vec<_> = compared.iter().map(|&(name, _, costs)| (name, costs)).chain(special(&units)).chain(complex(&complex_units))
        .chain(inner_products(&units)).chain(orderings(&units)).chain(decompositions(&units, &complex_units))
        .flat_map(|(name, costs)| ["f64", "f32"].into_iter().zip(costs).map(move |(width, cost)| (width, name, cost)))
        .chain(arrays(&units.1)).collect();
    let print_all = std::env::var_os("RUSTYMATH_COSTS").is_some();
    let mut stderr = std::io::stderr();
    if print_all {
        writeln!(stderr, "| Function | f64 | std f64 | f32 | std f32 |\n|---|---|---|---|---|").unwrap();
        for (name, t, _) in compared.iter() { writeln!(stderr, "| `{name}` | {:.2} | {:.2} | {:.2} | {:.2} |", t[0], t[1], t[2], t[3]).unwrap(); }
        writeln!(stderr).unwrap();
    }
    let budgets = budgets(include_str!("speed.txt"));
    let mut failures = Vec::new();
    for &(width, name, cost) in &costed {
        let budget = budgets.get(&(width, name)).copied();
        let line = match budget {
            Some(budget) => format!("{width} {name}: {cost:.2}, budget {budget}"), None => format!("{width} {name}: {cost:.2}, with no budget in speed.txt")
        };
        if print_all { writeln!(stderr, "{line}").unwrap(); }
        if budget.is_none_or(|budget| cost > budget) { failures.push(line); }
    }
    for &(width, name) in budgets.keys().filter(|&&(w, n)| !costed.iter().any(|&(width, name, _)| (width, name) == (w, n))) {
        failures.push(format!("{width} {name} has a budget in speed.txt, but the test doesn't time it"));
    }
    assert!(failures.is_empty(), "{failures:#?}");
}
