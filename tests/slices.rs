pub mod support;

mod parallel {
    use rustymath::parallel::{ranges, set_threads, split, threads};
    use std::{sync::{Mutex, atomic::{AtomicBool, Ordering}}, thread, time::Duration};

    #[test]
    fn ranges_and_outputs() {
        let empty = Mutex::new(Vec::new());
        assert!(!ranges(0, 8, &|range| {
            empty.lock().unwrap().push(range);
            false
        }));
        assert_eq!(empty.into_inner().unwrap(), [0..0]);
        assert!(std::panic::catch_unwind(|| ranges(0, 0, &|_| true)).is_err());
        for limit in [1, 4] {
            let previous = set_threads(limit);
            let (caller, worker, calls) = (thread::current().id(), AtomicBool::new(false), Mutex::new(Vec::new()));
            let mut out = vec![usize::MAX; 129];
            let ok = split(&mut out, 8, &|start, items| {
                calls.lock().unwrap().push(start..start + items.len());
                // Give the first unit enough work to select pieces, without asserting an elapsed time.
                if start == 0 && limit > 1 { thread::sleep(Duration::from_millis(1)); }
                if thread::current().id() != caller {
                    worker.store(true, Ordering::Relaxed);
                    assert_eq!(threads(), 1);
                    assert!(ranges(17, 1, &|range| range == (0..17)));
                }
                for (i, item) in items.iter_mut().enumerate() { *item = start + i; }
                start != 0
            });
            set_threads(previous);
            assert!(!ok);
            assert_eq!(out, (0..129).collect::<Vec<_>>());
            let mut calls = calls.into_inner().unwrap();
            calls.sort_by_key(|range| range.start);
            let mut end = 0;
            for range in &calls {
                assert_eq!(range.start, end);
                end = range.end;
            }
            assert_eq!(end, 129);
            if limit == 1 { assert_eq!(calls, [0..129]); }
            #[cfg(not(target_arch = "wasm32"))]
            if limit > 1 { assert!(worker.load(Ordering::Relaxed)); }
        }
    }
}

mod forms {
    use half::f16;
    use num_complex::Complex;
    use rustymath::{reduce::{Maximum, Minimum}, *};
    use std::fmt::Debug;

    /// Whether `a` and `b` agree: both NaN, the same value with the same sign, or finite and nonzero within 8 units in the last place of each other. A scalar form and its slice form can round differently, as `Pow`'s do at f32, where only the scalar form reads tables. Zeros, infinities and signs must match exactly.
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

    /// `k.many` against `k.one` for every pairing of `One` and `Many` among `items`. It must append after what `out` holds, and say whether every item succeeded, as `checked` does item by item. The NaN rule must agree too. That catches a function marked total that gives NaN from arguments that aren't NaN.
    fn binary<F: Float + Debug, K: Binary<F, Out = F>>(k: K, name: &str) {
        let items = items::<F>();
        let check = |x: Arg<'_, F>, y: Arg<'_, F>, n: usize| {
            let at = |a: Arg<'_, F>, i| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
            let mut out = vec![F::one()];
            let ok = k.many(x, y, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()), "{name}");
            for i in 0..n { assert!(same(out[i + 1], k.one(at(x, i), at(y, i))), "{name}({:?}, {:?})", at(x, i), at(y, i)) }
            assert_eq!(ok, (0..n).all(|i| k.checked(at(x, i), at(y, i)).is_some()), "{name}");
            assert_eq!(ok, (0..n).all(|i| !out[i + 1].is_nan() || [at(x, i), at(y, i)].iter().any(|v| v.is_nan())), "{name} breaks the NaN rule");
        };
        let (xs, ys): (Vec<F>, Vec<F>) = items.iter().flat_map(|&x| items.iter().map(move |&y| (x, y))).unzip();
        check(Arg::Many(&xs), Arg::Many(&ys), xs.len());
        for &v in &items {
            check(Arg::One(v), Arg::Many(&items), items.len());
            check(Arg::Many(&items), Arg::One(v), items.len());
        }
    }

    /// `k.many` against `k.one` for each of `items`, and for each item as `Arg::One`, as for `binary`.
    fn unary<F: Float + Debug, K: Unary<F, Out = F>>(k: K, name: &str, items: &[F]) {
        let check = |x: Arg<'_, F>, n: usize| {
            let at = |i| match x { Arg::One(v) => v, Arg::Many(s) => s[i] };
            let mut out = vec![F::one()];
            let ok = k.many(x, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()), "{name}");
            for i in 0..n { assert!(same(out[i + 1], k.one(at(i))), "{name}({:?})", at(i)) }
            assert_eq!(ok, (0..n).all(|i| k.checked(at(i)).is_some()), "{name}");
            assert_eq!(ok, (0..n).all(|i| !out[i + 1].is_nan() || at(i).is_nan()), "{name} breaks the NaN rule");
        };
        check(Arg::Many(items), items.len());
        for &v in items { check(Arg::One(v), 3) }
    }

    /// `k.many` against `k.one` for every triple of `items`, as `Many` arguments and with each argument in turn as `One`, as for `binary`.
    fn ternary<F: Float + Debug, K: Ternary<F, Out = F>>(k: K, name: &str) {
        let items = items::<F>();
        let at = |a: Arg<'_, F>, i| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
        let check = |[x, y, z]: [Arg<'_, F>; 3], n: usize| {
            let mut out = vec![F::one()];
            let ok = k.many(x, y, z, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()), "{name}");
            for i in 0..n {
                assert!(same(out[i + 1], k.one(at(x, i), at(y, i), at(z, i))), "{name}({:?}, {:?}, {:?})", at(x, i), at(y, i), at(z, i))
            }
            assert_eq!(ok, (0..n).all(|i| k.checked(at(x, i), at(y, i), at(z, i)).is_some()), "{name}");
            assert_eq!(ok, (0..n).all(|i| !out[i + 1].is_nan() || [at(x, i), at(y, i), at(z, i)].iter().any(|v| v.is_nan())), "{name} breaks the NaN rule");
        };
        let (mut xs, mut ys, mut zs) = (Vec::new(), Vec::new(), Vec::new());
        for &x in &items {
            for &y in &items {
                for &z in &items {
                    xs.push(x);
                    ys.push(y);
                    zs.push(z);
                }
            }
        }
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
                $(unary::<f64, _>($u, stringify!($u), &items::<f64>()); unary::<f32, _>($u, stringify!($u), &items::<f32>()); unary::<f16, _>($u, stringify!($u), &items::<f16>());)+
                $(binary::<f64, _>($b, stringify!($b)); binary::<f32, _>($b, stringify!($b)); binary::<f16, _>($b, stringify!($b));)+
                $(ternary::<f64, _>($t, stringify!($t)); ternary::<f32, _>($t, stringify!($t)); ternary::<f16, _>($t, stringify!($t));)+
            };
        }
        each_width!(
            unary: Exp, Expm1, Ln, Log1p, Log2, Log10, Sin, Cos, Tan, Asin, Acos, Atan, Sinh, Cosh, Tanh, Asinh, Acosh, Atanh, Cbrt, Gamma,
                Lgamma, Erf, Erfc, Erfinv, Erfcinv, LambertW, Sqrt, Sqrt1mx2, Sqrt1px2, Sqrtx2m1, Abs, Sign, Angle;
            binary: Pow, Root, Hypot, Atan2, Gammainc, Gammaincc, Gammaincinv, Binom, Logb, Maximum, Minimum; ternary: Betainc, Betaincinv
        );
        // `Erfinv`'s and `Erfcinv`'s f64 slice forms place each block's items by region. Several blocks of uniform items, in random order, reach every region.
        use rand::RngExt;
        let mut rng = crate::support::seeded(1);
        let q: Vec<f64> = (0..2000).map(|_| 2.0 * rng.random::<f64>()).collect();
        unary(Erfcinv, "Erfcinv", &q);
        unary(Erfinv, "Erfinv", &q.iter().map(|&q| q - 1.0).collect::<Vec<_>>());
    }

    /// An output type other than the item type, at each float type, f16 included, and at concrete complex types. A `bool` output is never NaN, so it never fails.
    #[test]
    fn outputs_can_differ_from_items() {
        use rustymath::tolerant::{Greater, Tolerance};
        fn positive<F: Float>(x: &[F]) -> Vec<bool> where Greater: Binary<F, Out = bool> {
            let mut out = Vec::new();
            assert!(Greater(Tolerance::EXACT).many(Arg::Many(x), Arg::One(F::zero()), x.len(), &mut out));
            out
        }
        assert_eq!(positive(&[1.0f64, -1.0, f64::NAN]), [true, false, false]);
        assert_eq!(positive(&[f16::from_f32(2.0), f16::NAN]), [true, false]);
        assert_eq!(Greater(Tolerance::EXACT).checked(f32::NAN, 0.0), Some(false));
        let mut out = Vec::new();
        assert!(Abs.many(Arg::Many(&[Complex::new(3.0f32, 4.0), Complex::new(f32::INFINITY, f32::NAN)]), 2, &mut out));
        assert_eq!(out, [5.0, f32::INFINITY]);
    }
}

mod reductions {
    use half::f16;
    use num_complex::Complex;
    use rustymath::reduce::*;

    /// Each reduction and scan against the same taken item by item, in order: sums and running sums of `sums`, and the product of `prods`. Their items keep every sum and product exact at every width, so any grouping gives the same result.
    fn exact<T: Copy + PartialEq + std::fmt::Debug, S: Copy + PartialEq + std::fmt::Debug>(
        sums: &[T], prods: &[T], add: impl Fn(S, T) -> Option<S>, mul: impl Fn(S, T) -> Option<S>, zero: S, one: S) where Sum: Reduction<T, Total = S>,
        Product: Reduction<T, Total = S> {
        let mut total = Some(zero);
        let running: Vec<_> = sums.iter().map(|&v| { total = total.and_then(|t| add(t, v)); total.unwrap() }).collect();
        assert_eq!(sum(sums), (total.unwrap(), true));
        let mut out = Vec::new();
        assert!(cumsum(sums, None, &mut out));
        assert_eq!(out, running);
        assert_eq!(prod(prods), (prods.iter().try_fold(one, |t, &v| mul(t, v)).unwrap(), true));
    }

    #[test]
    fn folds_match_item_by_item() {
        // 101 items cover the running results, the remainder, and the scans' groups of four or sixteen and their tails. Ten 2s keep the product within f16.
        let sums: Vec<i64> = (0..101).map(|i| (i * 7919) % 23 - 11).collect();
        let prods: Vec<i64> = (0..101).map(|i| { if i % 10 == 3 { 2 } else if i % 3 == 0 { -1 } else { 1 } }).collect();
        let at = |f: fn(i64) -> f64| (sums.iter().map(|&v| f(v)).collect::<Vec<_>>(), prods.iter().map(|&v| f(v)).collect::<Vec<_>>());
        exact(&sums, &prods, |t, v| t.checked_add(v), |t, v| t.checked_mul(v), 0, 1);
        let (s, p): (Vec<i16>, Vec<i16>) = (sums.iter().map(|&v| v as i16).collect(), prods.iter().map(|&v| v as i16).collect());
        exact(&s, &p, |t, v| t.checked_add(v.into()), |t, v| t.checked_mul(v.into()), 0, 1);
        let (s, p) = at(|v| v as f64);
        exact(&s, &p, |t, v| Some(t + v), |t, v| Some(t * v), 0.0, 1.0);
        let (s, p): (Vec<f32>, Vec<f32>) = (s.iter().map(|&v| v as f32).collect(), p.iter().map(|&v| v as f32).collect());
        exact(&s, &p, |t, v| Some(t + v), |t, v| Some(t * v), 0.0, 1.0);
        let (s, p): (Vec<f16>, Vec<f16>) = (s.iter().map(|&v| f16::from_f32(v)).collect(), p.iter().map(|&v| f16::from_f32(v)).collect());
        exact(&s, &p, |t, v| Some(t + v.to_f32()), |t, v| Some(t * v.to_f32()), 0.0, 1.0);
        assert_eq!(extremes(&sums), (*sums.iter().min().unwrap(), *sums.iter().max().unwrap()));
        let mut maxima = Vec::new();
        cummax(&sums, Some(-5), &mut maxima);
        assert_eq!(
            maxima, sums.iter().scan(-5, |m, &v| {
                *m = (*m).max(v);
                Some(*m)
            }).collect::<Vec<_>>()
        );
        assert_eq!((max(&sums), min(&sums)), (*sums.iter().max().unwrap(), *sums.iter().min().unwrap()));
        // Complex sums of whole numbers are exact in any order.
        let z: Vec<Complex<f64>> = (0..101).map(|k| Complex::new(k as f64, -(k as f64) / 2.0)).collect();
        assert_eq!(sum(&z), (z.iter().fold(Complex::new(0.0, 0.0), |s, &v| s + v), true));
        let mut out = Vec::new();
        assert!(cumsum(&z, Some(Complex::new(1.0, 1.0)), &mut out));
        assert_eq!(
            out, z.iter().scan(Complex::new(1.0, 1.0), |s, &v| {
                *s += v;
                Some(*s)
            }).collect::<Vec<_>>()
        );
    }

    #[test]
    fn edges() {
        assert_eq!((sum::<f64>(&[]), prod::<i32>(&[]), max::<f32>(&[]), min::<i16>(&[])), ((0.0, true), (1, true), f32::NEG_INFINITY, i16::MAX));
        // NaN at item 40 of 101 lies inside a fold's running results, which vectorize, and not in the remainder. AArch64's `fmax` gives NaN there, and other targets skip it.
        let mut v = vec![1.0f64; 101];
        v[40] = f64::NAN;
        let v32: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let (most, least, half) = (max(&v), min(&v32), max(&[f16::ONE, f16::NAN]));
        assert!((most.is_nan() || most == 1.0) && (least.is_nan() || least == 1.0) && (half.is_nan() || half == f16::ONE));
        assert_eq!((maximum(1.0, f64::NAN).is_nan(), minimum(f32::NAN, 1.0), maximum(-1.0, 2.0), minimum(f16::ONE, f16::ZERO)), (true, 1.0, 2.0, f16::ZERO));
        let mut out = Vec::new();
        cummax(&[1.0, 3.0, f64::NAN, 5.0, 2.0, 1.0], None, &mut out);
        assert!(out.iter().zip([1.0, 3.0, 3.0, 5.0, 5.0, 5.0]).all(|(&got, want)| got.is_nan() || got == want), "{out:?}");
        assert_eq!((sum(&[i64::MAX, 1]), sum(&[i64::MAX, 1, -2]), prod(&[i64::MAX, 2])), ((i64::MIN, false), (i64::MAX - 1, true), (-2, false)));
        assert!(!cumsum(&[i64::MAX, 1], None, &mut Vec::new()));
        assert!(!cumprod(&[i32::MAX, i32::MAX, i32::MAX], None, &mut Vec::new()));
        let mut out = vec![7.0];
        assert!(cumsum(&[1.0f64, 2.0], Some(10.0), &mut out));
        assert_eq!(out, [7.0, 11.0, 13.0]);
    }

    /// Each running maximum is NaN or `maximum`'s result with the running result as `y`, either side of the scan's block boundaries.
    #[test]
    fn scans_beside_nan() {
        let x: Vec<f64> = (0..200).map(|i| (i * 37 % 101) as f64 - 50.0).collect();
        for nan in [None, Some(0), Some(3), Some(63), Some(64), Some(130), Some(199)] {
            for start in [None, Some(10.0), Some(f64::NAN)] {
                let mut y = x.clone();
                if let Some(k) = nan { y[k] = f64::NAN }
                let mut out = Vec::new();
                cummax(&y, start, &mut out);
                let mut total = start;
                let want = y.iter().map(|&v| {
                    let t = total.map_or(v, |t| maximum(v, t));
                    total = Some(t);
                    t
                });
                assert!(out.len() == y.len() && out.iter().zip(want).all(|(&a, b)| a == b || a.is_nan()), "{nan:?} {start:?}");
            }
        }
    }

    /// Every lane of an array of shape `[outer, len, inner]`, its items in order.
    fn lanes_of<T: Copy>(x: &[T], [outer, len, inner]: [usize; 3]) -> Vec<Vec<T>> {
        (0..outer).flat_map(|i| (0..inner).map(move |k| (0..len).map(|j| x[(i * len + j) * inner + k]).collect())).collect()
    }

    /// Folds and scans along the middle axis against the same taken lane by lane, item by item, for lanes of consecutive items and otherwise, short and long, and empty. Folding in blocks of 13 splits lanes at many offsets. Booleans fold by each of their reductions, and an `i64` sum that wraps reports it, along a lane and down the rows.
    #[test]
    fn axes_match_lane_by_lane() {
        for shape in [[3, 37, 5], [15, 37, 1], [4, 3, 2], [2, 0, 3], [2, 200, 1], [70, 3, 1], [5, 64, 1], [65, 1, 1], [3, 50, 2]] {
            let [outer, len, inner] = shape;
            let x: Vec<i64> = (0..outer * len * inner).map(|i| (i * 37 % 101) as i64 - 50).collect();
            let sums: Vec<i64> = lanes_of(&x, shape).iter().map(|l| l.iter().sum()).collect();
            assert_eq!(along::<i64, Sum>(&x, shape), (sums.clone(), true));
            let mut totals = vec![0; sums.len()];
            for (b, block) in x.chunks(13).enumerate() { assert!(fold_into::<i64, Sum>(&mut totals, shape, b * 13, block)) }
            assert_eq!(totals, sums, "{shape:?}");
            let mut want = vec![0; x.len()];
            for i in 0..outer {
                for k in 0..inner {
                    (0..len).fold(10, |t, j| {
                        let p = (i * len + j) * inner + k;
                        want[p] = t + x[p];
                        want[p]
                    });
                }
            }
            let mut out = Vec::new();
            assert!(scan_along::<i64, Sum>(&x, shape, Some(10), &mut out));
            assert_eq!(out, want, "{shape:?}");
            // `SumAs`'s running sums of Booleans and of narrow integers match `Sum`'s, which run at 64 bits.
            let wide = |items: Vec<i64>, start: Option<i64>| {
                let mut out = Vec::new();
                scan_along::<i64, Sum>(&items, shape, start, &mut out);
                out.into_iter().map(|v| v as i32).collect::<Vec<_>>()
            };
            let mut counts = Vec::new();
            scan_along::<bool, SumAs<i32>>(&x.iter().map(|v| v % 3 == 0).collect::<Vec<_>>(), shape, Some(3), &mut counts);
            assert_eq!(counts, wide(x.iter().map(|v| i64::from(v % 3 == 0)).collect(), Some(3)), "{shape:?}");
            let mut sums = Vec::new();
            scan_along::<i16, SumAs<i32>>(&x.iter().map(|&v| v as i16).collect::<Vec<_>>(), shape, None, &mut sums);
            assert_eq!(sums, wide(x.clone(), None), "{shape:?}");
            let b: Vec<bool> = x.iter().map(|v| v % 3 == 0).collect();
            let lanes = lanes_of(&b, shape);
            let ones = |l: &Vec<bool>| l.iter().filter(|&&v| v).count();
            assert_eq!(along::<bool, Max>(&b, shape).0, lanes.iter().map(|l| l.contains(&true)).collect::<Vec<_>>());
            assert_eq!(along::<bool, Min>(&b, shape).0, lanes.iter().map(|l| !l.contains(&false)).collect::<Vec<_>>());
            assert_eq!(along::<bool, Xor>(&b, shape).0, lanes.iter().map(|l| ones(l) % 2 == 1).collect::<Vec<_>>());
            assert_eq!(along::<bool, Xnor>(&b, shape).0, lanes.iter().map(|l| (l.len() - ones(l)) % 2 == 0).collect::<Vec<_>>());
            assert_eq!(along::<bool, SumAs<u8>>(&b, shape).0, lanes.iter().map(|l| ones(l) as u8).collect::<Vec<_>>());
            for start in [None, Some(false), Some(true)] {
                for code in 0..16 {
                    let table = Logical([[code & 1 != 0, code & 2 != 0], [code & 4 != 0, code & 8 != 0]]);
                    let mut want = b.clone();
                    for i in 0..outer {
                        for k in 0..inner {
                            (0..len).fold(start, |t, j| {
                                let p = (i * len + j) * inner + k;
                                want[p] = t.map_or(b[p], |t| table.0[usize::from(t)][usize::from(b[p])]);
                                Some(want[p])
                            });
                        }
                    }
                    let mut out = Vec::new();
                    table.scan_along(&b, shape, start, &mut out);
                    assert_eq!(out, want, "{shape:?} {start:?} {code}");
                    // The Boolean reductions scan by their tables: `or` is 14, `and` 8, `!=` 6 and `==` 9.
                    let mut reduced = Vec::new();
                    match code {
                        14 => scan_along::<bool, Max>(&b, shape, start, &mut reduced), 8 => scan_along::<bool, Min>(&b, shape, start, &mut reduced),
                        6 => scan_along::<bool, Xor>(&b, shape, start, &mut reduced), 9 => scan_along::<bool, Xnor>(&b, shape, start, &mut reduced),
                        _ => continue
                    };
                    assert_eq!(reduced, want, "{shape:?} {start:?} {code}");
                }
            }
            // `chosen` picks each item's result from `low` or `high` by the result before it, from either carry. Item `first + i·len` takes `fixed[i]` whatever the result before it.
            let high: Vec<bool> = x.iter().map(|v| v % 5 != 0).collect();
            for (carry, len, first, restarts) in [(false, 1, 0, false), (true, 1, 0, false), (true, 3, 2, true), (false, 4, 1, true), (true, 70, 5, true)] {
                let fixed: Vec<bool> = if restarts { (first..b.len()).step_by(len).map(|p| x[p] % 3 == 0).collect() } else { Vec::new() };
                let mut t = carry;
                let want: Vec<bool> = (0..b.len()).map(|k| {
                    let restart = (k >= first && (k - first) % len == 0).then(|| fixed.get((k - first) / len)).flatten();
                    t = restart.copied().unwrap_or(if t { high[k] } else { b[k] });
                    t
                }).collect();
                let mut out = Vec::new();
                chosen(carry, &b, &high, len, first, &fixed, &mut out);
                assert_eq!(out, want, "{shape:?} {carry} {len} {first}");
            }
        }
        assert!(!along::<i64, Sum>(&[i64::MAX, 1], [1, 2, 1]).1);
        assert!(!along::<i64, Sum>(&[i64::MAX, 0, 1, 0], [1, 2, 2]).1);
        assert!(!scan_along::<i64, Sum>(&[i64::MAX, 0, 1, 0], [1, 2, 2], None, &mut Vec::new()));
    }

    /// Folds that run in pieces, of long lanes and of single matrices down their columns, and folds of many short lanes or groups, which split across threads, match the sums taken lane by lane, on one thread and on four.
    #[test]
    fn long_and_many_lanes() {
        for shape in [[1, (3 << 18) + 5, 1], [2, (1 << 18) + 7, 3], [5000, 200, 1], [300, 50, 70], [1, 2000, 300], [1, 600, 2000]] {
            let x: Vec<i64> = (0..shape.iter().product::<usize>()).map(|i| (i * 37 % 101) as i64 - 50).collect();
            let lanes = lanes_of(&x, shape);
            let sums: Vec<i64> = lanes.iter().map(|l| l.iter().sum()).collect();
            let sizes: Vec<f64> = lanes.iter().map(|l| l.iter().map(|&v| v.abs() as f64 / 7.0).sum()).collect();
            let f: Vec<f64> = x.iter().map(|&v| v as f64 / 7.0).collect();
            for threads in [1, 4] {
                rustymath::parallel::set_threads(threads);
                assert_eq!(along::<i64, Sum>(&x, shape), (sums.clone(), true), "{shape:?} {threads}");
                let floats = along::<f64, Sum>(&f, shape).0;
                for ((&got, &sum), &size) in floats.iter().zip(&sums).zip(&sizes) {
                    assert!((got - sum as f64 / 7.0).abs() <= 1e-10 * size, "{shape:?} {threads}: {got} against {}", sum as f64 / 7.0);
                }
            }
            rustymath::parallel::set_threads(0);
        }
    }

    #[test]
    fn alternating_folds_match_the_right_fold() {
        fn right<T: Copy>(x: &[T], unit: T, f: impl Fn(T, T) -> T) -> T { x.iter().rev().copied().reduce(|t, v| f(v, t)).unwrap_or(unit) }
        for n in [0, 1, 2, 3, 64, 101] {
            let x: Vec<i64> = (0..n).map(|i| (i * 7919) % 23 - 11).collect();
            let want = right(&x, 0, |a, b| a - b);
            assert_eq!(<Sum as Alternating<i64>>::alternating(&x), (want, true));
            assert_eq!(<Sum as Alternating<i16>>::alternating(&x.iter().map(|&v| v as i16).collect::<Vec<_>>()), (want, true));
            // Whole numbers, and powers of 2, keep every sum and quotient exact in any order.
            let f: Vec<f64> = x.iter().map(|&v| v as f64).collect();
            assert_eq!(<Sum as Alternating<f64>>::alternating(&f), (want as f64, true));
            let z: Vec<Complex<f64>> = f.iter().map(|&v| Complex::new(v, -v)).collect();
            assert_eq!(<Sum as Alternating<Complex<f64>>>::alternating(&z), (Complex::new(want as f64, -want as f64), true));
            let p: Vec<f64> = x.iter().map(|&v| 2f64.powi(v as i32 % 5)).collect();
            let q = right(&p, 1.0, |a, b| a / b);
            assert_eq!(<Product as Alternating<f64>>::alternating(&p), (q, true));
            let pz: Vec<Complex<f64>> = p.iter().map(|&v| Complex::new(v, 0.0)).collect();
            assert_eq!(<Product as Alternating<Complex<f64>>>::alternating(&pz), (Complex::new(q, 0.0), true));
        }
        assert!(!<Sum as Alternating<i64>>::alternating(&[i64::MAX, -1]).1);
        // The pairs' ratios alternate between 1e200 and 1e-200, so the right fold is 1. Each running product takes every 16th ratio, or every 8th in wasm, so half of them overflow and half underflow.
        let x: Vec<f64> = (0..256).map(|i| if (i % 4 == 0) | (i % 4 == 3) { 1e200 } else { 1.0 }).collect();
        assert_eq!(right(&x, 1.0, |a, b| a / b), 1.0);
        assert!(!<Product as Alternating<f64>>::alternating(&x).1);
        assert!(!<Product as Alternating<f64>>::alternating(&[1.0, 0.0]).1);
    }
}

mod search {
    use rustymath::search::*;

    #[test]
    fn search_and_sort() {
        let x = [30, 10, 20, 10];
        assert_eq!(index_of(&x, &[10, 40]), [1, 4]);
        // More than `SCANNED` needles build the index in place of scanning.
        let many: Vec<i32> = (0..20).map(|i| [10, 40, 30][i % 3]).collect();
        assert_eq!(index_of(&x, &many), (0..20).map(|i| [1, 4, 0][i % 3]).collect::<Vec<_>>());
        assert_eq!(numbered(&classes(&x)), Numbered { firsts: vec![0, 1, 2], numbers: vec![0, 1, 2, 1], counts: vec![1, 2, 1] });
        assert_eq!(classes(&x), [0, 1, 2, 1]);
        assert_eq!(sort_rows(x.len(), vec![x.iter().map(|&v| v as u32).collect()]), [1, 3, 2, 0]);
        // 300 rows take the counting sort. Equal keys keep their order of position, also under complemented keys, which reverse it.
        let y: Vec<u32> = (0..300).map(|i| i * 7 % 5).collect();
        let mut stable: Vec<u32> = (0..300).collect();
        stable.sort_by_key(|&r| y[r as usize]);
        assert_eq!(sort_rows(y.len(), vec![y.clone()]), stable);
        stable.sort_by_key(|&r| !y[r as usize]);
        assert_eq!(sort_rows(y.len(), vec![y.iter().map(|&v| !v).collect()]), stable);
        // Two keys take the pass by masks of 8 items. Here group 0's block writes past its end, onto group 1's first places. In the second, group 1's only item follows the last block.
        assert_eq!(grouped(&[1, 0, 0, 0, 0, 0, 1, 1], 2), (vec![1, 2, 3, 4, 5, 0, 6, 7], vec![5, 3]));
        assert_eq!(grouped(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2), ((0..10).collect(), vec![9, 1]));
        let b: Vec<u16> = (0..300).map(|i| u16::from(i % 3 == 0)).collect();
        let mut two: Vec<u32> = (0..300).collect();
        two.sort_by_key(|&r| b[r as usize]);
        assert_eq!(sort_rows(b.len(), vec![b.clone()]), two);
        // Keys too wide for a table take the radix sort, one column at a time from the last. Rows that tie on the first column follow the second, and rows that tie on both keep their order of position.
        let wide: Vec<u64> = (0..300u64).map(|i| (i % 7).wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let second: Vec<u64> = (0..300u64).map(|i| (i % 11) << 40).collect();
        let mut rows: Vec<u32> = (0..300).collect();
        rows.sort_by_key(|&r| wide[r as usize]);
        assert_eq!(sort_rows(300, vec![wide.clone()]), rows);
        rows.sort_by_key(|&r| (wide[r as usize], second[r as usize]));
        assert_eq!(sort_rows(300, vec![wide, second]), rows);
        assert_eq!((grouped(&[1, 0, 1, 1], 2), grouped(&[0, 0], 1)), ((vec![1, 0, 2, 3], vec![1, 3]), (vec![0, 1], vec![2])));
        // `grouped_keys` gives the classes that `classes` and `numbered` give, each with its positions. The five keys first appear out of their order, and a range of 2^20 + 1 is too wide for a table.
        let together = |k: &[i32]| {
            let n = numbered(&classes(k));
            let mut each = vec![Vec::new(); n.firsts.len()];
            for (i, &g) in n.numbers.iter().enumerate() { each[g].push(i as u32) }
            let g = grouped_keys(k).expect("a table holds the keys");
            assert_eq!((&g.firsts, g.each().collect::<Vec<_>>()), (&n.firsts, each.iter().map(Vec::as_slice).collect()));
            assert_eq!(g.into_classes(), each);
        };
        together(&(0..300).map(|i| i32::from(i % 3 == 0)).collect::<Vec<_>>());
        together(&(0..300).map(|i| 3 + (i * 7 + 2) % 5).collect::<Vec<_>>());
        assert_eq!(grouped_keys(&[0, 1 << 20]), None);
        // Keys 0, 2 and 4 fill the first 595 places, and key 6 first appears at 595, after four blocks of 64 with no new key make `write_first` count the keys `x` holds.
        let late: Vec<i32> = (0..600).map(|i| if i == 595 { 6 } else { i % 3 * 2 }).collect();
        assert_eq!((index_of(&late, &[6; 20]), classes(&late)[595]), (vec![595; 20], 595));
        // 299 items of 5 keys go to four tables, with 3 left over. A range of 2^20 + 1 is too wide for a table.
        let z = &y[..299];
        assert_eq!(counts(z), Some((0, (0..5).map(|k| z.iter().filter(|&&v| v == k).count()).collect())));
        assert_eq!((counts(&[3, 1, 3, 2]), counts(&[0, 1 << 20])), (Some((1, vec![1, 1, 2])), None));
        // Repeated boundaries, values beyond both ends, and no boundaries at all.
        let (bounds, values) = ([1, 3, 3, 3, 7, 9], (0..12).collect::<Vec<i32>>());
        assert_eq!(bins(&bounds, &values), values.iter().map(|&v| bounds.partition_point(|&b| b <= v) as u32).collect::<Vec<_>>());
        assert_eq!(bins::<i32>(&[], &[5]), [0]);
        // Boundaries too wide for a table take the binary search.
        let wide = [-70000, 5, 5, 90000];
        let values = [-80000, -70000, 0, 5, 6, 90000, 95000];
        assert_eq!(bins(&wide, &values), values.iter().map(|&v| wide.partition_point(|&b| b <= v) as u32).collect::<Vec<_>>());
    }

    /// Over more than one 65,536-key piece, the radix sort's passes and the lookups split across threads and give one thread's results. Equal keys keep their order of position.
    #[test]
    fn pieces_across_threads() {
        use rustymath::tolerant::{self, Tolerance};
        let n = 200_000;
        // 5,000 keys, about 40 of each, spread too wide for a table, so the sort takes the radix passes.
        let keys: Vec<u64> = (0..n as u64).map(|i| (i * 2_654_435_761 % 5_000) << 40).collect();
        let second: Vec<u64> = (0..n as u64).map(|i| i % 7).collect();
        let mut stable: Vec<u32> = (0..n as u32).collect();
        stable.sort_by_key(|&r| keys[r as usize]);
        let mut rows = stable.clone();
        rows.sort_by_key(|&r| (keys[r as usize], second[r as usize]));
        let haystack: Vec<i64> = keys.iter().map(|&k| (k >> 40) as i64).collect();
        let needles: Vec<i64> = (0..n as i64).map(|i| i * 7 % 6_000).collect();
        let mut first = std::collections::HashMap::new();
        for (i, &k) in haystack.iter().enumerate() { first.entry(k).or_insert(i as u32); }
        let found: Vec<u32> = needles.iter().map(|k| first.get(k).copied().unwrap_or(n as u32)).collect();
        // Sevenths that aren't whole numbers take the buckets, in more than one round of lookups.
        let floats: Vec<f64> = haystack.iter().map(|&k| k as f64 / 7.0).collect();
        let near: Vec<f64> = needles.iter().map(|&k| k as f64 / 7.0 + 1e-15).collect();
        let mut by_threads = Vec::new();
        for threads in [1, 4] {
            rustymath::parallel::set_threads(threads);
            assert_eq!(sort_rows(n, vec![keys.clone()]), stable, "{threads}");
            assert_eq!(sort_rows(n, vec![keys.clone(), second.clone()]), rows, "{threads}");
            assert_eq!(index_of(&haystack, &needles), found, "{threads}");
            by_threads.push(tolerant::index_of(Tolerance::DEFAULT, &floats, &near));
        }
        rustymath::parallel::set_threads(0);
        assert_eq!(by_threads[0], by_threads[1]);
    }
}

mod tolerant {
    use half::f16;
    use num_complex::Complex64;
    use num_traits::Float as _;
    use rustymath::{Arg, Binary, Float, Unary, search, tolerant::*, with_tolerance};
    use std::cmp::Ordering;

    const T: Tolerance = Tolerance::DEFAULT;
    fn c(re: f64, im: f64) -> Complex64 { Complex64::new(re, im) }

    #[test]
    fn equal_and_match() {
        // The rounded pair 0.3 and 0.1+0.2 lies inside the default tolerance, and 1 and 1+3e-14 outside it.
        assert!(T.equal(0.3, 0.1 + 0.2) && T.equal(-0.3, -(0.1 + 0.2)) && !Exactly.equal(0.3, 0.1 + 0.2));
        assert!(!T.equal(1.0, 1.0 + 3e-14));
        assert_eq!([3.1, 3.0, -2.0, -3.0].map(|x| T.equal(3.0, x)), [false, true, false, false]);
        let wide = Tolerance::new(1e-10).unwrap();
        assert!(wide.equal(1.0, 1.000000000001) && !wide.equal(1.0, 1.0000001));
        // A tolerance has no effect beside zero.
        assert!(!T.equal(0.0, 1e-300) && T.equal(0.0, -0.0));
        assert!(T.equal(f64::INFINITY, f64::INFINITY) && !T.equal(f64::INFINITY, f64::MAX) && !T.equal(f64::NAN, f64::NAN));
        assert!(T.matches(f64::NAN, f64::NAN) && !T.matches(f64::NAN, 1.0));
        assert_eq!((T.compare(0.3, 0.1 + 0.2), T.compare(1.0, 2.0), T.compare(f64::NAN, 1.0)), (Some(Ordering::Equal), Some(Ordering::Less), None));
        // Below and above leave out what's equal within the tolerance, and comparisons with NaN are false.
        assert!(!T.less(0.3, 0.1 + 0.2) && T.less_equal(0.1 + 0.2, 0.3) && Exactly.less(0.3, 0.1 + 0.2) && !Exactly.greater(0.3, 0.1 + 0.2));
        assert!(!T.less(f64::NAN, 1.0) && !T.less_equal(f64::NAN, f64::NAN) && !T.greater_equal(1.0, f64::NAN));
        // At f32, a tolerance of 4 units in the last place.
        let t32 = Tolerance::new(4.0 * f32::EPSILON as f64).unwrap();
        assert!(t32.equal(1.0f32, 1.0 + 4.0 * f32::EPSILON) && !t32.equal(1.0f32, 1.0 + 5.0 * f32::EPSILON));
        assert_eq!((Tolerance::new(-1e-14), Tolerance::new(1.0), Tolerance::new(f64::NAN)), (None, None, None));
        // `with_tolerance!` runs its body at `Exactly` for a tolerance of 0.
        let count = |t: Tolerance| with_tolerance!(t, |t| [0.3, 0.1 + 0.2, 0.31].iter().filter(|&&x| t.equal(x, 0.3)).count());
        assert_eq!((count(T), count(Tolerance::EXACT)), (2, 1));
    }

    #[test]
    fn floor_ceil_and_whole() {
        assert_eq!([-2.3, 0.1, 100.0, 3.3, 0.5 + 0.4, 0.5 + 0.5, 0.5 + 0.6].map(|y| T.floor(y)), [-3.0, 0.0, 100.0, 3.0, 0.0, 1.0, 1.0]);
        assert_eq!((T.floor(2.9999999999999996), Exactly.floor(2.9999999999999996)), (3.0, 2.0));
        assert_eq!((T.ceil(1.0000000000000002), Exactly.ceil(1.0000000000000002)), (1.0, 2.0));
        assert_eq!((T.floor(-0.9999999999999999), T.floor(-0.999999999999)), (-1.0, -1.0));
        assert_eq!((T.floor(0.999999999999), T.floor(f64::INFINITY)), (0.0, f64::INFINITY));
        assert!(T.floor(f64::NAN).is_nan());
        assert_eq!((T.floor(f16::from_f32(2.5)), T.ceil(2.5f32)), (f16::from_f32(2.0), 3.0));
        assert_eq!((T.whole(2.9999999999999996), T.whole(2.5), T.whole(f64::INFINITY), Exactly.whole(4.0)), (Some(3.0), None, None, Some(4.0)));
        // Below 1/(8t), where no float equals two whole numbers, the tolerant floor is the ordinary floor of the greatest equal float. Step up to 60 floats either side of whole numbers.
        for n in [1.0f64, 3.0, -3.0, 1e6, -1e6, 1e12, 4e12, -4e12] {
            let (mut down, mut up) = (n, n);
            for _ in 0..60 {
                for y in [down, up] { assert_eq!(T.floor(y), T.range(y).unwrap().1.floor(), "{y}"); }
                (down, up) = (down.next_down(), up.next_up());
            }
        }
    }

    #[test]
    fn residue_and_gcd() {
        assert_eq!([(3.0, -5.0), (3.0, 5.0), (-3.0, -4.0), (-3.0, 4.0)].map(|(x, y)| T.residue(x, y)), [1.0, 2.0, -1.0, -2.0]);
        let r = [3.12, -1.0, -0.6].map(|y| T.residue(0.5, y));
        assert!((r[0] - 0.12).abs() < 1e-15 && r[1] == 0.0 && (r[2] - 0.4).abs() < 1e-15);
        let r = [(-1.0, -5.25), (0.0, 0.0), (1.0, 2.41)].map(|(x, y)| T.residue(x, y));
        assert!(r[0] == -0.25 && r[1] == 0.0 && (r[2] - 0.41).abs() < 1e-15);
        // A quotient within tolerance of a whole number leaves no remainder, where exact comparison leaves a whole divisor.
        assert_eq!((T.residue(0.1, 0.3), T.residue(0.0, 2.5)), (0.0, 2.5));
        assert!((Exactly.residue(0.1, 0.3) - 0.1).abs() < 1e-15);
        assert!((T.gcd(0.6, 0.4) - 0.2).abs() < 1e-15 && Exactly.gcd(0.6, 0.4) < 1e-15);
        assert_eq!((T.gcd(12.0, 18.0), T.gcd(-12.0, 18.0), T.gcd(0.0, 2.5), T.gcd(0.0, 0.0), T.gcd(12.0f32, 18.0)), (6.0, 6.0, 2.5, 0.0, 6.0));
        assert!(T.gcd(f64::NAN, 1.0).is_nan() && T.gcd(f64::NAN, f64::NAN).is_nan() && T.gcd(f64::INFINITY, 1.0).is_nan());
    }

    #[test]
    fn complex() {
        let a = c(2.0, 1e-14);
        assert!(T.complex_equal(a, c(2.0, 0.00000000000001)) && !T.complex_equal(a, c(2.0, 0.0000000000001)));
        assert!(T.complex_equal(c(0.3, 0.0), c(0.1 + 0.2, 0.0)) && !T.complex_equal(c(f64::NAN, 1.0), c(f64::NAN, 1.0)));
        assert_eq!([c(1.0, 3.2), c(3.3, 2.5), c(-3.3, -2.5)].map(|y| T.complex_floor(y)), [c(1.0, 3.0), c(3.0, 2.0), c(-3.0, -3.0)]);
        assert_eq!(T.complex_ceil(c(1.0, 3.2)), c(1.0, 4.0));
        let r = [c(2.0, 3.0), c(3.0, 4.0), c(5.0, 6.0)].map(|y| T.complex_residue(c(1.0, 2.0), y));
        assert_eq!(r, [c(1.0, 1.0), c(-1.0, 1.0), c(0.0, 1.0)]);
        assert_eq!(T.complex_residue(c(0.0, 0.0), c(2.0, 1.0)), c(2.0, 1.0));
        // 5 = (2+i)(2-i) and 3+4i = (2+i)², so their gcd is 2+i, the associate in the first quadrant. Halving both halves it.
        assert_eq!((T.complex_gcd(c(5.0, 0.0), c(3.0, 4.0)), T.complex_gcd(c(2.5, 0.0), c(1.5, 2.0))), (c(2.0, 1.0), c(1.0, 0.5)));
        assert_eq!((T.complex_gcd(c(0.0, 0.0), c(-4.0, 3.0)), T.complex_gcd(c(12.0, 0.0), c(18.0, 0.0))), (c(3.0, 4.0), c(6.0, 0.0)));
        assert!(T.complex_gcd(c(f64::INFINITY, 1.0), c(2.0, 1.0)).re.is_nan());
        // Squared magnitudes would overflow at ±1e300 and lose the tolerance at 1e-300, so those pairs compare magnitudes themselves.
        let (huge, tiny) = (c(1e300, 1e300), c(1e-300, 1e-300));
        assert!(T.complex_equal(huge, huge * (1.0 + 1e-15)) && !T.complex_equal(c(1e300, 0.0), c(-1e300, 0.0)));
        assert!(T.complex_equal(tiny, tiny * (1.0 + 1e-15)) && !T.complex_equal(tiny, tiny * 2.0));
        // The slice forms decide most pairs from squared magnitudes, in a loop with no branch, then redo the rest.
        let zs = [c(2.0, 1.0), huge, tiny, c(0.0, 0.0), c(f64::INFINITY, 1.0), c(f64::NAN, 0.0), c(-1e300, 0.0)];
        let (xs, ys): (Vec<_>, Vec<_>) = zs.iter().flat_map(|&x| zs.iter().map(move |&y| (x, y * (1.0 + 1e-15)))).unzip();
        let (mut equal, mut unequal) = (Vec::new(), Vec::new());
        assert!(Equal(T).many(Arg::Many(&xs), Arg::Many(&ys), xs.len(), &mut equal));
        assert!(NotEqual(T).many(Arg::Many(&xs), Arg::Many(&ys), xs.len(), &mut unequal));
        for i in 0..xs.len() { assert!(equal[i] == T.complex_equal(xs[i], ys[i]) && unequal[i] != equal[i], "{:?} {:?}", xs[i], ys[i]) }
    }

    /// `range(c)` gives the last floats equal to `c` on each side.
    fn check_range<F: Float + std::fmt::Debug>(t: impl Tolerant, c: F) {
        let (lo, hi) = t.range(c).unwrap();
        assert!(lo <= c && c <= hi, "{c:?}");
        assert!(t.equal(lo, c) && t.equal(hi, c) && !t.equal(lo.next_down(), c) && !t.equal(hi.next_up(), c), "{c:?}: {lo:?} {hi:?}");
    }

    #[test]
    fn range() {
        let specials = [0.3, -0.3, 1.0, 1e-5, 5e-324, 2.2250738585072014e-308, 8796093022208.0, -1e20, f64::MAX, -f64::MAX];
        for c in specials {
            check_range(T, c);
            check_range(Exactly, c);
        }
        for c in (0..100_000).map(|i| f64::from_bits(crate::support::random(i))).filter(|c| !c.is_nan()) { check_range(T, c); }
        let t32 = Tolerance::new(4.0 * f32::EPSILON as f64).unwrap();
        for c in [0.3f32, -7.5, 1e-40, f32::MAX] { check_range(t32, c); }
        let t16 = Tolerance::new(4.0 * f16::EPSILON.to_f64()).unwrap();
        for c in [0.3, -7.5, 65504.0, 1e-7].map(f16::from_f64) { check_range(t16, c); }
        assert_eq!((T.range(f64::NAN), T.range(0.0), T.range(f64::NEG_INFINITY)), (None, Some((0.0, 0.0)), Some((f64::NEG_INFINITY, f64::NEG_INFINITY))));
        // At f16 the default tolerance rounds to 0, so every float equals only itself.
        assert_eq!(T.range(f16::from_f32(0.3)), Some((f16::from_f32(0.3), f16::from_f32(0.3))));
    }

    /// Checks each real comparison's slice form against its scalar form for many items against one number, in either order, with each item of `items` as the number.
    fn against_one<F: Float + std::fmt::Debug>(t: Tolerance, items: &[f64]) where Equal: Binary<F, Out = bool>, NotEqual: Binary<F, Out = bool>,
        Less: Binary<F, Out = bool>, LessEqual: Binary<F, Out = bool>, Greater: Binary<F, Out = bool>, GreaterEqual: Binary<F, Out = bool>,
        Matches: Binary<F, Out = bool> {
        fn check<F: Copy + std::fmt::Debug, K: Binary<F, Out = bool> + std::fmt::Debug>(k: K, x: &[F]) {
            for &c in x {
                let (mut left, mut right) = (Vec::new(), Vec::new());
                assert!(k.many(Arg::Many(x), Arg::One(c), x.len(), &mut left) & k.many(Arg::One(c), Arg::Many(x), x.len(), &mut right));
                assert_eq!(left, x.iter().map(|&v| k.one(v, c)).collect::<Vec<_>>(), "{k:?} with {c:?} on the right");
                assert_eq!(right, x.iter().map(|&v| k.one(c, v)).collect::<Vec<_>>(), "{k:?} with {c:?} on the left");
            }
        }
        let x: Vec<F> = items.iter().map(|&v| F::narrow(v)).collect();
        check(Equal(t), &x);
        check(NotEqual(t), &x);
        check(Less(t), &x);
        check(LessEqual(t), &x);
        check(Greater(t), &x);
        check(GreaterEqual(t), &x);
        check(Matches(t), &x);
    }
    /// Each tolerant object gives its method's result for each item, at the default tolerance and at 0, for slices beside slices and one item beside a slice, at f32, f64 and complex items. Only `Gcd`'s infinite argument fails.
    #[test]
    fn objects() {
        let (xs, ys) = (&[3.0, -3.0, 0.3, 0.0, f64::NAN, 2.0][..], &[-5.0, 5.0, 0.1 + 0.2, 7.0, 1.0, 2.9999999999999996][..]);
        let bits = |v: &[f64]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        for t in [T, Tolerance::EXACT] {
            let (mut floors, mut residues, mut equal, mut order) = (Vec::new(), Vec::new(), Vec::new(), Vec::new());
            assert!(Floor(t).many(Arg::Many(ys), 6, &mut floors) & Ceil(t).many(Arg::Many(ys), 6, &mut floors));
            assert!(Residue(t).many(Arg::Many(xs), Arg::Many(ys), 6, &mut residues) & Residue(t).many(Arg::One(3.0), Arg::Many(ys), 6, &mut residues));
            assert!(Equal(t).many(Arg::Many(xs), Arg::Many(ys), 6, &mut equal) & Matches(t).many(Arg::Many(xs), Arg::One(f64::NAN), 6, &mut equal));
            assert!(Compare(t).many(Arg::Many(xs), Arg::Many(ys), 6, &mut order));
            let floor: Vec<f64> = ys.iter().map(|&y| t.floor(y)).chain(ys.iter().map(|&y| t.ceil(y))).collect();
            let residue: Vec<f64> = (0..6).map(|i| t.residue(xs[i], ys[i])).chain(ys.iter().map(|&y| t.residue(3.0, y))).collect();
            let eq: Vec<bool> = (0..6).map(|i| t.equal(xs[i], ys[i])).chain(xs.iter().map(|&x| t.matches(x, f64::NAN))).collect();
            assert_eq!((bits(&floors), bits(&residues), equal), (bits(&floor), bits(&residue), eq));
            assert_eq!(order, (0..6).map(|i| t.compare(xs[i], ys[i]).unwrap_or(Ordering::Greater)).collect::<Vec<_>>());
            let check = |many: &dyn Fn(&mut Vec<bool>) -> bool, one: &dyn Fn(f64, f64) -> bool| {
                let mut got = Vec::new();
                assert!(many(&mut got));
                assert_eq!(got, (0..6).map(|i| one(xs[i], ys[i])).collect::<Vec<_>>());
            };
            check(&|o| NotEqual(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| !t.equal(x, y));
            check(&|o| Less(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.less(x, y));
            check(&|o| LessEqual(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.less_equal(x, y));
            check(&|o| Greater(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.greater(x, y));
            check(&|o| GreaterEqual(t).many(Arg::Many(xs), Arg::Many(ys), 6, o), &|x, y| t.greater_equal(x, y));
            let (ys32, mut floors32) = (ys.iter().map(|&y| y as f32).collect::<Vec<_>>(), Vec::new());
            assert!(Floor(t).many(Arg::Many(&ys32), 6, &mut floors32));
            assert_eq!(floors32, ys32.iter().map(|&y| t.floor(y)).collect::<Vec<_>>());
            let (zs, mut complex) = (&[c(2.3, -1.7), c(0.5, 0.5), c(-1.0, 2.0)][..], Vec::new());
            assert!(Floor(t).many(Arg::Many(zs), 3, &mut complex) & Residue(t).many(Arg::One(c(1.0, 1.0)), Arg::Many(zs), 3, &mut complex));
            let want: Vec<Complex64> = zs.iter().map(|&z| t.complex_floor(z)).chain(zs.iter().map(|&z| t.complex_residue(c(1.0, 1.0), z))).collect();
            assert_eq!(complex, want);
        }
        // Many items against one number take the range of floats equal to it, in either order. Each test item serves as the number.
        let items = [3.0, -3.0, 0.3, 0.1 + 0.2, 0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 2.0, 2.9999999999999996, 1e-310, 1.0 + 3e-14];
        for t in [T, Tolerance::EXACT] {
            against_one::<f64>(t, &items);
            against_one::<f32>(t, &items);
            against_one::<f16>(t, &items);
        }
        assert_eq!(Gcd(T).one(12.0, 18.0), 6.0);
        assert!(!Gcd(T).many(Arg::One(f64::INFINITY), Arg::Many(&[2.0][..]), 1, &mut Vec::new()));
        // Lcm has the sign of x y, gives 0 where the gcd is 0, and divides at the tolerance's gcd.
        assert_eq!([Lcm(T).one(-4.0, 6.0), Lcm(T).one(0.0, 5.0), Lcm(T).one(0.6, 0.4)], [-12.0, 0.0, 1.2000000000000002]);
        assert_eq!(Lcm(T).one(c(2.0, 0.0), c(0.0, 3.0)), c(0.0, 6.0));
    }

    /// Tolerant search finds the first match that a brute-force search finds, at every width and at small and large tolerances, among floats crowded near a few values, and among zeros, infinities and NaN, whether it scans for a few items or indexes many.
    #[test]
    fn slice_search() {
        fn check<F: Float + std::fmt::Debug>(t: Tolerance) {
            let base = [1.0, -1.0, 3.0, 1e-3, 0.0, 1000.0];
            let x: Vec<F> = (0..400u64).map(|i| {
                let r = crate::support::random(i);
                F::narrow(base[(r % 6) as usize] * (1.0 + (((r >> 8) % 9) as f64 - 4.0) * t.value() / 2.0))
            }).chain([F::nan(), F::infinity(), F::neg_infinity(), F::neg_zero()]).collect();
            let y: Vec<F> = x.iter().rev().copied().chain([F::narrow(7.0)]).collect();
            let brute: Vec<u32> = y.iter().map(|&v| x.iter().position(|&h| t.matches(h, v)).unwrap_or(x.len()) as u32).collect();
            assert_eq!(index_of(t, &x, &y), brute, "{t:?}");
            assert_eq!(index_of(t, &x, &y[..5]), brute[..5], "{t:?}, scanning");
        }
        for t in [T, Tolerance::new(1e-3).unwrap(), Tolerance::new(0.3).unwrap()] {
            check::<f64>(t);
            check::<f32>(t);
            check::<f16>(t);
        }
        let x = [0.3, 1.0, 0.1 + 0.2];
        assert_eq!((index_of(T, &x, &[0.1 + 0.2, 2.0]), index_of(Exactly, &x, &[0.1 + 0.2, 2.0])), (vec![0, 3], vec![2, 3]));
        // Floats that each equal a whole number index as integers. At a tolerance of 0.3, which makes 3 equal 4, `whole_limit` lets only 0 qualify, so the search falls back to buckets.
        let wx: Vec<f64> = (0..300u64).map(|i| (crate::support::random(i) % 50) as f64 - 25.0).collect();
        let wy: Vec<f64> = (0..40).map(|i| i as f64 - 30.0).collect();
        for t in [T, Tolerance::EXACT, Tolerance::new(0.3).unwrap()] {
            let brute: Vec<u32> = wy.iter().map(|&v| wx.iter().position(|&h| t.matches(h, v)).unwrap_or(wx.len()) as u32).collect();
            assert_eq!(index_of(t, &wx, &wy), brute, "{t:?}");
        }
        // Floats near one whole number each equal it, but needn't equal each other. An indexed search finds only true matches, as a scan of 16 or fewer does.
        let near = [1.0 - 8e-15, 1.0, 1.0 + 8e-15];
        assert_eq!((index_of(T, &near[..1], &[near[2]; 17]), index_of(T, &near[..2], &[near[2]; 17])), (vec![1; 17], vec![1; 17]));
        let ints: Vec<i64> = wx.iter().map(|&v| v as i64).collect();
        assert_eq!((classes(T, &wx), firsts(T, &wx)), (search::classes(&ints), search::firsts(&ints)));
    }
}

mod gather {
    use rustymath::gather::{walk, walk_range, Same, Steps};
    /// The items of `s` that `steps` reach from `base`, with -1 for a fill.
    fn read(s: &[i64], base: usize, steps: &[Steps]) -> Vec<i64> {
        let mut out = vec![];
        walk(&mut out, s, base, steps, -1, Same);
        out
    }
    fn along(len: usize, step: isize) -> Steps { Steps::Stride { start: 0, len, step } }
    #[test]
    fn walks_reach_and_run_agree() {
        // Each item of `s` is its own offset, so a walk shows the offsets it read.
        let s: Vec<i64> = (0..600).collect();
        let cases = [
            (0, vec![along(3, 20), along(20, 1)]),
            (0, vec![along(3, 20), Steps::Stride { start: 19, len: 20, step: -1 }]),
            (0, vec![along(2, 50), Steps::Rotated { shift: 7, len: 30, stride: 1 }]),
            (0, vec![along(2, 50), Steps::Clipped { start: -3, len: 40, size: 30, stride: 1 }]),
            (0, vec![Steps::Table(vec![Some(4), None, Some(1)]), along(3, 1)]),
        ];
        for (base, steps) in &cases {
            let whole = read(&s, *base, steps);
            let mut mapped = vec![];
            walk(&mut mapped, &s, *base, steps, -1, |x: i64| x);
            assert_eq!(whole, mapped);
            for (i, &x) in whole.iter().enumerate() { assert_eq!(Steps::reach(*base, steps, i).map_or(-1, |o| o as i64), x) }
            for (from, n) in [(1, 7), (5, whole.len() - 5), (0, whole.len())] {
                let mut part = vec![];
                walk_range(&mut part, &s, *base, steps, -1, from, n, Same);
                assert_eq!(part, whole[from..from + n]);
            }
        }
        assert_eq!((Steps::run(0, &cases[0].1, 3, 5), Steps::run(0, &cases[0].1, 18, 5)), (Some(3), None));
    }
}
