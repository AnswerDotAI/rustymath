pub mod support;

mod parallel {
    use rustymath::parallel::{ranges, set_threads, split, threads};
    use std::{sync::{Mutex, atomic::{AtomicBool, Ordering}}, thread, time::Duration};

    #[test]
    fn ranges_and_outputs() {
        let _threads = crate::support::threads_lock();
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

    /// Whether `a` and `b` agree: both NaN, the same value, or finite and nonzero within 8 units in the last place of each other. A scalar form and its slice form can round differently, as `Pow`'s do at f32, where only the scalar form reads tables. Infinities and signs must match exactly, apart from a zero's sign, which can differ between routes: a vector maximum orders `-0` below `0`, where the scalar form gives its second argument on a tie.
    fn same<F: Float>(a: F, b: F) -> bool {
        if a.is_nan() || b.is_nan() { return a.is_nan() && b.is_nan(); }
        if a == b { return true; }
        let near = (a - b).abs() <= F::narrow(8.0) * F::epsilon() * a.abs().max(b.abs()).max(F::min_positive_value());
        a.is_finite() && b.is_finite() && a != F::zero() && b != F::zero() && near
    }

    /// Items with every special value, both signs, a subnormal, and magnitudes beyond the trig functions' fast reduction.
    fn items<F: Float>() -> Vec<F> {
        let wide = [0.0, 1.0, 0.5, 2.0, 3.0, 0.1, 7.25, 1e-310, 1e-30, 30.0, 700.0, 2e6, 1e10, 3e8, f64::INFINITY, f64::NAN];
        wide.iter().flat_map(|&x| [x, -x]).map(F::narrow).collect()
    }

    /// `k.many` against `k.one` for every pairing of `One` and `Many` among `items`. It must append after what `out` holds, and say whether every item succeeded, as `checked` does item by item. Where `K` isn't total, the verdict follows the NaN rule: an item fails exactly where it gives NaN from arguments that aren't NaN. A function whose `TOTAL` is true must succeed on every item, including where NaN is its value, as sin ∞ is.
    fn binary<F: Float + Debug, K: Binary<F, Out = F>>(k: K, name: &str) {
        let items = items::<F>();
        let check = |x: Arg<'_, F>, y: Arg<'_, F>, n: usize| {
            let at = |a: Arg<'_, F>, i| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
            let mut out = vec![F::one()];
            let ok = k.many(x, y, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()), "{name}");
            for i in 0..n { assert!(same(out[i + 1], k.one(at(x, i), at(y, i))), "{name}({:?}, {:?})", at(x, i), at(y, i)) }
            assert_eq!(ok, (0..n).all(|i| k.checked(at(x, i), at(y, i)).is_some()), "{name}");
            if !K::TOTAL {
                assert_eq!(ok, (0..n).all(|i| !out[i + 1].is_nan() || [at(x, i), at(y, i)].iter().any(|v| v.is_nan())), "{name} breaks the NaN rule")
            }
            assert!(ok || !K::TOTAL, "{name} is total but fails");
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
            if !K::TOTAL { assert_eq!(ok, (0..n).all(|i| !out[i + 1].is_nan() || at(i).is_nan()), "{name} breaks the NaN rule") }
            assert!(ok || !K::TOTAL, "{name} is total but fails");
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
            if !K::TOTAL {
                assert_eq!(ok, (0..n).all(|i| !out[i + 1].is_nan() || [at(x, i), at(y, i), at(z, i)].iter().any(|v| v.is_nan())), "{name} breaks the NaN rule")
            }
            assert!(ok || !K::TOTAL, "{name} is total but fails");
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
        // The logarithms' slice forms take each block of 64 items whose every item is positive, normal and finite through a form with no special values. Blocks of such items, some holding a special value, reach both forms.
        let mut x: Vec<f64> = (0..2000).map(|_| 1000.0 * rng.random::<f64>()).collect();
        for (i, v) in [0.0, -1.0, 1e-310, f64::INFINITY, f64::NAN, 1.0].into_iter().enumerate() { x[300 * i + 5] = v }
        unary(Ln, "Ln", &x);
        unary(Log2, "Log2", &x);
        unary(Log10, "Log10", &x);
        unary(Log1p, "Log1p", &x.iter().map(|&v| v - 1.0).collect::<Vec<_>>());
        // `Exp` and `Expm1` take their fast form for blocks whose items all lie within ±708 at f64 and ±87 at f32. The same items, shifted into that range, put ∞ and NaN in two blocks.
        let (e64, e32): (Vec<f64>, Vec<f32>) = x.iter().map(|&v| (v - 700.0, (v / 10.0 - 50.0) as f32)).unzip();
        unary(Exp, "Exp", &e64);
        unary(Exp, "Exp", &e32);
        unary(Expm1, "Expm1", &e64);
        unary(Expm1, "Expm1", &e32);
        // `Erfc` computes one formula for a block of 512 items whose magnitudes all lie on one side of its split, and a mixed block takes both. A ramp from -5 to 5 holds blocks of each kind.
        let ramp: Vec<f64> = (0..4000).map(|i| i as f64 / 400.0 - 5.0).collect();
        let ramp32: Vec<f32> = ramp.iter().map(|&v| v as f32).collect();
        unary(Erfc, "Erfc", &ramp);
        unary(Erfc, "Erfc", &ramp32);
        // Complex `Sqrt` takes a plain form for blocks of 64 items whose parts are all 0 or of moderate size, which gives exactly the full form's parts. Random items, with special or extreme items in some blocks and a run of real items, reach both forms.
        fn roots<F: Float + Debug>(z: &[Complex<F>]) where Sqrt: Unary<Complex<F>, Out = Complex<F>> {
            let exact = |a: F, b: F| (a == b && a.is_sign_negative() == b.is_sign_negative()) || (a.is_nan() && b.is_nan());
            let mut out = Vec::new();
            Sqrt.many(Arg::Many(z), z.len(), &mut out);
            for (&z, &r) in z.iter().zip(&out) {
                let s = Sqrt.one(z);
                assert!(exact(r.re, s.re) && exact(r.im, s.im), "complex Sqrt({z:?}): {r:?} against {s:?}")
            }
        }
        let mut z: Vec<Complex<f64>> = (0..2000).map(|_| Complex::new(8.0 * rng.random::<f64>() - 4.0, 8.0 * rng.random::<f64>() - 4.0)).collect();
        for (i, (re, im)) in [(0.0, 0.0), (f64::INFINITY, 1.0), (1.0, f64::NAN), (1e-300, 0.0), (2.0, 1e-160), (1e160, 1.0)].into_iter().enumerate() {
            z[300 * i + 5] = Complex::new(re, im)
        }
        for v in &mut z[1600..1700] { v.im = 0.0 }
        roots(&z);
        roots(&z.iter().map(|z| Complex::new(z.re as f32, z.im as f32)).collect::<Vec<_>>());
        for b in [2.0, 10.0] {
            let mut out = Vec::new();
            let ok = Logb.many(Arg::One(b), Arg::Many(&x), x.len(), &mut out);
            assert!(x.iter().zip(&out).all(|(&v, &r)| same(r, Logb.one(b, v))), "Logb at base {b}");
            assert_eq!(ok, x.iter().all(|&v| Logb.checked(b, v).is_some()), "Logb at base {b}");
        }
        assert!(
            <Exp as Results<f64, 1>>::TOTAL && <Maximum as Results<f32, 2>>::TOTAL && !<Pow as Results<f64, 2>>::TOTAL && !<Sqrt as Results<f16, 1>>::TOTAL
        );
        assert!(<Add as Results<i64, 2>>::WRAPS && <Pow as Results<u8, 2>>::WRAPS && !<Add as Results<f64, 2>>::WRAPS && !<Binom as Results<i64, 2>>::WRAPS);
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
    use rustymath::{Add, Arg, Binary, Subtract, tolerant::{Less, Tolerance}};

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
        assert_eq!((maximum(1.0, f64::NAN).is_nan(), minimum(f32::NAN, 1.0), maximum(-1.0, 2.0), minimum(f16::ONE, f16::ZERO)), (true, 1.0, 2.0, f16::ZERO));
        assert_eq!((sum(&[i64::MAX, 1]), sum(&[i64::MAX, 1, -2]), prod(&[i64::MAX, 2])), ((i64::MIN, false), (i64::MAX - 1, true), (-2, false)));
        assert!(!cumsum(&[i64::MAX, 1], None, &mut Vec::new()));
        assert!(!cumprod(&[i32::MAX, i32::MAX, i32::MAX], None, &mut Vec::new()));
        let mut out = vec![7.0];
        assert!(cumsum(&[1.0f64, 2.0], Some(10.0), &mut out));
        assert_eq!(out, [7.0, 11.0, 13.0]);
    }

    /// A float maximum or minimum at f64, f32 and f16, by fold, scan, column fold, window and the pieces of a long lane, with NaNs where those routes split their items. Each result must be NaN or the greatest or least of its items that aren't NaN, and never any other number. A scan's seed counts as one more item. The cases include the fold that once gave 671743.5: a long lane whose pieces fold apart, with a NaN beside its greatest item. They also include a leading NaN in short folds and scans, and NaNs at the first and third items of the groups of four that x86's and f16's scans fold apart.
    #[test]
    fn extrema_beside_nan() {
        fn routes<T: Copy + Into<f64> + Send + Sync>(from: fn(f64) -> T) where Max: Reduction<T, Total = T>, Min: Reduction<T, Total = T> {
            let step = |a: f64, v: f64, greatest: bool| {
                if v.is_nan() { a } else if greatest { a.max(v) } else { a.min(v) }
            };
            let unit = |greatest: bool| if greatest { f64::NEG_INFINITY } else { f64::INFINITY };
            let want = |items: &mut dyn Iterator<Item = T>, greatest: bool| items.fold(unit(greatest), |a, v| step(a, v.into(), greatest));
            let check = |got: T, want: f64, case: &str| {
                let got: f64 = got.into();
                assert!(got.is_nan() || got == want, "{case}: {got}, where {want} or NaN");
            };
            let piece = 1 << 18;
            for (len, nans) in [
                (3, vec![0]), (3, vec![0, 1, 2]), (64, vec![0]), (64, vec![2, 4]), (200, vec![63, 64, 130]),
                (3 * piece + 5, vec![piece - 1, piece, 2 * piece + 7]),
            ] {
                let mut x: Vec<T> = (0..len).map(|i| from((i * 37 % 101) as f64 - 50.0)).collect();
                let last = nans[nans.len() - 1];
                if last + 1 < len { x[last + 1] = from(1e4) }
                if last > 0 { x[last - 1] = from(-1e4) }
                for &k in &nans { x[k] = from(f64::NAN) }
                for greatest in [true, false] {
                    let case = format!("{len} items, NaN at {nans:?}, maximum {greatest}");
                    let all = want(&mut x.iter().copied(), greatest);
                    check(if greatest { max(&x) } else { min(&x) }, all, &case);
                    check(if greatest { along::<T, Max>(&x, [1, len, 1]).0[0] } else { along::<T, Min>(&x, [1, len, 1]).0[0] }, all, &case);
                    for start in [None, Some(from(10.0)), Some(from(f64::NAN))] {
                        let mut out = Vec::new();
                        if greatest { <Max as Reduction<T>>::scan(&x, start, &mut out); }
                        else { <Min as Reduction<T>>::scan(&x, start, &mut out); }
                        let mut seen = start.map_or(unit(greatest), |s| step(unit(greatest), s.into(), greatest));
                        for (i, &got) in out.iter().enumerate() {
                            seen = step(seen, x[i].into(), greatest);
                            check(got, seen, &format!("{case}, scan from {:?}, item {i}", start.map(Into::<f64>::into)));
                        }
                    }
                    for inner in [1, 3, 7] {
                        let rows = len / inner;
                        let data = &x[..rows * inner];
                        let column = |j: usize| (0..rows).map(move |r| data[r * inner + j]);
                        let folds = if greatest { along::<T, Max>(data, [1, rows, inner]).0 } else { along::<T, Min>(data, [1, rows, inner]).0 };
                        for j in 0..inner { check(folds[j], want(&mut column(j), greatest), &format!("{case}, column {j} of {inner}")); }
                        for width in [3, 50].into_iter().filter(|&w| w <= rows && len <= 200) {
                            let windows =
                                if greatest { sliding::<T, Max>(data, [1, rows, inner], width).0 } else { sliding::<T, Min>(data, [1, rows, inner], width).0 };
                            for w in 0..=rows - width {
                                for j in 0..inner {
                                    check(
                                        windows[w * inner + j], want(&mut (w..w + width).map(|r| data[r * inner + j]), greatest),
                                        &format!("{case}, window {w} of {width} rows of {inner}"));
                                }
                            }
                        }
                    }
                }
            }
        }
        routes::<f64>(|v| v);
        routes::<f32>(|v| v as f32);
        routes::<f16>(f16::from_f64);
    }

    /// Every lane of an array of shape `[outer, len, inner]`, its items in order.
    fn lanes_of<T: Copy>(x: &[T], [outer, len, inner]: [usize; 3]) -> Vec<Vec<T>> {
        (0..outer).flat_map(|i| (0..inner).map(move |k| (0..len).map(|j| x[(i * len + j) * inner + k]).collect())).collect()
    }

    /// Folds of an array that a mapping lays over its source, against folds of the same array built. Transposes in two and three dimensions, reversed axes, a block of rows, and 2 by 2 windows of a 4 by 6 matrix that don't overlap each read a contiguous block, which folds in its own order. Overlapping windows and a broadcast read the source block by block. Each folds along every axis. `Mapping::block` lays out the windows' block as window row, row within a window, window column and column within a window.
    #[test]
    fn mapped_folds_match_built_copies() {
        use rustymath::{agreement::{Block, Mapping}, gather::{Same, Steps, walk}, layout::Lanes};
        let x: Vec<i64> = (0..120).map(|i| (i * 37 % 101) - 50).collect();
        let s = |start, len, step| Steps::Stride { start, len, step };
        for (shape, steps) in [
            (vec![7, 6], vec![s(0, 7, 1), s(0, 6, 7)]), (vec![6, 7], vec![s(0, 6, 7), s(6, 7, -1)]), (vec![4, 7], vec![s(14, 4, 7), s(0, 7, 1)]),
            (vec![6, 4, 5], vec![s(0, 6, 1), s(0, 4, 30), s(0, 5, 6)]), (vec![6, 7], vec![s(35, 6, -7), s(0, 7, 1)]),
            (vec![5, 3], vec![s(0, 5, 1), s(0, 3, 1)]), (vec![2, 3, 2, 2], vec![s(0, 2, 12), s(0, 3, 2), s(0, 2, 6), s(0, 2, 1)]),
            (vec![4, 5], vec![s(0, 4, 0), s(0, 5, 1)]),
        ] {
            let mut built = Vec::new();
            walk(&mut built, &x, 0, &steps, 0, Same);
            let mapping = Mapping::Steps { base: 0, steps: steps.into() };
            for axis in 0..shape.len() {
                let dims = Lanes::new(&shape, axis).unwrap().dims();
                let (mut totals, fits, order) = along_mapped::<i64, Sum>(&x, &mapping, &shape, axis);
                if let Some(tables) = order {
                    let mut moved = Vec::new();
                    walk(&mut moved, &totals, 0, &tables, 0, Same);
                    totals = moved;
                }
                assert_eq!((totals, fits), along::<i64, Sum>(&built, dims), "{shape:?} {axis}");
            }
        }
        let pool = Mapping::Steps { base: 0, steps: vec![s(0, 2, 12), s(0, 3, 2), s(0, 2, 6), s(0, 2, 1)].into() };
        let block = Block { range: 0..24, dims: vec![2, 2, 3, 2], axes: vec![Some((0, false)), Some((2, false)), Some((1, false)), Some((3, false))] };
        assert_eq!(pool.block(&[2, 3, 2, 2]), Some(block));
    }

    /// Folds and scans along the middle axis against the same taken lane by lane, item by item, for lanes of consecutive items and otherwise, short and long, and empty. Folding in blocks of 13 splits lanes at many offsets. Booleans fold by each of their reductions, and an `i64` sum that wraps reports it, along a lane and down the rows.
    #[test]
    fn axes_match_lane_by_lane() {
        for shape in
            [[3, 37, 5], [15, 37, 1], [4, 3, 2], [2, 0, 3], [2, 200, 1], [70, 3, 1], [5, 64, 1], [65, 1, 1], [70, 2, 1], [3, 50, 2], [2, 5000, 1], [1, 5000, 2]]
        {
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
            let mut ordered = Vec::new();
            assert!(left_scan(&Add, &x, shape, Some(10), &mut ordered));
            assert_eq!(ordered, want, "{shape:?}");
            ordered.clear();
            assert!(left_scan(&Subtract, &x, shape, None, &mut ordered));
            let mut running = Vec::new();
            difference_scan_along(&x, shape, None, &mut running);
            assert_eq!(running, ordered, "{shape:?}");
            running.clear();
            difference_scan_as::<_, i64>(&x, shape, None, &mut running);
            assert_eq!(running, ordered, "{shape:?}");
            let mut differences = Vec::new();
            assert!(adjacent(
                &x, shape, &mut differences, |_, _| true, |before, next, out| {
                    out.extend(next.iter().zip(before).map(|(&a, &b)| a - b));
                    true
                }
            ));
            let want: Vec<_> = x.chunks((len * inner).max(1)).flat_map(|lane| lane[inner..].iter().zip(lane).map(|(&a, &b)| a - b)).collect();
            assert_eq!(differences, want, "{shape:?}");
            for seed in [None, Some(10)] {
                ordered.clear();
                assert!(left_scan(&Less(Tolerance::EXACT), &x, shape, seed, &mut ordered));
                let starts = seed.map(|s| x.chunks((len * inner).max(1)).flat_map(|lane| lane[..inner].iter().map(move |&v| s < v)).collect::<Vec<_>>());
                let (bits, ok) =
                    comparison_scan(&x, shape, starts.as_deref(), 0, 1, &|t, xs, out| Less(Tolerance::EXACT).many(t, Arg::Many(xs), xs.len(), out));
                assert!(ok && bits.len() == ordered.len());
                assert!(
                    bits.iter().zip(&ordered).enumerate().all(|(i, (&bit, &v))| seed.is_none() && i % (len * inner) < inner || i64::from(bit) == v),
                    "{shape:?} {seed:?}"
                );
            }
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
                    // `fold` gives each lane's right fold by the table, `x₀ f (x₁ f (… f last))`, with `start` as every lane's tail where it's given.
                    if Logical::folds(shape) {
                        let folds: Vec<bool> = lanes.iter().map(|l| {
                            let (last, items) = match start { Some(s) => (s, &l[..]), None => (l[len - 1], &l[..len - 1]) };
                            items.iter().rev().fold(last, |r, &v| table.0[usize::from(v)][usize::from(r)])
                        }).collect();
                        assert_eq!(table.fold(&b, len, start.map(|s| vec![s; outer]).as_deref()), folds, "{shape:?} {start:?} {code}");
                    }
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
        assert!(!left_scan(&Add, &[i64::MAX, 1], [1, 2, 1], None, &mut Vec::new()));
        assert!(!left_scan(&Add, &[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, i64::MAX, 1], [8, 2, 1], None, &mut Vec::new()));
    }

    /// Each window's fold matches the same reduction of that window's items alone, for windows folded directly, across the blocks of a long lane and by van Herk-Gil-Werman, a last block cut short, a window as wide as its lane, and windows down rows. A float maximum skips each NaN item, as `Max::step` does. Each window's pick is its first row that gives its maximum or minimum, and so is each whole lane's pick and each running pick. A sum that wraps reports it.
    #[test]
    fn windows_fold_as_their_items_do() {
        let same = |a: f64, b: f64| a == b || a.is_nan() && b.is_nan();
        for shape in [[1usize, 100, 1], [3, 37, 2], [2, 25, 1], [1, 9, 4], [2, 0, 3], [1, 2500, 1], [2, 600, 3]] {
            let [outer, len, inner] = shape;
            let x: Vec<i64> = (0..outer * len * inner).map(|i| (i * 37 % 101) as i64 - 50).collect();
            let f: Vec<f64> = x.iter().map(|&v| if v == 7 { f64::NAN } else { v as f64 / 7.0 }).collect();
            let b: Vec<bool> = x.iter().map(|v| v % 3 == 0).collect();
            for width in [1, 2, 3, 8, 9, 10, 25, len] {
                if width == 0 || width > len.max(1) { continue; }
                let count = (len + 1).saturating_sub(width);
                // The items of window `a` of the lane at `(i, k)`.
                let window = |a: usize, i: usize, k: usize| (a..a + width).map(move |j| (i * len + j) * inner + k);
                let each = |push: &mut dyn FnMut(usize, usize, usize)| {
                    for i in 0..outer { for a in 0..count { for k in 0..inner { push(a, i, k) } } }
                };
                let (mut sums, mut maxima, mut anys) = (Vec::new(), Vec::new(), Vec::new());
                each(&mut |a, i, k| {
                    sums.push(window(a, i, k).map(|p| x[p]).sum::<i64>());
                    maxima.push(window(a, i, k).map(|p| f[p]).fold(<Max as Reduction<f64>>::unit(), <Max as Reduction<f64>>::step));
                    anys.push(window(a, i, k).any(|p| b[p]));
                });
                assert_eq!(sliding::<i64, Sum>(&x, shape, width), (sums, true), "{shape:?} {width}");
                let (got, ok) = sliding::<f64, Max>(&f, shape, width);
                assert!(
                    ok && got.len() == maxima.len() && got.iter().zip(&maxima).all(|(&g, &m)| same(g, m)), "{shape:?} {width} {:?}",
                    got.iter().zip(&maxima).find(|&(&g, &m)| !same(g, m))
                );
                assert_eq!(sliding::<bool, Max>(&b, shape, width).0, anys, "{shape:?} {width}");
                // Each pick is the first row of its window whose item, folded alone, gives the window's fold.
                let pick = |r: &dyn Fn(f64) -> f64, folds: &[f64]| {
                    let mut at = 0;
                    let mut picks = Vec::new();
                    each(&mut |a, i, k| {
                        picks.push(window(a, i, k).position(|p| same(r(f[p]), folds[at])).map_or(a, |j| a + j));
                        at += 1;
                    });
                    picks
                };
                let (maxima, minima) = (sliding::<f64, Max>(&f, shape, width).0, sliding::<f64, Min>(&f, shape, width).0);
                assert_eq!(picks::<f64, Max>(&f, shape, Picks::Windows(width)), pick(&|x| <Max as Reduction<f64>>::one(x).0, &maxima), "{shape:?} {width}");
                assert_eq!(picks::<f64, Min>(&f, shape, Picks::Windows(width)), pick(&|x| <Min as Reduction<f64>>::one(x).0, &minima), "{shape:?} {width}");
            }
            if len == 0 { continue; }
            assert_eq!(picks::<f64, Max>(&f, shape, Picks::Lanes), picks::<f64, Max>(&f, shape, Picks::Windows(len)), "{shape:?}");
            // A running pick is the first row so far whose item, folded alone, gives the running maximum.
            let one = |p: usize| <Max as Reduction<f64>>::one(f[p]).0;
            let running: Vec<usize> = (0..outer * len * inner).map(|p| {
                let (i, r, k) = (p / (len * inner), p / inner % len, p % inner);
                let rows = (0..=r).map(|j| (i * len + j) * inner + k);
                let m = rows.clone().map(one).fold(f64::NEG_INFINITY, f64::max);
                rows.clone().position(|q| one(q) == m).unwrap_or(0)
            }).collect();
            assert_eq!(picks::<f64, Max>(&f, shape, Picks::Running), running, "{shape:?}");
        }
        assert!(!sliding::<i64, Sum>(&[i64::MAX, 1, 0], [1, 3, 1], 2).1);
        assert!(!sliding::<i64, Sum>(&[i64::MAX; 20], [1, 20, 1], 10).1);
    }

    /// Each window along two axes folds as its items do, for windows that slide or tile along either axis, a window of one row along one of them, and a float maximum beside NaN. A window wider than the direct folds take gives `None`.
    #[test]
    fn windows_along_two_axes_fold_as_their_items_do() {
        for (dims, widths, every) in
            [([2, 7, 3, 9, 2], [3, 2], [1, 1]), ([1, 8, 1, 6, 4], [2, 2], [2, 2]), ([3, 6, 2, 5, 1], [2, 3], [2, 1]), ([1, 4, 1, 4, 3], [4, 1], [1, 1])] {
                let [outer, len1, mid, len2, inner] = dims;
                let x: Vec<i64> = (0..dims.iter().product::<usize>()).map(|i| (i * 37 % 101) as i64 - 50).collect();
                let f: Vec<f64> = x.iter().map(|&v| if v == 7 { f64::NAN } else { v as f64 / 7.0 }).collect();
                let at = |o: usize, r: usize, m: usize, s: usize, c: usize| (((o * len1 + r) * mid + m) * len2 + s) * inner + c;
                let (c1, c2) = ((len1 - widths[0]) / every[0] + 1, (len2 - widths[1]) / every[1] + 1);
                let (mut sums, mut maxima) = (Vec::new(), Vec::new());
                for o in 0..outer {
                    for i in 0..c1 {
                        for m in 0..mid {
                            for j in 0..c2 {
                                for c in 0..inner {
                                    let items: Vec<usize> = (0..widths[0]).flat_map(|a| (0..widths[1]).map(move |b| (a, b)))
                                        .map(|(a, b)| at(o, every[0] * i + a, m, every[1] * j + b, c)).collect();
                                    sums.push(items.iter().map(|&p| x[p]).sum::<i64>());
                                    maxima.push(items.iter().map(|&p| f[p]).fold(<Max as Reduction<f64>>::unit(), <Max as Reduction<f64>>::step));
                                }
                            }
                        }
                    }
                }
                assert_eq!(sliding_2d::<i64, Sum>(&x, dims, widths, every), Some((sums, true)), "{dims:?} {widths:?} {every:?}");
                let (got, ok) = sliding_2d::<f64, Max>(&f, dims, widths, every).expect("windows of at most 8 rows fold directly");
                assert!(
                    ok && got.len() == maxima.len() && got.iter().zip(&maxima).all(|(&g, &m)| g == m || g.is_nan() && m.is_nan()),
                    "{dims:?} {widths:?} {every:?}"
                );
            }
        assert_eq!(sliding_2d::<i64, Sum>(&[0; 80], [1, 4, 1, 20, 1], [2, 9], [1, 1]), None);
    }

    /// Folds that run in pieces, of long lanes and of single matrices down their columns, folds of many short lanes or groups, and folds of a few wide rows by columns, which split across threads, match the sums taken lane by lane, on one thread and on four.
    #[test]
    fn long_and_many_lanes() {
        let _threads = crate::support::threads_lock();
        for shape in
            [[1, (3 << 18) + 5, 1], [2, (1 << 18) + 7, 3], [5000, 200, 1], [300, 50, 70], [1, 2000, 300], [1, 600, 2000], [1, 8, 100000], [3, 5, 60000]] {
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
        assert_eq!(alternating_lanes::<_, Sum>(&[1i64, 2, 3, 4, 5, 6], 3), Some(vec![2, 5]));
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
    fn selected_order() {
        use rustymath::buffer::{Element, View};
        let x: Vec<i64> = (0..1000).map(|i| (i * 37 % 13) - 6).collect();
        for down in [false, true] {
            let order = view_grade(i64::view(&x), down).unwrap();
            assert_eq!(i64::vec(view_sort(i64::view(&x), down).unwrap().unwrap()).unwrap(), order.iter().map(|&i| x[i as usize]).collect::<Vec<_>>());
            for range in [0..10, 495..505, 999..1000] {
                let selected = view_selected(i64::view(&x), range.clone(), down).unwrap();
                assert_eq!(selected, order[range]);
            }
        }
        let mut x = [9., 1., 5., 3., 7.];
        assert_eq!(quantiles(&mut x, &[0., 0.25, 0.5, 1.]).unwrap(), [1., 3., 5., 9.]);
        assert!(quantiles(&mut x, &[1.5]).is_err() && quantiles(&mut x, &[f64::NAN]).is_err());
        let part = view_partition(View::Floats(rustymath::buffer::Floats::F64(&[f64::NAN, -0., 0., 3.])), 1).unwrap();
        assert_eq!(f64::of(part.view()).unwrap()[1], 0.);
    }

    #[test]
    fn float_sorts_follow_total_keys() {
        use rustymath::{Float, buffer::Element};
        fn check<F: Float + Element>(x: &[F], dims @ [outer, n, inner]: [usize; 3], bits: impl Fn(F) -> u64) {
            for down in [false, true] {
                let got = F::vec(view_sort_along(F::view(x), dims, down).unwrap().unwrap()).unwrap();
                for (o, i) in (0..outer).flat_map(|o| (0..inner).map(move |i| (o, i))) {
                    let at = |k: usize| (o * n + k) * inner + i;
                    let mut keys: Vec<_> = (0..n).map(|k| x[at(k)].total_key()).collect();
                    keys.sort();
                    if down { keys.reverse() }
                    let want: Vec<u64> = keys.into_iter().map(|k| bits(F::from_order_key(k))).collect();
                    assert_eq!((0..n).map(|k| bits(got[at(k)])).collect::<Vec<_>>(), want, "{dims:?} {down}");
                }
            }
        }
        // Every 11th item is special: NaNs of both signs and a signalling one, both zeros, both infinities, a subnormal. The rest are whole numbers with many repeats.
        let specials = [f64::NAN, -f64::NAN, f64::from_bits(0x7ff0_0000_0000_0001), -0.0, 0.0, f64::INFINITY, -f64::INFINITY, 5e-324, -1.5, 1.5];
        // Lengths below, inside and above the range that the network sorts, and around its powers of two.
        for n in [0, 1, 7, 31, 32, 33, 63, 64, 65, 100, 255, 256, 257, 1000, 1024, 8191, 8192, 8193, 16384, 16385] {
            for dims in [[1, n, 1], [3, n, 2]] {
                let x: Vec<f64> = (0..dims.iter().product())
                    .map(|i| if i % 11 == 0 { specials[i / 11 % specials.len()] } else { (i as f64 * 0.618034 % 1.0 * 64.0).round() - 32.0 }).collect();
                check(&x, dims, f64::to_bits);
                check(&x.iter().map(|&v| v as f32).collect::<Vec<_>>(), dims, |v: f32| v.to_bits().into());
            }
        }
    }

    #[test]
    fn search_and_sort() {
        use rustymath::{Float, buffer::Element, tolerant::Tolerance};
        let x = [30, 10, 20, 10];
        assert_eq!(index_of(&x, &[10, 40]), [1, 4]);
        // More than `SCANNED` needles build the index in place of scanning.
        let many: Vec<i32> = (0..20).map(|i| [10, 40, 30][i % 3]).collect();
        assert_eq!(index_of(&x, &many), (0..20).map(|i| [1, 4, 0][i % 3]).collect::<Vec<_>>());
        let n = numbered(&classes(&x));
        assert_eq!((&n, n.counts()), (&Numbered { firsts: vec![0, 1, 2], numbers: vec![0, 1, 2, 1] }, vec![1, 2, 1]));
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
        assert_eq!(grouped(&[1, 0, 0, 0, 0, 0, 1, 1], 2).each().collect::<Vec<_>>(), [&[1, 2, 3, 4, 5][..], &[0, 6, 7][..]]);
        assert_eq!(grouped(&[0, 0, 0, 0, 0, 0, 0, 0, 0, 1], 2).each().collect::<Vec<_>>(), [&(0..9).collect::<Vec<u32>>()[..], &[9][..]]);
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
        assert_eq!(
            (
                grouped(&[1, 0, 1, 1], 2).each().collect::<Vec<_>>(), grouped(&[0, 0], 1).each().collect::<Vec<_>>(),
                grouped(&[0, 0], 2).each().collect::<Vec<_>>()
            ), (vec![&[1][..], &[0, 2, 3][..]], vec![&[0, 1][..]], vec![&[0, 1][..], &[][..]])
        );
        // `grouped_keys` gives the classes that `classes` and `numbered` give, each with its positions. The five keys first appear out of their order, and a range of 2^20 + 1 is too wide for a table.
        let together = |k: &[i32]| {
            let n = numbered(&classes(k));
            let mut each = vec![Vec::new(); n.firsts.len()];
            for (i, &g) in n.numbers.iter().enumerate() { each[g as usize].push(i as u32) }
            let g = grouped_keys(k).expect("a table holds the keys");
            assert_eq!((&g.firsts, g.groups.each().collect::<Vec<_>>()), (&n.firsts, each.iter().map(Vec::as_slice).collect()));
            let buffers = g.groups.into_buffers();
            assert!(buffers == each || buffers == [each.concat()]);
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
        // Heterogeneous, strided columns share native-width key construction, with a single comparison sort for small tables and radix passes for larger ones.
        for n in [32, 300] {
            let rows: Vec<i16> = (0..n).flat_map(|i| [i as i16 % 9 - 4, 0, i as i16 % 5]).collect();
            let second: Vec<f32> = (0..n).map(|i| if i % 19 == 0 { f32::NAN } else { (i % 17) as f32 / 7. }).collect();
            for down in [false, true] {
                let mut expected: Vec<u32> = (0..n as u32).collect();
                expected.sort_by_key(|&i| {
                    let k = (rows[i as usize * 3] as u16 ^ 1 << 15, second[i as usize].order_key());
                    if down { (!k.0, !k.1) } else { k }
                });
                let order = view_lexgrade(n, [(i16::view(&rows), 0, 3), (f32::view(&second), 0, 1)].into_iter(), down).unwrap();
                assert_eq!(order, expected);
                let ranks = inverse_permutation::<u32>(order.iter().map(|&i| i as usize));
                for (rank, &position) in order.iter().enumerate() { assert_eq!(ranks[position as usize], rank as u32); }
            }
        }
        let x = [1u8, 2, 1];
        assert_eq!(view_index_of(u8::view(&x), f64::view(&[1., 256., -1., 2.]), Tolerance::DEFAULT).unwrap().unwrap(), [0, 3, 3, 1]);
        assert!(view_index_of(u8::view(&x), f64::view(&[1. + 1e-15]), Tolerance::DEFAULT).is_none());
        let groups = numbered(&[0, 1, 0]);
        assert_eq!(groups.fold::<_, rustymath::reduce::Sum>(&[i64::MAX, 2, 1]), (vec![i64::MIN, 2], false));
        assert_eq!(find_with(&[2], &[2, 3], |i, j| Ok::<_, ()>([1, 2][i] == [1, 2, 1, 1, 2, 3][j])).unwrap(), [true, false, false, true, false, false]);
        assert_eq!(
            view_find(i16::view(&[1, 2]), &[2], u8::view(&[1, 2, 1, 1, 2, 3]), &[2, 3], Tolerance::EXACT), Some(vec![true, false, false, true, false, false])
        );
        assert_eq!(view_find(bool::view(&[true, false]), &[2], bool::view(&[true, false, true]), &[3], Tolerance::EXACT), Some(vec![true, false, false]));
        assert_eq!(view_find(char::view(&['a']), &[1], char::view(&['b', 'a']), &[2], Tolerance::EXACT), Some(vec![false, true]));
        for (t, first) in [(Tolerance::DEFAULT, true), (Tolerance::EXACT, false)] {
            assert_eq!(
                view_find(f32::view(&[1., f32::NAN]), &[2], f64::view(&[1. + 1e-15, f64::NAN, 1., f64::NAN]), &[4], t), Some(vec![first, false, true, false])
            );
        }
        assert!(view_find(u8::view(&[1]), &[1], f64::view(&[1.]), &[1], Tolerance::DEFAULT).is_none());
        for size in [4, 1_000_000] { assert_eq!(occurrences(&[1, 2, 1, 3, 1], size), Some((vec![0, 0, 1, 0, 2], 3))); }
        assert_eq!((occurrences(&[1, 2, 3], 4), occurrences(&[], 0)), (None, None));
    }

    /// Over more than one 65,536-key piece, the radix sort's passes and the lookups split across threads and give one thread's results. Equal keys keep their order of position.
    #[test]
    fn pieces_across_threads() {
        let _threads = crate::support::threads_lock();
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
            by_threads.push(tolerant::index_of(Tolerance::DEFAULT, &floats, &near).unwrap());
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
        let (x, y) = (2.2500000000000018, 0.7500000000000007);
        assert_eq!((FloorDivide(Tolerance::EXACT).one(x, y), Residue(Tolerance::EXACT).one(y, x)), (2.0, x - 2.0 * y));
        if rustymath::FUSED { assert_ne!(Exactly.residue(0.1, 1.0), 0.0); }
        assert_eq!(Exactly.residue(1e300, 1e-300), 1e-300);
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

    /// An `Indicator` gives each comparison's Booleans as 1 and 0 at each float width, over more items than its buffer holds, with one number on either side and with slices on both.
    #[test]
    fn indicator() {
        fn check<F: Float + std::fmt::Debug, K: Binary<F, Out = bool> + Copy>(k: K) where Indicator<K>: Binary<F, Out = F> {
            let x: Vec<F> = (0..700).map(|i| F::narrow(f64::from(i % 7) - 3.0)).collect();
            let zero = F::narrow(0.0);
            for (a, b) in [(Arg::Many(&x[..]), Arg::One(zero)), (Arg::One(zero), Arg::Many(&x[..])), (Arg::Many(&x[..]), Arg::Many(&x[1..]))] {
                let (mut bools, mut lanes) = (Vec::new(), Vec::new());
                assert!(k.many(a, b, 699, &mut bools) & Indicator(k).many(a, b, 699, &mut lanes));
                assert_eq!(lanes, bools.iter().map(|&b| F::narrow(if b { 1.0 } else { 0.0 })).collect::<Vec<_>>());
            }
        }
        macro_rules! comparisons {
            ($f:ty, $t:expr) => {
                check::<$f, _>(Equal($t));
                check::<$f, _>(NotEqual($t));
                check::<$f, _>(Less($t));
                check::<$f, _>(LessEqual($t));
                check::<$f, _>(Greater($t));
                check::<$f, _>(GreaterEqual($t));
                check::<$f, _>(Matches($t));
            };
        }
        for t in [T, Tolerance::EXACT] {
            comparisons!(f64, t);
            comparisons!(f32, t);
        }
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
            assert_eq!(index_of(t, &x, &y).unwrap(), brute, "{t:?}");
            assert_eq!(index_of(t, &x, &y[..5]).unwrap(), brute[..5], "{t:?}, scanning");
        }
        for t in [T, Tolerance::new(1e-3).unwrap(), Tolerance::new(0.3).unwrap()] {
            check::<f64>(t);
            check::<f32>(t);
            check::<f16>(t);
        }
        let x = [0.3, 1.0, 0.1 + 0.2];
        assert_eq!((index_of(T, &x, &[0.1 + 0.2, 2.0]).unwrap(), index_of(Exactly, &x, &[0.1 + 0.2, 2.0]).unwrap()), (vec![0, 3], vec![2, 3]));
        // Floats that each equal a whole number index as integers. At a tolerance of 0.3, which makes 3 equal 4, `whole_limit` lets only 0 qualify, so the search falls back to buckets.
        let wx: Vec<f64> = (0..300u64).map(|i| (crate::support::random(i) % 50) as f64 - 25.0).collect();
        let wy: Vec<f64> = (0..40).map(|i| i as f64 - 30.0).collect();
        for t in [T, Tolerance::EXACT, Tolerance::new(0.3).unwrap()] {
            let brute: Vec<u32> = wy.iter().map(|&v| wx.iter().position(|&h| t.matches(h, v)).unwrap_or(wx.len()) as u32).collect();
            assert_eq!(index_of(t, &wx, &wy).unwrap(), brute, "{t:?}");
        }
        // Floats near one whole number each equal it, but needn't equal each other. An indexed search finds only true matches, as a scan of 16 or fewer does.
        let near = [1.0 - 8e-15, 1.0, 1.0 + 8e-15];
        assert_eq!((index_of(T, &near[..1], &[near[2]; 17]).unwrap(), index_of(T, &near[..2], &[near[2]; 17]).unwrap()), (vec![1; 17], vec![1; 17]));
        let ints: Vec<i64> = wx.iter().map(|&v| v as i64).collect();
        assert_eq!((classes(T, &wx).unwrap(), firsts(T, &wx).unwrap()), (search::classes(&ints), search::firsts(&ints)));
    }
}

mod gather {
    use rustymath::{agreement::{Mapping, expanded_items}, buffer::Element, gather::{Same, Steps, scatter, scatter_walk, walk}, structure::Walk};
    /// The items of `s` that `steps` reach from `base`, with -1 for a fill.
    fn read(s: &[i64], base: usize, steps: &[Steps]) -> Vec<i64> {
        let mut out = vec![];
        walk(&mut out, s, base, steps, -1, Same);
        out
    }
    fn along(len: usize, step: isize) -> Steps { Steps::Stride { start: 0, len, step } }
    #[test]
    fn walks_reach_and_run_agree() {
        assert_eq!(read(&[], 1, &[Steps::table(vec![Steps::FILL; 2]), along(2, 3)]), [-1; 4]);
        assert!(read(&[], 0, &[Steps::padded(0, 0, 0, 1, rustymath::structure::Edge::Wrap)]).is_empty());
        // Each item of `s` is its own offset, so a walk shows the offsets it read.
        let s: Vec<i64> = (0..600).collect();
        let cases = [
            (0, vec![along(3, 20), along(20, 1)]),
            (2, vec![along(11, 3)]),
            (2, vec![Steps::Stride { start: 30, len: 11, step: -3 }]),
            (0, vec![along(3, 20), Steps::Stride { start: 19, len: 20, step: -1 }]),
            (0, vec![along(2, 50), Steps::Rotated { shift: 7, len: 30, stride: 1 }]),
            (0, vec![along(2, 50), Steps::clipped(-3, 40, 30, 1)]),
            (0, vec![Steps::Rotated { shift: 4, len: 9, stride: 3 }]),
            (5, vec![along(3, 40), Steps::padded(-4, 12, 6, 0, rustymath::structure::Edge::Wrap)]),
            (0, vec![Steps::padded(-3, 20, 8, 2, rustymath::structure::Edge::Reflect)]),
            (0, vec![along(2, 100), Steps::clipped(-2, 9, 5, 3)]),
            (0, vec![Steps::table(vec![4, Steps::FILL, 1]), along(3, 1)]),
            (0, vec![Steps::table((0..10).collect()), along(4, 20)]),
            (0, vec![Steps::table(vec![3, Steps::FILL, 9, 0, 5, 1, 2, 7, 8, 4]), along(3, 40)]),
            (0, vec![along(9, 1), along(10, 30)]),
            (0, vec![along(4, 10), Steps::Stride { start: 2, len: 3, step: 0 }]),
            // Short cells whose runs pass `RUN` items: forwards along a stride, backwards, and along a table.
            (0, vec![along(5, 7), along(3, 40), along(9, 1)]),
            (0, vec![along(5, 7), along(2, 40), Steps::Stride { start: 19, len: 12, step: -1 }]),
            (0, vec![Steps::table(vec![3, 9, 0, 5]), along(2, 40), along(10, 1)]),
        ];
        for (base, steps) in &cases {
            let whole = read(&s, *base, steps);
            let mut mapped = vec![];
            walk(&mut mapped, &s, *base, steps, -1, |x: i64| x);
            assert_eq!(whole, mapped);
            for (i, &x) in whole.iter().enumerate() { assert_eq!(Steps::reach(*base, steps, i).map_or(-1, |o| o as i64), x) }
            // A mapping reads parts of the walk through the plan that it keeps, with the type's fill.
            let mapping = Mapping::Steps { base: *base, steps: steps.clone().into() };
            let filled: Vec<i64> = whole.iter().map(|&x| if x == -1 { i64::FILL } else { x }).collect();
            for (from, n) in [(1, 7), (5, whole.len() - 5), (0, whole.len())] {
                let mut buf = vec![];
                assert_eq!(expanded_items(&mapping, &s, from, n, &mut buf), &filled[from..from + n]);
            }
            // `scatter_walk` writes where the walk reads, in the same order, so it agrees with `scatter` at the listed offsets, for many items and for one.
            let offsets = Steps::combined(*base, steps);
            let many: Vec<i64> = (1000..1000 + whole.len() as i64).collect();
            for items in [&many[..], &[7]] {
                let (mut walked, mut listed) = (vec![-1; 600], vec![-1; 600]);
                scatter_walk(&mut walked, items, *base, steps, |x| x);
                scatter(&mut listed, &offsets, items, |x| x);
                assert_eq!(walked, listed);
            }
        }
        assert_eq!((Steps::run(0, &cases[0].1, 3, 5), Steps::run(0, &cases[0].1, 18, 5)), (Some((3, 1)), Some((18, 1))));
        assert_eq!(Steps::run(0, &[along(30, 2), along(1, 0)], 3, 20), Some((6, 2)));
        for (base, steps) in &cases {
            let len: usize = steps.iter().map(Steps::len).product();
            for from in 0..len.min(20) {
                for n in 1..=(len - from).min(20) {
                    if let Some((first, step)) = Steps::run(*base, steps, from, n) {
                        for i in 0..n { assert_eq!(Steps::reach(*base, steps, from + i), first.checked_add_signed(step * i as isize)); }
                    }
                }
            }
        }
    }
    /// Index arrays pair by broadcasting and take the place of the first one, while a range keeps its axis in place. With one index array, pairing is the cross product.
    #[test]
    fn index_arrays_pair() {
        use rustymath::{layout::Layout, structure::{Selector, select}};
        let s: Vec<i64> = (0..24).collect();
        let layout = Layout::from([2, 3, 4]);
        let ix = |positions: Vec<usize>| Selector::Indices { layout: Layout::from([positions.len()]), positions: rustymath::gather::Table::new(positions) };
        let paired = select(&layout, vec![ix(vec![0, 1]), Selector::all(3), ix(vec![1, 2])], true).unwrap();
        assert_eq!(paired.layout.shape(), &[2, 3]);
        assert_eq!(read(&s, 0, &paired.steps), [1, 5, 9, 14, 18, 22]);
        let one = [ix(vec![1, 0]), Selector::At(2)];
        assert_eq!(read(&s, 0, &select(&layout, one.to_vec(), true).unwrap().steps), read(&s, 0, &select(&layout, one.to_vec(), false).unwrap().steps));
    }
    /// Index arrays read as strides wherever their positions step evenly along each of their axes, alone or paired. Paired arrays that share no axis keep their own positions, as `⌷` keeps them, so a data-dependent array beside an evenly spaced one reads as a table and a stride. Arrays that share axes sum their positions, and an array that lacks one of those axes repeats along it.
    #[test]
    fn index_arrays_read_as_strides() {
        use rustymath::{gather::{Steps, Table}, layout::Layout, structure::{Selector, select}};
        let ix = |names: &[&str], shape: &[usize], positions: Vec<usize>| {
            let layout = Layout::from(shape.to_vec()).with_names(names.iter().map(|&n| Some(n.into())).collect()).unwrap();
            Selector::Indices { layout, positions: Table::new(positions) }
        };
        let s: Vec<i64> = (0..24).collect();
        let m = Layout::from([4, 6]);
        let whole = select(&m, vec![ix(&["r"], &[4], (0..4).collect()), ix(&["c"], &[6], (0..6).collect())], true).unwrap();
        assert!(Steps::in_order(&whole.steps));
        let diagonal = select(&Layout::from([4, 4]), vec![ix(&["i"], &[4], (0..4).collect()), ix(&["i"], &[4], (0..4).collect())], true).unwrap();
        assert_eq!(diagonal.steps, [Steps::Stride { start: 0, len: 4, step: 5 }]);
        let windows = select(&m, vec![Selector::all(4), ix(&["p", "q"], &[2, 3], vec![0, 1, 2, 2, 3, 4])], true).unwrap();
        assert!(windows.steps.iter().all(|s| matches!(s, Steps::Stride { .. })));
        assert_eq!(read(&s, 0, &windows.steps)[..6], [0, 1, 2, 2, 3, 4]);
        let rows = select(&m, vec![ix(&["r"], &[4], vec![2, 0, 3, 1]), ix(&["c"], &[6], (0..6).collect())], true).unwrap();
        assert!(matches!(rows.steps[..], [Steps::Table(_), Steps::Stride { .. }]));
        assert_eq!(read(&s, 0, &rows.steps), [12..18, 0..6, 18..24, 6..12].into_iter().flatten().collect::<Vec<i64>>());
        let sum = select(&m, vec![ix(&["i", "j"], &[2, 3], vec![0, 1, 2, 1, 2, 3]), ix(&["j"], &[3], vec![0, 1, 2])], true).unwrap();
        assert_eq!(sum.steps, [Steps::Stride { start: 0, len: 2, step: 6 }, Steps::Stride { start: 0, len: 3, step: 7 }]);
        let spread = select(&m, vec![ix(&["i", "j"], &[2, 3], vec![3, 0, 2, 1, 1, 0]), ix(&["j"], &[3], vec![5, 0, 4])], true).unwrap();
        assert_eq!(read(&s, 0, &spread.steps), [23, 0, 16, 11, 6, 4]);
        // An inner axis that steps down starts at its far end, so that each stride's own offsets stay at or above 0.
        let back = select(&m, vec![ix(&["p", "q"], &[2, 2], vec![1, 0, 1, 0])], true).unwrap();
        assert_eq!(back.steps[..2], [Steps::Stride { start: 0, len: 2, step: 0 }, Steps::Stride { start: 6, len: 2, step: -6 }]);
        assert_eq!(read(&s, 0, &back.steps), [6..12, 0..6, 6..12, 0..6].into_iter().flatten().collect::<Vec<i64>>());
    }
    /// An index array whose positions read a stretch of one axis past its ends, as `•edge`'s modes and positions that count back from the end give them, reads as one `Padded` entry, or as a `Rotated` one where it wraps once round the whole axis. Each reads what its table reads. Positions in any other order stay a table.
    #[test]
    fn index_arrays_read_past_edges() {
        use rustymath::{gather::{Steps, Table}, layout::Layout, structure::{Edge, Selector, select}};
        const F: usize = Steps::FILL;
        let s: Vec<i64> = (0..24).collect();
        // The rows of a 4 by 6 array at `positions`, checked against a table of them.
        let rows = |positions: &[usize]| {
            let layout = Layout::from([positions.len()]).with_names(vec![Some("r".into())]).unwrap();
            let plan =
                select(&Layout::from([4, 6]), vec![Selector::Indices { layout, positions: Table::new(positions.to_vec()) }, Selector::all(6)], true).unwrap();
            let table = Steps::table(positions.iter().map(|&p| if p == F { F } else { 6 * p }).collect());
            assert_eq!(read(&s, 0, &plan.steps), read(&s, 0, &[table, along(6, 1)]));
            plan.steps[0].clone()
        };
        let padded = |start, len, edge| Steps::Padded { start, len, size: 4, stride: 6, edge };
        assert_eq!(rows(&[3, 0, 1, 2]), Steps::Rotated { shift: 3, len: 4, stride: 6 });
        assert_eq!(rows(&[3, 0, 1]), padded(-1, 3, Edge::Wrap));
        assert_eq!(rows(&[F, 0, 1, 2, 3, F]), padded(-1, 6, Edge::Fill));
        assert_eq!(rows(&[1, 2, 3, F]), padded(1, 4, Edge::Fill));
        assert_eq!(rows(&[0, 0, 0, 1]), padded(-2, 4, Edge::Clamp));
        assert_eq!(rows(&[2, 1, 0, 1]), padded(-2, 4, Edge::Reflect));
        assert_eq!(rows(&[1, 0, 0, 1]), padded(-2, 4, Edge::Symmetric));
        assert!(matches!(rows(&[2, 0, 3, 1]), Steps::Table(_)));
    }
    /// A padded axis reads its edge mode's positions past its ends. Centred windows of 3 over 4 items read one position before the axis and one after it. A padded size of 0 is an error.
    #[test]
    fn window_padding() {
        use rustymath::{gather::{Same, walk}, structure::{Edge, Windows}};
        let read = |edge| {
            let mut out = vec![];
            walk(&mut out, &[10i64, 11, 12, 13], 0, &Windows::new(&[3], &[], |_| Some(edge), &[4]).unwrap().padding().unwrap().steps, -1, Same);
            out
        };
        assert_eq!(read(Edge::Wrap), [13, 10, 11, 12, 13, 10]);
        assert_eq!(read(Edge::Clamp), [10, 10, 11, 12, 13, 13]);
        assert_eq!(read(Edge::Fill), [-1, 10, 11, 12, 13, -1]);
        assert!(Windows::new(&[0], &[], |_| Some(Edge::Fill), &[4]).is_err());
    }
    /// A composed walk reads what walking one walk and then the other reads. `spreads` and `rereads` find walks that read an item more than once or read a fill.
    #[test]
    fn walks_compose() {
        let s: Vec<i64> = (0..600).collect();
        // An array of shape 6 by 10 stored from offset 5 with strides 30 and 2, read through reversed rows, a take, a rotation, a clipped read, and one row with fill around it, whose entry walks no axis.
        let (shape, inner) = ([6, 10], [along(6, 30), along(10, 2)]);
        let array = read(&s, 5, &inner);
        for outer in [
            vec![Steps::Stride { start: 50, len: 6, step: -10 }, along(10, 1)],
            vec![Steps::Stride { start: 10, len: 4, step: 10 }, Steps::Stride { start: 2, len: 5, step: 1 }],
            vec![along(6, 10), Steps::Rotated { shift: 3, len: 10, stride: 1 }], vec![along(6, 10), Steps::clipped(-2, 12, 10, 1)],
            vec![Steps::clipped(-2, 6, 1, 0), along(10, 1)],
        ] {
            let (base, steps) = Steps::through(&outer, &shape, 5, &inner).expect("these reads compose");
            assert_eq!(read(&s, base, &steps), read(&array, 0, &outer));
        }
        // A rotation of a fill-padded read composes where it carries only fill across the end, as `¯1⌽(n+2)↑v` does. Where it splits the source's items, no regular entry reads them.
        let (shape, inner) = ([8], [Steps::clipped(0, 8, 6, 1)]);
        let padded = read(&s, 5, &inner);
        let (base, steps) = Steps::through(&[Steps::rotated(7, 8, 1)], &shape, 5, &inner).expect("only fill crosses the end");
        assert_eq!(read(&s, base, &steps), read(&padded, 0, &[Steps::rotated(7, 8, 1)]));
        assert!(Steps::through(&[Steps::rotated(1, 8, 1)], &shape, 5, &inner).is_none());
        // An entry over a frame of 2 by 3 splits into one entry for each axis, which read the same items.
        let tables = [along(6, 4), along(4, 1)];
        let (base, steps) = Steps::per_axis(&tables, &[2, 3, 4]).expect("a stride splits");
        assert_eq!(read(&s, base, &steps), read(&s, 0, &tables));
        // `Walk::new` moves a walked axis's position into the first entry along it, and leaves the position on an axis no entry walks in the base. A read that crosses a row, or a rotation over part of an axis, has no form by axis.
        let walk = Walk::new(&[Steps::Stride { start: 12, len: 1, step: 0 }, along(3, 1)], &[3], &[4, 5]).expect("part of one row");
        assert!(walk.base == 10 && walk.along == [Some(1)] && matches!(walk.steps[..], [Steps::Stride { start: 2, len: 3, step: 1 }]));
        assert!(Walk::new(&[Steps::Stride { start: 3, len: 1, step: 0 }, along(3, 1)], &[3], &[4, 5]).is_none());
        assert!(Walk::new(&[Steps::Stride { start: 2, len: 1, step: 0 }, Steps::Rotated { shift: 1, len: 6, stride: 1 }], &[6], &[10]).is_none());
        assert!(along(3, 0).spreads() && Steps::table(vec![1, 1]).spreads() && !Steps::Rotated { shift: 1, len: 4, stride: 1 }.spreads());
        assert!(Steps::rereads(&[along(8, 1), along(3, 1)], &[10]) && !Steps::rereads(&[along(10, 1)], &[10]));
    }
    #[test]
    fn windows_are_recognised() {
        // `Walk::windows` gives each window axis's frames, window size and move, and refuses windows that don't start at 0.
        let windows = |t: &[Steps], shape: &[usize], source: &[usize]| Walk::new(t, shape, source).and_then(|w| w.windows(source));
        assert_eq!(windows(&[along(3, 6), along(4, 1), along(2, 6), along(3, 1)], &[3, 4, 2, 3], &[4, 6]), Some(vec![[3, 2, 1], [4, 3, 1]]));
        assert_eq!(windows(&[along(3, 2), along(3, 1)], &[3, 3], &[7]), Some(vec![[3, 3, 2]]));
        assert_eq!(windows(&[Steps::Stride { start: 2, len: 4, step: 1 }, along(3, 1)], &[4, 3], &[8]), None);
    }
}
