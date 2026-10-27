//! Matrix products and decompositions through their public APIs.
pub mod support;

use num_complex::Complex;
use rustymath::{Native, linalg::*};

fn identity<T: num_traits::Zero + num_traits::One>(n: usize) -> Vec<T> { (0..n * n).map(|k| if k % (n + 1) == 0 { T::one() } else { T::zero() }).collect() }

fn transpose<T: Copy>(a: &[T], m: usize, n: usize) -> Vec<T> { (0..m * n).map(|k| a[k % m * n + k / m]).collect() }

fn adjoint<T: Scalar>(a: &[T], m: usize, n: usize) -> Vec<T> { transpose(a, m, n).into_iter().map(T::conj).collect() }
mod products {
    use super::*;

    #[test]
    fn native_inner_pairs_keep_right_folds_and_bounded_terms() {
        use rustymath::{buffer::Element, kernels, tolerant::{Equal, Greater, Tolerance}};
        macro_rules! inner {
            ($f:expr, $g:expr, $a:expr, $b:expr, $size:expr) => {{
                let (f, g) = ($f, $g);
                let (fk, gk) = (kernels!(RightFold of f), kernels!(Binary of g));
                let checked = |(buffer, ok)| if ok { Ok(buffer) } else { Err(()) };
                inner_with($a, $b, $size,
                    |x, y, agreement| {
                        use rustymath::{array::ArrayView, layout::Layout};
                        assert!(agreement.len <= rustymath::CHUNK);
                        checked(gk.binary(&g, ArrayView::compact(&Layout::from([x.len()]), x), ArrayView::compact(&Layout::from([y.len()]), y), agreement).unwrap())
                    },
                    |x, dims, tail| checked(fk.run(&f, x, dims, tail).unwrap()),
                )
            }};
        }
        let [m, k, n] = [2, rustymath::CHUNK + 3, 3];
        let a = (0..m * k).map(|i| (i % 7) as i64 - 3).collect::<Vec<_>>();
        let b = (0..k * n).map(|i| (i % 5) as i64 - 2).collect::<Vec<_>>();
        let out = inner!(rustymath::Subtract, rustymath::Multiply, i64::view(&a), i64::view(&b), [m, k, n]).unwrap();
        let expected = (0..m * n).map(|ij| (0..k - 1).rev().fold(a[ij / n * k + k - 1] * b[(k - 1) * n + ij % n], |t, p| a[ij / n * k + p] * b[p * n + ij % n] - t)).collect::<Vec<_>>();
        assert_eq!(i64::of(out.view()).unwrap(), expected);
        let out = inner!(Equal(Tolerance::EXACT), rustymath::Add, i64::view(&a), i64::view(&b), [m, k, n]).unwrap();
        let expected = (0..m * n).map(|ij| {
            let term = |p| a[ij / n * k + p] + b[p * n + ij % n];
            let t = term(k - 2) == term(k - 1);
            (0..k - 2).rev().fold(t, |t, p| term(p) == i64::from(t))
        }).collect::<Vec<_>>();
        assert_eq!(bool::of(out.view()).unwrap(), expected);
        let wide = vec![1i64; 3 * (rustymath::CHUNK + 1)];
        let out = inner!(rustymath::Subtract, rustymath::Multiply, i64::view(&[1, 2, 3]), i64::view(&wide), [1, 3, rustymath::CHUNK + 1]).unwrap();
        assert!(i64::of(out.view()).unwrap().iter().all(|&x| x == 2));
        let out = inner!(rustymath::Add, Greater(Tolerance::EXACT), i64::view(&[2, 3, 4]), i64::view(&[1, 5, 2]), [1, 3, 1]).unwrap();
        assert_eq!(i64::of(out.view()).unwrap(), [2]);
        let out = inner!(Equal(Tolerance::EXACT), rustymath::Subtract, i64::view(&[3]), i64::view(&[2, 1]), [1, 1, 2]).unwrap();
        assert_eq!(i64::of(out.view()).unwrap(), [1, 2]);
        assert!(inner!(rustymath::Pow, rustymath::Add, f64::view(&[-1., 0.5]), f64::view(&[0., 2., 0., 1.5]), [1, 2, 2]).is_err());
        assert!(inner!(rustymath::Add, rustymath::Multiply, i64::view(&[i64::MAX, 1]), i64::view(&[2, 2]), [1, 2, 1]).is_err());
        assert!(inner!(rustymath::Add, rustymath::Multiply, i64::view(&[]), i64::view(&[2, 2]), [0, 2, 1]).unwrap().is_empty());
        let sum = kernels!(Fold rustymath::reduce::Sum);
        let (out, ok) = sum.run(&rustymath::reduce::Sum, bool::view(&[true, false]), [1, 2, 1], Some(i64::view(&[7]))).unwrap();
        assert!(ok && i64::of(out.view()).unwrap() == [8]);
        assert!(!sum.run(&rustymath::reduce::Sum, i64::view(&[i64::MAX]), [1, 1, 1], Some(i64::view(&[1]))).unwrap().1);
    }

    /// An inner product taken in order, item by item.
    fn folded<T: Copy, O: Copy>(a: &[T], b: &[T], [m, k, n]: [usize; 3], unit: O, step: impl Fn(O, T, T) -> O) -> Vec<O> {
        (0..m * n).map(|ij| (0..k).fold(unit, |acc, p| step(acc, a[ij / n * k + p], b[p * n + ij % n]))).collect()
    }

    /// The number of steps where the items differ, for each result.
    fn differing<T: Copy + PartialEq>(a: &[T], b: &[T], [m, k, n]: [usize; 3]) -> Vec<i64> {
        (0..m * n).map(|ij| (0..k).filter(|&p| a[ij / n * k + p] != b[p * n + ij % n]).count() as i64).collect()
    }

    #[test]
    fn transposed_operands_read_as_their_transposes() {
        // Float `PlusTimes` reads a transposed operand in place, and `MaxPlus` takes the default, which copies it. `inner_mapped` uses an operand where its map reads it: through a transposing map, through padded rows, or in a transposed orientation. A map that reads backwards is read in blocks. On rustymath's own kernels, a product with a vector takes both of `by_vector`'s routes: rows dotted four at a time with a remainder, and stored rows added a block at a time into a whole strip of results and a tail. Small whole numbers keep every sum exact.
        fn check<P: Inner<f64, Out = f64>>(p: P, a: &[f64], b: &[f64], [m, k, n]: [usize; 3]) {
            use rustymath::{agreement::Mapping, gather::Steps};
            let (at, bt, expected) = (transpose(a, m, k), transpose(b, k, n), p.inner(a, b, [m, k, n]));
            assert_eq!(p.inner_ops(&at, b, [Orientation::Transpose, Orientation::Normal], [m, k, n]), expected);
            assert_eq!(p.inner_ops(a, &bt, [Orientation::Normal, Orientation::Transpose], [m, k, n]), expected);
            assert_eq!(p.inner_ops(&at, &bt, [Orientation::Transpose, Orientation::Transpose], [m, k, n]), expected);
            let read = |[(s0, l0, d0), (s1, l1, d1)]: [(usize, usize, isize); 2]| Mapping::Steps { base: 0, steps: vec![Steps::Stride { start: s0, len: l0, step: d0 }, Steps::Stride { start: s1, len: l1, step: d1 }].into() };
            let (held, (mi, ki, ni)) = (Mapping::Linear(1), (m as isize, k as isize, n as isize));
            let padded: Vec<f64> = b.chunks(n).flat_map(|row| row.iter().copied().chain([0.0; 2])).collect();
            let reversed: Vec<f64> = a.iter().rev().copied().collect();
            let (from_at, padded_rows) = (read([(0, m, 1), (0, k, mi)]), read([(0, k, ni + 2), (0, n, 1)]));
            let (backwards, stored_t) = (read([((m - 1) * k, m, -ki), (k - 1, k, -1)]), read([(0, k, 1), (0, m, ki)]));
            let normal = [Orientation::Normal; 2];
            assert_eq!(p.inner_mapped(Mapped { data: &at, mapping: &from_at }, Mapped { data: &padded, mapping: &padded_rows }, normal, [m, k, n]), expected);
            assert_eq!(p.inner_mapped(Mapped { data: &reversed, mapping: &backwards }, Mapped { data: b, mapping: &held }, normal, [m, k, n]), expected);
            assert_eq!(p.inner_mapped(Mapped { data: a, mapping: &stored_t }, Mapped { data: &bt, mapping: &held }, [Orientation::Transpose; 2], [m, k, n]), expected);
        }
        for [m, k, n] in [[5, 7, 9], [1, 11, 37], [37, 11, 1]] {
            let a: Vec<f64> = (0..m * k).map(|i| (i * 7 % 17) as f64 - 8.0).collect();
            let b: Vec<f64> = (0..k * n).map(|i| (i * 5 % 13) as f64 - 6.0).collect();
            check(PlusTimes, &a, &b, [m, k, n]);
            check(MaxPlus, &a, &b, [m, k, n]);
        }
    }
    #[test]
    fn vector_products_read_windows_by_columns() {
        // `+.×` of windows and a vector reads each column of the windows where it lies, when the windows' rows are adjacent in long runs: the windows of a vector, of a single-channel image whose first row of windows reads fills, and long windows transposed. Each matches the product of the gathered windows. Small whole numbers keep every sum exact.
        use rustymath::{agreement::{Mapping, expanded_items}, gather::Steps};
        let along = |start, len, step| Steps::Stride { start, len, step };
        let check = |data: &[f64], steps: Vec<Steps>, o: Orientation, [m, k]: [usize; 2]| {
            let mapping = Mapping::Steps { base: 0, steps: steps.into() };
            let mut buf = Vec::new();
            let stored = expanded_items(&mapping, data, 0, m * k, &mut buf).to_vec();
            let a = if o == Orientation::Normal { stored } else { transpose(&stored, k, m) };
            let b: Vec<f64> = (0..k).map(|i| (i * 5 % 13) as f64 - 6.0).collect();
            let got = PlusTimes.inner_mapped(Mapped { data, mapping: &mapping }, Mapped { data: &b, mapping: &Mapping::Linear(1) }, [o, Orientation::Normal], [m, k, 1]);
            assert_eq!(got, PlusTimes.inner(&a, &b, [m, k, 1]));
        };
        let x: Vec<f64> = (0..300).map(|i| (i * 7 % 17) as f64 - 8.0).collect();
        check(&x, vec![along(0, 296, 1), along(0, 5, 1)], Orientation::Normal, [296, 5]);
        check(&x, vec![Steps::clipped(-1, 3, 4, 75), along(0, 73, 1), along(0, 3, 75), along(0, 3, 1)], Orientation::Normal, [219, 9]);
        check(&x, vec![along(0, 231, 1), along(0, 70, 1)], Orientation::Transpose, [70, 231]);
    }
    #[test]
    fn products_match_folds_in_order() {
        // The sizes leave partial tiles, give no steps or a single one, and cross each block size. Small whole numbers keep every float sum exact, in any order.
        let sizes = [[0, 3, 4], [3, 0, 4], [3, 4, 0], [1, 1, 1], [5, 7, 9], [MC + 3, KC + 5, 37], [MC + 3, KC + 5, 1], [17, 3, NC + 5]];
        for size @ [m, k, n] in sizes {
            let a64: Vec<f64> = (0..m * k).map(|i| (i * 7 % 17) as f64 - 8.0).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i * 5 % 13) as f64 - 6.0).collect();
            let (a32, b32): (Vec<f32>, Vec<f32>) = (a64.iter().map(|&x| x as f32).collect(), b64.iter().map(|&x| x as f32).collect());
            let expected = folded(&a64, &b64, size, 0.0, |s, x, y| s + x * y);
            let mut into = vec![1.0; m * n];
            PlusTimes.inner_into(&a64, &b64, size, &mut into);
            assert_eq!(into, expected);
            assert_eq!(PlusTimes.inner(&a64, &b64, size), expected);
            assert_eq!(PlusTimes.inner(&a32, &b32, size), folded(&a32, &b32, size, 0.0, |s, x, y| s + x * y));
            assert_eq!(MaxPlus.inner(&a64, &b64, size), folded(&a64, &b64, size, f64::NEG_INFINITY, |s, x, y| s.max(x + y)));
            assert_eq!(MaxPlus.inner(&a32, &b32, size), folded(&a32, &b32, size, f32::NEG_INFINITY, |s, x, y| s.max(x + y)));
            assert_eq!(MinPlus.inner(&a64, &b64, size), folded(&a64, &b64, size, f64::INFINITY, |s, x, y| s.min(x + y)));
            assert_eq!(MinPlus.inner(&a32, &b32, size), folded(&a32, &b32, size, f32::INFINITY, |s, x, y| s.min(x + y)));
            let (ab, bb): (Vec<bool>, Vec<bool>) = (a64.iter().map(|&x| x > 4.0).collect(), b64.iter().map(|&x| x > 4.0).collect());
            assert_eq!(OrAnd.inner(&ab, &bb, size), folded(&ab, &bb, size, false, |s, x, y| s | (x & y)));
            assert_eq!(AndOr.inner(&ab, &bb, size), folded(&ab, &bb, size, true, |s, x, y| s & (x | y)));
            let (ai, bi): (Vec<i64>, Vec<i64>) = (a64.iter().map(|&x| x as i64).collect(), b64.iter().map(|&x| x as i64).collect());
            // Items too large for 32 bits take `+.=`'s 64-bit kernel. Items that fit 32 bits but no window of 2^16 values take its 32-bit kernel. Small items, and items in a narrow window far from 0, take its 16-bit kernel.
            let (wa, wb): (Vec<i64>, Vec<i64>) = (ai.iter().map(|&x| x << 40).collect(), bi.iter().map(|&x| x << 40).collect());
            let (ma, mb): (Vec<i64>, Vec<i64>) = (ai.iter().map(|&x| x << 20).collect(), bi.iter().map(|&x| x << 20).collect());
            let (fa, fb): (Vec<i64>, Vec<i64>) = (ai.iter().map(|&x| x - (1 << 50)).collect(), bi.iter().map(|&x| x - (1 << 50)).collect());
            let (a3, b3): (Vec<i32>, Vec<i32>) = (ai.iter().map(|&x| x as i32).collect(), bi.iter().map(|&x| x as i32).collect());
            assert_eq!(PlusEqual.inner(&ai, &bi, size), folded(&ai, &bi, size, 0i64, |s, x, y| s + i64::from(x == y)));
            assert_eq!(PlusEqual.inner(&wa, &wb, size), folded(&wa, &wb, size, 0i64, |s, x, y| s + i64::from(x == y)));
            assert_eq!(PlusEqual.inner(&ma, &mb, size), folded(&ma, &mb, size, 0i64, |s, x, y| s + i64::from(x == y)));
            assert_eq!(PlusEqual.inner(&fa, &fb, size), folded(&fa, &fb, size, 0i64, |s, x, y| s + i64::from(x == y)));
            assert_eq!(PlusEqual.inner(&a3, &b3, size), folded(&a3, &b3, size, 0i32, |s, x, y| s + i32::from(x == y)));
            assert_eq!(PlusEqual.inner(&a64, &b64, size), folded(&a64, &b64, size, 0i64, |s, x, y| s + i64::from(x == y)));
            assert_eq!(PlusEqual.inner(&a32, &b32, size), folded(&a32, &b32, size, 0i32, |s, x, y| s + i32::from(x == y)));
            let wide = |counts: Vec<i32>| counts.into_iter().map(i64::from).collect::<Vec<_>>();
            assert_eq!(PlusNotEqual.inner(&ai, &bi, size), differing(&ai, &bi, size));
            assert_eq!(PlusNotEqual.inner(&wa, &wb, size), differing(&wa, &wb, size));
            assert_eq!(wide(PlusNotEqual.inner(&a3, &b3, size)), differing(&a3, &b3, size));
            assert_eq!(PlusNotEqual.inner(&a64, &b64, size), differing(&a64, &b64, size));
            assert_eq!(wide(PlusNotEqual.inner(&a32, &b32, size)), differing(&a32, &b32, size));
        }
    }

    /// `p`'s `checked_into` as the results it writes, or `None` where it gives `false`.
    fn checked<P: Inner<i64, Out = i64>>(p: &P, a: &[i64], b: &[i64], size @ [m, _, n]: [usize; 3]) -> Option<Vec<i64>> {
        let mut out = vec![0; m * n];
        p.checked_into(a, b, size, &mut out).then_some(out)
    }

    #[test]
    fn integer_products_match_exact_folds() {
        // Each pair of scales for `a` and `b` reaches one path. `+.×` runs at f32, f64, i64, then in 128 bits, overflowing. `⌈.+` and `⌊.+` run at f32 twice, then f64, then in 128 bits, overflowing. Odd scales give products that a narrower path would round, where a power of two would keep them exact. `checked_into` gives `false` only where a result doesn't fit `i64`, and `inner` gives the same results elsewhere.
        let scales = [(1, 1), (4097, 4093), ((1 << 22) + 1, (1 << 22) + 3), ((1 << 59) + 1, (1 << 60) + 1)];
        for (sa, sb) in scales {
            for size @ [m, k, n] in [[0, 3, 4], [3, 0, 4], [3, 4, 0], [5, 7, 9], [MC + 3, KC + 5, 37]] {
                let a: Vec<i64> = (0..m * k).map(|i| ((i * 7 % 17) as i64 - 8) * sa).collect();
                let b: Vec<i64> = (0..k * n).map(|i| ((i * 5 % 13) as i64 - 6) * sb).collect();
                let overflows = sa > 1 << 59 && !a.is_empty() && !b.is_empty();
                let products = (!overflows).then(|| folded(&a, &b, size, 0i64, |s, x, y| s + x * y));
                assert_eq!(checked(&PlusTimes, &a, &b, size), products);
                if let Some(products) = products { assert_eq!(PlusTimes.inner(&a, &b, size), products); }
                let exact = |unit: i64, step: fn(i128, i128) -> i128| {
                    folded(&a, &b, size, i128::from(unit), |s, x, y| step(s, i128::from(x) + i128::from(y))).into_iter().map(|t| i64::try_from(t).ok()).collect::<Option<Vec<_>>>()
                };
                let maxima = exact(i64::MIN, i128::max);
                assert_eq!(checked(&MaxPlus, &a, &b, size), maxima);
                if let Some(maxima) = maxima { assert_eq!(MaxPlus.inner(&a, &b, size), maxima); }
                let minima = exact(i64::MAX, i128::min);
                assert_eq!(checked(&MinPlus, &a, &b, size), minima);
                if let Some(minima) = minima { assert_eq!(MinPlus.inner(&a, &b, size), minima); }
            }
        }
        let ([a, b], size) = ([1i64 << 40, -(1 << 40)], [1, 2, 1]);
        assert_eq!(checked(&PlusTimes, &[a, b], &[1 << 30, 1 << 30], size), Some(vec![0]), "products beyond i64 whose sum fits");
    }

    #[test]
    fn wrapped_and_half_products_match_folds() {
        // `Wrapped(PlusTimes)` folds with wrapping `i32` steps whatever its items' type, whether they compute as exact floats or on the `i32` kernel: `u8`, `i16` and `i32` items give the folds of their `i32` values, and the large `i16` and `i32` items overflow 32 bits. Small and moderate `i32` items take the exact float route, and the moderate ones still wrap at 32 bits. f16 `+.×` gives the f32 folds of the same values, which small whole numbers keep exact.
        use rustymath::Wrapped;
        fn widened<T: Copy + Into<i32>>(x: &[T]) -> Vec<i32> { x.iter().map(|&v| v.into()).collect() }
        for size @ [m, k, n] in [[0, 3, 4], [3, 0, 4], [3, 4, 0], [5, 7, 9], [MC + 3, KC + 5, 37]] {
            let wrapped = |a: &[i32], b: &[i32]| folded(a, b, size, 0i32, |s, x, y| s.wrapping_add(x.wrapping_mul(y)));
            let a8: Vec<u8> = (0..m * k).map(|i| (i * 37 % 256) as u8).collect();
            let b8: Vec<u8> = (0..k * n).map(|i| (i * 91 % 256) as u8).collect();
            let a16: Vec<i16> = (0..m * k).map(|i| (i * 4099 % 65536) as u16 as i16).collect();
            let b16: Vec<i16> = (0..k * n).map(|i| (i * 8191 % 65536) as u16 as i16).collect();
            let a32: Vec<i32> = (0..m * k).map(|i| (i as i32).wrapping_mul(-1_640_531_527)).collect();
            let b32: Vec<i32> = (0..k * n).map(|i| (i as i32).wrapping_mul(1_013_904_223)).collect();
            assert_eq!(Wrapped(PlusTimes).inner(&a8, &b8, size), wrapped(&widened(&a8), &widened(&b8)));
            assert_eq!(Wrapped(PlusTimes).inner(&a16, &b16, size), wrapped(&widened(&a16), &widened(&b16)));
            assert_eq!(Wrapped(PlusTimes).inner(&a32, &b32, size), wrapped(&a32, &b32));
            let small = |len: usize| (0..len).map(|i| (i * 37 % 127) as i32 - 63).collect::<Vec<i32>>();
            let moderate = |len: usize| (0..len).map(|i| (i * 7919 % 2_000_001) as i32 - 1_000_000).collect::<Vec<i32>>();
            assert_eq!(Wrapped(PlusTimes).inner(&small(m * k), &small(k * n), size), wrapped(&small(m * k), &small(k * n)));
            assert_eq!(Wrapped(PlusTimes).inner(&moderate(m * k), &moderate(k * n), size), wrapped(&moderate(m * k), &moderate(k * n)));
            let ah: Vec<half::f16> = (0..m * k).map(|i| half::f16::from_f32((i * 7 % 17) as f32 - 8.0)).collect();
            let bh: Vec<half::f16> = (0..k * n).map(|i| half::f16::from_f32((i * 5 % 13) as f32 - 6.0)).collect();
            let single = |x: &[half::f16]| x.iter().map(|&v| f32::from(v)).collect::<Vec<f32>>();
            assert_eq!(PlusTimes.inner(&ah, &bh, size), folded(&single(&ah), &single(&bh), size, 0.0, |s, x, y| s + x * y));
        }
        // A sum of more than 2^21 `i16` products can pass the 53 bits that `f64` holds exactly, so it runs on the `i32` kernel.
        let size = [1, (1 << 21) + 1, 1];
        let a: Vec<i16> = (0..size[1]).map(|i| (i * 4099 % 65536) as u16 as i16).collect();
        assert_eq!(Wrapped(PlusTimes).inner(&a, &a, size), folded(&widened(&a), &widened(&a), size, 0i32, |s, x, y| s.wrapping_add(x.wrapping_mul(y))));
    }

    #[test]
    fn finite_complex_products_match_four_products() {
        // Every product of parts is exact for these values. Native complex products must then give the same bits as `*`. x(1+i) squared, with x² above a quarter of the largest float, and max(1+i) times 1 have finite results, though sums of their parts overflow.
        macro_rules! check {
            ($r:ty, $x:expr) => {{
                let (c, x, max) = (Complex::<$r>::new, $x, <$r>::MAX);
                for (p, q) in [(c(x, x), c(x, x)), (c(max, max), c(1.0, 0.0)), (c(1.0, 2.0), c(3.0, 4.0))] {
                    assert_eq!(PlusTimes.inner(&[p], &[q], [1, 1, 1]), [p * q]);
                    let mut into = [c(9.0, 9.0)];
                    PlusTimes.inner_into(&[p], &[q], [1, 1, 1], &mut into);
                    assert_eq!(into, [p * q]);
                }
                // Complex-step differentiation reads an imaginary part far smaller than its real part. A real matrix times one with such parts must keep them.
                let tiny = (2.0 as $r).powi(-100);
                let (a, b) = ([c(1.0, 0.0), c(2.0, 0.0)], [c(3.0, tiny), c(5.0, tiny)]);
                assert_eq!(PlusTimes.inner(&a, &b, [1, 2, 1]), [c(13.0, 3.0 * tiny)]);
            }};
        }
        check!(f64, 2f64.powi(511));
        check!(f32, 2f32.powi(63));
    }

    #[test]
    fn nan_terms_stay_in_their_rows() {
        // Rows 0 and 3 each have a NaN term, in the first step and in a later one. Each of their results is NaN or the fold of their other terms. The other rows give their folds exactly.
        let size @ [m, k, n] = [9, 300, 11];
        let mut a: Vec<f64> = (0..m * k).map(|i| (i % 23) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 19) as f64).collect();
        (a[0], a[3 * k + 280]) = (f64::NAN, f64::NAN);
        let fold = |i: usize, j: usize, max: bool| (0..k).map(|p| a[i * k + p] + b[p * n + j]).filter(|t| !t.is_nan()).fold(if max { f64::NEG_INFINITY } else { f64::INFINITY }, |s, t| if max { s.max(t) } else { s.min(t) });
        let nan_row = |i: usize| (0..k).any(|p| a[i * k + p].is_nan());
        let check = |c: &[f64], max: bool| for i in 0..m { for j in 0..n { let (got, want) = (c[i * n + j], fold(i, j, max)); assert!(got == want || (nan_row(i) && got.is_nan()), "row {i}: {got}, where {want}") } };
        check(&MaxPlus.inner(&a, &b, size), true);
        check(&MinPlus.inner(&a, &b, size), false);
        let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        for (c, max) in [(MaxPlus.inner(&a32, &b32, size), true), (MinPlus.inner(&a32, &b32, size), false)] { check(&c.iter().map(|&x| x as f64).collect::<Vec<_>>(), max) }
    }
}
mod factor {
    use super::*;
    use rand::RngExt;

    /// A real item type of the tests: `f32` or `f64`, whose real quantities, such as singular values, have its own type.
    pub(super) trait RealItem: Factor + Native + Scalar<Real = Self> + std::fmt::Debug {}
    impl<T: Factor + Native + Scalar<Real = T> + std::fmt::Debug> RealItem for T {}

    /// An `m`×`n` matrix of numbers from -1 to 1.
    pub(super) fn random<T: RealItem>(m: usize, n: usize, seed: u64) -> Vec<T> {
        let mut rng = support::seeded(seed);
        (0..m * n).map(|_| T::narrow(2.0 * rng.random::<f64>() - 1.0)).collect()
    }

    pub(super) fn wide<T: RealItem>(x: &[T]) -> Vec<f64> { x.iter().map(|v| v.wide()).collect() }

    /// The product of the `m`×`k` matrix `a` and the `k`×`n` matrix `b`, in f64, item by item.
    pub(super) fn product(a: &[f64], b: &[f64], [m, k, n]: [usize; 3]) -> Vec<f64> {
        (0..m * n).map(|ij| (0..k).map(|p| a[ij / n * k + p] * b[p * n + ij % n]).sum()).collect()
    }

    pub(super) fn norm(x: &[f64]) -> f64 { x.iter().map(|v| v * v).sum::<f64>().sqrt() }

    /// The norm of `x - y`, relative to `scale`.
    pub(super) fn distance(x: &[f64], y: &[f64], scale: f64) -> f64 { norm(&x.iter().zip(y).map(|(a, b)| a - b).collect::<Vec<_>>()) / scale }

    /// Whether `x` differs from `y` by less than `bound` times `y`'s norm, as `gap` measures it.
    pub(super) fn near(x: &[f64], y: &[f64], bound: f64) -> bool { gap(x, y) < bound }

    /// How far `x` is from `y`, relative to `y`'s norm. It divides both by `y`'s largest magnitude first, keeping huge items' squares finite.
    pub(super) fn gap(x: &[f64], y: &[f64]) -> f64 {
        let largest = y.iter().fold(f64::MIN_POSITIVE, |m, v| m.max(v.abs()));
        let unit = |v: &[f64]| v.iter().map(|w| w / largest).collect::<Vec<f64>>();
        distance(&unit(x), &unit(y), norm(&unit(y)).max(f64::MIN_POSITIVE))
    }

    /// How far the `rows`×`cols` matrix `q` is from having orthonormal columns.
    pub(super) fn orthogonality(q: &[f64], rows: usize, cols: usize) -> f64 {
        let qt = transpose(q, rows, cols);
        distance(&product(&qt, q, [cols, rows, cols]), &identity(cols), 1.0)
    }

    /// Checks that `u`, `s` and `v` form an SVD of the `m`×`c` matrix `a`: a residual below `bound` relative to `a`'s norm, and U and V orthonormal to within four times `bound`.
    fn check_svd(a: &[f64], [m, c]: [usize; 2], u: &[f64], s: &[f64], v: &[f64], bound: f64, what: &str) {
        let p = s.len();
        let us: Vec<f64> = (0..m * p).map(|k| u[k] * s[k % p]).collect();
        assert!(distance(&product(&us, &transpose(v, c, p), [m, p, c]), a, norm(a).max(1.0)) < bound, "svd {what}");
        // The accumulated rotations' rounding errors add up over the sweeps, in V, or in U for a wide matrix.
        let (ou, ov) = (orthogonality(u, m, p), orthogonality(v, c, p));
        assert!(ou < 4.0 * bound && ov < 4.0 * bound, "svd's u and v {what}: {ou:e} and {ov:e}, against {bound:e}");
    }

    /// Checks each decomposition at `T` on matrices either side of the block width, where `tolerance` is a few times `T`'s epsilon.
    fn decompositions<T: RealItem>(tolerance: f64) {
        for n in [0, 1, 2, 5, 31, 32, 33, 71] {
            let bound = tolerance * (n as f64).max(1.0);
            let a = random::<T>(n, n, n as u64 + 1);
            let aw = wide(&a);
            // L U is A with its rows permuted.
            let f = lu(&a, n);
            let (l, u) = (wide(&f.lower()), wide(&f.upper()));
            let pa: Vec<f64> = f.perm.iter().flat_map(|&p| aw[p * n..(p + 1) * n].to_vec()).collect();
            assert!(distance(&product(&l, &u, [n; 3]), &pa, norm(&aw).max(1.0)) < bound, "lu at {n}");
            // Solutions have small residuals for their size.
            for k in [1, 3] {
                let b = random::<T>(n, k, 99);
                let x = wide(&f.solve(&b, k));
                assert!(distance(&product(&aw, &x, [n, n, k]), &wide(&b), (norm(&aw) * norm(&x)).max(1.0)) < bound, "solve at {n}, {k}");
            }
            let inverse = wide(&f.inverse());
            assert!(distance(&product(&aw, &inverse, [n; 3]), &identity(n), (norm(&aw) * norm(&inverse)).max(1.0)) < bound, "inverse at {n}");
            // Cholesky of a positive definite matrix, and none of one that isn't.
            let spd: Vec<T> = product(&aw, &transpose(&aw, n, n), [n; 3]).iter().enumerate()
                .map(|(k, &v)| T::narrow(v + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
            let l = wide(&cholesky(&spd, n).expect("a positive definite matrix"));
            assert!((0..n * n).all(|k| k % n <= k / n || l[k] == 0.0), "cholesky's upper triangle at {n}");
            assert!(distance(&product(&l, &transpose(&l, n, n), [n; 3]), &wide(&spd), norm(&wide(&spd)).max(1.0)) < bound, "cholesky at {n}");
            if n > 0 {
                assert_eq!(cholesky(&spd.iter().map(|&v| -v).collect::<Vec<_>>(), n), None, "cholesky of a negative definite matrix")
            }
            // QR and the SVD, of square, tall and wide matrices.
            for (m, c) in [(n, n), (n + 5, n), (n, n + 5)] {
                let a = random::<T>(m, c, (m * 31 + c) as u64);
                let (aw, p, scale) = (wide(&a), m.min(c), norm(&wide(&a)).max(1.0));
                let (q, r) = qr(&a, [m, c]);
                let (q, r) = (wide(&q), wide(&r));
                assert!(distance(&product(&q, &r, [m, p, c]), &aw, scale) < bound, "qr at {m}×{c}");
                assert!(orthogonality(&q, m, p) < bound, "qr's q at {m}×{c}");
                assert!((0..p * c).all(|k| k % c >= k / c || r[k] == 0.0), "qr's r at {m}×{c}");
                let f = svd(&a, [m, c]);
                let s = wide(&f.s);
                assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0), "svd's s at {m}×{c}");
                check_svd(&aw, [m, c], &wide(&f.u), &s, &wide(&f.v), bound, &format!("at {m}×{c}"));
                assert!(distance(&wide(&singular_values(&a, [m, c])), &wide(&f.s), scale) < bound, "singular values alone at {m}×{c}");
                // A least-squares solution leaves a residual orthogonal to A's columns, and the pseudoinverse inverts A on its range.
                if m >= c && c > 0 {
                    let b = random::<T>(m, 2, 7);
                    let x = wide(&f.solve(&b, 2));
                    let residual: Vec<f64> = product(&aw, &x, [m, c, 2]).iter().zip(wide(&b)).map(|(ax, b)| ax - b).collect();
                    let normal = product(&transpose(&aw, m, c), &residual, [c, m, 2]);
                    assert!(norm(&normal) / (scale * (scale * norm(&x) + norm(&wide(&b)))) < bound, "svd's solve at {m}×{c}");
                }
                // `lstsq` leaves its residual orthogonal to A's columns too, and gives the SVD's singular values.
                if m > 0 && c > 0 {
                    let b = random::<T>(m, 2, 11);
                    let (x, s2) = lstsq(&a, &b, [m, c], 2);
                    let x = wide(&x);
                    let residual: Vec<f64> = product(&aw, &x, [m, c, 2]).iter().zip(wide(&b)).map(|(ax, b)| ax - b).collect();
                    let normal = product(&transpose(&aw, m, c), &residual, [c, m, 2]);
                    assert!(norm(&normal) / (scale * (scale * norm(&x) + norm(&wide(&b)))) < bound, "lstsq at {m}×{c}");
                    assert!(distance(&wide(&s2), &s, s[0]) < bound, "lstsq's singular values at {m}×{c}");
                }
                let pinv = wide(&f.pseudoinverse());
                assert!(distance(&product(&product(&aw, &pinv, [m, c, m]), &aw, [m, m, c]), &aw, scale) < bound, "pseudoinverse at {m}×{c}");
            }
        }
        // A zero column gives a singular value within rounding of zero, and U stays orthonormal.
        let n = 9;
        let a: Vec<T> = random::<T>(n, n, 5).iter().enumerate().map(|(k, &v)| if k % n == 3 { T::zero() } else { v }).collect();
        let f = svd(&a, [n, n]);
        assert!(f.s[n - 1].wide() <= tolerance * f.s[0].wide(), "a zero column's singular value");
        assert!(orthogonality(&wide(&f.u), n, n) < tolerance * n as f64, "u with a zero column");
        // A bidiagonal matrix with a zero on its diagonal reaches the QR algorithm unchanged, which rotates that row's superdiagonal item away before it iterates.
        let (d, e) = ([3.0, 0.0, 2.0, 0.5, 1.0], [1.0, 1.0, 0.25, 2.0]);
        let b: Vec<T> = (0..25).map(|k| {
            T::narrow(match (k / 5, k % 5) { (i, j) if i == j => d[i], (i, j) if j == i + 1 => e[i], _ => 0.0 })
        }).collect();
        let f = svd(&b, [5, 5]);
        let us: Vec<f64> = (0..25).map(|k| wide(&f.u)[k] * wide(&f.s)[k % 5]).collect();
        assert!(
            distance(&product(&us, &transpose(&wide(&f.v), 5, 5), [5; 3]), &wide(&b), norm(&wide(&b))) < 8.0 * tolerance, "svd of a bidiagonal with a zero"
        );
        assert!(orthogonality(&wide(&f.u), 5, 5) < 8.0 * tolerance, "u of a bidiagonal with a zero");
    }

    #[test]
    fn real_decompositions() {
        decompositions::<f64>(4.0 * f64::EPSILON);
        decompositions::<f32>(4.0 * f32::EPSILON as f64);
        hard_matrices::<f64>(4.0 * f64::EPSILON);
        hard_matrices::<f32>(4.0 * f32::EPSILON as f64);
        solves_and_determinants::<f64>(4.0 * f64::EPSILON);
        solves_and_determinants::<f32>(4.0 * f32::EPSILON as f64);
    }

    /// Checks the decompositions at `T` on LAPACK's hard matrix types, where `tolerance` is a few times `T`'s epsilon. A matrix scaled by a power of two near underflow or overflow gives LU, Cholesky and QR factors scaled exactly in step, and singular values and a least-squares solution that match to rounding. Its singular vectors pass their own residual and orthogonality checks. A zero column gives finite LU factors with a zero pivot. A subnormal pivot gives finite multipliers.
    fn hard_matrices<T: RealItem>(tolerance: f64) {
        let n = 40;
        let tiny = T::min_positive_value() / T::epsilon();
        let (a, b) = (random::<T>(n, n, 17), random::<T>(n, 3, 19));
        let aw = wide(&a);
        let spd: Vec<T> = product(&aw, &transpose(&aw, n, n), [n; 3]).iter().enumerate()
            .map(|(k, &v)| T::narrow(v + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
        // Fourth powers of items scaled by `quartic` overflow, though sums of their squares don't, so `safe_scale` leaves them alone.
        let quartic = T::narrow(2f64.powi((T::max_value().log2().wide() / 4.0) as i32 + 6));
        // On macOS the SVD and least squares go through LAPACK, whose own scaling matches a scaled copy's results only to rounding.
        let drift = 64.0 * tolerance * n as f64;
        for s in [tiny, tiny.recip(), quartic] {
            let times = |x: &[T], c: T| x.iter().map(|&v| v * c).collect::<Vec<T>>();
            let (f, g) = (lu(&a, n), lu(&times(&a, s), n));
            assert_eq!(g.perm, f.perm, "lu's permutation scaled by {s:?}");
            assert_eq!(g.factors, f.factors.iter().enumerate().map(|(k, &v)| if k % n <= k / n { v * s } else { v }).collect::<Vec<T>>(), "lu scaled by {s:?}");
            assert_eq!(g.solve(&times(&b, s), 3), f.solve(&b, 3), "lu's solve scaled by {s:?}");
            assert_eq!(times(&g.inverse(), s), f.inverse(), "lu's inverse scaled by {s:?}");
            // An even power of two has an exact square root.
            let c = if s.sqrt() * s.sqrt() == s { s } else { s + s };
            assert_eq!(cholesky(&times(&spd, c), n), cholesky(&spd, n).map(|l| times(&l, c.sqrt())), "cholesky scaled by {c:?}");
            for (rows, cols) in [(n, n), (n + 5, n), (n, n + 5)] {
                let a = random::<T>(rows, cols, (rows * 31 + cols) as u64);
                let b = random::<T>(rows, 2, 23);
                let ((q, r), (qs, rs)) = (qr(&a, [rows, cols]), qr(&times(&a, s), [rows, cols]));
                assert_eq!((qs, rs), (q, times(&r, s)), "qr scaled by {s:?} at {rows}×{cols}");
                let (f, g) = (svd(&a, [rows, cols]), svd(&times(&a, s), [rows, cols]));
                assert!(near(&wide(&g.s), &wide(&times(&f.s, s)), drift), "svd's singular values scaled by {s:?} at {rows}×{cols}");
                let unscaled: Vec<f64> = wide(&g.s).iter().map(|v| v / s.wide()).collect();
                check_svd(&wide(&a), [rows, cols], &wide(&g.u), &unscaled, &wide(&g.v), drift, &format!("scaled by {s:?} at {rows}×{cols}"));
                let ((x, sv), (xs, svs)) = (lstsq(&a, &b, [rows, cols], 2), lstsq(&times(&a, s), &times(&b, s), [rows, cols], 2));
                assert!(near(&wide(&xs), &wide(&x), drift) && near(&wide(&svs), &wide(&times(&sv, s)), drift), "lstsq scaled by {s:?} at {rows}×{cols}");
            }
        }
        let n = 71;
        for zero in [0, n / 2, n - 1] {
            let a: Vec<T> = random::<T>(n, n, 29).iter().enumerate().map(|(k, &v)| if k % n == zero { T::zero() } else { v }).collect();
            let (aw, f) = (wide(&a), lu(&a, n));
            let factors = wide(&f.factors);
            assert!(factors.iter().all(|v| v.is_finite()) && f.diagonal().any(|d| d == T::zero()), "lu with zero column {zero}");
            let (l, u) = (wide(&f.lower()), wide(&f.upper()));
            let pa: Vec<f64> = f.perm.iter().flat_map(|&p| aw[p * n..(p + 1) * n].to_vec()).collect();
            assert!(distance(&product(&l, &u, [n; 3]), &pa, norm(&aw)) < tolerance * n as f64, "lu's residual with zero column {zero}");
        }
        let (p, q) = (T::min_positive_value() / T::narrow(1024.0), T::min_positive_value() / T::narrow(1048576.0));
        assert_eq!(lu(&[p, T::one(), q, T::one()], 2).factors, [p, q / p, T::one(), T::one() - q / p], "lu with a subnormal pivot");
    }

    /// Checks `solve`, `pinv`, `qr_complete`, the determinants and `hermitian_eigenvalues` at `T`, where `tolerance` is a few times `T`'s epsilon. The singular matrices repeat columns, so their rank is exact.
    fn solves_and_determinants<T: RealItem>(tolerance: f64) {
        let bound = 64.0 * tolerance;
        let base = random::<T>(9, 4, 7);
        // Columns 4 to 6 repeat columns 0 to 2, so each matrix has rank 4: tall, square, wide, and a regular square one.
        let repeat = |rows: usize| -> Vec<T> { (0..rows * 7).map(|k| base[k / 7 * 4 + k % 7 % 4]).collect() };
        let regular = random::<T>(6, 6, 11);
        for (a, [m, n], r) in [(repeat(9), [9, 7], 4), (repeat(7), [7, 7], 4), (transpose(&repeat(9), 9, 7), [7, 9], 4), (regular.clone(), [6, 6], 6)] {
            let (aw, scale) = (wide(&a), norm(&wide(&a)));
            assert_eq!(rank(&a, [m, n]), r, "rank at {m}×{n}");
            // The pseudoinverse meets the Moore-Penrose conditions A A⁺ A = A and A⁺ A A⁺ = A⁺.
            let p = wide(&pinv(&a, [m, n]));
            let ap = product(&aw, &p, [m, n, m]);
            assert!(distance(&product(&ap, &aw, [m, m, n]), &aw, scale) < bound, "A A⁺ A at {m}×{n}");
            assert!(distance(&product(&p, &ap, [n, m, m]), &p, norm(&p)) < bound, "A⁺ A A⁺ at {m}×{n}");
            // `solve` gives the least-norm least-squares solution, A⁺ b.
            let b = random::<T>(m, 2, 13);
            let x = wide(&solve(&a, &b, [m, n], 2));
            let expected = product(&p, &wide(&b), [n, m, 2]);
            assert!(distance(&x, &expected, norm(&expected)) < bound, "solve at {m}×{n}");
            // The complete QR: Q is square and orthogonal, R's rows below the thin ones are zero, and Q R is A.
            let (q, r) = qr_complete(&a, [m, n]);
            let (q, r, thin) = (wide(&q), wide(&r), m.min(n));
            assert!(orthogonality(&q, m, m) < bound, "complete Q at {m}×{n}");
            assert!(r[thin * n..].iter().all(|&v| v == 0.0), "complete R's lower rows at {m}×{n}");
            assert!(distance(&product(&q, &r, [m, m, n]), &aw, scale) < bound, "complete Q R at {m}×{n}");
        }
        // A regular matrix's determinant times its inverse's is 1, and the log-determinant agrees with it.
        let det = |a: &[T], n: usize| lu(a, n).determinant().wide();
        let inverse = pinv(&regular, [6, 6]);
        assert!((det(&regular, 6) * det(&inverse, 6) - 1.0).abs() < bound, "determinants of a matrix and its inverse");
        let mut x = vec![T::narrow(0.0); 36];
        let d = LuWork::default().inverse_into(&regular, [6, 6], &mut x).expect("a regular matrix has an inverse");
        assert!(x == inverse && d.wide() == det(&regular, 6), "a regular matrix's inverse, which pinv gives, and its determinant, which lu gives");
        assert!(LuWork::default().inverse_into(&[1., 2., 2., 4.].map(T::narrow), [2, 2], &mut [T::narrow(0.0); 4]).is_none(), "a singular matrix has no inverse");
        let (sign, log) = lu(&regular, 6).log_determinant();
        assert!((sign.wide() * log.wide().exp() / det(&regular, 6) - 1.0).abs() < bound, "log-determinant");
        // diag(1, 2, 3, 4) with two rows swapped has determinant -24, and with a zero column, 0 and a logarithm of -∞.
        let swapped: Vec<T> = [0., 2., 0., 0., 1., 0., 0., 0., 0., 0., 3., 0., 0., 0., 0., 4.].map(T::narrow).to_vec();
        assert_eq!(det(&swapped, 4), -24.0, "an odd permutation's sign");
        let singular: Vec<T> = [1., 0., 3., 0.].map(T::narrow).to_vec();
        let (sign, log) = lu(&singular, 2).log_determinant();
        assert_eq!((det(&singular, 2), sign.wide(), log.wide()), (0.0, 0.0, f64::NEG_INFINITY), "a singular matrix");
        // Eigenvalues alone match `eigh`'s, either side of the size where `eigh` switches to divide and conquer.
        for n in [6, 70] {
            let a = random::<T>(n, n, n as u64);
            let s: Vec<T> = (0..n * n).map(|k| a[k] + a[k % n * n + k / n]).collect();
            let (values, expected) = (wide(&hermitian_eigenvalues(&s, n).unwrap()), wide(&eigh(&s, n).unwrap().values));
            assert!(distance(&values, &expected, norm(&expected)) < bound, "eigenvalues alone at {n}");
        }
    }

    /// Checks LU, Cholesky and QR at `T` on matrices large enough for the wider blocks, each leaving a partial last block, where `tolerance` is a few times `T`'s epsilon. Products with random vectors keep each check to O(n²) time.
    fn blocked<T: RealItem>(tolerance: f64) {
        let times = |a: &[f64], v: &[f64]| product(a, v, [a.len() / v.len(), v.len(), 1]);
        for n in [530, 1030] {
            let bound = tolerance * n as f64;
            let a = random::<T>(n, n, n as u64);
            let (aw, x) = (wide(&a), wide(&random::<T>(n, 1, 3)));
            let f = lu(&a, n);
            let (l, u) = (wide(&f.lower()), wide(&f.upper()));
            let ax = times(&aw, &x);
            let pax: Vec<f64> = f.perm.iter().map(|&p| ax[p]).collect();
            assert!(distance(&times(&l, &times(&u, &x)), &pax, norm(&aw) * norm(&x)) < bound, "lu at {n}");
            let b = random::<T>(n, 3, 5);
            let solved = wide(&f.solve(&b, 3));
            assert!(distance(&product(&aw, &solved, [n, n, 3]), &wide(&b), norm(&aw) * norm(&solved)) < bound, "solve at {n}");
            let inverse = wide(&f.inverse());
            assert!(distance(&times(&aw, &times(&inverse, &x)), &x, norm(&aw) * norm(&inverse) * norm(&x)) < bound, "inverse at {n}");
            // (A + Aᵀ)/2 + n I is diagonally dominant with a positive diagonal, so it's positive definite.
            let spd: Vec<T> = (0..n * n).map(|k| T::narrow((aw[k] + aw[k % n * n + k / n]) / 2.0 + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
            let (sw, l) = (wide(&spd), wide(&cholesky(&spd, n).expect("a positive definite matrix")));
            assert!((0..n * n).all(|k| k % n <= k / n || l[k] == 0.0), "cholesky's upper triangle at {n}");
            assert!(distance(&times(&l, &times(&transpose(&l, n, n), &x)), &times(&sw, &x), norm(&sw) * norm(&x)) < bound, "cholesky at {n}");
            for rows in [n, n + 5] {
                let a = random::<T>(rows, n, rows as u64 + 7);
                let (aw, (q, r)) = (wide(&a), qr(&a, [rows, n]));
                let (q, r) = (wide(&q), wide(&r));
                assert!(distance(&times(&q, &times(&r, &x)), &times(&aw, &x), norm(&aw) * norm(&x)) < bound, "qr at {rows}×{n}");
                assert!((0..n * n).all(|k| k % n >= k / n || r[k] == 0.0), "qr's r at {rows}×{n}");
                let (y, qt) = (wide(&random::<T>(n, 1, 9)), transpose(&q, rows, n));
                assert!(distance(&times(&qt, &times(&q, &y)), &y, norm(&y)) < bound, "qr's q at {rows}×{n}");
            }
        }
        // Least squares applies Qᴴ to both few and many right-hand sides, without exposing the stored reflections.
        let n = 71;
        let rows = n + 5;
        let a = random::<T>(rows, n, rows as u64 + 7);
        let (q, r) = qr(&a, [rows, n]);
        let (qt, r) = (transpose(&wide(&q), rows, n), wide(&r));
        for k in [2, FEW_COLUMNS + 1] {
            let b = random::<T>(rows, k, 13);
            let (x, _) = lstsq(&a, &b, [rows, n], k);
            let bw = wide(&b);
            assert!(
                distance(&product(&r, &wide(&x), [n, n, k]), &product(&qt, &bw, [n, rows, k]), norm(&bw)) < tolerance * n as f64,
                "lstsq's Qᴴ b at {rows}×{n}×{k}"
            );
        }
    }

    #[test]
    fn blocked_decompositions_have_small_residuals() {
        blocked::<f64>(4.0 * f64::EPSILON);
        blocked::<f32>(4.0 * f32::EPSILON as f64);
    }

    /// The `m`×`n` matrix U diag(s) Vᵀ, with orthonormal U and V from fixed seeds and the singular values `s`.
    pub(super) fn with_spectrum(m: usize, n: usize, s: &[f64]) -> Vec<f64> {
        let (u, v) = (qr(&random::<f64>(m, n, 3), [m, n]).0, qr(&random::<f64>(n, n, 5), [n, n]).0);
        let us: Vec<f64> = (0..m * n).map(|k| u[k] * s[k % n]).collect();
        PlusTimes.inner(&us, &transpose(&v, n, n), [m, n, n])
    }

    /// QR, least squares and the SVD of tall matrices, with condition numbers from 1 to `worst`, which Cholesky QR factors, and `beyond`, which falls back to Householder QR. Each matrix's singular values run from 1 down to 1/condition. Its first item is positive, so R's first item is positive on the Cholesky path and negative on Householder's.
    fn tall<T: RealItem>(tolerance: f64, worst: f64, beyond: f64) {
        let tiny = T::min_positive_value() / T::epsilon();
        let quartic = T::narrow(2f64.powi((T::max_value().log2().wide() / 4.0) as i32 + 6));
        let times = |x: &[T], c: T| x.iter().map(|&v| v * c).collect::<Vec<T>>();
        for (m, n) in [(4000, 20), (8000, 60)] {
            let bound = tolerance * n as f64;
            for condition in [1.0, worst.sqrt(), worst, beyond] {
                let s: Vec<f64> = (0..n).map(|j| condition.powf(-(j as f64) / (n - 1) as f64)).collect();
                let mut aw = with_spectrum(m, n, &s);
                if aw[0] < 0.0 { aw[..n].iter_mut().for_each(|x| *x = -*x) }
                let a: Vec<T> = aw.iter().map(|&x| T::narrow(x)).collect();
                let (aw, at) = (wide(&a), format!("{m}×{n}, condition {condition:e}"));
                let (q, r) = qr(&a, [m, n]);
                let (qw, rw) = (wide(&q), wide(&r));
                assert_eq!(rw[0] > 0.0, condition < beyond, "qr's path at {at}");
                assert!(
                    distance(&product(&qw, &rw, [m, n, n]), &aw, norm(&aw)) < bound, "tall qr at {at}: {:e} of {bound:e}",
                    distance(&product(&qw, &rw, [m, n, n]), &aw, norm(&aw))
                );
                assert!(orthogonality(&qw, m, n) < bound, "tall qr's q at {at}: {:e} of {bound:e}", orthogonality(&qw, m, n));
                assert!((0..n * n).all(|k| k % n >= k / n || rw[k] == 0.0), "tall qr's r at {at}");
                // `lstsq` leaves its residual orthogonal to A's columns, and gives A's singular values.
                let b = random::<T>(m, 2, 11);
                let (x, sv) = lstsq(&a, &b, [m, n], 2);
                let xw = wide(&x);
                let residual: Vec<f64> = product(&aw, &xw, [m, n, 2]).iter().zip(wide(&b)).map(|(ax, b)| ax - b).collect();
                let normal = product(&transpose(&aw, m, n), &residual, [n, m, 2]);
                let scale = norm(&aw);
                assert!(norm(&normal) / (scale * (scale * norm(&xw) + norm(&wide(&b)))) < bound, "tall lstsq at {at}");
                let svd_s = wide(&singular_values(&a, [m, n]));
                assert!(distance(&wide(&sv), &svd_s, svd_s[0]) < bound, "tall lstsq's singular values at {at}");
                // The SVD reduces to R by `qr`'s path, and gives the singular values A was built with.
                let f = svd(&a, [m, n]);
                let (uw, sw, vw) = (wide(&f.u), wide(&f.s), wide(&f.v));
                let usw: Vec<f64> = (0..m * n).map(|k| uw[k] * sw[k % n]).collect();
                assert!(distance(&product(&usw, &transpose(&vw, n, n), [m, n, n]), &aw, norm(&aw)) < bound, "tall svd at {at}");
                assert!(orthogonality(&uw, m, n) < 4.0 * bound, "tall svd's u at {at}");
                assert!(distance(&sw, &s, 1.0) < bound, "tall svd's singular values at {at}");
            }
            // Scaling by a power of two scales R exactly, and the singular values to rounding. Unlike R of a matrix with orthonormal columns, a random matrix's R has no items small enough to lose bits as subnormals when scaled down.
            let (a, b) = (random::<T>(m, n, 7), random::<T>(m, 2, 13));
            let ((q, r), (x, sv)) = (qr(&a, [m, n]), lstsq(&a, &b, [m, n], 2));
            for c in [tiny, tiny.recip(), quartic] {
                assert_eq!(qr(&times(&a, c), [m, n]), (q.clone(), times(&r, c)), "tall qr scaled by {c:?} at {m}×{n}");
                let (xs, svs) = lstsq(&times(&a, c), &times(&b, c), [m, n], 2);
                assert!(near(&wide(&xs), &wide(&x), bound) && near(&wide(&svs), &wide(&times(&sv, c)), bound), "tall lstsq scaled by {c:?} at {m}×{n}");
            }
        }
    }

    #[test]
    fn tall_matrices() {
        tall::<f64>(4.0 * f64::EPSILON, 1e7, 1e12);
        tall::<f32>(4.0 * f32::EPSILON as f64, 1e2, 1e12);
    }

    /// A tall matrix 520 columns wide takes the Cholesky path, with the same accuracy as the narrow ones in `tall_matrices`.
    #[test]
    fn tall_wide_matrices() {
        let (m, n) = (8320, 520);
        let bound = 4.0 * f64::EPSILON * n as f64;
        let s: Vec<f64> = (0..n).map(|j| 1e3f64.powf(-(j as f64) / (n - 1) as f64)).collect();
        let mut a = with_spectrum(m, n, &s);
        if a[0] < 0.0 { a[..n].iter_mut().for_each(|x| *x = -*x) }
        let (q, r) = qr(&a, [m, n]);
        assert!(r[0] > 0.0, "wide tall qr takes the Cholesky path");
        assert!(distance(&PlusTimes.inner(&q, &r, [m, n, n]), &a, norm(&a)) < bound, "wide tall qr");
        assert!(distance(&PlusTimes.inner(&transpose(&q, m, n), &q, [n, m, n]), &identity::<f64>(n), 1.0) < bound, "wide tall qr's q");
        let b = random::<f64>(m, 2, 11);
        let (x, sv) = lstsq(&a, &b, [m, n], 2);
        let residual: Vec<f64> = product(&a, &x, [m, n, 2]).iter().zip(&b).map(|(ax, b)| ax - b).collect();
        let scale = norm(&a);
        assert!(norm(&product(&transpose(&a, m, n), &residual, [n, m, 2])) / (scale * (scale * norm(&x) + norm(&b))) < bound, "wide tall lstsq");
        assert!(distance(&sv, &s, 1.0) < bound, "wide tall lstsq's singular values");
    }

    /// The truncated SVD of matrices whose singular values decay fast and slowly: its values at or just below the ones they estimate, its vectors orthonormal, and its error within 10% of the best rank-`k` approximation's.
    fn truncated<T: RealItem>(tolerance: f64) {
        let (m, n, k) = (2000, 300, 10);
        let bound = tolerance * n as f64;
        let fast: Vec<f64> = (0..n).map(|j| 0.5f64.powi(j as i32)).collect();
        let slow: Vec<f64> = (0..n).map(|j| 1.0 / (1 + j) as f64).collect();
        for (decay, s) in [("fast", fast), ("slow", slow)] {
            let tall: Vec<T> = with_spectrum(m, n, &s).iter().map(|&x| T::narrow(x)).collect();
            // The wide case works on a view of the conjugate transpose.
            for (a, [m, n]) in [(transpose(&tall, m, n), [n, m]), (tall, [m, n])] {
                let f = truncated_svd(&a, [m, n], k);
                assert_eq!((f.u.len(), f.s.len(), f.v.len()), (m * k, k, n * k), "truncated svd's sizes, {decay} decay, {m}×{n}");
                let (uw, sw, vw) = (wide(&f.u), wide(&f.s), wide(&f.v));
                assert!(orthogonality(&uw, m, k) < bound && orthogonality(&vw, n, k) < bound, "truncated svd's vectors, {decay} decay, {m}×{n}");
                assert!(sw.iter().zip(&s).all(|(&e, &t)| e <= t * (1.0 + bound) && e >= 0.9 * t), "truncated svd's values, {decay} decay, {m}×{n}");
                let usw: Vec<f64> = (0..m * k).map(|i| uw[i] * sw[i % k]).collect();
                let error = distance(&product(&usw, &transpose(&vw, n, k), [m, k, n]), &wide(&a), 1.0);
                assert!(error <= 1.1 * norm(&s[k..]) + bound, "truncated svd's error, {decay} decay, {m}×{n}");
            }
        }
        // A test matrix as wide as the smaller side gives the thin SVD's leading values, for a tall matrix and for its transpose.
        let small = random::<T>(40, 30, 19);
        let leading = svd(&small, [40, 30]).s[..25].to_vec();
        assert_eq!(truncated_svd(&small, [40, 30], 25).s, leading, "truncated svd of a small tall matrix");
        assert_eq!(truncated_svd(&transpose(&small, 40, 30), [30, 40], 25).s, leading, "truncated svd of a small wide matrix");
        // A matrix holding a NaN or an infinity gives NaN for every result.
        for bad in [T::nan(), T::infinity()] {
            let mut a = random::<T>(100, 60, 23);
            a[61] = bad;
            let f = truncated_svd(&a, [100, 60], 5);
            assert!(f.u.iter().chain(&f.s).chain(&f.v).all(|x| x.is_nan()), "truncated svd of a matrix holding {bad:?}");
        }
    }

    #[test]
    fn truncated_svds() {
        truncated::<f64>(4.0 * f64::EPSILON);
        truncated::<f32>(4.0 * f32::EPSILON as f64);
    }

    type C64 = num_complex::Complex<f64>;

    /// An `m`×`n` matrix of complex numbers whose parts lie between -1 and 1.
    pub(super) fn complex_random<R: RealItem>(m: usize, n: usize, seed: u64) -> Vec<num_complex::Complex<R>> {
        random::<R>(m, n, seed).into_iter().zip(random::<R>(m, n, seed + 1000)).map(|(re, im)| num_complex::Complex::new(re, im)).collect()
    }

    pub(super) fn complex_wide<R: RealItem>(x: &[num_complex::Complex<R>]) -> Vec<C64> { x.iter().map(|z| C64::new(z.re.wide(), z.im.wide())).collect() }

    /// The product of the `m`×`k` matrix `a` and the `k`×`n` matrix `b`, item by item.
    fn complex_product(a: &[C64], b: &[C64], [m, k, n]: [usize; 3]) -> Vec<C64> {
        (0..m * n).map(|ij| (0..k).map(|p| a[ij / n * k + p] * b[p * n + ij % n]).sum()).collect()
    }

    fn complex_norm(x: &[C64]) -> f64 { x.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt() }

    /// The norm of `x - y`, relative to `scale`.
    fn complex_distance(x: &[C64], y: &[C64], scale: f64) -> f64 { x.iter().zip(y).map(|(a, b)| (a - b).norm_sqr()).sum::<f64>().sqrt() / scale }

    /// Checks LU with its solves and inverse, Cholesky and QR at complex items with parts of type `R`, on the matrices `decompositions` takes, where `tolerance` is a few times `R`'s epsilon.
    fn complex_decompositions<R: RealItem>(tolerance: f64) where Complex<R>: Factor + Scalar<Real = R> {
        let zero = C64::new(0.0, 0.0);
        for n in [0, 1, 2, 5, 31, 32, 33, 71] {
            let bound = tolerance * (n as f64).max(1.0);
            let a = complex_random::<R>(n, n, n as u64 + 1);
            let (aw, i) = (complex_wide(&a), identity::<C64>(n));
            let scale = complex_norm(&aw).max(1.0);
            // L U is A with its rows permuted.
            let f = lu(&a, n);
            let (l, u) = (complex_wide(&f.lower()), complex_wide(&f.upper()));
            let pa: Vec<C64> = f.perm.iter().flat_map(|&p| aw[p * n..(p + 1) * n].to_vec()).collect();
            assert!(complex_distance(&complex_product(&l, &u, [n; 3]), &pa, scale) < bound, "complex lu at {n}");
            for k in [1, 3] {
                let b = complex_random::<R>(n, k, 99);
                let x = complex_wide(&f.solve(&b, k));
                assert!(
                    complex_distance(&complex_product(&aw, &x, [n, n, k]), &complex_wide(&b), (scale * complex_norm(&x)).max(1.0)) < bound,
                    "complex solve at {n}, {k}"
                );
            }
            let inverse = complex_wide(&f.inverse());
            assert!(complex_distance(&complex_product(&aw, &inverse, [n; 3]), &i, (scale * complex_norm(&inverse)).max(1.0)) < bound, "complex inverse at {n}");
            // Cholesky of a Hermitian positive definite matrix gives L with a real, positive diagonal. A negative definite one has none.
            let hermitian: Vec<C64> = complex_product(&aw, &adjoint(&aw, n, n), [n; 3]).iter().zip(&i).map(|(&v, &d)| v + d * n as f64).collect();
            let hpd: Vec<num_complex::Complex<R>> = hermitian.iter().map(|v| num_complex::Complex::new(R::narrow(v.re), R::narrow(v.im))).collect();
            let hw = complex_wide(&hpd);
            let l = complex_wide(&cholesky(&hpd, n).expect("a Hermitian positive definite matrix"));
            assert!((0..n * n).all(|k| k % n <= k / n || l[k] == zero), "complex cholesky's upper triangle at {n}");
            assert!((0..n).all(|j| l[j * (n + 1)].im == 0.0 && l[j * (n + 1)].re > 0.0), "complex cholesky's diagonal at {n}");
            assert!(complex_distance(&complex_product(&l, &adjoint(&l, n, n), [n; 3]), &hw, complex_norm(&hw).max(1.0)) < bound, "complex cholesky at {n}");
            if n > 0 {
                assert!(cholesky(&hpd.iter().map(|&v| -v).collect::<Vec<_>>(), n).is_none(), "complex cholesky of a negative definite matrix")
            }
            // QR of square, tall and wide matrices gives orthonormal columns of Q, and R with a real, non-negative diagonal.
            for (m, c) in [(n, n), (n + 5, n), (n, n + 5)] {
                let a = complex_random::<R>(m, c, (m * 31 + c) as u64);
                let (aw, p) = (complex_wide(&a), m.min(c));
                let (q, r) = qr(&a, [m, c]);
                let (q, r) = (complex_wide(&q), complex_wide(&r));
                assert!(complex_distance(&complex_product(&q, &r, [m, p, c]), &aw, complex_norm(&aw).max(1.0)) < bound, "complex qr at {m}×{c}");
                assert!(complex_distance(&complex_product(&adjoint(&q, m, p), &q, [p, m, p]), &identity::<C64>(p), 1.0) < bound, "complex qr's q at {m}×{c}");
                assert!((0..p * c).all(|k| k % c >= k / c || r[k] == zero), "complex qr's r at {m}×{c}");
                assert!((0..p).all(|j| r[j * c + j].im == 0.0), "complex qr's diagonal at {m}×{c}");
            }
        }
    }

    #[test]
    fn complex_factorizations() {
        complex_decompositions::<f64>(4.0 * f64::EPSILON);
        complex_decompositions::<f32>(4.0 * f32::EPSILON as f64);
        complex_svd_and_eigen::<f64>(4.0 * f64::EPSILON);
        complex_svd_and_eigen::<f32>(4.0 * f32::EPSILON as f64);
        complex_tall::<f64>(4.0 * f64::EPSILON);
        complex_tall::<f32>(4.0 * f32::EPSILON as f64);
    }

    /// QR and least squares of a tall, well-conditioned complex matrix, which Cholesky QR factors, giving R a real, positive diagonal. The matrix's first item has a positive real part, which Householder QR would turn into a negative item of R.
    fn complex_tall<R: RealItem>(tolerance: f64) where Complex<R>: Factor + Scalar<Real = R> {
        let (m, n) = (2000, 24);
        let bound = tolerance * n as f64;
        let mut a = complex_random::<R>(m, n, 41);
        if a[0].re < R::zero() { a[..n].iter_mut().for_each(|z| *z = -*z) }
        let (q, r) = qr(&a, [m, n]);
        let (aw, q, r) = (complex_wide(&a), complex_wide(&q), complex_wide(&r));
        assert!(complex_distance(&complex_product(&q, &r, [m, n, n]), &aw, complex_norm(&aw)) < bound, "tall complex qr");
        assert!(complex_distance(&complex_product(&adjoint(&q, m, n), &q, [n, m, n]), &identity::<C64>(n), 1.0) < bound, "tall complex qr's q");
        assert!((0..n).all(|j| r[j * n + j].im == 0.0 && r[j * n + j].re > 0.0), "tall complex qr's diagonal");
        let b = complex_random::<R>(m, 2, 43);
        let x = complex_wide(&lstsq(&a, &b, [m, n], 2).0);
        let residual: Vec<C64> = complex_product(&aw, &x, [m, n, 2]).iter().zip(complex_wide(&b)).map(|(ax, b)| ax - b).collect();
        let normal = complex_product(&adjoint(&aw, m, n), &residual, [n, m, 2]);
        let scale = complex_norm(&aw);
        assert!(complex_norm(&normal) / (scale * (scale * complex_norm(&x) + complex_norm(&complex_wide(&b)))) < bound, "tall complex lstsq");
    }

    /// Checks the SVD and the Hermitian eigen decomposition at complex items with parts of type `R`, either side of each block width, where `tolerance` is a few times `R`'s epsilon.
    fn complex_svd_and_eigen<R: RealItem>(tolerance: f64) where Complex<R>: Factor + Scalar<Real = R> {
        type C<R> = num_complex::Complex<R>;
        for n in [0, 1, 2, 5, 31, 64, 129] {
            let bound = tolerance * (n as f64).max(1.0);
            // Square, tall, very tall and wide matrices: U Σ Vᴴ is A, U and V have orthonormal columns, and the singular values descend.
            for (m, c) in [(n, n), (n + 5, n), (2 * n + 5, n), (n, n + 5)] {
                let a = complex_random::<R>(m, c, (m * 37 + c) as u64);
                let (aw, p) = (complex_wide(&a), m.min(c));
                let Svd { u, s, v, .. } = svd(&a, [m, c]);
                assert!(
                    distance(&wide(&singular_values(&a, [m, c])), &wide(&s), complex_norm(&aw).max(1.0)) < bound, "complex singular values alone at {m}×{c}"
                );
                let (u, v, s) = (complex_wide(&u), complex_wide(&v), wide(&s));
                let us: Vec<C64> = (0..m * p).map(|k| u[k] * s[k % p]).collect();
                assert!(
                    complex_distance(&complex_product(&us, &adjoint(&v, c, p), [m, p, c]), &aw, complex_norm(&aw).max(1.0)) < bound, "complex svd at {m}×{c}"
                );
                assert!(complex_distance(&complex_product(&adjoint(&u, m, p), &u, [p, m, p]), &identity::<C64>(p), 1.0) < bound, "complex svd's u at {m}×{c}");
                assert!(complex_distance(&complex_product(&adjoint(&v, c, p), &v, [p, c, p]), &identity::<C64>(p), 1.0) < bound, "complex svd's v at {m}×{c}");
                assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0), "complex singular values descend at {m}×{c}");
            }
            // A Hermitian matrix, read from its lower triangle: H V is V Λ, V is unitary, and the eigenvalues ascend.
            let b = complex_random::<R>(n, n, n as u64 + 7);
            let half = R::narrow(0.5);
            let h: Vec<C<R>> = (0..n * n).map(|k| (b[k] + b[k % n * n + k / n].conj()).mul_real(half)).collect();
            let junk = C::new(R::narrow(9.0), R::narrow(-9.0));
            let lower: Vec<C<R>> = (0..n * n).map(|k| if k % n > k / n { junk } else { h[k] }).collect();
            let Eigh { values, vectors } = eigh(&lower, n).expect("a Hermitian matrix converges");
            let (hw, vectors, values) = (complex_wide(&h), complex_wide(&vectors), wide(&values));
            assert!(values.windows(2).all(|w| w[0] <= w[1]), "complex eigenvalues ascend at {n}");
            let vl: Vec<C64> = (0..n * n).map(|k| vectors[k] * values[k % n]).collect();
            assert!(complex_distance(&complex_product(&hw, &vectors, [n; 3]), &vl, complex_norm(&hw).max(1.0)) < bound, "complex hermitian eigen at {n}");
            assert!(
                complex_distance(&complex_product(&adjoint(&vectors, n, n), &vectors, [n; 3]), &identity::<C64>(n), 1.0) < bound, "unitary eigenvectors at {n}"
            );
            // A complex matrix with real items gives the real code's singular values and eigenvalues, within rounding.
            let re: Vec<R> = h.iter().map(|z| z.re).collect();
            let real: Vec<C<R>> = re.iter().map(|&x| C::new(x, R::zero())).collect();
            let close = |x: &[f64], y: &[f64]| x.iter().zip(y).all(|(a, b)| (a - b).abs() <= bound * x.iter().fold(1.0, |m: f64, v| m.max(v.abs())));
            assert!(close(&wide(&singular_values(&real, [n, n])), &wide(&singular_values(&re, [n, n]))), "real singular values at {n}");
            assert!(close(&wide(&eigh(&real, n).unwrap().values), &wide(&eigh(&re, n).unwrap().values)), "real eigenvalues at {n}");
            if n > 1 {
                // A NaN gives NaN for every singular value and no eigen decomposition.
                let mut nan = h.clone();
                nan[n] = C::new(R::nan(), R::zero());
                assert!(svd(&nan, [n, n]).s.iter().all(|x| x.is_nan()), "complex svd of a matrix holding NaN at {n}");
                assert!(eigh(&nan, n).is_none(), "complex hermitian eigen of a matrix holding NaN at {n}");
            }
        }
        // A wide complex matrix of rank 5 takes the truncated SVD's randomized path on a view of its conjugate transpose. U Σ Vᴴ recovers it to rounding.
        let (m, c, k) = (40, 300, 5);
        let low = complex_product(&complex_wide(&complex_random::<R>(m, k, 51)), &complex_wide(&complex_random::<R>(k, c, 53)), [m, k, c]);
        let a: Vec<C<R>> = low.iter().map(|z| C::new(R::narrow(z.re), R::narrow(z.im))).collect();
        let (aw, f) = (complex_wide(&a), truncated_svd(&a, [m, c], k));
        let (u, v, s) = (complex_wide(&f.u), complex_wide(&f.v), wide(&f.s));
        let us: Vec<C64> = (0..m * k).map(|i| u[i] * s[i % k]).collect();
        let bound = tolerance * c as f64;
        assert!(
            complex_distance(&complex_product(&us, &adjoint(&v, c, k), [m, k, c]), &aw, complex_norm(&aw)) < bound,
            "truncated svd of a wide complex matrix of rank {k}"
        );
        // Both parts of `z` exceed half the largest float, so `abs1(z)` overflows, though `z`'s modulus and reciprocal fit.
        let z = C::new(R::max_value() * R::narrow(0.6), R::max_value() * R::narrow(0.6));
        let (zero, one, size) = (C::new(R::zero(), R::zero()), C::new(R::one(), R::zero()), z.modulus().wide());
        let s = wide(&singular_values(&[z, zero, zero, z], [2, 2]));
        assert!(s.iter().all(|&x| (x / size - 1.0).abs() < 16.0 * tolerance), "singular values of items whose abs1 overflows");
        assert_eq!(z.quotient(z), one, "z ÷ z where abs1(z) overflows");
        let r = z.reciprocal();
        assert!(r.im == -r.re && (r.re.wide() * 1.2 * R::max_value().wide() - 1.0).abs() < 64.0 * tolerance, "z's subnormal reciprocal");
    }
}

mod eigen {
    use super::factor::{RealItem, distance, near, norm, orthogonality, product, random, wide};
    use super::*;

    /// The symmetric `n`×`n` matrix whose lower triangle comes from `lower`, with `junk` in its upper triangle, which `eigh` never reads.
    fn symmetric<T: RealItem>(lower: &[T], n: usize, junk: T) -> Vec<T> { (0..n * n).map(|k| if k % n > k / n { junk } else { lower[k] }).collect() }

    /// The symmetric tridiagonal `n`×`n` matrix with diagonal `d` and subdiagonal `e`, in full.
    fn tridiagonal_matrix(d: &[f64], e: &[f64]) -> Vec<f64> {
        let n = d.len();
        (0..n * n).map(|k| match (k / n, k % n) { (i, j) if i == j => d[i], (i, j) if i == j + 1 || j == i + 1 => e[i.min(j)], _ => 0.0 }).collect()
    }

    /// Checks that `f` decomposes the symmetric `n`×`n` matrix `a`: ascending values, orthonormal vectors, and a small residual `A V - V Λ`, all within `bound`.
    fn check(a: &[f64], n: usize, f: &Eigh<f64>, bound: f64, what: &str) {
        assert!(f.values.windows(2).all(|w| w[0] <= w[1]), "values ascend for {what}");
        let (v, scale) = (&f.vectors, norm(a).max(f64::MIN_POSITIVE));
        let vl: Vec<f64> = (0..n * n).map(|k| v[k] * f.values[k % n]).collect();
        assert!(distance(&product(a, v, [n; 3]), &vl, scale) < bound, "residual for {what}");
        assert!(orthogonality(v, n, n) < bound, "orthogonal vectors for {what}");
    }

    /// Checks `eigh` at `T` on random symmetric matrices, where `tolerance` is a few times `T`'s epsilon.
    fn random_matrices<T: RealItem>(tolerance: f64) {
        for n in [0, 1, 2, 5, 31, 64, 129] {
            let lower = random::<T>(n, n, n as u64 + 3);
            let a = symmetric(&lower, n, T::zero());
            let full: Vec<T> = (0..n * n).map(|k| lower[if k % n > k / n { k % n * n + k / n } else { k }]).collect();
            let f = eigh(&a, n).expect("a symmetric matrix converges");
            assert_eq!(eigh(&symmetric(&lower, n, T::narrow(1e30)), n).as_ref(), Some(&f), "eigh reads only the lower triangle at {n}");
            let g = Eigh { values: wide(&f.values), vectors: wide(&f.vectors) };
            check(&wide(&full), n, &g, tolerance * (n as f64).max(1.0), &format!("{n}×{n}"));
        }
    }

    #[test]
    fn symmetric_eigen() {
        random_matrices::<f64>(4.0 * f64::EPSILON);
        random_matrices::<f32>(4.0 * f32::EPSILON as f64);
        hard_matrices::<f64>();
        hard_matrices::<f32>();
    }

    #[test]
    fn symmetric_eigen_survives_hard_spectra() {
        let bound = 64.0 * f64::EPSILON;
        // Repeated eigenvalues: the identity, a zero matrix and a diagonal matrix with ties.
        for (a, n, values) in [
            (identity::<f64>(4), 4, vec![1.0; 4]), (vec![0.0; 9], 3, vec![0.0; 3]),
            (tridiagonal_matrix(&[2.0, 1.0, 2.0, 1.0], &[0.0; 3]), 4, vec![1.0, 1.0, 2.0, 2.0]),
        ] {
            let f = eigh(&a, n).unwrap();
            assert_eq!(f.values, values, "eigenvalues of a matrix with ties");
            check(&a, n, &f, bound, "a matrix with ties");
        }
        // Wilkinson's matrix W7⁺, whose largest eigenvalues come in close pairs, and two copies of it joined weakly.
        let wilkinson = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let glued_d: Vec<f64> = wilkinson.iter().chain(&wilkinson).copied().collect();
        let glued_e: Vec<f64> = (0..13).map(|i| if i == 6 { 1e-6 } else { 1.0 }).collect();
        for (d, e) in [(wilkinson.to_vec(), vec![1.0; 6]), (glued_d, glued_e)] {
            let (a, n) = (tridiagonal_matrix(&d, &e), d.len());
            check(&a, n, &eigh(&a, n).unwrap(), bound, "a Wilkinson matrix");
        }
        // A 2×2 block whose subdiagonal is barely above the negligible size.
        let a = tridiagonal_matrix(&[1.0000000000000002, 1.0000000000000002], &[7.216449660063518e-16]);
        check(&a, 2, &eigh(&a, 2).unwrap(), bound, "a nearly diagonal 2×2 matrix");
        // In a matrix of rank 4, Householder reduction leaves only rounding noise below the leading rows. The iterations must count that noise as negligible against the matrix's norm. The 50-row case runs QL. The 300-row case runs divide and conquer.
        for n in [50, 300] {
            let p: Vec<f64> = (0..n * n).map(|k| (k as f64).sin()).collect();
            let a: Vec<f64> = (0..n * n).map(|k| p[k] + p[k % n * n + k / n]).collect();
            check(&a, n, &eigh(&a, n).expect("a matrix of rank 4 converges"), 64.0 * f64::EPSILON * n as f64, "a matrix of rank 4");
            assert!(hermitian_eigenvalues(&a, n).is_some(), "eigenvalues alone of a matrix of rank 4 at {n}");
        }
    }

    #[test]
    fn divide_and_conquer_survives_hard_spectra() {
        // Hard spectra reach the public solver, at sizes above its divide-and-conquer cutoff.
        let wilkinson = [3.0, 2.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let glued_d: Vec<f64> = wilkinson.iter().chain(&wilkinson).copied().collect();
        let glued_e: Vec<f64> = (0..13).map(|i| if i == 6 { 1e-6 } else { 1.0 }).collect();
        let random_d: Vec<f64> = random::<f64>(1, 100, 7);
        let random_e: Vec<f64> = random::<f64>(1, 99, 8);
        let clustered_d: Vec<f64> = (0..60).map(|i| if i < 30 { 1.0 } else { 2.0 }).collect();
        let clustered_e: Vec<f64> = (0..59).map(|i| if i % 7 == 0 { 0.5 } else { 1e-18 }).collect();
        let cases = [
            (wilkinson.to_vec(), vec![1.0; 6], "a Wilkinson matrix"), (glued_d, glued_e, "two glued Wilkinson matrices"),
            (random_d, random_e, "a random tridiagonal matrix"), (clustered_d, clustered_e, "a matrix with two clusters of eigenvalues"),
            (vec![1.0; 50], vec![0.0; 49], "the identity"), ((0..40).map(|i| (i % 2 + 1) as f64).collect(), vec![0.0; 39], "a diagonal matrix with ties"),
        ];
        for (d, e, what) in cases {
            let copies = (EIGH_CONQUER + 1).div_ceil(d.len());
            let diagonal: Vec<f64> = (0..copies).flat_map(|_| d.iter().copied()).collect();
            let join = if e.iter().all(|&v| v == 0.0) { 0.0 } else { 1e-6 };
            let sub: Vec<f64> = (0..diagonal.len() - 1).map(|i| if i % d.len() == d.len() - 1 { join } else { e[i % d.len()] }).collect();
            let (a, n) = (tridiagonal_matrix(&diagonal, &sub), diagonal.len());
            let bound = 64.0 * f64::EPSILON * n as f64;
            check(&a, n, &eigh(&a, n).expect("a tridiagonal matrix converges"), bound, what);
        }
    }

    /// Checks that `eigh` at `T` handles a matrix scaled by a power of two near underflow or overflow, and gives `None` for a matrix holding NaN. The scaled eigenvalues match the unscaled ones to rounding. The scaled eigenvectors, whose signs can change, pass their own residual and orthogonality checks.
    fn hard_matrices<T: RealItem>() {
        let n = 40;
        let lower = random::<T>(n, n, 41);
        let a = symmetric(&lower, n, T::zero());
        let full: Vec<f64> = (0..n * n).map(|k| lower[if k % n > k / n { k % n * n + k / n } else { k }].wide()).collect();
        let f = eigh(&a, n).unwrap();
        let tiny = T::min_positive_value() / T::epsilon();
        let drift = 64.0 * n as f64 * T::epsilon().wide();
        for s in [tiny, tiny.recip()] {
            let g = eigh(&a.iter().map(|&v| v * s).collect::<Vec<T>>(), n).unwrap();
            assert!(near(&wide(&g.values), &wide(&f.values.iter().map(|&v| v * s).collect::<Vec<T>>()), drift), "eigenvalues scaled by {s:?}");
            let values = wide(&g.values).iter().map(|v| v / s.wide()).collect();
            check(&full, n, &Eigh { values, vectors: wide(&g.vectors) }, drift, &format!("eigh scaled by {s:?}"));
        }
        let mut nan = a.clone();
        nan[3 * n + 1] = T::nan();
        assert!(eigh(&nan, n).is_none(), "a matrix holding NaN");
    }
}

mod general {
    use super::factor::{RealItem, complex_random, complex_wide, random, wide};
    use super::*;
    use num_traits::Float as _;
    use rustymath::Float as _;

    /// `a` with each item as a complex number.
    fn complex(a: &[f64]) -> Vec<Complex<f64>> { a.iter().map(|&x| Complex::new(x, 0.0)).collect() }

    /// `values` sorted by real part, then imaginary part.
    fn sorted(values: &[Complex<f64>]) -> Vec<Complex<f64>> {
        let mut v = values.to_vec();
        v.sort_by(|a, b| a.re.total_cmp(&b.re).then(a.im.total_cmp(&b.im)));
        v
    }

    /// Checks what `eig` promises for any `n`×`n` matrix `a`: a relative residual `‖A V - V Λ‖ / ‖A‖` below `bound`, and each eigenvector of unit length, to within `bound`.
    fn check_units(a: &[Complex<f64>], n: usize, f: &Eig<f64>, bound: f64, what: &str) {
        assert_eq!((f.values.len(), f.vectors.len()), (n, n * n), "sizes for {what}");
        let mut sum = 0.0;
        for j in 0..n {
            for i in 0..n {
                let av = (0..n).fold(Complex::new(0.0, 0.0), |s, p| s + a[i * n + p] * f.vectors[p * n + j]);
                sum += (av - f.values[j] * f.vectors[i * n + j]).norm_sqr();
            }
        }
        let residual = sum.sqrt() / a.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt().max(f64::MIN_POSITIVE);
        assert!(residual < bound, "residual {residual} for {what}");
        for j in 0..n {
            let column: Vec<Complex<f64>> = (0..n).map(|i| f.vectors[i * n + j]).collect();
            let length = column.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
            assert!((length - 1.0).abs() < bound, "vector {j} has unit length for {what}");
        }
    }

    /// Checks that `f` decomposes the real `n`×`n` matrix `a` as `Eig` promises, with a residual and errors in length below `bound`.
    fn check(a: &[f64], n: usize, f: &Eig<f64>, bound: f64, what: &str) {
        check_units(&complex(a), n, f, bound, what);
        let mut j = 0;
        while j < n {
            if f.values[j].im == 0.0 {
                assert!((0..n).all(|i| f.vectors[i * n + j].im == 0.0), "a real eigenvalue has a real vector for {what}");
                j += 1;
            } else {
                assert!(f.values[j].im > 0.0 && f.values[j + 1] == f.values[j].conj(), "a conjugate pair, positive first, for {what}");
                assert!((0..n).all(|i| f.vectors[i * n + j + 1] == f.vectors[i * n + j].conj()), "conjugate vectors for {what}");
                j += 2;
            }
        }
    }

    /// `z` at f64.
    fn widen<T: RealItem>(z: &Complex<T>) -> Complex<f64> { Complex::new(z.re.wide(), z.im.wide()) }

    /// `f` at f64, from a decomposition at `T`.
    fn widened<T: RealItem>(f: &Eig<T>) -> Eig<f64> { Eig { values: f.values.iter().map(widen).collect(), vectors: f.vectors.iter().map(widen).collect() } }

    /// Whether each value in `x` lies within `bound` times the largest magnitude of some value in `y`, and each value in `y` of some value in `x`. Their orders can differ.
    fn same_values(x: &[Complex<f64>], y: &[Complex<f64>], bound: f64) -> bool {
        let largest = x.iter().chain(y).fold(f64::MIN_POSITIVE, |m, z| m.max(z.norm()));
        let within = |p: &[Complex<f64>], q: &[Complex<f64>]| p.iter().all(|z| q.iter().any(|w| (z - w).norm() < bound * largest));
        x.len() == y.len() && within(x, y) && within(y, x)
    }

    /// Checks `eig` at `T` on random matrices, where `tolerance` is a few times `T`'s epsilon. On the Mac, 256 rows reach LAPACK's `?geev`.
    fn random_matrices<T: RealItem>(tolerance: f64) {
        for n in [0, 1, 2, 3, 5, 31, 64, 129, 256] {
            let a = random::<T>(n, n, n as u64 + 7);
            let (f, bound) = (widened(&eig(&a, n).expect("a random matrix converges")), tolerance * (n as f64).max(1.0));
            let alone: Vec<Complex<f64>> = eigenvalues(&a, n).expect("a random matrix converges").iter().map(widen).collect();
            assert!(same_values(&alone, &f.values, bound), "eigenvalues alone match eig's at {n}");
            check(&wide(&a), n, &f, bound, &format!("{n}×{n}"));
        }
    }

    /// Checks the complex general eigen problem at parts `R` on random matrices, where `tolerance` is a few times `R`'s epsilon.
    fn complex_matrices<R: RealItem>(tolerance: f64) where Complex<R>: Factor + Scalar<Real = R> {
        for n in [0, 1, 2, 3, 5, 31, 64, 129] {
            let a = complex_random::<R>(n, n, n as u64 + 11);
            let (f, bound) = (widened(&eig(&a, n).expect("a random complex matrix converges")), tolerance * (n as f64).max(1.0));
            let alone: Vec<Complex<f64>> = eigenvalues(&a, n).expect("a random complex matrix converges").iter().map(widen).collect();
            assert!(same_values(&alone, &f.values, bound), "eigenvalues alone match at complex {n}");
            check_units(&complex_wide(&a), n, &f, bound, &format!("complex {n}×{n}"));
        }
    }

    #[test]
    fn general_eigen() {
        random_matrices::<f64>(8.0 * f64::EPSILON);
        random_matrices::<f32>(8.0 * f32::EPSILON as f64);
        complex_matrices::<f64>(8.0 * f64::EPSILON);
        complex_matrices::<f32>(8.0 * f32::EPSILON as f64);
        let n = 30;
        scaled(&random::<f64>(n, n, 43).iter().map(|&v| v + v).collect::<Vec<_>>(), n);
        scaled(&random::<f32>(n, n, 43).iter().map(|&v| v + v).collect::<Vec<_>>(), n);
        scaled(&complex_random::<f64>(n, n, 43), n);
        scaled(&complex_random::<f32>(n, n, 43), n);
    }

    #[test]
    fn general_eigen_survives_hard_matrices() {
        let bound = 256.0 * f64::EPSILON;
        // A rotation's eigenvalues are ±i exactly.
        let f = eig(&[0.0, -1.0, 1.0, 0.0], 2).unwrap();
        assert_eq!(f.values, [Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)]);
        check(&[0.0, -1.0, 1.0, 0.0], 2, &f, bound, "a rotation");
        // A cyclic permutation, whose standard shifts stall, so that only the exceptional shifts make progress.
        let n = 4;
        let cyclic: Vec<f64> = (0..n * n).map(|k| f64::from(k / n == (k % n + 1) % n)).collect();
        let f = eig(&cyclic, n).unwrap();
        check(&cyclic, n, &f, bound, "a cyclic permutation");
        for root in [Complex::new(1.0, 0.0), Complex::new(-1.0, 0.0), Complex::new(0.0, 1.0), Complex::new(0.0, -1.0)] {
            assert!(f.values.iter().any(|z| (z - root).norm() < bound), "{root} is an eigenvalue of a cyclic permutation");
        }
        // A defective matrix, whose eigenvectors are almost parallel, a zero matrix and a triangular one.
        let defective = [1.0, 1.0, 0.0, 1.0];
        let f = eig(&defective, 2).unwrap();
        assert_eq!(f.values, [Complex::new(1.0, 0.0); 2]);
        check(&defective, 2, &f, bound, "a defective matrix");
        check(&[0.0; 9], 3, &eig(&[0.0; 9], 3).unwrap(), bound, "a zero matrix");
        let triangular = [3.0, 1.0, 4.0, 0.0, 1.0, 5.0, 0.0, 0.0, 2.0];
        let f = eig(&triangular, 3).unwrap();
        assert_eq!(sorted(&f.values), [1.0, 2.0, 3.0].map(|x| Complex::new(x, 0.0)), "a triangular matrix's diagonal");
        check(&triangular, 3, &f, bound, "a triangular matrix");
        // Grcar's matrix, whose eigenvalues are very sensitive, and Frank's, whose small ones are badly conditioned. The residual measures backward stability, which holds for both.
        let n = 32;
        let grcar: Vec<f64> = (0..n * n).map(|k| match (k / n, k % n) {
            (i, j) if j + 1 == i => -1.0, (i, j) if j >= i && j <= i + 3 => 1.0, _ => 0.0
        }).collect();
        check(&grcar, n, &eig(&grcar, n).unwrap(), bound * n as f64, "Grcar's matrix");
        let n = 12;
        let frank: Vec<f64> = (0..n * n).map(|k| {
            let (i, j) = (k / n, k % n);
            if j + 1 < i { 0.0 } else { (n - i.max(j)) as f64 }
        }).collect();
        check(&frank, n, &eig(&frank, n).unwrap(), bound * n as f64, "Frank's matrix");
        // A symmetric matrix's eigenvalues are real, and match eigh's.
        let n = 20;
        let lower = random::<f64>(n, n, 5);
        let symmetric: Vec<f64> = (0..n * n).map(|k| {
            let (i, j) = (k / n, k % n);
            lower[i.max(j) * n + i.min(j)]
        }).collect();
        let mut values: Vec<f64> = eig(&symmetric, n).unwrap().values.iter().map(|z| {
            assert_eq!(z.im, 0.0);
            z.re
        }).collect();
        values.sort_by(f64::total_cmp);
        let want = eigh(&symmetric, n).unwrap().values;
        assert!(values.iter().zip(&want).all(|(a, b)| (a - b).abs() < bound * n as f64), "symmetric eigenvalues match eigh's");
        // A complex triangular matrix gives its diagonal, and a complex defective one repeats its eigenvalue.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let triangular = [c(1.0, 1.0), c(2.0, 0.0), c(3.0, -1.0), c(0.0, 0.0), c(0.0, -2.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.5, 0.0)];
        let f = eig(&triangular, 3).unwrap();
        assert_eq!(sorted(&f.values), sorted(&[triangular[0], triangular[4], triangular[8]]), "a complex triangular matrix's diagonal");
        check_units(&triangular, 3, &f, bound, "a complex triangular matrix");
        let defective = [c(0.0, 1.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 1.0)];
        let f = eig(&defective, 2).unwrap();
        assert_eq!(f.values, [c(0.0, 1.0); 2]);
        check_units(&defective, 2, &f, bound, "a complex defective matrix");
        // In complex form, the cyclic permutation's single shifts make progress too.
        let f = eig(&complex(&cyclic), 4).unwrap();
        check_units(&complex(&cyclic), 4, &f, bound, "a cyclic permutation in complex form");
        for root in [c(1.0, 0.0), c(-1.0, 0.0), c(0.0, 1.0), c(0.0, -1.0)] {
            assert!(f.values.iter().any(|z| (z - root).norm() < bound), "{root} is an eigenvalue of a cyclic permutation in complex form");
        }
        // A Hermitian matrix's eigenvalues are real.
        let a = complex_random::<f64>(n, n, 9);
        let hermitian: Vec<Complex<f64>> = (0..n * n).map(|k| a[k] + a[k % n * n + k / n].conj()).collect();
        let f = eig(&hermitian, n).unwrap();
        check_units(&hermitian, n, &f, bound * n as f64, "a Hermitian matrix");
        assert!(f.values.iter().all(|z| z.im.abs() < bound * n as f64), "a Hermitian matrix's eigenvalues are real");
    }

    /// Checks `eig` on the `n`×`n` matrix `a` scaled by a power of two near underflow and near overflow. Each scaled decomposition, with its eigenvalues divided by the scale, passes `a`'s residual and unit-length checks. Its eigenvalues also sum to `a`'s trace. A matrix holding NaN or an infinity gives `None`.
    fn scaled<T: Factor>(a: &[T], n: usize) where T::Real: RealItem {
        let wide: Vec<Complex<f64>> = a.iter().map(|v| Complex::new(v.re().wide(), v.im().wide())).collect();
        let trace: Complex<f64> = (0..n).map(|i| wide[i * n + i]).sum();
        let size = wide.iter().map(|z| z.norm_sqr()).sum::<f64>().sqrt();
        let tiny = T::Real::min_positive_value() / T::Real::epsilon();
        let bound = 64.0 * n as f64 * T::Real::epsilon().wide();
        for s in [tiny, tiny.recip()] {
            let g = widened(&eig(&a.iter().map(|v| v.mul_real(s)).collect::<Vec<T>>(), n).unwrap());
            let f = Eig { values: g.values.iter().map(|&z| z / s.wide()).collect(), vectors: g.vectors };
            check_units(&wide, n, &f, bound, &format!("scaled by {s:?}"));
            let sum: Complex<f64> = f.values.iter().sum();
            assert!((sum - trace).norm() < bound * size, "eigenvalues sum to the trace when scaled by {s:?}");
        }
        for bad in [T::Real::nan(), T::Real::infinity()] {
            let mut b = a.to_vec();
            b[2 * n + 5] = T::from_real(bad);
            assert!(eig(&b, n).is_none() && eigenvalues(&b, n).is_none(), "a matrix holding {bad:?}");
        }
    }
}

mod singular {
    use super::factor::{distance, norm, orthogonality, product, random};
    use super::*;

    /// The (n+1)×n lower bidiagonal matrix with diagonal `d` and subdiagonal `sub`.
    fn lower(d: &[f64], sub: &[f64]) -> Vec<f64> {
        let n = d.len();
        (0..(n + 1) * n).map(|k| match (k / n, k % n) { (i, j) if i == j => d[j], (i, j) if i == j + 1 => sub[j], _ => 0.0 }).collect()
    }

    #[test]
    fn divide_and_conquer_survives_hard_bidiagonals() {
        // Both sides of the public divide-and-conquer cutoff: zero singular values, negligible items of z, runs of equal singular values, and total deflation.
        for n in [40, SVD_SPLIT + 7] {
            let ramp = |f: fn(usize) -> f64| (0..n).map(f).collect::<Vec<f64>>();
            let cases = [
                (random::<f64>(n, 1, 3), random::<f64>(n, 1, 5)), (ramp(|_| 1.0), ramp(|_| 0.0)), (ramp(|_| 0.0), ramp(|_| 0.0)),
                (ramp(|i| if i % 3 == 0 { 0.0 } else { 1.0 + i as f64 }), ramp(|i| (i % 2) as f64)),
                (ramp(|i| 10f64.powf(-(i as f64) / 4.0)), ramp(|i| 10f64.powf(-(i as f64) / 4.0 - 1.0))), (ramp(|i| 1.0 + 1e-15 * i as f64), ramp(|_| 1e-12)),
                (ramp(|i| (i % 5) as f64), ramp(|i| 1.0 / (1.0 + i as f64))),
            ];
            for (d, sub) in &cases {
                let b = lower(d, sub);
                let Svd { u, s, v, .. } = svd(&b, [n + 1, n]);
                assert!(s.iter().all(|&x| x >= 0.0) && s.windows(2).all(|w| w[0] >= w[1]), "singular values descend at {n}: {s:?}");
                let us: Vec<f64> = (0..(n + 1) * n).map(|k| u[k] * s[k % n]).collect();
                let bound = 64.0 * f64::EPSILON * n as f64;
                assert!(distance(&product(&us, &transpose(&v, n, n), [n + 1, n, n]), &b, norm(&b).max(1.0)) < bound, "U Σ Vᵀ is B at {n}");
                assert!(orthogonality(&u, n + 1, n) < bound && orthogonality(&v, n, n) < bound, "U and V are orthogonal at {n}");
            }
        }
    }
}
mod exact {
    use num_rational::BigRational;
    use rustymath::linalg::exact::*;

    fn q(items: &[i64]) -> Vec<BigRational> { items.iter().map(|&k| BigRational::from_integer(k.into())).collect() }

    /// Row reduction, solves and determinants on exact rationals, checked against results worked out by hand.
    #[test]
    fn exact_linear_algebra() -> Result<(), rustymath::error::Error> {
        let a = q(&[2, 1, 1, 3]);
        assert_eq!(determinant(&a, 2)?, q(&[5])[0]);
        assert_eq!(determinant(&q(&[1, 2, 2, 4]), 2)?, q(&[0])[0]);
        assert_eq!(determinant::<BigRational>(&[], 0)?, q(&[1])[0]);
        // A row swap negates: the determinant of [0 1; 1 0] is -1.
        assert_eq!(determinant(&q(&[0, 1, 1, 0]), 2)?, q(&[-1])[0]);
        let x = solve(&a, &q(&[3, 5]), [2, 2], 1)?;
        assert_eq!(x, vec![BigRational::new(4.into(), 5.into()), BigRational::new(7.into(), 5.into())]);
        // A singular matrix: the least-norm solution of x + y = 2, 2x + 2y = 4 is (1, 1).
        assert_eq!(solve(&q(&[1, 1, 2, 2]), &q(&[2, 4]), [2, 2], 1)?, q(&[1, 1]));
        // A wide matrix: the least-norm solution of x + y + z = 3 is (1, 1, 1).
        assert_eq!(solve(&q(&[1, 1, 1]), &q(&[3]), [1, 3], 1)?, q(&[1, 1, 1]));
        // The pseudoinverse of a column (1, 1) is the row (1/2, 1/2).
        let half = BigRational::new(1.into(), 2.into());
        assert_eq!(pinv(&q(&[1, 1]), [2, 1])?, vec![half.clone(), half.clone()]);
        // An inverse exists only for a matrix of full rank.
        assert_eq!(inverse(&q(&[1, 1, 2, 2]), 2)?, None);
        assert_eq!(inverse(&q(&[2, 0, 0, 1]), 2)?, Some((vec![half, BigRational::default(), BigRational::default(), BigRational::from_integer(1.into())], BigRational::from_integer(2.into()))));
        let e = echelon(q(&[1, 2, 3, 2, 4, 7]), [2, 3], 3)?;
        assert_eq!((e.pivots, e.rows), (vec![0, 2], q(&[1, 2, 0, 0, 0, 1])));
        Ok(())
    }

    /// The permanent against its definition, a sum over permutations, and on the all-ones matrix, whose permanent is n!.
    #[test]
    fn permanents() -> Result<(), rustymath::error::Error> {
        assert_eq!(permanent(&q(&[1, 2, 3, 4]), 2)?, q(&[10])[0]);
        assert_eq!(permanent::<BigRational>(&[], 0)?, q(&[1])[0]);
        let a = q(&[2, -1, 3, 0, 5, 7, -2, 4, 1]);
        let by_definition: i64 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]].iter()
            .map(|p: &[usize; 3]| (0..3).map(|i| [2, -1, 3, 0, 5, 7, -2, 4, 1][i * 3 + p[i]]).product::<i64>()).sum();
        assert_eq!(permanent(&a, 3)?, q(&[by_definition])[0]);
        assert_eq!(permanent(&vec![1.0f64; 36], 6)?, 720.0);
        Ok(())
    }
}
