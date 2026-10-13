//! Matrix products and decompositions through their public APIs.
pub mod support;

use num_complex::Complex;
use rustymath::{linalg::*, Native};

fn identity<T: num_traits::Zero + num_traits::One>(n: usize) -> Vec<T> {
    (0..n * n).map(|k| if k % (n + 1) == 0 { T::one() } else { T::zero() }).collect()
}

fn transpose<T: Copy>(a: &[T], m: usize, n: usize) -> Vec<T> {
    (0..m * n).map(|k| a[k % m * n + k / m]).collect()
}

fn adjoint<T: Scalar>(a: &[T], m: usize, n: usize) -> Vec<T> { transpose(a, m, n).into_iter().map(T::conj).collect() }
mod products {
    use super::*;

    /// An inner product taken in order, item by item.
    fn folded<T: Copy, O: Copy>(a: &[T], b: &[T], [m, k, n]: [usize; 3], unit: O, step: impl Fn(O, T, T) -> O) -> Vec<O> {
        (0..m * n).map(|ij| (0..k).fold(unit, |acc, p| step(acc, a[ij / n * k + p], b[p * n + ij % n]))).collect()
    }

    /// The number of steps where the items differ, for each result.
    fn differing<T: Copy + PartialEq>(a: &[T], b: &[T], [m, k, n]: [usize; 3]) -> Vec<i64> {
        (0..m * n).map(|ij| (0..k).filter(|&p| a[ij / n * k + p] != b[p * n + ij % n]).count() as i64).collect()
    }

    #[test]
    fn products_match_folds_in_order() {
        // The sizes leave partial tiles, give no steps or a single one, and cross each block size. Small whole numbers keep every
        // float sum exact, in any order.
        let sizes = [[0, 3, 4], [3, 0, 4], [3, 4, 0], [1, 1, 1], [5, 7, 9], [MC + 3, KC + 5, 37], [17, 3, NC + 5]];
        for size @ [m, k, n] in sizes {
            let a64: Vec<f64> = (0..m * k).map(|i| (i * 7 % 17) as f64 - 8.0).collect();
            let b64: Vec<f64> = (0..k * n).map(|i| (i * 5 % 13) as f64 - 6.0).collect();
            let (a32, b32): (Vec<f32>, Vec<f32>) = (a64.iter().map(|&x| x as f32).collect(), b64.iter().map(|&x| x as f32).collect());
            assert_eq!(PlusTimes.inner(&a64, &b64, size), folded(&a64, &b64, size, 0.0, |s, x, y| s + x * y));
            assert_eq!(PlusTimes.inner(&a32, &b32, size), folded(&a32, &b32, size, 0.0, |s, x, y| s + x * y));
            assert_eq!(MaxPlus.inner(&a64, &b64, size), folded(&a64, &b64, size, f64::NEG_INFINITY, |s, x, y| s.max(x + y)));
            assert_eq!(MaxPlus.inner(&a32, &b32, size), folded(&a32, &b32, size, f32::NEG_INFINITY, |s, x, y| s.max(x + y)));
            assert_eq!(MinPlus.inner(&a64, &b64, size), folded(&a64, &b64, size, f64::INFINITY, |s, x, y| s.min(x + y)));
            assert_eq!(MinPlus.inner(&a32, &b32, size), folded(&a32, &b32, size, f32::INFINITY, |s, x, y| s.min(x + y)));
            let (ab, bb): (Vec<bool>, Vec<bool>) = (a64.iter().map(|&x| x > 4.0).collect(), b64.iter().map(|&x| x > 4.0).collect());
            assert_eq!(OrAnd.inner(&ab, &bb, size), folded(&ab, &bb, size, false, |s, x, y| s | (x & y)));
            assert_eq!(AndOr.inner(&ab, &bb, size), folded(&ab, &bb, size, true, |s, x, y| s & (x | y)));
            let (ai, bi): (Vec<i64>, Vec<i64>) = (a64.iter().map(|&x| x as i64).collect(), b64.iter().map(|&x| x as i64).collect());
            // Items too large for 32 bits take `+.=`'s 64-bit kernel. Items that fit 32 bits but no window of 2^16 values take its 32-bit
            // kernel. Small items, and items in a narrow window far from 0, take its 16-bit kernel.
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

    #[test]
    fn integer_products_match_wrapping_folds() {
        // Each pair of scales for `a` and `b` reaches one path. `+.×` runs at f32, f64, then i64 twice, the last time overflowing.
        // `⌈.+` and `⌊.+` run at f32 twice, then f64, then i64, overflowing. Odd scales give products that a narrower path would
        // round, where a power of two would keep them exact.
        let scales = [(1, 1), (4097, 4093), ((1 << 22) + 1, (1 << 22) + 3), ((1 << 59) + 1, (1 << 60) + 1)];
        for (sa, sb) in scales {
            for size @ [m, k, n] in [[0, 3, 4], [3, 0, 4], [3, 4, 0], [5, 7, 9], [MC + 3, KC + 5, 37]] {
                let a: Vec<i64> = (0..m * k).map(|i| ((i * 7 % 17) as i64 - 8) * sa).collect();
                let b: Vec<i64> = (0..k * n).map(|i| ((i * 5 % 13) as i64 - 6) * sb).collect();
                let overflows = sa > 1 << 59 && !a.is_empty() && !b.is_empty();
                let products = folded(&a, &b, size, 0i64, |s, x, y| s.wrapping_add(x.wrapping_mul(y)));
                assert_eq!(PlusTimes.inner(&a, &b, size), products);
                assert_eq!(PlusTimes.checked(&a, &b, size), (!overflows).then_some(products));
                let maxima = folded(&a, &b, size, i64::MIN, |s, x, y| s.max(x.wrapping_add(y)));
                assert_eq!(MaxPlus.inner(&a, &b, size), maxima);
                assert_eq!(MaxPlus.checked(&a, &b, size), (!overflows).then_some(maxima));
                let minima = folded(&a, &b, size, i64::MAX, |s, x, y| s.min(x.wrapping_add(y)));
                assert_eq!(MinPlus.inner(&a, &b, size), minima);
                assert_eq!(MinPlus.checked(&a, &b, size), (!overflows).then_some(minima));
            }
        }
    }

    #[test]
    fn complex_products_match_four_products() {
        // Each case leaves 3M's `s - p - q` non-finite where four real products give the right parts. `s` overflows for x(1+i)
        // squared with x² above a quarter of the largest float. Its sum `aᵣ + aᵢ` overflows at max(1+i). ∞ times 1+i gives ∞ - ∞.
        // The values keep every product exact, so `*`, which forms four real products, gives each result's bits.
        macro_rules! check {
            ($r:ty, $x:expr) => {{
                let (c, x, max, inf) = (Complex::<$r>::new, $x, <$r>::MAX, <$r>::INFINITY);
                for (p, q) in [(c(x, x), c(x, x)), (c(max, max), c(1.0, 0.0)), (c(inf, 0.0), c(1.0, 1.0)), (c(1.0, 2.0), c(3.0, 4.0))] {
                    assert_eq!(PlusTimes.inner(&[p], &[q], [1, 1, 1]), [p * q]);
                }
                // Results beside a non-finite one keep 3M's parts, which are exact here.
                let (a, b) = ([c(inf, 0.0), c(1.0, 2.0)], [c(1.0, 1.0), c(3.0, 4.0)]);
                assert_eq!(PlusTimes.inner(&a, &b, [2, 1, 2]), [a[0] * b[0], a[0] * b[1], a[1] * b[0], a[1] * b[1]]);
            }};
        }
        check!(f64, 2f64.powi(511));
        check!(f32, 2f32.powi(63));
    }

    #[test]
    fn extremes_give_nan_beside_nan() {
        // A NaN in the first step, and one in a later step, each turn their whole row of results to NaN.
        let size @ [m, k, n] = [9, 300, 11];
        let mut a: Vec<f64> = (0..m * k).map(|i| (i % 23) as f64).collect();
        let b: Vec<f64> = (0..k * n).map(|i| (i % 19) as f64).collect();
        (a[0], a[3 * k + 280]) = (f64::NAN, f64::NAN);
        for c in [MaxPlus.inner(&a, &b, size), MinPlus.inner(&a, &b, size)] {
            for (i, row) in c.chunks(n).enumerate() { assert!(row.iter().all(|x| x.is_nan() == (i == 0 || i == 3)), "row {i}: {row:?}") }
        }
        let a32: Vec<f32> = a.iter().map(|&x| x as f32).collect();
        let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        for c in [MaxPlus.inner(&a32, &b32, size), MinPlus.inner(&a32, &b32, size)] {
            for (i, row) in c.chunks(n).enumerate() { assert!(row.iter().all(|x| x.is_nan() == (i == 0 || i == 3)), "row {i}: {row:?}") }
        }
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

    /// How far the `rows`×`cols` matrix `q` is from having orthonormal columns.
    pub(super) fn orthogonality(q: &[f64], rows: usize, cols: usize) -> f64 {
        let qt = transpose(q, rows, cols);
        distance(&product(&qt, q, [cols, rows, cols]), &identity(cols), 1.0)
    }

    /// Checks each decomposition at `T` on matrices either side of the block width, where `tolerance` is a few times `T`'s epsilon.
    fn decompositions<T: RealItem>(tolerance: f64) {
        let nb = lu_block(0);
        for n in [0, 1, 2, 5, nb - 1, nb, nb + 1, 2 * nb + 7] {
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
            let spd: Vec<T> = product(&aw, &transpose(&aw, n, n), [n; 3]).iter().enumerate().map(|(k, &v)| T::narrow(v + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
            let l = wide(&cholesky(&spd, n).expect("a positive definite matrix"));
            assert!((0..n * n).all(|k| k % n <= k / n || l[k] == 0.0), "cholesky's upper triangle at {n}");
            assert!(distance(&product(&l, &transpose(&l, n, n), [n; 3]), &wide(&spd), norm(&wide(&spd)).max(1.0)) < bound, "cholesky at {n}");
            if n > 0 { assert_eq!(cholesky(&spd.iter().map(|&v| -v).collect::<Vec<_>>(), n), None, "cholesky of a negative definite matrix") }
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
                let (u, s, v) = (wide(&f.u), wide(&f.s), wide(&f.v));
                assert!(s.windows(2).all(|w| w[0] >= w[1]) && s.iter().all(|&x| x >= 0.0), "svd's s at {m}×{c}");
                let us: Vec<f64> = (0..m * p).map(|k| u[k] * s[k % p]).collect();
                assert!(distance(&product(&us, &transpose(&v, c, p), [m, p, c]), &aw, scale) < bound, "svd at {m}×{c}");
                // The accumulated rotations' rounding errors add up over the sweeps, in V, or in U for a wide matrix.
                let (ou, ov) = (orthogonality(&u, m, p), orthogonality(&v, c, p));
                assert!(ou < 4.0 * bound && ov < 4.0 * bound, "svd's u and v at {m}×{c}: {ou:e} and {ov:e}, against {bound:e}");
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
        // A bidiagonal matrix with a zero on its diagonal reaches the QR algorithm unchanged, which rotates that row's superdiagonal
        // item away before it iterates.
        let (d, e) = ([3.0, 0.0, 2.0, 0.5, 1.0], [1.0, 1.0, 0.25, 2.0]);
        let b: Vec<T> = (0..25).map(|k| T::narrow(match (k / 5, k % 5) { (i, j) if i == j => d[i], (i, j) if j == i + 1 => e[i], _ => 0.0 })).collect();
        let f = svd(&b, [5, 5]);
        let us: Vec<f64> = (0..25).map(|k| wide(&f.u)[k] * wide(&f.s)[k % 5]).collect();
        assert!(distance(&product(&us, &transpose(&wide(&f.v), 5, 5), [5; 3]), &wide(&b), norm(&wide(&b))) < 8.0 * tolerance, "svd of a bidiagonal with a zero");
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

    /// Checks the decompositions at `T` on LAPACK's hard matrix types, where `tolerance` is a few times `T`'s epsilon. A matrix scaled
    /// by a power of two near underflow or overflow gives factors scaled exactly in step. A zero column gives finite LU factors with a
    /// zero pivot. A subnormal pivot gives finite multipliers.
    fn hard_matrices<T: RealItem>(tolerance: f64) {
        let n = 40;
        let tiny = T::min_positive_value() / T::epsilon();
        let (a, b) = (random::<T>(n, n, 17), random::<T>(n, 3, 19));
        let aw = wide(&a);
        let spd: Vec<T> = product(&aw, &transpose(&aw, n, n), [n; 3]).iter().enumerate().map(|(k, &v)| T::narrow(v + if k % (n + 1) == 0 { n as f64 } else { 0.0 })).collect();
        // Fourth powers of items scaled by `quartic` overflow, though sums of their squares don't, so `safe_scale` leaves them alone.
        let quartic = T::narrow(2f64.powi((T::max_value().log2().wide() / 4.0) as i32 + 6));
        for s in [tiny, tiny.recip(), quartic] {
            let times = |x: &[T], c: T| x.iter().map(|&v| v * c).collect::<Vec<T>>();
            let (f, g) = (lu(&a, n), lu(&times(&a, s), n));
            assert_eq!(g.perm, f.perm, "lu's permutation scaled by {s:?}");
            assert_eq!(g.factors, f.factors.iter().enumerate().map(|(k, &v)| if k % n >= k / n { v * s } else { v }).collect::<Vec<T>>(), "lu scaled by {s:?}");
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
                assert_eq!((g.u, g.s, g.v), (f.u, times(&f.s, s), f.v), "svd scaled by {s:?} at {rows}×{cols}");
                let ((x, sv), (xs, svs)) = (lstsq(&a, &b, [rows, cols], 2), lstsq(&times(&a, s), &times(&b, s), [rows, cols], 2));
                assert_eq!((xs, svs), (x, times(&sv, s)), "lstsq scaled by {s:?} at {rows}×{cols}");
            }
        }
        let n = 2 * lu_block(0) + 7;
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
        assert_eq!(lu(&[p, T::one(), q, T::one()], 2).factors, [p, T::one(), q / p, T::one() - q / p], "lu with a subnormal pivot");
    }

    /// Checks `solve`, `pinv`, `qr_complete`, the determinants and `hermitian_eigenvalues` at `T`, where `tolerance` is a few times
    /// `T`'s epsilon. The singular matrices repeat columns, so their rank is exact.
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

    /// Checks LU, Cholesky and QR at `T` on matrices large enough for the wider blocks, each leaving a partial last block, where
    /// `tolerance` is a few times `T`'s epsilon. Products with random vectors keep each check to O(n²) time.
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
        let n = 2 * lu_block(0) + 7;
        let rows = n + 5;
        let a = random::<T>(rows, n, rows as u64 + 7);
        let (q, r) = qr(&a, [rows, n]);
        let (qt, r) = (transpose(&wide(&q), rows, n), wide(&r));
        for k in [2, FEW_COLUMNS + 1] {
            let b = random::<T>(rows, k, 13);
            let (x, _) = lstsq(&a, &b, [rows, n], k);
            let bw = wide(&b);
            assert!(distance(&product(&r, &wide(&x), [n, n, k]), &product(&qt, &bw, [n, rows, k]), norm(&bw)) < tolerance * n as f64, "lstsq's Qᴴ b at {rows}×{n}×{k}");
        }
    }

    #[test]
    fn blocked_decompositions_have_small_residuals() {
        blocked::<f64>(4.0 * f64::EPSILON);
        blocked::<f32>(4.0 * f32::EPSILON as f64);
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

    /// Checks LU with its solves and inverse, Cholesky and QR at complex items with parts of type `R`, on the matrices `decompositions`
    /// takes, where `tolerance` is a few times `R`'s epsilon.
    fn complex_decompositions<R: RealItem>(tolerance: f64)
    where
        Complex<R>: Factor + Scalar<Real = R>,
    {
        let zero = C64::new(0.0, 0.0);
        let nb = lu_block(0);
        for n in [0, 1, 2, 5, nb - 1, nb, nb + 1, 2 * nb + 7] {
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
                assert!(complex_distance(&complex_product(&aw, &x, [n, n, k]), &complex_wide(&b), (scale * complex_norm(&x)).max(1.0)) < bound, "complex solve at {n}, {k}");
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
            if n > 0 { assert!(cholesky(&hpd.iter().map(|&v| -v).collect::<Vec<_>>(), n).is_none(), "complex cholesky of a negative definite matrix") }
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
    }


    /// Checks the SVD and the Hermitian eigen decomposition at complex items with parts of type `R`, either side of each block width,
    /// where `tolerance` is a few times `R`'s epsilon.
    fn complex_svd_and_eigen<R: RealItem>(tolerance: f64)
    where
        Complex<R>: Factor + Scalar<Real = R>,
    {
        type C<R> = num_complex::Complex<R>;
        for n in [0, 1, 2, 5, 31, 64, 129] {
            let bound = tolerance * (n as f64).max(1.0);
            // Square, tall, very tall and wide matrices: U Σ Vᴴ is A, U and V have orthonormal columns, and the singular values descend.
            for (m, c) in [(n, n), (n + 5, n), (2 * n + 5, n), (n, n + 5)] {
                let a = complex_random::<R>(m, c, (m * 37 + c) as u64);
                let (aw, p) = (complex_wide(&a), m.min(c));
                let Svd { u, s, v, .. } = svd(&a, [m, c]);
                assert!(distance(&wide(&singular_values(&a, [m, c])), &wide(&s), complex_norm(&aw).max(1.0)) < bound, "complex singular values alone at {m}×{c}");
                let (u, v, s) = (complex_wide(&u), complex_wide(&v), wide(&s));
                let us: Vec<C64> = (0..m * p).map(|k| u[k] * s[k % p]).collect();
                assert!(complex_distance(&complex_product(&us, &adjoint(&v, c, p), [m, p, c]), &aw, complex_norm(&aw).max(1.0)) < bound, "complex svd at {m}×{c}");
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
            assert!(complex_distance(&complex_product(&adjoint(&vectors, n, n), &vectors, [n; 3]), &identity::<C64>(n), 1.0) < bound, "unitary eigenvectors at {n}");
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
    use super::factor::{distance, norm, orthogonality, product, random, RealItem, wide};
    use super::*;

    /// The symmetric `n`×`n` matrix whose lower triangle comes from `lower`, with `junk` in its upper triangle, which `eigh` never reads.
    fn symmetric<T: RealItem>(lower: &[T], n: usize, junk: T) -> Vec<T> {
        (0..n * n).map(|k| if k % n > k / n { junk } else { lower[k] }).collect()
    }

    /// The symmetric tridiagonal `n`×`n` matrix with diagonal `d` and subdiagonal `e`, in full.
    fn tridiagonal_matrix(d: &[f64], e: &[f64]) -> Vec<f64> {
        let n = d.len();
        (0..n * n)
            .map(|k| match (k / n, k % n) {
                (i, j) if i == j => d[i],
                (i, j) if i == j + 1 || j == i + 1 => e[i.min(j)],
                _ => 0.0,
            })
            .collect()
    }

    /// Checks that `f` decomposes the symmetric `n`×`n` matrix `a`: ascending values, orthonormal vectors, and a small residual
    /// `A V - V Λ`, all within `bound`.
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
        for (a, n, values) in [(identity::<f64>(4), 4, vec![1.0; 4]), (vec![0.0; 9], 3, vec![0.0; 3]), (tridiagonal_matrix(&[2.0, 1.0, 2.0, 1.0], &[0.0; 3]), 4, vec![1.0, 1.0, 2.0, 2.0])] {
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
            (wilkinson.to_vec(), vec![1.0; 6], "a Wilkinson matrix"),
            (glued_d, glued_e, "two glued Wilkinson matrices"),
            (random_d, random_e, "a random tridiagonal matrix"),
            (clustered_d, clustered_e, "a matrix with two clusters of eigenvalues"),
            (vec![1.0; 50], vec![0.0; 49], "the identity"),
            ((0..40).map(|i| (i % 2 + 1) as f64).collect(), vec![0.0; 39], "a diagonal matrix with ties"),
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

    /// Checks that `eigh` at `T` scales exactly with a matrix scaled by a power of two near underflow or overflow, and gives `None` for
    /// a matrix holding NaN.
    fn hard_matrices<T: RealItem>() {
        let n = 40;
        let a = symmetric(&random::<T>(n, n, 41), n, T::zero());
        let f = eigh(&a, n).unwrap();
        let tiny = T::min_positive_value() / T::epsilon();
        for s in [tiny, tiny.recip()] {
            let g = eigh(&a.iter().map(|&v| v * s).collect::<Vec<T>>(), n).unwrap();
            assert_eq!(g.vectors, f.vectors, "eigenvectors scaled by {s:?}");
            assert_eq!(g.values, f.values.iter().map(|&v| v * s).collect::<Vec<T>>(), "eigenvalues scaled by {s:?}");
        }
        let mut nan = a.clone();
        nan[3 * n + 1] = T::nan();
        assert!(eigh(&nan, n).is_none(), "a matrix holding NaN");
    }

}

mod general {
    use super::factor::{complex_random, complex_wide, random, RealItem, wide};
    use super::*;
    use num_traits::Float as _;

    /// `a` with each item as a complex number.
    fn complex(a: &[f64]) -> Vec<Complex<f64>> { a.iter().map(|&x| Complex::new(x, 0.0)).collect() }

    /// Checks what `eig` promises for any `n`×`n` matrix `a`: a relative residual `‖A V - V Λ‖ / ‖A‖` below `bound`, and each
    /// eigenvector of unit length, to within `bound`.
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

    /// `f` at f64, from a decomposition at `T`.
    fn widened<T: RealItem>(f: &Eig<T>) -> Eig<f64> {
        let wide = |z: &Complex<T>| Complex::new(z.re.wide(), z.im.wide());
        Eig { values: f.values.iter().map(wide).collect(), vectors: f.vectors.iter().map(wide).collect() }
    }

    /// Checks `eig` at `T` on random matrices, where `tolerance` is a few times `T`'s epsilon.
    fn random_matrices<T: RealItem>(tolerance: f64) {
        for n in [0, 1, 2, 3, 5, 31, 64, 129] {
            let a = random::<T>(n, n, n as u64 + 7);
            let f = eig(&a, n).expect("a random matrix converges");
            assert_eq!(eigenvalues(&a, n).as_ref(), Some(&f.values), "eigenvalues alone match eig's at {n}");
            check(&wide(&a), n, &widened(&f), tolerance * (n as f64).max(1.0), &format!("{n}×{n}"));
        }
    }

    /// Checks the complex general eigen problem at parts `R` on random matrices, where `tolerance` is a few times `R`'s epsilon.
    fn complex_matrices<R: RealItem>(tolerance: f64)
    where
        Complex<R>: Factor + Scalar<Real = R>,
    {
        for n in [0, 1, 2, 3, 5, 31, 64, 129] {
            let a = complex_random::<R>(n, n, n as u64 + 11);
            let f = eig(&a, n).expect("a random complex matrix converges");
            assert_eq!(eigenvalues(&a, n).as_ref(), Some(&f.values), "eigenvalues alone match at complex {n}");
            check_units(&complex_wide(&a), n, &widened(&f), tolerance * (n as f64).max(1.0), &format!("complex {n}×{n}"));
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
        assert_eq!(f.values, [3.0, 1.0, 2.0].map(|x| Complex::new(x, 0.0)), "a triangular matrix's diagonal, in order");
        check(&triangular, 3, &f, bound, "a triangular matrix");
        // Grcar's matrix, whose eigenvalues are very sensitive, and Frank's, whose small ones are badly conditioned. The residual
        // measures backward stability, which holds for both.
        let n = 32;
        let grcar: Vec<f64> = (0..n * n).map(|k| match (k / n, k % n) { (i, j) if j + 1 == i => -1.0, (i, j) if j >= i && j <= i + 3 => 1.0, _ => 0.0 }).collect();
        check(&grcar, n, &eig(&grcar, n).unwrap(), bound * n as f64, "Grcar's matrix");
        let n = 12;
        let frank: Vec<f64> = (0..n * n).map(|k| { let (i, j) = (k / n, k % n); if j + 1 < i { 0.0 } else { (n - i.max(j)) as f64 } }).collect();
        check(&frank, n, &eig(&frank, n).unwrap(), bound * n as f64, "Frank's matrix");
        // A symmetric matrix's eigenvalues are real, and match eigh's.
        let n = 20;
        let lower = random::<f64>(n, n, 5);
        let symmetric: Vec<f64> = (0..n * n).map(|k| { let (i, j) = (k / n, k % n); lower[i.max(j) * n + i.min(j)] }).collect();
        let mut values: Vec<f64> = eig(&symmetric, n).unwrap().values.iter().map(|z| { assert_eq!(z.im, 0.0); z.re }).collect();
        values.sort_by(f64::total_cmp);
        let want = eigh(&symmetric, n).unwrap().values;
        assert!(values.iter().zip(&want).all(|(a, b)| (a - b).abs() < bound * n as f64), "symmetric eigenvalues match eigh's");
        // A complex triangular matrix gives its diagonal, in order, and a complex defective one repeats its eigenvalue.
        let c = |re: f64, im: f64| Complex::new(re, im);
        let triangular = [c(1.0, 1.0), c(2.0, 0.0), c(3.0, -1.0), c(0.0, 0.0), c(0.0, -2.0), c(1.0, 0.0), c(0.0, 0.0), c(0.0, 0.0), c(0.5, 0.0)];
        let f = eig(&triangular, 3).unwrap();
        assert_eq!(f.values, [triangular[0], triangular[4], triangular[8]], "a complex triangular matrix's diagonal, in order");
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

    /// Checks that `eig` scales exactly with the `n`×`n` matrix `a` scaled by a power of two near underflow or overflow, and
    /// gives `None` for a matrix holding NaN or an infinity. `a`'s largest item lies between 1 and 2, where `safe_scale` brings each
    /// scaled copy back to `a`. The iterations compare quantities in the matrix's own units, so only that matrix gives the same steps.
    fn scaled<T: Factor>(a: &[T], n: usize)
    where
        T::Real: std::fmt::Debug,
    {
        let f = eig(a, n).unwrap();
        let tiny = T::Real::min_positive_value() / T::Real::epsilon();
        for s in [tiny, tiny.recip()] {
            let g = eig(&a.iter().map(|v| v.mul_real(s)).collect::<Vec<T>>(), n).unwrap();
            assert_eq!(g.vectors, f.vectors, "eigenvectors scaled by {s:?}");
            assert_eq!(g.values, f.values.iter().map(|&z| z * s).collect::<Vec<_>>(), "eigenvalues scaled by {s:?}");
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
        // Both sides of the public divide-and-conquer cutoff: zero singular values, negligible items of z, runs of equal singular
        // values, and total deflation.
        for n in [40, SVD_SPLIT + 7] {
            let ramp = |f: fn(usize) -> f64| (0..n).map(f).collect::<Vec<f64>>();
            let cases = [
                (random::<f64>(n, 1, 3), random::<f64>(n, 1, 5)),
                (ramp(|_| 1.0), ramp(|_| 0.0)),
                (ramp(|_| 0.0), ramp(|_| 0.0)),
                (ramp(|i| if i % 3 == 0 { 0.0 } else { 1.0 + i as f64 }), ramp(|i| (i % 2) as f64)),
                (ramp(|i| 10f64.powf(-(i as f64) / 4.0)), ramp(|i| 10f64.powf(-(i as f64) / 4.0 - 1.0))),
                (ramp(|i| 1.0 + 1e-15 * i as f64), ramp(|_| 1e-12)),
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
    use rustymath::linalg::exact::*;
    use num_rational::BigRational;

    fn q(items: &[i64]) -> Vec<BigRational> { items.iter().map(|&k| BigRational::from_integer(k.into())).collect() }

    /// Row reduction, solves and determinants on exact rationals, checked against results worked out by hand.
    #[test]
    fn exact_linear_algebra() {
        let a = q(&[2, 1, 1, 3]);
        assert_eq!(determinant(&a, 2), q(&[5])[0]);
        assert_eq!(determinant(&q(&[1, 2, 2, 4]), 2), q(&[0])[0]);
        assert_eq!(determinant::<BigRational>(&[], 0), q(&[1])[0]);
        // A row swap negates: the determinant of [0 1; 1 0] is -1.
        assert_eq!(determinant(&q(&[0, 1, 1, 0]), 2), q(&[-1])[0]);
        let x = solve(&a, &q(&[3, 5]), [2, 2], 1);
        assert_eq!(x, vec![BigRational::new(4.into(), 5.into()), BigRational::new(7.into(), 5.into())]);
        // A singular matrix: the least-norm solution of x + y = 2, 2x + 2y = 4 is (1, 1).
        assert_eq!(solve(&q(&[1, 1, 2, 2]), &q(&[2, 4]), [2, 2], 1), q(&[1, 1]));
        // A wide matrix: the least-norm solution of x + y + z = 3 is (1, 1, 1).
        assert_eq!(solve(&q(&[1, 1, 1]), &q(&[3]), [1, 3], 1), q(&[1, 1, 1]));
        // The pseudoinverse of a column (1, 1) is the row (1/2, 1/2).
        let half = BigRational::new(1.into(), 2.into());
        assert_eq!(pinv(&q(&[1, 1]), [2, 1]), vec![half.clone(), half]);
        let e = echelon(q(&[1, 2, 3, 2, 4, 7]), [2, 3], 3);
        assert_eq!((e.pivots, e.rows), (vec![0, 2], q(&[1, 2, 0, 0, 0, 1])));
    }

    /// The permanent against its definition, a sum over permutations, and on the all-ones matrix, whose permanent is n!.
    #[test]
    fn permanents() {
        assert_eq!(permanent(&q(&[1, 2, 3, 4]), 2), q(&[10])[0]);
        assert_eq!(permanent::<BigRational>(&[], 0), q(&[1])[0]);
        let a = q(&[2, -1, 3, 0, 5, 7, -2, 4, 1]);
        let by_definition: i64 = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]]
            .iter()
            .map(|p: &[usize; 3]| (0..3).map(|i| [2, -1, 3, 0, 5, 7, -2, 4, 1][i * 3 + p[i]]).product::<i64>())
            .sum();
        assert_eq!(permanent(&a, 3), q(&[by_definition])[0]);
        assert_eq!(permanent(&vec![1.0f64; 36], 6), 720.0);
    }
}

