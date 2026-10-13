pub mod support;

mod arith {
    use rustymath::Overflowing;

    /// Each operation at type `$t`, on every pair of edge values, against the exact result in `i128`: the wrapped result, and whether
    /// it fits.
    macro_rules! check {
        ($($t:ty),+) => {$({
            let edges: Vec<$t> = [<$t>::MIN, <$t>::MAX, 0, 1, 2, 3, 7].into_iter().chain([-1i128, -2, -3].map(|v| v as $t))
                .chain([<$t>::MIN / 2, <$t>::MAX / 2, <$t>::MAX - 1, <$t>::MIN.wrapping_add(1)]).collect();
            let exact = |r: i128| (r as $t, <$t>::try_from(r).is_ok());
            for &x in &edges {
                for &y in &edges {
                    let (a, b) = (x as i128, y as i128);
                    assert_eq!(x.plus(y), exact(a + b), "{x} + {y}");
                    assert_eq!(x.minus(y), exact(a - b), "{x} - {y}");
                    assert_eq!(x.times(y), exact(a * b), "{x} × {y}");
                }
                assert_eq!(x.magnitude(), exact((x as i128).abs()), "|{x}|");
                assert_eq!(x.sign() as i128, (x as i128).signum(), "sign {x}");
            }
        })+};
    }

    #[test]
    fn overflowing_matches_exact_arithmetic() { check!(i8, i16, i32, i64, u8, u16, u32, u64); }
}

mod convert {
    use rustymath::convert::*;
    use half::f16;

    /// Each conversion against `as` and exact arithmetic, at the edges of each type's range and of the wide limit.
    #[test]
    fn conversions() {
        assert_eq!((int::<i32, f64>(-2147483648.0), int::<i32, f32>(-2147483648.0), int::<i32, f32>(2147483648.0)), (Some(i32::MIN), None, None));
        assert_eq!((int::<u8, f64>(255.0), int::<u8, f64>(256.0), int::<u8, f32>(-1.0), int::<u16, f32>(65535.0)), (Some(255), None, None, Some(65535)));
        assert_eq!((int::<i16, f16>(f16::from_f32(-300.0)), int::<i8, f64>(f64::NAN), int::<i8, f32>(f32::INFINITY)), (Some(-300), None, None));
        assert_eq!((int::<i64, f64>(-2251799813685247.0), int::<i64, f64>(2251799813685248.0)), (Some(-2251799813685247), None));
        assert_eq!((int::<i64, f32>(-1e15), int::<u32, f64>(4294967295.0), int::<u32, f64>(4294967296.0)), (Some(-999999986991104), Some(u32::MAX), None));
        for x in [0.0, -1.0, 2251799813685247.0, -2251799813685247.0, 1e15] { assert_eq!(int::<i64, _>(x), Some(x as i64), "{x}"); }
        for i in [0, 1, -1, 255, 1 << 31, -(1 << 31) - 1, (1 << 53) + 1, (1 << 53) + 3, i64::MAX, i64::MIN, 0x7fff_ffff_ffff_fdff] {
            assert_eq!((float::<f64, i64>(i), float::<f32, i64>(i)), (i as f64, i as f64 as f32), "{i}");
        }
        assert_eq!((float::<f64, u8>(255), float::<f32, u16>(65535), float::<f64, u32>(u32::MAX), float::<f16, i32>(-3)), (255.0, 65535.0, 4294967295.0, f16::from_f32(-3.0)));
        let mut counts = Vec::new();
        count(&mut counts, 0, false, 21);
        count(&mut counts, 20, true, 21);
        assert_eq!(counts, (0..21).chain((0..21).rev()).map(|i| i as f64).collect::<Vec<_>>());
    }
}

mod distribution {
    use rustymath::distribution::*;
    use rustymath::{Arg, Unary};
    use rand::RngExt;
    use crate::support::seeded;

    /// One of each distribution, with each sampling method's case: binomials that draw by inversion, by BTRD and reflected, and
    /// Poisson distributions that draw by multiplying and by PTRS.
    fn cases() -> Vec<Distribution> {
        let d = Distribution::binomial;
        [
            Distribution::normal(1.0, 2.0), Distribution::uniform(-1.0, 3.0), Distribution::beta(0.5, 2.0), Distribution::beta(3.0, 4.0),
            Distribution::cauchy(0.0, 1.0), Distribution::chi_squared(3.0), Distribution::exponential(2.0), Distribution::fisher(4.0, 7.0),
            Distribution::gamma(0.5, 2.0), Distribution::gamma(5.0, 1.0), Distribution::inverse_gamma(3.0, 2.0), Distribution::laplace(1.0, 2.0),
            Distribution::logistic(0.0, 1.0), Distribution::log_normal(0.0, 0.5), Distribution::student_t(3.0), Distribution::weibull(1.5, 2.0),
            d(10, 0.3), d(1000, 0.4), d(100, 0.9), Distribution::poisson(3.0), Distribution::poisson(50.0),
        ]
        .into_iter()
        .map(Result::unwrap)
        .collect()
    }

    #[test]
    fn quantiles_and_pdfs_match_cdfs() {
        for d in cases() {
            for p in [0.001, 0.1, 0.5, 0.9, 0.999] {
                let x = d.quantile(p);
                if d.is_discrete() {
                    assert!(d.cdf(x) >= p * (1.0 - 1e-14) && d.cdf(x - 1.0) < p, "{d:?} at {p}: {x}");
                } else {
                    assert!((d.cdf(x) - p).abs() < 1e-12, "{d:?} at {p}: cdf({x}) is {}", d.cdf(x));
                }
            }
            for p in [0.1, 0.5, 0.9] {
                let x = d.quantile(p);
                if d.is_discrete() {
                    let sum: f64 = (0..=x as u64).map(|k| d.pdf(k as f64)).sum();
                    assert!((sum - d.cdf(x)).abs() < 1e-12, "{d:?} at {x}: masses sum to {sum}, cdf is {}", d.cdf(x));
                } else {
                    let h = 1e-5 * x.abs().max(1e-3);
                    let slope = (d.cdf(x + h) - d.cdf(x - h)) / (2.0 * h);
                    assert!((slope - d.pdf(x)).abs() < 1e-5 * d.pdf(x), "{d:?} at {x}: pdf {} against slope {slope}", d.pdf(x));
                }
            }
        }
    }

    /// The fraction of draws at or below each of three quantiles matches the cdf there, within five standard errors.
    #[test]
    fn samples_follow_cdfs() {
        const N: usize = 20_000;
        let mut rng = seeded(1);
        let mut out = vec![0.0; N];
        for d in cases() {
            d.samples(&mut rng, &mut out);
            for p in [0.1, 0.5, 0.9] {
                let x = d.quantile(p);
                let c = d.cdf(x);
                let below = out.iter().filter(|&&v| v <= x).count() as f64 / N as f64;
                assert!((below - c).abs() < 5.0 * (c * (1.0 - c) / N as f64).sqrt(), "{d:?}: {below} of draws at most {x}, where the cdf is {c}");
            }
        }
        let mut one = [0.0; 1];
        Distribution::poisson(0.0).unwrap().samples(&mut rng, &mut one);
        assert_eq!(one[0], 0.0);
    }

    /// `f`'s slice form against `scalar` for each item of `xs`, at f64 and at f32. Gives whether the slice form reported every item
    /// as succeeding.
    fn matches_scalar(f: impl Unary<f64, Out = f64> + Unary<f32, Out = f32>, scalar: impl Fn(f64) -> f64, xs: &[f64]) -> bool {
        let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
        let (mut out, mut out32) = (Vec::new(), Vec::new());
        let ok = f.many(Arg::Many(xs), xs.len(), &mut out);
        let xs32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
        assert_eq!(f.many(Arg::Many(&xs32), xs32.len(), &mut out32), ok);
        assert!(out.iter().zip(xs).all(|(&r, &x)| same(r, scalar(x))), "{out:?}");
        assert!(out32.iter().zip(&xs32).all(|(&r, &x)| same(r.into(), scalar(x.into()) as f32 as f64)), "{out32:?}");
        ok
    }

    /// Each distribution's function objects give their scalar functions' results. A quantile fails where p lies outside 0 to 1.
    #[test]
    fn objects_match_scalars() {
        let (xs, ps) = ([-1.0, 0.0, 0.25, 0.5, 1.0, 3.0, f64::NAN], [0.001, 0.25, 0.5, 0.999, f64::NAN]);
        for d in cases() {
            assert!(matches_scalar(Pdf(d), |x| d.pdf(x), &xs) && matches_scalar(Cdf(d), |x| d.cdf(x), &xs), "{d:?}");
            assert!(matches_scalar(Quantile(d), |p| d.quantile(p), &ps), "{d:?}");
            assert!(!matches_scalar(Quantile(d), |p| d.quantile(p), &xs), "{d:?}");
        }
        // Several blocks of uniform p reach every region of the inverse error function, in random order.
        let mut rng = seeded(1);
        let uniform: Vec<f64> = (0..5000).map(|_| rng.random()).collect();
        for d in [Distribution::normal(1.0, 2.0).unwrap(), Distribution::log_normal(0.0, 0.5).unwrap()] {
            assert!(matches_scalar(Quantile(d), |p| d.quantile(p), &uniform), "{d:?}");
        }
    }
}

mod exact {
    use rustymath::exact::*;
    use num_rational::BigRational;

    fn q(n: i64, d: i64) -> BigRational { BigRational::new(n.into(), d.into()) }

    #[test]
    fn exact_functions() {
        assert_eq!((residue(&q(3, 1), &q(-7, 1)), residue(&q(1, 2), &q(5, 3)), residue(&q(0, 1), &q(5, 3))), (q(2, 1), q(1, 6), q(5, 3)));
        assert_eq!((gcd(&q(1, 2), &q(1, 3)), gcd(&q(-4, 1), &q(6, 1)), gcd(&q(0, 1), &q(0, 1))), (q(1, 6), q(2, 1), q(0, 1)));
        assert_eq!((lcm(&q(1, 2), &q(1, 3)), lcm(&q(-4, 1), &q(6, 1)), lcm(&q(0, 1), &q(5, 1))), (q(1, 1), q(-12, 1), q(0, 1)));
        // C(5, 2), C(5, 7), C(1/2, 2) = -1/8, C(-1, 2) = 1, C(-3, -5) = (-1)^2 C(4, 2) = 6, C(-3, -1) = 0 and C(1/2, -1) = 0.
        let c: Vec<BigRational> = [(q(5, 1), 2), (q(5, 1), 7), (q(1, 2), 2), (q(-1, 1), 2), (q(-3, 1), -5), (q(-3, 1), -1), (q(1, 2), -1)]
            .iter()
            .map(|(n, k)| binom(n, *k))
            .collect();
        assert_eq!(c, [q(10, 1), q(0, 1), q(-1, 8), q(1, 1), q(6, 1), q(0, 1), q(0, 1)]);
    }
}

mod integer {
    use rustymath::integer::*;
    use crate::support::random;
    use rustymath::{tolerant::{Ceil, Equal, Floor, Gcd, Greater, GreaterEqual, Lcm, Less, LessEqual, NotEqual, Residue}, Abs, Arg, Binary, Binom, Pow, Sign, Unary};
    use num_bigint::BigInt;

    /// Integers that exercise overflow and sign: the extremes, small numbers, powers of two and random bits.
    fn samples() -> Vec<i64> {
        let mut v = vec![0, 1, -1, 2, -2, 3, -3, 7, -12, 18, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1, 1 << 32, -(1 << 62)];
        v.extend((0..200).map(|i| random(i) as i64 >> (random(i + 1000) % 64)));
        v
    }

    #[test]
    fn gcd_lcm_power_residue() {
        // Hours, minutes and seconds; a base of 0 that takes what remains; and a quotient that leaves `i64`.
        assert_eq!((encode(&[24, 60, 60], &[3661, 86399]), encode(&[0, 10], &[123])), (Some(vec![1, 1, 1, 23, 59, 59]), Some(vec![12, 3])));
        assert_eq!((encode(&[-1], &[i64::MIN]), decode(&[24, 60, 60], &[1, 1, 1, 23, 59, 59])), (None, Some(vec![3661, 86399])));
        assert_eq!(decode(&[1, 1 << 62], &[2, 0]), None);
        let fit = |v: i128| i64::try_from(v).ok();
        for &x in &samples() {
            for &y in &samples() {
                let (a, b) = (x as i128, y as i128);
                let g = num_integer::Integer::gcd(&a, &b);
                assert_eq!(gcd(x, y), fit(g), "gcd({x}, {y})");
                assert_eq!(lcm(x, y), if g == 0 { Some(0) } else { fit(a / g * b) }, "lcm({x}, {y})");
                if x == 0 {
                    assert!(residue(0, y) == y && Divisor::new(0).is_none());
                    continue;
                }
                // The floored residue, with x's sign, and the quotient it leaves.
                let r = (b % a + a) % a;
                let d = Divisor::new(x).unwrap();
                assert!(residue(x, y) as i128 == r && d.residue(y) as i128 == r, "residue({x}, {y})");
                assert_eq!(d.div_mod(y), fit((b - r) / a).map(|q| (q, r as i64)), "div_mod({x}, {y})");
            }
            for y in -3..70 {
                let want = if y < 0 { (x.abs() == 1).then(|| if y % 2 == 0 { 1 } else { x }) } else { BigInt::from(x).pow(y as u32).try_into().ok() };
                assert_eq!(power(x, y), want, "power({x}, {y})");
            }
        }
        assert_eq!([power(0, 0), power(0, -1), power(-1, i64::MIN), power(-2, 63), power(2, 63)], [Some(1), None, Some(1), Some(i64::MIN), None]);
    }

    #[test]
    fn binomials() {
        // C(n, k) as the product of n - i for i below k, divided by k!, which holds for every integer n.
        let exact = |n: i64, k: i64| -> Option<i64> {
            let p = (0..k).fold(BigInt::from(1), |p, i| p * (n - i)) / (1..=k).fold(BigInt::from(1), |p, i| p * i);
            p.try_into().ok()
        };
        for n in -40..=70 {
            for k in 0..=75 {
                let c = exact(n, k);
                assert_eq!(binom(n, k), c, "binom({n}, {k})");
                // The float binomial coefficient of whole arguments is exact while the result times |n| + k fits the type's whole
                // numbers.
                let fits = |bits: u32| c.is_some_and(|c| (c.unsigned_abs() as u128) * (n.unsigned_abs() + k as u64) as u128 >> bits == 0);
                if fits(53) { assert_eq!(rustymath::binom(n as f64, k as f64), c.unwrap() as f64, "f64 binom({n}, {k})") }
                if fits(24) { assert_eq!(rustymath::binom(n as f32, k as f32), c.unwrap() as f32, "f32 binom({n}, {k})") }
            }
            assert_eq!(binom(n, -1), None);
        }
        assert_eq!(binom(i64::MIN, 1), Some(i64::MIN));
        assert_eq!(binom(i64::MAX, 1), Some(i64::MAX));
        assert_eq!(binom(i64::MAX, 2), None);
        assert_eq!(binom(-2, 1_000_000_000_000), Some(1_000_000_000_001));
        assert_eq!(binom(-1, i64::MAX), Some(-1));
    }

    #[test]
    fn slice_forms() {
        let (xs, ys) = (samples(), samples().into_iter().rev().collect::<Vec<_>>());
        // The integer forms ignore the tolerance.
        let t = rustymath::tolerant::Tolerance::DEFAULT;
        for d in [7, -12, 1, -1, 0, i64::MIN] {
            let mut out = Vec::new();
            assert!(Residue(t).many(Arg::One(d), Arg::Many(&ys), ys.len(), &mut out));
            assert_eq!(out, ys.iter().map(|&y| residue(d, y)).collect::<Vec<_>>());
        }
        let ys16: Vec<i16> = (i16::MIN..=i16::MAX).collect();
        for d in [7i16, -12, 1, -1, 0, 3, i16::MAX, i16::MIN] {
            let mut out = Vec::new();
            assert!(Residue(t).many(Arg::One(d), Arg::Many(&ys16), ys16.len(), &mut out));
            assert_eq!(out, ys16.iter().map(|&y| residue(d.into(), y.into()) as i16).collect::<Vec<_>>());
        }
        let ys16u: Vec<u16> = (0..=u16::MAX).collect();
        for d in [2u16, 3, 1000, u16::MAX] {
            let mut out = Vec::new();
            assert!(Residue(t).many(Arg::One(d), Arg::Many(&ys16u), ys16u.len(), &mut out));
            assert_eq!(out, ys16u.iter().map(|&y| y % d).collect::<Vec<_>>());
        }
        let mut out = vec![5];
        let ok = Gcd(t).many(Arg::Many(&xs), Arg::Many(&ys), xs.len(), &mut out);
        assert_eq!(ok, xs.iter().zip(&ys).all(|(&x, &y)| gcd(x, y).is_some()));
        assert_eq!(out[0], 5);
        assert_eq!(Binary::<i64>::checked(&Pow, 3, 4), Some(81));
        assert_eq!(Binary::<i64>::checked(&Pow, 3, 40), None);
        assert_eq!(Binom.one(5i64, 2), 10);
        assert_eq!(Lcm(t).one(-4i64, 6), -12);
    }

    #[test]
    fn other_widths() {
        let t = rustymath::tolerant::Tolerance::DEFAULT;
        // Every pair of `i8`s agrees with the `i64` functions where the result fits, and fails where it doesn't.
        for x in i8::MIN..=i8::MAX {
            for y in i8::MIN..=i8::MAX {
                let (a, b) = (i64::from(x), i64::from(y));
                let fit = |r: Option<i64>| r.and_then(|r| i8::try_from(r).ok());
                assert_eq!(Binary::<i8>::checked(&Gcd(t), x, y), fit(gcd(a, b)), "gcd({x}, {y})");
                assert_eq!(Binary::<i8>::checked(&Lcm(t), x, y), fit(lcm(a, b)), "lcm({x}, {y})");
                assert_eq!(Binary::<i8>::checked(&Pow, x, y), fit(power(a, b)), "power({x}, {y})");
                assert_eq!(i64::from(Residue(t).one(x, y)), residue(a, b), "residue({x}, {y})");
            }
        }
        // `u64`s, some beyond `i64::MAX`, against exact arithmetic.
        let big = [0, 1, 2, 3, 12, 18, 1 << 32, 1 << 63, (1 << 63) + 1, u64::MAX - 1, u64::MAX];
        for &x in &big {
            for &y in &big {
                let (a, b) = (u128::from(x), u128::from(y));
                let g = num_integer::Integer::gcd(&a, &b);
                assert_eq!(u128::from(Gcd(t).one(x, y)), g, "gcd({x}, {y})");
                assert_eq!(Binary::<u64>::checked(&Lcm(t), x, y), if g == 0 { Some(0) } else { u64::try_from(a / g * b).ok() }, "lcm({x}, {y})");
                assert_eq!(Residue(t).one(x, y), if x == 0 { y } else { y % x }, "residue({x}, {y})");
                assert_eq!((Equal(t).one(x, y), Less(t).one(x, y)), (x == y, x < y), "comparisons of {x} and {y}");
            }
            for y in 0..70u64 {
                assert_eq!(Binary::<u64>::checked(&Pow, x, y), BigInt::from(x).pow(y as u32).try_into().ok(), "power({x}, {y})");
            }
        }
        // Integers compare exactly, however close they are within the tolerance.
        let (n, m) = (1_000_000_000_000_000i64, 1_000_000_000_000_001i64);
        assert_eq!([Equal(t).one(n, m), NotEqual(t).one(n, m), LessEqual(t).one(m, n), GreaterEqual(t).one(m, n), Greater(t).one(m, n)], [false, true, false, true, true]);
        assert_eq!((Floor(t).one(m), Ceil(t).one(-m), Floor(t).one(true)), (m, -m, true));
        assert_eq!((Abs.one(i8::MIN), Abs.one(-5i16), Abs.one(7u8), Abs.one(false)), (i8::MIN, 5, 7, false));
        assert_eq!((Sign.one(-9i32), Sign.one(0i64), Sign.one(200u8)), (-1, 0, 1));
    }
}

mod number_theory {
    use rustymath::number_theory::*;
    use num_bigint::BigUint;

    const M61: u64 = (1 << 61) - 1;
    const M89: u128 = (1 << 89) - 1;
    const M127: u128 = (1 << 127) - 1;
    /// The largest prime below 2^64.
    const TOP: u64 = u64::MAX - 58;

    #[test]
    fn primality() {
        let mut primes = Primes::new(0).peekable();
        for n in 0..1u64 << 17 {
            let prime = primes.next_if_eq(&n).is_some();
            assert!(is_prime(&n) == prime && is_prime(&(n as u128)) == prime && is_prime(&BigUint::from(n)) == prime, "{n}");
        }
        assert!(is_prime(&M61) && is_prime(&TOP) && is_prime(&M89) && is_prime(&M127));
        assert!(is_prime(&((BigUint::from(1u8) << 521) - 1u8)));
        // Strong pseudoprimes to every prime base up to 23, 37 and 41.
        assert!(!is_prime(&3825123056546413051u64));
        assert!(!is_prime(&318665857834031151167461u128));
        assert!(!is_prime(&3317044064679887385961981u128));
        assert!(!is_prime(&(BigUint::from(M89) * BigUint::from(M127))));
        // Strong Lucas pseudoprimes with Selfridge's parameters, from OEIS A217255.
        for n in [5459u64, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519] { assert!(!is_prime(&n), "{n}"); }
    }

    #[test]
    fn factors() {
        assert!(factor(&1u64).is_empty());
        assert_eq!(factor(&u64::MAX), [(3, 1), (5, 1), (17, 1), (257, 1), (641, 1), (65537, 1), (6700417, 1)]);
        assert_eq!(factor(&((1u128 << 64) + 1)), [(274177, 1), (67280421310721, 1)]);
        assert_eq!(factor(&(((1u128 << 31) - 1) * M61 as u128)), [((1 << 31) - 1, 1), (M61 as u128, 1)]);
        let big = BigUint::from(1000003u64).pow(2) * 1000033u64 * BigUint::from(M127);
        assert_eq!(factor(&big), [(1000003u64.into(), 2), (1000033u64.into(), 1), (M127.into(), 1)]);
        assert_eq!(totient(&36u64), 12);
        assert_eq!(totient(&u64::MAX), 2 * 4 * 16 * 256 * 640 * 65536 * 6700416);
    }

    #[test]
    fn neighbours() {
        assert_eq!([next_prime(&0u64), next_prime(&2), next_prime(&13)], [Some(2), Some(3), Some(17)]);
        assert_eq!([prev_prime(&2u64), prev_prime(&3), prev_prime(&100)], [None, Some(2), Some(97)]);
        assert_eq!(next_prime(&(u64::MAX - 70)), Some(TOP));
        assert_eq!(next_prime(&TOP), None);
        assert_eq!(next_prime(&(TOP as u128)), Some((1 << 64) + 13));
        assert_eq!(prev_prime(&(1u128 << 64)), Some(TOP as u128));
    }

    #[test]
    fn counting() {
        let mut n = 1;
        for expected in [4, 25, 168, 1229, 9592, 78498, 664579] {
            n *= 10;
            assert_eq!(prime_count(n), expected);
        }
        // Counting and enumeration agree on each side of the first segments' ends.
        let (mut end, mut size) = (0, FIRST as u64);
        for _ in 0..4 {
            end += 30 * size;
            size = (2 * size).min(BYTES as u64);
            for n in [end - 1, end, end + 1] { assert_eq!(prime_count(n), Primes::new(0).take_while(|&p| p < n).count() as u64); }
        }
        let starts = [0, 3, 4, 6, 7, 29, 30, 98].map(|n| Primes::new(n).next().unwrap());
        assert_eq!(starts, [2, 3, 5, 7, 7, 29, 31, 101]);
        assert_eq!(Primes::new(0).nth(999_999), Some(15_485_863));
        let mut primes = Primes::new(0);
        assert_eq!([primes.nth(3), primes.nth(0), primes.nth(2)], [Some(7), Some(11), Some(19)]);
    }

    #[test]
    fn cancellation() {
        let stop = || Err("stopped");
        assert_eq!(factor_with(&((1u128 << 64) + 1), stop), Err("stopped"));
        assert_eq!(prime_count_with(1 << 20, stop), Err("stopped"));
        assert_eq!(Primes::new(0).nth_with(0, stop), Ok(Some(2)));
    }
}

mod polynomial {
    use rustymath::polynomial::*;
    use half::f16;
    use num_complex::Complex;

    #[test]
    fn polynomials() {
        // Mixed radix: each step multiplies by its own base, and the first base goes unused.
        assert_eq!(decode(&[9.0, 2.0, 2.0], &[1.0, 0.0, 1.5, 1.0, 1.0, 1.0]), [5.5, 7.0]);
        // (x - 1)(x - 2), with a trailing zero that infinity never meets.
        let c = [2.0, -3.0, 1.0, 0.0];
        let mut x = [0.0, 1.5, 3.0, f64::INFINITY];
        evaluate(&c, &mut x);
        assert_eq!(x, [2.0, -0.25, 2.0, f64::INFINITY]);
        let mut x = [1.0, 2.0];
        evaluate(&[0.0], &mut x);
        assert_eq!(x, [0.0; 2]);

        // Across block boundaries, at f32 and f16, against f64.
        let k = [0.5, -1.25, 0.75, 2.0, -0.125];
        let x: Vec<f64> = (0..1000).map(|i| i as f64 / 250.0 - 2.0).collect();
        let mut want = x.clone();
        evaluate(&k, &mut want);
        let mut got: Vec<f32> = x.iter().map(|&x| x as f32).collect();
        evaluate(&k.map(|k| k as f32), &mut got);
        assert!(got.iter().zip(&want).all(|(&g, &w)| (g as f64 - w).abs() <= 1e-5 * w.abs().max(1.0)));
        let mut half: Vec<f16> = x.iter().map(|&x| f16::from_f64(x)).collect();
        evaluate(&k.map(f16::from_f64), &mut half);
        assert!(half.iter().zip(&want).all(|(&g, &w)| (g.to_f64() - w).abs() <= 1e-2 * w.abs().max(1.0)));

        let r = roots(&c.map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r[0].im == 0.0 && r[1].im == 0.0 && (r[0].re - 1.0).abs() < 1e-12 && (r[1].re - 2.0).abs() < 1e-12);
        // x² + 1, and a polynomial with complex coefficients, (x - i)(x - 2).
        let r = roots(&[1.0f32, 0.0, 1.0].map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r.iter().all(|z| z.re.abs() < 1e-6 && (z.im.abs() - 1.0).abs() < 1e-6));
        let r = roots(&from_roots(Complex::new(1.0, 0.0), &[Complex::new(0.0, 1.0), Complex::new(2.0, 0.0)])).unwrap();
        assert!((r[0] - Complex::new(0.0, 1.0)).norm() < 1e-12 && (r[1] - Complex::new(2.0, 0.0)).norm() < 1e-12);
        assert!(roots::<f64>(&[Complex::new(3.0, 0.0)]).unwrap().is_empty());
        // 1 + ∞x has the root -1/∞, which is -0.
        let r = roots(&[1.0, f64::INFINITY].map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r.len() == 1 && r[0].re == 0.0 && r[0].re.is_sign_negative() && r[0].im == 0.0);

        assert_eq!(from_roots(1.0, &[1.0, 2.0]), [2.0, -3.0, 1.0]);
        assert_eq!(derivative(&[2.0, -3.0, 1.0]), [-3.0, 2.0]);
        let f = fit(&[0.0f64, 1.0, 2.0, 3.0, 4.0], &[2.0, 0.0, 0.0, 2.0, 6.0], 2);
        assert!(f.iter().zip([2.0, -3.0, 1.0]).all(|(&a, b)| (a - b).abs() < 1e-12));
    }
}

mod random {
    use rustymath::random::*;
    use rustymath::Arg;
    use crate::support::seeded;

    #[test]
    fn rolls_and_deals() {
        let mut rng = seeded(1);
        let mut out = vec![9];
        roll(Arg::One(6), 60_000, &mut rng, &mut out);
        assert_eq!(out[0], 9);
        let counts = (0..6).map(|k| out[1..].iter().filter(|&&r| r == k).count());
        assert!(counts.into_iter().all(|c| (9_000..11_000).contains(&c)));
        let bounds: Vec<u64> = (1..=1000).collect();
        let mut rolls = Vec::new();
        roll(Arg::Many(&bounds), 1000, &mut rng, &mut rolls);
        assert!(rolls.iter().zip(&bounds).all(|(r, b)| r < b));
        let mut dealt = Vec::new();
        deal(50, 60, &mut rng, &mut dealt);
        dealt.sort_unstable();
        dealt.dedup();
        assert!(dealt.len() == 50 && dealt.iter().all(|&d| d < 60));
    }
}
