pub mod support;

mod arith {
    use rustymath::{Abs, Add, Binary, Multiply, Sign, Subtract, Unary};

    /// The integer forms of the arithmetic objects at type `$t`, on every pair of edge values, against the exact result in `i128`. Each form fails where that doesn't fit `$t`.
    macro_rules! check {
        ($($t:ty),+) => {$({
            let edges: Vec<$t> = [<$t>::MIN, <$t>::MAX, 0, 1, 2, 3, 7].into_iter().chain([-1i128, -2, -3].map(|v| v as $t))
                .chain([<$t>::MIN / 2, <$t>::MAX / 2, <$t>::MAX - 1, <$t>::MIN.wrapping_add(1)]).collect();
            let fits = |r: i128| <$t>::try_from(r).ok();
            for &x in &edges {
                for &y in &edges {
                    let (a, b) = (x as i128, y as i128);
                    assert_eq!((Add.checked(x, y), Subtract.checked(x, y), Multiply.checked(x, y)), (fits(a + b), fits(a - b), fits(a * b)), "{x} and {y}");
                }
                assert_eq!(Unary::<$t>::checked(&Abs, x), fits((x as i128).abs()), "|{x}|");
                assert_eq!(Sign.one(x) as i128, (x as i128).signum(), "sign {x}");
            }
        })+};
    }

    #[test]
    fn integer_forms_match_exact_arithmetic() { check!(u8, i16, i32, i64); }
}

mod convert {
    use half::f16;
    use rustymath::convert::*;

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
        assert_eq!(
            (float::<f64, u8>(255), float::<f32, u16>(65535), float::<f64, u32>(u32::MAX), float::<f16, i32>(-3)),
            (255.0, 65535.0, 4294967295.0, f16::from_f32(-3.0))
        );
        let mut counts = Vec::<f64>::new();
        count(&mut counts, 0, false, 21);
        count(&mut counts, 20, true, 21);
        assert_eq!(counts, (0..21).chain((0..21).rev()).map(|i| i as f64).collect::<Vec<_>>());
        let mut narrow = Vec::<f32>::new();
        count(&mut narrow, (1 << 24) - 3, false, 8);
        count(&mut narrow, 1 << 31, true, 3);
        let expected = ((1i64 << 24) - 3..(1 << 24) + 5).chain(((1 << 31) - 2..=1 << 31).rev());
        assert_eq!(narrow, expected.map(|i| i as f64 as f32).collect::<Vec<_>>());
    }
}

mod distribution {
    use crate::support::seeded;
    use rand::RngExt;
    use rustymath::distribution::*;
    use rustymath::{Arg, Unary};

    /// One of each distribution, with each sampling method's case: binomials that draw by inversion, by BTRD and reflected, and Poisson distributions that draw by multiplying and by PTRS.
    fn cases() -> Vec<Distribution> {
        let d = Distribution::binomial;
        [
            Distribution::normal(1.0, 2.0), Distribution::uniform(-1.0, 3.0), Distribution::beta(0.5, 2.0), Distribution::beta(3.0, 4.0),
            Distribution::cauchy(0.0, 1.0), Distribution::chi_squared(3.0), Distribution::exponential(2.0), Distribution::fisher(4.0, 7.0),
            Distribution::gamma(0.5, 2.0), Distribution::gamma(5.0, 1.0), Distribution::inverse_gamma(3.0, 2.0), Distribution::laplace(1.0, 2.0),
            Distribution::logistic(0.0, 1.0), Distribution::log_normal(0.0, 0.5), Distribution::student_t(3.0), Distribution::weibull(1.5, 2.0), d(10, 0.3),
            d(1000, 0.4), d(100, 0.9), Distribution::poisson(3.0), Distribution::poisson(50.0),
        ].into_iter().map(Result::unwrap).collect()
    }

    #[test]
    fn quantiles_and_pdfs_match_cdfs() {
        for d in cases() {
            for p in [0.001, 0.1, 0.5, 0.9, 0.999] {
                let x = d.quantile(p);
                if d.is_discrete() { assert!(d.cdf(x) >= p * (1.0 - 1e-14) && d.cdf(x - 1.0) < p, "{d:?} at {p}: {x}"); }
                else { assert!((d.cdf(x) - p).abs() < 1e-12, "{d:?} at {p}: cdf({x}) is {}", d.cdf(x)); }
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

    /// `f`'s slice form against `scalar` for each item of `xs` at f64, and against `f`'s own scalar form at f32, which needn't compute in f64. Gives whether the slice form reported every item as succeeding.
    fn matches_scalar(f: impl Unary<f64, Out = f64> + Unary<f32, Out = f32>, scalar: impl Fn(f64) -> f64, xs: &[f64]) -> bool {
        let same = |a: f64, b: f64| a == b || (a.is_nan() && b.is_nan());
        let (mut out, mut out32) = (Vec::new(), Vec::new());
        let ok = f.many(Arg::Many(xs), xs.len(), &mut out);
        let xs32: Vec<f32> = xs.iter().map(|&x| x as f32).collect();
        assert_eq!(f.many(Arg::Many(&xs32), xs32.len(), &mut out32), ok);
        assert!(out.iter().zip(xs).all(|(&r, &x)| same(r, scalar(x))), "{out:?}");
        assert!(out32.iter().zip(&xs32).all(|(&r, &x)| same(r.into(), Unary::<f32>::one(&f, x).into())), "{out32:?}");
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
    use num_rational::BigRational;
    use rustymath::exact::*;

    fn q(n: i64, d: i64) -> BigRational { BigRational::new(n.into(), d.into()) }

    #[test]
    fn exact_functions() -> Result<(), rustymath::error::Error> {
        assert_eq!((residue(&q(3, 1), &q(-7, 1)), residue(&q(1, 2), &q(5, 3)), residue(&q(0, 1), &q(5, 3))), (q(2, 1), q(1, 6), q(5, 3)));
        assert_eq!((gcd(&q(1, 2), &q(1, 3)), gcd(&q(-4, 1), &q(6, 1)), gcd(&q(0, 1), &q(0, 1))), (q(1, 6), q(2, 1), q(0, 1)));
        assert_eq!((lcm(&q(1, 2), &q(1, 3)), lcm(&q(-4, 1), &q(6, 1)), lcm(&q(0, 1), &q(5, 1))), (q(1, 1), q(-12, 1), q(0, 1)));
        // C(5, 2), C(5, 7), C(1/2, 2) = -1/8, C(-1, 2) = 1, C(-3, -5) = (-1)^2 C(4, 2) = 6, C(-3, -1) = 0 and C(1/2, -1) = 0.
        let c: Vec<BigRational> = [(q(5, 1), 2), (q(5, 1), 7), (q(1, 2), 2), (q(-1, 1), 2), (q(-3, 1), -5), (q(-3, 1), -1), (q(1, 2), -1)].iter()
            .map(|(n, k)| binom(n, *k)).collect::<Result<_, _>>()?;
        assert_eq!(c, [q(10, 1), q(0, 1), q(-1, 8), q(1, 1), q(6, 1), q(0, 1), q(0, 1)]);
        Ok(())
    }
}

mod integer {
    use crate::support::random;
    use num_bigint::BigInt;
    use rustymath::integer::*;
    use rustymath::{
        Abs, Arg, Binary, Binom, Pow, Sign, Unary, tolerant::{Ceil, Equal, Floor, Gcd, Greater, GreaterEqual, Lcm, LessEqual, NotEqual, Residue}
    };

    /// Integers that exercise overflow and sign: the extremes, small numbers, powers of two and random bits.
    fn samples() -> Vec<i64> {
        let mut v = vec![0, 1, -1, 2, -2, 3, -3, 7, -12, 18, i64::MAX, i64::MIN, i64::MAX - 1, i64::MIN + 1, 1 << 32, -(1 << 62)];
        v.extend((0..200).map(|i| random(i) as i64 >> (random(i + 1000) % 64)));
        v
    }

    #[test]
    fn gcd_lcm_power_residue() {
        assert_eq!(sum_bounds([-2, 3], 4, Some(10), false), Some([2, 22]));
        assert_eq!((sum_bounds([-2, 3], 4, None, true), sum_bounds([-2, 3], 4, Some(10), true)), (Some([-11, 9]), Some([-2, 18])));
        assert_eq!((product_bounds([-3, 2], [-5, 7]), product_bounds([i64::MAX, i64::MAX], [2, 2])), (Some([-21, 15]), None));
        assert_eq!(sum_bounds([i64::MIN, i64::MAX], 2, None, false), None);
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
                if x == 0 { assert!(residue(0, y) == y && Divisor::new(0).is_none()); continue; }
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
                // The float binomial coefficient of whole arguments is exact while the result times |n| + k fits the type's whole numbers.
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
        // A whole k with a non-whole n takes the falling factorial, which stays accurate beside the polynomial's roots, where Γ(n + 1)/(Γ(k + 1) Γ(n - k + 1)) cancels: C(n, k) and its slope in n against mpmath, within the documented 2k ulps.
        let near = |got: f64, want: f64, k: f64| assert!((got - want).abs() <= 2.0 * k * f64::EPSILON * want.abs(), "{got} against {want}");
        for (n, k, c, slope) in [(0.999999999, 2.0, -4.9999998535903429657e-10, 0.49999999900000002828), (3.000000001, 5.0, -5.0000004178685223475e-11, -0.050000000083333340103),
                                 (-21.20622351098135, 64.0, 14366238820115916784.0, -20237195289673071829.0), (4.001, 20.0, -1.2883957254018030871e-8, -0.000012868008206733345107)] {
            near(rustymath::binom(n, k), c, k);
            near(rustymath::binom_dn(n, k), slope, k);
        }
        // Other real arguments take the Γ ratio in double words, which keeps C(n, k) within 4 ulps of mpmath for large n, negative arguments and a whole k above 64, and its slope in n where both ψ arguments are from 10 up.
        let close = |got: f64, want: f64| assert!((got - want).abs() <= 4.0 * f64::EPSILON * want.abs(), "{got} against {want}");
        for (n, k, c) in [(1.3e16, 2.5, 5.798052885632725e39), (1e36, 1.64, 7.439939859224348e58), (-35.249400585757215, -16.771288022576677, -6.434749405414238e-12),
                          (0.8465836218811786, 3.865135317059345, -0.0018154754878552631), (92.57318691468267, 117.0, 1.3012421741507206e-27), (7.25, 1.5, 13.909723913795338)] {
            close(rustymath::binom(n, k), c);
        }
        for (n, k, slope) in [(1.3e16, 2.5, 1.1150101703139856e24), (1e36, 1.64, 1.220150136912793e23), (92.57318691468267, 117.0, 8.106595581093816e-28)] {
            close(rustymath::binom_dn(n, k), slope);
        }
        // A result the type holds stays finite where a step's product would overflow, in the whole route at f64 and f32, and in the falling factorial and its slopes.
        close(rustymath::binom(1028.0, 514.0), 7.156051054877897e307);
        let c32 = rustymath::binom(130.0f32, 65.0f32);
        assert!((c32 - 9.50676258279607e37).abs() <= 4.0 * f32::EPSILON * 9.50676258279607e37, "{c32}");
        near(rustymath::binom(1_550_000.5, 64.0), 1.1946917951960594e307, 64.0);
        near(rustymath::binom_dn(1_550_000.5, 64.0), 4.933019621902672e302, 64.0);
        near(rustymath::binom_dk(1_550_000.5, 64.0), 1.205091539975857e308, 64.0);
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
        // Every pair of `u8`s agrees with the `i64` functions where the result fits, and fails where it doesn't.
        for x in 0..=u8::MAX {
            for y in 0..=u8::MAX {
                let (a, b) = (i64::from(x), i64::from(y));
                let fit = |r: Option<i64>| r.and_then(|r| u8::try_from(r).ok());
                assert_eq!(Binary::<u8>::checked(&Gcd(t), x, y), fit(gcd(a, b)), "gcd({x}, {y})");
                assert_eq!(Binary::<u8>::checked(&Lcm(t), x, y), fit(lcm(a, b)), "lcm({x}, {y})");
                assert_eq!(Binary::<u8>::checked(&Pow, x, y), fit(power(a, b)), "power({x}, {y})");
                assert_eq!(i64::from(Residue(t).one(x, y)), residue(a, b), "residue({x}, {y})");
            }
        }
        // Integers compare exactly, however close they are within the tolerance.
        let (n, m) = (1_000_000_000_000_000i64, 1_000_000_000_000_001i64);
        assert_eq!(
            [Equal(t).one(n, m), NotEqual(t).one(n, m), LessEqual(t).one(m, n), GreaterEqual(t).one(m, n), Greater(t).one(m, n)],
            [false, true, false, true, true]
        );
        assert_eq!((Floor(t).one(m), Ceil(t).one(-m), Floor(t).one(true)), (m, -m, true));
        assert_eq!((Unary::<i16>::checked(&Abs, i16::MIN), Abs.one(-5i16), Abs.one(7u8), Abs.one(false)), (None, 5, 7, false));
        assert_eq!((Sign.one(-9i32), Sign.one(0i64), Sign.one(200u8)), (-1, 0, 1));
    }
}

mod number_theory {
    use num_bigint::BigUint;
    use rustymath::error::{Error, Kind};
    use rustymath::number_theory::*;
    use std::cell::Cell;

    const M61: u64 = (1 << 61) - 1;
    const M89: u128 = (1 << 89) - 1;
    const M127: u128 = (1 << 127) - 1;
    /// The largest prime below 2^64.
    const TOP: u64 = u64::MAX - 58;

    #[test]
    fn primality() -> Result<(), Error> {
        let mut primes = Primes::new(0).peekable();
        for n in 0..1u64 << 17 {
            let prime = primes.next_if_eq(&Ok(n)).is_some();
            assert!(is_prime(&n)? == prime && is_prime(&(n as u128))? == prime && is_prime(&BigUint::from(n))? == prime, "{n}");
        }
        assert!(is_prime(&M61)? && is_prime(&TOP)? && is_prime(&M89)? && is_prime(&M127)?);
        assert!(is_prime(&((BigUint::from(1u8) << 521) - 1u8))?);
        // Strong pseudoprimes to every prime base up to 23, 37 and 41.
        assert!(!is_prime(&3825123056546413051u64)?);
        assert!(!is_prime(&318665857834031151167461u128)?);
        assert!(!is_prime(&3317044064679887385961981u128)?);
        assert!(!is_prime(&(BigUint::from(M89) * BigUint::from(M127)))?);
        // Strong Lucas pseudoprimes with Selfridge's parameters, from OEIS A217255.
        for n in [5459u64, 5777, 10877, 16109, 18971, 22499, 24569, 25199, 40309, 58519] { assert!(!is_prime(&n)?, "{n}"); }
        Ok(())
    }

    #[test]
    fn factors() -> Result<(), Error> {
        assert!(factor(&1u64)?.is_empty());
        assert_eq!(factor(&u64::MAX)?, [(3, 1), (5, 1), (17, 1), (257, 1), (641, 1), (65537, 1), (6700417, 1)]);
        assert_eq!(factor(&((1u128 << 64) + 1))?, [(274177, 1), (67280421310721, 1)]);
        assert_eq!(factor(&(((1u128 << 31) - 1) * M61 as u128))?, [((1 << 31) - 1, 1), (M61 as u128, 1)]);
        let big = BigUint::from(1000003u64).pow(2) * 1000033u64 * BigUint::from(M127);
        assert_eq!(factor(&big)?, [(1000003u64.into(), 2), (1000033u64.into(), 1), (M127.into(), 1)]);
        assert_eq!(totient(&36u64)?, 12);
        assert_eq!(totient(&u64::MAX)?, 2 * 4 * 16 * 256 * 640 * 65536 * 6700416);
        Ok(())
    }

    #[test]
    fn neighbours() -> Result<(), Error> {
        assert_eq!([next_prime(&0u64)?, next_prime(&2)?, next_prime(&13)?], [Some(2), Some(3), Some(17)]);
        assert_eq!([prev_prime(&2u64)?, prev_prime(&3)?, prev_prime(&100)?], [None, Some(2), Some(97)]);
        assert_eq!(next_prime(&(u64::MAX - 70))?, Some(TOP));
        assert_eq!(next_prime(&TOP)?, None);
        assert_eq!(next_prime(&(TOP as u128))?, Some((1 << 64) + 13));
        assert_eq!(prev_prime(&(1u128 << 64))?, Some(TOP as u128));
        Ok(())
    }

    #[test]
    fn counting() -> Result<(), Error> {
        let mut n = 1;
        for expected in [4, 25, 168, 1229, 9592, 78498, 664579] {
            n *= 10;
            assert_eq!(prime_count(n)?, expected);
        }
        // Counting and enumeration agree on each side of the first segments' ends.
        let (mut end, mut size) = (0, FIRST as u64);
        for _ in 0..4 {
            end += 30 * size;
            size = (2 * size).min(BYTES as u64);
            for n in [end - 1, end, end + 1] { assert_eq!(prime_count(n)?, Primes::new(0).map(Result::unwrap).take_while(|&p| p < n).count() as u64); }
        }
        assert_eq!([0, 3, 4, 6, 7, 29, 30, 98].map(|n| Primes::new(n).next()), [2, 3, 5, 7, 7, 29, 31, 101].map(|p| Some(Ok(p))));
        assert_eq!(Primes::new(0).nth(999_999), Some(Ok(15_485_863)));
        let mut primes = Primes::new(0);
        assert_eq!([primes.nth(3), primes.nth(0), primes.nth(2)], [7, 11, 19].map(|p| Some(Ok(p))));
        Ok(())
    }

    #[test]
    fn cancellation() {
        thread_local! { static STOP: Cell<bool> = const { Cell::new(false) }; }
        rustymath::interrupt::install(|| if STOP.get() { Err(Error::new(Kind::Interrupt, "stopped")) } else { Ok(()) });
        STOP.set(true);
        assert_eq!(factor(&((1u128 << 64) + 1)).map_err(|e| e.kind), Err(Kind::Interrupt));
        assert_eq!(prime_count(1 << 20).map_err(|e| e.kind), Err(Kind::Interrupt));
        assert_eq!(Primes::new(0).nth(0), Some(Ok(2)));
        STOP.set(false);
    }
}

mod operation {
    use rustymath::agreement::Agreement;
    use rustymath::array::ArrayView;
    use rustymath::buffer::{Buffer, ComplexWidth, DType, FloatWidth, Width};
    use rustymath::layout::Layout;
    use rustymath::{kernels, reduce};
    #[test]
    fn kernels_follow_forms() {
        let (b, i16, i64, f16, f32, f64, c64) = (DType::Boolean, DType::Integer(Width::I16), DType::Integer(Width::I64), DType::Float(FloatWidth::F16),
            DType::Float(FloatWidth::F32), DType::Float(FloatWidth::F64), DType::Complex(ComplexWidth::F32));
        // `Exp` has real and complex forms, so Booleans and integers compute in the narrowest float that holds them exactly, but at least `f32`, and characters have no kernel.
        let exp = kernels!(Unary rustymath::Exp);
        assert_eq!([b, i16, i64, f16, f32, c64].map(|d| exp.computes(d)), [Some(f32), Some(f32), Some(f64), Some(f16), Some(f32), Some(c64)]);
        assert_eq!(exp.computes(DType::Character), None);
        // `Add` has integer forms, so Booleans compute as `i64`.
        assert_eq!(kernels!(Binary rustymath::Add).computes(b), Some(i64));
        // `Cis` has `f32` and `f64` forms only, so `f16` widens to `f32`.
        assert_eq!(kernels!(Unary rustymath::Cis).computes(f16), Some(f32));
        // A value names its object too. `Equal` has Boolean, integer, real and complex forms, so Booleans compute as themselves. Characters have no kernel of their own: they compare as code points (`characters::binary`).
        let equal = kernels!(Binary of rustymath::tolerant::Equal(rustymath::tolerant::Tolerance::DEFAULT));
        assert_eq!([b, DType::Character].map(|d| equal.computes(d)), [Some(b), None]);
        // A sum of a narrow integer type has its own fold, whose totals are `i64`.
        let sum = kernels!(Fold reduce::Sum);
        let (totals, ok) = sum.run(&reduce::Sum, Buffer::Integers(rustymath::buffer::IntBuf::I16(vec![1, 2, 3, 4])).view(), [2, 2, 1], None).unwrap();
        assert!(ok && matches!(totals, Buffer::Integers(rustymath::buffer::IntBuf::I64(ref t)) if *t == [3, 7]));
        // An integer argument to `Exp` casts to `f64`, and broadcasting pairs a scalar with every item.
        let x = Buffer::Integers(rustymath::buffer::IntBuf::I64(vec![0, 1]));
        let (r, _) = exp.unary(&rustymath::Exp, ArrayView::compact(&Layout::from([2]), x.view())).unwrap();
        assert!(matches!(r, Buffer::Floats(rustymath::buffer::FloatBuf::F64(ref v)) if v[0] == 1.0 && (v[1] - std::f64::consts::E).abs() < 1e-15));
        let (two, row) = (Layout::from([1usize].as_slice()), Layout::from([3usize].as_slice()));
        let agreement = Agreement::new(&two, &row).unwrap();
        let (s, ok) = kernels!(Binary rustymath::Add).binary(&rustymath::Add, ArrayView::compact(&two, rustymath::buffer::View::Booleans(&[true])), ArrayView::compact(&row, rustymath::buffer::View::Booleans(&[true, false, true])), &agreement).unwrap();
        assert!(ok && matches!(s, Buffer::Integers(rustymath::buffer::IntBuf::I64(ref v)) if *v == [2, 1, 2]));
        // An integer result that doesn't fit fails.
        let big = Buffer::Integers(rustymath::buffer::IntBuf::I64(vec![i64::MAX]));
        let big = ArrayView::compact(&two, big.view());
        assert!(!kernels!(Binary rustymath::Add).binary(&rustymath::Add, big, big, &Agreement::new(&two, &two).unwrap()).unwrap().1);
        let data = Buffer::Integers(rustymath::buffer::IntBuf::I16(vec![10, 2, 3]));
        let (scan, ok) = kernels!(LeftScan rustymath::Subtract).run(&rustymath::Subtract, data.view(), [1, 3, 1], None).unwrap();
        assert!(ok && matches!(scan, Buffer::Integers(rustymath::buffer::IntBuf::I16(ref v)) if *v == [10, 8, 5]));
        let seed = Buffer::Integers(rustymath::buffer::IntBuf::I64(vec![100]));
        let (scan, ok) = kernels!(Scan reduce::Sum).run(&reduce::Sum, data.view(), [1, 3, 1], Some(seed.view())).unwrap();
        assert!(ok && matches!(scan, Buffer::Integers(rustymath::buffer::IntBuf::I64(ref v)) if *v == [110, 112, 115]));
        // Ordered folds keep a local total for consecutive lanes, and read a differently typed tail in the computing dtype.
        let right = kernels!(RightFold rustymath::Subtract);
        let (fold, ok) = right.run(&rustymath::Subtract, data.view(), [1, 3, 1], Some(seed.view())).unwrap();
        assert!(ok && matches!(fold, Buffer::Integers(rustymath::buffer::IntBuf::I16(ref v)) if *v == [-89]));
        // A failed scalar step retains its actual value and the healthy lanes in the raw kernel result.
        let (fold, ok) = kernels!(RightFold rustymath::Pow).run(&rustymath::Pow, Buffer::Floats(rustymath::buffer::FloatBuf::F64(vec![-1., 1., 0.5, 2.])).view(), [1, 2, 2], None).unwrap();
        assert!(!ok && matches!(fold, Buffer::Floats(rustymath::buffer::FloatBuf::F64(ref v)) if v[0].is_nan() && v[1] == 1.));
    }

    #[test]
    fn logical_views_compose_prepare_and_run() {
        use rustymath::{agreement::Mapping, buffer::Element, gather::Steps};
        let (shape, data) = (Layout::from([3, 2]), [0., 1., 2., 3., 4., 5.]);
        let transpose = Mapping::Steps { base: 0, steps: vec![Steps::along(3, 1), Steps::along(2, 3)] };
        let x = ArrayView::new(&shape, f64::view(&data), &transpose);
        assert!(x.contiguous().is_none());
        assert_eq!(x.items::<f64>().unwrap().as_ref(), &[0., 3., 1., 4., 2., 5.]);
        assert_eq!(transpose.affine(shape.shape()), Some((0, vec![1, 3])));
        let reversed = Mapping::Steps { base: 0, steps: vec![Steps::Stride { start: 4, len: 3, step: -2 }, Steps::along(2, 1)] };
        let composed = transpose.compose(&reversed, shape.shape(), shape.shape()).unwrap();
        assert_eq!(ArrayView::new(&shape, f64::view(&data), &composed).items::<f64>().unwrap().as_ref(), &[2., 5., 1., 4., 0., 3.]);
        let compact = ArrayView::compact(&shape, f64::view(&data));
        let (sum, ok) = kernels!(Binary rustymath::Add).binary(&rustymath::Add, x, compact, &Agreement::new(&shape, &shape).unwrap()).unwrap();
        assert!(ok); assert_eq!(f64::of(sum.view()).unwrap(), &[0., 4., 3., 7., 6., 10.]);
        let (one, two) = (Agreement::new(&shape, &Layout::EMPTY).unwrap(), Agreement::new(&shape, &Layout::EMPTY).unwrap());
        let args = [x, ArrayView::compact(&Layout::EMPTY, f64::view(&[1.])), ArrayView::compact(&Layout::EMPTY, f64::view(&[4.]))];
        let (clipped, ok) = kernels!(Ternary rustymath::Clip).ternary(&rustymath::Clip, args, &one, &two).unwrap();
        assert!(ok); assert_eq!(f64::of(clipped.view()).unwrap(), &[1., 3., 1., 4., 2., 4.]);
        // A table of positions reads through its offsets before broadcasting.
        let row = Layout::from([3]);
        let positions = Mapping::Steps { base: 0, steps: vec![Steps::table(vec![5, 0, 3])] };
        let row_view = ArrayView::new(&row, f64::view(&data), &positions);
        let (sum, _) = kernels!(Binary rustymath::Add).binary(&rustymath::Add, row_view, compact, &Agreement::new(&row, &shape).unwrap()).unwrap();
        assert_eq!(f64::of(sum.view()).unwrap(), &[5., 6., 2., 3., 7., 8.]);
        // Checked casts inspect only selected items; a compact subrange borrows its exact storage.
        let sub = Mapping::Steps { base: 1, steps: vec![Steps::along(3, 1)] };
        let selected = ArrayView::new(&row, f64::view(&[f64::NAN, 1., 2., 3., f64::INFINITY]), &sub);
        assert_eq!(selected.items::<i16>().unwrap().as_ref(), &[1, 2, 3]);
        assert_eq!(selected.contiguous().unwrap().as_ptr(), selected.storage().as_ptr().wrapping_add(8));
        let strided = Mapping::Steps { base: 0, steps: vec![Steps::Stride { start: 4, len: 3, step: -2 }] };
        assert_eq!(ArrayView::new(&row, f64::view(&[1., f64::NAN, 2., f64::INFINITY, 3.]), &strided).items::<i16>().unwrap().as_ref(), &[3, 2, 1]);
        let prefix = Layout::from([2]);
        let prefix = ArrayView::new(&prefix, f64::view(&[1., f64::NAN, 2., f64::INFINITY, 3.]), &strided);
        assert_eq!(prefix.items::<f64>().unwrap().as_ref(), &[3., 2.]);
        assert_eq!(prefix.items::<i16>().unwrap().as_ref(), &[3, 2]);
        let empty = Layout::from([0, 3]);
        assert!(ArrayView::new(&empty, f64::view(&data), &transpose).items::<f64>().unwrap().is_empty());
        assert!(Mapping::Linear(2).affine(&[3]).is_none());
        let table = |items| Mapping::Steps { base: 2, steps: vec![Steps::table(items)] };
        assert_eq!(table(vec![3]).affine(&[1]), Some((5, vec![0])));
        assert_eq!(table(vec![3, Steps::FILL]).affine(&[0]), Some((0, vec![0])));
        assert!(table(vec![3, Steps::FILL]).affine(&[2]).is_none() && table(vec![Steps::FILL]).affine(&[1]).is_none());
    }

    #[test]
    fn owned_views_share_detach_and_release_their_base() {
        use rustymath::{array::Array, buffer::Element, structure::{self, Selector}};
        let mut base = Array::new([2, 3].into(), i64::buffer((0..6).collect())).unwrap();
        let transposed = base.selected(structure::transpose(base.layout(), &[1, 0]).unwrap()).unwrap();
        assert_eq!(base.view().storage().as_ptr(), transposed.view().storage().as_ptr());
        assert_eq!(transposed.view().items::<i64>().unwrap().as_ref(), &[0, 3, 1, 4, 2, 5]);
        let reshaped = base.reshape([6].into()).unwrap();
        assert_eq!(base.view().storage().as_ptr(), reshaped.view().storage().as_ptr());
        let mut tail = reshaped.selected(structure::select_axis(reshaped.layout(), 0, Selector::Range { start: 2, len: 3, step: 1 }).unwrap()).unwrap();
        assert_eq!(tail.view().storage().len(), 6);
        let compact = tail.compact();
        assert_eq!(compact.view().storage().len(), 3);
        assert_eq!(compact.view().items::<i64>().unwrap().as_ref(), &[2, 3, 4]);
        let (_, moved) = tail.clone().into_compact();
        assert_eq!(moved.len(), 3);
        i64::of_mut(base.make_mut()).unwrap()[2] = 20;
        assert_eq!(tail.view().items::<i64>().unwrap().as_ref(), &[2, 3, 4]);
        let scalar = Array::new(Layout::EMPTY, i64::buffer(vec![99])).unwrap();
        tail.assign(&structure::select_axis(tail.layout(), 0, Selector::At(1)).unwrap(), &scalar).unwrap();
        assert_eq!(tail.view().items::<i64>().unwrap().as_ref(), &[2, 99, 4]);
        assert_eq!(reshaped.view().items::<i64>().unwrap().as_ref(), &[0, 1, 2, 3, 4, 5]);
        assert_eq!(Array::differences(transposed.view(), 1).unwrap().view().items::<i64>().unwrap().as_ref(), &[3, 3, 3]);
        assert_eq!(transposed.reshape([6].into()).unwrap().view().items::<i64>().unwrap().as_ref(), &[0, 3, 1, 4, 2, 5]);
    }

    /// `binary` of characters and `x` or `y`, laid out as vectors.
    fn characters<K>(k: K, x: &Buffer, y: &Buffer) -> Option<(Buffer, bool)>
    where K: rustymath::characters::Characters + rustymath::Binary<i32, Out: rustymath::buffer::Element> + rustymath::Binary<i64, Out: rustymath::buffer::Element> + Sync {
        let (lx, ly) = (Layout::from([x.len()]), Layout::from([y.len()]));
        let agreement = Agreement::new(&lx, &ly).unwrap();
        rustymath::characters::binary(&k, ArrayView::compact(&lx, x.view()), ArrayView::compact(&ly, y.view()), agreement.layout.shape(), &agreement)
    }

    #[test]
    fn characters_compute_as_code_points() {
        use rustymath::{Add, Subtract, buffer::{Element, IntBuf}, reduce::{Maximum, Minimum}, tolerant::{Less, Tolerance}};
        let text = |s: &str| Buffer::Characters(s.chars().collect());
        let (ab, ba, ints) = (text("ab"), text("ba"), Buffer::Integers(IntBuf::I64(vec![1, 2])));
        let run = |result: Option<(Buffer, bool)>| { let (b, ok) = result.unwrap(); assert!(ok); b };
        // Characters order by code point, and the maximum gives a character.
        assert_eq!(bool::vec(run(characters(Less(Tolerance::DEFAULT), &ab, &ba))), Some(vec![true, false]));
        assert_eq!(char::vec(run(characters(Maximum, &ab, &ba))), Some(vec!['b', 'b']));
        // A character minus a character is an integer, and an integer offsets a character on either side of `+`.
        assert_eq!(i32::vec(run(characters(Subtract, &ba, &ab))), Some(vec![1, -1]));
        assert_eq!(char::vec(run(characters(Add, &ab, &ints))), Some(vec!['b', 'd']));
        assert_eq!(char::vec(run(characters(Add, &ints, &ab))), Some(vec!['b', 'd']));
        assert!(characters(Add, &ab, &ba).is_none() && characters(Subtract, &ints, &ab).is_none());
        // Numbers order before characters, whatever the items.
        assert_eq!(bool::vec(run(characters(Less(Tolerance::DEFAULT), &ints, &ab))), Some(vec![true, true]));
        assert_eq!(char::vec(run(characters(Maximum, &ints, &ab))), Some(vec!['a', 'b']));
        assert_eq!(i64::vec(run(characters(Minimum, &ab, &ints))), Some(vec![1, 2]));
        // A result past the last code point fails.
        assert!(!characters(Add, &text("\u{10FFFF}"), &Buffer::Integers(IntBuf::I64(vec![1]))).unwrap().1);
    }
}

mod optimize {
    use rustymath::optimize::*;
    use std::convert::Infallible;

    /// Rosenbrock's function, whose curved valley has its minimum of 0 at (1, 1).
    fn rosenbrock(x: &[f64]) -> Result<f64, Infallible> { Ok((1.0 - x[0]).powi(2) + 100.0 * (x[1] - x[0] * x[0]).powi(2)) }

    fn rosenbrock_gradient(x: &[f64], g: &mut [f64]) -> Result<(), Infallible> {
        (g[0], g[1]) = (-2.0 * (1.0 - x[0]) - 400.0 * x[0] * (x[1] - x[0] * x[0]), 200.0 * (x[1] - x[0] * x[0]));
        Ok(())
    }

    /// The largest distance of `x` from `to` in any variable.
    fn off(x: &[f64], to: &[f64]) -> f64 { x.iter().zip(to).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) }

    #[test]
    fn methods_find_minimums() {
        let (lower, upper) = ([-2.0, -2.0], [2.0, 2.0]);
        let boxed = Some(Bounds { lower: &lower, upper: &upper });
        for (method, bounds) in
            [(Method::NelderMead, None), (Method::NelderMead, boxed), (Method::Powell, None), (Method::Powell, boxed), (Method::Evolution, boxed)] {
                let found = minimize(&mut rosenbrock, &[-1.2, 1.0], &Options { method: Some(method), bounds, ..Default::default() }).unwrap();
                assert!(found.converged && off(&found.x, &[1.0, 1.0]) < 1e-6, "{method:?} {bounds:?}: {found:?}");
            }
        // With a gradient and no bounds, `minimize` runs L-BFGS.
        let found = minimize(&mut Differentiable { value: rosenbrock, derivative: rosenbrock_gradient }, &[-1.2, 1.0], &Options::default()).unwrap();
        assert!(found.converged && off(&found.x, &[1.0, 1.0]) < 1e-6, "L-BFGS: {found:?}");
        // One variable runs Brent's method, which brackets the minimum or stops at a bound.
        let mut parabola = |x: &[f64]| Ok::<_, Infallible>((x[0] - 2.0).powi(2) + 1.0);
        let found = minimize(&mut parabola, &[10.0], &Options::default()).unwrap();
        assert!(found.converged && off(&found.x, &[2.0]) < 1e-6 && found.evaluations < 40, "{found:?}");
        let found = minimize(&mut parabola, &[0.0], &Options { bounds: Some(Bounds { lower: &[-1.0], upper: &[1.5] }), ..Default::default() }).unwrap();
        assert!(found.converged && found.x == [1.5] && found.value == 1.25, "{found:?}");
        // A minimum outside the bounds gives the nearest bound, and f32 works at its own precision.
        let mut bowl = |x: &[f32]| Ok::<_, Infallible>((x[0] - 3.0).powi(2) + (x[1] + 1.0).powi(2));
        let corner = Some(Bounds { lower: &[0.0, 0.0], upper: &[2.0, 2.0] });
        for method in [Method::Powell, Method::NelderMead] {
            let found = minimize(&mut bowl, &[0.0f32, 0.0], &Options { method: Some(method), bounds: corner, ..Default::default() }).unwrap();
            assert!(found.converged && found.x[0] == 2.0 && found.x[1].abs() < 1e-3, "{method:?}: {found:?}");
        }
        // A start at a bound, in a range narrower than the first step, still moves away from that bound.
        let narrow = Some(Bounds { lower: &[0.0, 0.0], upper: &[0.05, 0.05] });
        let found = minimize(&mut bowl, &[0.0f32, 0.0], &Options { method: Some(Method::NelderMead), bounds: narrow, ..Default::default() }).unwrap();
        assert!(found.converged && found.x == [0.05, 0.0], "narrow: {found:?}");
        // Powell's replaced directions can point out of an active bound, so a stalled round restarts from the coordinate directions.
        let (lower, upper) = ([f64::NEG_INFINITY, 0.0], [0.8, f64::INFINITY]);
        for method in [Method::Powell, Method::NelderMead] {
            let bounds = Some(Bounds { lower: &lower, upper: &upper });
            let found = minimize(&mut rosenbrock, &[0.5, 0.5], &Options { method: Some(method), bounds, ..Default::default() }).unwrap();
            assert!(found.converged && off(&found.x, &[0.8, 0.64]) < 1e-6, "{method:?}: {found:?}");
        }
        let found = minimize(&mut bowl, &[0.0f32, 0.0], &Options::default()).unwrap();
        assert!(found.converged && (found.x[0] - 3.0).abs() < 1e-3 && (found.x[1] + 1.0).abs() < 1e-3, "{found:?}");
    }

    #[test]
    fn defaults_limits_and_failures() {
        use Method::*;
        let minimizing = [(1, true, false), (2, true, false), (2, true, true), (2, false, true), (2, false, false)];
        let choices = minimizing.map(|(n, gradient, bounded)| Method::choose(Task::Minimize { gradient }, n, bounded));
        assert_eq!(choices, [Brent, Lbfgs, Powell, Powell, NelderMead]);
        // Solving picks Brent's method for one equation in one variable, and solving linear equations picks MINRES for least squares.
        let solving = [(Task::Solve { residuals: 1 }, 1), (Task::Solve { residuals: 3 }, 2), (Task::Linear { equations: 3, adjoint: false }, 3)];
        let least_squares = [(Task::Linear { equations: 3, adjoint: true }, 3), (Task::Linear { equations: 5, adjoint: true }, 3)];
        let choices = solving.into_iter().chain(least_squares).map(|(task, n)| Method::choose(task, n, false)).collect::<Vec<_>>();
        assert_eq!(choices, [Brent, LevenbergMarquardt, Gmres, Minres, Minres]);
        assert_eq!(
            Options::<f64> { method: Some(Cg), ..Default::default() }.check(Task::Minimize { gradient: false }, 2).unwrap_err().message,
            "that method solves equations, and doesn't minimize"
        );
        // A search that runs out of evaluations returns its best point so far, unconverged.
        let found = minimize(&mut rosenbrock, &[-1.2, 1.0], &Options { evaluations: Some(25), ..Default::default() }).unwrap();
        assert!(!found.converged && found.evaluations == 25 && found.value < rosenbrock(&[-1.2, 1.0]).unwrap(), "{found:?}");
        // An error from the objective ends the search.
        let mut calls = 0;
        let mut failing = |x: &[f64]| {
            calls += 1;
            if calls > 10 { Err("stopped") } else { Ok(x[0] * x[0] + x[1] * x[1]) }
        };
        assert_eq!(minimize(&mut failing, &[3.0, 1.0], &Options::default()), Err("stopped"));
        // NaN counts as +∞, so the search stays where the objective is defined.
        let mut holed = |x: &[f64]| Ok::<_, Infallible>(if x[0] < 0.5 { f64::NAN } else { x[0] * x[0] + x[1] * x[1] });
        let found = minimize(&mut holed, &[2.0, 2.0], &Options::default()).unwrap();
        assert!(found.converged && off(&found.x, &[0.5, 0.0]) < 1e-6, "{found:?}");
        // Reversed bounds, and a tolerance that isn't a positive number, are errors that a binding reports.
        let reversed = Options { bounds: Some(Bounds { lower: &[1.0], upper: &[0.0] }), ..Default::default() };
        assert_eq!(reversed.check(Task::Minimize { gradient: false }, 1).unwrap_err().message, "each lower limit must be at most its upper limit");
        assert_eq!(
            Options { tolerance: Some(0.0), ..Default::default() }.check(Task::Minimize { gradient: false }, 1).unwrap_err().message, "the tolerance must be a positive number"
        );
    }

    /// The sum of squares, as an objective that records how many points each call evaluates.
    struct Recorded(Vec<usize>);

    impl Objective<f64> for Recorded {
        type Error = Infallible;
        fn value(&mut self, x: &[f64]) -> Result<f64, Infallible> { self.0.push(1); Ok(x.iter().map(|v| v * v).sum()) }
        fn batch(&self) -> bool { true }
        fn values(&mut self, xs: &[f64], out: &mut [f64]) -> Result<(), Infallible> {
            self.0.push(out.len());
            let n = xs.len() / out.len();
            for (y, x) in out.iter_mut().zip(xs.chunks_exact(n)) { *y = x.iter().map(|v| v * v).sum() }
            Ok(())
        }
    }

    #[test]
    fn batches() {
        // Nelder–Mead evaluates its first simplex in one call, and differential evolution each generation.
        let mut simplex = Recorded(Vec::new());
        minimize(&mut simplex, &[1.0, 2.0, 3.0], &Options::default()).unwrap();
        assert_eq!(simplex.0[0], 4);
        let (lower, upper) = ([-5.0; 3], [5.0; 3]);
        let mut evolving = Recorded(Vec::new());
        let options =
            Options { method: Some(Method::Evolution), bounds: Some(Bounds { lower: &lower, upper: &upper }), population: Some(12), ..Default::default() };
        let found = minimize(&mut evolving, &[1.0, 2.0, 3.0], &options).unwrap();
        assert!(found.converged && off(&found.x, &[0.0; 3]) < 1e-6 && evolving.0.iter().filter(|&&k| k == 12).count() > 2, "{found:?}");
    }

    /// `f` as a mapping of one variable to one value.
    fn scalar<F: Copy>(f: impl Fn(F) -> F) -> impl FnMut(&[F], &mut [F]) -> Result<(), Infallible> { move |x, out| { out[0] = f(x[0]); Ok(()) } }

    /// `solve` from `start`, after evaluating `mapping` there for its `residuals` first values, as a caller does.
    fn solved<F: rustymath::Native + rustymath::linalg::Factor, M: Mapping<F>>(
        mapping: &mut M, start: &[F], residuals: usize, options: &Options<'_, F>) -> Result<Found<Vec<F>, F>, M::Error> {
        let mut first = vec![F::zero(); residuals];
        mapping.value(start, &mut first)?;
        solve(mapping, start, &first, options)
    }

    #[test]
    fn roots() {
        let options = Options::default();
        // Finite bounds are the bracket. Without them, the search steps outward from the start until the signs differ.
        let bracket = Options { bounds: Some(Bounds { lower: &[0.0], upper: &[1.0] }), ..options };
        let found = solved(&mut scalar(|x: f64| x.cos() - x), &[0.0], 1, &bracket).unwrap();
        assert!(found.converged && (found.x[0] - 0.739_085_133_215_160_6).abs() < 1e-8 && found.evaluations < 12, "{found:?}");
        let found = solved(&mut scalar(|x: f64| x.cos() - x), &[-20.0], 1, &options).unwrap();
        assert!(found.converged && (found.x[0] - 0.739_085_133_215_160_6).abs() < 1e-8, "{found:?}");
        let found = solved(&mut scalar(|x: f32| x * x * x - 2.0), &[0.0f32], 1, &Options::default()).unwrap();
        assert!(found.converged && (found.x[0] - 2f32.cbrt()).abs() < 1e-3, "{found:?}");
        let found = solved(&mut scalar(|x: f64| x.sinh()), &[3.0], 1, &options).unwrap();
        assert!(found.converged && found.x[0].abs() < 1e-15, "{found:?}");
        // A step into negative numbers, where √ is NaN, halves back until the value is defined, and so finds the root at 1/9.
        let found = solved(&mut scalar(|x: f64| 1.0 / x.sqrt() - 3.0), &[1.0], 1, &options).unwrap();
        assert!(found.converged && (found.x[0] - 1.0 / 9.0).abs() < 1e-8, "{found:?}");
        // A bracket whose ends have the same sign fails, and gives the point nearest zero of the three evaluated: the start and the ends.
        let same = Options { bounds: Some(Bounds { lower: &[-1.0], upper: &[2.0] }), ..options };
        let found = solved(&mut scalar(|x: f64| x * x + 1.0), &[0.0], 1, &same).unwrap();
        assert!(!found.converged && found.x == [0.0] && found.evaluations == 3, "{found:?}");
        assert_eq!(solved(&mut |_: &[f64], _: &mut [f64]| Err::<(), _>("stopped"), &[0.0], 1, &options), Err("stopped"));
    }

    #[test]
    fn nonlinear_systems() {
        let options = Options::default();
        // y = a e^(b x) through five points that it fits exactly, from a start far off.
        let xs = [0.0, 0.5, 1.0, 1.5, 2.0];
        let mut curve = |p: &[f64], r: &mut [f64]| {
            for (r, &x) in r.iter_mut().zip(&xs) { *r = p[0] * (p[1] * x).exp() - 2.0 * (0.3 * x).exp() }
            Ok::<_, Infallible>(())
        };
        let found = solved(&mut curve, &[1.0, 1.0], 5, &options).unwrap();
        assert!(found.converged && off(&found.x, &[2.0, 0.3]) < 1e-8 && found.value < 1e-10, "{found:?}");
        // With a Jacobian–vector product and more residuals than variables, the Jacobian comes from it, column by column, in place of forward differences.
        struct Curve<'a>(&'a [f64]);
        impl Mapping<f64> for Curve<'_> {
            type Error = Infallible;
            fn value(&mut self, p: &[f64], r: &mut [f64]) -> Result<(), Infallible> {
                for (r, &x) in r.iter_mut().zip(self.0) { *r = p[0] * (p[1] * x).exp() - 2.0 * (0.3 * x).exp() }
                Ok(())
            }
            fn has_jvp(&self) -> bool { true }
            fn jvp(&mut self, v: &[f64], p: &[f64], out: &mut [f64]) -> Result<bool, Infallible> {
                for (o, &x) in out.iter_mut().zip(self.0) { *o = (v[0] + v[1] * p[0] * x) * (p[1] * x).exp() }
                Ok(true)
            }
        }
        let forward = solved(&mut Curve(&xs), &[1.0, 1.0], 5, &options).unwrap();
        assert!(forward.converged && off(&forward.x, &[2.0, 0.3]) < 1e-8 && forward.evaluations < found.evaluations, "{forward:?}");
        // Rosenbrock's function as two equations, 10 (y - x²) = 0 and 1 - x = 0, whose root is (1, 1). With a vector–Jacobian product, the Jacobian comes from it, row by row, in place of forward differences.
        let rosen = |p: &[f64], r: &mut [f64]| {
            (r[0], r[1]) = (10.0 * (p[1] - p[0] * p[0]), 1.0 - p[0]);
            Ok::<_, Infallible>(())
        };
        let differenced = solved(&mut { rosen }, &[-1.2, 1.0], 2, &options).unwrap();
        assert!(differenced.converged && off(&differenced.x, &[1.0, 1.0]) < 1e-10, "{differenced:?}");
        let vjp = |u: &[f64], p: &[f64], g: &mut [f64]| {
            (g[0], g[1]) = (-20.0 * p[0] * u[0] - u[1], 10.0 * u[0]);
            Ok::<_, Infallible>(())
        };
        let exact = solved(&mut Differentiable { value: rosen, derivative: vjp }, &[-1.2, 1.0], 2, &options).unwrap();
        assert!(exact.converged && off(&exact.x, &[1.0, 1.0]) < 1e-10 && exact.evaluations < differenced.evaluations, "{exact:?}");
        // Data that no line fits gives the least-squares line, as `lstsq` gives it.
        let (t, y) = ([0.0, 1.0, 2.0, 3.0, 4.0], [1.0, 2.9, 5.2, 6.8, 9.1]);
        let mut line = |c: &[f64], r: &mut [f64]| {
            for i in 0..5 { r[i] = c[0] + c[1] * t[i] - y[i] }
            Ok::<_, Infallible>(())
        };
        let found = solved(&mut line, &[0.0, 0.0], 5, &options).unwrap();
        let design: Vec<f64> = t.iter().flat_map(|&t| [1.0, t]).collect();
        let (best, _) = rustymath::linalg::lstsq(&design, &y, [5, 2], 1);
        assert!(found.converged && off(&found.x, &best) < 1e-7, "{found:?}");
        // Residuals a million times larger change nothing: the damping scales with the Jacobian's columns.
        let mut steep = |p: &[f64], r: &mut [f64]| {
            for i in 0..2 { r[i] = 1e6 * (p[i] - 2.0) }
            Ok::<_, Infallible>(())
        };
        let found = solved(&mut steep, &[1.0, 1.0], 2, &options).unwrap();
        assert!(found.converged && off(&found.x, &[2.0, 2.0]) < 1e-8, "{found:?}");
        // Where the residuals are infinite, or every difference for the Jacobian is NaN, the gradient test can't pass.
        let mut pole = |p: &[f64], r: &mut [f64]| {
            (r[0], r[1]) = (1.0 / p[0] - 1.0, p[1] - 1.0);
            Ok::<_, Infallible>(())
        };
        let found = solved(&mut pole, &[0.0, 0.0], 2, &options).unwrap();
        assert!(!found.converged, "{found:?}");
        let mut edge = |p: &[f64], r: &mut [f64]| {
            for i in 0..2 { r[i] = (1.0 - p[i]).sqrt() - 0.5 }
            Ok::<_, Infallible>(())
        };
        let found = solved(&mut edge, &[1.0, 1.0], 2, &options).unwrap();
        assert!(!found.converged, "{found:?}");
        // A search that runs out of evaluations returns its best point so far, and an error from the function ends it.
        let found = solved(&mut curve, &[1.0, 1.0], 5, &Options { evaluations: Some(6), ..options }).unwrap();
        assert!(!found.converged && found.evaluations == 6, "{found:?}");
        assert_eq!(solved(&mut |_: &[f64], _: &mut [f64]| Err::<(), _>("stopped"), &[0.0, 0.0], 2, &options), Err("stopped"));
        let bounded = Options { bounds: Some(Bounds { lower: &[0.0; 2], upper: &[1.0; 2] }), ..options };
        assert_eq!(bounded.check(Task::Solve { residuals: 5 }, 2).unwrap_err().message, "Levenberg–Marquardt takes no bounds");
    }

    /// A uniform number from -0.5 to 0.5, the `i`th of a fixed sequence.
    fn uniform(i: usize) -> f64 { (super::support::random(i as u64) >> 11) as f64 / (1u64 << 53) as f64 - 0.5 }

    /// AᵀA + I for an `n`×`n` matrix A of the `uniform` numbers from the `first`th on, which is symmetric positive definite.
    fn positive_definite(n: usize, first: usize) -> Vec<f64> {
        let a: Vec<f64> = (first..first + n * n).map(uniform).collect();
        (0..n * n).map(|k| (0..n).map(|l| a[l * n + k / n] * a[l * n + k % n]).sum::<f64>() + if k % (n + 1) == 0 { 1.0 } else { 0.0 }).collect()
    }

    /// The products of the matrix `a`, whose rows hold `cols` items, with vectors.
    fn product(a: &[f64], cols: usize) -> impl FnMut(&[f64], &mut [f64]) -> Result<(), Infallible> + '_ {
        move |v, out| {
            for (i, o) in out.iter_mut().enumerate() { *o = (0..cols).map(|j| a[i * cols + j] * v[j]).sum() }
            Ok(())
        }
    }

    /// The products of the transpose of `a`, whose rows hold `cols` items, with vectors, as a vector–Jacobian product.
    fn transposed(a: &[f64], cols: usize) -> impl FnMut(&[f64], &[f64], &mut [f64]) -> Result<(), Infallible> + '_ {
        move |u, _, out| {
            for (j, o) in out.iter_mut().enumerate() { *o = u.iter().enumerate().map(|(i, &u)| a[i * cols + j] * u).sum() }
            Ok(())
        }
    }

    #[test]
    fn linear_systems() {
        let n = 30;
        let tight = Options { tolerance: Some(1e-12), ..Default::default() };
        let random: Vec<f64> = (0..n * n).map(uniform).collect();
        let spd = positive_definite(n, 0);
        let general: Vec<f64> = (0..n * n).map(|k| random[k] + if k % (n + 1) == 0 { 4.0 } else { 0.0 }).collect();
        let b: Vec<f64> = (1000..1000 + n).map(uniform).collect();
        let zeros = vec![0.0; n];
        // CG, MINRES and GMRES on a symmetric positive definite system, and GMRES, the default for a square system, on a general one.
        for method in [Method::Cg, Method::Minres, Method::Gmres] {
            let found = solve_linear(&mut product(&spd, n), &b, &zeros, &Options { method: Some(method), ..tight }).unwrap();
            let exact = rustymath::linalg::solve(&spd, &b, [n, n], 1);
            assert!(found.converged && off(&found.x, &exact) < 1e-9 && found.value < 1e-10, "{method:?}: {found:?}");
        }
        let found = solve_linear(&mut product(&general, n), &b, &zeros, &tight).unwrap();
        let exact = rustymath::linalg::solve(&general, &b, [n, n], 1);
        assert!(found.converged && off(&found.x, &exact) < 1e-9 && found.evaluations > 20, "{found:?}");
        // Least squares on a tall system, by MINRES, the default, and by CG on the normal equations, against `lstsq`.
        let (m, k) = (60, 20);
        let tall: Vec<f64> = (5000..5000 + m * k).map(uniform).collect();
        let y: Vec<f64> = (9000..9000 + m).map(uniform).collect();
        let (best, _) = rustymath::linalg::lstsq(&tall, &y, [m, k], 1);
        for method in [None, Some(Method::Cg)] {
            let mut a = Differentiable { value: product(&tall, k), derivative: transposed(&tall, k) };
            let found = solve_linear(&mut a, &y, &vec![0.0; k], &Options { method, ..tight }).unwrap();
            assert!(found.converged && off(&found.x, &best) < 1e-8, "{method:?}: {found:?}");
        }
        assert_eq!(Options::<f64>::default().check(Task::Linear { equations: m, adjoint: false }, k).unwrap_err().message, "least squares needs products with the transpose");
        // A right-hand side of zeros gives zeros, from any start.
        let found = solve_linear(&mut product(&spd, n), &zeros, &b, &tight).unwrap();
        assert!(found.converged && found.x == zeros, "{found:?}");
        // A solve that runs out of products returns its last iterate, and an error from a product ends it.
        let found = solve_linear(&mut product(&general, n), &b, &zeros, &Options { evaluations: Some(5), ..tight }).unwrap();
        assert!(!found.converged && found.evaluations == 6, "{found:?}");
        assert_eq!(solve_linear(&mut |_: &[f64], _: &mut [f64]| Err::<(), _>("stopped"), &b, &zeros, &tight), Err("stopped"));
    }

    /// The matrix `a`, whose rows hold `cols` items, as a mapping that gives its transpose's products where `adjoint`. Where `diagonal` isn't empty, its preconditioner is Jacobi's, which divides by `diagonal`.
    struct Dense<'a> { a: &'a [f64], cols: usize, adjoint: bool, diagonal: Vec<f64> }

    impl Mapping<f64> for Dense<'_> {
        type Error = Infallible;
        fn value(&mut self, v: &[f64], out: &mut [f64]) -> Result<(), Infallible> { product(self.a, self.cols)(v, out) }
        fn has_vjp(&self) -> bool { self.adjoint }
        fn vjp(&mut self, u: &[f64], x: &[f64], out: &mut [f64]) -> Result<(), Infallible> { transposed(self.a, self.cols)(u, x, out) }
        fn has_preconditioner(&self) -> bool { !self.diagonal.is_empty() }
        fn precondition(&mut self, v: &[f64], out: &mut [f64]) -> Result<(), Infallible> { for i in 0..v.len() { out[i] = v[i] / self.diagonal[i] } Ok(()) }
    }

    #[test]
    fn preconditioners() {
        // A symmetric positive definite matrix whose diagonal spans four orders of magnitude. Jacobi's preconditioner, the reciprocal of the diagonal, cuts the products that each method needs.
        let n = 40;
        let scale: Vec<f64> = (0..n).map(|i| 10f64.powf(2.0 * i as f64 / (n - 1) as f64)).collect();
        let core = positive_definite(n, 20_000);
        let a: Vec<f64> = (0..n * n).map(|k| core[k] * scale[k / n] * scale[k % n]).collect();
        let b: Vec<f64> = (30_000..30_000 + n).map(uniform).collect();
        let diagonal: Vec<f64> = (0..n).map(|i| a[i * (n + 1)]).collect();
        let options = Options { tolerance: Some(1e-10), evaluations: Some(10_000), ..Default::default() };
        let exact = rustymath::linalg::solve(&a, &b, [n, n], 1);
        let size = exact.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        for method in [Method::Cg, Method::Minres, Method::Gmres] {
            let options = Options { method: Some(method), ..options };
            let plain = solve_linear(&mut Dense { a: &a, cols: n, adjoint: false, diagonal: vec![] }, &b, &vec![0.0; n], &options).unwrap();
            let fast = solve_linear(&mut Dense { a: &a, cols: n, adjoint: false, diagonal: diagonal.clone() }, &b, &vec![0.0; n], &options).unwrap();
            assert!(fast.converged && off(&fast.x, &exact) < 1e-6 * size && 2 * fast.evaluations < plain.evaluations, "{method:?}: {plain:?} {fast:?}");
        }
        // Least squares with columns of very different sizes, where M approximates (AᵀA)⁻¹ by the reciprocals of the columns' squared norms.
        let (m, k) = (200, 60);
        let tall: Vec<f64> = (40_000..40_000 + m * k).map(|i| uniform(i) * 10f64.powf(3.0 * (i % k) as f64 / (k - 1) as f64)).collect();
        let y: Vec<f64> = (50_000..50_000 + m).map(uniform).collect();
        let squares: Vec<f64> = (0..k).map(|j| (0..m).map(|i| tall[i * k + j] * tall[i * k + j]).sum()).collect();
        let (best, _) = rustymath::linalg::lstsq(&tall, &y, [m, k], 1);
        let size = best.iter().fold(0.0f64, |m, v| m.max(v.abs()));
        let plain = solve_linear(&mut Dense { a: &tall, cols: k, adjoint: true, diagonal: vec![] }, &y, &vec![0.0; k], &options).unwrap();
        let fast = solve_linear(&mut Dense { a: &tall, cols: k, adjoint: true, diagonal: squares }, &y, &vec![0.0; k], &options).unwrap();
        assert!(fast.converged && off(&fast.x, &best) < 1e-6 * size && 2 * fast.evaluations < plain.evaluations, "{plain:?} {fast:?}");
    }
}

mod polynomial {
    use half::f16;
    use num_complex::Complex;
    use rustymath::polynomial::*;

    #[test]
    fn polynomials() {
        // Mixed radix: each step multiplies by its own base, and the first base goes unused.
        assert_eq!(decode(&[9.0, 2.0, 2.0], &[1.0, 0.0, 1.5, 1.0, 1.0, 1.0]), [5.5, 7.0]);
        // (x - 1)(x - 2), with a trailing zero that infinity never meets.
        let (c, mut y) = ([2.0, -3.0, 1.0, 0.0], vec![]);
        evaluate(&c, &[0.0, 1.5, 3.0, f64::INFINITY], &mut y);
        assert_eq!(y, [2.0, -0.25, 2.0, f64::INFINITY]);
        let mut y = vec![];
        evaluate(&[0.0], &[1.0, 2.0], &mut y);
        assert_eq!(y, [0.0; 2]);

        // Across block boundaries, at f32 and f16, against f64.
        let k = [0.5, -1.25, 0.75, 2.0, -0.125];
        let x: Vec<f64> = (0..1000).map(|i| i as f64 / 250.0 - 2.0).collect();
        let (mut want, mut got, mut half) = (vec![], vec![], vec![]);
        evaluate(&k, &x, &mut want);
        evaluate(&k.map(|k| k as f32), &x.iter().map(|&x| x as f32).collect::<Vec<f32>>(), &mut got);
        assert!(got.iter().zip(&want).all(|(&g, &w)| (g as f64 - w).abs() <= 1e-5 * w.abs().max(1.0)));
        evaluate(&k.map(f16::from_f64), &x.iter().map(|&x| f16::from_f64(x)).collect::<Vec<f16>>(), &mut half);
        assert!(half.iter().zip(&want).all(|(&g, &w)| (g.to_f64() - w).abs() <= 1e-2 * w.abs().max(1.0)));

        let r = roots(&c.map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r[0].im == 0.0 && r[1].im == 0.0 && (r[0].re - 1.0).abs() < 1e-12 && (r[1].re - 2.0).abs() < 1e-12);
        // x² + 1, and a polynomial with complex coefficients, (x - i)(x - 2).
        let r = roots(&[1.0f32, 0.0, 1.0].map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r.iter().all(|z| z.re.abs() < 1e-6 && (z.im.abs() - 1.0).abs() < 1e-6));
        let r = roots(&from_roots(Complex::new(1.0, 0.0), &[Complex::new(0.0, 1.0), Complex::new(2.0, 0.0)]).unwrap()).unwrap();
        assert!((r[0] - Complex::new(0.0, 1.0)).norm() < 1e-12 && (r[1] - Complex::new(2.0, 0.0)).norm() < 1e-12);
        assert!(roots::<f64>(&[Complex::new(3.0, 0.0)]).unwrap().is_empty());
        // 1 + ∞x has the root -1/∞, which is -0.
        let r = roots(&[1.0, f64::INFINITY].map(|k| Complex::new(k, 0.0))).unwrap();
        assert!(r.len() == 1 && r[0].re == 0.0 && r[0].re.is_sign_negative() && r[0].im == 0.0);

        assert_eq!(from_roots(1.0, &[1.0, 2.0]).unwrap(), [2.0, -3.0, 1.0]);
        assert_eq!(derivative(&[2.0, -3.0, 1.0]).unwrap(), [-3.0, 2.0]);
        let f = fit(&[0.0f64, 1.0, 2.0, 3.0, 4.0], &[2.0, 0.0, 0.0, 2.0, 6.0], 2);
        assert!(f.iter().zip([2.0, -3.0, 1.0]).all(|(&a, b)| (a - b).abs() < 1e-12));
    }
}

mod random {
    use crate::support::seeded;
    use rustymath::Arg;
    use rustymath::random::*;

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

    // A seed gives the numbers that `rand`'s `Uniform` gives for each bound, from one bound or one for each draw, also for bounds near 2⁶³, where about half the draws reject a word.
    #[test]
    fn rolls_match_rand() {
        use rand::distr::{Distribution, Uniform};
        let mixed: Vec<u64> = (0..2000).map(|k| if k % 7 == 0 { (1 << 63) + 1 } else { k + 1 }).collect();
        for b in [1, 6, 1 << 63, (1 << 63) + 1, u64::MAX] {
            let (mut ours, mut theirs) = (seeded(2), seeded(2));
            let ones = vec![b; 2000];
            let mut out = Vec::new();
            roll(Arg::One(b), 2000, &mut ours, &mut out);
            roll(Arg::Many(&ones), 2000, &mut ours, &mut out);
            roll(Arg::Many(&mixed), 2000, &mut ours, &mut out);
            out.push(below(b, &mut ours));
            let bounds = [&ones[..], &ones, &mixed, &[b]].concat();
            assert_eq!(out, bounds.iter().map(|&b| Uniform::new(0, b).unwrap().sample(&mut theirs)).collect::<Vec<_>>());
        }
    }
}
