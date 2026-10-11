import numpy as np, pytest, rustymath as rm

def ulps(got, want):
    "Distance from `got` to `want` in units in the last place of `want`'s type"
    return np.abs(got.astype(np.float64) - want) / np.spacing(np.abs(want).astype(got.dtype)).astype(np.float64)

def test_ufuncs():
    x = np.linspace(0.01, 0.99, 1000)
    for name in ['exp', 'expm1', 'log', 'log1p', 'log2', 'log10', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh',
                 'tanh', 'arcsinh', 'arctanh', 'cbrt']:
        for dt in [np.float16, np.float32, np.float64]:
            got = getattr(rm, name)(x.astype(dt))
            assert got.dtype == dt, name
            assert ulps(got, getattr(np, name)(x.astype(dt).astype(np.float64))).max() <= 4, (name, dt)
    assert ulps(rm.arccosh(1 + x), np.arccosh(1 + x)).max() <= 4
    assert ulps(rm.power(x, 2.5), x**2.5).max() <= 4 and ulps(rm.hypot(x, 3), np.hypot(x, 3)).max() <= 4
    assert ulps(rm.root(3, x[::7]), np.cbrt(x[::7])).max() <= 4
    assert rm.exp(np.float32(1)).dtype == np.float32 and rm.exp(1).dtype == np.float64
    out = np.empty(3, np.float32)
    assert rm.sin(np.zeros(3, np.float32), out=out) is out

def test_tolerant():
    assert rm.equal(0.1 + 0.2, 0.3) and not rm.equal(0.1 + 0.2, 0.3, ct=0)
    assert not rm.equal(1, 1 + 3e-14) and rm.equal([1, 2], 1 + 3e-14, ct=1e-13).tolist() == [True, False]
    assert rm.floor(0.5 + 0.5 - 1e-16) == 1 and rm.ceil(1 + 1e-15) == 1 and rm.floor(1 - 1e-15, ct=0) == 0
    assert rm.remainder([-5.0, 5.0], 3).tolist() == np.remainder([-5.0, 5.0], 3).tolist() and rm.remainder(0.3, 0.1) == 0
    assert abs(rm.gcd(0.6, 0.4) - 0.2) < 1e-15
    with pytest.raises(ValueError): rm.equal(1, 1, ct=1)

def test_tolerant_search():
    a = np.random.default_rng(0).integers(0, 50, (20, 30))
    got, want = rm.unique(a, True, True, True, ct=0), np.unique(a, True, True, True)
    assert all(np.array_equal(g, w) for g, w in zip(got, want)) and got[0].dtype == a.dtype
    x = [0.3, 1.0, 0.1 + 0.2, 2.0]
    assert rm.unique(x).tolist() == [0.3, 1.0, 2.0] and len(rm.unique(x, ct=0)) == 4
    assert [v.tolist() for v in rm.unique(x, return_inverse=True, return_counts=True)[1:]] == [[0, 1, 0, 2], [2, 1, 1]]
    assert rm.index(x, [0.1 + 0.2, 5.0]).tolist() == [0, 4] and rm.index(x, 0.1 + 0.2, ct=0) == 2
    assert rm.isin(np.float32([0.3, 5]), np.float32([0.3])).tolist() == [True, False] and rm.isin(5.0, x, invert=True)
    with pytest.raises(TypeError): rm.index(x, [1j])

def test_polynomial():
    from numpy.polynomial import polynomial as P
    c, x = [2.0, -3.0, 1.0], np.linspace(-2, 3, 50)
    assert rm.polynomial.polyval(2.0, [1, 2, 3]) == 17 and rm.polynomial.polyval(np.float32(x), np.float32(c)).dtype == np.float32
    assert np.allclose(rm.polynomial.polyval(x.reshape(5, 10), c), P.polyval(x, c).reshape(5, 10))
    assert np.allclose(rm.polynomial.polyroots(c), [1, 2]) and rm.polynomial.polyroots(np.float16(c)).dtype == np.float16
    assert np.allclose(rm.polynomial.polyroots([1.0, 0, 1]), [-1j, 1j]) and len(rm.polynomial.polyroots([5.0])) == 0
    assert rm.polynomial.polyfromroots([1, 2]).tolist() == [2, -3, 1] and rm.polynomial.polyfromroots([1j, -1j]).tolist() == [1, 0, 1]
    for args in [([3.0],), ([1, 2, 3, 4], 2, 3)]: assert rm.polynomial.polyder(*args).tolist() == P.polyder(*args).tolist()
    assert np.allclose(rm.polynomial.polyfit([0, 1, 2, 3], [1, 3, 7, 13], 2), [1, 1, 1])

def test_number_theory():
    nt = rm.number_theory
    assert nt.is_prime(2**127 - 1) and not nt.is_prime(2**128 + 1) and nt.factor(600851475143) == [(71, 1), (839, 1), (1471, 1), (6857, 1)]
    assert nt.next_prime(2**64) == 2**64 + 13 and nt.prev_prime(2) is None and nt.totient(36) == 12
    assert nt.prime_count(10**6) == 78498 and nt.primes(20).tolist() == [2, 3, 5, 7, 11, 13, 17, 19] and nt.primes(7, 12).tolist() == [7, 11]
    with pytest.raises(ValueError): nt.factor(0)

def test_special():
    from scipy import special
    x = np.linspace(0.1, 5, 200)
    for name in ['gamma', 'erf']: assert ulps(getattr(rm, name)(x), getattr(special, name)(x)).max() <= 4, name
    want = special.gammaln(x)
    assert (np.abs(rm.gammaln(x) - want) <= 4 * np.spacing(np.maximum(np.abs(want), 2))).all()
    p = np.linspace(0.01, 0.99, 99)
    assert np.allclose(rm.erfinv(p), special.erfinv(p)) and np.allclose(rm.erfcinv(p), special.erfcinv(p))
    assert np.allclose(rm.gammainc(2.5, x), special.gammainc(2.5, x)) and np.allclose(rm.gammaincc(2.5, x), special.gammaincc(2.5, x))
    assert np.allclose(rm.gammaincinv(2.5, p), special.gammaincinv(2.5, p))
    assert np.allclose(rm.betainc(2, 3, p), special.betainc(2, 3, p)) and np.allclose(rm.betaincinv(2, 3, p), special.betaincinv(2, 3, p))
    # SciPy's erfc is up to 16 ulps from mpmath near 4.8, so this check is loose. The Rust tests compare erfc with mpmath.
    assert np.allclose(rm.erfc(x), special.erfc(x), rtol=1e-14, atol=0)

def test_distribution():
    from scipy import stats
    from rustymath.distribution import binomial, normal
    d, x = normal(1, 2), np.linspace(-3, 5, 9)
    assert np.allclose(d.pdf(x), stats.norm(1, 2).pdf(x)) and np.allclose(d.cdf(x), stats.norm(1, 2).cdf(x)) and isinstance(d.cdf(1.0), float)
    assert np.allclose(d.quantile([0.1, 0.5]), stats.norm(1, 2).ppf([0.1, 0.5])) and np.isnan(d.quantile(2))
    assert np.allclose(binomial(10, 0.3).pdf(np.arange(11)), stats.binom(10, 0.3).pmf(np.arange(11))) and binomial(10, 0.3).is_discrete
    assert d.sample((2, 3)).shape == (2, 3) and (d.sample(5, seed=1) == d.sample(5, seed=1)).all()
    with pytest.raises(ValueError): normal(0, -1)
    assert normal().cdf(0.0) == 0.5
