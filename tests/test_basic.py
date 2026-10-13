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
    assert ulps(rm.arctan2(x, x - 0.5), np.arctan2(x, x - 0.5)).max() <= 4
    assert ulps(rm.root(3, x[::7]), np.cbrt(x[::7])).max() <= 4
    assert rm.exp(np.float32(1)).dtype == np.float32 and rm.exp(1).dtype == np.float64
    out = np.empty(3, np.float32)
    assert rm.sin(np.zeros(3, np.float32), out=out) is out
    # Strided inputs and outputs, and a broadcast item, give what contiguous ones give.
    y = np.linspace(-3, 3, 10_000)
    assert (rm.sin(y[::3]) == rm.sin(y)[::3]).all() and (rm.power(2.0, y[::-2]) == rm.power(np.full(5000, 2.0), y[::-2])).all()
    out = np.zeros(20_000)
    rm.exp(y, out=out[::2])
    assert (out[::2] == rm.exp(y)).all() and (out[1::2] == 0).all()

def test_complex():
    z = (np.linspace(-2, 2, 40)[:, None] + 1j * np.linspace(-1.9, 2.1, 40)).ravel()
    for name in ['exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan', 'arcsinh', 'arccosh',
                 'arctanh']:
        for dt in [np.complex64, np.complex128]:
            got, want = getattr(rm, name)(z.astype(dt)), getattr(np, name)(z.astype(dt).astype(np.complex128))
            assert got.dtype == dt, name
            # NumPy's complex functions err by an ulp or two themselves, so this bound is loose. The Rust tests compare with mpmath.
            assert (np.abs(got - want) <= 16 * np.finfo(got.real.dtype).eps * np.abs(want)).all(), (name, dt)
    assert np.allclose(rm.power(z, 0.5 + 1j), z**(0.5 + 1j)) and rm.absolute(z).dtype == np.float64 and rm.abs is rm.absolute
    assert np.allclose(rm.absolute(z), np.abs(z)) and np.allclose(rm.angle(z), np.angle(z)) and np.allclose(rm.sign(z), z / np.abs(z))
    assert rm.sign(-2.5) == -1 and rm.angle(-1.0) == np.pi

def test_integers():
    a, b = np.arange(-30, 30), np.arange(1, 61)[::-1] - 25
    assert (rm.gcd(a, b) == np.gcd(a, b)).all() and (rm.lcm(a, b) == np.sign(a * b) * np.lcm(a, b)).all() and rm.gcd(a, b).dtype == np.int64
    assert (rm.remainder(a, b + 100) == np.remainder(a, b + 100)).all() and rm.mod is rm.remainder
    assert (rm.power(a, 3) == a**3).all() and rm.power(a, 3).dtype == np.int64
    with np.errstate(over='raise'), pytest.raises(FloatingPointError): rm.power(np.int64(10), 30)
    # Every integer width keeps its type, as in NumPy, with add and multiply wrapping at the narrow widths.
    for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
        x, y = np.arange(1, 40).astype(t), np.arange(40, 1, -1).astype(t)
        for name in ['add', 'multiply', 'maximum', 'minimum', 'remainder', 'gcd']:
            got, want = getattr(rm, name)(x, y), getattr(np, name)(x, y)
            assert got.dtype == want.dtype and (got == want).all(), (name, t)
        for name in ['floor', 'ceil', 'absolute', 'sign']:
            got, want = getattr(rm, name)(x), getattr(np, name)(x)
            assert got.dtype == want.dtype and (got == want).all(), (name, t)
        assert rm.equal(x, y).dtype == bool and (rm.less(x, y) == (x < y)).all(), t
    # Integers compare exactly, where float64 would round them to one value.
    assert not rm.equal(10**15, 10**15 + 1) and rm.less(2**62, 2**62 + 1)
    # Beside a float, integers still take the float loops and the tolerance, as NumPy picks the first loop they cast to safely.
    assert rm.add(np.arange(3, dtype=np.int8), 0.5).dtype == np.add(np.arange(3, dtype=np.int8), 0.5).dtype
    assert rm.equal(np.arange(3), 1 + 1e-15).tolist() == [False, True, False]
    big = np.array([2**63 + 1], np.uint64)
    assert rm.add(big, big)[0] == np.add(big, big)[0] and rm.not_equal(big, big + np.uint64(1))[0]
    assert rm.maximum.reduce(np.array([3, -7, 5], np.int8)).dtype == np.int8 and rm.add.reduce(np.array([100, 100], np.int8)) == 200
    # A signed type's least value has no magnitude of its type, so `absolute` wraps it to itself, as NumPy's arrays do.
    assert rm.absolute(np.array([-128], np.int8))[0] == -128
    p, q = np.array([True, True, False, False]), np.array([True, False, True, False])
    for name in ['add', 'multiply', 'maximum', 'minimum']:
        got, want = getattr(rm, name)(p, q), getattr(np, name)(p, q)
        assert got.dtype == bool and (got == want).all(), name
    assert rm.floor(p).dtype == bool and rm.absolute(p).dtype == bool and rm.add.reduce(p) == 2

def test_reductions():
    x = np.random.default_rng(0).random(10_001)
    assert np.isclose(rm.add.reduce(x), x.sum()) and np.allclose(rm.add.accumulate(x), np.cumsum(x))
    assert np.isclose(rm.multiply.reduce(1 + x[:100] / 100), np.prod(1 + x[:100] / 100))
    m = x[:10_000].reshape(100, 100)
    assert np.allclose(rm.add.reduce(m, axis=0), m.sum(0)) and np.allclose(rm.add.reduce(m, axis=1), m.sum(1))
    v = np.array([3.0, 1.0, np.nan, 5.0])
    assert np.isnan(rm.maximum.reduce(v)) and np.array_equal(rm.minimum.accumulate(v), np.minimum.accumulate(v), equal_nan=True)
    assert rm.maximum.reduce(x[::7]) == x[::7].max() and np.isnan(rm.minimum(np.nan, 1.0))
    assert rm.power.reduce([2.0, 3.0, 2.0]) == 64 and rm.power.accumulate([2.0, 3.0, 2.0]).tolist() == [2, 8, 64]
    big = np.array([2**62, 2**62, 2**62])
    assert rm.add.reduce(big) == np.add.reduce(big) and (rm.add.accumulate(big) == np.add.accumulate(big)).all()

def test_tolerant():
    assert rm.equal(0.1 + 0.2, 0.3) and not rm.equal(0.1 + 0.2, 0.3, ct=0)
    assert not rm.equal(1, 1 + 3e-14) and rm.equal([1, 2], 1 + 3e-14, ct=1e-13).tolist() == [True, False]
    assert not rm.less(0.3, 0.1 + 0.2) and rm.less(0.3, 0.1 + 0.2, ct=0) and rm.greater_equal(0.3, 0.1 + 0.2) and not rm.not_equal(0.3, 0.1 + 0.2)
    assert rm.floor(0.5 + 0.5 - 1e-16) == 1 and rm.ceil(1 + 1e-15) == 1 and rm.floor(1 - 1e-15, ct=0) == 0
    assert rm.remainder([-5.0, 5.0], 3).tolist() == np.remainder([-5.0, 5.0], 3).tolist() and rm.remainder(0.3, 0.1) == 0
    assert abs(rm.gcd(0.6, 0.4) - 0.2) < 1e-15 and abs(rm.lcm(-0.6, 0.4) + 1.2) < 1e-14
    with pytest.raises(ValueError): rm.equal(1, 1, ct=1)
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): rm.log(-1.0)

def test_tolerant_search():
    a = np.random.default_rng(0).integers(0, 50, (20, 30))
    vals, first, inverse, counts = rm.unique(a)
    flat = a.ravel()
    assert np.array_equal(first, np.sort(np.unique(flat, return_index=True)[1])) and np.array_equal(vals, flat[first]) and vals.dtype == a.dtype
    assert np.array_equal(vals[inverse], flat) and np.array_equal(counts, np.bincount(inverse))
    # Integers match exactly, where floats this large would match within the tolerance.
    assert rm.index(np.int64([10**15]), 10**15 + 1) == 1 and rm.index(np.uint8([3, 7]), [7, 9]).tolist() == [1, 2]
    x = [0.3, 1.0, 0.1 + 0.2, 2.0]
    assert rm.unique(x).values.tolist() == [0.3, 1.0, 2.0] and len(rm.unique(x, ct=0).values) == 4
    assert [v.tolist() for v in rm.unique(x)[2:]] == [[0, 1, 0, 2], [2, 1, 1]]
    assert rm.index(x, [0.1 + 0.2, 5.0]).tolist() == [0, 4] and rm.index(x, 0.1 + 0.2, ct=0) == 2
    assert rm.isin(np.float32([0.3, 5]), np.float32([0.3])).tolist() == [True, False] and not rm.isin(5.0, x)
    with pytest.raises(TypeError): rm.index(x, [1j])

def test_polynomial():
    from numpy.polynomial import polynomial as P
    c, x = [2.0, -3.0, 1.0], np.linspace(-2, 3, 50)
    assert rm.polynomial.polyval(2.0, [1, 2, 3]) == 17 and rm.polynomial.polyval(np.float32(x), np.float32(c)).dtype == np.float32
    assert np.allclose(rm.polynomial.polyval(x.reshape(5, 10), c), P.polyval(x, c).reshape(5, 10))
    assert np.allclose(rm.polynomial.polyroots(c), [1, 2]) and rm.polynomial.polyroots(np.float16(c)).dtype == np.complex64
    assert np.allclose(rm.polynomial.polyroots([1.0, 0, 1]), [-1j, 1j]) and len(rm.polynomial.polyroots([5.0])) == 0
    # The eigenvalue solver fails on NaN coefficients, so each root is NaN and NumPy reports an invalid value.
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): rm.polynomial.polyroots([np.nan, 0, 1.0])
    with np.errstate(invalid='ignore'): assert np.isnan(rm.polynomial.polyroots([np.nan, 0, 1.0, 0])).sum() == 2
    assert rm.polynomial.polyfromroots([1, 2]).tolist() == [2, -3, 1] and rm.polynomial.polyfromroots([1j, -1j]).tolist() == [1, 0, 1]
    assert rm.polynomial.polyder([1, 2, 3, 4]).tolist() == [2, 6, 12] and len(rm.polynomial.polyder([3.0])) == 0
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
    w = np.linspace(-0.36, 20, 300)
    assert np.allclose(rm.lambertw(w), special.lambertw(w).real)
    with pytest.warns(RuntimeWarning, match='invalid value'): assert np.isnan(rm.lambertw(-1.0))
    assert np.allclose(rm.lambertw(w + 1j), special.lambertw(w + 1j))
    n, k = np.linspace(-5.5, 30, 50), np.linspace(0, 12, 50)
    assert np.allclose(rm.binom(n, k), special.binom(n, k)) and rm.binom(10.0, 3.0) == 120
    z = x[::10] + 1j * x[::-10]
    assert np.allclose(rm.loggamma(z), special.loggamma(z)) and np.allclose(rm.gamma(z), special.gamma(z))

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

@pytest.mark.parametrize('kind', [float, complex])
def test_linalg(kind):
    la, rng = rm.linalg, np.random.default_rng(0)
    H = lambda x: np.conj(np.swapaxes(x, -1, -2))
    rand = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape) if kind is complex else rng.standard_normal(shape)
    a, b = rand(4, 5, 5), rand(4, 5, 3)
    v = b[0, :, 0]
    assert np.allclose(la.solve(a, b), np.linalg.solve(a, b)) and np.allclose(la.solve(a[0], v), np.linalg.solve(a[0], v))
    assert np.allclose(la.inv(a), np.linalg.inv(a)) and np.allclose(la.det(a), np.linalg.det(a))
    sign, log = la.slogdet(a)
    assert np.allclose(sign, np.linalg.slogdet(a).sign) and np.allclose(log, np.linalg.slogdet(a).logabsdet)
    l, u, p = la.lu(a)
    assert np.allclose(l @ u, np.take_along_axis(a, p[..., :, None], axis=-2)) and np.allclose(np.tril(l), l) and np.allclose(np.triu(u), u)
    assert np.allclose(rm.matmul(a, b), a @ b) and np.allclose(rm.matmul(a, v), a @ v) and np.isclose(rm.matmul(v, v), v @ v)
    spd = a @ H(a) + 5 * np.eye(5)
    assert np.allclose(la.cholesky(spd), np.linalg.cholesky(spd))
    # Signs and phases that NumPy leaves to LAPACK can differ, so factorizations are checked by what they multiply back to.
    for x in rand(3, 4, 7), rand(3, 7, 4):
        q, r = la.qr(x)
        d = np.diagonal(r, axis1=-2, axis2=-1)
        assert np.allclose(q @ r, x) and (d.imag == 0).all()
        q, r = la.qr(x, complete=True)
        assert np.allclose(q @ r, x) and np.allclose(H(q) @ q, np.eye(x.shape[-2]))
        u, s, v = la.svd(x)
        assert np.allclose(u * s[..., None, :] @ H(v), x) and np.allclose(H(u) @ u, np.eye(u.shape[-1])) and np.allclose(H(v) @ v, np.eye(v.shape[-1]))
        assert np.allclose(s, np.linalg.svd(x, compute_uv=False)) and np.allclose(la.svdvals(x), s) and np.allclose(la.pinv(x), np.linalg.pinv(x))
    sym = a + H(a)
    w, vecs = la.eigh(sym)
    assert np.allclose(w, np.linalg.eigvalsh(sym)) and np.allclose(vecs * w[..., None, :] @ H(vecs), sym) and np.allclose(la.eigvalsh(sym), w)
    w, vecs = la.eig(a)
    assert np.allclose(a @ vecs, vecs * w[..., None, :]) and np.allclose(np.sort_complex(w), np.sort_complex(np.linalg.eigvals(a)))
    assert la.eig(sym).values.dtype == la.eigvals(a).dtype == np.result_type(a, 1j)
    # A rank-deficient or non-square matrix gives the least-norm least-squares solution, and its pseudoinverse.
    A, B = rand(9, 4), rand(9, 2)
    deficient = A[:, [0, 1, 1, 3]]
    for x, y in (A, B), (A, B[:, 0]), (deficient, B):
        (got, s), want = la.lstsq(x, y), np.linalg.lstsq(x, y)
        assert np.allclose(got, want[0]) and np.allclose(s, want[3]) and np.allclose(la.solve(x, y), want[0])
    singular = deficient[:4]
    assert np.allclose(la.inv(singular), np.linalg.pinv(singular)) and la.matrix_rank(deficient) == 3 and (la.matrix_rank(a) == 5).all()
    if kind is complex:
        # complex64 computes at its own width, with real results, such as singular values, in float32.
        eye = np.eye(3, dtype=np.complex64)
        assert la.inv(eye).dtype == np.complex64 and la.svd(eye).s.dtype == np.float32 and la.slogdet(eye)[1].dtype == np.float32
        with pytest.raises(TypeError, match='real matrices'): la.semiring(np.maximum, np.add)(a, b)
        return
    # Min-plus products give shortest paths, and or-and products give reachability.
    d = np.array([[0, 3, np.inf], [np.inf, 0, 1], [2, np.inf, 0]])
    assert (la.semiring(np.minimum, np.add)(d, d) == np.min(d[:, :, None] + d[None], axis=1)).all()
    assert np.allclose(la.semiring(rm.maximum, rm.add)(a, b), np.max(a[..., :, :, None] + b[..., None, :, :], axis=-2))
    g = rng.random((6, 6)) < 0.3
    assert (rm.matmul(g, g) == np.matmul(g, g)).all() and (la.semiring(np.logical_or, np.logical_and)(g, g) == np.matmul(g, g)).all()
    # float16 computes in float32 and rounds back.
    assert la.inv(np.eye(3, dtype=np.float16)).dtype == np.float16 and la.svd(np.eye(3, dtype=np.float32)).s.dtype == np.float32
    # Integer products compute in int64. An int8 argument converts to it, so these sums of 3 million don't overflow.
    i8 = np.full((2, 300), 100, np.int8)
    assert rm.matmul(i8, i8.T).dtype == np.int64 and (rm.matmul(i8, i8.T) == 3_000_000).all()
    # Integer max-plus and min-plus products, and the counts of equal and unequal items, match NumPy's results built item by item.
    n, p = rng.integers(-5, 5, (5, 7)), rng.integers(-5, 5, (7, 4))
    assert (la.semiring(np.maximum, np.add)(n, p) == np.max(n[:, :, None] + p[None], axis=1)).all()
    assert (la.semiring(np.minimum, np.add)(n, p) == np.min(n[:, :, None] + p[None], axis=1)).all()
    equal = (n[:, :, None] == p[None]).sum(axis=1)
    assert (la.semiring(np.add, np.equal)(n, p) == equal).all() and (la.semiring(np.add, np.not_equal)(n, p) == 7 - equal).all()
    assert (la.semiring(np.add, np.equal)(n / 2, p / 2) == equal).all()
    # Integer arithmetic wraps on overflow: 2^62 times 4 is 2^64, which wraps to 0.
    assert rm.matmul([[2**62]], [[4]])[0, 0] == 0
    # A matrix that can't be factored gives NaN and NumPy's "invalid value" warning, as a failing item of a ufunc does.
    for f, x in (la.cholesky, -np.eye(2)), (la.eig, [[1., np.nan], [0, 1]]):
        with pytest.warns(RuntimeWarning, match='invalid value'): assert np.isnan(f(np.array(x))[0]).all()
    assert la.slogdet([[1., 2], [2, 4]]) == (0, -np.inf) and la.inv(np.zeros((0, 0))).shape == (0, 0)
