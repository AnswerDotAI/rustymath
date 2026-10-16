import math
import numpy as np, pytest, rustymath as rm

A = np.asarray

def ulps(got, want):
    "Distance from `got` to `want` in units in the last place of `want`'s type"
    got = A(got)
    return np.abs(got.astype(np.float64) - want) / np.spacing(np.abs(want).astype(got.dtype)).astype(np.float64)

def test_elementwise():
    x = np.linspace(0.01, 0.99, 1000)
    for name in ['exp', 'expm1', 'log', 'log1p', 'log2', 'log10', 'sin', 'cos', 'tan', 'arcsin', 'arccos', 'arctan', 'sinh', 'cosh',
                 'tanh', 'arcsinh', 'arctanh', 'cbrt']:
        for dt in [np.float16, np.float32, np.float64]:
            got = getattr(rm, name)(x.astype(dt))
            assert A(got).dtype == dt, name
            assert ulps(got, getattr(np, name)(x.astype(dt).astype(np.float64))).max() <= 4, (name, dt)
    assert ulps(rm.arccosh(1 + x), np.arccosh(1 + x)).max() <= 4
    assert ulps(rm.power(x, 2.5), x**2.5).max() <= 4 and ulps(rm.hypot(x, 3), np.hypot(x, 3)).max() <= 4
    assert ulps(rm.arctan2(x, x - 0.5), np.arctan2(x, x - 0.5)).max() <= 4
    assert ulps(rm.root(3, x[::7]), np.cbrt(x[::7])).max() <= 4
    # A Python scalar gives a Python number, and an array of rank 0, NumPy scalars included, gives one of its dtype.
    assert isinstance(rm.exp(1), float) and rm.exp(np.float32(1)).dtype == 'f32' and rm.exp(np.float32(1)).shape == ()
    # Booleans and integers compute in the narrowest float that holds them exactly, but at least float32, and lists convert.
    assert [A(rm.exp(np.array([0, 1], t))).dtype for t in (np.bool_, np.uint8, np.int16, np.int32, np.int64)] == [np.float32] * 3 + [np.float64] * 2
    assert A(rm.exp([0, 1])).dtype == np.float64 and np.allclose(rm.exp([0, 1]), [1, np.e])
    # Strided inputs and a broadcast item give what contiguous ones give.
    y = np.linspace(-3, 3, 10_000)
    assert (A(rm.sin(y[::3])) == A(rm.sin(y))[::3]).all() and (A(rm.power(2.0, y[::-2])) == A(rm.power(np.full(5000, 2.0), y[::-2]))).all()

def test_complex():
    z = (np.linspace(-2, 2, 40)[:, None] + 1j * np.linspace(-1.9, 2.1, 40)).ravel()
    for name in ['exp', 'log', 'sqrt', 'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh', 'arcsin', 'arccos', 'arctan', 'arcsinh', 'arccosh',
                 'arctanh']:
        for dt in [np.complex64, np.complex128]:
            got, want = A(getattr(rm, name)(z.astype(dt))), getattr(np, name)(z.astype(dt).astype(np.complex128))
            assert got.dtype == dt, name
            # NumPy's complex functions err by an ulp or two themselves, so this bound is loose. The Rust tests compare with mpmath.
            assert (np.abs(got - want) <= 16 * np.finfo(got.real.dtype).eps * np.abs(want)).all(), (name, dt)
    assert np.allclose(rm.power(z, 0.5 + 1j), z**(0.5 + 1j)) and A(rm.absolute(z)).dtype == np.float64
    assert np.allclose(rm.absolute(z), np.abs(z)) and np.allclose(rm.angle(z), np.angle(z)) and np.allclose(rm.sign(z), z / np.abs(z))
    assert rm.sign(-2.5) == -1 and rm.angle(-1.0) == np.pi

def test_integers():
    a, b = np.arange(-30, 30), np.arange(1, 61)[::-1] - 25
    assert (A(rm.gcd(a, b)) == np.gcd(a, b)).all() and (A(rm.lcm(a, b)) == np.sign(a * b) * np.lcm(a, b)).all() and rm.gcd(a, b).dtype == 'i64'
    assert (A(rm.remainder(a, b + 100)) == np.remainder(a, b + 100)).all()
    assert (A(rm.power(a, 3)) == a**3).all() and rm.power(a, 3).dtype == 'i64'
    # An integer result that doesn't fit raises, and a negative integer exponent has no integer power.
    with pytest.raises(OverflowError): rm.power(np.int64(10), 30)
    with pytest.raises(ValueError, match='negative'): rm.power(np.array([2, 3]), np.array([1, -1]))
    # Each integer width keeps its type.
    for t in [np.uint8, np.int16, np.int32, np.int64]:
        x, y = np.arange(1, 12).astype(t), np.arange(12, 1, -1).astype(t)
        for name in ['add', 'multiply', 'maximum', 'minimum', 'remainder', 'gcd']:
            got, want = A(getattr(rm, name)(x, y)), getattr(np, name)(x, y)
            assert got.dtype == want.dtype and (got == want).all(), (name, t)
        for name in ['floor', 'ceil', 'absolute', 'sign']:
            got, want = A(getattr(rm, name)(x)), getattr(np, name)(x)
            assert got.dtype == want.dtype and (got == want).all(), (name, t)
        assert A(rm.equal(x, y)).dtype == bool and (A(rm.less(x, y)) == (x < y)).all(), t
        with pytest.raises(OverflowError): rm.add(np.array([np.iinfo(t).max], t), np.array([1], t))
    # Integers compare exactly, where float64 would round them to one value.
    assert not rm.equal(10**15, 10**15 + 1) and rm.less(2**62, 2**62 + 1)
    # A Python float beside integers promotes them to float64, and the comparison then uses the tolerance.
    assert A(rm.add(np.arange(3, dtype=np.int16), 0.5)).dtype == np.float64
    assert rm.equal(np.arange(3), 1 + 1e-15).tolist() == [False, True, False]
    # Arrays of different dtypes promote as NumPy's do. NumPy's other integer widths have no rustymath dtype.
    for t, u in [(np.int16, np.int32), (np.bool_, np.uint8), (np.uint8, np.float16), (np.int16, np.float16), (np.int32, np.float32), (np.float64, np.complex64)]:
        x, y = np.arange(3).astype(t), np.arange(3).astype(u)
        assert A(rm.add(x, y)).dtype == np.result_type(x, y), (t, u)
    with pytest.raises(ValueError): rm.add(np.arange(3, dtype=np.uint64), 1)
    # A signed type's least value has no magnitude of its type.
    with pytest.raises(OverflowError): rm.absolute(np.array([-32768], np.int16))
    p, q = np.array([True, True, False, False]), np.array([True, False, True, False])
    for name in ['maximum', 'minimum', 'logical_and', 'logical_or', 'logical_xor', 'logical_xnor']:
        got, want = A(getattr(rm, name)(p, q)), getattr(np, 'equal' if name == 'logical_xnor' else name)(p, q)
        assert got.dtype == bool and (got == want).all(), name
    # Booleans add and multiply as int64. Logical functions take only Booleans.
    assert rm.add(p, q).tolist() == [2, 1, 1, 0] and rm.multiply(p, q).dtype == 'i64'
    with pytest.raises(ValueError): rm.logical_and([1, 2], [3, 4])
    assert A(rm.floor(p)).dtype == bool and A(rm.absolute(p)).dtype == bool

def test_reductions(threads):
    x = np.random.default_rng(0).random(10_001)
    assert np.isclose(rm.sum(x), x.sum()) and isinstance(rm.sum(x), float) and np.allclose(rm.cumsum(x), np.cumsum(x))
    assert np.isclose(rm.prod(1 + x[:100] / 100), np.prod(1 + x[:100] / 100))
    m = x[:10_000].reshape(100, 100)
    assert np.allclose(rm.sum(m, axis=0), m.sum(0)) and np.allclose(rm.sum(m, axis=-1), m.sum(1)) and np.isclose(rm.sum(m, axis=(0, 1)), m.sum())
    assert np.allclose(rm.cumsum(m, axis=1), m.cumsum(1)) and rm.cumsum(m).shape == (10_000,)
    v = np.array([3.0, 1.0, np.nan, 5.0])
    most = rm.max(v)
    assert np.isnan(most) or most == 5
    assert rm.max(x[::7]) == x[::7].max() and rm.min(x) == x.min() and rm.minimum(np.nan, 1.0) == 1 and np.isnan(rm.minimum(1.0, np.nan))
    # Integers and Booleans sum as int64, and a total that doesn't fit raises.
    assert rm.sum(np.array([100, 100], np.uint8)) == 200 and rm.sum([True, True, False]) == 2
    with pytest.raises(OverflowError): rm.sum(np.full(3, 2**62))
    assert rm.max(np.array([3, -7, 5], np.int16)) == 5
    b = np.random.default_rng(0).integers(0, 2, (17, 1001)).astype(bool)
    for axis in [0, 1]:
        assert np.array_equal(rm.all(b, axis=axis), b.all(axis)) and np.array_equal(rm.any(b, axis=axis), b.any(axis))
    assert rm.all(np.array([], bool)) and not rm.any(np.array([], bool))
    with pytest.raises(ValueError): rm.all([1, 2])
    for dt in [np.complex64, np.complex128]:
        z = (1 + np.arange(9001) * (1e-5 + 2e-5j)).astype(dt)
        assert isinstance(rm.sum(z), complex) and np.allclose(rm.sum(z), z.sum(), rtol=2e-5)
        assert A(rm.sum(z[None], axis=1)).dtype == dt
        assert np.allclose(rm.cumprod(z[:100][::-2]), np.cumprod(z[:100][::-2]), rtol=2e-5)
    x = np.linspace(1, 2, 1_048_580)
    for view in [x, x[::2], x[::-1]]:
        assert np.isclose(rm.sum(view), view.sum())
    # float16 sums and products run in float32, and give float32.
    half = np.resize(np.array([1, 2**-11, -1, 2**-11], np.float16), len(x))
    assert rm.sum(half) == half.astype(np.float32).sum()
    factors = 1 + (x - 1.5) * 1e-5
    assert np.isclose(rm.prod(factors), factors.prod())
    assert rm.max(x[::-2]) == x[-1] and rm.min(x) == x[0]

def test_tolerant():
    assert rm.equal(0.1 + 0.2, 0.3) and not rm.equal(0.1 + 0.2, 0.3, ct=0)
    assert not rm.equal(1, 1 + 3e-14) and rm.equal([1, 2], 1 + 3e-14, ct=1e-13).tolist() == [True, False]
    assert not rm.less(0.3, 0.1 + 0.2) and rm.less(0.3, 0.1 + 0.2, ct=0) and rm.greater_equal(0.3, 0.1 + 0.2) and not rm.not_equal(0.3, 0.1 + 0.2)
    assert rm.floor(0.5 + 0.5 - 1e-16) == 1 and rm.ceil(1 + 1e-15) == 1 and rm.floor(1 - 1e-15, ct=0) == 0
    assert rm.remainder([-5.0, 5.0], 3).tolist() == np.remainder([-5.0, 5.0], 3).tolist() and rm.remainder(0.3, 0.1) == 0
    assert abs(rm.gcd(0.6, 0.4) - 0.2) < 1e-15 and abs(rm.lcm(-0.6, 0.4) + 1.2) < 1e-14
    with pytest.raises(ValueError): rm.equal(1, 1, ct=1)

def test_special():
    from scipy import special
    x = np.linspace(0.1, 5, 200)
    for name in ['gamma', 'erf']: assert ulps(getattr(rm, name)(x), getattr(special, name)(x)).max() <= 4, name
    want = special.gammaln(x)
    assert (np.abs(A(rm.gammaln(x)) - want) <= 4 * np.spacing(np.maximum(np.abs(want), 2))).all()
    p = np.linspace(0.01, 0.99, 99)
    assert np.allclose(rm.erfinv(p), special.erfinv(p)) and np.allclose(rm.erfcinv(p), special.erfcinv(p))
    assert np.allclose(rm.gammainc(2.5, x), special.gammainc(2.5, x)) and np.allclose(rm.gammaincc(2.5, x), special.gammaincc(2.5, x))
    assert np.allclose(rm.gammaincinv(2.5, p), special.gammaincinv(2.5, p))
    assert np.allclose(rm.betainc(2, 3, p), special.betainc(2, 3, p)) and np.allclose(rm.betaincinv(2, 3, p), special.betaincinv(2, 3, p))
    # SciPy's erfc is up to 16 ulps from mpmath near 4.8, so this check is loose. The Rust tests compare erfc with mpmath.
    assert np.allclose(rm.erfc(x), special.erfc(x), rtol=1e-14, atol=0)
    w = np.linspace(-0.36, 20, 300)
    assert np.allclose(rm.lambertw(w), special.lambertw(w).real) and np.isnan(rm.lambertw(-1.0))
    assert np.allclose(rm.lambertw(w + 1j), special.lambertw(w + 1j))
    n, k = np.linspace(-5.5, 30, 50), np.linspace(0, 12, 50)
    assert np.allclose(rm.binom(n, k), special.binom(n, k)) and rm.binom(10.0, 3.0) == 120
    z = x[::10] + 1j * x[::-10]
    assert np.allclose(rm.loggamma(z), special.loggamma(z)) and np.allclose(rm.gamma(z), special.gamma(z))
    # A real argument to `loggamma` computes as complex, and `gammaln` takes only reals.
    assert rm.loggamma(x).dtype == 'c128' and np.allclose(rm.loggamma(x), special.loggamma(x))
    with pytest.raises(ValueError): rm.gammaln(z)

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

def test_scalar_functions_and_integer_binomial():
    for dt in [np.float16, np.float32, np.float64]:
        x = np.array([0, 1, 2, 5], dtype=dt)
        assert A(rm.factorial(x)).dtype == dt and np.allclose(rm.factorial(x), [1, 1, 2, 120])
        assert A(rm.logb(dt(2), dt([1, 2, 8]))).dtype == dt and np.allclose(rm.logb(dt(2), dt([1, 2, 8])), [0, 1, 3])
        assert A(rm.cis(x)).dtype == np.result_type(dt, np.complex64) and np.allclose(rm.cis(x), np.exp(1j * x.astype(float)), atol=1e-6)
    for dt in [np.float32, np.float64]:
        e = A(rm.cis(dt([-0.0, np.pi / 2, np.nan])))
        assert np.signbit(e[0].imag) and e[1] == 1j and np.isnan(e[2])
        assert np.isnan(A(rm.cis(dt([np.inf])))).all()
        assert np.signbit(A(rm.tanh(np.array([complex(1, -0.0)]).astype(np.result_type(dt, np.complex64)))).imag).all()
    z = np.array([1 + 2j, 2 - 1j])
    assert np.allclose(rm.logb(2 + 1j, z), np.log(z) / np.log(2 + 1j)) and np.allclose(rm.cis(z), np.exp(1j * z))
    assert np.allclose(rm.divide(z, 2 - 1j), z / (2 - 1j)) and np.allclose(rm.divide([1., 2.], 2), [0.5, 1.])
    # Integers divide as floats.
    assert rm.divide([1, 2], 4).tolist() == [0.25, 0.5]
    for zero in [0., 0j]:
        assert np.isnan(rm.divide(zero, zero)) and np.isnan(rm.divide(np.nan + zero, zero))
    for dt in [np.uint8, np.int16, np.int32, np.int64]:
        got = rm.binom(dt([10, 5, 3]), dt([3, 0, 4]))
        assert A(got).dtype == dt and got.tolist() == [120, 1, 0]
    assert rm.binom(-3, 3) == -10 and rm.binom(66, 33) == math.comb(66, 33)
    for n, k in (np.uint8(11), np.uint8(5)), (np.int64(67), np.int64(33)):
        with pytest.raises(OverflowError): rm.binom(n, k)
    with pytest.raises(ValueError, match='negative'): rm.binom(3, -1)

def test_minmax_and_truth_table_scan():
    for dt in [np.bool_, np.uint8, np.int16, np.float16, np.float32, np.float64]:
        x = np.arange(30).reshape(5, 6).astype(dt)[:, ::-2]
        for axis in [0, -1]:
            lo, hi = map(A, rm.minmax(x, axis=axis))
            assert lo.dtype == hi.dtype == dt and np.array_equal(lo, x.min(axis=axis)) and np.array_equal(hi, x.max(axis=axis))
    lo, hi = rm.minmax([1., np.nan, 2.])
    assert (np.isnan(lo) or lo == 1) and (np.isnan(hi) or hi == 2)
    b = np.array([[True, False, True], [False, True, True]])
    for bits in range(16):
        table = np.array([(bits >> k) & 1 for k in range(4)], dtype=bool).reshape(2, 2)
        for axis in [0, 1]:
            for start in [None, True]:
                want = np.moveaxis(b, axis, -1).copy()
                for row in want:
                    if start is not None: row[0] = table[int(start), int(row[0])]
                    for i in range(1, len(row)): row[i] = table[int(row[i - 1]), int(row[i])]
                assert np.array_equal(rm.logical_scan(b, table.tolist(), axis=axis, start=start), np.moveaxis(want, -1, axis))
    # With no axis, the scan runs along the flattened array.
    xor = [[False, True], [True, False]]
    assert rm.logical_scan(b, xor).tolist() == np.logical_xor.accumulate(b.ravel()).tolist()
    assert rm.logical_scan(np.empty((2, 0), bool), xor, axis=-1).shape == (2, 0)
    with pytest.raises(IndexError, match='axis'): rm.logical_scan(b, xor, axis=2)

def test_elementwise_layouts(threads):
    x = np.linspace(0.1, 1, 262_147)
    for view in [x, x[::2], x[::-1], np.broadcast_to(0.5, x.shape)]:
        assert np.allclose(rm.exp(view), np.exp(view), rtol=2e-15)
    # Arguments pair their leading axes, so a vector pairs with the rows of a matrix.
    m = x[:6].reshape(2, 3)
    assert np.allclose(rm.add(m, [10, 20]), m + np.array([[10], [20]]))

def test_parallel_errors(threads):
    x = np.ones(262_147)
    x[-1] = -1
    powers = np.full(len(x), 3, np.int64)
    powers[-1] = 63
    with pytest.raises(OverflowError): rm.power(np.full(len(x), 2, np.int64), powers)
    y = A(rm.log(x))
    assert np.isnan(y[-1]) and np.allclose(y[:-1], 0)
    assert np.isnan(A(rm.log(np.full(len(x), np.nan)))).all()
