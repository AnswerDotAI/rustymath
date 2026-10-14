import math
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
    assert np.allclose(rm.power(z, 0.5 + 1j), z**(0.5 + 1j)) and rm.absolute(z).dtype == np.float64
    assert np.allclose(rm.absolute(z), np.abs(z)) and np.allclose(rm.angle(z), np.angle(z)) and np.allclose(rm.sign(z), z / np.abs(z))
    assert rm.sign(-2.5) == -1 and rm.angle(-1.0) == np.pi

def test_integers():
    a, b = np.arange(-30, 30), np.arange(1, 61)[::-1] - 25
    assert (rm.gcd(a, b) == np.gcd(a, b)).all() and (rm.lcm(a, b) == np.sign(a * b) * np.lcm(a, b)).all() and rm.gcd(a, b).dtype == np.int64
    assert (rm.remainder(a, b + 100) == np.remainder(a, b + 100)).all()
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

def test_reductions(threads):
    x = np.random.default_rng(0).random(10_001)
    assert np.isclose(rm.add.reduce(x), x.sum()) and np.allclose(rm.add.accumulate(x), np.cumsum(x))
    assert np.isclose(rm.multiply.reduce(1 + x[:100] / 100), np.prod(1 + x[:100] / 100))
    m = x[:10_000].reshape(100, 100)
    assert np.allclose(rm.add.reduce(m, axis=0), m.sum(0)) and np.allclose(rm.add.reduce(m, axis=1), m.sum(1))
    v = np.array([3.0, 1.0, np.nan, 5.0])
    most, least = rm.maximum.reduce(v), rm.minimum.accumulate(v)
    assert (np.isnan(most) or most == 5) and all(np.isnan(g) or g == w for g, w in zip(least, [3.0, 1.0, 1.0, 1.0]))
    assert rm.maximum.reduce(x[::7]) == x[::7].max() and rm.minimum(np.nan, 1.0) == 1 and np.isnan(rm.minimum(1.0, np.nan))
    assert rm.power.reduce([2.0, 3.0, 2.0]) == 64 and rm.power.accumulate([2.0, 3.0, 2.0]).tolist() == [2, 8, 64]
    big = np.array([2**62, 2**62, 2**62])
    assert rm.add.reduce(big) == np.add.reduce(big) and (rm.add.accumulate(big) == np.add.accumulate(big)).all()
    b = np.random.default_rng(0).integers(0, 2, (17, 1001)).astype(bool)
    for name in ['logical_and', 'logical_or', 'logical_xor', 'logical_xnor']:
        op = getattr(rm, name)
        ref = np.equal if name == 'logical_xnor' else getattr(np, name)
        assert op(b, ~b).dtype == bool and np.array_equal(op(b, ~b), ref(b, ~b))
        for axis in [0, 1]:
            assert np.array_equal(op.reduce(b[:, ::-1], axis=axis), ref.reduce(b[:, ::-1], axis=axis))
            assert np.array_equal(op.accumulate(b[:, ::-1], axis=axis), ref.accumulate(b[:, ::-1], axis=axis))
        assert op.reduce(np.array([], dtype=bool)) == (name in ['logical_and', 'logical_xnor'])
    for dt in [np.complex64, np.complex128]:
        z = (1 + np.arange(9001) * (1e-5 + 2e-5j)).astype(dt)
        for op, ref in [(rm.add, np.add), (rm.multiply, np.multiply)]:
            x = z if op is rm.add else z[:100]
            assert op.reduce(x).dtype == dt and np.allclose(op.reduce(x), ref.reduce(x), rtol=2e-5)
            assert np.allclose(op.accumulate(x[::-2]), ref.accumulate(x[::-2]), rtol=2e-5)
    x = np.linspace(1, 2, 1_048_580)
    half = np.resize(np.array([1, 2**-11, -1, 2**-11], np.float16), len(x))
    for view in [x, x[::2], x[::-1], x * (1 + 2j)]:
        assert np.allclose(rm.add.reduce(view, initial=3), view.sum() + 3)
    assert rm.add.reduce(half, initial=np.float16(0.25)) == np.float16(half.astype(np.float32).sum() + 0.25)
    ints = np.full(len(x), 2**62, np.int64)
    assert rm.add.reduce(ints) == np.add.reduce(ints)
    factors = 1 + (x - 1.5) * 1e-5
    assert np.isclose(rm.multiply.reduce(factors), factors.prod())
    assert rm.maximum.reduce(x[::-2]) == x[-1] and rm.minimum.reduce(x) == x[0]
    b = np.arange(len(x)) % 3 == 0
    assert rm.logical_xor.reduce(b) == np.logical_xor.reduce(b)
    assert np.array_equal(rm.logical_xnor.accumulate(b), np.equal.accumulate(b))
    assert np.allclose(rm.subtract.accumulate(x[:8193]), np.subtract.accumulate(x[:8193]))
    out = x[:8193].copy()
    rm.add.accumulate(out, out=out)
    assert np.allclose(out, x[:8193].cumsum())

def test_tolerant():
    assert rm.equal(0.1 + 0.2, 0.3) and not rm.equal(0.1 + 0.2, 0.3, ct=0)
    assert not rm.equal(1, 1 + 3e-14) and rm.equal([1, 2], 1 + 3e-14, ct=1e-13).tolist() == [True, False]
    assert not rm.less(0.3, 0.1 + 0.2) and rm.less(0.3, 0.1 + 0.2, ct=0) and rm.greater_equal(0.3, 0.1 + 0.2) and not rm.not_equal(0.3, 0.1 + 0.2)
    assert rm.floor(0.5 + 0.5 - 1e-16) == 1 and rm.ceil(1 + 1e-15) == 1 and rm.floor(1 - 1e-15, ct=0) == 0
    assert rm.remainder([-5.0, 5.0], 3).tolist() == np.remainder([-5.0, 5.0], 3).tolist() and rm.remainder(0.3, 0.1) == 0
    assert abs(rm.gcd(0.6, 0.4) - 0.2) < 1e-15 and abs(rm.lcm(-0.6, 0.4) + 1.2) < 1e-14
    with pytest.raises(ValueError): rm.equal(1, 1, ct=1)
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): rm.log(-1.0)

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

def test_scalar_functions_and_integer_binomial():
    for dt in [np.float16, np.float32, np.float64]:
        x = np.array([0, 1, 2, 5], dtype=dt)
        assert rm.factorial(x).dtype == dt and np.allclose(rm.factorial(x), [1, 1, 2, 120])
        assert rm.logb(dt(2), dt([1, 2, 8])).dtype == dt and np.allclose(rm.logb(dt(2), dt([1, 2, 8])), [0, 1, 3])
        assert rm.cis(x).dtype == np.result_type(dt, np.complex64) and np.allclose(rm.cis(x), np.exp(1j * x.astype(float)), atol=1e-6)
    for dt in [np.float32, np.float64]:
        e = rm.cis(dt([-0.0, np.pi / 2, np.nan]))
        assert np.signbit(e[0].imag) and e[1] == 1j and np.isnan(e[2])
        with pytest.warns(RuntimeWarning): assert np.isnan(rm.cis(dt([np.inf]))).all()
        assert np.signbit(rm.tanh(np.array([complex(1, -0.0)]).astype(np.result_type(dt, np.complex64))).imag).all()
    z = np.array([1 + 2j, 2 - 1j])
    assert np.allclose(rm.logb(2 + 1j, z), np.log(z) / np.log(2 + 1j)) and np.allclose(rm.cis(z), np.exp(1j * z))
    assert np.allclose(rm.divide(z, 2 - 1j), z / (2 - 1j)) and np.allclose(rm.divide([1., 2.], 2), [0.5, 1.])
    for zero in [0., 0j]:
        with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): rm.divide(zero, zero)
        with np.errstate(invalid='raise'): assert np.isnan(rm.divide(np.nan + zero, zero))
    assert rm.subtract.reduce([10, 3, 2]) == 5 and rm.subtract.accumulate([10, 3, 2]).tolist() == [10, 7, 5]
    for dt in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32, np.int64, np.uint64]:
        got = rm.binom(dt([10, 5, 3]), dt([3, 0, 4]))
        assert got.dtype == dt and got.tolist() == [120, 1, 0]
    assert rm.binom(np.int64(-3), np.int64(3)) == -10
    assert rm.binom(np.uint64(67), np.uint64(33)) == math.comb(67, 33)
    for n, k in (np.int8(11), np.int8(5)), (np.int64(67), np.int64(33)), (np.uint64(68), np.uint64(34)), (np.int64(3), np.int64(-1)):
        with np.errstate(over='raise'), pytest.raises(FloatingPointError): rm.binom(n, k)

def test_minmax_and_truth_table_scan():
    for dt in [np.bool_, np.int8, np.uint64, np.float16, np.float32, np.float64]:
        x = np.arange(30).reshape(5, 6).astype(dt)[:, ::-2]
        for axis in [0, -1]:
            lo, hi = rm.minmax(x, axis=axis)
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
    assert rm.logical_scan(np.empty((2, 0), bool), [[False, True], [True, False]]).shape == (2, 0)
    with pytest.raises(ValueError, match='axis'): rm.logical_scan(b, [[False, True], [True, False]], axis=2)

def test_elementwise_layouts(threads):
    x = np.linspace(0.1, 1, 262_147)
    for view in [x, x[::2], x[::-1], np.broadcast_to(0.5, x.shape)]:
        assert np.allclose(rm.exp(view), np.exp(view), rtol=2e-15)
    out = x.copy()
    rm.exp(out, out=out)
    assert np.allclose(out, np.exp(x), rtol=2e-15)
    out = np.zeros(2 * len(x))
    rm.power(1.001, x, out=out[::2])
    assert np.allclose(out[::2], 1.001**x) and (out[1::2] == 0).all()
    rm.exp(x, out=out[:len(x)][::-1])
    assert np.allclose(out[:len(x)][::-1], np.exp(x), rtol=2e-15)
    # Shifted aliases need NumPy's copy; exact aliases and negative strides need no cross-range dependency.
    out = np.arange(len(x) + 1, dtype=float)
    rm.add(out[:-1], 1, out=out[1:])
    assert np.array_equal(out, np.arange(len(out), dtype=float))
    out = np.zeros(1)
    repeated = np.lib.stride_tricks.as_strided(out, shape=x.shape, strides=(0,))
    rm.exp(x, out=repeated)
    assert np.isclose(out[0], np.exp(x[-1]))
    z = x + 1j * x
    rm.absolute(z, out=z.real)
    assert np.allclose(z.real, np.sqrt(2) * x)
    out, mask = np.full_like(x, -2), np.arange(len(x)) % 3 == 0
    rm.sin(x, out=out, where=mask)
    assert np.allclose(out[mask], np.sin(x[mask])) and (out[~mask] == -2).all()

def test_parallel_and_buffered_errors(threads):
    x = np.ones(262_147)
    x[-1] = -1
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): rm.log(x)
    powers = np.full(len(x), 3, np.int64)
    powers[-1] = 63
    with np.errstate(over='raise'), pytest.raises(FloatingPointError): rm.power(np.full(len(x), 2, np.int64), powers)
    # Integer-to-float casting splits the call into NumPy buffers; a later successful callback must not erase this failure.
    buffered = np.zeros(100_003, np.int64)
    buffered[0] = -1
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): rm.lambertw(buffered)
    with np.errstate(all='raise'):
        assert np.allclose(rm.log(np.ones(len(x))), 0)
        assert np.isnan(rm.log(np.full(len(x), np.nan))).all()
