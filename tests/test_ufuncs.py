import math
import numpy as np, pytest, rustymath as rm
from numpy.lib.stride_tricks import sliding_window_view as windows

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
    assert isinstance(rm.exp(1), float) and rm.exp(np.float32(1)).dtype == 'float32' and rm.exp(np.float32(1)).shape == ()
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
    assert (A(rm.gcd(a, b)) == np.gcd(a, b)).all() and (A(rm.lcm(a, b)) == np.sign(a * b) * np.lcm(a, b)).all() and rm.gcd(a, b).dtype == 'int64'
    assert (A(rm.remainder(a, b + 100)) == np.remainder(a, b + 100)).all()
    assert (A(rm.power(a, 3)) == a**3).all() and rm.power(a, 3).dtype == 'int64'
    # An integer power that doesn't fit wraps, as NumPy's does. A negative exponent has no integer power beside a base other than 1 or -1, whatever other pairs give, and a padded view's fill counts as an exponent.
    assert rm.power(np.array([10, 3]), 30).tolist() == np.power(np.array([10, 3]), 30).tolist() and rm.power(np.array([3], np.uint8), 6).tolist() == [729 % 256]
    with pytest.raises(ValueError, match='negative'): rm.power(np.array([2, 3]), np.array([1, -1]))
    with pytest.raises(ValueError, match='negative'): rm.power(2, rm.pad(rm.array([1, 2]), 1, fill=-1))
    assert rm.power(np.array([1, 2]), np.array([-1, 63])).tolist() == [1, -2**63]
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
        assert rm.add(np.array([np.iinfo(t).max], t), np.array([1], t)).tolist() == [np.iinfo(t).min], t
    # Integers compare exactly, where float64 would round them to one value.
    assert not rm.equal(10**15, 10**15 + 1) and rm.less(2**62, 2**62 + 1)
    # A Python float beside integers promotes them to float64, so a tolerant comparison applies to them.
    assert A(rm.add(np.arange(3, dtype=np.int16), 0.5)).dtype == np.float64
    assert rm.equal(np.arange(3), 1 + 1e-15).tolist() == [False, False, False]
    # Arrays of different dtypes promote to the higher kind, wide enough for both. An integer beside a float takes the float's dtype, as in JAX, where NumPy widens the float. NumPy's other integer widths have no rustymath dtype.
    for t, u, want in [(np.int16, np.int32, np.int32), (np.bool_, np.uint8, np.uint8), (np.uint8, np.float16, np.float16), (np.int16, np.float16, np.float16), (np.int32, np.float32, np.float32), (np.float64, np.complex64, np.complex128)]:
        x, y = np.arange(3).astype(t), np.arange(3).astype(u)
        assert A(rm.add(x, y)).dtype == want, (t, u)
    assert rm.add(np.arange(3, dtype=np.uint64), 1).tolist() == [1, 2, 3]
    # A signed type's least value has no magnitude of its type, and wraps to itself.
    assert rm.absolute(np.array([-32768], np.int16)).tolist() == [-32768]
    p, q = np.array([True, True, False, False]), np.array([True, False, True, False])
    for name in ['maximum', 'minimum', 'logical_and', 'logical_or', 'logical_xor', 'logical_xnor']:
        got, want = A(getattr(rm, name)(p, q)), getattr(np, 'equal' if name == 'logical_xnor' else name)(p, q)
        assert got.dtype == bool and (got == want).all(), name
    # Booleans add and multiply as int64. Logical functions take only Booleans.
    assert rm.add(p, q).tolist() == [2, 1, 1, 0] and rm.multiply(p, q).dtype == 'int64'
    with pytest.raises(ValueError): rm.logical_and([1, 2], [3, 4])
    assert A(rm.floor(p)).dtype == bool and A(rm.absolute(p)).dtype == bool

def test_characters():
    "Characters compute as their code points. A number orders before a character, and an integer offsets one."
    t = lambda x: A(x).tolist()
    assert t(rm.less('abc', 'bcb')) == [True, True, False] and t(rm.maximum('abc', 'bab')) == ['b', 'b', 'c']
    assert t(rm.less(3, 'ab')) == [True, True] and t(rm.minimum('ab', 3)) == [3, 3] and t(rm.equal(97, 'a')) == [False]
    assert t(rm.add('ab', 1)) == ['b', 'c'] and t(rm.add(1, 'ab')) == ['b', 'c'] and t(rm.subtract('bc', 'ab')) == [1, 1]
    for bad in [lambda: rm.add('ab', 'ab'), lambda: rm.add('a', 1.5), lambda: rm.subtract(1, 'a')]:
        with pytest.raises(ValueError): bad()

def test_reductions(threads):
    x = np.random.default_rng(0).random(10_001)
    assert np.isclose(rm.sum(x), x.sum()) and isinstance(rm.sum(x), float) and np.allclose(rm.cumsum(x), np.cumsum(x))
    assert np.isclose(rm.prod(1 + x[:100] / 100), np.prod(1 + x[:100] / 100))
    m = x[:10_000].reshape(100, 100)
    assert np.allclose(rm.sum(m, axis=0), m.sum(0)) and np.allclose(rm.sum(m, axis=-1), m.sum(1)) and np.isclose(rm.sum(m, axis=(0, 1)), m.sum())
    assert np.allclose(rm.cumsum(m, axis=1), m.cumsum(1)) and rm.cumsum(m).shape == (10_000,)
    v = np.array([3.0, 1.0, np.nan, 5.0])
    got = [rm.max(v), rm.minimum(np.nan, 1.0), rm.minimum(1.0, np.nan)]
    assert all(np.isnan(g) or g == w for g, w in zip(got, [5, 1, 1]))
    assert rm.max(x[::7]) == x[::7].max() and rm.min(x) == x.min()
    # Integers and Booleans sum as int64, and a total that doesn't fit wraps.
    assert rm.sum(np.array([100, 100], np.uint8)) == 200 and rm.sum([True, True, False]) == 2
    assert rm.sum(np.full(3, 2**62)) == np.full(3, 2**62).sum()
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
    # A view that reads a block of its storage, such as a transpose, a reversal or swapped axes, folds the block in place. Its totals come back in the view's order.
    s = np.random.default_rng(1).random((3, 4, 5))
    t, n = rm.array(s), rm.array(np.arange(60).reshape(3, 4, 5))
    for view, ref in [(t.T, s.T), (t[::-1, :, ::-1], s[::-1, :, ::-1]), (rm.swapaxes(t, 0, 2), s.swapaxes(0, 2))]:
        for axis in [None, 0, 1, 2, (0, 2)]:
            assert np.allclose(rm.sum(view, axis=axis), ref.sum(axis)) and np.array_equal(A(rm.max(view, axis=axis)), ref.max(axis))
            assert np.array_equal(A(rm.minmax(view, axis=axis)[0]), ref.min(axis))
    assert np.array_equal(A(rm.sum(rm.swapaxes(n, 0, 2), axis=1)), np.arange(60).reshape(3, 4, 5).swapaxes(0, 2).sum(1))
    assert np.array_equal(A(rm.any(rm.array(b).T, axis=0)), b.T.any(0))

def test_mathish():
    from rustymath import mathish as mm
    assert not rm.equal(0.1 + 0.2, 0.3)
    assert np.isnan(rm.divide(0., 0.)) and np.isnan(rm.floor_divide(0., 0.)) and np.isnan(rm.logb(1., 1.))
    assert mm.equal(0.1 + 0.2, 0.3) and not mm.equal(1, 1 + 3e-14)
    assert mm.equal(np.arange(3), 1 + 1e-15).tolist() == [False, True, False]
    assert not mm.less(0.3, 0.1 + 0.2) and mm.greater_equal(0.3, 0.1 + 0.2) and not mm.not_equal(0.3, 0.1 + 0.2)
    assert mm.floor(0.5 + 0.5 - 1e-16) == 1 and mm.ceil(1 + 1e-15) == 1
    assert mm.remainder([-5.0, 5.0], 3).tolist() == np.remainder([-5.0, 5.0], 3).tolist() and mm.remainder(0.3, 0.1) == 0
    assert abs(mm.gcd(0.6, 0.4) - 0.2) < 1e-15 and abs(mm.lcm(-0.6, 0.4) + 1.2) < 1e-14
    assert len(mm.unique([0.3, 0.1 + 0.2]).values) == 1 and 0.1 + 0.2 not in rm.array([0.3])
    assert mm.divide([0., 1.], [0., 0.]).tolist() == [0., np.inf] and mm.floor_divide(0., 0.) == 0
    assert mm.divide(0, 0) == 0 and mm.divide(np.zeros(1, np.complex128), 0).tolist() == [0j]
    assert mm.logb(1., 1.) == 0 and mm.logb(np.ones(2), np.ones(2)).tolist() == [0., 0.]
    assert mm.logb(1., np.ones(2)).tolist() == mm.logb(np.ones(2), 1.).tolist() == [0., 0.]
    assert mm.logb(np.ones(1, np.complex128), 1).tolist() == [0j]
    assert np.isnan(mm.logb(np.float32(1), np.float32(1)))
    for dtype in (np.float16, np.float32, np.complex64): assert np.isnan(mm.divide(np.zeros(1, dtype), 0)).all()
    for dtype in (np.float16, np.float32): assert np.isnan(mm.floor_divide(np.zeros(1, dtype), 0)).all()
    with pytest.raises(OverflowError): mm.floor_divide(0, 0)
    assert rm.linalg.inner(rm.add, mm.divide)([0., 2.], [0., 2.]) == 1
    assert rm.linalg.inner(mm.divide, rm.add)([0., 0.], [0., 0.]) == 0
    assert rm.accumulate(mm.divide, [0., 0.]).tolist() == [0., 0.] and rm.at(mm.divide, [0.], [0], 0.).tolist() == [0.]
    assert np.isnan(mm.accumulate(rm.divide, [0., 0.])[-1]) and not rm.equal(0.1 + 0.2, 0.3)

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
    assert rm.loggamma(x).dtype == 'complex128' and np.allclose(rm.loggamma(x), special.loggamma(x))
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
    # Each running maximum or minimum is NaN or that of the items so far that aren't NaN: in a short lane, and in a long one whose float16 scan regroups its items.
    x = np.arange(80.)
    x[5] = np.nan
    for dt in [np.float16, np.float32, np.float64]:
        for a in [x.astype(dt), x[5:8].astype(dt)]:
            for name, f, start in [('maximum', np.fmax, -np.inf), ('minimum', np.fmin, np.inf)]:
                got, want = A(rm.accumulate(getattr(rm, name), a)), f.accumulate(np.r_[start, a])[1:].astype(dt)
                assert got.dtype == dt and (np.isnan(got) | (got == want)).all()
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

def test_native_accumulation():
    for dtype in [np.int16, np.float16, np.float32, np.float64, np.complex64]:
        x = np.arange(1, 7, dtype=dtype).reshape(2, 3)[:, ::-1]
        for name in ['add', 'subtract', 'multiply', 'divide']:
            for axis in [None, 0, -1]:
                a = x.ravel() if axis is None else x
                want = getattr(np, name).accumulate(a, axis=0 if axis is None else axis, dtype=np.float32 if dtype == np.float16 else None)
                assert np.allclose(rm.accumulate(getattr(rm, name), x, axis), want, rtol=1e-3)
    x = rm.array({'a': [3., 2, 1], 'b': [6., 5, 4]})
    x.names = ('row', 'column')
    result = rm.accumulate(rm.subtract, x, 'column', start=10)
    assert result.tolist() == [[7., 5, 4], [4., -1, -5]] and result.keys == x.keys and result.names == x.names
    assert rm.accumulate(rm.add, [1, 2, 3], start=10).tolist() == [11, 13, 16]
    assert rm.accumulate(rm.less, [3, 2, 1]).tolist() == [3, 0, 1]
    assert rm.accumulate(rm.less, [3, 2, 1], start=1).tolist() == [1, 1, 0]
    assert rm.accumulate(rm.logical_xor, [True, False, True], start=True).tolist() == [False, False, True]
    for name in ['bitwise_and', 'bitwise_or', 'bitwise_xor']:
        for x in [np.array([12, 10, 14, -3], np.int16), np.array([200, 100, 7], np.uint8), np.array([True, False, True])]:
            assert rm.accumulate(getattr(rm, name), x).tolist() == getattr(np, name).accumulate(x).tolist()
    assert rm.accumulate(rm.subtract, np.empty((2, 0)), axis=-1).shape == (2, 0)
    assert rm.accumulate(rm.add, [2**62] * 3).tolist() == np.cumsum([2**62] * 3).tolist()
    for f, start in [(np.add, None), (lambda a, b: a + b, None), (rm.add, [1])]:
        with pytest.raises(ValueError): rm.accumulate(f, [1, 2], start=start)
    # Each window of a sliding fold matches the same reduction of that window alone, narrow or wide, with each fold keyed as its window's first item.
    x = np.random.default_rng(0).random((6, 40))
    for f, g in [(rm.add, np.sum), (rm.multiply, np.prod), (rm.maximum, np.max), (rm.minimum, np.min)]:
        for width, axis in [(3, -1), (12, -1), (4, 0)]:
            assert np.allclose(rm.sliding(f, x, width, axis), g(windows(x, width, axis=axis), axis=-1))
    assert rm.sliding(rm.add, np.arange(10), 4).tolist() == windows(np.arange(10), 4).sum(-1).tolist()
    assert rm.sliding(rm.logical_or, [False, True, False, False], 2).tolist() == [True, True, False]
    assert rm.sliding(rm.maximum, rm.array({'a': 1., 'b': 3, 'c': 2}), 2).keys == (('a', 'b'),)
    # `sliding_argmax` and `sliding_argmin` give the position along the axis of each window's first maximum or minimum.
    ties = np.random.default_rng(1).integers(0, 4, (6, 40))
    for data in [x, ties, ties.astype(np.float16), ties > 1]:
        for width, axis in [(3, -1), (12, -1), (4, 0)]:
            w = windows(data, width, axis=axis)
            starts = np.indices(w.shape[:-1])[axis]
            assert np.array_equal(A(rm.sliding_argmax(data, width, axis)), w.argmax(-1) + starts)
            assert np.array_equal(A(rm.sliding_argmin(data, width, axis)), w.argmin(-1) + starts)
    # A window holding NaN picks its greatest other item or the NaN.
    picks = rm.sliding_argmax([1., np.nan, 0.], 2).tolist()
    assert picks[0] in (0, 1) and picks[1] in (1, 2)
    for width in [0, 41]:
        with pytest.raises(ValueError, match='window'): rm.sliding(rm.add, x, width)
    with pytest.raises(ValueError): rm.sliding(rm.subtract, x, 3)

def test_derivative():
    x = np.linspace(-2, 2, 9)
    value, slope = rm.value_and_derivative(rm.tanh, x)
    assert np.allclose(value, np.tanh(x)) and np.allclose(slope, 1 - np.tanh(x) ** 2)
    assert np.allclose(rm.derivative(rm.sin, x), np.cos(x))
    y = np.linspace(0.5, 3, 9)
    dx, dy = rm.derivative(rm.power, y, 2.5)
    assert np.allclose(dx, 2.5 * y ** 1.5) and np.allclose(dy, y ** 2.5 * np.log(y))
    assert rm.derivative(rm.multiply, 3., 4.) == (4., 3.)
    assert np.isclose(rm.derivative(rm.erf, 0.5), 2 / np.sqrt(np.pi) * np.exp(-0.25))
    da, dx = rm.derivative(rm.gammainc, 2., 1.)
    assert np.isnan(da) and np.isclose(dx, np.exp(-1)) and np.isclose(rm.derivative(rm.gammainc, 1., 0.)[1], 1)
    with pytest.raises(ValueError, match='floor has no derivative'): rm.derivative(rm.floor, x)
    with pytest.raises(ValueError, match='native elementwise'): rm.derivative(np.sin, x)

def test_explain():
    x = rm.array([[1., 2., 3.], [4., 5., 6.]])
    with rm.explain() as outer:
        rm.add(x.T, 1.)
        with rm.explain() as inner: rm.add(x.T, 2.)
        rm.add(x.T, 3.)
    for report, calls in [(outer, 3), (inner, 1)]:
        assert {'vector kernel', 'mapped kernel'} <= set(report.routes)
        record = next(r for r in report.records if r.route == 'mapped kernel')
        assert (record.kind, record.calls, record.items) == ('taken', calls, 6 * calls)

def test_view_folds(threads):
    # A view that isn't a block of its storage folds through its mapping, without being built. Runs of at least 128 items fold where they lie, and shorter runs are copied a piece at a time.
    m = np.random.default_rng(0).random((1024, 1024))
    r = rm.array(m)
    for cols, in_place in [(slice(None, 512), True), (slice(None, None, 2), False)]:
        with rm.explain() as report:
            assert np.isclose(rm.sum(r[:, cols]), m[:, cols].sum())
            assert all(np.allclose(rm.sum(r[:, cols], axis=a), m[:, cols].sum(a)) for a in [0, 1])
        assert 'mapped fold' in report.routes and ('in place' in report.routes) == in_place

def test_window_folds(threads):
    # A fold along an axis of a view of windows slides over the view's storage, with no windows built.
    x = np.random.default_rng(0).random((300, 400))
    with rm.explain() as report:
        assert np.allclose(rm.sum(rm.windows(x[0], 3), axis=1), windows(x[0], 3).sum(1))
        assert np.allclose(rm.max(rm.windows(x, (5, 5)), axis=3), windows(x, (5, 5)).max(3))
    assert 'sliding fold' in report.routes

def test_elementwise_layouts(threads):
    x = np.linspace(0.1, 1, 262_147)
    for view in [x, x[::2], x[::-1], np.broadcast_to(0.5, x.shape)]:
        assert np.allclose(rm.exp(view), np.exp(view), rtol=2e-15)
    # Arguments pair their leading axes, so a vector pairs with the rows of a matrix.
    m = x[:6].reshape(2, 3)
    assert np.allclose(rm.add(m, [10, 20]), m + np.array([[10], [20]]))
    for dtype in ('float32', 'float64'):
        a = rm.array(m, dtype=dtype)
        for view in (a.T, a[::-1], a[:, ::2]):
            dense = view.compact()
            for f in (rm.exp, rm.sum, rm.cumsum, rm.mean, rm.sort): assert np.allclose(f(view), f(dense))
            assert np.allclose(view + view, dense + dense) and np.allclose(rm.clip(view, 0.2, 0.8), rm.clip(dense, 0.2, 0.8))
            assert np.array_equal(rm.where(view > 0.5, view, -view), rm.where(dense > 0.5, dense, -dense))

def test_parallel_errors(threads):
    x = np.ones(262_147)
    x[-1] = -1
    counts = np.full(len(x), 3, np.int64)
    counts[-1] = 64
    with pytest.raises(OverflowError): rm.left_shift(np.ones(len(x), np.int64), counts)
    y = A(rm.log(x))
    assert np.isnan(y[-1]) and np.allclose(y[:-1], 0)
    assert np.isnan(A(rm.log(np.full(len(x), np.nan)))).all()
