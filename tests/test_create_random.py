import numpy as np, pytest, rustymath as rm

def test_constructors_and_dtype_facts():
    a = rm.array({'a': [1, 2], 'b': [3, 4]}, dtype='float32')
    a.names = ('row', 'col')
    for fn, value in ((rm.zeros_like, 0), (rm.ones_like, 1)):
        b = fn(a)
        assert b.dtype == a.dtype and b.shape == a.shape and b.keys == a.keys and b.names == a.names
        assert b.tolist() == [[value, value], [value, value]]
    assert rm.zeros((2, 0), dtype='int16').shape == (2, 0) and rm.ones(3, dtype='bool').tolist() == [True] * 3
    assert rm.full((2, 2), 2j).tolist() == [[2j, 2j], [2j, 2j]] and rm.full_like(a, 5, dtype='int16').dtype == 'int16'
    assert rm.zeros((), dtype='float16').shape == () and rm.zeros(2, dtype='character').tolist() == [' ', ' ']
    assert rm.arange(5).tolist() == list(range(5)) and rm.arange(5, -2, -2).tolist() == [5, 3, 1, -1]
    assert rm.arange(2**60, 2**60 + 3).tolist() == [2**60, 2**60 + 1, 2**60 + 2]
    assert rm.array([10**20], dtype='float64').tolist() == [1e20] and rm.full(2, 10**20, dtype='float64').tolist() == [1e20, 1e20]
    assert rm.array([1.7], dtype='int16').tolist() == [1]
    assert rm.arange(0, 1, .25, dtype='float32').tolist() == [0, .25, .5, .75]
    assert rm.linspace(0, 1, 5).tolist() == [0, .25, .5, .75, 1] and rm.linspace(0, 1, 4, endpoint=False).tolist() == [0, .25, .5, .75]
    assert rm.linspace(2, 3, 1).tolist() == [2] and rm.linspace(2, 3, 0).shape == (0,)
    np.testing.assert_allclose(rm.logspace(0, 3, 4), [1, 10, 100, 1000])
    for dtype in ('float16', 'float32', 'float64'):
        info, expected = rm.finfo(dtype), np.finfo(DTYPES[dtype])
        assert (info.bits, info.eps, info.tiny, info.max) == (expected.bits, expected.eps, expected.tiny, expected.max)
    assert rm.iinfo('uint8').min == 0 and rm.iinfo('int64').max == 2**63 - 1 and rm.finfo('complex64').bits == 32
    assert rm.issubdtype('float32', 'floating') and not rm.issubdtype('complex64', 'floating') and rm.result_type('int16', 'float16') == 'float16'
    for fn, args in ((rm.arange, (0, 1, 0)), (rm.full, (3, [1, 2])), (rm.finfo, ('int64',)), (rm.iinfo, ('float32',))):
        with pytest.raises(ValueError): fn(*args)
    with pytest.raises(OverflowError): rm.full(2, 300, dtype='uint8')

DTYPES = {'float16': np.float16, 'float32': np.float32, 'float64': np.float64}

def test_persistent_generator():
    from rustymath.random import default_rng
    g, same = default_rng(7), default_rng(7)
    first, second = g.normal(size=5), g.normal(size=5)
    assert first.tolist() == same.normal(size=5).tolist() and second.tolist() == same.normal(size=5).tolist()
    assert first.tolist() != second.tolist() and isinstance(g.random(), float) and g.random((2, 3)).shape == (2, 3)
    values = g.integers(-2, 3, size=100)
    assert values.dtype == 'int64' and values.min() >= -2 and values.max() < 3 and 0 <= g.integers(5) < 5
    assert g.integers(-(2**63), 2**63 - 1, size=10).shape == (10,)
    assert g.sample(rm.distribution.beta(2, 3), (2, 2)).shape == (2, 2) and g.exponential(size=3).min() >= 0
    assert sorted(g.permutation(10).tolist()) == list(range(10))
    keyed = rm.array({'a': 10, 'b': 20})
    assert set(g.permutation(keyed).keys[0]) == {'a', 'b'} and g.choice(keyed, 8).keys == (None,)
    assert len(set(g.choice(10, 5, replace=False).tolist())) == 5 and g.choice([10, 20], (2, 3)).shape == (2, 3)
    assert g.choice(10**12, 3).shape == (3,) and len(set(g.choice(10**12, 3, replace=False).tolist())) == 3
    assert g.choice([[1, 2], [3, 4]], 3).shape == (3, 2) and g.choice([[1, 2], [3, 4]]).shape == (2,)
    assert g.choice([], 0).shape == (0,) and g.permutation([]).shape == (0,)
    for fn, args in ((g.integers, (0,)), (g.choice, ([],)), (g.choice, (2, 3, False))):
        with pytest.raises(ValueError): fn(*args)

def test_multivariate_sampling_uses_existing_sampler_and_products():
    from rustymath.random import Generator
    a, b = Generator(7), rm.default_rng(7)
    mean, cov = [1., 2.], [[4., 1.], [1., 3.]]
    actual = a.multivariate_normal(mean, cov, 20_000)
    expected = np.asarray(b.normal(size=(20_000, 2))) @ np.linalg.cholesky(cov).T + mean
    assert np.allclose(actual, expected) and np.allclose(rm.mean(actual, axis=0), mean, atol=.05)
    assert np.allclose(rm.cov(actual, rowvar=False), cov, atol=.1)
    assert a.multivariate_normal(mean, cov).shape == (2,) and a.multivariate_normal(mean, cov, (2, 3)).shape == (2, 3, 2)
    assert a.multivariate_normal(mean, cov, 0).shape == (0, 2)
    with pytest.raises(ValueError): a.multivariate_normal([[1., 2.]], cov)
    with pytest.raises(ValueError): a.multivariate_normal(mean, [[1.]])
    with pytest.raises(ValueError): a.multivariate_normal([1j, 2j], cov)
    with pytest.raises(ValueError): a.multivariate_normal(mean, [[1 + 0j, 0j], [0j, 1 + 0j]])
    with pytest.raises(ValueError): a.multivariate_normal({'a': 1., 'b': 2.}, cov)
    with pytest.raises(ValueError): a.multivariate_normal(mean, {'b': [4., 1.], 'a': [1., 3.]})
    named = rm.array(cov)
    named.names = ('row', 'column')
    with pytest.raises(ValueError): a.multivariate_normal(mean, named)
