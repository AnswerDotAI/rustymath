import numpy as np, pytest, rustymath as rm

def test_parts_rounding_and_clip():
    for dtype in [np.bool_, np.uint8, np.int16, np.float16, np.float32, np.float64, np.complex64, np.complex128]:
        x = np.array([0, 1, 2], dtype=dtype)
        if np.issubdtype(dtype, np.inexact): x = np.array([0, np.inf, np.nan], dtype=dtype)
        for name in ['isnan', 'isfinite', 'isinf', 'real', 'imag', 'conj']:
            assert np.array_equal(getattr(rm, name)(x), getattr(np, name)(x), equal_nan=True), (name, dtype)
        assert rm.real(x).dtype == rm.array(np.real(x)).dtype and rm.conj(x).dtype == rm.array(x).dtype
    x = np.array([-2.5, -1.5, -0.2, 0.5, 1.5, 2.5], np.float32)
    assert np.array_equal(rm.rint(x), np.rint(x)) and np.array_equal(rm.trunc(x), np.trunc(x))
    assert rm.rint(np.array([1.5 + 2.5j])).tolist() == [2 + 2j]
    x = np.arange(12).reshape(3, 4).astype(np.int16)
    assert np.array_equal(rm.clip(x, [2, 5, 9], 10), np.minimum(np.maximum(x, np.array([2, 5, 9])[:, None]), 10))
    assert rm.clip(x, 2, 10).dtype == 'int16' and rm.clip(3, 5, 2) == 2

def test_statistical_reductions(threads):
    x = np.linspace(-2, 3, 60_000).reshape(100, 20, 30)
    for dtype in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
        a = x.astype(dtype)
        if np.iscomplexobj(a): a += 1j * (a * a)
        reference = a.astype(np.float32) if dtype == np.float16 else a
        for axis in [None, 0, -1, (0, 2), ()]:
            assert np.allclose(rm.mean(a, axis), np.mean(reference, axis=axis), rtol=2e-5, atol=2e-6), (dtype, axis)
            for ddof in [0, 1]:
                if axis == () and ddof: continue
                assert np.allclose(rm.var(a, axis, ddof), np.var(reference, axis=axis, ddof=ddof), rtol=2e-5, atol=2e-6), (dtype, axis)
        assert np.allclose(rm.std(a, axis=1), np.std(reference, axis=1), rtol=2e-5, atol=2e-6)
        assert rm.mean(a, axis=0).dtype == ('complex64' if dtype == np.complex64 else 'complex128' if dtype == np.complex128 else 'float64' if dtype == np.float64 else 'float32')
    a = rm.array([[1., 2., 3.], [4., 5., 6.]])
    a.names = ('row', 'column')
    assert a.mean(axis='column').tolist() == [2., 5.] and a.mean(axis='row').names == ('column',)
    assert np.allclose(a.std(axis='column'), [np.sqrt(2/3)] * 2)
    assert np.isnan(rm.mean(np.empty((2, 0)), axis=1)).all() and np.isnan(rm.var([1.], ddof=1))
    with pytest.raises(IndexError): rm.mean(a, axis=2)

def test_weighted_average_norms_and_covariance():
    x = np.arange(24.).reshape(2, 3, 4)
    weights = np.array([1., 3.])
    assert np.allclose(rm.average(x, axis=0, weights=weights), np.average(x, axis=0, weights=weights))
    weights = np.arange(1., 9.).reshape(2, 1, 4)
    expected = (x * weights).sum(axis=(0, 2)) / np.broadcast_to(weights, x.shape).sum(axis=(0, 2))
    assert np.allclose(rm.average(x, axis=(0, 2), weights=weights), expected)
    a = np.array([1 + 2j, 3 - 1j, 0j])
    for order in [0, 1, 2, 3, np.inf, -np.inf]:
        assert np.isclose(rm.norm(a, ord=order), np.linalg.norm(a, ord=order))
    # An f32 norm computes its squares in f64, where the square of `1e20` doesn't overflow and the square of `1e-30` doesn't underflow.
    big, small = np.float32([1e20]), np.float32([1e-30])
    assert rm.norm(big) == big[0] and rm.norm(small) == small[0] and rm.linalg.norm(big[None]) == big[0]
    x = np.array([[1., 2., 4., 8.], [2., 1., 3., 6.]])
    for a in [x, x + 1j * x[:, ::-1]]:
        for rowvar in [True, False]:
            assert np.allclose(rm.cov(a, rowvar=rowvar), np.cov(a, rowvar=rowvar))
            assert np.allclose(rm.corrcoef(a, rowvar=rowvar), np.corrcoef(a, rowvar=rowvar))
    assert rm.cov([1., 2., 3.]) == 1 and np.allclose(rm.average([1., 2., 3.]), 2)

def test_histogram_edges_and_counts():
    x = np.array([-3., -2., -1., 0., 1., 2., 3., 4., np.nan])
    for bins, span in [(3, (-2, 3)), ([-2., 0., 1., 3.], None)]:
        got, edges = rm.histogram(x, bins=bins, range=span)
        want, boundaries = np.histogram(x, bins=bins, range=span)
        assert np.array_equal(got, want) and np.allclose(edges, boundaries)
    counts, edges = rm.histogram(np.array([], np.float32))
    assert counts.tolist() == [0] * 10 and len(edges) == 11 and edges.dtype == 'float32'
    with pytest.raises(ValueError): rm.histogram(x, bins=[0., 1., 1.])
    with pytest.raises(ValueError): rm.histogram(x, bins=0)
    with pytest.raises(ValueError): rm.histogram(x, bins=[0., 1.], range=(0, 1))

def test_scalar_compositions():
    assert rm.percentile([1, 2, 3, 4], [0, 50, 100]).tolist() == [1, 2.5, 4]

def test_bitwise_decimal_rounding_and_angles():
    a = rm.array([-7, -1, 0, 3, 7], dtype='int16')
    for fn, op in ((rm.bitwise_and, lambda x: x & 3), (rm.bitwise_or, lambda x: x | 3), (rm.bitwise_xor, lambda x: x ^ 3),
                   (rm.left_shift, lambda x: x << 3), (rm.right_shift, lambda x: x >> 3), (rm.floor_divide, lambda x: x // 3)):
        assert fn(a, 3).tolist() == op(a).tolist() == [op(x) for x in a.tolist()]
    assert (~a).tolist() == [~x for x in a.tolist()] and (3 // a[3:]).tolist() == [1, 0]
    assert rm.floor_divide([-7, 7], -3).tolist() == [2, -3] and rm.floor_divide([-7., 7.], -3.).tolist() == [2., -3.]
    # 0.3 / 0.1 is 2.9999999999999996, which floors to 2 exactly, and to 3 within the tolerance 1e-14.
    x = rm.array([0.3, 1.])
    assert (x // .1).tolist() == [2., 10.] and (x % .1).tolist() == [0.3 % .1, 0.]
    rm.set_tolerance(rm.CT)
    try: assert (x // .1).tolist() == [3., 10.] and (x % .1).tolist() == [0., 0.]
    finally: rm.set_tolerance(0)
    for dtype in ('float16', 'float32', 'float64'):
        x = rm.array([1.25, -1.25, 0., np.inf], dtype=dtype)
        assert np.allclose(rm.round(x, 1), [1.2, -1.2, 0., np.inf], atol=.001)
        assert np.allclose(rm.radians(rm.degrees(x[:3])), x[:3], atol=.001)
    assert rm.round(rm.array([15, 25, -15, -25], dtype='int16'), -1).tolist() == [20, 20, -20, -20]
    assert rm.array([1, 2], dtype='int16').round(2).dtype == 'int16'
    assert rm.round([True, False], -1).tolist() == [0, 0]
    assert rm.round([1.25 + 2.75j], 1).tolist() == [1.2 + 2.8j]
    for fn, args in ((rm.left_shift, (a, -1)), (rm.floor_divide, (a, 0)), (rm.bitwise_and, ([1.], 2))):
        with pytest.raises((ValueError, OverflowError)): fn(*args)
