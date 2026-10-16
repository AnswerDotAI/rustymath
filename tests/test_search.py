import numpy as np, pytest, rustymath as rm

def test_set_operations_retain_first_occurrences():
    x, y = [3, 1, 3, 2], [2, 4, 2]
    assert rm.union1d(x, y).tolist() == [3, 1, 2, 4]
    assert rm.intersect1d(x, y).tolist() == [2] and rm.setdiff1d(x, y).tolist() == [3, 1]
    assert rm.union1d([], y).tolist() == [2, 4] and rm.intersect1d(x, []).shape == (0,)
    assert rm.setdiff1d('abac', 'b').tolist() == ['a', 'c']
    assert rm.intersect1d([1., 2.], [1. + 1e-14]).tolist() == [1.]
    assert rm.intersect1d([1., 2.], [1. + 1e-14], ct=0).shape == (0,)

def test_sort_and_bins():
    s = rm.search
    for dt in [np.bool_, np.uint8, np.int16, np.int64, np.float16, np.float32, np.float64]:
        a = np.array([[3, 1, 1], [2, 0, 3]], dt)
        if np.dtype(dt).kind == 'f': a[0, 0] = np.nan
        for axis in [0, 1, -1, None]: assert np.array_equal(s.argsort(a, axis=axis), np.argsort(a, axis=axis, stable=True))
        assert np.array_equal(s.argsort(a), np.argsort(a, stable=True))
        a = np.array([0, 1, 1, 3], dt)
        v = np.array([[-1, 1], [2, 4]] if np.dtype(dt).kind in 'if' else [[0, 1], [2, 4]], dt)
        for side in ['left', 'right']: assert np.array_equal(s.searchsorted(a, v, side), np.searchsorted(a, v, side))
        assert np.array_equal(s.bins(a, v), np.searchsorted(a, v, side='right'))
    # Each key keeps its own dtype.
    keys = [np.array([[3, 1, 2], [1, 1, 0]], np.int64) + 2**62, np.array([[0, 0, 1], [0, 1, 1]], np.int16)]
    for axis in [0, -1]: assert np.array_equal(s.lexsort(keys, axis=axis), np.lexsort(keys, axis=axis))
    assert np.array_equal(s.lexsort(keys), np.lexsort(keys)) and np.array_equal(s.lexsort(keys, axis=None), np.lexsort([k.ravel() for k in keys]))
    a = [-np.inf, -0.0, 0.0, np.inf, np.nan, np.nan]
    for side in ['left', 'right']: assert np.array_equal(s.searchsorted(a, a, side), np.searchsorted(a, a, side))
    assert s.argsort(np.empty((2, 0))).shape == (2, 0) and s.argsort(5).tolist() == [0] and s.searchsorted([], 5) == 0
    for f, args in [(s.argsort, ([1j],)), (s.searchsorted, ([[1, 2]], 1)), (s.searchsorted, ([1], 1, 'bad')), (s.lexsort, ([],))]:
        with pytest.raises(ValueError): f(*args)

def test_counts_and_groups():
    s = rm.search
    for a in [[3, 1, 3, 0, 1, 3], [False, True, False], np.array([0, 2**62, 0], np.int64), [0.3, 1, 0.1 + 0.2, 1], list('abca')]:
        u, groups = s.unique(a), s.group_indices(a)
        assert np.array_equal(s.counts(a), u.counts) and [len(g) for g in groups] == u.counts.tolist()
        assert all(np.array_equal(g, np.flatnonzero(np.asarray(u.numbers) == i)) for i, g in enumerate(groups))
    assert len(s.group_indices([])) == len(s.counts([])) == 0 and len(s.group_indices([0.3, 0.1 + 0.2], ct=0)) == 2
    for x in [np.array([1, 3, 1, 0], np.uint8), np.array([100_000], np.int32), np.array([], np.int64), [True, False, True]]:
        assert np.array_equal(s.bincount(x, minlength=5), np.bincount(np.asarray(x, np.int64), minlength=5))
    for args in [([-1],), ([0.5],), ([[1, 2]],), ([1], -1)]:
        with pytest.raises(ValueError): s.bincount(*args)

def test_selection_and_statistics(threads):
    s = rm.search
    for dt in [np.bool_, np.uint8, np.int16, np.int64, np.float16, np.float32, np.float64]:
        a = np.array([[8, 2, 4, 2, 6], [1, 7, 5, 9, 3]], dt)
        for axis in [0, 1, None]:
            expected = np.sort(a, axis=axis)
            assert np.array_equal(s.sort(a, axis=axis), expected)
            assert np.array_equal(s.sort(a, axis=axis, descending=True), np.flip(expected, axis=axis))
            assert np.array_equal(s.argmin(a, axis=axis), np.argmin(a, axis=axis))
            assert np.array_equal(s.argmax(a, axis=axis), np.argmax(a, axis=axis))
            assert np.allclose(s.median(a, axis=axis), np.median(a, axis=axis))
            if dt != np.bool_: assert np.allclose(s.quantile(a, [0, 0.25, 0.5, 1], axis=axis), np.quantile(a, [0, 0.25, 0.5, 1], axis=axis))
            source = a.ravel() if axis is None else a
            actual_axis = 0 if axis is None else axis
            for rank in [0, -1]:
                p = np.asarray(s.partition(a, rank, axis=axis))
                order = np.asarray(s.argpartition(a, rank, axis=axis))
                assert np.array_equal(np.take(p, rank, axis=actual_axis), np.take(expected, rank, axis=actual_axis))
                assert np.array_equal(np.sort(p, axis=actual_axis), expected)
                assert np.array_equal(np.take_along_axis(source, order, axis=actual_axis), p)
    assert s.sort(list('cab')).tolist() == ['a', 'b', 'c']
    keyed = rm.array({'b': 2, 'a': 1})
    assert s.sort(keyed).keys == (('a', 'b'),)
    rows = rm.array({'b': [2, 1], 'a': [1, 2]})
    rows.names = ('row', 'column')
    sorted_rows = s.sort(rows, axis='row')
    assert sorted_rows.keys == (None, None) and sorted_rows.names == rows.names
    assert s.quantile([8, 2, 4, 6], 0.5) == 5 and np.isnan(s.median([]))
    rows = np.arange(4096 * 7, dtype=np.float32).reshape(4096, 7)[:, ::-1]
    for f, expected in [(s.argmin, np.full(4096, 6)), (s.median, np.arange(4096) * 7 + 3)]:
        result = np.asarray(f(rows, axis=1))
        assert np.array_equal(result, expected)
    assert np.asarray(s.quantile(rows, [0.25, 0.5], axis=1)).dtype == np.float32
    assert np.array_equal(s.argpartition([2**62 + 1, 2**62], 0), [1, 0])
    assert np.array_equal(s.argsort([0., -0., np.nan, np.nan]), [0, 1, 2, 3])
    for f, args in [(s.partition, ([1], 1)), (s.argmin, ([],)), (s.quantile, ([1], -0.1)), (s.sort, ([1j],))]:
        with pytest.raises((ValueError, IndexError)): f(*args)

def test_tolerant_search():
    a = np.random.default_rng(0).integers(0, 50, (20, 30))
    vals, first, inverse, counts = map(np.asarray, rm.unique(a))
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
    # A Python scalar gives a Python number, and a NumPy scalar a rank-0 array.
    assert rm.index('abc', 'cb').tolist() == [2, 1] and rm.index(np.float32([1, 2]), np.float32(2)).shape == ()
    with pytest.raises(ValueError): rm.index(x, [1j])
