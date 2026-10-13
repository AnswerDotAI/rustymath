import numpy as np, pytest, rustymath as rm

def test_sort_and_bins():
    s = rm.search
    for dt in [np.bool_, np.int8, np.uint32, np.int64, np.uint64, np.float16, np.float32, np.float64]:
        a = np.array([[3, 1, 1], [2, 0, 3]], dt)
        if np.dtype(dt).kind == 'f': a[0, 0] = np.nan
        for axis in [0, 1, None]: assert np.array_equal(s.argsort(a, axis=axis), np.argsort(a, axis=axis, stable=True))
        a = np.array([0, 1, 1, 3], dt)
        v = np.array([[-1, 1], [2, 4]]) if np.dtype(dt).kind not in 'bu' else np.array([[0, 1], [2, 4]], dt)
        for side in ['left', 'right']: assert np.array_equal(s.searchsorted(a, v, side), np.searchsorted(a, v, side))
        assert np.array_equal(s.bins(a, v), np.searchsorted(a, v, side='right'))
    keys = [np.array([[3, 1, 2], [1, 1, 0]], np.uint64) + 2**63, np.array([[0, 0, 1], [0, 1, 1]], np.int8)]
    assert np.array_equal(s.lexsort(keys), np.lexsort(keys)) and np.array_equal(s.lexsort(keys, axis=0), np.lexsort(keys, axis=0))
    a = [-np.inf, -0.0, 0.0, np.inf, np.nan, np.nan]
    for side in ['left', 'right']: assert np.array_equal(s.searchsorted(a, a, side), np.searchsorted(a, a, side))
    assert s.argsort(np.empty((2, 0))).shape == (2, 0) and s.searchsorted([], 5) == 0
    for f, args in [(s.argsort, ([1j],)), (s.searchsorted, ([[1, 2]], 1)), (s.searchsorted, ([1], 1, 'bad')), (s.lexsort, ([],))]:
        with pytest.raises((ValueError, TypeError)): f(*args)

def test_counts_and_groups():
    s = rm.search
    for a in [[3, 1, 3, 0, 1, 3], [False, True, False], np.array([0, 2**63, 0], np.uint64), [0.3, 1, 0.1 + 0.2, 1]]:
        u, groups = s.unique(a), s.group_indices(a)
        assert np.array_equal(s.counts(a), u.counts) and [len(g) for g in groups] == u.counts.tolist()
        assert all(np.array_equal(g, np.flatnonzero(u.numbers == i)) for i, g in enumerate(groups))
    assert len(s.group_indices([])) == len(s.counts([])) == 0 and len(s.group_indices([0.3, 0.1 + 0.2], ct=0)) == 2
    for x in [np.array([1, 3, 1, 0], np.int8), np.array([100_000], np.uint32), np.array([], np.int64)]:
        assert np.array_equal(s.bincount(x, minlength=5), np.bincount(x.astype(np.int64), minlength=5))
    for args in [([-1],), ([0.5],), ([[1, 2]],), ([1], -1)]:
        with pytest.raises((ValueError, TypeError)): s.bincount(*args)

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
