import numpy as np, pytest, rustymath as rm

def test_selection_and_metadata():
    a = rm.array([[10, 20, 30], [40, 50, 60]])
    assert a[[1, 0], [2, 0]].tolist() == [60, 10]
    assert np.array_equal(a[[[1], [0]], [[0, 2]]], np.asarray(a)[np.ix_([1, 0], [0, 2])])
    assert a[:, ::-1].tolist() == [[30, 20, 10], [60, 50, 40]] and a[None, ...].shape == (1, 2, 3)
    assert a[[True, False], [0, 2]].tolist() == [10, 30] and a[1, -1] == 60
    assert a[np.array([[1, 0]])].shape == (1, 2, 3) and a[:0, ::-1].shape == (0, 3)
    assert rm.array(3)[()] == 3 and rm.array(3)[None].tolist() == [3]
    keyed = rm.array({'a': [1, 2], 'b': [3, 4]})
    keyed.names = ('row', 'column')
    picked = keyed[['b', 'a'], :1]
    assert picked.tolist() == [[3], [1]] and picked.keys == (('b', 'a'), None) and picked.names == keyed.names
    assert keyed['b', :].names == ('column',)
    for index in ([2], (..., ...), (slice(None), [True]), (0, 0, 0)):
        with pytest.raises(IndexError): a[index]
    assert a[[1.0, np.nan]].tolist() == [[40, 50, 60], [0, 0, 0]]
    with pytest.raises(ValueError): a[[0.5]]
    with pytest.raises(ValueError, match='duplicate key: a'): keyed[['a', 'a']]

def test_named_index_arrays_pair_and_address_by_name():
    a = rm.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]])
    i, j = rm.arange(3), rm.arange(3)
    i.names, j.names = ['i'], ['j']
    assert a[i, i].tolist() == [0, 4, 8] and a[i, i].names == ('i',)
    assert a[i, j].shape == (3, 3) and a[i, j].names == ('i', 'j')
    b, c = rm.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]]), rm.array([2, 0])
    b.names, c.names = ['r', 'c'], ['c']
    assert b[c].tolist() == [[2, 0], [5, 3], [8, 6]] and b[c].names == ('r', 'c')
    k, cj = rm.array([2, 0, 1]), rm.arange(3)
    k.names, cj.names = ['c'], ['c']
    assert b[k, cj].tolist() == [6, 1, 5]

def test_assignment_uses_selection_without_resizing_or_aliasing():
    a = rm.array([[1, 2, 3], [4, 5, 6]])
    view = np.asarray(a)
    a[:, [0, 2]] = [10, 20]
    assert a.tolist() == [[10, 2, 10], [20, 5, 20]] and view.tolist() == [[1, 2, 3], [4, 5, 6]]
    a[::-1] = a
    assert a.tolist() == [[20, 5, 20], [10, 2, 10]]
    a[:, ::-1] = view
    a[[0, 0], 1] = [7, 8]
    assert a[0, 1] == 8
    a[[0, 1], [2, 0]] = [30, 40]
    assert a.tolist() == [[3, 8, 30], [40, 5, 4]]
    before = a.tolist()
    with pytest.raises(ValueError): a[:, 0] = [1, 2, 3]
    assert a.tolist() == before
    c = rm.array([10., 20, 30])
    c[np.array([1.0, np.nan])] = [5, 6]
    assert c.tolist() == [10, 5, 30]
    small = rm.array([1, 2], dtype='uint8')
    with pytest.raises(OverflowError): small[:] = [3, 300]
    assert small.tolist() == [1, 2]
    for values, item in (([True, False], True), ('ab', 'z')):
        b = rm.array(values)
        b[-1] = item
        assert b[-1] == item and not np.asarray(b).flags.writeable

def test_at_combines_at_each_selected_item():
    a = rm.array([[1, 2, 3], [4, 5, 6]])
    expected = np.asarray(a).copy()
    np.add.at(expected, ([0, 0, 1], [2, 2, 0]), [10, 20, 30])
    assert rm.at(rm.add, a, ([0, 0, 1], [2, 2, 0]), [10, 20, 30]).tolist() == expected.tolist()
    assert a.tolist() == [[1, 2, 3], [4, 5, 6]]
    assert rm.at(rm.add, rm.array([1, 2], dtype='uint8'), [1, 1], 200).tolist() == [1, 146]
    assert rm.at(rm.add, a, (0, slice(None)), 0.5).dtype == 'float64'
    x = rm.array([[8., 2, 3], [4, 5, 6]])
    for name in ['subtract', 'multiply', 'divide', 'maximum', 'minimum']:
        expected = np.asarray(x).copy()
        getattr(np, name).at(expected, ([0, 0, 1], [0, 0, 0]), [2, 4, 3])
        assert rm.at(getattr(rm, name), x, ([0, 0, 1], [0, 0, 0]), [2, 4, 3]).tolist() == expected.tolist()
    with pytest.raises(ValueError): rm.at(rm.power, a, 0, 1)

def test_shape_operations_keep_independent_values():
    a = rm.array(range(24)).reshape(2, 3, 4)
    for actual, expected in ((a.T, np.asarray(a).T), (rm.transpose(a, (1, 2, 0)), np.asarray(a).transpose(1, 2, 0)),
        (rm.swapaxes(a, 0, 2), np.asarray(a).swapaxes(0, 2)), (rm.expand_dims(a, -1), np.expand_dims(np.asarray(a), -1))):
        assert actual.tolist() == expected.tolist()
    assert a.flatten().shape == a.ravel().shape == (24,) and rm.reshape(a, (6, -1)).shape == (6, 4)
    assert rm.squeeze(a[None], axis=0).tolist() == a.tolist()
    b = a.T
    b[0, 0, 0] = 99
    assert a[0, 0, 0] == 0
    for source, shape in ((np.array(7), (3, 17)), (np.arange(3), (3, 64)), (np.arange(17)[None], (3, 17)),
        (np.arange(6).reshape(2, 1, 3), (2, 4, 3)), (np.array([]), (0, 3))):
        expected = np.broadcast_to(source.reshape(source.shape + (1,) * (len(shape) - source.ndim)), shape)
        assert np.array_equal(rm.broadcast_to(source, shape), expected)
    with pytest.raises(ValueError): rm.squeeze(a, axis=0)
    with pytest.raises(ValueError): rm.transpose(a, (0, 0, 1))

@pytest.mark.parametrize('a', (np.array([[1, 2], [3, 4]]), np.array([[1, 3], [2, 4]]).T, rm.array([[1, 2], [3, 4]]),
    rm.array([[1, 3], [2, 4]]).T, rm.array([[0, 0], [3, 4], [1, 2]])[2:0:-1]))
def test_join_split_repeat_and_padding(a):
    assert rm.concatenate([a, a], axis=1).tolist() == [[1, 2, 1, 2], [3, 4, 3, 4]] and rm.concatenate([a, a]).shape == (4, 2)
    assert rm.concatenate([a, [5]], axis=None).tolist() == [1, 2, 3, 4, 5]
    assert rm.stack([a, a], axis=-1).tolist() == [[[1, 1], [2, 2]], [[3, 3], [4, 4]]]
    assert [x.tolist() for x in rm.split(a, 2, axis=1)] == [[[1], [3]], [[2], [4]]]
    assert [x.tolist() for x in rm.split(a, 2, axis=None)] == [[1, 2], [3, 4]]
    assert [x.tolist() for x in rm.split([1, 2, 3], [1, 2])] == [[1], [2], [3]]
    assert rm.repeat(a, [2, 0], axis=1).tolist() == [[1, 1], [3, 3]]
    assert rm.repeat(a, 2).tolist() == [1, 1, 2, 2, 3, 3, 4, 4]
    assert rm.repeat([10, 20, 30], [2, -1, 0]).tolist() == [10, 10, 0] and rm.repeat([5], [1, 0, 2]).tolist() == [5, 5, 5]
    assert rm.pad(a, [(1, 0), (0, 1)]).tolist() == [[0, 0, 0], [1, 2, 0], [3, 4, 0]]
    assert rm.pad('ab', 1).tolist() == [' ', 'a', 'b', ' ']
    assert rm.take(a, [1, 0], axis=0).tolist() == [[3, 4], [1, 2]] and rm.take(a, [-1]).tolist() == [4]
    assert rm.roll(a, 1, axis=1).tolist() == [[2, 1], [4, 3]] and rm.flip(a).tolist() == [[4, 3], [2, 1]]

def test_nonzero_and_where_use_shared_broadcasting():
    a = rm.array([[0, 2, 0], [3, 0, 4]])
    assert [x.tolist() for x in rm.nonzero(a)] == [[0, 1, 1], [1, 0, 2]]
    assert rm.argwhere(a).tolist() == [[0, 1], [1, 0], [1, 2]]
    assert rm.where(a > 0, [10, 20], a).tolist() == [[0, 10, 0], [20, 0, 20]]
    assert rm.where(True, 2, 3) == 2 and rm.argwhere(rm.array([])).shape == (0, 1)
    assert rm.where([True, False, True], "abc", "xyz").tolist() == ["a", "y", "c"]
    # A key the condition lacks picks y, for characters as for numbers. The result takes its keys from x and y: a key that only the condition has is ignored, as `⍚` ignores a selector's extra keys.
    c = rm.array({'a': True, 'c': False})
    picked = rm.where(c, rm.array({'b': 1, 'a': 2}), rm.array({'b': 3, 'a': 4}))
    assert picked.tolist() == [3, 2] and picked.keys == (('b', 'a'),)
    picked = rm.where(c, rm.array({'b': 'p', 'a': 'q'}), rm.array({'b': 'r', 'a': 's'}))
    assert picked.tolist() == ['r', 'q'] and picked.keys == (('b', 'a'),)
    with pytest.raises(TypeError): rm.where([0, 1], 2, 3)

def test_diagonals_triangles_and_conveniences():
    a = np.arange(24).reshape(2, 3, 4)
    for k in (-5, -1, 0, 2, 5):
        assert np.array_equal(rm.diagonal(a, k), np.diagonal(a, k, axis1=-2, axis2=-1))
        assert np.array_equal(rm.diag([1, 2, 3], k), np.diag([1, 2, 3], k))
        assert np.array_equal(rm.eye(3, 4, k, dtype='int16'), np.eye(3, 4, k, dtype=np.int16))
        for fn in ('tril', 'triu'): assert np.array_equal(getattr(rm, fn)(a, k), getattr(np, fn)(a, k))
    assert rm.identity(0).shape == (0, 0) and rm.eye(2, 0).shape == (2, 0) and rm.diag([]).shape == (0, 0)
    assert np.array_equal(rm.diag(a[0]), np.diag(a[0])) and np.array_equal(rm.trace(a), np.trace(a, axis1=-2, axis2=-1))
    assert rm.diagonal(np.empty((0, 3, 4))).shape == (0, 3)
    keyed = rm.array({'a': [1, 2], 'b': [3, 4]})
    keyed.names = ('row', 'column')
    assert rm.tril(keyed).keys == keyed.keys and rm.tril(keyed).names == keyed.names
    assert rm.broadcast_to(keyed[:, 0], (2, 3)).tolist() == [[1] * 3, [3] * 3]
    assert rm.broadcast_to([1, 2], (2, 3)).tolist() == [[1] * 3, [2] * 3]
    for n in (0, 1, 2, 5): assert np.array_equal(rm.diff(a, n, axis=1), np.diff(a, n, axis=1))
    for shape in ((7, 2), (2, 3, 4), (3, 1), (3, 0)):
        for dtype in (np.float16, np.float32, np.float64, np.int16, np.complex64):
            x = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
            for axis in range(x.ndim): assert np.array_equal(rm.diff(x, axis=axis), np.diff(x, axis=axis))
    assert rm.diff([True, True, False]).tolist() == [False, True]
    assert rm.diff([[True, False], [True, True], [False, True]]).tolist() == [[True], [False], [True]]
    assert rm.diff(keyed, axis='row').keys == (('b',), None) and rm.ptp(a) == np.ptp(a)
    for u8 in (np.array([0, 255], np.uint8)[::-1], np.array([[255, 0], [0, 255]], np.uint8)): assert np.array_equal(rm.diff(u8), np.diff(u8))
    for fn, args in ((rm.broadcast_to, ([1, 2], (3, 2))), (rm.diagonal, ([1, 2],)), (rm.diff, (a, -1))):
        with pytest.raises((ValueError, IndexError)): fn(*args)
