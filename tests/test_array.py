import numpy as np, pytest, rustymath as rm

DTYPES = {'bool': np.bool_, 'uint8': np.uint8, 'int16': np.int16, 'int32': np.int32, 'int64': np.int64, 'float16': np.float16, 'float32': np.float32,
    'float64': np.float64, 'complex64': np.complex64, 'complex128': np.complex128, 'character': np.dtype('<U1')}

def test_numpy_views_share_items():
    for name, dt in DTYPES.items():
        x = np.array(['a', 'b', 'c'] if name == 'character' else [1, 0, 1], dtype=dt)
        a = rm.array(x)
        assert a.dtype == name and a.shape == (3,)
        v = np.asarray(a)
        assert v.dtype == np.dtype(dt) and (v == x).all()
        assert not v.flags.writeable
        with pytest.raises(ValueError): v[0] = v[1]
        a[0] = a[1]
        assert a[0] == x[1] and np.array_equal(v, x)
    v = np.asarray(rm.array(np.float32(2.5)))
    assert v.shape == () and v.dtype == np.float32 and v == 2.5
    a = rm.arange(24).reshape(4, 6)
    base = np.asarray(a)
    for view, expected in ((a.T, base.T), (a[::-1], base[::-1]), (a[:, ::2], base[:, ::2]), (a[1:3], base[1:3])):
        exported = np.asarray(view)
        assert np.shares_memory(exported, base) and exported.strides == expected.strides
        assert np.array_equal(exported, expected) and view.tolist() == expected.tolist()
        compact = view.compact()
        assert not np.shares_memory(np.asarray(compact), base) and compact.tolist() == view.tolist()
    view = a[::-1]
    snapshot = np.asarray(view)
    a[0, 0] = 99
    view[0, 0] = 88
    assert a[3, 0] == 18 and view[3, 0] == 0 and snapshot[0, 0] == 18 and base[0, 0] == 0

def test_views_that_strides_cant_describe_keep_what_they_gather():
    image = rm.arange(256.).reshape(16, 16)
    rows = rm.windows(image, (3, 3)).reshape(-1, 9)
    declines = lambda report: [(r.route, r.reason, r.calls) for r in report.records if r.kind == 'declined']
    with rm.explain() as report: row = rows[2]
    assert row.tolist() == [2, 3, 4, 18, 19, 20, 34, 35, 36] and declines(report) == []
    with rm.explain() as report: totals = {rows.sum(), rows.sum(), np.asarray(rows).sum(), sum(map(sum, rows.tolist()))}
    assert len(totals) == 1 and declines(report) == [('in place', 'a view that gathers', 1)]
    with rm.explain() as report: repeated = rm.repeat(image[:, ::2], 2, axis=1)
    assert repeated.tolist() == np.repeat(np.asarray(image)[:, ::2], 2, axis=1).tolist() and declines(report) == []
    padded = rm.pad(rm.arange(4.), 1, fill=9.)
    assert padded.tolist() == np.asarray(padded).tolist() == [9, 0, 1, 2, 3, 9] and repr(padded).startswith('Array([9.0, 0.0')

def test_array_copies_and_reads_any_layout():
    x = np.arange(12.).reshape(3, 4)
    a = rm.array(x)
    x[0, 0] = 99
    assert a.tolist()[0][0] == 0.0
    unaligned = np.frombuffer(bytearray(1) + np.array([1.5, 2.5]).tobytes(), dtype=np.float64, offset=1)
    for y in (x[:, ::2], x[::-1], x.T, unaligned, np.array(7, np.int16), np.float32(2.5), np.arange(3, dtype=np.int8),
              np.arange(3, dtype=np.uint16), np.arange(3, dtype=np.uint32), np.arange(3, dtype=np.uint64),
              np.array([1, -2], '>i4'), np.array([1.5, 2.5], '>f8'), np.array([1+2j, 3-4j], '>c8')):
        b = rm.array(y)
        assert b.dtype == {np.float64: 'float64', np.float32: 'float32', np.int8: 'int16', np.int16: 'int16', np.int32: 'int32',
                           np.uint16: 'int64', np.uint32: 'int64', np.uint64: 'int64', np.complex64: 'complex64'}[y.dtype.type] and b.shape == np.shape(y)
        assert np.array_equal(np.asarray(b), y)
    assert rm.array(np.array([0, 2, 1], np.uint8).view(np.bool_)).tolist() == [False, True, True]

def test_unsupported_buffers_raise():
    for x in (np.zeros(2, np.longdouble), np.array(['ab']), np.array([b'ab'])):
        with pytest.raises(ValueError): rm.array(x)
    with pytest.raises(OverflowError): rm.array(np.array([2**63], np.uint64))
    with pytest.raises(ValueError, match='Unicode'): rm.array(np.array([0xD800], np.uint32).view('<U1'))

def test_python_values():
    assert [rm.array(v).dtype for v in ([1, 2], [1, 2.5], [True, False], [1j], 'ab', [], (1, 2), range(3))] == ['int64', 'float64', 'bool', 'complex128', 'character', 'float64', 'int64', 'int64']
    m = rm.array([[1, 2], [3, 4]])
    assert m.shape == (2, 2) and len(m) == 2 and m[1].tolist() == [3, 4] and m[-1][0] == 3 and isinstance(m[0][1], int)
    assert rm.array(range(2, 8, 3)).tolist() == [2, 5]
    assert rm.array([np.float32(1.5), 2]).dtype == 'float32' and rm.array([np.array([1, 2], np.int16), [3, 4]]).dtype == 'int16'
    assert rm.array([np.float32(1), np.float64(2)]).dtype == 'float64' and rm.array([np.uint8(1), np.float16(2)]).dtype == 'float16'
    with pytest.raises(OverflowError): rm.array(range(2**63, 2**63 + 2))
    t = rm.array({'a': [1, 2], 'b': [3, 4]})
    assert t.keys == (('a', 'b'), None) and t[rm.array('b')].tolist() == [3, 4] and t.names == (None, None)
    assert np.isnan(rm.array([1, None]).tolist()[1])
    with pytest.raises(TypeError): rm.array(None)
    assert repr(rm.array([1.5])) == 'Array([1.5], dtype=float64)'
    with pytest.raises(ValueError, match='ragged'): rm.array([[1], [2, 3]])
    with pytest.raises(ValueError, match='ragged'): rm.array([1, [2, 3]])
    cyclic = [1]
    cyclic.append(cyclic)
    with pytest.raises(ValueError, match='cyclic'): rm.array(cyclic)
    with pytest.raises(ValueError): rm.array([1, 'a'])
    with pytest.raises(ValueError): rm.array(['ab'])
    with pytest.raises(OverflowError): rm.array([2**70])
    with pytest.raises(TypeError): rm.array(x for x in [1])
    with pytest.raises(KeyError): t['c']
    with pytest.raises(IndexError): m[2]

def test_data_frames():
    import pandas as pd
    df = pd.DataFrame({'a': [1, 2], 'b': [1.5, None]}, index=pd.Index(['r', 's'], name='row'))
    df.columns.name = 'col'
    t = rm.array(df)
    assert t.keys == (('r', 's'), ('a', 'b')) and t.names == ('row', 'col') and t.dtype == 'float64'
    assert t['s', 'a'] == 2 and np.isnan(t['s', 'b'])
    with pytest.raises(ValueError, match='column "t" holds text'): rm.array(pd.DataFrame({'t': ['ab', 'c']}))

def test_repr_precision():
    for dtype, text in (('float16', '1.23'), ('float32', '1.23457'), ('float64', '1.2345679')):
        a = rm.array([1.234567891234, -0.0], dtype=dtype)
        assert f'[{text}, -0.0]' in repr(a) and a.tolist() == np.array([1.234567891234, -0.0], dtype=DTYPES[dtype]).tolist()
    for dtype, text in (('complex64', '(1.23457+2.34568j)'), ('complex128', '(1.2345679+2.3456789j)')):
        assert text in repr(rm.array([1.234567891234 + 2.345678912345j], dtype=dtype))
    roots = [1.000000000000002+0j, complex(1.9999999999999984, -0.0), 3.0000000000000036+0j]
    a = rm.array(roots)
    assert '[(1+0j), (2-0j), (3+0j)]' in repr(a) and a.tolist() == roots
    assert '[1.2345679e-20, 1.2345679e+20]' in repr(rm.array([1.234567891e-20, 1.234567891e20]))

def test_methods():
    m = rm.array([[1, 2], [3, 4]])
    t = rm.array({'a': [1, 2], 'b': [3, 4]})
    assert not bool(rm.array(0)) and bool(rm.array([1])) and not bool(rm.array([[0]]))
    for value in ([], [0, 1]):
        with pytest.raises(ValueError): bool(rm.array(value))
    assert m[1, 0] == 3 and isinstance(m[1, 0], int) and t['b', -1] == 4 and m[()].tolist() == m.tolist()
    with pytest.raises(IndexError): m[0, 0, 0]
    assert t.astype('float32').dtype == 'float32' and t.astype('float32').keys == t.keys and rm.array([True]).astype('uint8').tolist() == [1]
    assert rm.array([1.5]).astype('complex128').tolist() == [1.5 + 0j] and m.astype('int64').tolist() == m.tolist()
    assert rm.array([1.7, -1.7]).astype('int64').tolist() == [1, -1] and rm.array([5]).astype('int16').tolist() == [5] and rm.array([2 + 0j]).astype('uint8').tolist() == [2]
    assert rm.array([0, 1]).astype('bool').tolist() == rm.array([0.0, 1.0]).astype('bool').tolist() == [False, True] and rm.array([1 + 2j]).astype('complex64').dtype == 'complex64'
    for source, values, targets in (('uint8', [0, 255], ('int16', 'int32', 'int64')), ('int16', [-32768, 32767], ('int32', 'int64')), ('int32', [-2**31, 2**31-1], ('int64',))):
        a = rm.array(values, dtype=source)
        for target in targets: assert a.astype(target).tolist() == a.astype(target).astype(source).tolist() == values
    for x, dtype in ((-1, 'uint8'), (300, 'uint8'), (2**40, 'int32'), (float('nan'), 'int64'), (float('inf'), 'int16'), (2, 'bool'), (0.5, 'bool'), (1.5, 'bool'), (1e19, 'int64')):
        with pytest.raises(OverflowError): rm.array([x]).astype(dtype)
    with pytest.raises(ValueError, match='imaginary'): rm.array([1 + 1j]).astype('float64')
    with pytest.raises(ValueError, match='characters'): rm.array('ab').astype('int64')
    with pytest.raises(ValueError, match='unknown dtype'): m.astype('f64')
    v = rm.array(range(6))
    r = v.reshape(2, 3)
    assert r.shape == (2, 3) and v.reshape((3, -1)).shape == (3, 2) and v.reshape([6]).tolist() == v.tolist() and t.reshape(4).keys == (None,)
    r[0, 0] = 9
    assert v[0] == 0
    with pytest.raises(ValueError): v.reshape(-1, -1)
    with pytest.raises(ValueError): v.reshape(4, -1)
    with pytest.raises(ValueError): v.reshape(4)
    s = rm.array([[10, 20, 30], [40, 50, 60]])
    s.names = ('city', 'month')
    assert s.names == ('city', 'month') and s.sum(axis='month').tolist() == [60, 150] and "names=('city', 'month')" in repr(s)
    s.names = ()
    assert s.names == (None, None)
    for names in (('city',), ('a', 'a')):
        with pytest.raises(ValueError): s.names = names
    assert '[0.1]' in repr(rm.array([0.1]).astype('float32')) and '[0.1]' in repr(rm.array([0.1]).astype('float16')) and '(0.1+0.2j)' in repr(rm.array([0.1 + 0.2j]).astype('complex64'))
    # Cast only selected items, not invalid values elsewhere in the shared base.
    assert rm.array([float('nan'), 2.5])[1:].astype('int64').tolist() == [2]

def test_operators_and_reductions():
    a, m = rm.array([1, 2, 3]), rm.array([[1, 2, 3], [4, 5, 6]])
    assert (a + 1).tolist() == [2, 3, 4] and (1 + a).tolist() == [2, 3, 4] and (10 - a).tolist() == [9, 8, 7] and (-a).tolist() == [-1, -2, -3]
    assert (a * a).tolist() == [1, 4, 9] and (a ** 2).tolist() == [1, 4, 9] and (2 ** a).tolist() == [2, 4, 8] and (a % 2).tolist() == [1, 0, 1]
    assert (a / 2).tolist() == [0.5, 1, 1.5] and abs(-a).tolist() == [1, 2, 3] and (a + np.ones(3, np.int64)).tolist() == [2, 3, 4]
    assert (m + rm.array([10, 20])).tolist() == [[11, 12, 13], [24, 25, 26]]
    assert (a == 2).tolist() == [False, True, False] and (a >= 2).tolist() == [False, True, True] and (a == None) is False
    with pytest.raises(TypeError): a + None
    with pytest.raises(TypeError): hash(a)
    assert (rm.array([2**62]) * 4).tolist() == [0]
    assert m.sum() == 21 and isinstance(m.sum(), int) and m.sum(axis=0).tolist() == [5, 7, 9] and m.sum(axis=(0, 1)) == 21
    assert m.prod(axis=1).tolist() == [6, 120] and m.max() == 6 and m.min(axis=1).tolist() == [1, 4]
    assert m.cumsum().tolist() == [1, 3, 6, 10, 15, 21] and m.cumsum(axis=1).tolist() == [[1, 3, 6], [4, 9, 15]] and m.cumprod(axis=0).tolist() == [[1, 2, 3], [4, 10, 18]]
    b = rm.array([[True, False], [True, True]])
    assert b.all() is False and b.any() is True and b.all(axis=0).tolist() == [True, False]

def test_matmul_operator():
    m, v = rm.array([[1, 2], [3, 4]]), rm.array([1, 1])
    assert (m @ m).tolist() == [[7, 10], [15, 22]] and (m @ v).tolist() == [3, 7] and (v @ m).tolist() == [4, 6] and v @ v == 2
    assert (np.eye(2, dtype=np.int64) @ m).tolist() == [[1, 2], [3, 4]] and ([[1, 0], [0, 1]] @ m).tolist() == [[1, 2], [3, 4]]
    with pytest.raises(TypeError): m @ None
