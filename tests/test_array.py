import numpy as np, pytest, rustymath as rm

DTYPES = {'bool': np.bool_, 'u8': np.uint8, 'i16': np.int16, 'i32': np.int32, 'i64': np.int64, 'f16': np.float16, 'f32': np.float32,
    'f64': np.float64, 'c64': np.complex64, 'c128': np.complex128, 'char': np.dtype('<U1')}

def test_numpy_views_share_items():
    for name, dt in DTYPES.items():
        x = np.array(['a', 'b', 'c'] if name == 'char' else [1, 0, 1], dtype=dt)
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

def test_array_copies_and_reads_any_layout():
    x = np.arange(12.).reshape(3, 4)
    a = rm.array(x)
    x[0, 0] = 99
    assert a.tolist()[0][0] == 0.0
    unaligned = np.frombuffer(bytearray(1) + np.array([1.5, 2.5]).tobytes(), dtype=np.float64, offset=1)
    for y in (x[:, ::2], x[::-1], x.T, unaligned, np.array(7, np.int16), np.float32(2.5)):
        b = rm.array(y)
        assert b.dtype == {np.float64: 'f64', np.int16: 'i16', np.float32: 'f32'}[y.dtype.type] and b.shape == np.shape(y)
        assert np.array_equal(np.asarray(b), y)
    assert rm.array(np.array([0, 2, 1], np.uint8).view(np.bool_)).tolist() == [False, True, True]

def test_unsupported_buffers_raise():
    for x in (np.zeros(2, np.uint16), np.zeros(2, np.uint32), np.zeros(2, np.uint64), np.zeros(2, np.int8), np.zeros(2, np.longdouble),
              np.array(['ab']), np.array([b'ab']), np.zeros(2, '>f8')):
        with pytest.raises(ValueError): rm.array(x)
    with pytest.raises(ValueError, match='Unicode'): rm.array(np.array([0xD800], np.uint32).view('<U1'))

def test_python_values():
    assert [rm.array(v).dtype for v in ([1, 2], [1, 2.5], [True, False], [1j], 'ab', [], (1, 2), range(3))] == ['i64', 'f64', 'bool', 'c128', 'char', 'f64', 'i64', 'i64']
    m = rm.array([[1, 2], [3, 4]])
    assert m.shape == (2, 2) and len(m) == 2 and m[1].tolist() == [3, 4] and m[-1][0] == 3 and isinstance(m[0][1], int)
    assert rm.array(range(2, 8, 3)).tolist() == [2, 5]
    assert rm.array([np.float32(1.5), 2]).dtype == 'f32' and rm.array([np.array([1, 2], np.int16), [3, 4]]).dtype == 'i16'
    assert rm.array([np.float32(1), np.float64(2)]).dtype == 'f64' and rm.array([np.uint8(1), np.float16(2)]).dtype == 'f16'
    with pytest.raises(OverflowError): rm.array(range(2**63, 2**63 + 2))
    t = rm.array({'a': [1, 2], 'b': [3, 4]})
    assert t.keys == (('a', 'b'), None) and t['b'].tolist() == [3, 4] and t.names == (None, None)
    assert repr(rm.array([1.5])) == 'Array([1.5], dtype=f64)'
    with pytest.raises(ValueError, match='ragged'): rm.array([[1], [2, 3]])
    with pytest.raises(ValueError): rm.array([1, 'a'])
    with pytest.raises(ValueError): rm.array(['ab'])
    with pytest.raises(OverflowError): rm.array([2**70])
    with pytest.raises(TypeError): rm.array(x for x in [1])
    with pytest.raises(KeyError): t['c']
    with pytest.raises(IndexError): m[2]

def test_repr_precision():
    for dtype, text in (('f16', '1.23'), ('f32', '1.23457'), ('f64', '1.2345679')):
        a = rm.array([1.234567891234, -0.0], dtype=dtype)
        assert f'[{text}, -0.0]' in repr(a) and a.tolist() == np.array([1.234567891234, -0.0], dtype=DTYPES[dtype]).tolist()
    for dtype, text in (('c64', '(1.23457+2.34568j)'), ('c128', '(1.2345679+2.3456789j)')):
        assert text in repr(rm.array([1.234567891234 + 2.345678912345j], dtype=dtype))
    roots = [1.000000000000002+0j, complex(1.9999999999999984, -0.0), 3.0000000000000036+0j]
    a = rm.array(roots)
    assert '[(1+0j), (2-0j), (3+0j)]' in repr(a) and a.tolist() == roots
    assert '[1.2345679e-20, 1.2345679e+20]' in repr(rm.array([1.234567891e-20, 1.234567891e20]))

def test_methods():
    m = rm.array([[1, 2], [3, 4]])
    t = rm.array({'a': [1, 2], 'b': [3, 4]})
    assert m[1, 0] == 3 and isinstance(m[1, 0], int) and t['b', -1] == 4 and m[()].tolist() == m.tolist()
    with pytest.raises(IndexError): m[0, 0, 0]
    assert t.astype('f32').dtype == 'f32' and t.astype('f32').keys == t.keys and rm.array([True]).astype('u8').tolist() == [1]
    assert rm.array([1.5]).astype('c128').tolist() == [1.5 + 0j] and m.astype('i64').tolist() == m.tolist()
    assert rm.array([1.7, -1.7]).astype('i64').tolist() == [1, -1] and rm.array([5]).astype('i16').tolist() == [5] and rm.array([2 + 0j]).astype('u8').tolist() == [2]
    assert rm.array([0, 1]).astype('bool').tolist() == rm.array([0.0, 1.0]).astype('bool').tolist() == [False, True] and rm.array([1 + 2j]).astype('c64').dtype == 'c64'
    for source, values, targets in (('u8', [0, 255], ('i16', 'i32', 'i64')), ('i16', [-32768, 32767], ('i32', 'i64')), ('i32', [-2**31, 2**31-1], ('i64',))):
        a = rm.array(values, dtype=source)
        for target in targets: assert a.astype(target).tolist() == a.astype(target).astype(source).tolist() == values
    for x, dtype in ((-1, 'u8'), (300, 'u8'), (2**40, 'i32'), (float('nan'), 'i64'), (float('inf'), 'i16'), (2, 'bool'), (0.5, 'bool'), (1.5, 'bool'), (1e19, 'i64')):
        with pytest.raises(OverflowError): rm.array([x]).astype(dtype)
    with pytest.raises(ValueError, match='imaginary'): rm.array([1 + 1j]).astype('f64')
    with pytest.raises(ValueError, match='characters'): rm.array('ab').astype('i64')
    with pytest.raises(ValueError, match='unknown dtype'): m.astype('float64')
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
    assert '[0.1]' in repr(rm.array([0.1]).astype('f32')) and '[0.1]' in repr(rm.array([0.1]).astype('f16')) and '(0.1+0.2j)' in repr(rm.array([0.1 + 0.2j]).astype('c64'))
    # Cast only selected items, not invalid values elsewhere in the shared base.
    assert rm.array([float('nan'), 2.5])[1:].astype('i64').tolist() == [2]

def test_operators_and_reductions():
    a, m = rm.array([1, 2, 3]), rm.array([[1, 2, 3], [4, 5, 6]])
    assert (a + 1).tolist() == [2, 3, 4] and (1 + a).tolist() == [2, 3, 4] and (10 - a).tolist() == [9, 8, 7] and (-a).tolist() == [-1, -2, -3]
    assert (a * a).tolist() == [1, 4, 9] and (a ** 2).tolist() == [1, 4, 9] and (2 ** a).tolist() == [2, 4, 8] and (a % 2).tolist() == [1, 0, 1]
    assert (a / 2).tolist() == [0.5, 1, 1.5] and abs(-a).tolist() == [1, 2, 3] and (a + np.ones(3, np.int64)).tolist() == [2, 3, 4]
    assert (m + rm.array([10, 20])).tolist() == [[11, 12, 13], [24, 25, 26]]
    assert (a == 2).tolist() == [False, True, False] and (a >= 2).tolist() == [False, True, True] and (a == None) is False
    with pytest.raises(TypeError): a + None
    with pytest.raises(TypeError): hash(a)
    with pytest.raises(OverflowError): rm.array([2**62]) * 4
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
