import itertools
import numpy as np, pytest, rustymath as rm

A = np.asarray

def test_lu(threads):
    la = rm.linalg
    base = np.array([[0., 2, 1], [3, 1, -1], [1, 0, 4]])
    b = np.arange(12.).reshape(3, 4)[:, ::-1]
    for dt in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
        a = np.stack([base + k / 8 * np.eye(3) for k in range(64)]).astype(dt)
        if np.issubdtype(dt, np.complexfloating): a += 0.25j
        f = la.lu(a)
        assert A(f.factors).dtype == dt and A(f.p).dtype == np.int64 and np.allclose(np.tril(A(f.l)), f.l) and np.allclose(np.triu(A(f.u)), f.u)
        assert np.allclose(f.l @ f.u, np.take_along_axis(a, A(f.p)[..., :, None], -2), rtol=0.002, atol=0.002)
        wide = a.astype(np.complex128 if np.iscomplexobj(a) else np.float64)
        # The right-hand side takes the matrices' dtype, because arrays of different dtypes raise.
        bt = b.astype(dt)
        assert np.allclose(f.solve(bt), np.linalg.solve(wide, b), rtol=0.003, atol=0.003)
        assert np.allclose(la.solve(a, bt), np.linalg.solve(wide, b), rtol=0.003, atol=0.003)
        assert np.allclose(f.solve(bt[:, 0]), np.linalg.solve(wide, b[:, :1])[..., 0], rtol=0.003, atol=0.003)
        assert np.allclose(f.inverse(), np.linalg.inv(wide), rtol=0.003, atol=0.003)
        assert np.allclose(f.determinant(), np.linalg.det(wide), rtol=0.003, atol=0.003)
        sign, log = f.log_determinant()
        want = np.linalg.slogdet(wide)
        assert np.allclose(sign, want.sign, rtol=0.003, atol=0.003) and np.allclose(log, want.logabsdet, rtol=0.003, atol=0.003) and A(log).dtype == a.real.dtype
    # Factors can be strided, and a row position out of range raises.
    f = la.lu(base)
    packed, perm = np.zeros((6, 6)), np.zeros(6, np.int64)
    packed[::2, ::2], perm[::2] = f
    assert np.allclose(la.Lu(packed[::2, ::2], perm[::2]).solve(b), f.solve(b))
    with pytest.raises(IndexError): la.Lu(f.factors, np.array([-1, 0, 1])).inverse()

def test_svd(threads):
    la = rm.linalg
    a = np.array([[1., 2, 1], [2, 1, 2], [3, 0, 3], [0, 2, 0]])
    for dt in [np.float32, np.float64, np.complex64, np.complex128]:
        for x in [a, a.T]:
            x = (x * np.resize([1., 2.], 64)[:, None, None]).astype(dt)
            if np.iscomplexobj(x): x *= 1 + 0.25j
            f, b = la.svd(x), np.arange(x.shape[-2] * 2.).reshape(-1, 2).astype(dt)
            inverse = np.linalg.pinv(x, rcond=np.finfo(x.real.dtype).eps * max(x.shape[-2:]))
            assert np.allclose(f.solve(b), inverse @ b, rtol=2e-5, atol=2e-5)
            assert np.allclose(f.solve(b[:, 0]), (inverse @ b)[..., 0], rtol=2e-5, atol=2e-5)
            assert np.allclose(f.pseudoinverse(), inverse, rtol=2e-5, atol=2e-5)
            u, s, v = map(A, f)
            assert s.dtype == x.real.dtype and np.allclose((u * s[..., None, :]) @ v.conj().swapaxes(-1, -2), x, rtol=2e-5, atol=2e-5)
    f = la.svd(np.eye(3, dtype=np.float16))
    assert A(f.solve(np.ones(3, np.float16))).dtype == np.float16 and A(f.pseudoinverse()).dtype == np.float16

def test_permanent_and_echelon():
    la = rm.linalg
    a = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 10]])
    for dt in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
        x = a.astype(dt)
        if np.iscomplexobj(x): x += 0.5j
        want = sum(np.prod(x[np.arange(3), p]) for p in itertools.permutations(range(3)))
        assert np.allclose(la.permanent(np.stack([x, x.T])), want, rtol=0.002)
        f = la.echelon(x)
        assert np.allclose(f.rows, np.eye(3), atol=1e-5) and f.pivots.tolist() == [0, 1, 2] and A(f.rows).dtype == dt
    f = la.echelon([[0., 2, 4], [0, 1, 2], [0, 0, 0]], columns=2)
    assert f.rows.tolist() == [[0, 1, 2]] and f.pivots.tolist() == [1] and f.determinant == 2
    assert la.permanent(np.empty((0, 0))) == 1 and la.echelon(np.empty((0, 3))).rows.shape == (0, 3)
    with pytest.raises(ValueError): la.permanent(np.eye(64))

def test_and_or_product():
    a = np.array([[1, 0, 1], [0, 1, 0]], bool)
    b = np.array([[0, 1], [1, 0], [0, 0]], bool)
    product = rm.linalg.inner(rm.logical_and, rm.logical_or)
    assert np.array_equal(product(a, b), np.all(a[:, :, None] | b[None], axis=1))
    assert product(np.empty((2, 0), bool), np.empty((0, 3), bool)).all()

def test_general_inner(threads):
    inner = rm.linalg.inner
    a, b = np.arange(24.).reshape(2, 3, 4), np.arange(8.).reshape(4, 2)
    terms = a[..., None] - b
    assert np.array_equal(inner(rm.add, rm.subtract)(a, b), terms.sum(axis=-2))
    half = inner(rm.add, rm.subtract)(a.astype(np.float16), b.astype(np.float16))
    assert half.dtype == 'float16' and np.array_equal(half, terms.sum(axis=-2))
    counts = inner(rm.add, rm.less)(a, b)
    assert A(counts).dtype == np.int64 and np.array_equal(counts, (a[..., None] < b).sum(axis=-2))
    # Right association must survive the bounded term blocks.
    v = np.arange(1., 20_001)
    assert inner(rm.subtract, rm.multiply)(v, np.ones_like(v)) == -10_000
    assert inner(rm.subtract, rm.multiply)([1., 2, 3, 4], [1., 1, 1, 1]) == -2
    singleton = inner(rm.less, rm.add)([[2.]], [[3.]])
    assert singleton.dtype == 'float64' and singleton.tolist() == [[5.]]
    assert inner(rm.add, rm.subtract)(np.empty((2, 0)), np.empty((0, 3))).tolist() == [[0.] * 3] * 2
    with pytest.raises(ValueError, match='no identity'): inner(rm.power, rm.add)(np.empty((2, 0)), np.empty((0, 3)))
    def add(x, y): return x + y
    for f in [np.add, add, rm.sin]:
        with pytest.raises(ValueError, match='native binary'): inner(f, rm.add)

@pytest.mark.parametrize('kind', [float, complex])
def test_linalg(kind):
    la, rng = rm.linalg, np.random.default_rng(0)
    H = lambda x: np.conj(np.swapaxes(x, -1, -2))
    rand = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape) if kind is complex else rng.standard_normal(shape)
    a, b = rand(4, 5, 5), rand(4, 5, 3)
    v = b[0, :, 0]
    assert np.allclose(la.solve(a, b), np.linalg.solve(a, b)) and np.allclose(la.solve(a[0], v), np.linalg.solve(a[0], v))
    # Frames pair from the front, as rustymath broadcasts, where NumPy pairs them from the back.
    c = rand(4, 2, 5, 3)
    assert np.allclose(la.solve(a, c), np.linalg.solve(a[:, None], c))
    assert np.allclose(la.pinv(a), np.linalg.inv(a)) and np.allclose(la.det(a), np.linalg.det(a))
    sign, log = la.slogdet(a)
    assert np.allclose(sign, np.linalg.slogdet(a).sign) and np.allclose(log, np.linalg.slogdet(a).logabsdet)
    assert np.allclose(rm.matmul(a, b), a @ b) and np.allclose(rm.matmul(a, v), a @ v) and np.isclose(rm.matmul(v, v), v @ v)
    spd = a @ H(a) + 5 * np.eye(5)
    assert np.allclose(la.cholesky(spd), np.linalg.cholesky(spd))
    # Signs and phases that NumPy leaves to LAPACK can differ, so factorizations are checked by what they multiply back to.
    for x in rand(3, 4, 7), rand(3, 7, 4):
        q, r = map(A, la.qr(x))
        d = np.diagonal(r, axis1=-2, axis2=-1)
        assert np.allclose(q @ r, x) and (d.imag == 0).all()
        q, r = map(A, la.qr(x, complete=True))
        assert np.allclose(q @ r, x) and np.allclose(H(q) @ q, np.eye(x.shape[-2]))
        u, s, v = map(A, la.svd(x))
        assert np.allclose(u * s[..., None, :] @ H(v), x) and np.allclose(H(u) @ u, np.eye(u.shape[-1])) and np.allclose(H(v) @ v, np.eye(v.shape[-1]))
        assert np.allclose(s, np.linalg.svd(x, compute_uv=False)) and np.allclose(la.svdvals(x), s) and np.allclose(la.pinv(x), np.linalg.pinv(x))
        tu, ts, tv = map(A, la.svd(x, rank=2))
        assert np.allclose(tu * ts[..., None, :] @ H(tv), u[..., :2] * s[..., None, :2] @ H(v[..., :2])) and la.svd(x, rank=9).s.shape == s.shape
    sym = a + H(a)
    w, vecs = map(A, la.eigh(sym))
    assert np.allclose(w, np.linalg.eigvalsh(sym)) and np.allclose(vecs * w[..., None, :] @ H(vecs), sym) and np.allclose(la.eigvalsh(sym), w)
    w, vecs = map(A, la.eig(a))
    assert np.allclose(a @ vecs, vecs * w[..., None, :]) and np.allclose(np.sort_complex(w), np.sort_complex(np.linalg.eigvals(a)))
    assert A(la.eig(sym).values).dtype == A(la.eigvals(a)).dtype == np.result_type(a, 1j)
    # A rank-deficient or non-square matrix gives the least-norm least-squares solution, and its pseudoinverse.
    P, Q = rand(9, 4), rand(9, 2)
    deficient = P[:, [0, 1, 1, 3]]
    for x, y in (P, Q), (P, Q[:, 0]), (deficient, Q):
        (got, s), want = la.lstsq(x, y), np.linalg.lstsq(x, y)
        assert np.allclose(got, want[0]) and np.allclose(s, want[3]) and np.allclose(la.solve(x, y), want[0])
    singular = deficient[:4]
    assert np.allclose(la.pinv(singular), np.linalg.pinv(singular)) and la.matrix_rank(deficient) == 3 and (la.matrix_rank(a) == 5).all()
    # The norms and condition numbers that NumPy's `ord` and `p` name, for each matrix of a stack. A singular matrix's condition number is infinite.
    for p in None, 1, 2, np.inf, 'fro':
        assert np.allclose(la.norm(a, p), np.linalg.norm(a, p, axis=(-2, -1))) and np.allclose(la.cond(a, p), np.linalg.cond(a, p))
    assert np.isinf(la.cond([[1., 0], [0, 0]])) and np.isinf(la.cond([[1., 0], [0, 0]], 1)) and np.isinf(la.cond(np.zeros((2, 2))))
    if kind is complex:
        # complex64 computes at its own width, with real results, such as singular values, in float32.
        eye = np.eye(3, dtype=np.complex64)
        assert A(la.pinv(eye)).dtype == np.complex64 and A(la.svd(eye).s).dtype == np.float32 and A(la.slogdet(eye[None])[1]).dtype == np.float32
        with pytest.raises(ValueError, match="doesn't take"): la.inner(rm.maximum, rm.add)(a, b)
        return
    # Min-plus products give shortest paths, and or-and products give reachability.
    d = np.array([[0, 3, np.inf], [np.inf, 0, 1], [2, np.inf, 0]])
    assert np.array_equal(la.inner(rm.minimum, rm.add)(d, d), np.min(d[:, :, None] + d[None], axis=1))
    assert np.allclose(la.inner(rm.maximum, rm.add)(a, b), np.max(a[..., :, :, None] + b[..., None, :, :], axis=-2))
    g = rng.random((6, 6)) < 0.3
    assert np.array_equal(rm.matmul(g, g), np.matmul(g, g)) and np.array_equal(la.inner(rm.logical_or, rm.logical_and)(g, g), np.matmul(g, g))
    # float16 computes in float32 and rounds back.
    assert A(la.pinv(np.eye(3, dtype=np.float16))).dtype == np.float16 and A(la.svd(np.eye(3, dtype=np.float32)).s).dtype == np.float32
    # Integer products compute in int64. An int16 argument converts to it, so these sums of 3 million don't overflow.
    i16 = np.full((2, 300), 100, np.int16)
    assert A(rm.matmul(i16, i16.T)).dtype == np.int64 and (rm.matmul(i16, i16.T) == 3_000_000).all()
    # Counts of int16 items compute in int64 too, so a count past int16's range is exact.
    zeros = np.zeros((1, 40_000), np.int16)
    assert la.inner(rm.add, rm.equal)(zeros, zeros.T) == 40_000
    # Integer max-plus and min-plus products, and the counts of equal and unequal items, match NumPy's results built item by item.
    n, p = rng.integers(-5, 5, (5, 7)), rng.integers(-5, 5, (7, 4))
    assert np.array_equal(la.inner(rm.maximum, rm.add)(n, p), np.max(n[:, :, None] + p[None], axis=1))
    assert np.array_equal(la.inner(rm.minimum, rm.add)(n, p), np.min(n[:, :, None] + p[None], axis=1))
    equal = (n[:, :, None] == p[None]).sum(axis=1)
    assert np.array_equal(la.inner(rm.add, rm.equal)(n, p), equal) and np.array_equal(la.inner(rm.add, rm.not_equal)(n, p), 7 - equal)
    assert np.array_equal(la.inner(rm.add, rm.equal)(n / 2, p / 2), equal)
    # A pair with no kernel at a dtype computes through the native functions, as counts of complex numbers do.
    assert la.inner(rm.add, rm.equal)(np.array([[1+1j, 2]]), np.array([[1+1j], [3]])) == 1
    # Mathish comparisons also apply through generic product callbacks.
    count = la.inner(rm.add, rm.equal)
    assert count([0.3, 1.0], [0.1 + 0.2, 1.0]) == 1 and count([[0.3 + 1j]], [[0.1 + 0.2 + 1j]]) == 0
    count = la.inner(rm.add, rm.mathish.equal)
    assert count([0.3, 1.0], [0.1 + 0.2, 1.0]) == 2 and count([[0.3 + 1j]], [[0.1 + 0.2 + 1j]]) == 1
    # NumPy's ufuncs aren't rustymath's functions.
    with pytest.raises(ValueError): la.inner(np.add, np.multiply)
    # An integer product that doesn't fit int64 raises: 2^62 times 4 is 2^64.
    with pytest.raises(OverflowError): rm.matmul([[2**62]], [[4]])
    # A matrix that can't be factored gives NaN, with no warning.
    for f, x in (la.cholesky, -np.eye(2)), (la.eig, [[1., np.nan], [0, 1]]):
        with np.errstate(all='raise'): assert np.isnan(A(f(np.array(x))[0])).all()
    assert la.slogdet([[1., 2], [2, 4]]) == (0, -np.inf) and la.pinv(np.zeros((0, 0))).shape == (0, 0)

def test_keyed_axes():
    # Contracted axes with keys pair by key, as rustymath broadcasts, and results keep the keys of the argument axes they index.
    la, v = rm.linalg, rm.array({'a': 1, 'b': 2})
    assert v @ rm.array({'b': 10, 'a': 20}) == 40
    assert la.inner(rm.add, rm.subtract)(v, rm.array({'b': 10, 'a': 20})) == -27
    m = rm.array({'x': {'a': 1., 'b': 2}, 'y': {'a': 3., 'b': 4}})
    mv = m @ rm.array({'b': 10., 'a': 20})
    assert mv.tolist() == [40, 100] and mv.keys == (('x', 'y'),)
    with pytest.raises(ValueError): v @ rm.array({'b': 1, 'c': 2})
    a = rm.array({'x': {'a': 2., 'b': 0}, 'y': {'a': 0., 'b': 4}})
    x = la.solve(a, rm.array({'y': 8., 'x': 2}))
    assert x.tolist() == [1, 2] and x.keys == (('a', 'b'),)
    assert la.pinv(a).keys == (('a', 'b'), ('x', 'y'))

def test_matrix_batches(threads):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(64, 6, 6))
    a = a @ a.swapaxes(-1, -2) + np.eye(6)
    w, vectors = map(A, rm.linalg.eig(a))
    assert w.dtype == np.complex128 and np.allclose(a @ vectors, vectors * w[..., None, :], atol=1e-9)
    for x in [a[:1], a[:3], np.eye(128)[None]]:
        assert np.allclose(rm.linalg.matmul(x, x), x @ x)

def test_matrix_layouts(threads):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(64, 8, 8))[:, ::-1, ::-1]
    b = rng.normal(size=(8, 5))[::-1]
    assert np.allclose(rm.linalg.matmul(a, b), a @ b)
    for dtype in [bool, np.int64]:
        x, y = (a > 0).astype(dtype), (b > 0).astype(dtype)
        assert np.array_equal(rm.linalg.matmul(x, y), x @ y)
    x, y = (a > 0).astype(np.float16), (b > 0).astype(np.float16)
    counts = rm.linalg.inner(rm.add, rm.equal)(x, y)
    assert A(counts).dtype == np.int64 and np.array_equal(counts, np.sum(x[..., None] == y[None, None], axis=-2))
    spd = a @ a.swapaxes(-1, -2) + np.eye(8)
    assert np.allclose(rm.linalg.slogdet(spd)[1], np.linalg.slogdet(spd).logabsdet)
    s = rng.normal(size=(3, 8, 5))
    for dtype in ('float32', 'float64', 'complex64', 'complex128'):
        m, st = rm.array(b, dtype=dtype), rm.array(s, dtype=dtype)
        for view in (m.T, m[::-1], m[:, ::2]):
            dense = view.compact()
            assert np.allclose(view.T @ view, dense.T @ dense, rtol=1e-5, atol=1e-6)
        for p, q in [(m[:, 1] @ m, b[:, 1] @ b), (m.T @ m[:, 1], b.T @ b[:, 1]), (rm.swapaxes(st, -1, -2) @ st, s.swapaxes(-1, -2) @ s), (st[:, ::-1] @ m.T, s[:, ::-1] @ b.T)]:
            assert np.allclose(p, q, rtol=1e-5, atol=1e-5)

def test_matrix_failures(threads):
    a = np.broadcast_to(np.eye(3), (2, 64, 3, 3)).copy()
    a[0, -1, 0, 0] = -1
    l = A(rm.linalg.cholesky(a))
    assert np.isnan(l[0, -1]).all() and np.isfinite(l[1]).all() and np.isfinite(l[0, :-1]).all()

