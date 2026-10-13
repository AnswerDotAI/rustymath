import itertools
import numpy as np, pytest, rustymath as rm
from rustymath._core import linalg as gufunc

def test_lu(threads):
    la = rm.linalg
    base = np.array([[0., 2, 1], [3, 1, -1], [1, 0, 4]])
    b = np.arange(12.).reshape(3, 4)[:, ::-1]
    for dt in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
        a = np.stack([base + k / 8 * np.eye(3) for k in range(64)]).astype(dt)
        if np.issubdtype(dt, np.complexfloating): a += 0.25j
        f = la.lu(a)
        assert f.factors.dtype == dt and f.p.dtype == np.int64 and np.allclose(np.tril(f.l), f.l) and np.allclose(np.triu(f.u), f.u)
        assert np.allclose(f.l @ f.u, np.take_along_axis(a, f.p[..., :, None], -2), rtol=0.002, atol=0.002)
        wide = a.astype(np.complex128 if np.iscomplexobj(a) else np.float64)
        assert np.allclose(f.solve(b), np.linalg.solve(wide, b), rtol=0.003, atol=0.003)
        assert np.allclose(la.solve(a, b), np.linalg.solve(wide, b), rtol=0.003, atol=0.003)
        assert np.allclose(f.solve(b[:, 0]), np.linalg.solve(wide, b[:, :1])[..., 0], rtol=0.003, atol=0.003)
        assert np.allclose(f.inverse(), np.linalg.inv(wide), rtol=0.003, atol=0.003)
        assert np.allclose(f.determinant(), np.linalg.det(wide), rtol=0.003, atol=0.003)
        sign, log = f.log_determinant()
        want = np.linalg.slogdet(wide)
        assert np.allclose(sign, want.sign, rtol=0.003, atol=0.003) and np.allclose(log, want.logabsdet, rtol=0.003, atol=0.003) and log.dtype == a.real.dtype
    # Factors can be strided too, and malformed row positions must not panic in a NumPy loop.
    f = la.lu(base)
    packed, perm = np.zeros((6, 6)), np.zeros(6, np.int64)
    packed[::2, ::2], perm[::2] = f
    assert np.allclose(la.Lu(packed[::2, ::2], perm[::2]).solve(b), f.solve(b))
    bad = la.Lu(f.factors, np.array([-1, 0, 1]))
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): bad.inverse()

def test_svd(threads):
    la = rm.linalg
    a = np.array([[1., 2, 1], [2, 1, 2], [3, 0, 3], [0, 2, 0]])
    for dt in [np.float32, np.float64, np.complex64, np.complex128]:
        for x in [a, a.T]:
            x = (x * np.resize([1., 2.], 64)[:, None, None]).astype(dt)
            if np.iscomplexobj(x): x *= 1 + 0.25j
            f, b = la.svd(x), np.arange(x.shape[-2] * 2.).reshape(-1, 2)
            inverse = np.linalg.pinv(x, rcond=np.finfo(x.real.dtype).eps * max(x.shape[-2:]))
            assert np.allclose(f.solve(b), inverse @ b, rtol=2e-5, atol=2e-5)
            assert np.allclose(f.solve(b[:, 0]), (inverse @ b)[..., 0], rtol=2e-5, atol=2e-5)
            assert np.allclose(f.pseudoinverse(), inverse, rtol=2e-5, atol=2e-5)
            assert f.s.dtype == x.real.dtype and np.allclose((f.u * f.s[..., None, :]) @ f.v.conj().swapaxes(-1, -2), x, rtol=2e-5, atol=2e-5)
    f = la.svd(np.eye(3, dtype=np.float16))
    assert f.solve(np.ones(3, np.float16)).dtype == np.float16 and f.pseudoinverse().dtype == np.float16

def test_permanent_and_echelon():
    la = rm.linalg
    a = np.array([[1., 2, 3], [4, 5, 6], [7, 8, 10]])
    for dt in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
        x = a.astype(dt)
        if np.iscomplexobj(x): x += 0.5j
        want = sum(np.prod(x[np.arange(3), p]) for p in itertools.permutations(range(3)))
        assert np.allclose(la.permanent(np.stack([x, x.T])), want, rtol=0.002)
        f = la.echelon(x)
        assert np.allclose(f.rows, np.eye(3), atol=1e-5) and f.pivots.tolist() == [0, 1, 2] and f.rows.dtype == dt
    f = la.echelon([[0., 2, 4], [0, 1, 2], [0, 0, 0]], columns=2)
    assert f.rows.tolist() == [[0, 1, 2]] and f.pivots.tolist() == [1] and f.determinant == 2
    assert la.permanent(np.empty((0, 0))) == 1 and la.echelon(np.empty((0, 3))).rows.shape == (0, 3)
    with pytest.raises(ValueError): la.permanent(np.eye(64))

def test_and_or_product():
    a = np.array([[1, 0, 1], [0, 1, 0]], bool)
    b = np.array([[0, 1], [1, 0], [0, 0]], bool)
    product = rm.linalg.semiring(np.logical_and, np.logical_or)
    assert np.array_equal(product(a, b), np.all(a[:, :, None] | b[None], axis=1))
    assert product(np.empty((2, 0), bool), np.empty((0, 3), bool)).all()

@pytest.mark.parametrize('kind', [float, complex])
def test_linalg(kind):
    la, rng = rm.linalg, np.random.default_rng(0)
    H = lambda x: np.conj(np.swapaxes(x, -1, -2))
    rand = lambda *shape: rng.standard_normal(shape) + 1j * rng.standard_normal(shape) if kind is complex else rng.standard_normal(shape)
    a, b = rand(4, 5, 5), rand(4, 5, 3)
    v = b[0, :, 0]
    assert np.allclose(la.solve(a, b), np.linalg.solve(a, b)) and np.allclose(la.solve(a[0], v), np.linalg.solve(a[0], v))
    assert np.allclose(la.pinv(a), np.linalg.inv(a)) and np.allclose(la.det(a), np.linalg.det(a))
    sign, log = la.slogdet(a)
    assert np.allclose(sign, np.linalg.slogdet(a).sign) and np.allclose(log, np.linalg.slogdet(a).logabsdet)
    assert np.allclose(rm.matmul(a, b), a @ b) and np.allclose(rm.matmul(a, v), a @ v) and np.isclose(rm.matmul(v, v), v @ v)
    spd = a @ H(a) + 5 * np.eye(5)
    assert np.allclose(la.cholesky(spd), np.linalg.cholesky(spd))
    # Signs and phases that NumPy leaves to LAPACK can differ, so factorizations are checked by what they multiply back to.
    for x in rand(3, 4, 7), rand(3, 7, 4):
        q, r = la.qr(x)
        d = np.diagonal(r, axis1=-2, axis2=-1)
        assert np.allclose(q @ r, x) and (d.imag == 0).all()
        q, r = la.qr(x, complete=True)
        assert np.allclose(q @ r, x) and np.allclose(H(q) @ q, np.eye(x.shape[-2]))
        u, s, v = la.svd(x)
        assert np.allclose(u * s[..., None, :] @ H(v), x) and np.allclose(H(u) @ u, np.eye(u.shape[-1])) and np.allclose(H(v) @ v, np.eye(v.shape[-1]))
        assert np.allclose(s, np.linalg.svd(x, compute_uv=False)) and np.allclose(la.svdvals(x), s) and np.allclose(la.pinv(x), np.linalg.pinv(x))
        t = la.svd(x, rank=2)
        assert np.allclose(t.u * t.s[..., None, :] @ H(t.v), u[..., :2] * s[..., None, :2] @ H(v[..., :2])) and la.svd(x, rank=9).s.shape == s.shape
    sym = a + H(a)
    w, vecs = la.eigh(sym)
    assert np.allclose(w, np.linalg.eigvalsh(sym)) and np.allclose(vecs * w[..., None, :] @ H(vecs), sym) and np.allclose(la.eigvalsh(sym), w)
    w, vecs = la.eig(a)
    assert np.allclose(a @ vecs, vecs * w[..., None, :]) and np.allclose(np.sort_complex(w), np.sort_complex(np.linalg.eigvals(a)))
    assert la.eig(sym).values.dtype == la.eigvals(a).dtype == np.result_type(a, 1j)
    # A rank-deficient or non-square matrix gives the least-norm least-squares solution, and its pseudoinverse.
    A, B = rand(9, 4), rand(9, 2)
    deficient = A[:, [0, 1, 1, 3]]
    for x, y in (A, B), (A, B[:, 0]), (deficient, B):
        (got, s), want = la.lstsq(x, y), np.linalg.lstsq(x, y)
        assert np.allclose(got, want[0]) and np.allclose(s, want[3]) and np.allclose(la.solve(x, y), want[0])
    singular = deficient[:4]
    assert np.allclose(la.pinv(singular), np.linalg.pinv(singular)) and la.matrix_rank(deficient) == 3 and (la.matrix_rank(a) == 5).all()
    if kind is complex:
        # complex64 computes at its own width, with real results, such as singular values, in float32.
        eye = np.eye(3, dtype=np.complex64)
        assert la.pinv(eye).dtype == np.complex64 and la.svd(eye).s.dtype == np.float32 and la.slogdet(eye)[1].dtype == np.float32
        with pytest.raises(TypeError, match='real matrices'): la.semiring(np.maximum, np.add)(a, b)
        return
    # Min-plus products give shortest paths, and or-and products give reachability.
    d = np.array([[0, 3, np.inf], [np.inf, 0, 1], [2, np.inf, 0]])
    assert (la.semiring(np.minimum, np.add)(d, d) == np.min(d[:, :, None] + d[None], axis=1)).all()
    assert np.allclose(la.semiring(rm.maximum, rm.add)(a, b), np.max(a[..., :, :, None] + b[..., None, :, :], axis=-2))
    g = rng.random((6, 6)) < 0.3
    assert (rm.matmul(g, g) == np.matmul(g, g)).all() and (la.semiring(np.logical_or, np.logical_and)(g, g) == np.matmul(g, g)).all()
    # float16 computes in float32 and rounds back.
    assert la.pinv(np.eye(3, dtype=np.float16)).dtype == np.float16 and la.svd(np.eye(3, dtype=np.float32)).s.dtype == np.float32
    # Integer products compute in int64. An int8 argument converts to it, so these sums of 3 million don't overflow.
    i8 = np.full((2, 300), 100, np.int8)
    assert rm.matmul(i8, i8.T).dtype == np.int64 and (rm.matmul(i8, i8.T) == 3_000_000).all()
    # Integer max-plus and min-plus products, and the counts of equal and unequal items, match NumPy's results built item by item.
    n, p = rng.integers(-5, 5, (5, 7)), rng.integers(-5, 5, (7, 4))
    assert (la.semiring(np.maximum, np.add)(n, p) == np.max(n[:, :, None] + p[None], axis=1)).all()
    assert (la.semiring(np.minimum, np.add)(n, p) == np.min(n[:, :, None] + p[None], axis=1)).all()
    equal = (n[:, :, None] == p[None]).sum(axis=1)
    assert (la.semiring(np.add, np.equal)(n, p) == equal).all() and (la.semiring(np.add, np.not_equal)(n, p) == 7 - equal).all()
    assert (la.semiring(np.add, np.equal)(n / 2, p / 2) == equal).all()
    # Integer arithmetic wraps on overflow: 2^62 times 4 is 2^64, which wraps to 0.
    assert rm.matmul([[2**62]], [[4]])[0, 0] == 0
    # A matrix that can't be factored gives NaN and NumPy's "invalid value" warning, as a failing item of a ufunc does.
    for f, x in (la.cholesky, -np.eye(2)), (la.eig, [[1., np.nan], [0, 1]]):
        with pytest.warns(RuntimeWarning, match='invalid value'): assert np.isnan(f(np.array(x))[0]).all()
    assert la.slogdet([[1., 2], [2, 4]]) == (0, -np.inf) and la.pinv(np.zeros((0, 0))).shape == (0, 0)

def test_matrix_batches(threads):
    rng = np.random.default_rng(3)
    a = rng.normal(size=(64, 6, 6))
    a = a @ a.swapaxes(-1, -2) + np.eye(6)
    eigen = rm.linalg.eig(a)
    assert eigen.values.dtype == np.complex128 and np.allclose(a @ eigen.vectors, eigen.vectors * eigen.values[..., None, :], atol=1e-9)
    for x in [a[:1], a[:3], np.eye(128)[None]]:
        assert np.allclose(rm.linalg.matmul(x, x), x @ x)

def test_matrix_layouts(threads):
    rng = np.random.default_rng(4)
    a = rng.normal(size=(64, 8, 8))[:, ::-1, ::-1]
    b = rng.normal(size=(8, 5))[::-1]
    out = np.empty((64, 8, 10))[:, ::-1, ::-2]
    assert rm.linalg.matmul(a, b, out=out) is out and np.allclose(out, a @ b)
    for dtype in [bool, np.int64]:
        x, y = (a > 0).astype(dtype), (b > 0).astype(dtype)
        assert np.array_equal(rm.linalg.matmul(x, y), x @ y)
    x, y = (a > 0).astype(np.float16), (b > 0).astype(np.float16)
    counts = rm.linalg.semiring(np.add, np.equal)(x, y)
    assert counts.dtype == np.int64 and np.array_equal(counts, np.sum(x[..., None] == y[None, None], axis=-2))
    x = a.copy()
    assert rm.linalg.matmul(x, np.eye(8) * 2, out=x) is x and np.allclose(x, a * 2)
    repeated = np.broadcast_to(np.empty((8, 5)), (64, 8, 5))
    repeated.flags.writeable = True
    rm.linalg.matmul(a, b, out=repeated)
    assert np.allclose(repeated[-1], a[-1] @ b)
    shared = np.empty(64)
    gufunc.slogdet(a @ a.swapaxes(-1, -2) + np.eye(8), out=(shared, shared))
    assert np.allclose(shared, np.linalg.slogdet(a @ a.swapaxes(-1, -2) + np.eye(8)).logabsdet)

def test_matrix_failures(threads):
    a = np.broadcast_to(np.eye(3), (2, 64, 3, 3)).copy()
    a[0, -1, 0, 0] = -1
    out = np.empty((2, 129, 3, 3))[:, 1::2]
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): gufunc.cholesky(a, out=out)
    assert np.isnan(out[0, -1]).all() and np.isfinite(out[1]).all()
