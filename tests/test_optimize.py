import numpy as np, pytest
from rustymath.optimize import minimize, root_scalar, least_squares, cg, minres, gmres, lsmr

def rosen(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosen_grad(x): return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

def test_minimize():
    for method, kw in [('nelder-mead', {}), ('Powell', {}), ('l-bfgs', dict(jac=rosen_grad)), ('differential-evolution', dict(bounds=[(-2, 2)]*2))]:
        r = minimize(rosen, [-1.2, 1], method=method, **kw)
        assert r.success and np.allclose(r.x, 1, atol=1e-4), (method, r)
    assert minimize(rosen, [-1.2, 1], jac=rosen_grad).nfev < 100
    r = minimize(rosen, np.array([-1.2, 1], np.float32))
    assert r.x.dtype == 'float32' and np.allclose(r.x, 1, atol=1e-2)
    assert np.allclose(minimize(rosen, [0.5, 0.5], bounds=[(None, 0.8), (0, None)]).x, [0.8, 0.64], atol=1e-6)
    shapes = []
    def columns(x): shapes.append(x.shape); return rosen(x)
    r = minimize(columns, [0, 0], method='differential-evolution', bounds=[(-2, 2)]*2, options=dict(popsize=10, seed=1), vectorized=True)
    assert np.allclose(r.x, 1, atol=1e-4) and shapes[0] == (2, 20)
    with pytest.raises(ValueError, match='gradient'): minimize(rosen, [0, 0], method='l-bfgs')
    with pytest.raises(ValueError, match='unknown method'): minimize(rosen, [0, 0], method='bfgs')
    with pytest.raises(TypeError, match='maxiter'): minimize(rosen, [0, 0], options=dict(maxiter=5))
    with pytest.raises(ZeroDivisionError): minimize(lambda x: 1/0, [0, 0])

def test_root_scalar():
    r = root_scalar(lambda x: np.cos(x) - x, [0, 1])
    assert r.converged and r.root == pytest.approx(0.7390851332151607, abs=1e-8) and r.function_calls < 20
    assert root_scalar(lambda x: np.cos(x) - x, x0=-20).root == pytest.approx(0.7390851332151607, abs=1e-8)
    assert not root_scalar(lambda x: x*x + 1, [0, 1]).converged
    with pytest.raises(TypeError, match='bracket or x0'): root_scalar(np.cos)

def test_least_squares():
    t = np.linspace(0, 2, 5)
    calls = []
    def fit(p):
        calls.append(p)
        return p[0]*np.exp(p[1]*t) - 2*np.exp(0.3*t)
    def jac(p): return np.column_stack([np.exp(p[1]*t), p[0]*t*np.exp(p[1]*t)])
    r = least_squares(fit, [1, 1])
    assert r.success and np.allclose(r.x, [2, 0.3]) and r.cost < 1e-20 and r.nfev == len(calls)
    assert np.allclose(least_squares(fit, [1, 1], jac=jac).x, [2, 0.3])
    y, s = np.array([1, 2.9, 5.2, 6.8, 9.1]), np.arange(5.)
    assert np.allclose(least_squares(lambda c: c[0] + c[1]*s - y, [0, 0]).x, np.polyfit(s, y, 1)[::-1])

def test_linear():
    rng = np.random.default_rng(0)
    m, b = rng.uniform(-0.5, 0.5, (30, 30)), rng.uniform(size=30)
    spd, general = m.T @ m + np.eye(30), m + 4*np.eye(30)
    for solve in (cg, minres, gmres):
        r = solve(spd, b, rtol=1e-12)
        assert r.success and np.allclose(r.x, np.linalg.solve(spd, b)) and r.normr < 1e-9, solve.__name__
    assert np.allclose(gmres(general, b, rtol=1e-12).x, np.linalg.solve(general, b))
    # `A` and `b` convert together, so a float32 `b` beside a float64 `A` promotes to float64 rather than rounding `A`.
    assert cg(spd.astype(np.float32), b.astype(np.float32)).x.dtype == 'float32' and cg(spd, b.astype(np.float32)).x.dtype == 'float64'
    # `maxiter` counts iterations as SciPy does. Four CG steps take four products, plus the residual's at the start and at the end. One
    # GMRES cycle takes 20 steps and its own residual's product.
    assert [cg(spd, b, maxiter=4).nmatvec, gmres(general, b, rtol=1e-14, maxiter=1).nmatvec] == [6, 22]
    tall, y = rng.uniform(size=(60, 20)), rng.uniform(size=60)
    r = lsmr(tall, y, rtol=1e-12)
    assert r.success and np.allclose(r.x, np.linalg.lstsq(tall, y, rcond=None)[0])
    # Three LSMR steps take a product with `A` and one with its transpose each, after one for Aᵀb and two for the starting residual,
    # and the residual at the end takes one more.
    assert lsmr(tall, y, maxiter=3).nmatvec == 1 + 2 + 3*2 + 1
    # Jacobi's preconditioner, the reciprocal of the diagonal, cuts the products for a matrix whose diagonal spans four orders.
    d = 10**np.linspace(0, 2, 30)
    scaled = d[:, None] * spd * d
    plain, fast = cg(scaled, b, rtol=1e-10), cg(scaled, b, rtol=1e-10, M=np.diag(1/np.diag(scaled)))
    assert fast.success and 2*fast.nmatvec < plain.nmatvec and np.allclose(fast.x, np.linalg.solve(scaled, b))
    class Products:
        "A matrix that gives only its products, as SciPy's `LinearOperator` can"
        shape = tall.shape
        def matvec(self, v): return tall @ v
    with pytest.raises(TypeError, match='rmatvec'): lsmr(Products(), y)
    with pytest.raises(ValueError, match='transpose'): cg(tall, y)
