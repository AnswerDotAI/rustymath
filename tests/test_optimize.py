import numpy as np, pytest
from rustymath.optimize import minimize, root_scalar

def rosen(x): return (1-x[0])**2 + 100*(x[1]-x[0]**2)**2
def rosen_grad(x): return np.array([-2*(1-x[0]) - 400*x[0]*(x[1]-x[0]**2), 200*(x[1]-x[0]**2)])

def test_minimize():
    for method, kw in [('nelder-mead', {}), ('Powell', {}), ('l-bfgs', dict(jac=rosen_grad)), ('differential-evolution', dict(bounds=[(-2, 2)]*2))]:
        r = minimize(rosen, [-1.2, 1], method=method, **kw)
        assert r.success and np.allclose(r.x, 1, atol=1e-4), (method, r)
    assert minimize(rosen, [-1.2, 1], jac=rosen_grad).nfev < 100
    r = minimize(rosen, np.array([-1.2, 1], np.float32))
    assert r.x.dtype == np.float32 and np.allclose(r.x, 1, atol=1e-2)
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
    assert not root_scalar(lambda x: x*x + 1, [0, 1]).converged
