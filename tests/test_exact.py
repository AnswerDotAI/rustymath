import math
from fractions import Fraction as F
import numpy as np, pytest, rustymath as rm

def test_rational_arithmetic():
    ex = rm.exact
    assert ex.gcd(F(2, 3), F(4, 5)) == F(2, 15) and isinstance(ex.gcd(6, 4), F)
    assert ex.lcm(F(-2, 3), F(4, 5)) == -4 and ex.remainder(F(-5, 2), F(3, 2)) == F(1, 2)
    assert ex.remainder(F(7, 3), 0) == F(7, 3) and ex.binom(F(3, 2), 2) == F(3, 8)
    assert ex.binom(-2, -3) == -2 and ex.binom(100, 50) == F(100891344545564193334812497256)
    got = ex.gcd([[F(2, 3)], [F(4, 3)]], np.array([F(4, 5), F(8, 5)], dtype=object))
    assert got.dtype == object and got.tolist() == [[F(2, 15), F(2, 15)], [F(4, 15), F(4, 15)]]
    assert ex.gcd(np.arange(6)[::-2], 2).tolist() == [F(1), F(1), F(1)] and ex.gcd([], 2).shape == (0,)
    with pytest.raises(TypeError): ex.gcd(0.5, F(1, 2))
    with pytest.raises(ValueError): ex.binom(3, F(1, 2))

def test_exact_matrices():
    ex = rm.exact
    a = np.array([[2, 1], [1, 2]], dtype=object)
    assert ex.solve(a, [1, 0]).tolist() == [F(2, 3), F(-1, 3)] and ex.det(a) == 3 and ex.permanent(a) == 5
    inverse = ex.pinv(a)
    assert np.array_equal(a @ inverse, np.eye(2)) and np.array_equal(ex.solve(a, np.eye(2, dtype=int)), inverse)
    singular = np.array([[1, 2], [2, 4]], dtype=object)
    p = ex.pinv(singular)
    assert np.array_equal(singular @ p @ singular, singular) and ex.solve(singular, [1, 2]).tolist() == [F(1, 5), F(2, 5)]
    rows, pivots, determinant = ex.echelon([[0, 2, 4], [1, 3, 5]], columns=2)
    assert rows.tolist() == [[F(1), F(0), F(-1)], [F(0), F(1), F(2)]] and pivots.tolist() == [0, 1] and determinant == -2
    assert ex.pinv(np.empty((0, 2), dtype=object)).shape == (2, 0) and ex.det(np.empty((0, 0), dtype=object)) == 1
    with pytest.raises(ValueError): ex.solve(np.zeros((2, 2, 2), dtype=int), [1, 2])

def test_exact_polynomials():
    ex = rm.exact
    roots, x = [F(1, 3), F(2, 3)], np.array([0, F(1, 2), 1], dtype=object)
    c = ex.polyfromroots(roots, m=3)
    assert c.tolist() == [F(2, 3), F(-3), F(3)] and ex.polyder(c).tolist() == [F(-3), F(6)]
    assert np.array_equal(ex.polyval(x, c), ex.polyvalfromroots(x, roots, m=3)) and ex.polyval(F(1, 3), c) == 0
    assert ex.polyval(7, []) == 0 and ex.polyder([3]).shape == (0,)

def test_polynomial():
    from numpy.polynomial import polynomial as P
    c, x = [2.0, -3.0, 1.0], np.linspace(-2, 3, 50)
    assert rm.polynomial.polyval(2.0, [1, 2, 3]) == 17 and rm.polynomial.polyval(np.float32(x), np.float32(c)).dtype == np.float32
    assert np.allclose(rm.polynomial.polyval(x.reshape(5, 10), c), P.polyval(x, c).reshape(5, 10))
    assert np.allclose(rm.polynomial.polyroots(c), [1, 2]) and rm.polynomial.polyroots(np.float16(c)).dtype == np.complex64
    assert np.allclose(rm.polynomial.polyroots([1.0, 0, 1]), [-1j, 1j]) and len(rm.polynomial.polyroots([5.0])) == 0
    # The eigenvalue solver fails on NaN coefficients, so each root is NaN and NumPy reports an invalid value.
    with np.errstate(invalid='raise'), pytest.raises(FloatingPointError): rm.polynomial.polyroots([np.nan, 0, 1.0])
    with np.errstate(invalid='ignore'): assert np.isnan(rm.polynomial.polyroots([np.nan, 0, 1.0, 0])).sum() == 2
    assert rm.polynomial.polyfromroots([1, 2]).tolist() == [2, -3, 1] and rm.polynomial.polyfromroots([1j, -1j]).tolist() == [1, 0, 1]
    assert rm.polynomial.polyder([1, 2, 3, 4]).tolist() == [2, 6, 12] and len(rm.polynomial.polyder([3.0])) == 0
    assert np.allclose(rm.polynomial.polyfit([0, 1, 2, 3], [1, 3, 7, 13], 2), [1, 1, 1])

def test_number_theory():
    nt = rm.number_theory
    assert nt.is_prime(2**127 - 1) and not nt.is_prime(2**128 + 1) and nt.factor(600851475143) == [(71, 1), (839, 1), (1471, 1), (6857, 1)]
    assert nt.next_prime(2**64) == 2**64 + 13 and nt.prev_prime(2) is None and nt.totient(36) == 12
    assert nt.prime_count(10**6) == 78498 and nt.primes(20).tolist() == [2, 3, 5, 7, 11, 13, 17, 19] and nt.primes(7, 12).tolist() == [7, 11]
    with pytest.raises(ValueError): nt.factor(0)
    assert nt.factorial(100) == math.factorial(100) and nt.factorial(0) == 1

def test_polynomial_fields():
    p = rm.polynomial
    for dt in [np.float16, np.float32, np.float64, np.complex64, np.complex128]:
        roots = np.array([1, 2], dt)
        x = np.array([[0, 1], [2, 3]], dt)
        if np.dtype(dt).kind == 'c': roots += np.array([1j, -1j], dt); x += dt(0.5j)
        c = p.polyfromroots(roots, m=dt(2))
        want = 2 * (x - roots[0]) * (x - roots[1])
        for got in [p.polyval(x[:, ::-1], c), p.polyvalfromroots(x[:, ::-1], roots, m=dt(2))]:
            assert got.dtype == dt and np.allclose(got, want[:, ::-1], rtol=2e-3)
        assert c.dtype == dt and np.allclose(c, [2 * roots.prod(), -2 * roots.sum(), 2])
        d = p.polyder(c)
        assert d.dtype == dt and np.allclose(d, [c[1], 2 * c[2]])
    assert p.polyval(1j, [1, 2j]) == -1 and p.polyvalfromroots(3, [], m=2) == 2
    assert p.polyfromroots([], m=3).tolist() == [3] and p.polyval([np.inf], [2, 0]).tolist() == [2]
    with pytest.raises(ValueError, match='scalar'): p.polyfromroots([1, 2], m=[1, 2])
    with pytest.raises(TypeError, match='real points'): p.polyfit([1j, 2j], [1, 2], 1)

def test_mixed_radix():
    bases, values = [0, 24, 60], np.array([[1563, 3064], [-1, 0]])
    digits = rm.integer.encode(bases, values)
    assert digits.shape == (2, 2, 3) and digits[0].tolist() == [[1, 2, 3], [2, 3, 4]]
    assert np.array_equal(rm.integer.decode(bases, digits), values) and rm.integer.encode([], [1, 2]).shape == (2, 0)
    for decode in [rm.integer.decode, rm.polynomial.decode]: assert decode(bases, [1, 2, 3]) == 1563
    assert rm.exact.decode(bases, [[1, 2, 3], [2, 3, 4]]).tolist() == [F(1563), F(3064)]
    digits = np.array([[[1, 2, 3], [2, 3, 4]], [[3, 4, 5], [4, 5, 6]]], np.float32)
    got = rm.polynomial.decode(np.array([0, 24, 60], np.float32), digits[:, ::-1])
    assert got.dtype == np.float32 and np.array_equal(got, ((digits[..., 0] * 24 + digits[..., 1]) * 60 + digits[..., 2])[:, ::-1])
    assert rm.polynomial.decode(np.complex64([0, 1j]), np.complex64([2, 3])).dtype == np.complex64
    assert rm.polynomial.decode([0, 10], np.empty((0, 2))).shape == (0,)
    for decode in [rm.integer.decode, rm.polynomial.decode, rm.exact.decode]:
        for bases, digits in [([], []), ([[1, 2]], [1, 2]), ([1, 2], [1]), ([1, 2], 1)]:
            with pytest.raises(ValueError): decode(bases, digits)
    with pytest.raises(OverflowError): rm.integer.decode([0, 2], [2**62, 0])
    with pytest.raises(OverflowError): rm.integer.encode([0], np.array([2**63], dtype=np.uint64))
    with pytest.raises(TypeError): rm.integer.encode([10], [1.5])
