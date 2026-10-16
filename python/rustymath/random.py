"""Persistent random generators. Rust owns the RNG and bulk samplers; Python composes the conveniences."""
from operator import index
from . import _core

class Generator(_core.random.Generator):
    "A persistent generator, optionally initialized with one seed"

    def random(self, size=None):
        "Uniform values in [0, 1)"
        return self.uniform(size=size)

    def uniform(self, low=0., high=1., size=None):
        "Uniform values between low and high"
        return self.sample(_core.distribution.uniform(low, high), size)

    def normal(self, mean=0., sd=1., size=None):
        "Normal values with this mean and standard deviation"
        return self.sample(_core.distribution.normal(mean, sd), size)

    def exponential(self, rate=1., size=None):
        "Exponential values with this rate"
        return self.sample(_core.distribution.exponential(rate), size)

    def multivariate_normal(self, mean, cov, size=None):
        "Float64 samples from a real mean vector and positive-definite covariance matrix, both without axis keys or names"
        mean, cov = (x if isinstance(x, _core.Array) else _core.array(x) for x in (mean, cov))
        if mean.ndim != 1: raise ValueError('mean must be a vector')
        if any(_core.issubdtype(x.dtype, 'complex') or any(k is not None for k in (*x.keys, *x.names)) for x in (mean, cov)):
            raise ValueError('multivariate normal needs a real mean and covariance without axis keys or names')
        factor = _core.linalg.cholesky(cov)
        n = mean.size
        if factor.shape != (n, n): raise ValueError('covariance must be one square matrix matching mean')
        shape = () if size is None else (index(size),) if hasattr(size, '__index__') else tuple(size)
        draws = self.normal(size=(*shape, n))
        shifted = _core.reshape(mean, (*([1] * len(shape)), n))
        return _core.add(_core.linalg.matmul(draws, factor.T), shifted)

def default_rng(seed=None):
    "Create a generator; subsequent calls advance its state"
    return Generator(seed)
