

# Accuracy

Float functions give results within 4 units in the last place (ulps) of
the correctly rounded result, apart from the exceptions below. Where
rustymath gives up a few ulps for real speed or a simpler
implementation, the loss is listed below. An exactly representable
result need not be returned exactly.

## Exceptions

- Near its zeros at 1 and 2, `lgamma` has an absolute error of up to 4
  ulps of 2, not 4 ulps of the result. Near its zeros, `digamma` has an
  absolute error of up to 4 ulps of 1, and below -20 up to 4 ulps of
  ln(1 - x).
- For a non-whole n and a whole k up to 64, `binom` and `binom_dn` are
  within 2k ulps.
- `binom_dk` has no relative bound near n = 2k, where its two ψ terms
  cancel.
- `beta` is within 6 ulps at `f64`. It adds three ln Γ terms. Where two
  of them have arguments of 10 or more, it computes those two as one
  difference. For each other term, `beta` rounds the logarithm of its
  Lanczos sum to a single `f64`. That rounding causes the extra error.
- A positive whole power from 1 to 8 is computed by squaring and
  multiplying, without compensation, in the scalar form, and in the
  slice form when one exponent applies to every item. The rounding error
  of the step that gives x^k is multiplied by about m/k in x^m. The
  worst errors measured over every `f32` significand and a million
  random `f64` inputs are 4.4 ulps for x^7 and 5.2 for x^8.
- Normal and log-normal evaluation of `f32` inputs uses `f32`
  arithmetic, with parameters and precomputed coefficients rounded to
  `f32`. Here z is the standard score: of x for the normal distribution,
  and of ln x for the log-normal. Where |z| \< 3, their densities are
  within 12 ulps and their cdfs within 14. As z² grows, the errors grow,
  to about 80 ulps for densities and 170 for cdfs at |z| = 12. Their
  quantiles are within 13 ulps, except where the normal quantile, mean +
  sd·z, is near 0. There cancellation leaves an absolute error of up to
  8 ulps of the standard deviation. The log-normal quantile is the
  exponential of a normal quantile, so this cancellation affects it near
  a quantile of 1. These figures exclude additional error from parameter
  rounding and ill-conditioned location/scale transformations.
- Uniform, exponential, Cauchy, Laplace, logistic and Weibull evaluation
  of `f32` inputs uses `f32` arithmetic too, with parameters and
  precomputed coefficients rounded to `f32`. For both `f32` and `f64`,
  their densities and cdfs are within 4 ulps, with two exceptions. The
  logistic density at `f64` is within 4.4 ulps because of rounding in
  its exponential, square and quotient. For the Weibull density, z is
  x/scale. The density is within 4 ulps up to z^shape = 4. Above that,
  its error grows to about z^shape ulps, such as 15, 50 and 240 ulps at
  z^shape = 16, 64 and 256. Rounding z^shape before computing
  e<sup>(-z</sup>shape) causes that much error, however z^shape is
  computed. Their quantiles are within 4 ulps, except for the Cauchy,
  Laplace and logistic laws near a quantile of 0. There the quantile is
  location + scale·q, where q is the standard law’s quantile.
  Cancellation in that sum leaves an absolute error of up to 4 ulps of
  the scale. These figures exclude additional error from parameter
  rounding, ill-conditioned location/scale transformations, and
  arguments whose standardized value is subnormal.
- The error of a complex function is measured in ulps of the result’s
  magnitude: |f(z) - exact| is within 4 ulps of |exact|. When one part
  of the result is much smaller than the other, that part can therefore
  have a large relative error. Complex ln Γ is measured in ulps of the
  larger of its magnitude and 2. Rustymath computes complex powers, Γ
  and binomial coefficients as the exponential of an exponent e, which
  is w ln z for z^w and ln Γ for Γ and binomial coefficients. Their
  error is measured in ulps of their magnitude times 1 + max(|e|, 2).

## Not promised

- Reproducibility. Floating-point last bits and seeded random draws can
  differ with batching, fusion, JIT, thread counts, array sizes,
  execution routes, platforms, builds and runs. A fold such as a sum can
  combine its items in any order.
- Infinities and NaNs. Results involving infinities or NaNs can differ
  by function, code path and platform. FMA and reassociation can
  overflow differently near the range limits. Given a NaN argument, a
  function can return NaN or another argument. The maximum or minimum of
  several values, whether two arguments or an array’s items, is NaN or
  the extreme of the values that aren’t NaN: the maximum of 5, NaN and 3
  can be 5 or NaN, never 3. Which NaN is returned, and its sign and
  payload, aren’t promised. Finite accuracy bounds still apply to
  subnormal results.
- The sign of zero. The sign of a zero result can differ by function,
  code path and platform.
