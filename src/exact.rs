//! Exact numbers: rationals of any integer type, such as `BigRational`, which is `Ratio<BigInt>`. These functions give exact results,
//! where the float functions round. [`linalg::exact`](crate::linalg::exact) solves matrices of them, [`polynomial`](crate::polynomial)
//! evaluates polynomials with them, and [`number_theory`](crate::number_theory) works on whole numbers.
use crate::{ok, BETWEEN_CHECKS};
use num_integer::Integer;
use num_rational::Ratio;
use num_traits::{One, Signed, ToPrimitive, Zero};

/// `y` modulo `x` with the sign of `x`, `y - x⌊y/x⌋`, which APL writes `x|y`. A zero `x` gives `y`.
pub fn residue<T: Clone + Integer>(x: &Ratio<T>, y: &Ratio<T>) -> Ratio<T> {
    if x.is_zero() { y.clone() } else { y - x * (y / x).floor() }
}

/// The greatest common divisor of `x` and `y`: the largest rational that divides each a whole number of times. It is the gcd of their
/// numerators over the lcm of their denominators. It is never negative, and the gcd of 0 and 0 is 0.
pub fn gcd<T: Clone + Integer>(x: &Ratio<T>, y: &Ratio<T>) -> Ratio<T> { Ratio::new(x.numer().gcd(y.numer()), x.denom().lcm(y.denom())) }

/// The least common multiple of `x` and `y`, `x y / gcd(x, y)`, with the sign of `x y`. It is 0 where either is 0.
pub fn lcm<T: Clone + Integer>(x: &Ratio<T>, y: &Ratio<T>) -> Ratio<T> {
    let g = gcd(x, y);
    if g.is_zero() { g } else { x / &g * y }
}

/// The binomial coefficient C(n, k) for a rational `n` and a whole `k`, with J's extension to negative `k`. For a whole `n` from 0, a
/// `k` above `n` gives 0. A negative `k` gives 0, unless `n` is a whole number with `k ≤ n < 0`, where the result is
/// (-1)^(n-k) C(-k-1, n-k). Otherwise the result is the product of (n-i)/(i+1) for each `i` from 0 below `k`.
/// It calls `check` once every 1,024 steps of that product.
pub fn binom_with<T, E>(n: &Ratio<T>, k: i64, mut check: impl FnMut() -> Result<(), E>) -> Result<Ratio<T>, E>
where
    T: Clone + Integer + Signed + ToPrimitive + From<i64>,
{
    let whole = n.is_integer().then(|| n.to_integer());
    if k < 0 {
        let Some(n) = whole else { return Ok(Ratio::zero()) };
        if !n.is_negative() || T::from(k) > n { return Ok(Ratio::zero()) }
        let count = (n - T::from(k)).to_i64().expect("n - k lies below -k");
        let c = binom_with(&Ratio::from_integer(T::from(-(k + 1))), count, check)?;
        return Ok(if count % 2 == 0 { c } else { -c });
    }
    let k = match whole {
        Some(n) if !n.is_negative() => {
            if T::from(k) > n { return Ok(Ratio::zero()) }
            let rest = n - T::from(k);
            if rest < T::from(k) { rest.to_i64().expect("n - k lies below k") } else { k }
        }
        _ => k,
    };
    let mut value = Ratio::one();
    for i in 0..k {
        if i % BETWEEN_CHECKS as i64 == 0 { check()?; }
        value = value * (n - Ratio::from_integer(T::from(i))) / Ratio::from_integer(T::from(i + 1));
    }
    Ok(value)
}

/// [`binom_with`] with no check.
pub fn binom<T: Clone + Integer + Signed + ToPrimitive + From<i64>>(n: &Ratio<T>, k: i64) -> Ratio<T> {
    let Ok(c) = binom_with(n, k, ok);
    c
}
