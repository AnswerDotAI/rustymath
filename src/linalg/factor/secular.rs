//! The roots of a secular equation. Divide and conquer solves one for each eigenvalue or singular value of a rank-one update. Each root lies between two poles, or above the largest pole for the last root. A root is held as one of its poles, `shift`, and its distance `mu` from that pole. The distance from the nearer pole then keeps its full relative accuracy, even for a root very close to the pole. The search brackets the root, narrows the bracket by halving, then refines it by a secant method on the function's rational shape, falling back to bisection. The search follows faer's root search in `tridiag_evd.rs`, with its `secular_eq` and `batch_secular_eq`. See `THIRD-PARTY.md`.
use super::*;

/// The root of the increasing function `f` between the poles `left` and `right`, as `(shift, mu)`: the root is `shift + mu`, where `shift` is `left` or `right`. `f(shift, mu)` evaluates the function at `shift + mu`, and `four` evaluates it at four such points at once. For the `last` root, `right` bounds the root from above and isn't a pole.
pub(super) fn root<R: Native>(f: impl Fn(R, R) -> R, four: impl Fn([R; 4], [R; 4]) -> [R; 4], left: R, right: R, last: bool) -> (R, R) {
    let (zero, one, two, eight, half, eps) = (R::zero(), R::one(), R::narrow(2.0), R::narrow(8.0), R::narrow(0.5), R::epsilon());
    let (width, mid) = (right - left, left + (right - left) * half);
    let [mut f_mid, f_max, f_left_mid, f_right_mid] =
        four([zero, left, left, right], [mid, if last { width } else { width * half }, width * half, -width * half]);
    let (mut shift, mu) = if last || f_mid > zero { (left, width * half) } else { (right, -width * half) };
    // Rounding can give the midpoint, measured from either pole, values of both signs. The root is then the midpoint.
    if f_left_mid <= zero && f_right_mid > zero { return (shift, mu); }
    if !last {
        if shift == left {
            if f_left_mid < zero {
                shift = right;
                f_mid = f_right_mid;
            }
        } else if f_right_mid > zero {
            shift = left;
            f_mid = f_left_mid;
        }
    }
    let from_left = shift == left;
    // The secant method on the model `f ≈ a/mu + b`, which holds near a pole. It records each side of the root it finds, and stops early, asking for bisection, when a step leaves the interval or makes `f` larger.
    let secant = |mut mu: R, mut mu_prev: R, mut f_mu: R, mut f_prev: R| {
        if f_prev.abs() < f_mu.abs() {
            std::mem::swap(&mut f_prev, &mut f_mu);
            std::mem::swap(&mut mu_prev, &mut mu);
        }
        let (mut below, mut above) = (None, None);
        if (f_prev > zero) != (f_mu > zero) {
            below = Some(mu.min(mu_prev));
            above = Some(mu.max(mu_prev));
        }
        let mut bisect = false;
        while f_mu != zero && (mu - mu_prev).abs() > eight * eps * mu.abs().max(mu_prev.abs()) && (f_mu - f_prev).abs() > eps && !bisect {
            let a = (f_mu - f_prev) * (mu_prev * mu) / (mu_prev - mu);
            let b = f_mu - a / mu;
            let next = -a / b;
            let f_next = f(shift, next);
            if f_next < zero { below = Some(next) }
            else { above = Some(next) }
            (mu_prev, f_prev, mu, f_mu) = (mu, f_mu, next, f_next);
            if (from_left && (mu < zero || mu > width)) || (!from_left && (mu > zero || mu < -width)) { bisect = true }
            if f_mu.abs() > f_prev.abs() {
                // The step overshot. Steps of the model's other branch look for the root's other side before bisection takes over.
                let mut k = one;
                for _ in 0..4 {
                    let opposite = -a / (k * f_next + b);
                    let f_opposite = f(shift, opposite);
                    if f_next < zero && f_opposite >= zero { above = Some(opposite); break; }
                    if f_next > zero && f_opposite <= zero { below = Some(opposite); break; }
                    k = k * two;
                }
                bisect = true;
            }
        }
        (bisect, mu, below, above)
    };
    // The bracket, as distances from `shift`, with `f` at each end.
    let (mut lo, mut f_lo, mut hi, mut f_hi) = if from_left {
        (zero, R::neg_infinity(), if last { width } else { width * half }, if last { f_max } else { f_mid })
    } else { (-width * half, f_mid, zero, R::infinity()) };
    let mut f_prev = f_mid;
    // Four points between the pole and the bracket's far end, at 1/256, 1/16, 1/4 and 1/2 of the way, narrow the bracket at once.
    let far = if from_left { hi } else { lo };
    let probes = [far * R::narrow(1.0 / 256.0), far * R::narrow(1.0 / 16.0), far * R::narrow(0.25), far * half];
    let values = four([shift; 4], probes);
    let mut i = 0;
    for (k, (&mu, &v)) in probes.iter().zip(&values).enumerate() {
        if from_left && v < zero { (lo, f_lo, i) = (mu, v, k + 1); } else if !from_left && v > zero { (hi, f_hi, i) = (mu, v, k + 1); }
    }
    if i < 4 { if from_left { (hi, f_hi) = (probes[i], values[i]) } else { (lo, f_lo) = (probes[i], values[i]) } }
    // Up to five halvings, geometric where both ends lie away from the pole.
    for _ in 0..5 {
        if hi - lo <= two * eps * lo.abs().max(hi.abs()) { break; }
        let mut m = lo.abs().sqrt() * hi.abs().sqrt();
        if lo < zero { m = -m }
        if m == zero { m = (lo + hi) * half }
        let f_m = f(shift, m);
        if f_m == zero { return (shift, m); }
        if f_m > zero { (hi, f_prev, f_hi) = (m, f_hi, f_m); } else { (lo, f_prev, f_lo) = (m, f_lo, f_m); }
    }
    let (start, next, f_start, f_next) = if lo == zero { (hi * two, hi, f_prev, f_hi) }
    else if hi == zero { (lo * two, lo, f_prev, f_lo) }
    else { (lo, hi, f_lo, f_hi) };
    let (bisect, mut mu, below, above) = secant(start, next, f_start, f_next);
    if let (Some(b), Some(a)) = (below, above) {
        if b < a {
            lo = lo.max(b);
            hi = hi.min(a);
        }
    }
    if bisect {
        while hi - lo > two * eps * lo.abs().max(hi.abs()) {
            let m = (lo + hi) * half;
            let f_m = f(shift, m);
            if f_m == zero { break; }
            if f_m > zero { hi = m } else { lo = m }
        }
        mu = (lo + hi) * half;
    }
    (shift, mu)
}
