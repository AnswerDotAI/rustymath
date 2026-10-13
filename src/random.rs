//! Random integers from a caller's generator: rolls, each uniform below its own bound, and deals, distinct integers below a total.
use crate::Arg;
use rand::{distr::{Distribution, Uniform}, Rng, RngExt};

/// Appends `n` integers to `out`, each uniform from 0 up to but not including its bound: the one bound, or each of the bounds in
/// turn. One bound for every item draws through one sampler, set up once. Each bound must be at least 1.
pub fn roll<R: Rng + ?Sized>(bounds: Arg<'_, u64>, n: usize, rng: &mut R, out: &mut Vec<u64>) {
    match bounds {
        Arg::One(b) => out.extend(Uniform::new(0, b).expect("a roll's bound must be at least 1").sample_iter(rng).take(n)),
        Arg::Many(bs) => out.extend(bs[..n].iter().map(|&b| rng.random_range(0..b))),
    }
}

/// Appends `n` distinct integers below `total` to `out`, in random order: every choice of them, in every order, is equally likely.
/// `n` must be at most `total`.
pub fn deal<R: Rng + ?Sized>(n: u64, total: u64, rng: &mut R, out: &mut Vec<u64>) {
    assert!(n <= total, "a deal can't take more items than its total");
    out.extend(rand::seq::index::sample(rng, total as usize, n as usize).into_iter().map(|i| i as u64));
}
