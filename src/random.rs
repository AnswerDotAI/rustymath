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

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};

    #[test]
    fn rolls_and_deals() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut out = vec![9];
        roll(Arg::One(6), 60_000, &mut rng, &mut out);
        assert_eq!(out[0], 9);
        let counts = (0..6).map(|k| out[1..].iter().filter(|&&r| r == k).count());
        assert!(counts.into_iter().all(|c| (9_000..11_000).contains(&c)));
        let bounds: Vec<u64> = (1..=1000).collect();
        let mut rolls = Vec::new();
        roll(Arg::Many(&bounds), 1000, &mut rng, &mut rolls);
        assert!(rolls.iter().zip(&bounds).all(|(r, b)| r < b));
        let mut dealt = Vec::new();
        deal(50, 60, &mut rng, &mut dealt);
        dealt.sort_unstable();
        dealt.dedup();
        assert!(dealt.len() == 50 && dealt.iter().all(|&d| d < 60));
    }
}
