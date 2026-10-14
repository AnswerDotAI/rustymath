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

/// SplitMix64 from its state, for rustymath's own seeded draws: differential evolution's, and the truncated SVD's test matrix.
pub(crate) struct SplitMix64(pub(crate) u64);

impl rand::TryRng for SplitMix64 {
    type Error = std::convert::Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Self::Error> { self.try_next_u64().map(|z| (z >> 32) as u32) }
    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let z = (self.0 ^ (self.0 >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        let z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        Ok(z ^ (z >> 31))
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Self::Error> {
        rand::rand_core::utils::fill_bytes_via_next_word(dst, || self.try_next_u64())
    }
}
