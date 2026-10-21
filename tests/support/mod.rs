use rand::TryRng;
use std::convert::Infallible;

/// SplitMix64's output at index `i`, shared by the tests' sampled inputs.
pub fn random(i: u64) -> u64 {
    let z = i.wrapping_add(1).wrapping_mul(0x9e3779b97f4a7c15);
    let z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    let z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
    z ^ (z >> 31)
}

/// A generator of successive `random(i)` outputs, starting at `seed`.
pub fn seeded(seed: u64) -> impl rand::Rng { Seeded(seed) }

struct Seeded(u64);
impl TryRng for Seeded {
    type Error = Infallible;
    fn try_next_u32(&mut self) -> Result<u32, Infallible> { self.try_next_u64().map(|x| x as u32) }
    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        let value = random(self.0);
        self.0 = self.0.wrapping_add(1);
        Ok(value)
    }
    fn try_fill_bytes(&mut self, dst: &mut [u8]) -> Result<(), Infallible> { rand::rand_core::utils::fill_bytes_via_next_word(dst, || self.try_next_u64()) }
}

/// Held by each test that sets rustymath's thread count, which holds for the whole process, so that tests running at the same time leave it as each set it.
pub fn threads_lock() -> std::sync::MutexGuard<'static, ()> {
    static LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());
    LOCK.lock().unwrap_or_else(std::sync::PoisonError::into_inner)
}
