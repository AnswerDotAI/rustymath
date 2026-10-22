//! The hash maps and sets rustymath uses. `Hashing` is foldhash's fast `RandomState`, hashbrown's default hasher, with `hash_one` marked `#[inline(always)]`. hashbrown 0.17 hashes every key through `BuildHasher::hash_one`, and core's default body for it has no inline hint, so incremental builds can call it out of line on every lookup.
use std::hash::{BuildHasher, Hash, Hasher};

#[derive(Clone, Default, Debug)]
pub struct Hashing(foldhash::fast::RandomState);

impl BuildHasher for Hashing {
    type Hasher = foldhash::fast::FoldHasher<'static>;
    #[inline(always)]
    fn build_hasher(&self) -> Self::Hasher { self.0.build_hasher() }
    #[inline(always)]
    fn hash_one<T: Hash>(&self, x: T) -> u64 { let mut h = self.build_hasher(); x.hash(&mut h); h.finish() }
}

pub type HashMap<K, V> = hashbrown::HashMap<K, V, Hashing>;
pub type HashSet<K> = hashbrown::HashSet<K, Hashing>;

/// `new` and `with_capacity` for `HashMap`, which hashbrown gives only for its default hasher.
pub trait HashMapExt { fn new() -> Self; fn with_capacity(capacity: usize) -> Self; }
impl<K, V> HashMapExt for HashMap<K, V> {
    fn new() -> Self { Self::with_hasher(Hashing::default()) }
    fn with_capacity(capacity: usize) -> Self { Self::with_capacity_and_hasher(capacity, Hashing::default()) }
}

/// `new` and `with_capacity` for `HashSet`, which hashbrown gives only for its default hasher.
pub trait HashSetExt { fn new() -> Self; fn with_capacity(capacity: usize) -> Self; }
impl<K> HashSetExt for HashSet<K> {
    fn new() -> Self { Self::with_hasher(Hashing::default()) }
    fn with_capacity(capacity: usize) -> Self { Self::with_capacity_and_hasher(capacity, Hashing::default()) }
}
