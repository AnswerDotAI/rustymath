use super::*;

/// Appends the running folds by `R` along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`, in the same layout. Where `seed` is given, each lane's first fold combines `seed`'s item for its column, or its only item. Gives whether every fold fits. Lanes of consecutive items scan by [`Reduction::short_lanes`] where the reduction takes them, else by `scan`. Along any other axis, each row's folds step the row before's by `checked`.
pub fn scan_along<T: Copy, R: Reduction<T>>(data: &[T], dims: [usize; 3], seed: Option<&[R::Total]>, out: &mut Vec<R::Total>) -> bool {
    let start = |i| seed.map(|s| s[if s.len() == 1 { 0 } else { i }]);
    if let [_, len, 1] = dims && let Some(fits) = R::short_lanes(data, len, start(0), out) { return fits; }
    rows_scan(data, dims, out, |lane, o| R::scan(lane, start(0), o), |i, x| start(i).map_or_else(|| R::one(x), |s| R::checked(s, x)), R::checked)
}

/// Appends the ordered left scan of each lane of `[outer, len, inner]` in `data` to `out`, in the same layout: along the last axis by `lane`, one call for every lane, stepping from `seed`'s item; along any other axis by `block`, one call for each row, stepping the row before it, or `seed`'s row, by the row's items. Without a seed, a lane's first item or row is kept as it is. Gives the kind of the first step that failed.
pub(crate) fn ordered_scan<T: Copy>(
    data: &[T], [_, len, inner]: [usize; 3], seed: Option<&[T]>, out: &mut Vec<T>, lane: &dyn Fn(Option<T>, &[T], usize, &mut Vec<T>) -> Option<Kind>,
    block: &dyn Fn(&[T], &[T], &mut [std::mem::MaybeUninit<T>]) -> Result<(), Kind>) -> Option<Kind> {
    if inner == 1 { return lane(seed.map(|s| s[0]), data, len, out); }
    appended(out, data.len(), |slots| {
        let mut failure = None;
        for (items, outs) in data.chunks(len * inner).zip(slots.chunks_mut(len * inner)) {
            match seed {
                Some(s) => failure = failure.or(block(s, &items[..inner], &mut outs[..inner]).err()),
                None => for (slot, &x) in outs.iter_mut().zip(&items[..inner]) { slot.write(x); },
            }
            for r in 1..len {
                let (done, next) = outs.split_at_mut(r * inner);
                failure = failure.or(block(written(&mut done[(r - 1) * inner..]), &items[r * inner..(r + 1) * inner], &mut next[..inner]).err());
            }
        }
        failure
    })
}

/// Appends the running results along each lane of `len` consecutive items in `data` to `out`, and gives whether every one fits: `first` gives each lane's first result, and `step` each later one from the one before. Lanes shorter than [`SHORT_LANE`] run [`LANES`] side by side (`rows_scan`).
pub(crate) fn lanes_scan<T: Copy>(data: &[T], len: usize, out: &mut Vec<T>, first: impl Fn(T) -> (T, bool) + Copy, step: impl Fn(T, T) -> (T, bool) + Copy) -> bool {
    rows_scan(data, [0, len, 1], out, |lane, out| {
        let (total, ok) = first(lane[0]);
        out.push(total);
        checked_scan(&lane[1..], total, out, step) & ok
    }, move |_, x| first(x), step)
}

/// Appends the running results along the middle axis of `data`, an array of shape `[outer, len, inner]` held row by row, to `out`, in the same layout, and gives whether every one fits. A lane of at least [`SHORT_LANE`] consecutive items runs through `lane`. Shorter lanes run [`LANES`] side by side, and lanes along any other axis run a row at a time: `first` gives each lane's first item or row, and `step` steps each from the one before. `first` and `step` go to `side_by_side` by value, because a build of many codegen units can leave core's `<&F as Fn>::call` for a reference out of line, at a call for each item.
#[inline(always)]
fn rows_scan<T: Copy, S: Copy>(
    data: &[T], [_, len, inner]: [usize; 3], out: &mut Vec<S>, lane: impl Fn(&[T], &mut Vec<S>) -> bool, first: impl Fn(usize, T) -> (S, bool) + Copy,
    step: impl Fn(S, T) -> (S, bool) + Copy) -> bool {
    if data.is_empty() { return true; }
    if inner == 1 && len >= SHORT_LANE { return data.chunks(len).fold(true, |fits, l| lane(l, out) & fits); }
    if inner == 1 {
        return appended(out, data.len(), |slots| {
            let whole = data.len() / (LANES * len) * (LANES * len);
            let (groups, rest) = slots.split_at_mut(whole);
            // SAFETY: `side_by_side` gives each position of the data it runs on, which holds as many items as the slots written here.
            side_by_side::<LANES, _, _>(&data[..whole], len, |x| first(0, x), step, |at, s| {
                unsafe { groups.get_unchecked_mut(at) }.write(s);
            }, |_, _| {}) & side_by_side::<1, _, _>(&data[whole..], len, |x| first(0, x), step, |at, s| {
                unsafe { rest.get_unchecked_mut(at) }.write(s);
            }, |_, _| {})
        });
    }
    // Each row steps from the row before it, which `out` already holds.
    appended(out, data.len(), |slots| {
        let mut fits = true;
        let mut at = 0;
        while at < data.len() {
            let (row, outs) = (&data[at..at + inner], &mut slots[at..at + inner]);
            for k in 0..inner {
                let (t, f) = first(k, row[k]);
                outs[k].write(t);
                fits &= f;
            }
            let mut r = at + inner;
            while r < at + len * inner {
                let (done, rest) = slots.split_at_mut(r);
                let (before, row, outs) = (written(&mut done[r - inner..]), &data[r..r + inner], &mut rest[..inner]);
                for k in 0..inner {
                    let (t, f) = step(before[k], row[k]);
                    outs[k].write(t);
                    fits &= f;
                }
                r += inner;
            }
            at += len * inner;
        }
        fits
    })
}

/// A function of two Booleans, held as its truth table: `Logical(t)` gives `t[r][x]` for the running result `r` and the next item `x`. Its scans run any such function.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Logical(pub [[bool; 2]; 2]);

impl Logical {
    /// `r f x`, with no branch.
    #[inline(always)]
    fn apply(self, r: bool, x: bool) -> bool { let [[f00, f01], [f10, f11]] = self.0; (r & ((x & f11) | (!x & f10))) | (!r & ((x & f01) | (!x & f00))) }

    /// Appends the running results of `x` to `out`: the first item, or `start` combined with it where `start` is given, then each result combined with the next item.
    ///
    /// For each value of the next item, the function maps the running result to itself, to its negation, or to a constant. Where one value maps it to itself and the other to a constant, as for `and` and `or`, the results change at most once. `first_where` finds that place, and two runs fill the rest. Otherwise each map is `r ↦ (s ∧ r) ≠ t` for two bits `s` and `t`, and the maps compose as `r ↦ (s₂ ∧ s₁ ∧ r) ≠ ((s₂ ∧ t₁) ≠ t₂)`. Each block of 64 items packs into a word of bits, whose maps compose in six steps of shifts, with no branch and no step for each item.
    pub fn scan(self, x: &[bool], start: Option<bool>, out: &mut Vec<bool>) {
        let Some((&head, rest)) = x.split_first() else { return };
        let first = self.first(start, head);
        out.push(first);
        // `maps[v]` gives the results of an item `v` for a running result of false and of true.
        let [[f00, f01], [f10, f11]] = self.0;
        let maps = [[f00, f10], [f01, f11]];
        for v in [false, true] {
            let [lo, hi] = maps[usize::from(!v)];
            if maps[usize::from(v)] == [false, true] && lo == hi {
                let at = first_where(rest, |x| x != v);
                out.extend(std::iter::repeat_n(first, at));
                out.extend(std::iter::repeat_n(lo, rest.len() - at));
                return;
            }
        }
        let bit_maps = self.bit_maps();
        mapped(rest.len(), first, |start| bit_maps(packed(rest, start)), out);
    }

    /// A lane's first result: its first item `x`, combined with `start` where it's given.
    #[inline(always)]
    fn first(self, start: Option<bool>, x: bool) -> bool { start.map_or(x, |s| self.apply(s, x)) }

    /// The maps of 64 items given as the bits of a word, as [`mapped`] takes them: item `k` maps `r` to `(s ∧ r) ≠ t` for bit `k` of each word of `(s, t)`.
    #[inline(always)]
    fn bit_maps(self) -> impl Fn(u64) -> (u64, u64) + Copy {
        let [[f00, f01], [f10, f11]] = self.0;
        let word = |b: bool| if b { u64::MAX } else { 0 };
        // An item `v` maps a result of false to `f0v` and a result of true to `f1v`.
        let (s0, t0, s1, t1) = (word(f00 != f10), word(f00), word(f01 != f11), word(f01));
        move |bits| ((bits & s1) | (!bits & s0), (bits & t1) | (!bits & t0))
    }

    /// Appends the running results of `data`'s lanes of `len` consecutive items, 1 to 64, each lane from `start`, to `out` in one [`mapped`] call. The Boolean reductions' [`Reduction::short_lanes`] runs through it. Each word restarts the lanes that start in it with constant maps, at places that shift by `64 % len` from one word to the next. A lane's first result depends only on its first item, so it comes from the same bits.
    pub(super) fn short_lanes(self, data: &[bool], len: usize, start: Option<bool>, out: &mut Vec<bool>) {
        let (bit_maps, word) = (self.bit_maps(), |b: bool| if b { u64::MAX } else { 0 });
        let (yes, no) = (word(self.first(start, true)), word(self.first(start, false)));
        let starts = (0..64).step_by(len).fold(0u64, |p, k| p | 1 << k);
        let (shift, at) = (64 % len, std::cell::Cell::new(0));
        mapped(data.len(), false, |from| {
            let (bits, k) = (packed(data, from), at.get());
            at.set(if k >= shift { k - shift } else { k + len - shift });
            let (places, (s, t)) = (starts << k, bit_maps(bits));
            (s & !places, (t & !places) | ((bits & yes) | (!bits & no)) & places)
        }, out);
    }
}

/// The 64 Booleans of `block` as the bits of a word, the first in the lowest bit. Each 8 Booleans, one byte each, gather into a byte through one multiplication.
#[inline(always)]
fn pack(block: &[bool; 64]) -> u64 {
    let mut bits = 0;
    for (k, b) in block.as_chunks::<8>().0.iter().enumerate() {
        let bytes = (0..8).fold(0u64, |w, i| w | u64::from(b[i]) << (8 * i));
        bits |= (bytes.wrapping_mul(0x0102_0408_1020_4080) >> 56) << (8 * k);
    }
    bits
}

/// Writes the 64 bits of `r` to `out` as Booleans, the lowest bit first. Each byte of `r` becomes 8 bytes of 0 or 1, bit `j` in byte `j`, through one multiplication, two masks, an add and a shift that's the same for every byte. LLVM compiles a test of bit `j` in byte `j` to a different shift for each byte, which wasm's SIMD can only emulate.
#[inline(always)]
fn unpack(r: u64, out: &mut [std::mem::MaybeUninit<bool>; 64]) {
    for (k, lanes) in out.as_chunks_mut::<8>().0.iter_mut().enumerate() {
        let bits = (r >> (8 * k) & 0xff).wrapping_mul(0x0101_0101_0101_0101) & 0x8040_2010_0804_0201;
        let bytes = ((bits + 0x7f7f_7f7f_7f7f_7f7f) >> 7 & 0x0101_0101_0101_0101).to_le_bytes();
        *lanes = std::array::from_fn(|j| std::mem::MaybeUninit::new(bytes[j] != 0));
    }
}

/// The results of 64 items, bit by bit, from the running result `carry`, where item `k`'s map is `r ↦ (s ∧ r) ≠ t` for bit `k` of `s` and `t`. Each step composes each bit's map with the map it holds from the bits before it. A bit with none before it takes the identity map, whose `s` is 1 and whose `t` is 0.
#[inline(always)]
fn composed(mut s: u64, mut t: u64, carry: bool) -> u64 {
    for k in [1, 2, 4, 8, 16, 32] {
        t ^= s & (t << k);
        s &= (s << k) | ((1u64 << k) - 1);
    }
    (s & if carry { u64::MAX } else { 0 }) ^ t
}

/// The Booleans of `x` from `start`, at most 64, as the bits of a word, the first in the lowest bit. Bits past the end of `x` are 0.
#[inline(always)]
fn packed(x: &[bool], start: usize) -> u64 {
    match x[start..].first_chunk::<64>() {
        Some(block) => pack(block),
        None => {
            let mut block = [false; 64];
            block[..x.len() - start].copy_from_slice(&x[start..]);
            pack(&block)
        }
    }
}

/// Appends the running results of `n` items to `out`, from the running result `carry` before the first. `maps(start)` gives the maps of the 64 items from `start` as two words of bits `(s, t)`: item `start + k` maps `r` to `(s ∧ r) ≠ t` for bit `k` of each. Bits past the last item are ignored.
#[inline(always)]
fn mapped(n: usize, mut carry: bool, maps: impl Fn(usize) -> (u64, u64), out: &mut Vec<bool>) {
    appended(out, n, |slots| {
        let (outs, last) = slots.as_chunks_mut::<64>();
        for (i, o) in outs.iter_mut().enumerate() {
            let (s, t) = maps(64 * i);
            let r = composed(s, t, carry);
            unpack(r, o);
            carry = r >> 63 != 0;
        }
        if !last.is_empty() {
            let (s, t) = maps(n - last.len());
            let r = composed(s, t, carry);
            for (k, slot) in last.iter_mut().enumerate() { slot.write(r >> k & 1 != 0); }
        }
    });
}
