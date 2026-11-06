//! What a kernel computes over: a call's dtype and overflow policy (`Variant`), and the values that an argument or a result can take (`Range`). Formulas (`facts`), recipes and kernels read both.
use crate::buffer::{DType, FloatWidth};

/// The values other than NaN that the items of an argument or a result can take: each lies from `lo` to `hi`. ¯0 and +0 count as one value, because the sign of a zero isn't promised and can differ by route. Any range can also hold NaN. A rule's range is sound where it holds every result that the object's kernels give, within their accuracy.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Range { pub lo: f64, pub hi: f64 }

impl Range {
    /// Every value.
    pub const ANY: Self = Self { lo: f64::NEG_INFINITY, hi: f64::INFINITY };
    /// Every value from 0 up.
    pub const NONNEGATIVE: Self = Self { lo: 0.0, hi: f64::INFINITY };
    /// No value, as an array of no items holds. It's `join`'s identity: its `lo` is ∞ and its `hi` ¯∞.
    pub const EMPTY: Self = Self { lo: f64::INFINITY, hi: f64::NEG_INFINITY };
    /// The range of an argument that is the one number `c`, or `ANY` where `c` is NaN or `None`, as for an argument that isn't one number.
    #[inline]
    pub fn of(c: Option<f64>) -> Self { c.filter(|c| !c.is_nan()).map_or(Self::ANY, |c| Self { lo: c, hi: c }) }
    /// The values that items of `dtype` hold: 0 and 1 for Booleans, from each integer width's least integer to its greatest, and any value for other dtypes.
    #[inline]
    pub fn of_dtype(dtype: DType) -> Self {
        match dtype {
            DType::Boolean => Self { lo: 0.0, hi: 1.0 }, DType::Integer(width) => with_width!(width, T => Self { lo: T::MIN as f64, hi: T::MAX as f64 }),
            _ => Self::ANY
        }
    }
    /// The one number that the range holds, if it holds only one.
    #[inline]
    pub fn constant(self) -> Option<f64> { (self.lo == self.hi).then_some(self.lo) }
    /// Whether the range holds no negative number.
    #[inline]
    pub fn nonnegative(self) -> bool { self.lo >= 0.0 }
    /// Whether the range holds no value (`EMPTY`).
    #[inline]
    pub fn is_empty(self) -> bool { self.lo > self.hi }
    /// Whether every value lies from `lo` to `hi`.
    #[inline]
    pub fn within(self, lo: f64, hi: f64) -> bool { lo <= self.lo && self.hi <= hi }
    /// Whether no value equals `c`.
    #[inline]
    pub fn excludes(self, c: f64) -> bool { c < self.lo || self.hi < c }
    /// The values that both ranges hold.
    #[inline]
    pub fn meet(self, other: Self) -> Self { Self { lo: self.lo.max(other.lo), hi: self.hi.min(other.hi) } }
    /// The least range that holds both ranges.
    #[inline]
    pub fn join(self, other: Self) -> Self { Self { lo: self.lo.min(other.lo), hi: self.hi.max(other.hi) } }
    /// The range of |x| for the values x of this range.
    pub fn magnitude(self) -> Self {
        if self.lo >= 0.0 { self } else if self.hi <= 0.0 { Self { lo: -self.hi, hi: -self.lo } } else { Self { lo: 0.0, hi: self.hi.max(-self.lo) } }
    }
    /// The range of x × x for the values x of this range, as `⊼x` squares them.
    pub fn square(self) -> Self { let m = self.magnitude(); Self { lo: m.lo * m.lo, hi: m.hi * m.hi } }
    /// The range of a fold's results, from its items' range, its seed's, if it has one, and `step`, which gives the range of a step's result from the ranges of its two arguments. Every lane holds an item. One bound holds for any fold: the least range that holds the items and the seed, where a step's results stay in it. Where they don't, as a sum's don't, each bound widens to an infinity, except that a bound on the far side of 0 stops at 0: [2, 3] widens to [0, ∞]. Where a step's results leave that range too, that bound is `ANY`.
    ///
    /// Where `lane` gives the most items that a lane holds and `step` is an associative function's, as `+ × ⌈ ⌊ ∧ ∨` are, a second bound holds, and the range is the meet of both. It starts at the items' range, and joins in `step` of the range on itself while the lane length that it covers doubles, up to `lane`: [2, 3] over lanes of up to 4 items gives [2, 12]. It holds because a lane of up to 2m items splits into two lanes of up to m items whose results, by associativity, one step combines, and `step` bounds that step for any arguments within its two ranges. A scan's running results are the results of shorter lanes. A seed then combines with that bound in one step, as associativity lets it combine with a lane's result, and its own range joins in for a lane of no items.
    pub fn folded(items: Self, seed: Option<Self>, lane: Option<usize>, step: impl Fn([Self; 2]) -> Self) -> Self {
        let start = seed.map_or(items, |s| s.join(items));
        let (lo, hi) = (if start.lo >= 0.0 { 0.0 } else { f64::NEG_INFINITY }, if start.hi <= 0.0 { 0.0 } else { f64::INFINITY });
        let kept = [start, Self { lo, hi }].into_iter().find(|&r| step([r, r]).inside(r)).unwrap_or(Self::ANY);
        let Some(lane) = lane else { return kept };
        let (mut covered, mut m) = (items, 1);
        while m < lane {
            covered = covered.join(step([covered, covered]));
            m = m.saturating_mul(2);
        }
        seed.map_or(covered, |s| s.join(step([s, covered]))).meet(kept)
    }
    /// Whether `other` holds every value of this range.
    fn inside(self, other: Self) -> bool { other.lo <= self.lo && self.hi <= other.hi }
    /// This range with each bound moved `ULPS` units in the last place outward, for bounds that a kernel computes within its accuracy, as the transcendental functions do. A bound never moves past 0, and 0 or an infinite bound stays.
    pub(crate) fn loose(self) -> Self { Self { lo: outward(self.lo, false, FloatWidth::F64), hi: outward(self.hi, true, FloatWidth::F64) } }
    /// This range as a kernel's results of `dtype` hold it. The bounds are f64, and a kernel at a narrower float rounds each result to that width, within its accuracy there: each bound moves `ULPS` units in that width's last place outward, as `loose` moves it at f64. A kernel at an integer dtype gives whole results, flooring a quotient as integer `÷` does: each bound rounds outward to a whole number. Other dtypes keep the range.
    pub fn at_width(self, dtype: DType) -> Self {
        match dtype {
            DType::Float(width @ (FloatWidth::F32 | FloatWidth::F16)) => Self { lo: outward(self.lo, false, width), hi: outward(self.hi, true, width) },
            DType::Integer(_) => Self { lo: self.lo.floor(), hi: self.hi.ceil() }, _ => self
        }
    }
}

/// How far a computed bound can lie from a kernel's result on the same argument, in units in the last place: a kernel lies within 4 of the correctly rounded result, and so does the function that computes the bound.
const ULPS: u64 = 8;

/// `x` moved `ULPS` units in the last place of floats of `width` up, or down, but not past 0. A zero or an infinity stays. A bound between two floats of `width` starts from the nearer, which the move covers.
fn outward(x: f64, up: bool, width: FloatWidth) -> f64 {
    if x == 0.0 || x.is_infinite() { return x; }
    let (bits, infinity): (u64, u64) = match width {
        FloatWidth::F64 => (x.abs().to_bits(), f64::INFINITY.to_bits()), FloatWidth::F32 => ((x.abs() as f32).to_bits().into(), f32::INFINITY.to_bits().into()),
        FloatWidth::F16 => (half::f16::from_f64(x.abs()).to_bits().into(), half::f16::INFINITY.to_bits().into())
    };
    let moved = if up == x.is_sign_positive() { (bits + ULPS).min(infinity) } else { bits.saturating_sub(ULPS) };
    let moved = match width {
        FloatWidth::F64 => f64::from_bits(moved), FloatWidth::F32 => f32::from_bits(moved as u32).into(),
        FloatWidth::F16 => half::f16::from_bits(moved as u16).to_f64()
    };
    moved.copysign(x)
}

/// Policies captured when the call is built; never read from force-time preferences.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Variant { pub dtype: DType, pub checked: bool, pub wrapping: bool }

impl Variant {
    #[inline]
    pub const fn checked(dtype: DType) -> Self { Self { dtype, checked: true, wrapping: false } }
    /// Whether an integer result that doesn't fit its type fails, as checked arithmetic that doesn't wrap gives it, rather than wrapping or filling. It covers a computed result and a result narrowed to a smaller output type.
    #[inline]
    pub const fn fails_on_overflow(self) -> bool { self.checked && !self.wrapping }
}
