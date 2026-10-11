//! Slice forms of rustymath's functions. Each function object, such as [`Pow`], has a scalar form `one` and a slice form `many`.
//! Both run code that rustymath compiles once for each float type, inside its own crate. A caller's build settings can't change that
//! code, and rustymath's speed test times what callers run. An edit to a function's body rebuilds only rustymath.
use crate::float::{
    sealed::{BinaryForms, Dispatch},
    Float,
};
use half::f16;

/// How many items a caller handles at a time where it converts or expands arguments into buffers for a slice form. Buffers of this
/// many items stay in cache.
pub const CHUNK: usize = 4096;

/// One argument of a slice function: one item that every result pairs with, or one item for each result.
#[derive(Clone, Copy, Debug)]
pub enum Arg<'a, T> {
    One(T),
    Many(&'a [T]),
}

impl<T: Copy> Arg<'_, T> {
    /// Items `start..start + n` of this argument, each converted by `f`. A `Many` argument converts into `buf`.
    #[inline]
    pub fn converted<'b, A>(self, start: usize, n: usize, buf: &'b mut Vec<A>, f: impl Fn(T) -> A) -> Arg<'b, A> {
        match self {
            Arg::One(v) => Arg::One(f(v)),
            Arg::Many(s) => {
                buf.clear();
                pairs(Arg::Many(&s[start..start + n]), Arg::One(()), n, buf, |v, ()| (f(v), true));
                Arg::Many(buf)
            }
        }
    }
}

/// A function of two floats as a value, with a scalar form and a slice form. rustymath's function objects, such as [`Pow`],
/// implement it, and nothing else can.
pub trait Binary: BinaryForms {
    /// The function of one pair of items.
    #[inline]
    fn one<F: Float>(self, x: F, y: F) -> F { F::binary_one::<Self>(x, y) }
    /// Appends `one` for each of `n` results to `out`. A `Many` argument has at least `n` items. The result is `false` where some
    /// appended item is NaN and neither of its arguments is. rustymath gives NaN for an argument outside a function's real domain.
    #[inline]
    fn many<F: Float>(self, x: Arg<'_, F>, y: Arg<'_, F>, n: usize, out: &mut Vec<F>) -> bool { F::binary_many::<Self>(x, y, n, out) }
    /// `one`, or `None` where its result is NaN and neither argument is, as `many` reports.
    #[inline]
    fn checked<F: Float>(self, x: F, y: F) -> Option<F> {
        let r = self.one(x, y);
        kept(r, x, y).then_some(r)
    }
}
impl<K: BinaryForms> Binary for K {}

/// Whether `r`, the result for `x` and `y`, keeps the rule that a NaN result needs a NaN argument.
#[inline(always)]
fn kept<F: Float>(r: F, x: F, y: F) -> bool { !r.is_nan() | x.is_nan() | y.is_nan() }

/// Runs `$body` for each item `$i` of a result of `$n` items, with `$a` and `$b` the items of `$x` and `$y` that pair with it. Each
/// pairing of `One` and `Many` has a loop of its own. The loops index slices of the result's length with no iterator adapters, and
/// vectorize at any build settings.
macro_rules! each_pair {
    ($x:expr, $y:expr, $n:expr, |$i:ident, $a:ident, $b:ident| $body:expr) => {{
        let n = $n;
        match ($x, $y) {
            (Arg::One($a), Arg::One($b)) => for $i in 0..n { $body },
            (Arg::One($a), Arg::Many(ys)) => {
                let ys = &ys[..n];
                for $i in 0..n { let $b = ys[$i]; $body }
            }
            (Arg::Many(xs), Arg::One($b)) => {
                let xs = &xs[..n];
                for $i in 0..n { let $a = xs[$i]; $body }
            }
            (Arg::Many(xs), Arg::Many(ys)) => {
                let (xs, ys) = (&xs[..n], &ys[..n]);
                for $i in 0..n { let ($a, $b) = (xs[$i], ys[$i]); $body }
            }
        }
    }};
}

/// Appends `f` on each of `n` pairs of items to `out`, and gives whether `f` said `true` for every pair. An `Arg::One` pairs with
/// every item of the other argument. The loops write straight into `out`'s spare capacity, which is never zeroed first. This is the
/// loop that rustymath's slice forms run, and a caller's own kernels can run it too, as basedpl's arithmetic does.
#[inline]
pub fn pairs<X: Copy, Y: Copy, B>(x: Arg<'_, X>, y: Arg<'_, Y>, n: usize, out: &mut Vec<B>, f: impl Fn(X, Y) -> (B, bool)) -> bool {
    out.reserve(n);
    let (slots, mut ok) = (&mut out.spare_capacity_mut()[..n], true);
    each_pair!(x, y, n, |i, a, b| {
        let (r, good) = f(a, b);
        slots[i].write(r);
        ok &= good;
    });
    // SAFETY: each pairing's loop writes all `n` slots.
    unsafe { out.set_len(out.len() + n) }
    ok
}

/// Appends `f` for each of `n` pairs, for a function whose special cases are rare. `fast` computes every item, and gives `f`'s
/// result wherever `plain` holds. A second pass recomputes with `f` the items where `plain` fails, if there are any. No item where
/// `plain` holds can break the rule that a NaN result needs a NaN argument.
#[inline(always)]
pub(crate) fn two_pass<F: Float>(
    x: Arg<'_, F>, y: Arg<'_, F>, n: usize, out: &mut Vec<F>, f: impl Fn(F, F) -> F, fast: impl Fn(F, F) -> F,
    plain: impl Fn(F, F) -> bool,
) -> bool {
    let start = out.len();
    if pairs(x, y, n, out, |a, b| (fast(a, b), plain(a, b))) { return true; }
    let (out, mut ok) = (&mut out[start..], true);
    each_pair!(x, y, n, |i, a, b| if !plain(a, b) {
        let r = f(a, b);
        ok &= kept(r, a, b);
        out[i] = r;
    });
    ok
}

/// `f`, a slice form at `F`'s working type, for narrower floats: each `CHUNK` items convert to the working type, run, and round once.
fn working<F: Float>(
    x: Arg<'_, F>, y: Arg<'_, F>, n: usize, out: &mut Vec<F>,
    f: fn(Arg<'_, F::Working>, Arg<'_, F::Working>, usize, &mut Vec<F::Working>) -> bool,
) -> bool {
    let (mut xs, mut ys, mut rs) = (Vec::new(), Vec::new(), Vec::new());
    let mut ok = true;
    for start in (0..n).step_by(CHUNK) {
        let m = CHUNK.min(n - start);
        rs.clear();
        ok &= f(x.converted(start, m, &mut xs, F::working), y.converted(start, m, &mut ys, F::working), m, &mut rs);
        pairs(Arg::Many(&rs), Arg::One(()), m, out, |r, ()| (F::from_working(r), true));
    }
    ok
}

impl Dispatch for f32 {
    #[inline]
    fn binary_one<K: BinaryForms>(x: f32, y: f32) -> f32 { K::one_f32(x, y) }
    #[inline]
    fn binary_many<K: BinaryForms>(x: Arg<'_, f32>, y: Arg<'_, f32>, n: usize, out: &mut Vec<f32>) -> bool { K::many_f32(x, y, n, out) }
}

impl Dispatch for f64 {
    #[inline]
    fn binary_one<K: BinaryForms>(x: f64, y: f64) -> f64 { K::one_f64(x, y) }
    #[inline]
    fn binary_many<K: BinaryForms>(x: Arg<'_, f64>, y: Arg<'_, f64>, n: usize, out: &mut Vec<f64>) -> bool { K::many_f64(x, y, n, out) }
}

impl Dispatch for f16 {
    #[inline]
    fn binary_one<K: BinaryForms>(x: f16, y: f16) -> f16 { f16::from_working(K::one_f32(x.working(), y.working())) }
    #[inline]
    fn binary_many<K: BinaryForms>(x: Arg<'_, f16>, y: Arg<'_, f16>, n: usize, out: &mut Vec<f16>) -> bool { working(x, y, n, out, K::many_f32) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Pow;

    /// `Pow.many` against `pow` for each item, for every pairing of `One` and `Many`, with every special value among the items. It must
    /// append after what `out` holds, and say whether any item broke the rule that a NaN result needs a NaN argument.
    fn pow_many<F: Float + std::fmt::Debug>() {
        let items = [0.0, -0.0, 1.0, -1.0, 0.5, -0.5, 2.0, -2.0, 3.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN].map(F::narrow);
        let same = |a: F, b: F| (a == b && a.is_sign_negative() == b.is_sign_negative()) || (a.is_nan() && b.is_nan());
        let check = |x: Arg<'_, F>, y: Arg<'_, F>, n: usize| {
            let at = |a: Arg<'_, F>, i| match a { Arg::One(v) => v, Arg::Many(s) => s[i] };
            let mut out = vec![F::one()];
            let ok = Pow.many(x, y, n, &mut out);
            assert_eq!((out.len(), out[0]), (n + 1, F::one()));
            for i in 0..n { assert!(same(out[i + 1], crate::pow(at(x, i), at(y, i))), "{:?}^{:?}", at(x, i), at(y, i)) }
            assert_eq!(ok, (0..n).all(|i| Pow.checked(at(x, i), at(y, i)).is_some()));
        };
        let (xs, ys): (Vec<F>, Vec<F>) = items.iter().flat_map(|&x| items.map(|y| (x, y))).unzip();
        check(Arg::Many(&xs), Arg::Many(&ys), xs.len());
        for v in items {
            check(Arg::One(v), Arg::Many(&items), items.len());
            check(Arg::Many(&items), Arg::One(v), items.len());
            for w in items { check(Arg::One(v), Arg::One(w), 2) }
        }
    }

    #[test]
    fn pow_many_matches_pow() {
        pow_many::<f64>();
        pow_many::<f32>();
        pow_many::<f16>();
    }
}
