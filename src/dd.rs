//! Double-word arithmetic: a value held as an unevaluated sum `hi + lo`, with `lo` below half an ulp of `hi`. It carries about twice
//! a float's precision through `pow`'s logarithm, whose error the exponent multiplies. The operations and their forms with and
//! without FMA come from SLEEF's `dd.h`. See `THIRD-PARTY.md`.
use crate::float::Native;

#[derive(Clone, Copy)]
pub(crate) struct Double<R> {
    pub(crate) hi: R,
    pub(crate) lo: R,
}

impl<R: Native> Double<R> {
    #[inline]
    pub(crate) fn new(hi: R, lo: R) -> Self { Self { hi, lo } }
    /// a + b exactly, whatever their magnitudes.
    #[inline]
    pub(crate) fn sum(a: R, b: R) -> Self {
        let s = a + b;
        let v = s - a;
        Self::new(s, (a - (s - v)) + (b - v))
    }
    /// The same value with `lo` below half an ulp of `hi` again, when |lo| is below |hi|.
    #[inline]
    pub(crate) fn normalize(self) -> Self {
        let s = self.hi + self.lo;
        Self::new(s, (self.hi - s) + self.lo)
    }
    /// self + y, when |self| is at least |y|.
    #[inline]
    pub(crate) fn add(self, y: Self) -> Self {
        let s = self.hi + y.hi;
        Self::new(s, (self.hi - s) + y.hi + self.lo + y.lo)
    }
    /// self × y.
    #[inline]
    pub(crate) fn mul_float<const FUSED: bool>(self, y: R) -> Self {
        let s = self.hi * y;
        if FUSED { return Self::new(s, self.lo.mla::<true>(y, self.hi.mla::<true>(y, -s))); }
        let (xh, yh) = (self.hi.upper(), y.upper());
        let (xl, yl) = (self.hi - xh, y - yh);
        Self::new(s, xh * yh - s + xl * yh + xh * yl + xl * yl + self.lo * y)
    }
    /// self ÷ d.
    #[inline]
    pub(crate) fn div<const FUSED: bool>(self, d: Self) -> Self {
        let t = R::one() / d.hi;
        let s = self.hi * t;
        if FUSED {
            let u = t.mla::<true>(self.hi, -s);
            let v = (-d.lo).mla::<true>(t, (-d.hi).mla::<true>(t, R::one()));
            return Self::new(s, s.mla::<true>(v, self.lo.mla::<true>(t, u)));
        }
        let (dh, th, nh) = (d.hi.upper(), t.upper(), self.hi.upper());
        let (dl, tl, nl) = (d.hi - dh, t - th, self.hi - nh);
        let u = nh * th - s + nh * tl + nl * th + nl * tl + s * (R::one() - dh * th - dh * tl - dl * th - dl * tl);
        Self::new(s, t.mla::<false>(self.lo - s * d.lo, u))
    }
}
