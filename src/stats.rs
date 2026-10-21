//! Statistics through the shared reductions and matrix products. Means and norms use the ordinary parallel folds. Variance makes two passes, without an array-sized buffer of deviations. No compensated sums or reproducibility rules are imposed.
use crate::agreement::Mapping;
use crate::buffer::Element;
use crate::error::{Error, Kind};
use crate::gather::{self, Steps};
use crate::layout::Layout;
use crate::linalg::{Inner, Orientation, PlusTimes, Scalar};
use crate::reduce::{self, Reduction, Sum};
use crate::{Float, Native};
use num_traits::{Float as _, One as _, Zero as _};
use std::borrow::Cow;
use std::ops::Range;

/// Groups the selected axes into one reduction axis, and keeps the other axes' metadata. One axis, or consecutive axes, needs no copy. Nonconsecutive axes gather through the ordinary transpose plan.
pub struct Axes { pub dims: [usize; 3], pub layout: Layout, shape: Vec<usize>, steps: Option<Vec<Steps>> }

impl Axes {
    pub fn new(layout: &Layout, axes: &[usize]) -> Result<Self, Error> {
        let shape = layout.shape();
        let mut axes = axes.to_vec();
        axes.sort_unstable();
        axes.dedup();
        if axes.last().is_some_and(|&a| a >= shape.len()) { return Err(Error::new(Kind::Index, "reduction axis is out of range")); }
        let remaining: Vec<_> = (0..shape.len()).filter(|a| !axes.contains(a)).collect();
        let result = layout.axes(remaining.iter().copied());
        let (first, end) = axes.first().zip(axes.last()).map_or((0, 0), |(&a, &b)| (a, b + 1));
        if end - first == axes.len() {
            return Ok(Self { dims: [shape[..first].iter().product(), shape[first..end].iter().product(), shape[end..].iter().product()], layout: result, shape: shape.to_vec(), steps: None });
        }
        let mut destinations = vec![0; shape.len()];
        for (to, &from) in remaining.iter().chain(&axes).enumerate() { destinations[from] = to; }
        let plan = crate::structure::transpose(layout, &destinations)?;
        Ok(Self { dims: [remaining.iter().map(|&a| shape[a]).product(), axes.iter().map(|&a| shape[a]).product(), 1], layout: result, shape: shape.to_vec(), steps: Some(plan.steps) })
    }

    pub fn read<'a, T: Element>(&self, data: &'a [T]) -> Cow<'a, [T]> {
        let Some(steps) = &self.steps else { return Cow::Borrowed(data) };
        let mut out = Vec::with_capacity(data.len());
        gather::walk(&mut out, data, 0, steps, T::FILL, gather::Same);
        Cow::Owned(out)
    }

    /// The offsets that `mapping` reads, in the order the reduction reads its items: one entry per axis where the mapping and the axes compose, otherwise a table.
    fn plan(&self, mapping: &Mapping) -> (usize, Vec<Steps>) {
        let frame: Vec<usize> = self.steps.as_ref().map_or_else(|| self.shape.clone(), |steps| steps.iter().map(Steps::len).collect());
        let ordered = match &self.steps {
            Some(steps) => mapping.composed(&Mapping::Steps { base: 0, steps: steps.clone() }, &self.shape, &frame, true).expect("a table always composes"),
            None => mapping.clone(),
        };
        ordered.entries(&frame).unwrap_or_else(|| (0, vec![Steps::table((0..frame.iter().product()).map(|j| ordered.get(j).unwrap_or(Steps::FILL)).collect())]))
    }
}

/// Means along the middle axis of `[outer, len, inner]`. Empty lanes give NaN.
pub fn mean<T: Scalar>(data: &[T], dims: [usize; 3]) -> Vec<T> where Sum: Reduction<T, Total = T> {
    let mut out = reduce::along::<T, Sum>(data, dims).0;
    let count = T::Real::narrow(dims[1] as f64);
    for v in &mut out { *v = v.div_real(count); }
    out
}

/// Sums a mapped array through the same parallel axis traversal as ordinary reductions, materializing only cache-sized chunks.
#[inline]
fn mapped<F: Copy, R: Reduction<F>>(dims: [usize; 3], f: impl Fn(Range<usize>, &mut Vec<F>) + Sync) -> Vec<R::Total> {
    let mut totals = vec![R::unit(); dims[0] * dims[2]];
    if dims[1] == 0 || totals.is_empty() { return totals; }
    reduce::fold_pieces::<F, R>(&mut totals, dims, &|range, fold| {
        let mut work = Vec::with_capacity(crate::CHUNK);
        for start in (range.start..range.end).step_by(crate::CHUNK) {
            let end = range.end.min(start + crate::CHUNK);
            work.clear();
            f(start..end, &mut work);
            fold(start, &work);
        }
    });
    totals
}

#[inline]
fn values<T: Copy + Sync, F: Copy, R: Reduction<F>>(data: &[T], dims: [usize; 3], f: impl Fn(T) -> F + Sync) -> Vec<R::Total> {
    mapped::<F, R>(dims, |range, out| out.extend(data[range].iter().map(|&x| f(x))))
}

#[inline(always)]
fn square<T: Scalar>(x: T) -> T::Real { x.re().algebraic_mul(x.re()).algebraic_add(x.im().algebraic_mul(x.im())) }

/// Two-pass variances, real for complex data, with denominator `len - ddof`. Empty lanes and nonpositive denominators give NaN.
pub fn variance<T: Scalar>(data: &[T], dims: [usize; 3], ddof: usize) -> Vec<T::Real> where Sum: Reduction<T, Total = T> {
    let means = mean(data, dims);
    let [_, len, inner] = dims;
    let mut out = mapped::<T::Real, Sum>(dims, |range, out| {
        let mut start = range.start;
        while start < range.end {
            if inner == 1 {
                let m = means[start / len];
                let end = range.end.min((start / len + 1) * len);
                out.extend(data[start..end].iter().map(#[inline(always)] |&x| square(x - m)));
                start = end;
            } else {
                let m = start / (len * inner) * inner + start % inner;
                let end = range.end.min((start / inner + 1) * inner);
                out.extend(data[start..end].iter().enumerate().map(#[inline(always)] |(k, &x)| square(x - means[m + k])));
                start = end;
            }
        }
    });
    let count = T::Real::narrow(len.saturating_sub(ddof) as f64);
    for v in &mut out { *v = if ddof >= len { T::Real::nan() } else { *v / count }; }
    out
}

/// Standard deviations, the square roots of [`variance`].
pub fn stddev<T: Scalar>(data: &[T], dims: [usize; 3], ddof: usize) -> Vec<T::Real> where Sum: Reduction<T, Total = T> {
    let mut out = variance(data, dims, ddof);
    for v in &mut out { *v = v.sqrt(); }
    out
}

/// The elementwise p-norm along the selected axes: 0 counts nonzero values, ±infinity selects the greatest/least magnitude, and a finite p gives `(sum |x|^p)^(1/p)`. A pair of axes still gives this norm, not a matrix-induced norm.
pub fn norm<T: Scalar>(data: &[T], dims: [usize; 3], p: f64) -> Vec<T::Real> {
    if p == 1.0 { return values::<_, _, Sum>(data, dims, #[inline(always)] |x| x.modulus()); }
    if p == 0.0 { return values::<_, _, Sum>(data, dims, #[inline(always)] |x| if x == T::zero() { T::Real::zero() } else { T::Real::one() }); }
    if p == f64::INFINITY { return values::<_, _, reduce::Max>(data, dims, #[inline(always)] |x| x.modulus()); }
    if p == f64::NEG_INFINITY { return values::<_, _, reduce::Min>(data, dims, #[inline(always)] |x| x.modulus()); }
    if p == 2.0 {
        let mut out = values::<_, _, Sum>(data, dims, #[inline(always)] |x| square(x));
        for v in &mut out { *v = v.sqrt(); }
        return out;
    }
    let mut out = values::<_, _, Sum>(data, dims, #[inline(always)] |x| crate::pow(x.modulus(), T::Real::narrow(p)));
    for v in &mut out { *v = crate::pow(*v, T::Real::narrow(1.0 / p)); }
    out
}

/// Weighted means. The two mappings broadcast `data` and `weights` into an array of `dims`. Weights may be real or complex; a zero total weight gives the ordinary floating-point quotient.
pub fn average<T: Scalar + Element>(data: &[T], weights: &[T], axes: &Axes, mappings: [&Mapping; 2]) -> Vec<T> where Sum: Reduction<T, Total = T> {
    let [(xbase, xsteps), (wbase, wsteps)] = mappings.map(|mapping| axes.plan(mapping));
    let read = |x: &[T], base, steps: &[Steps], range: Range<usize>, out: &mut Vec<T>| gather::walk_range(out, x, base, steps, T::zero(), range.start, range.len(), gather::Same);
    let mut sums = mapped::<_, Sum>(axes.dims, |range, out| {
        read(data, xbase, &xsteps, range.clone(), out);
        let mut w = Vec::with_capacity(range.len());
        read(weights, wbase, &wsteps, range, &mut w);
        for (x, w) in out.iter_mut().zip(w) { *x = *x * w; }
    });
    let totals = mapped::<_, Sum>(axes.dims, |range, out| read(weights, wbase, &wsteps, range, out));
    for (sum, weight) in sums.iter_mut().zip(totals) { *sum = sum.quotient(weight); }
    sums
}

/// The covariance or correlation of a row-major `variables × observations` matrix. The centered Gram product uses the same native/Accelerate product as linear algebra. Complex covariance conjugates the second variable.
pub fn covariance<T: Scalar>(data: &[T], [rows, cols]: [usize; 2], rowvar: bool, ddof: usize, correlation: bool) -> Vec<T>
where Sum: Reduction<T, Total = T>, PlusTimes: Inner<T, Out = T> {
    let (variables, observations, dims) = if rowvar { (rows, cols, [rows, cols, 1]) } else { (cols, rows, [1, rows, cols]) };
    let means = mean(data, dims);
    let mut centered = Vec::with_capacity(data.len());
    if rowvar { centered.extend_from_slice(data); }
    else {
        centered.resize(data.len(), T::zero());
        crate::linalg::transpose_into(data, cols, &mut centered, rows, [rows, cols], |x| x);
    }
    if observations > 0 {
        for (row, &m) in centered.chunks_mut(observations).zip(&means) { for x in row { *x = *x - m; } }
    }
    let conjugate = if size_of::<T>() == size_of::<T::Real>() { Cow::Borrowed(&centered[..]) }
        else { Cow::Owned(centered.iter().map(|&x| x.conj()).collect()) };
    let mut out = PlusTimes.inner_ops(&centered, &conjugate, [Orientation::Normal, Orientation::Transpose], [variables, observations, variables]);
    let count = T::Real::narrow(observations.saturating_sub(ddof) as f64);
    if correlation {
        let scales: Vec<_> = (0..variables).map(|j| out[j * (variables + 1)].re().sqrt()).collect();
        for (ij, x) in out.iter_mut().enumerate() { *x = x.div_real(scales[ij / variables] * scales[ij % variables]); }
    } else {
        for x in &mut out { *x = if ddof >= observations { T::from_real(T::Real::nan()) } else { x.div_real(count) }; }
    }
    out
}

/// Equally spaced histogram edges, from the data's extrema where no range is given. Constant data expands by 0.5 on each side; empty data uses 0 to 1.
pub fn histogram_edges<F: Native>(data: &[F], bins: usize, range: Option<(f64, f64)>) -> Result<Vec<F>, Error> {
    if bins == 0 { return Err(Error::new(Kind::Domain, "histogram needs a positive bin count")); }
    let (mut lo, mut hi) = range.unwrap_or_else(|| if data.is_empty() { (0.0, 1.0) } else { (reduce::min(data).wide(), reduce::max(data).wide()) });
    if lo == hi { lo -= 0.5; hi += 0.5; }
    if !lo.is_finite() || !hi.is_finite() || lo >= hi { return Err(Error::new(Kind::Domain, "histogram needs a finite increasing range")); }
    Ok((0..=bins).map(|j| F::narrow(lo + (hi - lo) * (j as f64 / bins as f64))).collect())
}

/// Counts in increasing bin edges. Bins include their lower edge and exclude their upper edge, except that the last bin includes both. Values outside the edges and NaNs aren't counted.
pub fn histogram<F: Native>(data: &[F], edges: &[F]) -> Result<Vec<i64>, Error> {
    if edges.len() < 2 || edges.windows(2).any(|p| p[0].is_nan() || p[0] >= p[1]) || edges.last().is_some_and(|v| v.is_nan()) {
        return Err(Error::new(Kind::Domain, "histogram needs at least two strictly increasing bin edges"));
    }
    let bins = crate::search::bins(&crate::tolerant::order_keys(edges), &crate::tolerant::order_keys(data));
    let n = edges.len() - 1;
    let mut out = vec![0; n];
    for (&x, b) in data.iter().zip(bins) {
        let b = b as usize;
        if b > 0 && b <= n { out[b - 1] += 1; }
        else if x == edges[n] { out[n - 1] += 1; }
    }
    Ok(out)
}
